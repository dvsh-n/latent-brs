import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drake import lcmt_iiwa_status

from pydrake.all import (
    DiagramBuilder,
    IiwaControlMode,
    LeafSystem,
    Simulator,
)

from pydrake.all import DrakeLcm

from iiwa_hardware import MakeFullBimanualStation
from iiwa_cartesian_ik import (
    SingleIiwaPositionIK,
    make_cartesian_line_waypoints,
    solve_cartesian_path,
)
from rope.real.collision_guard import MujocoArmCollisionGuard
from rope.shared.lab_env import LabEnv


def validate_bimanual_joint_path(q0_path, q1_path, segment_duration, label):
    guard = MujocoArmCollisionGuard(LabEnv(), min_arm_arm_distance=0.06, control_path_samples=2)
    path = np.hstack([np.asarray(q0_path, dtype=float), np.asarray(q1_path, dtype=float)])
    guard.validate_sequence(path, segment_duration=segment_duration, label=label)


def read_current_iiwa_positions(timeout_sec=5.0):
    """
    Read one IIWA_STATUS and one IIWA_STATUS_2 message directly through LCM.
    """
    lcm = DrakeLcm()

    q = {
        "": None,
        "_2": None,
    }

    def make_handler(suffix):
        def handler(data):
            msg = lcmt_iiwa_status.decode(data)
            q[suffix] = np.asarray(msg.joint_position_measured, dtype=float).reshape(7)
        return handler

    lcm.Subscribe("IIWA_STATUS", make_handler(""))
    lcm.Subscribe("IIWA_STATUS_2", make_handler("_2"))

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        lcm.HandleSubscriptions(100)
        if q[""] is not None and q["_2"] is not None:
            return q[""], q["_2"]

    raise RuntimeError(
        f"Failed to receive both IIWA_STATUS and IIWA_STATUS_2 within {timeout_sec} s"
    )


class BimanualWaypointTrajectory(LeafSystem):
    """
    Time-indexed waypoint trajectory for two arms.
    Uses zero-order hold between dense waypoints.
    """

    def __init__(self, q0_path, q1_path, duration):
        super().__init__()

        self.q0_path = np.asarray(q0_path, dtype=float)
        self.q1_path = np.asarray(q1_path, dtype=float)
        self.duration = float(duration)

        assert self.q0_path.ndim == 2 and self.q0_path.shape[1] == 7
        assert self.q1_path.ndim == 2 and self.q1_path.shape[1] == 7
        assert self.q0_path.shape[0] == self.q1_path.shape[0]

        self.num_waypoints = self.q0_path.shape[0]

        self.DeclareVectorOutputPort("q0_cmd", 7, self.CalcQ0)
        self.DeclareVectorOutputPort("q1_cmd", 7, self.CalcQ1)

    def _index(self, t):
        if self.duration <= 0.0:
            return self.num_waypoints - 1

        s = np.clip(t / self.duration, 0.0, 1.0)
        idx = int(round(s * (self.num_waypoints - 1)))
        idx = int(np.clip(idx, 0, self.num_waypoints - 1))
        return idx

    def CalcQ0(self, context, output):
        idx = self._index(context.get_time())
        output.SetFromVector(self.q0_path[idx])

    def CalcQ1(self, context, output):
        idx = self._index(context.get_time())
        output.SetFromVector(self.q1_path[idx])


class JointMonitor(LeafSystem):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclarePeriodicPublishEvent(
            period_sec=1.0,
            offset_sec=0.0,
            publish=self.Publish,
        )

    def Publish(self, context):
        q = self.get_input_port().Eval(context)
        print(f"[{self.name}] measured deg = {np.round(np.rad2deg(q), 2)}")


def main():
    # -------------------------
    # 1. Read current robot q
    # -------------------------
    print("Reading current joint positions...")
    q0_now, q1_now = read_current_iiwa_positions(timeout_sec=5.0)

    print("arm0 q_now deg =", np.round(np.rad2deg(q0_now), 2))
    print("arm1 q_now deg =", np.round(np.rad2deg(q1_now), 2))

    # -------------------------
    # 2. Build independent IK models
    # -------------------------
    ik0 = SingleIiwaPositionIK()
    ik1 = SingleIiwaPositionIK()

    p0_start = ik0.fk_position(q0_now)
    p1_start = ik1.fk_position(q1_now)

    print("arm0 p_start =", np.round(p0_start, 4))
    print("arm1 p_start =", np.round(p1_start, 4))

    # IMPORTANT:
    # These are in each arm's independent local/world frame.
    # Start with very small deltas.
    p0_goal = p0_start + np.array([-0.0, -0.0, 0.1])
    p1_goal = p1_start + np.array([-0.0, 0.0, 0.1])

    print("arm0 p_goal  =", np.round(p0_goal, 4))
    print("arm1 p_goal  =", np.round(p1_goal, 4))

    # -------------------------
    # 3. Cartesian waypoints
    # -------------------------
    duration = 15.0
    control_rate = 50.0
    num_waypoints = int(duration * control_rate) + 1

    p0_waypoints = make_cartesian_line_waypoints(
        p_start=p0_start,
        p_goal=p0_goal,
        num_waypoints=num_waypoints,
    )
    p1_waypoints = make_cartesian_line_waypoints(
        p_start=p1_start,
        p_goal=p1_goal,
        num_waypoints=num_waypoints,
    )

    # -------------------------
    # 4. Sequential IK for each path
    # -------------------------
    print("Solving arm0 IK path...")
    q0_path, info0 = solve_cartesian_path(
        ik_solver=ik0,
        p_waypoints=p0_waypoints,
        q_seed=q0_now,
        position_tol=0.005,
        max_joint_move_from_seed=np.deg2rad(2.0),
        min_sigma=1e-4,
        max_cond=1e4,
    )

    print("Solving arm1 IK path...")
    q1_path, info1 = solve_cartesian_path(
        ik_solver=ik1,
        p_waypoints=p1_waypoints,
        q_seed=q1_now,
        position_tol=0.005,
        max_joint_move_from_seed=np.deg2rad(2.0),
        min_sigma=1e-4,
        max_cond=1e4,
    )

    print("IK path solved.")
    print("arm0 max IK pos err =", max(x["pos_err"] for x in info0))
    print("arm1 max IK pos err =", max(x["pos_err"] for x in info1))
    print("arm0 min sigma =", min(x["sigma_min"] for x in info0))
    print("arm1 min sigma =", min(x["sigma_min"] for x in info1))

    print("arm0 max joint step deg =", np.rad2deg(np.max(np.abs(np.diff(q0_path, axis=0)))))
    print("arm1 max joint step deg =", np.rad2deg(np.max(np.abs(np.diff(q1_path, axis=0)))))

    # Extra safety check: total move should be small for the first test.
    print("arm0 total joint move deg =", np.round(np.rad2deg(q0_path[-1] - q0_path[0]), 2))
    print("arm1 total joint move deg =", np.round(np.rad2deg(q1_path[-1] - q1_path[0]), 2))

    print("Checking bimanual arm-arm collision path...")
    validate_bimanual_joint_path(
        q0_path,
        q1_path,
        segment_duration=1.0 / control_rate,
        label="small Cartesian test",
    )
    print("Collision path check passed.")

    # -------------------------
    # 5. Send to hardware station
    # -------------------------
    builder = DiagramBuilder()

    station = builder.AddSystem(
        MakeFullBimanualStation(control_mode=IiwaControlMode.kPositionOnly)
    )

    traj = builder.AddSystem(
        BimanualWaypointTrajectory(
            q0_path=q0_path,
            q1_path=q1_path,
            duration=duration,
        )
    )

    builder.Connect(traj.GetOutputPort("q0_cmd"), station.GetInputPort("position_0"))
    builder.Connect(traj.GetOutputPort("q1_cmd"), station.GetInputPort("position_1"))

    mon0 = builder.AddSystem(JointMonitor("arm0"))
    mon1 = builder.AddSystem(JointMonitor("arm1"))

    builder.Connect(
        station.GetOutputPort("position_measured_0"),
        mon0.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("position_measured_1"),
        mon1.get_input_port(),
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    print("Starting Cartesian waypoint motion...")
    simulator.AdvanceTo(duration + 2.0)
    print("Done.")


if __name__ == "__main__":
    main()
