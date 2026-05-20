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
    DrakeLcm,
    IiwaControlMode,
    LeafSystem,
    Simulator,
)

from iiwa_hardware import MakeFullBimanualStation
from rope.real.collision_guard import MujocoArmCollisionGuard
from rope.shared.lab_env import LabEnv, TaskState


def validate_bimanual_home_path(q0_start, q1_start, q0_goal, q1_goal, duration):
    guard = MujocoArmCollisionGuard(LabEnv(), min_arm_arm_distance=0.06)
    start = np.concatenate([q0_start, q1_start])
    goal = np.concatenate([q0_goal, q1_goal])
    guard.validate_path(start, goal, duration=duration, label="home")


def mirrored_rope_home_goal():
    env = LabEnv()
    task_target = env.task_bounds.clip(env.nominal_state).as_array()
    env.reset(TaskState.from_array(task_target))
    q_goal = env.get_arm_joint_positions()
    left_site = env.data.site_xpos[env.arm1_site_id].copy()
    right_site = env.data.site_xpos[env.arm2_site_id].copy()
    return q_goal[:7], q_goal[7:], task_target, left_site, right_site


def read_current_iiwa_positions(timeout_sec=5.0):
    lcm = DrakeLcm()
    q = {"0": None, "1": None}

    def handler0(data):
        msg = lcmt_iiwa_status.decode(data)
        q["0"] = np.asarray(msg.joint_position_measured, dtype=float).reshape(7)

    def handler1(data):
        msg = lcmt_iiwa_status.decode(data)
        q["1"] = np.asarray(msg.joint_position_measured, dtype=float).reshape(7)

    lcm.Subscribe("IIWA_STATUS", handler0)
    lcm.Subscribe("IIWA_STATUS_2", handler1)

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        lcm.HandleSubscriptions(100)
        if q["0"] is not None and q["1"] is not None:
            return q["0"], q["1"]

    raise RuntimeError("Failed to receive both IIWA_STATUS and IIWA_STATUS_2.")


def smoothstep5(s):
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


class SafeBimanualHomeTrajectory(LeafSystem):
    def __init__(self, q0_start, q1_start, q0_goal, q1_goal, duration):
        super().__init__()

        self.q0_start = np.asarray(q0_start, dtype=float).reshape(7)
        self.q1_start = np.asarray(q1_start, dtype=float).reshape(7)
        self.q0_goal = np.asarray(q0_goal, dtype=float).reshape(7)
        self.q1_goal = np.asarray(q1_goal, dtype=float).reshape(7)
        self.duration = float(duration)

        self.DeclareVectorOutputPort("q0_cmd", 7, self.CalcQ0)
        self.DeclareVectorOutputPort("q1_cmd", 7, self.CalcQ1)

    def _alpha(self, t):
        if self.duration <= 0:
            return 1.0
        return smoothstep5(t / self.duration)

    def CalcQ0(self, context, output):
        a = self._alpha(context.get_time())
        q = self.q0_start + a * (self.q0_goal - self.q0_start)
        output.SetFromVector(q)

    def CalcQ1(self, context, output):
        a = self._alpha(context.get_time())
        q = self.q1_start + a * (self.q1_goal - self.q1_start)
        output.SetFromVector(q)


class JointMonitor(LeafSystem):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclarePeriodicPublishEvent(1.0, 0.0, self.Publish)

    def Publish(self, context):
        q = self.get_input_port().Eval(context)
        # print(f"[{self.name}] measured deg = {np.round(np.rad2deg(q), 2)}")


def main():
    q0_goal, q1_goal, task_target, left_site, right_site = mirrored_rope_home_goal()
    q0_goal_deg = np.rad2deg(q0_goal)
    q1_goal_deg = np.rad2deg(q1_goal)

    # Safety settings.
    max_speed_deg_s = 3.0       # conservative homing speed
    min_duration = 12.0         # never move faster than this
    extra_hold_time = 2.0

    print("Reading current joint positions...")
    q0_start, q1_start = read_current_iiwa_positions()

    print("arm0 start deg =", np.round(np.rad2deg(q0_start), 2))
    print("arm1 start deg =", np.round(np.rad2deg(q1_start), 2))
    print("rope task home [reach, height, width] =", np.round(task_target, 4))
    print("expected left attachment site =", np.round(left_site, 4))
    print("expected right attachment site =", np.round(right_site, 4))
    print("arm0 goal  deg =", np.round(q0_goal_deg, 2))
    print("arm1 goal  deg =", np.round(q1_goal_deg, 2))

    max_move0_deg = np.max(np.abs(np.rad2deg(q0_goal - q0_start)))
    max_move1_deg = np.max(np.abs(np.rad2deg(q1_goal - q1_start)))
    max_move_deg = max(max_move0_deg, max_move1_deg)

    duration = max(min_duration, max_move_deg / max_speed_deg_s)

    print(f"arm0 max joint move: {max_move0_deg:.2f} deg")
    print(f"arm1 max joint move: {max_move1_deg:.2f} deg")
    print(f"planned duration: {duration:.2f} s")

    # Optional hard stop for accidental huge target.
    if max_move_deg > 90.0:
        raise RuntimeError(
            f"Target is too far for a safe home motion: max_move={max_move_deg:.2f} deg"
        )

    print("Checking bimanual arm-arm collision path...")
    validate_bimanual_home_path(q0_start, q1_start, q0_goal, q1_goal, duration)
    print("Collision path check passed.")

    builder = DiagramBuilder()

    station = builder.AddSystem(
        MakeFullBimanualStation(control_mode=IiwaControlMode.kPositionOnly)
    )

    traj = builder.AddSystem(
        SafeBimanualHomeTrajectory(
            q0_start=q0_start,
            q1_start=q1_start,
            q0_goal=q0_goal,
            q1_goal=q1_goal,
            duration=duration,
        )
    )

    builder.Connect(traj.GetOutputPort("q0_cmd"), station.GetInputPort("position_0"))
    builder.Connect(traj.GetOutputPort("q1_cmd"), station.GetInputPort("position_1"))

    mon0 = builder.AddSystem(JointMonitor("arm0"))
    mon1 = builder.AddSystem(JointMonitor("arm1"))

    builder.Connect(station.GetOutputPort("position_measured_0"), mon0.get_input_port())
    builder.Connect(station.GetOutputPort("position_measured_1"), mon1.get_input_port())

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    print("Starting safe bimanual home motion...")
    simulator.AdvanceTo(duration + extra_hold_time)
    print("Done.")


if __name__ == "__main__":
    main()
