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

from rope.data.iiwa_hardware import MakeFullBimanualStation
from rope.data.iiwa_cartesian_ik import SingleIiwaPositionIK, make_cartesian_line_waypoints, solve_cartesian_path
from rope.real.collision_guard import MujocoArmCollisionGuard
from rope.shared.lab_env import LabEnv


HARDWARE_HOME_Q0_DEG = np.array([-65.0, 3.0, -167.0, 113.0, 8.0, 5.0, 0.0])
HARDWARE_HOME_Q1_DEG = np.array([65.0, 3.0, -13.0, -113.0, -8.0, -5.0, 0.0])
ARM1_EE_Z_ROTATION_OFFSET_RAD = 0.0
ARM1_EE_POSITION_OFFSET_B = np.array([0.0, 0.0, 0.03], dtype=float)


def validate_bimanual_home_path(q0_start, q1_start, q0_goal, q1_goal, duration):
    guard = MujocoArmCollisionGuard(LabEnv(), min_arm_arm_distance=0.06)
    start = np.concatenate([q0_start, q1_start])
    goal = np.concatenate([q0_goal, q1_goal])
    guard.validate_path(start, goal, duration=duration, label="home")


def validate_bimanual_home_sequence(q0_path, q1_path, segment_duration):
    guard = MujocoArmCollisionGuard(LabEnv(), min_arm_arm_distance=0.06)
    path = np.concatenate([q0_path, q1_path], axis=1)
    guard.validate_sequence(path, segment_duration=segment_duration, label="oriented home")


def hardware_home_goal():
    q0_nominal = np.deg2rad(HARDWARE_HOME_Q0_DEG)
    q1_nominal = np.deg2rad(HARDWARE_HOME_Q1_DEG)
    ik0 = SingleIiwaPositionIK()
    ik1 = SingleIiwaPositionIK(
        ee_z_rotation_offset_rad=ARM1_EE_Z_ROTATION_OFFSET_RAD,
        ee_position_offset_B=ARM1_EE_POSITION_OFFSET_B,
    )
    p0_home = ik0.fk_position(q0_nominal)
    p1_home = ik1.fk_position(q1_nominal)
    q0_goal, info0 = ik0.solve_position_ik(
        p0_home,
        q0_nominal,
        position_tol=0.002,
        max_joint_move_from_seed=None,
    )
    q1_goal, info1 = ik1.solve_position_ik(
        p1_home,
        q1_nominal,
        position_tol=0.002,
        max_joint_move_from_seed=None,
    )
    if q0_goal is None or q1_goal is None:
        raise RuntimeError(f"Could not solve oriented hardware home: arm0={info0}, arm1={info1}")
    print(
        "oriented home IK errors: "
        f"arm0 pos={info0['pos_err']:.5f}m orient={np.rad2deg(info0['orientation_err']):.2f}deg; "
        f"arm1 pos={info1['pos_err']:.5f}m orient={np.rad2deg(info1['orientation_err']):.2f}deg"
    )
    return q0_goal, q1_goal


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


class BimanualJointPathTrajectory(LeafSystem):
    def __init__(self, q0_path, q1_path, duration):
        super().__init__()
        self.q0_path = np.asarray(q0_path, dtype=float)
        self.q1_path = np.asarray(q1_path, dtype=float)
        self.duration = float(duration)
        if self.q0_path.shape != self.q1_path.shape or self.q0_path.ndim != 2 or self.q0_path.shape[1] != 7:
            raise ValueError(f"Expected q paths with shape (T, 7), got {self.q0_path.shape} and {self.q1_path.shape}.")
        self.DeclareVectorOutputPort("q0_cmd", 7, self.CalcQ0)
        self.DeclareVectorOutputPort("q1_cmd", 7, self.CalcQ1)

    def _index(self, t):
        if self.duration <= 0.0 or self.q0_path.shape[0] == 1:
            return self.q0_path.shape[0] - 1
        alpha = np.clip(t / self.duration, 0.0, 1.0)
        return min(int(round(alpha * (self.q0_path.shape[0] - 1))), self.q0_path.shape[0] - 1)

    def CalcQ0(self, context, output):
        output.SetFromVector(self.q0_path[self._index(context.get_time())])

    def CalcQ1(self, context, output):
        output.SetFromVector(self.q1_path[self._index(context.get_time())])


class JointMonitor(LeafSystem):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclarePeriodicPublishEvent(1.0, 0.0, self.Publish)

    def Publish(self, context):
        q = self.get_input_port().Eval(context)
        # print(f"[{self.name}] measured deg = {np.round(np.rad2deg(q), 2)}")


def run_bimanual_command_system(command_system, duration, extra_hold_time):
    builder = DiagramBuilder()
    station = builder.AddSystem(
        MakeFullBimanualStation(control_mode=IiwaControlMode.kPositionOnly)
    )
    traj = builder.AddSystem(command_system)

    builder.Connect(traj.GetOutputPort("q0_cmd"), station.GetInputPort("position_0"))
    builder.Connect(traj.GetOutputPort("q1_cmd"), station.GetInputPort("position_1"))

    mon0 = builder.AddSystem(JointMonitor("arm0"))
    mon1 = builder.AddSystem(JointMonitor("arm1"))

    builder.Connect(station.GetOutputPort("position_measured_0"), mon0.get_input_port())
    builder.Connect(station.GetOutputPort("position_measured_1"), mon1.get_input_port())

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration + extra_hold_time)


def main():
    q0_goal, q1_goal = hardware_home_goal()
    q0_goal_deg = np.rad2deg(q0_goal)
    q1_goal_deg = np.rad2deg(q1_goal)

    # Safety settings.
    max_speed_deg_s = 3.0       # conservative homing speed
    min_duration = 12.0         # never move faster than this
    max_home_joint_move_deg = 170.0
    extra_hold_time = 2.0

    print("Reading current joint positions...")
    q0_start, q1_start = read_current_iiwa_positions()

    print("arm0 start deg =", np.round(np.rad2deg(q0_start), 2))
    print("arm1 start deg =", np.round(np.rad2deg(q1_start), 2))
    print("home source = fixed hardware manipulation home position with constrained link7 orientation")
    print("arm0 goal  deg =", np.round(q0_goal_deg, 2))
    print("arm1 goal  deg =", np.round(q1_goal_deg, 2))

    ik0 = SingleIiwaPositionIK()
    ik1 = SingleIiwaPositionIK(
        ee_z_rotation_offset_rad=ARM1_EE_Z_ROTATION_OFFSET_RAD,
        ee_position_offset_B=ARM1_EE_POSITION_OFFSET_B,
    )
    p0_start = ik0.fk_position(q0_start)
    p1_start = ik1.fk_position(q1_start)
    p0_goal = ik0.fk_position(q0_goal)
    p1_goal = ik1.fk_position(q1_goal)

    q0_oriented_start, info0 = ik0.solve_position_ik(
        p0_start,
        q0_start,
        position_tol=0.002,
        max_joint_move_from_seed=None,
    )
    q1_oriented_start, info1 = ik1.solve_position_ik(
        p1_start,
        q1_start,
        position_tol=0.002,
        max_joint_move_from_seed=None,
    )
    if q0_oriented_start is None or q1_oriented_start is None:
        raise RuntimeError(f"Could not solve oriented start: arm0={info0}, arm1={info1}")

    align_move0_deg = np.max(np.abs(np.rad2deg(q0_oriented_start - q0_start)))
    align_move1_deg = np.max(np.abs(np.rad2deg(q1_oriented_start - q1_start)))
    align_move_deg = max(align_move0_deg, align_move1_deg)
    if align_move_deg > max_home_joint_move_deg:
        raise RuntimeError(
            f"Oriented start is too far for a safe home motion: max_move={align_move_deg:.2f} deg"
        )

    if align_move_deg > 0.5:
        align_duration = max(min_duration, align_move_deg / max_speed_deg_s)
        print(
            "Initial orientation alignment is required because the current pose is not already aligned: "
            f"max_move={align_move_deg:.2f} deg, duration={align_duration:.2f} s"
        )
        validate_bimanual_home_path(q0_start, q1_start, q0_oriented_start, q1_oriented_start, align_duration)
        print("Starting initial orientation alignment...")
        run_bimanual_command_system(
            SafeBimanualHomeTrajectory(
                q0_start=q0_start,
                q1_start=q1_start,
                q0_goal=q0_oriented_start,
                q1_goal=q1_oriented_start,
                duration=align_duration,
            ),
            align_duration,
            extra_hold_time=0.5,
        )
        q0_start = q0_oriented_start
        q1_start = q1_oriented_start

    max_move0_deg = np.max(np.abs(np.rad2deg(q0_goal - q0_start)))
    max_move1_deg = np.max(np.abs(np.rad2deg(q1_goal - q1_start)))
    max_move_deg = max(max_move0_deg, max_move1_deg)
    duration = max(min_duration, max_move_deg / max_speed_deg_s)

    print(f"arm0 oriented home max joint move: {max_move0_deg:.2f} deg")
    print(f"arm1 oriented home max joint move: {max_move1_deg:.2f} deg")
    print(f"planned oriented home duration: {duration:.2f} s")

    # Optional hard stop for accidental huge target.
    if max_move_deg > max_home_joint_move_deg:
        raise RuntimeError(
            f"Target is too far for a safe home motion: max_move={max_move_deg:.2f} deg"
        )

    control_rate = 50.0
    num_waypoints = int(duration * control_rate) + 1
    p0_waypoints = make_cartesian_line_waypoints(p0_start, p0_goal, num_waypoints)
    p1_waypoints = make_cartesian_line_waypoints(p1_start, p1_goal, num_waypoints)
    print("Solving oriented Cartesian home path...")
    q0_path, info0_path = solve_cartesian_path(
        ik_solver=ik0,
        p_waypoints=p0_waypoints,
        q_seed=q0_start,
        position_tol=0.002,
        max_joint_move_from_seed=np.deg2rad(2.0),
    )
    q1_path, info1_path = solve_cartesian_path(
        ik_solver=ik1,
        p_waypoints=p1_waypoints,
        q_seed=q1_start,
        position_tol=0.002,
        max_joint_move_from_seed=np.deg2rad(2.0),
    )
    print(
        "oriented home path max errors: "
        f"arm0 pos={max(x['pos_err'] for x in info0_path):.5f}m orient={np.rad2deg(max(x['orientation_err'] for x in info0_path)):.2f}deg; "
        f"arm1 pos={max(x['pos_err'] for x in info1_path):.5f}m orient={np.rad2deg(max(x['orientation_err'] for x in info1_path)):.2f}deg"
    )
    print("Checking bimanual arm-arm collision path...")
    validate_bimanual_home_sequence(q0_path, q1_path, segment_duration=1.0 / control_rate)
    print("Collision path check passed.")

    print("Starting safe bimanual home motion...")
    run_bimanual_command_system(
        BimanualJointPathTrajectory(q0_path, q1_path, duration),
        duration,
        extra_hold_time,
    )
    print("Done.")


if __name__ == "__main__":
    main()
