#!/usr/bin/env python3
"""Rotate only A7 on IIWA_STATUS / arm 0 by a commanded offset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drake import lcmt_iiwa_status
from pydrake.all import DiagramBuilder, DrakeLcm, IiwaControlMode, LeafSystem, Simulator

from rope.data.iiwa_cartesian_ik import SingleIiwaPositionIK
from rope.data.iiwa_hardware import _MakeIiwaRobot, _MakeLcm


A7_HOME_OFFSET_DEG = 90.0
IIWA_A7_LIMIT_DEG = 175.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-speed-deg-s", type=float, default=3.0)
    parser.add_argument("--min-duration", type=float, default=12.0)
    parser.add_argument("--max-home-joint-move-deg", type=float, default=170.0)
    parser.add_argument("--extra-hold-time", type=float, default=2.0)
    parser.add_argument("--orientation-theta-bound-deg", type=float, default=3.0)
    parser.add_argument("--orientation-constraint-mode", choices=("z_axis_down", "full"), default="z_axis_down")
    parser.add_argument("--a7-home-offset-deg", type=float, default=A7_HOME_OFFSET_DEG)
    parser.add_argument("--i-understand-this-moves-real-robots", action="store_true")
    return parser.parse_args()


def a7_offset_goal(
    orientation_theta_bound_deg: float,
    orientation_constraint_mode: str,
    a7_home_offset_deg: float,
    q0_start: np.ndarray,
) -> np.ndarray:
    q0_start = np.asarray(q0_start, dtype=float).reshape(7)
    ik0 = SingleIiwaPositionIK(
        orientation_theta_bound=np.deg2rad(orientation_theta_bound_deg),
        orientation_constraint_mode=orientation_constraint_mode,
    )
    start_orientation_err = ik0.orientation_constraint_error(q0_start)
    q0_goal = q0_start.copy()
    q0_goal[6] = q0_start[6] + np.deg2rad(a7_home_offset_deg)
    if abs(q0_goal[6]) > np.deg2rad(IIWA_A7_LIMIT_DEG):
        raise RuntimeError(
            "A7 offset target exceeds the iiwa joint limit: "
            f"a7_start={np.rad2deg(q0_start[6]):.2f}deg, "
            f"a7_home_offset={a7_home_offset_deg:.2f}deg, "
            f"a7_target={np.rad2deg(q0_goal[6]):.2f}deg, "
            f"limit=+/-{IIWA_A7_LIMIT_DEG:.2f}deg"
        )
    goal_orientation_err = ik0.orientation_constraint_error(q0_goal)
    print(
        "arm0/status1 A7 offset target: "
        f"{orientation_constraint_mode}_start={np.rad2deg(start_orientation_err):.4f}deg "
        f"{orientation_constraint_mode}_goal={np.rad2deg(goal_orientation_err):.4f}deg "
        f"bound={orientation_theta_bound_deg:.4f}deg "
        f"a7_home_offset={a7_home_offset_deg:.2f}deg "
        f"a7_start={np.rad2deg(q0_start[6]):.2f}deg "
        f"a7_target={np.rad2deg(q0_goal[6]):.2f}deg"
    )
    return q0_goal


def read_current_iiwa_status_1(timeout_sec: float) -> np.ndarray:
    lcm = DrakeLcm()
    q0 = None

    def handler(data: bytes) -> None:
        nonlocal q0
        msg = lcmt_iiwa_status.decode(data)
        q0 = np.asarray(msg.joint_position_measured, dtype=float).reshape(7)

    lcm.Subscribe("IIWA_STATUS", handler)

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        lcm.HandleSubscriptions(100)
        if q0 is not None:
            return q0

    raise RuntimeError(f"Failed to receive IIWA_STATUS within {timeout_sec:.1f} s.")


def smoothstep5(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


class SafeArmHomeTrajectory(LeafSystem):
    def __init__(self, q_start: np.ndarray, q_goal: np.ndarray, duration: float):
        super().__init__()
        self.q_start = np.asarray(q_start, dtype=float).reshape(7)
        self.q_goal = np.asarray(q_goal, dtype=float).reshape(7)
        self.duration = float(duration)
        self.DeclareVectorOutputPort("q_cmd", 7, self.CalcQ)

    def CalcQ(self, context, output) -> None:
        alpha = 1.0 if self.duration <= 0.0 else smoothstep5(context.get_time() / self.duration)
        output.SetFromVector(self.q_start + alpha * (self.q_goal - self.q_start))


class JointMonitor(LeafSystem):
    def __init__(self, q_goal: np.ndarray):
        super().__init__()
        self.q_goal = np.asarray(q_goal, dtype=float).reshape(7)
        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclarePeriodicPublishEvent(1.0, 0.0, self.Publish)

    def Publish(self, context) -> None:
        q = self.get_input_port().Eval(context)
        err_deg = np.rad2deg(np.max(np.abs(q - self.q_goal)))
        print(f"[home arm0/status1] t={context.get_time():.1f}, max_err_deg={err_deg:.2f}")


def run_arm0_command_system(command_system: LeafSystem, q_goal: np.ndarray, duration: float, extra_hold_time: float) -> None:
    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)
    iiwa = builder.AddSystem(
        _MakeIiwaRobot(
            lcm=lcm,
            control_mode=IiwaControlMode.kPositionOnly,
            lcm_channel_suffix="",
        )
    )
    traj = builder.AddSystem(command_system)
    monitor = builder.AddSystem(JointMonitor(q_goal))

    builder.Connect(traj.GetOutputPort("q_cmd"), iiwa.GetInputPort("position"))
    builder.Connect(iiwa.GetOutputPort("position_measured"), monitor.get_input_port())

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration + extra_hold_time)


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_moves_real_robots:
        raise SystemExit("Refusing to move hardware without --i-understand-this-moves-real-robots")
    if args.max_speed_deg_s <= 0.0:
        raise SystemExit("--max-speed-deg-s must be positive")
    if args.min_duration <= 0.0:
        raise SystemExit("--min-duration must be positive")
    if args.orientation_theta_bound_deg < 0.0:
        raise SystemExit("--orientation-theta-bound-deg cannot be negative")
    print("Reading current arm0 / IIWA_STATUS joint position...")
    q0_start = read_current_iiwa_status_1(args.status_timeout)
    q0_goal = a7_offset_goal(
        args.orientation_theta_bound_deg,
        args.orientation_constraint_mode,
        args.a7_home_offset_deg,
        q0_start,
    )
    q0_goal_deg = np.rad2deg(q0_goal)
    goal_delta_deg = np.rad2deg(q0_goal - q0_start)

    print("arm0 start deg =", np.round(np.rad2deg(q0_start), 2))
    print(
        "home source = current measured arm0 pose with only A7 offset; "
        f"a7_home_offset={args.a7_home_offset_deg:.2f}deg "
        f"orientation_constraint_mode={args.orientation_constraint_mode}"
    )
    print("arm0 goal  deg =", np.round(q0_goal_deg, 2))
    print("commanded delta deg =", np.round(goal_delta_deg, 2))

    max_move_deg = float(np.max(np.abs(np.rad2deg(q0_goal - q0_start))))
    duration = max(args.min_duration, max_move_deg / args.max_speed_deg_s)
    print(f"arm0 A7-offset max joint move: {max_move_deg:.2f} deg")
    print(f"planned A7-offset duration: {duration:.2f} s")

    if max_move_deg > args.max_home_joint_move_deg:
        raise RuntimeError(f"Target is too far for a safe home motion: max_move={max_move_deg:.2f} deg")

    print("Starting safe arm0/status1 A7 offset motion...")
    run_arm0_command_system(
        SafeArmHomeTrajectory(q0_start, q0_goal, duration),
        q0_goal,
        duration,
        args.extra_hold_time,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
