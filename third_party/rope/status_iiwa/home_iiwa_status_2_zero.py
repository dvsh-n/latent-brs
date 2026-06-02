#!/usr/bin/env python3
"""Command IIWA_STATUS_2 / arm 1 to q = 0 with visible command telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from drake import lcmt_iiwa_status
from pydrake.all import DiagramBuilder, DrakeLcm, IiwaControlMode, LeafSystem, Simulator

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rope.data.iiwa_hardware import _MakeIiwaRobot, _MakeLcm


def smoothstep5(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def read_status_2(timeout_sec: float) -> np.ndarray:
    lcm = DrakeLcm()
    q_measured: np.ndarray | None = None

    def handler(data: bytes) -> None:
        nonlocal q_measured
        msg = lcmt_iiwa_status.decode(data)
        q_measured = np.asarray(msg.joint_position_measured, dtype=float).reshape(7)

    lcm.Subscribe("IIWA_STATUS_2", handler)
    start = time.time()
    while time.time() - start < timeout_sec:
        lcm.HandleSubscriptions(100)
        if q_measured is not None:
            return q_measured
    raise RuntimeError(f"Did not receive IIWA_STATUS_2 within {timeout_sec:.1f} s")


class ZeroTrajectory(LeafSystem):
    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration: float,
        *,
        jog_joint: int | None,
        jog_deg: float,
        max_speed_rad_s: float,
        min_segment_duration: float,
    ):
        super().__init__()
        self.points = [np.asarray(q_start, dtype=float).reshape(7)]
        if jog_joint is not None and abs(jog_deg) > 0.0:
            q_jog = self.points[0].copy()
            q_jog[jog_joint] += np.deg2rad(jog_deg)
            self.points.append(q_jog)
        self.points.append(np.asarray(q_goal, dtype=float).reshape(7))

        self.segment_durations = []
        for q0, q1 in zip(self.points[:-1], self.points[1:]):
            max_move = float(np.max(np.abs(q1 - q0)))
            self.segment_durations.append(max(min_segment_duration, max_move / max_speed_rad_s))
        if len(self.segment_durations) == 1:
            self.segment_durations[0] = max(self.segment_durations[0], duration)

        self.cumulative = np.cumsum([0.0, *self.segment_durations])
        self.DeclareVectorOutputPort("q_cmd", 7, self.CalcOutput)

    @property
    def total_duration(self) -> float:
        return float(self.cumulative[-1])

    def CalcOutput(self, context, output) -> None:
        t = context.get_time()
        index = int(np.searchsorted(self.cumulative, t, side="right") - 1)
        index = min(max(index, 0), len(self.points) - 2)
        t0 = self.cumulative[index]
        duration = self.segment_durations[index]
        alpha = smoothstep5((t - t0) / duration if duration > 0.0 else 1.0)
        q = self.points[index] + alpha * (self.points[index + 1] - self.points[index])
        output.SetFromVector(q)


class StatusPrinter(LeafSystem):
    def __init__(self, q_goal: np.ndarray):
        super().__init__()
        self.q_goal = np.asarray(q_goal, dtype=float).reshape(7)
        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclareVectorInputPort("q_commanded_status", 7)
        self.DeclareVectorInputPort("q_command_output", 7)
        self.DeclarePeriodicPublishEvent(0.5, 0.0, self.Publish)

    def Publish(self, context) -> None:
        q_measured = self.get_input_port(0).Eval(context)
        q_commanded_status = self.get_input_port(1).Eval(context)
        q_command_output = self.get_input_port(2).Eval(context)
        measured_err_deg = np.rad2deg(np.max(np.abs(q_measured - self.q_goal)))
        status_gap_deg = np.rad2deg(np.max(np.abs(q_commanded_status - q_measured)))
        output_gap_deg = np.rad2deg(np.max(np.abs(q_command_output - q_measured)))
        print(
            f"[status2 zero] t={context.get_time():.1f} "
            f"measured_err_deg={measured_err_deg:.2f} "
            f"status_cmd_gap_deg={status_gap_deg:.2f} "
            f"output_cmd_gap_deg={output_gap_deg:.2f} "
            f"measured_deg={np.round(np.rad2deg(q_measured), 2)} "
            f"output_cmd_deg={np.round(np.rad2deg(q_command_output), 2)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--min-segment-duration", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=5.0, help="Minimum duration for the zeroing segment.")
    parser.add_argument("--hold", type=float, default=10.0, help="Extra hold time at exact q=0 command after the motion.")
    parser.add_argument("--tolerance-deg", type=float, default=0.05)
    parser.add_argument("--jog-joint", type=int, default=None, choices=range(7))
    parser.add_argument("--jog-first-deg", type=float, default=0.0)
    parser.add_argument("--i-understand-this-moves-real-robots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.i_understand_this_moves_real_robots:
        raise SystemExit("Refusing to move hardware without --i-understand-this-moves-real-robots")
    if args.max_joint_speed_deg_s <= 0.0:
        raise SystemExit("--max-joint-speed-deg-s must be positive")
    if args.min_segment_duration <= 0.0:
        raise SystemExit("--min-segment-duration must be positive")
    if args.duration < 0.0 or args.hold < 0.0:
        raise SystemExit("--duration and --hold cannot be negative")
    if args.jog_first_deg and args.jog_joint is None:
        raise SystemExit("--jog-first-deg requires --jog-joint")

    q_goal = np.zeros(7, dtype=float)
    q_start = read_status_2(args.status_timeout)
    start_err_deg = float(np.rad2deg(np.max(np.abs(q_start - q_goal))))
    print(f"[status2 zero] start_deg={np.round(np.rad2deg(q_start), 2)}")
    print(f"[status2 zero] target_deg={np.zeros(7)} start_max_err_deg={start_err_deg:.2f}")

    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)
    iiwa = builder.AddSystem(
        _MakeIiwaRobot(
            lcm=lcm,
            control_mode=IiwaControlMode.kPositionOnly,
            lcm_channel_suffix="_2",
        )
    )
    trajectory = builder.AddSystem(
        ZeroTrajectory(
            q_start=q_start,
            q_goal=q_goal,
            duration=args.duration,
            jog_joint=args.jog_joint,
            jog_deg=args.jog_first_deg,
            max_speed_rad_s=np.deg2rad(args.max_joint_speed_deg_s),
            min_segment_duration=args.min_segment_duration,
        )
    )
    printer = builder.AddSystem(StatusPrinter(q_goal))

    builder.Connect(trajectory.GetOutputPort("q_cmd"), iiwa.GetInputPort("position"))
    builder.Connect(iiwa.GetOutputPort("position_measured"), printer.get_input_port(0))
    builder.Connect(iiwa.GetOutputPort("position_commanded"), printer.get_input_port(1))
    builder.Connect(trajectory.GetOutputPort("q_cmd"), printer.get_input_port(2))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    total_time = trajectory.total_duration + args.hold
    print(f"[status2 zero] commanding IIWA_COMMAND_2 for {total_time:.2f} s")
    simulator.AdvanceTo(total_time)

    q_final = read_status_2(args.status_timeout)
    final_err_deg = float(np.rad2deg(np.max(np.abs(q_final - q_goal))))
    print(f"[status2 zero] final_deg={np.round(np.rad2deg(q_final), 2)} final_max_err_deg={final_err_deg:.2f}")
    return 0 if final_err_deg <= args.tolerance_deg else 1


if __name__ == "__main__":
    raise SystemExit(main())
