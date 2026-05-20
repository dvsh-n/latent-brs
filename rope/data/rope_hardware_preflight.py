#!/usr/bin/env python3
"""No-rope hardware preflight for the bimanual rope collector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rope.real.drake_lcm_backend import DrakeLCMBimanualRobotBackend
from rope.real.real_rope_env import RealRopeEnv


class NullCamera:
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read_rgb_224(self) -> np.ndarray:
        return np.zeros((224, 224, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-mapping", choices=("robot0-left", "robot1-left"), default="robot0-left")
    parser.add_argument("--drake-publish-period", type=float, default=0.005)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--reset-duration", type=float, default=12.0)
    parser.add_argument("--step-duration", type=float, default=0.5)
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--max-control-joint-step-deg", type=float, default=5.0)
    parser.add_argument("--max-reset-joint-move-deg", type=float, default=90.0)
    parser.add_argument("--arm-arm-min-distance", type=float, default=0.06)
    parser.add_argument("--collision-control-samples", type=int, default=5)
    parser.add_argument("--collision-reset-samples", type=int, default=35)
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--wiggle-reach", type=float, default=0.015)
    parser.add_argument("--wiggle-height", type=float, default=0.015)
    parser.add_argument("--wiggle-width", type=float, default=0.030)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--skip-wiggle", action="store_true")
    parser.add_argument(
        "--i-understand-this-moves-real-robots",
        action="store_true",
        help="Required safety acknowledgement before sending hardware commands.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.i_understand_this_moves_real_robots:
        raise RuntimeError("Refusing to move hardware without --i-understand-this-moves-real-robots.")
    if args.drake_publish_period <= 0.0:
        raise ValueError("--drake-publish-period must be positive.")
    if args.status_timeout <= 0.0:
        raise ValueError("--status-timeout must be positive.")
    if args.reset_duration <= 0.5:
        raise ValueError("--reset-duration must be greater than 0.5 s so reset limits are used.")
    if args.step_duration <= 0.0 or args.step_duration > 0.5:
        raise ValueError("--step-duration must be in (0, 0.5] so per-step control limits are used.")
    if args.settle < 0.0:
        raise ValueError("--settle cannot be negative.")
    if args.max_control_joint_step_deg <= 0.0:
        raise ValueError("--max-control-joint-step-deg must be positive.")
    if args.max_reset_joint_move_deg <= 0.0:
        raise ValueError("--max-reset-joint-move-deg must be positive.")
    if args.arm_arm_min_distance < 0.0:
        raise ValueError("--arm-arm-min-distance cannot be negative.")
    if args.collision_control_samples < 1:
        raise ValueError("--collision-control-samples must be positive.")
    if args.collision_reset_samples < 1:
        raise ValueError("--collision-reset-samples must be positive.")
    if args.cycles < 1:
        raise ValueError("--cycles must be positive.")


def make_robot(args: argparse.Namespace) -> DrakeLCMBimanualRobotBackend:
    return DrakeLCMBimanualRobotBackend(
        arm_mapping=args.arm_mapping,
        publish_period=args.drake_publish_period,
        status_timeout=args.status_timeout,
        max_control_joint_step=np.deg2rad(args.max_control_joint_step_deg),
        max_reset_joint_move=np.deg2rad(args.max_reset_joint_move_deg),
    )


def task_center(env: RealRopeEnv, args: argparse.Namespace) -> np.ndarray:
    bounds = env.task_bounds
    center = np.array(
        [
            0.5 * (bounds.reach[0] + bounds.reach[1]),
            0.5 * (bounds.height[0] + bounds.height[1]),
            0.5 * (bounds.width[0] + bounds.width[1]),
        ],
        dtype=np.float64,
    )
    overrides = [args.reach, args.height, args.width]
    for index, value in enumerate(overrides):
        if value is not None:
            center[index] = value
    return env.clip_task_target(center)


def print_joint_summary(prefix: str, qpos: np.ndarray) -> None:
    left = np.round(np.rad2deg(qpos[:7]), 2)
    right = np.round(np.rad2deg(qpos[7:]), 2)
    print(f"{prefix} left deg  = {left}")
    print(f"{prefix} right deg = {right}")


def move_to_task(env: RealRopeEnv, target: np.ndarray, *, duration: float, label: str) -> None:
    start = env.robot.read_qpos_14()
    q_cmd = env.solve_task_to_joints(target)
    max_move_deg = float(np.rad2deg(np.max(np.abs(q_cmd - start))))
    print(f"{label}: task target [reach, height, width] = {np.round(target, 4)}")
    print(f"{label}: max joint move = {max_move_deg:.2f} deg over {duration:.2f} s")
    env.validate_command_path(q_cmd, duration=duration, label=label)
    env.robot.command_joint_positions(q_cmd, duration=duration, blocking=True)
    env.current_task_target = target.astype(np.float64)
    time.sleep(0.05)


def wiggle_targets(center: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    offsets = [
        np.array([args.wiggle_reach, 0.0, 0.0]),
        np.array([-args.wiggle_reach, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, args.wiggle_height, 0.0]),
        np.array([0.0, -args.wiggle_height, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, args.wiggle_width]),
        np.array([0.0, 0.0, -args.wiggle_width]),
        np.zeros(3),
    ]
    return [center + offset for offset in offsets]


def main() -> None:
    args = parse_args()
    validate_args(args)

    robot = make_robot(args)
    env = RealRopeEnv(
        robot=robot,
        camera=NullCamera(),
        command_duration=args.step_duration,
        reset_duration=args.reset_duration,
        enable_collision_guard=True,
        arm_arm_min_distance=args.arm_arm_min_distance,
        collision_control_samples=args.collision_control_samples,
        collision_reset_samples=args.collision_reset_samples,
    )

    try:
        print("Connecting to Drake LCM iiwa station...")
        env.connect()
        measured = env.robot.read_qpos_14()
        print_joint_summary("measured", measured)

        center = task_center(env, args)
        move_to_task(env, center, duration=args.reset_duration, label="preflight reset to rope workspace center")
        time.sleep(args.settle)

        if not args.skip_wiggle:
            for cycle in range(args.cycles):
                print(f"Starting guarded no-rope wiggle cycle {cycle + 1}/{args.cycles}...")
                for index, target in enumerate(wiggle_targets(center, args), start=1):
                    move_to_task(
                        env,
                        env.clip_task_target(target),
                        duration=args.step_duration,
                        label=f"wiggle {cycle + 1}.{index}",
                    )
                    time.sleep(args.settle)

        final_qpos = env.robot.read_qpos_14()
        print_joint_summary("final", final_qpos)
        print("Preflight complete. Arms moved through guarded rope-task poses.")
    finally:
        env.stop()
        env.close()


if __name__ == "__main__":
    main()
