#!/usr/bin/env python3
"""Collect real-hardware rope trajectories into the same HDF5 schema as sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rope.data.rope_data_gen import (
    ACTION_DIM,
    DEFAULT_MODE,
    DEFAULT_OUTPUT_NAME,
    append_rows,
    create_resizable_dataset,
    make_policy,
    policy_duration,
    policy_goal_state,
    should_continue,
    valid_training_windows,
)
from rope.real.camera_backend import OpenCVCamera
from rope.real.drake_lcm_backend import DrakeLCMBimanualRobotBackend
from rope.real.real_rope_env import RealRopeEnv
from rope.shared.lab_env import RANDOM_CUBIC_SPLINE_MODE, RANDOM_WAYPOINT_MODE, WS_EDGE_SAMPLE_MODE


DEFAULT_REAL_OUTDIR = "rope/data/real_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_REAL_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-transitions", type=int, default=None)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--reset-duration", type=float, default=3.0)
    parser.add_argument("--reset-settle", type=float, default=0.25)
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=(RANDOM_CUBIC_SPLINE_MODE, RANDOM_WAYPOINT_MODE, WS_EDGE_SAMPLE_MODE),
    )
    parser.add_argument("--segment-duration", type=float, default=4.0)
    parser.add_argument("--num-waypoints", type=int, default=6)
    parser.add_argument("--midpoint-inflation-scale", type=float, default=0.08)
    parser.add_argument("--goal-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--robot-backend",
        choices=("drake-lcm",),
        default="drake-lcm",
    )
    parser.add_argument("--arm-mapping", choices=("robot0-left", "robot1-left"), default="robot0-left")
    parser.add_argument("--drake-publish-period", type=float, default=0.005)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-control-joint-step-deg", type=float, default=5.0)
    parser.add_argument("--max-reset-joint-move-deg", type=float, default=90.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-capture-width", type=int, default=None)
    parser.add_argument("--camera-capture-height", type=int, default=None)
    parser.add_argument(
        "--i-understand-this-moves-real-robots",
        action="store_true",
        help="Required safety acknowledgement before sending hardware commands.",
    )
    return parser.parse_args()


def make_robot(args: argparse.Namespace):
    return DrakeLCMBimanualRobotBackend(
        arm_mapping=args.arm_mapping,
        publish_period=args.drake_publish_period,
        status_timeout=args.status_timeout,
        max_control_joint_step=np.deg2rad(args.max_control_joint_step_deg),
        max_reset_joint_move=np.deg2rad(args.max_reset_joint_move_deg),
    )


def collect_trajectory(
    *,
    env: RealRopeEnv,
    trajectory_seed: int,
    max_episode_steps: int,
    control_timestep: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    policy = make_policy(args, env.task_bounds, seed=trajectory_seed)
    initial_target = policy.sample(0.0)
    final_goal = policy_goal_state(policy)
    total_duration = policy_duration(policy)
    env.reset(initial_target)
    time.sleep(args.reset_settle)

    step_info = env.get_step_info(elapsed_time=0.0)
    observations = [step_info["observation"]]
    frames = [env.get_rgb_frame()]
    actions: list[np.ndarray] = []
    task_target = [step_info["task_target"]]
    qpos = [step_info["qpos"]]
    qvel = [step_info["qvel"]]
    control = [step_info["control"]]
    left_attachment_pos = [step_info["left_attachment_pos"]]
    right_attachment_pos = [step_info["right_attachment_pos"]]
    rope_length = [step_info["rope_length"]]
    times = [step_info["time"]]

    current_target = env.current_task_target.astype(np.float64)
    terminated = False
    truncated = False
    next_tick = time.monotonic()

    for step_idx in range(max_episode_steps):
        sample_time = min((step_idx + 1) * control_timestep, total_duration)
        desired = policy.sample(sample_time).as_array()
        requested_delta = desired - current_target
        applied_delta = env.apply_task_delta(requested_delta)
        actions.append(applied_delta)
        current_target = env.current_task_target.astype(np.float64)

        next_tick += control_timestep
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)

        step_info = env.get_step_info(elapsed_time=sample_time)
        observations.append(step_info["observation"])
        frames.append(env.get_rgb_frame())
        task_target.append(step_info["task_target"])
        qpos.append(step_info["qpos"])
        qvel.append(step_info["qvel"])
        control.append(step_info["control"])
        left_attachment_pos.append(step_info["left_attachment_pos"])
        right_attachment_pos.append(step_info["right_attachment_pos"])
        rope_length.append(step_info["rope_length"])
        times.append(step_info["time"])

        if np.linalg.norm(step_info["task_target"] - final_goal) <= args.goal_tolerance:
            terminated = True
            break

    if not terminated:
        truncated = True

    return (
        {
            "observation": np.stack(observations, axis=0),
            "pixels": np.stack(frames, axis=0),
            "action": np.stack(actions, axis=0) if actions else np.zeros((0, ACTION_DIM), dtype=np.float32),
            "task_target": np.stack(task_target, axis=0),
            "qpos": np.stack(qpos, axis=0),
            "qvel": np.stack(qvel, axis=0),
            "control": np.stack(control, axis=0),
            "left_attachment_pos": np.stack(left_attachment_pos, axis=0),
            "right_attachment_pos": np.stack(right_attachment_pos, axis=0),
            "rope_length": np.stack(rope_length, axis=0),
            "time": np.stack(times, axis=0),
        },
        terminated,
        truncated,
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.i_understand_this_moves_real_robots:
        raise RuntimeError("Refusing to move hardware without --i-understand-this-moves-real-robots.")
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.control_timestep <= 0.0:
        raise ValueError("--control-timestep must be positive.")
    if args.segment_duration <= 0.0:
        raise ValueError("--segment-duration must be positive.")
    if args.num_waypoints < 2:
        raise ValueError("--num-waypoints must be at least 2.")
    if args.midpoint_inflation_scale < 0.0:
        raise ValueError("--midpoint-inflation-scale cannot be negative.")
    if args.goal_tolerance < 0.0:
        raise ValueError("--goal-tolerance cannot be negative.")
    if args.drake_publish_period <= 0.0:
        raise ValueError("--drake-publish-period must be positive.")
    if args.status_timeout <= 0.0:
        raise ValueError("--status-timeout must be positive.")
    if args.max_control_joint_step_deg <= 0.0:
        raise ValueError("--max-control-joint-step-deg must be positive.")
    if args.max_reset_joint_move_deg <= 0.0:
        raise ValueError("--max-reset-joint-move-deg must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    robot = make_robot(args)
    camera = OpenCVCamera(
        index=args.camera_index,
        output_width=args.width,
        output_height=args.height,
        capture_width=args.camera_capture_width,
        capture_height=args.camera_capture_height,
    )
    env = RealRopeEnv(robot=robot, camera=camera, command_duration=args.control_timestep, reset_duration=args.reset_duration)
    compression = None if args.compression == "none" else args.compression

    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0
    control_freq_hz = 1.0 / args.control_timestep

    try:
        env.connect()
        env.reset()
        sample_step_info = env.get_step_info(elapsed_time=0.0)
        sample_frame = env.get_rgb_frame()
        obs_dim = int(sample_step_info["observation"].shape[0])
        qpos_dim = int(sample_step_info["qpos"].shape[0])
        qvel_dim = int(sample_step_info["qvel"].shape[0])
        control_dim = int(sample_step_info["control"].shape[0])

        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "stable_worldmodel_hdf5"
            h5.attrs["source"] = "rope/data/rope_real_data_gen.py"
            h5.attrs["hardware"] = True
            h5.attrs["robot_transport"] = "FRI"
            h5.attrs["robot_backend"] = args.robot_backend
            h5.attrs["arm_mapping"] = args.arm_mapping
            h5.attrs["drake_publish_period"] = args.drake_publish_period
            h5.attrs["max_control_joint_step_deg"] = args.max_control_joint_step_deg
            h5.attrs["max_reset_joint_move_deg"] = args.max_reset_joint_move_deg
            h5.attrs["seed"] = args.seed
            h5.attrs["video_dir"] = str(video_dir)
            h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
            h5.attrs["save_mp4"] = args.save_mp4
            h5.attrs["camera"] = f"opencv:{args.camera_index}"
            h5.attrs["mode"] = args.mode
            h5.attrs["video_fps"] = control_freq_hz
            h5.attrs["control_timestep"] = args.control_timestep
            h5.attrs["max_episode_steps"] = args.max_episode_steps
            h5.attrs["segment_duration"] = args.segment_duration
            h5.attrs["num_waypoints"] = args.num_waypoints
            h5.attrs["goal_tolerance"] = args.goal_tolerance
            h5.attrs["observation_dim"] = obs_dim
            h5.attrs["action_dim"] = ACTION_DIM
            h5.attrs["qpos_dim"] = qpos_dim
            h5.attrs["qvel_dim"] = qvel_dim
            h5.attrs["control_dim"] = control_dim
            h5.attrs["task_bounds"] = json.dumps(
                {
                    "reach": list(env.task_bounds.reach),
                    "height": list(env.task_bounds.height),
                    "width": list(env.task_bounds.width),
                }
            )
            h5.attrs["nominal_task_state"] = json.dumps(env.nominal_state.as_array().tolist())

            ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
            ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
            reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
            seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
            terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
            truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
            pixels_ds = create_resizable_dataset(
                h5,
                "pixels",
                sample_frame.shape,
                np.uint8,
                compression=compression,
                chunks=(1, *sample_frame.shape),
            )
            action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
            obs_ds = create_resizable_dataset(h5, "observation", (obs_dim,), np.float32, chunks=True)
            task_target_ds = create_resizable_dataset(h5, "task_target", (3,), np.float32, chunks=True)
            qpos_ds = create_resizable_dataset(h5, "qpos", (qpos_dim,), np.float32, chunks=True)
            qvel_ds = create_resizable_dataset(h5, "qvel", (qvel_dim,), np.float32, chunks=True)
            control_ds = create_resizable_dataset(h5, "control", (control_dim,), np.float32, chunks=True)
            left_attachment_pos_ds = create_resizable_dataset(h5, "left_attachment_pos", (3,), np.float32, chunks=True)
            right_attachment_pos_ds = create_resizable_dataset(h5, "right_attachment_pos", (3,), np.float32, chunks=True)
            rope_length_ds = create_resizable_dataset(h5, "rope_length", (1,), np.float32, chunks=True)
            time_ds = create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
            episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
            step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

            progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
            progress_desc = "Collecting real transitions" if args.target_transitions is not None else "Collecting real trajectories"
            progress_unit = "step" if args.target_transitions is not None else "traj"

            with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
                while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                    trajectory_seed = args.seed + seed_offset
                    seed_offset += 1
                    trajectory, terminated, truncated = collect_trajectory(
                        env=env,
                        trajectory_seed=trajectory_seed,
                        max_episode_steps=args.max_episode_steps,
                        control_timestep=args.control_timestep,
                        args=args,
                    )
                    num_actions = int(trajectory["action"].shape[0])
                    if num_actions < args.min_steps:
                        skipped_short += 1
                        continue

                    episode_idx = len(step_counts)
                    if args.save_mp4:
                        video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
                        imageio.mimwrite(
                            video_path,
                            trajectory["pixels"],
                            fps=control_freq_hz,
                            quality=args.quality,
                            macro_block_size=1,
                        )

                    padded_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                    padded_actions[:-1] = trajectory["action"]
                    padded_actions[-1] = np.nan

                    offset, _ = append_rows(pixels_ds, trajectory["pixels"])
                    append_rows(action_ds, padded_actions)
                    append_rows(obs_ds, trajectory["observation"])
                    append_rows(task_target_ds, trajectory["task_target"])
                    append_rows(qpos_ds, trajectory["qpos"])
                    append_rows(qvel_ds, trajectory["qvel"])
                    append_rows(control_ds, trajectory["control"])
                    append_rows(left_attachment_pos_ds, trajectory["left_attachment_pos"])
                    append_rows(right_attachment_pos_ds, trajectory["right_attachment_pos"])
                    append_rows(rope_length_ds, trajectory["rope_length"])
                    append_rows(time_ds, trajectory["time"])
                    append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
                    append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
                    append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
                    append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                    append_rows(reward_ds, np.asarray([0.0], dtype=np.float32))
                    append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                    append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                    append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))

                    rewards.append(0.0)
                    step_counts.append(num_actions)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    progress.update(num_actions if args.target_transitions is not None else 1)
                    progress.set_postfix(
                        episodes=len(step_counts),
                        transitions=int(np.sum(step_counts, dtype=np.int64)),
                    )

            ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
            h5.attrs["num_episodes"] = len(step_counts)
            h5.attrs["total_frames"] = int(pixels_ds.shape[0])
            h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
            h5.attrs["skipped_short_episodes"] = skipped_short
            h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
            h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
            h5.attrs["terminated_fraction"] = float(np.mean(terminated_flags)) if terminated_flags else 0.0
            h5.attrs["truncated_fraction"] = float(np.mean(truncated_flags)) if truncated_flags else 0.0
            h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)
    finally:
        try:
            env.stop()
        finally:
            env.close()

    summary = {
        "output_path": str(output_path),
        "video_dir": str(video_dir),
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "mean_episode_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "control_timestep": args.control_timestep,
        "control_fps": control_freq_hz,
        "mode": args.mode,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
