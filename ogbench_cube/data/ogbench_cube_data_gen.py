#!/usr/bin/env python3
"""Collect single-goal OGBench cube expert trajectories into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm


def import_ogbench_modules():
    try:
        import gymnasium
        import ogbench.manipspace  # noqa: F401
        from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

        return gymnasium, CubePlanOracle
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        local_ogbench_root = repo_root / "third_party" / "ogbench"
        if local_ogbench_root.is_dir():
            sys.path.insert(0, str(local_ogbench_root))
            import gymnasium
            import ogbench.manipspace  # noqa: F401
            from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle

            return gymnasium, CubePlanOracle
        raise


DEFAULT_OUTDIR = "ogbench_cube/data/expert_data"
DEFAULT_OUTPUT_NAME = "ogbench_cube_expert.h5"
DEFAULT_ENV_NAME = "cube-single-v0"
DEFAULT_SIM_FREQ_HZ = 500.0
DEFAULT_CONTROL_DECIMATION = 20
XY_SAMPLING_BOUNDS = np.asarray([[0.30, -0.25], [0.5, 0.25]], dtype=np.float32) # x (front back), y (left right), [x_min, y_min] and [x_max, y_max]
MIN_START_GOAL_SAMPLING_DIST = 0.10
ACTION_DIM = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--target-transitions",
        type=int,
        default=None,
        help="Collect variable-length trajectories until this many action transitions are stored.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise-smoothing", type=float, default=0.5)
    parser.add_argument("--segment-dt", type=float, default=0.4)
    parser.add_argument("--goal-threshold", type=float, default=0.04)
    parser.add_argument(
        "--post-goal-steps",
        type=int,
        default=15,
        help="Number of extra control steps to record after the cube first reaches the goal threshold.",
    )
    parser.add_argument("--camera", default="front_pixels")
    return parser.parse_args()


def create_resizable_dataset(
    h5: h5py.File,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = True,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def append_rows(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def valid_training_windows(ep_len: np.ndarray, *, history_size: int = 3, num_preds: int = 1, frameskip: int = 1) -> int:
    num_steps = history_size + num_preds
    required_last_frame_offset = (num_steps - 1) * frameskip
    required_action_end_offset = history_size * frameskip
    required_offset = max(required_last_frame_offset, required_action_end_offset)
    return int(np.maximum(ep_len - 1 - required_offset + 1, 0).sum())


def should_continue(args: argparse.Namespace, num_trajectories: int, total_transitions: int) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def compute_env_timing(sim_freq_hz: float, control_decimation: int) -> tuple[float, float, float]:
    if sim_freq_hz <= 0.0:
        raise ValueError("--sim-freq-hz must be positive.")
    if control_decimation < 1:
        raise ValueError("--control-decimation must be positive.")
    physics_timestep = 1.0 / sim_freq_hz
    control_timestep = physics_timestep * control_decimation
    control_freq_hz = 1.0 / control_timestep
    return physics_timestep, control_timestep, control_freq_hz


def apply_xy_sampling_bounds(env: object) -> None:
    xy_bounds = np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float64)
    if xy_bounds.shape != (2, 2):
        raise ValueError(f"XY_SAMPLING_BOUNDS must have shape (2, 2), got {xy_bounds.shape}.")
    env.unwrapped._object_sampling_bounds = xy_bounds
    env.unwrapped._target_sampling_bounds = xy_bounds


def sample_valid_reset(env: object, trajectory_seed: int, min_sampling_dist: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if min_sampling_dist < 0.0:
        raise ValueError(f"MIN_SAMPLING_DIST cannot be negative, got {min_sampling_dist}.")

    seed_offset = 0
    while True:
        ob, info = env.reset(seed=trajectory_seed + seed_offset)
        start_xy = np.asarray(info["privileged/block_0_pos"][:2], dtype=np.float64)
        goal_xy = np.asarray(info["privileged/target_block_pos"][:2], dtype=np.float64)
        if np.linalg.norm(start_xy - goal_xy) >= min_sampling_dist:
            return ob, info
        seed_offset += 1


def extract_step_info(info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "qpos": np.asarray(info["qpos"], dtype=np.float32),
        "qvel": np.asarray(info["qvel"], dtype=np.float32),
        "control": np.asarray(info["control"], dtype=np.float32),
        "effector_pos": np.asarray(info["proprio/effector_pos"], dtype=np.float32),
        "effector_yaw": np.asarray(info["proprio/effector_yaw"], dtype=np.float32),
        "gripper_opening": np.asarray(info["proprio/gripper_opening"], dtype=np.float32),
        "gripper_contact": np.asarray(info["proprio/gripper_contact"], dtype=np.float32),
        "block_pos": np.asarray(info["privileged/block_0_pos"], dtype=np.float32),
        "block_quat": np.asarray(info["privileged/block_0_quat"], dtype=np.float32),
        "block_yaw": np.asarray(info["privileged/block_0_yaw"], dtype=np.float32),
        "target_block_pos": np.asarray(info["privileged/target_block_pos"], dtype=np.float32),
        "target_block_yaw": np.asarray(info["privileged/target_block_yaw"], dtype=np.float32),
        "time": np.asarray(info["time"], dtype=np.float32),
    }


def reached_goal(info: dict[str, np.ndarray], threshold: float) -> bool:
    return bool(np.linalg.norm(info["privileged/block_0_pos"] - info["privileged/target_block_pos"]) <= threshold)


def collect_trajectory(
    *,
    env: object,
    oracle: object,
    trajectory_seed: int,
    camera: str,
    goal_threshold: float,
    post_goal_steps: int,
) -> tuple[dict[str, np.ndarray], float, bool, bool]:
    ob, info = sample_valid_reset(env, trajectory_seed, MIN_START_GOAL_SAMPLING_DIST)
    oracle.reset(ob, info)

    observations = [np.asarray(ob, dtype=np.float32)]
    frames = [np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)]
    actions: list[np.ndarray] = []
    qpos = [np.asarray(info["qpos"], dtype=np.float32)]
    qvel = [np.asarray(info["qvel"], dtype=np.float32)]
    control = [np.asarray(info["control"], dtype=np.float32)]
    effector_pos = [np.asarray(info["proprio/effector_pos"], dtype=np.float32)]
    effector_yaw = [np.asarray(info["proprio/effector_yaw"], dtype=np.float32)]
    gripper_opening = [np.asarray(info["proprio/gripper_opening"], dtype=np.float32)]
    gripper_contact = [np.asarray(info["proprio/gripper_contact"], dtype=np.float32)]
    block_pos = [np.asarray(info["privileged/block_0_pos"], dtype=np.float32)]
    block_quat = [np.asarray(info["privileged/block_0_quat"], dtype=np.float32)]
    block_yaw = [np.asarray(info["privileged/block_0_yaw"], dtype=np.float32)]
    target_block_pos = [np.asarray(info["privileged/target_block_pos"], dtype=np.float32)]
    target_block_yaw = [np.asarray(info["privileged/target_block_yaw"], dtype=np.float32)]
    time = [np.asarray(info["time"], dtype=np.float32)]

    total_reward = 0.0
    terminated = False
    truncated = False
    success = reached_goal(info, goal_threshold)
    goal_seen = success
    post_goal_step_count = 0

    while not (terminated or truncated or (goal_seen and post_goal_step_count >= post_goal_steps)):
        action = np.asarray(oracle.select_action(ob, info), dtype=np.float32)
        next_ob, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        actions.append(action)
        observations.append(np.asarray(next_ob, dtype=np.float32))
        frames.append(np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8))

        step_info = extract_step_info(next_info)
        qpos.append(step_info["qpos"])
        qvel.append(step_info["qvel"])
        control.append(step_info["control"])
        effector_pos.append(step_info["effector_pos"])
        effector_yaw.append(step_info["effector_yaw"])
        gripper_opening.append(step_info["gripper_opening"])
        gripper_contact.append(step_info["gripper_contact"])
        block_pos.append(step_info["block_pos"])
        block_quat.append(step_info["block_quat"])
        block_yaw.append(step_info["block_yaw"])
        target_block_pos.append(step_info["target_block_pos"])
        target_block_yaw.append(step_info["target_block_yaw"])
        time.append(step_info["time"])

        ob = next_ob
        info = next_info
        success = reached_goal(info, goal_threshold)
        if goal_seen:
            post_goal_step_count += 1
        elif success:
            goal_seen = True

    data = {
        "observation": np.stack(observations, axis=0),
        "pixels": np.stack(frames, axis=0),
        "action": np.stack(actions, axis=0) if actions else np.zeros((0, ACTION_DIM), dtype=np.float32),
        "qpos": np.stack(qpos, axis=0),
        "qvel": np.stack(qvel, axis=0),
        "control": np.stack(control, axis=0),
        "effector_pos": np.stack(effector_pos, axis=0),
        "effector_yaw": np.stack(effector_yaw, axis=0),
        "gripper_opening": np.stack(gripper_opening, axis=0),
        "gripper_contact": np.stack(gripper_contact, axis=0),
        "block_pos": np.stack(block_pos, axis=0),
        "block_quat": np.stack(block_quat, axis=0),
        "block_yaw": np.stack(block_yaw, axis=0),
        "target_block_pos": np.stack(target_block_pos, axis=0),
        "target_block_yaw": np.stack(target_block_yaw, axis=0),
        "time": np.stack(time, axis=0),
    }
    return data, total_reward, bool(terminated), bool(truncated)


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.post_goal_steps < 0:
        raise ValueError("--post-goal-steps cannot be negative.")
    physics_timestep, control_timestep, control_freq_hz = compute_env_timing(
        args.sim_freq_hz,
        args.control_decimation,
    )

    gymnasium, CubePlanOracle = import_ogbench_modules()

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=False,
        max_episode_steps=args.max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=args.width,
        height=args.height,
    )
    apply_xy_sampling_bounds(env)
    oracle = CubePlanOracle(
        env=env,
        segment_dt=args.segment_dt,
        noise=args.noise,
        noise_smoothing=args.noise_smoothing,
    )

    compression = None if args.compression == "none" else args.compression
    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0

    sample_ob, sample_info = env.reset(seed=args.seed)
    sample_frame = np.asarray(env.unwrapped.render(camera=args.camera), dtype=np.uint8)
    obs_dim = int(np.asarray(sample_ob).shape[0])
    qpos_dim = int(np.asarray(sample_info["qpos"]).shape[0])
    qvel_dim = int(np.asarray(sample_info["qvel"]).shape[0])
    control_dim = int(np.asarray(sample_info["control"]).shape[0])
    env.close()

    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=False,
        max_episode_steps=args.max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=args.width,
        height=args.height,
    )
    apply_xy_sampling_bounds(env)
    oracle = CubePlanOracle(
        env=env,
        segment_dt=args.segment_dt,
        noise=args.noise,
        noise_smoothing=args.noise_smoothing,
    )

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "ogbench_cube/data/ogbench_cube_data_gen.py"
        h5.attrs["env_name"] = args.env_name
        h5.attrs["seed"] = args.seed
        h5.attrs["goal_threshold"] = args.goal_threshold
        h5.attrs["post_goal_steps"] = args.post_goal_steps
        h5.attrs["video_dir"] = str(video_dir)
        h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
        h5.attrs["camera"] = args.camera
        h5.attrs["target_visualization"] = False
        h5.attrs["video_fps"] = control_freq_hz
        h5.attrs["sim_freq_hz"] = args.sim_freq_hz
        h5.attrs["control_freq_hz"] = control_freq_hz
        h5.attrs["control_decimation"] = args.control_decimation
        h5.attrs["physics_timestep"] = physics_timestep
        h5.attrs["control_timestep"] = control_timestep
        h5.attrs["max_episode_steps"] = args.max_episode_steps
        h5.attrs["xy_sampling_bounds"] = json.dumps(XY_SAMPLING_BOUNDS.tolist())
        h5.attrs["min_sampling_dist"] = MIN_START_GOAL_SAMPLING_DIST
        h5.attrs["noise"] = args.noise
        h5.attrs["noise_smoothing"] = args.noise_smoothing
        h5.attrs["segment_dt"] = args.segment_dt
        h5.attrs["observation_dim"] = obs_dim
        h5.attrs["action_dim"] = ACTION_DIM
        h5.attrs["qpos_dim"] = qpos_dim
        h5.attrs["qvel_dim"] = qvel_dim
        h5.attrs["control_dim"] = control_dim

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
        qpos_ds = create_resizable_dataset(h5, "qpos", (qpos_dim,), np.float32, chunks=True)
        qvel_ds = create_resizable_dataset(h5, "qvel", (qvel_dim,), np.float32, chunks=True)
        control_ds = create_resizable_dataset(h5, "control", (control_dim,), np.float32, chunks=True)
        effector_pos_ds = create_resizable_dataset(h5, "effector_pos", (3,), np.float32, chunks=True)
        effector_yaw_ds = create_resizable_dataset(h5, "effector_yaw", (1,), np.float32, chunks=True)
        gripper_opening_ds = create_resizable_dataset(h5, "gripper_opening", (1,), np.float32, chunks=True)
        gripper_contact_ds = create_resizable_dataset(h5, "gripper_contact", (1,), np.float32, chunks=True)
        block_pos_ds = create_resizable_dataset(h5, "block_pos", (3,), np.float32, chunks=True)
        block_quat_ds = create_resizable_dataset(h5, "block_quat", (4,), np.float32, chunks=True)
        block_yaw_ds = create_resizable_dataset(h5, "block_yaw", (1,), np.float32, chunks=True)
        target_block_pos_ds = create_resizable_dataset(h5, "target_block_pos", (3,), np.float32, chunks=True)
        target_block_yaw_ds = create_resizable_dataset(h5, "target_block_yaw", (1,), np.float32, chunks=True)
        time_ds = create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
        episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
        step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

        progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
        progress_desc = "Collecting transitions" if args.target_transitions is not None else "Collecting trajectories"
        progress_unit = "step" if args.target_transitions is not None else "traj"

        with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
            while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                trajectory_seed = args.seed + seed_offset
                seed_offset += 1

                trajectory, total_reward, terminated, truncated = collect_trajectory(
                    env=env,
                    oracle=oracle,
                    trajectory_seed=trajectory_seed,
                    camera=args.camera,
                    goal_threshold=args.goal_threshold,
                    post_goal_steps=args.post_goal_steps,
                )
                num_actions = int(trajectory["action"].shape[0])
                if num_actions < args.min_steps:
                    skipped_short += 1
                    continue

                episode_idx = len(step_counts)
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
                append_rows(qpos_ds, trajectory["qpos"])
                append_rows(qvel_ds, trajectory["qvel"])
                append_rows(control_ds, trajectory["control"])
                append_rows(effector_pos_ds, trajectory["effector_pos"])
                append_rows(effector_yaw_ds, trajectory["effector_yaw"])
                append_rows(gripper_opening_ds, trajectory["gripper_opening"])
                append_rows(gripper_contact_ds, trajectory["gripper_contact"])
                append_rows(block_pos_ds, trajectory["block_pos"])
                append_rows(block_quat_ds, trajectory["block_quat"])
                append_rows(block_yaw_ds, trajectory["block_yaw"])
                append_rows(target_block_pos_ds, trajectory["target_block_pos"])
                append_rows(target_block_yaw_ds, trajectory["target_block_yaw"])
                append_rows(time_ds, trajectory["time"])
                append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
                append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
                append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([total_reward], dtype=np.float32))
                append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))

                rewards.append(total_reward)
                step_counts.append(num_actions)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                progress.update(num_actions if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    skipped=skipped_short,
                )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        total_transitions = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = total_transitions
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
        h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
        h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)

    env.close()

    summary = {
        "output_path": str(output_path),
        "video_dir": str(video_dir),
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "total_frames": int(np.sum(step_counts, dtype=np.int64) + len(step_counts)),
        "min_episode_steps": int(np.min(step_counts)) if step_counts else 0,
        "mean_episode_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "max_episode_steps": int(np.max(step_counts)) if step_counts else 0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "terminated_episodes": int(np.sum(terminated_flags, dtype=np.int64)),
        "truncated_episodes": int(np.sum(truncated_flags, dtype=np.int64)),
        "skipped_short_episodes": skipped_short,
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
