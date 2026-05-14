#!/usr/bin/env python3
"""Collect expert Two Room trajectories directly into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_room.plan.expert_plan import expert_action
from two_room.shared import TwoRoomEnv, make_two_room_env

DEFAULT_OUTDIR = "two_room/data/train_data"
DEFAULT_OUTPUT_NAME = "two_room_train.h5"
ACTION_DIM = 2
STATE_DIM = 2
MAX_DOORS = TwoRoomEnv.MAX_DOOR
RATIOS = (0.4, 0.3, 0.2, 0.1)  # expert, expert_plus_noise, random_smooth, random
ROLLOUT_MODES = ("expert", "expert_plus_noise", "random_smooth", "random")
DEFAULT_MAX_STEPS_BY_MODE = (70, 70, 70, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-transitions",
        type=int,
        default=None,
        help="Collect variable-length trajectories until this many action transitions are stored.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=25000,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-expert", type=int, default=DEFAULT_MAX_STEPS_BY_MODE[0])
    parser.add_argument("--max-steps-expert-noisy", type=int, default=DEFAULT_MAX_STEPS_BY_MODE[1])
    parser.add_argument("--max-steps-random-smooth", type=int, default=DEFAULT_MAX_STEPS_BY_MODE[2])
    parser.add_argument("--max-steps-random", type=int, default=DEFAULT_MAX_STEPS_BY_MODE[3])
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--render-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agent-speed", type=float, default=5.0)
    parser.add_argument("--expert-noise-std", type=float, default=0.35)
    parser.add_argument("--smooth-hold-min", type=int, default=3)
    parser.add_argument("--smooth-hold-max", type=int, default=8)
    parser.add_argument("--num-videos", type=int, default=0, help="Save the first N trajectories as MP4 videos.")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-quality", type=int, default=8)
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


def valid_training_windows(ep_len: np.ndarray, *, history_size: int = 2, num_preds: int = 6, frameskip: int = 1) -> int:
    num_steps = history_size + num_preds
    required_last_frame_offset = (num_steps - 1) * frameskip
    required_action_end_offset = (history_size - 1 + num_preds) * frameskip
    required_offset = max(required_last_frame_offset, required_action_end_offset)
    return int(np.maximum(ep_len - 1 - required_offset + 1, 0).sum())


def should_continue(args: argparse.Namespace, num_trajectories: int, total_transitions: int) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def _resize_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame.astype(np.uint8, copy=False)
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _episode_layout(env: TwoRoomEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    door_positions = np.zeros((MAX_DOORS,), dtype=np.float32)
    door_sizes = np.zeros((MAX_DOORS,), dtype=np.float32)
    door_positions[: env.num_doors] = env.door_positions[: env.num_doors].detach().cpu().numpy()
    door_sizes[: env.num_doors] = env.door_sizes[: env.num_doors].detach().cpu().numpy()
    wall_axis = np.full((1,), env.wall_axis, dtype=np.int64)
    wall_thickness = np.full((1,), env.wall_thickness, dtype=np.int64)
    return wall_axis, wall_thickness, door_positions, door_sizes


def _sample_rollout_mode(rng: np.random.Generator) -> str:
    ratios = np.asarray(RATIOS, dtype=np.float64)
    if ratios.shape != (len(ROLLOUT_MODES),):
        raise ValueError(f"RATIOS must have length {len(ROLLOUT_MODES)}.")
    if np.any(ratios < 0.0):
        raise ValueError("RATIOS cannot contain negative values.")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("RATIOS must sum to a positive value.")
    probabilities = ratios / total
    return str(rng.choice(ROLLOUT_MODES, p=probabilities))


def _mode_max_steps(args: argparse.Namespace, mode: str) -> int:
    mapping = {
        "expert": args.max_steps_expert,
        "expert_plus_noise": args.max_steps_expert_noisy,
        "random_smooth": args.max_steps_random_smooth,
        "random": args.max_steps_random,
    }
    return int(mapping[mode])


def _sample_uniform_action(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)


def collect_trajectory(
    *,
    env: TwoRoomEnv,
    args: argparse.Namespace,
    trajectory_seed: int,
    width: int,
    height: int,
    rollout_mode: str,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    _, info = env.reset(seed=trajectory_seed)
    rng = np.random.default_rng(trajectory_seed)
    max_steps = _mode_max_steps(args, rollout_mode)
    held_action: np.ndarray | None = None
    hold_steps_remaining = 0

    states = [np.asarray(info["state"], dtype=np.float32)]
    goals = [np.asarray(info["goal_state"], dtype=np.float32)]
    frames = [_resize_frame(np.asarray(env.render(), dtype=np.uint8), width=width, height=height)]
    actions: list[np.ndarray] = []

    terminated = False
    truncated = False
    for _ in range(max_steps):
        state = np.asarray(info["state"], dtype=np.float32)
        goal_state = np.asarray(info["goal_state"], dtype=np.float32)
        if rollout_mode == "expert":
            action = np.asarray(expert_action(env, state, goal_state), dtype=np.float32)
        elif rollout_mode == "expert_plus_noise":
            expert = np.asarray(expert_action(env, state, goal_state), dtype=np.float32)
            noise = rng.normal(loc=0.0, scale=args.expert_noise_std, size=(ACTION_DIM,)).astype(np.float32)
            action = np.clip(expert + noise, -1.0, 1.0).astype(np.float32)
        elif rollout_mode == "random_smooth":
            if held_action is None or hold_steps_remaining <= 0:
                held_action = _sample_uniform_action(rng)
                hold_steps_remaining = int(
                    rng.integers(args.smooth_hold_min, args.smooth_hold_max + 1)
                )
            action = held_action.copy()
            hold_steps_remaining -= 1
        elif rollout_mode == "random":
            action = _sample_uniform_action(rng)
        else:
            raise ValueError(f"Unknown rollout mode: {rollout_mode}")

        _, _, terminated, env_truncated, info = env.step(action)
        actions.append(action)
        states.append(np.asarray(info["state"], dtype=np.float32))
        goals.append(np.asarray(info["goal_state"], dtype=np.float32))
        frames.append(_resize_frame(np.asarray(env.render(), dtype=np.uint8), width=width, height=height))

        if terminated:
            break

        if env_truncated:
            truncated = True
            break

    if not terminated and not truncated and len(actions) >= max_steps:
        truncated = True

    wall_axis, wall_thickness, door_positions, door_sizes = _episode_layout(env)
    episode_length = len(states)
    data = {
        "pixels": np.stack(frames, axis=0),
        "action": np.stack(actions, axis=0) if actions else np.zeros((0, ACTION_DIM), dtype=np.float32),
        "observation": np.stack(states, axis=0),
        "state": np.stack(states, axis=0),
        "goal_state": np.stack(goals, axis=0),
        "proprio": np.stack(states, axis=0),
        "wall_axis": np.repeat(wall_axis[None, :], episode_length, axis=0),
        "wall_thickness": np.repeat(wall_thickness[None, :], episode_length, axis=0),
        "door_positions": np.repeat(door_positions[None, :], episode_length, axis=0),
        "door_sizes": np.repeat(door_sizes[None, :], episode_length, axis=0),
        "rollout_mode": np.full((episode_length,), ROLLOUT_MODES.index(rollout_mode), dtype=np.int64),
    }
    return data, bool(terminated), bool(truncated)


def store_trajectory(
    *,
    args: argparse.Namespace,
    trajectory_seed: int,
    rollout_mode: str,
    trajectory: dict[str, np.ndarray],
    terminated: bool,
    truncated: bool,
    step_counts: list[int],
    terminated_flags: list[bool],
    truncated_flags: list[bool],
    mode_counts: dict[str, int],
    video_dir: Path,
    pixels_ds: h5py.Dataset,
    action_ds: h5py.Dataset,
    obs_ds: h5py.Dataset,
    state_ds: h5py.Dataset,
    goal_ds: h5py.Dataset,
    proprio_ds: h5py.Dataset,
    wall_axis_ds: h5py.Dataset,
    wall_thickness_ds: h5py.Dataset,
    door_positions_ds: h5py.Dataset,
    door_sizes_ds: h5py.Dataset,
    rollout_mode_ds: h5py.Dataset,
    episode_idx_ds: h5py.Dataset,
    step_idx_ds: h5py.Dataset,
    ep_len_ds: h5py.Dataset,
    ep_offset_ds: h5py.Dataset,
    seed_ds: h5py.Dataset,
    terminated_ds: h5py.Dataset,
    truncated_ds: h5py.Dataset,
) -> int:
    num_actions = int(trajectory["action"].shape[0])
    if num_actions < args.min_steps:
        return 0

    episode_idx = len(step_counts)
    if episode_idx < args.num_videos:
        video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
        imageio.mimwrite(
            video_path,
            trajectory["pixels"],
            fps=args.video_fps,
            quality=args.video_quality,
            macro_block_size=1,
        )

    padded_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
    padded_actions[:-1] = trajectory["action"]
    padded_actions[-1] = np.nan

    offset, _ = append_rows(pixels_ds, trajectory["pixels"])
    append_rows(action_ds, padded_actions)
    append_rows(obs_ds, trajectory["observation"])
    append_rows(state_ds, trajectory["state"])
    append_rows(goal_ds, trajectory["goal_state"])
    append_rows(proprio_ds, trajectory["proprio"])
    append_rows(wall_axis_ds, trajectory["wall_axis"])
    append_rows(wall_thickness_ds, trajectory["wall_thickness"])
    append_rows(door_positions_ds, trajectory["door_positions"])
    append_rows(door_sizes_ds, trajectory["door_sizes"])
    append_rows(rollout_mode_ds, trajectory["rollout_mode"])
    append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
    append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
    append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
    append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
    append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
    append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
    append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))

    step_counts.append(num_actions)
    terminated_flags.append(terminated)
    truncated_flags.append(truncated)
    mode_counts[rollout_mode] += 1
    return num_actions


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if min(
        args.max_steps_expert,
        args.max_steps_expert_noisy,
        args.max_steps_random_smooth,
        args.max_steps_random,
    ) < 1:
        raise ValueError("All max-step arguments must be positive.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive.")
    if args.num_videos < 0:
        raise ValueError("--num-videos cannot be negative.")
    if args.expert_noise_std < 0.0:
        raise ValueError("--expert-noise-std must be non-negative.")
    if args.agent_speed <= 0.0:
        raise ValueError("--agent-speed must be positive.")
    if args.smooth_hold_min < 1:
        raise ValueError("--smooth-hold-min must be positive.")
    if args.smooth_hold_max < args.smooth_hold_min:
        raise ValueError("--smooth-hold-max must be >= --smooth-hold-min.")

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    if args.num_videos > 0:
        video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    compression = None if args.compression == "none" else args.compression
    if output_path.exists():
        output_path.unlink()

    env = make_two_room_env(
        render_mode="rgb_array",
        render_target=args.render_target,
        agent_speed=args.agent_speed,
    )

    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "two_room/data/two_room_data_gen.py"
        h5.attrs["seed"] = args.seed
        h5.attrs["render_target"] = args.render_target
        h5.attrs["agent_speed"] = args.agent_speed
        h5.attrs["video_dir"] = str(video_dir)
        h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
        h5.attrs["video_fps"] = args.video_fps
        h5.attrs["max_steps_expert"] = args.max_steps_expert
        h5.attrs["max_steps_expert_noisy"] = args.max_steps_expert_noisy
        h5.attrs["max_steps_random_smooth"] = args.max_steps_random_smooth
        h5.attrs["max_steps_random"] = args.max_steps_random
        h5.attrs["rollout_ratios"] = json.dumps(RATIOS)
        h5.attrs["rollout_modes"] = json.dumps(ROLLOUT_MODES)
        h5.attrs["expert_noise_std"] = args.expert_noise_std
        h5.attrs["smooth_hold_min"] = args.smooth_hold_min
        h5.attrs["smooth_hold_max"] = args.smooth_hold_max
        h5.attrs["success_distance"] = float(env.SUCCESS_DISTANCE)
        h5.attrs["usable_train_history_default"] = 2
        h5.attrs["usable_train_num_preds_default"] = 6

        ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
        ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
        seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
        terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
        truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
        pixels_ds = create_resizable_dataset(
            h5,
            "pixels",
            (args.height, args.width, 3),
            np.uint8,
            compression=compression,
            chunks=(1, args.height, args.width, 3),
        )
        action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
        obs_ds = create_resizable_dataset(h5, "observation", (STATE_DIM,), np.float32, chunks=True)
        state_ds = create_resizable_dataset(h5, "state", (STATE_DIM,), np.float32, chunks=True)
        goal_ds = create_resizable_dataset(h5, "goal_state", (STATE_DIM,), np.float32, chunks=True)
        proprio_ds = create_resizable_dataset(h5, "proprio", (STATE_DIM,), np.float32, chunks=True)
        wall_axis_ds = create_resizable_dataset(h5, "wall_axis", (1,), np.int64, chunks=True)
        wall_thickness_ds = create_resizable_dataset(h5, "wall_thickness", (1,), np.int64, chunks=True)
        door_positions_ds = create_resizable_dataset(h5, "door_positions", (MAX_DOORS,), np.float32, chunks=True)
        door_sizes_ds = create_resizable_dataset(h5, "door_sizes", (MAX_DOORS,), np.float32, chunks=True)
        rollout_mode_ds = create_resizable_dataset(h5, "rollout_mode", (), np.int64, chunks=True)
        episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
        step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)
        mode_counts = {mode: 0 for mode in ROLLOUT_MODES}

        progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
        progress_desc = "Collecting transitions" if args.target_transitions is not None else "Collecting trajectories"
        progress_unit = "step" if args.target_transitions is not None else "traj"
        with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
            while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                trajectory_seed = args.seed + seed_offset
                seed_offset += 1
                rollout_mode = _sample_rollout_mode(np.random.default_rng(trajectory_seed))
                trajectory, terminated, truncated = collect_trajectory(
                    env=env,
                    args=args,
                    trajectory_seed=trajectory_seed,
                    width=args.width,
                    height=args.height,
                    rollout_mode=rollout_mode,
                )
                num_actions = store_trajectory(
                    args=args,
                    trajectory_seed=trajectory_seed,
                    rollout_mode=rollout_mode,
                    trajectory=trajectory,
                    terminated=terminated,
                    truncated=truncated,
                    step_counts=step_counts,
                    terminated_flags=terminated_flags,
                    truncated_flags=truncated_flags,
                    mode_counts=mode_counts,
                    video_dir=video_dir,
                    pixels_ds=pixels_ds,
                    action_ds=action_ds,
                    obs_ds=obs_ds,
                    state_ds=state_ds,
                    goal_ds=goal_ds,
                    proprio_ds=proprio_ds,
                    wall_axis_ds=wall_axis_ds,
                    wall_thickness_ds=wall_thickness_ds,
                    door_positions_ds=door_positions_ds,
                    door_sizes_ds=door_sizes_ds,
                    rollout_mode_ds=rollout_mode_ds,
                    episode_idx_ds=episode_idx_ds,
                    step_idx_ds=step_idx_ds,
                    ep_len_ds=ep_len_ds,
                    ep_offset_ds=ep_offset_ds,
                    seed_ds=seed_ds,
                    terminated_ds=terminated_ds,
                    truncated_ds=truncated_ds,
                )
                if num_actions == 0:
                    skipped_short += 1
                    continue
                progress.update(num_actions if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    skipped=skipped_short,
                )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["skipped_short_episodes"] = skipped_short
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
        "terminated_episodes": int(np.sum(terminated_flags, dtype=np.int64)),
        "truncated_episodes": int(np.sum(truncated_flags, dtype=np.int64)),
        "skipped_short_episodes": skipped_short,
        "episodes_by_mode": mode_counts,
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
