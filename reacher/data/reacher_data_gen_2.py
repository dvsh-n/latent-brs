#!/usr/bin/env python3
"""Collect double-integrator-planner Reacher trajectories into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from reacher.plan.plan_double_integrator import (
    angular_distance,
    compute_control_substeps,
    compute_joint2_clearance_limit,
    configure_dm_control_timing,
    configure_offscreen_framebuffer,
    double_integrator_plan,
    hide_target,
    nearest_equivalent_goal,
    reset_env_to_state,
    sample_start_goal,
)
from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_OUTDIR = "reacher/data/double_integrator_rollouts"
DEFAULT_OUTPUT_NAME = "reacher_train.h5"
ROLLOUT_MODES = ("expert", "expert_plus_noise")
DEFAULT_ROLLOUT_RATIOS = (0.6, 0.4)
EXPERT_NOISE_STD = 10.0

PHYSICS_FREQ_HZ = 50.0
CONTROL_FREQ_HZ = 50.0
STATE_DIM = 6
QPOS_DIM = 2
QVEL_DIM = 2
ACTION_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("easy", "hard"), default="hard")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing readable output file instead of starting a new dataset.",
    )
    parser.add_argument(
        "--target-transitions",
        type=int,
        default=None,
        help="Collect variable-length trajectories until this many action transitions are stored.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=15_000,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=PHYSICS_FREQ_HZ)
    parser.add_argument("--control-freq-hz", type=float, default=CONTROL_FREQ_HZ)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--goal-threshold", type=float, default=0.05)
    parser.add_argument("--goal-velocity-threshold", type=float, default=0.25)
    parser.add_argument("--max-blank-steps", type=int, default=10)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--min-goal-distance", type=float, default=0.75)
    parser.add_argument("--min-tip-root-distance", type=float, default=0.045)
    parser.add_argument("--control-gain", type=float, default=0.06)
    parser.add_argument("--velocity-gain", type=float, default=0.015)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--expert-ratio",
        type=float,
        default=DEFAULT_ROLLOUT_RATIOS[0],
        help="Relative amount of clean planner trajectories to collect.",
    )
    parser.add_argument(
        "--expert-plus-noise-ratio",
        type=float,
        default=DEFAULT_ROLLOUT_RATIOS[1],
        help="Relative amount of noised planner trajectories to collect.",
    )
    parser.add_argument("--expert-noise-std", type=float, default=EXPERT_NOISE_STD)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--save-videos", action="store_true", default=True)
    parser.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush HDF5 metadata every N accepted episodes. Use 1 for safest resume behavior.",
    )
    return parser.parse_args()


def make_env(args: argparse.Namespace, *, env_seed: int) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name=args.task,
        seed=env_seed,
        time_limit=args.time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=env_seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / args.physics_freq_hz, time_limit=args.time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, args.width, args.height)
    return env


def planner_observation(qpos: np.ndarray, qvel: np.ndarray, goal_qpos: np.ndarray) -> np.ndarray:
    goal_delta = np.asarray(goal_qpos - qpos, dtype=np.float32)
    goal_delta = ((goal_delta + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    return np.concatenate(
        (
            np.asarray(qpos, dtype=np.float32),
            goal_delta,
            np.asarray(qvel, dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def reached_goal(qpos: np.ndarray, qvel: np.ndarray, goal_qpos: np.ndarray, distance_threshold: float, velocity_threshold: float) -> bool:
    return bool(angular_distance(qpos, goal_qpos) <= distance_threshold and np.linalg.norm(qvel) <= velocity_threshold)


def rollout_probabilities(args: argparse.Namespace) -> np.ndarray:
    ratios = np.asarray([args.expert_ratio, args.expert_plus_noise_ratio], dtype=np.float64)
    if np.any(ratios < 0.0):
        raise ValueError("Rollout ratios cannot be negative.")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one rollout ratio must be positive.")
    return ratios / total


def sample_rollout_mode(args: argparse.Namespace, rng: np.random.Generator) -> str:
    return str(rng.choice(ROLLOUT_MODES, p=rollout_probabilities(args)))


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


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    if name not in h5:
        raise KeyError(f"Cannot resume because dataset '{name}' is missing.")
    return h5[name]


def load_resume_state(h5: h5py.File) -> tuple[list[float], list[int], list[bool], list[str], list[str], int, int]:
    ep_len = np.asarray(require_dataset(h5, "ep_len")[:], dtype=np.int64)
    rewards = [float(value) for value in np.asarray(require_dataset(h5, "reward")[:], dtype=np.float32)]
    terminated_flags = [bool(value) for value in np.asarray(require_dataset(h5, "terminated")[:], dtype=np.bool_)]
    step_counts = [int(length - 1) for length in ep_len]

    rollout_modes: list[str] = []
    rollout_mode_ds = require_dataset(h5, "rollout_mode")
    ep_offset = np.asarray(require_dataset(h5, "ep_offset")[:], dtype=np.int64)
    for offset in ep_offset:
        rollout_modes.append(ROLLOUT_MODES[int(rollout_mode_ds[int(offset)])])

    stop_reasons = [decode_h5_string(value) for value in require_dataset(h5, "stop_reason")[:]]
    skipped_short = int(h5.attrs.get("skipped_short_episodes", 0))
    collection_attempts = int(h5.attrs.get("collection_attempts", len(ep_len) + skipped_short))
    return rewards, step_counts, terminated_flags, rollout_modes, stop_reasons, skipped_short, collection_attempts


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


def collect_trajectory(
    *,
    env: DmControlGymEnv,
    trajectory_seed: int,
    rng: np.random.Generator,
    width: int,
    height: int,
    max_steps: int,
    physics_timestep: float,
    control_timestep: float,
    control_substeps: int,
    time_limit: float,
    horizon: int,
    regularization: float,
    goal_threshold: float,
    goal_velocity_threshold: float,
    max_blank_steps: int,
    min_goal_distance: float,
    min_tip_root_distance: float,
    max_abs_joint2: float,
    control_gain: float,
    velocity_gain: float,
    rollout_mode: str,
    expert_noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool, str]:
    configure_dm_control_timing(env, physics_timestep=physics_timestep, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)

    start_qpos, goal_qpos = sample_start_goal(
        env,
        rng,
        min_goal_distance=min_goal_distance,
        min_tip_root_distance=min_tip_root_distance,
        max_abs_joint2=max_abs_joint2,
    )
    zero_qvel = np.zeros(QVEL_DIM, dtype=np.float32)
    reset_env_to_state(env, seed=trajectory_seed, qpos=start_qpos, qvel=zero_qvel, height=height, width=width)

    physics = env._env.physics
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    qpos = np.asarray(physics.data.qpos[:QPOS_DIM], dtype=np.float32).copy()
    qvel = np.asarray(physics.data.qvel[:QVEL_DIM], dtype=np.float32).copy()
    states = [planner_observation(qpos, qvel, goal_qpos)]
    qpos_log = [qpos]
    qvel_log = [qvel]
    goal_qpos_log = [goal_qpos.copy()]
    actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    frames = [physics.render(height=height, width=width, camera_id=0)]

    total_reward = 0.0
    goal_seen = reached_goal(qpos, qvel, goal_qpos, goal_threshold, goal_velocity_threshold)
    post_goal_steps = 0
    terminated = False
    stop_reason = "max_steps"

    accel_cmd: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    for physics_step in range(max_steps):
        was_already_at_goal = goal_seen
        if physics_step % control_substeps == 0:
            qpos = np.asarray(physics.data.qpos[:QPOS_DIM], dtype=np.float32).copy()
            qvel = np.asarray(physics.data.qvel[:QVEL_DIM], dtype=np.float32).copy()
            goal_for_plan = nearest_equivalent_goal(qpos, goal_qpos, max_abs_joint2=max_abs_joint2)
            accel_plan = double_integrator_plan(
                qpos,
                qvel,
                goal_for_plan,
                dt=control_timestep,
                horizon=horizon,
                regularization=regularization,
            )
            accel_cmd = accel_plan[0].astype(np.float32)
            if rollout_mode == "expert_plus_noise":
                accel_cmd = accel_cmd + rng.normal(loc=0.0, scale=expert_noise_std, size=accel_cmd.shape).astype(np.float32)
            executed_action = control_gain * accel_cmd - velocity_gain * qvel
            executed_action = np.clip(executed_action, action_low, action_high).astype(np.float32)

        if accel_cmd is None or executed_action is None:
            raise RuntimeError("Controller action was not initialized.")

        _, reward, env_terminated, env_truncated, _ = env.step(executed_action)
        total_reward += float(reward)

        next_qpos = np.asarray(physics.data.qpos[:QPOS_DIM], dtype=np.float32).copy()
        next_qvel = np.asarray(physics.data.qvel[:QVEL_DIM], dtype=np.float32).copy()
        states.append(planner_observation(next_qpos, next_qvel, goal_qpos))
        qpos_log.append(next_qpos)
        qvel_log.append(next_qvel)
        goal_qpos_log.append(goal_qpos.copy())
        actions.append(accel_cmd.copy())
        executed_actions.append(executed_action.copy())
        frames.append(physics.render(height=height, width=width, camera_id=0))

        if was_already_at_goal:
            post_goal_steps += 1
        elif reached_goal(next_qpos, next_qvel, goal_qpos, goal_threshold, goal_velocity_threshold):
            goal_seen = True
            stop_reason = "goal_reached"

        if env_terminated or env_truncated:
            terminated = bool(env_terminated)
            stop_reason = "terminated" if env_terminated else "truncated"
            goal_seen = True
        if goal_seen and post_goal_steps >= max_blank_steps:
            break

    return (
        np.stack(states, axis=0),
        np.stack(actions, axis=0),
        np.stack(frames, axis=0),
        np.stack(qpos_log, axis=0),
        np.stack(qvel_log, axis=0),
        np.stack(goal_qpos_log, axis=0),
        np.stack(executed_actions, axis=0),
        total_reward,
        terminated,
        stop_reason,
    )


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.max_steps_per_episode < 1:
        raise ValueError("--max-steps-per-episode must be positive.")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.physics_freq_hz <= 0.0:
        raise ValueError("--physics-freq-hz must be positive.")
    if args.control_freq_hz <= 0.0:
        raise ValueError("--control-freq-hz must be positive.")
    if args.max_blank_steps < 0:
        raise ValueError("--max-blank-steps cannot be negative.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive.")
    if args.control_gain <= 0.0:
        raise ValueError("--control-gain must be positive.")
    if args.velocity_gain < 0.0:
        raise ValueError("--velocity-gain cannot be negative.")
    if args.expert_noise_std < 0.0:
        raise ValueError("--expert-noise-std cannot be negative.")
    if args.min_tip_root_distance < 0.0:
        raise ValueError("--min-tip-root-distance cannot be negative.")

    rollout_mode_probs = rollout_probabilities(args)
    physics_timestep = 1.0 / args.physics_freq_hz
    control_timestep = 1.0 / args.control_freq_hz
    control_substeps = compute_control_substeps(args.physics_freq_hz, args.control_freq_hz)
    video_fps = args.physics_freq_hz

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and not output_path.exists():
        raise FileNotFoundError(f"Cannot resume because output file does not exist: {output_path}")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if args.resume:
        try:
            with h5py.File(output_path, "r"):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"Cannot resume because the HDF5 file is not readable: {output_path}. "
                "If the process died during an HDF5 metadata write, the file may need h5clear recovery "
                "or a fresh output file."
            ) from exc

    env = make_env(args, env_seed=args.seed)
    max_abs_joint2 = compute_joint2_clearance_limit(env, float(args.min_tip_root_distance))

    compression = None if args.compression == "none" else args.compression
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0
    rollout_mode_by_episode: list[str] = []
    stop_reason_by_episode: list[str] = []

    h5_mode = "a" if args.resume else "w"
    with h5py.File(output_path, h5_mode) as h5:
        if args.resume:
            (
                rewards,
                step_counts,
                terminated_flags,
                rollout_mode_by_episode,
                stop_reason_by_episode,
                skipped_short,
                seed_offset,
            ) = load_resume_state(h5)
            ep_len_ds = require_dataset(h5, "ep_len")
            ep_offset_ds = require_dataset(h5, "ep_offset")
            reward_ds = require_dataset(h5, "reward")
            seed_ds = require_dataset(h5, "episode_seed")
            terminated_ds = require_dataset(h5, "terminated")
            stop_reason_ds = require_dataset(h5, "stop_reason")
            rollout_mode_ds = require_dataset(h5, "rollout_mode")
            pixels_ds = require_dataset(h5, "pixels")
            action_ds = require_dataset(h5, "action")
            executed_action_ds = require_dataset(h5, "executed_action")
            obs_ds = require_dataset(h5, "observation")
            qpos_ds = require_dataset(h5, "qpos")
            qvel_ds = require_dataset(h5, "qvel")
            goal_qpos_ds = require_dataset(h5, "goal_qpos")
            episode_idx_ds = require_dataset(h5, "episode_idx")
            step_idx_ds = require_dataset(h5, "step_idx")
        else:
            h5.attrs["format"] = "stable_worldmodel_hdf5"
            h5.attrs["source"] = "data/reacher_data_gen_2.py"
            h5.attrs["state_keys"] = json.dumps(["position(2)", "to_target(2)", "velocity(2)"])
            h5.attrs["state_to_target_semantics"] = "wrapped_joint_delta_to_goal_qpos"
            h5.attrs["action_semantics"] = "double_integrator_accel_cmd"
            h5.attrs["executed_action_semantics"] = "clipped_low_level_action_from_accel_cmd"
            h5.attrs["task"] = args.task
            h5.attrs["seed"] = args.seed
            h5.attrs["video_dir"] = str(video_dir) if args.save_videos else ""
            h5.attrs["save_videos"] = bool(args.save_videos)
            h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
            h5.attrs["video_fps"] = video_fps
            h5.attrs["physics_freq_hz"] = args.physics_freq_hz
            h5.attrs["control_freq_hz"] = args.control_freq_hz
            h5.attrs["physics_timestep"] = physics_timestep
            h5.attrs["control_timestep"] = control_timestep
            h5.attrs["control_substeps"] = control_substeps
            h5.attrs["frames_per_physics_step"] = 1
            h5.attrs["time_limit"] = args.time_limit
            h5.attrs["goal_threshold"] = args.goal_threshold
            h5.attrs["goal_velocity_threshold"] = args.goal_velocity_threshold
            h5.attrs["max_blank_steps"] = args.max_blank_steps
            h5.attrs["max_steps_per_episode"] = args.max_steps_per_episode
            h5.attrs["horizon"] = args.horizon
            h5.attrs["regularization"] = args.regularization
            h5.attrs["control_gain"] = args.control_gain
            h5.attrs["velocity_gain"] = args.velocity_gain
            h5.attrs["min_goal_distance"] = args.min_goal_distance
            h5.attrs["min_tip_root_distance"] = args.min_tip_root_distance
            h5.attrs["max_abs_joint2_without_overlap"] = float(max_abs_joint2)
            h5.attrs["rollout_modes"] = json.dumps(list(ROLLOUT_MODES))
            h5.attrs["rollout_mode_probabilities"] = json.dumps(rollout_mode_probs.tolist())
            h5.attrs["expert_noise_std"] = args.expert_noise_std

            ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
            ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
            reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
            seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
            terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
            stop_reason_ds = create_resizable_dataset(h5, "stop_reason", (), h5py.string_dtype(encoding="utf-8"), chunks=True)
            rollout_mode_ds = create_resizable_dataset(h5, "rollout_mode", (), np.int64, chunks=True)
            pixels_ds = create_resizable_dataset(
                h5,
                "pixels",
                (args.height, args.width, 3),
                np.uint8,
                compression=compression,
                chunks=(1, args.height, args.width, 3),
            )
            action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
            executed_action_ds = create_resizable_dataset(h5, "executed_action", (ACTION_DIM,), np.float32, chunks=True)
            obs_ds = create_resizable_dataset(h5, "observation", (STATE_DIM,), np.float32, chunks=True)
            qpos_ds = create_resizable_dataset(h5, "qpos", (QPOS_DIM,), np.float32, chunks=True)
            qvel_ds = create_resizable_dataset(h5, "qvel", (QVEL_DIM,), np.float32, chunks=True)
            goal_qpos_ds = create_resizable_dataset(h5, "goal_qpos", (QPOS_DIM,), np.float32, chunks=True)
            episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
            step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

        progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
        progress_initial = int(np.sum(step_counts, dtype=np.int64)) if args.target_transitions is not None else len(step_counts)
        progress_desc = "Collecting transitions" if args.target_transitions is not None else "Collecting trajectories"
        progress_unit = "step" if args.target_transitions is not None else "traj"
        with tqdm(total=progress_total, initial=progress_initial, desc=progress_desc, unit=progress_unit) as progress:
            while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                trajectory_seed = args.seed + seed_offset
                seed_offset += 1
                trajectory_rng = np.random.default_rng(trajectory_seed)
                rollout_mode = sample_rollout_mode(args, trajectory_rng)
                (
                    states,
                    actions,
                    frames,
                    qpos,
                    qvel,
                    goal_qpos,
                    executed_actions,
                    total_reward,
                    terminated,
                    stop_reason,
                ) = collect_trajectory(
                    env=env,
                    trajectory_seed=trajectory_seed,
                    rng=trajectory_rng,
                    width=args.width,
                    height=args.height,
                    max_steps=args.max_steps_per_episode,
                    physics_timestep=physics_timestep,
                    control_timestep=control_timestep,
                    control_substeps=control_substeps,
                    time_limit=args.time_limit,
                    horizon=args.horizon,
                    regularization=args.regularization,
                    goal_threshold=args.goal_threshold,
                    goal_velocity_threshold=args.goal_velocity_threshold,
                    max_blank_steps=args.max_blank_steps,
                    min_goal_distance=args.min_goal_distance,
                    min_tip_root_distance=args.min_tip_root_distance,
                    max_abs_joint2=max_abs_joint2,
                    control_gain=args.control_gain,
                    velocity_gain=args.velocity_gain,
                    rollout_mode=rollout_mode,
                    expert_noise_std=args.expert_noise_std,
                )
                if actions.shape[0] < args.min_steps:
                    skipped_short += 1
                    h5.attrs["skipped_short_episodes"] = skipped_short
                    h5.attrs["collection_attempts"] = seed_offset
                    if skipped_short % args.flush_every == 0:
                        h5.flush()
                    continue

                episode_idx = len(step_counts)
                if args.save_videos:
                    video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
                    imageio.mimwrite(
                        video_path,
                        frames,
                        fps=video_fps,
                        quality=args.quality,
                        macro_block_size=1,
                    )

                padded_actions = np.empty((states.shape[0], ACTION_DIM), dtype=np.float32)
                padded_actions[:-1] = actions
                padded_actions[-1] = np.nan

                padded_executed_actions = np.empty((states.shape[0], ACTION_DIM), dtype=np.float32)
                padded_executed_actions[:-1] = executed_actions
                padded_executed_actions[-1] = np.nan

                offset, _ = append_rows(pixels_ds, frames)
                append_rows(action_ds, padded_actions)
                append_rows(executed_action_ds, padded_executed_actions)
                append_rows(obs_ds, states)
                append_rows(qpos_ds, qpos)
                append_rows(qvel_ds, qvel)
                append_rows(goal_qpos_ds, goal_qpos)
                append_rows(episode_idx_ds, np.full((states.shape[0],), episode_idx, dtype=np.int64))
                append_rows(step_idx_ds, np.arange(states.shape[0], dtype=np.int64))
                append_rows(
                    rollout_mode_ds,
                    np.full((states.shape[0],), ROLLOUT_MODES.index(rollout_mode), dtype=np.int64),
                )
                append_rows(ep_len_ds, np.asarray([states.shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([total_reward], dtype=np.float32))
                append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                append_rows(stop_reason_ds, np.asarray([stop_reason], dtype=object))

                rewards.append(total_reward)
                step_counts.append(int(actions.shape[0]))
                terminated_flags.append(terminated)
                rollout_mode_by_episode.append(rollout_mode)
                stop_reason_by_episode.append(stop_reason)
                progress.update(actions.shape[0] if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    skipped=skipped_short,
                )
                if len(step_counts) % args.flush_every == 0:
                    ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
                    h5.attrs["num_episodes"] = len(step_counts)
                    h5.attrs["total_frames"] = int(pixels_ds.shape[0])
                    h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
                    h5.attrs["skipped_short_episodes"] = skipped_short
                    h5.attrs["collection_attempts"] = seed_offset
                    h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
                    h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
                    h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)
                    h5.flush()

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        total_transitions = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = total_transitions
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["collection_attempts"] = seed_offset
        h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
        h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
        h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)

    env.close()

    summary = {
        "output_path": str(output_path),
        "video_dir": str(video_dir) if args.save_videos else None,
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "total_frames": int(np.sum(step_counts, dtype=np.int64) + len(step_counts)),
        "min_episode_steps": int(np.min(step_counts)) if step_counts else 0,
        "mean_episode_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "max_episode_steps": int(np.max(step_counts)) if step_counts else 0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "terminated_episodes": int(np.sum(terminated_flags, dtype=np.int64)),
        "skipped_short_episodes": skipped_short,
        "expert_episodes": int(sum(mode == "expert" for mode in rollout_mode_by_episode)),
        "expert_plus_noise_episodes": int(sum(mode == "expert_plus_noise" for mode in rollout_mode_by_episode)),
        "stop_reasons": {reason: stop_reason_by_episode.count(reason) for reason in sorted(set(stop_reason_by_episode))},
        "expert_noise_std": args.expert_noise_std,
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
