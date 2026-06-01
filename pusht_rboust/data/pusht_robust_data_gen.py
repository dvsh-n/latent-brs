#!/usr/bin/env python3
"""Generate PushT robust insertion rollouts with randomized near-upright starts."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from pusht_rboust.data.pusht_data_gen import H5EpisodeWriter
from pusht_rboust.test.pusht_insertion_scripted_expert import ScriptedExpertConfig, ScriptedInsertionExpert
from pusht_rboust.test.pusht_insertion_test import InsertionGeometry, get_state, make_env, success_metrics


DEFAULT_TRAIN_OUTPUT_PATH = Path("pusht_rboust/data/pusht_robust_train.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_TRAIN_OUTPUT_PATH)
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--keep-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--pixel-compression", choices=("blosc", "lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--pixel-chunk-frames", type=int, default=100)

    parser.add_argument("--buffer", type=float, default=25.0)
    parser.add_argument("--obstacle-width", type=float, default=512.0 / 4.0)
    parser.add_argument("--obstacle-height", type=float, default=135.0)
    parser.add_argument("--goal-y", type=float, default=330.0)
    parser.add_argument("--start-dx-range", type=float, default=18.0)
    parser.add_argument("--start-dy-min", type=float, default=-130.0)
    parser.add_argument("--start-dy-max", type=float, default=-95.0)
    parser.add_argument(
        "--start-dtheta-deg-range",
        type=float,
        default=30.0,
        help="Sample initial T yaw uniformly in [-range, range] degrees around upright.",
    )
    parser.add_argument("--start-agent-y-offset", type=float, default=-70.0)

    parser.add_argument("--x-tol", type=float, default=8.0)
    parser.add_argument("--y-tol", type=float, default=16.0)
    parser.add_argument("--yaw-tol", type=float, default=0.20)
    parser.add_argument("--approach-y", type=float, default=66.0)
    parser.add_argument("--center-approach-y", type=float, default=42.0)
    parser.add_argument("--yaw-side-x", type=float, default=43.0)
    parser.add_argument("--yaw-push-y", type=float, default=46.0)
    parser.add_argument("--side-contact-x", type=float, default=48.0)
    parser.add_argument("--side-push-x", type=float, default=74.0)
    parser.add_argument("--side-y", type=float, default=-2.0)
    parser.add_argument("--insert-push-y", type=float, default=86.0)
    parser.add_argument("--waypoint-tol", type=float, default=32.0)
    parser.add_argument("--center-drive-x", type=float, default=24.0)
    parser.add_argument("--insert-drive-y", type=float, default=28.0)
    parser.add_argument("--yaw-drive-y", type=float, default=80.0)
    return parser.parse_args()


def make_geometry(args: argparse.Namespace, rng: np.random.Generator) -> InsertionGeometry:
    return InsertionGeometry(
        buffer=args.buffer,
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        goal_block_y=args.goal_y,
        start_dx=float(rng.uniform(-args.start_dx_range, args.start_dx_range)),
        start_dy=float(rng.uniform(args.start_dy_min, args.start_dy_max)),
        start_dtheta=math.radians(float(rng.uniform(-args.start_dtheta_deg_range, args.start_dtheta_deg_range))),
        start_agent_y_offset=args.start_agent_y_offset,
    )


def make_controller_config(args: argparse.Namespace) -> ScriptedExpertConfig:
    return ScriptedExpertConfig(
        x_tol=args.x_tol,
        y_tol=args.y_tol,
        yaw_tol=args.yaw_tol,
        approach_y=args.approach_y,
        center_approach_y=args.center_approach_y,
        yaw_side_x=args.yaw_side_x,
        yaw_push_y=args.yaw_push_y,
        side_contact_x=args.side_contact_x,
        side_push_x=args.side_push_x,
        side_y=args.side_y,
        insert_push_y=args.insert_push_y,
        waypoint_tol=args.waypoint_tol,
        center_drive_x=args.center_drive_x,
        insert_drive_y=args.insert_drive_y,
        yaw_drive_y=args.yaw_drive_y,
    )


def extract_pixels(observation: dict[str, Any], env: Any) -> np.ndarray:
    pixels = observation.get("pixels")
    if pixels is None:
        pixels = env.render()
    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB pixels, got shape {pixels.shape}.")
    return pixels


def make_model_state(state: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
    block_pose = np.asarray(state[2:5], dtype=np.float32)
    goal_pose = np.asarray(goal_pose, dtype=np.float32)
    return np.asarray(
        [
            block_pose[0],
            block_pose[1],
            np.cos(block_pose[2]),
            np.sin(block_pose[2]),
            goal_pose[0],
            goal_pose[1],
            goal_pose[2],
        ],
        dtype=np.float32,
    )


def logged_action_from_target(target_xy: np.ndarray, agent_xy: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(target_xy, dtype=np.float32) - np.asarray(agent_xy, dtype=np.float32)) / 100.0, -1.0, 1.0)


def make_proprio(state: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
    agent_xy = np.asarray(state[:2], dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    return np.asarray([agent_xy[0], agent_xy[1], previous_action[0], previous_action[1]], dtype=np.float32)


def rollout_episode(args: argparse.Namespace, seed: int) -> tuple[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], int] | None, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    geometry = make_geometry(args, rng)
    controller_config = make_controller_config(args)
    expert = ScriptedInsertionExpert(geometry, controller_config)
    env = make_env(geometry, render_size=args.image_width, obs_size=args.image_width)

    pixels: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    previous_action = np.zeros(2, dtype=np.float32)
    metrics: dict[str, Any] = {}
    phase_counts: dict[str, int] = {}

    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(args.max_steps):
            state = get_state(env)
            target_action, meta = expert.select_action(state)
            logged_action = logged_action_from_target(target_action, state[:2])

            pixels.append(extract_pixels(observation, env))
            actions.append(logged_action.astype(np.float32))
            states.append(make_model_state(state, geometry.goal_block_pose))
            proprios.append(make_proprio(state, previous_action))
            previous_action = logged_action.astype(np.float32)
            phase = str(meta.get("phase", "unknown"))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            observation, reward, terminated, truncated, info = env.step(target_action.astype(np.float32))
            metrics = success_metrics(get_state(env), geometry.goal_block_pose)
            metrics["reward"] = float(reward)
            if metrics["success"] or terminated or truncated:
                break

        if not metrics:
            metrics = success_metrics(get_state(env), geometry.goal_block_pose)
    finally:
        env.close()

    summary = {
        "seed": seed,
        "steps": len(actions),
        "success": bool(metrics["success"]),
        "metrics": metrics,
        "geometry": asdict(geometry),
        "controller": asdict(controller_config),
        "phase_counts": phase_counts,
    }
    if not actions or (not summary["success"] and not args.keep_failures):
        return None, summary
    return (pixels, actions, states, proprios, 0), summary


def write_metadata(writer: H5EpisodeWriter, args: argparse.Namespace) -> None:
    writer.h5.attrs["task"] = "pusht_robust_insertion"
    writer.h5.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    writer.h5.attrs["args_json"] = json.dumps(
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        sort_keys=True,
    )


def write_sidecar_summary(args: argparse.Namespace, summaries: list[dict[str, Any]], *, saved: int, attempted: int) -> Path:
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    success_rate = float(np.mean([summary["success"] for summary in summaries])) if summaries else 0.0
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out": str(args.out),
        "saved": saved,
        "attempted": attempted,
        "success_rate": success_rate,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "episodes": summaries,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")
    if args.image_height != args.image_width:
        raise ValueError("The current insertion env uses square observations; set --image-height equal to --image-width.")
    if args.pixel_chunk_frames < 1:
        raise ValueError("--pixel-chunk-frames must be >= 1.")
    if args.start_dy_max < args.start_dy_min:
        raise ValueError("--start-dy-max must be >= --start-dy-min.")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")

    writer = H5EpisodeWriter(
        args.out,
        pixel_compression=args.pixel_compression,
        pixel_chunk_frames=args.pixel_chunk_frames,
    )
    summaries: list[dict[str, Any]] = []
    saved = 0
    attempted = 0
    try:
        with tqdm(total=args.num_episodes, desc="Collecting robust PushT episodes", unit="ep") as pbar:
            while saved < args.num_episodes:
                seed = args.start_seed + attempted
                attempted += 1
                episode, summary = rollout_episode(args, seed)
                summaries.append(summary)
                success_label = "yes" if summary["success"] else "no"
                if episode is None:
                    pbar.set_postfix(seed=seed, saved=saved, success=success_label, refresh=False)
                    continue
                writer.append_episodes([episode])
                saved += 1
                pbar.update(1)
                pbar.set_postfix(
                    seed=seed,
                    steps=summary["steps"],
                    success=success_label,
                    refresh=False,
                )
        write_metadata(writer, args)
    finally:
        writer.close()

    success_rate = float(np.mean([summary["success"] for summary in summaries])) if summaries else 0.0
    summary_path = write_sidecar_summary(args, summaries, saved=saved, attempted=attempted)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "summary": str(summary_path),
                "saved": saved,
                "attempted": attempted,
                "success_rate": success_rate,
                "frames": writer.num_frames,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
