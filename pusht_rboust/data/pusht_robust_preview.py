#!/usr/bin/env python3
"""Render a few robust PushT insertion examples before collecting a dataset."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from pusht_rboust.data.pusht_robust_data_gen import (
    logged_action_from_target,
    make_controller_config,
    make_geometry,
)
from pusht_rboust.test.pusht_insertion_scripted_expert import ScriptedInsertionExpert, save_video
from pusht_rboust.test.pusht_insertion_test import ENV_FPS, get_state, make_env, success_metrics


DEFAULT_OUT_DIR = Path("pusht_rboust/data/pusht_robust_preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--obs-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=ENV_FPS)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument("--buffer", type=float, default=25.0)
    parser.add_argument("--obstacle-width", type=float, default=512.0 / 4.0)
    parser.add_argument("--obstacle-height", type=float, default=135.0)
    parser.add_argument("--goal-y", type=float, default=330.0)
    parser.add_argument("--start-dx-range", type=float, default=18.0)
    parser.add_argument("--start-dy-min", type=float, default=-130.0)
    parser.add_argument("--start-dy-max", type=float, default=-95.0)
    parser.add_argument("--start-dtheta-deg-range", type=float, default=30.0)
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


def rollout_preview(args: argparse.Namespace, episode_idx: int) -> dict[str, object]:
    seed = args.start_seed + episode_idx
    rng = np.random.default_rng(seed)
    geometry = make_geometry(args, rng)
    controller_config = make_controller_config(args)
    expert = ScriptedInsertionExpert(geometry, controller_config)
    env = make_env(geometry, render_size=args.render_size, obs_size=args.obs_size)

    episode_dir = args.out_dir / f"example_{episode_idx:03d}_seed_{seed}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    logged_actions: list[np.ndarray] = []
    step_meta: list[dict[str, object]] = []
    metrics: dict[str, object] = {}

    try:
        env.reset(seed=seed)
        frames.append(np.asarray(env.render(), dtype=np.uint8))
        imageio.imwrite(episode_dir / "initial.png", frames[-1])
        states.append(get_state(env))

        for step_idx in range(args.max_steps):
            state = get_state(env)
            target_action, meta = expert.select_action(state)
            logged_action = logged_action_from_target(target_action, state[:2])
            _, reward, terminated, truncated, info = env.step(target_action.astype(np.float32))

            next_state = get_state(env)
            metrics = success_metrics(next_state, geometry.goal_block_pose)
            target_actions.append(target_action.astype(np.float32))
            logged_actions.append(logged_action.astype(np.float32))
            states.append(next_state)
            meta.update(
                {
                    "step": step_idx,
                    "reward": float(reward),
                    "n_contacts": int(info.get("n_contacts", 0)),
                    "metrics": metrics,
                }
            )
            step_meta.append(meta)

            if (step_idx + 1) % args.save_every == 0 or metrics["success"]:
                frames.append(np.asarray(env.render(), dtype=np.uint8))
            if metrics["success"] or terminated or truncated:
                break

        if not metrics:
            metrics = success_metrics(get_state(env), geometry.goal_block_pose)
        imageio.imwrite(episode_dir / "final.png", frames[-1])
        save_video(episode_dir / "rollout.mp4", frames, fps=args.fps)
        np.savez_compressed(
            episode_dir / "trajectory.npz",
            states=np.asarray(states, dtype=np.float32),
            target_actions=np.asarray(target_actions, dtype=np.float32),
            logged_actions=np.asarray(logged_actions, dtype=np.float32),
            goal_block_pose=geometry.goal_block_pose.astype(np.float32),
            start_state=geometry.start_state.astype(np.float32),
        )
    finally:
        env.close()

    summary = {
        "episode": episode_idx,
        "seed": seed,
        "steps": len(target_actions),
        "success": bool(metrics["success"]),
        "metrics": metrics,
        "geometry": asdict(geometry),
        "controller": asdict(controller_config),
        "video_path": str(episode_dir / "rollout.mp4"),
        "initial_image_path": str(episode_dir / "initial.png"),
        "final_image_path": str(episode_dir / "final.png"),
        "trajectory_path": str(episode_dir / "trajectory.npz"),
        "steps_meta": step_meta,
    }
    with (episode_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.examples < 1:
        raise ValueError("--examples must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1.")
    if args.start_dy_max < args.start_dy_min:
        raise ValueError("--start-dy-max must be >= --start-dy-min.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode_idx in tqdm(range(args.examples), desc="Rendering robust PushT previews", unit="example"):
        summaries.append(rollout_preview(args, episode_idx))

    aggregate = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "examples": summaries,
        "success_rate": float(np.mean([summary["success"] for summary in summaries])),
        "mean_steps": float(np.mean([summary["steps"] for summary in summaries])),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "examples": len(summaries),
                "success_rate": aggregate["success_rate"],
                "mean_steps": aggregate["mean_steps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
