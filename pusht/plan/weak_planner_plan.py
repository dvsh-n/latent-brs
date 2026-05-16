#!/usr/bin/env python3
"""Visualize weak-policy PushT rollouts using this repo's PushT environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
)
from pusht.shared.utils import render_frame

DEFAULT_OUT_DIR = Path("pusht/plan/weak_planner_plan")
DEFAULT_MAX_STEPS = 500
DEFAULT_DIST_CONSTRAINT = 100.0
ENV_ACTION_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--video-name", default="weak_planner_plan.mp4")
    parser.add_argument(
        "--control-interval",
        type=int,
        default=1,
        help="Sample a new weak-policy action every N env steps and hold it in between.",
    )
    parser.add_argument(
        "--dist-constraint",
        type=float,
        default=DEFAULT_DIST_CONSTRAINT,
        help="Constrain sampled action targets to stay within this pixel box around the block center.",
    )
    return parser.parse_args()


def _clip_action_to_space(action: np.ndarray, env: Any) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != action.shape or low.shape != action.shape:
        return np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def _target_xy_to_env_action(env: Any, agent_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        high = np.asarray(action_space.high)
        low = np.asarray(action_space.low)
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)
    return target_xy.astype(np.float32)


def _sample_weak_action(env: Any, rng: np.random.Generator, *, dist_constraint: float) -> tuple[np.ndarray, dict[str, list[float] | float]]:
    agent_xy = get_pusht_agent_pos(env)
    block_xy = get_pusht_block_pose(env)[:2]

    delta = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32) * ENV_ACTION_SCALE
    proposed_target = agent_xy + delta
    constrained_target = np.clip(
        proposed_target,
        block_xy - dist_constraint,
        block_xy + dist_constraint,
    ).astype(np.float32)
    env_action = _target_xy_to_env_action(env, agent_xy, constrained_target)
    env_action = _clip_action_to_space(env_action, env)
    return env_action, {
        "agent_xy": agent_xy.tolist(),
        "block_xy": block_xy.tolist(),
        "sampled_delta": delta.tolist(),
        "proposed_target_xy": proposed_target.tolist(),
        "constrained_target_xy": constrained_target.tolist(),
        "dist_constraint": float(dist_constraint),
    }


def _get_video_fps(env: Any, control_interval: int) -> int:
    metadata = getattr(env, "metadata", None) or getattr(getattr(env, "unwrapped", None), "metadata", None) or {}
    base_fps = metadata.get("render_fps")
    if base_fps is None:
        dt = getattr(getattr(env, "unwrapped", None), "dt", None)
        if dt is not None and dt > 1e-6:
            base_fps = float(round(1.0 / dt))
    if base_fps is None:
        base_fps = 10.0
    return max(1, int(round(float(base_fps) / float(control_interval))))


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).item())
    if "success" in info:
        return bool(np.asarray(info["success"]).item())
    return bool(terminated or reward >= 0.95)


def rollout_episode(args: argparse.Namespace, episode_idx: int) -> dict[str, Any]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.dist_constraint <= 0.0:
        raise ValueError("--dist-constraint must be > 0.")

    episode_seed = None if args.seed is None else args.seed + episode_idx
    rng = np.random.default_rng(episode_seed)
    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    try:
        _, _ = env.reset(seed=episode_seed)

        frames = [render_frame(env)]
        rewards: list[float] = []
        contact_counts: list[int] = []
        action_history: list[dict[str, list[float] | float]] = []
        action = None
        success = False
        terminated = False
        truncated = False
        steps_taken = 0
        control_updates = 0

        for step_idx in range(args.max_steps):
            if action is None or step_idx % args.control_interval == 0:
                action, action_meta = _sample_weak_action(
                    env,
                    rng,
                    dist_constraint=args.dist_constraint,
                )
                action_history.append(action_meta)
                control_updates += 1

            _, reward, terminated, truncated, info = env.step(action)
            success = success or _extract_success(terminated, float(reward), info)
            steps_taken = step_idx + 1

            if (step_idx + 1) % args.control_interval == 0 or terminated or truncated:
                frames.append(render_frame(env))
                rewards.append(float(reward))
                contact_counts.append(int(np.asarray(info.get("n_contacts", 0)).item()))

            if terminated or truncated:
                break

        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            video_path,
            frames,
            fps=_get_video_fps(env, args.control_interval),
            quality=8,
            macro_block_size=1,
        )

        final_block_pose = get_pusht_block_pose(env)
        final_agent_xy = get_pusht_agent_pos(env)
        goal_pose = np.asarray(getattr(env.unwrapped, "goal_pose", None), dtype=np.float32).tolist()
        return {
            "episode": episode_idx,
            "seed": episode_seed,
            "env_steps": steps_taken,
            "stored_steps": len(frames),
            "control_updates": control_updates,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "dist_constraint": float(args.dist_constraint),
            "goal_pose": goal_pose,
            "final_agent_xy": final_agent_xy.tolist(),
            "final_block_pose": final_block_pose.tolist(),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_contacts": float(np.mean(contact_counts)) if contact_counts else 0.0,
            "actions": action_history,
            "video_path": str(video_path),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [rollout_episode(args, episode_idx) for episode_idx in range(args.episodes)]
    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} env_steps={result['env_steps']} "
            f"stored_steps={result['stored_steps']} control_updates={result['control_updates']} "
            f"success={result['success']} mean_reward={result['mean_reward']:.3f} "
            f"mean_contacts={result['mean_contacts']:.3f} video={result['video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
