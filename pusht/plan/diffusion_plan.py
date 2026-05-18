#!/usr/bin/env python3
"""Roll out the LeRobot PushT diffusion policy in this repo's PushT env."""

from __future__ import annotations

import argparse
import json
from typing import Any
from pathlib import Path

import numpy as np
from tqdm import trange

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_agent_pos,
    get_pusht_block_pose,
    make_pusht_env,
    reset_pusht_env_to_state,
)
from pusht.shared.utils import load_expert_policy_bundle, pusht_observation_to_policy_batch, select_expert_action

DEFAULT_MODEL_DIR = Path("pusht/models")
DEFAULT_OUT_DIR = Path("pusht/plan/diffusion_plan")
PUSHT_WALL_MIN = 5.0
PUSHT_WALL_MAX = 506.0
PUSHT_WALL_RADIUS = 2.0
PUSHT_AGENT_RADIUS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--video-name", default="diffusion_plan.mp4")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-noise-std", type=float, default=8.5)
    parser.add_argument(
        "--edge-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the T initialization from env.reset() but respawn the pusher on a random legal edge.",
    )
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Query the policy every N env steps and hold the last action in between.",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def maybe_display(frame: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install opencv-python or run without --display.") from exc

    cv2.imshow("PushT diffusion plan", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to save videos, or pass --no-video.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def _policy_view_frame(bundle, observation: dict[str, Any], env: Any) -> np.ndarray:
    batch = pusht_observation_to_policy_batch(
        observation,
        env=env,
        image_shape=tuple(bundle.policy.config.input_features["observation.image"].shape),
    )
    image = batch["observation.image"][0].permute(1, 2, 0).cpu().numpy()
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def _clip_action_to_space(action: np.ndarray, env: Any) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != action.shape or low.shape != action.shape:
        return np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def _pusht_agent_bounds() -> tuple[np.ndarray, np.ndarray]:
    min_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    low = np.full((2,), min_coord, dtype=np.float32)
    high = np.full((2,), max_coord, dtype=np.float32)
    return low, high


def _sample_agent_pos_on_edge(rng: np.random.Generator) -> np.ndarray:
    low, high = _pusht_agent_bounds()
    edge = int(rng.integers(4))
    coord = float(rng.uniform(float(low[0]), float(high[0])))
    if edge == 0:
        return np.asarray([low[0], coord], dtype=np.float32)
    if edge == 1:
        return np.asarray([high[0], coord], dtype=np.float32)
    if edge == 2:
        return np.asarray([coord, low[1]], dtype=np.float32)
    return np.asarray([coord, high[1]], dtype=np.float32)


def _refresh_observation(
    observation: dict[str, Any],
    *,
    pixels: np.ndarray,
    agent_pos: np.ndarray,
) -> dict[str, Any]:
    updated = dict(observation)
    if "pixels" in updated:
        updated["pixels"] = pixels
    if "image" in updated:
        updated["image"] = pixels
    if "agent_pos" in updated:
        updated["agent_pos"] = agent_pos.astype(np.float32, copy=True)
    if "proprio" in updated:
        proprio = np.asarray(updated["proprio"], dtype=np.float32).copy()
        if proprio.shape[0] < 2:
            raise ValueError("Expected PushT proprio observations with at least 2 entries for agent xy.")
        proprio[:2] = agent_pos
        updated["proprio"] = proprio
    if "state" in updated:
        state = np.asarray(updated["state"], dtype=np.float32).copy()
        if state.shape[0] < 2:
            raise ValueError("Expected PushT state observations with at least 2 entries for agent xy.")
        state[:2] = agent_pos
        updated["state"] = state
    return updated


def _maybe_edge_sample_observation(
    env: Any,
    observation: dict[str, Any],
    *,
    enabled: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    initial_agent_pos = get_pusht_agent_pos(env).astype(np.float32, copy=True)
    if not enabled:
        return observation, initial_agent_pos
    if not isinstance(observation, dict):
        raise ValueError("--edge-sample requires dict observations containing agent position fields.")

    block_pose = get_pusht_block_pose(env)
    sampled_agent_pos = _sample_agent_pos_on_edge(rng)
    state = np.asarray(
        [sampled_agent_pos[0], sampled_agent_pos[1], block_pose[0], block_pose[1], block_pose[2], 0.0, 0.0],
        dtype=np.float64,
    )
    pixels = reset_pusht_env_to_state(env, state)
    return _refresh_observation(observation, pixels=pixels, agent_pos=sampled_agent_pos), sampled_agent_pos


def extract_goal_pose(env) -> list[float] | None:
    goal_pose = getattr(env.unwrapped, "goal_pose", None)
    if goal_pose is None:
        return None
    return np.asarray(goal_pose, dtype=np.float32).tolist()


def rollout_episode(args: argparse.Namespace, bundle, episode_idx: int) -> dict[str, object]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.control_noise_std < 0.0:
        raise ValueError("--control-noise-std must be >= 0.")

    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    bundle.policy.reset()
    episode_seed = None if args.seed is None else args.seed + episode_idx
    observation, info = env.reset(seed=episode_seed)
    rng = np.random.default_rng(episode_seed)
    observation, initial_agent_pos = _maybe_edge_sample_observation(
        env,
        observation,
        enabled=args.edge_sample,
        rng=rng,
    )

    frames = [_policy_view_frame(bundle, observation, env)]
    total_reward = 0.0
    success = False
    steps = 0
    control_updates = 0
    action = None

    for step_idx in trange(args.max_steps, desc=f"episode {episode_idx}", unit="step"):
        if action is None or step_idx % args.control_interval == 0:
            action = select_expert_action(bundle, observation, env=env, action_mode=args.action_mode)
            if args.control_noise:
                noise = rng.normal(loc=0.0, scale=args.control_noise_std, size=action.shape).astype(np.float32)
                action = _clip_action_to_space(action + noise, env)
            control_updates += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = bool(terminated or info.get("is_success", False) or info.get("success", False))
        steps += 1

        if steps % args.control_interval == 0 or terminated or truncated:
            frame = _policy_view_frame(bundle, next_observation, env)
            frames.append(frame)
            maybe_display(frame, args.display)

        observation = next_observation
        if terminated or truncated:
            break

    final_block_pose = get_pusht_block_pose(env).tolist()
    final_agent_pos = get_pusht_agent_pos(env).tolist()
    goal_pose = extract_goal_pose(env)
    env.close()
    stored_steps = len(frames)

    video_path = None
    if not args.no_video:
        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.out_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        save_video(video_path, frames, max(1, int(args.fps)))

    return {
        "episode": episode_idx,
        "seed": episode_seed,
        "env_steps": steps,
        "stored_steps": stored_steps,
        "control_updates": control_updates,
        "total_reward": total_reward,
        "success": success,
        "control_noise": bool(args.control_noise),
        "control_noise_std": float(args.control_noise_std),
        "goal_pose": goal_pose,
        "initial_agent_pos": initial_agent_pos.tolist(),
        "final_block_pose": final_block_pose,
        "final_agent_pos": final_agent_pos,
        "edge_sample": bool(args.edge_sample),
        "video_path": str(video_path) if video_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading diffusion policy from: {args.model_dir}")
    bundle = load_expert_policy_bundle(args.model_dir, device=args.device)
    results = [rollout_episode(args, bundle, episode_idx) for episode_idx in range(args.episodes)]

    metrics_path = args.out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} stored_steps={result['stored_steps']} "
            f"env_steps={result['env_steps']} control_updates={result['control_updates']} "
            f"reward={result['total_reward']:.3f} "
            f"success={result['success']} video={result['video_path']}"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
