#!/usr/bin/env python3
"""Roll out the LeRobot PushT diffusion policy in this repo's PushT env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import trange

from pusht.shared.pusht_env import DEFAULT_PUSHT_ENV_ID, get_pusht_agent_pos, get_pusht_block_pose, make_pusht_env
from pusht.shared.utils import load_expert_policy_bundle, render_frame, select_expert_action

DEFAULT_MODEL_DIR = Path("pusht/models")
DEFAULT_OUT_DIR = Path("pusht/plan/diffusion_plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video-name", default="diffusion_plan.mp4")
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
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


def extract_goal_pose(env) -> list[float] | None:
    goal_pose = getattr(env.unwrapped, "goal_pose", None)
    if goal_pose is None:
        return None
    return np.asarray(goal_pose, dtype=np.float32).tolist()


def rollout_episode(args: argparse.Namespace, bundle, episode_idx: int) -> dict[str, object]:
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")

    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    bundle.policy.reset()
    observation, info = env.reset()

    frames = [render_frame(env)]
    total_reward = 0.0
    success = False
    steps = 0
    control_updates = 0
    action = None

    for step_idx in trange(args.max_steps, desc=f"episode {episode_idx}", unit="step"):
        if action is None or step_idx % args.control_interval == 0:
            action = select_expert_action(bundle, observation, env=env, action_mode=args.action_mode)
            control_updates += 1

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = bool(terminated or info.get("is_success", False) or info.get("success", False))
        steps += 1

        if steps % args.control_interval == 0 or terminated or truncated:
            frame = render_frame(env)
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
        save_video(video_path, frames, args.fps)

    return {
        "episode": episode_idx,
        "env_steps": steps,
        "stored_steps": stored_steps,
        "control_updates": control_updates,
        "total_reward": total_reward,
        "success": success,
        "goal_pose": goal_pose,
        "final_block_pose": final_block_pose,
        "final_agent_pos": final_agent_pos,
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
