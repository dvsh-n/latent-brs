#!/usr/bin/env python
"""Roll out and visualize the trained PushT diffusion expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import trange

from pusht.shared.utils import (
    DEFAULT_EXPERT_MODEL_DIR,
    DEFAULT_PUSHT_ENV_ID,
    load_expert_policy_bundle,
    make_pusht_env,
    render_frame,
    select_expert_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_EXPERT_MODEL_DIR)
    parser.add_argument("--run-dir", type=Path, default=Path("pusht/runs/pusht_diffusion_expert"))
    parser.add_argument("--video-name", default="expert_viz.mp4")
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--obs-type", default="pixels_agent_pos")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--display", action="store_true", help="Show frames live with OpenCV while rolling out.")
    return parser.parse_args()


def maybe_display(frame: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install opencv-python or run without --display.") from exc

    cv2.imshow("PushT expert", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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


def rollout_episode(args: argparse.Namespace, bundle, episode_idx: int) -> dict:
    env = make_pusht_env(
        args.env_id,
        obs_type=args.obs_type,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    bundle.policy.reset()
    observation, info = env.reset(seed=args.seed + episode_idx)

    frames = [render_frame(env)]
    total_reward = 0.0
    success = False
    steps = 0

    for _ in trange(args.max_steps, desc=f"episode {episode_idx}", unit="step"):
        action = select_expert_action(bundle, observation, env=env, action_mode=args.action_mode)
        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        success = bool(terminated or info.get("is_success", False) or info.get("success", False))
        steps += 1

        frame = render_frame(env)
        frames.append(frame)
        maybe_display(frame, args.display)

        observation = next_observation
        if terminated or truncated:
            break

    env.close()

    video_path = None
    if not args.no_video:
        suffix = "" if args.episodes == 1 else f"_episode_{episode_idx:03d}"
        video_path = args.run_dir / f"{Path(args.video_name).stem}{suffix}{Path(args.video_name).suffix}"
        save_video(video_path, frames, args.fps)

    return {
        "episode": episode_idx,
        "seed": args.seed + episode_idx,
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "video_path": str(video_path) if video_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading expert policy from: {args.model_dir}")
    bundle = load_expert_policy_bundle(args.model_dir, device=args.device)

    results = [rollout_episode(args, bundle, episode_idx) for episode_idx in range(args.episodes)]
    metrics_path = args.run_dir / "expert_viz_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    for result in results:
        print(
            f"episode={result['episode']} steps={result['steps']} "
            f"reward={result['total_reward']:.3f} success={result['success']} "
            f"video={result['video_path']}"
        )
    print(f"Saved rollout metrics to {metrics_path}")


if __name__ == "__main__":
    main()
