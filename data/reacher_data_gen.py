#!/usr/bin/env python3
"""Collect expert DM Control Reacher trajectories with the same rollout path as policy viz."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from eval.reacher_policy_viz import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VECNORMALIZE_PATH,
    configure_offscreen_framebuffer,
    get_render_env,
    make_eval_env,
    require_device,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "data" / "test_data"

CONTROL_FREQ_HZ = 50
STEPS_PER_EPISODE = 50
STATE_DIM = 6
ACTION_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", type=Path, default=DEFAULT_VECNORMALIZE_PATH)
    parser.add_argument("--task", choices=("easy", "hard"), default="hard")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--num-trajectories", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=252)
    parser.add_argument("--height", type=int, default=252)
    parser.add_argument("--fps", type=int, default=CONTROL_FREQ_HZ)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy output by default. Pass --no-deterministic for stochastic rollouts.",
    )
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return parser.parse_args()


def get_original_obs(env: object) -> np.ndarray:
    return np.asarray(env.get_original_obs()[0], dtype=np.float64)


def hide_target(render_env: object) -> None:
    target_geom_id = render_env._env.physics.model.name2id("target", "geom")
    render_env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def collect_trajectory(
    *,
    model: SAC,
    env: object,
    render_env: object,
    trajectory_seed: int,
    deterministic: bool,
    width: int,
    height: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    env.seed(trajectory_seed)
    obs = env.reset()
    hide_target(render_env)
    configure_offscreen_framebuffer(render_env, width, height)

    states = np.empty((STEPS_PER_EPISODE + 1, STATE_DIM), dtype=np.float64)
    actions = np.empty((STEPS_PER_EPISODE, ACTION_DIM), dtype=np.float64)
    states[0] = get_original_obs(env)
    frames = [render_env._env.physics.render(height=height, width=width, camera_id=0)]

    total_reward = 0.0
    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, _ = env.step(action)
        total_reward += float(rewards[0])
        actions[step_idx] = np.asarray(action[0], dtype=np.float64)
        states[step_idx + 1] = get_original_obs(env)
        frames.append(render_env._env.physics.render(height=height, width=width, camera_id=0))
        if bool(dones[0]):
            break

    if len(frames) != STEPS_PER_EPISODE + 1:
        raise RuntimeError(
            f"Trajectory for seed {trajectory_seed} produced {len(frames) - 1} steps; "
            f"expected exactly {STEPS_PER_EPISODE}."
        )

    return states, actions, np.stack(frames, axis=0), total_reward


def main() -> None:
    args = parse_args()
    device = require_device(args.device)

    model_path = args.model_path.expanduser().resolve()
    vecnormalize_path = args.vecnormalize_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(
        task=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        vecnormalize_path=vecnormalize_path,
    )
    render_env = get_render_env(env)
    model = SAC.load(str(model_path), env=env, device=device)

    states_np = np.empty((args.num_trajectories, STEPS_PER_EPISODE + 1, STATE_DIM), dtype=np.float64)
    actions_np = np.empty((args.num_trajectories, STEPS_PER_EPISODE, ACTION_DIM), dtype=np.float64)
    rewards_np = np.empty((args.num_trajectories,), dtype=np.float64)

    for traj_idx in tqdm(range(args.num_trajectories), desc="Collecting trajectories", unit="traj"):
        trajectory_seed = args.seed + traj_idx
        video_path = video_dir / f"trajectory_{traj_idx:07d}.mp4"
        states, actions, frames, total_reward = collect_trajectory(
            model=model,
            env=env,
            render_env=render_env,
            trajectory_seed=trajectory_seed,
            deterministic=args.deterministic,
            width=args.width,
            height=args.height,
            max_steps=STEPS_PER_EPISODE,
        )
        imageio.mimwrite(
            video_path,
            frames,
            fps=args.fps,
            quality=args.quality,
            macro_block_size=1,
        )
        states_np[traj_idx] = states
        actions_np[traj_idx] = actions
        rewards_np[traj_idx] = total_reward

    states_tensor = torch.from_numpy(states_np)
    actions_tensor = torch.from_numpy(actions_np)
    rewards_tensor = torch.from_numpy(rewards_np)

    dataset = {
        "states": states_tensor,
        "actions": actions_tensor,
        "rewards": rewards_tensor,
        "num_trajectories": args.num_trajectories,
        "steps_per_episode": STEPS_PER_EPISODE,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "time_limit": args.time_limit,
        "state_dim": int(states_tensor.shape[-1]),
        "action_dim": int(actions_tensor.shape[-1]),
        "state_dtype": str(states_tensor.dtype),
        "action_dtype": str(actions_tensor.dtype),
        "state_keys": ["position(2)", "to_target(2)", "velocity(2)"],
        "task": args.task,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path),
        "video_dir": str(video_dir),
        "video_resolution": [args.height, args.width],
        "video_fps": args.fps,
        "video_quality": args.quality,
        "mean_reward": float(np.mean(rewards_np)) if len(rewards_np) else 0.0,
    }

    output_path = outdir / "expert_data.pt"
    torch.save(dataset, output_path)
    print(f"Saved {args.num_trajectories} trajectories to {output_path}")
    print(f"  states:  {tuple(states_tensor.shape)} {states_tensor.dtype}")
    print(f"  actions: {tuple(actions_tensor.shape)} {actions_tensor.dtype}")
    print(f"  rewards: {tuple(rewards_tensor.shape)} {rewards_tensor.dtype}")
    print(f"  videos:  {video_dir}")
    print(json.dumps({"mean_reward": dataset["mean_reward"], "time_limit": args.time_limit}, indent=2))

    env.close()


if __name__ == "__main__":
    main()
