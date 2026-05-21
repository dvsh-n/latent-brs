#!/usr/bin/env python3
"""Render rollouts from the random-goal DM Control Reacher SAC policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm.auto import tqdm

from reacher.train.reacher_policy_train_random import RandomGoalDmControlGymEnv

DEFAULT_OUTPUT_DIR = "reacher/models/reacher-dm-control-sac-random-goal"
DEFAULT_MODEL_PATH = "reacher/models/reacher-dm-control-sac-random-goal/best_model/best_model.zip"
DEFAULT_VECNORMALIZE_PATH = "reacher/models/reacher-dm-control-sac-random-goal/vecnormalize.pkl"
DEFAULT_OUTDIR = "reacher/eval/reacher_random_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained SAC zip file.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=DEFAULT_VECNORMALIZE_PATH,
        help="Path to VecNormalize statistics from training.",
    )
    parser.add_argument(
        "--task",
        choices=("easy", "hard"),
        default="hard",
        help="DM Control task variant used during training.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where MP4s will be written.",
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=20,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum control steps per episode.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Episode time limit in seconds.",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=None,
        help=(
            "Success threshold on ||qpos - goal_qpos||_2. Defaults to the value in "
            "train_config.json when available, otherwise 0.05."
        ),
    )
    parser.add_argument(
        "--goal-bonus",
        type=float,
        default=None,
        help=(
            "Reward bonus added on success. Defaults to the value in "
            "train_config.json when available, otherwise 5.0."
        ),
    )
    parser.add_argument(
        "--action-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic action penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--action-rate-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic action-change penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--velocity-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic velocity penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output video FPS. DM Control reacher runs at 50 Hz by default.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="ImageIO/ffmpeg quality for MP4 output.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy output.",
    )
    return parser.parse_args()


def require_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device_arg


def load_train_config(model_path: Path, vecnormalize_path: Path) -> dict[str, Any]:
    config_candidates = [
        model_path.parent.parent / "train_config.json",
        model_path.parent / "train_config.json",
        vecnormalize_path.parent / "train_config.json",
    ]
    for config_path in config_candidates:
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def resolve_vecnormalize_path(requested_path: Path, model_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    run_dir = model_path.parent.parent if model_path.parent.name == "best_model" else model_path.parent
    candidates = [
        run_dir / "vecnormalize.pkl",
        run_dir / "eval_vecnormalize.pkl",
    ]
    candidates.extend(sorted((run_dir / "checkpoints").glob("*vecnormalize*.pkl")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"VecNormalize stats not found: {requested_path}. "
            "Pass --vecnormalize-path for the run that produced this model."
        )

    best_path = max(existing, key=lambda path: path.stat().st_mtime)
    print(f"VecNormalize stats not found at {requested_path}; using {best_path}")
    return best_path


def make_eval_env(
    *,
    task: str,
    seed: int,
    time_limit: float,
    goal_threshold: float,
    goal_bonus: float,
    vecnormalize_path: Path,
    action_cost_weight: float = 0.0,
    action_rate_cost_weight: float = 0.0,
    velocity_cost_weight: float = 0.0,
) -> VecNormalize:
    def _factory() -> Monitor:
        env = RandomGoalDmControlGymEnv(
            domain_name="reacher",
            task_name=task,
            seed=seed,
            time_limit=time_limit,
            goal_threshold=goal_threshold,
            goal_bonus=goal_bonus,
            action_cost_weight=action_cost_weight,
            action_rate_cost_weight=action_rate_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
        )
        env.reset(seed=seed)
        return Monitor(env)

    base_env = DummyVecEnv([_factory])
    vec_env = VecNormalize.load(str(vecnormalize_path), base_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def get_render_env(vec_env: VecNormalize) -> RandomGoalDmControlGymEnv:
    monitor_env = vec_env.venv.envs[0]
    raw_env = getattr(monitor_env, "env", None)
    if raw_env is None or not isinstance(raw_env, RandomGoalDmControlGymEnv):
        raise TypeError("Expected VecNormalize -> DummyVecEnv -> Monitor -> RandomGoalDmControlGymEnv")
    return raw_env


def configure_offscreen_framebuffer(
    render_env: RandomGoalDmControlGymEnv,
    width: int,
    height: int,
) -> None:
    """Expand the MuJoCo offscreen framebuffer to fit the requested dimensions."""
    global_ = render_env._env.physics.model.vis.global_
    global_.offheight = max(height, int(global_.offheight))
    global_.offwidth = max(width, int(global_.offwidth))


def hide_target(render_env: RandomGoalDmControlGymEnv) -> None:
    target_geom_id = render_env._env.physics.model.name2id("target", "geom")
    render_env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def render_frame(
    render_env: RandomGoalDmControlGymEnv,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    # DM Control can rebuild physics on reset/auto-reset, so reapply the buffer size
    # immediately before every render call.
    configure_offscreen_framebuffer(render_env, width, height)
    hide_target(render_env)
    return render_env._env.physics.render(height=height, width=width, camera_id=0)


def render_start_goal_frames(
    render_env: RandomGoalDmControlGymEnv,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    start_qpos, start_qvel = render_env._physics_obs()
    start_frame = render_frame(render_env, width=width, height=height)

    goal_qpos = render_env._goal_qpos.copy()
    goal_qvel = np.zeros(render_env.qvel_dim, dtype=np.float32)
    render_env._reset_physics_state(goal_qpos, goal_qvel)
    goal_frame = render_frame(render_env, width=width, height=height)

    render_env._reset_physics_state(start_qpos, start_qvel)
    return start_frame, goal_frame


def parse_observation_state(
    render_env: RandomGoalDmControlGymEnv,
    observation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    qpos_dim = render_env.qpos_dim
    qvel_dim = render_env.qvel_dim
    qpos = obs[:qpos_dim].copy()
    qvel = obs[qpos_dim : qpos_dim + qvel_dim].copy()
    goal_qpos = obs[qpos_dim + qvel_dim : (2 * qpos_dim) + qvel_dim].copy()
    return qpos, qvel, goal_qpos


def render_terminal_frame(
    vec_env: VecNormalize,
    render_env: RandomGoalDmControlGymEnv,
    terminal_observation: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    raw_terminal_obs = vec_env.unnormalize_obs(np.asarray(terminal_observation))
    qpos, qvel, goal_qpos = parse_observation_state(render_env, raw_terminal_obs)
    render_env._goal_qpos = goal_qpos.astype(np.float32)
    render_env._reset_physics_state(qpos.astype(np.float32), qvel.astype(np.float32))
    return render_frame(render_env, width=width, height=height)


def render_episode_frames(
    *,
    model: SAC,
    vec_env: VecNormalize,
    render_env: RandomGoalDmControlGymEnv,
    episode_seed: int,
    deterministic: bool,
    max_steps: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, float, int, bool, float, np.ndarray, np.ndarray]:
    vec_env.seed(episode_seed)
    obs = vec_env.reset()
    start_frame, goal_frame = render_start_goal_frames(render_env, width=width, height=height)
    frames = [render_frame(render_env, width=width, height=height)]

    total_reward = 0.0
    num_steps = 0
    success = False
    final_goal_distance = float("nan")
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += float(rewards[0])
        num_steps += 1

        info = infos[0]
        success = bool(info.get("success", False))
        final_goal_distance = float(info.get("goal_distance", float("nan")))
        if bool(dones[0]):
            terminal_observation = info.get("terminal_observation")
            if terminal_observation is not None:
                frames.append(
                    render_terminal_frame(
                        vec_env,
                        render_env,
                        terminal_observation,
                        width=width,
                        height=height,
                    )
                )
            break
        frames.append(render_frame(render_env, width=width, height=height))

    return np.stack(frames, axis=0), total_reward, num_steps, success, final_goal_distance, start_frame, goal_frame


def main() -> None:
    args = parse_args()
    device = require_device(args.device)

    model_path = args.model_path.expanduser().resolve()
    vecnormalize_path = args.vecnormalize_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    vecnormalize_path = resolve_vecnormalize_path(vecnormalize_path, model_path)

    train_config = load_train_config(model_path, vecnormalize_path)
    goal_threshold = (
        float(train_config.get("goal_threshold", 0.05))
        if args.goal_threshold is None
        else float(args.goal_threshold)
    )
    goal_bonus = (
        float(train_config.get("goal_bonus", 5.0))
        if args.goal_bonus is None
        else float(args.goal_bonus)
    )
    action_cost_weight = (
        float(train_config.get("action_cost_weight", 0.0))
        if args.action_cost_weight is None
        else float(args.action_cost_weight)
    )
    action_rate_cost_weight = (
        float(train_config.get("action_rate_cost_weight", 0.0))
        if args.action_rate_cost_weight is None
        else float(args.action_rate_cost_weight)
    )
    velocity_cost_weight = (
        float(train_config.get("velocity_cost_weight", 0.0))
        if args.velocity_cost_weight is None
        else float(args.velocity_cost_weight)
    )

    if goal_threshold <= 0.0:
        raise ValueError("--goal-threshold must be positive")
    if goal_bonus < 0.0:
        raise ValueError("--goal-bonus must be non-negative")
    if action_cost_weight < 0.0:
        raise ValueError("--action-cost-weight must be non-negative")
    if action_rate_cost_weight < 0.0:
        raise ValueError("--action-rate-cost-weight must be non-negative")
    if velocity_cost_weight < 0.0:
        raise ValueError("--velocity-cost-weight must be non-negative")

    vec_env = make_eval_env(
        task=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        goal_threshold=goal_threshold,
        goal_bonus=goal_bonus,
        vecnormalize_path=vecnormalize_path,
        action_cost_weight=action_cost_weight,
        action_rate_cost_weight=action_rate_cost_weight,
        velocity_cost_weight=velocity_cost_weight,
    )
    render_env = get_render_env(vec_env)
    model = SAC.load(str(model_path), env=vec_env, device=device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []
    final_goal_distances: list[float] = []
    digits = max(2, len(str(args.videos - 1)))

    for episode_index in tqdm(range(args.videos), desc="Rendering", unit="video"):
        episode_seed = args.seed + episode_index
        (
            frames,
            total_reward,
            num_steps,
            success,
            final_goal_distance,
            start_frame,
            goal_frame,
        ) = render_episode_frames(
            model=model,
            vec_env=vec_env,
            render_env=render_env,
            episode_seed=episode_seed,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
        )
        output_path = outdir / f"reacher_random_{episode_index:0{digits}d}.mp4"
        start_png_path = outdir / f"reacher_random_{episode_index:0{digits}d}_start.png"
        goal_png_path = outdir / f"reacher_random_{episode_index:0{digits}d}_goal.png"
        imageio.mimwrite(output_path, frames, fps=args.fps, quality=args.quality)
        imageio.imwrite(start_png_path, start_frame)
        imageio.imwrite(goal_png_path, goal_frame)
        episode_rewards.append(total_reward)
        episode_lengths.append(num_steps)
        episode_successes.append(success)
        final_goal_distances.append(final_goal_distance)
        print(
            f"saved {output_path} "
            f"(start_png={start_png_path.name}, goal_png={goal_png_path.name}, "
            f"reward={total_reward:.3f}, steps={num_steps}, success={success}, "
            f"final_goal_distance={final_goal_distance:.4f}, frames={frames.shape[0]})"
        )

    summary = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path),
        "task": args.task,
        "videos": args.videos,
        "outdir": str(outdir),
        "device": device,
        "deterministic": not args.stochastic,
        "goal_threshold": goal_threshold,
        "goal_bonus": goal_bonus,
        "action_cost_weight": action_cost_weight,
        "action_rate_cost_weight": action_rate_cost_weight,
        "velocity_cost_weight": velocity_cost_weight,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "mean_final_goal_distance": float(np.mean(final_goal_distances)) if final_goal_distances else 0.0,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    print(json.dumps(summary, indent=2))
    vec_env.close()


if __name__ == "__main__":
    main()
