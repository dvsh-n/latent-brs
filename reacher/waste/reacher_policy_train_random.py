#!/usr/bin/env python3
"""Train a goal-conditioned DM Control Reacher policy with SAC.

Each episode samples:
- a random arm start configuration
- a random joint-angle goal configuration

The policy observes the current arm state together with the goal and is
rewarded for reducing joint-space distance to that goal.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
import torch as th
from dm_control import suite
from dm_env import TimeStep
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm


DEFAULT_OUTPUT_DIR = "reacher/models/reacher-dm-control-sac-random-goal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("easy", "hard"), default="hard")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints, logs, normalization stats, and metadata.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps across all workers.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=24,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes per evaluation pass.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N environment steps.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Save a checkpoint every N environment steps.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Episode time limit in seconds. dm_control default is 20.0.",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.05,
        help="Terminate when ||qpos - goal_qpos||_2 falls below this threshold.",
    )
    parser.add_argument(
        "--goal-bonus",
        type=float,
        default=5.0,
        help="Reward bonus added on successful goal completion.",
    )
    parser.add_argument(
        "--action-cost-weight",
        type=float,
        default=0.0,
        help="Quadratic control-effort penalty weight: reward -= weight * ||action||_2^2.",
    )
    parser.add_argument(
        "--action-rate-cost-weight",
        type=float,
        default=1e-2,
        help="Quadratic action-change penalty weight: reward -= weight * ||action_t - action_{t-1}||_2^2.",
    )
    parser.add_argument(
        "--velocity-cost-weight",
        type=float,
        default=1e-2,
        help="Quadratic joint-velocity penalty weight: reward -= weight * ||velocity||_2^2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SAC. Defaults to `cuda`.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=8,
        help="Gradient updates after each training trigger.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=8,
        help="Run one SAC training phase every N collected steps per worker batch.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--net-width",
        type=int,
        default=256,
        help="Hidden width for actor/critic MLP layers.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of hidden layers in actor/critic MLPs.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="SB3 logging interval in training iterations.",
    )
    return parser.parse_args()


def require_cuda(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device_arg


def spec_to_box(spec: Any) -> spaces.Box:
    minimum = np.asarray(spec.minimum, dtype=np.float32)
    maximum = np.asarray(spec.maximum, dtype=np.float32)
    return spaces.Box(low=minimum, high=maximum, dtype=np.float32)


class RandomGoalDmControlGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        domain_name: str,
        task_name: str,
        seed: int,
        time_limit: float,
        goal_threshold: float,
        goal_bonus: float,
        action_cost_weight: float,
        action_rate_cost_weight: float,
        velocity_cost_weight: float,
        terminate_on_success: bool = True,
    ) -> None:
        super().__init__()
        self.domain_name = domain_name
        self.task_name = task_name
        self.base_seed = int(seed)
        self.time_limit = float(time_limit)
        self.goal_threshold = float(goal_threshold)
        self.goal_bonus = float(goal_bonus)
        self.action_cost_weight = float(action_cost_weight)
        self.action_rate_cost_weight = float(action_rate_cost_weight)
        self.velocity_cost_weight = float(velocity_cost_weight)
        self.terminate_on_success = bool(terminate_on_success)
        self._episode_index = 0
        self._reset_counter = 0
        self._rng = np.random.default_rng(self.base_seed)
        self._env = self._build_env(self.base_seed)

        action_spec = self._env.action_spec()
        self.action_space = spec_to_box(action_spec)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        physics = self._env.physics
        self.qpos_dim = int(physics.model.nq)
        self.qvel_dim = int(physics.model.nv)
        self._qpos_lower, self._qpos_upper = self._infer_qpos_sampling_bounds()
        self._goal_qpos = np.zeros(self.qpos_dim, dtype=np.float32)

        obs_low = np.full((3 * self.qpos_dim + self.qvel_dim,), -np.inf, dtype=np.float32)
        obs_high = np.full_like(obs_low, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _build_env(self, seed: int) -> Any:
        return suite.load(
            self.domain_name,
            self.task_name,
            task_kwargs={"random": seed, "time_limit": self.time_limit},
            environment_kwargs={"flat_observation": True},
        )

    def _infer_qpos_sampling_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        model = self._env.physics.model
        joint_ranges = np.asarray(model.jnt_range[: self.qpos_dim], dtype=np.float32)
        joint_limited = np.asarray(model.jnt_limited[: self.qpos_dim], dtype=bool)
        lower = joint_ranges[:, 0]
        upper = joint_ranges[:, 1]
        valid_limited = joint_limited & np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        lower = np.where(valid_limited, lower, -np.pi).astype(np.float32)
        upper = np.where(valid_limited, upper, np.pi).astype(np.float32)
        return lower, upper

    def _sample_qpos(self) -> np.ndarray:
        return self._rng.uniform(self._qpos_lower, self._qpos_upper).astype(np.float32)

    def _sample_qvel(self) -> np.ndarray:
        return np.zeros(self.qvel_dim, dtype=np.float32)

    def _physics_obs(self) -> tuple[np.ndarray, np.ndarray]:
        physics = self._env.physics
        qpos = np.asarray(physics.data.qpos[: self.qpos_dim], dtype=np.float32).copy()
        qvel = np.asarray(physics.data.qvel[: self.qvel_dim], dtype=np.float32).copy()
        return qpos, qvel

    def _goal_distance(self, qpos: np.ndarray) -> float:
        return float(np.linalg.norm(qpos - self._goal_qpos))

    def _build_observation(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        goal_delta = self._goal_qpos - qpos
        return np.concatenate((qpos, qvel, self._goal_qpos, goal_delta), axis=0).astype(np.float32)

    def _reset_physics_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        physics = self._env.physics
        with physics.reset_context():
            physics.data.qpos[: self.qpos_dim] = qpos
            physics.data.qvel[: self.qvel_dim] = qvel

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
            self._episode_index = 0
            self._reset_counter = 0
            self._rng = np.random.default_rng(self.base_seed)
            self._env = self._build_env(self.base_seed)

        self._reset_counter += 1
        episode_seed = self.base_seed + self._reset_counter
        self._env = self._build_env(episode_seed)
        self._env.reset()

        self._goal_qpos = self._sample_qpos()
        start_qpos = self._sample_qpos()
        start_qvel = self._sample_qvel()
        self._reset_physics_state(start_qpos, start_qvel)

        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        qpos, qvel = self._physics_obs()
        obs = self._build_observation(qpos, qvel)
        info = {
            "goal_qpos": self._goal_qpos.copy(),
            "start_qpos": qpos.copy(),
            "goal_distance": self._goal_distance(qpos),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        time_step: TimeStep = self._env.step(clipped)

        qpos, qvel = self._physics_obs()
        obs = self._build_observation(qpos, qvel)
        goal_distance = self._goal_distance(qpos)
        action_cost = self.action_cost_weight * float(np.dot(clipped, clipped))
        action_delta = clipped - self._last_action
        action_rate_cost = self.action_rate_cost_weight * float(np.dot(action_delta, action_delta))
        velocity_cost = self.velocity_cost_weight * float(np.dot(qvel, qvel))
        reward = -goal_distance - action_cost - action_rate_cost - velocity_cost

        success = goal_distance <= self.goal_threshold
        if success:
            reward += self.goal_bonus

        terminated = success and self.terminate_on_success
        truncated = bool(time_step.last() and not success)
        info = {
            "goal_qpos": self._goal_qpos.copy(),
            "goal_distance": goal_distance,
            "success": success,
            "dm_reward": 0.0 if time_step.reward is None else float(time_step.reward),
            "action_cost": action_cost,
            "action_rate_cost": action_rate_cost,
            "velocity_cost": velocity_cost,
        }
        self._last_action = clipped
        if terminated or truncated:
            self._episode_index += 1
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._env.physics.render(camera_id=0)

    def close(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None:
            self._env = None


def make_env(
    *,
    task_name: str,
    seed: int,
    time_limit: float,
    goal_threshold: float,
    goal_bonus: float,
    action_cost_weight: float,
    action_rate_cost_weight: float,
    velocity_cost_weight: float,
) -> Callable[[], gym.Env]:
    def _factory() -> gym.Env:
        env = RandomGoalDmControlGymEnv(
            domain_name="reacher",
            task_name=task_name,
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

    return _factory


def build_vec_env(
    *,
    num_envs: int,
    task_name: str,
    seed: int,
    time_limit: float,
    goal_threshold: float,
    goal_bonus: float,
    action_cost_weight: float,
    action_rate_cost_weight: float,
    velocity_cost_weight: float,
    training: bool,
) -> VecNormalize:
    env_fns = [
        make_env(
            task_name=task_name,
            seed=seed + env_index,
            time_limit=time_limit,
            goal_threshold=goal_threshold,
            goal_bonus=goal_bonus,
            action_cost_weight=action_cost_weight,
            action_rate_cost_weight=action_rate_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
        )
        for env_index in range(num_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if num_envs == 1 else SubprocVecEnv(env_fns, start_method="spawn")
    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )


def write_metadata(args: argparse.Namespace, output_dir: Path, device: str) -> None:
    metadata = {
        "script": str(Path(__file__).resolve()),
        "task": args.task,
        "seed": args.seed,
        "time_limit": args.time_limit,
        "goal_threshold": args.goal_threshold,
        "goal_bonus": args.goal_bonus,
        "action_cost_weight": args.action_cost_weight,
        "action_rate_cost_weight": args.action_rate_cost_weight,
        "velocity_cost_weight": args.velocity_cost_weight,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "device": device,
        "torch_version": th.__version__,
        "cuda_available": th.cuda.is_available(),
        "cuda_device_name": th.cuda.get_device_name(0) if th.cuda.is_available() else None,
        "started_at_unix": time.time(),
    }
    (output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int) -> None:
        super().__init__()
        self.total_timesteps = int(total_timesteps)
        self._progress_bar: tqdm | None = None
        self._last_n = 0

    def _on_training_start(self) -> None:
        self._progress_bar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")
        self._last_n = 0

    def _on_step(self) -> bool:
        if self._progress_bar is None:
            return True
        current_n = min(int(self.num_timesteps), self.total_timesteps)
        delta = current_n - self._last_n
        if delta > 0:
            self._progress_bar.update(delta)
            self._last_n = current_n
        return True

    def _on_training_end(self) -> None:
        if self._progress_bar is not None:
            current_n = min(int(self.num_timesteps), self.total_timesteps)
            if current_n > self._last_n:
                self._progress_bar.update(current_n - self._last_n)
            self._progress_bar.close()
            self._progress_bar = None


def main() -> None:
    args = parse_args()
    device = require_cuda(args.device)

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.time_limit <= 0.0:
        raise ValueError("--time-limit must be positive")
    if args.goal_threshold <= 0.0:
        raise ValueError("--goal-threshold must be positive")
    if args.action_cost_weight < 0.0:
        raise ValueError("--action-cost-weight must be non-negative")
    if args.action_rate_cost_weight < 0.0:
        raise ValueError("--action-rate-cost-weight must be non-negative")
    if args.velocity_cost_weight < 0.0:
        raise ValueError("--velocity-cost-weight must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    write_metadata(args, output_dir, device)

    training_env = build_vec_env(
        num_envs=args.num_envs,
        task_name=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        goal_threshold=args.goal_threshold,
        goal_bonus=args.goal_bonus,
        action_cost_weight=args.action_cost_weight,
        action_rate_cost_weight=args.action_rate_cost_weight,
        velocity_cost_weight=args.velocity_cost_weight,
        training=True,
    )
    eval_env = build_vec_env(
        num_envs=1,
        task_name=args.task,
        seed=args.seed + 100_000,
        time_limit=args.time_limit,
        goal_threshold=args.goal_threshold,
        goal_bonus=args.goal_bonus,
        action_cost_weight=args.action_cost_weight,
        action_rate_cost_weight=args.action_rate_cost_weight,
        velocity_cost_weight=args.velocity_cost_weight,
        training=False,
    )

    policy_kwargs = {
        "net_arch": [args.net_width] * args.num_layers,
    }

    model = SAC(
        policy="MlpPolicy",
        env=training_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        device=device,
        seed=args.seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix="reacher_dm_sac_random_goal",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(1, args.eval_freq // args.num_envs),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    tqdm_callback = TqdmProgressCallback(total_timesteps=args.total_timesteps)
    callbacks = CallbackList([tqdm_callback, checkpoint_callback, eval_callback])

    print(
        json.dumps(
            {
                "task": args.task,
                "device": device,
                "num_envs": args.num_envs,
                "total_timesteps": args.total_timesteps,
                "batch_size": args.batch_size,
                "gradient_steps": args.gradient_steps,
                "time_limit_seconds": args.time_limit,
                "goal_threshold": args.goal_threshold,
                "goal_bonus": args.goal_bonus,
                "action_cost_weight": args.action_cost_weight,
                "action_rate_cost_weight": args.action_rate_cost_weight,
                "velocity_cost_weight": args.velocity_cost_weight,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
        )
    finally:
        model.save(str(output_dir / "final_model"))
        training_env.save(str(output_dir / "vecnormalize.pkl"))
        eval_env.save(str(output_dir / "eval_vecnormalize.pkl"))
        training_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
