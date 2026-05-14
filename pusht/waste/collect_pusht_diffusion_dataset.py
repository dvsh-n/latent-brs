#!/usr/bin/env python

"""Collect fixed-goal PushT rollouts from a diffusion policy.

The output is an HDF5 dataset with exactly these root datasets:

    ep_len      (num_episodes,) int64
    ep_offset   (num_episodes,) int64
    pixels      (num_frames, H, W, 3) uint8 RGB
    action      (num_frames, 2) float32
    state       (num_frames, 7) float32
    proprio     (num_frames, 4) float32
    episode_idx (num_frames,) int64
    step_idx    (num_frames,) int64

Field convention:
    state   = [block_x, block_y, cos(block_theta), sin(block_theta), goal_x, goal_y, goal_theta]
    proprio = [agent_x, agent_y, previous_action_x, previous_action_y]

PushT's default green T region is used as the one fixed goal pose. Random starts
come from env.reset(seed=...).
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import logging
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from pusht.shared.pusht_env import DEFAULT_PUSHT_ENV_ID, make_pusht_env

from lerobot.configs import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import PushtEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor.pipeline import ProcessorMigrationError
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PushT random-start, fixed-green-goal rollouts into an HDF5 dataset."
    )
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        default="pretrained/diffusion_pusht_migrated",
        help="Local migrated LeRobot policy path or Hub id.",
    )
    parser.add_argument(
        "--out",
        default="outputs/datasets/pusht_diffusion_fixed_goal.h5",
        help="Output .h5 path.",
    )
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to save.")
    parser.add_argument("--max-attempts", type=int, default=1000, help="Maximum rollout attempts.")
    parser.add_argument("--start-seed", type=int, default=0, help="First PushT reset seed.")
    parser.add_argument("--seed", type=int, default=1000, help="Global random seed.")
    parser.add_argument("--episode-length", type=int, default=300, help="Max steps per rollout.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of PushT envs to roll out in parallel with batched policy inference.",
    )
    parser.add_argument(
        "--async-envs",
        action="store_true",
        help="Run PushT envs in subprocesses. Faster for many envs, but heavier to start.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override diffusion denoising steps. Try 10-20 for faster collection.",
    )
    parser.add_argument(
        "--h5-compression",
        choices=["none", "lzf", "gzip"],
        default="none",
        help="HDF5 compression for pixels. 'none' is fastest; 'lzf' is a good compromise.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=32,
        help="Maximum successful episodes to flatten and append to HDF5 at once. Lower uses less RAM.",
    )
    parser.add_argument(
        "--batch-timeout-s",
        type=int,
        default=180,
        help="Skip and recreate envs if one vector rollout batch takes longer than this. Use 0 to disable.",
    )
    parser.add_argument("--device", default="cuda", help="Policy inference device.")
    parser.add_argument("--image-height", type=int, default=96, help="PushT pixel observation height.")
    parser.add_argument("--image-width", type=int, default=96, help="PushT pixel observation width.")
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Save failed rollouts too. Default saves only successful goal-reaching episodes.",
    )
    return parser.parse_args()


class BatchTimeoutError(TimeoutError):
    pass


@contextmanager
def batch_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise BatchTimeoutError(f"Vector rollout batch exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _as_xy(value: Any) -> np.ndarray:
    if hasattr(value, "x") and hasattr(value, "y"):
        return np.asarray([value.x, value.y], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected an xy-like value, got {value!r}")
    return arr[:2]


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got {value!r}")
    return float(arr[0])


def _extract_agent_xy(raw_observation: Any, unwrapped_env: gym.Env | None) -> np.ndarray:
    if isinstance(raw_observation, dict) and "agent_pos" in raw_observation:
        return np.asarray(raw_observation["agent_pos"], dtype=np.float32).reshape(-1)[:2]
    if unwrapped_env is not None and hasattr(unwrapped_env, "agent") and hasattr(unwrapped_env.agent, "position"):
        return _as_xy(unwrapped_env.agent.position)
    raise AttributeError("Could not extract PushT agent position.")


def _extract_block_pose(unwrapped_env: gym.Env) -> tuple[float, float, float]:
    if hasattr(unwrapped_env, "block") and hasattr(unwrapped_env.block, "position"):
        block_xy = _as_xy(unwrapped_env.block.position)
        block_theta = _scalar(getattr(unwrapped_env.block, "angle"))
        return float(block_xy[0]), float(block_xy[1]), block_theta

    if hasattr(unwrapped_env, "get_state"):
        state = np.asarray(unwrapped_env.get_state(), dtype=np.float32).reshape(-1)
        if state.size >= 5:
            return float(state[2]), float(state[3]), float(state[4])

    raise AttributeError("Could not extract PushT block pose from the environment.")


def _extract_goal_pose(unwrapped_env: gym.Env) -> tuple[float, float, float]:
    for name in ("goal_pose", "_goal_pose"):
        if hasattr(unwrapped_env, name):
            goal = np.asarray(getattr(unwrapped_env, name), dtype=np.float32).reshape(-1)
            if goal.size >= 3:
                return float(goal[0]), float(goal[1]), float(goal[2])
            if goal.size >= 2:
                return float(goal[0]), float(goal[1]), 0.0

    if hasattr(unwrapped_env, "get_goal_pose_body"):
        body = unwrapped_env.get_goal_pose_body()
        if hasattr(body, "position"):
            goal_xy = _as_xy(body.position)
            goal_theta = _scalar(getattr(body, "angle", 0.0))
            return float(goal_xy[0]), float(goal_xy[1]), goal_theta

    raise AttributeError("Could not extract PushT fixed goal pose from the environment.")


class PushTPrivilegedObsWrapper(gym.Wrapper):
    """Adds simulator pose fields to observations for fast HDF5 logging."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "block_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _augment(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        observation["block_pose"] = np.asarray(_extract_block_pose(self.unwrapped), dtype=np.float32)
        observation["goal_pose"] = np.asarray(_extract_goal_pose(self.unwrapped), dtype=np.float32)
        return observation

    def reset(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info


def _extract_pixels(raw_observation: dict[str, Any]) -> np.ndarray:
    if "pixels" not in raw_observation:
        raise KeyError("Expected PushT observation to contain 'pixels'.")
    pixels = raw_observation["pixels"]
    if isinstance(pixels, dict):
        if len(pixels) != 1:
            raise ValueError(f"Expected one PushT pixel camera, got {list(pixels)}")
        pixels = next(iter(pixels.values()))

    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB pixels, got shape {pixels.shape}")
    return pixels


def _make_state(raw_observation: dict[str, Any], unwrapped_env: gym.Env | None) -> np.ndarray:
    if "block_pose" in raw_observation:
        block_x, block_y, block_theta = np.asarray(raw_observation["block_pose"]).reshape(-1)[:3]
    else:
        if unwrapped_env is None:
            raise AttributeError("Missing block_pose observation and no local env is available.")
        block_x, block_y, block_theta = _extract_block_pose(unwrapped_env)
    if "goal_pose" in raw_observation:
        goal_x, goal_y, goal_theta = np.asarray(raw_observation["goal_pose"]).reshape(-1)[:3]
    else:
        if unwrapped_env is None:
            raise AttributeError("Missing goal_pose observation and no local env is available.")
        goal_x, goal_y, goal_theta = _extract_goal_pose(unwrapped_env)
    return np.asarray(
        [
            block_x,
            block_y,
            np.cos(block_theta),
            np.sin(block_theta),
            goal_x,
            goal_y,
            goal_theta,
        ],
        dtype=np.float32,
    )


def _make_proprio(
    raw_observation: dict[str, Any],
    unwrapped_env: gym.Env | None,
    previous_action: np.ndarray,
) -> np.ndarray:
    agent_xy = _extract_agent_xy(raw_observation, unwrapped_env)
    return np.asarray(
        [agent_xy[0], agent_xy[1], previous_action[0], previous_action[1]],
        dtype=np.float32,
    )


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).item())
    return bool(terminated or reward >= 0.95)


def _extract_vector_success(
    terminated: np.ndarray,
    rewards: np.ndarray,
    info: dict[str, Any],
    env_index: int,
) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            success = final_info["is_success"]
            if hasattr(success, "tolist"):
                success = success.tolist()
            if isinstance(success, list):
                return bool(success[env_index])
            return bool(success)
        if isinstance(final_info, (list, tuple)) and final_info[env_index] is not None:
            return bool(final_info[env_index].get("is_success", False))

    if "is_success" in info:
        success = info["is_success"]
        if hasattr(success, "tolist"):
            success = success.tolist()
        if isinstance(success, list):
            return bool(success[env_index])
        return bool(success)

    return bool(terminated[env_index] or rewards[env_index] >= 0.95)


def _make_env(args: argparse.Namespace) -> gym.Env:
    env = make_pusht_env(
        DEFAULT_PUSHT_ENV_ID,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=args.image_width,
        observation_height=args.image_height,
        max_episode_steps=args.episode_length,
        visualization_width=args.image_width,
        visualization_height=args.image_height,
    )
    return PushTPrivilegedObsWrapper(env)


def _make_vector_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    env_fns = [lambda: _make_env(args) for _ in range(args.num_envs)]
    env_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    try:
        from gymnasium.vector import AutoresetMode

        return env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    except (ImportError, TypeError):
        return env_cls(env_fns)


def _load_policy_and_processors(args: argparse.Namespace, env_cfg: PushtEnv):
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path) if Path(args.policy_path).exists() else args.policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = False
    if args.num_inference_steps is not None:
        policy_cfg.num_inference_steps = args.num_inference_steps

    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    policy.eval()

    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_cfg.pretrained_path),
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
    except ProcessorMigrationError:
        raise
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "The policy is missing policy_preprocessor.json / policy_postprocessor.json. "
            "If this is lerobot/diffusion_pusht, migrate the downloaded model first:\n"
            f"  uv run python src/lerobot/processor/migrate_policy_normalization.py "
            f"--pretrained-path {args.policy_path} "
            f"--output-dir {args.policy_path}_migrated\n"
        ) from exc

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _select_action(
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    raw_observation: dict[str, Any],
) -> np.ndarray:
    policy_observation = preprocess_observation(raw_observation)
    policy_observation = env_preprocessor(policy_observation)
    policy_observation = preprocessor(policy_observation)

    with torch.inference_mode():
        action = policy.select_action(policy_observation)

    action = postprocessor(action)
    action = env_postprocessor({ACTION: action})[ACTION]
    action = action.cpu().numpy()
    if action.shape != (1, 2):
        raise ValueError(f"Expected policy action shape (1, 2), got {action.shape}")
    return action[0].astype(np.float32)


def _select_actions(
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    env_postprocessor,
    raw_observations: dict[str, Any],
) -> np.ndarray:
    policy_observations = preprocess_observation(raw_observations)
    policy_observations = env_preprocessor(policy_observations)
    policy_observations = preprocessor(policy_observations)

    with torch.inference_mode():
        actions = policy.select_action(policy_observations)

    actions = postprocessor(actions)
    actions = env_postprocessor({ACTION: actions})[ACTION]
    actions = actions.cpu().numpy().astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected policy action shape (num_envs, 2), got {actions.shape}")
    return actions


def _slice_observation(raw_observations: dict[str, Any], env_index: int) -> dict[str, Any]:
    obs = {}
    for key, value in raw_observations.items():
        if isinstance(value, dict):
            obs[key] = {subkey: subvalue[env_index] for subkey, subvalue in value.items()}
        else:
            obs[key] = value[env_index]
    return obs


def _stack_or_empty(values: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if values:
        return np.stack(values).astype(dtype, copy=False)
    return np.empty(shape, dtype=dtype)


class H5EpisodeWriter:
    def __init__(self, out_path: Path, compression: str):
        if compression == "none":
            self.pixel_compression = None
        else:
            self.pixel_compression = compression

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.num_episodes = 0
        self.num_frames = 0

        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to write .h5 datasets. Install it in this environment, for example:\n"
                "  uv pip install h5py"
            ) from exc

        self.h5 = h5py.File(out_path, "w")
        self.h5.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("action", shape=(0, 2), maxshape=(None, 2), dtype=np.float32, chunks=True)
        self.h5.create_dataset("state", shape=(0, 7), maxshape=(None, 7), dtype=np.float32, chunks=True)
        self.h5.create_dataset("proprio", shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=True)
        self.h5.create_dataset("episode_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("step_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.attrs["state_convention"] = (
            "[block_x, block_y, cos(block_theta), sin(block_theta), goal_x, goal_y, goal_theta]"
        )
        self.h5.attrs["proprio_convention"] = "[agent_x, agent_y, previous_action_x, previous_action_y]"
        self.h5.attrs["goal"] = "fixed PushT green T region"

    def _ensure_pixels_dataset(self, image_shape: tuple[int, int, int]) -> None:
        if "pixels" in self.h5:
            return
        self.h5.create_dataset(
            "pixels",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            dtype=np.uint8,
            chunks=(1, *image_shape),
            compression=self.pixel_compression,
            shuffle=self.pixel_compression is not None,
        )

    def append_episodes(
        self,
        episodes: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]],
    ) -> None:
        episodes = [episode for episode in episodes if len(episode[1]) > 0]
        if not episodes:
            return

        ep_lengths = np.asarray([len(ep_actions) for _, ep_actions, _, _ in episodes], dtype=np.int64)
        flat_pixels = [frame for ep_pixels, _, _, _ in episodes for frame in ep_pixels]
        flat_actions = [frame for _, ep_actions, _, _ in episodes for frame in ep_actions]
        flat_states = [frame for _, _, ep_states, _ in episodes for frame in ep_states]
        flat_proprios = [frame for _, _, _, ep_proprios in episodes for frame in ep_proprios]

        pixel_array = _stack_or_empty(flat_pixels, (0, *flat_pixels[0].shape), np.uint8)
        action_array = _stack_or_empty(flat_actions, (0, 2), np.float32)
        state_array = _stack_or_empty(flat_states, (0, 7), np.float32)
        proprio_array = _stack_or_empty(flat_proprios, (0, 4), np.float32)

        num_new_episodes = len(episodes)
        num_new_frames = int(ep_lengths.sum())
        episode_idx_array = np.repeat(
            np.arange(self.num_episodes, self.num_episodes + num_new_episodes, dtype=np.int64),
            ep_lengths,
        )
        step_idx_array = np.concatenate([np.arange(length, dtype=np.int64) for length in ep_lengths])
        ep_offset_array = self.num_frames + np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(ep_lengths[:-1], dtype=np.int64)]
        )

        self._ensure_pixels_dataset(pixel_array.shape[1:])

        frame_start = self.num_frames
        frame_end = frame_start + num_new_frames
        ep_start = self.num_episodes
        ep_end = ep_start + num_new_episodes

        self.h5["ep_len"].resize((ep_end,))
        self.h5["ep_offset"].resize((ep_end,))
        self.h5["ep_len"][ep_start:ep_end] = ep_lengths
        self.h5["ep_offset"][ep_start:ep_end] = ep_offset_array

        for name, array in {
            "pixels": pixel_array,
            "action": action_array,
            "state": state_array,
            "proprio": proprio_array,
            "episode_idx": episode_idx_array,
            "step_idx": step_idx_array,
        }.items():
            dataset = self.h5[name]
            dataset.resize((frame_end, *dataset.shape[1:]))
            dataset[frame_start:frame_end] = array

        self.num_episodes = ep_end
        self.num_frames = frame_end

    def append_episode(
        self,
        ep_pixels: list[np.ndarray],
        ep_actions: list[np.ndarray],
        ep_states: list[np.ndarray],
        ep_proprios: list[np.ndarray],
    ) -> None:
        episode_len = len(ep_actions)
        if episode_len == 0:
            return

        pixel_array = _stack_or_empty(ep_pixels, (0, *ep_pixels[0].shape), np.uint8)
        action_array = _stack_or_empty(ep_actions, (0, 2), np.float32)
        state_array = _stack_or_empty(ep_states, (0, 7), np.float32)
        proprio_array = _stack_or_empty(ep_proprios, (0, 4), np.float32)
        episode_idx_array = np.full(episode_len, self.num_episodes, dtype=np.int64)
        step_idx_array = np.arange(episode_len, dtype=np.int64)

        self._ensure_pixels_dataset(pixel_array.shape[1:])

        frame_start = self.num_frames
        frame_end = frame_start + episode_len
        ep_end = self.num_episodes + 1

        self.h5["ep_len"].resize((ep_end,))
        self.h5["ep_offset"].resize((ep_end,))
        self.h5["ep_len"][self.num_episodes] = episode_len
        self.h5["ep_offset"][self.num_episodes] = frame_start

        for name, array in {
            "pixels": pixel_array,
            "action": action_array,
            "state": state_array,
            "proprio": proprio_array,
            "episode_idx": episode_idx_array,
            "step_idx": step_idx_array,
        }.items():
            dataset = self.h5[name]
            dataset.resize((frame_end, *dataset.shape[1:]))
            dataset[frame_start:frame_end] = array

        self.num_episodes = ep_end
        self.num_frames = frame_end

    def close(self) -> None:
        self.h5.attrs["num_episodes"] = self.num_episodes
        self.h5.attrs["num_frames"] = self.num_frames
        if "pixels" not in self.h5:
            self.h5.create_dataset(
                "pixels",
                shape=(0, 0, 0, 3),
                maxshape=(None, 0, 0, 3),
                dtype=np.uint8,
            )
        self.h5.close()



def main() -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)

    env_cfg = PushtEnv(
        obs_type="pixels_agent_pos",
        episode_length=args.episode_length,
        observation_height=args.image_height,
        observation_width=args.image_width,
        render_mode="rgb_array",
    )
    if args.num_envs < 1:
        raise ValueError(f"--num-envs must be >= 1, got {args.num_envs}")

    env = _make_vector_env(args)
    unwrapped_envs = [subenv.unwrapped for subenv in env.envs] if hasattr(env, "envs") else [None] * args.num_envs

    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = _load_policy_and_processors(
        args, env_cfg
    )

    saved = 0
    attempts = 0
    total_frames = 0
    writer = H5EpisodeWriter(Path(args.out), compression=args.h5_compression)

    try:
        with tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep") as pbar:
            while saved < args.num_episodes and attempts < args.max_attempts:
                reset_seeds = [
                    args.start_seed + attempts + env_idx
                    for env_idx in range(min(args.num_envs, args.max_attempts - attempts))
                ]
                if len(reset_seeds) < args.num_envs:
                    break
                attempts += args.num_envs
                try:
                    with batch_timeout(args.batch_timeout_s):
                        policy.reset()
                        raw_observations, _ = env.reset(seed=reset_seeds)
                        previous_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                        active = np.ones(args.num_envs, dtype=bool)

                        batch_pixels: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                        batch_actions: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                        batch_states: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                        batch_proprios: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                        batch_success = np.zeros(args.num_envs, dtype=bool)

                        for _step in range(args.episode_length):
                            batch_action = _select_actions(
                                policy=policy,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                raw_observations=raw_observations,
                            )

                            for env_idx in range(args.num_envs):
                                if not active[env_idx]:
                                    continue
                                raw_observation = _slice_observation(raw_observations, env_idx)
                                batch_pixels[env_idx].append(_extract_pixels(raw_observation))
                                batch_actions[env_idx].append(batch_action[env_idx])
                                batch_states[env_idx].append(
                                    _make_state(raw_observation, unwrapped_envs[env_idx])
                                )
                                batch_proprios[env_idx].append(
                                    _make_proprio(
                                        raw_observation, unwrapped_envs[env_idx], previous_actions[env_idx]
                                    )
                                )

                            raw_observations, rewards, terminated, truncated, info = env.step(batch_action)
                            for env_idx in range(args.num_envs):
                                if not active[env_idx]:
                                    continue
                                batch_success[env_idx] = batch_success[env_idx] or _extract_vector_success(
                                    terminated=terminated,
                                    rewards=rewards,
                                    info=info,
                                    env_index=env_idx,
                                )
                                if terminated[env_idx] or truncated[env_idx]:
                                    active[env_idx] = False
                            previous_actions = batch_action

                            if not active.any():
                                break
                except BatchTimeoutError:
                    pbar.set_postfix(
                        attempts=attempts,
                        frames=total_frames,
                        timeout=f"{reset_seeds[0]}-{reset_seeds[-1]}",
                        refresh=False,
                    )
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = _make_vector_env(args)
                    unwrapped_envs = (
                        [subenv.unwrapped for subenv in env.envs]
                        if hasattr(env, "envs")
                        else [None] * args.num_envs
                    )
                    gc.collect()
                    continue

                episodes_to_write = []
                saved_seeds = []
                discarded = 0
                for env_idx, reset_seed in enumerate(reset_seeds):
                    if saved >= args.num_episodes:
                        break
                    if not batch_success[env_idx] and not args.keep_failures:
                        discarded += 1
                        continue

                    episodes_to_write.append(
                        (
                            batch_pixels[env_idx],
                            batch_actions[env_idx],
                            batch_states[env_idx],
                            batch_proprios[env_idx],
                        )
                    )
                    saved_seeds.append(reset_seed)

                if episodes_to_write:
                    if saved + len(episodes_to_write) > args.num_episodes:
                        keep = args.num_episodes - saved
                        episodes_to_write = episodes_to_write[:keep]
                        saved_seeds = saved_seeds[:keep]

                    frames_written = 0
                    for start in range(0, len(episodes_to_write), args.write_batch_size):
                        chunk = episodes_to_write[start : start + args.write_batch_size]
                        writer.append_episodes(chunk)
                        frames_written += sum(len(ep_actions) for _, ep_actions, _, _ in chunk)
                    total_frames += frames_written
                    saved += len(episodes_to_write)
                    pbar.update(len(episodes_to_write))
                    pbar.set_postfix(
                        attempts=attempts,
                        frames=total_frames,
                        saved_batch=len(episodes_to_write),
                        discarded=discarded,
                        last=f"seed {saved_seeds[-1]}",
                        refresh=False,
                    )
                elif discarded:
                    pbar.set_postfix(
                        attempts=attempts,
                        frames=total_frames,
                        saved_batch=0,
                        discarded=discarded,
                        refresh=False,
                    )
                del batch_pixels, batch_actions, batch_states, batch_proprios, batch_success
                del raw_observations, previous_actions, active
                gc.collect()
    finally:
        writer.close()
        env.close()

    if saved == 0:
        raise RuntimeError("No episodes were saved. Try increasing --max-attempts or pass --keep-failures.")

    logging.info("Wrote %s with %d episodes and %d frames.", Path(args.out).resolve(), saved, total_frames)


if __name__ == "__main__":
    main()
