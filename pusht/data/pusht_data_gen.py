#!/usr/bin/env python3
"""Generate PushT rollouts with the diffusion policy for latent dynamics training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pusht.shared.pusht_env import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from pusht.shared.utils import env_action_from_policy_action, load_expert_policy_bundle

DEFAULT_MODEL_DIR = Path("pusht/models")
DEFAULT_OUTPUT_PATH = Path("pusht/data/pusht_diffusion_train.h5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Query the policy every N env steps and store samples at that same interval.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--async-envs", action="store_true", default=True)
    parser.add_argument("--keep-failures", action="store_true", default=True)
    parser.add_argument("--hide-target", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _as_xy(value: Any) -> np.ndarray:
    if hasattr(value, "x") and hasattr(value, "y"):
        return np.asarray([value.x, value.y], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected an xy-like value, got {value!r}")
    return arr[:2]


def _extract_pixels(raw_observation: dict[str, Any]) -> np.ndarray:
    if "pixels" not in raw_observation:
        raise KeyError("Expected PushT observation to contain 'pixels'.")
    pixels = raw_observation["pixels"]
    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB pixels, got shape {pixels.shape}.")
    return pixels


def _extract_agent_xy(raw_observation: dict[str, Any], env) -> np.ndarray:
    if "agent_pos" in raw_observation:
        return np.asarray(raw_observation["agent_pos"], dtype=np.float32).reshape(-1)[:2]
    base_env = getattr(env, "unwrapped", env)
    return _as_xy(base_env.agent.position)


def _extract_block_pose(env) -> tuple[float, float, float]:
    base_env = getattr(env, "unwrapped", env)
    block_xy = _as_xy(base_env.block.position)
    block_theta = float(np.asarray(base_env.block.angle, dtype=np.float32).reshape(-1)[0])
    return float(block_xy[0]), float(block_xy[1]), block_theta


def _extract_goal_pose(env) -> tuple[float, float, float]:
    base_env = getattr(env, "unwrapped", env)
    goal = np.asarray(base_env.goal_pose, dtype=np.float32).reshape(-1)
    return float(goal[0]), float(goal[1]), float(goal[2])


class PushTPrivilegedObsWrapper(gym.Wrapper):
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

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info


def _make_state(raw_observation: dict[str, Any], env) -> np.ndarray:
    if "block_pose" in raw_observation:
        block_x, block_y, block_theta = np.asarray(raw_observation["block_pose"], dtype=np.float32).reshape(-1)[:3]
    else:
        block_x, block_y, block_theta = _extract_block_pose(env)
    if "goal_pose" in raw_observation:
        goal_x, goal_y, goal_theta = np.asarray(raw_observation["goal_pose"], dtype=np.float32).reshape(-1)[:3]
    else:
        goal_x, goal_y, goal_theta = _extract_goal_pose(env)
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


def _make_proprio(raw_observation: dict[str, Any], env, previous_action: np.ndarray) -> np.ndarray:
    agent_xy = _extract_agent_xy(raw_observation, env)
    return np.asarray(
        [agent_xy[0], agent_xy[1], previous_action[0], previous_action[1]],
        dtype=np.float32,
    )


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).item())
    if "success" in info:
        return bool(np.asarray(info["success"]).item())
    return bool(terminated or reward >= 0.95)


def _policy_action_to_relative_action(policy_action: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    return np.clip((policy_action - agent_pos) / 100.0, -1.0, 1.0).astype(np.float32)


class H5EpisodeWriter:
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.num_episodes = 0
        self.num_frames = 0
        self.h5 = h5py.File(out_path, "w")
        self.h5.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
        self.h5.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("action", shape=(0, 2), maxshape=(None, 2), dtype=np.float32, chunks=True)
        self.h5.create_dataset("state", shape=(0, 7), maxshape=(None, 7), dtype=np.float32, chunks=True)
        self.h5.create_dataset("proprio", shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=True)
        self.h5.create_dataset("episode_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("step_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)

    def _ensure_pixels_dataset(self, image_shape: tuple[int, int, int]) -> None:
        if "pixels" in self.h5:
            return
        self.h5.create_dataset(
            "pixels",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            dtype=np.uint8,
            chunks=(1, *image_shape),
        )

    def append_episodes(
        self,
        episodes: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]],
    ) -> None:
        episodes = [episode for episode in episodes if len(episode[1]) > 0]
        if not episodes:
            return

        ep_lengths = np.asarray([len(ep_actions) for _, ep_actions, _, _ in episodes], dtype=np.int32)
        flat_pixels = [frame for ep_pixels, _, _, _ in episodes for frame in ep_pixels]
        flat_actions = [frame for _, ep_actions, _, _ in episodes for frame in ep_actions]
        flat_states = [frame for _, _, ep_states, _ in episodes for frame in ep_states]
        flat_proprios = [frame for _, _, _, ep_proprios in episodes for frame in ep_proprios]

        pixel_array = np.stack(flat_pixels).astype(np.uint8, copy=False)
        action_array = np.stack(flat_actions).astype(np.float32, copy=False)
        state_array = np.stack(flat_states).astype(np.float32, copy=False)
        proprio_array = np.stack(flat_proprios).astype(np.float32, copy=False)
        episode_idx_array = np.repeat(
            np.arange(self.num_episodes, self.num_episodes + len(episodes), dtype=np.int64),
            ep_lengths,
        )
        step_idx_array = np.concatenate([np.arange(length, dtype=np.int64) for length in ep_lengths])
        ep_offset_array = self.num_frames + np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(ep_lengths[:-1], dtype=np.int64)]
        )

        num_new_frames = int(ep_lengths.sum())
        frame_start = self.num_frames
        frame_end = frame_start + num_new_frames
        ep_start = self.num_episodes
        ep_end = ep_start + len(episodes)

        self._ensure_pixels_dataset(tuple(pixel_array.shape[1:]))

        self.h5["ep_len"].resize((ep_end,))
        self.h5["ep_offset"].resize((ep_end,))
        self.h5["ep_len"][ep_start:ep_end] = ep_lengths
        self.h5["ep_offset"][ep_start:ep_end] = ep_offset_array

        arrays = {
            "pixels": pixel_array,
            "action": action_array,
            "state": state_array,
            "proprio": proprio_array,
            "episode_idx": episode_idx_array,
            "step_idx": step_idx_array,
        }
        for name, array in arrays.items():
            dataset = self.h5[name]
            dataset.resize((frame_end, *dataset.shape[1:]))
            dataset[frame_start:frame_end] = array

        self.num_episodes = ep_end
        self.num_frames = frame_end

    def close(self) -> None:
        if "pixels" not in self.h5:
            self.h5.create_dataset(
                "pixels",
                shape=(0, 0, 0, 3),
                maxshape=(None, 0, 0, 3),
                dtype=np.uint8,
            )
        self.h5.close()


def make_env(args: argparse.Namespace):
    env = make_pusht_env(
        args.env_id,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=args.episode_length,
        observation_width=args.image_width,
        observation_height=args.image_height,
        visualization_width=args.image_width,
        visualization_height=args.image_height,
        hide_target=args.hide_target,
    )
    return PushTPrivilegedObsWrapper(env)


def _resize_hwc_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _slice_observation(raw_observations: dict[str, Any], env_index: int) -> dict[str, Any]:
    observation = {}
    for key, value in raw_observations.items():
        if isinstance(value, dict):
            observation[key] = {subkey: subvalue[env_index] for subkey, subvalue in value.items()}
        else:
            observation[key] = value[env_index]
    return observation


def _select_expert_actions(
    bundle,
    raw_observations: dict[str, Any],
    action_space: gym.Space | None,
) -> tuple[np.ndarray, np.ndarray]:
    image_shape = tuple(bundle.policy.config.input_features["observation.image"].shape)
    channels, height, width = image_shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel policy input, got {image_shape=}.")

    pixels = raw_observations["pixels"]
    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    resized = np.stack([_resize_hwc_uint8(frame, height, width) for frame in pixels], axis=0)
    images = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous().float() / 255.0
    states = torch.from_numpy(np.asarray(raw_observations["agent_pos"], dtype=np.float32))
    batch = {
        "observation.image": images,
        "observation.state": states,
    }
    batch = bundle.preprocessor(batch)
    action = bundle.policy.select_action(batch)
    action = bundle.postprocessor(action)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    policy_actions = np.asarray(action, dtype=np.float32).reshape(states.shape[0], 2)
    agent_pos = states.numpy()
    logged_actions = np.stack(
        [
            _policy_action_to_relative_action(policy_action, agent_pos[idx])
            for idx, policy_action in enumerate(policy_actions)
        ],
        axis=0,
    )

    if action_space is not None:
        high = np.asarray(getattr(action_space, "high", None))
        low = np.asarray(getattr(action_space, "low", None))
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return logged_actions, logged_actions

    env_actions = np.stack(
        [
            env_action_from_policy_action(
                policy_action,
                env=None,
                observation=_slice_observation(raw_observations, env_idx),
                action_mode="absolute",
            )
            for env_idx, policy_action in enumerate(policy_actions)
        ],
        axis=0,
    ).astype(np.float32)
    return env_actions, logged_actions


def _extract_vector_success(
    terminated: np.ndarray,
    rewards: np.ndarray,
    info: dict[str, Any],
    env_index: int,
) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, (list, tuple)) and final_info[env_index] is not None:
            env_info = final_info[env_index]
            if "is_success" in env_info:
                return bool(env_info["is_success"])
            if "success" in env_info:
                return bool(env_info["success"])
    if "is_success" in info:
        return bool(np.asarray(info["is_success"])[env_index])
    if "success" in info:
        return bool(np.asarray(info["success"])[env_index])
    return bool(terminated[env_index] or rewards[env_index] >= 0.95)


def make_vector_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    env_fns = [lambda args=args: make_env(args) for _ in range(args.num_envs)]
    env_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    try:
        from gymnasium.vector import AutoresetMode

        return env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    except (ImportError, TypeError):
        return env_cls(env_fns)


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1.")
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1.")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")

    bundle = load_expert_policy_bundle(args.model_dir, device=args.device)
    writer = H5EpisodeWriter(args.out)
    env = make_vector_env(args)
    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    saved = 0
    attempts = 0

    try:
        with tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep") as pbar:
            while saved < args.num_episodes:
                reset_seeds = [args.start_seed + attempts + env_idx for env_idx in range(args.num_envs)]
                attempts += args.num_envs
                bundle.policy.reset()
                raw_observations, _ = env.reset(seed=reset_seeds)
                previous_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                active = np.ones(args.num_envs, dtype=bool)
                env_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                logged_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                batch_pixels: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_actions: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_states: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_proprios: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_success = np.zeros(args.num_envs, dtype=bool)

                for step_idx in range(args.episode_length):
                    if step_idx % args.control_interval == 0:
                        env_actions, logged_actions = _select_expert_actions(bundle, raw_observations, action_space)

                    for env_idx in range(args.num_envs):
                        if not active[env_idx]:
                            continue
                        if step_idx % args.control_interval == 0:
                            raw_observation = _slice_observation(raw_observations, env_idx)
                            batch_pixels[env_idx].append(_extract_pixels(raw_observation))
                            batch_actions[env_idx].append(logged_actions[env_idx])
                            batch_states[env_idx].append(_make_state(raw_observation, env=None))
                            batch_proprios[env_idx].append(
                                _make_proprio(raw_observation, env=None, previous_action=previous_actions[env_idx])
                            )

                    raw_observations, rewards, terminated, truncated, info = env.step(env_actions)
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
                    if step_idx % args.control_interval == 0:
                        previous_actions = logged_actions.copy()
                    if not active.any():
                        break

                episodes_to_write = []
                last_saved_seed = None
                discarded = 0
                for env_idx, seed in enumerate(reset_seeds):
                    if saved + len(episodes_to_write) >= args.num_episodes:
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
                    last_saved_seed = seed

                if episodes_to_write:
                    writer.append_episodes(episodes_to_write)
                    saved += len(episodes_to_write)
                    pbar.update(len(episodes_to_write))
                    pbar.set_postfix(
                        last_seed=last_saved_seed,
                        saved_batch=len(episodes_to_write),
                        stored_steps=sum(len(ep_actions) for _, ep_actions, _, _ in episodes_to_write),
                        discarded=discarded,
                        refresh=False,
                    )
                else:
                    pbar.set_postfix(
                        last_seed=reset_seeds[-1],
                        saved_batch=0,
                        stored_steps=0,
                        discarded=discarded,
                        refresh=False,
                    )
    finally:
        writer.close()
        env.close()

    if saved == 0:
        raise RuntimeError("No episodes were saved. Pass --keep-failures if you want to save unsuccessful rollouts.")

    print(f"Wrote {args.out.resolve()} with {saved} episodes and {writer.num_frames} frames.")


if __name__ == "__main__":
    main()
