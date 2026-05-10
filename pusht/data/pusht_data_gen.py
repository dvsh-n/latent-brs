#!/usr/bin/env python3
"""Generate PushT rollouts with the diffusion policy for latent dynamics training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from pusht.shared.pusht_env import DEFAULT_PUSHT_ENV_ID, make_pusht_env
from pusht.shared.utils import load_expert_policy_bundle, select_expert_action

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
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
    parser.add_argument("--keep-failures", action="store_true", default=False)
    parser.add_argument("--render-goal-t", action=argparse.BooleanOptionalAction, default=False)
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


def _make_state(raw_observation: dict[str, Any], env) -> np.ndarray:
    block_x, block_y, block_theta = _extract_block_pose(env)
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


class H5EpisodeWriter:
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.num_episodes = 0
        self.num_frames = 0
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

    def append_episode(
        self,
        ep_pixels: list[np.ndarray],
        ep_actions: list[np.ndarray],
        ep_states: list[np.ndarray],
        ep_proprios: list[np.ndarray],
    ) -> None:
        if not ep_actions:
            return

        pixel_array = np.stack(ep_pixels).astype(np.uint8, copy=False)
        action_array = np.stack(ep_actions).astype(np.float32, copy=False)
        state_array = np.stack(ep_states).astype(np.float32, copy=False)
        proprio_array = np.stack(ep_proprios).astype(np.float32, copy=False)
        episode_len = action_array.shape[0]
        frame_start = self.num_frames
        frame_end = frame_start + episode_len
        ep_end = self.num_episodes + 1

        self._ensure_pixels_dataset(tuple(pixel_array.shape[1:]))

        self.h5["ep_len"].resize((ep_end,))
        self.h5["ep_offset"].resize((ep_end,))
        self.h5["ep_len"][self.num_episodes] = episode_len
        self.h5["ep_offset"][self.num_episodes] = frame_start

        arrays = {
            "pixels": pixel_array,
            "action": action_array,
            "state": state_array,
            "proprio": proprio_array,
            "episode_idx": np.full(episode_len, self.num_episodes, dtype=np.int64),
            "step_idx": np.arange(episode_len, dtype=np.int64),
        }
        for name, array in arrays.items():
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


def make_env(args: argparse.Namespace):
    return make_pusht_env(
        args.env_id,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=args.episode_length,
        observation_width=args.image_width,
        observation_height=args.image_height,
        visualization_width=args.image_width,
        visualization_height=args.image_height,
        hide_target=not args.render_goal_t,
    )


def collect_episode(args: argparse.Namespace, bundle, seed: int):
    env = make_env(args)
    try:
        bundle.policy.reset()
        raw_observation, _ = env.reset(seed=seed)
        previous_action = np.zeros(2, dtype=np.float32)
        ep_pixels: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_states: list[np.ndarray] = []
        ep_proprios: list[np.ndarray] = []
        success = False

        for _ in range(args.episode_length):
            action = select_expert_action(bundle, raw_observation, env=env, action_mode=args.action_mode)
            ep_pixels.append(_extract_pixels(raw_observation))
            ep_actions.append(np.asarray(action, dtype=np.float32).reshape(2))
            ep_states.append(_make_state(raw_observation, env))
            ep_proprios.append(_make_proprio(raw_observation, env, previous_action))

            raw_observation, reward, terminated, truncated, info = env.step(action)
            success = success or _extract_success(terminated, float(reward), info)
            previous_action = np.asarray(action, dtype=np.float32).reshape(2)
            if terminated or truncated:
                break

        return ep_pixels, ep_actions, ep_states, ep_proprios, success
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1.")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")

    bundle = load_expert_policy_bundle(args.model_dir, device=args.device)
    writer = H5EpisodeWriter(args.out)
    saved = 0
    attempts = 0

    try:
        with tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep") as pbar:
            while saved < args.num_episodes:
                seed = args.start_seed + attempts
                attempts += 1
                ep_pixels, ep_actions, ep_states, ep_proprios, success = collect_episode(args, bundle, seed)
                if success or args.keep_failures:
                    writer.append_episode(ep_pixels, ep_actions, ep_states, ep_proprios)
                    saved += 1
                    pbar.update(1)
                    pbar.set_postfix(seed=seed, ep_len=len(ep_actions), success=success, refresh=False)
                else:
                    pbar.set_postfix(seed=seed, discarded=True, refresh=False)
    finally:
        writer.close()

    if saved == 0:
        raise RuntimeError("No episodes were saved. Pass --keep-failures if you want to save unsuccessful rollouts.")

    print(f"Wrote {args.out.resolve()} with {saved} episodes and {writer.num_frames} frames.")


if __name__ == "__main__":
    main()
