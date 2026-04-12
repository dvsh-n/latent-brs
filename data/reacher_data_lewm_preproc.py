#!/usr/bin/env python3
"""Convert local Reacher rollouts to the HDF5 layout used by LE-WM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm


DEFAULT_DATASET_PATH = "data/test_data/test_data.pt"
DEFAULT_OUTPUT_PATH = "data/test_data/reacher_expert_test.h5"
DEFAULT_DATASET_NAME = "reacher_expert_test"
STATE_DIM = 6
ACTION_DIM = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize for stored pixels.")
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="lzf",
        help="HDF5 compression for the pixels dataset.",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path.expanduser().resolve()

    if args.cache_dir is None:
        return Path(DEFAULT_OUTPUT_PATH).expanduser().resolve()

    datasets_dir = args.cache_dir.expanduser().resolve() / "datasets"
    return datasets_dir / f"{args.dataset_name}.h5"


def load_video_frames(video_path: Path, resize: int | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video {video_path} contains no frames.")
    return np.stack(frames, axis=0)


def create_dataset(
    h5: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = None,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = resolve_output_path(args)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    dataset = torch.load(dataset_path, map_location="cpu")
    states = dataset["states"].to(torch.float32).contiguous().numpy()
    actions = dataset["actions"].to(torch.float32).contiguous().numpy()
    video_dir = Path(dataset["video_dir"]).expanduser().resolve()

    if states.ndim != 3 or states.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected states with shape [episodes, frames, {STATE_DIM}], got {states.shape}.")
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected actions with shape [episodes, steps, {ACTION_DIM}], got {actions.shape}.")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    num_episodes, frames_per_episode, state_dim = states.shape
    if actions.shape[0] != num_episodes or actions.shape[1] != frames_per_episode - 1:
        raise ValueError(
            "Expected actions to have one fewer timestep than states: "
            f"states={states.shape}, actions={actions.shape}."
        )
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive.")
        num_episodes = min(num_episodes, args.max_episodes)
        states = states[:num_episodes]
        actions = actions[:num_episodes]

    first_video = video_dir / "trajectory_0000000.mp4"
    first_frames = load_video_frames(first_video, args.resize)
    if first_frames.shape[0] != frames_per_episode:
        raise RuntimeError(
            f"{first_video.name} yielded {first_frames.shape[0]} frames, expected {frames_per_episode}."
        )

    height, width, channels = first_frames.shape[1:]
    if channels != 3:
        raise RuntimeError(f"Expected RGB video frames, got shape {first_frames.shape[1:]}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_frames = num_episodes * frames_per_episode
    ep_len = np.full((num_episodes,), frames_per_episode, dtype=np.int64)
    ep_offset = np.arange(num_episodes, dtype=np.int64) * frames_per_episode
    compression = None if args.compression == "none" else args.compression

    with h5py.File(output_path, "w") as h5:
        h5.attrs["source_dataset_path"] = str(dataset_path)
        h5.attrs["source_video_dir"] = str(video_dir)
        h5.attrs["dataset_name"] = args.dataset_name
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["state_keys"] = json.dumps(dataset.get("state_keys", []))

        h5.create_dataset("ep_len", data=ep_len)
        h5.create_dataset("ep_offset", data=ep_offset)

        pixels_ds = create_dataset(
            h5,
            "pixels",
            (total_frames, height, width, 3),
            np.uint8,
            compression=compression,
            chunks=(1, height, width, 3),
        )
        action_ds = create_dataset(h5, "action", (total_frames, ACTION_DIM), np.float32, chunks=True)
        obs_ds = create_dataset(h5, "observation", (total_frames, state_dim), np.float32, chunks=True)
        qpos_ds = create_dataset(h5, "qpos", (total_frames, 2), np.float32, chunks=True)
        qvel_ds = create_dataset(h5, "qvel", (total_frames, 2), np.float32, chunks=True)
        episode_idx_ds = create_dataset(h5, "episode_idx", (total_frames,), np.int64, chunks=True)
        step_idx_ds = create_dataset(h5, "step_idx", (total_frames,), np.int64, chunks=True)

        for episode_idx in tqdm(range(num_episodes), desc="Writing LE-WM HDF5", unit="episode"):
            video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
            frames = first_frames if episode_idx == 0 else load_video_frames(video_path, args.resize)
            if frames.shape != (frames_per_episode, height, width, 3):
                raise RuntimeError(
                    f"{video_path.name} yielded {frames.shape}, "
                    f"expected {(frames_per_episode, height, width, 3)}."
                )

            start = episode_idx * frames_per_episode
            end = start + frames_per_episode

            padded_actions = np.empty((frames_per_episode, ACTION_DIM), dtype=np.float32)
            padded_actions[:-1] = actions[episode_idx]
            padded_actions[-1] = np.nan

            pixels_ds[start:end] = frames
            action_ds[start:end] = padded_actions
            obs_ds[start:end] = states[episode_idx]
            qpos_ds[start:end] = states[episode_idx, :, :2]
            qvel_ds[start:end] = states[episode_idx, :, 4:6]
            episode_idx_ds[start:end] = episode_idx
            step_idx_ds[start:end] = np.arange(frames_per_episode, dtype=np.int64)

    summary = {
        "output_path": str(output_path),
        "dataset_name": args.dataset_name,
        "num_episodes": num_episodes,
        "frames_per_episode": frames_per_episode,
        "total_frames": total_frames,
        "pixels_shape": [total_frames, height, width, 3],
        "action_shape": [total_frames, ACTION_DIM],
        "observation_shape": [total_frames, state_dim],
        "compression": args.compression,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
