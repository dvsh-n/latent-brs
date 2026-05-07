#!/usr/bin/env python3
"""Save one OGBench HDF5 dataset episode as an MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    hdf5plugin = None

import h5py
import numpy as np


DEFAULT_DATASET_PATH = Path(__file__).with_name("cube_single_expert.h5")
DEFAULT_OUT_DIR = Path(__file__).with_name("visualizations")
REQUIRED_KEYS = ("ep_len", "ep_offset", "pixels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode", type=int, default=1200, help="Episode index to render.")
    parser.add_argument("--fps", type=int, default=20, help="FPS for the saved MP4.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap after striding.")
    parser.add_argument("--output-name", default=None, help="Optional MP4 filename.")
    return parser.parse_args()


def validate_dataset(h5: h5py.File) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in h5]
    if missing:
        raise KeyError(f"OGBench HDF5 dataset is missing required keys: {missing}")


def episode_rows(h5: h5py.File, episode_idx: int, *, stride: int, max_frames: int | None) -> np.ndarray:
    num_episodes = int(h5["ep_len"].shape[0])
    if not 0 <= episode_idx < num_episodes:
        raise IndexError(f"episode {episode_idx} is out of range [0, {num_episodes}).")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames must be positive when provided.")

    ep_len = int(h5["ep_len"][episode_idx])
    ep_offset = int(h5["ep_offset"][episode_idx])
    rows = np.arange(ep_offset, ep_offset + ep_len, stride, dtype=np.int64)
    if max_frames is not None:
        rows = rows[:max_frames]
    if rows.size == 0:
        raise ValueError(f"episode {episode_idx} produced no frames after stride/max-frame filtering.")
    return rows


def save_video(path: Path, frames: np.ndarray, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to save MP4 videos.") from exc

    if fps < 1:
        raise ValueError("--fps must be >= 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"ogbench_episode_{args.episode:05d}.mp4"
    output_path = args.out_dir / output_name

    with h5py.File(args.dataset_path, "r") as h5:
        validate_dataset(h5)
        rows = episode_rows(h5, args.episode, stride=args.stride, max_frames=args.max_frames)
        frames = np.asarray(h5["pixels"][rows], dtype=np.uint8)

    save_video(output_path, frames, args.fps)
    height, width = frames.shape[1:3]
    print(f"Saved video: {output_path}")
    print(f"Episode {args.episode}: {frames.shape[0]} frames, {width}x{height}, {args.fps} fps")


if __name__ == "__main__":
    main()
