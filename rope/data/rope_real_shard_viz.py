#!/usr/bin/env python3
"""Write a trajectory video from a real-data rope shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


DEFAULT_SHARD_PATH = "rope/data/real_data/rope_real_shard0010.h5"
DEFAULT_OUT_DIR = "rope/data/rope_real_viz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-path", type=Path, default=Path(DEFAULT_SHARD_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS. Defaults to shard timing.")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_video(frames: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".mp4":
        imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    else:
        imageio.mimwrite(out_path, frames, fps=fps)


def infer_fps(h5: h5py.File, start: int, stop: int) -> float:
    if stop - start < 2:
        return 20.0
    if "camera_frame_time" in h5:
        timestamps = np.asarray(h5["camera_frame_time"][start:stop], dtype=np.float64).reshape(-1)
    elif "time" in h5:
        timestamps = np.asarray(h5["time"][start:stop], dtype=np.float64).reshape(-1)
    else:
        return 20.0

    deltas = np.diff(timestamps)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if deltas.size == 0:
        return 20.0
    return float(1.0 / np.median(deltas))


def main() -> None:
    args = parse_args()
    shard_path = args.shard_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_path = out_dir / f"{shard_path.stem}_ep{args.episode_idx:04d}.{args.format}"

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Pass --overwrite to replace it: {out_path}")

    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    try:
        with h5py.File(shard_path, "r") as h5:
            ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if args.episode_idx < 0 or args.episode_idx >= len(ep_len):
                raise IndexError(f"episode_idx out of range [0, {len(ep_len) - 1}]: {args.episode_idx}")

            episode_len = int(ep_len[args.episode_idx])
            episode_offset = int(ep_offset[args.episode_idx])
            if args.start_step < 0 or args.start_step >= episode_len:
                raise IndexError(f"start_step out of range [0, {episode_len - 1}]: {args.start_step}")

            max_frames = episode_len - args.start_step if args.max_frames is None else min(args.max_frames, episode_len - args.start_step)
            if max_frames <= 0:
                raise ValueError("No frames selected for output.")

            start = episode_offset + args.start_step
            stop = start + max_frames
            frames = np.asarray(h5["pixels"][start:stop], dtype=np.uint8)
            fps = float(args.fps) if args.fps is not None else infer_fps(h5, start, stop)
    except OSError as exc:
        raise OSError(f"Failed to read shard: {shard_path}. The file may be corrupted.") from exc

    save_video(frames, out_path, fps)
    print(out_path)


if __name__ == "__main__":
    main()
