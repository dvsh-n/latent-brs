#!/usr/bin/env python3
"""Visualize one random Two Room episode from a chosen rollout mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

DEFAULT_DATASET = Path("two_room/data/train_data/two_room_train.h5")
DEFAULT_OUTDIR = Path("two_room/data/viz")
ROLLOUT_MODES = ("expert", "expert_plus_noise", "random_smooth", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=ROLLOUT_MODES, default=ROLLOUT_MODES[3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=None, help="Use a specific matching episode instead of sampling.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=None, help="Defaults to '<mode>_episode_<idx>.mp4'.")
    parser.add_argument("--fps", type=int, default=None, help="Defaults to the dataset video_fps attr if present.")
    return parser.parse_args()


def _load_rollout_modes(h5: h5py.File) -> tuple[str, ...]:
    raw = h5.attrs.get("rollout_modes")
    if raw is None:
        return ROLLOUT_MODES
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return tuple(json.loads(raw))
    return tuple(raw)


def _select_episode_indices(h5: h5py.File, target_mode_idx: int) -> np.ndarray:
    ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    rollout_mode = np.asarray(h5["rollout_mode"][:], dtype=np.int64)
    episode_mode = rollout_mode[ep_offset]
    if episode_mode.shape[0] != ep_len.shape[0]:
        raise ValueError("Episode metadata is inconsistent: rollout_mode and ep_offset sizes do not match.")
    return np.flatnonzero(episode_mode == target_mode_idx)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as h5:
        rollout_modes = _load_rollout_modes(h5)
        if args.mode not in rollout_modes:
            raise ValueError(f"Mode {args.mode!r} not found in dataset rollout_modes={rollout_modes}.")

        mode_idx = rollout_modes.index(args.mode)
        matching_episodes = _select_episode_indices(h5, mode_idx)
        if matching_episodes.size == 0:
            raise ValueError(f"No episodes found for mode {args.mode!r}.")

        if args.episode_idx is not None:
            if args.episode_idx not in set(matching_episodes.tolist()):
                raise ValueError(f"Episode {args.episode_idx} does not match mode {args.mode!r}.")
            episode_idx = int(args.episode_idx)
        else:
            rng = np.random.default_rng(args.seed)
            episode_idx = int(rng.choice(matching_episodes))

        ep_offset = int(h5["ep_offset"][episode_idx])
        ep_len = int(h5["ep_len"][episode_idx])
        pixels = np.asarray(h5["pixels"][ep_offset : ep_offset + ep_len], dtype=np.uint8)
        episode_seed = int(h5["episode_seed"][episode_idx]) if "episode_seed" in h5 else None
        terminated = bool(h5["terminated"][episode_idx]) if "terminated" in h5 else None
        truncated = bool(h5["truncated"][episode_idx]) if "truncated" in h5 else None
        fps = int(args.fps if args.fps is not None else h5.attrs.get("video_fps", 10))

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.mode}_episode_{episode_idx:07d}.mp4"
    output_path = outdir / output_name

    try:
        imageio.mimwrite(output_path, pixels, fps=fps, quality=8, macro_block_size=1)
    except Exception:
        output_path = output_path.with_suffix(".gif")
        imageio.mimwrite(output_path, pixels, fps=fps)

    summary = {
        "dataset": str(dataset_path),
        "mode": args.mode,
        "episode_idx": episode_idx,
        "episode_seed": episode_seed,
        "num_frames": ep_len,
        "terminated": terminated,
        "truncated": truncated,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
