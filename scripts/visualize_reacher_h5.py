#!/usr/bin/env python3
"""Render a Reacher episode from data/reacher.h5 as an mp4 video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401 — register HDF5 compression filters
import h5py
import imageio.v3 as iio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data/reacher.h5"
DEFAULT_OUTPUT = REPO_ROOT / "data/reacher_episode.mp4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    if not args.dataset.is_file():
        sys.exit(f"Dataset not found: {args.dataset}")

    with h5py.File(args.dataset, "r") as h5:
        n_ep = h5["ep_len"].shape[0]
        if not 0 <= args.episode < n_ep:
            sys.exit(f"episode must be in [0, {n_ep - 1}]")

        base = int(h5["ep_offset"][args.episode])
        length = int(h5["ep_len"][args.episode])
        frames = np.asarray(h5["pixels"][base : base + length], dtype=np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, frames, fps=args.fps, codec="libx264")
    print(f"Wrote {args.output}  ({length} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
