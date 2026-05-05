#!/usr/bin/env python3
"""Visualize PushT episodes from a single HDF5 dataset path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET_PATH = Path(__file__).with_name("pusht_expert_train.h5")
REQUIRED_KEYS = ("ep_len", "ep_offset", "pixels", "action")
FILTER_NAMES = {
    32001: "blosc",
}
NUM_SHEET_FRAMES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the PushT HDF5 file.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <dataset-stem>_viz.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Specific episode indices to render.")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="When --episodes is omitted, render this many spread-out episodes.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap per episode.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--fps", type=int, default=10, help="FPS for saved MP4 videos.")
    parser.add_argument("--no-video", action="store_true", help="Only save summary PNGs.")
    return parser.parse_args()


def default_out_dir(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_viz")


def require_hdf5_plugins() -> None:
    try:
        import hdf5plugin  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This dataset uses HDF5 compression plugins for `pixels`, but `hdf5plugin` is not installed. "
            f"Install it in the active environment with `{Path.cwd() / 'latent_brs_venv/bin/pip'} install hdf5plugin`."
        ) from exc


def validate_dataset(h5: h5py.File) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in h5]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {missing}")


def ensure_pixels_are_readable(h5: h5py.File) -> None:
    dataset = h5["pixels"]
    creation_plist = dataset.id.get_create_plist()
    for filter_idx in range(creation_plist.get_nfilters()):
        filter_id, _, _, filter_name = creation_plist.get_filter(filter_idx)
        if filter_id >= 256:
            require_hdf5_plugins()
            human_name = FILTER_NAMES.get(filter_id) or filter_name.decode("utf-8", errors="ignore") or f"id {filter_id}"
            try:
                dataset[:1]
            except OSError as exc:
                raise RuntimeError(
                    f"`pixels` uses the HDF5 filter `{human_name}` and it could not be loaded. "
                    "The plugin package is installed, so this points to a broken HDF5 plugin setup in the environment."
                ) from exc
            break


def pick_episode_indices(num_episodes: int, requested: list[int] | None, count: int) -> list[int]:
    if requested:
        return requested
    if count < 1:
        raise ValueError("--count must be >= 1")
    if num_episodes == 0:
        return []
    samples = np.linspace(0, num_episodes - 1, min(count, num_episodes), dtype=np.int64)
    return [int(idx) for idx in np.unique(samples)]


def episode_slice(h5: h5py.File, episode_idx: int, *, max_frames: int | None, stride: int) -> slice:
    num_episodes = int(h5["ep_len"].shape[0])
    if not 0 <= episode_idx < num_episodes:
        raise IndexError(f"episode {episode_idx} is out of range [0, {num_episodes}).")
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    ep_len = int(h5["ep_len"][episode_idx])
    start = int(h5["ep_offset"][episode_idx])
    stop = start + ep_len
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive when provided.")
        stop = min(stop, start + max_frames * stride)
    if start >= stop:
        raise ValueError(f"episode {episode_idx} produced no frames after filtering.")
    return slice(start, stop, stride)


def load_episode(
    h5: h5py.File,
    episode_idx: int,
    *,
    max_frames: int | None,
    stride: int,
) -> dict[str, np.ndarray]:
    rows = episode_slice(h5, episode_idx, max_frames=max_frames, stride=stride)
    episode = {
        "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
        "action": np.asarray(h5["action"][rows], dtype=np.float32),
    }
    for key in ("state", "proprio", "step_idx"):
        if key in h5:
            episode[key] = np.asarray(h5[key][rows])
    return episode


def sheet_frame_indices(num_frames: int) -> np.ndarray:
    count = min(num_frames, NUM_SHEET_FRAMES)
    return np.unique(np.linspace(0, num_frames - 1, count, dtype=np.int64))


def plot_summary(episode: dict[str, np.ndarray], episode_idx: int, out_path: Path) -> None:
    frames = episode["pixels"]
    actions = episode["action"]
    selected = sheet_frame_indices(frames.shape[0])
    num_cols = len(selected)

    trace_key = "state" if "state" in episode else "proprio" if "proprio" in episode else None
    num_trace_rows = 2 if trace_key is not None else 1
    height_ratios = [2.4] + [1.0] * num_trace_rows

    fig = plt.figure(figsize=(max(12, 2.0 * num_cols), 3.2 + 2.2 * num_trace_rows))
    grid = fig.add_gridspec(1 + num_trace_rows, num_cols, height_ratios=height_ratios)

    for col, frame_idx in enumerate(selected):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frames[frame_idx])
        ax.set_title(f"t={frame_idx}", fontsize=9)
        ax.axis("off")

    time = np.arange(actions.shape[0])
    action_ax = fig.add_subplot(grid[1, :])
    for dim in range(actions.shape[1]):
        action_ax.plot(time, actions[:, dim], label=f"a{dim}")
    action_ax.set_title("Actions")
    action_ax.set_xlabel("Frame")
    action_ax.grid(alpha=0.25)
    action_ax.legend(loc="upper right", ncol=min(actions.shape[1], 4))

    if trace_key is not None:
        trace = np.asarray(episode[trace_key], dtype=np.float32)
        trace_ax = fig.add_subplot(grid[2, :])
        for dim in range(trace.shape[1]):
            trace_ax.plot(time, trace[:, dim], label=f"{trace_key}{dim}")
        trace_ax.set_title(trace_key.capitalize())
        trace_ax.set_xlabel("Frame")
        trace_ax.grid(alpha=0.25)
        trace_ax.legend(loc="upper right", ncol=min(trace.shape[1], 7))

    fig.suptitle(f"PushT episode {episode_idx} ({frames.shape[0]} frames)", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_video(path: Path, frames: np.ndarray, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to save MP4 videos, or pass --no-video.") from exc

    if fps < 1:
        raise ValueError("--fps must be >= 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_dir = args.out_dir or default_out_dir(dataset_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as h5:
        validate_dataset(h5)
        ensure_pixels_are_readable(h5)

        num_episodes = int(h5["ep_len"].shape[0])
        episode_indices = pick_episode_indices(num_episodes, args.episodes, args.count)

        print(f"Dataset: {dataset_path}")
        print(f"Episodes available: {num_episodes}")
        print(f"Episodes selected: {episode_indices}")

        for episode_idx in episode_indices:
            episode = load_episode(h5, episode_idx, max_frames=args.max_frames, stride=args.stride)
            summary_path = out_dir / f"episode_{episode_idx:05d}_summary.png"
            plot_summary(episode, episode_idx, summary_path)
            print(f"Saved summary: {summary_path}")

            if not args.no_video:
                video_path = out_dir / f"episode_{episode_idx:05d}.mp4"
                save_video(video_path, episode["pixels"], fps=args.fps)
                print(f"Saved video:   {video_path}")


if __name__ == "__main__":
    main()
