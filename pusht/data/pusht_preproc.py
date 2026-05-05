#!/usr/bin/env python3
"""Re-render a PushT HDF5 dataset from state with the green target hidden.

This is cleaner than pixel masking: it uses the stored simulator state to draw
new image observations and skips the renderer step that draws the green target
T. All non-pixel datasets are copied unchanged.

By default this reads `pusht_expert_train.h5` next to this script and writes a
new dataset file named `pusht_expert_train_rerendered_no_target.h5`.

Run with the Python environment that has gym_pusht installed, for example:

    lerobot_venv/bin/python pusht/data/rerender_pusht_h5_without_target.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import hdf5plugin
except ModuleNotFoundError:
    hdf5plugin = None

import h5py
import numpy as np


DEFAULT_DATASET_PATH = Path(__file__).with_name("pusht_expert_train.h5")
DEFAULT_OUTPUT_SUFFIX = "_rerendered_no_target"
DEFAULT_CHUNK_FRAMES = 100
REQUIRED_KEYS = ("ep_len", "ep_offset", "pixels", "action", "state")


def default_output_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}{DEFAULT_OUTPUT_SUFFIX}{dataset_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination HDF5 path. Defaults to <dataset-stem>_rerendered_no_target.h5.",
    )
    parser.add_argument("--pixel-key", default="pixels")
    parser.add_argument("--state-key", default="state")
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument(
        "--compression",
        choices=("blosc", "lzf", "gzip", "none"),
        default="lzf",
        help="Compression for the rewritten pixels dataset. Defaults to lzf so the script runs without hdf5plugin.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Debug option: only re-render this many frames, then stop.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_frames < 1:
        raise ValueError("--chunk-frames must be >= 1.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive when provided.")


def validate_pusht_dataset(h5: h5py.File, *, pixel_key: str, state_key: str) -> None:
    required = tuple(key if key != "pixels" else pixel_key for key in REQUIRED_KEYS)
    required = tuple(key if key != "state" else state_key for key in required)
    missing = [key for key in required if key not in h5]
    if missing:
        raise KeyError(f"PushT HDF5 dataset is missing required keys: {missing}")
    pixels = h5[pixel_key]
    states = h5[state_key]
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.dtype("uint8"):
        raise ValueError(f"{pixel_key!r} must have shape (N, H, W, 3) and dtype uint8.")
    if states.ndim != 2 or states.shape[1] < 5:
        raise ValueError(f"{state_key!r} must have shape (N, >=5).")
    if pixels.shape[0] != states.shape[0]:
        raise ValueError(f"{pixel_key!r} and {state_key!r} must have the same frame count.")


def pixel_create_kwargs(src: h5py.Dataset, compression: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"chunks": src.chunks}
    if compression == "none":
        return kwargs
    if compression == "lzf":
        return {**kwargs, "compression": "lzf"}
    if compression == "gzip":
        return {**kwargs, "compression": "gzip", "compression_opts": 4, "shuffle": True}
    if hdf5plugin is None:
        raise ModuleNotFoundError("Install hdf5plugin or pass --compression lzf/gzip/none.")
    return {
        **kwargs,
        **hdf5plugin.Blosc(
            cname="lz4",
            clevel=5,
            shuffle=hdf5plugin.Blosc.SHUFFLE,
        ),
    }


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def copy_non_pixel_items(src_h5: h5py.File, dst_h5: h5py.File, *, pixel_key: str) -> None:
    for key in src_h5.keys():
        if key != pixel_key:
            src_h5.copy(key, dst_h5)


def import_pusht_renderer():
    try:
        import pygame
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv
        from gym_pusht.envs.pymunk_override import DrawOptions
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs gym_pusht, pygame, and pymunk. In this repo, run it with "
            "`lerobot_venv/bin/python pusht/data/rerender_pusht_h5_without_target.py`."
        ) from exc

    class PushTNoTargetEnv(PushTEnv):
        def _setup(self):
            self.space = pymunk.Space()
            self.space.gravity = 0, 0
            self.space.damping = self.damping if self.damping is not None else 0.0
            self.teleop = False

            walls = [
                self.add_segment(self.space, (5, 506), (5, 5), 2),
                self.add_segment(self.space, (5, 5), (506, 5), 2),
                self.add_segment(self.space, (506, 5), (506, 506), 2),
                self.add_segment(self.space, (5, 506), (506, 506), 2),
            ]
            self.space.add(*walls)

            self.agent = self.add_circle(self.space, (256, 400), 15)
            self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
            self.goal_pose = np.array([256, 256, np.pi / 4])
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            self.n_contact_points = 0

        def _draw(self):
            screen = pygame.Surface((512, 512))
            screen.fill((255, 255, 255))
            draw_options = DrawOptions(screen)
            self.space.debug_draw(draw_options)
            return screen

    return PushTNoTargetEnv


def make_no_target_env(*, height: int, width: int):
    env_cls = import_pusht_renderer()
    env = env_cls(
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
    )
    env.reset(seed=0)
    return env


def set_render_state(env: Any, state: np.ndarray) -> None:
    # Use the environment's legacy setter because the block has a non-default
    # center of gravity. Direct assignment to block.position does not reproduce
    # the rendered pose from the original PushT dataset.
    env.agent.velocity = [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._set_state(np.asarray(state[:5], dtype=np.float64))
    env.agent.velocity = [float(state[5]), float(state[6])] if state.shape[0] >= 7 else [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._last_action = None


def render_states_without_target(env: Any, states: np.ndarray) -> np.ndarray:
    frames: list[np.ndarray] = []
    for state in states:
        set_render_state(env, state)
        frames.append(np.asarray(env._render(visualize=False), dtype=np.uint8))
    return np.stack(frames, axis=0)


def rewrite_pixels_from_state(
    src_h5: h5py.File,
    dst_h5: h5py.File,
    *,
    pixel_key: str,
    state_key: str,
    args: argparse.Namespace,
) -> int:
    src_pixels = src_h5[pixel_key]
    src_states = src_h5[state_key]
    height, width = src_pixels.shape[1:3]
    dst_pixels = dst_h5.create_dataset(
        pixel_key,
        shape=src_pixels.shape,
        dtype=src_pixels.dtype,
        **pixel_create_kwargs(src_pixels, args.compression),
    )
    copy_attrs(src_pixels.attrs, dst_pixels.attrs)

    env = make_no_target_env(height=height, width=width)
    total_frames = int(src_pixels.shape[0])
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames)

    try:
        for start in range(0, total_frames, args.chunk_frames):
            stop = min(start + args.chunk_frames, total_frames)
            states = np.asarray(src_states[start:stop], dtype=np.float64)
            dst_pixels[start:stop] = render_states_without_target(env, states)
            print(f"Re-rendered frames {start:>8}-{stop:<8}", flush=True)

        if total_frames < src_pixels.shape[0]:
            dst_pixels[total_frames:] = src_pixels[total_frames:]
            print(f"Copied original pixels for remaining frames {total_frames}-{src_pixels.shape[0]}")
    finally:
        env.close()
    return total_frames


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_path = args.output_path or default_output_path(args.dataset_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    with h5py.File(args.dataset_path, "r") as src_h5:
        validate_pusht_dataset(src_h5, pixel_key=args.pixel_key, state_key=args.state_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as dst_h5:
            copy_attrs(src_h5.attrs, dst_h5.attrs)
            copy_non_pixel_items(src_h5, dst_h5, pixel_key=args.pixel_key)
            rendered = rewrite_pixels_from_state(
                src_h5,
                dst_h5,
                pixel_key=args.pixel_key,
                state_key=args.state_key,
                args=args,
            )

    print(f"Wrote: {output_path}")
    print(f"Re-rendered frames without target: {rendered}")


if __name__ == "__main__":
    main()
