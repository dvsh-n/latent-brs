#!/usr/bin/env python3
"""Export PushT obstacle dataset sample images with insertion obstacle overlays."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from pusht.plan.obs_data_collect import PUSHT_CANVAS_SIZE, insertion_visual_rects, rotation_matrix

DEFAULT_DATA_PATH = "pusht/plan/obstacle_data_insert/obstacle_classifier_data.pt"
DEFAULT_OUT_DIR = "pusht/plan/obstacle_data_insert/sample_viz"
DEFAULT_SAMPLES_PER_CLASS = 100
INSERTION_COLOR_RGBA = (255, 140, 0, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--first",
        action="store_true",
        help="Use the first N samples of each class instead of a seeded random subset.",
    )
    return parser.parse_args()


def local_rect_to_frame_polygon(
    goal_pose: np.ndarray,
    rect: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = rect
    corners = np.asarray(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=np.float64,
    )
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    world = goal_pose[:2] + corners @ rotation_matrix(float(goal_pose[2])).T
    height, width = frame_shape[:2]
    scale = np.asarray([float(width) / PUSHT_CANVAS_SIZE, float(height) / PUSHT_CANVAS_SIZE], dtype=np.float64)
    pixels = world * scale
    return [(float(x), float(y)) for x, y in pixels]


def render_insertion_obstacle_overlay(
    frame: np.ndarray,
    goal_pose: np.ndarray,
    *,
    buffer: float,
    thickness: float,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect in insertion_visual_rects(buffer=buffer, thickness=thickness, margin=0.0):
        polygon = local_rect_to_frame_polygon(goal_pose, rect, frame.shape)
        draw.polygon(polygon, fill=INSERTION_COLOR_RGBA)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def select_indices(labels: np.ndarray, label: int, count: int, *, first: bool, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(labels == int(label)).astype(np.int64)
    if indices.shape[0] < int(count):
        raise ValueError(f"Requested {count} samples for label {label}, but only found {indices.shape[0]}.")
    if first:
        return indices[:count]
    return rng.choice(indices, size=count, replace=False).astype(np.int64)


def save_samples(
    pixels: np.ndarray,
    indices: np.ndarray,
    *,
    goal_pose: np.ndarray,
    buffer: float,
    thickness: float,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_number, dataset_index in enumerate(tqdm(indices, desc=f"Saving {prefix}", unit="image")):
        frame = np.asarray(pixels[int(dataset_index)], dtype=np.uint8)
        overlay = render_insertion_obstacle_overlay(frame, goal_pose, buffer=buffer, thickness=thickness)
        name = f"{prefix}_{sample_number:03d}_idx_{int(dataset_index):06d}.png"
        imageio.imwrite(out_dir / name, np.ascontiguousarray(overlay))


def main() -> None:
    args = parse_args()
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")

    rng = np.random.default_rng(args.seed)
    data_path = args.data_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    dataset = payload["dataset"]
    metadata = payload.get("metadata", {})

    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    goal_pose = np.asarray(metadata["goal_pose"], dtype=np.float64).reshape(-1)
    buffer = float(metadata.get("insertion_visual_buffer", 12.0))
    thickness = float(metadata.get("insertion_visual_thickness", 80.0))

    obstacle_idx = select_indices(labels, 1, int(args.samples_per_class), first=bool(args.first), rng=rng)
    non_obstacle_idx = select_indices(labels, 0, int(args.samples_per_class), first=bool(args.first), rng=rng)

    save_samples(
        pixels,
        obstacle_idx,
        goal_pose=goal_pose,
        buffer=buffer,
        thickness=thickness,
        out_dir=out_dir / "obstacle",
        prefix="obstacle",
    )
    save_samples(
        pixels,
        non_obstacle_idx,
        goal_pose=goal_pose,
        buffer=buffer,
        thickness=thickness,
        out_dir=out_dir / "non_obstacle",
        prefix="non_obstacle",
    )

    print(f"Saved sample visualization: {out_dir}")
    print(f"Obstacle samples:          {out_dir / 'obstacle'}")
    print(f"Non-obstacle samples:      {out_dir / 'non_obstacle'}")


if __name__ == "__main__":
    main()
