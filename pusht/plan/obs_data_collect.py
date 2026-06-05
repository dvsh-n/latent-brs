#!/usr/bin/env python3
"""Collect a balanced PushT insertion obstacle dataset from analytic geometry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    PUSHT_AGENT_RADIUS,
    PUSHT_WALL_MAX,
    PUSHT_WALL_MIN,
    PUSHT_WALL_RADIUS,
    get_pusht_goal_pose,
    make_pusht_env,
    reset_pusht_env_to_state,
)

DEFAULT_OUT_DIR = "pusht/plan/obstacle_data_insert"
DEFAULT_DATASET_NAME = "obstacle_classifier_data.pt"
DEFAULT_DIAGNOSTIC_PLOT_NAME = "balanced_insert_obstacle_dataset.png"
DEFAULT_SAMPLES_PER_CLASS = 20_000
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRAIN_FRACTION = 0.9
DEFAULT_COLLISION_MARGIN = 0.0
DEFAULT_INSERTION_VISUAL_BUFFER = 12.0
DEFAULT_INSERTION_VISUAL_THICKNESS = 80.0
DEFAULT_INSERTION_DISTANCE_RANGE = (0.0, 125.0)
DEFAULT_LATERAL_RANGE = (-15.0, 15.0)
DEFAULT_THETA_DEG_RANGE = (-22.5, 22.5)
PUSHT_CANVAS_SIZE = 512.0
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
TEE_BAR_X_MIN = -TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_X_MAX = TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_Y_MIN = 0.0
TEE_BAR_Y_MAX = TEE_SCALE
TEE_STEM_X_MIN = -TEE_SCALE / 2.0
TEE_STEM_X_MAX = TEE_SCALE / 2.0
TEE_STEM_Y_MIN = TEE_SCALE
TEE_STEM_Y_MAX = TEE_LENGTH * TEE_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--collision-margin", type=float, default=DEFAULT_COLLISION_MARGIN)
    parser.add_argument("--insertion-visual-buffer", type=float, default=DEFAULT_INSERTION_VISUAL_BUFFER)
    parser.add_argument("--insertion-visual-thickness", type=float, default=DEFAULT_INSERTION_VISUAL_THICKNESS)
    parser.add_argument(
        "--insertion-distance-range",
        type=float,
        nargs=2,
        default=DEFAULT_INSERTION_DISTANCE_RANGE,
        metavar=("MIN", "MAX"),
        help="Positive distance from the goal T along the insertion approach axis.",
    )
    parser.add_argument("--lateral-range", type=float, nargs=2, default=DEFAULT_LATERAL_RANGE)
    parser.add_argument("--theta-deg-range", type=float, nargs=2, default=DEFAULT_THETA_DEG_RANGE)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--max-attempts", type=int, default=20_000_000)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def rotation_matrix(theta: float | np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def insertion_visual_rects(*, buffer: float, thickness: float, margin: float = 0.0) -> list[tuple[float, float, float, float]]:
    buffer = float(buffer)
    thickness = float(thickness)
    margin = float(margin)
    stem_xmin = -0.5 * TEE_SCALE
    stem_xmax = 0.5 * TEE_SCALE
    cap_ymax = TEE_SCALE
    stem_ymax = TEE_LENGTH * TEE_SCALE

    inner_xmin = stem_xmin - buffer
    inner_xmax = stem_xmax + buffer
    inner_ymin = cap_ymax + buffer
    inner_ymax = stem_ymax + buffer
    outer_xmin = inner_xmin - thickness
    outer_xmax = inner_xmax + thickness
    outer_ymax = inner_ymax + thickness

    return [
        (outer_xmin - margin, inner_xmin + margin, inner_ymin - margin, outer_ymax + margin),
        (inner_xmax - margin, outer_xmax + margin, inner_ymin - margin, outer_ymax + margin),
        (outer_xmin - margin, outer_xmax + margin, inner_ymax - margin, outer_ymax + margin),
    ]


def rect_polygon(rect: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = rect
    return np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64)


def tee_part_rects() -> list[tuple[float, float, float, float]]:
    return [
        (TEE_BAR_X_MIN, TEE_BAR_X_MAX, TEE_BAR_Y_MIN, TEE_BAR_Y_MAX),
        (TEE_STEM_X_MIN, TEE_STEM_X_MAX, TEE_STEM_Y_MIN, TEE_STEM_Y_MAX),
    ]


def transform_tee_part_to_goal_local(block_goal_local: np.ndarray, dtheta: float, rect: tuple[float, float, float, float]) -> np.ndarray:
    corners = rect_polygon(rect)
    rot = rotation_matrix(float(dtheta))
    return np.asarray(block_goal_local, dtype=np.float64).reshape(2) + corners @ rot.T


def projections_overlap(poly_a: np.ndarray, poly_b: np.ndarray, axis: np.ndarray) -> bool:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return True
    unit = axis / axis_norm
    a = poly_a @ unit
    b = poly_b @ unit
    return bool(np.max(a) >= np.min(b) and np.max(b) >= np.min(a))


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    for poly in (poly_a, poly_b):
        edges = np.roll(poly, shift=-1, axis=0) - poly
        for edge in edges:
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            if not projections_overlap(poly_a, poly_b, axis):
                return False
    return True


def classify_insert_obstacle(
    block_goal_local: np.ndarray,
    dtheta: np.ndarray,
    *,
    buffer: float,
    thickness: float,
    collision_margin: float,
) -> np.ndarray:
    obstacle_polys = [rect_polygon(rect) for rect in insertion_visual_rects(buffer=buffer, thickness=thickness, margin=collision_margin)]
    labels = np.zeros((block_goal_local.shape[0],), dtype=np.int64)
    for index, (xy_local, theta) in enumerate(zip(block_goal_local, dtheta, strict=True)):
        tee_polys = [transform_tee_part_to_goal_local(xy_local, float(theta), rect) for rect in tee_part_rects()]
        labels[index] = int(any(polygons_overlap(tee_poly, obstacle_poly) for tee_poly in tee_polys for obstacle_poly in obstacle_polys))
    return labels


def sample_lateral_offsets(
    count: int,
    *,
    lateral_range: tuple[float, float],
    buffer: float,
    thickness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rects = insertion_visual_rects(buffer=buffer, thickness=thickness, margin=0.0)
    boundaries = np.asarray([rects[0][1], rects[1][0], rects[1][1], rects[0][0]], dtype=np.float64)
    modes = rng.choice(3, size=count, p=np.asarray([0.50, 0.25, 0.25]))
    lateral = np.empty((count,), dtype=np.float64)

    boundary_mask = modes == 0
    if np.any(boundary_mask):
        centers = rng.choice(boundaries, size=int(np.sum(boundary_mask)))
        lateral[boundary_mask] = centers + rng.normal(0.0, 12.0, size=int(np.sum(boundary_mask)))

    centered_mask = modes == 1
    if np.any(centered_mask):
        safe_half_width = 0.5 * TEE_SCALE + float(buffer)
        lateral[centered_mask] = rng.uniform(-safe_half_width, safe_half_width, size=int(np.sum(centered_mask)))

    broad_mask = modes == 2
    if np.any(broad_mask):
        lateral[broad_mask] = rng.uniform(float(lateral_range[0]), float(lateral_range[1]), size=int(np.sum(broad_mask)))

    return np.clip(lateral, float(lateral_range[0]), float(lateral_range[1]))


def sample_block_goal_local(
    count: int,
    *,
    insertion_distance_range: tuple[float, float],
    lateral_range: tuple[float, float],
    theta_deg_range: tuple[float, float],
    buffer: float,
    thickness: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = sample_lateral_offsets(count, lateral_range=lateral_range, buffer=buffer, thickness=thickness, rng=rng)
    insertion_distance = rng.uniform(
        float(insertion_distance_range[0]),
        float(insertion_distance_range[1]),
        size=count,
    )
    y = -insertion_distance
    theta = np.deg2rad(rng.uniform(float(theta_deg_range[0]), float(theta_deg_range[1]), size=count))
    return np.stack((x, y), axis=1).astype(np.float64), theta.astype(np.float64)


def sample_pusher_block_local(count: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(TEE_BAR_X_MIN, TEE_BAR_X_MAX, size=count)
    y = np.full((count,), TEE_BAR_Y_MIN - float(PUSHT_AGENT_RADIUS), dtype=np.float64)
    return np.stack((x, y), axis=1)


def goal_local_to_world(goal_pose: np.ndarray, xy_local: np.ndarray) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    rot = rotation_matrix(float(goal_pose[2]))
    return goal_pose[:2] + np.asarray(xy_local, dtype=np.float64) @ rot.T


def build_env_states(goal_pose: np.ndarray, block_goal_local: np.ndarray, dtheta: np.ndarray, pusher_block_local: np.ndarray) -> np.ndarray:
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    block_xy = goal_local_to_world(goal_pose, block_goal_local)
    block_theta = goal_pose[2] + dtheta
    agent_xy = np.empty_like(block_xy)
    for index, local_agent in enumerate(pusher_block_local):
        agent_xy[index] = block_xy[index] + local_agent @ rotation_matrix(float(block_theta[index])).T
    return np.column_stack(
        (
            agent_xy[:, 0],
            agent_xy[:, 1],
            block_xy[:, 0],
            block_xy[:, 1],
            block_theta,
            np.zeros((block_xy.shape[0], 2), dtype=np.float64),
        )
    ).astype(np.float64)


def valid_agent_positions(states: np.ndarray) -> np.ndarray:
    min_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    return np.all((states[:, :2] >= min_coord) & (states[:, :2] <= max_coord), axis=1)


def sample_labeled_states(
    label: int,
    count: int,
    *,
    goal_pose: np.ndarray,
    insertion_distance_range: tuple[float, float],
    lateral_range: tuple[float, float],
    theta_deg_range: tuple[float, float],
    buffer: float,
    thickness: float,
    collision_margin: float,
    chunk_size: int,
    max_attempts: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    kept_states: list[np.ndarray] = []
    kept_block_goal_local: list[np.ndarray] = []
    kept_pusher_block_local: list[np.ndarray] = []
    kept_dtheta: list[np.ndarray] = []
    kept_overlap_no_margin: list[np.ndarray] = []
    attempts = 0
    progress = tqdm(total=count, desc=f"Sampling label {label}", unit="state")
    try:
        while sum(batch.shape[0] for batch in kept_states) < count:
            attempts += int(chunk_size)
            if attempts > int(max_attempts):
                raise RuntimeError(f"Exceeded max attempts while sampling label {label}.")

            block_goal_local, dtheta = sample_block_goal_local(
                int(chunk_size),
                insertion_distance_range=insertion_distance_range,
                lateral_range=lateral_range,
                theta_deg_range=theta_deg_range,
                buffer=buffer,
                thickness=thickness,
                rng=rng,
            )
            pusher_block_local = sample_pusher_block_local(int(chunk_size), rng)
            states = build_env_states(goal_pose, block_goal_local, dtheta, pusher_block_local)
            valid = valid_agent_positions(states)
            labels = classify_insert_obstacle(
                block_goal_local,
                dtheta,
                buffer=buffer,
                thickness=thickness,
                collision_margin=collision_margin,
            )
            overlap_no_margin = classify_insert_obstacle(
                block_goal_local,
                dtheta,
                buffer=buffer,
                thickness=thickness,
                collision_margin=0.0,
            )
            keep = valid & (labels == int(label))
            if not np.any(keep):
                continue
            kept_states.append(states[keep])
            kept_block_goal_local.append(block_goal_local[keep])
            kept_pusher_block_local.append(pusher_block_local[keep])
            kept_dtheta.append(dtheta[keep])
            kept_overlap_no_margin.append(overlap_no_margin[keep])
            progress.update(min(int(np.sum(keep)), count - progress.n))
    finally:
        progress.close()

    return {
        "task_target": np.concatenate(kept_states, axis=0)[:count],
        "block_goal_local": np.concatenate(kept_block_goal_local, axis=0)[:count],
        "pusher_block_local": np.concatenate(kept_pusher_block_local, axis=0)[:count],
        "dtheta": np.concatenate(kept_dtheta, axis=0)[:count],
        "overlap_no_margin": np.concatenate(kept_overlap_no_margin, axis=0)[:count],
    }


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size == 1:
        return shuffled.copy(), np.zeros((0,), dtype=np.int64)
    train_count = int(np.floor(float(train_fraction) * float(shuffled.size)))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    return shuffled[:train_count], shuffled[train_count:]


def build_balanced_dataset(obstacle: dict[str, np.ndarray], non_obstacle: dict[str, np.ndarray], rng: np.random.Generator, train_fraction: float) -> dict[str, np.ndarray]:
    labels = np.concatenate(
        (
            np.ones((obstacle["task_target"].shape[0],), dtype=np.int64),
            np.zeros((non_obstacle["task_target"].shape[0],), dtype=np.int64),
        ),
        axis=0,
    )
    obstacle_idx = np.flatnonzero(labels == 1)
    non_obstacle_idx = np.flatnonzero(labels == 0)
    obstacle_train, obstacle_cal = split_indices(obstacle_idx, rng, train_fraction)
    non_obstacle_train, non_obstacle_cal = split_indices(non_obstacle_idx, rng, train_fraction)
    train_idx = np.concatenate((obstacle_train, non_obstacle_train), axis=0)
    calibration_idx = np.concatenate((obstacle_cal, non_obstacle_cal), axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(calibration_idx)
    return {
        "task_target": np.concatenate((obstacle["task_target"], non_obstacle["task_target"]), axis=0).astype(np.float32),
        "label": labels.astype(np.int64),
        "block_goal_local": np.concatenate((obstacle["block_goal_local"], non_obstacle["block_goal_local"]), axis=0).astype(np.float32),
        "pusher_block_local": np.concatenate((obstacle["pusher_block_local"], non_obstacle["pusher_block_local"]), axis=0).astype(np.float32),
        "dtheta": np.concatenate((obstacle["dtheta"], non_obstacle["dtheta"]), axis=0).astype(np.float32),
        "overlap_no_margin": np.concatenate((obstacle["overlap_no_margin"], non_obstacle["overlap_no_margin"]), axis=0).astype(np.int64),
        "train_idx": train_idx.astype(np.int64),
        "calibration_idx": calibration_idx.astype(np.int64),
    }


def render_dataset_images(states: np.ndarray, *, env_id: str, goal_pose: np.ndarray, width: int, height: int) -> np.ndarray:
    env = make_pusht_env(
        env_id,
        obs_type="pixels",
        render_mode="rgb_array",
        max_episode_steps=300,
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
        hide_target=False,
    )
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=0)
        base_env = getattr(env, "unwrapped", env)
        if hasattr(base_env, "goal_pose"):
            base_env.goal_pose = np.asarray(goal_pose, dtype=np.float32).copy()
        for state in tqdm(states, desc="Rendering dataset images", unit="image"):
            frames.append(reset_pusht_env_to_state(base_env, state))
    finally:
        env.close()
    return np.stack(frames, axis=0).astype(np.uint8)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_diagnostic(path: Path, dataset: dict[str, np.ndarray], *, buffer: float, thickness: float) -> None:
    labels = dataset["label"]
    xy = dataset["block_goal_local"]
    obstacle_mask = labels == 1
    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=180)
    for rect in insertion_visual_rects(buffer=buffer, thickness=thickness, margin=0.0):
        poly = rect_polygon(rect)
        closed = np.vstack((poly, poly[0]))
        ax.plot(closed[:, 0], closed[:, 1], color="#d55e00", linewidth=1.2)
        ax.fill(poly[:, 0], poly[:, 1], color="#d55e00", alpha=0.08)
    ax.scatter(xy[~obstacle_mask, 0], xy[~obstacle_mask, 1], s=4.0, c="#009e73", alpha=0.35, edgecolors="none", label="non-obstacle")
    ax.scatter(xy[obstacle_mask, 0], xy[obstacle_mask, 1], s=4.0, c="#d55e00", alpha=0.35, edgecolors="none", label="obstacle")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("goal-local lateral offset")
    ax.set_ylabel("goal-local insertion offset")
    ax.set_title("PushT insertion obstacle samples")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    if float(args.collision_margin) < 0.0:
        raise ValueError("--collision-margin must be non-negative.")
    if int(args.chunk_size) <= 0:
        raise ValueError("--chunk-size must be positive.")
    for name in ("insertion_distance_range", "lateral_range", "theta_deg_range"):
        values = np.asarray(getattr(args, name), dtype=np.float64).reshape(-1)
        if values.shape != (2,) or values[1] <= values[0]:
            raise ValueError(f"--{name.replace('_', '-')} must be MIN MAX with MAX > MIN.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_env = make_pusht_env(
        args.env_id,
        obs_type="pixels",
        render_mode="rgb_array",
        max_episode_steps=300,
        observation_width=int(args.width),
        observation_height=int(args.height),
        visualization_width=int(args.width),
        visualization_height=int(args.height),
        hide_target=False,
    )
    try:
        setup_env.reset(seed=0)
        goal_pose = get_pusht_goal_pose(setup_env)
    finally:
        setup_env.close()

    common_sampling = {
        "goal_pose": goal_pose,
        "insertion_distance_range": tuple(float(v) for v in args.insertion_distance_range),
        "lateral_range": tuple(float(v) for v in args.lateral_range),
        "theta_deg_range": tuple(float(v) for v in args.theta_deg_range),
        "buffer": float(args.insertion_visual_buffer),
        "thickness": float(args.insertion_visual_thickness),
        "collision_margin": float(args.collision_margin),
        "chunk_size": int(args.chunk_size),
        "max_attempts": int(args.max_attempts),
        "rng": rng,
    }
    obstacle = sample_labeled_states(1, int(args.samples_per_class), **common_sampling)
    non_obstacle = sample_labeled_states(0, int(args.samples_per_class), **common_sampling)
    dataset = build_balanced_dataset(obstacle, non_obstacle, rng, float(args.train_fraction))
    dataset_pixels = render_dataset_images(
        dataset["task_target"].astype(np.float64),
        env_id=str(args.env_id),
        goal_pose=goal_pose,
        width=int(args.width),
        height=int(args.height),
    )
    dataset["pixels"] = dataset_pixels

    save_diagnostic(
        out_dir / DEFAULT_DIAGNOSTIC_PLOT_NAME,
        dataset,
        buffer=float(args.insertion_visual_buffer),
        thickness=float(args.insertion_visual_thickness),
    )
    save_rgb_image(out_dir / "sample_obstacle.png", dataset_pixels[int(np.flatnonzero(dataset["label"] == 1)[0])])
    save_rgb_image(out_dir / "sample_non_obstacle.png", dataset_pixels[int(np.flatnonzero(dataset["label"] == 0)[0])])

    artifact = {
        "metadata": {
            "seed": int(args.seed),
            "env_id": str(args.env_id),
            "image_width": int(args.width),
            "image_height": int(args.height),
            "train_fraction": float(args.train_fraction),
            "calibration_fraction": float(1.0 - float(args.train_fraction)),
            "goal_pose": goal_pose.astype(np.float32),
            "label_convention": "label=1 is obstacle/unsafe, label=0 is non-obstacle/safe",
            "rendering": "normal PushT render without orange insertion obstacle overlay",
            "label_rule": "analytic overlap between PushT T geometry and goal-local insertion guide rectangles",
            "collision_margin": float(args.collision_margin),
            "insertion_visual_buffer": float(args.insertion_visual_buffer),
            "insertion_visual_thickness": float(args.insertion_visual_thickness),
            "insertion_distance_range": np.asarray(args.insertion_distance_range, dtype=np.float32),
            "y_local_range": -np.asarray(args.insertion_distance_range, dtype=np.float32)[::-1],
            "lateral_range": np.asarray(args.lateral_range, dtype=np.float32),
            "theta_deg_range": np.asarray(args.theta_deg_range, dtype=np.float32),
            "samples_per_class": int(args.samples_per_class),
            "balanced_total_count": int(dataset["label"].shape[0]),
            "tee_part_rects": np.asarray(tee_part_rects(), dtype=np.float32),
            "insertion_visual_rects": np.asarray(
                insertion_visual_rects(
                    buffer=float(args.insertion_visual_buffer),
                    thickness=float(args.insertion_visual_thickness),
                    margin=0.0,
                ),
                dtype=np.float32,
            ),
        },
        "dataset": {
            "pixels": dataset["pixels"].astype(np.uint8),
            "task_target": dataset["task_target"].astype(np.float32),
            "label": dataset["label"].astype(np.int64),
            "block_goal_local": dataset["block_goal_local"].astype(np.float32),
            "pusher_block_local": dataset["pusher_block_local"].astype(np.float32),
            "dtheta": dataset["dtheta"].astype(np.float32),
            "overlap_no_margin": dataset["overlap_no_margin"].astype(np.int64),
            "train_idx": dataset["train_idx"].astype(np.int64),
            "calibration_idx": dataset["calibration_idx"].astype(np.int64),
        },
    }
    torch.save(artifact, out_dir / DEFAULT_DATASET_NAME, pickle_protocol=4)

    save_json(
        out_dir / "summary.json",
        {
            "out_dir": str(out_dir),
            "dataset_path": str(out_dir / DEFAULT_DATASET_NAME),
            "diagnostic_path": str(out_dir / DEFAULT_DIAGNOSTIC_PLOT_NAME),
            "counts": {
                "obstacle": int(np.sum(dataset["label"] == 1)),
                "non_obstacle": int(np.sum(dataset["label"] == 0)),
                "total": int(dataset["label"].shape[0]),
                "train": int(dataset["train_idx"].shape[0]),
                "calibration": int(dataset["calibration_idx"].shape[0]),
            },
            "goal_pose": goal_pose,
            "collision_margin": float(args.collision_margin),
            "sampling": {
                "samples_per_class": int(args.samples_per_class),
                "insertion_distance_range": args.insertion_distance_range,
                "y_local_range": (-np.asarray(args.insertion_distance_range, dtype=np.float64)[::-1]).tolist(),
                "lateral_range": args.lateral_range,
                "theta_deg_range": args.theta_deg_range,
                "chunk_size": int(args.chunk_size),
            },
        },
    )

    print(f"Saved diagnostic: {out_dir / DEFAULT_DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved dataset:    {out_dir / DEFAULT_DATASET_NAME}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
