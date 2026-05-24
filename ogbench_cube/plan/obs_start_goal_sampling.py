#!/usr/bin/env python3
"""Sample one OGBench start/goal pair around a center obstacle and save a 3D plot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

USER_SITE_FRAGMENT = ".local/lib/python"
sys.path = [path for path in sys.path if USER_SITE_FRAGMENT not in path]

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DEFAULT_OUT_DIR = Path("ogbench_cube/plan/random_endpoint_pairs")
DEFAULT_PLOT_NAME = "start_goal_speed_bump.png"
DEFAULT_DATASET_NAME = "start_goal_speed_bump.pt"
DEFAULT_NUM_POINTS = 256
FIXED_YAW = 0.0

X_BOUNDS = (0.30, 0.50)
Y_BOUNDS = (-0.25, 0.25)
TABLE_Z = 0.02
OBSTACLE_BASE_Z = 0.0

LEFT_BAND = (-0.2, -0.12)
RIGHT_BAND = (0.12, 0.2)

OBSTACLE_Y_BOUNDS = (-0.06, 0.06)
OBSTACLE_PEAK_Z = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot-name", default=DEFAULT_PLOT_NAME)
    parser.add_argument("--out-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    return parser.parse_args()


def sample_point(
    rng: np.random.Generator,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(float(x_bounds[0]), float(x_bounds[1])),
            rng.uniform(float(y_bounds[0]), float(y_bounds[1])),
            TABLE_Z,
        ],
        dtype=np.float64,
    )


def sample_points(
    rng: np.random.Generator,
    count: int,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> np.ndarray:
    if count <= 0:
        raise ValueError("num-points must be positive.")
    x = rng.uniform(float(x_bounds[0]), float(x_bounds[1]), size=int(count))
    y = rng.uniform(float(y_bounds[0]), float(y_bounds[1]), size=int(count))
    z = np.full((int(count),), TABLE_Z, dtype=np.float64)
    return np.stack((x, y, z), axis=1)


def half_ellipse_height(
    y_values: np.ndarray,
    *,
    y_bounds: tuple[float, float],
    base_z: float,
    peak_z: float,
) -> np.ndarray:
    y_values = np.asarray(y_values, dtype=np.float64)
    center_y = 0.5 * (float(y_bounds[0]) + float(y_bounds[1]))
    half_width = 0.5 * (float(y_bounds[1]) - float(y_bounds[0]))
    normalized = (y_values - center_y) / half_width
    profile = np.sqrt(np.clip(1.0 - normalized**2, 0.0, None))
    return float(base_z) + (float(peak_z) - float(base_z)) * profile


def save_plot(
    path: Path,
    *,
    start_points: np.ndarray,
    goal_points: np.ndarray,
) -> None:
    x_values = np.linspace(float(X_BOUNDS[0]), float(X_BOUNDS[1]), num=80, dtype=np.float64)
    y_values = np.linspace(float(Y_BOUNDS[0]), float(Y_BOUNDS[1]), num=160, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    obstacle_y_values = np.linspace(
        float(OBSTACLE_Y_BOUNDS[0]),
        float(OBSTACLE_Y_BOUNDS[1]),
        num=80,
        dtype=np.float64,
    )
    obstacle_grid_x, obstacle_grid_y = np.meshgrid(x_values, obstacle_y_values)
    obstacle_z = half_ellipse_height(
        obstacle_grid_y,
        y_bounds=OBSTACLE_Y_BOUNDS,
        base_z=OBSTACLE_BASE_Z,
        peak_z=OBSTACLE_PEAK_Z,
    )

    plane_z = np.full_like(grid_x, TABLE_Z, dtype=np.float64)

    fig = plt.figure(figsize=(8.0, 6.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        grid_x,
        grid_y,
        plane_z,
        color="#d9d9d9",
        alpha=0.25,
        linewidth=0.0,
        antialiased=True,
        shade=False,
    )
    ax.plot_surface(
        obstacle_grid_x,
        obstacle_grid_y,
        obstacle_z,
        color="#cc7a00",
        alpha=0.75,
        linewidth=0.0,
        antialiased=True,
    )

    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        start_points[:, 2],
        color="#0072b2",
        s=8.0,
        depthshade=False,
        label="start",
    )
    ax.scatter(
        goal_points[:, 0],
        goal_points[:, 1],
        goal_points[:, 2],
        color="#d55e00",
        s=8.0,
        depthshade=False,
        label="goal",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(X_BOUNDS)
    ax.set_ylim(Y_BOUNDS)
    ax.set_zlim(0.0, 0.30)
    ax.view_init(elev=28, azim=-25)
    ax.legend(loc="upper left")
    ax.set_title("OGBench start/goal sampler with center speed-bump obstacle")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    # plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    start_points = sample_points(rng, args.num_points, x_bounds=X_BOUNDS, y_bounds=LEFT_BAND)
    goal_points = sample_points(rng, args.num_points, x_bounds=X_BOUNDS, y_bounds=RIGHT_BAND)

    plot_path = args.out_dir / args.plot_name
    out_path = args.out_dir / args.out_name

    save_plot(plot_path, start_points=start_points, goal_points=goal_points)
    payload = {
        "metadata": {
            "seed": int(args.seed),
            "num_points": int(args.num_points),
            "x_bounds": np.asarray(X_BOUNDS, dtype=np.float32),
            "y_bounds": np.asarray(Y_BOUNDS, dtype=np.float32),
            "table_z": float(TABLE_Z),
            "obstacle_base_z": float(OBSTACLE_BASE_Z),
            "left_band": np.asarray(LEFT_BAND, dtype=np.float32),
            "right_band": np.asarray(RIGHT_BAND, dtype=np.float32),
            "obstacle_y_bounds": np.asarray(OBSTACLE_Y_BOUNDS, dtype=np.float32),
            "obstacle_peak_z": float(OBSTACLE_PEAK_Z),
            "fixed_yaw": float(FIXED_YAW),
            "plot_path": str(plot_path),
        },
        "start": {
            "task_target": start_points.astype(np.float32),
            "yaw": np.full((int(args.num_points),), FIXED_YAW, dtype=np.float32),
        },
        "goal": {
            "task_target": goal_points.astype(np.float32),
            "yaw": np.full((int(args.num_points),), FIXED_YAW, dtype=np.float32),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Saved plot to {plot_path}")
    print(f"Saved dataset to {out_path}")
    print(f"sampled {start_points.shape[0]} start points and {goal_points.shape[0]} goal points")
    print(f"first start = {np.array2string(start_points[0], precision=4)}")
    print(f"first goal  = {np.array2string(goal_points[0], precision=4)}")


if __name__ == "__main__":
    main()
