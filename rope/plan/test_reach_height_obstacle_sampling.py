#!/usr/bin/env python3
"""Sample rope task states from a reach-height obstacle set and its complement."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from rope.plan import plan_ilqr_mpc as planner
from rope.shared.lab_env import LabEnv, TaskState

DEFAULT_OUT_DIR = "rope/plan/reach_height_obstacle_sampling"
DISABLE_SHADOWS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obstacle-count", "--sample-count", dest="obstacle_count", type=int, default=256)
    parser.add_argument("--outside-count", type=int, default=32)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=DISABLE_SHADOWS,
        help="Disable shadows for the saved renders.",
    )

    parser.add_argument("--reach-min", type=float, default=0.02)
    parser.add_argument("--reach-max", type=float, default=0.18)
    parser.add_argument("--min-height", type=float, default=1.26)
    parser.add_argument("--width-margin", type=float, default=0.03)
    parser.add_argument(
        "--outside-in-band-frac",
        "--outside-low-frac",
        dest="outside_in_band_frac",
        type=float,
        default=0.5,
        help="Fraction of outside samples drawn within the reach band at heights >= min_height.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def task_bounds_arrays(env: LabEnv) -> tuple[np.ndarray, np.ndarray]:
    bounds = env.task_bounds
    lower = np.array([bounds.reach[0], bounds.height[0], bounds.width[0]], dtype=np.float64)
    upper = np.array([bounds.reach[1], bounds.height[1], bounds.width[1]], dtype=np.float64)
    return lower, upper


def validate_obstacle_spec(lower: np.ndarray, upper: np.ndarray, args: argparse.Namespace) -> None:
    reach_min = float(args.reach_min)
    reach_max = float(args.reach_max)
    min_height = float(args.min_height)
    if not (lower[0] <= reach_min < reach_max <= upper[0]):
        raise ValueError(
            f"Reach band [{reach_min}, {reach_max}] must lie within task reach bounds [{lower[0]}, {upper[0]}]."
        )
    if not (lower[1] < min_height <= upper[1]):
        raise ValueError(
            f"Minimum height {min_height} must lie within task height bounds ({lower[1]}, {upper[1]}]."
        )
    if not (0.0 <= float(args.outside_in_band_frac) <= 1.0):
        raise ValueError(f"outside_in_band_frac must be in [0, 1], got {args.outside_in_band_frac}.")


def clipped_width_interval(lower: np.ndarray, upper: np.ndarray, margin: float) -> tuple[float, float]:
    lo = float(lower[2] + margin)
    hi = float(upper[2] - margin)
    if lo >= hi:
        return float(lower[2]), float(upper[2])
    return lo, hi


def sample_obstacle_states(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reach_min: float,
    reach_max: float,
    min_height: float,
    width_margin: float,
    count: int,
) -> np.ndarray:
    width_lo, width_hi = clipped_width_interval(lower, upper, width_margin)
    height_hi = min_height
    if height_hi <= lower[1]:
        raise ValueError(
            f"No feasible obstacle states: min_height={min_height} leaves no room above task height lower bound."
        )
    reach = rng.uniform(reach_min, reach_max, size=count)
    height = rng.uniform(lower[1], height_hi, size=count)
    width = rng.uniform(width_lo, width_hi, size=count)
    return np.stack((reach, height, width), axis=1).astype(np.float64)


def sample_outside_states(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reach_min: float,
    reach_max: float,
    min_height: float,
    width_margin: float,
    count: int,
    outside_in_band_frac: float,
) -> np.ndarray:
    width_lo, width_hi = clipped_width_interval(lower, upper, width_margin)
    in_band_count = int(round(count * outside_in_band_frac))
    out_of_band_count = count - in_band_count
    pieces: list[np.ndarray] = []

    height_lo = min_height
    if in_band_count > 0:
        reach = rng.uniform(reach_min, reach_max, size=in_band_count)
        height = rng.uniform(height_lo, upper[1], size=in_band_count)
        width = rng.uniform(width_lo, width_hi, size=in_band_count)
        pieces.append(np.stack((reach, height, width), axis=1))

    if out_of_band_count > 0:
        left_width = max(reach_min - lower[0], 0.0)
        right_width = max(upper[0] - reach_max, 0.0)
        if left_width <= 1e-8 and right_width <= 1e-8:
            raise ValueError("No feasible out-of-band outside states: reach obstacle covers the full task reach range.")
        side_pick = rng.uniform(0.0, left_width + right_width, size=out_of_band_count)
        reach = np.empty((out_of_band_count,), dtype=np.float64)
        left_mask = side_pick < left_width
        right_mask = ~left_mask
        if np.any(left_mask):
            reach[left_mask] = rng.uniform(lower[0], reach_min, size=int(np.sum(left_mask)))
        if np.any(right_mask):
            reach[right_mask] = rng.uniform(reach_max, upper[0], size=int(np.sum(right_mask)))
        height = rng.uniform(lower[1], upper[1], size=out_of_band_count)
        width = rng.uniform(width_lo, width_hi, size=out_of_band_count)
        pieces.append(np.stack((reach, height, width), axis=1))

    merged = np.concatenate(pieces, axis=0).astype(np.float64)
    if merged.shape[0] != count:
        raise RuntimeError(f"Outside sampler produced {merged.shape[0]} states, expected {count}.")
    return merged


def states_to_qpos_and_control(env: LabEnv, task_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos_batch: list[np.ndarray] = []
    control_batch: list[np.ndarray] = []
    for state_vec in task_states:
        task_state = TaskState.from_array(state_vec)
        env.reset(task_state)
        qpos_batch.append(env.data.qpos.copy().astype(np.float32))
        control_batch.append(env.data.ctrl.copy().astype(np.float32))
    return np.stack(qpos_batch, axis=0), np.stack(control_batch, axis=0)


def render_state_batch(
    env: LabEnv,
    renderer: mujoco.Renderer,
    task_states: np.ndarray,
    qpos_batch: np.ndarray,
    control_batch: np.ndarray,
    *,
    camera_name: str,
    elapsed_time: float,
    disable_shadows: bool,
) -> np.ndarray:
    camera_id = env.model.camera(camera_name).id
    qvel = np.zeros((qpos_batch.shape[1],), dtype=np.float32)
    frames: list[np.ndarray] = []
    for state_vec, qpos, control in zip(task_states, qpos_batch, control_batch, strict=True):
        frame, _ = planner.reset_env_to_state(
            env,
            renderer,
            qpos=qpos,
            qvel=qvel,
            control=control,
            task_target=np.asarray(state_vec, dtype=np.float32),
            camera_id=camera_id,
            elapsed_time=elapsed_time,
            disable_shadows=disable_shadows,
        )
        frames.append(frame.copy())
    return np.stack(frames, axis=0)


def make_image_grid(images: np.ndarray, *, columns: int) -> np.ndarray:
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError(f"Expected non-empty image batch, got shape {images.shape}.")
    columns = max(1, min(columns, images.shape[0]))
    rows = int(math.ceil(images.shape[0] / columns))
    height, width, channels = images.shape[1:]
    grid = np.full((rows * height, columns * width, channels), 255, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return grid


def representative_indices(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count <= 9:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=9, dtype=np.int64)


def save_workspace_plot(
    *,
    out_path: Path,
    lower: np.ndarray,
    upper: np.ndarray,
    reach_min: float,
    reach_max: float,
    min_height: float,
    obstacle_states: np.ndarray,
    outside_states: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=160)
    obstacle_rect = plt.Rectangle(
        (reach_min, float(lower[1])),
        reach_max - reach_min,
        min_height - float(lower[1]),
        facecolor="#f5d8c2",
        edgecolor="#b24700",
        linewidth=2.0,
        alpha=0.55,
        label="obstacle region",
    )
    ax.add_patch(obstacle_rect)

    ax.scatter(
        outside_states[:, 0],
        outside_states[:, 1],
        s=22,
        c="#0072b2",
        alpha=0.55,
        label="outside samples",
    )
    ax.scatter(
        obstacle_states[:, 0],
        obstacle_states[:, 1],
        s=24,
        c="#d55e00",
        alpha=0.55,
        label="obstacle samples",
    )

    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.grid(alpha=0.2)
    ax.set_xlabel("reach")
    ax.set_ylabel("height")
    ax.set_title("Reach-height obstacle region and sampled states")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = LabEnv()
    lower, upper = task_bounds_arrays(env)
    validate_obstacle_spec(lower, upper, args)

    obstacle_states = sample_obstacle_states(
        rng,
        lower,
        upper,
        reach_min=float(args.reach_min),
        reach_max=float(args.reach_max),
        min_height=float(args.min_height),
        width_margin=float(args.width_margin),
        count=int(args.obstacle_count),
    )
    outside_states = sample_outside_states(
        rng,
        lower,
        upper,
        reach_min=float(args.reach_min),
        reach_max=float(args.reach_max),
        min_height=float(args.min_height),
        width_margin=float(args.width_margin),
        count=int(args.outside_count),
        outside_in_band_frac=float(args.outside_in_band_frac),
    )

    obstacle_qpos, obstacle_control = states_to_qpos_and_control(env, obstacle_states)
    outside_qpos, outside_control = states_to_qpos_and_control(env, outside_states)

    rep_idx = representative_indices(obstacle_states.shape[0])
    rep_states = obstacle_states[rep_idx]
    rep_qpos = obstacle_qpos[rep_idx]
    rep_control = obstacle_control[rep_idx]

    with mujoco.Renderer(env.model, height=int(args.height), width=int(args.width)) as renderer:
        front_obstacle_reps = render_state_batch(
            env,
            renderer,
            rep_states,
            rep_qpos,
            rep_control,
            camera_name="video_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
        )
        top_obstacle_reps = render_state_batch(
            env,
            renderer,
            rep_states,
            rep_qpos,
            rep_control,
            camera_name="ceiling_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
        )

    planner.save_rgb_image(out_dir / "front_obstacle_samples_grid.png", make_image_grid(front_obstacle_reps, columns=3))
    planner.save_rgb_image(out_dir / "top_obstacle_samples_grid.png", make_image_grid(top_obstacle_reps, columns=3))

    save_workspace_plot(
        out_path=out_dir / "workspace_samples.png",
        lower=lower,
        upper=upper,
        reach_min=float(args.reach_min),
        reach_max=float(args.reach_max),
        min_height=float(args.min_height),
        obstacle_states=obstacle_states,
        outside_states=outside_states,
    )

    torch.save(
        {
            "metadata": {
                "seed": int(args.seed),
                "width": int(args.width),
                "height": int(args.height),
                "disable_shadows": bool(args.disable_shadows),
                "obstacle_count_valid": int(obstacle_states.shape[0]),
                "outside_count_valid": int(outside_states.shape[0]),
            },
            "episode_data": {
                "qpos": np.concatenate((obstacle_qpos, outside_qpos), axis=0),
                "qvel": np.zeros((obstacle_qpos.shape[0] + outside_qpos.shape[0], obstacle_qpos.shape[1]), dtype=np.float32),
                "control": np.concatenate((obstacle_control, outside_control), axis=0),
                "task_target": np.concatenate((obstacle_states, outside_states), axis=0).astype(np.float32),
                "labels": np.concatenate(
                    (
                        np.ones((obstacle_states.shape[0],), dtype=np.int64),
                        np.zeros((outside_states.shape[0],), dtype=np.int64),
                    ),
                    axis=0,
                ),
            },
            "planner_data": {
                "obstacle_task_states": obstacle_states.astype(np.float32),
                "obstacle_qpos": obstacle_qpos.astype(np.float32),
                "obstacle_control": obstacle_control.astype(np.float32),
                "outside_task_states": outside_states.astype(np.float32),
                "outside_qpos": outside_qpos.astype(np.float32),
                "outside_control": outside_control.astype(np.float32),
                "reach_band": np.array([float(args.reach_min), float(args.reach_max)], dtype=np.float32),
                "min_height": float(args.min_height),
            },
        },
        out_dir / "obstacle_samples.pt",
    )

    save_json(
        out_dir / "summary.json",
        {
            "seed": int(args.seed),
            "obstacle_count_requested": int(args.obstacle_count),
            "obstacle_count_valid": int(obstacle_states.shape[0]),
            "outside_count_requested": int(args.outside_count),
            "outside_count_valid": int(outside_states.shape[0]),
            "outside_in_band_frac": float(args.outside_in_band_frac),
            "reach_min": float(args.reach_min),
            "reach_max": float(args.reach_max),
            "min_height": float(args.min_height),
            "width_margin": float(args.width_margin),
            "task_lower": lower.tolist(),
            "task_upper": upper.tolist(),
            "obstacle_min_height_observed": float(np.min(obstacle_states[:, 1])) if obstacle_states.size > 0 else None,
            "obstacle_max_height_observed": float(np.max(obstacle_states[:, 1])) if obstacle_states.size > 0 else None,
            "outside_min_height_observed": float(np.min(outside_states[:, 1])) if outside_states.size > 0 else None,
            "outside_max_height_observed": float(np.max(outside_states[:, 1])) if outside_states.size > 0 else None,
            "outside_task_states": outside_states.tolist(),
        },
    )

    print(f"Saved workspace: {out_dir / 'workspace_samples.png'}")
    print(f"Saved front grid: {out_dir / 'front_obstacle_samples_grid.png'}")
    print(f"Saved top grid:   {out_dir / 'top_obstacle_samples_grid.png'}")
    print(f"Saved payload:    {out_dir / 'obstacle_samples.pt'}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
