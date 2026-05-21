#!/usr/bin/env python3
"""Sample rope task states using proxy midpoint height obstacle labels."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.shared.lab_env import BaseEnvConfig, LabEnv, TABLE_TOP_Z, TaskState

DEFAULT_OUT_DIR = "rope/plan/reach_height_obstacle_sampling"
DISABLE_SHADOWS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outside-count", type=int, default=64)
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
    parser.add_argument("--width-margin", type=float, default=0.03)
    parser.add_argument("--width-steps", type=int, default=5)
    parser.add_argument("--reach-steps", type=int, default=11)
    parser.add_argument("--height-steps", type=int, default=11)
    parser.add_argument(
        "--midpoint-clearance",
        type=float,
        default=0.15,
        help="Desired rope midpoint clearance above the table before adding the model buffer.",
    )
    parser.add_argument(
        "--midpoint-buffer",
        type=float,
        default=0.025,
        help="Extra model-error buffer added to the desired midpoint clearance threshold.",
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


def validate_args(lower: np.ndarray, upper: np.ndarray, args: argparse.Namespace) -> None:
    if not (lower[0] <= float(args.reach_min) < float(args.reach_max) <= upper[0]):
        raise ValueError(
            f"Reach band [{args.reach_min}, {args.reach_max}] must lie within task reach bounds [{lower[0]}, {upper[0]}]."
        )
    if int(args.width_steps) <= 0 or int(args.reach_steps) <= 0 or int(args.height_steps) <= 0:
        raise ValueError("Grid step counts must all be positive.")
    if int(args.outside_count) < 0:
        raise ValueError(f"outside_count must be non-negative, got {args.outside_count}.")
    if float(args.midpoint_clearance) < 0.0 or float(args.midpoint_buffer) < 0.0:
        raise ValueError("Midpoint clearance and midpoint buffer must be non-negative.")


def clipped_width_interval(lower: np.ndarray, upper: np.ndarray, margin: float) -> tuple[float, float]:
    lo = float(lower[2] + margin)
    hi = float(upper[2] - margin)
    if lo >= hi:
        return float(lower[2]), float(upper[2])
    return lo, hi


def build_in_band_grid(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reach_min: float,
    reach_max: float,
    width_margin: float,
    width_steps: int,
    reach_steps: int,
    height_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width_lo, width_hi = clipped_width_interval(lower, upper, width_margin)
    width_values = np.linspace(width_lo, width_hi, num=width_steps, dtype=np.float64)
    reach_values = np.linspace(reach_min, reach_max, num=reach_steps, dtype=np.float64)
    height_values = np.linspace(float(upper[1]), float(lower[1]), num=height_steps, dtype=np.float64)

    states: list[np.ndarray] = []
    for width in width_values:
        for reach in reach_values:
            for height in height_values:
                states.append(np.array([reach, height, width], dtype=np.float64))
    return np.stack(states, axis=0), width_values, reach_values, height_values


def sample_out_of_band_states(
    rng: np.random.Generator,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reach_min: float,
    reach_max: float,
    width_margin: float,
    count: int,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)

    width_lo, width_hi = clipped_width_interval(lower, upper, width_margin)
    left_width = max(reach_min - float(lower[0]), 0.0)
    right_width = max(float(upper[0]) - reach_max, 0.0)
    if left_width <= 1e-8 and right_width <= 1e-8:
        raise ValueError("No feasible out-of-band states: reach obstacle covers the full task reach range.")

    side_pick = rng.uniform(0.0, left_width + right_width, size=count)
    reach = np.empty((count,), dtype=np.float64)
    left_mask = side_pick < left_width
    right_mask = ~left_mask
    if np.any(left_mask):
        reach[left_mask] = rng.uniform(float(lower[0]), reach_min, size=int(np.sum(left_mask)))
    if np.any(right_mask):
        reach[right_mask] = rng.uniform(reach_max, float(upper[0]), size=int(np.sum(right_mask)))

    height = rng.uniform(float(lower[1]), float(upper[1]), size=count)
    width = rng.uniform(width_lo, width_hi, size=count)
    return np.stack((reach, height, width), axis=1).astype(np.float64)


def states_to_qpos_and_control(
    env: LabEnv,
    task_states: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if task_states.shape[0] == 0:
        return (
            np.zeros((0, env.model.nq), dtype=np.float32),
            np.zeros((0, env.model.nu), dtype=np.float32),
        )
    qpos_batch: list[np.ndarray] = []
    control_batch: list[np.ndarray] = []
    iterator = (
        tqdm(task_states, desc=progress_desc, unit="state", leave=False)
        if progress_desc is not None
        else task_states
    )
    for state_vec in iterator:
        task_state = TaskState.from_array(state_vec)
        env.reset(task_state)
        qpos_batch.append(env.data.qpos.copy().astype(np.float32))
        control_batch.append(env.data.ctrl.copy().astype(np.float32))
    return np.stack(qpos_batch, axis=0), np.stack(control_batch, axis=0)


def compute_proxy_midpoint_heights(
    proxy_env: LabEnv,
    task_states: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> np.ndarray:
    midpoint_heights = np.zeros((task_states.shape[0],), dtype=np.float64)
    iterator = (
        tqdm(enumerate(task_states), total=task_states.shape[0], desc=progress_desc, unit="state")
        if progress_desc is not None
        else enumerate(task_states)
    )
    for index, state_vec in iterator:
        proxy_env.reset(TaskState.from_array(state_vec))
        midpoint_heights[index] = proxy_env.get_proxy_rope_midpoint_height()
    return midpoint_heights


def compute_lowest_height_reach_curve(
    proxy_env: LabEnv,
    *,
    reach_values: np.ndarray,
    width: float,
    task_height: float,
    progress_desc: str | None = None,
) -> np.ndarray:
    states = np.stack(
        (
            np.asarray(reach_values, dtype=np.float64),
            np.full_like(reach_values, fill_value=task_height, dtype=np.float64),
            np.full_like(reach_values, fill_value=width, dtype=np.float64),
        ),
        axis=1,
    )
    return compute_proxy_midpoint_heights(proxy_env, states, progress_desc=progress_desc)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def render_rgb_frame(
    renderer: mujoco.Renderer,
    env: LabEnv,
    camera_id: int,
    *,
    disable_shadows: bool = False,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def reset_env_to_state(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
    camera_id: int,
    elapsed_time: float,
    disable_shadows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    del elapsed_time
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    return render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows), {}


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
    qvel = np.zeros((env.model.nv,), dtype=np.float32)
    frames: list[np.ndarray] = []
    for state_vec, qpos, control in zip(task_states, qpos_batch, control_batch, strict=True):
        frame, _ = reset_env_to_state(
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


def project_world_to_image(
    env: LabEnv,
    camera_id: int,
    world_points: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}.")

    cam_pos = env.data.cam_xpos[camera_id].astype(np.float64)
    cam_rot = env.data.cam_xmat[camera_id].reshape(3, 3).astype(np.float64)
    rel = points - cam_pos[None, :]
    cam_points = rel @ cam_rot

    depth = -cam_points[:, 2]
    valid = depth > 1e-6
    projected = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if not np.any(valid):
        return projected

    fovy = math.radians(float(env.model.cam_fovy[camera_id]))
    aspect = float(image_width) / float(image_height)
    tan_half_fovy = math.tan(0.5 * fovy)
    x_ndc = cam_points[valid, 0] / (depth[valid] * tan_half_fovy * aspect)
    y_ndc = cam_points[valid, 1] / (depth[valid] * tan_half_fovy)

    projected[valid, 0] = (0.5 + 0.5 * x_ndc) * float(image_width)
    projected[valid, 1] = (0.5 - 0.5 * y_ndc) * float(image_height)
    return projected


def overlay_proxy_rope(
    frame: np.ndarray,
    env: LabEnv,
    *,
    camera_id: int,
    midpoint_height: float,
    midpoint_cutoff: float,
    task_state: np.ndarray,
) -> np.ndarray:
    proxy_points = env.get_proxy_rope_points()
    midpoint = env.get_proxy_rope_midpoint()[None, :]
    pixel_points = project_world_to_image(
        env,
        camera_id,
        proxy_points,
        image_width=int(frame.shape[1]),
        image_height=int(frame.shape[0]),
    )
    midpoint_px = project_world_to_image(
        env,
        camera_id,
        midpoint,
        image_width=int(frame.shape[1]),
        image_height=int(frame.shape[0]),
    )[0]

    fig = plt.figure(figsize=(frame.shape[1] / 100.0, frame.shape[0] / 100.0), dpi=100)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(frame)
    valid = np.isfinite(pixel_points[:, 0]) & np.isfinite(pixel_points[:, 1])
    if np.any(valid):
        ax.plot(
            pixel_points[valid, 0],
            pixel_points[valid, 1],
            color="#39d0ff",
            linewidth=2.0,
            alpha=0.9,
        )
        ax.scatter(
            pixel_points[valid, 0],
            pixel_points[valid, 1],
            s=12.0,
            c="#a5f3ff",
            alpha=0.75,
        )
    if np.all(np.isfinite(midpoint_px)):
        midpoint_color = "#d55e00" if midpoint_height < midpoint_cutoff else "#009e73"
        ax.scatter(
            [midpoint_px[0]],
            [midpoint_px[1]],
            s=70.0,
            c=midpoint_color,
            edgecolors="white",
            linewidths=1.2,
            alpha=0.95,
        )
    task_state = np.asarray(task_state, dtype=np.float64)
    annotation = (
        f"r={task_state[0]:.3f} h={task_state[1]:.3f} w={task_state[2]:.3f}\n"
        f"mid z={midpoint_height:.3f} cutoff={midpoint_cutoff:.3f}"
    )
    ax.text(
        8.0,
        14.0,
        annotation,
        color="white",
        fontsize=10.0,
        va="top",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 4.0, "edgecolor": "none"},
    )
    ax.set_axis_off()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    overlaid = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
    plt.close(fig)
    return overlaid


def render_proxy_overlay_batch(
    env: LabEnv,
    renderer: mujoco.Renderer,
    task_states: np.ndarray,
    qpos_batch: np.ndarray,
    control_batch: np.ndarray,
    midpoint_heights: np.ndarray,
    *,
    camera_name: str,
    elapsed_time: float,
    disable_shadows: bool,
    midpoint_cutoff: float,
) -> np.ndarray:
    camera_id = env.model.camera(camera_name).id
    qvel = np.zeros((env.model.nv,), dtype=np.float32)
    frames: list[np.ndarray] = []
    for state_vec, qpos, control, midpoint_height in zip(
        task_states,
        qpos_batch,
        control_batch,
        midpoint_heights,
        strict=True,
    ):
        frame, _ = reset_env_to_state(
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
        frames.append(
            overlay_proxy_rope(
                frame.copy(),
                env,
                camera_id=camera_id,
                midpoint_height=float(midpoint_height),
                midpoint_cutoff=float(midpoint_cutoff),
                task_state=state_vec,
            )
        )
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


def choose_representative_indices(
    midpoint_heights: np.ndarray,
    labels: np.ndarray,
    midpoint_cutoff: float,
    *,
    count: int = 9,
) -> np.ndarray:
    total = midpoint_heights.shape[0]
    if total <= count:
        return np.arange(total, dtype=np.int64)

    chosen: list[int] = []
    signed_distance = midpoint_heights - midpoint_cutoff
    obstacle_idx = np.flatnonzero(labels == 1)
    safe_idx = np.flatnonzero(labels == 0)

    for pool in (
        obstacle_idx[np.argsort(np.abs(signed_distance[obstacle_idx]))],
        safe_idx[np.argsort(np.abs(signed_distance[safe_idx]))],
        np.argsort(midpoint_heights),
        np.argsort(-midpoint_heights),
    ):
        for index in pool:
            int_index = int(index)
            if int_index not in chosen:
                chosen.append(int_index)
            if len(chosen) >= count:
                return np.array(chosen, dtype=np.int64)

    return np.array(chosen[:count], dtype=np.int64)


def save_midpoint_slice_plot(
    *,
    out_path: Path,
    width_values: np.ndarray,
    reach_values: np.ndarray,
    height_values: np.ndarray,
    midpoint_grid: np.ndarray,
    label_grid: np.ndarray,
    midpoint_cutoff: float,
) -> None:
    columns = min(3, int(width_values.shape[0]))
    rows = int(math.ceil(float(width_values.shape[0]) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4.2 * rows), dpi=160, squeeze=False)

    reach_mesh, height_mesh = np.meshgrid(reach_values, height_values, indexing="xy")
    height_desc = height_values
    midpoint_min = float(np.min(midpoint_grid))
    midpoint_max = float(np.max(midpoint_grid))
    image_handles = []

    for width_index, width_value in enumerate(width_values):
        row = width_index // columns
        col = width_index % columns
        ax = axes[row][col]
        image = midpoint_grid[width_index].T
        obstacle_mask = label_grid[width_index].T.astype(float)
        handle = ax.imshow(
            image,
            extent=[float(reach_values[0]), float(reach_values[-1]), float(height_desc[-1]), float(height_desc[0])],
            origin="upper",
            aspect="auto",
            vmin=midpoint_min,
            vmax=midpoint_max,
            cmap="viridis",
        )
        image_handles.append(handle)
        ax.contour(
            reach_mesh,
            height_mesh,
            obstacle_mask,
            levels=[0.5],
            colors=["#d55e00"],
            linewidths=2.0,
        )
        ax.set_title(f"width={width_value:.3f}")
        ax.set_xlabel("reach")
        ax.set_ylabel("height")
        ax.grid(alpha=0.15)

    for empty_index in range(width_values.shape[0], rows * columns):
        row = empty_index // columns
        col = empty_index % columns
        axes[row][col].set_axis_off()

    cbar = fig.colorbar(image_handles[0], ax=axes, shrink=0.92)
    cbar.set_label("proxy midpoint height")
    fig.suptitle(f"In-band midpoint height slices with obstacle contour at z < {midpoint_cutoff:.3f}", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.92, wspace=0.28, hspace=0.32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_min_height_diagnostic_plot(
    *,
    out_path: Path,
    reach_values: np.ndarray,
    line_reach_values: np.ndarray,
    obstacle_reach_min: float,
    obstacle_reach_max: float,
    in_band_states: np.ndarray,
    midpoint_heights: np.ndarray,
    labels: np.ndarray,
    min_height_curve: np.ndarray,
    midpoint_target: float,
    midpoint_cutoff: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=160)
    obstacle_band = plt.Rectangle(
        (obstacle_reach_min, TABLE_TOP_Z),
        obstacle_reach_max - obstacle_reach_min,
        midpoint_target - TABLE_TOP_Z,
        facecolor="#f5d8c2",
        edgecolor="none",
        alpha=0.65,
        label="obstacle band",
    )
    buffer_band = plt.Rectangle(
        (obstacle_reach_min, midpoint_target),
        obstacle_reach_max - obstacle_reach_min,
        midpoint_cutoff - midpoint_target,
        facecolor="#ffe599",
        edgecolor="none",
        alpha=0.8,
        label="buffer band",
    )
    ax.add_patch(obstacle_band)
    ax.add_patch(buffer_band)
    line_handle = ax.plot(
        line_reach_values,
        min_height_curve,
        color="#0072b2",
        linewidth=2.0,
        label="midpoint z at min task height",
    )[0]
    ax.axhline(TABLE_TOP_Z, color="#444444", linestyle="--", linewidth=1.5, label="table top")
    obstacle_mask = labels == 1
    safe_mask = ~obstacle_mask
    safe_handle = ax.scatter(
        in_band_states[safe_mask, 0],
        midpoint_heights[safe_mask],
        s=24.0,
        c="#009e73",
        alpha=0.7,
        edgecolors="none",
        label="safe samples",
    )
    obstacle_handle = ax.scatter(
        in_band_states[obstacle_mask, 0],
        midpoint_heights[obstacle_mask],
        s=28.0,
        c="#d55e00",
        alpha=0.85,
        edgecolors="none",
        label="obstacle samples",
    )
    ax.set_title("Reach vs proxy midpoint height")
    ax.set_xlabel("reach")
    ax.set_ylabel("proxy midpoint height")
    ax.set_xlim(float(reach_values[0]), float(reach_values[-1]))
    ax.grid(alpha=0.2)
    ax.legend(handles=[line_handle, obstacle_handle, safe_handle, obstacle_band, buffer_band], loc="best")
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
    proxy_env = LabEnv(base_config=BaseEnvConfig(enable_proxy_rope=True))
    lower, upper = task_bounds_arrays(env)
    validate_args(lower, upper, args)

    in_band_states, width_values, reach_values, height_values = build_in_band_grid(
        lower,
        upper,
        reach_min=float(args.reach_min),
        reach_max=float(args.reach_max),
        width_margin=float(args.width_margin),
        width_steps=int(args.width_steps),
        reach_steps=int(args.reach_steps),
        height_steps=int(args.height_steps),
    )
    midpoint_heights = compute_proxy_midpoint_heights(
        proxy_env,
        in_band_states,
        progress_desc="Classifying in-band midpoint heights",
    )

    midpoint_grid = midpoint_heights.reshape(
        int(args.width_steps),
        int(args.reach_steps),
        int(args.height_steps),
    )
    midpoint_target = float(TABLE_TOP_Z + float(args.midpoint_clearance))
    midpoint_cutoff = float(midpoint_target + float(args.midpoint_buffer))
    in_band_labels = (midpoint_heights < midpoint_cutoff).astype(np.int64)
    label_grid = in_band_labels.reshape(
        int(args.width_steps),
        int(args.reach_steps),
        int(args.height_steps),
    )
    diagnostic_reach_values = np.linspace(float(lower[0]), float(upper[0]), num=161, dtype=np.float64)
    smallest_width_min_height_curve = compute_lowest_height_reach_curve(
        proxy_env,
        reach_values=diagnostic_reach_values,
        width=float(width_values[0]),
        task_height=float(lower[1]),
        progress_desc="Tracing min-height reach curve",
    )
    smallest_width_midpoint_at_lowest_height = midpoint_grid[0, :, -1].copy()
    diagnostic_min_midpoint = float(np.min(smallest_width_min_height_curve))
    diagnostic_min_clearance = float(diagnostic_min_midpoint - TABLE_TOP_Z)

    obstacle_states = in_band_states[in_band_labels == 1]
    safe_in_band_states = in_band_states[in_band_labels == 0]
    out_of_band_states = sample_out_of_band_states(
        rng,
        lower,
        upper,
        reach_min=float(args.reach_min),
        reach_max=float(args.reach_max),
        width_margin=float(args.width_margin),
        count=int(args.outside_count),
    )
    outside_states = np.concatenate((safe_in_band_states, out_of_band_states), axis=0)

    obstacle_qpos, obstacle_control = states_to_qpos_and_control(
        env,
        obstacle_states,
        progress_desc="Packing obstacle states",
    )
    outside_qpos, outside_control = states_to_qpos_and_control(
        env,
        outside_states,
        progress_desc="Packing non-obstacle states",
    )

    representative_idx = choose_representative_indices(midpoint_heights, in_band_labels, midpoint_cutoff, count=9)
    representative_states = in_band_states[representative_idx]
    representative_midpoint_heights = midpoint_heights[representative_idx]
    representative_qpos, representative_control = states_to_qpos_and_control(
        env,
        representative_states,
        progress_desc="Packing representative states",
    )

    with mujoco.Renderer(env.model, height=int(args.height), width=int(args.width)) as renderer:
        front_representatives = render_state_batch(
            env,
            renderer,
            representative_states,
            representative_qpos,
            representative_control,
            camera_name="video_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
        )
        top_representatives = render_state_batch(
            env,
            renderer,
            representative_states,
            representative_qpos,
            representative_control,
            camera_name="ceiling_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
        )

    with mujoco.Renderer(proxy_env.model, height=int(args.height), width=int(args.width)) as proxy_renderer:
        front_proxy_overlay_representatives = render_proxy_overlay_batch(
            proxy_env,
            proxy_renderer,
            representative_states,
            representative_qpos,
            representative_control,
            representative_midpoint_heights,
            camera_name="video_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
            midpoint_cutoff=midpoint_cutoff,
        )
        top_proxy_overlay_representatives = render_proxy_overlay_batch(
            proxy_env,
            proxy_renderer,
            representative_states,
            representative_qpos,
            representative_control,
            representative_midpoint_heights,
            camera_name="ceiling_cam",
            elapsed_time=0.0,
            disable_shadows=bool(args.disable_shadows),
            midpoint_cutoff=midpoint_cutoff,
        )

    save_rgb_image(out_dir / "front_representative_states_grid.png", make_image_grid(front_representatives, columns=3))
    save_rgb_image(out_dir / "top_representative_states_grid.png", make_image_grid(top_representatives, columns=3))
    save_rgb_image(
        out_dir / "front_representative_proxy_overlay_grid.png",
        make_image_grid(front_proxy_overlay_representatives, columns=3),
    )
    save_rgb_image(
        out_dir / "top_representative_proxy_overlay_grid.png",
        make_image_grid(top_proxy_overlay_representatives, columns=3),
    )

    save_min_height_diagnostic_plot(
        out_path=out_dir / "smallest_width_lowest_height_diagnostic.png",
        reach_values=np.array([float(lower[0]), float(upper[0])], dtype=np.float64),
        line_reach_values=diagnostic_reach_values,
        obstacle_reach_min=float(args.reach_min),
        obstacle_reach_max=float(args.reach_max),
        in_band_states=in_band_states,
        midpoint_heights=midpoint_heights,
        labels=in_band_labels,
        min_height_curve=smallest_width_min_height_curve,
        midpoint_target=midpoint_target,
        midpoint_cutoff=midpoint_cutoff,
    )

    torch.save(
        {
            "metadata": {
                "seed": int(args.seed),
                "width": int(args.width),
                "height": int(args.height),
                "disable_shadows": bool(args.disable_shadows),
                "in_band_grid_count": int(in_band_states.shape[0]),
                "obstacle_count_valid": int(obstacle_states.shape[0]),
                "safe_in_band_count_valid": int(safe_in_band_states.shape[0]),
                "out_of_band_count_valid": int(out_of_band_states.shape[0]),
            },
            "episode_data": {
                "qpos": np.concatenate((obstacle_qpos, outside_qpos), axis=0),
                "qvel": np.zeros((obstacle_qpos.shape[0] + outside_qpos.shape[0], env.model.nv), dtype=np.float32),
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
                "in_band_task_states": in_band_states.astype(np.float32),
                "in_band_midpoint_heights": midpoint_heights.astype(np.float32),
                "in_band_labels": in_band_labels.astype(np.int64),
                "width_values": width_values.astype(np.float32),
                "reach_values": reach_values.astype(np.float32),
                "height_values_descending": height_values.astype(np.float32),
                "obstacle_task_states": obstacle_states.astype(np.float32),
                "obstacle_qpos": obstacle_qpos.astype(np.float32),
                "obstacle_control": obstacle_control.astype(np.float32),
                "outside_task_states": outside_states.astype(np.float32),
                "outside_qpos": outside_qpos.astype(np.float32),
                "outside_control": outside_control.astype(np.float32),
                "out_of_band_task_states": out_of_band_states.astype(np.float32),
                "reach_band": np.array([float(args.reach_min), float(args.reach_max)], dtype=np.float32),
                "table_top_z": float(TABLE_TOP_Z),
                "midpoint_target": float(midpoint_target),
                "midpoint_cutoff": float(midpoint_cutoff),
            },
        },
        out_dir / "obstacle_samples.pt",
    )

    save_json(
        out_dir / "summary.json",
        {
            "seed": int(args.seed),
            "grid": {
                "width_steps": int(args.width_steps),
                "reach_steps": int(args.reach_steps),
                "height_steps": int(args.height_steps),
                "in_band_grid_count": int(in_band_states.shape[0]),
            },
            "counts": {
                "obstacle_count_valid": int(obstacle_states.shape[0]),
                "safe_in_band_count_valid": int(safe_in_band_states.shape[0]),
                "out_of_band_count_requested": int(args.outside_count),
                "out_of_band_count_valid": int(out_of_band_states.shape[0]),
                "outside_count_valid": int(outside_states.shape[0]),
            },
            "reach_min": float(args.reach_min),
            "reach_max": float(args.reach_max),
            "width_margin": float(args.width_margin),
            "task_lower": lower.tolist(),
            "task_upper": upper.tolist(),
            "table_top_z": float(TABLE_TOP_Z),
            "midpoint_clearance": float(args.midpoint_clearance),
            "midpoint_buffer": float(args.midpoint_buffer),
            "midpoint_target": float(midpoint_target),
            "midpoint_cutoff": float(midpoint_cutoff),
            "diagnostics": {
                "smallest_sampled_width": float(width_values[0]),
                "smallest_width_lowest_height_midpoint_min": diagnostic_min_midpoint,
                "smallest_width_lowest_height_midpoint_max": float(np.max(smallest_width_min_height_curve)),
                "smallest_width_lowest_height_clearance_min": diagnostic_min_clearance,
            },
            "midpoint_height_stats": {
                "in_band_min": float(np.min(midpoint_heights)),
                "in_band_max": float(np.max(midpoint_heights)),
                "obstacle_midpoint_min": float(np.min(midpoint_heights[in_band_labels == 1]))
                if np.any(in_band_labels == 1)
                else None,
                "obstacle_midpoint_max": float(np.max(midpoint_heights[in_band_labels == 1]))
                if np.any(in_band_labels == 1)
                else None,
                "safe_midpoint_min": float(np.min(midpoint_heights[in_band_labels == 0]))
                if np.any(in_band_labels == 0)
                else None,
                "safe_midpoint_max": float(np.max(midpoint_heights[in_band_labels == 0]))
                if np.any(in_band_labels == 0)
                else None,
            },
            "representative_states": representative_states.tolist(),
            "representative_midpoint_heights": representative_midpoint_heights.tolist(),
        },
    )

    print(f"Saved min-height diagnostic: {out_dir / 'smallest_width_lowest_height_diagnostic.png'}")
    print(f"Saved front grid: {out_dir / 'front_representative_states_grid.png'}")
    print(f"Saved top grid:   {out_dir / 'top_representative_states_grid.png'}")
    print(f"Saved front proxy overlay grid: {out_dir / 'front_representative_proxy_overlay_grid.png'}")
    print(f"Saved top proxy overlay grid:   {out_dir / 'top_representative_proxy_overlay_grid.png'}")
    print(f"Saved payload:    {out_dir / 'obstacle_samples.pt'}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
