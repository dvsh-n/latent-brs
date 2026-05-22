#!/usr/bin/env python3
"""Collect a balanced rope obstacle image dataset from dense task-space samples."""

from __future__ import annotations

import argparse
import json
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
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.shared.lab_env import BaseEnvConfig, LabEnv, TABLE_TOP_Z, TaskState

DEFAULT_OUT_DIR = "rope/plan/obstacle_data"
DEFAULT_CAMERA = "video_cam"
DISABLE_SHADOWS = True
TRAIN_FRACTION = 0.9
DIAGNOSTIC_PLOT_NAME = "balanced_obstacle_dataset_reach_height.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        default=DISABLE_SHADOWS,
        help="Disable shadows for saved renders.",
    )
    parser.add_argument("--reach-steps", type=int, default=41)
    parser.add_argument("--height-steps", type=int, default=41)
    parser.add_argument("--width-steps", type=int, default=11)
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
        help="Extra model-error buffer added to the midpoint clearance threshold.",
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


def validate_args(lower: np.ndarray, upper: np.ndarray, args: argparse.Namespace, env: LabEnv) -> None:
    del lower, upper
    if int(args.reach_steps) <= 0 or int(args.height_steps) <= 0 or int(args.width_steps) <= 0:
        raise ValueError("Grid step counts must all be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("Render width and height must both be positive.")
    if float(args.midpoint_clearance) < 0.0 or float(args.midpoint_buffer) < 0.0:
        raise ValueError("Midpoint clearance and midpoint buffer must be non-negative.")
    try:
        env.model.camera(str(args.camera)).id
    except KeyError as exc:
        raise ValueError(f"Unknown camera {args.camera!r}.") from exc


def build_dense_task_grid(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    reach_steps: int,
    height_steps: int,
    width_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reach_values = np.linspace(float(lower[0]), float(upper[0]), num=reach_steps, dtype=np.float64)
    height_values = np.linspace(float(upper[1]), float(lower[1]), num=height_steps, dtype=np.float64)
    width_values = np.linspace(float(lower[2]), float(upper[2]), num=width_steps, dtype=np.float64)

    states: list[np.ndarray] = []
    for width in width_values:
        for reach in reach_values:
            for height in height_values:
                states.append(np.array([reach, height, width], dtype=np.float64))
    return np.stack(states, axis=0), reach_values, height_values, width_values


def states_to_qpos_and_control(
    task_states: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if task_states.shape[0] == 0:
        env = LabEnv()
        return (
            np.zeros((0, env.model.nq), dtype=np.float32),
            np.zeros((0, env.model.nu), dtype=np.float32),
        )
    env = LabEnv()
    qpos_batch: list[np.ndarray] = []
    control_batch: list[np.ndarray] = []
    iterator = tqdm(task_states, desc=progress_desc, unit="state", leave=False) if progress_desc else task_states
    for state_vec in iterator:
        env.reset(TaskState.from_array(state_vec))
        qpos_batch.append(env.data.qpos.copy().astype(np.float32))
        control_batch.append(env.data.ctrl.copy().astype(np.float32))
    return np.stack(qpos_batch, axis=0), np.stack(control_batch, axis=0)


def compute_proxy_midpoint_heights(
    task_states: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> np.ndarray:
    proxy_env = LabEnv(base_config=BaseEnvConfig(enable_proxy_rope=True))
    midpoint_heights = np.zeros((task_states.shape[0],), dtype=np.float64)
    iterator = (
        tqdm(enumerate(task_states), total=task_states.shape[0], desc=progress_desc, unit="state")
        if progress_desc
        else enumerate(task_states)
    )
    for index, state_vec in iterator:
        proxy_env.reset(TaskState.from_array(state_vec))
        midpoint_heights[index] = proxy_env.get_proxy_rope_midpoint_height()
    return midpoint_heights


def render_rgb_frame(
    renderer: mujoco.Renderer,
    env: LabEnv,
    camera_id: int,
    *,
    disable_shadows: bool,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def render_dataset_images(
    task_states: np.ndarray,
    qpos_batch: np.ndarray,
    control_batch: np.ndarray,
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
    disable_shadows: bool,
) -> np.ndarray:
    env = LabEnv()
    camera_id = env.model.camera(camera_name).id
    qvel = np.zeros((env.model.nv,), dtype=np.float32)
    frames: list[np.ndarray] = []
    with mujoco.Renderer(env.model, height=image_height, width=image_width) as renderer:
        iterator = tqdm(
            zip(task_states, qpos_batch, control_batch, strict=True),
            total=task_states.shape[0],
            desc="Rendering dataset images",
            unit="image",
        )
        for state_vec, qpos, control in iterator:
            env.reset(TaskState.from_array(state_vec))
            env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
            env.data.qvel[: qvel.shape[0]] = qvel
            env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
            env.task_controller.set_target(TaskState.from_array(state_vec))
            env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
            mujoco.mj_forward(env.model, env.data)
            frames.append(render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows))
    return np.stack(frames, axis=0)


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size == 1:
        return shuffled.copy(), np.zeros((0,), dtype=np.int64)
    train_count = int(np.floor(train_fraction * float(shuffled.size)))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    return shuffled[:train_count], shuffled[train_count:]


def build_balanced_dataset(
    states: np.ndarray,
    midpoint_heights: np.ndarray,
    midpoint_cutoff: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    labels = (midpoint_heights < midpoint_cutoff).astype(np.int64)
    obstacle_idx = np.flatnonzero(labels == 1)
    non_obstacle_idx = np.flatnonzero(labels == 0)
    if obstacle_idx.size == 0:
        raise ValueError("Dense sampling produced zero obstacle states.")
    if non_obstacle_idx.size < obstacle_idx.size:
        raise ValueError(
            f"Need at least as many non-obstacle states as obstacle states for balancing, got "
            f"{non_obstacle_idx.size} non-obstacle and {obstacle_idx.size} obstacle."
        )

    sampled_non_obstacle_idx = rng.choice(non_obstacle_idx, size=obstacle_idx.size, replace=False)
    selected_idx = np.concatenate((obstacle_idx, sampled_non_obstacle_idx), axis=0)
    selected_labels = labels[selected_idx]
    selected_midpoint_heights = midpoint_heights[selected_idx]
    selected_states = states[selected_idx].astype(np.float32)

    obstacle_local_idx = np.flatnonzero(selected_labels == 1)
    non_obstacle_local_idx = np.flatnonzero(selected_labels == 0)
    obstacle_train, obstacle_cal = split_indices(obstacle_local_idx, rng, TRAIN_FRACTION)
    non_obstacle_train, non_obstacle_cal = split_indices(non_obstacle_local_idx, rng, TRAIN_FRACTION)
    train_idx = np.concatenate((obstacle_train, non_obstacle_train), axis=0)
    cal_idx = np.concatenate((obstacle_cal, non_obstacle_cal), axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(cal_idx)

    return {
        "selected_idx": selected_idx.astype(np.int64),
        "task_target": selected_states,
        "label": selected_labels.astype(np.int64),
        "midpoint_height": selected_midpoint_heights.astype(np.float32),
        "train_idx": train_idx.astype(np.int64),
        "calibration_idx": cal_idx.astype(np.int64),
        "all_labels": labels.astype(np.int64),
    }


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_balanced_dataset_diagnostic(
    out_path: Path,
    task_states: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    calibration_idx: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    obstacle_mask = labels == 1
    safe_mask = ~obstacle_mask
    train_mask = np.zeros((labels.shape[0],), dtype=bool)
    cal_mask = np.zeros((labels.shape[0],), dtype=bool)
    train_mask[train_idx] = True
    cal_mask[calibration_idx] = True

    ax.scatter(
        task_states[safe_mask & train_mask, 0],
        task_states[safe_mask & train_mask, 1],
        s=18.0,
        c="#009e73",
        alpha=0.7,
        edgecolors="none",
        label="train non-obstacle",
    )
    ax.scatter(
        task_states[obstacle_mask & train_mask, 0],
        task_states[obstacle_mask & train_mask, 1],
        s=18.0,
        c="#d55e00",
        alpha=0.75,
        edgecolors="none",
        label="train obstacle",
    )
    ax.scatter(
        task_states[safe_mask & cal_mask, 0],
        task_states[safe_mask & cal_mask, 1],
        s=42.0,
        marker="x",
        c="#0072b2",
        alpha=0.9,
        linewidths=1.2,
        label="calibration non-obstacle",
    )
    ax.scatter(
        task_states[obstacle_mask & cal_mask, 0],
        task_states[obstacle_mask & cal_mask, 1],
        s=42.0,
        marker="x",
        c="#7a0019",
        alpha=0.9,
        linewidths=1.2,
        label="calibration obstacle",
    )
    ax.set_title("Balanced 2N obstacle dataset in reach-height space")
    ax.set_xlabel("reach")
    ax.set_ylabel("height")
    ax.grid(alpha=0.2)
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
    validate_args(lower, upper, args, env)

    dense_states, reach_values, height_values, width_values = build_dense_task_grid(
        lower,
        upper,
        reach_steps=int(args.reach_steps),
        height_steps=int(args.height_steps),
        width_steps=int(args.width_steps),
    )
    midpoint_heights = compute_proxy_midpoint_heights(
        dense_states,
        progress_desc="Classifying dense task grid",
    )

    midpoint_target = float(TABLE_TOP_Z + float(args.midpoint_clearance))
    midpoint_cutoff = float(midpoint_target + float(args.midpoint_buffer))
    balanced = build_balanced_dataset(dense_states, midpoint_heights, midpoint_cutoff, rng)

    dataset_states = balanced["task_target"].astype(np.float64)
    dataset_qpos, dataset_control = states_to_qpos_and_control(
        dataset_states,
        progress_desc="Packing balanced dataset states",
    )
    dataset_pixels = render_dataset_images(
        dataset_states,
        dataset_qpos,
        dataset_control,
        camera_name=str(args.camera),
        image_width=int(args.width),
        image_height=int(args.height),
        disable_shadows=bool(args.disable_shadows),
    )

    save_balanced_dataset_diagnostic(
        out_dir / DIAGNOSTIC_PLOT_NAME,
        balanced["task_target"],
        balanced["label"],
        balanced["train_idx"],
        balanced["calibration_idx"],
    )

    torch.save(
        {
            "metadata": {
                "seed": int(args.seed),
                "camera": str(args.camera),
                "image_width": int(args.width),
                "image_height": int(args.height),
                "disable_shadows": bool(args.disable_shadows),
                "train_fraction": float(TRAIN_FRACTION),
                "calibration_fraction": float(1.0 - TRAIN_FRACTION),
                "table_top_z": float(TABLE_TOP_Z),
                "midpoint_clearance": float(args.midpoint_clearance),
                "midpoint_buffer": float(args.midpoint_buffer),
                "midpoint_target": float(midpoint_target),
                "midpoint_cutoff": float(midpoint_cutoff),
                "task_lower": lower.astype(np.float32),
                "task_upper": upper.astype(np.float32),
                "grid_shape": np.array(
                    [int(args.width_steps), int(args.reach_steps), int(args.height_steps)],
                    dtype=np.int64,
                ),
                "dense_grid_count": int(dense_states.shape[0]),
                "obstacle_count_dense": int(np.sum(balanced["all_labels"] == 1)),
                "non_obstacle_count_dense": int(np.sum(balanced["all_labels"] == 0)),
                "balanced_total_count": int(balanced["task_target"].shape[0]),
            },
            "dataset": {
                "pixels": dataset_pixels.astype(np.uint8),
                "task_target": balanced["task_target"].astype(np.float32),
                "label": balanced["label"].astype(np.int64),
                "midpoint_height": balanced["midpoint_height"].astype(np.float32),
                "qpos": dataset_qpos.astype(np.float32),
                "control": dataset_control.astype(np.float32),
                "train_idx": balanced["train_idx"].astype(np.int64),
                "calibration_idx": balanced["calibration_idx"].astype(np.int64),
            },
            "dense_grid": {
                "task_target": dense_states.astype(np.float32),
                "midpoint_height": midpoint_heights.astype(np.float32),
                "label": balanced["all_labels"].astype(np.int64),
                "reach_values": reach_values.astype(np.float32),
                "height_values_descending": height_values.astype(np.float32),
                "width_values": width_values.astype(np.float32),
            },
        },
        out_dir / "obstacle_classifier_data.pt",
    )

    save_json(
        out_dir / "summary.json",
        {
            "out_dir": str(out_dir),
            "camera": str(args.camera),
            "grid": {
                "reach_steps": int(args.reach_steps),
                "height_steps": int(args.height_steps),
                "width_steps": int(args.width_steps),
                "dense_grid_count": int(dense_states.shape[0]),
            },
            "counts": {
                "dense_obstacle_count": int(np.sum(balanced["all_labels"] == 1)),
                "dense_non_obstacle_count": int(np.sum(balanced["all_labels"] == 0)),
                "balanced_obstacle_count": int(np.sum(balanced["label"] == 1)),
                "balanced_non_obstacle_count": int(np.sum(balanced["label"] == 0)),
                "balanced_total_count": int(balanced["label"].shape[0]),
                "train_count": int(balanced["train_idx"].shape[0]),
                "calibration_count": int(balanced["calibration_idx"].shape[0]),
            },
            "split": {
                "train_fraction": float(TRAIN_FRACTION),
                "calibration_fraction": float(1.0 - TRAIN_FRACTION),
            },
            "task_lower": lower.tolist(),
            "task_upper": upper.tolist(),
            "midpoint_target": float(midpoint_target),
            "midpoint_cutoff": float(midpoint_cutoff),
        },
    )

    print(f"Saved diagnostic: {out_dir / DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved dataset:    {out_dir / 'obstacle_classifier_data.pt'}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
