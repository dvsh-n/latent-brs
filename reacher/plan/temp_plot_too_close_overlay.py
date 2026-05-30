#!/usr/bin/env python3
"""Render a low-alpha overlay for a band of Reacher joint configurations."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from reacher.plan.obs_data_collect import (
    alpha_composite_masked,
    build_arm_mask,
    forward_kinematics,
    get_arm_geom_ids,
    hide_target,
    infer_planar_arm_geometry,
    make_render_env,
    make_segmentation_scene_option,
)

DEFAULT_OUT_PATH = Path("reacher/plan/too_close_overlay.png")
DEFAULT_Q1_MIN = 3.1415
DEFAULT_Q1_MAX = 0.0
DEFAULT_Q2_MIN = -2.88
DEFAULT_Q2_MAX = -2.45
DEFAULT_Q1_SAMPLES = 21
DEFAULT_Q2_SAMPLES = 81
DEFAULT_ALPHA = 0.028
DEFAULT_SEED = 0
DEFAULT_TIME_LIMIT = 10.0
DEFAULT_PHYSICS_FREQ_HZ = 100.0
DEFAULT_WIDTH = 224
DEFAULT_HEIGHT = 224
DEFAULT_WORKSPACE_PLOT_PATH = Path("reacher/plan/too_close_workspace_samples.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--physics-freq-hz", type=float, default=DEFAULT_PHYSICS_FREQ_HZ)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--q1-min", type=float, default=DEFAULT_Q1_MIN)
    parser.add_argument("--q1-max", type=float, default=DEFAULT_Q1_MAX)
    parser.add_argument("--q2-min", type=float, default=DEFAULT_Q2_MIN)
    parser.add_argument("--q2-max", type=float, default=DEFAULT_Q2_MAX)
    parser.add_argument("--q1-samples", type=int, default=DEFAULT_Q1_SAMPLES)
    parser.add_argument("--q2-samples", type=int, default=DEFAULT_Q2_SAMPLES)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--save-boundaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace-plot-path", type=Path, default=DEFAULT_WORKSPACE_PLOT_PATH)
    return parser.parse_args()


def build_qpos_grid(
    *,
    q1_min: float,
    q1_max: float,
    q2_min: float,
    q2_max: float,
    q1_samples: int,
    q2_samples: int,
) -> np.ndarray:
    q1_values = np.linspace(q1_min, q1_max, q1_samples, dtype=np.float32)
    q2_values = np.linspace(q2_min, q2_max, q2_samples, dtype=np.float32)
    q1_mesh, q2_mesh = np.meshgrid(q1_values, q2_values, indexing="xy")
    return np.stack((q1_mesh.reshape(-1), q2_mesh.reshape(-1)), axis=1)


def build_boundary_qpos_batches(
    *,
    q1_min: float,
    q1_max: float,
    q2_min: float,
    q2_max: float,
    q1_samples: int,
    q2_samples: int,
) -> dict[str, np.ndarray]:
    q1_values = np.linspace(q1_min, q1_max, q1_samples, dtype=np.float32)
    q2_values = np.linspace(q2_min, q2_max, q2_samples, dtype=np.float32)
    return {
        "q1_min": np.stack((np.full_like(q2_values, q1_min), q2_values), axis=1),
        "q1_max": np.stack((np.full_like(q2_values, q1_max), q2_values), axis=1),
        "q2_min": np.stack((q1_values, np.full_like(q1_values, q2_min)), axis=1),
        "q2_max": np.stack((q1_values, np.full_like(q1_values, q2_max)), axis=1),
    }


def render_uniform_overlay(
    qpos_batch: np.ndarray,
    *,
    seed: int,
    time_limit: float,
    physics_freq_hz: float,
    width: int,
    height: int,
    alpha: float,
) -> np.ndarray:
    env = make_render_env(
        seed=seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    try:
        physics = env._env.physics
        model = physics.model
        qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
        arm_geom_ids = get_arm_geom_ids(model)
        scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        try:
            for qpos in tqdm(qpos_batch, desc="Rendering overlay", unit="pose"):
                with physics.reset_context():
                    physics.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float32)
                    physics.data.qvel[: qvel.shape[0]] = qvel
                frame = physics.render(height=height, width=width, camera_id=0)
                segmentation = physics.render(
                    height=height,
                    width=width,
                    camera_id=0,
                    segmentation=True,
                    scene_option=scene_option,
                )
                mask = build_arm_mask(segmentation, arm_geom_ids)
                canvas = alpha_composite_masked(canvas, frame, mask, alpha=float(alpha))
        finally:
            model.geom_group[target_geom_id] = original_group
    finally:
        env.close()
    return canvas


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def boundary_output_path(out_path: Path, boundary_name: str) -> Path:
    return out_path.with_name(f"{out_path.stem}_{boundary_name}{out_path.suffix}")


def qpos_to_workspace_xy(qpos_batch: np.ndarray, *, base_xy: np.ndarray, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray]:
    elbow_xy = np.zeros((qpos_batch.shape[0], 2), dtype=np.float64)
    tip_xy = np.zeros((qpos_batch.shape[0], 2), dtype=np.float64)
    for idx, qpos in enumerate(qpos_batch):
        elbow_local, tip_local = forward_kinematics(qpos, link1=link1, link2=link2)
        elbow_xy[idx] = base_xy + elbow_local
        tip_xy[idx] = base_xy + tip_local
    return elbow_xy, tip_xy


def save_workspace_plot(
    *,
    out_path: Path,
    geom: dict[str, object],
    qpos_batch: np.ndarray,
) -> None:
    base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
    link1 = float(geom["link1"])
    link2 = float(geom["link2"])
    _, tip_xy = qpos_to_workspace_xy(qpos_batch, base_xy=base_xy, link1=link1, link2=link2)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    reach_max = float(geom["reach_max"])
    reach_min = float(geom["reach_min"])
    ax.add_patch(plt.Circle(base_xy, reach_max, color="#d9d9d9", fill=False, linestyle="--", linewidth=1.0))
    if reach_min > 1e-4:
        ax.add_patch(plt.Circle(base_xy, reach_min, color="#ececec", fill=False, linestyle=":", linewidth=1.0))

    ax.scatter(tip_xy[:, 0], tip_xy[:, 1], s=8, c="#d55e00", alpha=0.35, label="obstacle tips")
    ax.set_aspect("equal", adjustable="box")
    lim = reach_max + 0.04
    ax.set_xlim(base_xy[0] - lim, base_xy[0] + lim)
    ax.set_ylim(base_xy[1] - lim, base_xy[1] + lim)
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Reacher box-constraint workspace samples")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    qpos_batch = build_qpos_grid(
        q1_min=args.q1_min,
        q1_max=args.q1_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        q1_samples=args.q1_samples,
        q2_samples=args.q2_samples,
    )
    boundary_batches = build_boundary_qpos_batches(
        q1_min=args.q1_min,
        q1_max=args.q1_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        q1_samples=args.q1_samples,
        q2_samples=args.q2_samples,
    )
    overlay = render_uniform_overlay(
        qpos_batch,
        seed=args.seed,
        time_limit=args.time_limit,
        physics_freq_hz=args.physics_freq_hz,
        width=args.width,
        height=args.height,
        alpha=args.alpha,
    )
    save_rgb_image(args.out_path, overlay)
    print(
        f"Saved overlay to {args.out_path} using {qpos_batch.shape[0]} poses "
        f"(q1 in [{args.q1_min}, {args.q1_max}], q2 in [{args.q2_min}, {args.q2_max}]).",
        flush=True,
    )
    geom_env = make_render_env(
        seed=args.seed,
        time_limit=args.time_limit,
        width=args.width,
        height=args.height,
        physics_freq_hz=args.physics_freq_hz,
    )
    try:
        geom = infer_planar_arm_geometry(geom_env)
    finally:
        geom_env.close()
    save_workspace_plot(
        out_path=args.workspace_plot_path,
        geom=geom,
        qpos_batch=qpos_batch,
    )
    print(f"Saved workspace plot to {args.workspace_plot_path}.", flush=True)
    if args.save_boundaries:
        for boundary_name, boundary_qpos_batch in boundary_batches.items():
            boundary_overlay = render_uniform_overlay(
                boundary_qpos_batch,
                seed=args.seed,
                time_limit=args.time_limit,
                physics_freq_hz=args.physics_freq_hz,
                width=args.width,
                height=args.height,
                alpha=args.alpha,
            )
            boundary_path = boundary_output_path(args.out_path, boundary_name)
            save_rgb_image(boundary_path, boundary_overlay)
            print(
                f"Saved boundary overlay to {boundary_path} using {boundary_qpos_batch.shape[0]} poses.",
                flush=True,
            )


if __name__ == "__main__":
    main()
