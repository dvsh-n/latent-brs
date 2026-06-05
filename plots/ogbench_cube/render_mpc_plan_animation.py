#!/usr/bin/env python3
"""Render a side-view video with executed rollout and planned MPC action horizons."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.lines import Line2D
import mujoco
import numpy as np
from tqdm.auto import tqdm

from plots.ogbench_cube import render_side_timelapse as side


DEFAULT_RUN_DIRS = (
    side.DEFAULT_ROOT / "ogbench_cube_safe_0",
    side.DEFAULT_ROOT / "ogbench_cube_unsafe_0",
)
DEFAULT_OUTPUT_NAME = "mpc_plan_animation.mp4"
DEFAULT_COMBINED_OUTPUT_NAME = "mpc_plan_animation_combined.mp4"
DEFAULT_BOTTOM_OUTPUT_NAME = "mpc_plan_animation_latent_tubes.mp4"
SAFE_COLOR = np.array([38.0, 166.0, 91.0], dtype=np.float32)
UNSAFE_COLOR = np.array([230.0, 126.0, 34.0], dtype=np.float32)
PLAN_COLOR = np.array([0.0, 188.0, 212.0], dtype=np.float32)
CURRENT_COLOR = np.array([20.0, 20.0, 20.0], dtype=np.float32)
FIGURE_GREEN_DARK = "#167a43"
FIGURE_GREEN = "#2ca25f"
FIGURE_CYAN = "#00bcd4"
LABEL_SAFE_COLOR = (19, 138, 54)
LABEL_UNSAFE_COLOR = (193, 18, 31)

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "figure.dpi": 300,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Render a single run directory. Overrides --run-dirs when provided.",
    )
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--combined-output-name", default=DEFAULT_COMBINED_OUTPUT_NAME)
    parser.add_argument("--combined-output-dir", type=Path, default=side.DEFAULT_ROOT)
    parser.add_argument(
        "--bottom-video",
        type=Path,
        default=None,
        help="Output path for the regenerated latent-tube bottom video.",
    )
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", default=side.DEFAULT_ENV_NAME)
    parser.add_argument("--sim-freq-hz", type=float, default=side.DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=side.DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--max-oracle-steps", type=int, default=80)
    parser.add_argument("--oracle-segment-dt", type=float, default=0.4)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=0.5)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.425, 0.0, 0.14))
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=0.0)
    parser.add_argument("--front-camera", default="front_pixels")
    parser.add_argument("--fovy", type=float, default=45.0)
    parser.add_argument("--height-threshold", type=float, default=None)
    parser.add_argument("--threshold-x-min", type=float, default=0.24)
    parser.add_argument("--threshold-x-max", type=float, default=0.56)
    parser.add_argument("--threshold-alpha", type=float, default=0.92)
    parser.add_argument("--threshold-width", type=int, default=2)
    parser.add_argument("--executed-alpha", type=float, default=0.92)
    parser.add_argument("--executed-width", type=int, default=3)
    parser.add_argument("--plan-alpha", type=float, default=0.95)
    parser.add_argument("--plan-width", type=int, default=3)
    parser.add_argument("--plan-ghost-alpha", type=float, default=0.30)
    parser.add_argument("--current-marker-radius", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-sync-duration",
        action="store_true",
        help="Do not cut batch renders to the shortest run's frame count.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--plan-horizon", type=int, default=None)
    parser.add_argument("--bottom-run-dir", type=Path, default=None)
    parser.add_argument("--bottom-dims", type=str, default=None)
    parser.add_argument("--bottom-start-step", type=int, default=1)
    parser.add_argument("--bottom-plan-stride", type=int, default=5)
    parser.add_argument("--bottom-max-plans", type=int, default=None)
    parser.add_argument("--bottom-alpha", type=float, default=0.50)
    parser.add_argument("--bottom-alpha-decay", type=float, default=1.0)
    parser.add_argument("--bottom-horizon-alpha-decay", type=float, default=0.86)
    parser.add_argument("--bottom-dpi", type=int, default=150)
    parser.add_argument("--bottom-quality", type=int, default=8)
    parser.add_argument("--bottom-data-axis-padding", type=float, default=0.08)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.fps) <= 0:
        raise ValueError("--fps must be positive.")
    if int(args.frame_stride) <= 0:
        raise ValueError("--frame-stride must be positive.")
    if int(args.max_oracle_steps) < 0:
        raise ValueError("--max-oracle-steps must be non-negative.")
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if args.plan_horizon is not None and int(args.plan_horizon) <= 0:
        raise ValueError("--plan-horizon must be positive when provided.")
    if float(args.threshold_x_min) >= float(args.threshold_x_max):
        raise ValueError("--threshold-x-min must be less than --threshold-x-max.")
    if not 0.0 <= float(args.plan_ghost_alpha) <= 1.0:
        raise ValueError(f"--plan-ghost-alpha must be in [0, 1], got {args.plan_ghost_alpha}.")
    if args.run_dir is None and not args.run_dirs:
        raise ValueError("At least one run directory is required.")
    if int(args.bottom_plan_stride) <= 0:
        raise ValueError("--bottom-plan-stride must be positive.")
    if args.bottom_max_plans is not None and int(args.bottom_max_plans) <= 0:
        raise ValueError("--bottom-max-plans must be positive when provided.")


def load_planned_actions(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    npz_path = run_dir / "planned_actions.npz"
    if npz_path.is_file():
        payload = np.load(npz_path)
        return (
            np.asarray(payload["mpc_planned_action_steps"], dtype=np.int64),
            np.asarray(payload["mpc_planned_actions_raw"], dtype=np.float32),
            {"source": str(npz_path), "format": "npz"},
        )

    tube_data_path = run_dir / "tube_data.npz"
    if tube_data_path.is_file():
        payload = np.load(tube_data_path)
        if "plan_steps" in payload and "nominal_actions" in payload:
            plan_actions, action_metadata = tube_nominal_actions_to_raw(run_dir, np.asarray(payload["nominal_actions"]))
            return (
                np.asarray(payload["plan_steps"], dtype=np.int64),
                plan_actions,
                {
                    "source": str(tube_data_path),
                    "format": "tube_data.npz",
                    "steps_key": "plan_steps",
                    "actions_key": "nominal_actions",
                    **action_metadata,
                },
            )

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing planned_actions.npz, tube_data.npz, and metrics.json in {run_dir}.")
    metrics = side.load_json(metrics_path)
    return (
        np.asarray(metrics["mpc_planned_action_steps"], dtype=np.int64),
        np.asarray(metrics["mpc_planned_actions_raw"], dtype=np.float32),
        {"source": str(metrics_path), "format": "metrics.json"},
    )


def tube_nominal_actions_to_raw(run_dir: Path, nominal_actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    actions_norm = np.asarray(nominal_actions, dtype=np.float32).copy()
    actions_path = run_dir / "executed_actions.npz"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Cannot convert {run_dir / 'tube_data.npz'} nominal_actions: missing executed_actions.npz.")

    executed = np.load(actions_path)
    if "executed_actions_norm" not in executed or "executed_actions_raw" not in executed:
        raise KeyError("executed_actions.npz must contain executed_actions_norm and executed_actions_raw.")

    executed_norm = np.asarray(executed["executed_actions_norm"], dtype=np.float64)
    executed_raw = np.asarray(executed["executed_actions_raw"], dtype=np.float64)
    if executed_norm.shape != executed_raw.shape or executed_norm.ndim != 2:
        raise ValueError(
            f"Expected executed action arrays with matching shape (T, action_dim), got "
            f"{executed_norm.shape} and {executed_raw.shape}."
        )
    if actions_norm.shape[-1] != executed_norm.shape[-1]:
        raise ValueError(
            f"Tube action dim {actions_norm.shape[-1]} does not match executed action dim {executed_norm.shape[-1]}."
        )

    summary_path = run_dir / "trajectory_summary.json"
    fixed_grasp = {}
    if summary_path.is_file():
        fixed_grasp = side.load_json(summary_path).get("metadata", {}).get("fixed_grasp", {})

    grip_idx = actions_norm.shape[-1] - 1
    fixed_grasp_norm = fixed_grasp.get("u4_norm")
    if fixed_grasp_norm is None and executed_norm.shape[0] > 0 and np.isfinite(executed_norm[:, grip_idx]).any():
        fixed_grasp_norm = float(np.nanmedian(executed_norm[:, grip_idx]))
    if fixed_grasp_norm is not None:
        missing_grip = ~np.isfinite(actions_norm[..., grip_idx])
        actions_norm[..., grip_idx][missing_grip] = float(fixed_grasp_norm)

    action_std = np.zeros(executed_norm.shape[1], dtype=np.float64)
    action_mean = np.zeros(executed_norm.shape[1], dtype=np.float64)
    for dim in range(executed_norm.shape[1]):
        finite = np.isfinite(executed_norm[:, dim]) & np.isfinite(executed_raw[:, dim])
        if not np.any(finite):
            raise ValueError(f"No finite executed action pairs available for action dim {dim}.")
        x = executed_norm[finite, dim]
        y = executed_raw[finite, dim]
        if np.ptp(x) <= 1e-8:
            action_std[dim] = 0.0
            action_mean[dim] = float(np.nanmedian(y))
        else:
            design = np.column_stack([x, np.ones_like(x)])
            action_std[dim], action_mean[dim] = np.linalg.lstsq(design, y, rcond=None)[0]

    finite_rows = np.isfinite(actions_norm).all(axis=-1)
    actions_norm = np.where(finite_rows[..., None], actions_norm, 0.0)
    actions_raw = actions_norm.astype(np.float64) * action_std.reshape(1, 1, -1) + action_mean.reshape(1, 1, -1)
    actions_raw = np.where(finite_rows[..., None], actions_raw, np.nan).astype(np.float32)
    return actions_raw, {
        "action_space": "raw",
        "converted_from": "normalized_nominal_actions",
        "conversion_stats_source": str(actions_path),
        "invalid_horizon_rows": int(np.size(finite_rows) - np.count_nonzero(finite_rows)),
    }


def load_gripper_heights(run_dir: Path, qpos: np.ndarray, env: Any, site_id: int) -> tuple[np.ndarray, dict[str, Any]]:
    loaded = side.load_logged_gripper_heights(run_dir)
    if loaded is not None and loaded[0].ndim == 1 and loaded[0].shape[0] >= qpos.shape[0]:
        return np.asarray(loaded[0][: qpos.shape[0]], dtype=np.float64), loaded[1]

    gripper_positions = side.collect_gripper_positions(env, qpos, site_id)
    return gripper_positions[:, 2].astype(np.float64), {"source": "simulated_pinch_site_z"}


def project_points(points: np.ndarray, projection: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    return side.project_world_points_to_pixels(
        points,
        camera_position=projection["position"],
        camera_right=projection["right"],
        camera_up=projection["up"],
        camera_forward=projection["forward"],
        fovy_deg=float(args.fovy),
        width=int(args.width),
        height=int(args.height),
    )


def draw_colored_executed_trace(
    image: np.ndarray,
    points_px: np.ndarray,
    heights: np.ndarray,
    threshold: float,
    *,
    alpha: float,
    width: int,
) -> np.ndarray:
    composite = image
    if points_px.shape[0] < 2:
        return composite
    for idx in range(points_px.shape[0] - 1):
        segment = points_px[idx : idx + 2]
        if not np.isfinite(segment).all():
            continue
        height = float(max(heights[idx], heights[min(idx + 1, heights.shape[0] - 1)]))
        color = SAFE_COLOR if height <= float(threshold) else UNSAFE_COLOR
        composite = side.draw_polyline(composite, segment, color=color, alpha=alpha, width=width)
    return composite


def draw_trace_legend(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    dpi = 100
    frame = np.clip(np.rint(image[:, :, :3]), 0, 255).astype(np.uint8)

    fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(frame)
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], color=SAFE_COLOR / 255.0, linewidth=2.0, label="Safe"),
        Line2D([0], [0], color=UNSAFE_COLOR / 255.0, linewidth=2.0, label="Unsafe"),
        Line2D([0], [0], color=PLAN_COLOR / 255.0, linewidth=2.0, label="MPC plan"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        framealpha=0.78,
        facecolor="white",
        edgecolor="0.15",
        fontsize=max(14, int(round(width / 40))),
        prop={"weight": "semibold", "size": max(14, int(round(width / 40)))},
        handlelength=2.3,
        borderpad=0.55,
        labelspacing=0.45,
    )

    canvas.draw()
    rendered = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    return rendered.copy()


def draw_current_marker(image: np.ndarray, point_px: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0 or not np.isfinite(point_px).all():
        return image
    overlay = image.copy()
    center = tuple(int(round(value)) for value in point_px)
    cv2.circle(overlay, center, int(radius), tuple(float(v) for v in CURRENT_COLOR), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius), (255.0, 255.0, 255.0), thickness=1, lineType=cv2.LINE_AA)
    return side.alpha_composite_image(image, overlay, 0.95)


def planned_rollout_states(env: Any, qpos_start: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos_history: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    env.reset()
    side.set_env_state(env, qpos_start)
    qpos_history.append(env.unwrapped._data.qpos.copy())
    info = env.unwrapped.get_step_info()
    target_block = int(info.get("privileged/target_block", 0))
    positions.append(np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64).copy())
    for action in actions:
        if not np.isfinite(action).all():
            break
        terminated = truncated = False
        _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        side.hide_target_cube(env)
        qpos_history.append(env.unwrapped._data.qpos.copy())
        info = env.unwrapped.get_step_info()
        target_block = int(info.get("privileged/target_block", 0))
        positions.append(np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64).copy())
        if terminated or truncated:
            break
    return np.stack(qpos_history, axis=0), np.stack(positions, axis=0)


def draw_plan_ghost_overlays(
    image: np.ndarray,
    env: Any,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    qpos_history: np.ndarray,
    geom_ids: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    composite = image
    if qpos_history.shape[0] <= 1 or geom_ids.size == 0 or float(alpha) <= 0.0:
        return composite
    for qpos_row in qpos_history[1:]:
        side.set_env_state(env, qpos_row)
        rgb = side.render_frame(renderer, env, camera).astype(np.float32)
        segmentation = side.render_segmentation(renderer, env, camera)
        mask = side.mask_from_geom_ids(segmentation, geom_ids)
        if not np.any(mask):
            continue
        composite = side.alpha_composite_mask(composite, rgb, mask, float(alpha))
    return composite


def draw_current_dynamic_geoms(
    image: np.ndarray,
    env: Any,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    geom_ids: np.ndarray,
) -> np.ndarray:
    if geom_ids.size == 0:
        return image
    rgb = side.render_frame(renderer, env, camera).astype(np.float32)
    segmentation = side.render_segmentation(renderer, env, camera)
    mask = side.mask_from_geom_ids(segmentation, geom_ids)
    return side.alpha_composite_mask(image, rgb, mask, 1.0)


def selected_frame_indices(plan_steps: np.ndarray, args: argparse.Namespace, frame_limit: int | None = None) -> np.ndarray:
    frame_indices = np.arange(plan_steps.shape[0], dtype=np.int64)
    frame_indices = frame_indices[:: int(args.frame_stride)]
    if frame_limit is not None:
        frame_indices = frame_indices[: int(frame_limit)]
    if args.max_frames is not None:
        frame_indices = frame_indices[: int(args.max_frames)]
    return frame_indices


def front_output_path_for(side_output_path: Path) -> Path:
    return side_output_path.with_name(f"{side_output_path.stem}_front{side_output_path.suffix or '.mp4'}")


def render_animation(run_dir: Path, args: argparse.Namespace, frame_limit: int | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    qpos, qpos_metadata = side.load_rollout_qpos(run_dir, args)
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"Expected qpos with shape (T, nq), got {qpos.shape}.")
    plan_steps, plan_actions, plan_metadata = load_planned_actions(run_dir)
    if plan_steps.shape[0] != plan_actions.shape[0]:
        raise ValueError(f"Plan steps/actions length mismatch: {plan_steps.shape[0]} vs {plan_actions.shape[0]}.")
    if args.plan_horizon is not None:
        plan_actions = plan_actions[:, : int(args.plan_horizon)]

    frame_indices = selected_frame_indices(plan_steps, args, frame_limit)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for animation.")

    height_threshold, threshold_source = side.resolve_height_threshold(run_dir, args)
    render_env = side.make_render_env(args)
    plan_env = side.make_render_env(args)
    frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []
    try:
        camera = side.side_camera(args)
        site_id = side.pinch_site_id(render_env.unwrapped._model)
        plan_ghost_geom_ids = side.dynamic_geom_ids(render_env.unwrapped._model)
        cube_positions = side.collect_cube_center_positions(render_env, qpos)
        gripper_heights, height_metadata = load_gripper_heights(run_dir, qpos, render_env, site_id)

        with mujoco.Renderer(render_env.unwrapped._model, height=int(args.height), width=int(args.width)) as renderer:
            renderer.scene.camera[0].frustum_center = 0.0
            renderer.scene.camera[0].frustum_width = 0.0
            side.set_env_state(render_env, qpos[0])
            side.render_frame(renderer, render_env, camera)
            cam_pos, cam_right, cam_up, cam_forward = side.camera_projection_from_scene(renderer)
            projection = {"position": cam_pos, "right": cam_right, "up": cam_up, "forward": cam_forward}
            threshold_world = np.array(
                [
                    [float(args.threshold_x_min), 0.0, float(height_threshold)],
                    [float(args.threshold_x_max), 0.0, float(height_threshold)],
                ],
                dtype=np.float64,
            )
            threshold_px = project_points(threshold_world, projection, args)

            for plan_index in tqdm(frame_indices, desc=f"Rendering {run_dir.name} MPC plan animation"):
                global_step = int(plan_steps[plan_index])
                qpos_index = min(max(global_step, 0), qpos.shape[0] - 1)
                side.set_env_state(render_env, qpos[qpos_index])
                front_frames.append(np.asarray(render_env.unwrapped.render(camera=str(args.front_camera)), dtype=np.uint8).copy())
                with side.hidden_geoms(render_env.unwrapped._model, plan_ghost_geom_ids):
                    frame = side.render_frame(renderer, render_env, camera).astype(np.float32)

                planned_qpos, planned_positions = planned_rollout_states(plan_env, qpos[qpos_index], plan_actions[plan_index])
                frame = draw_plan_ghost_overlays(
                    frame,
                    render_env,
                    renderer,
                    camera,
                    planned_qpos,
                    plan_ghost_geom_ids,
                    alpha=float(args.plan_ghost_alpha),
                )
                side.set_env_state(render_env, qpos[qpos_index])
                frame = draw_current_dynamic_geoms(
                    frame,
                    render_env,
                    renderer,
                    camera,
                    plan_ghost_geom_ids,
                )

                executed_until = min(qpos_index + 1, cube_positions.shape[0])
                executed_px = project_points(cube_positions[:executed_until], projection, args)
                frame = draw_colored_executed_trace(
                    frame,
                    executed_px,
                    gripper_heights[:executed_until],
                    height_threshold,
                    alpha=float(args.executed_alpha),
                    width=int(args.executed_width),
                )
                planned_px = project_points(planned_positions, projection, args)
                frame = side.draw_polyline(
                    frame,
                    planned_px,
                    color=PLAN_COLOR,
                    alpha=float(args.plan_alpha),
                    width=int(args.plan_width),
                )
                frame = side.draw_polyline(
                    frame,
                    threshold_px,
                    color=side.THRESHOLD_COLOR,
                    alpha=float(args.threshold_alpha),
                    width=int(args.threshold_width),
                )
                frame = draw_trace_legend(frame)
                frame = draw_current_marker(frame, executed_px[-1], int(args.current_marker_radius))
                frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))
    finally:
        plan_env.close()
        render_env.close()

    output_path = run_dir / str(args.output_name)
    front_output_path = front_output_path_for(output_path)
    imageio.mimwrite(output_path, frames, fps=int(args.fps), quality=8, macro_block_size=1)
    imageio.mimwrite(front_output_path, front_frames, fps=int(args.fps), quality=8, macro_block_size=1)
    metadata = {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "front_output_path": str(front_output_path),
        "qpos": qpos_metadata,
        "planned_actions": plan_metadata,
        "num_video_frames": int(len(frames)),
        "frame_limit": None if frame_limit is None else int(frame_limit),
        "sync_duration": frame_limit is not None,
        "selected_plan_indices": [int(index) for index in frame_indices],
        "selected_global_steps": [int(plan_steps[index]) for index in frame_indices],
        "plan_actions_shape": [int(value) for value in plan_actions.shape],
        "height_threshold": float(height_threshold),
        "height_threshold_source": threshold_source,
        "gripper_height": height_metadata,
        "colors": {
            "executed_safe": SAFE_COLOR.astype(int).tolist(),
            "executed_unsafe": UNSAFE_COLOR.astype(int).tolist(),
            "planned_mpc": PLAN_COLOR.astype(int).tolist(),
        },
        "plan_ghost_overlay": {
            "alpha": float(args.plan_ghost_alpha),
            "geoms": "dynamic",
            "color": "renderer",
            "skips_current_state": True,
        },
        "camera": {
            "lookat": [float(value) for value in args.camera_lookat],
            "distance": float(args.camera_distance),
            "azimuth": float(args.camera_azimuth),
            "elevation": float(args.camera_elevation),
            "fovy": float(args.fovy),
        },
        "front_camera": str(args.front_camera),
    }
    side.save_json(run_dir / "mpc_plan_animation_metadata.json", metadata)
    return {"side": output_path, "front": front_output_path}, metadata


def resolve_tube_data(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tube_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find tube data file: {path}")
    return path


def parse_latent_dims(value: str | None, state_dim: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(state_dim // 2))
    dims: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim < 0 or dim >= state_dim:
            raise ValueError(f"Bottom dimension {dim} is outside [0, {state_dim - 1}].")
        dims.append(dim)
    if not dims:
        raise ValueError("At least one bottom latent dimension must be selected.")
    return dims


def select_latent_plan_indices(
    plan_steps: np.ndarray,
    *,
    start_step: int,
    plan_stride: int,
    max_plans: int | None,
) -> np.ndarray:
    selected = np.flatnonzero(plan_steps >= int(start_step))[:: int(plan_stride)]
    if max_plans is not None:
        selected = selected[: int(max_plans)]
    if selected.size == 0:
        raise ValueError(
            f"No bottom tube plans selected. Available step range is "
            f"{int(plan_steps.min())}..{int(plan_steps.max())}."
        )
    return selected


def data_axis_limits(values: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanmin(values, axis=0)
    high = np.nanmax(values, axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - float(padding) * span, high + float(padding) * span


def alpha_for_order(order: int, start: float, decay: float) -> float:
    return float(max(0.015, min(1.0, float(start) * (float(decay) ** int(order)))))


def add_tube_artists(
    ax: Any,
    horizon_x: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
    *,
    fill_color: str,
    fill_alpha: float,
    horizon_alpha_decay: float,
    line_color: str | None = None,
    line_alpha: float = 1.0,
    line_width: float = 1.0,
) -> list[Any]:
    valid = np.isfinite(center) & np.isfinite(width)
    if not np.any(valid):
        return []

    horizon_x = horizon_x[valid]
    center = center[valid]
    width = np.maximum(width[valid], 0.0)
    lower = center - width
    upper = center + width
    artists: list[Any] = []
    if horizon_x.shape[0] < 2:
        artists.append(ax.fill_between(horizon_x, lower, upper, color=fill_color, alpha=fill_alpha, linewidth=0.0))
    else:
        for horizon_idx in range(horizon_x.shape[0] - 1):
            segment_alpha = float(max(0.01, min(1.0, float(fill_alpha) * (float(horizon_alpha_decay) ** horizon_idx))))
            segment = slice(horizon_idx, horizon_idx + 2)
            artists.append(
                ax.fill_between(
                    horizon_x[segment],
                    lower[segment],
                    upper[segment],
                    color=fill_color,
                    alpha=segment_alpha,
                    linewidth=0.0,
                )
            )
    if line_color is not None:
        (line,) = ax.plot(
            horizon_x,
            center,
            color=line_color,
            linestyle=":",
            linewidth=float(line_width),
            alpha=float(line_alpha),
        )
        artists.append(line)
    return artists


def frame_from_matplotlib(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height_px, width_px, 4)[:, :, :3]
    pad_h = frame.shape[0] % 2
    pad_w = frame.shape[1] % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return frame.copy()


def render_bottom_latent_video(
    run_dir: Path,
    args: argparse.Namespace,
    frame_steps: list[int],
    out_path: Path,
) -> tuple[Path, dict[str, Any]]:
    tube_path = resolve_tube_data(run_dir)
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)
    state_dim = int(np.asarray(data["state_dim"]))
    dt = float(np.asarray(data["control_timestep"]))

    dims = parse_latent_dims(args.bottom_dims, state_dim)
    selected = select_latent_plan_indices(
        plan_steps,
        start_step=int(args.bottom_start_step),
        plan_stride=int(args.bottom_plan_stride),
        max_plans=args.bottom_max_plans,
    )
    step_to_plan_idx = {int(step): int(idx) for idx, step in enumerate(plan_steps)}
    exec_x = np.arange(executed.shape[0])

    stacked_for_limits = np.concatenate(
        [
            executed,
            centers.reshape(-1, state_dim),
            (centers - widths).reshape(-1, state_dim),
            (centers + widths).reshape(-1, state_dim),
        ],
        axis=0,
    )
    axis_low, axis_high = data_axis_limits(stacked_for_limits, float(args.bottom_data_axis_padding))

    first_half_selected = dims == list(range(state_dim // 2))
    compact_dims_selected = state_dim == 24 and (dims == list(range(state_dim)) or first_half_selected)
    n_cols = 6 if compact_dims_selected else min(3, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    panel_width = 2.45 if compact_dims_selected else 4.4
    panel_height = 1.35 if compact_dims_selected else 2.65

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=int(args.fps), quality=int(args.bottom_quality), macro_block_size=1)

    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        dpi=int(args.bottom_dpi),
        sharex=True,
    )
    axes = np.atleast_1d(axes_grid).reshape(-1)
    executed_lines: list[Any] = []
    persistent_artists: list[dict[str, Any]] = []
    current_artists_by_plan: dict[int, list[Any]] = {}
    current_visible_idx: int | None = None
    needed_current_indices = sorted(
        {
            step_to_plan_idx[int(frame_step)]
            for frame_step in frame_steps
            if int(frame_step) in step_to_plan_idx and int(frame_step) < executed.shape[0] - 1
        }
    )
    horizon_len = int(centers.shape[1])
    x_limit = float(max(executed.shape[0] - 1, 1))
    if selected.size:
        x_limit = max(x_limit, float(np.nanmax(plan_steps[selected] + horizon_len - 1)))
    if needed_current_indices:
        x_limit = max(x_limit, float(np.nanmax(plan_steps[needed_current_indices] + horizon_len - 1)))

    try:
        for panel_idx, dim in enumerate(dims):
            ax = axes[panel_idx]
            (line,) = ax.plot([], [], color=FIGURE_GREEN_DARK, linewidth=2.0, label="executed" if panel_idx == 0 else None)
            executed_lines.append(line)

            for order, plan_idx in enumerate(selected):
                horizon_x = plan_steps[plan_idx] + np.arange(centers.shape[1])
                artists = add_tube_artists(
                    ax,
                    horizon_x,
                    centers[plan_idx, :, dim],
                    widths[plan_idx, :, dim],
                    fill_color=FIGURE_GREEN,
                    fill_alpha=alpha_for_order(order, float(args.bottom_alpha), float(args.bottom_alpha_decay)),
                    horizon_alpha_decay=float(args.bottom_horizon_alpha_decay),
                )
                for artist in artists:
                    artist.set_visible(False)
                persistent_artists.append(
                    {
                        "plan_idx": int(plan_idx),
                        "start_step": int(plan_steps[plan_idx]),
                        "artists": artists,
                    }
                )

            for plan_idx in needed_current_indices:
                horizon_x = plan_steps[plan_idx] + np.arange(centers.shape[1])
                artists = add_tube_artists(
                    ax,
                    horizon_x,
                    centers[plan_idx, :, dim],
                    widths[plan_idx, :, dim],
                    fill_color=FIGURE_GREEN,
                    fill_alpha=0.34,
                    horizon_alpha_decay=float(args.bottom_horizon_alpha_decay),
                    line_color=FIGURE_CYAN,
                    line_alpha=0.85,
                    line_width=1.2,
                )
                for artist in artists:
                    artist.set_visible(False)
                current_artists_by_plan.setdefault(int(plan_idx), []).extend(artists)

            ax.text(
                0.97,
                0.91,
                f"Dim. {dim}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                color="0.15",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            ax.set_ylim(float(axis_low[dim]), float(axis_high[dim]))
            ax.set_xlim(0.0, x_limit)

        for ax in axes[len(dims) :]:
            ax.axis("off")

        fig.supxlabel("MPC step", y=0.005 if first_half_selected else (0.025 if compact_dims_selected else 0.02), fontsize=13)
        fig.suptitle(
            "Safe Latent Rollout Tubes",
            y=0.965,
            fontsize=13,
        )

        if first_half_selected:
            fig.subplots_adjust(left=0.038, right=0.962, bottom=0.22, top=0.875, wspace=0.34, hspace=0.30)
        elif compact_dims_selected:
            fig.subplots_adjust(left=0.055, right=0.945, bottom=0.105, top=0.885, wspace=0.34, hspace=0.28)
        else:
            fig.tight_layout(rect=(0.04, 0.055, 0.96, 0.885))

        for frame_step in tqdm(frame_steps, desc=f"Rendering {run_dir.name} latent tube bottom video"):
            frame_step = int(frame_step)
            executed_until = min(max(frame_step + 1, 1), executed.shape[0])
            for line, dim in zip(executed_lines, dims):
                line.set_data(exec_x[:executed_until], executed[:executed_until, dim])

            current_idx = step_to_plan_idx.get(frame_step)
            show_current = current_idx is not None and frame_step < executed.shape[0] - 1
            if current_visible_idx is not None and current_visible_idx != current_idx:
                for artist in current_artists_by_plan.get(current_visible_idx, []):
                    artist.set_visible(False)
                current_visible_idx = None
            for entry in persistent_artists:
                visible = entry["start_step"] <= frame_step and entry["plan_idx"] != current_idx
                for artist in entry["artists"]:
                    artist.set_visible(visible)

            if show_current:
                for artist in current_artists_by_plan.get(current_idx, []):
                    artist.set_visible(True)
                current_visible_idx = current_idx

            writer.append_data(frame_from_matplotlib(fig))
    finally:
        plt.close(fig)
        writer.close()

    metadata = {
        "run_dir": str(run_dir),
        "tube_data": str(tube_path),
        "output_path": str(out_path),
        "num_video_frames": int(len(frame_steps)),
        "frame_steps": [int(step) for step in frame_steps],
        "selected_plan_indices": [int(index) for index in selected],
        "dims": [int(dim) for dim in dims],
        "dt": float(dt),
        "colors": {
            "executed": FIGURE_GREEN_DARK,
            "tube_fill": FIGURE_GREEN,
            "current_imagination_line": FIGURE_CYAN,
        },
    }
    side.save_json(out_path.with_suffix(".json"), metadata)
    return out_path, metadata


def reader_frame_count(reader: Any) -> int | None:
    try:
        count = int(reader.count_frames())
        return count if count > 0 else None
    except Exception:
        pass
    try:
        count = int(reader.get_length())
        return count if count > 0 and count < 10**9 else None
    except Exception:
        return None


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == int(height):
        return image[:, :, :3]
    scale = float(height) / float(image.shape[0])
    width = max(1, int(round(float(image.shape[1]) * scale)))
    return cv2.resize(image[:, :, :3], (width, int(height)), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == int(width):
        return image[:, :, :3]
    scale = float(width) / float(image.shape[1])
    height = max(1, int(round(float(image.shape[0]) * scale)))
    return cv2.resize(image[:, :, :3], (int(width), height), interpolation=cv2.INTER_AREA)


def draw_frame_border(image: np.ndarray, color: tuple[int, int, int], thickness: int) -> np.ndarray:
    bordered = image.copy()
    height, width = bordered.shape[:2]
    if height <= 0 or width <= 0 or int(thickness) <= 0:
        return bordered
    cv2.rectangle(
        bordered,
        (0, 0),
        (width - 1, height - 1),
        tuple(int(value) for value in color),
        thickness=int(thickness),
        lineType=cv2.LINE_AA,
    )
    return bordered


def border_color_for_top_video(path: Path) -> tuple[int, int, int]:
    run_name = path.parent.name.lower()
    if "unsafe" in run_name:
        return LABEL_UNSAFE_COLOR
    if "safe" in run_name:
        return LABEL_SAFE_COLOR
    return (0, 0, 0)


def label_for_top_video(path: Path) -> tuple[str, tuple[int, int, int]] | None:
    run_name = path.parent.name.lower()
    if "unsafe" in run_name:
        return "UNSAFE", LABEL_UNSAFE_COLOR
    if "safe" in run_name:
        return "SAFE", LABEL_SAFE_COLOR
    return None


def draw_status_label(image: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    labeled = image.copy()
    height, width = labeled.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.62, float(width) / 640.0)
    thickness = max(2, int(round(float(width) / 260.0)))
    margin = max(14, int(round(float(width) * 0.035)))
    pad_x = max(9, int(round(float(width) * 0.018)))
    pad_y = max(7, int(round(float(height) * 0.018)))
    icon_gap = max(5, int(round(float(width) * 0.010)))
    icon_size = max(12, int(round(float(width) * 0.030)))
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    content_w = icon_size + icon_gap + text_size[0]
    box_w = content_w + 2 * pad_x
    box_h = max(icon_size, text_size[1]) + 2 * pad_y
    x2 = width - margin
    y2 = height - margin
    x1 = max(0, x2 - box_w)
    y1 = max(0, y2 - box_h)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    labeled = side.alpha_composite_image(labeled, overlay, 0.92).astype(np.uint8)
    cv2.rectangle(labeled, (x1, y1), (x2, y2), color, thickness=max(2, thickness - 1), lineType=cv2.LINE_AA)

    icon_x1 = x1 + pad_x
    icon_y1 = y1 + (box_h - icon_size) // 2
    icon_x2 = icon_x1 + icon_size
    icon_y2 = icon_y1 + icon_size
    if label == "SAFE":
        points = np.array(
            [
                [icon_x1 + int(0.18 * icon_size), icon_y1 + int(0.52 * icon_size)],
                [icon_x1 + int(0.42 * icon_size), icon_y1 + int(0.74 * icon_size)],
                [icon_x1 + int(0.84 * icon_size), icon_y1 + int(0.22 * icon_size)],
            ],
            dtype=np.int32,
        )
        cv2.polylines(labeled, [points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        cv2.line(labeled, (icon_x1 + 3, icon_y1 + 3), (icon_x2 - 3, icon_y2 - 3), color, thickness, cv2.LINE_AA)
        cv2.line(labeled, (icon_x2 - 3, icon_y1 + 3), (icon_x1 + 3, icon_y2 - 3), color, thickness, cv2.LINE_AA)

    text_x = icon_x2 + icon_gap
    text_y = y1 + (box_h + text_size[1]) // 2 - max(1, baseline // 2)
    cv2.putText(labeled, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return labeled


def method_label_for_top_index(index: int, color: tuple[int, int, int]) -> tuple[str, tuple[int, int, int]]:
    return ("SLS2" if int(index) < 2 else "Nominal iLQR", color)


def text_width_sls_squared(font: int, font_scale: float, thickness: int) -> tuple[int, int, int]:
    base_size, base_baseline = cv2.getTextSize("SLS", font, font_scale, thickness)
    sup_scale = font_scale * 0.62
    sup_size, sup_baseline = cv2.getTextSize("2", font, sup_scale, max(1, thickness - 1))
    return base_size[0] + sup_size[0], max(base_size[1], sup_size[1]), max(base_baseline, sup_baseline)


def draw_boxed_text_label(image: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    labeled = image.copy()
    height, width = labeled.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.58, float(width) / 690.0)
    thickness = max(2, int(round(float(width) / 285.0)))
    margin = max(14, int(round(float(width) * 0.035)))
    pad_x = max(9, int(round(float(width) * 0.018)))
    pad_y = max(7, int(round(float(height) * 0.018)))
    if label == "SLS2":
        text_w, text_h, baseline = text_width_sls_squared(font, font_scale, thickness)
    else:
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y
    x1 = margin
    y2 = height - margin
    x2 = min(width - 1, x1 + box_w)
    y1 = max(0, y2 - box_h)
    overlay = labeled.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    labeled = side.alpha_composite_image(labeled, overlay, 0.92).astype(np.uint8)
    cv2.rectangle(labeled, (x1, y1), (x2, y2), color, thickness=max(2, thickness - 1), lineType=cv2.LINE_AA)

    text_x = x1 + pad_x
    text_y = y1 + (box_h + text_h) // 2 - max(1, baseline // 2)
    if label == "SLS2":
        cv2.putText(labeled, "SLS", (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
        base_size, _ = cv2.getTextSize("SLS", font, font_scale, thickness)
        sup_scale = font_scale * 0.62
        sup_thickness = max(1, thickness - 1)
        sup_x = text_x + base_size[0]
        sup_y = text_y - max(5, int(round(float(text_h) * 0.45)))
        cv2.putText(labeled, "2", (sup_x, sup_y), font, sup_scale, color, sup_thickness, lineType=cv2.LINE_AA)
    else:
        cv2.putText(labeled, label, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return labeled


def compose_all_in_one_frame(
    top_frames: list[np.ndarray],
    bottom_frame: np.ndarray,
    *,
    top_border_colors: list[tuple[int, int, int]] | None = None,
    top_labels: list[tuple[str, tuple[int, int, int]] | None] | None = None,
    top_method_labels: list[tuple[str, tuple[int, int, int]]] | None = None,
    gap: int = 10,
    border_thickness: int = 6,
) -> np.ndarray:
    top_height = min(int(frame.shape[0]) for frame in top_frames)
    resized_top = [resize_to_height(frame, top_height) for frame in top_frames]
    if top_border_colors is not None:
        resized_top = [
            draw_frame_border(frame, color, int(border_thickness))
            for frame, color in zip(resized_top, top_border_colors, strict=False)
        ]
    if top_labels is not None:
        labeled_top: list[np.ndarray] = []
        for frame, label_color in zip(resized_top, top_labels, strict=False):
            if label_color is None:
                labeled_top.append(frame)
            else:
                label, color = label_color
                labeled_top.append(draw_status_label(frame, label, color))
        resized_top = labeled_top
    if top_method_labels is not None:
        resized_top = [
            draw_boxed_text_label(frame, label, color)
            for frame, (label, color) in zip(resized_top, top_method_labels, strict=False)
        ]
    top_width = sum(int(frame.shape[1]) for frame in resized_top) + int(gap) * (len(resized_top) - 1)
    top_row = np.full((top_height, top_width, 3), 255, dtype=np.uint8)
    x = 0
    for index, frame in enumerate(resized_top):
        if index > 0:
            x += int(gap)
        top_row[:, x : x + frame.shape[1]] = frame
        x += frame.shape[1]

    bottom = draw_frame_border(resize_to_width(bottom_frame, top_width), (0, 0, 0), int(border_thickness))
    canvas_h = top_row.shape[0] + int(gap) + bottom.shape[0]
    canvas_w = top_width
    if canvas_h % 2:
        canvas_h += 1
    if canvas_w % 2:
        canvas_w += 1
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[: top_row.shape[0], : top_row.shape[1]] = top_row
    canvas[top_row.shape[0] + int(gap) : top_row.shape[0] + int(gap) + bottom.shape[0], : bottom.shape[1]] = bottom
    return canvas


def write_combined_video(top_video_paths: list[Path], bottom_video_path: Path, output_path: Path, fps: int) -> dict[str, Any]:
    readers = [imageio.get_reader(path) for path in [*top_video_paths, bottom_video_path]]
    top_border_colors = [border_color_for_top_video(path) for path in top_video_paths]
    top_labels = [label_for_top_video(path) for path in top_video_paths]
    top_method_labels = [method_label_for_top_index(index, color) for index, color in enumerate(top_border_colors)]
    try:
        frame_counts = [reader_frame_count(reader) for reader in readers]
        finite_counts = [count for count in frame_counts if count is not None]
        if len(finite_counts) != len(readers):
            raise ValueError("Could not determine all input video frame counts.")
        num_frames = min(finite_counts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_path, fps=int(fps), quality=8, macro_block_size=1)
        try:
            for frame_idx in tqdm(range(num_frames), desc="Writing combined MPC plan animation"):
                top_frames = [reader.get_data(frame_idx) for reader in readers[:-1]]
                bottom_frame = readers[-1].get_data(frame_idx)
                writer.append_data(
                    compose_all_in_one_frame(
                        top_frames,
                        bottom_frame,
                        top_border_colors=top_border_colors,
                        top_labels=top_labels,
                        top_method_labels=top_method_labels,
                    )
                )
        finally:
            writer.close()
    finally:
        for reader in readers:
            reader.close()

    metadata = {
        "combined_output_path": str(output_path),
        "top_videos": [str(path) for path in top_video_paths],
        "bottom_video": str(bottom_video_path),
        "num_video_frames": int(num_frames),
        "fps": int(fps),
        "layout": "top videos in one row, latent tube video spanning the bottom",
        "borders": {
            "safe": list(LABEL_SAFE_COLOR),
            "unsafe": list(LABEL_UNSAFE_COLOR),
            "bottom": [0, 0, 0],
        },
        "labels": {
            "safe": "SAFE",
            "unsafe": "UNSAFE",
            "anchor": "bottom-right",
            "methods": ["SLS^2", "SLS^2", "Nominal iLQR", "Nominal iLQR"],
            "method_anchor": "bottom-left",
        },
    }
    side.save_json(output_path.with_suffix(".json"), metadata)
    return metadata


def order_safe_then_unsafe(entries: list[dict[str, Path]]) -> list[dict[str, Path]]:
    safe = [entry for entry in entries if "safe" in entry["side"].parent.name and "unsafe" not in entry["side"].parent.name]
    unsafe = [entry for entry in entries if "unsafe" in entry["side"].parent.name]
    other = [entry for entry in entries if entry not in safe and entry not in unsafe]
    return [*safe, *unsafe, *other]


def flatten_top_video_entries(entries: list[dict[str, Path]]) -> list[Path]:
    flattened: list[Path] = []
    for entry in entries:
        flattened.extend([entry["side"], entry["front"]])
    return flattened


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_dirs = [args.run_dir] if args.run_dir is not None else args.run_dirs
    resolved_run_dirs = [run_dir.expanduser().resolve() for run_dir in run_dirs]
    frame_limit = None
    if args.run_dir is None and not args.no_sync_duration and len(resolved_run_dirs) > 1:
        frame_counts = []
        for run_dir in resolved_run_dirs:
            plan_steps, _, _ = load_planned_actions(run_dir)
            frame_counts.append(selected_frame_indices(plan_steps, args).shape[0])
        frame_limit = min(frame_counts)
        print(f"Syncing batch duration to {frame_limit} frames.")
    output_entries: list[dict[str, Path]] = []
    output_metadata: dict[Path, dict[str, Any]] = {}
    for run_dir in resolved_run_dirs:
        outputs, metadata = render_animation(run_dir, args, frame_limit=frame_limit)
        output_entries.append(outputs)
        output_metadata[outputs["side"]] = metadata
        print(f"Wrote MPC plan animation: {outputs['side']}")
        print(f"Wrote plain front-view animation: {outputs['front']}")

    if not bool(args.no_combined) and len(output_entries) >= 2:
        ordered_entries = order_safe_then_unsafe(output_entries)[:2]
        ordered_top_paths = flatten_top_video_entries(ordered_entries)
        reference_metadata = output_metadata[ordered_entries[0]["side"]]
        bottom_frame_steps = [int(step) for step in reference_metadata["selected_global_steps"]]
        bottom_run_dir = (
            args.bottom_run_dir.expanduser().resolve()
            if args.bottom_run_dir is not None
            else ordered_top_paths[0].parent
        )
        bottom_video = (
            args.bottom_video.expanduser().resolve()
            if args.bottom_video is not None
            else bottom_run_dir / DEFAULT_BOTTOM_OUTPUT_NAME
        )
        bottom_video, _ = render_bottom_latent_video(bottom_run_dir, args, bottom_frame_steps, bottom_video)
        print(f"Wrote regenerated latent tube bottom video: {bottom_video}")
        combined_dir = args.combined_output_dir.expanduser().resolve()
        combined_path = combined_dir / str(args.combined_output_name)
        write_combined_video(ordered_top_paths, bottom_video, combined_path, int(args.fps))
        print(f"Wrote combined MPC plan animation: {combined_path}")


if __name__ == "__main__":
    main()
