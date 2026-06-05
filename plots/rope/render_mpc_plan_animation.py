#!/usr/bin/env python3
"""Render Rope MPC plan animations with side/front views and latent tube panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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

from rope.shared.lab_env import BaseEnvConfig, LabEnv, TaskState
from plots.rope import render_side_timelapse as side


DEFAULT_RUN_DIRS = (
    REPO_ROOT / "plots" / "rope" / "rope_safe_7",
    REPO_ROOT / "plots" / "rope" / "rope_unsafe_7",
)
DEFAULT_OUTPUT_NAME = "mpc_plan_animation.mp4"
DEFAULT_COMBINED_OUTPUT_NAME = "mpc_plan_animation_combined.mp4"
DEFAULT_BOTTOM_OUTPUT_NAME = "mpc_plan_animation_latent_tubes.mp4"

ROPE_TENDON_NAME = side.ROPE_TENDON_NAME
SIDE_CAMERA_NAME = side.SIDE_TIMELAPSE_CAMERA_NAME
FRONT_CAMERA_NAME = side.FRONT_TIMELAPSE_CAMERA_NAME
GEOM_OBJTYPE = side.GEOM_OBJTYPE
SITE_OBJTYPE = side.SITE_OBJTYPE
TENDON_OBJTYPE = side.TENDON_OBJTYPE

SAFE_COLOR = np.array([38.0, 166.0, 91.0], dtype=np.float32)
UNSAFE_COLOR = np.array([230.0, 126.0, 34.0], dtype=np.float32)
PLAN_COLOR = np.array([0.0, 188.0, 212.0], dtype=np.float32)
CURRENT_COLOR = np.array([20.0, 20.0, 20.0], dtype=np.float32)
FIGURE_GREEN_DARK = "#167a43"
FIGURE_GREEN = "#2ca25f"
FIGURE_CYAN = "#00bcd4"
LABEL_SAFE_COLOR = (19, 138, 54)
LABEL_UNSAFE_COLOR = (193, 18, 31)


plt.rcParams.update({"text.usetex": False, "font.family": "serif", "figure.dpi": 300})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--combined-output-name", default=DEFAULT_COMBINED_OUTPUT_NAME)
    parser.add_argument("--combined-output-dir", type=Path, default=REPO_ROOT / "plots" / "rope")
    parser.add_argument("--bottom-video", type=Path, default=None)
    parser.add_argument("--bottom-run-dir", type=Path, default=None)
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--control-decimation", type=int, default=None)
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.15, -1.5, 0.95))
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--front-camera-name", default=FRONT_CAMERA_NAME)
    parser.add_argument("--plan-ghost-alpha", type=float, default=0.30)
    parser.add_argument("--executed-alpha", type=float, default=0.92)
    parser.add_argument("--executed-width", type=int, default=3)
    parser.add_argument("--plan-alpha", type=float, default=0.95)
    parser.add_argument("--plan-width", type=int, default=3)
    parser.add_argument("--current-marker-radius", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--plan-horizon", type=int, default=None)
    parser.add_argument("--no-sync-duration", action="store_true")
    parser.add_argument("--show-obstacle", action="store_true", default=True)
    parser.add_argument("--no-show-obstacle", action="store_false", dest="show_obstacle")
    parser.add_argument("--obstacle-data-dir", type=Path, default=side.DEFAULT_OBSTACLE_DATA_DIR)
    parser.add_argument("--obstacle-y-radius", type=float, default=0.68)
    parser.add_argument("--obstacle-rgba", nargs=4, type=float, default=(0.28, 0.28, 0.28, 1.0))
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
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive when provided.")
    if args.plan_horizon is not None and int(args.plan_horizon) <= 0:
        raise ValueError("--plan-horizon must be positive when provided.")
    if args.control_decimation is not None and int(args.control_decimation) <= 0:
        raise ValueError("--control-decimation must be positive when provided.")
    if args.run_dir is None and not args.run_dirs:
        raise ValueError("At least one run directory is required.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_task_targets(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing executed states: {states_path}")
    payload = np.load(states_path, allow_pickle=False)
    task_targets = np.asarray(payload["task_targets"], dtype=np.float64)
    if task_targets.ndim != 2 or task_targets.shape[1] != 3:
        raise ValueError(f"Expected task_targets with shape (T, 3), got {task_targets.shape}.")
    return task_targets, {"source": str(states_path), "source_key": "task_targets"}


def raw_from_nominal_actions(run_dir: Path, nominal_actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    actions_norm = np.asarray(nominal_actions, dtype=np.float32).copy()
    actions_path = run_dir / "executed_actions.npz"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Cannot convert nominal tube actions without {actions_path}.")
    payload = np.load(actions_path, allow_pickle=False)
    executed_norm = np.asarray(payload["executed_actions_norm"], dtype=np.float64)
    executed_raw = np.asarray(payload["executed_actions_raw"], dtype=np.float64)
    if executed_norm.shape != executed_raw.shape or executed_norm.ndim != 2:
        raise ValueError(f"Expected matching executed action arrays, got {executed_norm.shape} and {executed_raw.shape}.")
    if actions_norm.shape[-1] != executed_norm.shape[-1]:
        raise ValueError(f"Action dim mismatch: {actions_norm.shape[-1]} vs {executed_norm.shape[-1]}.")

    action_std = np.zeros(executed_norm.shape[1], dtype=np.float64)
    action_mean = np.zeros(executed_norm.shape[1], dtype=np.float64)
    for dim in range(executed_norm.shape[1]):
        finite = np.isfinite(executed_norm[:, dim]) & np.isfinite(executed_raw[:, dim])
        if not np.any(finite):
            raise ValueError(f"No finite executed action pairs for dim {dim}.")
        x = executed_norm[finite, dim]
        y = executed_raw[finite, dim]
        if np.ptp(x) <= 1e-8:
            action_std[dim] = 0.0
            action_mean[dim] = float(np.nanmedian(y))
        else:
            action_std[dim], action_mean[dim] = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]

    finite_rows = np.isfinite(actions_norm).all(axis=-1)
    filled = np.where(finite_rows[..., None], actions_norm, 0.0)
    actions_raw = filled.astype(np.float64) * action_std.reshape(1, 1, -1) + action_mean.reshape(1, 1, -1)
    actions_raw = np.where(finite_rows[..., None], actions_raw, np.nan).astype(np.float32)
    return actions_raw, {"converted_from": "tube_data.nominal_actions", "conversion_stats_source": str(actions_path)}


def load_planned_actions(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    planned_path = run_dir / "planned_actions.npz"
    if planned_path.is_file():
        payload = np.load(planned_path, allow_pickle=False)
        return (
            np.asarray(payload["mpc_planned_action_steps"], dtype=np.int64),
            np.asarray(payload["mpc_planned_actions_raw"], dtype=np.float32),
            {"source": str(planned_path), "format": "planned_actions.npz"},
        )

    tube_path = run_dir / "tube_data.npz"
    if tube_path.is_file():
        payload = np.load(tube_path, allow_pickle=False)
        actions_raw, metadata = raw_from_nominal_actions(run_dir, np.asarray(payload["nominal_actions"]))
        return (
            np.asarray(payload["plan_steps"], dtype=np.int64),
            actions_raw,
            {"source": str(tube_path), "format": "tube_data.npz", **metadata},
        )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        return (
            np.asarray(metrics["mpc_planned_action_steps"], dtype=np.int64),
            np.asarray(metrics["mpc_planned_actions_raw"], dtype=np.float32),
            {"source": str(metrics_path), "format": "metrics.json"},
        )
    raise FileNotFoundError(f"Missing planned_actions.npz, tube_data.npz, and metrics.json in {run_dir}.")


def resolve_control_decimation(run_dir: Path, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if args.control_decimation is not None:
        return int(args.control_decimation), {"source": "cli"}
    for path in (run_dir / "trajectory_summary.json", run_dir / "metrics.json"):
        if not path.is_file():
            continue
        data = load_json(path)
        if path.name == "trajectory_summary.json":
            value = data.get("metadata", {}).get("control_decimation")
        else:
            value = data.get("control_decimation")
        if value is not None:
            return int(value), {"source": str(path)}
    return 10, {"source": "default", "note": "No saved control_decimation found."}


def make_env(args: argparse.Namespace, *, enable_proxy_rope: bool = False) -> tuple[LabEnv, dict[str, Any]]:
    asset_xml, worldbody_xml, obstacle_metadata = side.obstacle_speedbump_xml(args)
    camera_xml, camera_metadata = side.camera_xml(args)
    merged_worldbody_xml = "\n".join(part for part in (worldbody_xml, camera_xml) if part.strip())
    env = LabEnv(
        base_config=BaseEnvConfig(
            enable_proxy_rope=bool(enable_proxy_rope),
            asset_extra_xml=asset_xml,
            worldbody_extra_xml=merged_worldbody_xml,
            offscreen_width=max(int(args.width), 640),
            offscreen_height=max(int(args.height), 480),
        )
    )
    return env, {"obstacle": obstacle_metadata, "camera": camera_metadata}


def proxy_rope_bottom_point(env: LabEnv) -> np.ndarray:
    points = env.get_proxy_rope_points()
    return points[int(np.argmin(points[:, 2]))].astype(np.float64, copy=True)


def proxy_rope_bottom_points_for_task_targets(env: LabEnv, task_targets: np.ndarray) -> np.ndarray:
    bottom_points: list[np.ndarray] = []
    for target in np.asarray(task_targets, dtype=np.float64):
        env.reset(TaskState.from_array(target))
        bottom_points.append(proxy_rope_bottom_point(env))
    return np.stack(bottom_points, axis=0)


def set_env_to_task_target(env: LabEnv, task_target: np.ndarray, *, first_frame: bool) -> None:
    side.set_env_to_task_target_continuous(env, np.asarray(task_target, dtype=np.float64), first_frame=first_frame)


def planned_rollout_task_targets(
    env: LabEnv,
    start_task_target: np.ndarray,
    actions: np.ndarray,
    *,
    control_decimation: int,
) -> np.ndarray:
    targets: list[np.ndarray] = []
    env.reset(TaskState.from_array(start_task_target))
    targets.append(env.task_controller.desired_state.as_array().astype(np.float64))
    for action in actions:
        if not np.isfinite(action).all():
            break
        env.apply_task_delta(np.asarray(action, dtype=np.float64))
        env.step(int(control_decimation))
        targets.append(env.task_controller.desired_state.as_array().astype(np.float64))
    return np.stack(targets, axis=0)


@contextmanager
def hidden_dynamic_objects(env: LabEnv, arm_geoms: np.ndarray, arm_sites: np.ndarray, rope_tendon_id: int) -> Iterator[None]:
    with side.hidden_dynamic_objects(env, arm_geoms, arm_sites, rope_tendon_id):
        yield


def alpha_composite_image(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base_float = np.asarray(base, dtype=np.float32)
    layer_float = np.asarray(layer, dtype=np.float32)
    return ((1.0 - alpha) * base_float + alpha * layer_float).astype(np.float32)


def alpha_composite_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    output = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output[mask] = (1.0 - alpha) * output[mask] + alpha * layer.astype(np.float32)[mask]
    return output


def draw_polyline(image: np.ndarray, points_px: np.ndarray, *, color: np.ndarray, alpha: float, width: int) -> np.ndarray:
    valid = np.isfinite(points_px).all(axis=1)
    if int(np.sum(valid)) < 2:
        return image
    overlay = image.copy()
    points = np.rint(points_px[valid]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [points], False, tuple(float(v) for v in color), int(width), lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


def half_ellipse_height_at_reach(reach: np.ndarray, obstacle_metadata: dict[str, Any]) -> np.ndarray:
    reach_values = np.asarray(reach, dtype=np.float64)
    obstacle_reach = np.asarray(obstacle_metadata["obstacle_reach"], dtype=np.float64)
    base_height = float(obstacle_metadata["obstacle_base_height"])
    peak_height = float(obstacle_metadata["obstacle_peak_height"])
    center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    half_width = 0.5 * (float(obstacle_reach[1]) - float(obstacle_reach[0]))
    if half_width <= 0.0:
        return np.full_like(reach_values, base_height, dtype=np.float64)
    normalized = (reach_values - center) / half_width
    profile = np.sqrt(np.clip(1.0 - normalized * normalized, 0.0, None))
    return base_height + (peak_height - base_height) * profile


def obstacle_containment_mask(points_world: np.ndarray, obstacle_metadata: dict[str, Any] | None) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if obstacle_metadata is None or points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    obstacle_reach = np.asarray(obstacle_metadata["obstacle_reach"], dtype=np.float64)
    y_radius = float(obstacle_metadata.get("size", [0.0, np.inf, 0.0])[1])
    in_reach = (points[:, 0] >= float(obstacle_reach[0])) & (points[:, 0] <= float(obstacle_reach[1]))
    in_y = np.abs(points[:, 1]) <= y_radius
    below_profile = points[:, 2] <= half_ellipse_height_at_reach(points[:, 0], obstacle_metadata)
    return in_reach & in_y & below_profile


def draw_colored_executed_trace(
    image: np.ndarray,
    points_px: np.ndarray,
    points_world: np.ndarray,
    obstacle_metadata: dict[str, Any] | None,
    *,
    alpha: float,
    width: int,
) -> np.ndarray:
    composite = image
    if points_px.shape[0] < 2:
        return composite
    contained = obstacle_containment_mask(points_world, obstacle_metadata)
    for idx in range(points_px.shape[0] - 1):
        segment = points_px[idx : idx + 2]
        if not np.isfinite(segment).all():
            continue
        unsafe = bool(contained[idx]) or bool(contained[min(idx + 1, contained.shape[0] - 1)])
        composite = draw_polyline(
            composite,
            segment,
            color=UNSAFE_COLOR if unsafe else SAFE_COLOR,
            alpha=alpha,
            width=width,
        )
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
        loc="upper left",
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
    cv2.circle(overlay, center, int(radius), tuple(float(v) for v in CURRENT_COLOR), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius), (255.0, 255.0, 255.0), 1, lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, 0.95)


def project_world_points(env: LabEnv, camera_id: int, points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    camera_position = env.data.cam_xpos[camera_id].copy()
    camera_rotation = env.data.cam_xmat[camera_id].reshape(3, 3).copy()
    camera_points = (np.asarray(points, dtype=np.float64) - camera_position) @ camera_rotation
    depth = -camera_points[:, 2]
    fovy = np.deg2rad(float(env.model.cam_fovy[camera_id]))
    focal = 0.5 * float(args.height) / np.tan(0.5 * fovy)
    pixels = np.full((camera_points.shape[0], 2), np.nan, dtype=np.float64)
    visible = depth > 1e-6
    pixels[visible, 0] = 0.5 * float(args.width) + focal * camera_points[visible, 0] / depth[visible]
    pixels[visible, 1] = 0.5 * float(args.height) - focal * camera_points[visible, 1] / depth[visible]
    return pixels


def render_plan_ghosts(
    image: np.ndarray,
    env: LabEnv,
    renderer: mujoco.Renderer,
    camera_id: int,
    planned_targets: np.ndarray,
    arm_geoms: np.ndarray,
    arm_sites: np.ndarray,
    rope_tendon_id: int,
    *,
    alpha: float,
) -> np.ndarray:
    composite = image
    if planned_targets.shape[0] <= 1 or float(alpha) <= 0.0:
        return composite
    for idx, target in enumerate(planned_targets[1:], start=1):
        set_env_to_task_target(env, target, first_frame=False)
        rgb = side.render_frame(renderer, env, camera_id).astype(np.float32)
        segmentation = side.render_segmentation(renderer, env, camera_id)
        arm_mask = side.mask_from_object_ids(segmentation, arm_geoms, GEOM_OBJTYPE) | side.mask_from_object_ids(
            segmentation, arm_sites, SITE_OBJTYPE
        )
        rope_mask = (segmentation[..., 1] == TENDON_OBJTYPE) & (segmentation[..., 0] == rope_tendon_id)
        horizon_alpha = float(alpha) * (0.92 ** (idx - 1))
        composite = alpha_composite_mask(composite, rgb, arm_mask | rope_mask, horizon_alpha)
    return composite


def selected_frame_indices(plan_steps: np.ndarray, args: argparse.Namespace, frame_limit: int | None = None) -> np.ndarray:
    frame_indices = np.arange(plan_steps.shape[0], dtype=np.int64)[:: int(args.frame_stride)]
    if frame_limit is not None:
        frame_indices = frame_indices[: int(frame_limit)]
    if args.max_frames is not None:
        frame_indices = frame_indices[: int(args.max_frames)]
    return frame_indices


def front_output_path_for(side_output_path: Path) -> Path:
    return side_output_path.with_name(f"{side_output_path.stem}_front{side_output_path.suffix or '.mp4'}")


def render_animation(run_dir: Path, args: argparse.Namespace, frame_limit: int | None = None) -> tuple[dict[str, Path], dict[str, Any]]:
    task_targets, target_metadata = load_task_targets(run_dir)
    plan_steps, plan_actions, plan_metadata = load_planned_actions(run_dir)
    if plan_steps.shape[0] != plan_actions.shape[0]:
        raise ValueError(f"Plan steps/actions length mismatch: {plan_steps.shape[0]} vs {plan_actions.shape[0]}.")
    if args.plan_horizon is not None:
        plan_actions = plan_actions[:, : int(args.plan_horizon)]
    frame_indices = selected_frame_indices(plan_steps, args, frame_limit)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for animation.")
    control_decimation, control_metadata = resolve_control_decimation(run_dir, args)

    render_env, env_metadata = make_env(args)
    plan_env, _ = make_env(args)
    proxy_trace_env, _ = make_env(args, enable_proxy_rope=True)
    frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []
    try:
        side_camera_id = render_env.model.camera(SIDE_CAMERA_NAME).id if SIDE_CAMERA_NAME in [
            mujoco.mj_id2name(render_env.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(render_env.model.ncam)
        ] else None
        if side_camera_id is None:
            raise RuntimeError(f"Camera {SIDE_CAMERA_NAME!r} was not added to the Rope model.")
        front_camera_id = render_env.model.camera(str(args.front_camera_name)).id
        arm_geoms = side.arm_geom_ids(render_env.model)
        arm_sites = side.arm_site_ids(render_env.model)
        rope_tendon_id = render_env.model.tendon(ROPE_TENDON_NAME).id
        executed_world = proxy_rope_bottom_points_for_task_targets(proxy_trace_env, task_targets)

        with mujoco.Renderer(render_env.model, height=int(args.height), width=int(args.width)) as renderer:
            for plan_index in tqdm(frame_indices, desc=f"Rendering {run_dir.name} Rope MPC plan animation"):
                global_step = int(plan_steps[plan_index])
                target_index = min(max(global_step, 0), task_targets.shape[0] - 1)
                current_target = task_targets[target_index]

                set_env_to_task_target(render_env, current_target, first_frame=target_index == 0)
                front_frames.append(side.render_frame(renderer, render_env, front_camera_id))
                with hidden_dynamic_objects(render_env, arm_geoms, arm_sites, rope_tendon_id):
                    frame = side.render_frame(renderer, render_env, side_camera_id).astype(np.float32)

                planned_targets = planned_rollout_task_targets(
                    plan_env,
                    current_target,
                    plan_actions[plan_index],
                    control_decimation=control_decimation,
                )
                planned_world = proxy_rope_bottom_points_for_task_targets(proxy_trace_env, planned_targets)
                frame = render_plan_ghosts(
                    frame,
                    render_env,
                    renderer,
                    side_camera_id,
                    planned_targets,
                    arm_geoms,
                    arm_sites,
                    rope_tendon_id,
                    alpha=float(args.plan_ghost_alpha),
                )

                set_env_to_task_target(render_env, current_target, first_frame=False)
                rgb = side.render_frame(renderer, render_env, side_camera_id).astype(np.float32)
                segmentation = side.render_segmentation(renderer, render_env, side_camera_id)
                current_mask = side.mask_from_object_ids(segmentation, arm_geoms, GEOM_OBJTYPE) | side.mask_from_object_ids(
                    segmentation, arm_sites, SITE_OBJTYPE
                )
                current_mask |= (segmentation[..., 1] == TENDON_OBJTYPE) & (segmentation[..., 0] == rope_tendon_id)
                frame = alpha_composite_mask(frame, rgb, current_mask, 1.0)

                executed_points = executed_world[: target_index + 1]
                executed_px = project_world_points(render_env, side_camera_id, executed_points, args)
                frame = draw_colored_executed_trace(
                    frame,
                    executed_px,
                    executed_points,
                    env_metadata.get("obstacle"),
                    alpha=float(args.executed_alpha),
                    width=int(args.executed_width),
                )
                planned_px = project_world_points(render_env, side_camera_id, planned_world, args)
                frame = draw_polyline(
                    frame,
                    planned_px,
                    color=PLAN_COLOR,
                    alpha=float(args.plan_alpha),
                    width=int(args.plan_width),
                )
                if executed_px.shape[0]:
                    frame = draw_current_marker(frame, executed_px[-1], int(args.current_marker_radius))
                frame = draw_trace_legend(frame)
                frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))
    finally:
        del proxy_trace_env
        del plan_env
        del render_env

    output_path = run_dir / str(args.output_name)
    front_output_path = front_output_path_for(output_path)
    imageio.mimwrite(output_path, frames, fps=int(args.fps), quality=8, macro_block_size=1)
    imageio.mimwrite(front_output_path, front_frames, fps=int(args.fps), quality=8, macro_block_size=1)
    metadata = {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "front_output_path": str(front_output_path),
        "task_targets": target_metadata,
        "planned_actions": plan_metadata,
        "control_decimation": int(control_decimation),
        "control_decimation_metadata": control_metadata,
        "num_video_frames": int(len(frames)),
        "selected_plan_indices": [int(index) for index in frame_indices],
        "selected_global_steps": [int(plan_steps[index]) for index in frame_indices],
        "plan_actions_shape": [int(value) for value in plan_actions.shape],
        "front_camera": str(args.front_camera_name),
        **env_metadata,
    }
    save_json(run_dir / "mpc_plan_animation_metadata.json", metadata)
    return {"side": output_path, "front": front_output_path}, metadata


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


def data_axis_limits(values: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanmin(values, axis=0)
    high = np.nanmax(values, axis=0)
    span = np.maximum(high - low, 1e-6)
    return low - float(padding) * span, high + float(padding) * span


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
) -> list[Any]:
    valid = np.isfinite(center) & np.isfinite(width)
    if not np.any(valid):
        return []
    horizon_x = horizon_x[valid]
    center = center[valid]
    width = np.maximum(width[valid], 0.0)
    artists: list[Any] = []
    for horizon_idx in range(max(1, horizon_x.shape[0] - 1)):
        segment = slice(horizon_idx, min(horizon_idx + 2, horizon_x.shape[0]))
        alpha = float(max(0.01, min(1.0, fill_alpha * (float(horizon_alpha_decay) ** horizon_idx))))
        artists.append(ax.fill_between(horizon_x[segment], center[segment] - width[segment], center[segment] + width[segment], color=fill_color, alpha=alpha, linewidth=0.0))
    if line_color is not None:
        (line,) = ax.plot(horizon_x, center, color=line_color, linestyle=":", linewidth=1.2, alpha=0.85)
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
    tube_path = run_dir / "tube_data.npz"
    if not tube_path.is_file():
        raise FileNotFoundError(f"Bottom latent video requires tube_data.npz: {tube_path}")
    data = np.load(tube_path, allow_pickle=False)
    plan_steps = np.asarray(data["plan_steps"], dtype=np.int64)
    centers = np.asarray(data["nominal_centers"], dtype=np.float64)
    widths = np.asarray(data["tube_widths"], dtype=np.float64)
    executed = np.asarray(data["executed_markov_states"], dtype=np.float64)
    state_dim = int(np.asarray(data["state_dim"]))
    dims = parse_latent_dims(args.bottom_dims, state_dim)
    selected = np.flatnonzero(plan_steps >= int(args.bottom_start_step))[:: int(args.bottom_plan_stride)]
    if args.bottom_max_plans is not None:
        selected = selected[: int(args.bottom_max_plans)]
    step_to_plan_idx = {int(step): int(idx) for idx, step in enumerate(plan_steps)}

    stacked = np.concatenate([executed, centers.reshape(-1, state_dim), (centers - widths).reshape(-1, state_dim), (centers + widths).reshape(-1, state_dim)], axis=0)
    axis_low, axis_high = data_axis_limits(stacked, float(args.bottom_data_axis_padding))
    compact = state_dim == 24 and dims == list(range(state_dim // 2))
    n_cols = 6 if compact else min(3, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    panel_width = 2.45 if compact else 4.4
    panel_height = 1.35 if compact else 2.65
    horizon_len = int(centers.shape[1])
    x_limit = max(float(executed.shape[0] - 1), float(np.nanmax(plan_steps[selected] + horizon_len - 1)) if selected.size else 1.0)

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=int(args.fps), quality=int(args.bottom_quality), macro_block_size=1)
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(panel_width * n_cols, panel_height * n_rows), dpi=int(args.bottom_dpi), sharex=True)
    axes = np.atleast_1d(axes_grid).reshape(-1)
    executed_lines: list[Any] = []
    persistent_artists: list[dict[str, Any]] = []
    current_artists: dict[int, list[Any]] = {}
    try:
        for panel_idx, dim in enumerate(dims):
            ax = axes[panel_idx]
            (line,) = ax.plot([], [], color=FIGURE_GREEN_DARK, linewidth=2.0, label="executed" if panel_idx == 0 else None)
            executed_lines.append(line)
            for order, plan_idx in enumerate(selected):
                alpha = float(max(0.015, min(1.0, float(args.bottom_alpha) * (float(args.bottom_alpha_decay) ** order))))
                artists = add_tube_artists(ax, plan_steps[plan_idx] + np.arange(horizon_len), centers[plan_idx, :, dim], widths[plan_idx, :, dim], fill_color=FIGURE_GREEN, fill_alpha=alpha, horizon_alpha_decay=float(args.bottom_horizon_alpha_decay))
                for artist in artists:
                    artist.set_visible(False)
                persistent_artists.append({"plan_idx": int(plan_idx), "start_step": int(plan_steps[plan_idx]), "artists": artists})
            for frame_step in frame_steps:
                current_idx = step_to_plan_idx.get(int(frame_step))
                if current_idx is None or current_idx in current_artists:
                    continue
                artists = add_tube_artists(ax, plan_steps[current_idx] + np.arange(horizon_len), centers[current_idx, :, dim], widths[current_idx, :, dim], fill_color=FIGURE_GREEN, fill_alpha=0.34, horizon_alpha_decay=float(args.bottom_horizon_alpha_decay), line_color=FIGURE_CYAN)
                for artist in artists:
                    artist.set_visible(False)
                current_artists.setdefault(int(current_idx), []).extend(artists)
            ax.text(0.97, 0.91, f"Dim. {dim}", transform=ax.transAxes, ha="right", va="top", fontsize=13, color="0.15", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5})
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            ax.set_ylim(float(axis_low[dim]), float(axis_high[dim]))
            ax.set_xlim(0.0, x_limit)
        for ax in axes[len(dims) :]:
            ax.axis("off")
        axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
        fig.supxlabel("MPC step", y=0.005 if compact else 0.025, fontsize=13)
        fig.suptitle("Safe Latent Rollout Tubes", y=0.965, fontsize=13)
        if compact:
            fig.subplots_adjust(left=0.038, right=0.962, bottom=0.22, top=0.875, wspace=0.34, hspace=0.30)
        else:
            fig.tight_layout(rect=(0.04, 0.055, 0.96, 0.885))

        visible_current: int | None = None
        exec_x = np.arange(executed.shape[0])
        for frame_step in tqdm(frame_steps, desc=f"Rendering {run_dir.name} latent tube bottom video"):
            frame_step = int(frame_step)
            until = min(max(frame_step + 1, 1), executed.shape[0])
            for line, dim in zip(executed_lines, dims):
                line.set_data(exec_x[:until], executed[:until, dim])
            current_idx = step_to_plan_idx.get(frame_step)
            if visible_current is not None and visible_current != current_idx:
                for artist in current_artists.get(visible_current, []):
                    artist.set_visible(False)
                visible_current = None
            for entry in persistent_artists:
                visible = entry["start_step"] <= frame_step and entry["plan_idx"] != current_idx
                for artist in entry["artists"]:
                    artist.set_visible(visible)
            if current_idx is not None:
                for artist in current_artists.get(current_idx, []):
                    artist.set_visible(True)
                visible_current = current_idx
            writer.append_data(frame_from_matplotlib(fig))
    finally:
        plt.close(fig)
        writer.close()

    metadata = {"run_dir": str(run_dir), "tube_data": str(tube_path), "output_path": str(out_path), "num_video_frames": int(len(frame_steps)), "frame_steps": [int(step) for step in frame_steps], "dims": [int(dim) for dim in dims]}
    save_json(out_path.with_suffix(".json"), metadata)
    return out_path, metadata


def reader_frame_count(reader: Any) -> int | None:
    try:
        count = int(reader.count_frames())
        return count if count > 0 else None
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
    cv2.rectangle(bordered, (0, 0), (bordered.shape[1] - 1, bordered.shape[0] - 1), tuple(int(v) for v in color), int(thickness), lineType=cv2.LINE_AA)
    return bordered


def label_for_video(path: Path) -> tuple[str, tuple[int, int, int]] | None:
    name = path.parent.name.lower()
    if "unsafe" in name:
        return "UNSAFE", LABEL_UNSAFE_COLOR
    if "safe" in name:
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
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    labeled = alpha_composite_image(labeled, overlay, 0.92).astype(np.uint8)
    cv2.rectangle(labeled, (x1, y1), (x2, y2), color, max(2, thickness - 1), lineType=cv2.LINE_AA)

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
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    labeled = alpha_composite_image(labeled, overlay, 0.92).astype(np.uint8)
    cv2.rectangle(labeled, (x1, y1), (x2, y2), color, max(2, thickness - 1), lineType=cv2.LINE_AA)

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


def compose_frame(top_frames: list[np.ndarray], bottom_frame: np.ndarray, top_paths: list[Path], *, gap: int = 10, border_thickness: int = 6) -> np.ndarray:
    top_height = min(int(frame.shape[0]) for frame in top_frames)
    resized_top = []
    for index, (frame, path) in enumerate(zip(top_frames, top_paths)):
        color = LABEL_UNSAFE_COLOR if "unsafe" in path.parent.name.lower() else LABEL_SAFE_COLOR
        item = draw_frame_border(resize_to_height(frame, top_height), color, border_thickness)
        label = label_for_video(path)
        if label is not None:
            item = draw_status_label(item, *label)
        method_label, method_color = method_label_for_top_index(index, color)
        item = draw_boxed_text_label(item, method_label, method_color)
        resized_top.append(item)
    top_width = sum(frame.shape[1] for frame in resized_top) + gap * (len(resized_top) - 1)
    top_row = np.full((top_height, top_width, 3), 255, dtype=np.uint8)
    x = 0
    for idx, frame in enumerate(resized_top):
        if idx > 0:
            x += gap
        top_row[:, x : x + frame.shape[1]] = frame
        x += frame.shape[1]
    bottom = draw_frame_border(resize_to_width(bottom_frame, top_width), (0, 0, 0), border_thickness)
    canvas_h = top_row.shape[0] + gap + bottom.shape[0]
    canvas_w = top_width
    if canvas_h % 2:
        canvas_h += 1
    if canvas_w % 2:
        canvas_w += 1
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[: top_row.shape[0], : top_row.shape[1]] = top_row
    canvas[top_row.shape[0] + gap : top_row.shape[0] + gap + bottom.shape[0], : bottom.shape[1]] = bottom
    return canvas


def write_combined_video(top_video_paths: list[Path], bottom_video_path: Path, output_path: Path, fps: int) -> dict[str, Any]:
    readers = [imageio.get_reader(path) for path in [*top_video_paths, bottom_video_path]]
    try:
        counts = [reader_frame_count(reader) for reader in readers]
        if any(count is None for count in counts):
            raise ValueError("Could not determine all input video frame counts.")
        num_frames = min(int(count) for count in counts if count is not None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output_path, fps=int(fps), quality=8, macro_block_size=1)
        try:
            for frame_idx in tqdm(range(num_frames), desc="Writing combined Rope MPC plan animation"):
                top_frames = [reader.get_data(frame_idx) for reader in readers[:-1]]
                bottom_frame = readers[-1].get_data(frame_idx)
                writer.append_data(compose_frame(top_frames, bottom_frame, top_video_paths))
        finally:
            writer.close()
    finally:
        for reader in readers:
            reader.close()
    metadata = {"combined_output_path": str(output_path), "top_videos": [str(path) for path in top_video_paths], "bottom_video": str(bottom_video_path), "num_video_frames": int(num_frames), "fps": int(fps)}
    save_json(output_path.with_suffix(".json"), metadata)
    return metadata


def order_safe_then_unsafe(entries: list[dict[str, Path]]) -> list[dict[str, Path]]:
    safe = [entry for entry in entries if "safe" in entry["side"].parent.name.lower() and "unsafe" not in entry["side"].parent.name.lower()]
    unsafe = [entry for entry in entries if "unsafe" in entry["side"].parent.name.lower()]
    other = [entry for entry in entries if entry not in safe and entry not in unsafe]
    return [*safe, *unsafe, *other]


def flatten_entries(entries: list[dict[str, Path]]) -> list[Path]:
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
        frame_limit = min(selected_frame_indices(load_planned_actions(run_dir)[0], args).shape[0] for run_dir in resolved_run_dirs)
        print(f"Syncing batch duration to {frame_limit} frames.")

    outputs: list[dict[str, Path]] = []
    metadata_by_side: dict[Path, dict[str, Any]] = {}
    for run_dir in resolved_run_dirs:
        output_paths, metadata = render_animation(run_dir, args, frame_limit=frame_limit)
        outputs.append(output_paths)
        metadata_by_side[output_paths["side"]] = metadata
        print(f"Wrote Rope MPC plan animation: {output_paths['side']}")
        print(f"Wrote plain front-view animation: {output_paths['front']}")

    if not bool(args.no_combined) and len(outputs) >= 2:
        ordered_entries = order_safe_then_unsafe(outputs)[:2]
        top_paths = flatten_entries(ordered_entries)
        reference_metadata = metadata_by_side[ordered_entries[0]["side"]]
        frame_steps = [int(step) for step in reference_metadata["selected_global_steps"]]
        bottom_run_dir = args.bottom_run_dir.expanduser().resolve() if args.bottom_run_dir is not None else ordered_entries[0]["side"].parent
        bottom_video = args.bottom_video.expanduser().resolve() if args.bottom_video is not None else bottom_run_dir / DEFAULT_BOTTOM_OUTPUT_NAME
        bottom_video, _ = render_bottom_latent_video(bottom_run_dir, args, frame_steps, bottom_video)
        print(f"Wrote regenerated latent tube bottom video: {bottom_video}")
        combined_path = args.combined_output_dir.expanduser().resolve() / str(args.combined_output_name)
        write_combined_video(top_paths, bottom_video, combined_path, int(args.fps))
        print(f"Wrote combined Rope MPC plan animation: {combined_path}")


if __name__ == "__main__":
    main()
