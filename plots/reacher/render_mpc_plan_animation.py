#!/usr/bin/env python3
"""Render Reacher MPC plan animations with fingertip traces and Markov tube panel."""

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
import numpy as np
from tqdm.auto import tqdm

from plots.reacher import plot_obstacle_membership as obstacle_plot


DEFAULT_RUN_DIRS = (
    REPO_ROOT / "plots" / "reacher" / "reacher_safe_10",
    REPO_ROOT / "plots" / "reacher" / "reacher_unsafe_10",
)
DEFAULT_OBSTACLE_SUMMARY = REPO_ROOT / "reacher" / "plan" / "obstacle_data_joint_box" / "summary.json"
DEFAULT_OUTPUT_NAME = "mpc_plan_animation.mp4"
DEFAULT_COMBINED_OUTPUT_NAME = "mpc_plan_animation_combined.mp4"
DEFAULT_BOTTOM_OUTPUT_NAME = "mpc_plan_animation_markov_tubes.mp4"

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
    parser.add_argument("--combined-output-dir", type=Path, default=REPO_ROOT / "plots" / "reacher")
    parser.add_argument("--bottom-video", type=Path, default=None)
    parser.add_argument("--bottom-run-dir", type=Path, default=None)
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--plan-horizon", type=int, default=None)
    parser.add_argument("--no-sync-duration", action="store_true")
    parser.add_argument("--plan-ghost-alpha", type=float, default=0.22)
    parser.add_argument("--executed-alpha", type=float, default=0.94)
    parser.add_argument("--executed-width", type=int, default=3)
    parser.add_argument("--plan-alpha", type=float, default=0.95)
    parser.add_argument("--plan-width", type=int, default=3)
    parser.add_argument("--current-marker-radius", type=int, default=4)
    parser.add_argument("--obstacle-summary", type=Path, default=DEFAULT_OBSTACLE_SUMMARY)
    parser.add_argument("--background-alpha", type=float, default=1.0)
    parser.add_argument("--show-obstacle-circle", action="store_true", default=True)
    parser.add_argument("--no-obstacle-circle", action="store_false", dest="show_obstacle_circle")
    parser.add_argument("--obstacle-alpha", type=float, default=0.18)
    parser.add_argument("--obstacle-outline-alpha", type=float, default=0.78)
    parser.add_argument("--obstacle-outline-width", type=int, default=3)
    parser.add_argument("--bottom-dims", type=str, default="0,1,2,3,4")
    parser.add_argument("--bottom-start-step", type=int, default=1)
    parser.add_argument("--bottom-plan-stride", type=int, default=3)
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
    if args.run_dir is None and not args.run_dirs:
        raise ValueError("At least one run directory is required.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def first_present(payload: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    for key in keys:
        if key in payload.files:
            return np.asarray(payload[key])
    raise KeyError(f"None of these keys were found in {payload.files}: {keys}")


def load_executed_states(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "executed_states.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing executed states: {path}")
    payload = np.load(path, allow_pickle=False)
    qpos = np.asarray(payload["qpos"], dtype=np.float64)
    qvel = np.asarray(payload["qvel"], dtype=np.float64)
    states = np.asarray(payload["markov_states"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 2:
        raise ValueError(f"Expected qpos shape (T, >=2), got {qpos.shape}.")
    if qvel.shape[0] != qpos.shape[0]:
        raise ValueError(f"qpos/qvel length mismatch: {qpos.shape} vs {qvel.shape}.")
    return qpos, qvel, states


def raw_from_normalized_actions(run_dir: Path, actions_norm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    actions_path = run_dir / "executed_actions.npz"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Cannot convert nominal actions without {actions_path}.")
    payload = np.load(actions_path, allow_pickle=False)
    executed_norm = np.asarray(first_present(payload, "actions_norm", "executed_actions_norm"), dtype=np.float64)
    executed_raw = np.asarray(first_present(payload, "actions_raw", "executed_actions_raw"), dtype=np.float64)
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
    return actions_raw, {"converted_from": "nominal_actions", "conversion_stats_source": str(actions_path)}


def load_plan_data(run_dir: Path) -> dict[str, Any]:
    for name in ("mpc_plans.npz", "tube_data.npz"):
        path = run_dir / name
        if path.is_file():
            payload = np.load(path, allow_pickle=False)
            plan_steps = np.asarray(payload["plan_steps"], dtype=np.int64)
            centers = np.asarray(payload["nominal_centers"], dtype=np.float64)
            actions_norm = np.asarray(payload["nominal_actions"], dtype=np.float64)
            actions_raw, action_metadata = raw_from_normalized_actions(run_dir, actions_norm)
            widths = np.asarray(payload["tube_widths"], dtype=np.float64) if "tube_widths" in payload.files else None
            return {
                "source": str(path),
                "format": name,
                "plan_steps": plan_steps,
                "centers": centers,
                "widths": widths,
                "actions_raw": actions_raw,
                "action_metadata": action_metadata,
                "state_dim": int(np.asarray(payload["state_dim"])) if "state_dim" in payload.files else int(centers.shape[-1]),
            }
    raise FileNotFoundError(f"Missing mpc_plans.npz or tube_data.npz in {run_dir}.")


def selected_frame_indices(plan_steps: np.ndarray, args: argparse.Namespace, frame_limit: int | None = None) -> np.ndarray:
    frame_indices = np.arange(plan_steps.shape[0], dtype=np.int64)[:: int(args.frame_stride)]
    if frame_limit is not None:
        frame_indices = frame_indices[: int(frame_limit)]
    if args.max_frames is not None:
        frame_indices = frame_indices[: int(args.max_frames)]
    return frame_indices


def set_env_state(env: Any, qpos: np.ndarray, qvel: np.ndarray | None = None) -> None:
    physics = env.physics
    qpos = np.asarray(qpos, dtype=np.float64)
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        if qvel is None:
            physics.data.qvel[: qpos.shape[0]] = 0.0
        else:
            qvel = np.asarray(qvel, dtype=np.float64)
            physics.data.qvel[: qvel.shape[0]] = qvel


def fingertip_xy(env: Any, summary: dict[str, Any]) -> np.ndarray:
    physics = env.physics
    source = str(summary.get("tip_source", "geom:finger"))
    kind, _, name = source.partition(":")
    try:
        if kind == "site":
            idx = int(physics.model.name2id(name, "site"))
            return np.asarray(physics.data.site_xpos[idx, :2], dtype=np.float64).copy()
        idx = int(physics.model.name2id(name or "finger", "geom"))
        return np.asarray(physics.data.geom_xpos[idx, :2], dtype=np.float64).copy()
    except Exception:
        qpos = np.asarray(physics.data.qpos[:2], dtype=np.float64)
        link1 = float(summary.get("link1", 0.12))
        link2 = float(summary.get("link2", 0.12))
        return np.asarray(
            [
                link1 * np.cos(qpos[0]) + link2 * np.cos(qpos[0] + qpos[1]),
                link1 * np.sin(qpos[0]) + link2 * np.sin(qpos[0] + qpos[1]),
            ],
            dtype=np.float64,
        )


def fingertip_trace_for_qpos(env: Any, qpos: np.ndarray, qvel: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    points: list[np.ndarray] = []
    for qpos_item, qvel_item in zip(qpos, qvel, strict=True):
        set_env_state(env, qpos_item, qvel_item)
        points.append(fingertip_xy(env, summary))
    return np.stack(points, axis=0)


def rollout_plan_qpos_and_tips(
    env: Any,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    actions_raw: np.ndarray,
    summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    qpos_items: list[np.ndarray] = []
    tip_items: list[np.ndarray] = []
    set_env_state(env, start_qpos, start_qvel)
    qpos_items.append(np.asarray(env.physics.data.qpos[: start_qpos.shape[0]], dtype=np.float64).copy())
    tip_items.append(fingertip_xy(env, summary))
    for action in np.asarray(actions_raw, dtype=np.float64):
        if not np.isfinite(action).all():
            break
        env.step(action)
        qpos_items.append(np.asarray(env.physics.data.qpos[: start_qpos.shape[0]], dtype=np.float64).copy())
        tip_items.append(fingertip_xy(env, summary))
    return np.stack(qpos_items, axis=0), np.stack(tip_items, axis=0)


def world_xy_to_pixel(env: Any, xy: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    model = env.physics.model
    return obstacle_plot.world_xy_to_pixel(
        np.asarray(xy, dtype=np.float64),
        width=int(args.width),
        height=int(args.height),
        camera_z=float(model.cam_pos[int(args.camera_id)][2]),
        camera_fovy_deg=float(model.cam_fovy[int(args.camera_id)]),
    )


def alpha_composite_image(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(base, dtype=np.float32) + alpha * np.asarray(layer, dtype=np.float32)


def draw_polyline(image: np.ndarray, points_px: np.ndarray, *, color: np.ndarray, alpha: float, width: int) -> np.ndarray:
    valid = np.isfinite(points_px).all(axis=1)
    if int(np.sum(valid)) < 2:
        return image
    overlay = image.copy()
    points = np.rint(points_px[valid]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [points], False, tuple(float(v) for v in color), int(width), lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


def draw_current_marker(image: np.ndarray, point_px: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0 or not np.isfinite(point_px).all():
        return image
    overlay = image.copy()
    center = tuple(int(round(value)) for value in point_px)
    cv2.circle(overlay, center, int(radius), tuple(float(v) for v in CURRENT_COLOR), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius), (255.0, 255.0, 255.0), 1, lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, 0.95)


def obstacle_containment_mask(points_xy: np.ndarray, summary: dict[str, Any]) -> np.ndarray:
    radius = obstacle_plot.obstacle_radius_from_qpos_box(summary)
    return np.linalg.norm(np.asarray(points_xy, dtype=np.float64), axis=1) <= float(radius)


def draw_colored_executed_trace(
    image: np.ndarray,
    points_px: np.ndarray,
    points_xy: np.ndarray,
    summary: dict[str, Any],
    *,
    alpha: float,
    width: int,
) -> np.ndarray:
    composite = image
    if points_px.shape[0] < 2:
        return composite
    contained = obstacle_containment_mask(points_xy, summary)
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
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()


@contextmanager
def restored_geom_group(model: Any, geom_id: int, original_group: int) -> Iterator[None]:
    try:
        yield
    finally:
        model.geom_group[geom_id] = original_group


def render_plan_ghosts(
    image: np.ndarray,
    env: Any,
    plan_qpos: np.ndarray,
    *,
    arm_geom_ids: np.ndarray,
    scene_option: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    composite = image
    if plan_qpos.shape[0] <= 1 or float(args.plan_ghost_alpha) <= 0.0:
        return composite
    for idx, qpos in enumerate(plan_qpos[1:], start=1):
        reacher_layer, mask = obstacle_plot.render_reacher_layer(
            env,
            qpos,
            arm_geom_ids=arm_geom_ids,
            scene_option=scene_option,
            args=args,
        )
        alpha = float(args.plan_ghost_alpha) * (0.92 ** (idx - 1))
        composite = obstacle_plot.alpha_composite_mask(composite, reacher_layer, mask, alpha)
    return composite


def render_animation(run_dir: Path, args: argparse.Namespace, summary: dict[str, Any], frame_limit: int | None = None) -> tuple[Path, dict[str, Any]]:
    qpos, qvel, _ = load_executed_states(run_dir)
    plan_data = load_plan_data(run_dir)
    plan_steps = np.asarray(plan_data["plan_steps"], dtype=np.int64)
    plan_actions = np.asarray(plan_data["actions_raw"], dtype=np.float32)
    if args.plan_horizon is not None:
        plan_actions = plan_actions[:, : int(args.plan_horizon)]
    frame_indices = selected_frame_indices(plan_steps, args, frame_limit)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for animation.")

    env = obstacle_plot.make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    plan_env = obstacle_plot.make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    frames: list[np.ndarray] = []
    try:
        obstacle_plot.hide_target(env)
        obstacle_plot.hide_target(plan_env)
        model = env.physics.model
        arm_geom_ids = obstacle_plot.get_arm_geom_ids(model)
        scene_option, target_geom_id, original_group = obstacle_plot.make_segmentation_scene_option(model)
        with restored_geom_group(model, target_geom_id, original_group):
            executed_xy = fingertip_trace_for_qpos(env, qpos, qvel, summary)
            background = obstacle_plot.render_background(env, qpos[0], arm_geom_ids=arm_geom_ids, args=args)
            for plan_index in tqdm(frame_indices, desc=f"Rendering {run_dir.name} Reacher MPC plan animation"):
                global_step = int(plan_steps[plan_index])
                state_index = min(max(global_step, 0), qpos.shape[0] - 1)
                plan_qpos, plan_xy = rollout_plan_qpos_and_tips(
                    plan_env,
                    qpos[state_index],
                    qvel[state_index],
                    plan_actions[plan_index],
                    summary,
                )
                frame = render_plan_ghosts(background.copy(), env, plan_qpos, arm_geom_ids=arm_geom_ids, scene_option=scene_option, args=args)
                current_layer, current_mask = obstacle_plot.render_reacher_layer(
                    env,
                    qpos[state_index],
                    arm_geom_ids=arm_geom_ids,
                    scene_option=scene_option,
                    args=args,
                )
                frame = obstacle_plot.alpha_composite_mask(frame, current_layer, current_mask, 1.0)
                if bool(args.show_obstacle_circle):
                    frame = obstacle_plot.draw_obstacle_circle(frame, env, summary, args)
                executed_px = world_xy_to_pixel(env, executed_xy[: state_index + 1], args)
                plan_px = world_xy_to_pixel(env, plan_xy, args)
                frame = draw_colored_executed_trace(
                    frame,
                    executed_px,
                    executed_xy[: state_index + 1],
                    summary,
                    alpha=float(args.executed_alpha),
                    width=int(args.executed_width),
                )
                frame = draw_polyline(frame, plan_px, color=PLAN_COLOR, alpha=float(args.plan_alpha), width=int(args.plan_width))
                if executed_px.shape[0]:
                    frame = draw_current_marker(frame, executed_px[-1], int(args.current_marker_radius))
                frame = draw_trace_legend(frame)
                frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))
    finally:
        for item in (plan_env, env):
            close = getattr(item, "close", None)
            if close is not None:
                close()

    output_path = run_dir / str(args.output_name)
    imageio.mimwrite(output_path, frames, fps=int(args.fps), quality=8, macro_block_size=1)
    metadata = {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "plan_data": {
            "source": str(plan_data["source"]),
            "format": str(plan_data["format"]),
            "state_dim": int(plan_data["state_dim"]),
            "action_metadata": plan_data["action_metadata"],
        },
        "num_video_frames": int(len(frames)),
        "selected_plan_indices": [int(index) for index in frame_indices],
        "selected_global_steps": [int(plan_steps[index]) for index in frame_indices],
        "plan_actions_shape": [int(value) for value in plan_actions.shape],
        "obstacle_summary": str(args.obstacle_summary),
    }
    save_json(run_dir / "mpc_plan_animation_metadata.json", metadata)
    return output_path, metadata


def parse_latent_dims(value: str | None, state_dim: int) -> list[int]:
    if value is None or value.strip() == "":
        return list(range(min(5, state_dim)))
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
        raise ValueError("At least one bottom Markov dimension must be selected.")
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
    width: np.ndarray | None,
    *,
    fill_color: str,
    fill_alpha: float,
    horizon_alpha_decay: float,
    line_color: str | None = None,
) -> list[Any]:
    valid = np.isfinite(center)
    if width is not None:
        valid &= np.isfinite(width)
    if not np.any(valid):
        return []
    horizon_x = horizon_x[valid]
    center = center[valid]
    artists: list[Any] = []
    if width is not None:
        width = np.maximum(width[valid], 0.0)
        for horizon_idx in range(max(1, horizon_x.shape[0] - 1)):
            segment = slice(horizon_idx, min(horizon_idx + 2, horizon_x.shape[0]))
            alpha = float(max(0.01, min(1.0, fill_alpha * (float(horizon_alpha_decay) ** horizon_idx))))
            artists.append(
                ax.fill_between(
                    horizon_x[segment],
                    center[segment] - width[segment],
                    center[segment] + width[segment],
                    color=fill_color,
                    alpha=alpha,
                    linewidth=0.0,
                )
            )
    if line_color is not None or width is None:
        (line,) = ax.plot(horizon_x, center, color=line_color or fill_color, linestyle=":", linewidth=1.2, alpha=0.85)
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


def render_bottom_markov_video(run_dir: Path, args: argparse.Namespace, frame_steps: list[int], out_path: Path) -> tuple[Path, dict[str, Any]]:
    plan_data = load_plan_data(run_dir)
    plan_steps = np.asarray(plan_data["plan_steps"], dtype=np.int64)
    centers = np.asarray(plan_data["centers"], dtype=np.float64)
    widths = None if plan_data["widths"] is None else np.asarray(plan_data["widths"], dtype=np.float64)
    _, _, executed = load_executed_states(run_dir)
    state_dim = int(plan_data["state_dim"])
    dims = parse_latent_dims(args.bottom_dims, state_dim)
    selected = np.flatnonzero(plan_steps >= int(args.bottom_start_step))[:: int(args.bottom_plan_stride)]
    if args.bottom_max_plans is not None:
        selected = selected[: int(args.bottom_max_plans)]
    step_to_plan_idx = {int(step): int(idx) for idx, step in enumerate(plan_steps)}

    stacked_items = [executed, centers.reshape(-1, state_dim)]
    if widths is not None:
        stacked_items.extend([(centers - widths).reshape(-1, state_dim), (centers + widths).reshape(-1, state_dim)])
    axis_low, axis_high = data_axis_limits(np.concatenate(stacked_items, axis=0), float(args.bottom_data_axis_padding))
    n_cols = len(dims)
    n_rows = 1
    horizon_len = int(centers.shape[1])
    x_limit = max(float(executed.shape[0] - 1), float(np.nanmax(plan_steps[selected] + horizon_len - 1)) if selected.size else 1.0)

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=int(args.fps), quality=int(args.bottom_quality), macro_block_size=1)
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(2.55 * n_cols, 2.25), dpi=int(args.bottom_dpi), sharex=True)
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
                artists = add_tube_artists(
                    ax,
                    plan_steps[plan_idx] + np.arange(horizon_len),
                    centers[plan_idx, :, dim],
                    None if widths is None else widths[plan_idx, :, dim],
                    fill_color=FIGURE_GREEN,
                    fill_alpha=alpha,
                    horizon_alpha_decay=float(args.bottom_horizon_alpha_decay),
                )
                for artist in artists:
                    artist.set_visible(False)
                persistent_artists.append({"plan_idx": int(plan_idx), "start_step": int(plan_steps[plan_idx]), "artists": artists})
            for frame_step in frame_steps:
                current_idx = step_to_plan_idx.get(int(frame_step))
                if current_idx is None or current_idx in current_artists:
                    continue
                artists = add_tube_artists(
                    ax,
                    plan_steps[current_idx] + np.arange(horizon_len),
                    centers[current_idx, :, dim],
                    None if widths is None else widths[current_idx, :, dim],
                    fill_color=FIGURE_GREEN,
                    fill_alpha=0.34,
                    horizon_alpha_decay=float(args.bottom_horizon_alpha_decay),
                    line_color=FIGURE_CYAN,
                )
                for artist in artists:
                    artist.set_visible(False)
                current_artists.setdefault(int(current_idx), []).extend(artists)
            ax.text(0.96, 0.90, f"Dim. {dim}", transform=ax.transAxes, ha="right", va="top", fontsize=11, color="0.15", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5})
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            ax.set_ylim(float(axis_low[dim]), float(axis_high[dim]))
            ax.set_xlim(0.0, x_limit)
        axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
        fig.supxlabel("MPC step", y=0.015, fontsize=12)
        fig.suptitle("Safe Latent Rollout Tubes", y=0.965, fontsize=13)
        fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))

        visible_current: int | None = None
        exec_x = np.arange(executed.shape[0])
        for frame_step in tqdm(frame_steps, desc=f"Rendering {run_dir.name} Markov tube bottom video"):
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

    metadata = {
        "run_dir": str(run_dir),
        "plan_data_source": str(plan_data["source"]),
        "output_path": str(out_path),
        "num_video_frames": int(len(frame_steps)),
        "frame_steps": [int(step) for step in frame_steps],
        "dims": [int(dim) for dim in dims],
        "has_tube_widths": bool(widths is not None),
    }
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


def method_label_for_path(path: Path) -> tuple[str, tuple[int, int, int]]:
    name = path.parent.name.lower()
    color = LABEL_UNSAFE_COLOR if "unsafe" in name else LABEL_SAFE_COLOR
    return ("Nominal iLQR" if "unsafe" in name else "SLS2", color)


def compose_frame(top_frames: list[np.ndarray], bottom_frame: np.ndarray, top_paths: list[Path], *, gap: int = 10, border_thickness: int = 6) -> np.ndarray:
    top_height = min(int(frame.shape[0]) for frame in top_frames)
    resized_top = []
    for frame, path in zip(top_frames, top_paths, strict=True):
        color = LABEL_UNSAFE_COLOR if "unsafe" in path.parent.name.lower() else LABEL_SAFE_COLOR
        item = draw_frame_border(resize_to_height(frame, top_height), color, border_thickness)
        label = label_for_video(path)
        if label is not None:
            item = draw_status_label(item, *label)
        method_label, method_color = method_label_for_path(path)
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
            for frame_idx in tqdm(range(num_frames), desc="Writing combined Reacher MPC plan animation"):
                top_frames = [reader.get_data(frame_idx) for reader in readers[:-1]]
                bottom_frame = readers[-1].get_data(frame_idx)
                writer.append_data(compose_frame(top_frames, bottom_frame, top_video_paths))
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
    }
    save_json(output_path.with_suffix(".json"), metadata)
    return metadata


def order_safe_then_unsafe(paths: list[Path]) -> list[Path]:
    safe = [path for path in paths if "safe" in path.parent.name.lower() and "unsafe" not in path.parent.name.lower()]
    unsafe = [path for path in paths if "unsafe" in path.parent.name.lower()]
    other = [path for path in paths if path not in safe and path not in unsafe]
    return [*safe, *unsafe, *other]


def main() -> None:
    args = parse_args()
    validate_args(args)
    summary = load_json(args.obstacle_summary.expanduser().resolve())
    run_dirs = [args.run_dir] if args.run_dir is not None else args.run_dirs
    resolved_run_dirs = [run_dir.expanduser().resolve() for run_dir in run_dirs]
    frame_limit = None
    if args.run_dir is None and not args.no_sync_duration and len(resolved_run_dirs) > 1:
        frame_limit = min(selected_frame_indices(load_plan_data(run_dir)["plan_steps"], args).shape[0] for run_dir in resolved_run_dirs)
        print(f"Syncing batch duration to {frame_limit} frames.")

    outputs: list[Path] = []
    metadata_by_path: dict[Path, dict[str, Any]] = {}
    for run_dir in resolved_run_dirs:
        output_path, metadata = render_animation(run_dir, args, summary, frame_limit=frame_limit)
        outputs.append(output_path)
        metadata_by_path[output_path] = metadata
        print(f"Wrote Reacher MPC plan animation: {output_path}")

    if not bool(args.no_combined) and len(outputs) >= 2:
        top_paths = order_safe_then_unsafe(outputs)[:2]
        reference_metadata = metadata_by_path[top_paths[0]]
        frame_steps = [int(step) for step in reference_metadata["selected_global_steps"]]
        bottom_run_dir = args.bottom_run_dir.expanduser().resolve() if args.bottom_run_dir is not None else top_paths[0].parent
        bottom_video = args.bottom_video.expanduser().resolve() if args.bottom_video is not None else bottom_run_dir / DEFAULT_BOTTOM_OUTPUT_NAME
        bottom_video, _ = render_bottom_markov_video(bottom_run_dir, args, frame_steps, bottom_video)
        print(f"Wrote regenerated Markov tube bottom video: {bottom_video}")
        combined_path = args.combined_output_dir.expanduser().resolve() / str(args.combined_output_name)
        write_combined_video(top_paths, bottom_video, combined_path, int(args.fps))
        print(f"Wrote combined Reacher MPC plan animation: {combined_path}")


if __name__ == "__main__":
    main()
