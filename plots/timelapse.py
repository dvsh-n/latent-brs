#!/usr/bin/env python3
"""Render Reacher safe/unsafe rollout timelapse composites.

The output stacks a MuJoCo background with the reacher hidden, a synthetic
workspace circle obstacle, translucent segmented reacher renders over time,
and a trace of the fingertip path.
"""

from __future__ import annotations

import argparse
import json
import math
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from reacher.plan.obs_data_collect import (
    configure_offscreen_framebuffer,
    forward_kinematics,
    get_arm_geom_ids,
    hide_target,
    make_render_env,
    make_segmentation_scene_option,
)


DEFAULT_ROOT = REPO_ROOT / "plots" / "reacher"
DEFAULT_TRAJECTORIES = ("reacher_safe", "reacher_unsafe")
DEFAULT_OBSTACLE_SUMMARY = REPO_ROOT / "reacher" / "plan" / "obstacle_data_joint_box" / "summary.json"
DEFAULT_OUTPUT_NAME = "timelapse.png"
DEFAULT_COMBINED_OUTPUT_NAME = "reacher_timelapse.png"
OBSTACLE_COLOR = np.array([230.0, 83.0, 35.0], dtype=np.float32)
LINE_COLOR = np.array([20.0, 20.0, 20.0], dtype=np.float32)
TIP_COLOR = np.array([255.0, 245.0, 80.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--obstacle-summary", type=Path, default=DEFAULT_OBSTACLE_SUMMARY)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--combined-output-name", default=DEFAULT_COMBINED_OUTPUT_NAME)
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--sample-by", choices=("motion", "index"), default="motion")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--background-alpha", type=float, default=1.0)
    parser.add_argument(
        "--obstacle-center-source",
        choices=("base", "summary", "manual"),
        default="base",
        help="Where to place the artificial obstacle circle. Default: reacher base/center.",
    )
    parser.add_argument("--obstacle-center-x", type=float, default=None)
    parser.add_argument("--obstacle-center-y", type=float, default=None)
    parser.add_argument(
        "--obstacle-radius-source",
        choices=("max-tip-distance", "summary", "manual"),
        default="max-tip-distance",
        help="How to choose the obstacle circle radius. Default: max fingertip distance in the obstacle qpos box.",
    )
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--obstacle-alpha", type=float, default=0.22)
    parser.add_argument("--obstacle-outline-alpha", type=float, default=0.78)
    parser.add_argument("--obstacle-outline-width", type=int, default=3)
    parser.add_argument("--reacher-min-alpha", "--mask-min-alpha", dest="reacher_min_alpha", type=float, default=0.20)
    parser.add_argument("--reacher-max-alpha", "--mask-max-alpha", dest="reacher_max_alpha", type=float, default=0.82)
    parser.add_argument("--line-alpha", type=float, default=0.88)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--tip-marker-alpha", type=float, default=0.95)
    parser.add_argument("--tip-marker-radius", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    for name in (
        "background_alpha",
        "obstacle_alpha",
        "obstacle_outline_alpha",
        "reacher_min_alpha",
        "reacher_max_alpha",
        "line_alpha",
        "tip_marker_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}.")
    if args.obstacle_center_source == "manual" and (args.obstacle_center_x is None or args.obstacle_center_y is None):
        raise ValueError("--obstacle-center-source manual requires --obstacle-center-x and --obstacle-center-y.")
    if args.obstacle_radius_source == "manual" and args.obstacle_radius is None:
        raise ValueError("--obstacle-radius-source manual requires --obstacle-radius.")
    if args.obstacle_radius is not None and float(args.obstacle_radius) <= 0.0:
        raise ValueError("--obstacle-radius must be positive.")
    if float(args.reacher_min_alpha) > float(args.reacher_max_alpha):
        raise ValueError("--reacher-min-alpha must be less than or equal to --reacher-max-alpha.")
    if int(args.line_width) <= 0:
        raise ValueError("--line-width must be positive.")
    if int(args.tip_marker_radius) < 0:
        raise ValueError("--tip-marker-radius must be non-negative.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_rollout_qpos(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    states_path = run_dir / "executed_states.npz"
    if states_path.is_file():
        states = np.load(states_path)
        if "qpos" not in states:
            raise ValueError(f"Expected qpos in {states_path}.")
        qpos = np.asarray(states["qpos"], dtype=np.float64)
        return qpos, {"source": str(states_path), "source_key": "qpos"}

    qpos_json_path = run_dir / "executed_qpos.json"
    if qpos_json_path.is_file():
        payload = load_json(qpos_json_path)
        if "qpos_over_time" not in payload:
            raise ValueError(f"Expected qpos_over_time in {qpos_json_path}.")
        qpos = np.asarray(payload["qpos_over_time"], dtype=np.float64)
        return qpos, {"source": str(qpos_json_path), "source_key": "qpos_over_time"}

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        payload = load_json(summary_path)
        records = payload.get("step_records", [])
        qpos = np.asarray([record["qpos"] for record in records if "qpos" in record], dtype=np.float64)
        return qpos, {"source": str(summary_path), "source_key": "step_records[].qpos"}

    raise FileNotFoundError(f"No supported rollout qpos artifact found in {run_dir}.")


def ensure_qpos_shape(qpos: np.ndarray, run_dir: Path) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] < 2:
        raise ValueError(f"Expected qpos with shape (T, >=2), got {qpos.shape} in {run_dir}.")
    return np.asarray(qpos[:, :2], dtype=np.float64)


def tip_xy_from_qpos(qpos: np.ndarray, *, link1: float, link2: float, base_xy: np.ndarray) -> np.ndarray:
    tips: list[np.ndarray] = []
    for row in qpos:
        _, tip_local = forward_kinematics(row[:2], link1=float(link1), link2=float(link2))
        tips.append(base_xy + tip_local)
    return np.stack(tips, axis=0)


def max_tip_distance_from_qpos_box(
    obstacle_summary: dict[str, Any],
    *,
    link1: float,
    link2: float,
) -> float:
    if "box_lower" not in obstacle_summary or "box_upper" not in obstacle_summary:
        raise ValueError("Obstacle summary must contain box_lower and box_upper for max-tip-distance radius.")
    lower = np.asarray(obstacle_summary["box_lower"], dtype=np.float64)
    upper = np.asarray(obstacle_summary["box_upper"], dtype=np.float64)
    if lower.shape[0] < 2 or upper.shape[0] < 2:
        raise ValueError(f"Expected 2D qpos box bounds, got lower={lower}, upper={upper}.")

    q1_values = np.linspace(float(lower[0]), float(upper[0]), 256, dtype=np.float64)
    q2_values = np.linspace(float(lower[1]), float(upper[1]), 256, dtype=np.float64)
    q1_mesh, q2_mesh = np.meshgrid(q1_values, q2_values, indexing="xy")
    qpos = np.stack((q1_mesh.reshape(-1), q2_mesh.reshape(-1)), axis=1)
    tip_xy = tip_xy_from_qpos(qpos, link1=link1, link2=link2, base_xy=np.zeros(2, dtype=np.float64))
    return float(np.max(np.linalg.norm(tip_xy, axis=1)))


def resolve_obstacle_circle(
    *,
    args: argparse.Namespace,
    obstacle_summary: dict[str, Any],
    link1: float,
    link2: float,
    base_xy: np.ndarray,
) -> tuple[np.ndarray, float, str, str]:
    if args.obstacle_center_source == "manual":
        center_xy = np.array([float(args.obstacle_center_x), float(args.obstacle_center_y)], dtype=np.float64)
        center_source = "manual"
    elif args.obstacle_center_source == "summary":
        if "circle_center_xy" in obstacle_summary:
            center_xy = np.asarray(obstacle_summary["circle_center_xy"], dtype=np.float64)
        else:
            center_xy = base_xy.copy()
        center_source = "summary"
    else:
        center_xy = base_xy.copy()
        center_source = "base"

    if args.obstacle_radius_source == "manual":
        radius = float(args.obstacle_radius)
        radius_source = "manual"
    elif args.obstacle_radius_source == "summary":
        if "circle_radius" not in obstacle_summary:
            raise ValueError("Obstacle summary does not contain circle_radius.")
        radius = float(obstacle_summary["circle_radius"])
        radius_source = "summary"
    else:
        radius = max_tip_distance_from_qpos_box(obstacle_summary, link1=link1, link2=link2)
        radius_source = "max-tip-distance"
    return center_xy, radius, center_source, radius_source


def evenly_spaced_indices(frame_count: int, sample_count: int) -> np.ndarray:
    return np.unique(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=np.int64))


def motion_spaced_indices(points: np.ndarray, sample_count: int) -> np.ndarray:
    if points.shape[0] == 1:
        return np.array([0], dtype=np.int64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.array([0], dtype=np.int64)
    targets = np.linspace(0.0, total, min(sample_count, points.shape[0]))
    return np.unique(np.clip(np.searchsorted(cumulative, targets), 0, points.shape[0] - 1).astype(np.int64))


def select_sample_indices(tip_xy: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.sample_by == "index":
        return evenly_spaced_indices(tip_xy.shape[0], int(args.num_frames))
    return motion_spaced_indices(tip_xy, int(args.num_frames))


def ramp_alpha(index: int, frame_count: int, min_alpha: float, max_alpha: float) -> float:
    if frame_count <= 1:
        return float(max_alpha)
    t = float(index) / float(frame_count - 1)
    return float((1.0 - t) * min_alpha + t * max_alpha)


def alpha_composite_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    out = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - alpha) * out[mask] + alpha * layer.astype(np.float32)[mask]
    return out


def alpha_composite_image(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * base + alpha * layer.astype(np.float32)


def world_xy_to_pixel(
    xy: np.ndarray,
    *,
    width: int,
    height: int,
    camera_z: float,
    camera_fovy_deg: float,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    half_height = float(camera_z) * math.tan(math.radians(float(camera_fovy_deg)) / 2.0)
    half_width = half_height * float(width) / float(height)
    px = (xy[:, 0] / half_width + 1.0) * 0.5 * float(width)
    py = (1.0 - xy[:, 1] / half_height) * 0.5 * float(height)
    return np.stack((px, py), axis=1)


def draw_alpha_circle(
    image: np.ndarray,
    *,
    center_px: np.ndarray,
    radius_px: float,
    color: np.ndarray,
    alpha: float,
    thickness: int,
) -> np.ndarray:
    overlay = image.copy()
    center = tuple(int(round(value)) for value in center_px)
    radius = max(1, int(round(float(radius_px))))
    cv2.circle(overlay, center, radius, tuple(float(value) for value in color), thickness=int(thickness), lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


def draw_polyline(
    image: np.ndarray,
    points_px: np.ndarray,
    *,
    color: np.ndarray,
    alpha: float,
    width: int,
) -> np.ndarray:
    if points_px.shape[0] < 2:
        return image
    overlay = image.copy()
    points = np.rint(points_px).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [points], isClosed=False, color=tuple(float(value) for value in color), thickness=int(width), lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


def draw_tip_marker(
    image: np.ndarray,
    center_px: np.ndarray,
    *,
    color: np.ndarray,
    alpha: float,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return image
    overlay = image.copy()
    center = tuple(int(round(value)) for value in center_px)
    cv2.circle(overlay, center, int(radius), tuple(float(value) for value in color), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, center, int(radius), (20.0, 20.0, 20.0), thickness=1, lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


@contextmanager
def hidden_arm_geoms(model: Any, arm_geom_ids: np.ndarray) -> Iterator[None]:
    geom_rgba = np.asarray(model.geom_rgba).copy()
    try:
        model.geom_rgba[arm_geom_ids, 3] = 0.0
        yield
    finally:
        model.geom_rgba[:] = geom_rgba


def set_env_qpos(env: Any, qpos: np.ndarray) -> None:
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[:2] = np.asarray(qpos[:2], dtype=np.float64)
        physics.data.qvel[:2] = 0.0
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)


def render_background(env: Any, qpos: np.ndarray, *, arm_geom_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    set_env_qpos(env, qpos)
    with hidden_arm_geoms(env._env.physics.model, arm_geom_ids):
        frame = env._env.physics.render(height=int(args.height), width=int(args.width), camera_id=int(args.camera_id))
    if float(args.background_alpha) >= 1.0:
        return frame.astype(np.float32)
    white = np.full_like(frame, 255, dtype=np.float32)
    return alpha_composite_image(white, frame, float(args.background_alpha))


def render_reacher_layer(
    env: Any,
    qpos: np.ndarray,
    *,
    arm_geom_ids: np.ndarray,
    scene_option: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    set_env_qpos(env, qpos)
    frame = env._env.physics.render(height=int(args.height), width=int(args.width), camera_id=int(args.camera_id))
    segmentation = env._env.physics.render(
        height=int(args.height),
        width=int(args.width),
        camera_id=int(args.camera_id),
        segmentation=True,
        scene_option=scene_option,
    )
    mask = np.zeros(segmentation.shape[:2], dtype=bool)
    for geom_id in arm_geom_ids:
        mask |= segmentation[..., 0] == int(geom_id)
    return frame.astype(np.float32), mask


def render_timelapse(run_dir: Path, args: argparse.Namespace, obstacle_summary: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    qpos_raw, qpos_metadata = load_rollout_qpos(run_dir)
    qpos = ensure_qpos_shape(qpos_raw, run_dir)

    link1 = float(obstacle_summary.get("link1", 0.12))
    link2 = float(obstacle_summary.get("link2", 0.12))
    base_xy = np.zeros(2, dtype=np.float64)
    tip_xy = tip_xy_from_qpos(qpos, link1=link1, link2=link2, base_xy=base_xy)
    obstacle_center_xy, obstacle_radius, obstacle_center_source, obstacle_radius_source = resolve_obstacle_circle(
        args=args,
        obstacle_summary=obstacle_summary,
        link1=link1,
        link2=link2,
        base_xy=base_xy,
    )
    sampled_indices = select_sample_indices(tip_xy, args)

    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        hide_target(env)
        configure_offscreen_framebuffer(env, int(args.width), int(args.height))
        physics = env._env.physics
        model = physics.model
        arm_geom_ids = get_arm_geom_ids(model)
        scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)

        try:
            camera_z = float(model.cam_pos[int(args.camera_id)][2])
            camera_fovy = float(model.cam_fovy[int(args.camera_id)])
            tip_px = world_xy_to_pixel(
                tip_xy,
                width=int(args.width),
                height=int(args.height),
                camera_z=camera_z,
                camera_fovy_deg=camera_fovy,
            )
            obstacle_center_px = world_xy_to_pixel(
                obstacle_center_xy[None, :],
                width=int(args.width),
                height=int(args.height),
                camera_z=camera_z,
                camera_fovy_deg=camera_fovy,
            )[0]
            obstacle_edge_px = world_xy_to_pixel(
                (obstacle_center_xy + np.array([obstacle_radius, 0.0], dtype=np.float64))[None, :],
                width=int(args.width),
                height=int(args.height),
                camera_z=camera_z,
                camera_fovy_deg=camera_fovy,
            )[0]
            obstacle_radius_px = float(np.linalg.norm(obstacle_edge_px - obstacle_center_px))

            composite = render_background(env, qpos[0], arm_geom_ids=arm_geom_ids, args=args)

            sampled_tip_px = tip_px[sampled_indices]
            for local_index, frame_index in enumerate(tqdm(sampled_indices, desc=f"Rendering {run_dir.name} timelapse")):
                alpha = ramp_alpha(
                    local_index,
                    int(sampled_indices.shape[0]),
                    float(args.reacher_min_alpha),
                    float(args.reacher_max_alpha),
                )
                reacher_layer, mask = render_reacher_layer(
                    env,
                    qpos[frame_index],
                    arm_geom_ids=arm_geom_ids,
                    scene_option=scene_option,
                    args=args,
                )
                composite = alpha_composite_mask(composite, reacher_layer, mask, alpha)

            composite = draw_alpha_circle(
                composite,
                center_px=obstacle_center_px,
                radius_px=obstacle_radius_px,
                color=OBSTACLE_COLOR,
                alpha=float(args.obstacle_alpha),
                thickness=-1,
            )
            composite = draw_alpha_circle(
                composite,
                center_px=obstacle_center_px,
                radius_px=obstacle_radius_px,
                color=OBSTACLE_COLOR,
                alpha=float(args.obstacle_outline_alpha),
                thickness=int(args.obstacle_outline_width),
            )

            composite = draw_polyline(
                composite,
                sampled_tip_px,
                color=LINE_COLOR,
                alpha=float(args.line_alpha),
                width=int(args.line_width),
            )
            for point in sampled_tip_px:
                composite = draw_tip_marker(
                    composite,
                    point,
                    color=TIP_COLOR,
                    alpha=float(args.tip_marker_alpha),
                    radius=int(args.tip_marker_radius),
                )
        finally:
            model.geom_group[target_geom_id] = original_group
    finally:
        env.close()

    output_path = run_dir / str(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, np.clip(np.rint(composite), 0, 255).astype(np.uint8))

    metadata = {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "qpos": qpos_metadata,
        "source_frame_count": int(qpos.shape[0]),
        "sampled_indices": [int(index) for index in sampled_indices],
        "sample_by": str(args.sample_by),
        "num_requested_frames": int(args.num_frames),
        "num_rendered_frames": int(sampled_indices.shape[0]),
        "alphas": {
            "background": float(args.background_alpha),
            "obstacle": float(args.obstacle_alpha),
            "obstacle_outline": float(args.obstacle_outline_alpha),
            "reacher_min": float(args.reacher_min_alpha),
            "reacher_max": float(args.reacher_max_alpha),
            "line": float(args.line_alpha),
            "tip_marker": float(args.tip_marker_alpha),
        },
        "obstacle": {
            "summary_path": str(args.obstacle_summary),
            "center_source": obstacle_center_source,
            "radius_source": obstacle_radius_source,
            "circle_center_xy": [float(value) for value in obstacle_center_xy],
            "circle_radius": float(obstacle_radius),
        },
        "tip_xy": tip_xy.astype(np.float64).tolist(),
    }
    save_json(run_dir / "timelapse_metadata.json", metadata)
    return output_path, metadata


def write_combined_image(output_paths: list[Path], combined_path: Path) -> None:
    images = [imageio.imread(path) for path in output_paths]
    if not images:
        return
    min_height = min(int(image.shape[0]) for image in images)
    resized = []
    for image in images:
        if int(image.shape[0]) == min_height:
            resized.append(image)
            continue
        scale = min_height / float(image.shape[0])
        width = max(1, int(round(float(image.shape[1]) * scale)))
        resized.append(cv2.resize(image, (width, min_height), interpolation=cv2.INTER_AREA))
    spacer = np.full((min_height, 10, 3), 255, dtype=np.uint8)
    combined_parts: list[np.ndarray] = []
    for index, image in enumerate(resized):
        if index > 0:
            combined_parts.append(spacer)
        combined_parts.append(image)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(combined_path, np.concatenate(combined_parts, axis=1))


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = args.root.expanduser().resolve()
    obstacle_summary = load_json(args.obstacle_summary.expanduser().resolve())
    output_paths: list[Path] = []
    metadata: list[dict[str, Any]] = []
    for name in args.trajectories:
        output_path, run_metadata = render_timelapse(root / name, args, obstacle_summary)
        output_paths.append(output_path)
        metadata.append(run_metadata)
        print(f"Wrote timelapse for {name}: {output_path}")

    if not bool(args.no_combined) and len(output_paths) > 1:
        combined_path = root / str(args.combined_output_name)
        write_combined_image(output_paths, combined_path)
        save_json(
            root / "reacher_timelapse_metadata.json",
            {
                "combined_output_path": str(combined_path),
                "inputs": [str(path) for path in output_paths],
                "runs": metadata,
            },
        )
        print(f"Wrote combined timelapse: {combined_path}")


if __name__ == "__main__":
    main()
