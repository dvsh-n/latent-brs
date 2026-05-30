#!/usr/bin/env python3
"""Render fixed-alpha MuJoCo composites for rope obstacle and safe states."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.shared.lab_env import BaseEnvConfig, LabEnv, TaskState


DEFAULT_DATA_DIR = REPO_ROOT / "rope" / "plan" / "obstacle_data"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "obstacle_membership"
DEFAULT_DATASET_NAME = "obstacle_classifier_data.pt"
DEFAULT_SUMMARY_NAME = "summary.json"
DEFAULT_CAMERA_NAME = "video_cam"
ROPE_TENDON_NAME = "rope_tendon"

# Label controls.
LABEL_FONT_SIZE = 20.0
LABEL_ICON_SIZE = 8.0
LABEL_ICON_LINEWIDTH = 2.8
LABEL_ICON_TEXT_GAP = 3.0
LABEL_ANCHOR_Y = 0.035
LABEL_ANCHOR_X = 1.0 - LABEL_ANCHOR_Y
LABEL_TEXT_Y_OFFSET = 5.0
LABEL_BOX_PAD = 0.24
LABEL_BOX_ROUNDING = 0.18
LABEL_BOX_LINEWIDTH = 1.15
LABEL_BOX_ALPHA = 0.92
LABEL_SAFE_COLOR = "#138a36"
LABEL_UNSAFE_COLOR = "#c1121f"

GEOM_OBJTYPE = int(mujoco.mjtObj.mjOBJ_GEOM)
SITE_OBJTYPE = int(mujoco.mjtObj.mjOBJ_SITE)
TENDON_OBJTYPE = int(mujoco.mjtObj.mjOBJ_TENDON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--safe-clearance", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="MuJoCo camera to render from. The default video_cam is the front dataset camera.",
    )
    parser.add_argument("--obstacle-y-radius", type=float, default=0.68)
    parser.add_argument("--obstacle-rgba", nargs=4, type=float, default=(1.0, 0.58, 0.22, 1.0))
    parser.add_argument("--show-boundary-line", action="store_true", default=True)
    parser.add_argument("--no-boundary-line", action="store_false", dest="show_boundary_line")
    parser.add_argument("--boundary-line-width", type=float, default=1.2)
    parser.add_argument("--boundary-line-color", default="white")
    args = parser.parse_args()
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if not 0.0 <= float(args.alpha) <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if float(args.safe_clearance) < 0.0:
        raise ValueError("--safe-clearance must be non-negative.")
    if float(args.obstacle_y_radius) <= 0.0:
        raise ValueError("--obstacle-y-radius must be positive.")
    return args


def install_numpy_core_pickle_aliases() -> None:
    try:
        import numpy.core

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    except Exception:
        return


def torch_load_local(path: Path) -> dict[str, Any]:
    install_numpy_core_pickle_aliases()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_inputs(data_dir: Path, dataset_name: str, summary_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = data_dir / dataset_name
    summary_path = data_dir / summary_name
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing obstacle dataset: {dataset_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing obstacle summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return torch_load_local(dataset_path), summary


def format_xml_vec(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def half_ellipse_cylinder_mesh_xml(
    *,
    name: str,
    x_radius: float,
    half_length_y: float,
    z_height: float,
    arc_segments: int = 48,
) -> str:
    cross_section = [
        (x_radius * float(np.cos(np.pi * index / arc_segments)), z_height * float(np.sin(np.pi * index / arc_segments)))
        for index in range(arc_segments + 1)
    ]

    vertices: list[tuple[float, float, float]] = []
    front: list[int] = []
    back: list[int] = []
    for y, indices in ((-half_length_y, front), (half_length_y, back)):
        for x, z in cross_section:
            indices.append(len(vertices))
            vertices.append((x, y, z))

    front_bottom_center = len(vertices)
    vertices.append((0.0, -half_length_y, 0.0))
    back_bottom_center = len(vertices)
    vertices.append((0.0, half_length_y, 0.0))

    faces: list[tuple[int, int, int]] = []
    for index in range(arc_segments):
        faces.append((front[index], back[index], back[index + 1]))
        faces.append((front[index], back[index + 1], front[index + 1]))
        faces.append((front_bottom_center, front[index + 1], front[index]))
        faces.append((back_bottom_center, back[index], back[index + 1]))
    faces.append((front_bottom_center, back[0], front[0]))
    faces.append((front_bottom_center, back_bottom_center, back[0]))
    faces.append((front_bottom_center, front[-1], back[-1]))
    faces.append((front_bottom_center, back[-1], back_bottom_center))
    faces.extend((c, b, a) for a, b, c in list(faces))

    vertex_text = " ".join(format_xml_vec(vertex) for vertex in vertices)
    face_text = " ".join(" ".join(str(index) for index in face) for face in faces)
    return f'    <mesh name="{name}" vertex="{vertex_text}" face="{face_text}"/>'


def obstacle_speedbump_xml(summary: dict[str, Any], args: argparse.Namespace) -> tuple[str, str, dict[str, object]]:
    obstacle_reach = summary["obstacle_reach"]
    obstacle_base_height = float(summary["obstacle_base_height"])
    obstacle_peak_height = float(summary["obstacle_height"])
    x_center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    x_radius = 0.5 * (float(obstacle_reach[1]) - float(obstacle_reach[0]))
    z_height = obstacle_peak_height - obstacle_base_height
    if x_radius <= 0.0 or z_height <= 0.0:
        raise ValueError(f"Invalid obstacle dimensions: x_radius={x_radius}, z_height={z_height}.")

    mesh_name = "task_obstacle_speedbump_mesh"
    position = (x_center, 0.0, obstacle_base_height)
    rgba = tuple(float(value) for value in args.obstacle_rgba)
    asset_xml = half_ellipse_cylinder_mesh_xml(
        name=mesh_name,
        x_radius=x_radius,
        half_length_y=float(args.obstacle_y_radius),
        z_height=z_height,
    )
    worldbody_xml = (
        f'    <geom name="task_obstacle_speedbump" type="mesh" mesh="{mesh_name}" '
        f'pos="{format_xml_vec(position)}" rgba="{format_xml_vec(rgba)}" contype="0" conaffinity="0"/>'
    )
    return asset_xml, worldbody_xml, {
        "position": [float(value) for value in position],
        "obstacle_reach": [float(value) for value in obstacle_reach],
        "obstacle_base_height": obstacle_base_height,
        "obstacle_peak_height": obstacle_peak_height,
        "rgba": [float(value) for value in rgba],
    }


def half_ellipse_obstacle_height(
    reach: np.ndarray,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
) -> np.ndarray:
    reach_values = np.asarray(reach, dtype=np.float64)
    center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    half_width = 0.5 * (float(obstacle_reach[1]) - float(obstacle_reach[0]))
    normalized = (reach_values - center) / half_width
    profile = np.sqrt(np.clip(1.0 - normalized**2, 0.0, None))
    return float(obstacle_base_height) + (float(obstacle_peak_height) - float(obstacle_base_height)) * profile


def model_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, objtype, index)
    return "" if name is None else name


def arm_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        geom_name = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = model_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if body_name.startswith(("arm1_", "arm2_")) or geom_name.startswith(("arm1_", "arm2_")):
            ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def arm_site_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for site_id in range(model.nsite):
        site_name = model_name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        body_name = model_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site_id]))
        if body_name.startswith(("arm1_", "arm2_")) or site_name.startswith(("arm1_", "arm2_")):
            ids.append(site_id)
    return np.asarray(ids, dtype=np.int32)


@contextmanager
def hidden_dynamic_objects(env: LabEnv, arm_geoms: np.ndarray, arm_sites: np.ndarray, rope_tendon_id: int) -> Iterator[None]:
    geom_rgba = env.model.geom_rgba.copy()
    site_rgba = env.model.site_rgba.copy()
    tendon_rgba = env.model.tendon_rgba.copy()
    try:
        env.model.geom_rgba[arm_geoms, 3] = 0.0
        env.model.site_rgba[arm_sites, 3] = 0.0
        env.model.tendon_rgba[rope_tendon_id, 3] = 0.0
        yield
    finally:
        env.model.geom_rgba[:] = geom_rgba
        env.model.site_rgba[:] = site_rgba
        env.model.tendon_rgba[:] = tendon_rgba


def render_frame(renderer: mujoco.Renderer, env: LabEnv, camera_id: int) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def render_segmentation(renderer: mujoco.Renderer, env: LabEnv, camera_id: int) -> np.ndarray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera_id)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
    renderer.disable_segmentation_rendering()
    return segmentation


def mask_from_object_ids(segmentation: np.ndarray, object_ids: np.ndarray, object_type: int) -> np.ndarray:
    if object_ids.size == 0:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    return (segmentation[..., 1] == object_type) & np.isin(segmentation[..., 0], object_ids)


def alpha_composite(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    output = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output[mask] = (1.0 - alpha) * output[mask] + alpha * layer[mask]
    return output


def project_world_points_to_pixels(
    env: LabEnv,
    camera_id: int,
    points: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    camera_position = env.data.cam_xpos[camera_id].copy()
    camera_rotation = env.data.cam_xmat[camera_id].reshape(3, 3).copy()
    camera_points = (np.asarray(points, dtype=np.float64) - camera_position) @ camera_rotation
    depth = -camera_points[:, 2]
    fovy = np.deg2rad(float(env.model.cam_fovy[camera_id]))
    focal = 0.5 * float(height) / np.tan(0.5 * fovy)
    pixels = np.full((camera_points.shape[0], 2), np.nan, dtype=np.float64)
    visible = depth > 1e-6
    pixels[visible, 0] = 0.5 * float(width) + focal * camera_points[visible, 0] / depth[visible]
    pixels[visible, 1] = 0.5 * float(height) - focal * camera_points[visible, 1] / depth[visible]
    return pixels


def front_boundary_pixels(
    env: LabEnv,
    camera_id: int,
    *,
    obstacle_reach: tuple[float, float],
    obstacle_peak_height: float,
    obstacle_y_radius: float,
    width: int,
    height: int,
) -> np.ndarray:
    x_center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    y_values = np.linspace(-float(obstacle_y_radius), float(obstacle_y_radius), num=256)
    points = np.column_stack(
        [
            np.full_like(y_values, x_center),
            y_values,
            np.full_like(y_values, float(obstacle_peak_height)),
        ]
    )
    return project_world_points_to_pixels(env, camera_id, points, width=width, height=height)


def make_label_icon(kind: str, color: str) -> DrawingArea:
    size = float(LABEL_ICON_SIZE)
    area = DrawingArea(size, size, 0.0, 0.0)
    if kind == "safe":
        segments = [
            ((0.18 * size, 0.48 * size), (0.42 * size, 0.25 * size)),
            ((0.42 * size, 0.25 * size), (0.82 * size, 0.76 * size)),
        ]
    elif kind == "unsafe":
        segments = [
            ((0.22 * size, 0.22 * size), (0.78 * size, 0.78 * size)),
            ((0.78 * size, 0.22 * size), (0.22 * size, 0.78 * size)),
        ]
    else:
        raise ValueError(f"Unknown label icon kind {kind!r}.")
    for start, end in segments:
        area.add_artist(
            Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=float(LABEL_ICON_LINEWIDTH),
                solid_capstyle="round",
            )
        )
    return area


def add_sample_label(ax: plt.Axes, *, text: str, color: str, icon_kind: str) -> None:
    icon = make_label_icon(icon_kind, color)
    label = TextArea(
        text,
        textprops={
            "color": color,
            "fontsize": float(LABEL_FONT_SIZE),
            "fontweight": "bold",
            "fontfamily": "DejaVu Sans",
            "va": "center",
        },
    )
    label._text.set_y(float(LABEL_TEXT_Y_OFFSET))
    packed = HPacker(
        children=[icon, label],
        align="center",
        pad=0.0,
        sep=float(LABEL_ICON_TEXT_GAP),
    )
    annotation = AnnotationBbox(
        packed,
        (float(LABEL_ANCHOR_X), float(LABEL_ANCHOR_Y)),
        xycoords=ax.transAxes,
        box_alignment=(1.0, 0.0),
        frameon=True,
        bboxprops={
            "boxstyle": f"round,pad={float(LABEL_BOX_PAD)},rounding_size={float(LABEL_BOX_ROUNDING)}",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": float(LABEL_BOX_LINEWIDTH),
            "alpha": float(LABEL_BOX_ALPHA),
        },
    )
    ax.add_artist(annotation)


def set_env_to_dataset_state(env: LabEnv, task_target: np.ndarray, arm_qpos: np.ndarray | None) -> None:
    env.reset(TaskState.from_array(task_target))
    if arm_qpos is not None:
        env.set_arm_joint_positions(np.asarray(arm_qpos, dtype=np.float64))
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def select_evenly_by_reach(
    candidates: np.ndarray,
    task_target: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    shuffled = np.asarray(candidates, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    sorted_candidates = shuffled[np.argsort(task_target[shuffled, 0])]
    if sorted_candidates.size <= count:
        return sorted_candidates
    positions = np.linspace(0, sorted_candidates.size - 1, count, dtype=np.int64)
    return sorted_candidates[positions]


def select_plot_indices(
    task_target: np.ndarray,
    labels: np.ndarray,
    low_rope_height: np.ndarray,
    *,
    obstacle_reach: tuple[float, float],
    obstacle_base_height: float,
    obstacle_peak_height: float,
    safe_clearance: float,
    num_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = half_ellipse_obstacle_height(task_target[:, 0], obstacle_reach, obstacle_base_height, obstacle_peak_height)
    margin = low_rope_height - boundary
    in_reach = (task_target[:, 0] >= obstacle_reach[0]) & (task_target[:, 0] <= obstacle_reach[1])

    obstacle_candidates = np.flatnonzero((labels == 1) & in_reach & (margin <= 0.0))
    safe_candidates = np.flatnonzero((labels == 0) & in_reach & (margin >= float(safe_clearance)))
    if obstacle_candidates.size < num_frames:
        raise ValueError(f"Only {obstacle_candidates.size} obstacle candidates available for {num_frames} frames.")
    if safe_candidates.size < num_frames:
        raise ValueError(
            f"Only {safe_candidates.size} safe-above-obstacle candidates available for {num_frames} frames. "
            "Lower --safe-clearance or --num-frames."
        )
    return (
        select_evenly_by_reach(obstacle_candidates, task_target, num_frames, rng),
        select_evenly_by_reach(safe_candidates, task_target, num_frames, rng),
        margin,
    )


def render_composite(
    env: LabEnv,
    camera_id: int,
    arm_geoms: np.ndarray,
    arm_sites: np.ndarray,
    rope_tendon_id: int,
    *,
    task_target: np.ndarray,
    arm_qpos: np.ndarray | None,
    indices: np.ndarray,
    alpha: float,
    width: int,
    height: int,
    desc: str,
) -> np.ndarray:
    set_env_to_dataset_state(env, task_target[indices[0]], None if arm_qpos is None else arm_qpos[indices[0]])
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(width))
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(height))
    with mujoco.Renderer(env.model, height=height, width=width) as renderer:
        with hidden_dynamic_objects(env, arm_geoms, arm_sites, rope_tendon_id):
            composite = render_frame(renderer, env, camera_id).astype(np.float32)

        for index in tqdm(indices, desc=desc, unit="frame"):
            set_env_to_dataset_state(env, task_target[index], None if arm_qpos is None else arm_qpos[index])
            rgb = render_frame(renderer, env, camera_id).astype(np.float32)
            segmentation = render_segmentation(renderer, env, camera_id)
            arm_mask = mask_from_object_ids(segmentation, arm_geoms, GEOM_OBJTYPE) | mask_from_object_ids(
                segmentation, arm_sites, SITE_OBJTYPE
            )
            rope_mask = (segmentation[..., 1] == TENDON_OBJTYPE) & (segmentation[..., 0] == rope_tendon_id)
            composite = alpha_composite(composite, rgb, arm_mask | rope_mask, alpha)
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


def save_image(
    path: Path,
    image: np.ndarray,
    dpi: int,
    *,
    boundary_pixels: np.ndarray | None = None,
    line_width: float = 2.2,
    line_color: str = "white",
    label_text: str | None = None,
    label_color: str = "white",
    label_icon: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    if boundary_pixels is not None:
        finite = np.isfinite(boundary_pixels).all(axis=1)
        if np.any(finite):
            (line,) = ax.plot(
                boundary_pixels[finite, 0],
                boundary_pixels[finite, 1],
                linestyle=(0.0, (0.45, 1.25)),
                linewidth=float(line_width),
                color=str(line_color),
                solid_capstyle="round",
                dash_capstyle="round",
            )
            line.set_path_effects(
                [
                    patheffects.Stroke(linewidth=float(line_width) + 1.3, foreground="black"),
                    patheffects.Normal(),
                ]
            )
    if label_text is not None and label_icon is not None:
        add_sample_label(ax, text=label_text, color=label_color, icon_kind=label_icon)
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, summary = load_inputs(data_dir, args.dataset_name, args.summary_name)
    dataset = data["dataset"]
    task_target = np.asarray(dataset["task_target"], dtype=np.float64)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    low_rope_height = np.asarray(dataset["low_rope_height"], dtype=np.float64)
    arm_qpos = np.asarray(dataset["qpos"], dtype=np.float64) if "qpos" in dataset else None

    obstacle_reach = tuple(float(value) for value in summary["obstacle_reach"])
    obstacle_base_height = float(summary["obstacle_base_height"])
    obstacle_peak_height = float(summary["obstacle_height"])
    rng = np.random.default_rng(int(args.seed))
    obstacle_indices, safe_indices, margin = select_plot_indices(
        task_target,
        labels,
        low_rope_height,
        obstacle_reach=obstacle_reach,
        obstacle_base_height=obstacle_base_height,
        obstacle_peak_height=obstacle_peak_height,
        safe_clearance=float(args.safe_clearance),
        num_frames=int(args.num_frames),
        rng=rng,
    )

    env = LabEnv(
        base_config=BaseEnvConfig(
            asset_extra_xml="",
            worldbody_extra_xml="",
        )
    )
    try:
        camera_id = env.model.camera(str(args.camera_name)).id
    except KeyError as exc:
        raise ValueError(f"Unknown camera {args.camera_name!r}.") from exc
    camera_metadata = {"name": str(args.camera_name)}
    arm_geoms = arm_geom_ids(env.model)
    arm_sites = arm_site_ids(env.model)
    rope_tendon_id = env.model.tendon(ROPE_TENDON_NAME).id
    boundary_pixels = None
    if bool(args.show_boundary_line):
        mujoco.mj_forward(env.model, env.data)
        boundary_pixels = front_boundary_pixels(
            env,
            camera_id,
            obstacle_reach=obstacle_reach,
            obstacle_peak_height=obstacle_peak_height,
            obstacle_y_radius=float(args.obstacle_y_radius),
            width=int(args.width),
            height=int(args.height),
        )

    obstacle_image = render_composite(
        env,
        camera_id,
        arm_geoms,
        arm_sites,
        rope_tendon_id,
        task_target=task_target,
        arm_qpos=arm_qpos,
        indices=obstacle_indices,
        alpha=float(args.alpha),
        width=int(args.width),
        height=int(args.height),
        desc="Rendering obstacle composite",
    )
    safe_image = render_composite(
        env,
        camera_id,
        arm_geoms,
        arm_sites,
        rope_tendon_id,
        task_target=task_target,
        arm_qpos=arm_qpos,
        indices=safe_indices,
        alpha=float(args.alpha),
        width=int(args.width),
        height=int(args.height),
        desc="Rendering safe composite",
    )

    for fmt in args.formats:
        save_image(
            out_dir / f"obstacle_alpha_overlap.{fmt}",
            obstacle_image,
            int(args.dpi),
            boundary_pixels=boundary_pixels,
            line_width=float(args.boundary_line_width),
            line_color=str(args.boundary_line_color),
            label_text="UNSAFE",
            label_color=LABEL_UNSAFE_COLOR,
            label_icon="unsafe",
        )
        save_image(
            out_dir / f"safe_alpha_overlap.{fmt}",
            safe_image,
            int(args.dpi),
            boundary_pixels=boundary_pixels,
            line_width=float(args.boundary_line_width),
            line_color=str(args.boundary_line_color),
            label_text="SAFE",
            label_color=LABEL_SAFE_COLOR,
            label_icon="safe",
        )

    metadata = {
        "data_dir": str(data_dir),
        "num_frames": int(args.num_frames),
        "alpha": float(args.alpha),
        "safe_clearance": float(args.safe_clearance),
        "obstacle_indices": [int(index) for index in obstacle_indices],
        "safe_indices": [int(index) for index in safe_indices],
        "obstacle_margins": [float(margin[index]) for index in obstacle_indices],
        "safe_margins": [float(margin[index]) for index in safe_indices],
        "obstacle_states": [[float(value) for value in task_target[index]] for index in obstacle_indices],
        "safe_states": [[float(value) for value in task_target[index]] for index in safe_indices],
        "obstacle_rendered": False,
        "obstacle_reach": [float(value) for value in obstacle_reach],
        "obstacle_base_height": obstacle_base_height,
        "obstacle_peak_height": obstacle_peak_height,
        "boundary_line": {
            "shown": bool(args.show_boundary_line),
            "style": "dotted",
            "height": obstacle_peak_height,
            "world_x": 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1])),
            "world_y_radius": float(args.obstacle_y_radius),
            "color": str(args.boundary_line_color),
            "line_width": float(args.boundary_line_width),
        },
        "camera": camera_metadata,
        "outputs": [f"{prefix}_alpha_overlap.{fmt}" for prefix in ("obstacle", "safe") for fmt in args.formats],
    }
    with (out_dir / "alpha_overlap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote fixed-alpha obstacle composites to {out_dir}")


if __name__ == "__main__":
    main()
