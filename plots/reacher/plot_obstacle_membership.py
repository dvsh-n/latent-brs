#!/usr/bin/env python3
"""Render fixed-alpha MuJoCo composites for Reacher joint-box safe/unsafe states."""

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
from dm_control import suite
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
import numpy as np
import torch
from tqdm.auto import tqdm


DEFAULT_DATA_DIR = REPO_ROOT / "reacher" / "plan" / "obstacle_data_joint_box"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "reacher"
DEFAULT_DATASET_NAME = "obstacle_classifier_data.pt"
DEFAULT_SUMMARY_NAME = "summary.json"

LABEL_FONT_SIZE = 15.0
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
OBSTACLE_COLOR = np.array([230.0, 83.0, 35.0], dtype=np.float32)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--background-alpha", type=float, default=1.0)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--show-obstacle-circle", action="store_true", default=True)
    parser.add_argument("--no-obstacle-circle", action="store_false", dest="show_obstacle_circle")
    parser.add_argument("--obstacle-alpha", type=float, default=0.18)
    parser.add_argument("--obstacle-outline-alpha", type=float, default=0.78)
    parser.add_argument("--obstacle-outline-width", type=int, default=3)
    args = parser.parse_args()
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    for name in ("alpha", "background_alpha", "obstacle_alpha", "obstacle_outline_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if int(args.obstacle_outline_width) <= 0:
        raise ValueError("--obstacle-outline-width must be positive.")
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs(data_dir: Path, dataset_name: str, summary_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = data_dir / dataset_name
    summary_path = data_dir / summary_name
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing obstacle dataset: {dataset_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing obstacle summary: {summary_path}")
    return torch_load_local(dataset_path), load_json(summary_path)


def make_render_env(*, seed: int, time_limit: float, width: int, height: int, physics_freq_hz: float) -> Any:
    env = suite.load(
        domain_name="reacher",
        task_name="hard",
        task_kwargs={"random": int(seed)},
    )
    env.reset()
    physics = env.physics
    physics.model.opt.timestep = 1.0 / float(physics_freq_hz)
    physics.model.vis.global_.offwidth = max(int(physics.model.vis.global_.offwidth), int(width))
    physics.model.vis.global_.offheight = max(int(physics.model.vis.global_.offheight), int(height))
    return env


def hide_target(env: Any) -> None:
    target_geom_id = env.physics.model.name2id("target", "geom")
    env.physics.model.geom_rgba[target_geom_id] = [0.0, 0.0, 0.0, 0.0]


def get_arm_geom_ids(model: Any) -> np.ndarray:
    arm_body_names = ("arm", "hand", "finger")
    arm_body_ids = {int(model.name2id(name, "body")) for name in arm_body_names}
    arm_geom_ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        geom_name = model.id2name(geom_id, "geom")
        geom_body_id = int(model.geom_bodyid[geom_id])
        if geom_name == "root" or geom_body_id in arm_body_ids:
            arm_geom_ids.append(geom_id)
    if not arm_geom_ids:
        raise ValueError("Failed to identify arm geoms for segmentation.")
    return np.asarray(sorted(set(arm_geom_ids)), dtype=np.int32)


def make_segmentation_scene_option(model: Any) -> tuple[Any, int, int]:
    from dm_control.mujoco.wrapper import core as dm_core

    target_geom_id = int(model.name2id("target", "geom"))
    original_group = int(model.geom_group[target_geom_id])
    model.geom_group[target_geom_id] = 3
    scene_option = dm_core.MjvOption()
    scene_option.geomgroup[:] = 1
    scene_option.geomgroup[3] = 0
    return scene_option, target_geom_id, original_group


@contextmanager
def hidden_arm_geoms(model: Any, arm_geom_ids: np.ndarray) -> Iterator[None]:
    geom_rgba = np.asarray(model.geom_rgba).copy()
    try:
        model.geom_rgba[arm_geom_ids, 3] = 0.0
        yield
    finally:
        model.geom_rgba[:] = geom_rgba


def set_env_qpos(env: Any, qpos: np.ndarray) -> None:
    physics = env.physics
    with physics.reset_context():
        physics.data.qpos[:2] = np.asarray(qpos[:2], dtype=np.float64)
        physics.data.qvel[:2] = 0.0


def alpha_composite_image(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * base + alpha * layer.astype(np.float32)


def alpha_composite_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    out = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - alpha) * out[mask] + alpha * layer.astype(np.float32)[mask]
    return out


def render_background(env: Any, qpos: np.ndarray, *, arm_geom_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    set_env_qpos(env, qpos)
    with hidden_arm_geoms(env.physics.model, arm_geom_ids):
        frame = env.physics.render(height=int(args.height), width=int(args.width), camera_id=int(args.camera_id))
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
    frame = env.physics.render(height=int(args.height), width=int(args.width), camera_id=int(args.camera_id))
    segmentation = env.physics.render(
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


def world_xy_to_pixel(
    xy: np.ndarray,
    *,
    width: int,
    height: int,
    camera_z: float,
    camera_fovy_deg: float,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    half_height = float(camera_z) * np.tan(np.deg2rad(float(camera_fovy_deg)) / 2.0)
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


def select_evenly_by_q1(candidates: np.ndarray, qpos: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if candidates.size == 0:
        return candidates
    shuffled = np.asarray(candidates, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    sorted_candidates = shuffled[np.argsort(qpos[shuffled, 0])]
    if sorted_candidates.size <= count:
        return sorted_candidates
    positions = np.linspace(0, sorted_candidates.size - 1, count, dtype=np.int64)
    return sorted_candidates[positions]


def select_plot_indices(qpos: np.ndarray, labels: np.ndarray, num_frames: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    unsafe_candidates = np.flatnonzero(labels == 1)
    safe_candidates = np.flatnonzero(labels == 0)
    if unsafe_candidates.size < num_frames:
        raise ValueError(f"Only {unsafe_candidates.size} unsafe candidates available for {num_frames} frames.")
    if safe_candidates.size < num_frames:
        raise ValueError(f"Only {safe_candidates.size} safe candidates available for {num_frames} frames.")
    return (
        select_evenly_by_q1(unsafe_candidates, qpos, num_frames, rng),
        select_evenly_by_q1(safe_candidates, qpos, num_frames, rng),
    )


def obstacle_radius_from_qpos_box(summary: dict[str, Any]) -> float:
    lower = np.asarray(summary["box_lower"], dtype=np.float64)
    upper = np.asarray(summary["box_upper"], dtype=np.float64)
    link1 = float(summary.get("link1", 0.12))
    link2 = float(summary.get("link2", 0.12))
    q1_values = np.linspace(float(lower[0]), float(upper[0]), 256, dtype=np.float64)
    q2_values = np.linspace(float(lower[1]), float(upper[1]), 256, dtype=np.float64)
    q1_mesh, q2_mesh = np.meshgrid(q1_values, q2_values, indexing="xy")
    tip_x = link1 * np.cos(q1_mesh) + link2 * np.cos(q1_mesh + q2_mesh)
    tip_y = link1 * np.sin(q1_mesh) + link2 * np.sin(q1_mesh + q2_mesh)
    return float(np.max(np.sqrt(tip_x**2 + tip_y**2)))


def draw_obstacle_circle(composite: np.ndarray, env: Any, summary: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    physics = env.physics
    model = physics.model
    camera_z = float(model.cam_pos[int(args.camera_id)][2])
    camera_fovy = float(model.cam_fovy[int(args.camera_id)])
    center_xy = np.zeros(2, dtype=np.float64)
    radius = obstacle_radius_from_qpos_box(summary)
    center_px = world_xy_to_pixel(
        center_xy[None, :],
        width=int(args.width),
        height=int(args.height),
        camera_z=camera_z,
        camera_fovy_deg=camera_fovy,
    )[0]
    edge_px = world_xy_to_pixel(
        (center_xy + np.array([radius, 0.0], dtype=np.float64))[None, :],
        width=int(args.width),
        height=int(args.height),
        camera_z=camera_z,
        camera_fovy_deg=camera_fovy,
    )[0]
    radius_px = float(np.linalg.norm(edge_px - center_px))
    composite = draw_alpha_circle(
        composite,
        center_px=center_px,
        radius_px=radius_px,
        color=OBSTACLE_COLOR,
        alpha=float(args.obstacle_alpha),
        thickness=-1,
    )
    return draw_alpha_circle(
        composite,
        center_px=center_px,
        radius_px=radius_px,
        color=OBSTACLE_COLOR,
        alpha=float(args.obstacle_outline_alpha),
        thickness=int(args.obstacle_outline_width),
    )


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
    packed = HPacker(children=[icon, label], align="center", pad=0.0, sep=float(LABEL_ICON_TEXT_GAP))
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


def save_image(path: Path, image: np.ndarray, dpi: int, *, label_text: str, label_color: str, label_icon: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    add_sample_label(ax, text=label_text, color=label_color, icon_kind=label_icon)
    ax.set_axis_off()
    fig.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def render_composite(
    qpos: np.ndarray,
    indices: np.ndarray,
    summary: dict[str, Any],
    args: argparse.Namespace,
    *,
    desc: str,
) -> np.ndarray:
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        hide_target(env)
        physics = env.physics
        model = physics.model
        arm_geom_ids = get_arm_geom_ids(model)
        scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)
        try:
            composite = render_background(env, qpos[indices[0]], arm_geom_ids=arm_geom_ids, args=args)
            for index in tqdm(indices, desc=desc, unit="frame"):
                reacher_layer, mask = render_reacher_layer(
                    env,
                    qpos[index],
                    arm_geom_ids=arm_geom_ids,
                    scene_option=scene_option,
                    args=args,
                )
                composite = alpha_composite_mask(composite, reacher_layer, mask, float(args.alpha))
            if bool(args.show_obstacle_circle):
                composite = draw_obstacle_circle(composite, env, summary, args)
        finally:
            model.geom_group[target_geom_id] = original_group
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, summary = load_inputs(data_dir, str(args.dataset_name), str(args.summary_name))
    dataset = data["dataset"]
    qpos = np.asarray(dataset["qpos"], dtype=np.float64)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    if qpos.ndim != 2 or qpos.shape[1] < 2:
        raise ValueError(f"Expected qpos with shape (N, >=2), got {qpos.shape}.")
    if labels.shape[0] != qpos.shape[0]:
        raise ValueError(f"Expected one label per qpos, got qpos={qpos.shape} labels={labels.shape}.")

    rng = np.random.default_rng(int(args.seed))
    unsafe_indices, safe_indices = select_plot_indices(qpos, labels, int(args.num_frames), rng)
    unsafe_image = render_composite(qpos, unsafe_indices, summary, args, desc="Rendering unsafe composite")
    safe_image = render_composite(qpos, safe_indices, summary, args, desc="Rendering safe composite")

    outputs: list[str] = []
    for fmt in args.formats:
        unsafe_path = out_dir / f"unsafe_alpha_overlap.{fmt}"
        safe_path = out_dir / f"safe_alpha_overlap.{fmt}"
        save_image(
            unsafe_path,
            unsafe_image,
            int(args.dpi),
            label_text="UNSAFE",
            label_color=LABEL_UNSAFE_COLOR,
            label_icon="unsafe",
        )
        save_image(
            safe_path,
            safe_image,
            int(args.dpi),
            label_text="SAFE",
            label_color=LABEL_SAFE_COLOR,
            label_icon="safe",
        )
        outputs.extend([unsafe_path.name, safe_path.name])

    metadata = {
        "data_dir": str(data_dir),
        "num_frames": int(args.num_frames),
        "alpha": float(args.alpha),
        "unsafe_indices": [int(index) for index in unsafe_indices],
        "safe_indices": [int(index) for index in safe_indices],
        "unsafe_qpos": [[float(value) for value in qpos[index, :2]] for index in unsafe_indices],
        "safe_qpos": [[float(value) for value in qpos[index, :2]] for index in safe_indices],
        "label_rule": "unsafe label 1 iff qpos lies inside the configured joint-space box; safe label 0 otherwise",
        "box_lower": [float(value) for value in summary["box_lower"]],
        "box_upper": [float(value) for value in summary["box_upper"]],
        "obstacle_circle": {
            "shown": bool(args.show_obstacle_circle),
            "center_xy": [0.0, 0.0],
            "radius_source": "max fingertip distance over joint-space obstacle box",
            "radius": float(obstacle_radius_from_qpos_box(summary)),
        },
        "outputs": outputs,
    }
    with (out_dir / "alpha_overlap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote fixed-alpha Reacher obstacle composites to {out_dir}")


if __name__ == "__main__":
    main()
