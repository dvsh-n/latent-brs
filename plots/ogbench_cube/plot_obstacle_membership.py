#!/usr/bin/env python3
"""Render side-view fixed-alpha overlays for OGBench cube height-obstacle samples."""

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
OGBENCH_ROOT = REPO_ROOT / "third_party" / "ogbench"
for path in (REPO_ROOT, OGBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt
import mujoco
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
import numpy as np
import torch
from tqdm.auto import tqdm

from plots.ogbench_cube import render_side_timelapse as side_render


DEFAULT_DATA_DIR = REPO_ROOT / "ogbench_cube" / "plan" / "height_data"
DEFAULT_DATASET_NAME = "height_classifier_data.pt"
DEFAULT_SUMMARY_NAME = "summary.json"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "ogbench_cube"

LABEL_FONT_SIZE = 7.0
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
SAFE_LABEL = 0
UNSAFE_LABEL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", default=side_render.DEFAULT_ENV_NAME)
    parser.add_argument("--sim-freq-hz", type=float, default=side_render.DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=side_render.DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.425, 0.0, 0.14))
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=0.0)
    parser.add_argument("--fovy", type=float, default=45.0)
    parser.add_argument("--formats", nargs="+", default=["png"])
    args = parser.parse_args()
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive.")
    for name in ("alpha",):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive.")
    if float(args.fovy) <= 0.0:
        raise ValueError("--fovy must be positive.")
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


def load_height_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing height dataset: {path}")
    payload = torch_load_local(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("dataset"), dict):
        raise ValueError(f"Unexpected height dataset format in {path}.")
    dataset = payload["dataset"]
    required = ("label", "qpos", "block_pos", "gripper_height")
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError(f"Height dataset is missing required keys: {missing}")
    return payload


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_evenly_by_value(candidates: np.ndarray, values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if candidates.size == 0:
        raise ValueError("No candidate samples available for the requested class.")
    shuffled = np.asarray(candidates, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    sorted_candidates = shuffled[np.argsort(values[shuffled])]
    selected_count = min(int(count), int(sorted_candidates.size))
    if sorted_candidates.size <= selected_count:
        return sorted_candidates
    positions = np.linspace(0, sorted_candidates.size - 1, selected_count, dtype=np.int64)
    return sorted_candidates[positions]


def select_plot_indices(
    labels: np.ndarray,
    gripper_height: np.ndarray,
    num_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    unsafe_candidates = np.flatnonzero(labels == UNSAFE_LABEL)
    safe_candidates = np.flatnonzero(labels == SAFE_LABEL)
    if unsafe_candidates.size < num_frames:
        raise ValueError(f"Only {unsafe_candidates.size} unsafe samples available for {num_frames} frames.")
    if safe_candidates.size < num_frames:
        raise ValueError(f"Only {safe_candidates.size} safe samples available for {num_frames} frames.")
    return (
        select_evenly_by_value(unsafe_candidates, gripper_height, num_frames, rng),
        select_evenly_by_value(safe_candidates, gripper_height, num_frames, rng),
    )


def render_side_composite(qpos: np.ndarray, args: argparse.Namespace, *, desc: str) -> np.ndarray:
    env = side_render.make_render_env(args)
    try:
        camera = side_render.side_camera(args)
        dyn_geom_ids = side_render.dynamic_geom_ids(env.unwrapped._model)
        with mujoco.Renderer(env.unwrapped._model, height=int(args.height), width=int(args.width)) as renderer:
            renderer.scene.camera[0].frustum_center = 0.0
            renderer.scene.camera[0].frustum_width = 0.0
            side_render.set_env_state(env, qpos[0])
            with side_render.hidden_geoms(env.unwrapped._model, dyn_geom_ids):
                background = side_render.render_frame(renderer, env, camera).astype(np.float32)

            composite = background
            for qpos_row in tqdm(qpos, desc=desc, unit="frame"):
                side_render.set_env_state(env, qpos_row)
                rgb = side_render.render_frame(renderer, env, camera).astype(np.float32)
                segmentation = side_render.render_segmentation(renderer, env, camera)
                mask = side_render.mask_from_geom_ids(segmentation, dyn_geom_ids)
                composite = side_render.alpha_composite_mask(composite, rgb, mask, float(args.alpha))
    finally:
        env.close()
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


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


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    dataset_path = data_dir / str(args.dataset_name)
    summary_path = data_dir / str(args.summary_name)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_height_payload(dataset_path)
    summary = load_json_if_present(summary_path)
    dataset = payload["dataset"]
    qpos = np.asarray(dataset["qpos"], dtype=np.float64)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    block_pos = np.asarray(dataset["block_pos"], dtype=np.float64)
    gripper_height = np.asarray(dataset["gripper_height"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"Expected qpos with shape (N, nq), got {qpos.shape}.")
    if labels.shape[0] != qpos.shape[0] or gripper_height.shape[0] != qpos.shape[0]:
        raise ValueError(
            f"Expected one label and gripper height per qpos, got qpos={qpos.shape}, "
            f"labels={labels.shape}, gripper_height={gripper_height.shape}."
        )
    if block_pos.shape[0] != qpos.shape[0]:
        raise ValueError(f"Expected one block position per qpos, got qpos={qpos.shape}, block_pos={block_pos.shape}.")

    rng = np.random.default_rng(int(args.seed))
    unsafe_indices, safe_indices = select_plot_indices(labels, gripper_height, int(args.num_frames), rng)
    unsafe_qpos = qpos[unsafe_indices]
    safe_qpos = qpos[safe_indices]

    unsafe_image = render_side_composite(unsafe_qpos, args, desc="Rendering unsafe side overlay")
    safe_image = render_side_composite(
        safe_qpos,
        args,
        desc="Rendering safe side overlay",
    )

    outputs: list[str] = []
    for fmt in args.formats:
        unsafe_path = out_dir / f"ogbench_cube_unsafe_alpha_overlap.{fmt}"
        safe_path = out_dir / f"ogbench_cube_safe_alpha_overlap.{fmt}"
        save_image(
            unsafe_path,
            unsafe_image,
            int(args.dpi),
            label_text="UNSAFE SAMPLES",
            label_color=LABEL_UNSAFE_COLOR,
            label_icon="unsafe",
        )
        save_image(
            safe_path,
            safe_image,
            int(args.dpi),
            label_text="SAFE SAMPLES",
            label_color=LABEL_SAFE_COLOR,
            label_icon="safe",
        )
        outputs.extend([unsafe_path.name, safe_path.name])

    metadata = {
        "dataset_path": str(dataset_path),
        "summary_path": str(summary_path) if summary_path.is_file() else None,
        "num_frames": int(args.num_frames),
        "alpha": float(args.alpha),
        "label_semantics": {
            "unsafe": int(UNSAFE_LABEL),
            "safe": int(SAFE_LABEL),
            "source": "height_classifier_data label; 1=above threshold/unsafe, 0=below threshold/safe",
        },
        "unsafe_indices": [int(index) for index in unsafe_indices],
        "safe_indices": [int(index) for index in safe_indices],
        "unsafe_block_pos": block_pos[unsafe_indices].tolist(),
        "safe_block_pos": block_pos[safe_indices].tolist(),
        "unsafe_gripper_height": gripper_height[unsafe_indices].tolist(),
        "safe_gripper_height": gripper_height[safe_indices].tolist(),
        "height_threshold": summary.get("sampling", {}).get("height_threshold"),
        "rendering": "side_timelapse_style_alpha_overlays_without_traces",
        "camera": {
            "lookat": [float(value) for value in args.camera_lookat],
            "distance": float(args.camera_distance),
            "azimuth": float(args.camera_azimuth),
            "elevation": float(args.camera_elevation),
            "fovy": float(args.fovy),
        },
        "source_metadata": jsonable(payload.get("metadata", {})),
        "outputs": outputs,
    }
    with (out_dir / "ogbench_cube_alpha_overlap_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote OGBench cube height-obstacle overlays to {out_dir}")


if __name__ == "__main__":
    main()
