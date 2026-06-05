#!/usr/bin/env python3
"""Render a 2-row OGBench cube side-view timelapse PDF.

The top row is the safe rollout and the bottom row is the unsafe rollout. Each
panel is a raw MuJoCo side render with timestamp labels and grid styling
matching plots/rope/make_side_timelapse_safe_unsafe.py.
"""

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

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from plots.ogbench_cube import render_side_timelapse as side


DEFAULT_SAFE_RUN_DIR = side.DEFAULT_ROOT / "ogbench_cube_safe_0"
DEFAULT_UNSAFE_RUN_DIR = side.DEFAULT_ROOT / "ogbench_cube_unsafe_0"
DEFAULT_OUTPUT = side.DEFAULT_ROOT / "safe_unsafe_side_timelapse.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-run-dir", type=Path, default=DEFAULT_SAFE_RUN_DIR)
    parser.add_argument("--unsafe-run-dir", type=Path, default=DEFAULT_UNSAFE_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--sample-by", choices=("motion", "index"), default="motion")
    parser.add_argument("--trim-oracle", action="store_true", default=True)
    parser.add_argument("--no-trim-oracle", action="store_false", dest="trim_oracle")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--image-separator-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", default=side.DEFAULT_ENV_NAME)
    parser.add_argument("--sim-freq-hz", type=float, default=side.DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=side.DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--control-timestep", type=float, default=None)
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
    parser.add_argument("--fovy", type=float, default=45.0)
    parser.add_argument("--height-threshold", type=float, default=None)
    parser.add_argument("--threshold-x-min", type=float, default=0.24)
    parser.add_argument("--threshold-x-max", type=float, default=0.56)
    parser.add_argument("--threshold-alpha", type=float, default=0.92)
    parser.add_argument("--threshold-width", type=int, default=2)
    parser.add_argument("--show-height-threshold", action="store_true", default=True)
    parser.add_argument("--no-height-threshold", action="store_false", dest="show_height_threshold")
    args = parser.parse_args()

    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.image_separator_width) < 0:
        raise ValueError("--image-separator-width must be nonnegative.")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive.")
    if float(args.fovy) <= 0.0:
        raise ValueError("--fovy must be positive.")
    if float(args.sim_freq_hz) <= 0.0:
        raise ValueError("--sim-freq-hz must be positive.")
    if int(args.control_decimation) <= 0:
        raise ValueError("--control-decimation must be positive.")
    if args.control_timestep is not None and float(args.control_timestep) <= 0.0:
        raise ValueError("--control-timestep must be positive when provided.")
    if int(args.max_oracle_steps) < 0:
        raise ValueError("--max-oracle-steps must be non-negative.")
    if float(args.threshold_x_min) >= float(args.threshold_x_max):
        raise ValueError("--threshold-x-min must be less than --threshold-x-max.")
    if not 0.0 <= float(args.threshold_alpha) <= 1.0:
        raise ValueError("--threshold-alpha must be in [0, 1].")
    if int(args.threshold_width) <= 0:
        raise ValueError("--threshold-width must be positive.")
    return args


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_scalar_in_npz(run_dir: Path, filenames: tuple[str, ...], key: str) -> float | None:
    for filename in filenames:
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=False)
        if key in payload.files:
            value = np.asarray(payload[key])
            if value.shape == ():
                return float(value)
    return None


def resolve_control_timestep(run_dir: Path, args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    if args.control_timestep is not None:
        return float(args.control_timestep), {"source": "argument", "key": "control_timestep"}

    npz_value = first_scalar_in_npz(run_dir, ("tube_data.npz", "planned_actions.npz"), "control_timestep")
    if npz_value is not None:
        return float(npz_value), {"source": "npz", "key": "control_timestep"}

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        metadata = load_json(summary_path).get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("control_timestep") is not None:
            return float(metadata["control_timestep"]), {"source": str(summary_path), "key": "metadata.control_timestep"}

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        if metrics.get("control_timestep") is not None:
            return float(metrics["control_timestep"]), {"source": str(metrics_path), "key": "control_timestep"}

    timestep = float(args.control_decimation) / float(args.sim_freq_hz)
    return timestep, {
        "source": "control_decimation_over_sim_freq_hz",
        "control_decimation": int(args.control_decimation),
        "sim_freq_hz": float(args.sim_freq_hz),
    }


def trim_optional_series(series: np.ndarray | None, trim_metadata: dict[str, Any], used_length: int) -> np.ndarray | None:
    if series is None or series.ndim != 1:
        return None
    start_index = int(trim_metadata.get("start_index", 0))
    if series.shape[0] < start_index + used_length:
        return None
    return series[start_index : start_index + used_length]


def load_run_qpos(run_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    qpos_full, qpos_metadata = side.load_rollout_qpos(run_dir, args)
    qpos, trim_metadata = side.trim_oracle_prefix(qpos_full, run_dir, args)
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"Expected non-empty qpos with shape (T, nq), got {qpos.shape} in {run_dir}.")
    return qpos, qpos_metadata, trim_metadata


def draw_height_threshold(
    image: np.ndarray,
    projection: dict[str, np.ndarray],
    height_threshold: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if not bool(args.show_height_threshold):
        return image
    threshold_world = np.array(
        [
            [float(args.threshold_x_min), 0.0, float(height_threshold)],
            [float(args.threshold_x_max), 0.0, float(height_threshold)],
        ],
        dtype=np.float64,
    )
    threshold_px = side.project_world_points_to_pixels(
        threshold_world,
        camera_position=projection["position"],
        camera_right=projection["right"],
        camera_up=projection["up"],
        camera_forward=projection["forward"],
        fovy_deg=float(args.fovy),
        width=int(args.width),
        height=int(args.height),
    )
    return side.draw_polyline(
        image,
        threshold_px,
        color=side.THRESHOLD_COLOR,
        alpha=float(args.threshold_alpha),
        width=int(args.threshold_width),
    )


def render_run_row(
    *,
    run_dir: Path,
    env: Any,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    projection: dict[str, np.ndarray],
    height_threshold: float,
    args: argparse.Namespace,
) -> tuple[list[Image.Image], np.ndarray, list[float], int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    qpos, qpos_metadata, trim_metadata = load_run_qpos(run_dir, args)
    cube_positions = side.collect_cube_center_positions(env, qpos)
    sampled_indices = side.select_sample_indices(cube_positions, args)
    control_timestep, timing_metadata = resolve_control_timestep(run_dir, args)

    frames: list[np.ndarray] = []
    for frame_index in tqdm(sampled_indices, desc=f"Rendering {run_dir.name}"):
        side.set_env_state(env, qpos[int(frame_index)])
        frame = side.render_frame(renderer, env, camera).astype(np.float32)
        frame = draw_height_threshold(frame, projection, height_threshold, args)
        frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))

    start_index = int(trim_metadata.get("start_index", 0))
    timestamps = [float(start_index + int(index)) * float(control_timestep) for index in sampled_indices]
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    return images, sampled_indices, timestamps, int(qpos.shape[0]), timing_metadata, qpos_metadata, trim_metadata


def make_grid(rows: list[list[Image.Image]], row_timestamps: list[list[float]], image_separator_width: int) -> Image.Image:
    if len(rows) != 2:
        raise ValueError(f"Expected exactly two rows, got {len(rows)}.")
    if not rows[0] or not rows[1]:
        raise ValueError("Cannot tile empty rows.")
    if len(rows[0]) != len(rows[1]):
        raise ValueError(f"Rows must have equal length, got {len(rows[0])} and {len(rows[1])}.")
    if len(row_timestamps) != 2 or any(len(labels) != len(rows[0]) for labels in row_timestamps):
        raise ValueError("Timestamp rows must match rendered image rows.")

    width, height = rows[0][0].size
    cols = len(rows[0])
    separator_width = int(image_separator_width)
    label_height = max(22, height // 12)
    font_size = max(12, label_height // 2)
    font = load_font("DejaVuSans.ttf", font_size)

    panel_height = height + label_height
    grid_width = cols * width + (cols - 1) * separator_width
    grid_height = 2 * panel_height
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)

    for row_index, row in enumerate(rows):
        for col_index, image in enumerate(row):
            x = col_index * (width + separator_width)
            y = row_index * panel_height
            label = f"t={row_timestamps[row_index][col_index]:.2f}s"

            draw.rectangle([x, y, x + width, y + label_height], fill=(245, 245, 245))
            draw.text((x + 6, y + 4), label, fill=(20, 20, 20), font=font)
            grid.paste(image, (x, y + label_height))
            if separator_width > 0 and col_index < cols - 1:
                draw.rectangle(
                    [x + width, y + label_height, x + width + separator_width - 1, y + panel_height - 1],
                    fill=(0, 0, 0),
                )
    return grid


def main() -> None:
    args = parse_args()
    safe_run_dir = args.safe_run_dir.expanduser().resolve()
    unsafe_run_dir = args.unsafe_run_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = side.make_render_env(args)
    try:
        camera = side.side_camera(args)
        height_threshold, threshold_source = side.resolve_height_threshold(safe_run_dir, args)
        with mujoco.Renderer(env.unwrapped._model, height=int(args.height), width=int(args.width)) as renderer:
            renderer.scene.camera[0].frustum_center = 0.0
            renderer.scene.camera[0].frustum_width = 0.0
            side.set_env_state(env, load_run_qpos(safe_run_dir, args)[0][0])
            side.render_frame(renderer, env, camera)
            camera_position, camera_right, camera_up, camera_forward = side.camera_projection_from_scene(renderer)
            projection = {
                "position": camera_position,
                "right": camera_right,
                "up": camera_up,
                "forward": camera_forward,
            }

            safe_frames, safe_indices, safe_timestamps, safe_count, safe_timing_metadata, safe_qpos_metadata, safe_trim = render_run_row(
                run_dir=safe_run_dir,
                env=env,
                renderer=renderer,
                camera=camera,
                projection=projection,
                height_threshold=height_threshold,
                args=args,
            )
            unsafe_frames, unsafe_indices, unsafe_timestamps, unsafe_count, unsafe_timing_metadata, unsafe_qpos_metadata, unsafe_trim = render_run_row(
                run_dir=unsafe_run_dir,
                env=env,
                renderer=renderer,
                camera=camera,
                projection=projection,
                height_threshold=height_threshold,
                args=args,
            )
    finally:
        env.close()

    image = make_grid(
        [safe_frames, unsafe_frames],
        [safe_timestamps, unsafe_timestamps],
        int(args.image_separator_width),
    )
    image.save(output_path)

    metadata = {
        "output": str(output_path),
        "safe_run_dir": str(safe_run_dir),
        "unsafe_run_dir": str(unsafe_run_dir),
        "safe_qpos": safe_qpos_metadata,
        "unsafe_qpos": unsafe_qpos_metadata,
        "safe_source_frame_count": int(safe_count),
        "unsafe_source_frame_count": int(unsafe_count),
        "safe_oracle_trim": safe_trim,
        "unsafe_oracle_trim": unsafe_trim,
        "safe_sampled_indices": [int(index) for index in safe_indices],
        "unsafe_sampled_indices": [int(index) for index in unsafe_indices],
        "safe_timestamps_sec": [float(value) for value in safe_timestamps],
        "unsafe_timestamps_sec": [float(value) for value in unsafe_timestamps],
        "safe_timing_metadata": safe_timing_metadata,
        "unsafe_timing_metadata": unsafe_timing_metadata,
        "sample_by": str(args.sample_by),
        "num_frames_per_row": int(args.num_frames),
        "width": int(args.width),
        "height": int(args.height),
        "image_separator_width": int(args.image_separator_width),
        "height_threshold": float(height_threshold),
        "height_threshold_source": threshold_source,
        "show_height_threshold": bool(args.show_height_threshold),
        "camera": {
            "lookat": [float(value) for value in args.camera_lookat],
            "distance": float(args.camera_distance),
            "azimuth": float(args.camera_azimuth),
            "elevation": float(args.camera_elevation),
            "fovy": float(args.fovy),
        },
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
