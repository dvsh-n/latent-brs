#!/usr/bin/env python3
"""Render a 2-row side-view rope timelapse PDF.

The top row is the safe rollout and the bottom row is the unsafe rollout. Each
panel is a raw MuJoCo render from the side camera with the rope, arms, and
obstacle visible. The grid styling matches plots/rope/make_rope_timelapse_example.py,
with timestamp labels and no goal panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from plots.rope import render_side_timelapse as side
from rope.shared.lab_env import BaseEnvConfig, LabEnv


DEFAULT_SAFE_RUN_DIR = REPO_ROOT / "plots" / "rope" / "rope_safe_7"
DEFAULT_UNSAFE_RUN_DIR = REPO_ROOT / "plots" / "rope" / "rope_unsafe_7"
DEFAULT_OUTPUT = REPO_ROOT / "plots" / "rope" / "safe_unsafe_side_timelapse.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-run-dir", type=Path, default=DEFAULT_SAFE_RUN_DIR)
    parser.add_argument("--unsafe-run-dir", type=Path, default=DEFAULT_UNSAFE_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--sample-by", choices=("motion", "index"), default="motion")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--image-separator-width", type=int, default=2)
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.15, -1.5, 0.95))
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--show-obstacle", action="store_true", default=True)
    parser.add_argument("--no-show-obstacle", action="store_false", dest="show_obstacle")
    parser.add_argument("--obstacle-data-dir", type=Path, default=side.DEFAULT_OBSTACLE_DATA_DIR)
    parser.add_argument("--obstacle-y-radius", type=float, default=0.68)
    parser.add_argument("--obstacle-rgba", nargs=4, type=float, default=(0.28, 0.28, 0.28, 1.0))
    args = parser.parse_args()

    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.image_separator_width) < 0:
        raise ValueError("--image-separator-width must be nonnegative.")
    if float(args.obstacle_y_radius) <= 0.0:
        raise ValueError("--obstacle-y-radius must be positive.")
    if any(not 0.0 <= float(channel) <= 1.0 for channel in args.obstacle_rgba):
        raise ValueError("--obstacle-rgba channels must be in [0, 1].")
    return args


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_control_timestep(run_dir: Path, env: LabEnv) -> tuple[float, dict[str, object]]:
    trajectory_summary_path = run_dir / "trajectory_summary.json"
    if trajectory_summary_path.is_file():
        summary = load_json(trajectory_summary_path)
        metadata = summary.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("control_timestep") is not None:
            return float(metadata["control_timestep"]), {"source": str(trajectory_summary_path), "key": "metadata.control_timestep"}

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        if metrics.get("control_timestep") is not None:
            return float(metrics["control_timestep"]), {"source": str(metrics_path), "key": "control_timestep"}

    control_decimation = 25
    for path in (trajectory_summary_path, metrics_path):
        if not path.is_file():
            continue
        payload = load_json(path)
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("control_decimation") is not None:
            control_decimation = int(metadata["control_decimation"])
            break
        if payload.get("control_decimation") is not None:
            control_decimation = int(payload["control_decimation"])
            break

    timestep = float(env.model.opt.timestep) * float(control_decimation)
    return timestep, {"source": "model_timestep_x_control_decimation", "control_decimation": int(control_decimation)}


def load_task_targets(run_dir: Path) -> np.ndarray:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing executed states: {states_path}")
    states = np.load(states_path, allow_pickle=False)
    task_targets = np.asarray(states["task_targets"], dtype=np.float64)
    if task_targets.ndim != 2 or task_targets.shape[1] != 3:
        raise ValueError(f"Expected task_targets with shape (T, 3), got {task_targets.shape} in {states_path}")
    return task_targets


def select_sample_indices(task_targets: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.sample_by == "index":
        return side.evenly_spaced_indices(task_targets.shape[0], int(args.num_frames))
    return side.motion_spaced_indices(task_targets, int(args.num_frames))


def make_env(args: argparse.Namespace) -> LabEnv:
    obstacle_asset_xml, obstacle_worldbody_xml, _ = side.obstacle_speedbump_xml(args)
    camera_worldbody_xml, _ = side.camera_xml(args)
    worldbody_extra_xml = "\n".join(part for part in (obstacle_worldbody_xml, camera_worldbody_xml) if part)
    return LabEnv(
        base_config=BaseEnvConfig(
            asset_extra_xml=obstacle_asset_xml,
            worldbody_extra_xml=worldbody_extra_xml,
            offscreen_width=max(int(args.width), 640),
            offscreen_height=max(int(args.height), 480),
        )
    )


def render_run_row(
    *,
    run_dir: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
    camera_id: int,
    args: argparse.Namespace,
) -> tuple[list[Image.Image], np.ndarray, list[float], int, dict[str, object]]:
    task_targets = load_task_targets(run_dir)
    sampled_indices = select_sample_indices(task_targets, args)
    control_timestep, timing_metadata = resolve_control_timestep(run_dir, env)
    frames: list[np.ndarray] = []
    for local_index, frame_index in enumerate(tqdm(sampled_indices, desc=f"Rendering {run_dir.name}")):
        side.set_env_to_task_target_continuous(
            env,
            task_targets[int(frame_index)],
            first_frame=local_index == 0,
        )
        frames.append(side.render_frame(renderer, env, camera_id))
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    timestamps = [float(index) * float(control_timestep) for index in sampled_indices]
    return images, sampled_indices, timestamps, int(task_targets.shape[0]), timing_metadata


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

    env = make_env(args)
    camera_id = env.model.camera(side.SIDE_TIMELAPSE_CAMERA_NAME).id

    try:
        with mujoco.Renderer(env.model, height=int(args.height), width=int(args.width)) as renderer:
            safe_frames, safe_indices, safe_timestamps, safe_count, safe_timing_metadata = render_run_row(
                run_dir=safe_run_dir,
                env=env,
                renderer=renderer,
                camera_id=camera_id,
                args=args,
            )
            unsafe_frames, unsafe_indices, unsafe_timestamps, unsafe_count, unsafe_timing_metadata = render_run_row(
                run_dir=unsafe_run_dir,
                env=env,
                renderer=renderer,
                camera_id=camera_id,
                args=args,
            )
    finally:
        del env

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
        "safe_source_frame_count": int(safe_count),
        "unsafe_source_frame_count": int(unsafe_count),
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
        "camera": {
            "name": side.SIDE_TIMELAPSE_CAMERA_NAME,
            "position": [float(value) for value in args.camera_position],
            "yaw": float(args.camera_yaw),
            "pitch": float(args.camera_pitch),
            "roll": float(args.camera_roll),
        },
        "show_obstacle": bool(args.show_obstacle),
        "obstacle_rgba": [float(value) for value in args.obstacle_rgba],
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
