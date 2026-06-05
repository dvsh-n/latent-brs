#!/usr/bin/env python3
"""Render a 2-row Reacher timelapse PDF.

The top row is the safe rollout and the bottom row is the unsafe rollout. Each
panel is a MuJoCo render from the default Reacher camera, with timestamp labels
and grid styling matching plots/rope/make_side_timelapse_safe_unsafe.py.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from plots.reacher import plot_obstacle_membership as obstacle_plot


DEFAULT_SAFE_RUN_DIR = REPO_ROOT / "plots" / "reacher" / "reacher_safe_10"
DEFAULT_UNSAFE_RUN_DIR = REPO_ROOT / "plots" / "reacher" / "reacher_unsafe_10"
DEFAULT_OBSTACLE_SUMMARY = REPO_ROOT / "reacher" / "plan" / "obstacle_data_joint_box" / "summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "plots" / "reacher" / "safe_unsafe_side_timelapse.pdf"


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
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=50.0)
    parser.add_argument("--control-timestep", type=float, default=None)
    parser.add_argument("--show-obstacle-circle", action="store_true", default=True)
    parser.add_argument("--no-obstacle-circle", action="store_false", dest="show_obstacle_circle")
    parser.add_argument("--obstacle-summary", type=Path, default=DEFAULT_OBSTACLE_SUMMARY)
    parser.add_argument("--obstacle-alpha", type=float, default=0.18)
    parser.add_argument("--obstacle-outline-alpha", type=float, default=0.78)
    parser.add_argument("--obstacle-outline-width", type=int, default=3)
    parser.add_argument("--hide-target", action="store_true", default=True)
    parser.add_argument("--show-target", action="store_false", dest="hide_target")
    args = parser.parse_args()

    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.image_separator_width) < 0:
        raise ValueError("--image-separator-width must be nonnegative.")
    if float(args.physics_freq_hz) <= 0.0:
        raise ValueError("--physics-freq-hz must be positive.")
    if args.control_timestep is not None and float(args.control_timestep) <= 0.0:
        raise ValueError("--control-timestep must be positive when provided.")
    for name in ("obstacle_alpha", "obstacle_outline_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if int(args.obstacle_outline_width) <= 0:
        raise ValueError("--obstacle-outline-width must be positive.")
    return args


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_executed_states(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing executed states: {states_path}")
    states = np.load(states_path, allow_pickle=False)
    qpos = np.asarray(states["qpos"], dtype=np.float64)
    qvel = np.asarray(states["qvel"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 2:
        raise ValueError(f"Expected qpos with shape (T, >=2), got {qpos.shape} in {states_path}")
    if qvel.shape[0] != qpos.shape[0]:
        raise ValueError(f"qpos/qvel length mismatch in {states_path}: {qpos.shape} vs {qvel.shape}")
    return qpos, qvel


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample from an empty trajectory.")
    if count <= 1:
        return np.asarray([0], dtype=np.int64)
    if length <= count:
        return np.arange(length, dtype=np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def motion_spaced_indices(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return evenly_spaced_indices(points.shape[0], count)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    if not np.isfinite(cumulative).all() or cumulative[-1] <= 1e-12:
        return evenly_spaced_indices(points.shape[0], count)
    targets = np.linspace(0.0, cumulative[-1], min(count, points.shape[0]))
    return np.asarray([int(np.argmin(np.abs(cumulative - target))) for target in targets], dtype=np.int64)


def select_sample_indices(qpos: np.ndarray, qvel: np.ndarray, summary: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    if args.sample_by == "index":
        return evenly_spaced_indices(qpos.shape[0], int(args.num_frames))
    env = make_env(args)
    try:
        points = fingertip_trace_for_qpos(env, qpos, qvel, summary)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return motion_spaced_indices(points, int(args.num_frames))


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

    npz_value = first_scalar_in_npz(run_dir, ("tube_data.npz", "mpc_plans.npz"), "control_timestep")
    if npz_value is not None:
        return float(npz_value), {"source": "npz", "key": "control_timestep"}

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        metadata = summary.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("control_timestep") is not None:
            return float(metadata["control_timestep"]), {"source": str(summary_path), "key": "metadata.control_timestep"}

    return 1.0 / float(args.physics_freq_hz), {"source": "physics_freq_hz", "physics_freq_hz": float(args.physics_freq_hz)}


def make_env(args: argparse.Namespace) -> Any:
    return obstacle_plot.make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )


def set_env_state(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> None:
    physics = env.physics
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
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


def render_frame(env: Any, qpos: np.ndarray, qvel: np.ndarray, summary: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    set_env_state(env, qpos, qvel)
    frame = env.physics.render(height=int(args.height), width=int(args.width), camera_id=int(args.camera_id)).astype(np.float32)
    if bool(args.show_obstacle_circle):
        frame = obstacle_plot.draw_obstacle_circle(frame, env, summary, args)
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def render_run_row(
    *,
    run_dir: Path,
    env: Any,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[Image.Image], np.ndarray, list[float], int, dict[str, Any]]:
    qpos, qvel = load_executed_states(run_dir)
    sampled_indices = select_sample_indices(qpos, qvel, summary, args)
    control_timestep, timing_metadata = resolve_control_timestep(run_dir, args)
    frames: list[np.ndarray] = []
    for frame_index in tqdm(sampled_indices, desc=f"Rendering {run_dir.name}"):
        frames.append(render_frame(env, qpos[int(frame_index)], qvel[int(frame_index)], summary, args))
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    timestamps = [float(index) * float(control_timestep) for index in sampled_indices]
    return images, sampled_indices, timestamps, int(qpos.shape[0]), timing_metadata


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
    summary_path = args.obstacle_summary.expanduser().resolve()
    summary = load_json(summary_path)

    env = make_env(args)
    try:
        if bool(args.hide_target):
            obstacle_plot.hide_target(env)
        safe_frames, safe_indices, safe_timestamps, safe_count, safe_timing_metadata = render_run_row(
            run_dir=safe_run_dir,
            env=env,
            summary=summary,
            args=args,
        )
        unsafe_frames, unsafe_indices, unsafe_timestamps, unsafe_count, unsafe_timing_metadata = render_run_row(
            run_dir=unsafe_run_dir,
            env=env,
            summary=summary,
            args=args,
        )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

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
        "camera": {"id": int(args.camera_id)},
        "hide_target": bool(args.hide_target),
        "show_obstacle_circle": bool(args.show_obstacle_circle),
        "obstacle_summary": str(summary_path),
        "obstacle_circle": {
            "center_xy": [0.0, 0.0],
            "radius_source": "max fingertip distance over joint-space obstacle box",
            "radius": float(obstacle_plot.obstacle_radius_from_qpos_box(summary)),
        },
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
