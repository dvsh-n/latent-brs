#!/usr/bin/env python3
"""Render side-view timelapse composites for saved OGBench cube rollouts."""

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
OGBENCH_ROOT = REPO_ROOT / "third_party" / "ogbench"
for path in (REPO_ROOT, OGBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from ogbench.manipspace import lie

from ogbench_cube.data.ogbench_cube_data_gen import (
    DEFAULT_CONTROL_DECIMATION,
    DEFAULT_ENV_NAME,
    DEFAULT_SIM_FREQ_HZ,
    LocalCubePlanOracle,
)
from ogbench_cube.plan.obs_data_collect_3d_ellipsoid import hide_target_cube, make_env


DEFAULT_ROOT = REPO_ROOT / "plots" / "ogbench_cube"
DEFAULT_TRAJECTORIES = ("ogbench_cube_safe", "ogbench_cube_unsafe")
DEFAULT_OUTPUT_NAME = "side_timelapse.png"
DEFAULT_COMBINED_OUTPUT_NAME = "ogbench_cube_side_timelapse.png"
DEFAULT_HEIGHT_THRESHOLD = 0.09
GEOM_OBJTYPE = int(mujoco.mjtObj.mjOBJ_GEOM)
LINE_COLOR = np.array([15.0, 15.0, 15.0], dtype=np.float32)
TIP_COLOR = np.array([255.0, 238.0, 72.0], dtype=np.float32)
THRESHOLD_COLOR = np.array([220.0, 60.0, 36.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--combined-output-name", default=DEFAULT_COMBINED_OUTPUT_NAME)
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--sample-by", choices=("motion", "index"), default="motion")
    parser.add_argument("--trim-oracle", action="store_true", default=True)
    parser.add_argument("--no-trim-oracle", action="store_false", dest="trim_oracle")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
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
    parser.add_argument("--dynamic-min-alpha", type=float, default=0.22)
    parser.add_argument("--dynamic-max-alpha", type=float, default=0.82)
    parser.add_argument("--line-alpha", type=float, default=0.9)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--threshold-alpha", type=float, default=0.92)
    parser.add_argument("--threshold-width", type=int, default=3)
    parser.add_argument("--tip-marker-alpha", type=float, default=0.95)
    parser.add_argument("--tip-marker-radius", type=int, default=5)
    parser.add_argument("--height-threshold", type=float, default=None)
    parser.add_argument("--threshold-x-min", type=float, default=0.24)
    parser.add_argument("--threshold-x-max", type=float, default=0.56)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive.")
    if float(args.fovy) <= 0.0:
        raise ValueError("--fovy must be positive.")
    for name in ("dynamic_min_alpha", "dynamic_max_alpha", "line_alpha", "threshold_alpha", "tip_marker_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}.")
    if float(args.dynamic_min_alpha) > float(args.dynamic_max_alpha):
        raise ValueError("--dynamic-min-alpha must be less than or equal to --dynamic-max-alpha.")
    if int(args.line_width) <= 0 or int(args.threshold_width) <= 0:
        raise ValueError("Line widths must be positive.")
    if int(args.tip_marker_radius) < 0:
        raise ValueError("--tip-marker-radius must be non-negative.")
    if float(args.threshold_x_min) >= float(args.threshold_x_max):
        raise ValueError("--threshold-x-min must be less than --threshold-x-max.")
    if int(args.max_oracle_steps) < 0:
        raise ValueError("--max-oracle-steps must be non-negative.")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def qpos_from_npz(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        return None
    states = np.load(states_path)
    for key in ("qpos", "qpos_history", "executed_qpos"):
        if key in states:
            return np.asarray(states[key], dtype=np.float64), {"source": str(states_path), "source_key": key}
    return None


def qpos_from_metrics(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = load_json(metrics_path)
    if "qpos_history" not in metrics:
        return None
    return np.asarray(metrics["qpos_history"], dtype=np.float64), {"source": str(metrics_path), "source_key": "qpos_history"}


def make_render_env(args: argparse.Namespace) -> Any:
    env_args = argparse.Namespace(
        env_name=str(args.env_name),
        sim_freq_hz=float(args.sim_freq_hz),
        control_decimation=int(args.control_decimation),
        max_episode_steps=int(args.max_episode_steps),
        width=int(args.width),
        height=int(args.height),
    )
    return make_env(env_args)


def set_env_state(env: Any, qpos: np.ndarray, qvel: np.ndarray | None = None) -> None:
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    if qvel is None:
        unwrapped._data.qvel[:] = 0.0
    else:
        unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    hide_target_cube(env)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()


def restore_target_pose(env: Any, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    hide_target_cube(env)


def cube_is_grasped(env: Any, *, contact_threshold: float, alignment_threshold: float) -> bool:
    info = env.unwrapped.get_step_info()
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float64)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= float(contact_threshold) and block_alignment <= float(alignment_threshold))


def run_oracle_to_grasp(env: Any, args: argparse.Namespace) -> int:
    if cube_is_grasped(
        env,
        contact_threshold=float(args.grasp_contact_threshold),
        alignment_threshold=float(args.grasp_alignment_threshold),
    ):
        return 0

    oracle = LocalCubePlanOracle(
        env=env,
        segment_dt=float(args.oracle_segment_dt),
        noise=float(args.oracle_noise),
        noise_smoothing=float(args.oracle_noise_smoothing),
    )
    oracle.reset(None, env.unwrapped.get_step_info())
    for step in range(int(args.max_oracle_steps)):
        info = env.unwrapped.get_step_info()
        action = np.asarray(oracle.select_action(None, info), dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        hide_target_cube(env)
        if terminated or truncated:
            return step + 1
        if cube_is_grasped(
            env,
            contact_threshold=float(args.grasp_contact_threshold),
            alignment_threshold=float(args.grasp_alignment_threshold),
        ):
            return step + 1
    return int(args.max_oracle_steps)


def qpos_from_resimulation(run_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]] | None:
    summary_path = run_dir / "trajectory_summary.json"
    actions_path = run_dir / "executed_actions.npz"
    if not summary_path.is_file() or not actions_path.is_file():
        return None

    summary = load_json(summary_path)
    start_goal = summary.get("start_goal", {})
    if "qpos_init" not in start_goal:
        return None
    actions_npz = np.load(actions_path)
    if "executed_actions_raw" not in actions_npz:
        return None

    qpos_init = np.asarray(start_goal["qpos_init"], dtype=np.float64)
    qvel_init = np.asarray(start_goal.get("qvel_init", np.zeros(max(qpos_init.shape[0] - 1, 0))), dtype=np.float64)
    actions = np.asarray(actions_npz["executed_actions_raw"], dtype=np.float32)
    env = make_render_env(args)
    qpos_history: list[np.ndarray] = []
    try:
        env.reset(seed=int(summary.get("metadata", {}).get("episode_seed", args.seed)))
        set_env_state(env, qpos_init, qvel_init)
        target_pos = start_goal.get("target_block_pos_goal", start_goal.get("target_block_pos_init"))
        target_yaw = start_goal.get("target_block_yaw_goal", start_goal.get("target_block_yaw_init", 0.0))
        if target_pos is not None:
            restore_target_pose(env, np.asarray(target_pos, dtype=np.float64), float(target_yaw))
        oracle_steps = run_oracle_to_grasp(env, args)
        qpos_history.append(env.unwrapped._data.qpos.copy())
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            hide_target_cube(env)
            qpos_history.append(env.unwrapped._data.qpos.copy())
            if terminated or truncated:
                break
    finally:
        env.close()

    return (
        np.stack(qpos_history, axis=0),
        {
            "source": str(actions_path),
            "source_key": "executed_actions_raw",
            "resimulated_from": str(summary_path),
            "oracle_prefix_steps": int(oracle_steps),
        },
    )


def load_rollout_qpos(run_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    for loader in (qpos_from_npz, qpos_from_metrics):
        loaded = loader(run_dir)
        if loaded is not None:
            return loaded
    loaded_resim = qpos_from_resimulation(run_dir, args)
    if loaded_resim is not None:
        return loaded_resim
    raise FileNotFoundError(
        f"No qpos history or resimulatable qpos/actions artifacts found in {run_dir}. "
        "Expected executed_states.npz qpos/qpos_history, metrics.json qpos_history, or "
        "trajectory_summary.json plus executed_actions.npz executed_actions_raw."
    )


def load_logged_gripper_heights(run_dir: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    states_path = run_dir / "executed_states.npz"
    if states_path.is_file():
        states = np.load(states_path)
        if "gripper_height" in states:
            return (
                np.asarray(states["gripper_height"], dtype=np.float64),
                {"source": str(states_path), "source_key": "gripper_height"},
            )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        if "gripper_heights" in metrics:
            return (
                np.asarray(metrics["gripper_heights"], dtype=np.float64),
                {"source": str(metrics_path), "source_key": "gripper_heights"},
            )

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        records = load_json(summary_path).get("step_records", [])
        heights = [record["gripper_height"] for record in records if "gripper_height" in record]
        if heights:
            return (
                np.asarray(heights, dtype=np.float64),
                {"source": str(summary_path), "source_key": "step_records[].gripper_height"},
            )

    return None


def resolve_height_threshold(run_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    if args.height_threshold is not None:
        return float(args.height_threshold), "manual"

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        metadata = load_json(summary_path).get("metadata", {})
        if "gripper_height_threshold" in metadata:
            return float(metadata["gripper_height_threshold"]), str(summary_path)

    log_path = run_dir / "latent_state_log.jsonl"
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                metrics = record.get("metrics", {})
                if "gripper_height_threshold" in metrics:
                    return float(metrics["gripper_height_threshold"]), str(log_path)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        if "gripper_height_threshold" in metrics:
            return float(metrics["gripper_height_threshold"]), str(metrics_path)

    return DEFAULT_HEIGHT_THRESHOLD, "default"


def resolve_planner_start_index(run_dir: Path) -> tuple[int, dict[str, Any]]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = load_json(metrics_path)
        handoff_step = metrics.get("handoff_step")
        if handoff_step is not None:
            return max(0, int(handoff_step)), {"source": str(metrics_path), "source_key": "handoff_step"}

    summary_path = run_dir / "trajectory_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        records = summary.get("step_records", [])
        for index, record in enumerate(records):
            phase = str(record.get("phase", ""))
            if phase not in {"oracle", "oracle_grasp", "grasp", "grasping"}:
                return int(index), {"source": str(summary_path), "source_key": "step_records[].phase", "phase": phase}

    return 0, {"source": "none", "source_key": None}


def trim_oracle_prefix(qpos: np.ndarray, run_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(args.trim_oracle):
        return qpos, {"enabled": False, "start_index": 0}
    start_index, source = resolve_planner_start_index(run_dir)
    start_index = min(max(0, int(start_index)), max(0, int(qpos.shape[0]) - 1))
    return qpos[start_index:], {"enabled": True, "start_index": int(start_index), **source}


def trim_optional_series(series: np.ndarray | None, start_index: int, used_length: int) -> np.ndarray | None:
    if series is None:
        return None
    if series.ndim != 1:
        return None
    if series.shape[0] < start_index + used_length:
        return None
    return series[start_index : start_index + used_length]


def model_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, objtype, index)
    return "" if name is None else name


def dynamic_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        geom_name = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = model_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if body_name.startswith("ur5e/") or body_name == "object_0" or geom_name == "object_0":
            ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def pinch_site_id(model: mujoco.MjModel) -> int:
    return int(model.site("ur5e/robotiq/pinch").id)


@contextmanager
def hidden_geoms(model: mujoco.MjModel, geom_ids: np.ndarray) -> Iterator[None]:
    geom_rgba = np.asarray(model.geom_rgba).copy()
    try:
        model.geom_rgba[geom_ids, 3] = 0.0
        yield
    finally:
        model.geom_rgba[:] = geom_rgba


def side_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(args.camera_lookat, dtype=np.float64)
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.camera_azimuth)
    camera.elevation = float(args.camera_elevation)
    return camera


def render_frame(renderer: mujoco.Renderer, env: Any, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.disable_segmentation_rendering()
    renderer.update_scene(env.unwrapped._data, camera=camera)
    return renderer.render()


def render_segmentation(renderer: mujoco.Renderer, env: Any, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.unwrapped._data, camera=camera)
    segmentation = renderer.render()
    renderer.disable_segmentation_rendering()
    return segmentation


def mask_from_geom_ids(segmentation: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    if geom_ids.size == 0:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    return (segmentation[..., 1] == GEOM_OBJTYPE) & np.isin(segmentation[..., 0], geom_ids)


def alpha_composite_mask(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    output = base.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output[mask] = (1.0 - alpha) * output[mask] + alpha * layer.astype(np.float32)[mask]
    return output


def alpha_composite_image(base: np.ndarray, layer: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * base + alpha * layer.astype(np.float32)


def ramp_alpha(index: int, frame_count: int, min_alpha: float, max_alpha: float) -> float:
    if frame_count <= 1:
        return float(max_alpha)
    t = float(index) / float(frame_count - 1)
    return float((1.0 - t) * min_alpha + t * max_alpha)


def evenly_spaced_indices(frame_count: int, sample_count: int) -> np.ndarray:
    return np.unique(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=np.int64))


def motion_spaced_indices(points: np.ndarray, sample_count: int) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.array([0], dtype=np.int64)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.array([0], dtype=np.int64)
    targets = np.linspace(0.0, total, min(sample_count, points.shape[0]))
    return np.unique(np.clip(np.searchsorted(cumulative, targets), 0, points.shape[0] - 1).astype(np.int64))


def select_sample_indices(gripper_positions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.sample_by == "index":
        return evenly_spaced_indices(gripper_positions.shape[0], int(args.num_frames))
    return motion_spaced_indices(gripper_positions[:, [0, 2]], int(args.num_frames))


def collect_gripper_positions(env: Any, qpos: np.ndarray, site_id: int) -> np.ndarray:
    positions: list[np.ndarray] = []
    for qpos_row in qpos:
        set_env_state(env, qpos_row)
        positions.append(env.unwrapped._data.site_xpos[site_id].copy())
    return np.stack(positions, axis=0)


def collect_cube_center_positions(env: Any, qpos: np.ndarray) -> np.ndarray:
    positions: list[np.ndarray] = []
    for qpos_row in qpos:
        set_env_state(env, qpos_row)
        info = env.unwrapped.get_step_info()
        target_block = int(info.get("privileged/target_block", 0))
        positions.append(np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64).copy())
    return np.stack(positions, axis=0)


def camera_projection_from_scene(renderer: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_camera = renderer.scene.camera[0]
    right_camera = renderer.scene.camera[1]
    position = 0.5 * (
        np.asarray(left_camera.pos, dtype=np.float64) + np.asarray(right_camera.pos, dtype=np.float64)
    )
    forward = 0.5 * (
        np.asarray(left_camera.forward, dtype=np.float64) + np.asarray(right_camera.forward, dtype=np.float64)
    )
    up = 0.5 * (
        np.asarray(left_camera.up, dtype=np.float64) + np.asarray(right_camera.up, dtype=np.float64)
    )
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    return position, right, up, forward


def project_world_points_to_pixels(
    points: np.ndarray,
    *,
    camera_position: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
    camera_forward: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64) - camera_position[None, :]
    camera_x = rel @ camera_right
    camera_y = rel @ camera_up
    depth = rel @ camera_forward
    focal = 0.5 * float(height) / np.tan(0.5 * np.deg2rad(float(fovy_deg)))
    pixels = np.full((rel.shape[0], 2), np.nan, dtype=np.float64)
    visible = depth > 1e-6
    pixels[visible, 0] = 0.5 * float(width) + focal * camera_x[visible] / depth[visible]
    pixels[visible, 1] = 0.5 * float(height) - focal * camera_y[visible] / depth[visible]
    return pixels


def draw_polyline(image: np.ndarray, points_px: np.ndarray, *, color: np.ndarray, alpha: float, width: int) -> np.ndarray:
    valid = np.isfinite(points_px).all(axis=1)
    if int(np.sum(valid)) < 2:
        return image
    overlay = image.copy()
    points = np.rint(points_px[valid]).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [points], isClosed=False, color=tuple(float(value) for value in color), thickness=int(width), lineType=cv2.LINE_AA)
    return alpha_composite_image(image, overlay, alpha)


def draw_markers(image: np.ndarray, points_px: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if int(args.tip_marker_radius) <= 0:
        return image
    composite = image
    for point in points_px:
        if not np.isfinite(point).all():
            continue
        overlay = composite.copy()
        center = tuple(int(round(value)) for value in point)
        cv2.circle(overlay, center, int(args.tip_marker_radius), tuple(float(v) for v in TIP_COLOR), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, center, int(args.tip_marker_radius), (15.0, 15.0, 15.0), thickness=1, lineType=cv2.LINE_AA)
        composite = alpha_composite_image(composite, overlay, float(args.tip_marker_alpha))
    return composite


def render_timelapse(run_dir: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    qpos_full, qpos_metadata = load_rollout_qpos(run_dir, args)
    qpos, trim_metadata = trim_oracle_prefix(qpos_full, run_dir, args)
    logged_heights_loaded = load_logged_gripper_heights(run_dir)
    logged_heights = None if logged_heights_loaded is None else logged_heights_loaded[0]
    height_metadata = None if logged_heights_loaded is None else logged_heights_loaded[1]
    trimmed_logged_heights = trim_optional_series(
        logged_heights,
        int(trim_metadata["start_index"]),
        int(qpos.shape[0]),
    )
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"Expected non-empty qpos with shape (T, nq), got {qpos.shape} in {run_dir}.")
    height_threshold, threshold_source = resolve_height_threshold(run_dir, args)

    env = make_render_env(args)
    try:
        camera = side_camera(args)
        site_id = pinch_site_id(env.unwrapped._model)
        dyn_geom_ids = dynamic_geom_ids(env.unwrapped._model)
        gripper_positions = collect_gripper_positions(env, qpos, site_id)
        cube_center_positions = collect_cube_center_positions(env, qpos)
        sampled_indices = select_sample_indices(cube_center_positions, args)

        with mujoco.Renderer(env.unwrapped._model, height=int(args.height), width=int(args.width)) as renderer:
            renderer.scene.camera[0].frustum_center = 0.0
            renderer.scene.camera[0].frustum_width = 0.0
            set_env_state(env, qpos[0])
            with hidden_geoms(env.unwrapped._model, dyn_geom_ids):
                background = render_frame(renderer, env, camera).astype(np.float32)
            camera_position, camera_right, camera_up, camera_forward = camera_projection_from_scene(renderer)

            composite = background
            for local_index, frame_index in enumerate(tqdm(sampled_indices, desc=f"Rendering {run_dir.name} side timelapse")):
                alpha = ramp_alpha(
                    local_index,
                    int(sampled_indices.shape[0]),
                    float(args.dynamic_min_alpha),
                    float(args.dynamic_max_alpha),
                )
                set_env_state(env, qpos[frame_index])
                rgb = render_frame(renderer, env, camera).astype(np.float32)
                segmentation = render_segmentation(renderer, env, camera)
                mask = mask_from_geom_ids(segmentation, dyn_geom_ids)
                composite = alpha_composite_mask(composite, rgb, mask, alpha)

            sampled_cube_center_positions = cube_center_positions[sampled_indices]
            gripper_px = project_world_points_to_pixels(
                sampled_cube_center_positions,
                camera_position=camera_position,
                camera_right=camera_right,
                camera_up=camera_up,
                camera_forward=camera_forward,
                fovy_deg=float(args.fovy),
                width=int(args.width),
                height=int(args.height),
            )
            threshold_world = np.array(
                [
                    [float(args.threshold_x_min), 0.0, float(height_threshold)],
                    [float(args.threshold_x_max), 0.0, float(height_threshold)],
                ],
                dtype=np.float64,
            )
            threshold_px = project_world_points_to_pixels(
                threshold_world,
                camera_position=camera_position,
                camera_right=camera_right,
                camera_up=camera_up,
                camera_forward=camera_forward,
                fovy_deg=float(args.fovy),
                width=int(args.width),
                height=int(args.height),
            )
            composite = draw_polyline(
                composite,
                threshold_px,
                color=THRESHOLD_COLOR,
                alpha=float(args.threshold_alpha),
                width=int(args.threshold_width),
            )
            composite = draw_polyline(composite, gripper_px, color=LINE_COLOR, alpha=float(args.line_alpha), width=int(args.line_width))
            composite = draw_markers(composite, gripper_px, args)
    finally:
        env.close()

    output_path = run_dir / str(args.output_name)
    imageio.imwrite(output_path, np.clip(np.rint(composite), 0, 255).astype(np.uint8))
    metadata = {
        "run_dir": str(run_dir),
        "output_path": str(output_path),
        "qpos": qpos_metadata,
        "source_frame_count": int(qpos_full.shape[0]),
        "used_frame_count": int(qpos.shape[0]),
        "oracle_trim": trim_metadata,
        "sampled_indices": [int(index) for index in sampled_indices],
        "sample_by": str(args.sample_by),
        "num_requested_frames": int(args.num_frames),
        "num_rendered_frames": int(sampled_indices.shape[0]),
        "height_threshold": float(height_threshold),
        "height_threshold_source": threshold_source,
        "gripper_height": height_metadata if trimmed_logged_heights is not None else {"source": "simulated_pinch_site_z"},
        "trace_point_source": "target_cube_center",
        "threshold_world_points": threshold_world.tolist(),
        "gripper_positions": gripper_positions.astype(np.float64).tolist(),
        "cube_center_positions": cube_center_positions.astype(np.float64).tolist(),
        "gripper_trace_positions": cube_center_positions.astype(np.float64).tolist(),
        "camera": {
            "lookat": [float(value) for value in args.camera_lookat],
            "distance": float(args.camera_distance),
            "azimuth": float(args.camera_azimuth),
            "elevation": float(args.camera_elevation),
            "fovy": float(args.fovy),
            "position": [float(value) for value in camera_position],
            "right": [float(value) for value in camera_right],
            "up": [float(value) for value in camera_up],
            "forward": [float(value) for value in camera_forward],
        },
    }
    save_json(run_dir / "side_timelapse_metadata.json", metadata)
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
    parts: list[np.ndarray] = []
    for index, image in enumerate(resized):
        if index > 0:
            parts.append(spacer)
        parts.append(image)
    imageio.imwrite(combined_path, np.concatenate(parts, axis=1))


def main() -> None:
    args = parse_args()
    validate_args(args)
    root = args.root.expanduser().resolve()
    output_paths: list[Path] = []
    metadata: list[dict[str, Any]] = []
    for name in args.trajectories:
        output_path, run_metadata = render_timelapse(root / name, args)
        output_paths.append(output_path)
        metadata.append(run_metadata)
        print(f"Wrote side timelapse for {name}: {output_path}")

    if not bool(args.no_combined) and len(output_paths) > 1:
        combined_path = root / str(args.combined_output_name)
        write_combined_image(output_paths, combined_path)
        save_json(
            root / "ogbench_cube_side_timelapse_metadata.json",
            {"combined_output_path": str(combined_path), "inputs": [str(path) for path in output_paths], "runs": metadata},
        )
        print(f"Wrote combined side timelapse: {combined_path}")


if __name__ == "__main__":
    main()
