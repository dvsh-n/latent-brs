#!/usr/bin/env python3
"""Render side- and front-view timelapse composites for saved rope plot trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from rope.shared.lab_env import BaseEnvConfig, LabEnv

from plots.rope.render_side_views import (
    DEFAULT_ROOT,
    DEFAULT_TRAJECTORIES,
    find_stall_cutoff,
    set_env_to_task_target_continuous,
)


ROPE_TENDON_NAME = "rope_tendon"
SIDE_TIMELAPSE_CAMERA_NAME = "side_timelapse_camera"
FRONT_TIMELAPSE_CAMERA_NAME = "video_cam"
DEFAULT_OBSTACLE_DATA_DIR = REPO_ROOT / "rope" / "plan" / "obstacle_data"
GEOM_OBJTYPE = int(mujoco.mjtObj.mjOBJ_GEOM)
SITE_OBJTYPE = int(mujoco.mjtObj.mjOBJ_SITE)
TENDON_OBJTYPE = int(mujoco.mjtObj.mjOBJ_TENDON)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--sample-by", choices=("motion", "index"), default="motion")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.15, -1.5, 0.95))
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--arms-min-alpha", type=float, default=0.24)
    parser.add_argument("--arms-max-alpha", dest="arms_max_alpha", type=float, default=0.8)
    parser.add_argument("--arm-lighten", type=float, default=0.0)
    parser.add_argument("--rope-min-alpha", type=float, default=0.24)
    parser.add_argument("--rope-max-alpha", "--rope-alpha", dest="rope_max_alpha", type=float, default=0.8)
    parser.add_argument("--output-name", default="side_timelapse.png")
    parser.add_argument("--front-output-name", default="front_timelapse.png")
    parser.add_argument("--front-camera-name", default=FRONT_TIMELAPSE_CAMERA_NAME)
    parser.add_argument("--front-timelapse", action="store_true", default=True)
    parser.add_argument("--no-front-timelapse", action="store_false", dest="front_timelapse")
    parser.add_argument("--trim-stalled", action="store_true", default=True)
    parser.add_argument("--no-trim-stalled", action="store_false", dest="trim_stalled")
    parser.add_argument("--trim-trajectories", nargs="+", default=["rope_unsafe"])
    parser.add_argument("--stall-delta-threshold", type=float, default=1e-3)
    parser.add_argument("--show-obstacle", action="store_true", default=True)
    parser.add_argument("--no-show-obstacle", action="store_false", dest="show_obstacle")
    parser.add_argument("--obstacle-data-dir", type=Path, default=DEFAULT_OBSTACLE_DATA_DIR)
    parser.add_argument("--obstacle-y-radius", type=float, default=0.68)
    parser.add_argument("--obstacle-rgba", nargs=4, type=float, default=(1.0, 0.58, 0.22, 1.0))
    args = parser.parse_args()
    for name in ("arms_min_alpha", "arms_max_alpha", "rope_min_alpha", "rope_max_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            option_name = name.replace("_", "-")
            raise ValueError(f"--{option_name} must be between 0 and 1, got {value}.")
    for prefix in ("arms", "rope"):
        min_alpha = float(getattr(args, f"{prefix}_min_alpha"))
        max_alpha = float(getattr(args, f"{prefix}_max_alpha"))
        if min_alpha > max_alpha:
            raise ValueError(
                f"--{prefix}-min-alpha must be less than or equal to --{prefix}-max-alpha, "
                f"got {min_alpha} > {max_alpha}."
            )
    if args.obstacle_y_radius <= 0.0:
        raise ValueError(f"--obstacle-y-radius must be positive, got {args.obstacle_y_radius}.")
    for channel in args.obstacle_rgba:
        if not 0.0 <= float(channel) <= 1.0:
            raise ValueError(f"--obstacle-rgba values must be between 0 and 1, got {args.obstacle_rgba}.")
    return args


def model_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, objtype, index)
    return "" if name is None else name


def arm_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        geom_name = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        body_name = model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name.startswith(("arm1_", "arm2_")) or geom_name.startswith(("arm1_", "arm2_")):
            ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)


def arm_site_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    for site_id in range(model.nsite):
        site_name = model_name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        body_id = int(model.site_bodyid[site_id])
        body_name = model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name.startswith(("arm1_", "arm2_")) or site_name.startswith(("arm1_", "arm2_")):
            ids.append(site_id)
    return np.asarray(ids, dtype=np.int32)


def evenly_spaced_indices(frame_count: int, sample_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Cannot sample an empty trajectory.")
    if sample_count <= 0:
        raise ValueError("--num-frames must be positive.")
    return np.unique(np.linspace(0, frame_count - 1, min(sample_count, frame_count), dtype=np.int64))


def motion_spaced_indices(task_targets: np.ndarray, sample_count: int) -> np.ndarray:
    if task_targets.shape[0] <= 0:
        raise ValueError("Cannot sample an empty trajectory.")
    if sample_count <= 0:
        raise ValueError("--num-frames must be positive.")
    if task_targets.shape[0] == 1:
        return np.array([0], dtype=np.int64)

    step_distances = np.linalg.norm(np.diff(task_targets, axis=0), axis=1)
    cumulative_distance = np.concatenate([[0.0], np.cumsum(step_distances)])
    total_distance = float(cumulative_distance[-1])
    if total_distance <= 1e-12:
        return np.array([0], dtype=np.int64)

    motion_targets = np.linspace(0.0, total_distance, min(sample_count, task_targets.shape[0]))
    indices = np.searchsorted(cumulative_distance, motion_targets, side="left")
    indices = np.clip(indices, 0, task_targets.shape[0] - 1)
    return np.unique(indices.astype(np.int64))


def select_sample_indices(task_targets: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.sample_by == "index":
        return evenly_spaced_indices(task_targets.shape[0], int(args.num_frames))
    return motion_spaced_indices(task_targets, int(args.num_frames))


def format_xml_vec(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def normalize_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"Cannot normalize zero-length {name}.")
    return vector / norm


def camera_axes_from_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(float(yaw_deg))
    pitch = np.deg2rad(float(pitch_deg))
    roll = np.deg2rad(float(roll_deg))

    forward = np.array(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward = normalize_vector(forward, name="camera forward vector")

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    right = normalize_vector(np.cross(forward, world_up), name="camera right vector")
    up = normalize_vector(np.cross(right, forward), name="camera up vector")

    if abs(roll) > 1e-12:
        cos_roll = float(np.cos(roll))
        sin_roll = float(np.sin(roll))
        rolled_right = cos_roll * right + sin_roll * up
        rolled_up = -sin_roll * right + cos_roll * up
        right = normalize_vector(rolled_right, name="rolled camera right vector")
        up = normalize_vector(rolled_up, name="rolled camera up vector")

    return right, up


def camera_xml(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    right, up = camera_axes_from_yaw_pitch_roll(args.camera_yaw, args.camera_pitch, args.camera_roll)
    position = tuple(float(value) for value in args.camera_position)
    xyaxes = tuple(float(value) for value in np.concatenate([right, up]))
    xml = (
        f'    <camera name="{SIDE_TIMELAPSE_CAMERA_NAME}" mode="fixed" '
        f'pos="{format_xml_vec(position)}" xyaxes="{format_xml_vec(xyaxes)}"/>'
    )
    metadata = {
        "name": SIDE_TIMELAPSE_CAMERA_NAME,
        "mode": "fixed_yaw_pitch_roll",
        "position": [float(value) for value in position],
        "yaw": float(args.camera_yaw),
        "pitch": float(args.camera_pitch),
        "roll": float(args.camera_roll),
        "xyaxes": [float(value) for value in xyaxes],
    }
    return xml, metadata


def fixed_camera_metadata(env: LabEnv, camera_id: int, camera_name: str) -> dict[str, object]:
    return {
        "name": str(camera_name),
        "mode": "fixed",
        "position": [float(value) for value in env.model.cam_pos[camera_id]],
        "fovy": float(env.model.cam_fovy[camera_id]),
    }


def half_ellipse_cylinder_mesh_xml(
    *,
    name: str,
    x_radius: float,
    half_length_y: float,
    z_height: float,
    arc_segments: int = 48,
) -> str:
    cross_section: list[tuple[float, float]] = []
    for index in range(arc_segments + 1):
        theta = np.pi * float(index) / float(arc_segments)
        x = x_radius * float(np.cos(theta))
        z = z_height * float(np.sin(theta))
        cross_section.append((x, z))

    vertices: list[tuple[float, float, float]] = []
    front: list[int] = []
    back: list[int] = []
    for y, index_list in ((-half_length_y, front), (half_length_y, back)):
        for x, z in cross_section:
            index_list.append(len(vertices))
            vertices.append((x, y, z))

    front_bottom_center = len(vertices)
    vertices.append((0.0, -half_length_y, 0.0))
    back_bottom_center = len(vertices)
    vertices.append((0.0, half_length_y, 0.0))

    faces: list[tuple[int, int, int]] = []
    for index in range(arc_segments):
        faces.append((front[index], back[index], back[index + 1]))
        faces.append((front[index], back[index + 1], front[index + 1]))

    for index in range(arc_segments):
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


def obstacle_speedbump_xml(args: argparse.Namespace) -> tuple[str, str, dict[str, object] | None]:
    if not bool(args.show_obstacle):
        return "", "", None

    summary_path = args.obstacle_data_dir.expanduser().resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing obstacle summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    obstacle_reach = summary["obstacle_reach"]
    obstacle_base_height = float(summary["obstacle_base_height"])
    obstacle_peak_height = float(summary["obstacle_height"])
    x_center = 0.5 * (float(obstacle_reach[0]) + float(obstacle_reach[1]))
    x_radius = 0.5 * (float(obstacle_reach[1]) - float(obstacle_reach[0]))
    z_height = obstacle_peak_height - obstacle_base_height
    if x_radius <= 0.0 or z_height <= 0.0:
        raise ValueError(
            f"Invalid obstacle speedbump dimensions from {summary_path}: "
            f"x_radius={x_radius}, z_height={z_height}."
        )

    mesh_name = "task_obstacle_speedbump_mesh"
    position = (x_center, 0.0, obstacle_base_height)
    size = (x_radius, float(args.obstacle_y_radius), z_height)
    rgba = tuple(float(value) for value in args.obstacle_rgba)
    asset_xml = half_ellipse_cylinder_mesh_xml(
        name=mesh_name,
        x_radius=x_radius,
        half_length_y=float(args.obstacle_y_radius),
        z_height=z_height,
    )
    worldbody_xml = (
        f'    <geom name="task_obstacle_speedbump" type="mesh" mesh="{mesh_name}" '
        f'pos="{format_xml_vec(position)}" '
        f'rgba="{format_xml_vec(rgba)}" contype="0" conaffinity="0"/>'
    )
    metadata = {
        "summary_path": str(summary_path),
        "profile": summary.get("obstacle_profile"),
        "geometry": "half_ellipse_cylinder_speedbump_mesh",
        "position": [float(value) for value in position],
        "size": [float(value) for value in size],
        "rgba": [float(value) for value in rgba],
        "obstacle_reach": [float(value) for value in obstacle_reach],
        "obstacle_base_height": obstacle_base_height,
        "obstacle_peak_height": obstacle_peak_height,
    }
    return asset_xml, worldbody_xml, metadata


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


def render_segmentation(renderer: mujoco.Renderer, env: LabEnv, camera_id: int) -> np.ndarray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(env.data, camera=camera_id)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
    renderer.disable_segmentation_rendering()
    return segmentation


def render_frame(renderer: mujoco.Renderer, env: LabEnv, camera_id: int) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def mask_from_object_ids(segmentation: np.ndarray, object_ids: np.ndarray, object_type: int) -> np.ndarray:
    if object_ids.size == 0:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    return (segmentation[..., 1] == object_type) & np.isin(segmentation[..., 0], object_ids)


def lighten(rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    rgb_float = rgb.astype(np.float32)
    return rgb_float + (255.0 - rgb_float) * amount


def alpha_composite(base: np.ndarray, layer: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return base
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = base.copy()
    output[mask] = (1.0 - alpha) * output[mask] + alpha * layer[mask]
    return output


def ramp_alpha(local_index: int, frame_count: int, min_alpha: float, max_alpha: float) -> float:
    if frame_count <= 1:
        return float(max_alpha)
    t = float(local_index) / float(frame_count - 1)
    return float((1.0 - t) * min_alpha + t * max_alpha)


def load_task_targets(run_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing executed states: {states_path}")

    states = np.load(states_path)
    task_targets = np.asarray(states["task_targets"], dtype=np.float64)
    if task_targets.ndim != 2 or task_targets.shape[1] != 3:
        raise ValueError(f"Expected task_targets with shape (T, 3), got {task_targets.shape} in {states_path}")

    metadata: dict[str, object] = {
        "source_frame_count": int(task_targets.shape[0]),
        "trimmed": False,
    }
    if bool(args.trim_stalled) and run_dir.name in set(args.trim_trajectories):
        cutoff_frame = find_stall_cutoff(task_targets, float(args.stall_delta_threshold))
        if cutoff_frame is not None:
            task_targets = task_targets[: cutoff_frame + 1]
            metadata.update(
                {
                    "trimmed": True,
                    "stall_delta_threshold": float(args.stall_delta_threshold),
                    "cutoff_frame": int(cutoff_frame),
                }
            )
    metadata["used_frame_count"] = int(task_targets.shape[0])
    return task_targets, metadata


def render_timelapse_composite(
    *,
    env: LabEnv,
    renderer: mujoco.Renderer,
    task_targets: np.ndarray,
    sampled_indices: np.ndarray,
    camera_id: int,
    arm_geoms: np.ndarray,
    arm_sites: np.ndarray,
    rope_tendon_id: int,
    args: argparse.Namespace,
    desc: str,
) -> np.ndarray:
    set_env_to_task_target_continuous(env, task_targets[0], first_frame=True)
    with hidden_dynamic_objects(env, arm_geoms, arm_sites, rope_tendon_id):
        background = render_frame(renderer, env, camera_id).astype(np.float32)

    composite = background
    for local_index, frame_index in enumerate(tqdm(sampled_indices, desc=desc)):
        arms_alpha = ramp_alpha(
            local_index,
            int(sampled_indices.shape[0]),
            float(args.arms_min_alpha),
            float(args.arms_max_alpha),
        )
        rope_alpha = ramp_alpha(
            local_index,
            int(sampled_indices.shape[0]),
            float(args.rope_min_alpha),
            float(args.rope_max_alpha),
        )
        set_env_to_task_target_continuous(env, task_targets[frame_index], first_frame=local_index == 0)
        rgb = render_frame(renderer, env, camera_id)
        segmentation = render_segmentation(renderer, env, camera_id)

        arm_mask = mask_from_object_ids(segmentation, arm_geoms, GEOM_OBJTYPE) | mask_from_object_ids(
            segmentation, arm_sites, SITE_OBJTYPE
        )
        rope_mask = (segmentation[..., 1] == TENDON_OBJTYPE) & (segmentation[..., 0] == rope_tendon_id)

        arm_layer = lighten(rgb, float(args.arm_lighten))
        composite = alpha_composite(composite, arm_layer, arm_mask, arms_alpha)
        composite = alpha_composite(composite, rgb.astype(np.float32), rope_mask, rope_alpha)
    return composite


def write_timelapse_output(
    *,
    run_dir: Path,
    output_name: str,
    metadata_name: str,
    metadata: dict[str, object],
    composite: np.ndarray,
    view_name: str,
) -> Path:
    output_path = run_dir / str(output_name)
    imageio.imwrite(output_path, np.clip(np.rint(composite), 0, 255).astype(np.uint8))
    metadata["output"] = output_path.name
    metadata["view"] = view_name
    with (run_dir / metadata_name).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return output_path


def timelapse_metadata(
    *,
    base_metadata: dict[str, object],
    task_targets: np.ndarray,
    sampled_indices: np.ndarray,
    args: argparse.Namespace,
    obstacle_metadata: dict[str, object] | None,
    camera_metadata: dict[str, object],
) -> dict[str, object]:
    metadata = dict(base_metadata)

    metadata.update(
        {
            "sampled_indices": [int(index) for index in sampled_indices],
            "sample_by": str(args.sample_by),
            "sampled_cumulative_motion": [
                float(value)
                for value in np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(task_targets, axis=0), axis=1))])[
                    sampled_indices
                ]
            ],
            "num_requested_frames": int(args.num_frames),
            "num_rendered_frames": int(sampled_indices.shape[0]),
            "obstacle": obstacle_metadata,
            "arms_min_alpha": float(args.arms_min_alpha),
            "arms_max_alpha": float(args.arms_max_alpha),
            "arm_lighten": float(args.arm_lighten),
            "rope_min_alpha": float(args.rope_min_alpha),
            "rope_max_alpha": float(args.rope_max_alpha),
            "arms_alpha_schedule": [
                ramp_alpha(index, int(sampled_indices.shape[0]), float(args.arms_min_alpha), float(args.arms_max_alpha))
                for index in range(int(sampled_indices.shape[0]))
            ],
            "rope_alpha_schedule": [
                ramp_alpha(index, int(sampled_indices.shape[0]), float(args.rope_min_alpha), float(args.rope_max_alpha))
                for index in range(int(sampled_indices.shape[0]))
            ],
            "camera": camera_metadata,
        }
    )
    return metadata


def render_timelapse(run_dir: Path, args: argparse.Namespace) -> None:
    task_targets, metadata = load_task_targets(run_dir, args)
    sampled_indices = select_sample_indices(task_targets, args)
    obstacle_asset_xml, obstacle_worldbody_xml, obstacle_metadata = obstacle_speedbump_xml(args)
    camera_worldbody_xml, side_camera_metadata = camera_xml(args)

    env = LabEnv(
        base_config=BaseEnvConfig(
            asset_extra_xml=obstacle_asset_xml,
            worldbody_extra_xml="\n".join(part for part in (obstacle_worldbody_xml, camera_worldbody_xml) if part),
        )
    )
    side_camera_id = env.model.camera(SIDE_TIMELAPSE_CAMERA_NAME).id
    arm_geoms = arm_geom_ids(env.model)
    arm_sites = arm_site_ids(env.model)
    rope_tendon_id = env.model.tendon(ROPE_TENDON_NAME).id

    with mujoco.Renderer(env.model, height=int(args.height), width=int(args.width)) as renderer:
        side_composite = render_timelapse_composite(
            env=env,
            renderer=renderer,
            task_targets=task_targets,
            sampled_indices=sampled_indices,
            camera_id=side_camera_id,
            arm_geoms=arm_geoms,
            arm_sites=arm_sites,
            rope_tendon_id=rope_tendon_id,
            args=args,
            desc=f"Rendering {run_dir.name} side timelapse",
        )
        side_output_path = write_timelapse_output(
            run_dir=run_dir,
            output_name=str(args.output_name),
            metadata_name="side_timelapse_metadata.json",
            metadata=timelapse_metadata(
                base_metadata=metadata,
                task_targets=task_targets,
                sampled_indices=sampled_indices,
                args=args,
                obstacle_metadata=obstacle_metadata,
                camera_metadata=side_camera_metadata,
            ),
            composite=side_composite,
            view_name="side",
        )
        print(f"Wrote side-view timelapse for {run_dir}: {side_output_path}")

        if bool(args.front_timelapse):
            front_camera_name = str(args.front_camera_name)
            front_camera_id = env.model.camera(front_camera_name).id
            front_composite = render_timelapse_composite(
                env=env,
                renderer=renderer,
                task_targets=task_targets,
                sampled_indices=sampled_indices,
                camera_id=front_camera_id,
                arm_geoms=arm_geoms,
                arm_sites=arm_sites,
                rope_tendon_id=rope_tendon_id,
                args=args,
                desc=f"Rendering {run_dir.name} front timelapse",
            )
            front_output_path = write_timelapse_output(
                run_dir=run_dir,
                output_name=str(args.front_output_name),
                metadata_name="front_timelapse_metadata.json",
                metadata=timelapse_metadata(
                    base_metadata=metadata,
                    task_targets=task_targets,
                    sampled_indices=sampled_indices,
                    args=args,
                    obstacle_metadata=obstacle_metadata,
                    camera_metadata=fixed_camera_metadata(env, front_camera_id, front_camera_name),
                ),
                composite=front_composite,
                view_name="front",
            )
            print(f"Wrote front-view timelapse for {run_dir}: {front_output_path}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    for name in args.trajectories:
        render_timelapse(root / name, args)


if __name__ == "__main__":
    main()
