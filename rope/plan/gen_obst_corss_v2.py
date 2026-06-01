#!/usr/bin/env python3
"""Generate rope HDF5 episodes that cross the obstacle reach band at safe height."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from rope.data.rope_data_gen import (
    ACTION_DIM,
    append_rows,
    create_resizable_dataset,
    extract_step_info,
    render_rgb_frame,
)
from rope.shared.lab_env import BaseEnvConfig, LabEnv, TaskState


DEFAULT_OBSTACLE_SUMMARY = "rope/models/obs_net/da270d7d1050f110/summary.json"
DEFAULT_OUTDIR = "rope/data/obstacle_crossing"
DEFAULT_OUTPUT_NAME = "rope_obstacle_crossing.h5"
DEFAULT_CAMERA = "video_cam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obstacle-summary", type=Path, default=Path(DEFAULT_OBSTACLE_SUMMARY))
    parser.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-segment", type=int, default=18)
    parser.add_argument("--control-decimation", type=int, default=25)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--disable-shadows", action="store_true", default=True)
    parser.add_argument(
        "--safe-height-margin",
        type=float,
        default=0.04,
        help="Extra proxy low-rope clearance above the obstacle cutoff when crossing the reach band.",
    )
    parser.add_argument(
        "--endpoint-low-margin",
        type=float,
        default=0.04,
        help="Target endpoint proxy low-rope height this far below the obstacle cutoff.",
    )
    parser.add_argument(
        "--width-search-samples",
        type=int,
        default=101,
        help="Number of candidate rope widths used to find one that can place endpoints below the cutoff. Smaller widths are tried first.",
    )
    parser.add_argument(
        "--straight-path-samples",
        type=int,
        default=81,
        help="Number of samples used to verify that the direct start-goal path collides with the obstacle.",
    )
    parser.add_argument(
        "--straight-collision-margin",
        type=float,
        default=0.0,
        help="Require the straight path to go this far below the low-rope cutoff inside the obstacle reach band.",
    )
    return parser.parse_args()


def load_obstacle_metadata(summary_path: Path) -> dict[str, object]:
    with summary_path.expanduser().open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    metadata = summary.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{summary_path} does not contain source_metadata.")
    return metadata


def metadata_array(metadata: dict[str, object], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"Obstacle metadata is missing {key!r}.")
    value = np.asarray(metadata[key], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"Expected metadata {key!r} shape {shape}, got {value.shape}.")
    return value


def endpoint_boxes(metadata: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = metadata_array(metadata, "task_lower", (3,))
    upper = metadata_array(metadata, "task_upper", (3,))
    reach = metadata_array(metadata, "obstacle_reach", (2,))
    low_lower = lower.copy()
    low_upper = upper.copy()
    low_upper[0] = reach[0]
    high_lower = lower.copy()
    high_lower[0] = reach[1]
    high_upper = upper.copy()
    return low_lower, low_upper, high_lower, high_upper


def proxy_low_rope_height(state: np.ndarray, proxy_env: LabEnv) -> float:
    proxy_env.reset(TaskState.from_array(state))
    points = proxy_env.get_proxy_rope_points()
    return float(np.min(points[:, 2]))


def crossing_height_for_width(
    *,
    width: float,
    low_rope_cutoff: float,
    task_height_upper: float,
    margin: float,
    proxy_env: LabEnv,
) -> float:
    state = np.asarray([0.5 * sum(proxy_env.task_bounds.reach), task_height_upper, width], dtype=np.float64)
    low_at_upper = proxy_low_rope_height(state, proxy_env)
    needed = task_height_upper + (float(low_rope_cutoff) + float(margin) - low_at_upper)
    return float(np.clip(needed, proxy_env.task_bounds.height[0], proxy_env.task_bounds.height[1]))


def task_height_for_target_low_rope(
    *,
    width: float,
    target_low_rope_height: float,
    proxy_env: LabEnv,
) -> float:
    task_height_upper = float(proxy_env.task_bounds.height[1])
    state = np.asarray([0.5 * sum(proxy_env.task_bounds.reach), task_height_upper, width], dtype=np.float64)
    low_at_upper = proxy_low_rope_height(state, proxy_env)
    needed = task_height_upper + (float(target_low_rope_height) - low_at_upper)
    return float(np.clip(needed, proxy_env.task_bounds.height[0], proxy_env.task_bounds.height[1]))


def choose_low_endpoint_width_and_height(
    rng: np.random.Generator,
    *,
    width_lower: float,
    width_upper: float,
    low_rope_cutoff: float,
    endpoint_low_margin: float,
    width_search_samples: int,
    proxy_env: LabEnv,
) -> tuple[float, float, float]:
    if width_search_samples < 2:
        raise ValueError("--width-search-samples must be at least 2.")

    target_low = float(low_rope_cutoff) - float(endpoint_low_margin)
    # Smaller endpoint separation leaves more slack in the fixed-length rope, so try close-arm
    # configurations first; they are the most likely to sag below the obstacle cutoff.
    widths = np.linspace(float(width_lower), float(width_upper), num=int(width_search_samples), dtype=np.float64)

    best: tuple[float, float, float] | None = None
    iterator = tqdm(widths, desc="Searching feasible endpoint width", unit="width", leave=False)
    for sample_index, width in enumerate(iterator, start=1):
        endpoint_height = task_height_for_target_low_rope(
            width=float(width),
            target_low_rope_height=target_low,
            proxy_env=proxy_env,
        )
        low_height = proxy_low_rope_height(np.asarray([0.0, endpoint_height, float(width)], dtype=np.float64), proxy_env)
        if best is None or low_height < best[2]:
            best = (float(width), float(endpoint_height), float(low_height))
            iterator.set_postfix(best_low=f"{best[2]:.4f}", cutoff=f"{low_rope_cutoff:.4f}", tried=sample_index)
        if low_height <= float(low_rope_cutoff):
            iterator.set_postfix(found=True, low=f"{low_height:.4f}", tried=sample_index)
            return float(width), float(endpoint_height), float(low_height)

    if best is None:
        raise RuntimeError("No candidate widths were checked.")
    raise RuntimeError(
        "Could not find any width whose lowest reachable endpoint is below the obstacle cutoff. "
        f"best_width={best[0]:.6g}, best_height={best[1]:.6g}, "
        f"best_low_rope_height={best[2]:.6g}, cutoff={low_rope_cutoff:.6g}. "
        "Try reducing --endpoint-low-margin, increasing the allowed task bounds, or checking the obstacle cutoff."
    )


def interpolate_waypoints(waypoints: np.ndarray, steps_per_segment: int) -> np.ndarray:
    states: list[np.ndarray] = [waypoints[0]]
    for start, goal in zip(waypoints[:-1], waypoints[1:], strict=True):
        for step in range(1, steps_per_segment + 1):
            alpha = step / float(steps_per_segment)
            states.append((1.0 - alpha) * start + alpha * goal)
    return np.stack(states, axis=0).astype(np.float32)


def straight_path_obstacle_diagnostic(
    start: np.ndarray,
    goal: np.ndarray,
    *,
    obstacle_reach: np.ndarray,
    low_rope_cutoff: float,
    num_samples: int,
    proxy_env: LabEnv,
) -> dict[str, object]:
    if num_samples < 2:
        raise ValueError("--straight-path-samples must be at least 2.")
    alphas = np.linspace(0.0, 1.0, num=int(num_samples), dtype=np.float64)
    states = (1.0 - alphas[:, None]) * np.asarray(start, dtype=np.float64) + alphas[:, None] * np.asarray(goal, dtype=np.float64)
    low_heights = np.asarray(
        [
            proxy_low_rope_height(state, proxy_env)
            for state in tqdm(states, desc="Checking straight path", unit="sample", leave=False)
        ],
        dtype=np.float64,
    )
    in_reach = (states[:, 0] >= float(obstacle_reach[0])) & (states[:, 0] <= float(obstacle_reach[1]))
    if np.any(in_reach):
        scoped_low = low_heights[in_reach]
        scoped_states = states[in_reach]
        local_index = int(np.argmin(scoped_low))
        min_low = float(scoped_low[local_index])
        min_state = scoped_states[local_index]
    else:
        min_low = float("inf")
        min_state = np.full((3,), np.nan, dtype=np.float64)
    return {
        "collides": bool(min_low <= float(low_rope_cutoff)),
        "min_low_rope_height_in_obstacle_reach": min_low,
        "clearance_to_cutoff": float(min_low - float(low_rope_cutoff)),
        "min_state_in_obstacle_reach": min_state.astype(float).tolist(),
        "num_samples": int(num_samples),
    }


def sample_waypoints(
    rng: np.random.Generator,
    metadata: dict[str, object],
    proxy_env: LabEnv,
    *,
    safe_height_margin: float,
    endpoint_low_margin: float,
    width_search_samples: int,
    straight_path_samples: int,
    straight_collision_margin: float,
) -> tuple[np.ndarray, dict[str, object]]:
    low_lower, low_upper, high_lower, high_upper = endpoint_boxes(metadata)
    obstacle_reach = metadata_array(metadata, "obstacle_reach", (2,))
    low_rope_cutoff = float(metadata["low_rope_cutoff"])

    width_lower = max(float(low_lower[2]), float(high_lower[2]))
    width_upper = min(float(low_upper[2]), float(high_upper[2]))
    endpoint_low_target = float(low_rope_cutoff) - float(endpoint_low_margin)
    width, endpoint_height, constructed_low_height = choose_low_endpoint_width_and_height(
        rng,
        width_lower=width_lower,
        width_upper=width_upper,
        low_rope_cutoff=low_rope_cutoff,
        endpoint_low_margin=endpoint_low_margin,
        width_search_samples=int(width_search_samples),
        proxy_env=proxy_env,
    )

    low_reach = float(rng.uniform(float(low_lower[0]), float(low_upper[0])))
    high_reach = float(rng.uniform(float(high_lower[0]), float(high_upper[0])))
    start_low_side = np.asarray([low_reach, endpoint_height, width], dtype=np.float64)
    goal_high_side = np.asarray([high_reach, endpoint_height, width], dtype=np.float64)

    low_side_low_height = proxy_low_rope_height(start_low_side, proxy_env)
    high_side_low_height = proxy_low_rope_height(goal_high_side, proxy_env)
    if low_side_low_height > low_rope_cutoff or high_side_low_height > low_rope_cutoff:
        raise RuntimeError(
            "Could not construct low endpoints below the obstacle cutoff. "
            f"low_side={low_side_low_height:.6g}, high_side={high_side_low_height:.6g}, "
            f"cutoff={low_rope_cutoff:.6g}."
        )

    if bool(rng.integers(0, 2)):
        start = start_low_side
        goal = goal_high_side
        before_reach = float(obstacle_reach[0])
        after_reach = float(obstacle_reach[1])
    else:
        start = goal_high_side
        goal = start_low_side
        before_reach = float(obstacle_reach[1])
        after_reach = float(obstacle_reach[0])

    straight_diagnostic = straight_path_obstacle_diagnostic(
        start,
        goal,
        obstacle_reach=obstacle_reach,
        low_rope_cutoff=low_rope_cutoff,
        num_samples=int(straight_path_samples),
        proxy_env=proxy_env,
    )
    required_clearance = -float(straight_collision_margin)
    if float(straight_diagnostic["clearance_to_cutoff"]) > required_clearance:
        raise RuntimeError(
            "Constructed endpoints did not make the straight-line path collide with the requested margin. "
            f"clearance_to_cutoff={float(straight_diagnostic['clearance_to_cutoff']):.6g}, "
            f"required<={required_clearance:.6g}."
        )
    straight_diagnostic["construction"] = "low endpoints outside obstacle reach, same low height and width"
    straight_diagnostic["endpoint_low_target"] = float(endpoint_low_target)
    straight_diagnostic["constructed_width"] = float(width)
    straight_diagnostic["constructed_endpoint_height"] = float(endpoint_height)
    straight_diagnostic["constructed_low_rope_height_at_search_state"] = float(constructed_low_height)
    straight_diagnostic["low_side_low_rope_height"] = float(low_side_low_height)
    straight_diagnostic["high_side_low_rope_height"] = float(high_side_low_height)

    crossing_width = float(0.5 * (start[2] + goal[2]))
    crossing_height = crossing_height_for_width(
        width=crossing_width,
        low_rope_cutoff=low_rope_cutoff,
        task_height_upper=float(proxy_env.task_bounds.height[1]),
        margin=safe_height_margin,
        proxy_env=proxy_env,
    )
    if crossing_height >= float(proxy_env.task_bounds.height[1]) - 1e-9:
        probe = np.asarray([0.5 * sum(obstacle_reach), crossing_height, crossing_width], dtype=np.float64)
        low_height = proxy_low_rope_height(probe, proxy_env)
        if low_height <= low_rope_cutoff:
            raise RuntimeError(
                "Could not make a high crossing state clear the obstacle cutoff. "
                f"low_height={low_height:.6g}, cutoff={low_rope_cutoff:.6g}."
            )

    bridge_start = np.asarray([before_reach, crossing_height, crossing_width], dtype=np.float64)
    bridge_end = np.asarray([after_reach, crossing_height, crossing_width], dtype=np.float64)
    return np.stack([start, bridge_start, bridge_end, goal], axis=0), straight_diagnostic


def collect_task_trajectory(
    *,
    env: LabEnv,
    renderer: mujoco.Renderer,
    camera_id: int,
    task_states: np.ndarray,
    control_decimation: int,
    control_timestep: float,
    disable_shadows: bool,
) -> dict[str, np.ndarray]:
    env.reset(TaskState.from_array(task_states[0]))
    step_info = extract_step_info(env, elapsed_time=0.0)
    observations = [step_info["observation"]]
    frames = [render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)]
    actions: list[np.ndarray] = []
    task_target = [step_info["task_target"]]
    qpos = [step_info["qpos"]]
    qvel = [step_info["qvel"]]
    control = [step_info["control"]]
    left_attachment_pos = [step_info["left_attachment_pos"]]
    right_attachment_pos = [step_info["right_attachment_pos"]]
    rope_length = [step_info["rope_length"]]
    time_values = [step_info["time"]]

    current_target = np.asarray(task_target[0], dtype=np.float64)
    iterator = tqdm(task_states[1:], desc="Rendering generated trajectory", unit="step", leave=False)
    for step_idx, desired in enumerate(iterator, start=1):
        desired = np.asarray(desired, dtype=np.float64)
        env.apply_task_delta(desired - current_target)
        applied_target = env.task_controller.desired_state.as_array().astype(np.float64)
        actions.append((applied_target - current_target).astype(np.float32))
        current_target = applied_target
        env.step(control_decimation)

        step_info = extract_step_info(env, elapsed_time=step_idx * control_timestep)
        observations.append(step_info["observation"])
        frames.append(render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows))
        task_target.append(step_info["task_target"])
        qpos.append(step_info["qpos"])
        qvel.append(step_info["qvel"])
        control.append(step_info["control"])
        left_attachment_pos.append(step_info["left_attachment_pos"])
        right_attachment_pos.append(step_info["right_attachment_pos"])
        rope_length.append(step_info["rope_length"])
        time_values.append(step_info["time"])

    return {
        "observation": np.stack(observations, axis=0),
        "pixels": np.stack(frames, axis=0),
        "action": np.stack(actions, axis=0),
        "task_target": np.stack(task_target, axis=0),
        "qpos": np.stack(qpos, axis=0),
        "qvel": np.stack(qvel, axis=0),
        "control": np.stack(control, axis=0),
        "left_attachment_pos": np.stack(left_attachment_pos, axis=0),
        "right_attachment_pos": np.stack(right_attachment_pos, axis=0),
        "rope_length": np.stack(rope_length, axis=0),
        "time": np.stack(time_values, axis=0),
    }


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be positive.")
    if args.steps_per_segment < 1:
        raise ValueError("--steps-per-segment must be positive.")
    if args.control_decimation < 1:
        raise ValueError("--control-decimation must be positive.")
    if args.width_search_samples < 2:
        raise ValueError("--width-search-samples must be at least 2.")
    if args.straight_path_samples < 2:
        raise ValueError("--straight-path-samples must be at least 2.")

    metadata = load_obstacle_metadata(args.obstacle_summary)
    outdir = args.outdir.expanduser().resolve()
    output_path = outdir / args.output_name
    video_dir = outdir / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    rng = np.random.default_rng(args.seed)
    env = LabEnv()
    proxy_env = LabEnv(base_config=BaseEnvConfig(enable_proxy_rope=True))
    camera_id = env.model.camera(args.camera).id
    control_timestep = float(env.model.opt.timestep) * float(args.control_decimation)
    video_fps = float(args.video_fps) if args.video_fps is not None else 1.0 / control_timestep
    compression = None if args.compression == "none" else args.compression

    with mujoco.Renderer(env.model, height=args.height, width=args.width) as renderer:
        env.reset()
        sample_info = extract_step_info(env, elapsed_time=0.0)
        sample_frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=args.disable_shadows)

        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "stable_worldmodel_hdf5"
            h5.attrs["source"] = "rope/plan/generate_obstacle_crossing_episode.py"
            h5.attrs["seed"] = int(args.seed)
            h5.attrs["video_dir"] = str(video_dir)
            h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
            h5.attrs["save_mp4"] = bool(args.save_mp4)
            h5.attrs["camera"] = str(args.camera)
            h5.attrs["disable_shadows"] = bool(args.disable_shadows)
            h5.attrs["mode"] = "OBSTACLE_CROSSING_BRIDGE_STRAIGHT_PATH_COLLIDES"
            h5.attrs["video_fps"] = video_fps
            h5.attrs["physics_timestep"] = float(env.model.opt.timestep)
            h5.attrs["control_timestep"] = control_timestep
            h5.attrs["control_decimation"] = int(args.control_decimation)
            h5.attrs["max_episode_steps"] = int(3 * args.steps_per_segment)
            h5.attrs["segment_duration"] = float(args.steps_per_segment * control_timestep)
            h5.attrs["num_waypoints"] = 4
            h5.attrs["goal_tolerance"] = 0.0
            h5.attrs["observation_dim"] = int(sample_info["observation"].shape[0])
            h5.attrs["action_dim"] = ACTION_DIM
            h5.attrs["qpos_dim"] = int(sample_info["qpos"].shape[0])
            h5.attrs["qvel_dim"] = int(sample_info["qvel"].shape[0])
            h5.attrs["control_dim"] = int(sample_info["control"].shape[0])
            h5.attrs["task_bounds"] = json.dumps(
                {
                    "reach": list(env.task_bounds.reach),
                    "height": list(env.task_bounds.height),
                    "width": list(env.task_bounds.width),
                }
            )
            h5.attrs["nominal_task_state"] = json.dumps(env.nominal_state.as_array().tolist())
            h5.attrs["obstacle_metadata"] = json.dumps(metadata)
            h5.attrs["straight_path_samples"] = int(args.straight_path_samples)
            h5.attrs["straight_collision_margin"] = float(args.straight_collision_margin)
            h5.attrs["endpoint_low_margin"] = float(args.endpoint_low_margin)
            h5.attrs["safe_height_margin"] = float(args.safe_height_margin)
            h5.attrs["width_search_samples"] = int(args.width_search_samples)

            ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
            ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
            reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
            seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
            terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
            truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
            pixels_ds = create_resizable_dataset(h5, "pixels", sample_frame.shape, np.uint8, compression=compression, chunks=(1, *sample_frame.shape))
            action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
            obs_ds = create_resizable_dataset(h5, "observation", sample_info["observation"].shape, np.float32, chunks=True)
            task_target_ds = create_resizable_dataset(h5, "task_target", (3,), np.float32, chunks=True)
            qpos_ds = create_resizable_dataset(h5, "qpos", sample_info["qpos"].shape, np.float32, chunks=True)
            qvel_ds = create_resizable_dataset(h5, "qvel", sample_info["qvel"].shape, np.float32, chunks=True)
            control_ds = create_resizable_dataset(h5, "control", sample_info["control"].shape, np.float32, chunks=True)
            left_attachment_pos_ds = create_resizable_dataset(h5, "left_attachment_pos", (3,), np.float32, chunks=True)
            right_attachment_pos_ds = create_resizable_dataset(h5, "right_attachment_pos", (3,), np.float32, chunks=True)
            rope_length_ds = create_resizable_dataset(h5, "rope_length", sample_info["rope_length"].shape, np.float32, chunks=True)
            time_ds = create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
            episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
            step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

            summaries = []
            for episode_idx in tqdm(range(args.num_episodes), desc="Generating obstacle-crossing episodes", unit="episode"):
                waypoints, straight_diagnostic = sample_waypoints(
                    rng,
                    metadata,
                    proxy_env,
                    safe_height_margin=float(args.safe_height_margin),
                    endpoint_low_margin=float(args.endpoint_low_margin),
                    width_search_samples=int(args.width_search_samples),
                    straight_path_samples=int(args.straight_path_samples),
                    straight_collision_margin=float(args.straight_collision_margin),
                )
                task_states = interpolate_waypoints(waypoints, int(args.steps_per_segment))
                proxy_low = np.asarray(
                    [
                        proxy_low_rope_height(state, proxy_env)
                        for state in tqdm(task_states, desc="Checking bridge path", unit="sample", leave=False)
                    ],
                    dtype=np.float32,
                )
                generated_path_diagnostic = straight_path_obstacle_diagnostic(
                    task_states[0],
                    task_states[-1],
                    obstacle_reach=np.asarray(metadata["obstacle_reach"], dtype=np.float64),
                    low_rope_cutoff=float(metadata["low_rope_cutoff"]),
                    num_samples=int(args.straight_path_samples),
                    proxy_env=proxy_env,
                )
                trajectory = collect_task_trajectory(
                    env=env,
                    renderer=renderer,
                    camera_id=camera_id,
                    task_states=task_states,
                    control_decimation=int(args.control_decimation),
                    control_timestep=control_timestep,
                    disable_shadows=bool(args.disable_shadows),
                )

                padded_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                padded_actions[:-1] = trajectory["action"]
                padded_actions[-1] = np.nan

                offset, _ = append_rows(pixels_ds, trajectory["pixels"])
                append_rows(action_ds, padded_actions)
                append_rows(obs_ds, trajectory["observation"])
                append_rows(task_target_ds, trajectory["task_target"])
                append_rows(qpos_ds, trajectory["qpos"])
                append_rows(qvel_ds, trajectory["qvel"])
                append_rows(control_ds, trajectory["control"])
                append_rows(left_attachment_pos_ds, trajectory["left_attachment_pos"])
                append_rows(right_attachment_pos_ds, trajectory["right_attachment_pos"])
                append_rows(rope_length_ds, trajectory["rope_length"])
                append_rows(time_ds, trajectory["time"])
                append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
                append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
                append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([0.0], dtype=np.float32))
                append_rows(seed_ds, np.asarray([args.seed + episode_idx], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([True], dtype=np.bool_))
                append_rows(truncated_ds, np.asarray([False], dtype=np.bool_))

                if args.save_mp4:
                    imageio.mimwrite(video_dir / f"trajectory_{episode_idx:07d}.mp4", trajectory["pixels"], fps=video_fps, macro_block_size=1)

                summaries.append(
                    {
                        "episode_idx": episode_idx,
                        "offset": int(offset),
                        "length": int(trajectory["observation"].shape[0]),
                        "waypoints": waypoints.astype(float).tolist(),
                        "straight_start_goal_path": straight_diagnostic,
                        "straight_start_goal_path_recomputed": generated_path_diagnostic,
                        "proxy_low_rope_min": float(np.min(proxy_low)),
                        "proxy_low_rope_min_in_obstacle_reach": float(
                            np.min(
                                proxy_low[
                                    (task_states[:, 0] >= float(metadata["obstacle_reach"][0]))
                                    & (task_states[:, 0] <= float(metadata["obstacle_reach"][1]))
                                ]
                            )
                        ),
                    }
                )

            h5.attrs["num_episodes"] = int(args.num_episodes)
            h5.attrs["total_frames"] = int(pixels_ds.shape[0])

    summary_path = outdir / "obstacle_crossing_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_path": str(output_path),
                "obstacle_summary": str(args.obstacle_summary),
                "low_rope_cutoff": float(metadata["low_rope_cutoff"]),
                "obstacle_reach": list(metadata["obstacle_reach"]),
                "endpoint_low_margin": float(args.endpoint_low_margin),
                "safe_height_margin": float(args.safe_height_margin),
                "width_search_samples": int(args.width_search_samples),
                "episodes": summaries,
            },
            handle,
            indent=2,
        )

    print(f"Saved obstacle-crossing dataset: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
