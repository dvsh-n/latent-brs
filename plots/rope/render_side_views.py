#!/usr/bin/env python3
"""Render side-view replays for the saved rope plot trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from rope.shared.lab_env import LabEnv, TaskState


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORIES = ("rope_safe", "rope_unsafe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trajectories", nargs="+", default=list(DEFAULT_TRAJECTORIES))
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--azimuth", type=float, default=-90.0)
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--lookat", nargs=3, type=float, default=(0.0, 0.0, 1.05))
    parser.add_argument("--trim-stalled", action="store_true", default=True)
    parser.add_argument("--no-trim-stalled", action="store_false", dest="trim_stalled")
    parser.add_argument("--trim-trajectories", nargs="+", default=["rope_unsafe"])
    parser.add_argument("--stall-delta-threshold", type=float, default=1e-3)
    return parser.parse_args()


def make_side_camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(args.lookat, dtype=np.float64)
    camera.distance = float(args.distance)
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)
    return camera


def render_frame(renderer: mujoco.Renderer, env: LabEnv, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def set_env_to_task_target_continuous(env: LabEnv, task_target: np.ndarray, *, first_frame: bool) -> None:
    if first_frame:
        env.reset(TaskState.from_array(task_target))
        return

    env.set_task_target(TaskState.from_array(task_target))
    env.set_arm_joint_positions(env.joint_controller.target)
    env.data.qvel[:] = 0.0
    if env.base_config.enable_proxy_rope:
        env._sync_proxy_anchors()
    mujoco.mj_forward(env.model, env.data)


def find_stall_cutoff(task_targets: np.ndarray, threshold: float) -> int | None:
    deltas = np.linalg.norm(np.diff(task_targets, axis=0), axis=1)
    for index in range(deltas.shape[0]):
        if np.all(deltas[index:] <= threshold):
            return index
    return None


def render_trajectory(run_dir: Path, args: argparse.Namespace) -> None:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing executed states: {states_path}")

    states = np.load(states_path)
    task_targets = np.asarray(states["task_targets"], dtype=np.float64)
    if task_targets.ndim != 2 or task_targets.shape[1] != 3:
        raise ValueError(f"Expected task_targets with shape (T, 3), got {task_targets.shape} in {states_path}")
    elapsed_time = np.asarray(states["elapsed_time"], dtype=np.float64) if "elapsed_time" in states else None

    cutoff_frame = None
    if bool(args.trim_stalled) and run_dir.name in set(args.trim_trajectories):
        cutoff_frame = find_stall_cutoff(task_targets, float(args.stall_delta_threshold))
        if cutoff_frame is not None:
            task_targets = task_targets[: cutoff_frame + 1]

    env = LabEnv()
    camera = make_side_camera(args)
    frames: list[np.ndarray] = []
    with mujoco.Renderer(env.model, height=int(args.height), width=int(args.width)) as renderer:
        for index, task_target in enumerate(tqdm(task_targets, desc=f"Rendering {run_dir.name} side view")):
            set_env_to_task_target_continuous(env, task_target, first_frame=index == 0)
            frames.append(render_frame(renderer, env, camera))

    imageio.mimwrite(run_dir / "side_view.mp4", frames, fps=int(args.fps), quality=8, macro_block_size=1)
    imageio.imwrite(run_dir / "side_view_start.png", frames[0])
    imageio.imwrite(run_dir / "side_view_goal.png", frames[-1])
    if cutoff_frame is not None:
        metadata = {
            "trimmed": True,
            "stall_delta_threshold": float(args.stall_delta_threshold),
            "cutoff_frame": int(cutoff_frame),
            "cutoff_time": float(elapsed_time[cutoff_frame]) if elapsed_time is not None else None,
            "saved_frame_count": int(len(frames)),
            "source_frame_count": int(states["task_targets"].shape[0]),
        }
    else:
        metadata = {
            "trimmed": False,
            "saved_frame_count": int(len(frames)),
            "source_frame_count": int(states["task_targets"].shape[0]),
        }
    with (run_dir / "side_view_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote side-view replay for {run_dir}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    for name in args.trajectories:
        render_trajectory(root / name, args)


if __name__ == "__main__":
    main()
