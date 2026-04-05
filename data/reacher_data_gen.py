#!/usr/bin/env python3
"""Collect Reacher-v5 trajectories first, then render videos from the saved data."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch as th
from tqdm import tqdm


TOP_DOWN_CAMERA = {
    "distance": 0.9,
    "azimuth": 90.0,
    "elevation": -90.0,
    "lookat": np.array([0.0, 0.0, 0.0]),
}

REACHER_XML = (
    "/home/devesh/latent-brs/latent_brs_venv/lib/python3.11/site-packages/"
    "gymnasium/envs/mujoco/assets/reacher.xml"
)


def configure_mujoco_gl(backend: str) -> None:
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend


def import_mujoco() -> Any:
    return importlib.import_module("mujoco")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="farama-minari/Reacher-v5-SAC-expert",
        help="Hugging Face repo containing the SB3 policy artifacts.",
    )
    parser.add_argument(
        "--subdir",
        default="reacher-v5-sac-expert",
        help="Subdirectory inside the repo that contains `policy.pth` and `data`.",
    )
    parser.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="Optional local path to `policy.pth`.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional local path to the SB3 metadata file named `data`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reacher_dataset.pt"),
        help="Output base path. With chunking, chunk indices are inserted into the filename.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10_000,
        help="Number of trajectories to collect.",
    )
    parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=256,
        help="Number of trajectories to simulate together in one MJX batch.",
    )
    parser.add_argument(
        "--physics-steps",
        type=int,
        default=100,
        help="Number of physics steps per trajectory.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="MuJoCo frame skip. `1` gives 100 Hz physics/control/render for Reacher-v5.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device for policy inference: `cuda` by default, or override with `cpu`.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of SAC sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for initial states and goals.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output chunks if they already exist.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory where per-trajectory MP4 files will be written. Defaults to `<output stem>_videos`.",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=8,
        help="ImageIO/ffmpeg quality value for trajectory MP4s.",
    )
    parser.add_argument(
        "--gl-backend",
        default="osmesa",
        choices=("egl", "osmesa", "glfw"),
        help="MuJoCo OpenGL backend used for this run. `osmesa` is the safest headless default.",
    )
    return parser.parse_args()


def sample_goal(rng: np.random.Generator) -> np.ndarray:
    while True:
        goal = rng.uniform(low=-0.2, high=0.2, size=2).astype(np.float32)
        if float(np.linalg.norm(goal)) < 0.2:
            return goal


def make_video_dir(base_path: Path, video_dir: Path | None) -> Path:
    if video_dir is not None:
        return video_dir.resolve()
    return base_path.with_name(f"{base_path.stem}_videos")


def make_trajectory_video_path(video_dir: Path, trajectory_index: int) -> Path:
    return video_dir / f"trajectory_{trajectory_index:07d}.mp4"


def ensure_offscreen_buffer(model: Any, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def render_trajectory(
    mujoco: Any,
    model: Any,
    qpos_trajectory: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = float(TOP_DOWN_CAMERA["distance"])
    camera.azimuth = float(TOP_DOWN_CAMERA["azimuth"])
    camera.elevation = float(TOP_DOWN_CAMERA["elevation"])
    camera.lookat[:] = TOP_DOWN_CAMERA["lookat"]
    ensure_offscreen_buffer(model, width=width, height=height)

    frames = np.empty((qpos_trajectory.shape[0], height, width, 3), dtype=np.uint8)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, qpos in enumerate(qpos_trajectory):
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames[index] = renderer.render()
    return frames


def write_trajectory_video(
    output_path: Path,
    images: np.ndarray,
    *,
    fps: int,
    quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output_path, images, fps=fps, quality=quality)


def build_header(
    *,
    args: argparse.Namespace,
    render_fps: int,
    physics_dt: float,
    control_dt: float,
    policy_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    return {
        "format": "reacher_data_collect_v1",
        "format_version": 4,
        "env_id": "Reacher-v5",
        "policy_file": str(policy_path),
        "data_file": str(data_path),
        "num_trajectories": args.num_trajectories,
        "physics_steps": args.physics_steps,
        "goal_schedule": "single_goal_fixed_horizon",
        "frame_skip": args.frame_skip,
        "physics_dt": physics_dt,
        "control_dt": control_dt,
        "render_fps": render_fps,
        "image_shape": [args.height, args.width, 3],
        "video_quality": int(args.video_quality),
        "gl_backend": args.gl_backend,
        "state_layout": ["qpos[0]", "qpos[1]", "qpos[2]", "qpos[3]", "qvel[0]", "qvel[1]", "qvel[2]", "qvel[3]"],
        "action_layout": ["action[0]", "action[1]"],
        "trajectories_are_streamed": True,
        "videos_are_external": True,
        "record_layout": {
            "dataset_metadata": [
                "num_trajectories",
            ],
            "trajectory_record": [
                "trajectory_index",
                "video_path",
                "video_num_frames",
                "video_fps",
                "qpos",
                "states",
                "actions",
            ],
        },
    }


def save_chunk(
    dataset_path: Path,
    header: dict[str, Any],
    dataset_meta: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("wb") as handle:
        th.save(header, handle, _use_new_zipfile_serialization=False)
        th.save(dataset_meta, handle, _use_new_zipfile_serialization=False)
        for record in records:
            th.save(record, handle, _use_new_zipfile_serialization=False)


def load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        header = th.load(handle, map_location="cpu")
        dataset_meta = th.load(handle, map_location="cpu")
        while True:
            try:
                records.append(th.load(handle, map_location="cpu"))
            except EOFError:
                break
    if not isinstance(header, dict) or not isinstance(dataset_meta, dict):
        raise TypeError(f"Unexpected dataset format in {path}")
    return header, dataset_meta, records


def simulate_dataset(args: argparse.Namespace, mujoco: Any) -> tuple[Path, dict[str, Any], float]:
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    from reacher_policy_viz import (
        build_policy,
        get_observation_batch,
        load_metadata,
        load_policy_weights,
        pick_device,
        pick_jax_device,
        resolve_file,
        sample_initial_state,
    )

    if args.num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if args.sim_batch_size <= 0:
        raise ValueError("sim_batch_size must be positive")

    device = pick_device(args.device)
    jax_device = pick_jax_device(args.device)
    repo_policy_name = f"{args.subdir}/policy.pth"
    repo_data_name = f"{args.subdir}/data"
    policy_path = resolve_file(args.policy_file, args.repo_id, repo_policy_name)
    data_path = resolve_file(args.data_file, args.repo_id, repo_data_name)
    metadata = load_metadata(data_path)
    policy = build_policy(metadata, device=device)
    load_policy_weights(policy, policy_path, device=device)

    model = mujoco.MjModel.from_xml_path(REACHER_XML)
    mjx_model = jax.device_put(mjx.put_model(model), device=jax_device)
    render_fps = int(round(1.0 / (model.opt.timestep * args.frame_skip)))
    physics_dt = float(model.opt.timestep)
    control_dt = float(model.opt.timestep * args.frame_skip)
    header = build_header(
        args=args,
        render_fps=render_fps,
        physics_dt=physics_dt,
        control_dt=control_dt,
        policy_path=policy_path,
        data_path=data_path,
    )

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    sim_total = 0.0
    dataset_path = args.output
    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(f"{dataset_path} already exists; pass --overwrite to replace it")

    records: list[dict[str, Any]] = []
    with tqdm(total=args.num_trajectories, desc="Simulate", dynamic_ncols=True) as progress:
        local_written = 0
        while local_written < args.num_trajectories:
            current_batch = min(args.sim_batch_size, args.num_trajectories - local_written)
            batch_seed = args.seed + local_written

            qpos0 = np.empty((current_batch, 4), dtype=np.float32)
            qvel0 = np.empty((current_batch, 4), dtype=np.float32)
            goal_rngs = [
                np.random.default_rng(batch_seed + 10_000_000 + batch_index)
                for batch_index in range(current_batch)
            ]
            for offset in range(current_batch):
                qpos0[offset], qvel0[offset] = sample_initial_state(batch_seed + offset)
            qpos0[:, -2:] = np.stack([sample_goal(rng) for rng in goal_rngs], axis=0).astype(np.float32)
            qvel0[:, -2:] = 0.0

            with jax.default_device(jax_device):
                data = jax.vmap(lambda _: mjx.make_data(model))(jp.arange(current_batch))
                data = data.replace(
                    qpos=jp.asarray(qpos0),
                    qvel=jp.asarray(qvel0),
                    ctrl=jp.zeros((current_batch, model.nu), dtype=jp.float32),
                )
                forward_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)), device=jax_device)
                step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)), device=jax_device)
                data = forward_fn(mjx_model, data)

            qpos_history = np.empty((current_batch, args.physics_steps + 1, model.nq), dtype=np.float32)
            qvel_history = np.empty((current_batch, args.physics_steps + 1, model.nv), dtype=np.float32)
            action_history = np.empty((current_batch, args.physics_steps, model.nu), dtype=np.float32)
            qpos_history[:, 0] = np.asarray(data.qpos)
            qvel_history[:, 0] = np.asarray(data.qvel)

            sim_start = time.perf_counter()
            for step in range(args.physics_steps):
                obs = np.asarray(get_observation_batch(data, target_body_id, fingertip_body_id))
                actions, _ = policy.predict(obs, deterministic=args.deterministic)
                actions = np.asarray(actions, dtype=np.float32)
                action_history[:, step] = actions
                data = data.replace(ctrl=jp.asarray(actions, dtype=jp.float32))
                for _ in range(args.frame_skip):
                    data = step_fn(mjx_model, data)
                qpos_history[:, step + 1] = np.asarray(data.qpos)
                qvel_history[:, step + 1] = np.asarray(data.qvel)
            sim_total += time.perf_counter() - sim_start

            state_history = np.concatenate([qpos_history, qvel_history], axis=-1)
            for batch_offset in range(current_batch):
                trajectory_index = local_written + batch_offset
                records.append(
                    {
                        "trajectory_index": trajectory_index,
                        "video_path": None,
                        "video_num_frames": None,
                        "video_fps": render_fps,
                        "qpos": th.from_numpy(qpos_history[batch_offset]),
                        "states": th.from_numpy(state_history[batch_offset]),
                        "actions": th.from_numpy(action_history[batch_offset]),
                    }
                )
            local_written += current_batch
            progress.update(current_batch)

    dataset_meta = {
        "num_trajectories": args.num_trajectories,
    }
    save_chunk(dataset_path, header, dataset_meta, records)
    return dataset_path, header, sim_total


def render_dataset(
    args: argparse.Namespace,
    mujoco: Any,
    dataset_path: Path,
    header: dict[str, Any],
) -> float:
    video_dir = make_video_dir(args.output, args.video_dir)
    model = mujoco.MjModel.from_xml_path(REACHER_XML)
    render_total = 0.0

    with tqdm(total=args.num_trajectories, desc="Encode", dynamic_ncols=True) as progress:
        dataset_header, dataset_meta, records = load_dataset(dataset_path)
        if dataset_header["render_fps"] != header["render_fps"]:
            raise RuntimeError(f"Unexpected render FPS mismatch in {dataset_path}")

        for record in records:
            qpos_trajectory = np.asarray(record["qpos"], dtype=np.float32)
            trajectory_index = int(record["trajectory_index"])
            render_start = time.perf_counter()
            images = render_trajectory(
                mujoco,
                model,
                qpos_trajectory,
                width=args.width,
                height=args.height,
            )
            video_path = make_trajectory_video_path(video_dir, trajectory_index)
            write_trajectory_video(
                video_path,
                images,
                fps=int(header["render_fps"]),
                quality=args.video_quality,
            )
            render_total += time.perf_counter() - render_start

            record["video_path"] = os.path.relpath(video_path, start=dataset_path.parent)
            record["video_num_frames"] = int(images.shape[0])
            record["video_fps"] = int(header["render_fps"])
            progress.update(1)

        save_chunk(dataset_path, dataset_header, dataset_meta, records)

    return render_total


def main() -> None:
    args = parse_args()
    if args.video_quality <= 0:
        raise ValueError("video_quality must be positive")

    configure_mujoco_gl(args.gl_backend)
    mujoco = import_mujoco()

    dataset_path, header, sim_total = simulate_dataset(args, mujoco)
    render_total = render_dataset(args, mujoco, dataset_path, header)

    print(
        {
            "num_trajectories": args.num_trajectories,
            "sim_batch_size": args.sim_batch_size,
            "gl_backend": args.gl_backend,
            "output": str(args.output.resolve()),
            "video_dir": str(make_video_dir(args.output, args.video_dir).resolve()),
            "simulation_seconds": round(sim_total, 3),
            "render_seconds": round(render_total, 3),
            "total_seconds": round(sim_total + render_total, 3),
        }
    )


if __name__ == "__main__":
    main()
