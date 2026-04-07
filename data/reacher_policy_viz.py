#!/usr/bin/env python3
"""Render batched top-down Reacher-v5 rollouts using MJX dynamics."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import torch as th
from huggingface_hub import hf_hub_download
from mujoco import mjx
from stable_baselines3.common.save_util import json_to_data
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import constant_fn


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
INIT_QPOS = np.array([0.0, 0.0, 0.1, -0.1], dtype=np.float32)
INIT_QVEL = np.zeros(4, dtype=np.float32)


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
        "--outdir",
        type=Path,
        default=Path("reacher_videos"),
        help="Directory where the rendered MP4 files will be written.",
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=100,
        help="Number of rollout videos to save.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rollouts to simulate together in one MJX batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Each rollout uses `seed + index`.",
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
        help="MuJoCo frame skip. `1` gives 100 Hz control/render for Reacher-v5.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference: `cuda` by default, or override with `cpu`.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of SAC sampling.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of control steps per rollout.",
    )
    return parser.parse_args()


def resolve_file(local_path: Path | None, repo_id: str, remote_name: str) -> Path:
    if local_path is not None:
        return local_path.expanduser().resolve()
    downloaded = hf_hub_download(repo_id=repo_id, filename=remote_name)
    return Path(downloaded)


def load_metadata(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    metadata = json_to_data(raw)
    if not isinstance(metadata, dict):
        raise TypeError(f"Expected dict metadata in {path}, got {type(metadata)!r}")
    return metadata


def pick_device(device_arg: str) -> th.device:
    if device_arg == "auto":
        device_arg = "cuda"
    if device_arg == "cuda" and not th.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested by default, but no CUDA-enabled PyTorch device is available. "
            "Install a CUDA-enabled PyTorch build or run with --device cpu."
        )
    return th.device(device_arg)


def pick_jax_device(device_arg: str) -> jax.Device:
    if device_arg == "auto":
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            return gpu_devices[0]
        return jax.devices("cpu")[0]

    platform = {"cuda": "gpu", "cpu": "cpu"}.get(device_arg, device_arg)
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(
            f"JAX device '{platform}' was requested, but no matching JAX device is available."
        )
    return devices[0]


def normalize_schedule(schedule: Any) -> Schedule:
    if isinstance(schedule, (int, float, np.floating)):
        return constant_fn(float(schedule))

    if not callable(schedule):
        raise TypeError(f"Expected callable or numeric schedule, got {type(schedule)!r}")

    try:
        float(schedule(1.0))
    except Exception:
        closure = getattr(schedule, "__closure__", None) or ()
        if len(closure) == 1 and callable(closure[0].cell_contents):
            inner = closure[0].cell_contents
            return lambda progress_remaining: float(inner(progress_remaining))
        raise TypeError("Unable to normalize the serialized SB3 schedule") from None

    return lambda progress_remaining: float(schedule(progress_remaining))


def build_policy(metadata: dict[str, Any], device: th.device) -> Any:
    policy_class = metadata["policy_class"]
    observation_space = metadata["observation_space"]
    action_space = metadata["action_space"]
    lr_schedule = normalize_schedule(metadata["lr_schedule"])
    policy_kwargs = metadata.get("policy_kwargs", {})

    policy = policy_class(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **policy_kwargs,
    )
    policy.to(device)
    policy.set_training_mode(False)
    return policy


def load_policy_weights(policy: Any, policy_path: Path, device: th.device) -> None:
    state_dict = th.load(policy_path, map_location=device)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Policy state dict mismatch.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )


def sample_initial_state(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    qpos = INIT_QPOS + rng.uniform(low=-0.1, high=0.1, size=4).astype(np.float32)
    while True:
        goal = rng.uniform(low=-0.2, high=0.2, size=2).astype(np.float32)
        if float(np.linalg.norm(goal)) < 0.2:
            break
    qpos[-2:] = goal
    qvel = INIT_QVEL + rng.uniform(low=-0.005, high=0.005, size=4).astype(np.float32)
    qvel[-2:] = 0.0
    return qpos, qvel


def batched_initial_state(start_seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.empty((count, 4), dtype=np.float32)
    qvel = np.empty((count, 4), dtype=np.float32)
    for offset in range(count):
        qpos[offset], qvel[offset] = sample_initial_state(start_seed + offset)
    return qpos, qvel


def camera_from_config() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.distance = float(TOP_DOWN_CAMERA["distance"])
    camera.azimuth = float(TOP_DOWN_CAMERA["azimuth"])
    camera.elevation = float(TOP_DOWN_CAMERA["elevation"])
    camera.lookat[:] = TOP_DOWN_CAMERA["lookat"]
    return camera


def ensure_offscreen_buffer(model: mujoco.MjModel, width: int, height: int) -> None:
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def get_observation_batch(data: mjx.Data, target_body_id: int, fingertip_body_id: int) -> jp.ndarray:
    return jp.concatenate(
        [
            jp.cos(data.qpos[:, :2]),
            jp.sin(data.qpos[:, :2]),
            data.qpos[:, 2:],
            data.qvel[:, :2],
            data.xpos[:, fingertip_body_id, :2] - data.xpos[:, target_body_id, :2],
        ],
        axis=1,
    )


def simulate_batch(
    policy: Any,
    model: mujoco.MjModel,
    mjx_model: mjx.Model,
    jax_device: jax.Device,
    batch_size: int,
    seed: int,
    deterministic: bool,
    max_steps: int,
    frame_skip: int,
) -> tuple[np.ndarray, float]:
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    qpos0, qvel0 = batched_initial_state(seed, batch_size)

    with jax.default_device(jax_device):
        data = jax.vmap(lambda _: mjx.make_data(model))(jp.arange(batch_size))
        data = data.replace(
            qpos=jp.asarray(qpos0),
            qvel=jp.asarray(qvel0),
            ctrl=jp.zeros((batch_size, model.nu), dtype=jp.float32),
        )

        forward_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)), device=jax_device)
        step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)), device=jax_device)
        data = forward_fn(mjx_model, data)

    trajectories = np.empty((batch_size, max_steps + 1, model.nq), dtype=np.float32)
    trajectories[:, 0, :] = np.asarray(data.qpos)

    start_time = time.perf_counter()
    for step in range(max_steps):
        obs = np.asarray(get_observation_batch(data, target_body_id, fingertip_body_id))
        actions, _ = policy.predict(obs, deterministic=deterministic)
        data = data.replace(ctrl=jp.asarray(actions, dtype=jp.float32))
        for _ in range(frame_skip):
            data = step_fn(mjx_model, data)
        trajectories[:, step + 1, :] = np.asarray(data.qpos)
    elapsed = time.perf_counter() - start_time

    return trajectories, elapsed


def render_video(
    model: mujoco.MjModel,
    qpos_trajectory: np.ndarray,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    camera = camera_from_config()
    ensure_offscreen_buffer(model, width=width, height=height)
    renderer: mujoco.Renderer | None = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames = []
        for qpos in qpos_trajectory:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        if renderer is not None:
            renderer.close()

    imageio.mimwrite(output_path, frames, fps=fps, quality=8)


def main() -> None:
    args = parse_args()
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
    fps = int(round(1.0 / (model.opt.timestep * args.frame_skip)))
    digits = max(2, len(str(args.videos - 1)))

    sim_total = 0.0
    render_total = 0.0
    video_index = 0
    num_batches = math.ceil(args.videos / args.batch_size)

    for batch_idx in range(num_batches):
        current_batch = min(args.batch_size, args.videos - video_index)
        trajectories, sim_elapsed = simulate_batch(
            policy=policy,
            model=model,
            mjx_model=mjx_model,
            jax_device=jax_device,
            batch_size=current_batch,
            seed=args.seed + video_index,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            frame_skip=args.frame_skip,
        )
        sim_total += sim_elapsed

        render_start = time.perf_counter()
        for local_idx in range(current_batch):
            outfile = args.outdir / f"reacher_topdown_{video_index:0{digits}d}.mp4"
            render_video(
                model=model,
                qpos_trajectory=trajectories[local_idx],
                output_path=outfile,
                width=args.width,
                height=args.height,
                fps=fps,
            )
            print(f"saved {outfile}")
            video_index += 1
        render_total += time.perf_counter() - render_start

        print(
            f"batch {batch_idx + 1}/{num_batches}: simulated {current_batch} rollouts "
            f"in {sim_elapsed:.3f}s"
        )

    summary = {
        "policy_file": str(policy_path),
        "data_file": str(data_path),
        "videos": args.videos,
        "batch_size": args.batch_size,
        "outdir": str(args.outdir.resolve()),
        "policy_device": str(device),
        "jax_default_device": str(jax_device),
        "jax_devices": [str(dev) for dev in jax.devices()],
        "simulation_seconds": round(sim_total, 3),
        "render_seconds": round(render_total, 3),
        "total_seconds": round(sim_total + render_total, 3),
        "fps": fps,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
