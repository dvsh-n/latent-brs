#!/usr/bin/env python3
"""Sample a circle obstacle instance, then train and conformalize a local latent obstacle classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_4"
DEFAULT_BACKGROUND_DATASET_PATH = "reacher/data/train_data_noisy.h5"
DEFAULT_OUT_DIR = "reacher/plan/obs_net"
DEFAULT_OVERLAY_PERTURB_ALPHA = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=32)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inside-sample-count", type=int, default=2048)
    parser.add_argument("--outside-sample-count", type=int, default=2048)
    parser.add_argument("--sampling-budget", dest="sampling_budget", type=int, default=8192)
    parser.add_argument("--ik-sampling-budget", dest="sampling_budget", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--inside-bend-sign", dest="inside_bend_sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--bend-sign", dest="inside_bend_sign", type=int, choices=(-1, 1), default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--circle-center-x", type=float, default=0.135)
    parser.add_argument("--circle-center-y", type=float, default=0.135)
    parser.add_argument("--circle-radius", type=float, default=0.035)
    parser.add_argument("--outside-margin", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)

    parser.add_argument("--joint-range-scale", type=float, default=1.15)
    parser.add_argument("--joint-range-min", type=float, default=0.05)
    parser.add_argument("--background-outside-sample-count", type=int, default=512)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--calibration-frac", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[obstacle_net_v3] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def make_render_env(
    *,
    seed: int,
    time_limit: float,
    width: int,
    height: int,
    physics_freq_hz: float,
) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name="hard",
        seed=seed,
        time_limit=time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env


def reset_env_to_state(
    env: DmControlGymEnv,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_torch_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def preprocess_pixels(
    pixels: np.ndarray | torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        tensor = pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return pixel_mean, pixel_std


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    return torch.cat(latents, dim=0)


@torch.no_grad()
def encode_single_frame(
    model: torch.nn.Module,
    pixel: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std).to(device)
    output = model.encoder(batch, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]


class ObstacleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Expected depth >= 1, got {depth}.")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    n = int(scores.shape[0])
    augmented = np.concatenate((np.sort(scores.astype(np.float64)), np.array([np.inf], dtype=np.float64)))
    rank = int(np.ceil((n + 1) * (1.0 - delta))) - 1
    rank = int(np.clip(rank, 0, augmented.shape[0] - 1))
    return float(augmented[rank])


def render_qpos_batch(
    env: DmControlGymEnv,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    frames = []
    qpos_iter = tqdm(qpos_batch, desc=progress_desc, leave=False) if progress_desc is not None else qpos_batch
    for qpos in qpos_iter:
        frame = reset_env_to_state(
            env,
            seed=seed,
            qpos=np.asarray(qpos, dtype=np.float32),
            qvel=qvel,
            height=height,
            width=width,
        )
        frames.append(frame.copy())
    return np.stack(frames, axis=0)


def sample_background_rows(dataset_path: Path, rng: np.random.Generator, count: int) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    valid_eps = np.flatnonzero(ep_len > 0)
    if valid_eps.size == 0:
        raise ValueError(f"No non-empty episodes in {dataset_path}.")
    ep_choices = rng.choice(valid_eps, size=count, replace=True)
    time_choices = np.array([rng.integers(0, int(ep_len[ep])) for ep in ep_choices], dtype=np.int64)
    return ep_offset[ep_choices] + time_choices


def encode_dataset_rows(
    model: torch.nn.Module,
    dataset_path: Path,
    rows: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    frame_batch_size: int,
) -> np.ndarray:
    rows_sorted_unique, inverse = np.unique(rows, return_inverse=True)
    with h5py.File(dataset_path, "r") as h5:
        pixels_np = np.asarray(h5["pixels"][rows_sorted_unique], dtype=np.uint8)
    pixels = preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    latents = encode_frames(model, pixels, device=device, frame_batch_size=frame_batch_size)
    latents_np = latents.detach().cpu().numpy().astype(np.float64)
    return latents_np[inverse]


def get_arm_geom_ids(model: Any) -> np.ndarray:
    arm_body_names = ("arm", "hand", "finger")
    arm_body_ids = {int(model.name2id(name, "body")) for name in arm_body_names}
    arm_geom_ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        geom_name = model.id2name(geom_id, "geom")
        geom_body_id = int(model.geom_bodyid[geom_id])
        if geom_name == "root" or geom_body_id in arm_body_ids:
            arm_geom_ids.append(geom_id)
    if not arm_geom_ids:
        raise ValueError("Failed to identify arm geoms for segmentation.")
    return np.asarray(sorted(set(arm_geom_ids)), dtype=np.int32)


def build_arm_mask(segmentation: np.ndarray, arm_geom_ids: np.ndarray) -> np.ndarray:
    mask = np.zeros(segmentation.shape[:2], dtype=bool)
    for geom_id in arm_geom_ids:
        mask |= segmentation[..., 0] == geom_id
    return mask


def make_segmentation_scene_option(model: Any) -> tuple[Any, int, int]:
    from dm_control.mujoco.wrapper import core as dm_core

    target_geom_id = int(model.name2id("target", "geom"))
    original_group = int(model.geom_group[target_geom_id])
    model.geom_group[target_geom_id] = 3
    scene_option = dm_core.MjvOption()
    scene_option.geomgroup[:] = 1
    scene_option.geomgroup[3] = 0
    return scene_option, target_geom_id, original_group


def render_masked_qpos_batch(
    env: DmControlGymEnv,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    physics = env._env.physics
    model = physics.model
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    arm_geom_ids = get_arm_geom_ids(model)
    scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    try:
        for qpos in qpos_batch:
            with physics.reset_context():
                physics.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float32)
                physics.data.qvel[: qvel.shape[0]] = qvel
            frame = physics.render(height=height, width=width, camera_id=0)
            segmentation = physics.render(
                height=height,
                width=width,
                camera_id=0,
                segmentation=True,
                scene_option=scene_option,
            )
            frames.append(frame.copy())
            masks.append(build_arm_mask(segmentation, arm_geom_ids))
    finally:
        model.geom_group[target_geom_id] = original_group
    return np.stack(frames, axis=0), np.stack(masks, axis=0)


def alpha_composite_masked(
    canvas: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    out = canvas.copy()
    if not np.any(mask):
        return out
    base = out[mask].astype(np.float32)
    src = frame[mask].astype(np.float32)
    out[mask] = np.clip((1.0 - alpha) * base + alpha * src, 0.0, 255.0).astype(np.uint8)
    return out


def make_obstacle_overlay_image(
    nominal_frame: np.ndarray,
    nominal_mask: np.ndarray,
    perturb_frames: np.ndarray,
    perturb_masks: np.ndarray,
    *,
    perturb_alpha: float,
) -> np.ndarray:
    canvas = np.full_like(nominal_frame, 255, dtype=np.uint8)
    for frame, mask in zip(perturb_frames, perturb_masks, strict=True):
        canvas = alpha_composite_masked(canvas, frame, mask, alpha=perturb_alpha)
    canvas = alpha_composite_masked(canvas, nominal_frame, nominal_mask, alpha=1.0)
    return canvas


def save_obstacle_overlay(
    *,
    rollout: dict[str, Any],
    center_qpos: np.ndarray,
    sample_qpos: np.ndarray,
    out_path: Path,
    perturb_alpha: float = DEFAULT_OVERLAY_PERTURB_ALPHA,
) -> None:
    overlay_qpos_batch = np.concatenate((center_qpos[None, :], sample_qpos), axis=0)
    overlay_env = make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    try:
        overlay_frames, overlay_masks = render_masked_qpos_batch(
            overlay_env,
            int(rollout["episode_seed"]),
            overlay_qpos_batch,
            height=int(rollout["height"]),
            width=int(rollout["width"]),
        )
    finally:
        overlay_env.close()
    obstacle_overlay = make_obstacle_overlay_image(
        nominal_frame=overlay_frames[0],
        nominal_mask=overlay_masks[0],
        perturb_frames=overlay_frames[1:],
        perturb_masks=overlay_masks[1:],
        perturb_alpha=float(perturb_alpha),
    )
    save_rgb_image(out_path, obstacle_overlay)


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def get_descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    descendants = {int(root_body_id)}
    parent_ids = np.asarray(model.body_parentid, dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            parent_id = int(parent_ids[body_id])
            if parent_id in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def infer_planar_arm_geometry(env: Any) -> dict[str, Any]:
    physics = env._env.physics
    model = physics.model
    with physics.reset_context():
        physics.data.qpos[:2] = 0.0
        physics.data.qvel[:2] = 0.0

    arm_body_id = int(model.name2id("arm", "body"))
    hand_body_id = int(model.name2id("hand", "body"))
    base_xy = np.asarray(physics.data.xpos[arm_body_id][:2], dtype=np.float64)
    hand_xy = np.asarray(physics.data.xpos[hand_body_id][:2], dtype=np.float64)
    hand_xy_local = hand_xy - base_xy
    link1 = float(np.linalg.norm(hand_xy_local))

    hand_descendants = get_descendant_body_ids(model, hand_body_id)
    candidates: list[tuple[float, np.ndarray, str]] = []

    for site_id in range(int(model.nsite)):
        name = model.id2name(site_id, "site")
        if name is None or "target" in name:
            continue
        if int(model.site_bodyid[site_id]) not in hand_descendants:
            continue
        xy_world = np.asarray(physics.data.site_xpos[site_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"site:{name}"))

    for geom_id in range(int(model.ngeom)):
        name = model.id2name(geom_id, "geom")
        if name is None or name in {"target", "root"}:
            continue
        if int(model.geom_bodyid[geom_id]) not in hand_descendants:
            continue
        xy_world = np.asarray(physics.data.geom_xpos[geom_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"geom:{name}"))

    for body_id in hand_descendants:
        xy_world = np.asarray(physics.data.xpos[body_id][:2], dtype=np.float64)
        xy_local = xy_world - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"body:{model.id2name(body_id, 'body')}"))

    if not candidates:
        raise RuntimeError("Failed to infer fingertip geometry from the DM Control reacher model.")

    link2, tip_xy, tip_source = max(candidates, key=lambda item: item[0])
    reach_min = abs(link1 - link2)
    reach_max = link1 + link2
    return {
        "base_xy": base_xy,
        "link1": link1,
        "link2": link2,
        "reach_min": reach_min,
        "reach_max": reach_max,
        "tip_source": tip_source,
        "tip_xy_at_zero": tip_xy,
    }


def resolve_circle_spec(
    geom: dict[str, Any],
    *,
    circle_center_x: float | None,
    circle_center_y: float | None,
    circle_radius: float | None,
) -> tuple[np.ndarray, float]:
    reach_min = float(geom["reach_min"])
    reach_max = float(geom["reach_max"])
    annulus_width = max(reach_max - reach_min, 1e-6)
    radius = float(circle_radius) if circle_radius is not None else min(0.03, 0.12 * annulus_width)
    radius = max(radius, 1e-4)

    default_center_radius = reach_min + 0.72 * max(reach_max - reach_min - radius, 1e-6)
    default_center_local = default_center_radius * np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0)
    center_local = np.array(
        [
            default_center_local[0] if circle_center_x is None else float(circle_center_x),
            default_center_local[1] if circle_center_y is None else float(circle_center_y),
        ],
        dtype=np.float64,
    )

    center_norm = float(np.linalg.norm(center_local))
    min_center_norm = reach_min + radius + 1e-4
    max_center_norm = reach_max - radius - 1e-4
    if center_norm < 1e-9:
        center_local = default_center_local.copy()
        center_norm = float(np.linalg.norm(center_local))
    if center_norm < min_center_norm:
        center_local *= min_center_norm / center_norm
    if center_norm > max_center_norm:
        center_local *= max_center_norm / center_norm
    if center_local[0] <= 0.0 or center_local[1] <= 0.0:
        center_local = np.abs(center_local)
        center_norm = float(np.linalg.norm(center_local))
        if center_norm > max_center_norm:
            center_local *= max_center_norm / center_norm
    center_world = np.asarray(geom["base_xy"], dtype=np.float64) + center_local
    return center_world, radius


def sample_points_in_circle(rng: np.random.Generator, center_xy: np.ndarray, radius: float, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=count)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    offsets = np.stack((radial * np.cos(theta), radial * np.sin(theta)), axis=1)
    return center_xy[None, :] + offsets


def solve_two_link_ik(
    target_xy: np.ndarray,
    *,
    link1: float,
    link2: float,
    bend_sign: int,
) -> np.ndarray:
    x = float(target_xy[0])
    y = float(target_xy[1])
    radius_sq = x * x + y * y
    cos_theta2 = (radius_sq - link1 * link1 - link2 * link2) / (2.0 * link1 * link2)
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    sin_theta2 = float(bend_sign) * math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = math.atan2(y, x) - math.atan2(link2 * sin_theta2, link1 + link2 * cos_theta2)
    return np.array([wrap_angle(theta1), wrap_angle(theta2)], dtype=np.float64)


def qpos_within_limits(qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-6) -> bool:
    if np.any(np.isfinite(lower) & (qpos < lower - tol)):
        return False
    if np.any(np.isfinite(upper) & (qpos > upper + tol)):
        return False
    return True


def joint_limits_with_unbounded_fixed(env: Any) -> tuple[np.ndarray, np.ndarray]:
    model = env._env.physics.model
    raw = np.asarray(model.jnt_range[:2], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[:2], dtype=bool)
    lower = raw[:, 0].copy()
    upper = raw[:, 1].copy()
    lower[~limited] = -np.inf
    upper[~limited] = np.inf
    return lower, upper


def forward_kinematics(qpos: np.ndarray, *, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray]:
    theta1 = float(qpos[0])
    theta2 = float(qpos[1])
    elbow_xy = np.array([link1 * math.cos(theta1), link1 * math.sin(theta1)], dtype=np.float64)
    tip_xy = elbow_xy + np.array(
        [link2 * math.cos(theta1 + theta2), link2 * math.sin(theta1 + theta2)],
        dtype=np.float64,
    )
    return elbow_xy, tip_xy


def solve_ik_batch(
    targets_xy: np.ndarray,
    *,
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_list: list[np.ndarray] = []
    elbow_list: list[np.ndarray] = []
    tip_list: list[np.ndarray] = []
    for target_xy_world in targets_xy:
        target_xy_local = np.asarray(target_xy_world, dtype=np.float64) - base_xy
        qpos = solve_two_link_ik(target_xy_local, link1=link1, link2=link2, bend_sign=bend_sign)
        if not qpos_within_limits(qpos, lower, upper):
            continue
        elbow_xy_local, tip_xy_local = forward_kinematics(qpos, link1=link1, link2=link2)
        if np.linalg.norm(tip_xy_local - target_xy_local) > 1e-5:
            continue
        elbow_xy = base_xy + elbow_xy_local
        tip_xy = base_xy + tip_xy_local
        qpos_list.append(qpos)
        elbow_list.append(elbow_xy)
        tip_list.append(tip_xy)
    if not qpos_list:
        raise RuntimeError("No valid IK solutions survived the joint-limit and reconstruction checks.")
    return np.stack(qpos_list, axis=0), np.stack(elbow_list, axis=0), np.stack(tip_list, axis=0)


def save_workspace_plot(
    *,
    out_path: Path,
    geom: dict[str, Any],
    circle_center_xy: np.ndarray,
    circle_radius: float,
    center_qpos: np.ndarray,
    sampled_elbows_xy: np.ndarray,
    sampled_tips_xy: np.ndarray,
    outside_elbows_xy: np.ndarray,
    outside_tips_xy: np.ndarray,
    inside_bend_sign: int,
) -> None:
    base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)

    reach_max = float(geom["reach_max"])
    reach_min = float(geom["reach_min"])
    outer = plt.Circle(base_xy, reach_max, color="#d9d9d9", fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(outer)
    if reach_min > 1e-4:
        inner = plt.Circle(base_xy, reach_min, color="#ececec", fill=False, linestyle=":", linewidth=1.0)
        ax.add_patch(inner)

    obstacle = plt.Circle(circle_center_xy, circle_radius, color="#d55e00", fill=False, linewidth=2.0)
    ax.add_patch(obstacle)

    for elbow_point, tip_point in zip(sampled_elbows_xy, sampled_tips_xy, strict=True):
        ax.plot([base_xy[0], elbow_point[0]], [base_xy[1], elbow_point[1]], color="#111111", linewidth=0.8, alpha=0.12)
        ax.plot([elbow_point[0], tip_point[0]], [elbow_point[1], tip_point[1]], color="#111111", linewidth=0.8, alpha=0.12)

    outside_colors = ("#009e73", "#0072b2")
    for idx, (elbow_point, tip_point) in enumerate(zip(outside_elbows_xy, outside_tips_xy, strict=True)):
        color = outside_colors[idx % len(outside_colors)]
        ax.plot([base_xy[0], elbow_point[0]], [base_xy[1], elbow_point[1]], color=color, linewidth=1.2, alpha=0.6)
        ax.plot([elbow_point[0], tip_point[0]], [elbow_point[1], tip_point[1]], color=color, linewidth=1.2, alpha=0.6)

    ax.scatter(sampled_tips_xy[:, 0], sampled_tips_xy[:, 1], s=10, c="#d55e00", alpha=0.5, label="inside tips")
    ax.scatter(sampled_elbows_xy[:, 0], sampled_elbows_xy[:, 1], s=10, c="#0072b2", alpha=0.35, label="inside elbows")
    if outside_tips_xy.size > 0:
        ax.scatter(outside_tips_xy[:, 0], outside_tips_xy[:, 1], s=48, marker="x", c="#009e73", linewidths=1.5, label="outside tips")
    if outside_elbows_xy.size > 0:
        ax.scatter(outside_elbows_xy[:, 0], outside_elbows_xy[:, 1], s=28, marker="D", c="#56b4e9", alpha=0.8, label="outside elbows")

    center_elbow_xy_local, center_tip_xy_local = forward_kinematics(
        center_qpos,
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
    )
    center_elbow_xy = base_xy + center_elbow_xy_local
    center_tip_xy = base_xy + center_tip_xy_local
    ax.plot([base_xy[0], center_elbow_xy[0]], [base_xy[1], center_elbow_xy[1]], color="#cc79a7", linewidth=2.0)
    ax.plot([center_elbow_xy[0], center_tip_xy[0]], [center_elbow_xy[1], center_tip_xy[1]], color="#cc79a7", linewidth=2.0)
    ax.scatter([base_xy[0]], [base_xy[1]], s=32, c="#000000")
    ax.scatter([circle_center_xy[0]], [circle_center_xy[1]], s=20, c="#d55e00")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Circle obstacle workspace samples (inside_bend_sign={inside_bend_sign:+d})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


@dataclass
class ObstacleNetV3Config:
    model_dir: str
    checkpoint_path: str
    background_dataset_path: str
    seed: int
    inside_sample_count: int
    outside_sample_count: int
    sampling_budget: int
    inside_bend_sign: int
    circle_center_x: float
    circle_center_y: float
    circle_radius: float
    outside_margin: float
    time_limit: float
    physics_freq_hz: float
    width: int
    height: int
    joint_range_scale: float
    joint_range_min: float
    background_outside_sample_count: int
    hidden_dim: int
    depth: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    margin: float
    delta: float
    calibration_frac: float
    embed_dim: int


@dataclass(frozen=True)
class ArtifactPaths:
    cache_dir: Path
    summary: Path
    model: Path
    splits: Path
    planner_payload: Path
    obstacle_overlay: Path
    outside_overlay: Path
    workspace_plot: Path

    @property
    def required_for_cache_hit(self) -> tuple[Path, ...]:
        return (self.summary, self.model, self.splits, self.planner_payload)


def build_run_config(
    args: argparse.Namespace,
    *,
    model_dir: Path,
    checkpoint_path: Path,
    background_dataset_path: Path,
    embed_dim: int,
) -> ObstacleNetV3Config:
    return ObstacleNetV3Config(
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        background_dataset_path=str(background_dataset_path),
        seed=int(args.seed),
        inside_sample_count=int(args.inside_sample_count),
        outside_sample_count=int(args.outside_sample_count),
        sampling_budget=int(args.sampling_budget),
        inside_bend_sign=int(args.inside_bend_sign),
        circle_center_x=float(args.circle_center_x),
        circle_center_y=float(args.circle_center_y),
        circle_radius=float(args.circle_radius),
        outside_margin=float(args.outside_margin),
        time_limit=float(args.time_limit),
        physics_freq_hz=float(args.physics_freq_hz),
        width=int(args.width),
        height=int(args.height),
        joint_range_scale=float(args.joint_range_scale),
        joint_range_min=float(args.joint_range_min),
        background_outside_sample_count=int(args.background_outside_sample_count),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        margin=float(args.margin),
        delta=float(args.delta),
        calibration_frac=float(args.calibration_frac),
        embed_dim=int(embed_dim),
    )


def cache_key_for_config(config: ObstacleNetV3Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def infer_cache_dir(out_root: Path, config: ObstacleNetV3Config) -> Path:
    return out_root / cache_key_for_config(config)


def artifact_paths(out_root: Path, config: ObstacleNetV3Config) -> ArtifactPaths:
    cache_dir = infer_cache_dir(out_root, config)
    return ArtifactPaths(
        cache_dir=cache_dir,
        summary=cache_dir / "summary.json",
        model=cache_dir / "model.pt",
        splits=cache_dir / "splits.pt",
        planner_payload=cache_dir / "planner_start_goal_obstacle.pt",
        obstacle_overlay=cache_dir / "obstacle_overlay.png",
        outside_overlay=cache_dir / "outside_overlay.png",
        workspace_plot=cache_dir / "workspace_samples.png",
    )


def infer_joint_ranges(
    center_qpos: np.ndarray,
    obstacle_qpos: np.ndarray,
    *,
    range_scale: float,
    range_min: float,
) -> np.ndarray:
    deltas = np.abs(np.asarray(obstacle_qpos, dtype=np.float64) - np.asarray(center_qpos, dtype=np.float64)[None, :])
    inferred = np.max(deltas, axis=0) if deltas.size > 0 else np.zeros_like(center_qpos, dtype=np.float64)
    return np.maximum(inferred * float(range_scale), float(range_min)).astype(np.float64)


def hinge_loss(scores: torch.Tensor, labels: torch.Tensor, *, margin: float) -> torch.Tensor:
    return torch.clamp(float(margin) - labels * scores, min=0.0).mean()


def compute_signed_metrics(scores: torch.Tensor, labels: torch.Tensor, *, threshold: float = 0.0) -> dict[str, float]:
    preds_obstacle = scores <= float(threshold)
    labels_obstacle = labels < 0.0
    accuracy = float((preds_obstacle == labels_obstacle).float().mean().item())
    pos_mask = labels_obstacle
    neg_mask = ~pos_mask
    recall = float(preds_obstacle[pos_mask].float().mean().item()) if torch.any(pos_mask) else 0.0
    specificity = float((~preds_obstacle[neg_mask]).float().mean().item()) if torch.any(neg_mask) else 0.0
    tp = float(torch.sum(preds_obstacle & pos_mask).item())
    fp = float(torch.sum(preds_obstacle & neg_mask).item())
    precision = tp / max(tp + fp, 1.0)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
    }


def evaluate_signed_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    margin: float,
) -> dict[str, Any]:
    model.eval()
    score_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            score_chunks.append(model(xb).cpu())
    scores = torch.cat(score_chunks, dim=0)
    loss = float(hinge_loss(scores, y, margin=margin).item())
    metrics = compute_signed_metrics(scores, y, threshold=0.0)
    return {
        "loss": loss,
        "scores": scores.numpy().astype(np.float32),
        **metrics,
    }


def compute_conformal_score_threshold(obstacle_nonconformity_cal: np.ndarray, delta: float) -> dict[str, float]:
    if obstacle_nonconformity_cal.ndim != 1:
        raise ValueError(
            f"Expected 1D obstacle calibration nonconformity scores, got shape {obstacle_nonconformity_cal.shape}."
        )
    if obstacle_nonconformity_cal.size == 0:
        raise ValueError("Need at least one obstacle calibration sample for conformal calibration.")
    threshold = conformal_quantile(obstacle_nonconformity_cal.astype(np.float64), delta=delta)
    return {
        "threshold": float(threshold),
        "score_quantile": float(threshold),
        "num_obstacle_calibration": int(obstacle_nonconformity_cal.size),
    }


def score_threshold_metrics(eval_payload: dict[str, Any], labels: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(eval_payload["scores"], dtype=np.float64)
    preds_obstacle = scores <= threshold
    pos_mask = labels < 0.0
    neg_mask = ~pos_mask
    obstacle_coverage = float(np.mean(preds_obstacle[pos_mask])) if np.any(pos_mask) else 0.0
    outside_activation_rate = float(np.mean(preds_obstacle[neg_mask])) if np.any(neg_mask) else 0.0
    return {
        "obstacle_coverage": obstacle_coverage,
        "outside_activation_rate": outside_activation_rate,
        "threshold": float(threshold),
    }


def collect_valid_ik_samples(
    *,
    count: int,
    target_sampler: Callable[[int], np.ndarray],
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
    sampling_budget: int,
    max_attempts: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sampling_budget < count:
        raise ValueError(f"{label} sampling budget ({sampling_budget}) must be at least requested count ({count}).")

    qpos_chunks: list[np.ndarray] = []
    elbow_chunks: list[np.ndarray] = []
    tip_chunks: list[np.ndarray] = []
    remaining = int(count)
    attempts = 0
    while remaining > 0 and attempts < max_attempts:
        batch_count = remaining
        targets_xy = target_sampler(batch_count)
        qpos_batch, elbow_xy, tip_xy = solve_ik_batch(
            targets_xy,
            base_xy=base_xy,
            link1=link1,
            link2=link2,
            bend_sign=bend_sign,
            lower=lower,
            upper=upper,
        )
        if qpos_batch.shape[0] > 0:
            take = min(remaining, qpos_batch.shape[0])
            qpos_chunks.append(qpos_batch[:take])
            elbow_chunks.append(elbow_xy[:take])
            tip_chunks.append(tip_xy[:take])
            remaining -= take
        attempts += 1
    if remaining > 0:
        raise RuntimeError(f"Failed to collect {count} valid {label} IK samples; missing {remaining}.")
    return (
        np.concatenate(qpos_chunks, axis=0),
        np.concatenate(elbow_chunks, axis=0),
        np.concatenate(tip_chunks, axis=0),
    )


def sample_valid_inside_qpos(
    rng: np.random.Generator,
    *,
    count: int,
    circle_center_xy: np.ndarray,
    circle_radius: float,
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
    sampling_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return collect_valid_ik_samples(
        count=count,
        target_sampler=lambda batch_count: sample_points_in_circle(rng, circle_center_xy, circle_radius, batch_count),
        base_xy=base_xy,
        link1=link1,
        link2=link2,
        bend_sign=bend_sign,
        lower=lower,
        upper=upper,
        sampling_budget=sampling_budget,
        max_attempts=32,
        label="inside-circle",
    )


def qpos_sampling_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_lower = np.asarray(lower, dtype=np.float64).copy()
    sample_upper = np.asarray(upper, dtype=np.float64).copy()
    unbounded = ~np.isfinite(sample_lower) | ~np.isfinite(sample_upper)
    sample_lower[unbounded] = -math.pi
    sample_upper[unbounded] = math.pi
    return sample_lower, sample_upper


def sample_valid_outside_qpos(
    rng: np.random.Generator,
    *,
    count: int,
    circle_center_xy: np.ndarray,
    circle_radius: float,
    base_xy: np.ndarray,
    outside_margin: float,
    link1: float,
    link2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    sampling_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count <= 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )

    sample_lower, sample_upper = qpos_sampling_bounds(lower, upper)
    min_tip_distance = float(circle_radius) + max(float(outside_margin), 1e-4)
    qpos_chunks: list[np.ndarray] = []
    elbow_chunks: list[np.ndarray] = []
    tip_chunks: list[np.ndarray] = []
    remaining = int(count)
    draws = 0
    while remaining > 0 and draws < sampling_budget:
        batch_count = min(max(count, 1024), sampling_budget - draws)
        qpos_batch = rng.uniform(sample_lower, sample_upper, size=(batch_count, sample_lower.shape[0]))
        draws += batch_count

        elbow_local = np.stack(
            (
                link1 * np.cos(qpos_batch[:, 0]),
                link1 * np.sin(qpos_batch[:, 0]),
            ),
            axis=1,
        )
        tip_local = elbow_local + np.stack(
            (
                link2 * np.cos(qpos_batch[:, 0] + qpos_batch[:, 1]),
                link2 * np.sin(qpos_batch[:, 0] + qpos_batch[:, 1]),
            ),
            axis=1,
        )
        elbow_xy = base_xy[None, :] + elbow_local
        tip_xy = base_xy[None, :] + tip_local
        outside_mask = np.linalg.norm(tip_xy - circle_center_xy[None, :], axis=1) >= min_tip_distance
        if np.any(outside_mask):
            take = min(remaining, int(np.sum(outside_mask)))
            qpos_chunks.append(qpos_batch[outside_mask][:take])
            elbow_chunks.append(elbow_xy[outside_mask][:take])
            tip_chunks.append(tip_xy[outside_mask][:take])
            remaining -= take

    if remaining > 0:
        raise RuntimeError(
            f"Failed to collect {count} outside-circle state samples after {draws} qpos draws; missing {remaining}."
        )
    return (
        np.concatenate(qpos_chunks, axis=0),
        np.concatenate(elbow_chunks, axis=0),
        np.concatenate(tip_chunks, axis=0),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    config = load_config(model_dir)
    embed_dim = int(config.get("embed_dim", 24))

    if args.background_dataset_path is not None:
        background_dataset_path = args.background_dataset_path.expanduser().resolve()
    else:
        background_dataset_path = Path(DEFAULT_BACKGROUND_DATASET_PATH).expanduser().resolve()

    run_config = build_run_config(
        args,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        background_dataset_path=background_dataset_path,
        embed_dim=embed_dim,
    )
    out_root = args.out_dir.expanduser().resolve()
    paths = artifact_paths(out_root, run_config)
    out_root.mkdir(parents=True, exist_ok=True)

    if all(path.is_file() for path in paths.required_for_cache_hit) and not args.force_retrain:
        log_progress(f"Using cached artifact at {paths.cache_dir}.")
        print(f"Cache dir:  {paths.cache_dir}")
        print(f"Model path: {paths.model}")
        return

    log_progress("Sampling circle obstacle geometry and IK configurations.")
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        geom = infer_planar_arm_geometry(env)
        lower, upper = joint_limits_with_unbounded_fixed(env)
        circle_center_xy, circle_radius = resolve_circle_spec(
            geom,
            circle_center_x=args.circle_center_x,
            circle_center_y=args.circle_center_y,
            circle_radius=args.circle_radius,
        )
        inside_qpos, inside_elbow_xy, inside_tip_xy = sample_valid_inside_qpos(
            rng,
            count=int(args.inside_sample_count),
            circle_center_xy=circle_center_xy,
            circle_radius=float(circle_radius),
            base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            bend_sign=int(args.inside_bend_sign),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
        )
        outside_qpos, outside_elbow_xy, outside_tip_xy = sample_valid_outside_qpos(
            rng,
            count=int(args.outside_sample_count),
            circle_center_xy=circle_center_xy,
            circle_radius=float(circle_radius),
            base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
            outside_margin=float(args.outside_margin),
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
        )
        validation_frac = 0.10
        val_inside_count = max(1, int(math.ceil(validation_frac * inside_qpos.shape[0])))
        val_outside_count = max(1, int(math.ceil(validation_frac * outside_qpos.shape[0])))
        val_inside_qpos, val_inside_elbow_xy, val_inside_tip_xy = sample_valid_inside_qpos(
            rng,
            count=val_inside_count,
            circle_center_xy=circle_center_xy,
            circle_radius=float(circle_radius),
            base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            bend_sign=int(args.inside_bend_sign),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
        )
        val_outside_qpos, val_outside_elbow_xy, val_outside_tip_xy = sample_valid_outside_qpos(
            rng,
            count=val_outside_count,
            circle_center_xy=circle_center_xy,
            circle_radius=float(circle_radius),
            base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
            outside_margin=float(args.outside_margin),
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
        )
    finally:
        env.close()

    if outside_qpos.shape[0] < 2:
        raise RuntimeError("Need at least two outside configurations to define start and goal.")

    start_qpos = np.asarray(outside_qpos[0], dtype=np.float32)
    goal_qpos = np.asarray(outside_qpos[1], dtype=np.float32)
    start_qvel = np.zeros_like(start_qpos, dtype=np.float32)
    goal_qvel = np.zeros_like(goal_qpos, dtype=np.float32)
    start_tip_xy = np.asarray(outside_tip_xy[0], dtype=np.float32)
    goal_tip_xy = np.asarray(outside_tip_xy[1], dtype=np.float32)
    base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
    circle_center_local = circle_center_xy - base_xy
    center_qpos = solve_two_link_ik(
        circle_center_local,
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.inside_bend_sign),
    )
    if not qpos_within_limits(center_qpos, lower, upper):
        raise RuntimeError("Circle-center IK solution violates joint limits.")

    rollout_stub = {
        "episode_seed": int(args.seed),
        "time_limit": float(args.time_limit),
        "width": int(args.width),
        "height": int(args.height),
        "physics_freq_hz": float(args.physics_freq_hz),
    }

    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    save_obstacle_overlay(
        rollout=rollout_stub,
        center_qpos=center_qpos,
        sample_qpos=inside_qpos,
        out_path=paths.obstacle_overlay,
    )
    save_obstacle_overlay(
        rollout=rollout_stub,
        center_qpos=center_qpos,
        sample_qpos=outside_qpos,
        out_path=paths.outside_overlay,
    )
    save_workspace_plot(
        out_path=paths.workspace_plot,
        geom=geom,
        circle_center_xy=circle_center_xy,
        circle_radius=float(circle_radius),
        center_qpos=center_qpos,
        sampled_elbows_xy=inside_elbow_xy,
        sampled_tips_xy=inside_tip_xy,
        outside_elbows_xy=outside_elbow_xy,
        outside_tips_xy=outside_tip_xy,
        inside_bend_sign=int(args.inside_bend_sign),
    )

    save_torch_payload(
        paths.planner_payload,
        {
            "metadata": {
                "episode_seed": int(args.seed),
                "time_limit": float(args.time_limit),
                "physics_freq_hz": float(args.physics_freq_hz),
                "height": int(args.height),
                "width": int(args.width),
                "inside_bend_sign": int(args.inside_bend_sign),
                "start_label": "green arm / start pos",
                "goal_label": "blue arm / goal pos",
                "inside_sample_count_valid": int(inside_qpos.shape[0]),
                "outside_sample_count_valid": int(outside_qpos.shape[0]),
                "validation_inside_sample_count": int(val_inside_qpos.shape[0]),
                "validation_outside_sample_count": int(val_outside_qpos.shape[0]),
                "validation_sample_frac": float(validation_frac),
                "sampling_budget": int(args.sampling_budget),
            },
            "episode_data": {
                "qpos": np.stack((start_qpos, goal_qpos), axis=0),
                "qvel": np.stack((start_qvel, goal_qvel), axis=0),
            },
            "planner_data": {
                "start_qpos": start_qpos,
                "goal_qpos": goal_qpos,
                "obstacle_center_qpos": np.asarray(center_qpos, dtype=np.float32),
                "obstacle_qpos": np.asarray(inside_qpos, dtype=np.float32),
                "outside_qpos": np.asarray(outside_qpos, dtype=np.float32),
                "validation_obstacle_qpos": np.asarray(val_inside_qpos, dtype=np.float32),
                "validation_outside_qpos": np.asarray(val_outside_qpos, dtype=np.float32),
                "start_tip_xy": start_tip_xy,
                "goal_tip_xy": goal_tip_xy,
                "circle_center_xy": np.asarray(circle_center_xy, dtype=np.float32),
                "circle_radius": float(circle_radius),
            },
        },
    )

    sampling_summary = {
        "seed": int(args.seed),
        "inside_sample_count_requested": int(args.inside_sample_count),
        "inside_sample_count_valid": int(inside_qpos.shape[0]),
        "outside_sample_count_requested": int(args.outside_sample_count),
        "outside_sample_count_valid": int(outside_qpos.shape[0]),
        "validation_sample_frac": float(validation_frac),
        "validation_inside_sample_count": int(val_inside_qpos.shape[0]),
        "validation_outside_sample_count": int(val_outside_qpos.shape[0]),
        "sampling_budget": int(args.sampling_budget),
        "outside_margin": float(args.outside_margin),
        "inside_bend_sign": int(args.inside_bend_sign),
        "circle_center_xy": circle_center_xy.tolist(),
        "circle_radius": float(circle_radius),
        "link1": float(geom["link1"]),
        "link2": float(geom["link2"]),
        "reach_min": float(geom["reach_min"]),
        "reach_max": float(geom["reach_max"]),
        "tip_source": str(geom["tip_source"]),
        "center_qpos": center_qpos.tolist(),
        "start_qpos": start_qpos.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "start_tip_xy": start_tip_xy.tolist(),
        "goal_tip_xy": goal_tip_xy.tolist(),
        "outside_tip_xy": outside_tip_xy.tolist(),
        "outside_qpos": outside_qpos.tolist(),
        "validation_inside_tip_xy": val_inside_tip_xy.tolist(),
        "validation_inside_qpos": val_inside_qpos.tolist(),
        "validation_outside_tip_xy": val_outside_tip_xy.tolist(),
        "validation_outside_qpos": val_outside_qpos.tolist(),
    }
    cal_obstacle_count = max(1, int(math.ceil(float(args.calibration_frac) * inside_qpos.shape[0])))
    cal_inside_qpos, _, _ = sample_valid_inside_qpos(
        rng,
        count=cal_obstacle_count,
        circle_center_xy=circle_center_xy,
        circle_radius=float(circle_radius),
        base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.inside_bend_sign),
        lower=lower,
        upper=upper,
        sampling_budget=int(args.sampling_budget),
    )
    sampling_summary["calibration_obstacle_sample_count"] = int(cal_inside_qpos.shape[0])

    log_progress("Rendering and encoding train, validation, and calibration states.")
    world_model = load_model(checkpoint_path, device)
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    img_size = int(config.get("img_size", 224))

    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    nominal_frame = reset_env_to_state(
        env,
        seed=int(args.seed),
        qpos=np.asarray(center_qpos, dtype=np.float32),
        qvel=np.zeros_like(np.asarray(center_qpos, dtype=np.float32)),
        height=int(args.height),
        width=int(args.width),
    )
    nominal_latent = encode_single_frame(
        world_model,
        nominal_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    ).detach().cpu().numpy().astype(np.float64)[:embed_dim]

    qpos_batch = np.concatenate((inside_qpos, outside_qpos, val_inside_qpos, val_outside_qpos, cal_inside_qpos), axis=0)
    local_frames = render_qpos_batch(
        env,
        int(args.seed),
        qpos_batch,
        height=int(args.height),
        width=int(args.width),
        progress_desc="Rendering train/validation/calibration states",
    )
    env.close()

    local_pixels = preprocess_pixels(
        local_frames,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    local_emb = encode_frames(
        world_model,
        local_pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    ).detach().cpu().numpy().astype(np.float64)
    obstacle_latents = local_emb[: inside_qpos.shape[0], :embed_dim]
    outside_start = inside_qpos.shape[0]
    outside_end = outside_start + outside_qpos.shape[0]
    val_inside_start = outside_end
    val_inside_end = val_inside_start + val_inside_qpos.shape[0]
    val_outside_start = val_inside_end
    val_outside_end = val_outside_start + val_outside_qpos.shape[0]
    cal_start = val_outside_end
    outside_latents = local_emb[outside_start:outside_end, :embed_dim]
    val_inside_latents = local_emb[val_inside_start:val_inside_end, :embed_dim]
    val_outside_latents = local_emb[val_outside_start:val_outside_end, :embed_dim]
    cal_obstacle_latents = local_emb[cal_start:, :embed_dim]

    background_latents = np.zeros((0, embed_dim), dtype=np.float64)
    if args.background_outside_sample_count > 0:
        log_progress(f"Sampling {args.background_outside_sample_count} background outside states from dataset latents.")
        background_rows = sample_background_rows(
            background_dataset_path,
            rng,
            int(args.background_outside_sample_count),
        )
        background_emb = encode_dataset_rows(
            world_model,
            background_dataset_path,
            background_rows,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            frame_batch_size=args.frame_batch_size,
        )
        background_latents = background_emb[:, :embed_dim]
    else:
        background_rows = np.zeros((0,), dtype=np.int64)

    x_obstacle = obstacle_latents.astype(np.float32)
    x_outside = np.concatenate((outside_latents, background_latents), axis=0).astype(np.float32)
    y_obstacle = -np.ones((x_obstacle.shape[0],), dtype=np.float32)
    y_outside = np.ones((x_outside.shape[0],), dtype=np.float32)
    x_all = np.concatenate((x_obstacle, x_outside), axis=0).astype(np.float32)
    y_all_signed = np.concatenate((y_obstacle, y_outside), axis=0).astype(np.float32)

    x_tensor = torch.from_numpy(x_all)
    y_tensor = torch.from_numpy(y_all_signed)
    full_dataset = TensorDataset(x_tensor, y_tensor)
    train_x = x_tensor
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0).clamp_min(1e-6)

    def normalize_features(x: torch.Tensor) -> torch.Tensor:
        return (x - train_mean) / train_std

    normalized = normalize_features(x_tensor)
    x_val = np.concatenate((val_inside_latents, val_outside_latents), axis=0).astype(np.float32)
    y_val_signed = np.concatenate(
        (
            -np.ones((val_inside_latents.shape[0],), dtype=np.float32),
            np.ones((val_outside_latents.shape[0],), dtype=np.float32),
        ),
        axis=0,
    )
    val_x_tensor = torch.from_numpy(x_val)
    val_y_tensor = torch.from_numpy(y_val_signed)
    normalized_val = normalize_features(val_x_tensor)
    normalized_cal_obstacle = normalize_features(torch.from_numpy(cal_obstacle_latents.astype(np.float32)))
    train_ds = TensorDataset(normalized, y_tensor)
    val_ds = TensorDataset(normalized_val, val_y_tensor)
    cal_y_tensor = -torch.ones((normalized_cal_obstacle.shape[0],), dtype=torch.float32)
    cal_ds = TensorDataset(normalized_cal_obstacle, cal_y_tensor)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(args.batch_size), max(1, len(train_ds))),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    model = ObstacleMLP(embed_dim, int(args.hidden_dim), int(args.depth), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_start = time.perf_counter()
    log_progress(f"Training classifier on {len(train_ds)} train / {len(val_ds)} validation / {len(cal_ds)} calibration samples.")

    for _ in range(int(args.epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            scores = model(xb)
            loss = hinge_loss(scores, yb, margin=float(args.margin))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - train_start

    train_eval = evaluate_signed_model(
        model,
        normalized,
        y_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    val_eval = evaluate_signed_model(
        model,
        normalized_val,
        val_y_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    cal_eval = evaluate_signed_model(
        model,
        normalized_cal_obstacle,
        cal_y_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )

    cal_scores = np.asarray(cal_eval["scores"], dtype=np.float64)
    cal_labels = -np.ones((cal_scores.shape[0],), dtype=np.float32)
    cal_obstacle_nonconformity = np.maximum(cal_scores, 0.0)
    conformal = compute_conformal_score_threshold(cal_obstacle_nonconformity, float(args.delta))
    safe_score_threshold = float(conformal["threshold"])

    train_cp = score_threshold_metrics(train_eval, y_all_signed, safe_score_threshold)
    val_cp = score_threshold_metrics(val_eval, y_val_signed, safe_score_threshold)
    cal_cp = score_threshold_metrics(cal_eval, cal_labels, safe_score_threshold)
    obstacle_ranges = infer_joint_ranges(
        np.asarray(center_qpos, dtype=np.float64),
        inside_qpos,
        range_scale=float(args.joint_range_scale),
        range_min=float(args.joint_range_min),
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": embed_dim,
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "feature_mean": train_mean.numpy().astype(np.float32),
            "feature_std": train_std.numpy().astype(np.float32),
            "nominal_position": nominal_latent.astype(np.float32),
            "score_sign_convention": {
                "obstacle": "negative",
                "non_obstacle": "positive",
            },
            "base_decision_threshold": 0.0,
            "conformal_safe_score_threshold": float(safe_score_threshold),
            "conformal_delta": float(args.delta),
            "conformal_nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
            "conformal_score_quantile": float(conformal["score_quantile"]),
            "cache_config": asdict(run_config),
        },
        paths.model,
    )
    torch.save(
        {
            "x_all": x_all.astype(np.float32),
            "y_all_signed": y_all_signed.astype(np.float32),
            "x_val": x_val.astype(np.float32),
            "y_val_signed": y_val_signed.astype(np.float32),
            "eval": {
                "train": train_eval,
                "val": val_eval,
                "cal": cal_eval,
            },
            "conformal": {
                "base_decision_threshold": 0.0,
                "safe_score_threshold": float(safe_score_threshold),
                "delta": float(args.delta),
                "nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
                "score_quantile": float(conformal["score_quantile"]),
                "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
                "metrics": {
                    "train": train_cp,
                    "val": val_cp,
                    "cal": cal_cp,
                },
            },
            "obstacle_center_qpos": np.asarray(center_qpos, dtype=np.float32),
            "obstacle_qpos": inside_qpos.astype(np.float32),
            "calibration_obstacle_qpos": cal_inside_qpos.astype(np.float32),
            "outside_qpos": outside_qpos.astype(np.float32),
            "validation_obstacle_qpos": val_inside_qpos.astype(np.float32),
            "validation_outside_qpos": val_outside_qpos.astype(np.float32),
            "background_rows": background_rows.astype(np.int64),
            "obstacle_latents": obstacle_latents.astype(np.float32),
            "calibration_obstacle_latents": cal_obstacle_latents.astype(np.float32),
            "outside_latents": outside_latents.astype(np.float32),
            "validation_obstacle_latents": val_inside_latents.astype(np.float32),
            "validation_outside_latents": val_outside_latents.astype(np.float32),
            "background_outside_latents": background_latents.astype(np.float32),
            "nominal_position": nominal_latent.astype(np.float32),
            "joint_ranges": obstacle_ranges.astype(np.float32),
        },
        paths.splits,
    )

    summary = {
        "cache_key": cache_key_for_config(run_config),
        "cache_config": asdict(run_config),
        "cache_dir": str(paths.cache_dir),
        "model_path": str(paths.model),
        "planner_payload_path": str(paths.planner_payload),
        "workspace_plot_path": str(paths.workspace_plot),
        "obstacle_overlay_path": str(paths.obstacle_overlay),
        "outside_overlay_path": str(paths.outside_overlay),
        "train_seconds": float(train_seconds),
        "sampling": sampling_summary,
        "dataset_sizes": {
            "total": int(len(full_dataset)),
            "obstacle": int(x_obstacle.shape[0]),
            "outside": int(outside_latents.shape[0]),
            "background_outside": int(background_latents.shape[0]),
            "validation_obstacle": int(val_inside_latents.shape[0]),
            "validation_outside": int(val_outside_latents.shape[0]),
            "calibration_obstacle": int(cal_obstacle_latents.shape[0]),
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "cal": int(len(cal_ds)),
        },
        "metrics": {
            "train": {k: float(v) for k, v in train_eval.items() if k not in {"scores"}},
            "val": {k: float(v) for k, v in val_eval.items() if k not in {"scores"}},
            "cal": {k: float(v) for k, v in cal_eval.items() if k not in {"scores"}},
        },
        "conformal": {
            "base_decision_threshold": 0.0,
            "safe_score_threshold": float(safe_score_threshold),
            "delta": float(args.delta),
            "nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
            "score_quantile": float(conformal["score_quantile"]),
            "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
            "metrics": {
                "train": train_cp,
                "val": val_cp,
                "cal": cal_cp,
            },
        },
        "score_sign_convention": {
            "obstacle": "negative",
            "non_obstacle": "positive",
        },
        "obstacle_center_qpos": np.asarray(center_qpos, dtype=np.float64).tolist(),
        "joint_ranges": obstacle_ranges.tolist(),
        "nominal_latent_norm": float(np.linalg.norm(nominal_latent)),
    }
    save_json(paths.summary, summary)

    log_progress("Obstacle net v3 complete.")
    print(f"Cache dir:  {paths.cache_dir}")
    print(f"Model path: {paths.model}")
    print(f"Train acc:  {train_eval['accuracy']:.4f}")
    print(f"Val acc:    {val_eval['accuracy']:.4f}")
    print(f"Cal acc:    {cal_eval['accuracy']:.4f}")
    print("Conformal nonconformity: max(0, NN(x)) on obstacle calibration samples")
    print(f"Applied safe score threshold: {safe_score_threshold:.6f}")


if __name__ == "__main__":
    main()
