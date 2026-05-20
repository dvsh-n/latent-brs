#!/usr/bin/env python3
"""Train and cache a local obstacle classifier in latent space for one Reacher obstacle instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from reacher.plan import plan_ilqr_mpc as planner

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_4"
DEFAULT_ROLLOUT_DIR = "reacher/plan/ilqr_mpc_mlpdyn/1779300191_episode_00163"
DEFAULT_NOMINAL_ROLLOUT_NAME = "nominal_rollout.pt"
DEFAULT_ROLLOUT_PATH = str(Path(DEFAULT_ROLLOUT_DIR) / DEFAULT_NOMINAL_ROLLOUT_NAME)
DEFAULT_OUT_DIR = str(Path(DEFAULT_ROLLOUT_DIR) / "obstacle_net")
DEFAULT_OBSTACLE_STEP = -1
DEFAULT_OVERLAY_PERTURB_ALPHA = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--rollout-dir", type=Path, default=Path(DEFAULT_ROLLOUT_DIR))
    parser.add_argument("--rollout-path", type=Path, default=Path(DEFAULT_ROLLOUT_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--obstacle-step", type=int, default=DEFAULT_OBSTACLE_STEP)
    parser.add_argument("--frame-batch-size", type=int, default=32)

    parser.add_argument("--joint1-range", type=float, default=0.25)
    parser.add_argument("--joint2-range", type=float, default=0.15)
    parser.add_argument("--obstacle-sample-count", type=int, default=2048)
    parser.add_argument("--outside-sample-count", type=int, default=2048)
    parser.add_argument("--background-outside-sample-count", type=int, default=512)
    parser.add_argument("--max-outside-resample-factor", type=int, default=12)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--calibration-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[obstacle_net_train] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_dataset_paths(dataset_config: object, default_dataset_path: Path) -> list[Path]:
    if dataset_config is None:
        raw_paths = [default_dataset_path]
    elif isinstance(dataset_config, (str, Path)):
        raw_paths = [dataset_config]
    elif isinstance(dataset_config, list):
        if not dataset_config:
            raise ValueError("Training config dataset_path is an empty list.")
        raw_paths = dataset_config
    else:
        raise TypeError(f"Unsupported dataset_path config type: {type(dataset_config).__name__}.")

    dataset_paths = [Path(str(path)).expanduser().resolve() for path in raw_paths]
    missing_paths = [path for path in dataset_paths if not path.is_file()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Training dataset file not found: {missing_str}")
    return dataset_paths


@dataclass
class ObstacleCacheConfig:
    model_dir: str
    checkpoint_path: str
    dataset_path: str
    background_dataset_path: str
    rollout_path: str
    episode_idx: int
    obstacle_step: int
    rollout_horizon: int
    rollout_max_mpc_steps: int
    q_terminal: float
    q_stage: float
    r_control: float
    ilqr_max_iters: int
    ilqr_tol: float
    ilqr_regularization: float
    joint_ranges: list[float]
    obstacle_sample_count: int
    outside_sample_count: int
    background_outside_sample_count: int
    max_outside_resample_factor: int
    hidden_dim: int
    depth: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    delta: float
    calibration_frac: float
    val_frac: float
    test_frac: float
    seed: int
    embed_dim: int


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


def build_cache_config(
    args: argparse.Namespace,
    *,
    model_dir: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    background_dataset_path: Path,
    rollout_path: Path,
    episode_idx: int,
    obstacle_step: int,
    embed_dim: int,
    rollout_metadata: dict[str, Any],
) -> ObstacleCacheConfig:
    return ObstacleCacheConfig(
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        dataset_path=str(dataset_path),
        background_dataset_path=str(background_dataset_path),
        rollout_path=str(rollout_path),
        episode_idx=int(episode_idx),
        obstacle_step=int(obstacle_step),
        rollout_horizon=int(rollout_metadata["horizon"]),
        rollout_max_mpc_steps=int(rollout_metadata["max_mpc_steps"]),
        q_terminal=float(rollout_metadata["q_terminal"]),
        q_stage=float(rollout_metadata["q_stage"]),
        r_control=float(rollout_metadata["r_control"]),
        ilqr_max_iters=int(rollout_metadata["ilqr_max_iters"]),
        ilqr_tol=float(rollout_metadata["ilqr_tol"]),
        ilqr_regularization=float(rollout_metadata["ilqr_regularization"]),
        joint_ranges=[float(args.joint1_range), float(args.joint2_range)],
        obstacle_sample_count=int(args.obstacle_sample_count),
        outside_sample_count=int(args.outside_sample_count),
        background_outside_sample_count=int(args.background_outside_sample_count),
        max_outside_resample_factor=int(args.max_outside_resample_factor),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        label_smoothing=float(args.label_smoothing),
        delta=float(args.delta),
        calibration_frac=float(args.calibration_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        embed_dim=int(embed_dim),
    )


def cache_key_for_config(config: ObstacleCacheConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def infer_cache_dir(out_root: Path, config: ObstacleCacheConfig) -> Path:
    key = cache_key_for_config(config)
    return out_root / f"episode_{config.episode_idx:05d}" / f"step_{config.obstacle_step:04d}" / key


def split_lengths(total: int, cal_frac: float, val_frac: float, test_frac: float) -> tuple[int, int, int, int]:
    if total < 4:
        raise ValueError(f"Need at least 4 samples, got {total}.")
    if min(cal_frac, val_frac, test_frac) < 0.0:
        raise ValueError("Split fractions must be non-negative.")
    holdout = cal_frac + val_frac + test_frac
    if holdout >= 1.0:
        raise ValueError("Expected calibration_frac + val_frac + test_frac < 1.")
    cal = int(round(total * cal_frac))
    val = int(round(total * val_frac))
    test = int(round(total * test_frac))
    train = total - cal - val - test
    while train < 1:
        if test > 0:
            test -= 1
        elif val > 0:
            val -= 1
        elif cal > 0:
            cal -= 1
        train = total - cal - val - test
    return train, cal, val, test


def stratified_split_indices(
    labels: np.ndarray,
    *,
    cal_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {labels.shape}.")

    rng = np.random.default_rng(seed)
    split_names = ("train", "cal", "val", "test")
    split_indices: dict[str, list[np.ndarray]] = {name: [] for name in split_names}

    for class_id in np.unique(labels.astype(np.int64)):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            continue
        shuffled = rng.permutation(class_indices)
        lengths = split_lengths(int(shuffled.size), cal_frac, val_frac, test_frac)
        offset = 0
        for split_name, length in zip(split_names, lengths, strict=True):
            next_offset = offset + int(length)
            split_indices[split_name].append(shuffled[offset:next_offset])
            offset = next_offset

    resolved: dict[str, np.ndarray] = {}
    for split_name in split_names:
        chunks = split_indices[split_name]
        merged = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.int64)
        resolved[split_name] = np.sort(merged.astype(np.int64, copy=False))
    return resolved


def sample_outside_qpos(
    center_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    obstacle_ranges: np.ndarray,
    count: int,
    max_resample_factor: int,
) -> np.ndarray:
    accepted: list[np.ndarray] = []
    remaining = int(count)
    attempts = 0
    while remaining > 0 and attempts < max_resample_factor:
        batch_count = max(remaining * 2, 256)
        sampled = rng.uniform(lower[None, :], upper[None, :], size=(batch_count, center_qpos.shape[0]))
        delta = np.abs(sampled - center_qpos[None, :])
        keep = np.any(delta > obstacle_ranges[None, :], axis=1)
        kept = sampled[keep]
        if kept.shape[0] > 0:
            take = min(remaining, kept.shape[0])
            accepted.append(kept[:take])
            remaining -= take
        attempts += 1
    if remaining > 0:
        raise RuntimeError(
            f"Failed to sample {count} outside states from the valid joint box. "
            f"Missing {remaining} after {attempts} rounds."
        )
    return np.concatenate(accepted, axis=0).astype(np.float64)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(labels.dtype)
    labels01 = labels.to(preds.dtype)
    accuracy = float((preds == labels01).float().mean().item())
    pos_mask = labels01 > 0.5
    neg_mask = ~pos_mask
    recall = float((preds[pos_mask] == 1).float().mean().item()) if torch.any(pos_mask) else 0.0
    specificity = float((preds[neg_mask] == 0).float().mean().item()) if torch.any(neg_mask) else 0.0
    tp = float(torch.sum((preds == 1) & pos_mask).item())
    fp = float(torch.sum((preds == 1) & neg_mask).item())
    precision = tp / max(tp + fp, 1.0)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
    }


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    n = int(scores.shape[0])
    augmented = np.concatenate((np.sort(scores.astype(np.float64)), np.array([np.inf], dtype=np.float64)))
    rank = int(np.ceil((n + 1) * (1.0 - delta))) - 1
    rank = int(np.clip(rank, 0, augmented.shape[0] - 1))
    return float(augmented[rank])


def compute_conformal_threshold(probs_pos_cal: np.ndarray, delta: float) -> dict[str, float]:
    if probs_pos_cal.ndim != 1:
        raise ValueError(f"Expected 1D positive calibration probabilities, got shape {probs_pos_cal.shape}.")
    if probs_pos_cal.size == 0:
        raise ValueError("Need at least one positive calibration sample for conformal calibration.")
    scores = 1.0 - np.clip(probs_pos_cal.astype(np.float64), 0.0, 1.0)
    score_q = conformal_quantile(scores, delta=delta)
    tau = float(np.clip(1.0 - score_q, 0.0, 1.0))
    return {
        "threshold": tau,
        "score_quantile": float(score_q),
        "num_obstacle_calibration": int(probs_pos_cal.size),
    }


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            logits_chunks.append(model(xb).cpu())
    logits = torch.cat(logits_chunks, dim=0)
    loss = float(F.binary_cross_entropy_with_logits(logits, y).item())
    metrics = compute_metrics(logits, y)
    return {
        "loss": loss,
        "logits": logits.numpy().astype(np.float32),
        "probs": torch.sigmoid(logits).numpy().astype(np.float32),
        **metrics,
    }


def joint_limits_from_env(env) -> tuple[np.ndarray, np.ndarray]:
    jnt_range = np.asarray(env._env.physics.model.jnt_range[:2], dtype=np.float64)
    return jnt_range[:, 0].copy(), jnt_range[:, 1].copy()


def sample_local_perturbations(
    center_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    joint_ranges: np.ndarray,
    count: int,
) -> np.ndarray:
    noise = rng.uniform(-joint_ranges, joint_ranges, size=(count, center_qpos.shape[0]))
    return np.clip(center_qpos[None, :] + noise, lower[None, :], upper[None, :]).astype(np.float64)


def render_qpos_batch(
    env,
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
        frame = planner.reset_env_to_state(
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


def choose_background_dataset_path(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_path: Path,
) -> Path:
    if args.background_dataset_path is not None:
        return args.background_dataset_path.expanduser().resolve()

    if dataset_path.is_file():
        return dataset_path

    configured_paths = resolve_dataset_paths(config.get("dataset_path"), dataset_path)
    return configured_paths[0]


def resolve_rollout_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    rollout_dir = args.rollout_dir.expanduser().resolve()

    rollout_path_arg = Path(args.rollout_path)
    if rollout_path_arg == Path(DEFAULT_ROLLOUT_PATH):
        rollout_path = rollout_dir / DEFAULT_NOMINAL_ROLLOUT_NAME
    elif rollout_path_arg.is_absolute():
        rollout_path = rollout_path_arg.resolve()
    else:
        rollout_path = (rollout_dir / rollout_path_arg).resolve()

    out_dir_arg = Path(args.out_dir)
    if out_dir_arg == Path(DEFAULT_OUT_DIR):
        out_dir = rollout_dir / "obstacle_net"
    elif out_dir_arg.is_absolute():
        out_dir = out_dir_arg.resolve()
    else:
        out_dir = (rollout_dir / out_dir_arg).resolve()

    return rollout_path, out_dir


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
    pixels = planner.preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    latents = planner.encode_frames(model, pixels, device=device, frame_batch_size=frame_batch_size)
    latents_np = latents.detach().cpu().numpy().astype(np.float64)
    return latents_np[inverse]


def get_arm_geom_ids(model) -> np.ndarray:
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


def make_segmentation_scene_option(model):
    from dm_control.mujoco.wrapper import core as dm_core

    target_geom_id = int(model.name2id("target", "geom"))
    original_group = int(model.geom_group[target_geom_id])
    model.geom_group[target_geom_id] = 3
    scene_option = dm_core.MjvOption()
    scene_option.geomgroup[:] = 1
    scene_option.geomgroup[3] = 0
    return scene_option, target_geom_id, original_group


def render_masked_qpos_batch(
    env,
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
    planner_module,
    rollout: dict[str, Any],
    center_qpos: np.ndarray,
    obstacle_qpos: np.ndarray,
    out_path: Path,
    perturb_alpha: float = DEFAULT_OVERLAY_PERTURB_ALPHA,
) -> None:
    overlay_qpos_batch = np.concatenate((center_qpos[None, :], obstacle_qpos), axis=0)
    overlay_env = planner_module.make_render_env(
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
    planner_module.save_rgb_image(out_path, obstacle_overlay)


def load_nominal_rollout(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    planner_data = payload.get("planner_data")
    executed_rollout = payload.get("executed_rollout")
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid nominal rollout payload: missing metadata in {path}.")
    if not isinstance(planner_data, dict):
        raise ValueError(f"Invalid nominal rollout payload: missing planner_data in {path}.")
    if not isinstance(executed_rollout, dict):
        raise ValueError(f"Invalid nominal rollout payload: missing executed_rollout in {path}.")
    embed_dim = int(np.asarray(planner_data["start_embedding"]).shape[0])
    if "qpos" not in executed_rollout or "embeddings" not in executed_rollout:
        raise ValueError(f"Invalid nominal rollout payload: executed_rollout must contain qpos and embeddings in {path}.")

    rollout_qpos = np.asarray(executed_rollout["qpos"], dtype=np.float64)
    rollout_emb = np.asarray(executed_rollout["embeddings"], dtype=np.float64)
    if rollout_qpos.ndim != 2 or rollout_qpos.shape[0] == 0:
        raise ValueError(f"Expected non-empty executed qpos trajectory in {path}, got shape {rollout_qpos.shape}.")
    if rollout_emb.ndim != 2 or rollout_emb.shape[0] != rollout_qpos.shape[0]:
        raise ValueError(
            f"Executed embedding trajectory must align with qpos in {path}, "
            f"got qpos {rollout_qpos.shape} and embeddings {rollout_emb.shape}."
        )

    return {
        "episode_idx": int(metadata["episode_idx"]),
        "episode_seed": int(metadata["episode_seed"]),
        "physics_freq_hz": float(metadata["physics_freq_hz"]),
        "time_limit": float(metadata["time_limit"]),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "rollout_qpos": rollout_qpos.astype(np.float32),
        "rollout_emb": rollout_emb[:, :embed_dim].astype(np.float32),
        "planner_args": {
            "horizon": int(metadata["horizon"]),
            "max_mpc_steps": int(metadata["max_mpc_steps"]),
            "q_terminal": float(metadata["q_terminal"]),
            "q_stage": float(metadata["q_stage"]),
            "r_control": float(metadata["r_control"]),
            "ilqr_max_iters": int(metadata["ilqr_max_iters"]),
            "ilqr_tol": float(metadata["ilqr_tol"]),
            "ilqr_regularization": float(metadata["ilqr_regularization"]),
        },
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    rollout_path, out_root = resolve_rollout_paths(args)
    out_root.mkdir(parents=True, exist_ok=True)

    config = planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else planner.latest_object_checkpoint(model_dir).resolve()
    )
    world_model = planner.load_model(checkpoint_path, device)
    background_dataset_path = choose_background_dataset_path(args, config, dataset_path)
    rollout = load_nominal_rollout(rollout_path)
    rollout_episode_idx = int(rollout["episode_idx"])
    log_progress(
        f"Preparing obstacle classifier training from nominal rollout {rollout_path} "
        f"(episode {rollout_episode_idx}, step {args.obstacle_step})."
    )

    rollout_qpos = np.asarray(rollout["rollout_qpos"], dtype=np.float64)
    rollout_emb = np.asarray(rollout["rollout_emb"], dtype=np.float64)
    obstacle_step = int(args.obstacle_step)
    if obstacle_step == -1:
        obstacle_step = int(rollout_qpos.shape[0] - 1)
        log_progress(f"Resolved obstacle step -1 to final rollout step {obstacle_step}.")
    if obstacle_step < 0 or obstacle_step >= rollout_qpos.shape[0]:
        raise ValueError(f"--obstacle-step must be in [0, {rollout_qpos.shape[0] - 1}], got {args.obstacle_step}.")

    embed_dim = int(config.get("embed_dim", 24))
    cache_config = build_cache_config(
        args,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        background_dataset_path=background_dataset_path,
        rollout_path=rollout_path,
        episode_idx=rollout_episode_idx,
        obstacle_step=obstacle_step,
        embed_dim=embed_dim,
        rollout_metadata=dict(rollout["planner_args"]),
    )
    cache_dir = infer_cache_dir(out_root, cache_config)
    summary_path = cache_dir / "summary.json"
    model_path = cache_dir / "model.pt"
    split_path = cache_dir / "splits.pt"
    overlay_path = cache_dir / "obstacle_overlay_all.png"
    if summary_path.is_file() and model_path.is_file() and split_path.is_file() and not args.force_retrain:
        if not overlay_path.is_file():
            log_progress(f"Cached obstacle model found at {cache_dir}; regenerating missing obstacle overlay.")
            cached = torch.load(split_path, map_location="cpu", weights_only=False)
            save_obstacle_overlay(
                planner_module=planner,
                rollout=rollout,
                center_qpos=rollout_qpos[obstacle_step],
                obstacle_qpos=np.asarray(cached["obstacle_qpos"], dtype=np.float64),
                out_path=overlay_path,
            )
        log_progress(f"Using cached obstacle model at {cache_dir}.")
        print(f"Cache dir:  {cache_dir}")
        print(f"Model path: {model_path}")
        return

    img_size = int(config.get("img_size", 224))
    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)
    env = planner.make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    lower, upper = joint_limits_from_env(env)
    center_qpos = rollout_qpos[obstacle_step]
    nominal_position = rollout_emb[obstacle_step, :embed_dim].astype(np.float64)

    obstacle_ranges = np.array([float(args.joint1_range), float(args.joint2_range)], dtype=np.float64)
    if obstacle_ranges.shape[0] != center_qpos.shape[0]:
        raise ValueError(f"Expected {center_qpos.shape[0]} joint ranges, got {obstacle_ranges.shape[0]}.")

    log_progress("Sampling obstacle states inside the box and outside states from the remaining valid joint space.")
    obstacle_qpos = sample_local_perturbations(
        center_qpos,
        lower,
        upper,
        rng,
        joint_ranges=obstacle_ranges,
        count=int(args.obstacle_sample_count),
    )
    outside_qpos = sample_outside_qpos(
        center_qpos,
        lower,
        upper,
        rng,
        obstacle_ranges=obstacle_ranges,
        count=int(args.outside_sample_count),
        max_resample_factor=int(args.max_outside_resample_factor),
    )
    qpos_batch = np.concatenate((obstacle_qpos, outside_qpos), axis=0)
    local_frames = render_qpos_batch(
        env,
        int(rollout["episode_seed"]),
        qpos_batch,
        height=int(rollout["height"]),
        width=int(rollout["width"]),
        progress_desc="Rendering obstacle/outside states",
    )
    env.close()

    local_pixels = planner.preprocess_pixels(
        local_frames,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    local_emb = planner.encode_frames(
        world_model,
        local_pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    ).detach().cpu().numpy().astype(np.float64)
    obstacle_latents = local_emb[: args.obstacle_sample_count, :embed_dim]
    outside_latents = local_emb[args.obstacle_sample_count :, :embed_dim]

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

    x_obstacle = obstacle_latents
    x_outside = np.concatenate((outside_latents, background_latents), axis=0)
    y_obstacle = np.ones((x_obstacle.shape[0],), dtype=np.float32)
    y_outside = np.zeros((x_outside.shape[0],), dtype=np.float32)
    x_all = np.concatenate((x_obstacle, x_outside), axis=0).astype(np.float32)
    y_all = np.concatenate((y_obstacle, y_outside), axis=0).astype(np.float32)

    x_tensor = torch.from_numpy(x_all)
    y_tensor = torch.from_numpy(y_all)
    full_dataset = TensorDataset(x_tensor, y_tensor)
    split_indices = stratified_split_indices(
        y_all,
        cal_frac=float(args.calibration_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    train_idx = split_indices["train"]
    cal_idx = split_indices["cal"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    if not np.any(y_all[cal_idx] > 0.5):
        raise RuntimeError("Calibration split contains no obstacle samples; increase sample counts or adjust split fractions.")

    train_x = x_tensor[train_idx]
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0).clamp_min(1e-6)

    def normalize_features(x: torch.Tensor) -> torch.Tensor:
        return (x - train_mean) / train_std

    normalized = normalize_features(x_tensor)
    normalized_dataset = TensorDataset(normalized, y_tensor)
    train_ds = torch.utils.data.Subset(normalized_dataset, train_idx.tolist())
    cal_ds = torch.utils.data.Subset(normalized_dataset, cal_idx.tolist())
    val_ds = torch.utils.data.Subset(normalized_dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(normalized_dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(args.batch_size), max(1, len(train_ds))),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    model = ObstacleMLP(embed_dim, int(args.hidden_dim), int(args.depth), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_labels = y_all[train_idx].astype(np.int64, copy=False)
    class_counts = np.bincount(train_labels, minlength=2)
    pos_weight = torch.tensor(
        float(class_counts[0] / max(class_counts[1], 1)),
        dtype=torch.float32,
        device=device,
    )
    best_state = None
    best_val_loss = math.inf
    train_start = time.perf_counter()
    log_progress(
        f"Training classifier on {len(train_ds)} train / {len(cal_ds)} cal / {len(val_ds)} val / {len(test_ds)} test samples."
    )

    for _ in range(int(args.epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if args.label_smoothing > 0.0:
                yb_loss = yb * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            else:
                yb_loss = yb
            loss = F.binary_cross_entropy_with_logits(logits, yb_loss, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_eval = evaluate_model(
            model,
            normalized[val_idx],
            y_tensor[val_idx],
            batch_size=int(args.batch_size),
            device=device,
        )
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - train_start

    train_eval = evaluate_model(
        model,
        normalized[train_idx],
        y_tensor[train_idx],
        batch_size=int(args.batch_size),
        device=device,
    )
    cal_eval = evaluate_model(
        model,
        normalized[cal_idx],
        y_tensor[cal_idx],
        batch_size=int(args.batch_size),
        device=device,
    )
    val_eval = evaluate_model(
        model,
        normalized[val_idx],
        y_tensor[val_idx],
        batch_size=int(args.batch_size),
        device=device,
    )
    test_eval = evaluate_model(
        model,
        normalized[test_idx],
        y_tensor[test_idx],
        batch_size=int(args.batch_size),
        device=device,
    )

    cal_labels = y_all[cal_idx]
    cal_probs = np.asarray(cal_eval["probs"], dtype=np.float64)
    cal_obstacle_probs = cal_probs[cal_labels > 0.5]
    conformal = compute_conformal_threshold(cal_obstacle_probs, float(args.delta))
    tau = float(conformal["threshold"])

    def threshold_metrics(eval_payload: dict[str, Any], labels: np.ndarray, threshold: float) -> dict[str, float]:
        probs = np.asarray(eval_payload["probs"], dtype=np.float64)
        preds = probs >= threshold
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        coverage_pos = float(np.mean(preds[pos_mask])) if np.any(pos_mask) else 0.0
        false_positive_rate = float(np.mean(preds[neg_mask])) if np.any(neg_mask) else 0.0
        return {
            "obstacle_coverage": coverage_pos,
            "outside_activation_rate": false_positive_rate,
            "threshold": float(threshold),
        }

    train_cp = threshold_metrics(train_eval, y_all[train_idx], tau)
    cal_cp = threshold_metrics(cal_eval, cal_labels, tau)
    val_cp = threshold_metrics(val_eval, y_all[val_idx], tau)
    test_cp = threshold_metrics(test_eval, y_all[test_idx], tau)

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_progress("Saving obstacle overlay image with nominal state emphasized.")
    save_obstacle_overlay(
        planner_module=planner,
        rollout=rollout,
        center_qpos=center_qpos,
        obstacle_qpos=obstacle_qpos,
        out_path=overlay_path,
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
            "nominal_position": nominal_position.astype(np.float32),
            "conformal_threshold": float(tau),
            "conformal_delta": float(args.delta),
            "conformal_score_quantile": float(conformal["score_quantile"]),
            "cache_config": asdict(cache_config),
        },
        model_path,
    )
    torch.save(
        {
            "x_all": x_all.astype(np.float32),
            "y_all": y_all.astype(np.float32),
            "indices": {
                "train": train_idx,
                "cal": cal_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "eval": {
                "train": train_eval,
                "cal": cal_eval,
                "val": val_eval,
                "test": test_eval,
            },
            "conformal": {
                "threshold": float(tau),
                "delta": float(args.delta),
                "score_quantile": float(conformal["score_quantile"]),
                "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
                "metrics": {
                    "train": train_cp,
                    "cal": cal_cp,
                    "val": val_cp,
                    "test": test_cp,
                },
            },
            "obstacle_qpos": obstacle_qpos.astype(np.float32),
            "outside_qpos": outside_qpos.astype(np.float32),
            "background_rows": background_rows.astype(np.int64),
            "obstacle_latents": obstacle_latents.astype(np.float32),
            "outside_latents": outside_latents.astype(np.float32),
            "background_outside_latents": background_latents.astype(np.float32),
            "nominal_position": nominal_position.astype(np.float32),
        },
        split_path,
    )
    summary = {
        "cache_key": cache_key_for_config(cache_config),
        "cache_config": asdict(cache_config),
        "cache_dir": str(cache_dir),
        "model_path": str(model_path),
        "rollout_path": str(rollout_path),
        "train_seconds": float(train_seconds),
        "dataset_sizes": {
            "total": int(len(full_dataset)),
            "obstacle": int(x_obstacle.shape[0]),
            "outside": int(outside_latents.shape[0]),
            "background_outside": int(background_latents.shape[0]),
            "train": int(len(train_ds)),
            "cal": int(len(cal_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "metrics": {
            "train": {k: float(v) for k, v in train_eval.items() if k not in {"logits", "probs"}},
            "cal": {k: float(v) for k, v in cal_eval.items() if k not in {"logits", "probs"}},
            "val": {k: float(v) for k, v in val_eval.items() if k not in {"logits", "probs"}},
            "test": {k: float(v) for k, v in test_eval.items() if k not in {"logits", "probs"}},
        },
        "conformal": {
            "threshold": float(tau),
            "delta": float(args.delta),
            "score_quantile": float(conformal["score_quantile"]),
            "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
            "metrics": {
                "train": train_cp,
                "cal": cal_cp,
                "val": val_cp,
                "test": test_cp,
            },
        },
        "obstacle_center_qpos": center_qpos.tolist(),
        "joint_ranges": obstacle_ranges.tolist(),
        "obstacle_center_latent_norm": float(np.linalg.norm(nominal_position)),
    }
    save_json(summary_path, summary)

    log_progress("Obstacle classifier training complete.")
    print(f"Cache dir:  {cache_dir}")
    print(f"Model path: {model_path}")
    print(f"Val acc:    {val_eval['accuracy']:.4f}")
    print(f"Test acc:   {test_eval['accuracy']:.4f}")


if __name__ == "__main__":
    main()
