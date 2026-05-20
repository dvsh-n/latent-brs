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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from reacher.plan import analyze_conformal_obstacle as obstacle_analysis
from reacher.plan import plan_ilqr_mpc as planner
from reacher.train.mlpdyn_train import LeWMReacherDataset

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_5"
DEFAULT_OUT_DIR = "reacher/plan/obstacle_nets"
DEFAULT_EPISODE_IDX = 520
DEFAULT_OBSTACLE_STEP = -1
DEFAULT_HORIZON = 20
DEFAULT_MAX_MPC_STEPS = 100
DEFAULT_OVERLAY_PERTURB_ALPHA = 0.035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-idx", type=int, default=DEFAULT_EPISODE_IDX)
    parser.add_argument("--obstacle-step", type=int, default=DEFAULT_OBSTACLE_STEP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=DEFAULT_MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--q-terminal", type=float, default=10.0)
    parser.add_argument("--q-stage", type=float, default=0.005)
    parser.add_argument("--r-control", type=float, default=0.1)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--force-rerun-rollout", action="store_true", default=False)

    parser.add_argument("--joint1-range", type=float, default=0.25)
    parser.add_argument("--joint2-range", type=float, default=0.0)
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


@dataclass
class ObstacleCacheConfig:
    model_dir: str
    checkpoint_path: str
    dataset_path: str
    background_dataset_path: str
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
    obstacle_step: int,
    embed_dim: int,
) -> ObstacleCacheConfig:
    return ObstacleCacheConfig(
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        dataset_path=str(dataset_path),
        background_dataset_path=str(background_dataset_path),
        episode_idx=int(args.episode_idx),
        obstacle_step=int(obstacle_step),
        rollout_horizon=int(args.horizon),
        rollout_max_mpc_steps=int(args.max_mpc_steps),
        q_terminal=float(args.q_terminal),
        q_stage=float(args.q_stage),
        r_control=float(args.r_control),
        ilqr_max_iters=int(args.ilqr_max_iters),
        ilqr_tol=float(args.ilqr_tol),
        ilqr_regularization=float(args.ilqr_regularization),
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
        overlay_frames, overlay_masks = obstacle_analysis.render_masked_qpos_batch(
            overlay_env,
            int(rollout["episode_seed"]),
            overlay_qpos_batch,
            height=int(rollout["height"]),
            width=int(rollout["width"]),
        )
    finally:
        overlay_env.close()
    obstacle_overlay = obstacle_analysis.make_obstacle_overlay_image(
        nominal_frame=overlay_frames[0],
        nominal_mask=overlay_masks[0],
        perturb_frames=overlay_frames[1:],
        perturb_masks=overlay_masks[1:],
        perturb_alpha=float(perturb_alpha),
    )
    planner_module.save_rgb_image(out_path, obstacle_overlay)


def main() -> None:
    args = parse_args()
    log_progress(f"Preparing obstacle classifier training for episode {args.episode_idx}, step {args.obstacle_step}.")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else planner.latest_object_checkpoint(model_dir).resolve()
    )
    world_model = planner.load_model(checkpoint_path, device)
    background_dataset_path = (
        args.background_dataset_path.expanduser().resolve()
        if args.background_dataset_path is not None
        else Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    )

    rollout_cache_path = obstacle_analysis.infer_rollout_cache_path(out_root, checkpoint_path, args.episode_idx, args)
    rollout = obstacle_analysis.run_or_load_rollout(
        planner=planner,
        dataset_cls=LeWMReacherDataset,
        cache_path=rollout_cache_path,
        force_rerun=args.force_rerun_rollout,
        model=world_model,
        config=config,
        dataset_path=dataset_path,
        episode_idx=args.episode_idx,
        device=device,
        frame_batch_size=args.frame_batch_size,
        args=args,
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
        obstacle_step=obstacle_step,
        embed_dim=embed_dim,
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
    lower, upper = obstacle_analysis.joint_limits_from_env(env)
    center_qpos = rollout_qpos[obstacle_step]
    nominal_position = rollout_emb[obstacle_step, :embed_dim].astype(np.float64)

    obstacle_ranges = np.array([float(args.joint1_range), float(args.joint2_range)], dtype=np.float64)
    if obstacle_ranges.shape[0] != center_qpos.shape[0]:
        raise ValueError(f"Expected {center_qpos.shape[0]} joint ranges, got {obstacle_ranges.shape[0]}.")

    log_progress("Sampling obstacle states inside the box and outside states from the remaining valid joint space.")
    obstacle_qpos = obstacle_analysis.sample_local_perturbations(
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
    local_frames = obstacle_analysis.render_qpos_batch(
        planner,
        env,
        int(rollout["episode_seed"]),
        qpos_batch,
        height=int(rollout["height"]),
        width=int(rollout["width"]),
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
        background_rows = obstacle_analysis.sample_background_rows(
            background_dataset_path,
            rng,
            int(args.background_outside_sample_count),
        )
        background_emb = obstacle_analysis.encode_dataset_rows(
            planner,
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
    lengths = split_lengths(len(full_dataset), args.calibration_frac, args.val_frac, args.test_frac)
    split_gen = torch.Generator().manual_seed(args.seed)
    train_ds, cal_ds, val_ds, test_ds = random_split(full_dataset, lengths, generator=split_gen)

    train_x = x_tensor[train_ds.indices]
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0).clamp_min(1e-6)

    def normalize_features(x: torch.Tensor) -> torch.Tensor:
        return (x - train_mean) / train_std

    normalized = normalize_features(x_tensor)
    normalized_dataset = TensorDataset(normalized, y_tensor)
    train_ds = torch.utils.data.Subset(normalized_dataset, train_ds.indices)
    cal_ds = torch.utils.data.Subset(normalized_dataset, cal_ds.indices)
    val_ds = torch.utils.data.Subset(normalized_dataset, val_ds.indices)
    test_ds = torch.utils.data.Subset(normalized_dataset, test_ds.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(args.batch_size), max(1, len(train_ds))),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    model = ObstacleMLP(embed_dim, int(args.hidden_dim), int(args.depth), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    class_counts = np.bincount(y_all.astype(np.int64), minlength=2)
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

    for epoch in range(int(args.epochs)):
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
            normalized[val_ds.indices],
            y_tensor[val_ds.indices],
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
        normalized[train_ds.indices],
        y_tensor[train_ds.indices],
        batch_size=int(args.batch_size),
        device=device,
    )
    cal_eval = evaluate_model(
        model,
        normalized[cal_ds.indices],
        y_tensor[cal_ds.indices],
        batch_size=int(args.batch_size),
        device=device,
    )
    val_eval = evaluate_model(
        model,
        normalized[val_ds.indices],
        y_tensor[val_ds.indices],
        batch_size=int(args.batch_size),
        device=device,
    )
    test_eval = evaluate_model(
        model,
        normalized[test_ds.indices],
        y_tensor[test_ds.indices],
        batch_size=int(args.batch_size),
        device=device,
    )

    cal_labels = y_all[np.asarray(cal_ds.indices, dtype=np.int64)]
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

    train_cp = threshold_metrics(train_eval, y_all[np.asarray(train_ds.indices, dtype=np.int64)], tau)
    cal_cp = threshold_metrics(cal_eval, cal_labels, tau)
    val_cp = threshold_metrics(val_eval, y_all[np.asarray(val_ds.indices, dtype=np.int64)], tau)
    test_cp = threshold_metrics(test_eval, y_all[np.asarray(test_ds.indices, dtype=np.int64)], tau)

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
                "train": np.asarray(train_ds.indices, dtype=np.int64),
                "cal": np.asarray(cal_ds.indices, dtype=np.int64),
                "val": np.asarray(val_ds.indices, dtype=np.int64),
                "test": np.asarray(test_ds.indices, dtype=np.int64),
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
        "rollout_cache_path": str(rollout_cache_path),
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
