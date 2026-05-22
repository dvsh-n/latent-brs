#!/usr/bin/env python3
"""Train and cache a local latent obstacle classifier from circle-obstacle samples."""

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
from torch.utils.data import DataLoader, TensorDataset

from reacher.plan import obstacle_net_train as obstacle_v1
from reacher.plan import plan_ilqr_mpc as planner

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_noisy.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_4"
DEFAULT_OBSTACLE_DIR = "reacher/plan/circle_obstacle_sampling"
DEFAULT_OBSTACLE_PAYLOAD_NAME = "planner_start_goal_obstacle.pt"
DEFAULT_OBSTACLE_SUMMARY_NAME = "summary.json"
DEFAULT_OUT_DIR = str(Path(DEFAULT_OBSTACLE_DIR) / "obstacle_net_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--obstacle-dir", type=Path, default=Path(DEFAULT_OBSTACLE_DIR))
    parser.add_argument("--obstacle-payload-name", type=str, default=DEFAULT_OBSTACLE_PAYLOAD_NAME)
    parser.add_argument("--obstacle-summary-name", type=str, default=DEFAULT_OBSTACLE_SUMMARY_NAME)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-batch-size", type=int, default=32)

    parser.add_argument("--joint-range-scale", type=float, default=1.15)
    parser.add_argument("--joint-range-min", type=float, default=0.05)
    parser.add_argument("--outside-sample-count", type=int, default=2048)
    parser.add_argument("--background-outside-sample-count", type=int, default=512)
    parser.add_argument("--max-outside-resample-factor", type=int, default=12)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--calibration-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[obstacle_net_train_v2] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class ObstacleCacheConfigV2:
    model_dir: str
    checkpoint_path: str
    background_dataset_path: str
    obstacle_payload_path: str
    obstacle_summary_path: str
    obstacle_dir: str
    obstacle_sample_count: int
    outside_seed_count: int
    outside_sample_count: int
    background_outside_sample_count: int
    max_outside_resample_factor: int
    joint_range_scale: float
    joint_range_min: float
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
    val_frac: float
    test_frac: float
    seed: int
    embed_dim: int


def cache_key_for_config(config: ObstacleCacheConfigV2) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def infer_cache_dir(out_root: Path, config: ObstacleCacheConfigV2) -> Path:
    return out_root / cache_key_for_config(config)


def load_circle_obstacle_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("metadata", "episode_data", "planner_data")
    missing = [key for key in required if key not in payload or not isinstance(payload[key], dict)]
    if missing:
        raise ValueError(f"Invalid obstacle payload {path}: missing sections {missing}.")

    planner_data = payload["planner_data"]
    metadata = payload["metadata"]
    if "obstacle_center_qpos" not in planner_data or "obstacle_qpos" not in planner_data:
        raise ValueError(f"Obstacle payload {path} must contain obstacle_center_qpos and obstacle_qpos.")
    if "episode_seed" not in metadata:
        raise ValueError(f"Obstacle payload {path} must contain episode_seed in metadata.")
    return payload


def load_circle_obstacle_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid obstacle summary {path}.")
    return summary


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
        raise ValueError(f"Expected 1D obstacle calibration nonconformity scores, got shape {obstacle_nonconformity_cal.shape}.")
    if obstacle_nonconformity_cal.size == 0:
        raise ValueError("Need at least one obstacle calibration sample for conformal calibration.")
    threshold = obstacle_v1.conformal_quantile(obstacle_nonconformity_cal.astype(np.float64), delta=delta)
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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    obstacle_dir = args.obstacle_dir.expanduser().resolve()
    obstacle_payload_path = (obstacle_dir / args.obstacle_payload_name).resolve()
    obstacle_summary_path = (obstacle_dir / args.obstacle_summary_name).resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else planner.latest_object_checkpoint(model_dir).resolve()
    )
    world_model = planner.load_model(checkpoint_path, device)
    default_background_dataset_path = Path(DEFAULT_TEST_DATASET_PATH).expanduser().resolve()
    background_dataset_path = obstacle_v1.choose_background_dataset_path(args, config, default_background_dataset_path)
    obstacle_payload = load_circle_obstacle_payload(obstacle_payload_path)
    obstacle_summary = load_circle_obstacle_summary(obstacle_summary_path)

    metadata = obstacle_payload["metadata"]
    planner_data = obstacle_payload["planner_data"]
    center_qpos = np.asarray(planner_data["obstacle_center_qpos"], dtype=np.float64)
    obstacle_qpos = np.asarray(planner_data["obstacle_qpos"], dtype=np.float64)
    if obstacle_qpos.ndim != 2 or obstacle_qpos.shape[0] == 0:
        raise ValueError(f"Expected non-empty obstacle_qpos batch in {obstacle_payload_path}, got {obstacle_qpos.shape}.")
    if center_qpos.shape != (obstacle_qpos.shape[1],):
        raise ValueError(
            f"Obstacle center shape mismatch: center {center_qpos.shape}, obstacle batch {obstacle_qpos.shape}."
        )

    outside_seed_qpos = np.asarray(obstacle_summary.get("outside_qpos", []), dtype=np.float64)
    if outside_seed_qpos.size == 0:
        outside_seed_qpos = np.zeros((0, center_qpos.shape[0]), dtype=np.float64)
    if outside_seed_qpos.ndim != 2 or outside_seed_qpos.shape[1] != center_qpos.shape[0]:
        raise ValueError(
            f"Expected outside_qpos in {obstacle_summary_path} to have shape (N, {center_qpos.shape[0]}), "
            f"got {outside_seed_qpos.shape}."
        )

    embed_dim = int(config.get("embed_dim", 24))
    cache_config = ObstacleCacheConfigV2(
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        background_dataset_path=str(background_dataset_path),
        obstacle_payload_path=str(obstacle_payload_path),
        obstacle_summary_path=str(obstacle_summary_path),
        obstacle_dir=str(obstacle_dir),
        obstacle_sample_count=int(obstacle_qpos.shape[0]),
        outside_seed_count=int(outside_seed_qpos.shape[0]),
        outside_sample_count=int(args.outside_sample_count),
        background_outside_sample_count=int(args.background_outside_sample_count),
        max_outside_resample_factor=int(args.max_outside_resample_factor),
        joint_range_scale=float(args.joint_range_scale),
        joint_range_min=float(args.joint_range_min),
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
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
        embed_dim=int(embed_dim),
    )
    cache_dir = infer_cache_dir(out_root, cache_config)
    summary_path = cache_dir / "summary.json"
    model_path = cache_dir / "model.pt"
    split_path = cache_dir / "splits.pt"
    overlay_path = cache_dir / "obstacle_overlay_all.png"

    if summary_path.is_file() and model_path.is_file() and split_path.is_file() and not args.force_retrain:
        if not overlay_path.is_file():
            log_progress(f"Cached obstacle model found at {cache_dir}; regenerating missing obstacle overlay.")
            obstacle_v1.save_obstacle_overlay(
                planner_module=planner,
                rollout=metadata,
                center_qpos=center_qpos,
                obstacle_qpos=obstacle_qpos,
                out_path=overlay_path,
            )
        log_progress(f"Using cached obstacle model at {cache_dir}.")
        print(f"Cache dir:  {cache_dir}")
        print(f"Model path: {model_path}")
        return

    img_size = int(config.get("img_size", 224))
    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)
    env = planner.make_render_env(
        seed=int(metadata["episode_seed"]),
        time_limit=float(metadata["time_limit"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        physics_freq_hz=float(metadata["physics_freq_hz"]),
    )
    lower, upper = obstacle_v1.joint_limits_from_env(env)
    nominal_frame = planner.reset_env_to_state(
        env,
        seed=int(metadata["episode_seed"]),
        qpos=center_qpos.astype(np.float32),
        qvel=np.zeros_like(center_qpos, dtype=np.float32),
        height=int(metadata["height"]),
        width=int(metadata["width"]),
    )
    nominal_latent = planner.encode_single_frame(
        world_model,
        nominal_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    ).detach().cpu().numpy().astype(np.float64)[:embed_dim]

    obstacle_ranges = infer_joint_ranges(
        center_qpos,
        obstacle_qpos,
        range_scale=float(args.joint_range_scale),
        range_min=float(args.joint_range_min),
    )
    extra_outside_count = max(0, int(args.outside_sample_count) - int(outside_seed_qpos.shape[0]))
    log_progress("Sampling outside states from the valid joint space around the circle obstacle.")
    sampled_outside_qpos = obstacle_v1.sample_outside_qpos(
        center_qpos,
        lower,
        upper,
        rng,
        obstacle_ranges=obstacle_ranges,
        count=extra_outside_count,
        max_resample_factor=int(args.max_outside_resample_factor),
    )
    outside_qpos = np.concatenate((outside_seed_qpos, sampled_outside_qpos), axis=0)

    qpos_batch = np.concatenate((obstacle_qpos, outside_qpos), axis=0)
    local_frames = obstacle_v1.render_qpos_batch(
        env,
        int(metadata["episode_seed"]),
        qpos_batch,
        height=int(metadata["height"]),
        width=int(metadata["width"]),
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
    obstacle_latents = local_emb[: obstacle_qpos.shape[0], :embed_dim]
    outside_latents = local_emb[obstacle_qpos.shape[0] :, :embed_dim]

    background_latents = np.zeros((0, embed_dim), dtype=np.float64)
    if args.background_outside_sample_count > 0:
        log_progress(f"Sampling {args.background_outside_sample_count} background outside states from dataset latents.")
        background_rows = obstacle_v1.sample_background_rows(
            background_dataset_path,
            rng,
            int(args.background_outside_sample_count),
        )
        background_emb = obstacle_v1.encode_dataset_rows(
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
    split_indices = obstacle_v1.stratified_split_indices(
        (y_all_signed < 0.0).astype(np.float32),
        cal_frac=float(args.calibration_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    train_idx = split_indices["train"]
    cal_idx = split_indices["cal"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    if not np.any(y_all_signed[cal_idx] < 0.0):
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

    model = obstacle_v1.ObstacleMLP(embed_dim, int(args.hidden_dim), int(args.depth), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
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
            scores = model(xb)
            loss = hinge_loss(scores, yb, margin=float(args.margin))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_eval = evaluate_signed_model(
            model,
            normalized[val_idx],
            y_tensor[val_idx],
            batch_size=int(args.batch_size),
            device=device,
            margin=float(args.margin),
        )
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - train_start

    train_eval = evaluate_signed_model(
        model,
        normalized[train_idx],
        y_tensor[train_idx],
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    cal_eval = evaluate_signed_model(
        model,
        normalized[cal_idx],
        y_tensor[cal_idx],
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    val_eval = evaluate_signed_model(
        model,
        normalized[val_idx],
        y_tensor[val_idx],
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    test_eval = evaluate_signed_model(
        model,
        normalized[test_idx],
        y_tensor[test_idx],
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )

    cal_labels = y_all_signed[cal_idx]
    cal_scores = np.asarray(cal_eval["scores"], dtype=np.float64)
    cal_obstacle_scores = cal_scores[cal_labels < 0.0]
    cal_obstacle_nonconformity = np.maximum(cal_obstacle_scores, 0.0)
    conformal = compute_conformal_score_threshold(cal_obstacle_nonconformity, float(args.delta))
    safe_score_threshold = float(conformal["threshold"])

    train_cp = score_threshold_metrics(train_eval, y_all_signed[train_idx], safe_score_threshold)
    cal_cp = score_threshold_metrics(cal_eval, cal_labels, safe_score_threshold)
    val_cp = score_threshold_metrics(val_eval, y_all_signed[val_idx], safe_score_threshold)
    test_cp = score_threshold_metrics(test_eval, y_all_signed[test_idx], safe_score_threshold)

    cache_dir.mkdir(parents=True, exist_ok=True)
    log_progress("Saving obstacle overlay image with nominal state emphasized.")
    obstacle_v1.save_obstacle_overlay(
        planner_module=planner,
        rollout=metadata,
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
            "cache_config": asdict(cache_config),
        },
        model_path,
    )
    torch.save(
        {
            "x_all": x_all.astype(np.float32),
            "y_all_signed": y_all_signed.astype(np.float32),
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
                "base_decision_threshold": 0.0,
                "safe_score_threshold": float(safe_score_threshold),
                "delta": float(args.delta),
                "nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
                "score_quantile": float(conformal["score_quantile"]),
                "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
                "metrics": {
                    "train": train_cp,
                    "cal": cal_cp,
                    "val": val_cp,
                    "test": test_cp,
                },
            },
            "obstacle_center_qpos": center_qpos.astype(np.float32),
            "obstacle_qpos": obstacle_qpos.astype(np.float32),
            "outside_seed_qpos": outside_seed_qpos.astype(np.float32),
            "sampled_outside_qpos": sampled_outside_qpos.astype(np.float32),
            "outside_qpos": outside_qpos.astype(np.float32),
            "background_rows": background_rows.astype(np.int64),
            "obstacle_latents": obstacle_latents.astype(np.float32),
            "outside_latents": outside_latents.astype(np.float32),
            "background_outside_latents": background_latents.astype(np.float32),
            "nominal_position": nominal_latent.astype(np.float32),
            "joint_ranges": obstacle_ranges.astype(np.float32),
        },
        split_path,
    )
    summary = {
        "cache_key": cache_key_for_config(cache_config),
        "cache_config": asdict(cache_config),
        "cache_dir": str(cache_dir),
        "model_path": str(model_path),
        "obstacle_payload_path": str(obstacle_payload_path),
        "obstacle_summary_path": str(obstacle_summary_path),
        "train_seconds": float(train_seconds),
        "dataset_sizes": {
            "total": int(len(full_dataset)),
            "obstacle": int(x_obstacle.shape[0]),
            "outside_seed": int(outside_seed_qpos.shape[0]),
            "outside_sampled": int(sampled_outside_qpos.shape[0]),
            "outside_total": int(outside_qpos.shape[0]),
            "background_outside": int(background_latents.shape[0]),
            "train": int(len(train_ds)),
            "cal": int(len(cal_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "metrics": {
            "train": {k: float(v) for k, v in train_eval.items() if k not in {"scores"}},
            "cal": {k: float(v) for k, v in cal_eval.items() if k not in {"scores"}},
            "val": {k: float(v) for k, v in val_eval.items() if k not in {"scores"}},
            "test": {k: float(v) for k, v in test_eval.items() if k not in {"scores"}},
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
                "cal": cal_cp,
                "val": val_cp,
                "test": test_cp,
            },
        },
        "score_sign_convention": {
            "obstacle": "negative",
            "non_obstacle": "positive",
        },
        "obstacle_center_qpos": center_qpos.tolist(),
        "joint_ranges": obstacle_ranges.tolist(),
        "nominal_latent_norm": float(np.linalg.norm(nominal_latent)),
    }
    save_json(summary_path, summary)

    log_progress("Obstacle classifier training complete.")
    print(f"Cache dir:  {cache_dir}")
    print(f"Model path: {model_path}")
    print(f"Val acc:    {val_eval['accuracy']:.4f}")
    print(f"Test acc:   {test_eval['accuracy']:.4f}")
    print("Conformal nonconformity: max(0, NN(x)) on obstacle calibration samples")
    print(f"Applied safe score threshold: {safe_score_threshold:.6f}")


if __name__ == "__main__":
    main()
