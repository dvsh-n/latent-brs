#!/usr/bin/env python3
"""Train a goal-conditioned latent value model on the first half of the rope test dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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


DEFAULT_DATASET_PATH = "rope/data/test_data_noshadow/rope_random_cubic_spline.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow_ft"
DEFAULT_OUT_DIR = "rope/models/goal_value_net"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--encoded-data-path", type=Path, default=None)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3072)

    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-episode-frac", type=float, default=0.5)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--discount", type=float, default=0.6)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=0.05)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[rope_goal_value_train] {message}", flush=True)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


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


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_world_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def required_markov_history(markov_deriv: int) -> int:
    if markov_deriv < 0:
        raise ValueError("markov_deriv must be non-negative.")
    return markov_deriv + 1


def build_markov_state(history_emb: torch.Tensor, markov_deriv: int) -> torch.Tensor:
    squeeze = False
    if history_emb.ndim == 2:
        history_emb = history_emb.unsqueeze(0)
        squeeze = True
    context_len = required_markov_history(markov_deriv)
    if history_emb.ndim != 3 or history_emb.shape[1] < context_len:
        raise ValueError(
            f"Expected history_emb with shape [batch, >= {context_len}, dim], got {tuple(history_emb.shape)}."
        )
    deriv_seq = history_emb[:, -context_len:]
    components = [deriv_seq[:, -1]]
    for _ in range(markov_deriv):
        deriv_seq = deriv_seq[:, 1:] - deriv_seq[:, :-1]
        components.append(deriv_seq[:, -1])
    state = torch.cat(components, dim=-1)
    return state[0] if squeeze else state


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def preprocess_pixels(
    pixels: np.ndarray,
    *,
    img_size: int,
    device: torch.device,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = F.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (tensor - pixel_mean) / pixel_std


@torch.no_grad()
def encode_dataset_pixels(
    model: torch.nn.Module,
    h5: h5py.File,
    *,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
) -> np.ndarray:
    num_rows = int(h5["pixels"].shape[0])
    latents = np.empty((num_rows, embed_dim), dtype=np.float32)
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    iterator = tqdm(range(0, num_rows, frame_batch_size), desc="Encoding frames", unit="batch")
    for start in iterator:
        stop = min(start + frame_batch_size, num_rows)
        batch = preprocess_pixels(
            np.asarray(h5["pixels"][start:stop], dtype=np.uint8),
            img_size=img_size,
            device=device,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents[start:stop] = emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32)
    return latents


def compute_action_stats(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = actions[~np.isnan(actions).any(axis=1)]
    if finite.size == 0:
        raise ValueError("No finite actions found in dataset.")
    action_mean = finite.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(finite.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
    return action_mean, action_std


def discounted_reverse_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    running = 0.0
    for idx in range(values.shape[0] - 1, -1, -1):
        running = float(values[idx]) + float(discount) * running
        out[idx] = running
    return out


def build_goal_value_cache(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    model_dir: Path,
    cache_path: Path,
    device: torch.device,
    frame_batch_size: int,
    discount: float,
    task_weight: float,
    control_weight: float,
) -> dict[str, Any]:
    config = load_config(model_dir)
    embed_dim = int(config.get("embed_dim", 12))
    img_size = int(config.get("img_size", 224))
    markov_deriv = int(config.get("markov_deriv", 1))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    action_dim = int(config.get("action_dim", 3))

    if frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")
    if not 0.0 < discount <= 1.0:
        raise ValueError("--discount must be in (0, 1].")

    log("Loading frozen rope world model.")
    model = load_world_model(checkpoint_path, device)
    encode_start = time.perf_counter()
    with h5py.File(dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]}.")
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        task_targets = np.asarray(h5["task_target"][:], dtype=np.float32)
        actions_raw = np.asarray(h5["action"][:], dtype=np.float32)
        action_mean, action_std = compute_action_stats(actions_raw)
        actions_norm = (np.nan_to_num(actions_raw, nan=0.0) - action_mean) / action_std
        latents = encode_dataset_pixels(
            model,
            h5,
            device=device,
            img_size=img_size,
            embed_dim=embed_dim,
            frame_batch_size=frame_batch_size,
        )
    del model

    context_len = required_markov_history(markov_deriv)
    num_rows = latents.shape[0]
    states = np.empty((num_rows, markov_state_dim), dtype=np.float32)
    for ep_idx, length in enumerate(tqdm(ep_len.tolist(), desc="Building Markov states", unit="episode")):
        offset = int(ep_offset[ep_idx])
        for local_t in range(int(length)):
            rows = offset + np.maximum(local_t - np.arange(context_len - 1, -1, -1), 0)
            history = torch.from_numpy(latents[rows].astype(np.float32))
            states[offset + local_t] = build_markov_state(history, markov_deriv).numpy().astype(np.float32)

    num_episodes = int(ep_len.shape[0])
    samples_x: list[np.ndarray] = []
    samples_goal: list[np.ndarray] = []
    samples_target: list[np.ndarray] = []
    samples_episode_idx: list[np.ndarray] = []
    samples_step_idx: list[np.ndarray] = []

    episode_iterator = tqdm(range(num_episodes), desc="Building goal-conditioned values", unit="episode")
    for ep_idx in episode_iterator:
        offset = int(ep_offset[ep_idx])
        length = int(ep_len[ep_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        goal_task = task_targets[rows[-1]]
        goal_state = states[rows[-1]]
        task_error = np.sum((task_targets[rows] - goal_task[None, :]) ** 2, axis=-1).astype(np.float64)
        control_cost = np.sum(actions_norm[rows] ** 2, axis=-1).astype(np.float64)
        control_cost[-1] = 0.0
        stage_cost = float(task_weight) * task_error + float(control_weight) * control_cost
        returns = discounted_reverse_cumsum(stage_cost, float(discount)).astype(np.float32)

        samples_x.append(states[rows].astype(np.float32))
        samples_goal.append(np.broadcast_to(goal_state, (length, goal_state.shape[0])).astype(np.float32))
        samples_target.append(returns.reshape(-1, 1))
        samples_episode_idx.append(np.full((length, 1), ep_idx, dtype=np.int64))
        samples_step_idx.append(np.arange(length, dtype=np.int64).reshape(-1, 1))

    x = np.concatenate(samples_x, axis=0)
    goal = np.concatenate(samples_goal, axis=0)
    values = np.concatenate(samples_target, axis=0)
    episode_idx = np.concatenate(samples_episode_idx, axis=0)
    step_idx = np.concatenate(samples_step_idx, axis=0)
    features = np.concatenate((x, goal), axis=-1).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "features": torch.from_numpy(features),
        "state": torch.from_numpy(x),
        "goal_state": torch.from_numpy(goal),
        "target_value": torch.from_numpy(values),
        "episode_idx": torch.from_numpy(episode_idx.reshape(-1)),
        "step_idx": torch.from_numpy(step_idx.reshape(-1)),
        "metadata": {
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint_path": str(checkpoint_path),
            "encode_seconds": float(time.perf_counter() - encode_start),
            "num_rows_source": int(num_rows),
            "num_samples": int(features.shape[0]),
            "num_episodes": int(num_episodes),
            "embed_dim": int(embed_dim),
            "img_size": int(img_size),
            "markov_deriv": int(markov_deriv),
            "markov_state_dim": int(markov_state_dim),
            "action_dim": int(action_dim),
            "discount": float(discount),
            "task_weight": float(task_weight),
            "control_weight": float(control_weight),
            "target_definition": (
                "discounted return to episode final goal: "
                "sum_k gamma^(k-t) * (task_weight * ||task_target_k - task_target_goal||_2^2 "
                "+ control_weight * ||normalized_action_k||_2^2)"
            ),
            "action_convention": "world-model normalized action: (raw_action - action_mean) / action_std",
            "action_mean": action_mean.reshape(-1).astype(np.float32),
            "action_std": action_std.reshape(-1).astype(np.float32),
        },
    }
    torch.save(payload, cache_path)
    log(f"Saved encoded goal-value cache to {cache_path}.")
    return payload


class GoalValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "huber":
        return F.huber_loss(pred, target, delta=float(huber_delta))
    raise ValueError(f"Unsupported loss: {loss_name}")


@torch.no_grad()
def evaluate_regression(
    model: GoalValueMLP,
    features: torch.Tensor,
    targets_norm: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    preds_norm_chunks: list[torch.Tensor] = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device=device, dtype=torch.float32)
        preds_norm_chunks.append(model(batch).cpu())
    preds_norm = torch.cat(preds_norm_chunks, dim=0)
    targets_norm = targets_norm.cpu()
    preds_raw = preds_norm * target_std.cpu() + target_mean.cpu()
    targets_raw = targets_norm * target_std.cpu() + target_mean.cpu()
    mse_norm = float(F.mse_loss(preds_norm, targets_norm).item())
    mae_norm = float(F.l1_loss(preds_norm, targets_norm).item())
    mse_raw = float(F.mse_loss(preds_raw, targets_raw).item())
    mae_raw = float(F.l1_loss(preds_raw, targets_raw).item())
    ss_res = float(torch.sum((preds_raw - targets_raw) ** 2).item())
    ss_tot = float(torch.sum((targets_raw - torch.mean(targets_raw)) ** 2).item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {
        "mse_norm": mse_norm,
        "mae_norm": mae_norm,
        "mse_raw": mse_raw,
        "mae_raw": mae_raw,
        "r2_raw": r2,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not 0.0 < float(args.train_episode_frac) < 1.0:
        raise ValueError("--train-episode-frac must be between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = require_device(args.device)
    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded_data_path = (
        args.encoded_data_path.expanduser().resolve()
        if args.encoded_data_path is not None
        else out_dir / "encoded_goal_value_pairs.pt"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if encoded_data_path.is_file() and not args.force_reencode:
        log(f"Loading encoded goal-value cache from {encoded_data_path}.")
        cache = torch.load(encoded_data_path, map_location="cpu", weights_only=False)
    else:
        cache = build_goal_value_cache(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            model_dir=model_dir,
            cache_path=encoded_data_path,
            device=device,
            frame_batch_size=int(args.frame_batch_size),
            discount=float(args.discount),
            task_weight=float(args.task_weight),
            control_weight=float(args.control_weight),
        )

    features = cache["features"].float()
    targets = cache["target_value"].float().reshape(-1)
    episode_idx = cache["episode_idx"].long().reshape(-1)
    metadata = dict(cache.get("metadata", {}))

    if features.ndim != 2:
        raise ValueError(f"Expected cached features with shape [N, D], got {tuple(features.shape)}.")
    if targets.ndim != 1:
        raise ValueError(f"Expected cached targets with shape [N], got {tuple(targets.shape)}.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("Feature and target counts do not match.")

    num_episodes = int(metadata["num_episodes"])
    train_episode_count = max(1, int(np.floor(num_episodes * float(args.train_episode_frac))))
    train_episode_count = min(train_episode_count, num_episodes - 1)
    train_mask = episode_idx < train_episode_count
    val_mask = ~train_mask
    if not bool(torch.any(train_mask)):
        raise ValueError("Episode split produced no training samples.")
    if not bool(torch.any(val_mask)):
        raise ValueError("Episode split produced no validation samples.")

    train_features = features[train_mask]
    val_features = features[val_mask]
    train_targets = targets[train_mask]
    val_targets = targets[val_mask]

    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-6)
    target_mean = train_targets.mean()
    target_std = train_targets.std().clamp_min(1e-6)

    train_features_norm = (train_features - feature_mean) / feature_std
    val_features_norm = (val_features - feature_mean) / feature_std
    train_targets_norm = (train_targets - target_mean) / target_std
    val_targets_norm = (val_targets - target_mean) / target_std

    train_dataset = TensorDataset(train_features_norm, train_targets_norm)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(int(args.batch_size), len(train_dataset)),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = GoalValueMLP(
        input_dim=int(train_features_norm.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log(
        f"Training goal value model on first {train_episode_count} / {num_episodes} episodes "
        f"({int(train_features.shape[0])} train samples, {int(val_features.shape[0])} val samples)."
    )
    best_val_mae = float("inf")
    best_state_dict = None
    for epoch in range(int(args.epochs)):
        model.train()
        train_losses = []
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for batch_features, batch_targets in iterator:
            batch_features = batch_features.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            batch_targets = batch_targets.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            preds = model(batch_features)
            loss = regression_loss(
                preds,
                batch_targets,
                loss_name=str(args.loss),
                huber_delta=float(args.huber_delta),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            train_losses.append(loss_value)
            iterator.set_postfix(loss=f"{loss_value:.4g}")

        metrics = evaluate_regression(
            model,
            val_features_norm,
            val_targets_norm,
            batch_size=int(args.batch_size),
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        log(
            f"epoch={epoch + 1} train_loss={float(np.mean(train_losses)):.6g} "
            f"val_mae_raw={metrics['mae_raw']:.6g} val_r2_raw={metrics['r2_raw']:.6g}"
        )
        if metrics["mae_raw"] < best_val_mae:
            best_val_mae = metrics["mae_raw"]
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_metrics = evaluate_regression(
        model,
        train_features_norm,
        train_targets_norm,
        batch_size=int(args.batch_size),
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_metrics = evaluate_regression(
        model,
        val_features_norm,
        val_targets_norm,
        batch_size=int(args.batch_size),
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    artifact = {
        "state_dict": model.state_dict(),
        "model_class": "GoalValueMLP",
        "input_dim": int(train_features_norm.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "feature_mean": feature_mean.numpy().astype(np.float32),
        "feature_std": feature_std.numpy().astype(np.float32),
        "target_mean": float(target_mean.item()),
        "target_std": float(target_std.item()),
        "prediction_semantics": {
            "name": "goal_conditioned_cost_to_go",
            "lower_is_better": True,
            "input": "[current_markov_state, goal_markov_state]",
            "target_definition": metadata["target_definition"],
        },
        "training": {
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint_path": str(checkpoint_path),
            "encoded_data_path": str(encoded_data_path),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "train_episode_frac": float(args.train_episode_frac),
            "train_episode_count": int(train_episode_count),
            "val_episode_count": int(num_episodes - train_episode_count),
            "train_samples": int(train_features.shape[0]),
            "val_samples": int(val_features.shape[0]),
            "loss": str(args.loss),
            "huber_delta": float(args.huber_delta),
            "precision": "float32",
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
        "source_metadata": metadata,
    }
    artifact_path = out_dir / "model.pt"
    torch.save(artifact, artifact_path)

    summary = {
        "artifact_path": str(artifact_path),
        "encoded_data_path": str(encoded_data_path),
        "train_episode_count": int(train_episode_count),
        "val_episode_count": int(num_episodes - train_episode_count),
        "train_samples": int(train_features.shape[0]),
        "val_samples": int(val_features.shape[0]),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log(f"Saved goal value artifact to {artifact_path}.")


if __name__ == "__main__":
    main()
