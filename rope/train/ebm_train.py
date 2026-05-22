#!/usr/bin/env python3
"""Encode rope data into latent Markov-state/action pairs and train a DSM EBM."""

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
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

DEFAULT_DATASET_PATH = "rope/data/train_data_noshadow.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow_ft"
DEFAULT_OUT_DIR = "rope/models/latent_ebm"


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
    parser.add_argument("--train-split", type=float, default=0.98)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=0.10,
        help="Optional upper bound for log-uniform sigma sampling. If omitted, uses fixed --sigma.",
    )
    parser.add_argument(
        "--energy-quantile",
        type=float,
        default=0.99,
        help="Clean validation energy quantile saved as an in-distribution threshold.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[rope_ebm_train] {message}", flush=True)


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


def build_pair_cache(
    *,
    dataset_path: Path,
    checkpoint_path: Path,
    model_dir: Path,
    cache_path: Path,
    device: torch.device,
    frame_batch_size: int,
) -> dict[str, Any]:
    config = load_config(model_dir)
    embed_dim = int(config.get("embed_dim", 12))
    img_size = int(config.get("img_size", 224))
    markov_deriv = int(config.get("markov_deriv", 1))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    action_dim = int(config.get("action_dim", 3))

    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv, got {markov_deriv}.")
    if frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")

    log("Loading frozen rope world model.")
    model = load_world_model(checkpoint_path, device)
    encode_start = time.perf_counter()
    with h5py.File(dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]}.")
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
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
    valid_mask = np.ones((num_rows,), dtype=bool)
    for ep_idx, length in enumerate(tqdm(ep_len.tolist(), desc="Building Markov states", unit="episode")):
        offset = int(ep_offset[ep_idx])
        for local_t in range(int(length)):
            rows = offset + np.maximum(local_t - np.arange(context_len - 1, -1, -1), 0)
            history = torch.from_numpy(latents[rows].astype(np.float32))
            states[offset + local_t] = build_markov_state(history, markov_deriv).numpy().astype(np.float32)
            if np.isnan(actions_raw[offset + local_t]).any():
                valid_mask[offset + local_t] = False

    states = states[valid_mask]
    actions_norm = actions_norm[valid_mask].astype(np.float32)
    pairs = np.concatenate((states, actions_norm), axis=-1).astype(np.float32)
    if pairs.shape[1] != markov_state_dim + action_dim:
        raise ValueError(f"Unexpected pair shape {pairs.shape}; expected dim {markov_state_dim + action_dim}.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "pairs": torch.from_numpy(pairs),
        "markov_state": torch.from_numpy(states),
        "action": torch.from_numpy(actions_norm),
        "metadata": {
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint_path": str(checkpoint_path),
            "encode_seconds": float(time.perf_counter() - encode_start),
            "num_rows_source": int(num_rows),
            "num_pairs": int(pairs.shape[0]),
            "num_filtered_nan_action_rows": int(num_rows - pairs.shape[0]),
            "embed_dim": int(embed_dim),
            "img_size": int(img_size),
            "markov_deriv": int(markov_deriv),
            "markov_state_dim": int(markov_state_dim),
            "action_dim": int(action_dim),
            "action_convention": "world-model normalized action: (raw_action - action_mean) / action_std",
            "action_mean": action_mean.reshape(-1).astype(np.float32),
            "action_std": action_std.reshape(-1).astype(np.float32),
        },
    }
    torch.save(payload, cache_path)
    log(f"Saved encoded pair cache to {cache_path}.")
    return payload


class EnergyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sample_sigma(batch_size: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    sigma_min = float(args.sigma)
    if sigma_min <= 0.0:
        raise ValueError("--sigma must be positive.")
    if args.sigma_max is None:
        return torch.full((batch_size, 1), sigma_min, dtype=torch.float32, device=device)
    sigma_max = float(args.sigma_max)
    if sigma_max < sigma_min:
        raise ValueError("--sigma-max must be greater than or equal to --sigma.")
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    return torch.empty((batch_size, 1), dtype=torch.float32, device=device).uniform_(log_min, log_max).exp_()


def dsm_loss(model: EnergyMLP, clean: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    sigma = sample_sigma(clean.shape[0], args, clean.device)
    noise = torch.randn_like(clean) * sigma
    noisy = (clean + noise).detach().requires_grad_(True)
    energy = model(noisy).sum()
    grad_energy = torch.autograd.grad(energy, noisy, create_graph=True)[0]
    target_score = -noise / sigma.square()
    predicted_score = -grad_energy
    return (predicted_score - target_score).square().sum(dim=-1).mean()


@torch.no_grad()
def evaluate_energy(
    model: EnergyMLP,
    data: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size].to(device)
        chunks.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not 0.0 < float(args.train_split) < 1.0:
        raise ValueError("--train-split must be between 0 and 1.")
    if not 0.0 < float(args.energy_quantile) < 1.0:
        raise ValueError("--energy-quantile must be between 0 and 1.")
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
        else out_dir / "encoded_markov_action_pairs.pt"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if encoded_data_path.is_file() and not args.force_reencode:
        log(f"Loading encoded pair cache from {encoded_data_path}.")
        cache = torch.load(encoded_data_path, map_location="cpu", weights_only=False)
    else:
        cache = build_pair_cache(
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            model_dir=model_dir,
            cache_path=encoded_data_path,
            device=device,
            frame_batch_size=int(args.frame_batch_size),
        )

    pairs = cache["pairs"].float()
    if pairs.ndim != 2:
        raise ValueError(f"Expected cached pairs with shape [N, D], got {tuple(pairs.shape)}.")
    if pairs.shape[0] < 2:
        raise ValueError(f"Need at least two pairs for train/validation split, got {pairs.shape[0]}.")

    pair_mean = pairs.mean(dim=0)
    pair_std = pairs.std(dim=0).clamp_min(1e-6)
    pairs_norm = (pairs - pair_mean) / pair_std

    dataset = TensorDataset(pairs_norm)
    train_len = max(1, int(len(dataset) * float(args.train_split)))
    val_len = len(dataset) - train_len
    if val_len < 1:
        val_len = 1
        train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=min(int(args.batch_size), train_len),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = EnergyMLP(
        input_dim=int(pairs_norm.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log(f"Training EBM on {train_len} train / {val_len} validation normalized Markov-state/action pairs.")
    for epoch in range(int(args.epochs)):
        model.train()
        losses = []
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
        for (batch,) in iterator:
            clean = batch.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            loss = dsm_loss(model, clean, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            iterator.set_postfix(loss=f"{loss_value:.4g}")
        log(f"epoch={epoch + 1} train_dsm_loss={float(np.mean(losses)):.6g}")

    val_indices = torch.as_tensor(val_set.indices, dtype=torch.long)
    val_pairs_norm = pairs_norm[val_indices]
    val_energy = evaluate_energy(model, val_pairs_norm, batch_size=int(args.batch_size), device=device)
    energy_threshold = float(np.quantile(val_energy.astype(np.float64), float(args.energy_quantile)))
    log(
        "validation clean energy: "
        f"mean={float(np.mean(val_energy)):.6g} "
        f"std={float(np.std(val_energy)):.6g} "
        f"min={float(np.min(val_energy)):.6g} "
        f"max={float(np.max(val_energy)):.6g} "
        f"q{float(args.energy_quantile):.3f}={energy_threshold:.6g} "
        f"score_threshold={-energy_threshold:.6g}"
    )

    metadata = dict(cache.get("metadata", {}))
    artifact = {
        "state_dict": model.state_dict(),
        "model_class": "EnergyMLP",
        "input_dim": int(pairs_norm.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "pair_mean": pair_mean.numpy().astype(np.float32),
        "pair_std": pair_std.numpy().astype(np.float32),
        "energy_threshold": energy_threshold,
        "energy_quantile": float(args.energy_quantile),
        "score_convention": {
            "energy": "lower is more in-distribution",
            "score": "-energy; larger is more in-distribution",
            "score_threshold": float(-energy_threshold),
        },
        "dsm": {
            "sigma": float(args.sigma),
            "sigma_max": None if args.sigma_max is None else float(args.sigma_max),
            "noise_space": "normalized concatenated tuple [markov_state, normalized_action]",
            "loss": "mean || -grad_x E(x_noisy) - (x_clean - x_noisy) / sigma^2 ||_2^2",
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
            "train_split": float(args.train_split),
            "train_pairs": int(train_len),
            "val_pairs": int(val_len),
            "precision": "float32",
        },
        "source_metadata": metadata,
    }
    artifact_path = out_dir / "model.pt"
    torch.save(artifact, artifact_path)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "artifact_path": str(artifact_path),
                "encoded_data_path": str(encoded_data_path),
                "num_pairs": int(pairs.shape[0]),
                "input_dim": int(pairs_norm.shape[1]),
                "energy_threshold": energy_threshold,
                "score_threshold": float(-energy_threshold),
                "val_energy_mean": float(np.mean(val_energy)),
                "val_energy_std": float(np.std(val_energy)),
                "val_energy_min": float(np.min(val_energy)),
                "val_energy_max": float(np.max(val_energy)),
            },
            handle,
            indent=2,
        )
    log(f"Saved EBM artifact to {artifact_path}.")


if __name__ == "__main__":
    main()
