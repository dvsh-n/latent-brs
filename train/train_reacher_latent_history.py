#!/usr/bin/env python3
"""Train a single-frame encoder with latent-history inverse and forward dynamics heads."""

from __future__ import annotations

import argparse
import functools
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def find_root(start_path: Path, marker: str = ".root") -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

from models import SingleFrameLatentDynamicsModel  # noqa: E402


DATA_PATH = "data/reacher_dataset_processed_manifest.pt"
SAVE_PATH = "models/reacher_latent_history.pt"
LOG_DIR = "runs/reacher_latent_history"

IMAGE_CHANNELS = 3
CONTROL_DIM = 2
STATE_DIM = 32
HISTORY_LENGTH = 8
IMAGE_SIZE = 128
ENCODER_FEATURE_DIM = 768
ENCODER_BASE_CHANNELS = 64
ADAPTER_HIDDEN_WIDTH = 512
DYNAMICS_HIDDEN_WIDTH = 512
DYNAMICS_DEPTH = 3
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 40
BATCH_SIZE = 512
NUM_WORKERS = 16
CHUNK_CACHE_SIZE = 2
PREFETCH_FACTOR = 4
VAL_FRACTION = 0.05
LAMBDA_FORWARD = 1.0
LAMBDA_STD = 1e-2
LAMBDA_COV = 1e-3
AMP_DTYPE = "bfloat16"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_activation_fn(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    return activations[name.lower()]


def get_amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_manifest(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_tensor_file(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


class ReacherLatentHistoryDataset(Dataset):
    """Samples frame windows of length H + 1 and the action at time t."""

    def __init__(
        self,
        manifest_path: Path,
        history_length: int,
        chunk_indices: list[int],
        chunk_cache_size: int,
        output_image_size: int | None = None,
    ):
        self.manifest_path = manifest_path
        self.history_length = history_length
        self.chunk_cache_size = chunk_cache_size
        self._chunk_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

        self.manifest = load_manifest(manifest_path)
        self.header = self.manifest["source_header"]
        self.chunks = [self.manifest["chunks"][idx] for idx in chunk_indices]
        self.control_dim = len(self.header["action_layout"])
        self.source_image_size = int(self.manifest["image_size"])
        self.image_size = self.source_image_size if output_image_size is None else int(output_image_size)
        self.samples, self.chunk_sample_indices = self._build_samples()

    def _build_samples(self) -> tuple[list[tuple[int, int, int]], list[list[int]]]:
        samples: list[tuple[int, int, int]] = []
        chunk_sample_indices: list[list[int]] = []
        for chunk_idx, chunk_info in enumerate(self.chunks):
            per_chunk: list[int] = []
            num_actions = int(chunk_info["num_actions"])
            num_trajectories = int(chunk_info["num_trajectories"])
            for local_traj_idx in range(num_trajectories):
                for timestep in range(num_actions):
                    per_chunk.append(len(samples))
                    samples.append((chunk_idx, local_traj_idx, timestep))
            chunk_sample_indices.append(per_chunk)
        return samples, chunk_sample_indices

    def _load_chunk(self, chunk_idx: int) -> dict[str, torch.Tensor]:
        if chunk_idx in self._chunk_cache:
            chunk = self._chunk_cache.pop(chunk_idx)
            self._chunk_cache[chunk_idx] = chunk
            return chunk

        chunk_path = Path(self.chunks[chunk_idx]["chunk_path"])
        chunk = load_tensor_file(chunk_path, mmap=True)
        self._chunk_cache[chunk_idx] = chunk
        while len(self._chunk_cache) > self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)
        return chunk

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk_idx, local_traj_idx, timestep = self.samples[idx]
        chunk = self._load_chunk(chunk_idx)
        frames = chunk["frames"][local_traj_idx]
        actions = chunk["actions"][local_traj_idx].float()

        start = timestep - self.history_length + 1
        indices = torch.arange(start, timestep + 2, dtype=torch.long)
        indices.clamp_(0, frames.shape[0] - 1)
        frame_window = frames.index_select(0, indices)

        return {
            "frame_window": frame_window,
            "action": actions[timestep],
        }


class ChunkBatchSampler:
    """Shuffle by chunk, then batch contiguous indices within each chunk for locality."""

    def __init__(self, chunk_sample_indices: list[list[int]], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.chunk_sample_indices = chunk_sample_indices
        self.batch_size = batch_size

    def __iter__(self):
        chunk_order = torch.randperm(len(self.chunk_sample_indices)).tolist()
        for chunk_idx in chunk_order:
            sample_indices = self.chunk_sample_indices[chunk_idx]
            if not sample_indices:
                continue
            order = torch.randperm(len(sample_indices)).tolist()
            shuffled = [sample_indices[i] for i in order]
            for start in range(0, len(shuffled), self.batch_size):
                yield shuffled[start:start + self.batch_size]

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.chunk_sample_indices if indices)


def collate_batch(batch: list[dict[str, torch.Tensor]], image_size: int) -> dict[str, torch.Tensor]:
    frame_windows = torch.stack([sample["frame_window"] for sample in batch], dim=0)
    actions = torch.stack([sample["action"] for sample in batch], dim=0)
    if frame_windows.shape[-2:] != (image_size, image_size):
        b, t, c, h, w = frame_windows.shape
        resized = F.interpolate(
            frame_windows.reshape(b * t, c, h, w).float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        frame_windows = resized.reshape(b, t, c, image_size, image_size).round().clamp_(0, 255).to(torch.uint8)
    return {
        "frame_window": frame_windows,
        "action": actions,
    }


def preprocess_frames(frame_windows: torch.Tensor) -> torch.Tensor:
    return frame_windows.float().div_(255.0)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    return off_diagonal(cov).pow(2).mean()


def split_encoded_histories(state_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_t = state_window[:, :-1, :]
    history_tp1 = state_window[:, 1:, :]
    next_state = state_window[:, -1, :]
    return history_t, history_tp1, next_state


def run_epoch(
    *,
    model: SingleFrameLatentDynamicsModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    amp_dtype: torch.dtype,
    lambda_forward: float,
    lambda_std: float,
    lambda_cov: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_inv_loss = 0.0
    total_fwd_loss = 0.0
    total_std_loss = 0.0
    total_cov_loss = 0.0
    total_examples = 0

    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    progress = tqdm(loader, dynamic_ncols=True, leave=False)
    for batch in progress:
        frame_windows = preprocess_frames(batch["frame_window"])
        actions = batch["action"]

        b, t, c, h, w = frame_windows.shape
        frame_windows = frame_windows.view(b * t, c, h, w).to(device=device, non_blocking=True)
        actions = actions.to(device=device, non_blocking=True)
        frame_windows = frame_windows.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_dtype = amp_dtype if use_amp else torch.float32
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            encoded = model.encode_frames(frame_windows)
            state_window = encoded.view(b, t, model.state_dim)
            history_t, history_tp1, next_state = split_encoded_histories(state_window)

            pred_action = model.predict_action(history_t, history_tp1)
            pred_next_state = model.predict_next_state(history_t, actions)

            inv_loss = F.mse_loss(pred_action, actions)
            fwd_loss = F.mse_loss(pred_next_state, next_state.detach())
            state_flat = state_window.reshape(-1, model.state_dim)
            std_loss = variance_loss(state_flat)
            cov_loss = covariance_loss(state_flat)
            loss = inv_loss + lambda_forward * fwd_loss + lambda_std * std_loss + lambda_cov * cov_loss

        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = actions.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_inv_loss += float(inv_loss.detach()) * batch_size
        total_fwd_loss += float(fwd_loss.detach()) * batch_size
        total_std_loss += float(std_loss.detach()) * batch_size
        total_cov_loss += float(cov_loss.detach()) * batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            inv=f"{total_inv_loss / total_examples:.4f}",
            fwd=f"{total_fwd_loss / total_examples:.4f}",
        )

    return {
        "loss": total_loss / max(total_examples, 1),
        "inverse_loss": total_inv_loss / max(total_examples, 1),
        "forward_loss": total_fwd_loss / max(total_examples, 1),
        "std_loss": total_std_loss / max(total_examples, 1),
        "cov_loss": total_cov_loss / max(total_examples, 1),
    }


def build_dataloader(
    dataset: ReacherLatentHistoryDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    batch_sampler = ChunkBatchSampler(dataset.chunk_sample_indices, batch_size=batch_size)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "collate_fn": functools.partial(collate_batch, image_size=dataset.image_size),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--image-channels", type=int, default=IMAGE_CHANNELS)
    parser.add_argument("--control-dim", type=int, default=CONTROL_DIM)
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--history-length", type=int, default=HISTORY_LENGTH)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--encoder-feature-dim", type=int, default=ENCODER_FEATURE_DIM)
    parser.add_argument("--encoder-base-channels", type=int, default=ENCODER_BASE_CHANNELS)
    parser.add_argument("--adapter-hidden-width", type=int, default=ADAPTER_HIDDEN_WIDTH)
    parser.add_argument("--dynamics-hidden-width", type=int, default=DYNAMICS_HIDDEN_WIDTH)
    parser.add_argument("--dynamics-depth", type=int, default=DYNAMICS_DEPTH)
    parser.add_argument("--activation", choices=("relu", "gelu", "silu"), default="gelu")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--chunk-cache-size", type=int, default=CHUNK_CACHE_SIZE)
    parser.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--lambda-forward", type=float, default=LAMBDA_FORWARD)
    parser.add_argument("--lambda-std", type=float, default=LAMBDA_STD)
    parser.add_argument("--lambda-cov", type=float, default=LAMBDA_COV)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16", "float32"), default=AMP_DTYPE)
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_length <= 0:
        raise ValueError("history-length must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("val-fraction must be in [0, 1)")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    manifest_path = Path(args.data_path)
    manifest = load_manifest(manifest_path)
    num_chunks = len(manifest["chunks"])
    if num_chunks == 0:
        raise RuntimeError(f"No chunks found in {manifest_path}")

    val_chunks = max(1, int(round(num_chunks * args.val_fraction))) if args.val_fraction > 0.0 else 0
    train_chunk_indices = list(range(0, num_chunks - val_chunks))
    val_chunk_indices = list(range(num_chunks - val_chunks, num_chunks))
    if not train_chunk_indices:
        raise RuntimeError("Validation split consumed all chunks; reduce --val-fraction")

    train_dataset = ReacherLatentHistoryDataset(
        manifest_path=manifest_path,
        history_length=args.history_length,
        chunk_indices=train_chunk_indices,
        chunk_cache_size=args.chunk_cache_size,
        output_image_size=args.image_size,
    )
    val_dataset = None
    if val_chunk_indices:
        val_dataset = ReacherLatentHistoryDataset(
            manifest_path=manifest_path,
            history_length=args.history_length,
            chunk_indices=val_chunk_indices,
            chunk_cache_size=max(1, min(args.chunk_cache_size, len(val_chunk_indices))),
            output_image_size=args.image_size,
        )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )

    activation_fn = get_activation_fn(args.activation)
    model = SingleFrameLatentDynamicsModel(
        image_channels=args.image_channels,
        control_dim=args.control_dim,
        state_dim=args.state_dim,
        history_length=args.history_length,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_base_channels=args.encoder_base_channels,
        adapter_hidden_width=args.adapter_hidden_width,
        dynamics_hidden_width=args.dynamics_hidden_width,
        dynamics_depth=args.dynamics_depth,
        activation_fn=activation_fn,
    ).to(device=device)
    model = model.to(memory_format=torch.channels_last)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    amp_dtype = get_amp_dtype(args.amp_dtype)
    scaler = None
    if device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            lambda_forward=args.lambda_forward,
            lambda_std=args.lambda_std,
            lambda_cov=args.lambda_cov,
        )
        scheduler.step()

        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    scaler=None,
                    device=device,
                    amp_dtype=amp_dtype,
                    lambda_forward=args.lambda_forward,
                    lambda_std=args.lambda_std,
                    lambda_cov=args.lambda_cov,
                )

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/inverse_loss", train_metrics["inverse_loss"], epoch)
        writer.add_scalar("train/forward_loss", train_metrics["forward_loss"], epoch)
        writer.add_scalar("train/std_loss", train_metrics["std_loss"], epoch)
        writer.add_scalar("train/cov_loss", train_metrics["cov_loss"], epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        if val_metrics is not None:
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/inverse_loss", val_metrics["inverse_loss"], epoch)
            writer.add_scalar("val/forward_loss", val_metrics["forward_loss"], epoch)

        metric_source = val_metrics if val_metrics is not None else train_metrics
        if metric_source["loss"] < best_val:
            best_val = metric_source["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                save_path,
            )

        summary = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_inverse_loss": round(train_metrics["inverse_loss"], 6),
            "train_forward_loss": round(train_metrics["forward_loss"], 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        if val_metrics is not None:
            summary["val_loss"] = round(val_metrics["loss"], 6)
            summary["val_inverse_loss"] = round(val_metrics["inverse_loss"], 6)
            summary["val_forward_loss"] = round(val_metrics["forward_loss"], 6)
        print(summary)

    writer.close()


if __name__ == "__main__":
    main()
