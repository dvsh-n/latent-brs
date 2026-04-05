import argparse
import functools
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def find_root(start_path: Path, marker: str = ".root") -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import DeepKoopmanImageNoDec


DATA_PATH = "data/reacher_dataset_processed_manifest.pt"
MODEL_SAVE_PATH = "models/reacher_koop_nodec.pt"
LOG_DIR = "runs/reacher_koop_nodec"

IMAGE_CHANNELS = 3
CONTROL_DIM = 2
HISTORY_LENGTH = 4
STATE_DIM = 16
OBSERVABLE_DIM = 48
HIDDEN_WIDTH = 256
HIDDEN_DEPTH = 2
ACTIVATION = "gelu"
IMAGE_SIZE = 128

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-6
EPOCHS = 30
BATCH_SIZE = 128
MULTI_STEP_HORIZON = 30
NUM_WORKERS = 0

LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.25
LAMBDA_STD = 0.0
LAMBDA_COV = 0.0
CHUNK_CACHE_SIZE = 1
PREFETCH_FACTOR = 1
TARGET_ENCODE_CHUNK_SIZE = 256
USE_AMP = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_activation_fn(name: str):
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    return activations[name.lower()]


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(ROOT_DIR / p)


def get_unique_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return str(p)
    counter = 1
    while True:
        candidate = p.parent / f"{p.stem}_{counter}{p.suffix}"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def print_config(config: dict[str, Any], title: str) -> None:
    print("\n" + "=" * 45)
    print(f"      {title}")
    print("=" * 45)
    for key, value in config.items():
        print(f"{key:<24s} | {value}")
    print("=" * 45 + "\n")


def load_processed_manifest(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_tensor_file(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", mmap=mmap, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


class ReacherVideoDataset(Dataset):
    """Samples current frame/history and future targets from processed frame chunks."""

    def __init__(
        self,
        data_path: str,
        history_length: int,
        multi_step_horizon: int,
        chunk_cache_size: int,
        output_image_size: int | None = None,
    ):
        self.data_path = Path(data_path)
        self.history_length = history_length
        self.multi_step_horizon = multi_step_horizon
        self.chunk_cache_size = chunk_cache_size
        self._chunk_cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

        self.manifest = load_processed_manifest(self.data_path)
        self.header = self.manifest["source_header"]
        self.meta = self.manifest["source_meta"]
        self.chunks = self.manifest["chunks"]
        self.control_dim = len(self.header["action_layout"])
        self.source_image_size = int(self.manifest["image_size"])
        self.image_size = self.source_image_size if output_image_size is None else int(output_image_size)
        self.samples, self.chunk_sample_indices = self._build_samples()

    def _build_samples(self) -> tuple[list[tuple[int, int, int]], list[list[int]]]:
        samples: list[tuple[int, int, int]] = []
        chunk_sample_indices: list[list[int]] = []
        h = self.history_length
        m = self.multi_step_horizon
        for chunk_idx, chunk_info in enumerate(self.chunks):
            chunk_indices: list[int] = []
            num_actions = int(chunk_info["num_actions"])
            max_timestep = num_actions - m
            for local_traj_idx in range(int(chunk_info["num_trajectories"])):
                for t in range(h - 1, max_timestep + 1):
                    chunk_indices.append(len(samples))
                    samples.append((chunk_idx, local_traj_idx, t))
            chunk_sample_indices.append(chunk_indices)
        return samples, chunk_sample_indices

    def _load_chunk(self, chunk_idx: int) -> dict[str, torch.Tensor]:
        if chunk_idx in self._chunk_cache:
            chunk = self._chunk_cache.pop(chunk_idx)
            self._chunk_cache[chunk_idx] = chunk
            return chunk

        chunk_path = Path(self.chunks[chunk_idx]["chunk_path"])
        chunk = load_tensor_file(chunk_path, mmap=True)
        history_windows = chunk["frames"].unfold(1, self.history_length, 1).permute(0, 1, 5, 2, 3, 4)
        chunk["history_windows"] = history_windows
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
        history_start = timestep - self.history_length + 1

        frame_k = frames[timestep].float().mul_(1.0 / 255.0)
        history_k = chunk["history_windows"][local_traj_idx, history_start].float().mul_(1.0 / 255.0)
        frame_next_seq = frames[timestep + 1:timestep + 1 + self.multi_step_horizon].float().mul_(1.0 / 255.0)
        history_next_seq = (
            chunk["history_windows"][local_traj_idx, history_start + 1:history_start + 1 + self.multi_step_horizon]
            .float()
            .mul_(1.0 / 255.0)
        )

        return {
            "frame_k": frame_k,
            "history_k": history_k,
            "u_seq": actions[timestep:timestep + self.multi_step_horizon],
            "frame_next_seq": frame_next_seq,
            "history_next_seq": history_next_seq,
        }


def resize_frame_batch(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    if frames.shape[-2:] == (image_size, image_size):
        return frames
    return F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)


def resize_history_batch(history: torch.Tensor, image_size: int) -> torch.Tensor:
    if history.shape[-2:] == (image_size, image_size):
        return history
    leading_shape = history.shape[:-3]
    channels = history.shape[-3]
    resized = F.interpolate(
        history.reshape(-1, channels, history.shape[-2], history.shape[-1]),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(*leading_shape, channels, image_size, image_size)


def collate_reacher_batch(batch: list[dict[str, torch.Tensor]], image_size: int) -> dict[str, torch.Tensor]:
    collated = {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in batch[0]
    }
    collated["frame_k"] = resize_frame_batch(collated["frame_k"], image_size)
    collated["history_k"] = resize_history_batch(collated["history_k"], image_size)
    collated["frame_next_seq"] = resize_history_batch(collated["frame_next_seq"], image_size)
    collated["history_next_seq"] = resize_history_batch(collated["history_next_seq"], image_size)
    return collated


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


def resolve_num_workers(requested_num_workers: int, dataset: ReacherVideoDataset) -> int:
    if requested_num_workers >= 0:
        return requested_num_workers
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 2, len(dataset.chunks)))


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


def latent_regularizer(s_target_seq: torch.Tensor, z_target_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s_flat = s_target_seq.reshape(-1, s_target_seq.shape[-1])
    z_flat = z_target_seq.reshape(-1, z_target_seq.shape[-1])
    std_loss = variance_loss(s_flat) + variance_loss(z_flat)
    cov_loss = covariance_loss(s_flat) + covariance_loss(z_flat)
    return std_loss, cov_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only two-encoder decoder-free Koopman model for Reacher.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--image_channels", type=int, default=IMAGE_CHANNELS)
    parser.add_argument("--control_dim", type=int, default=CONTROL_DIM)
    parser.add_argument("--history_length", type=int, default=HISTORY_LENGTH)
    parser.add_argument("--state_dim", type=int, default=STATE_DIM)
    parser.add_argument("--observable_dim", type=int, default=OBSERVABLE_DIM)
    parser.add_argument("--hidden_width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--hidden_depth", type=int, default=HIDDEN_DEPTH)
    parser.add_argument("--activation", type=str, default=ACTIVATION, choices=["relu", "gelu", "tanh"])
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min_lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--multi_step_horizon", type=int, default=MULTI_STEP_HORIZON)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lambda_state", type=float, default=LAMBDA_STATE)
    parser.add_argument("--lambda_latent", type=float, default=LAMBDA_LATENT)
    parser.add_argument("--lambda_std", type=float, default=LAMBDA_STD)
    parser.add_argument("--lambda_cov", type=float, default=LAMBDA_COV)
    parser.add_argument("--chunk_cache_size", type=int, default=CHUNK_CACHE_SIZE)
    parser.add_argument("--prefetch_factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--target_encode_chunk_size", type=int, default=TARGET_ENCODE_CHUNK_SIZE)
    parser.add_argument("--use_amp", type=str2bool, default=USE_AMP)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    data_path = resolve_path(args.data_path)
    model_save_path = get_unique_path(resolve_path(args.model_save_path))
    log_dir = get_unique_path(resolve_path(args.log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    activation_fn = get_activation_fn(args.activation)
    dataset = ReacherVideoDataset(
        data_path=data_path,
        history_length=args.history_length,
        multi_step_horizon=args.multi_step_horizon,
        chunk_cache_size=args.chunk_cache_size,
        output_image_size=args.image_size,
    )
    num_workers = resolve_num_workers(args.num_workers, dataset)
    batch_sampler = ChunkBatchSampler(dataset.chunk_sample_indices, batch_size=args.batch_size)
    use_amp = args.use_amp and device.type == "cuda"
    collate_fn = functools.partial(collate_reacher_batch, image_size=dataset.image_size)
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
    dataloader = DataLoader(**dataloader_kwargs)
    writer = SummaryWriter(log_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_config = {
        "image_channels": args.image_channels,
        "control_dim": args.control_dim,
        "history_length": args.history_length,
        "state_dim": args.state_dim,
        "observable_dim": args.observable_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
        "activation_fn": activation_fn,
    }
    model = DeepKoopmanImageNoDec(**model_config).to(device)

    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.min_lr,
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting training for image-only decoder-free Koopman model (Reacher)")
    print_config(
        {
            **model_config,
            "latent_dim": args.state_dim + args.observable_dim,
            "image_size": dataset.image_size,
            "source_image_size": dataset.source_image_size,
        },
        title="Model Configuration",
    )
    training_config = {
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "multi_step_horizon": args.multi_step_horizon,
        "target_encode_chunk_size": args.target_encode_chunk_size,
        "use_amp": use_amp,
        "num_workers": num_workers,
        "prefetch_factor": max(1, args.prefetch_factor) if num_workers > 0 else None,
        "lambda_state": args.lambda_state,
        "lambda_latent": args.lambda_latent,
        "lambda_std": args.lambda_std,
        "lambda_cov": args.lambda_cov,
        "chunk_cache_size": args.chunk_cache_size,
    }
    print_config(training_config, title="Training Configuration")

    global_step = 0
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for _ in pbar:
        for batch in dataloader:
            frame_k = batch["frame_k"].to(device, non_blocking=True)
            history_k = batch["history_k"].to(device, non_blocking=True)
            u_seq = batch["u_seq"].to(device, non_blocking=True)
            frame_next_seq = batch["frame_next_seq"].to(device, non_blocking=True)
            history_next_seq = batch["history_next_seq"].to(device, non_blocking=True)
            target_requires_grad = (args.lambda_std > 0.0) or (args.lambda_cov > 0.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z_pred_seq, s_pred_seq, z_target_seq, s_target_seq = model(
                    frame_k=frame_k,
                    history_k=history_k,
                    u_seq=u_seq,
                    frame_next_seq=frame_next_seq,
                    history_next_seq=history_next_seq,
                    target_encode_chunk_size=args.target_encode_chunk_size,
                    target_requires_grad=target_requires_grad,
                )

                z_target_detached = z_target_seq.detach()
                s_target_detached = s_target_seq.detach()

                loss_state = F.mse_loss(s_pred_seq, s_target_detached)
                loss_latent = F.mse_loss(z_pred_seq, z_target_detached)
                loss_std, loss_cov = latent_regularizer(s_target_seq, z_target_seq)
                loss = (
                    args.lambda_state * loss_state
                    + args.lambda_latent * loss_latent
                    + args.lambda_std * loss_std
                    + args.lambda_cov * loss_cov
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % 10 == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/State", loss_state.item(), global_step)
                writer.add_scalar("Loss/Latent", loss_latent.item(), global_step)
                writer.add_scalar("Loss/Std", loss_std.item(), global_step)
                writer.add_scalar("Loss/Cov", loss_cov.item(), global_step)
                writer.add_scalar("Hyperparams/Learning_Rate", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", state=f"{loss_state.item():.4f}", latent=f"{loss_latent.item():.4f}")
            global_step += 1

    writer.close()

    model_config_save = model_config.copy()
    model_config_save["activation_fn"] = args.activation
    save_data = {
        "model_config": model_config_save,
        "training_config": training_config,
        "dataset_header": dataset.header,
        "state_dict": model.state_dict(),
    }
    torch.save(save_data, model_save_path)

    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    train(parse_args())
