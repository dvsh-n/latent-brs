#!/usr/bin/env python3
"""Train a latent dynamics model from Reacher image rollouts."""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from shared.models import LatentDynamicsModel

DEFAULT_DATA_DIR = "data/expert_data/prepocessed"
DEFAULT_SAVE_PATH = "models/latent_dyn_reacher.pt"
DEFAULT_LOG_DIR = "runs/latent_dyn_reacher"
MODEL_ARCHITECTURE = "residual_next_latent_dynamics_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--dynamics-hidden-dim", type=int, default=256)
    parser.add_argument("--dynamics-depth", type=int, default=3)
    parser.add_argument("--encoder-proj-hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-proj-depth", type=int, default=1)
    parser.add_argument("--cache-episodes", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--curvature-weight", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReacherLatentDynamicsDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        *,
        history: int,
        rollout_steps: int,
        cache_episodes: int = 32,
    ) -> None:
        self.data_dir = data_dir
        self.obses_dir = data_dir / "obses"
        self.actions = torch.load(data_dir / "actions.pth", map_location="cpu").float()
        self.seq_lengths = torch.load(data_dir / "seq_lengths.pth", map_location="cpu").long()
        self.history = int(history)
        self.rollout_steps = int(rollout_steps)
        self.total_rollout_frames = self.rollout_steps + 1
        self.total_actions = self.rollout_steps
        self.cache_episodes = int(cache_episodes)
        self._episode_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.slices: list[tuple[int, int]] = []

        if self.history < 1:
            raise ValueError("history must be positive.")
        if self.rollout_steps < 1:
            raise ValueError("rollout_steps must be positive.")

        for traj_idx in range(len(self.seq_lengths)):
            frame_count = int(self.seq_lengths[traj_idx])
            max_start = frame_count - self.total_rollout_frames
            for start in range(max_start + 1):
                self.slices.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self.slices)

    def _load_episode(self, traj_idx: int) -> torch.Tensor:
        cached = self._episode_cache.get(traj_idx)
        if cached is not None:
            self._episode_cache.move_to_end(traj_idx)
            return cached

        episode = torch.load(self.obses_dir / f"episode_{traj_idx:05d}.pth", map_location="cpu").float()
        self._episode_cache[traj_idx] = episode
        if len(self._episode_cache) > self.cache_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        traj_idx, start = self.slices[idx]
        context_start = max(0, start - self.history + 1)
        stop = start + self.total_rollout_frames
        frames = self._load_episode(traj_idx)[context_start:stop].clone()
        rollout_start_idx = start - context_start
        action_start = start
        action_stop = action_start + self.total_actions
        actions = self.actions[traj_idx, action_start:action_stop].clone()
        return frames, actions, rollout_start_idx


def collate_latent_dynamics_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames, actions, rollout_start_indices = zip(*batch)
    max_frames = max(frame.shape[0] for frame in frames)
    padded_frames = []
    for frame in frames:
        pad_count = max_frames - frame.shape[0]
        if pad_count > 0:
            padding = frame[-1:].expand(pad_count, *frame.shape[1:])
            frame = torch.cat((frame, padding), dim=0)
        padded_frames.append(frame)
    return (
        torch.stack(padded_frames, dim=0),
        torch.stack(actions, dim=0),
        torch.tensor(rollout_start_indices, dtype=torch.long),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> DataLoader:
    persistent_workers = num_workers > 0
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_latent_dynamics_batch,
    }
    if persistent_workers:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def train_epoch(
    model: LatentDynamicsModel,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    log_every: int,
    amp_enabled: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    running = {"loss": 0.0, "dynamics_loss": 0.0, "curvature_loss": 0.0}
    count = 0

    for frames, actions, rollout_start_idx in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        rollout_start_idx = rollout_start_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(frames=frames, actions=actions, rollout_start_idx=rollout_start_idx)

        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += float(outputs.loss.item())
        running["dynamics_loss"] += float(outputs.dynamics_loss.item())
        running["curvature_loss"] += float(outputs.curvature_loss.item())
        count += 1
        global_step += 1

        if global_step % log_every == 0:
            for key, total in running.items():
                writer.add_scalar(f"train/{key}", total / count, global_step)

    metrics = {key: total / max(count, 1) for key, total in running.items()}
    return metrics, global_step


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    save_path = args.save_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")

    dataset = ReacherLatentDynamicsDataset(
        data_dir=data_dir,
        history=args.history,
        rollout_steps=args.rollout_steps,
        cache_episodes=args.cache_episodes,
    )
    train_loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor,
    )

    model = LatentDynamicsModel(
        latent_dim=args.latent_dim,
        history=args.history,
        action_dim=args.action_dim,
        encoder_proj_hidden_dim=args.encoder_proj_hidden_dim,
        encoder_proj_depth=args.encoder_proj_depth,
        dynamics_hidden_dim=args.dynamics_hidden_dim,
        dynamics_depth=args.dynamics_depth,
        curvature_weight=args.curvature_weight,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=args.amp and device.type == "cuda")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    global_step = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            writer=writer,
            global_step=global_step,
            log_every=args.log_every,
            amp_enabled=args.amp and device.type == "cuda",
        )

        for key, value in train_metrics.items():
            writer.add_scalar(f"epoch_train/{key}", value, epoch)

        epoch_log = {"epoch": epoch, "train": train_metrics}
        history.append(epoch_log)
        print(json.dumps(epoch_log, indent=2))

        ckpt = {
            "epoch": epoch,
            "model_architecture": MODEL_ARCHITECTURE,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "history": history,
        }

        if epoch % args.save_every == 0:
            torch.save(ckpt, save_path.with_name(f"{save_path.stem}_epoch_{epoch}{save_path.suffix}"))
        torch.save(ckpt, save_path)

    writer.close()


if __name__ == "__main__":
    main()
