#!/usr/bin/env python3
"""Train a GELU MLP baseline for LE-WM latent dynamics."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATA_PATH = REPO_ROOT / "data" / "expert_data" / "latent_traj_lewm_reacher_mlp.pt"
MODEL_SAVE_PATH = REPO_ROOT / "test" / "mlp_lewm_reacher_mlp.pt"
LOG_DIR = REPO_ROOT / "test" / "runs" / "mlp_lewm_reacher_mlp"

STATE_DIM = 24
ACTION_DIM = 2
HISTORY = 3
HIDDEN_WIDTH = 1024
HIDDEN_DEPTH = 4
MULTI_STEP_HORIZON = 35
BATCH_SIZE = 4096
EPOCHS = 50
LR = 1e-4
MIN_LR = 1e-7
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
TRAIN_SPLIT = 0.9
SEED = 3072


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--action-dim", type=int, default=ACTION_DIM)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--hidden-width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--hidden-depth", type=int, default=HIDDEN_DEPTH)
    parser.add_argument("--multi-step-horizon", type=int, default=MULTI_STEP_HORIZON)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--min-lr", type=float, default=MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-predict-delta", action="store_true")
    return parser.parse_args()


def make_mlp(input_dim: int, output_dim: int, hidden_width: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("hidden_depth must be at least 1.")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_width))
        layers.append(nn.GELU())
        current_dim = hidden_width
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        history: int,
        hidden_width: int,
        hidden_depth: int,
        predict_delta: bool,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.history_dim = self.history * self.state_dim
        self.predict_delta = bool(predict_delta)
        self.net = make_mlp(
            input_dim=self.history_dim + self.action_dim,
            output_dim=self.state_dim,
            hidden_width=hidden_width,
            depth=hidden_depth,
        )

    def forward_step(self, history_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        update = self.net(torch.cat((history_state, action), dim=-1))
        if self.predict_delta:
            return history_state[..., -self.state_dim :] + update
        return update

    def rollout(self, history_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        predictions = []
        current_history = history_state
        for step in range(actions.shape[1]):
            next_latent = self.forward_step(current_history, actions[:, step])
            predictions.append(next_latent)
            current_history = torch.cat((current_history[:, self.state_dim :], next_latent), dim=-1)
        return torch.stack(predictions, dim=1)


class LeWMLatentWindowDataset(Dataset):
    def __init__(
        self,
        payload: dict,
        *,
        episode_indices: list[int],
        multi_step_horizon: int,
        history: int,
    ) -> None:
        self.latents = payload["latents"].float()
        self.actions = payload.get("action_sequences", payload["actions"][:, :-1]).float()
        self.ep_len = payload["ep_len"].long()
        self.state_dim = int(self.latents.shape[-1])
        self.history = int(history)
        self.multi_step_horizon = int(multi_step_horizon)
        self.history_states = self._load_history_states(payload)
        self.valid_indices: list[tuple[int, int]] = []

        if self.history < 1:
            raise ValueError("history must be positive.")
        if self.multi_step_horizon < 1:
            raise ValueError("multi_step_horizon must be positive.")
        if self.history_states.shape != (self.latents.shape[0], self.latents.shape[1], self.history * self.state_dim):
            raise ValueError(f"history_states has unexpected shape: {self.history_states.shape}")

        for episode_idx in episode_indices:
            ep_frames = int(self.ep_len[episode_idx].item())
            max_start = ep_frames - self.multi_step_horizon - 1
            for start in range(max_start + 1):
                self.valid_indices.append((episode_idx, start))

        if not self.valid_indices:
            raise ValueError("No valid training windows found.")

    def _load_history_states(self, payload: dict) -> torch.Tensor:
        metadata = payload.get("metadata", {})
        history_states = payload.get("history_states")
        expected_shape = (self.latents.shape[0], self.latents.shape[1], self.history * self.state_dim)
        if (
            history_states is not None
            and int(metadata.get("history", self.history)) == self.history
            and tuple(history_states.shape) == expected_shape
        ):
            return history_states.float()

        _, num_frames, _ = self.latents.shape
        frame_idx = torch.arange(num_frames)
        slices = []
        for offset in range(self.history):
            source_idx = (frame_idx - self.history + 1 + offset).clamp_min(0)
            slices.append(self.latents[:, source_idx])
        return torch.stack(slices, dim=2).flatten(start_dim=2)

    def normalize_actions(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.actions = (torch.nan_to_num(self.actions, nan=0.0) - mean) / std

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx, start = self.valid_indices[idx]
        horizon = self.multi_step_horizon
        history_k = self.history_states[episode_idx, start]
        u_seq = self.actions[episode_idx, start : start + horizon]
        target_seq = self.latents[episode_idx, start + 1 : start + horizon + 1]
        return history_k, u_seq, target_seq


def split_episodes(num_episodes: int, train_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1.")
    indices = list(range(num_episodes))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(num_episodes - 1, int(num_episodes * train_split)))
    return indices[:train_count], indices[train_count:]


def action_stats(actions: torch.Tensor, valid_indices: list[tuple[int, int]], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = [actions[episode_idx, start : start + horizon] for episode_idx, start in valid_indices]
    flat = torch.cat(chunks, dim=0)
    flat = flat[torch.isfinite(flat).all(dim=-1)]
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def evaluate_loss(model: LatentDynamicsMLP, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for history_k, u_seq, target_seq in loader:
            history_k = history_k.to(device)
            u_seq = u_seq.to(device)
            target_seq = target_seq.to(device)
            pred_seq = model.rollout(history_k, u_seq)
            loss = nn.functional.mse_loss(pred_seq, target_seq, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(target_seq.numel())
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = args.data_path.expanduser().resolve()
    model_save_path = args.model_save_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    train_episodes, val_episodes = split_episodes(int(payload["latents"].shape[0]), args.train_split, args.seed)
    train_dataset = LeWMLatentWindowDataset(
        payload,
        episode_indices=train_episodes,
        multi_step_horizon=args.multi_step_horizon,
        history=args.history,
    )
    val_dataset = LeWMLatentWindowDataset(
        payload,
        episode_indices=val_episodes,
        multi_step_horizon=args.multi_step_horizon,
        history=args.history,
    )

    action_mean, action_std = action_stats(train_dataset.actions, train_dataset.valid_indices, args.multi_step_horizon)
    train_dataset.normalize_actions(action_mean, action_std)
    val_dataset.normalize_actions(action_mean, action_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model_config = {
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "history": args.history,
        "hidden_width": args.hidden_width,
        "hidden_depth": args.hidden_depth,
        "predict_delta": not args.no_predict_delta,
    }
    model = LatentDynamicsMLP(**model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=args.min_lr,
    )
    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    print(f"Using device: {device}")
    print(json.dumps({**model_config, "train_windows": len(train_dataset), "val_windows": len(val_dataset)}, indent=2))

    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for history_k, u_seq, target_seq in train_loader:
            history_k = history_k.to(device, non_blocking=device.type == "cuda")
            u_seq = u_seq.to(device, non_blocking=device.type == "cuda")
            target_seq = target_seq.to(device, non_blocking=device.type == "cuda")

            pred_seq = model.rollout(history_k, u_seq)
            loss = nn.functional.mse_loss(pred_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss_sum += float(loss.item()) * int(target_seq.numel())
            train_count += int(target_seq.numel())
            if global_step % 10 == 0:
                writer.add_scalar("Loss/TrainStep", loss.item(), global_step)
                writer.add_scalar("Hyperparams/LearningRate", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        writer.add_scalar("Loss/TrainEpoch", train_loss, epoch)
        writer.add_scalar("Loss/ValEpoch", val_loss, epoch)
        tqdm.write(f"epoch={epoch + 1} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "training_config": {
                "data_path": str(data_path),
                "multi_step_horizon": args.multi_step_horizon,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "train_split": args.train_split,
                "seed": args.seed,
                "best_val_mse": best_val_loss,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
            },
            "action_stats": {
                "mean": action_mean,
                "std": action_std,
            },
            "state_dict": model.state_dict(),
        },
        model_save_path,
    )
    print(f"Saved best model to {model_save_path}")
    print(f"Best validation MSE: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
