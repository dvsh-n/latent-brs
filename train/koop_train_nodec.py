#!/usr/bin/env python3
"""Train decoder-free Koopman dynamics on LE-WM Reacher latent trajectories."""

from __future__ import annotations

import argparse
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

from shared.models import HistoryDeepKoopmanNoDec


DATA_PATH = "data/expert_data/latent_traj_lewm_reacher_24D.pt"
MODEL_SAVE_PATH = "models/koop_lewm_reacher_24D_nodec.pt"
LOG_DIR = "runs/koop_lewm_reacher_24D_nodec"

STATE_DIM = 24
CONTROL_DIM = 2
EMBEDDING_DIM = 128
HISTORY = 3
HIDDEN_WIDTH = 1024
HIDDEN_DEPTH = 3
ACTIVATION_FN = nn.GELU

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-6
EPOCHS = 500
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 25
NUM_WORKERS = 20
PREFETCH_FACTOR = 4

LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.1


def get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def print_config(config: dict, title: str = "Configuration") -> None:
    print("\n" + "=" * 45)
    print(f"      {title}")
    print("=" * 45)
    for key, value in config.items():
        print(f"{key:>24}: {value}")
    print("=" * 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--state_dim", type=int, default=STATE_DIM)
    parser.add_argument("--control_dim", type=int, default=CONTROL_DIM)
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--hidden_width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--hidden_depth", type=int, default=HIDDEN_DEPTH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min_lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--multi_step_horizon", type=int, default=MULTI_STEP_HORIZON)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--lambda_state", type=float, default=LAMBDA_STATE)
    parser.add_argument("--lambda_latent", type=float, default=LAMBDA_LATENT)
    return parser.parse_args()


class LeWMLatentKoopmanDataset(Dataset):
    """Multi-step Koopman dataset over LE-WM latents and true actions."""

    def __init__(self, data_path: Path, *, multi_step_horizon: int, history: int) -> None:
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        self.latents = payload["latents"].float()
        self.state_dim = int(self.latents.shape[-1])
        if "action_sequences" in payload:
            self.actions = payload["action_sequences"].float()
        else:
            self.actions = payload["actions"][:, :-1].float()
        self.ep_len = payload["ep_len"].long()
        self.multi_step_horizon = int(multi_step_horizon)
        self.history = int(history)
        self.uses_precomputed_history = False
        self.history_states = self._load_history_states(payload)
        self.valid_indices: list[tuple[int, int]] = []

        if self.latents.ndim != 3:
            raise ValueError(f"Expected latents with shape [episodes, frames, dim], got {self.latents.shape}.")
        if self.actions.ndim != 3:
            raise ValueError(f"Expected action_sequences with shape [episodes, steps, dim], got {self.actions.shape}.")
        if self.history_states.shape != (self.latents.shape[0], self.latents.shape[1], self.history * self.state_dim):
            raise ValueError(
                "Expected history_states with shape "
                f"{(self.latents.shape[0], self.latents.shape[1], self.history * self.state_dim)}, "
                f"got {self.history_states.shape}."
            )
        if self.history < 1:
            raise ValueError("history must be positive.")
        if self.multi_step_horizon < 1:
            raise ValueError("multi_step_horizon must be positive.")
        if self.latents.shape[0] != self.actions.shape[0]:
            raise ValueError(f"Episode count mismatch: latents={self.latents.shape}, actions={self.actions.shape}.")

        for episode_idx, ep_frames in enumerate(self.ep_len.tolist()):
            max_start = ep_frames - self.multi_step_horizon - 1
            for start in range(max_start + 1):
                self.valid_indices.append((episode_idx, start))

        if not self.valid_indices:
            raise ValueError("No valid training windows found. Check episode length and multi_step_horizon.")

    def _load_history_states(self, payload: dict) -> torch.Tensor:
        metadata = payload.get("metadata", {})
        history_states = payload.get("history_states")
        expected_shape = (self.latents.shape[0], self.latents.shape[1], self.history * self.state_dim)
        if (
            history_states is not None
            and int(metadata.get("history", self.history)) == self.history
            and tuple(history_states.shape) == expected_shape
        ):
            self.uses_precomputed_history = True
            return history_states.float()

        _, num_frames, _ = self.latents.shape
        frame_idx = torch.arange(num_frames)
        history_slices = []
        for offset in range(self.history):
            source_idx = (frame_idx - self.history + 1 + offset).clamp_min(0)
            history_slices.append(self.latents[:, source_idx])
        return torch.stack(history_slices, dim=2).flatten(start_dim=2)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx, start = self.valid_indices[idx]
        horizon = self.multi_step_horizon

        history_k = self.history_states[episode_idx, start]
        u_seq = self.actions[episode_idx, start : start + horizon]
        x_kp1_seq = self.latents[episode_idx, start + 1 : start + horizon + 1]
        history_kp1_seq = self.history_states[episode_idx, start + 1 : start + horizon + 1]

        return history_k, u_seq, x_kp1_seq, history_kp1_seq


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    model_save_path = get_unique_path(Path(args.model_save_path))
    log_dir = get_unique_path(Path(args.log_dir))
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch_factor must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    activation_fn = nn.GELU

    writer = SummaryWriter(log_dir)
    dataset = LeWMLatentKoopmanDataset(
        data_path,
        multi_step_horizon=args.multi_step_horizon,
        history=args.history,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    model_config = {
        "state_dim": args.state_dim,
        "control_dim": args.control_dim,
        "embedding_dim": args.embedding_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
        "activation_fn": activation_fn,
        "history": args.history,
    }
    model = HistoryDeepKoopmanNoDec(**model_config).to(device)

    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.min_lr,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting training for decoder-free Koopman model on LE-WM Reacher latents")
    print_config(
        {
            **model_config,
            "activation_fn": activation_fn.__name__,
            "history_input_dim": args.history * args.state_dim,
            "total_latent_dim": args.state_dim + args.embedding_dim,
        },
        title="Model Configuration",
    )

    training_config = {
        "data_path": str(data_path),
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "multi_step_horizon": args.multi_step_horizon,
        "lambda_state": args.lambda_state,
        "lambda_latent": args.lambda_latent,
        "num_samples": len(dataset),
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "uses_precomputed_history": dataset.uses_precomputed_history,
    }
    print_config(training_config, title="Training Configuration")

    global_step = 0
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for _ in pbar:
        for history_k, u_seq, x_kp1_seq, history_kp1_seq in dataloader:
            non_blocking = device.type == "cuda"
            history_k = history_k.to(device, non_blocking=non_blocking)
            u_seq = u_seq.to(device, non_blocking=non_blocking)
            x_kp1_seq = x_kp1_seq.to(device, non_blocking=non_blocking)
            history_kp1_seq = history_kp1_seq.to(device, non_blocking=non_blocking)

            z_pred_seq, x_pred_seq, z_target_seq = model(history_k, u_seq, history_kp1_seq)

            loss_state = nn.functional.mse_loss(x_pred_seq, x_kp1_seq)
            loss_latent = nn.functional.mse_loss(z_pred_seq, z_target_seq)
            loss = args.lambda_state * loss_state + args.lambda_latent * loss_latent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 10 == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/State", loss_state.item(), global_step)
                writer.add_scalar("Loss/Latent", loss_latent.item(), global_step)
                writer.add_scalar("Hyperparams/Learning_Rate", scheduler.get_last_lr()[0], global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

    writer.close()

    model_config_save = model_config.copy()
    model_config_save["activation_fn"] = activation_fn.__name__

    save_data = {
        "model_config": model_config_save,
        "training_config": training_config,
        "state_dict": model.state_dict(),
        "normalization_stats": None,
        "data_metadata": {
            "state_order": "chronological history [z_{t-history+1}, ..., z_t], boundary repeated from z0",
            "action_order": "u_seq[:, i] is the true action a_{t+i} used for z_{t+i} -> z_{t+i+1}",
        },
    }
    torch.save(save_data, model_save_path)

    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    train(parse_args())
