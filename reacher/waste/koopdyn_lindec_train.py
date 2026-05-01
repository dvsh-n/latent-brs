#!/usr/bin/env python3
"""Train a linear-decoder Koopman model on offline Reacher Markov-state trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from reacher.shared.models import DeepKoopmanLinDec


DATA_PATH = "reacher/data/expert_data/reacher_koopman_markov.pt"
MODEL_SAVE_PATH = "reacher/models/koopdyn_lindec/koopman_lindec.pt"
LOG_DIR = "reacher/models/koopdyn_lindec/runs"

STATE_DIM = 36
CONTROL_DIM = 2
LATENT_DIM = 512
HIDDEN_WIDTH = 1024
HIDDEN_DEPTH = 3

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-6
EPOCHS = 150
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 35
NUM_WORKERS = 8
ENABLE_NORMALIZATION = False

LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.1


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if lowered in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--state-dim", type=int, default=STATE_DIM)
    parser.add_argument("--control-dim", type=int, default=CONTROL_DIM)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--hidden-width", type=int, default=HIDDEN_WIDTH)
    parser.add_argument("--hidden-depth", type=int, default=HIDDEN_DEPTH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--multi-step-horizon", type=int, default=MULTI_STEP_HORIZON)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--enable-normalization",
        type=str2bool,
        nargs="?",
        const=True,
        default=ENABLE_NORMALIZATION,
    )
    parser.add_argument("--lambda-state", type=float, default=LAMBDA_STATE)
    parser.add_argument("--lambda-latent", type=float, default=LAMBDA_LATENT)
    parser.add_argument("--seed", type=int, default=3072)
    return parser.parse_args()


def get_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        if path.suffix:
            candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        else:
            candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def print_config(config: dict[str, object], title: str) -> None:
    print(f"\n{'=' * 45}")
    print(f"      {title}")
    print(f"{'=' * 45}")
    for key, value in config.items():
        if isinstance(value, float):
            text = f"{value:.2e}" if value < 1e-3 else f"{value:.6f}"
        else:
            text = str(value)
        print(f"{key:<20s} | {text}")
    print(f"{'=' * 45}\n")


class ReacherMarkovDataset(Dataset):
    """Dataset for multi-step Koopman training on frozen visual Markov states."""

    def __init__(self, data_path: Path, multi_step_horizon: int = 1, enable_normalization: bool = False) -> None:
        payload = torch.load(data_path, weights_only=False)
        self.metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        demos = payload["demos"] if isinstance(payload, dict) else payload
        if not demos:
            raise ValueError(f"No demonstrations found in {data_path}.")

        self.states = []
        self.controls = []
        for demo in demos:
            states = torch.as_tensor(demo["states"], dtype=torch.float32).T.contiguous()
            controls = torch.as_tensor(demo["controls"], dtype=torch.float32).T.contiguous()
            if states.shape[0] != controls.shape[0] + 1:
                raise ValueError(
                    "Each demo must contain one more state than control, "
                    f"got states={states.shape} and controls={controls.shape}."
                )
            self.states.append(states)
            self.controls.append(controls)

        self.multi_step_horizon = int(multi_step_horizon)
        self.enable_normalization = bool(enable_normalization)
        self.x_k, self.u_k, self.x_kp1 = self._process_data()
        self._calculate_normalization_stats()

    def _process_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_k_list, u_k_list, x_kp1_list = [], [], []
        for episode_states, episode_controls in zip(self.states, self.controls):
            x_k_list.append(episode_states[:-1])
            u_k_list.append(episode_controls)
            x_kp1_list.append(episode_states[1:])

        episode_lengths = [len(ep) for ep in u_k_list]
        x_k = torch.cat(x_k_list, dim=0)
        u_k = torch.cat(u_k_list, dim=0)
        x_kp1 = torch.cat(x_kp1_list, dim=0)

        self.valid_indices = []
        current_start_index = 0
        horizon = self.multi_step_horizon
        for length in episode_lengths:
            valid_end_in_episode = length - horizon
            for idx in range(valid_end_in_episode + 1):
                self.valid_indices.append(current_start_index + idx)
            current_start_index += length
        if not self.valid_indices:
            raise ValueError("No valid training windows found. Check the dataset and multi-step horizon.")
        return x_k, u_k, x_kp1

    def _calculate_normalization_stats(self) -> None:
        self.min = self.x_k.min(dim=0).values
        self.max = self.x_k.max(dim=0).values
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-6

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        return 2 * (x - min_dev) / range_dev - 1

    def preprocess_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_normalization:
            return self.normalize(x)
        return x

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_start_idx = self.valid_indices[idx]
        horizon = self.multi_step_horizon
        x_k = self.x_k[actual_start_idx]
        u_seq = self.u_k[actual_start_idx : actual_start_idx + horizon]
        x_kp1_seq = self.x_kp1[actual_start_idx : actual_start_idx + horizon]
        return self.preprocess_state(x_k), u_seq, self.preprocess_state(x_kp1_seq)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    data_path = args.data_path.expanduser().resolve()
    model_save_path = get_unique_path(args.model_save_path.expanduser().resolve())
    log_dir = get_unique_path(args.log_dir.expanduser().resolve())
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir)
    dataset = ReacherMarkovDataset(
        data_path,
        multi_step_horizon=args.multi_step_horizon,
        enable_normalization=args.enable_normalization,
    )
    if dataset.x_k.shape[-1] != args.state_dim:
        raise ValueError(f"State dim mismatch: data has {dataset.x_k.shape[-1]}, args specify {args.state_dim}.")
    if dataset.u_k.shape[-1] != args.control_dim:
        raise ValueError(
            f"Control dim mismatch: data has {dataset.u_k.shape[-1]}, args specify {args.control_dim}."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_config = {
        "state_dim": args.state_dim,
        "control_dim": args.control_dim,
        "latent_dim": args.latent_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
    }
    model = DeepKoopmanLinDec(**model_config).to(device)

    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.min_lr,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting training for linear-decoder Koopman model (Reacher Markov state)")
    print_config({**model_config, "dataset_path": str(data_path)}, title="Model Configuration")
    training_config = {
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "multi_step_horizon": args.multi_step_horizon,
        "enable_normalization": args.enable_normalization,
        "lambda_state": args.lambda_state,
        "lambda_latent": args.lambda_latent,
    }
    print_config(training_config, title="Training Configuration")

    global_step = 0
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for _ in pbar:
        for x_k, u_seq, x_kp1_seq in dataloader:
            x_k = x_k.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            x_kp1_seq = x_kp1_seq.to(device, non_blocking=True)

            z_pred_seq, x_pred_seq, z_target_seq = model(x_k, u_seq, x_kp1_seq)
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

    save_data = {
        "model_config": model_config,
        "training_config": training_config,
        "state_dict": model.state_dict(),
        "normalization_stats": {
            "min": dataset.min,
            "max": dataset.max,
            "range": dataset.range,
        },
        "source_metadata": dataset.metadata,
    }
    torch.save(save_data, model_save_path)

    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
