# koop_train_lindec.py
import sys
from pathlib import Path


def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import DeepKoopmanLinDec
from utils import (
    parse_train_args,
    get_activation_fn,
    get_activation_name,
    resolve_path,
    print_config,
    get_unique_path,
)


DATA_PATH = "data/random_data.pt"
MODEL_SAVE_PATH = "models/koop_quad3d_lindec.pt"
LOG_DIR = "runs/koop_quad3d_lindec"

STATE_DIM = 12
CONTROL_DIM = 4
LATENT_DIM = 24
HIDDEN_WIDTH = 64
HIDDEN_DEPTH = 3
ACTIVATION_FN = nn.GELU

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-6
EPOCHS = 35
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 50
NUM_WORKERS = 20
ENABLE_NORMALIZATION = False

LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.1


class QuadrotorDataset(Dataset):
    """Dataset for multi-step Koopman training with optional normalization."""

    def __init__(self, data_path, multi_step_horizon=1, enable_normalization=True):
        demos = torch.load(data_path, weights_only=False)

        self.states = []
        self.controls = []
        for demo in demos:
            self.states.append(torch.tensor(demo["states"].T, dtype=torch.float32))
            self.controls.append(torch.tensor(demo["controls"].T, dtype=torch.float32))

        self.multi_step_horizon = multi_step_horizon
        self.enable_normalization = enable_normalization
        self.x_k, self.u_k, self.x_kp1 = self._process_data()
        self._calculate_normalization_stats()

    def _process_data(self):
        x_k_list, u_k_list, x_kp1_list = [], [], []
        for episode_states, episode_controls in zip(self.states, self.controls):
            x_k_list.append(episode_states[:-1])
            u_k_list.append(episode_controls)
            x_kp1_list.append(episode_states[1:])

        episode_lengths = [len(ep) for ep in u_k_list]
        x_k = torch.cat(x_k_list)
        u_k = torch.cat(u_k_list)
        x_kp1 = torch.cat(x_kp1_list)

        self.valid_indices = []
        current_start_index = 0
        M = self.multi_step_horizon
        for length in episode_lengths:
            valid_end_in_episode = length - M
            for i in range(valid_end_in_episode + 1):
                self.valid_indices.append(current_start_index + i)
            current_start_index += length

        return x_k, u_k, x_kp1

    def _calculate_normalization_stats(self):
        self.min = self.x_k.min(dim=0).values
        self.max = self.x_k.max(dim=0).values
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        return 2 * (x - min_dev) / range_dev - 1

    def preprocess_state(self, x):
        if self.enable_normalization:
            return self.normalize(x)
        return x

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_start_idx = self.valid_indices[idx]
        M = self.multi_step_horizon

        x_k = self.x_k[actual_start_idx]
        u_seq = self.u_k[actual_start_idx: actual_start_idx + M]
        x_kp1_seq = self.x_kp1[actual_start_idx: actual_start_idx + M]

        return self.preprocess_state(x_k), u_seq, self.preprocess_state(x_kp1_seq)


def train(args):
    data_path = resolve_path(args.data_path)
    model_save_path = resolve_path(args.model_save_path)
    log_dir = resolve_path(args.log_dir)

    model_save_path = get_unique_path(model_save_path)
    log_dir = get_unique_path(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    activation_fn = get_activation_fn(args.activation)

    writer = SummaryWriter(log_dir)
    dataset = QuadrotorDataset(
        data_path,
        multi_step_horizon=args.multi_step_horizon,
        enable_normalization=args.enable_normalization,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_config = {
        "state_dim": args.state_dim,
        "control_dim": args.control_dim,
        "latent_dim": args.latent_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
        "activation_fn": activation_fn,
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

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting training for linear-decoder Deep Koopman model (3D Quadrotor)")
    print_config(
        {
            **model_config,
            "total_latent_dim": args.latent_dim,
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
        "enable_normalization": args.enable_normalization,
        "lambda_state": args.lambda_state,
        "lambda_latent": args.lambda_latent,
    }
    print_config(training_config, title="Training Configuration")

    global_step = 0
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for _ in pbar:
        for x_k, u_seq, x_kp1_seq in dataloader:
            x_k = x_k.to(device)
            u_seq = u_seq.to(device)
            x_kp1_seq = x_kp1_seq.to(device)

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

    model_config_save = model_config.copy()
    model_config_save["activation_fn"] = args.activation

    save_data = {
        "model_config": model_config_save,
        "training_config": training_config,
        "state_dict": model.state_dict(),
        "normalization_stats": {
            "min": dataset.min,
            "max": dataset.max,
            "range": dataset.range,
        },
    }
    torch.save(save_data, model_save_path)

    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    defaults = {
        "data_path": DATA_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "log_dir": LOG_DIR,
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "hidden_depth": HIDDEN_DEPTH,
        "activation": get_activation_name(ACTIVATION_FN),
        "lr": LEARNING_RATE,
        "min_lr": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "multi_step_horizon": MULTI_STEP_HORIZON,
        "num_workers": NUM_WORKERS,
        "enable_normalization": ENABLE_NORMALIZATION,
        "lambda_state": LAMBDA_STATE,
        "lambda_latent": LAMBDA_LATENT,
    }
    args = parse_train_args(
        defaults,
        description="Train linear-decoder Deep Koopman model for 3D Quadrotor",
    )
    train(args)
