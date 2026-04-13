#!/usr/bin/env python3
"""Evaluate decoder-free Koopman rollouts on LE-WM Reacher latent trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.models import HistoryDeepKoopmanNoDec


DATA_PATH = "data/expert_data/latent_traj_lewm_reacher_24D.pt"
MODEL_SAVE_PATH = "models/koop_lewm_reacher_24D_nodec_1.pt"
PLOT_SAVE_DIR = "eval/koop_eval_nodec"
N_EVAL_EPISODES = 25
MAX_TIMESTEP = 50
PURE_LATENT_ROLLOUT = True
OVERALL_RMSE_STEPS = [1, 10, 25, 50]


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_save_path", type=str, default=MODEL_SAVE_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--plot_save_dir", type=str, default=PLOT_SAVE_DIR)
    parser.add_argument("--n_eval_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--max_timestep", type=int, default=MAX_TIMESTEP)
    parser.add_argument(
        "--pure_latent_rollout",
        action=argparse.BooleanOptionalAction,
        default=PURE_LATENT_ROLLOUT,
        help="Use pure Koopman latent rollout instead of recurrent re-lifting.",
    )
    return parser.parse_args()


def get_activation_fn(name: str) -> type[nn.Module]:
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    return activations[name.lower()]


def print_config_table(config: dict, title: str = "Configuration") -> None:
    print("\n" + "=" * 45)
    print(f"      {title}")
    print("=" * 45)
    for key, value in config.items():
        print(f"{key:>24}: {value}")
    print("=" * 45)


def print_table(title: str, headers: list[str], rows: list[tuple]) -> None:
    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(value)))

    print("\n" + title)
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    print(" | ".join(header.ljust(col_widths[idx]) for idx, header in enumerate(headers)))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print(" | ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(row)))


def make_padded_history_states(latents: torch.Tensor, history: int) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(f"Expected latents with shape [episodes, frames, dim], got {latents.shape}.")
    if history < 1:
        raise ValueError("history must be positive.")

    _, num_frames, _ = latents.shape
    frame_idx = torch.arange(num_frames)
    history_slices = []
    for offset in range(history):
        source_idx = (frame_idx - history + 1 + offset).clamp_min(0)
        history_slices.append(latents[:, source_idx])
    return torch.stack(history_slices, dim=2).flatten(start_dim=2)


class LeWMLatentKoopmanEvalDataset:
    def __init__(self, data_path: Path, history: int) -> None:
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        self.latents = payload["latents"].float()
        if "action_sequences" in payload:
            self.actions = payload["action_sequences"].float()
        else:
            self.actions = payload["actions"][:, :-1].float()
        self.ep_len = payload["ep_len"].long()
        self.history = int(history)
        self.state_dim = int(self.latents.shape[-1])
        self.history_states = self._load_history_states(payload)

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
        return make_padded_history_states(self.latents, self.history)

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def get_raw_episode(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(self.ep_len[idx].item())
        latents = self.latents[idx, :length]
        actions = self.actions[idx, : max(length - 1, 0)]
        history0 = self.history_states[idx, 0]
        return latents, actions, history0


@torch.no_grad()
def rollout_latent(
    model: HistoryDeepKoopmanNoDec,
    history0: torch.Tensor,
    u_raw: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history0 = history0.to(device).unsqueeze(0)
    u_raw = u_raw.to(device)

    z_k = model.lift_state(history0)
    x_hat_list = [history0[:, -model.state_dim :].squeeze(0)]
    for step in range(len(u_raw)):
        z_k = model.A(z_k) + model.B(u_raw[step].unsqueeze(0))
        x_hat_list.append(z_k[:, : model.state_dim].squeeze(0))

    return torch.stack(x_hat_list, dim=0).cpu().numpy()


@torch.no_grad()
def rollout_recurrent(
    model: HistoryDeepKoopmanNoDec,
    history0: torch.Tensor,
    u_raw: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    history = history0.to(device).unsqueeze(0)
    u_raw = u_raw.to(device)

    x_hat_list = [history[:, -model.state_dim :].squeeze(0)]
    for step in range(len(u_raw)):
        z_k = model.lift_state(history)
        z_kp1 = model.A(z_k) + model.B(u_raw[step].unsqueeze(0))
        x_kp1 = z_kp1[:, : model.state_dim]
        history = torch.cat((history[:, model.state_dim :], x_kp1), dim=-1)
        x_hat_list.append(x_kp1.squeeze(0))

    return torch.stack(x_hat_list, dim=0).cpu().numpy()


def calculate_rmse(x_traj: np.ndarray, x_hat_traj: np.ndarray) -> np.ndarray:
    horizon = min(len(x_traj), len(x_hat_traj))
    err = x_traj[:horizon] - x_hat_traj[:horizon]
    return np.sqrt(np.mean(err**2, axis=0))


def plot_rollout(x_traj: np.ndarray, x_hat_traj: np.ndarray, episode_idx: int, plot_save_dir: Path) -> None:
    horizon = min(len(x_traj), len(x_hat_traj))
    state_dim = x_traj.shape[-1]
    cols = 4
    rows = int(np.ceil(state_dim / cols))
    time_ax = np.arange(horizon)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.0 * rows), sharex=True, constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    fig.suptitle(f"Decoder-Free Koopman Latent Rollout vs. LE-WM Latents (Episode {episode_idx})", fontsize=16)

    for dim, ax in enumerate(axes_flat):
        if dim >= state_dim:
            ax.set_axis_off()
            continue
        ax.plot(time_ax, x_traj[:horizon, dim], label="LE-WM latent", color="blue", linewidth=1.4)
        ax.plot(time_ax, x_hat_traj[:horizon, dim], label="Koopman rollout", color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(f"z{dim}")
        ax.grid(True, linestyle=":", alpha=0.6)

    for ax in axes_flat[-cols:]:
        if ax.has_data():
            ax.set_xlabel("Time Step")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)

    plot_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_save_dir / f"episode_{episode_idx}.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def load_model(model_save_path: Path, device: torch.device) -> tuple[HistoryDeepKoopmanNoDec, dict, dict]:
    model_data = torch.load(model_save_path, map_location=device, weights_only=False)
    model_config = model_data.get("model_config", model_data.get("config"))
    if model_config is None:
        raise ValueError("Model checkpoint does not contain model_config.")
    training_config = model_data.get("training_config", {})

    model_config = model_config.copy()
    if isinstance(model_config.get("activation_fn"), str):
        model_config["activation_fn"] = get_activation_fn(model_config["activation_fn"])

    model_params = {
        key: value
        for key, value in model_config.items()
        if key in ["state_dim", "control_dim", "embedding_dim", "hidden_width", "depth", "activation_fn", "history"]
    }
    model = HistoryDeepKoopmanNoDec(**model_params).to(device)
    model.load_state_dict(model_data["state_dict"])
    model.eval()
    return model, model_config, training_config


def evaluate(
    model_save_path_arg: str | None = None,
    data_path_arg: str | None = None,
    plot_save_dir_arg: str | None = None,
    n_eval_episodes: int = N_EVAL_EPISODES,
    max_timestep: int | None = MAX_TIMESTEP,
    pure_latent_rollout: bool = PURE_LATENT_ROLLOUT,
) -> None:
    data_path = Path(data_path_arg or DATA_PATH)
    model_save_path = Path(model_save_path_arg or MODEL_SAVE_PATH)
    plot_save_dir = Path(plot_save_dir_arg or PLOT_SAVE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_save_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_save_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model, model_config, training_config = load_model(model_save_path, device)
    dataset = LeWMLatentKoopmanEvalDataset(data_path, history=model.history)

    printable_model_config = model_config.copy()
    if isinstance(printable_model_config.get("activation_fn"), type):
        printable_model_config["activation_fn"] = printable_model_config["activation_fn"].__name__
    print_config_table(printable_model_config, title="Model Configuration")
    if training_config:
        print_config_table(training_config, title="Training Configuration")

    n_eval_episodes = min(n_eval_episodes, len(dataset))
    state_errs = []
    step_mse = {step: [] for step in OVERALL_RMSE_STEPS}

    pbar = tqdm(range(n_eval_episodes), desc="Evaluating episodes")
    for ep_id in pbar:
        x_traj, u, history0 = dataset.get_raw_episode(ep_id)
        if max_timestep is not None and max_timestep > 0:
            u = u[:max_timestep]
            x_traj = x_traj[: max_timestep + 1]

        if pure_latent_rollout:
            x_hat_traj = rollout_latent(model, history0, u, device)
        else:
            x_hat_traj = rollout_recurrent(model, history0, u, device)

        x_traj_np = x_traj.numpy()
        plot_rollout(x_traj_np, x_hat_traj, ep_id, plot_save_dir)

        rmse_state = calculate_rmse(x_traj_np, x_hat_traj)
        state_errs.append(rmse_state)

        horizon = min(len(x_traj_np), len(x_hat_traj))
        err = x_traj_np[:horizon] - x_hat_traj[:horizon]
        for step in OVERALL_RMSE_STEPS:
            if step < horizon:
                step_mse[step].append(float(np.mean(err[step] ** 2)))

        pbar.set_postfix(mean_latent_rmse=f"{np.mean(state_errs):.4f}")

    state_errs_np = np.asarray(state_errs)
    rmse_per_dim = state_errs_np.mean(axis=0)
    dim_rows = [(f"z{idx}", f"{rmse:.6f}") for idx, rmse in enumerate(rmse_per_dim)]
    print_table("Decoder-Free Koopman LE-WM Latent RMSE", ["Dimension", "RMSE"], dim_rows)
    print(f"\nMean latent RMSE: {float(rmse_per_dim.mean()):.6f}")

    horizon_rows = []
    for step in OVERALL_RMSE_STEPS:
        count = len(step_mse[step])
        rmse_val = np.sqrt(np.mean(step_mse[step])) if count > 0 else np.nan
        rmse_str = f"{rmse_val:.6f}" if count > 0 else "N/A"
        horizon_rows.append((f"{step}-step", rmse_str, f"{count}/{n_eval_episodes}"))
    print_table("Overall Horizon RMSE", ["Horizon", "RMSE", "Episodes Used"], horizon_rows)


if __name__ == "__main__":
    args = parse_eval_args()
    evaluate(
        model_save_path_arg=args.model_save_path,
        data_path_arg=args.data_path,
        plot_save_dir_arg=args.plot_save_dir,
        n_eval_episodes=args.n_eval_episodes,
        max_timestep=args.max_timestep,
        pure_latent_rollout=args.pure_latent_rollout,
    )
