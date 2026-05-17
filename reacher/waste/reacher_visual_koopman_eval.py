#!/usr/bin/env python3
"""Evaluate a history-conditioned visual Koopman model on Reacher episodes."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from test.models_koopman import VisualHistoryKoopman
from test.reacher_visual_koopman_train import ChunkedDinoLatentStore, require_device


DEFAULT_DATASET_PATH = Path("data/expert_data/expert_data.pt")
DEFAULT_LATENT_METADATA_PATH = Path("data/expert_data/latents/dinov2_vits14_all_tokens/metadata.pt")
DEFAULT_MODEL_SAVE_PATH = Path("models/reacher_visual_koopman_epoch_1.pt")
DEFAULT_PLOT_SAVE_DIR = Path("eval/reacher_visual_koopman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-save-path", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--latent-metadata-path", type=Path, default=DEFAULT_LATENT_METADATA_PATH)
    parser.add_argument("--plot-save-dir", type=Path, default=DEFAULT_PLOT_SAVE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=10,
        help="Requested rollout horizon in control steps. Clamped to the maximum valid horizon for each episode.",
    )
    parser.add_argument(
        "--plots-per-figure",
        type=int,
        default=8,
        help="Number of dimensions to show per image.",
    )
    return parser.parse_args()


class EvalLatentStore(ChunkedDinoLatentStore):
    def __init__(self, dataset_path: Path, latent_metadata_path: Path) -> None:
        super().__init__(dataset_path, latent_metadata_path)
        self._cached_chunk_range: tuple[int, int] | None = None
        self._cached_chunk_latents: torch.Tensor | None = None

    def _load_chunk_for_episode(self, episode_idx: int) -> tuple[torch.Tensor, int]:
        for chunk_info in self.chunk_infos:
            start_idx = int(chunk_info["start_idx"])
            end_idx = int(chunk_info["end_idx"])
            if start_idx <= episode_idx < end_idx:
                chunk_range = (start_idx, end_idx)
                if self._cached_chunk_range != chunk_range:
                    chunk_data = torch.load(chunk_info["path"], map_location="cpu", weights_only=False)
                    self._cached_chunk_range = chunk_range
                    self._cached_chunk_latents = chunk_data["latents"].to(torch.float32)
                assert self._cached_chunk_latents is not None
                return self._cached_chunk_latents, episode_idx - start_idx
        raise IndexError(f"Episode index {episode_idx} is out of range")

    def get_episode(self, episode_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (0 <= episode_idx < self.num_trajectories):
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self.num_trajectories})")
        chunk_latents, local_idx = self._load_chunk_for_episode(episode_idx)
        return (
            chunk_latents[local_idx].clone(),
            self.actions[episode_idx].clone(),
            self.states[episode_idx].clone(),
        )


def rollout_episode(
    model: VisualHistoryKoopman,
    latent_episode: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_len: int,
    max_timestep: int | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        latent_episode = latent_episode.to(device)
        actions = actions.to(device)

        max_valid_horizon = int(actions.shape[0]) - history_len
        if max_valid_horizon <= 0:
            raise ValueError(
                f"Episode supports no valid rollout horizon with history_len={history_len} and "
                f"{int(actions.shape[0])} actions"
            )
        rollout_horizon = max_valid_horizon if max_timestep is None else min(int(max_timestep), max_valid_horizon)

        latent_seq = latent_episode[: history_len + rollout_horizon + 1].unsqueeze(0)
        action_seq = actions[history_len : history_len + rollout_horizon].unsqueeze(0)

        z_target_seq, x_target_seq = model.lift_latent_sequence(latent_seq)
        z0 = z_target_seq[:, 0, :]
        z_pred_list = [z0]
        z_current = z0
        for step in range(rollout_horizon):
            u_k = action_seq[:, step, :]
            z_current = model.A(z_current) + model.B(u_k)
            z_pred_list.append(z_current)

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        x_pred_seq = z_pred_seq[..., : model.state_dim]

        return (
            x_target_seq.squeeze(0).cpu().numpy(),
            x_pred_seq.squeeze(0).cpu().numpy(),
            z_target_seq.squeeze(0).cpu().numpy(),
            z_pred_seq.squeeze(0).cpu().numpy(),
        )


def plot_dimension_grid(
    target_seq: np.ndarray,
    pred_seq: np.ndarray,
    *,
    title_prefix: str,
    save_prefix: Path,
    plots_per_figure: int,
) -> None:
    n_dims = int(target_seq.shape[1])
    if pred_seq.shape != target_seq.shape:
        raise ValueError(f"Mismatched target/pred shapes: {target_seq.shape} vs {pred_seq.shape}")

    cols = 2
    rows = max(1, math.ceil(plots_per_figure / cols))
    time_ax = np.arange(target_seq.shape[0])
    n_figures = math.ceil(n_dims / plots_per_figure)

    for fig_idx in range(n_figures):
        start_dim = fig_idx * plots_per_figure
        end_dim = min(start_dim + plots_per_figure, n_dims)
        dims = list(range(start_dim, end_dim))

        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5), squeeze=False, constrained_layout=True)
        axes_flat = axes.flatten()
        fig.suptitle(f"{title_prefix} ({start_dim} to {end_dim - 1})", fontsize=16)

        for ax, dim in zip(axes_flat, dims):
            ax.plot(time_ax, target_seq[:, dim], color="blue", linewidth=2, label="Direct Encode")
            ax.plot(time_ax, pred_seq[:, dim], color="red", linewidth=2, linestyle="--", label="Koopman Rollout")
            ax.set_title(f"Dim {dim}", fontsize=11)
            ax.set_xlabel("Time Step")
            ax.grid(True, linestyle=":")

        for ax in axes_flat[len(dims) :]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        save_path = save_prefix.parent / f"{save_prefix.name}_part_{fig_idx + 1:02d}.png"
        plt.savefig(save_path)
        plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path.expanduser().resolve()
    latent_metadata_path = args.latent_metadata_path.expanduser().resolve()
    model_save_path = args.model_save_path.expanduser().resolve()
    plot_save_dir = args.plot_save_dir.expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not latent_metadata_path.is_file():
        raise FileNotFoundError(f"Latent metadata file not found: {latent_metadata_path}")
    if not model_save_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_save_path}")

    device = require_device(args.device)
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = VisualHistoryKoopman(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    store = EvalLatentStore(dataset_path, latent_metadata_path)
    history_len = int(model_config["history_len"])
    n_eval_episodes = min(int(args.n_eval_episodes), store.num_trajectories)
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Checkpoint: {model_save_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Latent metadata: {latent_metadata_path}")
    print(f"Plot dir: {plot_save_dir}")
    print(f"History length: {history_len}")

    state_mse_values: list[float] = []
    lifted_mse_values: list[float] = []

    for episode_idx in range(n_eval_episodes):
        latent_episode, actions, _states = store.get_episode(episode_idx)
        x_target_seq, x_pred_seq, z_target_seq, z_pred_seq = rollout_episode(
            model,
            latent_episode,
            actions,
            history_len=history_len,
            max_timestep=args.max_timestep,
            device=device,
        )

        state_mse = float(np.mean((x_target_seq - x_pred_seq) ** 2))
        lifted_mse = float(np.mean((z_target_seq - z_pred_seq) ** 2))
        state_mse_values.append(state_mse)
        lifted_mse_values.append(lifted_mse)

        episode_dir = plot_save_dir / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        plot_dimension_grid(
            x_target_seq,
            x_pred_seq,
            title_prefix=f"Episode {episode_idx} State Evolution",
            save_prefix=episode_dir / "state_evolution",
            plots_per_figure=args.plots_per_figure,
        )
        print(
            f"Episode {episode_idx}: "
            f"horizon={x_target_seq.shape[0] - 1} "
            f"state_mse={state_mse:.6f} "
            f"lifted_mse={lifted_mse:.6f}"
        )

    if state_mse_values:
        print(f"Mean state MSE across {n_eval_episodes} episodes: {float(np.mean(state_mse_values)):.6f}")
        print(f"Mean lifted MSE across {n_eval_episodes} episodes: {float(np.mean(lifted_mse_values)):.6f}")


if __name__ == "__main__":
    evaluate(parse_args())
