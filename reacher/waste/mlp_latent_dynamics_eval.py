#!/usr/bin/env python3
"""Evaluate the GELU MLP latent dynamics baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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

from test.mlp_latent_dynamics_train import LatentDynamicsMLP


DATA_PATH = REPO_ROOT / "data" / "expert_data" / "latent_traj_lewm_reacher_24D.pt"
MODEL_SAVE_PATH = REPO_ROOT / "test" / "mlp_lewm_reacher_24D.pt"
OUTPUT_DIR = REPO_ROOT / "test" / "mlp_latent_dynamics_eval"
N_EVAL_EPISODES = 25
MAX_TIMESTEP = 50
START_TIME = 2
OVERALL_RMSE_STEPS = [1, 10, 25, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--max-timestep", type=int, default=MAX_TIMESTEP)
    parser.add_argument("--start-time", type=int, default=START_TIME)
    parser.add_argument("--use-val-episodes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_padded_history_states(latents: torch.Tensor, history: int) -> torch.Tensor:
    _, num_frames, _ = latents.shape
    frame_idx = torch.arange(num_frames)
    slices = []
    for offset in range(history):
        source_idx = (frame_idx - history + 1 + offset).clamp_min(0)
        slices.append(latents[:, source_idx])
    return torch.stack(slices, dim=2).flatten(start_dim=2)


def load_history_states(payload: dict, history: int) -> torch.Tensor:
    latents = payload["latents"]
    state_dim = int(latents.shape[-1])
    expected_shape = (latents.shape[0], latents.shape[1], history * state_dim)
    metadata = payload.get("metadata", {})
    history_states = payload.get("history_states")
    if (
        history_states is not None
        and int(metadata.get("history", history)) == history
        and tuple(history_states.shape) == expected_shape
    ):
        return history_states.float()
    return make_padded_history_states(latents.float(), history)


def calculate_rmse(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    horizon = min(len(target), len(pred))
    err = target[:horizon] - pred[:horizon]
    return np.sqrt(np.mean(err**2, axis=0))


def plot_rollout(target: np.ndarray, pred: np.ndarray, episode_idx: int, output_dir: Path) -> None:
    horizon = min(len(target), len(pred))
    state_dim = target.shape[-1]
    cols = 4
    rows = int(np.ceil(state_dim / cols))
    time_ax = np.arange(horizon)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.0 * rows), sharex=True, constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    fig.suptitle(f"MLP Latent Rollout vs. LE-WM Latents (Episode {episode_idx})", fontsize=16)

    for dim, ax in enumerate(axes_flat):
        if dim >= state_dim:
            ax.set_axis_off()
            continue
        ax.plot(time_ax, target[:horizon, dim], label="LE-WM latent", color="blue", linewidth=1.4)
        ax.plot(time_ax, pred[:horizon, dim], label="MLP rollout", color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(f"z{dim}")
        ax.grid(True, linestyle=":", alpha=0.6)

    for ax in axes_flat[-cols:]:
        if ax.has_data():
            ax.set_xlabel("Time Step")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"episode_{episode_idx}.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def rollout_episode(
    model: LatentDynamicsMLP,
    history0: torch.Tensor,
    actions: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    normalized_actions = (torch.nan_to_num(actions, nan=0.0) - action_mean) / action_std
    history0 = history0.unsqueeze(0).to(device)
    normalized_actions = normalized_actions.unsqueeze(0).to(device)
    pred = model.rollout(history0, normalized_actions).squeeze(0).cpu()
    return torch.cat((history0.cpu()[:, -model.state_dim :].squeeze(0).unsqueeze(0), pred), dim=0).numpy()


def main() -> None:
    args = parse_args()
    data_path = args.data_path.expanduser().resolve()
    model_save_path = args.model_save_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not model_save_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)
    model = LatentDynamicsMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    action_mean = checkpoint["action_stats"]["mean"].float()
    action_std = checkpoint["action_stats"]["std"].float()

    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    latents = payload["latents"].float()
    actions = payload.get("action_sequences", payload["actions"][:, :-1]).float()
    ep_len = payload["ep_len"].long()
    history_states = load_history_states(payload, model.history)

    eval_episodes = checkpoint.get("training_config", {}).get("val_episodes", []) if args.use_val_episodes else []
    if not eval_episodes:
        eval_episodes = list(range(int(latents.shape[0])))
    eval_episodes = eval_episodes[: args.n_eval_episodes]

    output_dir.mkdir(parents=True, exist_ok=True)
    state_errs = []
    step_mse = {step: [] for step in OVERALL_RMSE_STEPS}
    episode_rows = []

    for episode_idx in tqdm(eval_episodes, desc="Evaluating episodes"):
        length = int(ep_len[episode_idx].item())
        if not 0 <= args.start_time < length:
            raise ValueError(f"start_time={args.start_time} is outside episode {episode_idx} length {length}.")
        horizon = length - args.start_time - 1
        if args.max_timestep is not None and args.max_timestep > 0:
            horizon = min(horizon, args.max_timestep)

        target = latents[episode_idx, args.start_time : args.start_time + horizon + 1]
        u_seq = actions[episode_idx, args.start_time : args.start_time + horizon]
        history0 = history_states[episode_idx, args.start_time]
        pred = rollout_episode(model, history0, u_seq, action_mean, action_std, device)
        target_np = target.numpy()

        plot_rollout(target_np, pred, int(episode_idx), output_dir)
        rmse_state = calculate_rmse(target_np, pred)
        state_errs.append(rmse_state)
        episode_rows.append({"episode": int(episode_idx), "mean_rmse": float(rmse_state.mean())})

        err = target_np[: len(pred)] - pred[: len(target_np)]
        for step in OVERALL_RMSE_STEPS:
            if step < len(err):
                step_mse[step].append(float(np.mean(err[step] ** 2)))

    rmse_per_dim = np.asarray(state_errs).mean(axis=0)
    horizon_rmse = {
        f"{step}_step": (float(np.sqrt(np.mean(values))) if values else None)
        for step, values in step_mse.items()
    }
    metrics = {
        "model_save_path": str(model_save_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "n_eval_episodes": len(eval_episodes),
        "start_time": args.start_time,
        "max_timestep": args.max_timestep,
        "mean_latent_rmse": float(rmse_per_dim.mean()),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "horizon_rmse": horizon_rmse,
        "episodes": episode_rows,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ["mean_latent_rmse", "horizon_rmse", "n_eval_episodes"]}, indent=2))
    print(f"Saved plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
