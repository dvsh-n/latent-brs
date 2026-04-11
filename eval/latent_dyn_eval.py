#!/usr/bin/env python3
"""Evaluate latent dynamics rollouts on preprocessed Reacher trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.models import LatentDynamicsModel, make_history_states, rollout_dynamics


DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "latent_dyn_reacher_epoch_1.pt"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "test_data" / "preprocessed"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "latent_dyn_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-eval", type=int, default=10, help="Number of test episodes to evaluate.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Deprecated alias for --num-eval.")
    parser.add_argument(
        "--plot-episodes",
        type=int,
        default=None,
        help="Number of evaluated episodes to plot. Defaults to --num-eval.",
    )
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # This repo's training checkpoint stores argparse Paths inside args, so PyTorch
    # 2.6's default weights_only=True cannot load it.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain model_state_dict.")
    return checkpoint


def build_model(checkpoint: dict, device: torch.device) -> LatentDynamicsModel:
    ckpt_args = checkpoint.get("args", {})
    model = LatentDynamicsModel(
        latent_dim=int(ckpt_args.get("latent_dim", 24)),
        history=int(ckpt_args.get("history", 3)),
        action_dim=int(ckpt_args.get("action_dim", 2)),
        encoder_proj_hidden_dim=int(ckpt_args.get("encoder_proj_hidden_dim", 512)),
        encoder_proj_depth=int(ckpt_args.get("encoder_proj_depth", 1)),
        dynamics_hidden_dim=int(ckpt_args.get("dynamics_hidden_dim", 1024)),
        dynamics_depth=int(ckpt_args.get("dynamics_depth", 3)),
        curvature_weight=float(ckpt_args.get("curvature_weight", 0.0)),
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        raise RuntimeError(
            "Checkpoint is incompatible with the current next-latent dynamics architecture. "
            "Retrain with train/latent_dyn_train.py before evaluating this model."
        ) from None
    model.eval()
    return model


@torch.no_grad()
def encode_episode(
    model: LatentDynamicsModel,
    frames: torch.Tensor,
    *,
    device: torch.device,
    frame_batch_size: int,
    amp_enabled: bool,
) -> torch.Tensor:
    latents = []
    for start in range(0, frames.shape[0], frame_batch_size):
        chunk = frames[start : start + frame_batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            latents.append(model.encoder(chunk))
    return torch.cat(latents, dim=0)


@torch.no_grad()
def rollout_episode(
    model: LatentDynamicsModel,
    frames: torch.Tensor,
    actions: torch.Tensor,
    *,
    seq_len: int,
    device: torch.device,
    frame_batch_size: int,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = model.history
    if seq_len <= history:
        raise ValueError(f"seq_len={seq_len} must be greater than history={history}.")

    latents = encode_episode(
        model,
        frames[:seq_len],
        device=device,
        frame_batch_size=frame_batch_size,
        amp_enabled=amp_enabled,
    )
    initial_state = latents[:history].reshape(1, -1)
    rollout_actions = actions[history - 1 : seq_len - 1].unsqueeze(0).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        pred_states = rollout_dynamics(model.dynamics, initial_state, rollout_actions)

    target_states = make_history_states(latents.unsqueeze(0), history)[:, 1:]
    latent_dim = latents.shape[-1]
    pred_next_latents = pred_states.reshape(1, -1, history, latent_dim)[0, :, -1]
    true_next_latents = latents[history:seq_len]
    return latents[:history], true_next_latents, pred_next_latents, target_states[0], pred_states[0]


def update_error_stats(
    stats: list[dict[str, float]],
    true_next: torch.Tensor,
    pred_next: torch.Tensor,
    target_states: torch.Tensor,
    pred_states: torch.Tensor,
) -> None:
    latent_rmse = (pred_next.float() - true_next.float()).pow(2).mean(dim=-1).sqrt().cpu()
    state_rmse = (pred_states.float() - target_states.float()).pow(2).mean(dim=-1).sqrt().cpu()

    for step_idx in range(latent_rmse.numel()):
        while len(stats) <= step_idx:
            stats.append(
                {
                    "count": 0.0,
                    "latent_sum": 0.0,
                    "latent_sumsq": 0.0,
                    "state_sum": 0.0,
                    "state_sumsq": 0.0,
                }
            )
        row = stats[step_idx]
        latent_value = float(latent_rmse[step_idx])
        state_value = float(state_rmse[step_idx])
        row["count"] += 1.0
        row["latent_sum"] += latent_value
        row["latent_sumsq"] += latent_value * latent_value
        row["state_sum"] += state_value
        row["state_sumsq"] += state_value * state_value


def summarize_errors(stats: list[dict[str, float]], history: int) -> list[dict[str, float]]:
    rows = []
    for step_idx, row in enumerate(stats):
        count = max(row["count"], 1.0)
        latent_mean = row["latent_sum"] / count
        state_mean = row["state_sum"] / count
        latent_var = max(row["latent_sumsq"] / count - latent_mean * latent_mean, 0.0)
        state_var = max(row["state_sumsq"] / count - state_mean * state_mean, 0.0)
        rows.append(
            {
                "rollout_step": step_idx + 1,
                "frame_index": history + step_idx,
                "count": int(row["count"]),
                "latent_rmse_mean": latent_mean,
                "latent_rmse_std": math.sqrt(latent_var),
                "state_rmse_mean": state_mean,
                "state_rmse_std": math.sqrt(state_var),
            }
        )
    return rows


def save_error_csv(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "rollout_step",
        "frame_index",
        "count",
        "latent_rmse_mean",
        "latent_rmse_std",
        "state_rmse_mean",
        "state_rmse_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_error_curves(rows: list[dict[str, float]], path: Path) -> None:
    steps = [row["rollout_step"] for row in rows]
    latent_mean = [row["latent_rmse_mean"] for row in rows]
    latent_std = [row["latent_rmse_std"] for row in rows]
    state_mean = [row["state_rmse_mean"] for row in rows]
    state_std = [row["state_rmse_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, latent_mean, label="next latent RMSE", color="#0072B2")
    ax.fill_between(
        steps,
        [mean - std for mean, std in zip(latent_mean, latent_std)],
        [mean + std for mean, std in zip(latent_mean, latent_std)],
        color="#0072B2",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(steps, state_mean, label="history state RMSE", color="#D55E00")
    ax.fill_between(
        steps,
        [mean - std for mean, std in zip(state_mean, state_std)],
        [mean + std for mean, std in zip(state_mean, state_std)],
        color="#D55E00",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("rollout step")
    ax.set_ylabel("RMSE")
    ax.set_title("Latent dynamics rollout error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_latent_rollout(
    initial_latents: torch.Tensor,
    true_next: torch.Tensor,
    pred_next: torch.Tensor,
    *,
    episode_idx: int,
    history: int,
    path: Path,
) -> None:
    true_latents = torch.cat((initial_latents, true_next), dim=0)
    pred_latents = torch.cat((initial_latents, pred_next), dim=0)
    true_np = true_latents.detach().float().cpu().numpy()
    pred_np = pred_latents.detach().float().cpu().numpy()
    rollout_steps, latent_dim = true_np.shape
    x = list(range(rollout_steps))
    ncols = 4
    nrows = math.ceil(latent_dim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(2.0 * nrows, 4.0)), sharex=True)
    flat_axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]

    for dim in range(latent_dim):
        ax = flat_axes[dim]
        ax.plot(x, true_np[:, dim], label="true", color="#0072B2", linewidth=1.3)
        ax.plot(x, pred_np[:, dim], label="pred", color="#D55E00", linewidth=1.1, linestyle="--")
        ax.axvline(history - 0.5, color="#666666", linewidth=0.8, alpha=0.5)
        ax.set_title(f"z[{dim}]")
        ax.grid(True, alpha=0.25)
    for ax in flat_axes[latent_dim:]:
        ax.axis("off")

    flat_axes[0].legend(loc="best")
    fig.supxlabel("frame index; prediction starts after the vertical line")
    fig.supylabel("latent value")
    fig.suptitle(f"Episode {episode_idx:05d}: true encoding vs predicted rollout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_report_rows(rows: list[dict[str, float]], report_every: int) -> list[dict[str, float]]:
    selected = []
    for row in rows:
        step = int(row["rollout_step"])
        if step == 1 or step % report_every == 0 or step == int(rows[-1]["rollout_step"]):
            selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    obses_dir = data_dir / "obses"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")
    if not obses_dir.is_dir():
        raise FileNotFoundError(f"Observation directory not found: {obses_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(checkpoint, device)
    history = model.history
    amp_enabled = args.amp and device.type == "cuda"

    actions = torch.load(data_dir / "actions.pth", map_location="cpu", weights_only=False).float()
    seq_lengths = torch.load(data_dir / "seq_lengths.pth", map_location="cpu", weights_only=False).long()
    total_episodes = int(seq_lengths.numel())
    requested_episodes = args.num_eval if args.max_episodes is None else args.max_episodes
    episodes_to_eval = min(requested_episodes, total_episodes)
    plot_episodes = episodes_to_eval if args.plot_episodes is None else min(args.plot_episodes, episodes_to_eval)

    stats: list[dict[str, float]] = []
    plotted = 0
    for episode_idx in tqdm(range(episodes_to_eval), desc="Evaluating", unit="episode"):
        seq_len = int(seq_lengths[episode_idx])
        frame_path = obses_dir / f"episode_{episode_idx:05d}.pth"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing preprocessed episode: {frame_path}")
        frames = torch.load(frame_path, map_location="cpu", weights_only=False).float()
        initial_latents, true_next, pred_next, target_states, pred_states = rollout_episode(
            model,
            frames,
            actions[episode_idx],
            seq_len=seq_len,
            device=device,
            frame_batch_size=args.frame_batch_size,
            amp_enabled=amp_enabled,
        )
        update_error_stats(stats, true_next, pred_next, target_states, pred_states)

        if not args.no_plots and plotted < plot_episodes:
            plot_latent_rollout(
                initial_latents,
                true_next,
                pred_next,
                episode_idx=episode_idx,
                history=history,
                path=out_dir / f"latent_rollout_episode_{episode_idx:05d}.png",
            )
            plotted += 1

    rows = summarize_errors(stats, history)
    save_error_csv(rows, out_dir / "rollout_errors_by_step.csv")
    if not args.no_plots:
        plot_error_curves(rows, out_dir / "rollout_error_by_step.png")

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "amp": amp_enabled,
        "episodes_evaluated": episodes_to_eval,
        "history": history,
        "latent_dim": model.latent_dim,
        "rollout_steps": len(rows),
        "plots_saved": plotted,
        "error_csv": str(out_dir / "rollout_errors_by_step.csv"),
        "error_plot": None if args.no_plots else str(out_dir / "rollout_error_by_step.png"),
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nErrors by rollout step:")
    for row in select_report_rows(rows, args.report_every):
        print(
            "step={rollout_step:02d} frame={frame_index:02d} "
            "latent_rmse={latent_rmse_mean:.6f}+/-{latent_rmse_std:.6f} "
            "state_rmse={state_rmse_mean:.6f}+/-{state_rmse_std:.6f} "
            "n={count}".format(**row)
        )


if __name__ == "__main__":
    main()
