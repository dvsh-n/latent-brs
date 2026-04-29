#!/usr/bin/env python3
"""Evaluate a linear-decoder Koopman model on offline Reacher Markov-state trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.shared.models import DeepKoopmanLinDec


DATA_PATH = "reacher/data/expert_data/reacher_koopman_markov.pt"
MODEL_SAVE_PATH = "reacher/models/koopdyn_lindec/koopman_lindec.pt"
PLOT_SAVE_DIR = "reacher/eval/koopdyn_lindec_eval"
N_EVAL_EPISODES = 25
MAX_TIMESTEP = 100
PURE_LATENT_ROLLOUT = True
OVERALL_RMSE_STEPS = [1, 10, 25, 50, 100]


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-save-path", type=Path, default=MODEL_SAVE_PATH)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--plot-save-dir", type=Path, default=PLOT_SAVE_DIR)
    parser.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--max-timestep", type=int, default=MAX_TIMESTEP)
    parser.add_argument(
        "--pure-latent-rollout",
        action=argparse.BooleanOptionalAction,
        default=PURE_LATENT_ROLLOUT,
        help="Use pure latent rollout instead of recurrent re-lifting through the encoder.",
    )
    return parser.parse_args()


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


def print_table(title: str, headers: list[str], rows: list[tuple[object, ...]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    total_width = sum(widths) + 3 * (len(headers) - 1)
    print(f"\n{title}")
    print("-" * total_width)
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-" * total_width)
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    print("-" * total_width)


class ReacherMarkovEvalDataset:
    """Evaluation dataset with optional min-max normalization."""

    def __init__(self, data_path: Path) -> None:
        payload = torch.load(data_path, weights_only=False)
        self.metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        demos = payload["demos"] if isinstance(payload, dict) else payload
        if not demos:
            raise ValueError(f"No demonstrations found in {data_path}.")

        self.states_data: list[torch.Tensor] = []
        self.controls_data: list[torch.Tensor] = []
        for demo in demos:
            self.states_data.append(torch.as_tensor(demo["states"], dtype=torch.float32).T.contiguous())
            self.controls_data.append(torch.as_tensor(demo["controls"], dtype=torch.float32).T.contiguous())

        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None
        self.range: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.states_data)

    def set_normalization_stats(self, stats: dict[str, torch.Tensor]) -> None:
        self.min = torch.as_tensor(stats["min"], dtype=torch.float32)
        self.max = torch.as_tensor(stats["max"], dtype=torch.float32)
        self.range = torch.as_tensor(stats.get("range", self.max - self.min), dtype=torch.float32)
        self.range[self.range == 0] = 1e-6

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        return 2 * (x - self.min.to(x.device)) / self.range.to(x.device) - 1

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.range is None:
            raise ValueError("Normalization stats not set. Call set_normalization_stats() first.")
        return (x + 1) / 2 * self.range.to(x.device) + self.min.to(x.device)

    def preprocess(self, x: torch.Tensor, enable_normalization: bool) -> torch.Tensor:
        return self.normalize(x) if enable_normalization else x

    def postprocess(self, x: torch.Tensor, enable_normalization: bool) -> torch.Tensor:
        return self.denormalize(x) if enable_normalization else x

    def get_raw_episode(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states_data[idx], self.controls_data[idx]


@torch.no_grad()
def rollout_latent(
    model: DeepKoopmanLinDec,
    x0_raw: torch.Tensor,
    u_raw: torch.Tensor,
    dataset: ReacherMarkovEvalDataset,
    device: torch.device,
    enable_normalization: bool,
) -> np.ndarray:
    horizon = len(u_raw)
    x0_proc = dataset.preprocess(x0_raw.unsqueeze(0).to(device), enable_normalization)
    z_k = model.encoder(x0_proc)

    x_hat_proc_list = [model.decode(z_k).squeeze(0)]
    for step in range(horizon):
        u_k = u_raw[step].unsqueeze(0).to(device)
        z_k = model.A(z_k) + model.B(u_k)
        x_hat_proc_list.append(model.decode(z_k).squeeze(0))

    x_hat_proc = torch.stack(x_hat_proc_list, dim=0)
    x_hat = dataset.postprocess(x_hat_proc, enable_normalization)
    return x_hat.cpu().numpy()


@torch.no_grad()
def rollout_recurrent(
    model: DeepKoopmanLinDec,
    x0_raw: torch.Tensor,
    u_raw: torch.Tensor,
    dataset: ReacherMarkovEvalDataset,
    device: torch.device,
    enable_normalization: bool,
) -> np.ndarray:
    horizon = len(u_raw)
    x_hat_raw_list = [x0_raw.to(device)]
    for step in range(horizon):
        x_k_proc = dataset.preprocess(x_hat_raw_list[-1].unsqueeze(0), enable_normalization)
        z_k = model.encoder(x_k_proc)
        u_k = u_raw[step].unsqueeze(0).to(device)
        z_kp1 = model.A(z_k) + model.B(u_k)
        x_kp1_proc = model.decode(z_kp1)
        x_kp1_raw = dataset.postprocess(x_kp1_proc, enable_normalization).squeeze(0)
        x_hat_raw_list.append(x_kp1_raw)
    return torch.stack(x_hat_raw_list, dim=0).cpu().numpy()


def calculate_rmse(x_traj: np.ndarray, x_hat_traj: np.ndarray) -> np.ndarray:
    horizon = min(len(x_traj), len(x_hat_traj))
    err = x_traj[:horizon] - x_hat_traj[:horizon]
    return np.sqrt(np.mean(err**2, axis=0))


def plot_rollout(x_traj: np.ndarray, x_hat_traj: np.ndarray, episode_idx: int, plot_save_dir: Path) -> Path:
    state_dim = x_traj.shape[1]
    split = state_dim // 2
    sections = [
        ("Embedding", 0, split),
        ("Delta", split, state_dim),
    ]

    fig, axes = plt.subplots(state_dim, 1, figsize=(14, max(10, 1.7 * state_dim)), sharex=True)
    if state_dim == 1:
        axes = [axes]
    horizon = min(len(x_traj), len(x_hat_traj))
    time_ax = np.arange(horizon)

    for dim, axis in enumerate(axes):
        axis.plot(time_ax, x_traj[:horizon, dim], label="Ground Truth", color="blue", linewidth=1.6)
        axis.plot(time_ax, x_hat_traj[:horizon, dim], label="Koopman Rollout", color="red", linestyle="--", linewidth=1.4)
        axis.set_ylabel(f"x{dim}")
        axis.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Time Step")
    axes[0].legend(loc="upper right")
    title_suffix = ", ".join(f"{name}: {start}-{end - 1}" for name, start, end in sections)
    fig.suptitle(f"Linear-Decoder Koopman Rollout vs. Ground Truth (Episode {episode_idx})\n{title_suffix}", fontsize=16)
    fig.tight_layout()

    plot_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_save_dir / f"episode_{episode_idx:05d}.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def evaluate(
    model_save_path_arg: Path | None = None,
    data_path_arg: Path | None = None,
    plot_save_dir_arg: Path | None = None,
    *,
    n_eval_episodes: int = N_EVAL_EPISODES,
    max_timestep: int = MAX_TIMESTEP,
    pure_latent_rollout: bool = PURE_LATENT_ROLLOUT,
) -> None:
    model_save_path = (model_save_path_arg or Path(MODEL_SAVE_PATH)).expanduser().resolve()
    data_path = (data_path_arg or Path(DATA_PATH)).expanduser().resolve()
    plot_save_dir = (plot_save_dir_arg or Path(PLOT_SAVE_DIR)).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(model_save_path, map_location=device, weights_only=False)

    dataset = ReacherMarkovEvalDataset(data_path)
    if "normalization_stats" not in model_data:
        raise ValueError(f"'normalization_stats' not found in model file: {model_save_path}")
    dataset.set_normalization_stats(model_data["normalization_stats"])

    model_config = model_data["model_config"]
    training_config = model_data.get("training_config", {})
    enable_normalization = bool(training_config.get("enable_normalization", False))

    print_config(model_config, title="Model Configuration")
    if training_config:
        print_config(training_config, title="Training Configuration")

    model = DeepKoopmanLinDec(**model_config).to(device)
    model.load_state_dict(model_data["state_dict"])
    model.eval()

    n_eval_episodes = min(n_eval_episodes, len(dataset))
    state_errs = []
    step_mse = {step: [] for step in OVERALL_RMSE_STEPS}
    plot_paths: list[str] = []

    pbar = tqdm(range(n_eval_episodes), desc="Evaluating episodes")
    for ep_id in pbar:
        x_traj, u = dataset.get_raw_episode(ep_id)
        if max_timestep is not None and max_timestep > 0:
            u = u[:max_timestep]
            x_traj = x_traj[: max_timestep + 1]
        x0 = x_traj[0]

        if pure_latent_rollout:
            x_hat_traj = rollout_latent(model, x0, u, dataset, device, enable_normalization)
        else:
            x_hat_traj = rollout_recurrent(model, x0, u, dataset, device, enable_normalization)

        x_traj_np = x_traj.cpu().numpy()
        plot_paths.append(str(plot_rollout(x_traj_np, x_hat_traj, ep_id, plot_save_dir)))

        rmse_state = calculate_rmse(x_traj_np, x_hat_traj)
        state_errs.append(rmse_state)

        horizon = min(len(x_traj_np), len(x_hat_traj))
        err = x_traj_np[:horizon] - x_hat_traj[:horizon]
        for step in OVERALL_RMSE_STEPS:
            if step < horizon:
                step_mse[step].append(float(np.mean(err[step] ** 2)))

        pbar.set_postfix(mean_embed_err=f"{np.mean(np.array(state_errs)[:, : model_config['state_dim'] // 2]):.4f}")

    state_errs_np = np.asarray(state_errs)
    rmse_state_mean = state_errs_np.mean(axis=0)
    split = model_config["state_dim"] // 2

    grouped_rows = [
        ("Embedding", f"{rmse_state_mean[:split].mean():.6f}"),
        ("Delta", f"{rmse_state_mean[split:].mean():.6f}"),
        ("All", f"{rmse_state_mean.mean():.6f}"),
    ]
    print_table("Linear-Decoder Koopman Reacher Metrics (RMSE)", ["Group", "RMSE"], grouped_rows)

    horizon_rows = []
    for step in OVERALL_RMSE_STEPS:
        count = len(step_mse[step])
        rmse_val = np.sqrt(np.mean(step_mse[step])) if count > 0 else np.nan
        horizon_rows.append((f"{step}-step", f"{rmse_val:.6f}" if count > 0 else "N/A", f"{count}/{n_eval_episodes}"))
    print_table("Overall Horizon RMSE", ["Horizon", "RMSE", "Episodes Used"], horizon_rows)

    summary = {
        "model_save_path": str(model_save_path),
        "data_path": str(data_path),
        "plot_save_dir": str(plot_save_dir),
        "n_eval_episodes": int(n_eval_episodes),
        "max_timestep": int(max_timestep) if max_timestep is not None else None,
        "pure_latent_rollout": bool(pure_latent_rollout),
        "enable_normalization": enable_normalization,
        "state_dim": int(model_config["state_dim"]),
        "control_dim": int(model_config["control_dim"]),
        "latent_dim": int(model_config["latent_dim"]),
        "mean_rmse_per_dim": rmse_state_mean.tolist(),
        "mean_rmse_embedding": float(rmse_state_mean[:split].mean()),
        "mean_rmse_delta": float(rmse_state_mean[split:].mean()),
        "mean_rmse_all": float(rmse_state_mean.mean()),
        "horizon_rmse": {
            str(step): (float(np.sqrt(np.mean(step_mse[step]))) if step_mse[step] else None)
            for step in OVERALL_RMSE_STEPS
        },
        "plot_paths": plot_paths,
    }
    summary_path = plot_save_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "metrics_summary_path": str(summary_path),
                "mean_rmse_embedding": summary["mean_rmse_embedding"],
                "mean_rmse_delta": summary["mean_rmse_delta"],
                "mean_rmse_all": summary["mean_rmse_all"],
                "plot_count": len(plot_paths),
            },
            indent=2,
        )
    )


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
