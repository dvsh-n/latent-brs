#!/usr/bin/env python3
"""Evaluate Koopman linear-decoder JEPA rollouts from `koopdyn_ft` object checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_DATASET_PATH = "reacher/data/test_data/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/koopdyn_ft"
DEFAULT_OUT_DIR = "reacher/eval/koopdyn_lindec_eval"
DEFAULT_OVERALL_RMSE_STEPS = (1, 10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--action-stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--num-preds", type=int, default=None)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument(
        "--pure-latent-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pure latent rollout instead of re-lifting decoded states after each step.",
    )
    return parser.parse_args()


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open() as handle:
        return json.load(handle)


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def apply_config_defaults(args: argparse.Namespace, config: dict[str, object]) -> None:
    defaults = {
        "history_size": 2,
        "num_preds": 15,
        "frameskip": 1,
        "img_size": 224,
        "action_dim": 2,
        "embed_dim": 18,
    }
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, int(config.get(key, fallback)))


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def compute_action_stats(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]}.")
        finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
    return action_mean, np.maximum(action_std, 1e-6)


def valid_episode_indices(dataset_path: Path, *, history_size: int, frameskip: int) -> np.ndarray:
    min_episode_len = history_size * frameskip + 1
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    return np.flatnonzero(ep_len >= min_episode_len)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    if not hasattr(model, "predictor"):
        raise TypeError(f"Expected a JEPA-like model with `.predictor`, got {type(model).__name__}.")
    return model


def load_episode(
    dataset_path: Path,
    episode_idx: int,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        pixels = torch.from_numpy(np.asarray(h5["pixels"][rows], dtype=np.uint8)).permute(0, 3, 1, 2).contiguous()
        actions_np = np.asarray(h5["action"][rows], dtype=np.float32)
    actions_np = (np.nan_to_num(actions_np, nan=0.0) - action_mean) / action_std
    actions = torch.from_numpy(actions_np).float()
    return pixels, actions


def build_markov_pairs(emb: torch.Tensor) -> torch.Tensor:
    current = emb[1:]
    previous = emb[:-1]
    delta = current - previous
    return torch.cat((current, delta), dim=-1)


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    frame_batch_size: int,
    img_size: int,
) -> torch.Tensor:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device, non_blocking=True).float().div_(255.0)
        if chunk.shape[-2:] != (img_size, img_size):
            chunk = torch.nn.functional.interpolate(
                chunk,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        chunk = (chunk - pixel_mean) / pixel_std
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    return torch.cat(latents, dim=0)


def rollout_markov_states(
    model: torch.nn.Module,
    true_markov_states: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_size: int,
    frameskip: int,
    max_rollout_steps: int | None,
    pure_latent_rollout: bool,
) -> tuple[torch.Tensor, int]:
    predictor = model.predictor
    start_idx = history_size - 2
    total_future_steps = (true_markov_states.shape[0] - 1 - start_idx) // frameskip
    if max_rollout_steps is not None:
        total_future_steps = min(total_future_steps, max_rollout_steps)
    if total_future_steps < 1:
        raise ValueError("Not enough timesteps for a rollout with the requested history/frameskip settings.")

    current_state = true_markov_states[start_idx].unsqueeze(0)
    z_k = predictor.lift_state(current_state)
    predicted_states = [current_state.squeeze(0)]

    for step in range(total_future_steps):
        action_start = (history_size - 1 + step) * frameskip
        action_stop = action_start + frameskip
        control = actions[action_start:action_stop].reshape(1, -1).to(current_state.device)
        z_k = predictor.A(z_k) + predictor.B(control)
        decoded_state = predictor.decode(z_k)
        predicted_states.append(decoded_state.squeeze(0))
        if not pure_latent_rollout:
            z_k = predictor.lift_state(decoded_state)

    return torch.stack(predicted_states, dim=0), start_idx


def compute_metrics(
    true_markov_states: torch.Tensor,
    pred_markov_states: torch.Tensor,
    *,
    start_idx: int,
) -> dict[str, object]:
    length = min(true_markov_states.shape[0] - start_idx, pred_markov_states.shape[0])
    if length <= 1:
        raise ValueError("Rollout is not longer than the warm-start context.")
    true = true_markov_states[start_idx : start_idx + length].float().cpu()
    pred = pred_markov_states[:length].float().cpu()
    err = pred[1:] - true[1:]
    rmse_per_step = err.pow(2).mean(dim=-1).sqrt()
    rmse_per_dim = err.pow(2).mean(dim=0).sqrt()
    split = true.shape[-1] // 2
    return {
        "num_context_steps": 1,
        "num_rollout_steps": int(err.shape[0]),
        "markov_state_dim": int(true.shape[-1]),
        "mean_rmse": float(rmse_per_step.mean()),
        "final_rmse": float(rmse_per_step[-1]),
        "max_rmse": float(rmse_per_step.max()),
        "mean_rmse_embedding": float(rmse_per_dim[:split].mean()),
        "mean_rmse_delta": float(rmse_per_dim[split:].mean()),
        "rmse_per_step": rmse_per_step.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
    }


def plot_rollout(
    true_markov_states: torch.Tensor,
    pred_markov_states: torch.Tensor,
    *,
    out_dir: Path,
    episode_idx: int,
    start_idx: int,
) -> Path:
    length = min(true_markov_states.shape[0] - start_idx, pred_markov_states.shape[0])
    true = true_markov_states[start_idx : start_idx + length].float().cpu().numpy()
    pred = pred_markov_states[:length].float().cpu().numpy()
    state_dim = true.shape[-1]
    split = state_dim // 2
    fig, axes = plt.subplots(state_dim, 1, figsize=(14, max(10, 1.7 * state_dim)), sharex=True)
    if state_dim == 1:
        axes = [axes]
    steps = np.arange(length)
    for dim, axis in enumerate(axes):
        axis.plot(steps, true[:, dim], label="true", linewidth=1.5)
        axis.plot(steps, pred[:, dim], label="rollout", linewidth=1.2, linestyle="--")
        axis.axvline(0.5, color="black", alpha=0.25, linewidth=1)
        axis.set_ylabel(f"x{dim}")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("markov timestep from rollout start")
    fig.suptitle(
        f"Koopman linear-decoder rollout episode {episode_idx}, "
        f"embedding dims 0-{split - 1}, delta dims {split}-{state_dim - 1}"
    )
    fig.tight_layout()
    path = out_dir / f"episode_{episode_idx:05d}_rollout.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
    elif args.checkpoint is not None:
        model_dir = args.checkpoint.expanduser().resolve().parent
    else:
        model_dir = Path(DEFAULT_MODEL_DIR).expanduser().resolve()

    config = load_config(model_dir)
    apply_config_defaults(args, config)
    dataset_path = args.dataset_path.expanduser().resolve()
    action_stats_dataset_path = (
        args.action_stats_dataset_path.expanduser().resolve()
        if args.action_stats_dataset_path is not None
        else Path(config.get("dataset_path", dataset_path)).expanduser().resolve()
    )
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.history_size < 2:
        raise ValueError("history_size must be at least 2 for Koopman markov-state evaluation.")
    device = require_device(args.device)
    model = load_model(checkpoint_path, device)
    action_mean, action_std = compute_action_stats(action_stats_dataset_path, args.action_dim)
    valid_episodes = valid_episode_indices(
        dataset_path,
        history_size=args.history_size,
        frameskip=args.frameskip,
    )
    if valid_episodes.size == 0:
        raise ValueError("No episodes are long enough for the requested history/frameskip settings.")

    n_eval_episodes = min(args.n_eval_episodes, int(valid_episodes.size))
    selected_episodes = valid_episodes[:n_eval_episodes]
    episode_summaries: list[dict[str, object]] = []
    rmse_by_dim: list[np.ndarray] = []
    horizon_mse: dict[int, list[float]] = {step: [] for step in DEFAULT_OVERALL_RMSE_STEPS}
    plot_paths: list[str] = []

    for episode_idx in selected_episodes.tolist():
        pixels, actions = load_episode(
            dataset_path,
            episode_idx,
            action_mean=action_mean,
            action_std=action_std,
        )
        latents = encode_frames(
            model,
            pixels,
            device=device,
            frame_batch_size=args.frame_batch_size,
            img_size=args.img_size,
        )
        markov_states = build_markov_pairs(latents)
        pred_markov_states, start_idx = rollout_markov_states(
            model,
            markov_states,
            actions,
            history_size=args.history_size,
            frameskip=args.frameskip,
            max_rollout_steps=args.max_rollout_steps,
            pure_latent_rollout=args.pure_latent_rollout,
        )
        metrics = compute_metrics(markov_states, pred_markov_states, start_idx=start_idx)
        metrics["episode_idx"] = int(episode_idx)
        episode_summaries.append(metrics)
        rmse_by_dim.append(np.asarray(metrics["rmse_per_dim"], dtype=np.float64))

        true_segment = markov_states[start_idx : start_idx + pred_markov_states.shape[0]].float().cpu()
        pred_segment = pred_markov_states[: true_segment.shape[0]].float().cpu()
        err = (pred_segment[1:] - true_segment[1:]).numpy()
        for step in DEFAULT_OVERALL_RMSE_STEPS:
            if step <= err.shape[0]:
                horizon_mse[step].append(float(np.mean(err[step - 1] ** 2)))

        plot_paths.append(
            str(
                plot_rollout(
                    markov_states,
                    pred_markov_states,
                    out_dir=out_dir,
                    episode_idx=episode_idx,
                    start_idx=start_idx,
                )
            )
        )

    rmse_by_dim_np = np.stack(rmse_by_dim, axis=0)
    mean_rmse_per_dim = rmse_by_dim_np.mean(axis=0)
    split = mean_rmse_per_dim.shape[0] // 2
    summary = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(model_dir / "config.json"),
        "dataset_path": str(dataset_path),
        "action_stats_dataset_path": str(action_stats_dataset_path),
        "n_eval_episodes": int(n_eval_episodes),
        "history_size": int(args.history_size),
        "num_preds": int(args.num_preds),
        "frameskip": int(args.frameskip),
        "img_size": int(args.img_size),
        "embed_dim": int(args.embed_dim),
        "markov_state_dim": int(2 * args.embed_dim),
        "pure_latent_rollout": bool(args.pure_latent_rollout),
        "max_rollout_steps": int(args.max_rollout_steps) if args.max_rollout_steps is not None else None,
        "mean_rmse_per_dim": mean_rmse_per_dim.tolist(),
        "mean_rmse_embedding": float(mean_rmse_per_dim[:split].mean()),
        "mean_rmse_delta": float(mean_rmse_per_dim[split:].mean()),
        "mean_rmse_all": float(mean_rmse_per_dim.mean()),
        "horizon_rmse": {
            str(step): (float(np.sqrt(np.mean(values))) if values else None)
            for step, values in horizon_mse.items()
        },
        "episodes": episode_summaries,
        "plot_paths": plot_paths,
    }
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        json.dumps(
            {
                "metrics_summary_path": str(summary_path),
                "mean_rmse_embedding": summary["mean_rmse_embedding"],
                "mean_rmse_delta": summary["mean_rmse_delta"],
                "mean_rmse_all": summary["mean_rmse_all"],
                "n_eval_episodes": n_eval_episodes,
                "plot_count": len(plot_paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
