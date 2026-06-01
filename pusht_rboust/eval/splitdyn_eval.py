#!/usr/bin/env python3
"""Evaluate split latent dynamics rollouts on one PushT trajectory."""

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

from pusht_rboust.train.splitdyn_ft import (
    LeWMPushTDataset,
    SplitBodyDynamicsPredictor,
    SplitDynamicsModel,
    build_markov_state,
    preprocess_pixels,
    required_markov_history,
)

DEFAULT_DATASET_PATH = "pusht_rboust/data/pusht_expert_train_preproc.h5"
DEFAULT_MODEL_DIR = "pusht_rboust/models/splitdyn_ft"
DEFAULT_OUT_DIR = "pusht_rboust/eval/splitdyn_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--markov-deriv", type=int, default=None)
    parser.add_argument("--num-preds", type=int, default=None)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    return parser.parse_args()


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open() as f:
        return json.load(f)


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
        "markov_deriv": 1,
        "num_preds": 5,
        "frameskip": 1,
        "img_size": 224,
        "action_dim": 2,
    }
    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, fallback))


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def valid_episode_indices(dataset_path: Path, *, args: argparse.Namespace) -> np.ndarray:
    if int(args.markov_deriv) < 0:
        raise ValueError("markov_deriv must be non-negative.")
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    num_steps = 1 + int(args.num_preds)
    required_last_frame_offset = (num_steps - 1) * int(args.frameskip)
    action_steps = int(args.num_preds)
    required_action_end_offset = action_steps * int(args.frameskip)
    required_offset = max(required_last_frame_offset, required_action_end_offset)
    return np.flatnonzero(ep_len - 1 - required_offset >= 0)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_episode(
    dataset_path: Path,
    episode_idx: int,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = LeWMPushTDataset(
        dataset_path,
        markov_deriv=args.markov_deriv,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        action_dim=args.action_dim,
    )
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()

        actions = np.asarray(h5["action"][rows], dtype=np.float32)
        actions = (np.nan_to_num(actions, nan=0.0) - dataset.action_mean) / dataset.action_std
        actions = torch.from_numpy(actions).float()
    return pixels, actions


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    img_size: int,
    frame_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_latents = []
    body_latents = []
    pusher_latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        chunk = preprocess_pixels(chunk.unsqueeze(0), img_size)[0]
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        full_latents.append(emb)
        body_latents.append(emb[:, : model.body_dim])
        pusher_latents.append(emb[:, model.body_dim :])
    return (
        torch.cat(full_latents, dim=0),
        torch.cat(body_latents, dim=0),
        torch.cat(pusher_latents, dim=0),
    )


@torch.no_grad()
def rollout_latents(
    model: torch.nn.Module,
    true_body_latents: torch.Tensor,
    true_pusher_latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    markov_deriv: int,
    frameskip: int,
    max_rollout_steps: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = true_body_latents.device
    rollout_steps = (true_body_latents.shape[0] - 1) // frameskip
    if max_rollout_steps is not None:
        rollout_steps = min(rollout_steps, max_rollout_steps)
    if rollout_steps < 1:
        raise ValueError("Not enough actions for a rollout with the requested history/frameskip.")
    if markov_deriv < 0:
        raise ValueError("markov_deriv must be non-negative.")

    history_len = required_markov_history(markov_deriv)
    body_history = true_body_latents[:1]
    pusher_history = true_pusher_latents[:1]
    if history_len > 1:
        body_history = torch.cat((body_history[:1].repeat(history_len - 1, 1), body_history), dim=0)
        pusher_history = torch.cat((pusher_history[:1].repeat(history_len - 1, 1), pusher_history), dim=0)
    body_state = build_markov_state(body_history.unsqueeze(0), markov_deriv)
    pusher_state = build_markov_state(pusher_history.unsqueeze(0), markov_deriv)
    pred_body = [true_body_latents[0]]
    pred_pusher = [true_pusher_latents[0]]

    for step in range(rollout_steps):
        action_start = step * frameskip
        action_stop = action_start + frameskip
        act = actions[action_start:action_stop].reshape(1, 1, -1).to(device)

        pred_pusher_state = model.predict_pusher(pusher_state.unsqueeze(1), act)[:, 0]
        pred_pusher_next = pred_pusher_state[..., : model.pusher_dim]
        pred_body_state = model.predict_body(body_state.unsqueeze(1), pred_pusher_state.unsqueeze(1))[:, 0]
        pred_body_next = pred_body_state[..., : model.body_dim]

        pred_body.append(pred_body_next[0])
        pred_pusher.append(pred_pusher_next[0])
        body_state = pred_body_state
        pusher_state = pred_pusher_state

    pred_body_tensor = torch.stack(pred_body, dim=0)
    pred_pusher_tensor = torch.stack(pred_pusher, dim=0)
    pred_full_tensor = torch.cat((pred_body_tensor, pred_pusher_tensor), dim=-1)
    return pred_full_tensor, pred_body_tensor, pred_pusher_tensor


def compute_metrics(true_latents: torch.Tensor, pred_latents: torch.Tensor, *, frameskip: int) -> dict[str, object]:
    pred_indices = np.arange(1, pred_latents.shape[0], dtype=np.int64) * frameskip
    pred_indices = pred_indices[pred_indices < true_latents.shape[0]]
    if pred_indices.size == 0:
        raise ValueError("Rollout is not longer than the warm-start history.")
    true = true_latents[pred_indices].float().cpu()
    pred = pred_latents[1 : 1 + pred_indices.size].float().cpu()
    err = pred - true
    rmse_per_step = err.pow(2).mean(dim=-1).sqrt()
    rmse_per_dim = err.pow(2).mean(dim=0).sqrt()
    return {
        "num_context_steps": 1,
        "num_rollout_steps": int(true.shape[0]),
        "embed_dim": int(true.shape[-1]),
        "mean_rmse": float(rmse_per_step.mean()),
        "final_rmse": float(rmse_per_step[-1]),
        "max_rmse": float(rmse_per_step.max()),
        "rmse_per_step": rmse_per_step.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
    }


def plot_latents(
    true_latents: torch.Tensor,
    pred_latents: torch.Tensor,
    *,
    out_dir: Path,
    stem: str,
    title: str,
    num_context_steps: int,
) -> list[Path]:
    length = min(true_latents.shape[0], pred_latents.shape[0])
    true = true_latents[:length].float().cpu().numpy()
    pred = pred_latents[:length].float().cpu().numpy()
    embed_dim = true.shape[-1]
    midpoint = (embed_dim + 1) // 2
    splits = [(0, midpoint), (midpoint, embed_dim)]
    paths: list[Path] = []

    for plot_idx, (start_dim, end_dim) in enumerate(splits, start=1):
        num_dims = max(end_dim - start_dim, 1)
        fig, axes = plt.subplots(num_dims, 1, figsize=(12, max(3, 1.8 * num_dims)), sharex=True)
        if num_dims == 1:
            axes = [axes]
        steps = np.arange(length)
        for axis, dim in zip(axes, range(start_dim, end_dim)):
            axis.plot(steps, true[:, dim], label="true", linewidth=1.5)
            axis.plot(steps, pred[:, dim], label="rollout", linewidth=1.2, linestyle="--")
            if num_context_steps > 0:
                axis.axvline(num_context_steps - 0.5, color="black", alpha=0.25, linewidth=1)
            axis.set_ylabel(f"z{dim}")
            axis.grid(True, alpha=0.25)
        if start_dim == end_dim:
            axes[0].text(0.5, 0.5, "no latent dims in this half", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_axis_off()
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("trajectory frame")
        fig.suptitle(f"{title}, dims {start_dim}-{end_dim - 1}")
        fig.tight_layout()
        path = out_dir / f"{stem}_latents_part_{plot_idx}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


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
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = require_device(args.device)
    valid_episodes = valid_episode_indices(dataset_path, args=args)
    with h5py.File(dataset_path, "r") as h5:
        num_episodes = int(h5["ep_len"].shape[0])
    if valid_episodes.size == 0:
        raise ValueError("No episodes are long enough for the requested markov_deriv/num_preds/frameskip settings.")
    episode_idx = args.episode_idx
    if episode_idx is None:
        episode_idx = int(np.random.default_rng().choice(valid_episodes))
    if not 0 <= episode_idx < num_episodes:
        raise IndexError(f"episode_idx {episode_idx} is out of range [0, {num_episodes}).")
    if episode_idx not in set(valid_episodes.tolist()):
        raise ValueError(
            f"episode_idx {episode_idx} is too short for markov_deriv={args.markov_deriv}, "
            f"num_preds={args.num_preds}, frameskip={args.frameskip}."
        )

    model = load_model(checkpoint_path, device)
    pixels, actions = load_episode(dataset_path, episode_idx, args=args)
    true_full_latents, true_body_latents, true_pusher_latents = encode_frames(
        model,
        pixels,
        device=device,
        img_size=args.img_size,
        frame_batch_size=args.frame_batch_size,
    )
    pred_full_latents, pred_body_latents, pred_pusher_latents = rollout_latents(
        model,
        true_body_latents,
        true_pusher_latents,
        actions.to(device),
        markov_deriv=args.markov_deriv,
        frameskip=args.frameskip,
        max_rollout_steps=args.max_rollout_steps,
    )

    full_metrics = compute_metrics(true_full_latents, pred_full_latents, frameskip=args.frameskip)
    body_metrics = compute_metrics(true_body_latents, pred_body_latents, frameskip=args.frameskip)
    pusher_metrics = compute_metrics(true_pusher_latents, pred_pusher_latents, frameskip=args.frameskip)

    metrics = {
        "episode_idx": episode_idx,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(model_dir / "config.json"),
        "dataset_path": str(dataset_path),
        "markov_deriv": args.markov_deriv,
        "action_history_size": 1,
        "state_space": "split_latent_plus_finite_differences",
        "embed_dim": int(true_full_latents.shape[-1]),
        "body_dim": int(true_body_latents.shape[-1]),
        "pusher_dim": int(true_pusher_latents.shape[-1]),
        "markov_state_dim": int(true_full_latents.shape[-1] * (args.markov_deriv + 1)),
        "body_markov_state_dim": int(true_body_latents.shape[-1] * (args.markov_deriv + 1)),
        "pusher_markov_state_dim": int(true_pusher_latents.shape[-1] * (args.markov_deriv + 1)),
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "full_metrics": full_metrics,
        "body_metrics": body_metrics,
        "pusher_metrics": pusher_metrics,
    }
    metrics_path = out_dir / f"episode_{episode_idx:05d}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    plot_paths = []
    plot_paths.extend(
        plot_latents(
            true_full_latents,
            pred_full_latents,
            out_dir=out_dir,
            stem=f"episode_{episode_idx:05d}_full",
            title=f"PushT splitdyn full latent rollout episode {episode_idx}",
            num_context_steps=1,
        )
    )
    plot_paths.extend(
        plot_latents(
            true_body_latents,
            pred_body_latents,
            out_dir=out_dir,
            stem=f"episode_{episode_idx:05d}_body",
            title=f"PushT splitdyn body latent rollout episode {episode_idx}",
            num_context_steps=1,
        )
    )
    plot_paths.extend(
        plot_latents(
            true_pusher_latents,
            pred_pusher_latents,
            out_dir=out_dir,
            stem=f"episode_{episode_idx:05d}_pusher",
            title=f"PushT splitdyn pusher latent rollout episode {episode_idx}",
            num_context_steps=1,
        )
    )

    print(
        json.dumps(
            {
                "episode_idx": episode_idx,
                "full_mean_rmse": full_metrics["mean_rmse"],
                "body_mean_rmse": body_metrics["mean_rmse"],
                "pusher_mean_rmse": pusher_metrics["mean_rmse"],
                "checkpoint": str(checkpoint_path),
                "action_history_size": 1,
                "state_space": "split_latent_plus_finite_differences",
                "metrics_path": str(metrics_path),
                "plot_paths": [str(path) for path in plot_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
