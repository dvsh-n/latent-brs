#!/usr/bin/env python3
"""Precompute expert Markov-state trajectories for offline Koopman training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DEFAULT_DATASET_PATH = "reacher/data/expert_data/reacher_expert.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft"
DEFAULT_OUTPUT_PATH = "reacher/data/expert_data/reacher_koopman_markov.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def preprocess_pixels(
    pixels: torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    pixels = pixels.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(pixels.shape[-2:]) != (img_size, img_size):
        pixels = F.interpolate(pixels, size=(img_size, img_size), mode="bilinear", align_corners=False)
    pixels = (pixels - pixel_mean) / pixel_std
    return pixels.to(device, non_blocking=True)


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    h5: h5py.File,
    *,
    total_frames: int,
    frame_batch_size: int,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    for start in tqdm(range(0, total_frames, frame_batch_size), desc="Encoding frames", unit="batch"):
        end = min(start + frame_batch_size, total_frames)
        batch = torch.from_numpy(np.ascontiguousarray(h5["pixels"][start:end]))
        batch = preprocess_pixels(
            batch,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            device=device,
        )
        output = model.encoder(batch, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]).cpu())
    return torch.cat(latents, dim=0)


def make_markov_states(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [time, dim], got {embeddings.shape}.")
    deltas = torch.zeros_like(embeddings)
    deltas[1:] = embeddings[1:] - embeddings[:-1]
    return torch.cat((embeddings, deltas), dim=-1)


def compute_action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)
    return action_mean, action_std


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive.")

    config = load_config(model_dir)
    img_size = int(config["img_size"])
    embed_dim = int(config["embed_dim"])
    action_dim = int(config["action_dim"])
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    if markov_state_dim != 2 * embed_dim:
        raise ValueError(
            f"Expected markov_state_dim to equal 2 * embed_dim, got {markov_state_dim} and {embed_dim}."
        )

    device = require_device(args.device)
    if checkpoint_path is None:
        checkpoint_path = latest_object_checkpoint(model_dir)
    model = load_model(checkpoint_path, device)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = torch.from_numpy(np.asarray(h5["ep_len"][:], dtype=np.int64))
        ep_offset = torch.from_numpy(np.asarray(h5["ep_offset"][:], dtype=np.int64))
        total_frames = int(ep_len.sum().item())
        latents_flat = encode_pixels(
            model,
            h5,
            total_frames=total_frames,
            frame_batch_size=args.frame_batch_size,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            device=device,
        )
        action_mean, action_std = compute_action_stats(h5)
        actions_flat = np.asarray(h5["action"][:total_frames], dtype=np.float32)

        demos: list[dict[str, torch.Tensor | int]] = []
        for episode_idx, (length, offset) in enumerate(zip(ep_len.tolist(), ep_offset.tolist(), strict=True)):
            end = offset + length
            episode_emb = latents_flat[offset:end]
            episode_states = make_markov_states(episode_emb)
            episode_actions = actions_flat[offset:end]
            episode_actions = (np.nan_to_num(episode_actions, nan=0.0) - action_mean) / action_std
            episode_controls = torch.from_numpy(episode_actions[:-1]).float()
            demos.append(
                {
                    "states": episode_states.transpose(0, 1).contiguous(),
                    "controls": episode_controls.transpose(0, 1).contiguous(),
                    "episode_idx": int(episode_idx),
                }
            )

        attrs = {key: (value.item() if hasattr(value, "item") else value) for key, value in h5.attrs.items()}

    payload = {
        "demos": demos,
        "metadata": {
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint_path),
            "img_size": img_size,
            "embed_dim": embed_dim,
            "state_dim": markov_state_dim,
            "action_dim": action_dim,
            "num_episodes": len(demos),
            "total_frames": total_frames,
            "frame_batch_size": int(args.frame_batch_size),
            "state_type": "markov_state_[e_t,delta_e_t]",
            "control_type": "globally_normalized_actions",
            "action_mean": torch.from_numpy(action_mean.squeeze(0)),
            "action_std": torch.from_numpy(action_std.squeeze(0)),
            "h5_attrs": attrs,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_episodes": len(demos),
                "state_dim": markov_state_dim,
                "action_dim": action_dim,
                "first_states_shape": list(demos[0]["states"].shape) if demos else None,
                "first_controls_shape": list(demos[0]["controls"].shape) if demos else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
