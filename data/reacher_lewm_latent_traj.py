#!/usr/bin/env python3
"""Encode Reacher expert videos into LE-WM latent trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "expert_data" / "reacher_expert.h5"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "lewm_reacher_24D" / "lewm_epoch_50_object.ckpt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "expert_data" / "latent_traj_lewm_reacher_24D.pt"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def preprocess_pixels(pixels: torch.Tensor, *, img_size: int, device: torch.device) -> torch.Tensor:
    pixels = pixels.permute(0, 3, 1, 2).float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        pixels = F.interpolate(pixels, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return ((pixels - IMAGENET_MEAN) / IMAGENET_STD).to(device, non_blocking=True)


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    h5: h5py.File,
    *,
    total_frames: int,
    batch_size: int,
    img_size: int,
    device: torch.device,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    for start in tqdm(range(0, total_frames, batch_size), desc="Encoding frames", unit="batch"):
        end = min(start + batch_size, total_frames)
        batch = torch.from_numpy(h5["pixels"][start:end])
        batch = preprocess_pixels(batch, img_size=img_size, device=device)
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb.cpu())
    return torch.cat(latents, dim=0)


def reshape_trajectories(flat: torch.Tensor, ep_len: torch.Tensor, ep_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_episodes = int(ep_len.numel())
    max_len = int(ep_len.max().item())
    latent_dim = int(flat.shape[-1])
    trajectories = flat.new_full((num_episodes, max_len, latent_dim), float("nan"))
    mask = torch.zeros((num_episodes, max_len), dtype=torch.bool)
    for episode_idx in range(num_episodes):
        start = int(ep_offset[episode_idx].item())
        length = int(ep_len[episode_idx].item())
        trajectories[episode_idx, :length] = flat[start : start + length]
        mask[episode_idx, :length] = True
    return trajectories, mask


def reshape_dataset_array(values: torch.Tensor, ep_len: torch.Tensor, ep_offset: torch.Tensor) -> torch.Tensor:
    num_episodes = int(ep_len.numel())
    max_len = int(ep_len.max().item())
    out = values.new_full((num_episodes, max_len, *values.shape[1:]), float("nan"))
    for episode_idx in range(num_episodes):
        start = int(ep_offset[episode_idx].item())
        length = int(ep_len[episode_idx].item())
        out[episode_idx, :length] = values[start : start + length]
    return out


def reshape_action_sequences(
    values: torch.Tensor,
    ep_len: torch.Tensor,
    ep_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_episodes = int(ep_len.numel())
    max_actions = max(int(ep_len.max().item()) - 1, 0)
    out = values.new_full((num_episodes, max_actions, values.shape[-1]), float("nan"))
    mask = torch.zeros((num_episodes, max_actions), dtype=torch.bool)
    for episode_idx in range(num_episodes):
        start = int(ep_offset[episode_idx].item())
        length = max(int(ep_len[episode_idx].item()) - 1, 0)
        out[episode_idx, :length] = values[start : start + length]
        mask[episode_idx, :length] = True
    return out, mask


def make_padded_history_states(latents: torch.Tensor, history: int) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(f"Expected latents with shape [episodes, frames, dim], got {latents.shape}.")
    if history < 1:
        raise ValueError("--history must be positive.")

    _, num_frames, _ = latents.shape
    frame_idx = torch.arange(num_frames)
    history_slices = []
    for offset in range(history):
        source_idx = (frame_idx - history + 1 + offset).clamp_min(0)
        history_slices.append(latents[:, source_idx])
    return torch.stack(history_slices, dim=2).flatten(start_dim=2)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.history < 1:
        raise ValueError("--history must be positive.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive.")

    device = require_device(args.device)
    model = load_model(checkpoint_path, device)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = torch.from_numpy(h5["ep_len"][:]).long()
        ep_offset = torch.from_numpy(h5["ep_offset"][:]).long()
        if args.max_episodes is not None:
            keep = min(args.max_episodes, int(ep_len.numel()))
            ep_len = ep_len[:keep]
            ep_offset = ep_offset[:keep]

        total_frames = int(ep_len.sum().item())
        latents_flat = encode_pixels(
            model,
            h5,
            total_frames=total_frames,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=device,
        )
        latents, mask = reshape_trajectories(latents_flat, ep_len, ep_offset)
        history_states = make_padded_history_states(latents, args.history)

        actions_flat = torch.from_numpy(h5["action"][:total_frames]).float()
        actions = reshape_dataset_array(actions_flat, ep_len, ep_offset)
        action_sequences, action_mask = reshape_action_sequences(actions_flat, ep_len, ep_offset)
        observations = reshape_dataset_array(torch.from_numpy(h5["observation"][:total_frames]).float(), ep_len, ep_offset)

        attrs = {}
        for key, value in h5.attrs.items():
            attrs[key] = value.item() if hasattr(value, "item") else value

    payload = {
        "latents": latents,
        "latents_flat": latents_flat,
        "history_states": history_states,
        "mask": mask,
        "ep_len": ep_len,
        "ep_offset": ep_offset,
        "actions_flat": actions_flat,
        "actions": actions,
        "action_sequences": action_sequences,
        "action_mask": action_mask,
        "observations": observations,
        "metadata": {
            "latent_type": "lewm_projected_cls",
            "action_type": "ground_truth_raw_actions",
            "actions_note": (
                "actions is frame-aligned with a padded final NaN action per episode; "
                "action_sequences excludes that padded final action."
            ),
            "latent_dim": int(latents.shape[-1]),
            "action_dim": int(action_sequences.shape[-1]),
            "history": int(args.history),
            "history_state_dim": int(history_states.shape[-1]),
            "num_episodes": int(ep_len.numel()),
            "max_episode_len": int(ep_len.max().item()),
            "total_frames": int(total_frames),
            "dataset_path": str(dataset_path),
            "checkpoint": str(checkpoint_path),
            "img_size": int(args.img_size),
            "batch_size": int(args.batch_size),
            "h5_attrs": attrs,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "latents_shape": list(latents.shape),
                "latents_flat_shape": list(latents_flat.shape),
                "history_states_shape": list(history_states.shape),
                "actions_shape": list(actions.shape),
                "action_sequences_shape": list(action_sequences.shape),
                "observations_shape": list(observations.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
