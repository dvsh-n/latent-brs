#!/usr/bin/env python3
"""Generate one-step latent error-predictor data from a Markov LE-WM model."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from reacher.train.lewm_train_mlp_markov import LeWMReacherDataset

DEFAULT_DATASET_PATH = "reacher/data/expert_data/reacher_expert.h5"
DEFAULT_RUN_DIR = "reacher/models/lewm_reacher"
DEFAULT_OUT_DIR = "reacher/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Output name stem. Defaults to config output_model_name, then the model directory name.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


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


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def make_loader(dataset: LeWMReacherDataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


@torch.no_grad()
def build_error_data(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    embed_dim: int,
) -> dict[str, torch.Tensor]:
    x_t_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    pred_next_chunks: list[torch.Tensor] = []
    true_next_chunks: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Generating one-step latent data"):
        batch = {
            "pixels": batch["pixels"].to(device, non_blocking=True),
            "action": torch.nan_to_num(batch["action"].to(device, non_blocking=True), 0.0),
        }
        output = model.encode(batch)
        emb = output["emb"]
        act = output["act_emb"][:, :1]

        past = emb[:, -3]
        current = emb[:, -2]
        true_next_latent = emb[:, -1]

        x_t = torch.cat((current, current - past), dim=-1)
        true_next = torch.cat((true_next_latent, true_next_latent - current), dim=-1)
        pred_next = model.predict(x_t.unsqueeze(1), act)[:, 0]

        if pred_next.shape[-1] != 2 * embed_dim:
            raise RuntimeError(
                f"Expected predicted Markov state dim {2 * embed_dim}, got {pred_next.shape[-1]}."
            )

        x_t_chunks.append(x_t.cpu())
        action_chunks.append(act[:, 0].cpu())
        pred_next_chunks.append(pred_next.cpu())
        true_next_chunks.append(true_next.cpu())

    x_t = torch.cat(x_t_chunks, dim=0).contiguous()
    action = torch.cat(action_chunks, dim=0).contiguous()
    pred_next = torch.cat(pred_next_chunks, dim=0).contiguous()
    true_next = torch.cat(true_next_chunks, dim=0).contiguous()
    return {
        "x_t": x_t,
        "a_t": action,
        "x_hat_t_plus_1": pred_next,
        "x_t_plus_1": true_next,
        "error": (true_next - pred_next).contiguous(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    model_dir = args.model_dir.expanduser().resolve()
    config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    dataset_path = args.dataset_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    model_name = args.model_name or str(config.get("output_model_name") or model_dir.name)
    output_path = out_dir / f"{model_name}_one_step_error_data.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    history_size = int(config.get("history_size", 2))
    if history_size < 2:
        raise ValueError("Markov one-step data requires history_size >= 2.")
    frameskip = int(config.get("frameskip", 1))
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 24))

    dataset = LeWMReacherDataset(
        dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=frameskip,
        img_size=img_size,
        action_dim=action_dim,
    )
    device = require_device(args.device)
    model = load_model(checkpoint_path, device)
    loader = make_loader(dataset, args, device)
    tensors = build_error_data(model, loader, device=device, embed_dim=embed_dim)

    payload = {
        **tensors,
        "metadata": {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint_path),
            "config_path": str(model_dir / "config.json"),
            "dataset_path": str(dataset_path),
            "num_samples": int(tensors["x_t"].shape[0]),
            "embed_dim": embed_dim,
            "state_dim": int(tensors["x_t"].shape[-1]),
            "action_dim": int(tensors["a_t"].shape[-1]),
            "history_size": history_size,
            "num_preds": 1,
            "frameskip": frameskip,
            "state_space": "latent_plus_latent_delta",
            "keys": ["x_t", "a_t", "x_hat_t_plus_1", "x_t_plus_1", "error"],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_samples": payload["metadata"]["num_samples"],
                "state_dim": payload["metadata"]["state_dim"],
                "action_dim": payload["metadata"]["action_dim"],
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
