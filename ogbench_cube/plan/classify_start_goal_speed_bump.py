#!/usr/bin/env python3
"""Classify start/goal endpoint images with the learned OGBench obstacle net."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


DEFAULT_DATASET_PATH = Path("ogbench_cube/plan/start_goal_speed_bump_new.pt")
DEFAULT_MODEL_DIR = Path("ogbench_cube/models/mlpdyn_embd_8")
DEFAULT_OBSTACLE_MODEL_PATH = Path("ogbench_cube/models/obs_net_small/7b9441ea22420e3b/model.pt")
DEFAULT_OUT_DIR = Path("ogbench_cube/plan/start_goal_speed_bump_obs_net_small_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--obstacle-model-path", type=Path, default=DEFAULT_OBSTACLE_MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--activation", default="auto", choices=("auto", "gelu", "tanh"))
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch[_=](\d+).*\.ckpt$")
    candidates = [(int(match.group(1)), path) for path in model_dir.glob("*.ckpt") for match in [pattern.match(path.name)] if match]
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def as_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def load_endpoint_pixels(path: Path, *, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "start" not in payload or "goal" not in payload:
        raise KeyError("Expected a .pt payload with top-level 'start' and 'goal' dictionaries.")
    start = payload["start"]
    goal = payload["goal"]
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("'start' and 'goal' entries must both be dictionaries.")
    start_pixels = as_numpy(start["pixels"], dtype=np.uint8)
    goal_pixels = as_numpy(goal["pixels"], dtype=np.uint8)
    if start_pixels.shape != goal_pixels.shape:
        raise ValueError(f"Start and goal pixel arrays must match, got {start_pixels.shape} and {goal_pixels.shape}.")
    if start_pixels.ndim != 4 or start_pixels.shape[-1] != 3:
        raise ValueError(f"Expected pixels shaped [N, H, W, 3], got {start_pixels.shape}.")
    if limit is not None:
        start_pixels = start_pixels[: int(limit)]
        goal_pixels = goal_pixels[: int(limit)]
    metadata = payload.get("metadata", {})
    return start_pixels, goal_pixels, metadata if isinstance(metadata, dict) else {}


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels_np: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    batch_size: int,
) -> torch.Tensor:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    latents: list[torch.Tensor] = []
    for start in tqdm(range(0, pixels_np.shape[0], batch_size), desc="Encoding", unit="batch"):
        chunk_np = pixels_np[start : start + batch_size]
        chunk = torch.from_numpy(chunk_np.copy()).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        if tuple(chunk.shape[-2:]) != (img_size, img_size):
            chunk = torch.nn.functional.interpolate(chunk, size=(img_size, img_size), mode="bilinear", align_corners=False)
        chunk = (chunk - pixel_mean) / pixel_std
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]).detach().cpu())
    return torch.cat(latents, dim=0)


def resolve_activation(artifact: dict[str, Any], path: Path, activation_arg: str) -> str:
    if activation_arg != "auto":
        return activation_arg
    activation = artifact.get("activation") or artifact.get("cache_config", {}).get("activation")
    if activation is not None:
        return str(activation).lower()
    # The small obstacle classifier was trained with tanh; older/default artifacts use GELU.
    return "tanh" if int(artifact["hidden_dim"]) == 6 or "obs_net_small" in str(path) else "gelu"


class ObstacleMLP(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int, dropout: float, activation: str):
        super().__init__()
        layers: list[torch.nn.Module] = []
        current_dim = int(input_dim)
        if activation == "gelu":
            activation_factory: type[torch.nn.Module] = torch.nn.GELU
        elif activation == "tanh":
            activation_factory = torch.nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        for _ in range(int(depth) - 1):
            layers.append(torch.nn.Linear(current_dim, int(hidden_dim)))
            layers.append(torch.nn.LayerNorm(int(hidden_dim)))
            layers.append(activation_factory())
            if float(dropout) > 0.0:
                layers.append(torch.nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(torch.nn.Linear(current_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_obstacle_model(
    path: Path,
    device: torch.device,
    *,
    activation_arg: str,
) -> tuple[ObstacleMLP, torch.Tensor, torch.Tensor, float, str, dict[str, Any]]:
    artifact = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    activation = resolve_activation(artifact, path, activation_arg)
    model = ObstacleMLP(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        depth=int(artifact["depth"]),
        dropout=float(artifact["dropout"]),
        activation=activation,
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device).eval()
    feature_mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.clamp(torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device), min=1e-6)
    threshold = float(artifact["conformal_safe_score_threshold"])
    return model, feature_mean, feature_std, threshold, activation, artifact


@torch.no_grad()
def score_latents(
    obstacle_model: ObstacleMLP,
    latents: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for start in tqdm(range(0, latents.shape[0], batch_size), desc="Classifying", unit="batch"):
        chunk = latents[start : start + batch_size].to(device)
        chunk = (chunk - feature_mean) / feature_std
        scores.append(obstacle_model(chunk).detach().cpu())
    return torch.cat(scores, dim=0)


def summarize(scores: torch.Tensor, required_score: float) -> dict[str, Any]:
    non_obstacle = scores.numpy() > float(required_score)
    return {
        "count": int(scores.numel()),
        "non_obstacle_count": int(non_obstacle.sum()),
        "obstacle_count": int((~non_obstacle).sum()),
        "non_obstacle_rate": float(non_obstacle.mean()) if non_obstacle.size else 0.0,
        "min_score": float(scores.min().item()) if scores.numel() else None,
        "mean_score": float(scores.mean().item()) if scores.numel() else None,
        "max_score": float(scores.max().item()) if scores.numel() else None,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = resolve_device(args.device)

    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve() if args.checkpoint_path else latest_object_checkpoint(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        model_config = json.load(f)
    img_size = int(model_config.get("img_size", 224))

    print(f"Loading world model checkpoint: {checkpoint_path}")
    world_model = torch.load(checkpoint_path, map_location=device, weights_only=False).eval()
    world_model.to(device)

    print(f"Loading endpoint pairs: {args.dataset_path}")
    start_pixels, goal_pixels, dataset_metadata = load_endpoint_pixels(args.dataset_path, limit=args.limit)
    print(f"Loaded {start_pixels.shape[0]} aligned start/goal pairs.")

    obstacle_model, feature_mean, feature_std, threshold, activation, obstacle_artifact = load_obstacle_model(
        args.obstacle_model_path,
        device,
        activation_arg=args.activation,
    )
    required_score = threshold + float(args.margin)
    print(
        f"Loaded obstacle classifier: {args.obstacle_model_path} "
        f"(activation={activation}, non-obstacle iff score > {required_score:.6g})"
    )

    start_latents = encode_frames(world_model, start_pixels, device=device, img_size=img_size, batch_size=args.batch_size)
    goal_latents = encode_frames(world_model, goal_pixels, device=device, img_size=img_size, batch_size=args.batch_size)
    if start_latents.shape[-1] != int(obstacle_artifact["input_dim"]):
        raise ValueError(
            f"Encoded latent dim {start_latents.shape[-1]} does not match classifier input_dim "
            f"{int(obstacle_artifact['input_dim'])}."
        )

    start_scores = score_latents(
        obstacle_model,
        start_latents,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
        batch_size=args.batch_size,
    )
    goal_scores = score_latents(
        obstacle_model,
        goal_latents,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
        batch_size=args.batch_size,
    )

    start_non_obstacle = start_scores.numpy() > required_score
    goal_non_obstacle = goal_scores.numpy() > required_score
    pair_clear = start_non_obstacle & goal_non_obstacle

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "start_goal_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_index",
                "start_score",
                "goal_score",
                "start_label",
                "goal_label",
                "pair_label",
            ],
        )
        writer.writeheader()
        for index, (start_score, goal_score, start_ok, goal_ok, clear) in enumerate(
            zip(start_scores.tolist(), goal_scores.tolist(), start_non_obstacle.tolist(), goal_non_obstacle.tolist(), pair_clear.tolist())
        ):
            writer.writerow(
                {
                    "pair_index": index,
                    "start_score": f"{start_score:.9g}",
                    "goal_score": f"{goal_score:.9g}",
                    "start_label": "non_obstacle" if start_ok else "obstacle",
                    "goal_label": "non_obstacle" if goal_ok else "obstacle",
                    "pair_label": "clear" if clear else "blocked",
                }
            )

    summary = {
        "dataset_path": str(args.dataset_path),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "obstacle_model_path": str(args.obstacle_model_path),
        "score_sign_convention": obstacle_artifact.get("score_sign_convention"),
        "threshold": threshold,
        "margin": float(args.margin),
        "activation": activation,
        "required_score": required_score,
        "num_pairs": int(start_scores.numel()),
        "pair_clear_count": int(pair_clear.sum()),
        "pair_blocked_count": int((~pair_clear).sum()),
        "pair_clear_rate": float(pair_clear.mean()) if pair_clear.size else 0.0,
        "start": summarize(start_scores, required_score),
        "goal": summarize(goal_scores, required_score),
        "dataset_metadata": jsonable(dataset_metadata),
        "csv_path": str(csv_path),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote per-pair scores to {csv_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        f"Clear pairs: {summary['pair_clear_count']}/{summary['num_pairs']} "
        f"({summary['pair_clear_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
