#!/usr/bin/env python3
"""Count rope HDF5 frames/trajectories classified as entering the obstacle zone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.safety.compat import register_legacy_checkpoint_aliases
from rope.safety.obstacle_classifier import (
    ObstacleMLP,
    imagenet_pixel_stats,
    load_config,
    load_world_model,
    preprocess_pixels,
)

DEFAULT_DATASET = REPO_ROOT / "rope/data/train_data_noshadow.h5"
DEFAULT_OBSTACLE_MODEL = REPO_ROOT / "rope/safety/obs_net/da270d7d1050f110/model.pt"


def parse_max_frames(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"all", "full", "none", "-1", "0"}:
        return None
    try:
        frames = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer frame count or 'all'.") from exc
    if frames < 1:
        raise argparse.ArgumentTypeError("--max-frames must be positive, or use 'all'.")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--obstacle-model", type=Path, default=DEFAULT_OBSTACLE_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-frames", type=parse_max_frames, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_artifact_path(path_string: str, fallback_model_dir: Path) -> Path:
    path = Path(path_string).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = fallback_model_dir / path.name
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint from classifier artifact was not found: {path_string}")


def episode_indices_from_lengths(ep_len: np.ndarray, total_frames: int) -> np.ndarray:
    episode_idx = np.empty((total_frames,), dtype=np.int64)
    cursor = 0
    for ep, length in enumerate(ep_len.tolist()):
        stop = min(cursor + int(length), total_frames)
        if stop <= cursor:
            break
        episode_idx[cursor:stop] = ep
        cursor = stop
        if cursor >= total_frames:
            break
    if cursor < total_frames:
        raise ValueError("Could not map all selected rows to episodes.")
    return episode_idx


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    obstacle_model_path = args.obstacle_model.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not obstacle_model_path.is_file():
        raise FileNotFoundError(f"Obstacle model not found: {obstacle_model_path}")

    register_legacy_checkpoint_aliases()
    device = require_device(args.device)
    artifact = torch.load(obstacle_model_path, map_location="cpu", weights_only=False)
    cache_config = artifact.get("cache_config", {})
    model_dir = Path(cache_config.get("model_dir", REPO_ROOT / "rope/models/mlpdyn_noshadow_ft")).expanduser()
    if not model_dir.is_dir():
        model_dir = REPO_ROOT / "rope/models/mlpdyn_noshadow_ft"
    model_dir = model_dir.resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else resolve_artifact_path(str(cache_config["checkpoint_path"]), model_dir)
    )
    model_config = load_config(model_dir)
    embed_dim = int(artifact["input_dim"])
    img_size = int(cache_config.get("img_size", model_config.get("img_size", 224)))
    threshold = float(artifact.get("conformal_safe_score_threshold", artifact.get("base_decision_threshold", 0.0)))

    print(f"device={device}", flush=True)
    print(f"dataset={dataset_path}", flush=True)
    print(f"obstacle_model={obstacle_model_path}", flush=True)
    print(f"world_checkpoint={checkpoint_path}", flush=True)
    print(f"threshold={threshold}", flush=True)

    world_model = load_world_model(checkpoint_path, device)
    obstacle_model = ObstacleMLP(
        embed_dim,
        int(artifact["hidden_dim"]),
        int(artifact["depth"]),
        float(artifact["dropout"]),
        head_style=str(artifact.get("head_style", "postnorm-gelu")),
    ).to(device)
    obstacle_model.load_state_dict(artifact["state_dict"])
    obstacle_model.eval()
    obstacle_model.requires_grad_(False)

    feature_mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(artifact["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    pixel_mean, pixel_std = imagenet_pixel_stats(device)

    start_time = time.perf_counter()
    frame_hits = 0
    frame_count = 0
    score_min = float("inf")
    score_max = float("-inf")

    with h5py.File(dataset_path, "r") as h5:
        available_frames = int(h5["pixels"].shape[0])
        num_frames = min(available_frames, int(args.max_frames)) if args.max_frames is not None else available_frames
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        if "episode_idx" in h5:
            episode_idx = np.asarray(h5["episode_idx"][:num_frames], dtype=np.int64)
        else:
            episode_idx = episode_indices_from_lengths(ep_len, num_frames)
        num_episodes = int(np.max(episode_idx)) + 1 if episode_idx.size else int(ep_len.shape[0])
        trajectory_hits = np.zeros((num_episodes,), dtype=bool)
        trajectory_min_score = np.full((num_episodes,), np.inf, dtype=np.float32)

        iterator = tqdm(range(0, num_frames, args.batch_size), desc="Scoring frames", unit="batch")
        for start in iterator:
            stop = min(start + args.batch_size, num_frames)
            pixels = np.asarray(h5["pixels"][start:stop], dtype=np.uint8)
            batch = preprocess_pixels(
                pixels,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            with torch.no_grad():
                encoded = world_model.encoder(batch, interpolate_pos_encoding=True)
                latents = world_model.projector(encoded.last_hidden_state[:, 0])[:, :embed_dim]
                scores = obstacle_model((latents - feature_mean) / feature_std).squeeze(-1)

            scores_np = scores.detach().cpu().numpy().astype(np.float32)
            hit_mask = scores_np <= threshold
            batch_episode_idx = episode_idx[start:stop]
            if np.any(hit_mask):
                trajectory_hits[np.unique(batch_episode_idx[hit_mask])] = True
            np.minimum.at(trajectory_min_score, batch_episode_idx, scores_np)

            frame_hits += int(np.sum(hit_mask))
            frame_count += int(stop - start)
            score_min = min(score_min, float(np.min(scores_np)))
            score_max = max(score_max, float(np.max(scores_np)))

    valid_episodes = np.zeros_like(trajectory_hits, dtype=bool)
    valid_episodes[np.unique(episode_idx)] = True
    unsafe_trajectories = int(np.sum(trajectory_hits[valid_episodes]))
    total_trajectories = int(np.sum(valid_episodes))
    safe_trajectories = total_trajectories - unsafe_trajectories
    safe_frames = int(frame_count - frame_hits)
    result = {
        "unsafe_trajectories": unsafe_trajectories,
        "safe_trajectories": safe_trajectories,
        "total_trajectories": total_trajectories,
        "unsafe_trajectory_fraction": float(unsafe_trajectories / total_trajectories) if total_trajectories else 0.0,
        "safe_trajectory_fraction": float(safe_trajectories / total_trajectories) if total_trajectories else 0.0,
        "unsafe_frames": int(frame_hits),
        "safe_frames": safe_frames,
        "total_frames": int(frame_count),
        "unsafe_frame_fraction": float(frame_hits / frame_count) if frame_count else 0.0,
        "safe_frame_fraction": float(safe_frames / frame_count) if frame_count else 0.0,
        "trajectories_through_obstacle": unsafe_trajectories,
        "trajectory_fraction": float(unsafe_trajectories / total_trajectories) if total_trajectories else 0.0,
        "obstacle_frames": int(frame_hits),
        "frame_fraction": float(frame_hits / frame_count) if frame_count else 0.0,
        "score_min": float(score_min),
        "score_max": float(score_max),
        "threshold": threshold,
        "elapsed_sec": float(time.perf_counter() - start_time),
    }

    print("RESULT")
    for key, value in result.items():
        print(f"{key}={value}")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(jsonable(result), handle, indent=2)
        print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
