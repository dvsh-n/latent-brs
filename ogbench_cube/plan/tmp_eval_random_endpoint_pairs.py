#!/usr/bin/env python3
"""Run sampled start/goal endpoint images through the OGBench cube obstacle encoder and classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from ogbench_cube.plan.obstacle_net import (
    ObstacleMLP,
    encode_pixels,
    load_config,
    load_world_model,
    require_device,
)

DEFAULT_PAIR_PATH = Path("ogbench_cube/plan/random_endpoint_pairs/start_goal_speed_bump.pt")
DEFAULT_OBS_MODEL_PATH = Path("ogbench_cube/plan/obs_net/7b9441ea22420e3b/model.pt")
DEFAULT_NUM_SAMPLES = 256
DEFAULT_FRAME_BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-path", type=Path, default=DEFAULT_PAIR_PATH)
    parser.add_argument("--obs-model-path", type=Path, default=DEFAULT_OBS_MODEL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=DEFAULT_FRAME_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def load_pair_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected pair payload type: {type(payload)}")
    for key in ("start", "goal"):
        if key not in payload:
            raise KeyError(f"Pair payload missing '{key}' section.")
        section = payload[key]
        if not isinstance(section, dict) or "pixels" not in section or "task_target" not in section:
            raise KeyError(f"Pair payload section '{key}' is missing required fields.")
    return payload


def load_classifier(path: Path, device: torch.device) -> tuple[ObstacleMLP, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ObstacleMLP(
        int(payload["input_dim"]),
        int(payload["hidden_dim"]),
        int(payload["depth"]),
        float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    model.eval()
    feature_mean = torch.from_numpy(np.asarray(payload["feature_mean"], dtype=np.float32)).to(device)
    feature_std = torch.from_numpy(np.asarray(payload["feature_std"], dtype=np.float32)).to(device)
    return model, feature_mean, feature_std, payload


def select_indices(total: int, requested: int, seed: int) -> np.ndarray:
    if requested <= 0:
        raise ValueError("--num-samples must be positive.")
    count = min(int(requested), int(total))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


@torch.no_grad()
def score_latents(
    classifier: ObstacleMLP,
    latents: np.ndarray,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    normalized = (torch.from_numpy(latents.astype(np.float32)).to(device) - feature_mean) / feature_std
    chunks: list[torch.Tensor] = []
    for start in range(0, normalized.shape[0], batch_size):
        chunks.append(classifier(normalized[start : start + batch_size]).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def summarize_scores(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(scores.shape[0]),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
        "obstacle_fraction_at_threshold": float(np.mean(scores <= float(threshold))),
    }


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    pair_path = args.pair_path.expanduser().resolve()
    obs_model_path = args.obs_model_path.expanduser().resolve()

    pair_payload = load_pair_payload(pair_path)
    classifier, feature_mean, feature_std, clf_payload = load_classifier(obs_model_path, device)

    cache_config = clf_payload["cache_config"]
    model_dir = Path(cache_config["model_dir"]).expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else Path(cache_config["checkpoint_path"]).expanduser().resolve()
    )
    config_dict = load_config(model_dir)
    embed_dim = int(config_dict.get("embed_dim", clf_payload["input_dim"]))
    img_size = int(config_dict.get("img_size", 224))

    total_pairs = int(pair_payload["start"]["pixels"].shape[0])
    indices = select_indices(total_pairs, int(args.num_samples), int(args.seed))
    world_model = load_world_model(checkpoint_path, device)
    try:
        start_latents = encode_pixels(
            world_model,
            np.asarray(pair_payload["start"]["pixels"], dtype=np.uint8),
            indices,
            device=device,
            img_size=img_size,
            embed_dim=embed_dim,
            frame_batch_size=int(args.frame_batch_size),
            progress_desc="Encoding start samples",
        )
        goal_latents = encode_pixels(
            world_model,
            np.asarray(pair_payload["goal"]["pixels"], dtype=np.uint8),
            indices,
            device=device,
            img_size=img_size,
            embed_dim=embed_dim,
            frame_batch_size=int(args.frame_batch_size),
            progress_desc="Encoding goal samples",
        )
    finally:
        del world_model

    start_scores = score_latents(
        classifier,
        start_latents,
        feature_mean,
        feature_std,
        device,
        batch_size=int(args.frame_batch_size),
    )
    goal_scores = score_latents(
        classifier,
        goal_latents,
        feature_mean,
        feature_std,
        device,
        batch_size=int(args.frame_batch_size),
    )
    threshold = float(clf_payload.get("conformal_safe_score_threshold", 0.0))

    result = {
        "pair_path": str(pair_path),
        "obs_model_path": str(obs_model_path),
        "checkpoint_path": str(checkpoint_path),
        "num_pairs_evaluated": int(indices.shape[0]),
        "indices": indices.tolist(),
        "decision_threshold": threshold,
        "score_sign_convention": clf_payload.get("score_sign_convention", {}),
        "start": {
            "summary": summarize_scores(start_scores, threshold),
            "scores_first_16": start_scores[:16].tolist(),
        },
        "goal": {
            "summary": summarize_scores(goal_scores, threshold),
            "scores_first_16": goal_scores[:16].tolist(),
        },
        "pairwise": {
            "mean_score_gap_goal_minus_start": float((goal_scores - start_scores).mean()),
            "pairs_with_either_endpoint_flagged_obstacle": float(np.mean((start_scores <= threshold) | (goal_scores <= threshold))),
            "pairs_with_both_endpoints_flagged_obstacle": float(np.mean((start_scores <= threshold) & (goal_scores <= threshold))),
        },
    }

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {out_path}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
