#!/usr/bin/env python3
"""Build a Reacher Latent Policy Barrier prototype bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.plan import plan_ilqr_mpc as ilqr_base
from reacher.safety.compat import register_legacy_checkpoint_aliases
from reacher.safety.latent_cache import (
    DEFAULT_CLASSIFIER_CHECKPOINT,
    compute_margins,
    encode_h5_pixels,
    make_episode_index,
    make_markov_states,
    parse_max_frames,
)

DEFAULT_DATASET_PATH = "reacher/data/expert_data_50hz/reacher_expert.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_1"
DEFAULT_OUTPUT_PATH = "reacher/safety/runs/lpb_reacher/lpb_bank.pt"
DEFAULT_Q1_MIN = 0.0
DEFAULT_Q1_MAX = 3.1415
DEFAULT_Q2_MIN = -2.88
DEFAULT_Q2_MAX = -2.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=parse_max_frames, default=4096)
    parser.add_argument("--num-prototypes", type=int, default=8192)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--safe-margin-min", type=float, default=0.0)
    parser.add_argument("--margin-source", choices=("none", "geometry", "dataset", "classifier"), default="geometry")
    parser.add_argument("--q1-min", type=float, default=DEFAULT_Q1_MIN)
    parser.add_argument("--q1-max", type=float, default=DEFAULT_Q1_MAX)
    parser.add_argument("--q2-min", type=float, default=DEFAULT_Q2_MIN)
    parser.add_argument("--q2-max", type=float, default=DEFAULT_Q2_MAX)
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER_CHECKPOINT))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("identity", "tanh", "tanh2"), default="identity")
    parser.add_argument("--allow-classifier-latent-slice", action="store_true")
    return parser.parse_args()


def geometry_box_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([float(args.q1_min), float(args.q2_min)], dtype=np.float32)
    upper = np.asarray([float(args.q1_max), float(args.q2_max)], dtype=np.float32)
    if np.any(lower > upper):
        raise ValueError("Reacher obstacle box lower bounds must be <= upper bounds.")
    return lower, upper


def signed_box_margin(qpos: np.ndarray, *, lower: np.ndarray, upper: np.ndarray) -> torch.Tensor:
    qpos_np = np.asarray(qpos, dtype=np.float32)
    if qpos_np.ndim != 2 or qpos_np.shape[1] < 2:
        raise ValueError(f"Expected qpos with shape [N, >=2], got {qpos_np.shape}.")
    q = qpos_np[:, :2]
    below = np.maximum(lower[None, :] - q, 0.0)
    above = np.maximum(q - upper[None, :], 0.0)
    outside_delta = below + above
    outside_distance = np.linalg.norm(outside_delta, axis=1)
    inside = np.all((q >= lower[None, :]) & (q <= upper[None, :]), axis=1)
    distance_to_faces = np.minimum(q - lower[None, :], upper[None, :] - q)
    inside_margin = np.min(distance_to_faces, axis=1)
    signed = outside_distance.astype(np.float32)
    signed[inside] = -inside_margin[inside].astype(np.float32)
    return torch.from_numpy(signed).float()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def choose_prototypes(states: torch.Tensor, *, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 1:
        raise ValueError("--num-prototypes must be positive.")
    total = int(states.shape[0])
    if total == 0:
        raise ValueError("Cannot choose prototypes from an empty state set.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if total <= count:
        indices = torch.arange(total, dtype=torch.long)
        return states, indices
    indices = torch.randperm(total, generator=generator)[:count]
    indices, _ = torch.sort(indices)
    return states[indices], indices


@torch.no_grad()
def nearest_distances(query: torch.Tensor, prototypes: torch.Tensor, *, chunk_size: int = 4096) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    for start in tqdm(range(0, query.shape[0], chunk_size), desc="Calibrating LPB threshold", unit="batch"):
        chunk = query[start : start + chunk_size]
        distances.append(torch.cdist(chunk, prototypes).min(dim=-1).values.cpu())
    return torch.cat(distances, dim=0)


def calibrate_threshold(
    whitened_states: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_indices: torch.Tensor,
    *,
    quantile: float,
) -> tuple[float, dict[str, Any]]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("--threshold-quantile must be between 0 and 1.")
    total = int(whitened_states.shape[0])
    mask = torch.ones(total, dtype=torch.bool)
    mask[prototype_indices] = False
    calibration = whitened_states[mask]
    calibration_source = "non_prototype_states"
    if calibration.shape[0] == 0:
        calibration_source = "all_states_leave_one_out"
        if total <= 1:
            distances = torch.zeros((total,), dtype=torch.float32)
        else:
            full_distances = torch.cdist(whitened_states, prototypes)
            if prototypes.shape[0] == total and torch.equal(prototype_indices.cpu(), torch.arange(total)):
                full_distances.fill_diagonal_(float("inf"))
            distances = full_distances.min(dim=-1).values.cpu()
    else:
        distances = nearest_distances(calibration, prototypes)
    threshold = float(torch.quantile(distances.float(), float(quantile)).item())
    stats = {
        "threshold_quantile": float(quantile),
        "threshold": threshold,
        "calibration_source": calibration_source,
        "calibration_count": int(distances.shape[0]),
        "distance_mean": float(distances.mean().item()),
        "distance_min": float(distances.min().item()),
        "distance_max": float(distances.max().item()),
    }
    return threshold, stats


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite.")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = ilqr_base.require_device(args.device)
    register_legacy_checkpoint_aliases()
    config = ilqr_base.load_config(model_dir)
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else ilqr_base.latest_object_checkpoint(model_dir).resolve()
    )
    model = ilqr_base.load_model(checkpoint, device)
    markov_deriv = int(config.get("markov_deriv", 1))
    img_size = int(config.get("img_size", 224))
    latent_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * latent_dim))

    with h5py.File(dataset_path, "r") as h5:
        total_available = int(h5["pixels"].shape[0])
        total_frames = min(total_available, int(args.max_frames)) if args.max_frames is not None else total_available
        rows = np.arange(total_frames, dtype=np.int64)
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        if "episode_idx" in h5 and "step_idx" in h5:
            episode_idx_np = np.asarray(h5["episode_idx"][rows], dtype=np.int64)
            step_idx_np = np.asarray(h5["step_idx"][rows], dtype=np.int64)
        else:
            episode_idx_np, step_idx_np = make_episode_index(ep_len, total_frames)
        dataset_margin_np = (
            np.asarray(h5["safety_margin"][rows], dtype=np.float32)
            if "safety_margin" in h5
            else None
        )
        qpos_np = np.asarray(h5["qpos"][rows], dtype=np.float32) if "qpos" in h5 else None
        latents = encode_h5_pixels(
            h5,
            rows,
            model,
            device=device,
            img_size=img_size,
            frame_batch_size=int(args.frame_batch_size),
        )

    markov_state = make_markov_states(latents[:, :latent_dim], episode_idx_np, markov_deriv)
    if int(markov_state.shape[-1]) != markov_state_dim:
        raise ValueError(f"Built Markov state dim {markov_state.shape[-1]}, expected {markov_state_dim}.")

    margin_metadata: dict[str, Any] = {"margin_source": "none"}
    safe_mask = torch.ones((markov_state.shape[0],), dtype=torch.bool)
    if args.margin_source == "geometry":
        if qpos_np is None:
            raise ValueError("--margin-source geometry requires an HDF5 'qpos' dataset.")
        box_lower, box_upper = geometry_box_bounds(args)
        margins = signed_box_margin(qpos_np, lower=box_lower, upper=box_upper)
        margin_metadata = {
            "margin_source": "geometry",
            "label_rule": "safe iff qpos lies outside the configured joint-space obstacle box",
            "obstacle_rule": "unsafe iff qpos lies inside the configured joint-space obstacle box",
            "box_lower": box_lower.tolist(),
            "box_upper": box_upper.tolist(),
        }
        safe_mask = margins > float(args.safe_margin_min)
    elif args.margin_source != "none":
        _, margins, margin_metadata = compute_margins(
            args,
            dataset_margin=dataset_margin_np,
            markov_state=markov_state,
            latent_dim=latent_dim,
            device=device,
        )
        safe_mask = margins > float(args.safe_margin_min)

    safe_states = markov_state[safe_mask].float()
    if safe_states.shape[0] == 0:
        raise ValueError("No states survived LPB safe-state filtering.")
    state_mean = safe_states.mean(dim=0)
    state_std = safe_states.std(dim=0).clamp_min(1e-6)
    whitened = (safe_states - state_mean) / state_std
    prototypes, prototype_indices = choose_prototypes(
        whitened.cpu(),
        count=int(args.num_prototypes),
        seed=int(args.seed),
    )
    threshold, threshold_stats = calibrate_threshold(
        whitened.cpu(),
        prototypes,
        prototype_indices,
        quantile=float(args.threshold_quantile),
    )

    metadata = {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "model_config": jsonable(config),
        "latent_dim": int(latent_dim),
        "markov_deriv": int(markov_deriv),
        "markov_state_dim": int(markov_state_dim),
        "total_frames": int(total_frames),
        "safe_state_count": int(safe_states.shape[0]),
        "num_prototypes": int(prototypes.shape[0]),
        "seed": int(args.seed),
        "safe_margin_min": float(args.safe_margin_min),
        "margin": jsonable(margin_metadata),
        "threshold": threshold_stats,
        "step_idx_min": int(step_idx_np.min()) if step_idx_np.size else None,
        "step_idx_max": int(step_idx_np.max()) if step_idx_np.size else None,
    }
    payload = {
        "prototypes": prototypes.float().cpu(),
        "state_mean": state_mean.float().cpu(),
        "state_std": state_std.float().cpu(),
        "threshold": float(threshold),
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(json.dumps(jsonable({"saved_to": output_path, **metadata}), indent=2))


if __name__ == "__main__":
    main()
