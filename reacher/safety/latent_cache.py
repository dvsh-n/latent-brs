#!/usr/bin/env python3
"""Build a Reacher latent cache for the latent-safety/PyHJ pipeline."""

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
from reacher.safety.classifier_oracle import ReacherObstacleClassifierMargin, transform_margin
from reacher.safety.compat import register_legacy_checkpoint_aliases
from reacher.train.lewm_train_mlp_markov import LeWMReacherDataset

DEFAULT_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/Haoran_obs_data/mlpdyn_ft_6"
DEFAULT_OUTPUT_PATH = "reacher/safety/cache/reacher_latent_safety_smoke.pt"
DEFAULT_CLASSIFIER_CHECKPOINT = "reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt"


def parse_max_frames(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"all", "none", "full", "-1", "0"}:
        return None
    try:
        frames = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer frame count or 'all'.") from exc
    if frames < 2:
        raise argparse.ArgumentTypeError("--max-frames must be at least 2, or use 'all'.")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-frames",
        type=parse_max_frames,
        default=4096,
        help="Number of frames to cache. Use 'all' to process the full HDF5.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--margin-source",
        choices=("dataset", "classifier"),
        default="classifier",
        help="Use existing HDF5 safety_margin values or the Reacher obstacle classifier.",
    )
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER_CHECKPOINT))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument(
        "--margin-transform",
        choices=("identity", "tanh", "tanh2"),
        default="identity",
        help=(
            "Transform applied after the signed margin is computed. "
            "Use tanh for Dreamer-margin style, tanh2 for latent-safety DINO failure-head style."
        ),
    )
    parser.add_argument("--allow-classifier-latent-slice", action="store_true")
    return parser.parse_args()


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


def stats_dataset_path(config: dict[str, object], dataset_path: Path) -> Path:
    configured = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    return configured if configured.is_file() else dataset_path


def load_action_stats(config: dict[str, object], dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats_dataset = LeWMReacherDataset(
        stats_dataset_path(config, dataset_path),
        history_size=int(config.get("history_size", 1)),
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=int(config.get("img_size", 224)),
        action_dim=int(config.get("action_dim", 2)),
    )
    return stats_dataset.action_mean.astype(np.float32), stats_dataset.action_std.astype(np.float32)


def make_episode_index(ep_len: np.ndarray, total_frames: int) -> tuple[np.ndarray, np.ndarray]:
    episode_idx = np.empty((total_frames,), dtype=np.int64)
    step_idx = np.empty((total_frames,), dtype=np.int64)
    cursor = 0
    for ep, length in enumerate(ep_len.tolist()):
        stop = min(cursor + int(length), total_frames)
        if stop <= cursor:
            break
        count = stop - cursor
        episode_idx[cursor:stop] = ep
        step_idx[cursor:stop] = np.arange(count, dtype=np.int64)
        cursor = stop
        if cursor >= total_frames:
            break
    if cursor < total_frames:
        raise ValueError("Could not map all selected rows to reacher episodes.")
    return episode_idx, step_idx


@torch.no_grad()
def encode_h5_pixels(
    h5: h5py.File,
    rows: np.ndarray,
    model: torch.nn.Module,
    *,
    device: torch.device,
    img_size: int,
    frame_batch_size: int,
) -> torch.Tensor:
    latents: list[torch.Tensor] = []
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    for start in tqdm(range(0, rows.shape[0], frame_batch_size), desc="Encoding reacher frames", unit="batch"):
        batch_rows = rows[start : start + frame_batch_size]
        pixels_np = np.asarray(h5["pixels"][batch_rows], dtype=np.uint8)
        pixels = ilqr_base.preprocess_pixels(
            pixels_np,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        latents.append(ilqr_base.encode_frames(model, pixels, device=device, frame_batch_size=frame_batch_size).cpu())
    return torch.cat(latents, dim=0).float()


def make_markov_states(latents: torch.Tensor, episode_idx: np.ndarray, markov_deriv: int) -> torch.Tensor:
    if markov_deriv < 0:
        raise ValueError("markov_deriv must be non-negative.")
    if latents.ndim != 2:
        raise ValueError(f"Expected latents with shape [N, D], got {tuple(latents.shape)}.")

    episode_idx_t = torch.from_numpy(np.asarray(episode_idx, dtype=np.int64))
    same_episode_prev = torch.zeros((latents.shape[0],), dtype=torch.bool)
    if latents.shape[0] > 1:
        same_episode_prev[1:] = episode_idx_t[1:] == episode_idx_t[:-1]

    components = [latents.float()]
    current = latents.float()
    for _ in range(markov_deriv):
        previous = torch.empty_like(current)
        previous[0] = current[0]
        previous[1:] = current[:-1]
        previous[~same_episode_prev] = current[~same_episode_prev]
        current = current - previous
        components.append(current)
    return torch.cat(components, dim=-1).float()


def make_valid_transition_mask(episode_idx: np.ndarray, actions: torch.Tensor) -> torch.Tensor:
    valid = torch.zeros((episode_idx.shape[0],), dtype=torch.bool)
    if episode_idx.shape[0] <= 1:
        return valid
    finite_action = torch.isfinite(actions).all(dim=-1)
    same_episode_next = torch.from_numpy(episode_idx[1:] == episode_idx[:-1])
    valid[:-1] = finite_action[:-1] & same_episode_next
    return valid


def compute_margins(
    args: argparse.Namespace,
    *,
    dataset_margin: np.ndarray | None,
    markov_state: torch.Tensor,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if args.margin_source == "dataset":
        if dataset_margin is None:
            raise ValueError("--margin-source dataset requires an HDF5 'safety_margin' dataset.")
        raw_margins = torch.from_numpy(np.asarray(dataset_margin, dtype=np.float32)).float()
        margins = transform_margin(raw_margins, str(args.margin_transform)).float()
        return raw_margins, margins, {"margin_source": "dataset"}

    if args.classifier_checkpoint is None:
        raise ValueError("--classifier-checkpoint is required when --margin-source classifier.")
    classifier = ReacherObstacleClassifierMargin(
        args.classifier_checkpoint,
        device=device,
        latent_dim=latent_dim,
        threshold=str(args.classifier_threshold),
        margin_transform=str(args.margin_transform),
        allow_latent_slice=bool(args.allow_classifier_latent_slice),
    )
    raw_margins = []
    margins = []
    for start in tqdm(range(0, markov_state.shape[0], args.frame_batch_size), desc="Scoring classifier margins", unit="batch"):
        batch = markov_state[start : start + args.frame_batch_size].to(device)
        raw_margins.append(classifier.raw_margin(batch).cpu())
        margins.append(classifier.margin(batch).cpu())
    return (
        torch.cat(raw_margins, dim=0).float(),
        torch.cat(margins, dim=0).float(),
        {"margin_source": "classifier", "classifier": classifier.metadata()},
    )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite.")
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive.")
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
    action_dim = int(config.get("action_dim", 2))
    latent_dim = int(config.get("embed_dim", 5))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * latent_dim))
    action_mean, action_std = load_action_stats(config, dataset_path)

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
        actions_np = np.asarray(h5["action"][rows], dtype=np.float32)
        dataset_margin_np = (
            np.asarray(h5["safety_margin"][rows], dtype=np.float32)
            if "safety_margin" in h5
            else None
        )
        observation_np = np.asarray(h5["observation"][rows], dtype=np.float32) if "observation" in h5 else None
        qpos_np = np.asarray(h5["qpos"][rows], dtype=np.float32) if "qpos" in h5 else None
        qvel_np = np.asarray(h5["qvel"][rows], dtype=np.float32) if "qvel" in h5 else None
        latents = encode_h5_pixels(
            h5,
            rows,
            model,
            device=device,
            img_size=img_size,
            frame_batch_size=int(args.frame_batch_size),
        )
        attrs = {key: jsonable(value) for key, value in h5.attrs.items()}

    markov_state = make_markov_states(latents, episode_idx_np, markov_deriv)
    if int(markov_state.shape[-1]) != markov_state_dim:
        raise ValueError(f"Built markov_state_dim={markov_state.shape[-1]}, config says {markov_state_dim}.")
    actions_raw = torch.from_numpy(actions_np).float()
    action_mean_t = torch.from_numpy(action_mean).float()
    action_std_t = torch.from_numpy(action_std).float()
    actions_norm = (torch.nan_to_num(actions_raw, nan=0.0) - action_mean_t) / action_std_t
    valid_transition = make_valid_transition_mask(episode_idx_np, actions_raw)
    raw_margins, margins, margin_metadata = compute_margins(
        args,
        dataset_margin=dataset_margin_np,
        markov_state=markov_state,
        latent_dim=latent_dim,
        device=device,
    )

    payload = {
        "latent": latents,
        "markov_state": markov_state,
        "action_raw": actions_raw,
        "action_norm": actions_norm,
        "raw_safety_margin": raw_margins,
        "safety_margin": margins,
        "failure": (margins <= 0.0).float(),
        "episode_idx": torch.from_numpy(episode_idx_np).long(),
        "step_idx": torch.from_numpy(step_idx_np).long(),
        "valid_transition": valid_transition,
        "metadata": {
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint),
            "num_frames": int(total_frames),
            "num_valid_transitions": int(valid_transition.sum().item()),
            "unsafe_fraction": float((margins <= 0.0).float().mean().item()),
            "latent_dim": int(latent_dim),
            "markov_deriv": int(markov_deriv),
            "markov_state_dim": int(markov_state_dim),
            "action_dim": int(action_dim),
            "action_mean": action_mean.reshape(-1).tolist(),
            "action_std": action_std.reshape(-1).tolist(),
            "model_config": config,
            "h5_attrs": attrs,
            "margin_transform": str(args.margin_transform),
            "raw_unsafe_fraction": float((raw_margins <= 0.0).float().mean().item()),
            **margin_metadata,
        },
    }
    if observation_np is not None:
        payload["observation"] = torch.from_numpy(observation_np).float()
    if qpos_np is not None:
        payload["qpos"] = torch.from_numpy(qpos_np).float()
    if qvel_np is not None:
        payload["qvel"] = torch.from_numpy(qvel_np).float()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "latent_shape": list(latents.shape),
                "markov_state_shape": list(markov_state.shape),
                "num_valid_transitions": int(valid_transition.sum().item()),
                "unsafe_fraction": float((margins <= 0.0).float().mean().item()),
                "raw_unsafe_fraction": float((raw_margins <= 0.0).float().mean().item()),
                "margin_source": margin_metadata["margin_source"],
                "margin_transform": str(args.margin_transform),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
