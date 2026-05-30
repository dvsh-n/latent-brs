#!/usr/bin/env python3
"""Encode obstacle/test frames into the embd-12 straightened latent space and visualize them with PCA."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from ogbench_cube.plan.obstacle_net_3d_ellipsoid import (
    ObstacleMLP,
    load_config,
    load_world_model,
    require_device,
)

DEFAULT_INPUT_PATH = Path("ogbench_cube/plan/obstacle_data_3d_ellipsoid/obstacle_classifier_data_3d_ellipsoid.pt")
DEFAULT_OBS_MODEL_PATH = Path("ogbench_cube/plan/obs_net_embd_8/be6d7840183a2948/model.pt")
DEFAULT_FRAME_BATCH_SIZE = 128
DEFAULT_CLF_BATCH_SIZE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--obs-model-path", type=Path, default=DEFAULT_OBS_MODEL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=DEFAULT_FRAME_BATCH_SIZE)
    parser.add_argument("--clf-batch-size", type=int, default=DEFAULT_CLF_BATCH_SIZE)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-npz", type=Path, default=None)
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


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return pixel_mean, pixel_std


def preprocess_pixels(
    pixels: np.ndarray,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


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


def load_pixels_from_pt(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected obstacle dataset format in {path}.")
    dataset = payload["dataset"]
    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    labels = None if "label" not in dataset else np.asarray(dataset["label"], dtype=np.int64)
    return pixels, labels, payload


@torch.no_grad()
def encode_dataset_latents(
    model: torch.nn.Module,
    dataset_path: Path,
    *,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
) -> np.ndarray:
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    latents: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as h5:
        pixels = h5["pixels"]
        total = int(pixels.shape[0])
        for start in tqdm(range(0, total, frame_batch_size), desc="Encoding test frames", unit="batch"):
            stop = min(start + frame_batch_size, total)
            batch = preprocess_pixels(
                np.asarray(pixels[start:stop], dtype=np.uint8),
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            output = model.encoder(batch, interpolate_pos_encoding=True)
            emb = model.projector(output.last_hidden_state[:, 0])
            latents.append(emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def encode_pixel_array(
    model: torch.nn.Module,
    pixels: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
) -> np.ndarray:
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    latents: list[np.ndarray] = []
    for start in tqdm(range(0, pixels.shape[0], frame_batch_size), desc="Encoding frames", unit="batch"):
        stop = min(start + frame_batch_size, pixels.shape[0])
        batch = preprocess_pixels(
            pixels[start:stop],
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def score_latents(
    classifier: ObstacleMLP,
    latents: np.ndarray,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    normalized = (torch.from_numpy(latents.astype(np.float32)).to(device) - feature_mean) / feature_std
    scores: list[torch.Tensor] = []
    for start in tqdm(range(0, normalized.shape[0], batch_size), desc="Scoring latents", unit="batch"):
        scores.append(classifier(normalized[start : start + batch_size]).detach().cpu())
    return torch.cat(scores, dim=0).numpy().astype(np.float32)


def resolve_outputs(args: argparse.Namespace, model_dir_name: str) -> tuple[Path, Path]:
    default_dir = Path("ogbench_cube/plan/obs_net_embd12_strtn")
    png_path = (
        args.out_png
        if args.out_png is not None
        else default_dir / f"{Path(args.input_path).stem}_{model_dir_name}_pca_classifier.png"
    )
    npz_path = (
        args.out_npz
        if args.out_npz is not None
        else png_path.with_suffix(".npz")
    )
    return png_path.expanduser().resolve(), npz_path.expanduser().resolve()


def plot_pca(
    pca_xy: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_obstacle = labels.astype(bool)
    safe_mask = ~is_obstacle

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    axes[0].scatter(
        pca_xy[safe_mask, 0],
        pca_xy[safe_mask, 1],
        s=6,
        c="#1f77b4",
        alpha=0.55,
        linewidths=0.0,
        label="predicted non-obstacle",
    )
    axes[0].scatter(
        pca_xy[is_obstacle, 0],
        pca_xy[is_obstacle, 1],
        s=6,
        c="#d62728",
        alpha=0.55,
        linewidths=0.0,
        label="predicted obstacle",
    )
    axes[0].set_title("PCA with classifier labels")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="best", markerscale=3, frameon=False)

    scatter = axes[1].scatter(
        pca_xy[:, 0],
        pca_xy[:, 1],
        s=6,
        c=scores,
        cmap="coolwarm",
        alpha=0.65,
        linewidths=0.0,
    )
    axes[1].set_title(f"PCA with classifier score heatmap (threshold={threshold:.3f})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=axes[1], shrink=0.92)
    cbar.set_label("classifier score")

    fig.suptitle("OGBench cube latents in embd-12 straightened space")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    input_path = args.input_path.expanduser().resolve()
    obs_model_path = args.obs_model_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if not obs_model_path.is_file():
        raise FileNotFoundError(f"Obstacle model not found: {obs_model_path}")

    classifier, feature_mean, feature_std, clf_payload = load_classifier(obs_model_path, device)
    cache_config = clf_payload["cache_config"]
    model_dir = Path(cache_config["model_dir"]).expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else Path(cache_config.get("checkpoint_path", latest_object_checkpoint(model_dir))).expanduser().resolve()
    )
    config_dict = load_config(model_dir)
    embed_dim = int(config_dict.get("embed_dim", clf_payload["input_dim"]))
    img_size = int(config_dict.get("img_size", 224))
    png_path, npz_path = resolve_outputs(args, model_dir.name)

    world_model = load_world_model(checkpoint_path, device)
    try:
        source_kind = input_path.suffix.lower()
        true_labels: np.ndarray | None = None
        if source_kind == ".pt":
            pixels, true_labels, source_payload = load_pixels_from_pt(input_path)
            latents = encode_pixel_array(
                world_model,
                pixels,
                device=device,
                img_size=img_size,
                embed_dim=embed_dim,
                frame_batch_size=int(args.frame_batch_size),
            )
        elif source_kind in {".h5", ".hdf5"}:
            source_payload = None
            latents = encode_dataset_latents(
                world_model,
                input_path,
                device=device,
                img_size=img_size,
                embed_dim=embed_dim,
                frame_batch_size=int(args.frame_batch_size),
            )
        else:
            raise ValueError(f"Unsupported input suffix '{input_path.suffix}'. Expected .pt or .h5/.hdf5.")
    finally:
        del world_model

    scores = score_latents(
        classifier,
        latents,
        feature_mean,
        feature_std,
        device=device,
        batch_size=int(args.clf_batch_size),
    )
    threshold = float(clf_payload.get("conformal_safe_score_threshold", 0.0))
    predicted_obstacle = (scores <= threshold).astype(np.uint8)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    pca_xy = pca.fit_transform(latents).astype(np.float32)
    explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float64)

    plot_pca(pca_xy, predicted_obstacle, scores, threshold, png_path)

    npz_payload = {
        "latents": latents.astype(np.float32),
        "pca_xy": pca_xy.astype(np.float32),
        "scores": scores.astype(np.float32),
        "predicted_obstacle": predicted_obstacle.astype(np.uint8),
        "threshold": np.asarray([threshold], dtype=np.float32),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float32),
    }
    if true_labels is not None:
        npz_payload["true_labels"] = true_labels.astype(np.int64)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **npz_payload)

    summary = {
        "input_path": str(input_path),
        "input_kind": input_path.suffix.lower(),
        "obs_model_path": str(obs_model_path),
        "checkpoint_path": str(checkpoint_path),
        "num_frames": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "threshold": threshold,
        "predicted_obstacle_fraction": float(predicted_obstacle.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "pca_explained_variance_ratio": explained_variance_ratio.tolist(),
        "out_png": str(png_path),
        "out_npz": str(npz_path),
    }
    if true_labels is not None:
        summary["ground_truth_obstacle_fraction"] = float(true_labels.mean())
        summary["prediction_agreement"] = float(np.mean(predicted_obstacle == true_labels.astype(np.uint8)))
        summary["source_metadata_keys"] = sorted(source_payload.get("metadata", {}).keys()) if isinstance(source_payload, dict) else []
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
