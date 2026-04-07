#!/usr/bin/env python3
"""Encode Reacher videos with DINOv2 and visualize CLS + patch token PCA trajectories."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV2_REPO = REPO_ROOT / "third_party" / "dinov2"
if str(DINOV2_REPO) not in sys.path:
    sys.path.insert(0, str(DINOV2_REPO))

import hubconf  # noqa: E402  (must come after sys.path insert)

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
VIDEO_DIR   = REPO_ROOT / "test" / "videos_768" 
VIZ_VIDEO   = VIDEO_DIR / "trajectory_0000051.mp4"   # trajectory to visualize
OUTPUT_DIR  = REPO_ROOT / "test"

N_TRAJECTORIES   = 50   # videos sampled to fit PCA
ENCODE_BATCH_SIZE = 101  # frames per forward pass
SEED = 0

PATCH_SIZE = 14  # DINOv2 ViT patch size

# ---------------------------------------------------------------------------

def _compute_input_size(video_dir: Path, patch_size: int = PATCH_SIZE) -> int:
    first = next(video_dir.glob("*.mp4"))
    cap = cv2.VideoCapture(str(first))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    side = min(w, h)
    return (side // patch_size) * patch_size

INPUT_SIZE = _compute_input_size(VIDEO_DIR)

MODELS = [
    {
        "name": "ViT-S/14",
        "hub_fn": "dinov2_vits14",
        "weights": REPO_ROOT / "models" / "dinov2_vits14_pretrain.pth",
        "embed_dim": 384,
    },
    # {
    #     "name": "ViT-S/14 reg4",
    #     "hub_fn": "dinov2_vits14_reg",
    #     "weights": REPO_ROOT / "models" / "dinov2_vits14_reg4_pretrain.pth",
    #     "embed_dim": 384,
    # },
    # {
    #     "name": "ViT-B/14",
    #     "hub_fn": "dinov2_vitb14",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitb14_pretrain.pth",
    #     "embed_dim": 768,
    # },
    # {
    #     "name": "ViT-B/14 reg4",
    #     "hub_fn": "dinov2_vitb14_reg",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitb14_reg4_pretrain.pth",
    #     "embed_dim": 768,
    # },
    # {
    #     "name": "ViT-L/14",
    #     "hub_fn": "dinov2_vitl14",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitl14_pretrain.pth",
    #     "embed_dim": 1024,
    # },
    # {
    #     "name": "ViT-L/14 reg4",
    #     "hub_fn": "dinov2_vitl14_reg",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitl14_reg4_pretrain.pth",
    #     "embed_dim": 1024,
    # },
    # {
    #     "name": "ViT-G/14",
    #     "hub_fn": "dinov2_vitg14",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitg14_pretrain.pth",
    #     "embed_dim": 1536,
    # },
    # {
    #     "name": "ViT-G/14 reg4",
    #     "hub_fn": "dinov2_vitg14_reg",
    #     "weights": REPO_ROOT / "models" / "dinov2_vitg14_reg4_pretrain.pth",
    #     "embed_dim": 1536,
    # },
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(hub_fn: str, weights: Path) -> torch.nn.Module:
    print(f"Building {hub_fn} model")
    model = getattr(hubconf, hub_fn)(pretrained=False)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    print(f"  Loaded weights from {weights}")
    return model


# ---------------------------------------------------------------------------
# Frame extraction & preprocessing
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def preprocess_batch(frames: list[np.ndarray]) -> torch.Tensor:
    """Resize and normalise a list of frames → (B, 3, H, W) on DEVICE."""
    tensors = []
    for frame in frames:
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(img).permute(2, 0, 1))  # (3, H, W)
    return torch.stack(tensors).to(DEVICE)  # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    batch_size: int = ENCODE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode frames in batches of `batch_size`.

    Returns:
        cls_tokens:   (T, D)
        patch_tokens: (T, num_patches, D)
    """
    cls_list, patch_list = [], []
    batches = range(0, len(frames), batch_size)
    for start in tqdm(batches, desc="  encoding", unit="batch", leave=False):
        batch = frames[start : start + batch_size]
        tensor = preprocess_batch(batch)          # (B, 3, H, W)
        out = model.forward_features(tensor)
        cls_list.append(out["x_norm_clstoken"].cpu().numpy())       # (B, D)
        patch_list.append(out["x_norm_patchtokens"].cpu().numpy())  # (B, P, D)
    return np.concatenate(cls_list, axis=0), np.concatenate(patch_list, axis=0)


def sample_video_paths(video_dir: Path, n: int, seed: int) -> list[Path]:
    all_videos = sorted(video_dir.glob("*.mp4"))
    rng = random.Random(seed)
    return rng.sample(all_videos, min(n, len(all_videos)))


def collect_pca_corpus(
    model: torch.nn.Module,
    video_paths: list[Path],
    batch_size: int = ENCODE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each video independently and accumulate only the embeddings."""
    all_cls, all_patch = [], []
    for vp in tqdm(video_paths, desc="  corpus", unit="video"):
        frames = extract_frames(vp)
        cls, patch = encode_frames(model, frames, batch_size)
        all_cls.append(cls)
        all_patch.append(patch)
    return np.concatenate(all_cls, axis=0), np.concatenate(all_patch, axis=0)


# ---------------------------------------------------------------------------
# PCA + plotting
# ---------------------------------------------------------------------------

def fit_pca(data: np.ndarray, n_components: int = 2) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


def plot_trajectory(ax: plt.Axes, traj: np.ndarray, title: str) -> None:
    T = len(traj)
    colors = cm.plasma(np.linspace(0, 1, T))
    for t in range(T - 1):
        ax.plot(traj[t:t+2, 0], traj[t:t+2, 1], color=colors[t], linewidth=1.2, alpha=0.8)
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(T), cmap="plasma", s=20, zorder=3)
    ax.scatter(*traj[0], color="green", s=80, zorder=5, label="start", marker="o")
    ax.scatter(*traj[-1], color="red",   s=80, zorder=5, label="end",   marker="X")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="frame index")


def compute_frame_delta(traj: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)


def compute_dist_to_goal(traj: np.ndarray) -> np.ndarray:
    return np.linalg.norm(traj - traj[-1], axis=1)


def plot_deltas(ax: plt.Axes, deltas: np.ndarray, label: str, color: str) -> None:
    ax.plot(deltas, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("frame")
    ax.set_ylabel("L2 distance to next frame")
    ax.set_title("Frame-to-frame dynamics (PCA space)")
    ax.legend()


def plot_goal_distances(ax: plt.Axes, cls_dist: np.ndarray, patch_dist: np.ndarray) -> None:
    ax.plot(cls_dist  / cls_dist.max(),   color="steelblue",  linewidth=1.5, label="CLS (normalised)")
    ax.plot(patch_dist / patch_dist.max(), color="darkorange", linewidth=1.5, label="Patch (normalised)")
    ax.set_xlabel("frame")
    ax.set_ylabel("L2 distance to goal frame")
    ax.set_title("Distance to goal frame (PCA space)")
    ax.legend()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(cfg: dict, pca_paths: list[Path], viz_frames: list[np.ndarray], axes: np.ndarray) -> None:
    model = load_model(cfg["hub_fn"], cfg["weights"])

    print(f"\nBuilding PCA corpus from {len(pca_paths)} trajectories...")
    corpus_cls, corpus_patch = collect_pca_corpus(model, pca_paths)
    print(f"  corpus_cls:   {corpus_cls.shape}")
    print(f"  corpus_patch: {corpus_patch.shape}")

    print(f"\nEncoding viz trajectory ({len(viz_frames)} frames)...")
    viz_cls, viz_patch = encode_frames(model, viz_frames)

    del model
    torch.cuda.empty_cache()

    T = len(viz_frames)
    corpus_patch_flat = corpus_patch.reshape(len(corpus_patch), -1)
    viz_patch_flat    = viz_patch.reshape(T, -1)

    print("\nFitting PCA on corpus...")
    cls_pca   = fit_pca(corpus_cls)
    patch_pca = fit_pca(corpus_patch_flat)

    cls_var   = cls_pca.explained_variance_ratio_
    patch_var = patch_pca.explained_variance_ratio_
    print(f"  CLS   PCA explained variance: PC1={cls_var[0]:.3f}, PC2={cls_var[1]:.3f}")
    print(f"  Patch PCA explained variance: PC1={patch_var[0]:.3f}, PC2={patch_var[1]:.3f}")

    viz_cls_proj   = cls_pca.transform(viz_cls)
    viz_patch_proj = patch_pca.transform(viz_patch_flat)

    plot_trajectory(
        axes[0], viz_cls_proj,
        f"CLS token trajectory (PCA)\nvar: {cls_var[0]:.2f}+{cls_var[1]:.2f}={cls_var.sum():.2f}",
    )
    plot_trajectory(
        axes[1], viz_patch_proj,
        f"Stacked patch tokens trajectory (PCA)\nvar: {patch_var[0]:.2f}+{patch_var[1]:.2f}={patch_var.sum():.2f}",
    )

    cls_deltas   = compute_frame_delta(viz_cls_proj)
    patch_deltas = compute_frame_delta(viz_patch_proj)
    plot_deltas(axes[2], cls_deltas   / cls_deltas.max(),   "CLS (normalised)",   "steelblue")
    plot_deltas(axes[2], patch_deltas / patch_deltas.max(), "Patch (normalised)", "darkorange")

    cls_goal   = compute_dist_to_goal(viz_cls_proj)
    patch_goal = compute_dist_to_goal(viz_patch_proj)
    plot_goal_distances(axes[3], cls_goal, patch_goal)


def main() -> None:
    print(f"Device:     {DEVICE}")
    print(f"Input size: {INPUT_SIZE}px  ({INPUT_SIZE // PATCH_SIZE}×{INPUT_SIZE // PATCH_SIZE} patches)")
    print(f"PCA corpus: {N_TRAJECTORIES} trajectories  |  batch size: {ENCODE_BATCH_SIZE}")

    pca_paths = sample_video_paths(VIDEO_DIR, N_TRAJECTORIES, SEED)
    print(f"\nSampled {len(pca_paths)} videos from {VIDEO_DIR}")

    print(f"\nExtracting viz frames from {VIZ_VIDEO.name}...")
    viz_frames = extract_frames(VIZ_VIDEO)
    print(f"  {len(viz_frames)} frames ({viz_frames[0].shape[1]}×{viz_frames[0].shape[0]})")

    n_models = len(MODELS)
    fig, all_axes = plt.subplots(n_models, 4, figsize=(22, 5 * n_models), squeeze=False)
    fig.suptitle(
        f"DINOv2 PCA trajectories — {VIZ_VIDEO.name}  (PCA fit on {len(pca_paths)} trajectories)",
        fontsize=14, y=1.01,
    )

    for row, cfg in enumerate(MODELS):
        print(f"\n{'='*60}")
        print(f"Model: {cfg['name']}")
        print('='*60)
        axes = all_axes[row]
        run_model(cfg, pca_paths, viz_frames, axes)
        axes[0].annotate(
            cfg["name"],
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.35, 0.5), textcoords="axes fraction",
            fontsize=12, fontweight="bold", va="center", ha="center",
            rotation=90,
        )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "dinov2_pca_all_models.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
