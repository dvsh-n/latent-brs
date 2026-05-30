#!/usr/bin/env python3
"""Plot rope latent t-SNE with safe/unsafe rollouts and saved SLS tubes."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Polygon
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_PATH = REPO_ROOT / "rope" / "plan" / "obstacle_data" / "obstacle_classifier_data.pt"
DEFAULT_MODEL_DIR = REPO_ROOT / "rope" / "models" / "mlpdyn_noshadow_ft_3"
DEFAULT_SAFE_DIR = REPO_ROOT / "plots" / "rope" / "rope_safe"
DEFAULT_UNSAFE_DIR = REPO_ROOT / "plots" / "rope" / "rope_unsafe"
DEFAULT_TUBE_PATH = REPO_ROOT / "plots" / "rope" / "idx7_rope_with_tubes.zip"
DEFAULT_OUT_PATH = REPO_ROOT / "plots" / "rope" / "tsne_latent_obs_tubes.png"
TUBE_MEMBER_NAME = "idx7_rope_with_tubes/tube_data.npz"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "figure.dpi": 300,
})

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--safe-dir", type=Path, default=DEFAULT_SAFE_DIR)
    parser.add_argument("--unsafe-dir", type=Path, default=DEFAULT_UNSAFE_DIR)
    parser.add_argument("--tube-data", type=Path, default=DEFAULT_TUBE_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=80.0)
    parser.add_argument("--learning-rate", type=str, default="auto")
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--neighbor-count", type=int, default=30)
    parser.add_argument("--tube-start-step", type=int, default=1)
    parser.add_argument("--tube-plan-stride", type=int, default=5)
    parser.add_argument("--tube-max-plans", type=int, default=None)
    parser.add_argument("--tube-horizon-stride", type=int, default=1)
    parser.add_argument("--tube-alpha", type=float, default=0.18)
    parser.add_argument("--tube-alpha-decay", type=float, default=0.86)
    parser.add_argument("--tube-horizon-alpha-decay", type=float, default=0.92)
    parser.add_argument("--tube-line-alpha", type=float, default=0.85)
    parser.add_argument("--font-size", type=float, default=15.0, help="Base font size for plot text.")
    parser.add_argument("--title-font-size", type=float, default=None, help="Title font size. Defaults to --font-size + 1.")
    parser.add_argument("--tick-font-size", type=float, default=None, help="Tick label font size. Defaults to --font-size.")
    parser.add_argument("--axis-label-font-size", type=float, default=None, help="Axis label font size. Defaults to --font-size.")
    parser.add_argument("--legend-font-size", type=float, default=12.0, help="Legend font size. Defaults to --font-size - 1.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def install_numpy_core_pickle_aliases() -> None:
    try:
        import numpy.core

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    except Exception:
        return


def torch_load_local(path: Path) -> Any:
    install_numpy_core_pickle_aliases()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoint matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_world_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    install_numpy_core_pickle_aliases()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_obstacle_pixels(data_path: Path, seed: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing obstacle dataset: {data_path}")
    payload = torch_load_local(data_path)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected obstacle dataset format in {data_path}")
    dataset = payload["dataset"]
    for key in ("pixels", "label"):
        if key not in dataset:
            raise KeyError(f"Obstacle dataset is missing dataset[{key!r}]")

    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    sample_indices = np.arange(labels.shape[0], dtype=np.int64)
    if sample_indices.size == 0:
        raise ValueError(f"No samples found in {data_path}")

    if max_samples is not None and max_samples > 0 and sample_indices.size > max_samples:
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(rng.choice(sample_indices, size=max_samples, replace=False))
    return pixels, sample_indices.astype(np.int64), labels[sample_indices].astype(np.int64)


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def preprocess_pixels(
    pixels: np.ndarray,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


@torch.no_grad()
def encode_dataset_latents(
    model: torch.nn.Module,
    pixels: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
) -> np.ndarray:
    if frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    latents: list[np.ndarray] = []
    for start in tqdm(range(0, indices.shape[0], frame_batch_size), desc="Encoding dataset images", unit="batch"):
        batch_idx = indices[start : start + frame_batch_size]
        batch = preprocess_pixels(pixels[batch_idx], img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def load_rollout_embeddings(run_dir: Path) -> np.ndarray:
    states_path = run_dir / "executed_states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"Missing rollout states: {states_path}")
    states = np.load(states_path)
    if "embeddings" not in states:
        raise KeyError(f"{states_path} is missing the 'embeddings' array.")
    embeddings = np.asarray(states["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected rollout embeddings with shape (T, D), got {embeddings.shape} in {states_path}")
    return embeddings


def fit_tsne(latents: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if latents.shape[0] < 3:
        raise ValueError("Need at least three dataset samples for t-SNE.")
    perplexity = min(float(args.perplexity), max(1.0, (latents.shape[0] - 1) / 3.0))
    learning_rate: str | float = "auto" if str(args.learning_rate) == "auto" else float(args.learning_rate)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=int(args.max_iter),
        init="pca",
        random_state=int(args.seed),
        metric="euclidean",
    )
    return tsne.fit_transform(latents).astype(np.float32)


def save_plot(out_path: Path, formats: list[str], fig: plt.Figure) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_path = out_path.with_suffix("")
    written: list[Path] = []
    for fmt in formats:
        path = stem_path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path)
        written.append(path)
    return written


def np_load_dict(file_obj: Any) -> dict[str, np.ndarray]:
    with np.load(file_obj, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def load_tube_data(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tube_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing tube data: {path}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            member = TUBE_MEMBER_NAME if TUBE_MEMBER_NAME in names else None
            if member is None:
                matches = [name for name in names if name.endswith("/tube_data.npz") or name == "tube_data.npz"]
                if not matches:
                    raise FileNotFoundError(f"No tube_data.npz found inside {path}")
                member = matches[0]
            return np_load_dict(io.BytesIO(archive.read(member)))

    return np_load_dict(path)


def select_tube_plan_indices(
    plan_steps: np.ndarray,
    *,
    start_step: int,
    plan_stride: int,
    max_plans: int | None,
) -> np.ndarray:
    if plan_stride <= 0:
        raise ValueError("--tube-plan-stride must be positive.")
    selected = np.flatnonzero(plan_steps >= int(start_step))[:: int(plan_stride)]
    if max_plans is not None:
        selected = selected[: int(max_plans)]
    if selected.size == 0:
        raise ValueError(
            f"No tube plans selected. Available step range is "
            f"{int(plan_steps.min())}..{int(plan_steps.max())}."
        )
    return selected.astype(np.int64)


def make_projector(dataset_scaled: np.ndarray, dataset_xy: np.ndarray, neighbor_count: int):
    k = min(max(1, int(neighbor_count)), dataset_scaled.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(dataset_scaled)

    def project(query_scaled: np.ndarray) -> np.ndarray:
        distances, indices = nbrs.kneighbors(query_scaled)
        weights = 1.0 / np.maximum(distances, 1e-8)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return np.einsum("nk,nkd->nd", weights, dataset_xy[indices])

    return project


def tube_cross_section_points(center: np.ndarray, width: np.ndarray) -> np.ndarray:
    dim = center.shape[0]
    width = np.maximum(width, 0.0)
    points = [center]
    eye_scaled = np.eye(dim, dtype=np.float64) * width.reshape(1, -1)
    points.extend(center + eye_scaled)
    points.extend(center - eye_scaled)
    return np.stack(points, axis=0)


def build_projected_tube_sections(
    centers: np.ndarray,
    widths: np.ndarray,
    selected_plans: np.ndarray,
    *,
    embed_dim: int,
    horizon_stride: int,
    scaler: StandardScaler,
    project,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if horizon_stride <= 0:
        raise ValueError("--tube-horizon-stride must be positive.")

    section_specs: list[tuple[int, int, int, int]] = []
    all_points: list[np.ndarray] = []
    cursor = 0
    for order, plan_idx in enumerate(selected_plans):
        for horizon_idx in range(0, centers.shape[1], int(horizon_stride)):
            center = np.asarray(centers[plan_idx, horizon_idx, :embed_dim], dtype=np.float64)
            width = np.asarray(widths[plan_idx, horizon_idx, :embed_dim], dtype=np.float64)
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(width)):
                continue
            points = tube_cross_section_points(center, width)
            all_points.append(points)
            section_specs.append((order, int(plan_idx), int(horizon_idx), cursor))
            cursor += points.shape[0]

    if not all_points:
        return [], []

    stacked = np.concatenate(all_points, axis=0)
    projected = project(scaler.transform(stacked))
    sections: list[dict[str, Any]] = []
    for order, plan_idx, horizon_idx, start in section_specs:
        count = 2 * embed_dim + 1
        sections.append(
            {
                "order": order,
                "plan_idx": plan_idx,
                "horizon_idx": horizon_idx,
                "xy": projected[start : start + count],
            }
        )

    plan_lines: list[np.ndarray] = []
    for plan_idx in selected_plans:
        line = np.asarray(centers[plan_idx, :, :embed_dim], dtype=np.float64)
        valid = np.all(np.isfinite(line), axis=1)
        if np.sum(valid) >= 2:
            plan_lines.append(project(scaler.transform(line[valid])))
    return sections, plan_lines


def add_tube_sections(
    ax: plt.Axes,
    sections: list[dict[str, Any]],
    plan_lines: list[np.ndarray],
    args: argparse.Namespace,
) -> None:
    del plan_lines
    tube_color = "#46b39d"
    for section in sections:
        xy = np.asarray(section["xy"], dtype=np.float64)
        if xy.shape[0] < 3:
            continue
        try:
            hull = ConvexHull(xy)
            polygon_xy = xy[hull.vertices]
        except Exception:
            continue
        alpha = (
            float(args.tube_alpha)
            * (float(args.tube_alpha_decay) ** int(section["order"]))
            * (float(args.tube_horizon_alpha_decay) ** int(section["horizon_idx"]))
        )
        alpha = float(max(0.006, min(0.45, alpha)))
        ax.add_patch(
            Polygon(
                polygon_xy,
                closed=True,
                facecolor=tube_color,
                edgecolor="none",
                alpha=alpha,
                zorder=2,
            )
        )


def draw_trajectory(ax: plt.Axes, xy: np.ndarray, *, color: str, label: str, linewidth: float = 2.2) -> None:
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=0.95, label=label, zorder=5)
    ax.scatter(xy[:, 0], xy[:, 1], s=15.0, color=color, edgecolors="white", linewidths=0.35, zorder=6)
    ax.scatter(xy[0, 0], xy[0, 1], s=66.0, marker="o", color="#0b2f6b", edgecolors="none", linewidths=0.0, zorder=7)
    ax.scatter(xy[-1, 0], xy[-1, 1], s=96.0, marker="X", color="#f2c94c", edgecolors="black", linewidths=0.6, zorder=7)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    config = load_model_config(model_dir)
    img_size = int(config.get("img_size", 224))
    embed_dim = int(config.get("embed_dim", 12))
    device = resolve_device(str(args.device))

    pixels, dataset_indices, dataset_labels = load_obstacle_pixels(
        args.data_path.expanduser().resolve(),
        int(args.seed),
        args.max_samples,
    )
    world_model = load_world_model(checkpoint_path, device)
    dataset_latents = encode_dataset_latents(
        world_model,
        pixels,
        dataset_indices,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
    )
    del world_model

    safe_embeddings = load_rollout_embeddings(args.safe_dir.expanduser().resolve())[:, :embed_dim]
    unsafe_embeddings = load_rollout_embeddings(args.unsafe_dir.expanduser().resolve())[:, :embed_dim]
    tube_data = load_tube_data(args.tube_data)
    tube_centers = np.asarray(tube_data["nominal_centers"], dtype=np.float64)
    tube_widths = np.asarray(tube_data["tube_widths"], dtype=np.float64)
    plan_steps = np.asarray(tube_data["plan_steps"], dtype=np.int64)
    selected_plans = select_tube_plan_indices(
        plan_steps,
        start_step=int(args.tube_start_step),
        plan_stride=int(args.tube_plan_stride),
        max_plans=args.tube_max_plans,
    )

    scaler = StandardScaler().fit(dataset_latents)
    dataset_scaled = scaler.transform(dataset_latents)
    dataset_xy = fit_tsne(dataset_scaled, args)
    project = make_projector(dataset_scaled, dataset_xy, int(args.neighbor_count))

    safe_xy = project(scaler.transform(safe_embeddings))
    unsafe_xy = project(scaler.transform(unsafe_embeddings))
    tube_sections, tube_plan_lines = build_projected_tube_sections(
        tube_centers,
        tube_widths,
        selected_plans,
        embed_dim=embed_dim,
        horizon_stride=int(args.tube_horizon_stride),
        scaler=scaler,
        project=project,
    )

    base_font_size = float(args.font_size)
    title_font_size = float(args.title_font_size) if args.title_font_size is not None else base_font_size + 1.0
    tick_font_size = float(args.tick_font_size) if args.tick_font_size is not None else base_font_size
    axis_label_font_size = (
        float(args.axis_label_font_size) if args.axis_label_font_size is not None else base_font_size
    )
    legend_font_size = float(args.legend_font_size) if args.legend_font_size is not None else base_font_size - 1.0
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": axis_label_font_size,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=int(args.dpi))
    safe_label_mask = dataset_labels == 0
    obstacle_label_mask = dataset_labels == 1
    ax.scatter(
        dataset_xy[safe_label_mask, 0],
        dataset_xy[safe_label_mask, 1],
        s=7.0,
        color="#4e79a7",
        alpha=0.24,
        edgecolors="none",
        label="non-obstacle samples",
        zorder=1,
    )
    ax.scatter(
        dataset_xy[obstacle_label_mask, 0],
        dataset_xy[obstacle_label_mask, 1],
        s=7.0,
        color="#f28e2b",
        alpha=0.30,
        edgecolors="none",
        label="obstacle samples",
        zorder=1,
    )
    add_tube_sections(ax, tube_sections, tube_plan_lines, args)
    draw_trajectory(ax, safe_xy, color="#16833a", label="safe rollout", linewidth=1.8)
    draw_trajectory(ax, unsafe_xy, color="#c62828", label="unsafe rollout", linewidth=2.0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#46b39d", edgecolor="none", alpha=0.28, label="SLS tube"))
    labels.append("SLS tube")
    # ax.set_title("Latent t-SNE with SLS tubes for Bimanual Rope Manipulation", fontsize=title_font_size)
    ax.set_xlabel("t-SNE 1", fontsize=axis_label_font_size)
    ax.set_ylabel("t-SNE 2", fontsize=axis_label_font_size)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(
        handles,
        labels,
        loc="lower left",
        ncol=2,
        frameon=True,
        framealpha=0.92,
        fontsize=legend_font_size,
    )
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    written = save_plot(args.out_path.expanduser().resolve(), list(args.formats), fig)
    plt.close(fig)

    metadata = {
        "data_path": str(args.data_path.expanduser().resolve()),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "safe_dir": str(args.safe_dir.expanduser().resolve()),
        "unsafe_dir": str(args.unsafe_dir.expanduser().resolve()),
        "tube_data": str(args.tube_data.expanduser().resolve()),
        "dataset_sample_count": int(dataset_indices.shape[0]),
        "label_0_count": int(np.sum(dataset_labels == 0)),
        "label_1_count": int(np.sum(dataset_labels == 1)),
        "safe_frame_count": int(safe_embeddings.shape[0]),
        "unsafe_frame_count": int(unsafe_embeddings.shape[0]),
        "tube_frame_count": int(np.asarray(tube_data["executed_markov_states"]).shape[0]),
        "selected_tube_plan_steps": [int(plan_steps[index]) for index in selected_plans],
        "tube_sections": int(len(tube_sections)),
        "embed_dim": int(embed_dim),
        "tsne_fit_samples": "all_obstacle_dataset_samples_both_labels",
        "tube_projection": "first_embed_dim_markov_dimensions_projected_by_inverse_distance_weighted_dataset_tsne_neighbors",
        "neighbor_count": int(args.neighbor_count),
        "perplexity": float(min(float(args.perplexity), max(1.0, (dataset_latents.shape[0] - 1) / 3.0))),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "font_size": base_font_size,
        "title_font_size": title_font_size,
        "tick_font_size": tick_font_size,
        "axis_label_font_size": axis_label_font_size,
        "legend_font_size": legend_font_size,
        "written": [str(path) for path in written],
    }
    metadata_path = args.out_path.expanduser().resolve().with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Encoded dataset samples: {dataset_indices.shape[0]}")
    print(f"Selected tube plans: {len(selected_plans)}")
    print(f"Projected tube sections: {len(tube_sections)}")
    print(f"Wrote plot(s): {', '.join(str(path) for path in written)}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
