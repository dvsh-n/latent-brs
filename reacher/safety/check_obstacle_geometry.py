#!/usr/bin/env python3
"""Compare Reacher obstacle classifier hits against the collector geometry rule."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.safety.count_obstacle_hits import (
    DEFAULT_MODEL_DIR,
    ObstacleMLP,
    episode_indices_from_lengths,
    imagenet_pixel_stats,
    load_config,
    load_world_model,
    preprocess_pixels,
    require_device,
    resolve_artifact_path,
)
from reacher.safety.compat import register_legacy_checkpoint_aliases

DEFAULT_DATASET = "reacher/data/expert_data_50hz/reacher_expert.h5"
DEFAULT_OBSTACLE_MODEL = "reacher/safety/obs_net_latent_safety_dino_20260524_151742/b5d63ff9bff51288/model.pt"


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--obstacle-model", type=Path, default=Path(DEFAULT_OBSTACLE_MODEL))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-frames", type=parse_max_frames, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--base-xy",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="Planar arm base xy. Haoran's collector stores circle/link geometry and uses a zero base for this Reacher model.",
    )
    parser.add_argument("--threshold", default="conformal", help="'conformal', 'base', or a numeric score threshold.")
    parser.add_argument(
        "--inside-bend-sign",
        type=int,
        choices=(-1, 1),
        default=None,
        help=(
            "If set, only count inside-circle frames from this IK branch as geometry unsafe. "
            "Haoran's collector defaults to -1; omit for the broad tip-inside-circle check."
        ),
    )
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


def score_threshold(artifact: dict[str, Any], threshold_arg: str) -> float:
    if threshold_arg == "conformal":
        return float(artifact.get("conformal_safe_score_threshold", artifact.get("base_decision_threshold", 0.0)))
    if threshold_arg == "base":
        return float(artifact.get("base_decision_threshold", 0.0))
    return float(threshold_arg)


def geometry_from_artifact(artifact: dict[str, Any], base_xy: tuple[float, float]) -> dict[str, Any]:
    metadata = artifact.get("source_metadata", {})
    missing = [key for key in ("circle_center_xy", "circle_radius", "link1", "link2") if key not in metadata]
    if missing:
        raise KeyError(f"Classifier artifact source_metadata is missing geometry keys: {missing}")
    return {
        "circle_center_xy": np.asarray(metadata["circle_center_xy"], dtype=np.float64),
        "circle_radius": float(metadata["circle_radius"]),
        "outside_margin": float(metadata.get("outside_margin", 0.0)),
        "link1": float(metadata["link1"]),
        "link2": float(metadata["link2"]),
        "base_xy": np.asarray(base_xy, dtype=np.float64),
        "label_rule": metadata.get("label_rule", "1 iff fingertip lies inside workspace circle"),
    }


def fingertip_xy(qpos: np.ndarray, *, base_xy: np.ndarray, link1: float, link2: float) -> np.ndarray:
    theta1 = qpos[:, 0].astype(np.float64)
    theta2 = qpos[:, 1].astype(np.float64)
    elbow = np.stack((link1 * np.cos(theta1), link1 * np.sin(theta1)), axis=1)
    tip = elbow + np.stack(
        (
            link2 * np.cos(theta1 + theta2),
            link2 * np.sin(theta1 + theta2),
        ),
        axis=1,
    )
    return base_xy[None, :] + tip


def selected_ik_branch(qpos: np.ndarray, inside_bend_sign: int | None) -> np.ndarray:
    if inside_bend_sign is None:
        return np.ones((qpos.shape[0],), dtype=bool)
    theta2 = qpos[:, 1].astype(np.float64)
    return (float(inside_bend_sign) * np.sin(theta2)) > 0.0


def rates(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    return {
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    dataset_path = args.dataset.expanduser().resolve()
    obstacle_model_path = args.obstacle_model.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not obstacle_model_path.is_file():
        raise FileNotFoundError(f"Obstacle model not found: {obstacle_model_path}")

    register_legacy_checkpoint_aliases()
    device = require_device(args.device)
    artifact = torch.load(obstacle_model_path, map_location="cpu", weights_only=False)
    cache_config = artifact.get("cache_config", {})
    model_dir = Path(cache_config.get("model_dir", DEFAULT_MODEL_DIR)).expanduser()
    if not model_dir.is_dir():
        model_dir = DEFAULT_MODEL_DIR
    model_dir = model_dir.resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else resolve_artifact_path(str(cache_config["checkpoint_path"]), model_dir)
    )
    model_config = load_config(model_dir)
    embed_dim = int(artifact["input_dim"])
    img_size = int(cache_config.get("img_size", model_config.get("img_size", 224)))
    threshold = score_threshold(artifact, str(args.threshold))
    geom = geometry_from_artifact(artifact, tuple(args.base_xy))

    print(f"device={device}", flush=True)
    print(f"dataset={dataset_path}", flush=True)
    print(f"obstacle_model={obstacle_model_path}", flush=True)
    print(f"world_checkpoint={checkpoint_path}", flush=True)
    print(f"threshold={threshold}", flush=True)
    print(f"geometry={json.dumps(jsonable(geom))}", flush=True)
    print(f"inside_bend_sign={args.inside_bend_sign}", flush=True)

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

    tp = fp = tn = fn = 0
    gray_total = gray_classifier_unsafe = 0
    inside_circle_total = 0
    other_branch_total = other_branch_classifier_unsafe = 0
    total_frames = 0
    score_min = math.inf
    score_max = -math.inf
    distance_min = math.inf
    distance_max = -math.inf
    examples: dict[str, list[dict[str, Any]]] = {"false_positive": [], "false_negative": []}

    start_time = time.perf_counter()
    with h5py.File(dataset_path, "r") as h5:
        available_frames = int(h5["pixels"].shape[0])
        num_frames = min(available_frames, int(args.max_frames)) if args.max_frames is not None else available_frames
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        if "episode_idx" in h5:
            episode_idx = np.asarray(h5["episode_idx"][:num_frames], dtype=np.int64)
        else:
            episode_idx = episode_indices_from_lengths(ep_len, num_frames)
        if "step_idx" in h5:
            step_idx = np.asarray(h5["step_idx"][:num_frames], dtype=np.int64)
        else:
            _, step_idx = episode_indices_from_lengths(ep_len, num_frames), np.arange(num_frames, dtype=np.int64)

        num_episodes = int(np.max(episode_idx)) + 1 if episode_idx.size else int(ep_len.shape[0])
        geometry_episode_unsafe = np.zeros((num_episodes,), dtype=bool)
        classifier_episode_unsafe = np.zeros((num_episodes,), dtype=bool)

        iterator = tqdm(range(0, num_frames, args.batch_size), desc="Geometry/classifier check", unit="batch")
        for start in iterator:
            stop = min(start + args.batch_size, num_frames)
            pixels = np.asarray(h5["pixels"][start:stop], dtype=np.uint8)
            qpos = np.asarray(h5["qpos"][start:stop], dtype=np.float32)

            tip = fingertip_xy(
                qpos,
                base_xy=np.asarray(geom["base_xy"], dtype=np.float64),
                link1=float(geom["link1"]),
                link2=float(geom["link2"]),
            )
            distances = np.linalg.norm(tip - np.asarray(geom["circle_center_xy"], dtype=np.float64)[None, :], axis=1)
            inside_circle = distances <= float(geom["circle_radius"])
            selected_branch = selected_ik_branch(qpos, args.inside_bend_sign)
            geometry_unsafe = inside_circle & selected_branch
            other_branch_inside = inside_circle & ~selected_branch
            gray = (distances > float(geom["circle_radius"])) & (
                distances < float(geom["circle_radius"]) + float(geom["outside_margin"])
            )

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
            classifier_unsafe = scores_np <= threshold

            batch_episode_idx = episode_idx[start:stop]
            geometry_episode_unsafe[np.unique(batch_episode_idx[geometry_unsafe])] = True
            classifier_episode_unsafe[np.unique(batch_episode_idx[classifier_unsafe])] = True

            tp_mask = classifier_unsafe & geometry_unsafe
            fp_mask = classifier_unsafe & ~geometry_unsafe
            tn_mask = ~classifier_unsafe & ~geometry_unsafe
            fn_mask = ~classifier_unsafe & geometry_unsafe
            tp += int(np.sum(tp_mask))
            fp += int(np.sum(fp_mask))
            tn += int(np.sum(tn_mask))
            fn += int(np.sum(fn_mask))
            gray_total += int(np.sum(gray))
            gray_classifier_unsafe += int(np.sum(classifier_unsafe & gray))
            inside_circle_total += int(np.sum(inside_circle))
            other_branch_total += int(np.sum(other_branch_inside))
            other_branch_classifier_unsafe += int(np.sum(classifier_unsafe & other_branch_inside))
            total_frames += int(stop - start)
            score_min = min(score_min, float(np.min(scores_np)))
            score_max = max(score_max, float(np.max(scores_np)))
            distance_min = min(distance_min, float(np.min(distances)))
            distance_max = max(distance_max, float(np.max(distances)))

            for name, mask in (("false_positive", fp_mask), ("false_negative", fn_mask)):
                if len(examples[name]) >= 20 or not np.any(mask):
                    continue
                local_idx = np.flatnonzero(mask)[: 20 - len(examples[name])]
                for idx in local_idx.tolist():
                    examples[name].append(
                        {
                            "row": int(start + idx),
                            "episode_idx": int(episode_idx[start + idx]),
                            "step_idx": int(step_idx[start + idx]),
                            "score": float(scores_np[idx]),
                            "tip_distance": float(distances[idx]),
                            "qpos": qpos[idx].astype(float).tolist(),
                            "tip_xy": tip[idx].astype(float).tolist(),
                        }
                    )

    episode_tp = int(np.sum(classifier_episode_unsafe & geometry_episode_unsafe))
    episode_fp = int(np.sum(classifier_episode_unsafe & ~geometry_episode_unsafe))
    episode_tn = int(np.sum(~classifier_episode_unsafe & ~geometry_episode_unsafe))
    episode_fn = int(np.sum(~classifier_episode_unsafe & geometry_episode_unsafe))

    result = {
        "dataset": str(dataset_path),
        "obstacle_model": str(obstacle_model_path),
        "world_checkpoint": str(checkpoint_path),
        "threshold": float(threshold),
        "geometry": jsonable(geom),
        "inside_bend_sign": args.inside_bend_sign,
        "branch_rule": (
            "broad_tip_inside_circle"
            if args.inside_bend_sign is None
            else f"tip_inside_circle_and_sin(theta2)_sign_{int(args.inside_bend_sign):+d}"
        ),
        "frame_counts": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total": int(total_frames),
            "geometry_unsafe": int(tp + fn),
            "classifier_unsafe": int(tp + fp),
            "inside_circle_frames": int(inside_circle_total),
            "inside_other_branch_frames": int(other_branch_total),
            "inside_other_branch_classifier_unsafe": int(other_branch_classifier_unsafe),
            "gray_band_frames": int(gray_total),
            "gray_band_classifier_unsafe": int(gray_classifier_unsafe),
        },
        "frame_metrics": rates(tp, fp, tn, fn),
        "episode_counts": {
            "tp": episode_tp,
            "fp": episode_fp,
            "tn": episode_tn,
            "fn": episode_fn,
            "total": int(episode_tp + episode_fp + episode_tn + episode_fn),
            "geometry_unsafe": int(episode_tp + episode_fn),
            "classifier_unsafe": int(episode_tp + episode_fp),
        },
        "episode_metrics": rates(episode_tp, episode_fp, episode_tn, episode_fn),
        "score_min": float(score_min),
        "score_max": float(score_max),
        "tip_distance_min": float(distance_min),
        "tip_distance_max": float(distance_max),
        "elapsed_sec": float(time.perf_counter() - start_time),
        "examples": examples,
    }

    print("RESULT")
    print(json.dumps(jsonable(result), indent=2))
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
        print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
