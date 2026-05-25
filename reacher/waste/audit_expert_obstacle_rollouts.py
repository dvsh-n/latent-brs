#!/usr/bin/env python3
"""Audit expert Reacher rollouts for obstacle crossings predicted by the learned classifier."""

from __future__ import annotations

import argparse
import json
import os
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
import imageio.v2 as imageio
import numpy as np
import torch

from rope.plan.obstacle_net import ObstacleMLP, preprocess_pixels
from reacher.plan.obs_data_collect import forward_kinematics
from reacher.plan.plan_ilqr_mpc import latest_object_checkpoint, load_config, require_device

DEFAULT_DATASET_PATH = "reacher/data/test_data_noisy.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_6"
DEFAULT_OBS_NET_DIR = "reacher/plan/obs_net"
DEFAULT_OBSTACLE_SUMMARY_PATH = "reacher/plan/obstacle_data/summary.json"
DEFAULT_OUT_DIR = "reacher/plan/expert_obstacle_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--obs-net-dir", type=Path, default=Path(DEFAULT_OBS_NET_DIR))
    parser.add_argument("--obs-model-path", type=Path, default=None)
    parser.add_argument("--obstacle-summary-path", type=Path, default=Path(DEFAULT_OBSTACLE_SUMMARY_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=1024)
    parser.add_argument("--save-limit", type=int, default=10)
    parser.add_argument("--video-fps", type=float, default=None)
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[audit_expert_obstacle_rollouts] {message}", flush=True)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def discover_latest_obs_model(obs_net_dir: Path) -> Path:
    candidates = sorted(obs_net_dir.rglob("model.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No obstacle classifier model.pt found under {obs_net_dir}")
    return candidates[0]


def load_obstacle_spec(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Obstacle summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    required = ("circle_center_xy", "circle_radius", "link1", "link2", "center_qpos")
    missing = [key for key in required if key not in summary]
    if missing:
        raise KeyError(f"Obstacle summary missing required keys: {missing}")
    circle_center_xy = np.asarray(summary["circle_center_xy"], dtype=np.float64)
    center_qpos = np.asarray(summary["center_qpos"], dtype=np.float64)
    link1 = float(summary["link1"])
    link2 = float(summary["link2"])
    _, center_tip_local = forward_kinematics(center_qpos, link1=link1, link2=link2)
    base_xy = circle_center_xy - center_tip_local
    return {
        "circle_center_xy": circle_center_xy,
        "circle_radius": float(summary["circle_radius"]),
        "link1": link1,
        "link2": link2,
        "center_qpos": center_qpos,
        "base_xy": base_xy.astype(np.float64),
        "raw_summary": summary,
    }


def load_world_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_obstacle_classifier(model_path: Path, device: torch.device) -> tuple[ObstacleMLP, dict[str, Any]]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ObstacleMLP(
        int(payload["input_dim"]),
        int(payload["hidden_dim"]),
        int(payload["depth"]),
        float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, payload


def analytic_obstacle_hits(qpos: np.ndarray, obstacle_spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if qpos.ndim != 2 or qpos.shape[1] < 2:
        raise ValueError(f"Expected qpos with shape [T, >=2], got {qpos.shape}.")
    qpos_2d = np.asarray(qpos[:, :2], dtype=np.float64)
    theta1 = qpos_2d[:, 0]
    theta2 = qpos_2d[:, 1]
    link1 = float(obstacle_spec["link1"])
    link2 = float(obstacle_spec["link2"])
    base_xy = np.asarray(obstacle_spec["base_xy"], dtype=np.float64)
    elbow_local = np.stack((link1 * np.cos(theta1), link1 * np.sin(theta1)), axis=1)
    tip_local = elbow_local + np.stack(
        (link2 * np.cos(theta1 + theta2), link2 * np.sin(theta1 + theta2)),
        axis=1,
    )
    tip_xy = base_xy[None, :] + tip_local
    distances = np.linalg.norm(tip_xy - np.asarray(obstacle_spec["circle_center_xy"], dtype=np.float64)[None, :], axis=1)
    hits = distances <= float(obstacle_spec["circle_radius"])
    return hits.astype(bool), distances.astype(np.float32), tip_xy.astype(np.float32)


@torch.no_grad()
def encode_and_score_frames(
    pixels_ds: h5py.Dataset,
    *,
    world_model: torch.nn.Module,
    obstacle_model: ObstacleMLP,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = int(pixels_ds.shape[0])
    scores = np.empty((total_frames,), dtype=np.float32)
    hits = np.empty((total_frames,), dtype=bool)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for start in range(0, total_frames, frame_batch_size):
        stop = min(start + frame_batch_size, total_frames)
        batch_pixels = np.asarray(pixels_ds[start:stop], dtype=np.uint8)
        batch = preprocess_pixels(
            batch_pixels,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        output = world_model.encoder(batch, interpolate_pos_encoding=True)
        latents = world_model.projector(output.last_hidden_state[:, 0])[:, :embed_dim]
        normalized = (latents - feature_mean) / feature_std
        batch_scores = obstacle_model(normalized).detach().cpu().numpy().astype(np.float32)
        scores[start:stop] = batch_scores
        hits[start:stop] = batch_scores <= float(threshold)
        if start == 0 or stop == total_frames or ((start // frame_batch_size) + 1) % 100 == 0:
            log_progress(f"Scored frames {stop}/{total_frames}.")
    return scores, hits


def summarize_flag_mask(flag_mask: np.ndarray, ep_offset: np.ndarray, ep_len: np.ndarray) -> dict[str, Any]:
    flagged_episode_lengths: list[int] = []
    flagged_step_counts: list[int] = []
    flagged_episodes = 0
    for offset, length in zip(ep_offset.tolist(), ep_len.tolist(), strict=True):
        start = int(offset)
        stop = start + int(length)
        episode_count = int(np.count_nonzero(flag_mask[start:stop]))
        if episode_count > 0:
            flagged_episodes += 1
            flagged_episode_lengths.append(int(length))
            flagged_step_counts.append(episode_count)
    total_episodes = int(ep_len.shape[0])
    return {
        "total_episodes": total_episodes,
        "flagged_episodes": int(flagged_episodes),
        "flagged_fraction": float(flagged_episodes / total_episodes) if total_episodes else 0.0,
        "total_frames": int(flag_mask.shape[0]),
        "flagged_frames": int(np.count_nonzero(flag_mask)),
        "flagged_frame_fraction": float(np.mean(flag_mask)) if flag_mask.size else 0.0,
        "mean_flagged_steps_per_flagged_episode": float(np.mean(flagged_step_counts)) if flagged_step_counts else 0.0,
        "median_flagged_steps_per_flagged_episode": float(np.median(flagged_step_counts)) if flagged_step_counts else 0.0,
        "mean_length_of_flagged_episode": float(np.mean(flagged_episode_lengths)) if flagged_episode_lengths else 0.0,
    }


def episode_metrics(
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    classifier_hit_mask: np.ndarray,
    classifier_score_array: np.ndarray,
    analytic_hit_mask: np.ndarray,
    analytic_distance_array: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_rows: list[dict[str, Any]] = []
    for episode_idx, (offset, length) in enumerate(zip(ep_offset.tolist(), ep_len.tolist(), strict=True)):
        start = int(offset)
        stop = start + int(length)
        classifier_hits = classifier_hit_mask[start:stop]
        classifier_scores = classifier_score_array[start:stop]
        analytic_hits = analytic_hit_mask[start:stop]
        analytic_distances = analytic_distance_array[start:stop]
        classifier_steps = np.flatnonzero(classifier_hits).astype(np.int64)
        analytic_steps = np.flatnonzero(analytic_hits).astype(np.int64)
        either_hits = classifier_hits | analytic_hits
        both_hits = classifier_hits & analytic_hits
        either_steps = np.flatnonzero(either_hits).astype(np.int64)
        episode_rows.append(
            {
                "episode_idx": int(episode_idx),
                "offset": start,
                "length": int(length),
                "classifier_flagged": bool(classifier_steps.shape[0] > 0),
                "analytic_flagged": bool(analytic_steps.shape[0] > 0),
                "either_flagged": bool(either_steps.shape[0] > 0),
                "both_flagged_same_step": bool(np.any(both_hits)),
                "num_classifier_flagged_steps": int(classifier_steps.shape[0]),
                "num_analytic_flagged_steps": int(analytic_steps.shape[0]),
                "num_union_flagged_steps": int(either_steps.shape[0]),
                "first_classifier_flagged_step": int(classifier_steps[0]) if classifier_steps.size else None,
                "first_analytic_flagged_step": int(analytic_steps[0]) if analytic_steps.size else None,
                "first_union_flagged_step": int(either_steps[0]) if either_steps.size else None,
                "min_classifier_score": float(np.min(classifier_scores)),
                "mean_classifier_score": float(np.mean(classifier_scores)),
                "min_analytic_distance": float(np.min(analytic_distances)),
                "mean_analytic_distance": float(np.mean(analytic_distances)),
            }
        )

    classifier_only_mask = classifier_hit_mask & ~analytic_hit_mask
    analytic_only_mask = analytic_hit_mask & ~classifier_hit_mask
    overlap_mask = classifier_hit_mask & analytic_hit_mask
    union_mask = classifier_hit_mask | analytic_hit_mask
    summary = {
        "classifier": summarize_flag_mask(classifier_hit_mask, ep_offset, ep_len),
        "analytic": summarize_flag_mask(analytic_hit_mask, ep_offset, ep_len),
        "classifier_only": summarize_flag_mask(classifier_only_mask, ep_offset, ep_len),
        "analytic_only": summarize_flag_mask(analytic_only_mask, ep_offset, ep_len),
        "overlap": summarize_flag_mask(overlap_mask, ep_offset, ep_len),
        "union": summarize_flag_mask(union_mask, ep_offset, ep_len),
    }
    return episode_rows, summary


def save_rollout_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        output_path,
        [np.ascontiguousarray(frame) for frame in frames],
        fps=fps,
        quality=8,
        macro_block_size=1,
    )


def save_flagged_rollouts(
    h5: h5py.File,
    episode_rows: list[dict[str, Any]],
    *,
    classifier_score_array: np.ndarray,
    classifier_hit_mask: np.ndarray,
    analytic_distance_array: np.ndarray,
    analytic_hit_mask: np.ndarray,
    tip_xy_array: np.ndarray,
    out_dir: Path,
    save_limit: int,
    fps: float,
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    flagged_rows = [row for row in episode_rows if row["either_flagged"]][:save_limit]
    for rank, row in enumerate(flagged_rows):
        episode_idx = int(row["episode_idx"])
        offset = int(row["offset"])
        length = int(row["length"])
        stop = offset + length
        episode_dir = out_dir / f"episode_{episode_idx:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        pixels = np.asarray(h5["pixels"][offset:stop], dtype=np.uint8)
        observations = np.asarray(h5["observation"][offset:stop], dtype=np.float32)
        qpos = np.asarray(h5["qpos"][offset:stop], dtype=np.float32) if "qpos" in h5 else None
        qvel = np.asarray(h5["qvel"][offset:stop], dtype=np.float32) if "qvel" in h5 else None
        actions = np.asarray(h5["action"][offset:stop], dtype=np.float32)
        rewards = np.asarray(h5["reward"][offset:stop], dtype=np.float32) if "reward" in h5 else None
        classifier_scores = np.asarray(classifier_score_array[offset:stop], dtype=np.float32)
        classifier_hits = np.asarray(classifier_hit_mask[offset:stop], dtype=bool)
        analytic_distances = np.asarray(analytic_distance_array[offset:stop], dtype=np.float32)
        analytic_hits = np.asarray(analytic_hit_mask[offset:stop], dtype=bool)
        tip_xy = np.asarray(tip_xy_array[offset:stop], dtype=np.float32)

        save_rollout_video(pixels, episode_dir / "rollout.mp4", fps=fps)
        np.savez_compressed(
            episode_dir / "rollout_data.npz",
            pixels=pixels,
            observation=observations,
            qpos=qpos,
            qvel=qvel,
            action=actions,
            reward=rewards,
            classifier_score=classifier_scores,
            classifier_hit=classifier_hits,
            analytic_distance=analytic_distances,
            analytic_hit=analytic_hits,
            tip_xy=tip_xy,
        )
        metadata = {
            "rank": int(rank),
            **row,
            "saved_video_path": str((episode_dir / "rollout.mp4").resolve()),
            "saved_npz_path": str((episode_dir / "rollout_data.npz").resolve()),
        }
        save_json(episode_dir / "metadata.json", metadata)
        saved.append(metadata)
    return saved


def main() -> None:
    args = parse_args()
    if args.frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")
    if args.save_limit < 0:
        raise ValueError("--save-limit must be non-negative.")

    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    obs_net_dir = args.obs_net_dir.expanduser().resolve()
    obs_model_path = (
        args.obs_model_path.expanduser().resolve()
        if args.obs_model_path is not None
        else discover_latest_obs_model(obs_net_dir).resolve()
    )
    obstacle_summary_path = args.obstacle_summary_path.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = require_device(args.device)

    config = load_config(model_dir)
    embed_dim = int(config.get("embed_dim", 5))
    img_size = int(config.get("img_size", 224))
    obstacle_spec = load_obstacle_spec(obstacle_summary_path)

    log_progress(f"Loading world model checkpoint {checkpoint_path}.")
    world_model = load_world_model(checkpoint_path, device)
    log_progress(f"Loading obstacle classifier {obs_model_path}.")
    obstacle_model, obstacle_payload = load_obstacle_classifier(obs_model_path, device)
    feature_mean = torch.as_tensor(obstacle_payload["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(obstacle_payload["feature_std"], dtype=torch.float32, device=device)
    threshold = float(obstacle_payload.get("conformal_safe_score_threshold", obstacle_payload.get("base_decision_threshold", 0.0)))

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        qpos = np.asarray(h5["qpos"][:], dtype=np.float32)
        if ep_len.ndim != 1 or ep_offset.ndim != 1 or ep_len.shape != ep_offset.shape:
            raise ValueError(f"Expected ep_len and ep_offset to be 1D arrays with matching shape, got {ep_len.shape} and {ep_offset.shape}.")

        log_progress(f"Scoring {int(h5['pixels'].shape[0])} frames from {len(ep_len)} episodes.")
        classifier_score_array, classifier_hit_mask = encode_and_score_frames(
            h5["pixels"],
            world_model=world_model,
            obstacle_model=obstacle_model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            device=device,
            img_size=img_size,
            embed_dim=embed_dim,
            frame_batch_size=int(args.frame_batch_size),
            threshold=threshold,
        )
        analytic_hit_mask, analytic_distance_array, tip_xy_array = analytic_obstacle_hits(qpos, obstacle_spec)
        episode_rows, summary = episode_metrics(
            ep_offset,
            ep_len,
            classifier_hit_mask,
            classifier_score_array,
            analytic_hit_mask,
            analytic_distance_array,
        )
        saved_rollouts = save_flagged_rollouts(
            h5,
            episode_rows,
            classifier_score_array=classifier_score_array,
            classifier_hit_mask=classifier_hit_mask,
            analytic_distance_array=analytic_distance_array,
            analytic_hit_mask=analytic_hit_mask,
            tip_xy_array=tip_xy_array,
            out_dir=out_dir / "flagged_rollouts",
            save_limit=int(args.save_limit),
            fps=float(args.video_fps) if args.video_fps is not None else float(h5.attrs.get("video_fps", 50.0)),
        )

    summary_payload = {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "obs_model_path": str(obs_model_path),
        "obstacle_summary_path": str(obstacle_summary_path),
        "decision_threshold": threshold,
        "frame_batch_size": int(args.frame_batch_size),
        "save_limit": int(args.save_limit),
        "obstacle_spec": jsonable(
            {
                "circle_center_xy": obstacle_spec["circle_center_xy"],
                "circle_radius": obstacle_spec["circle_radius"],
                "link1": obstacle_spec["link1"],
                "link2": obstacle_spec["link2"],
                "base_xy": obstacle_spec["base_xy"],
                "center_qpos": obstacle_spec["center_qpos"],
            }
        ),
        "summary": summary,
        "saved_rollouts": saved_rollouts,
        "first_25_classifier_flagged_episodes": [row for row in episode_rows if row["classifier_flagged"]][:25],
        "first_25_analytic_flagged_episodes": [row for row in episode_rows if row["analytic_flagged"]][:25],
        "first_25_disagreement_episodes": [row for row in episode_rows if row["classifier_flagged"] != row["analytic_flagged"]][:25],
        "score_sign_convention": jsonable(obstacle_payload.get("score_sign_convention", {})),
    }
    save_json(out_dir / "summary.json", summary_payload)

    print(json.dumps(summary_payload["summary"], indent=2))
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
