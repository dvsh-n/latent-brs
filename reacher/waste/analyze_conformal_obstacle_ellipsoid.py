#!/usr/bin/env python3
"""Analyze a local conformalized ellipsoidal obstacle for the Reacher planner."""

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
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_1"
DEFAULT_OUT_DIR = "reacher/plan/conformal_obstacle_analysis_ellipsoid"
DEFAULT_EPISODE_IDX = 829
DEFAULT_HORIZON = 20
DEFAULT_OBSTACLE_STEP = -1
DEFAULT_MAX_MPC_STEPS = 100
DEFAULT_Q_TERMINAL = 10.0
DEFAULT_Q_STAGE = 0.005
DEFAULT_R_CONTROL = 0.1
DEFAULT_VIDEO_FPS = 60
DEFAULT_OVERLAY_SAMPLE_COUNT = 96
DEFAULT_OVERLAY_PERTURB_ALPHA = 0.05


def load_runtime_dependencies():
    from reacher.plan import plan_ilqr_mpc as planner
    from reacher.train.mlpdyn_train import LeWMReacherDataset

    return planner, LeWMReacherDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-idx", type=int, default=DEFAULT_EPISODE_IDX)
    parser.add_argument("--obstacle-step", type=int, default=DEFAULT_OBSTACLE_STEP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=DEFAULT_MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=DEFAULT_Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=DEFAULT_Q_STAGE)
    parser.add_argument("--r-control", type=float, default=DEFAULT_R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--joint1-range", type=float, default=0.25)
    parser.add_argument("--joint2-range", type=float, default=0.12)
    parser.add_argument("--set-pca-count", type=int, default=192)
    parser.add_argument("--set-norm-count", type=int, default=192)
    parser.add_argument("--set-cal-count", type=int, default=192)
    parser.add_argument("--set-test-count", type=int, default=512)
    parser.add_argument("--background-samples", type=int, default=4000)
    parser.add_argument("--context-background-count", type=int, default=512)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--conformal-eps", type=float, default=1e-12)
    parser.add_argument("--eigval-floor", type=float, default=1e-10)
    parser.add_argument("--membership-tol", type=float, default=1e-8)
    parser.add_argument("--overlay-sample-count", type=int, default=DEFAULT_OVERLAY_SAMPLE_COUNT)
    parser.add_argument("--overlay-perturb-alpha", type=float, default=DEFAULT_OVERLAY_PERTURB_ALPHA)
    parser.add_argument("--force-rerun-rollout", action="store_true", default=False)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def log_progress(message: str) -> None:
    print(f"[analyze_conformal_obstacle] {message}", flush=True)


def infer_rollout_cache_path(
    out_dir: Path,
    checkpoint_path: Path,
    episode_idx: int,
    args: argparse.Namespace,
) -> Path:
    stem = checkpoint_path.stem
    name = (
        f"episode_{episode_idx:05d}_{stem}_h{args.horizon}_"
        f"mpc{args.max_mpc_steps}_qt{args.q_terminal:g}_qs{args.q_stage:g}_rc{args.r_control:g}.pt"
    )
    return out_dir / "rollout_cache" / name


def joint_limits_from_env(env) -> tuple[np.ndarray, np.ndarray]:
    model = env._env.physics.model
    joint_ranges = np.asarray(model.jnt_range[:2], dtype=np.float64)
    joint_limited = np.asarray(model.jnt_limited[:2], dtype=bool)
    lower = joint_ranges[:, 0]
    upper = joint_ranges[:, 1]
    valid = joint_limited & np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    lower = np.where(valid, lower, -np.pi)
    upper = np.where(valid, upper, np.pi)
    return lower, upper


def reconstruct_rollout_frames(
    planner,
    *,
    rollout: dict[str, Any],
) -> list[np.ndarray]:
    env = planner.make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    frames: list[np.ndarray] = []
    qpos_seq = np.asarray(rollout["rollout_qpos"], dtype=np.float32)
    qvel_seq = np.asarray(rollout["rollout_qvel"], dtype=np.float32)
    for qpos, qvel in zip(qpos_seq, qvel_seq, strict=True):
        frame = planner.reset_env_to_state(
            env,
            seed=int(rollout["episode_seed"]),
            qpos=qpos,
            qvel=qvel,
            height=int(rollout["height"]),
            width=int(rollout["width"]),
        )
        frames.append(frame.copy())
    env.close()
    return frames


def run_or_load_rollout(
    *,
    planner,
    dataset_cls,
    cache_path: Path,
    force_rerun: bool,
    model: torch.nn.Module,
    config: dict[str, Any],
    dataset_path: Path,
    episode_idx: int,
    device: torch.device,
    frame_batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if cache_path.is_file() and not force_rerun:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 24))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for this planner, got {history_size}.")

    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = dataset_cls(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

    episode = planner.load_dataset_episode(dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)
    dataset_pixels = planner.preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    dataset_latents = planner.encode_frames(
        model,
        dataset_pixels,
        device=device,
        frame_batch_size=frame_batch_size,
    ).detach().cpu()

    goal_emb = dataset_latents[-1].to(device)
    goal_state = planner.make_markov_state(goal_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)
    obs_dim = int(goal_obs.shape[0])

    env = planner.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    current_frame = planner.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        height=height,
        width=width,
    )

    dynamics = planner.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    mpc_solver = planner.ILQRMPCSolver(
        dynamics,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        device=device,
    )

    current_emb = planner.encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = planner.make_markov_state(current_emb)
    current_obs = planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)

    rollout_frames = [current_frame.copy()]
    rollout_qpos = [np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy()]
    rollout_qvel = [np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy()]
    rollout_emb = [current_emb.detach().cpu().numpy().astype(np.float32)]
    rollout_markov = [current_state.detach().cpu().numpy().astype(np.float32)]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    observation_goal_distances = [planner.compute_observation_goal_distance(current_obs, goal_obs)]
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    stop_reason = "max_mpc_steps"

    for _ in tqdm(range(args.max_mpc_steps), desc="MPC rollout"):
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, _, _, _ = mpc_solver.solve(current_state_np, goal_state_np)
        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = planner.normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = planner.encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = planner.make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        rollout_qpos.append(np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy())
        rollout_qvel.append(np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy())
        rollout_emb.append(current_emb.detach().cpu().numpy().astype(np.float32))
        rollout_markov.append(current_state.detach().cpu().numpy().astype(np.float32))
        observation_goal_distances.append(planner.compute_observation_goal_distance(current_obs, goal_obs))
        latent_goal_distances.append(float(torch.linalg.vector_norm(current_state - goal_state).item()))

        reached_goal, _ = planner.goal_reached(current_obs, goal_obs)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    env.close()

    payload = {
        "episode_idx": int(episode_idx),
        "episode_seed": int(episode_seed),
        "physics_freq_hz": float(physics_freq_hz),
        "time_limit": float(time_limit),
        "width": int(width),
        "height": int(height),
        "goal_obs": np.asarray(goal_obs, dtype=np.float32),
        "goal_qpos": np.asarray(qpos_np[-1], dtype=np.float32),
        "dataset_qpos": np.asarray(qpos_np, dtype=np.float32),
        "dataset_qvel": np.asarray(qvel_np, dtype=np.float32),
        "dataset_emb": dataset_latents.numpy().astype(np.float32),
        "dataset_start_pixel": np.asarray(pixels_np[0], dtype=np.uint8),
        "dataset_goal_pixel": np.asarray(pixels_np[-1], dtype=np.uint8),
        "dataset_pixels_shape": tuple(pixels_np.shape),
        "rollout_frames": np.asarray(rollout_frames, dtype=np.uint8),
        "rollout_qpos": np.asarray(rollout_qpos, dtype=np.float32),
        "rollout_qvel": np.asarray(rollout_qvel, dtype=np.float32),
        "rollout_emb": np.asarray(rollout_emb, dtype=np.float32),
        "rollout_markov": np.asarray(rollout_markov, dtype=np.float32),
        "executed_actions_raw": np.asarray(executed_actions_raw, dtype=np.float32),
        "executed_actions_norm": np.asarray(executed_actions_norm, dtype=np.float32),
        "observation_goal_distances": np.asarray(observation_goal_distances, dtype=np.float32),
        "latent_goal_distances": np.asarray(latent_goal_distances, dtype=np.float32),
        "stop_reason": stop_reason,
        "planner_args": {
            "horizon": int(args.horizon),
            "max_mpc_steps": int(args.max_mpc_steps),
            "q_terminal": float(args.q_terminal),
            "q_stage": float(args.q_stage),
            "r_control": float(args.r_control),
            "ilqr_max_iters": int(args.ilqr_max_iters),
            "ilqr_tol": float(args.ilqr_tol),
            "ilqr_regularization": float(args.ilqr_regularization),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def sample_background_rows(dataset_path: Path, rng: np.random.Generator, count: int) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    valid_eps = np.flatnonzero(ep_len > 0)
    if valid_eps.size == 0:
        raise ValueError(f"No non-empty episodes in {dataset_path}.")
    ep_choices = rng.choice(valid_eps, size=count, replace=True)
    time_choices = np.array([rng.integers(0, int(ep_len[ep])) for ep in ep_choices], dtype=np.int64)
    return ep_offset[ep_choices] + time_choices


def encode_dataset_rows(
    planner,
    model: torch.nn.Module,
    dataset_path: Path,
    rows: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    frame_batch_size: int,
) -> np.ndarray:
    rows_sorted_unique, inverse = np.unique(rows, return_inverse=True)
    with h5py.File(dataset_path, "r") as h5:
        pixels_np = np.asarray(h5["pixels"][rows_sorted_unique], dtype=np.uint8)
    pixels = planner.preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    latents = planner.encode_frames(model, pixels, device=device, frame_batch_size=frame_batch_size)
    latents_np = latents.detach().cpu().numpy().astype(np.float64)
    return latents_np[inverse]


def sample_local_perturbations(
    center_qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    *,
    joint_ranges: np.ndarray,
    count: int,
) -> np.ndarray:
    noise = rng.uniform(-joint_ranges, joint_ranges, size=(count, center_qpos.shape[0]))
    return np.clip(center_qpos[None, :] + noise, lower[None, :], upper[None, :]).astype(np.float64)


def render_qpos_batch(
    planner,
    env,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    frames = []
    for qpos in tqdm(qpos_batch, desc="Render perturbations"):
        frame = planner.reset_env_to_state(
            env,
            seed=seed,
            qpos=np.asarray(qpos, dtype=np.float32),
            qvel=qvel,
            height=height,
            width=width,
        )
        frames.append(frame.copy())
    return np.stack(frames, axis=0)


def get_arm_geom_ids(model) -> np.ndarray:
    arm_body_names = ("arm", "hand", "finger")
    arm_body_ids = {int(model.name2id(name, "body")) for name in arm_body_names}
    arm_geom_ids: list[int] = []
    for geom_id in range(int(model.ngeom)):
        geom_name = model.id2name(geom_id, "geom")
        geom_body_id = int(model.geom_bodyid[geom_id])
        if geom_name == "root" or geom_body_id in arm_body_ids:
            arm_geom_ids.append(geom_id)
    if not arm_geom_ids:
        raise ValueError("Failed to identify arm geoms for segmentation.")
    return np.asarray(sorted(set(arm_geom_ids)), dtype=np.int32)


def build_arm_mask(segmentation: np.ndarray, arm_geom_ids: np.ndarray) -> np.ndarray:
    mask = np.zeros(segmentation.shape[:2], dtype=bool)
    for geom_id in arm_geom_ids:
        mask |= segmentation[..., 0] == geom_id
    return mask


def make_segmentation_scene_option(model):
    from dm_control.mujoco.wrapper import core as dm_core

    target_geom_id = int(model.name2id("target", "geom"))
    original_group = int(model.geom_group[target_geom_id])
    model.geom_group[target_geom_id] = 3
    scene_option = dm_core.MjvOption()
    scene_option.geomgroup[:] = 1
    scene_option.geomgroup[3] = 0
    return scene_option, target_geom_id, original_group


def render_masked_qpos_batch(
    env,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    physics = env._env.physics
    model = physics.model
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    arm_geom_ids = get_arm_geom_ids(model)
    scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    try:
        for qpos in tqdm(qpos_batch, desc="Render obstacle masks"):
            with physics.reset_context():
                physics.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float32)
                physics.data.qvel[: qvel.shape[0]] = qvel
            frame = physics.render(height=height, width=width, camera_id=0)
            segmentation = physics.render(
                height=height,
                width=width,
                camera_id=0,
                segmentation=True,
                scene_option=scene_option,
            )
            frames.append(frame.copy())
            masks.append(build_arm_mask(segmentation, arm_geom_ids))
    finally:
        model.geom_group[target_geom_id] = original_group
    return np.stack(frames, axis=0), np.stack(masks, axis=0)


def alpha_composite_masked(
    canvas: np.ndarray,
    frame: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    out = canvas.copy()
    if not np.any(mask):
        return out
    base = out[mask].astype(np.float32)
    src = frame[mask].astype(np.float32)
    out[mask] = np.clip((1.0 - alpha) * base + alpha * src, 0.0, 255.0).astype(np.uint8)
    return out


def make_obstacle_overlay_image(
    nominal_frame: np.ndarray,
    nominal_mask: np.ndarray,
    perturb_frames: np.ndarray,
    perturb_masks: np.ndarray,
    *,
    perturb_alpha: float,
) -> np.ndarray:
    canvas = np.full_like(nominal_frame, 255, dtype=np.uint8)
    for frame, mask in zip(perturb_frames, perturb_masks, strict=True):
        canvas = alpha_composite_masked(canvas, frame, mask, alpha=perturb_alpha)
    canvas = alpha_composite_masked(canvas, nominal_frame, nominal_mask, alpha=1.0)
    return canvas


def fit_pca_2d(points: np.ndarray) -> dict[str, np.ndarray]:
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples for PCA, got shape {points.shape}.")
    mean = points.mean(axis=0)
    centered = points - mean
    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    if components.shape[0] < 2:
        pad = np.zeros((2 - components.shape[0], points.shape[1]), dtype=np.float64)
        components = np.concatenate((components, pad), axis=0)
    denom = max(points.shape[0] - 1, 1)
    variances = (svals**2) / denom
    total_var = float(np.sum(variances))
    explained = variances[:2] / total_var if total_var > 0.0 else np.zeros(2, dtype=np.float64)
    return {
        "mean": mean.astype(np.float64),
        "components": components.astype(np.float64),
        "explained_variance_ratio": explained.astype(np.float64),
    }


def project_points(points: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    return (points - pca["mean"]) @ pca["components"].T


def ellipsoid_coords(points: np.ndarray, ellipsoid: dict[str, np.ndarray | float]) -> np.ndarray:
    center = np.asarray(ellipsoid["center"], dtype=np.float64)
    right_pinv = np.asarray(ellipsoid["right_pinv"], dtype=np.float64)
    return (points - center) @ right_pinv


def select_nearest_points(points: np.ndarray, center: np.ndarray, count: int) -> np.ndarray:
    if points.ndim != 2 or center.ndim != 1 or points.shape[1] != center.shape[0]:
        raise ValueError(f"Shape mismatch: points {points.shape}, center {center.shape}.")
    if count <= 0:
        raise ValueError(f"Expected positive count, got {count}.")
    if points.shape[0] <= count:
        return points
    distances = np.linalg.norm(points - center[None, :], axis=1)
    order = np.argpartition(distances, count - 1)[:count]
    return points[order]


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    n = int(scores.shape[0])
    augmented = np.concatenate((np.sort(scores.astype(np.float64)), np.array([np.inf], dtype=np.float64)))
    rank = int(np.ceil((n + 1) * (1.0 - delta))) - 1
    rank = int(np.clip(rank, 0, augmented.shape[0] - 1))
    return float(augmented[rank])


def build_conformal_ellipsoid(
    pca_samples: np.ndarray,
    norm_samples: np.ndarray,
    cal_samples: np.ndarray,
    *,
    delta: float,
    eps: float,
    eigval_floor: float,
) -> dict[str, np.ndarray | float]:
    center = pca_samples.mean(axis=0)
    centered = pca_samples - center
    if pca_samples.shape[0] < 2:
        raise ValueError("Need at least two samples to estimate covariance.")
    cov = centered.T @ centered / float(pca_samples.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], eigval_floor)
    eigvecs = eigvecs[:, order]
    sqrt_eigs = np.sqrt(eigvals)
    inv_sqrt_eigs = 1.0 / sqrt_eigs

    base_axes = sqrt_eigs
    base_right_pinv = eigvecs @ np.diag(inv_sqrt_eigs)

    def whitened(points: np.ndarray) -> np.ndarray:
        return (points - center) @ base_right_pinv

    norm_coords = whitened(norm_samples)
    norm_excess = np.maximum(0.0, np.abs(norm_coords) - 1.0)
    direction_inflation = 1.0 + norm_excess.max(axis=0)

    cal_coords = whitened(cal_samples) / direction_inflation[None, :]
    cal_gauge = np.linalg.norm(cal_coords, axis=1)
    scores = np.maximum(0.0, cal_gauge - 1.0)
    conformal_radius = conformal_quantile(scores, delta=delta)

    inflation = direction_inflation * (1.0 + conformal_radius)
    semi_axes = base_axes * inflation
    generators = eigvecs @ np.diag(semi_axes)
    right_pinv = eigvecs @ np.diag(1.0 / np.maximum(semi_axes, eps))

    return {
        "center": center.astype(np.float64),
        "covariance": cov.astype(np.float64),
        "eigvals": eigvals.astype(np.float64),
        "eigvecs": eigvecs.astype(np.float64),
        "base_axes": base_axes.astype(np.float64),
        "direction_inflation": direction_inflation.astype(np.float64),
        "inflation": inflation.astype(np.float64),
        "semi_axes": semi_axes.astype(np.float64),
        "generators": generators.astype(np.float64),
        "conformal_radius": float(conformal_radius),
        "right_pinv": right_pinv.astype(np.float64),
    }


def ellipsoid_contains(points: np.ndarray, ellipsoid: dict[str, np.ndarray | float], tol: float) -> np.ndarray:
    center = np.asarray(ellipsoid["center"], dtype=np.float64)
    right_pinv = np.asarray(ellipsoid["right_pinv"], dtype=np.float64)
    coords = (points - center) @ right_pinv
    return np.linalg.norm(coords, axis=1) <= (1.0 + tol)


def project_ellipsoid(
    ellipsoid: dict[str, np.ndarray | float],
    pca: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(ellipsoid["center"], dtype=np.float64)
    generators = np.asarray(ellipsoid["generators"], dtype=np.float64)
    center_2d = (center - pca["mean"]) @ pca["components"].T
    projected_map = pca["components"] @ generators
    shape_2d = projected_map @ projected_map.T
    return center_2d, shape_2d.astype(np.float64)


def ellipsoid_boundary_2d(center: np.ndarray, shape_2d: np.ndarray, num_points: int = 256) -> np.ndarray:
    if shape_2d.shape != (2, 2):
        raise ValueError(f"Expected 2x2 ellipsoid shape matrix, got {shape_2d.shape}.")
    eigvals, eigvecs = np.linalg.eigh(shape_2d)
    radii = np.sqrt(np.maximum(eigvals, 0.0))
    angles = np.linspace(0.0, 2.0 * np.pi, num=num_points, endpoint=True)
    unit_circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    boundary = center[None, :] + unit_circle @ (eigvecs @ np.diag(radii)).T
    return boundary.astype(np.float64)


def make_plot(
    out_path: Path,
    *,
    title: str,
    background: np.ndarray,
    focus_background: np.ndarray | None,
    construction_samples: np.ndarray,
    test_inside_samples: np.ndarray,
    test_outside_samples: np.ndarray,
    dataset_traj: np.ndarray,
    rollout_traj: np.ndarray,
    start_point: np.ndarray,
    dataset_goal_point: np.ndarray,
    rollout_goal_point: np.ndarray,
    nominal_point: np.ndarray,
    ellipsoid_boundary: np.ndarray,
    explained_ratio: np.ndarray,
    add_zoom_inset: bool,
    focus_main_view: bool = False,
) -> None:
    if add_zoom_inset:
        fig, (ax, zoom_ax) = plt.subplots(
            1,
            2,
            figsize=(13, 6),
            gridspec_kw={"width_ratios": [3.2, 1.4]},
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 8))
        zoom_ax = None
    ax.scatter(background[:, 0], background[:, 1], s=3, alpha=0.18, color="0.5", label="background", zorder=1)
    if focus_background is not None and focus_background.shape[0] > 0:
        ax.scatter(
            focus_background[:, 0],
            focus_background[:, 1],
            s=7,
            alpha=0.28,
            color="0.45",
            label="nearby background",
            zorder=2,
        )
    if construction_samples.shape[0] > 0:
        ax.scatter(
            construction_samples[:, 0],
            construction_samples[:, 1],
            s=10,
            alpha=0.35,
            color="tab:red",
            label="obstacle samples",
            zorder=3,
        )
    if test_inside_samples.shape[0] > 0:
        ax.scatter(
            test_inside_samples[:, 0],
            test_inside_samples[:, 1],
            s=18,
            alpha=0.7,
            color="tab:orange",
            label="held-out inside",
            zorder=6,
        )
    if test_outside_samples.shape[0] > 0:
        ax.scatter(
            test_outside_samples[:, 0],
            test_outside_samples[:, 1],
            s=28,
            alpha=0.95,
            marker="x",
            linewidths=1.2,
            color="black",
            label="held-out outside",
            zorder=7,
        )
    if ellipsoid_boundary.shape[0] >= 2:
        ax.plot(ellipsoid_boundary[:, 0], ellipsoid_boundary[:, 1], color="tab:orange", linewidth=2.0, label="ellipsoid boundary", zorder=10)
    ax.plot(dataset_traj[:, 0], dataset_traj[:, 1], color="tab:blue", linewidth=1.6, label="dataset trajectory", zorder=4)
    ax.plot(rollout_traj[:, 0], rollout_traj[:, 1], color="tab:green", linewidth=1.6, label="executed MPC rollout", zorder=5)
    ax.scatter(start_point[0], start_point[1], s=90, marker="o", color="black", label="start", zorder=11)
    ax.scatter(
        dataset_goal_point[0],
        dataset_goal_point[1],
        s=120,
        marker="*",
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        label="dataset goal",
        zorder=11,
    )
    ax.scatter(
        rollout_goal_point[0],
        rollout_goal_point[1],
        s=95,
        marker="P",
        color="limegreen",
        edgecolor="black",
        linewidth=0.8,
        label="executed goal",
        zorder=11,
    )
    ax.scatter(nominal_point[0], nominal_point[1], s=80, marker="X", color="tab:red", edgecolor="black", linewidth=0.8, label="obstacle center", zorder=9)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({100.0 * explained_ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100.0 * explained_ratio[1]:.1f}%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    if focus_main_view:
        focus_items = [
            construction_samples,
            test_inside_samples,
            test_outside_samples,
            dataset_traj,
            rollout_traj,
            start_point[None, :],
            dataset_goal_point[None, :],
            rollout_goal_point[None, :],
            nominal_point[None, :],
        ]
        if focus_background is not None and focus_background.shape[0] > 0:
            focus_items.append(focus_background)
        if ellipsoid_boundary.shape[0] >= 2:
            focus_items.append(ellipsoid_boundary)
        focus_stack = np.concatenate(focus_items, axis=0)
        mins = focus_stack.min(axis=0)
        maxs = focus_stack.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.4 * spans
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])

    if add_zoom_inset:
        zoom_items = [nominal_point[None, :]]
        if construction_samples.shape[0] > 0:
            zoom_items.append(construction_samples)
        if test_inside_samples.shape[0] > 0:
            zoom_items.append(test_inside_samples)
        if test_outside_samples.shape[0] > 0:
            zoom_items.append(test_outside_samples)
        if ellipsoid_boundary.shape[0] >= 2:
            zoom_items.append(ellipsoid_boundary)
        zoom_stack = np.concatenate(zoom_items, axis=0)
        mins = zoom_stack.min(axis=0)
        maxs = zoom_stack.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.35 * spans
        x0, y0 = mins - pad
        x1, y1 = maxs + pad

        zoom_ax.scatter(background[:, 0], background[:, 1], s=2, alpha=0.14, color="0.56", zorder=1)
        if focus_background is not None and focus_background.shape[0] > 0:
            zoom_ax.scatter(focus_background[:, 0], focus_background[:, 1], s=5, alpha=0.18, color="0.5", zorder=2)
        if construction_samples.shape[0] > 0:
            zoom_ax.scatter(construction_samples[:, 0], construction_samples[:, 1], s=12, alpha=0.35, color="tab:red", zorder=3)
        if test_inside_samples.shape[0] > 0:
            zoom_ax.scatter(test_inside_samples[:, 0], test_inside_samples[:, 1], s=18, alpha=0.75, color="tab:orange", zorder=6)
        if test_outside_samples.shape[0] > 0:
            zoom_ax.scatter(
                test_outside_samples[:, 0],
                test_outside_samples[:, 1],
                s=26,
                alpha=0.95,
                marker="x",
                linewidths=1.2,
                color="black",
                zorder=7,
            )
        if ellipsoid_boundary.shape[0] >= 2:
            zoom_ax.plot(ellipsoid_boundary[:, 0], ellipsoid_boundary[:, 1], color="tab:orange", linewidth=2.0, zorder=10)
        zoom_ax.scatter(
            nominal_point[0],
            nominal_point[1],
            s=80,
            marker="X",
            color="tab:red",
            edgecolor="black",
            linewidth=0.8,
            zorder=9,
        )
        zoom_ax.set_xlim(x0, x1)
        zoom_ax.set_ylim(y0, y1)
        zoom_ax.set_title("Zoom")
        zoom_ax.set_xlabel(f"PC1 ({100.0 * explained_ratio[0]:.1f}%)")
        zoom_ax.set_ylabel(f"PC2 ({100.0 * explained_ratio[1]:.1f}%)")
        zoom_ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_ellipsoid_coords_plot(
    out_path: Path,
    *,
    construction_coords: np.ndarray,
    test_coords: np.ndarray,
    test_inside_mask: np.ndarray,
    nominal_coords: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    angles = np.linspace(0.0, 2.0 * np.pi, num=256, endpoint=True)
    unit_circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    ax.fill(unit_circle[:, 0], unit_circle[:, 1], color="tab:orange", alpha=0.08, zorder=0)
    ax.plot(unit_circle[:, 0], unit_circle[:, 1], color="tab:orange", linewidth=2.0, label="unit ball in ellipsoid coords", zorder=5)
    if construction_coords.shape[0] > 0:
        ax.scatter(
            construction_coords[:, 0],
            construction_coords[:, 1],
            s=12,
            alpha=0.35,
            color="tab:red",
            label="obstacle samples",
            zorder=1,
        )
    inside_coords = test_coords[test_inside_mask]
    outside_coords = test_coords[~test_inside_mask]
    if inside_coords.shape[0] > 0:
        ax.scatter(inside_coords[:, 0], inside_coords[:, 1], s=18, alpha=0.7, color="tab:orange", label="held-out inside", zorder=2)
    if outside_coords.shape[0] > 0:
        ax.scatter(
            outside_coords[:, 0],
            outside_coords[:, 1],
            s=28,
            alpha=0.95,
            marker="x",
            linewidths=1.2,
            color="black",
            label="held-out outside",
            zorder=3,
        )
    ax.scatter(
        nominal_coords[0],
        nominal_coords[1],
        s=80,
        marker="X",
        color="tab:red",
        edgecolor="black",
        linewidth=0.8,
        label="obstacle center",
        zorder=6,
    )
    ax.set_xlabel("Ellipsoid coord 1")
    ax.set_ylabel("Ellipsoid coord 2")
    ax.set_title("Coverage view in ellipsoid coordinates")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_progress(
        f"Starting analysis for episode {args.episode_idx}, requested obstacle step {args.obstacle_step}."
    )
    rng = np.random.default_rng(args.seed)
    planner, dataset_cls = load_runtime_dependencies()
    device = planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else planner.latest_object_checkpoint(model_dir).resolve()
    )
    model = planner.load_model(checkpoint_path, device)

    background_dataset_path = (
        args.background_dataset_path.expanduser().resolve()
        if args.background_dataset_path is not None
        else Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    )

    rollout_cache_path = infer_rollout_cache_path(out_root, checkpoint_path, args.episode_idx, args)
    rollout = run_or_load_rollout(
        planner=planner,
        dataset_cls=dataset_cls,
        cache_path=rollout_cache_path,
        force_rerun=args.force_rerun_rollout,
        model=model,
        config=config,
        dataset_path=dataset_path,
        episode_idx=args.episode_idx,
        device=device,
        frame_batch_size=args.frame_batch_size,
        args=args,
    )
    log_progress("Loaded rollout cache and model outputs.")

    rollout_qpos = np.asarray(rollout["rollout_qpos"], dtype=np.float64)
    rollout_qvel = np.asarray(rollout["rollout_qvel"], dtype=np.float64)
    rollout_emb = np.asarray(rollout["rollout_emb"], dtype=np.float64)
    dataset_emb = np.asarray(rollout["dataset_emb"], dtype=np.float64)
    if "dataset_start_pixel" in rollout and "dataset_goal_pixel" in rollout:
        start_pixel = np.asarray(rollout["dataset_start_pixel"], dtype=np.uint8)
        goal_pixel = np.asarray(rollout["dataset_goal_pixel"], dtype=np.uint8)
    else:
        episode_pixels = np.asarray(planner.load_dataset_episode(dataset_path, args.episode_idx)["pixels"], dtype=np.uint8)
        start_pixel = episode_pixels[0]
        goal_pixel = episode_pixels[-1]
    if "rollout_frames" in rollout:
        rollout_frames = [frame.copy() for frame in np.asarray(rollout["rollout_frames"], dtype=np.uint8)]
    else:
        rollout_frames = reconstruct_rollout_frames(planner, rollout=rollout)

    obstacle_step = int(args.obstacle_step)
    if obstacle_step == -1:
        obstacle_step = int(rollout_qpos.shape[0] - 1)
        log_progress(f"Resolved obstacle step -1 to final rollout step {obstacle_step}.")
    if obstacle_step < 0 or obstacle_step >= rollout_qpos.shape[0]:
        raise ValueError(
            f"--obstacle-step must be in [0, {rollout_qpos.shape[0] - 1}], got {args.obstacle_step}."
        )
    episode_dir = out_root / f"episode_{args.episode_idx:05d}" / f"step_{obstacle_step:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    img_size = int(config.get("img_size", 224))
    embed_dim = int(config.get("embed_dim", 24))
    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)

    env = planner.make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    lower, upper = joint_limits_from_env(env)

    center_qpos = rollout_qpos[obstacle_step]
    joint_ranges = np.array([args.joint1_range, args.joint2_range], dtype=np.float64)
    total_obstacle = args.set_pca_count + args.set_norm_count + args.set_cal_count + args.set_test_count
    log_progress(
        f"Sampling {total_obstacle} local perturbations and rendering obstacle frames."
    )
    sampled_qpos = sample_local_perturbations(
        center_qpos,
        lower,
        upper,
        rng,
        joint_ranges=joint_ranges,
        count=total_obstacle,
    )
    perturb_frames = render_qpos_batch(
        planner,
        env,
        int(rollout["episode_seed"]),
        sampled_qpos,
        height=int(rollout["height"]),
        width=int(rollout["width"]),
    )
    env.close()

    perturb_pixels = planner.preprocess_pixels(
        perturb_frames,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    perturb_emb = planner.encode_frames(
        model,
        perturb_pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    ).detach().cpu().numpy().astype(np.float64)
    log_progress("Encoded perturbation frames into latent positions.")

    position_samples = perturb_emb[:, :embed_dim]
    permutation = rng.permutation(total_obstacle)
    position_samples = position_samples[permutation]
    sampled_qpos = sampled_qpos[permutation]

    pca_end = args.set_pca_count
    norm_end = pca_end + args.set_norm_count
    cal_end = norm_end + args.set_cal_count
    pca_samples = position_samples[:pca_end]
    norm_samples = position_samples[pca_end:norm_end]
    cal_samples = position_samples[norm_end:cal_end]
    test_samples = position_samples[cal_end:]
    construction_samples = position_samples[:cal_end]
    ellipsoid = build_conformal_ellipsoid(
        pca_samples,
        norm_samples,
        cal_samples,
        delta=args.delta,
        eps=args.conformal_eps,
        eigval_floor=args.eigval_floor,
    )

    nominal_position = rollout_emb[obstacle_step, :embed_dim]
    heldout_inside = ellipsoid_contains(test_samples, ellipsoid, tol=args.membership_tol)
    empirical_coverage = float(np.mean(heldout_inside))
    log_progress(
        f"Built conformal ellipsoid and evaluated held-out coverage: {empirical_coverage:.4f}."
    )

    log_progress(
        f"Sampling {args.background_samples} background latents for global/context PCA."
    )
    background_rows = sample_background_rows(background_dataset_path, rng, args.background_samples)
    background_emb = encode_dataset_rows(
        planner,
        model,
        background_dataset_path,
        background_rows,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        frame_batch_size=args.frame_batch_size,
    )
    background_pos = background_emb[:, :embed_dim]

    dataset_pos = dataset_emb[:, :embed_dim]
    rollout_pos = rollout_emb[:, :embed_dim]
    all_obstacle_samples = position_samples
    construction_samples_global = construction_samples
    heldout_inside_samples = test_samples[heldout_inside]
    heldout_outside_samples = test_samples[~heldout_inside]

    global_pca = fit_pca_2d(background_pos)
    local_pca = fit_pca_2d(all_obstacle_samples)

    local_boundary = ellipsoid_boundary_2d(*project_ellipsoid(ellipsoid, local_pca))

    rollout_goal_pixel = np.asarray(rollout_frames[-1], dtype=np.uint8)

    planner.save_rgb_image(episode_dir / "start_image.png", start_pixel)
    planner.save_rgb_image(episode_dir / "goal_image_target.png", goal_pixel)
    planner.save_rgb_image(episode_dir / "goal_image_rollout.png", rollout_goal_pixel)
    planner.save_rollout_video(rollout_frames, episode_dir, fps=args.video_fps)
    overlay_count = min(int(args.overlay_sample_count), int(sampled_qpos.shape[0]))
    log_progress(
        f"Rendering obstacle overlay image using {overlay_count} perturbation masks."
    )
    overlay_indices = np.linspace(0, sampled_qpos.shape[0] - 1, num=overlay_count, dtype=np.int64)
    overlay_qpos_batch = np.concatenate((center_qpos[None, :], sampled_qpos[overlay_indices]), axis=0)
    overlay_env = planner.make_render_env(
        seed=int(rollout["episode_seed"]),
        time_limit=float(rollout["time_limit"]),
        width=int(rollout["width"]),
        height=int(rollout["height"]),
        physics_freq_hz=float(rollout["physics_freq_hz"]),
    )
    overlay_frames, overlay_masks = render_masked_qpos_batch(
        overlay_env,
        int(rollout["episode_seed"]),
        overlay_qpos_batch,
        height=int(rollout["height"]),
        width=int(rollout["width"]),
    )
    overlay_env.close()
    obstacle_overlay = make_obstacle_overlay_image(
        nominal_frame=overlay_frames[0],
        nominal_mask=overlay_masks[0],
        perturb_frames=overlay_frames[1:],
        perturb_masks=overlay_masks[1:],
        perturb_alpha=float(args.overlay_perturb_alpha),
    )
    planner.save_rgb_image(episode_dir / "obstacle_overlay_all.png", obstacle_overlay)
    log_progress("Saved obstacle overlay image and summary plots.")

    make_plot(
        episode_dir / "pca_global.png",
        title=f"Global PCA trajectory/object view | episode {args.episode_idx} step {obstacle_step}",
        background=project_points(background_pos, global_pca),
        focus_background=None,
        construction_samples=project_points(all_obstacle_samples, global_pca),
        test_inside_samples=np.zeros((0, 2), dtype=np.float64),
        test_outside_samples=np.zeros((0, 2), dtype=np.float64),
        dataset_traj=project_points(dataset_pos, global_pca),
        rollout_traj=project_points(rollout_pos, global_pca),
        start_point=project_points(dataset_pos[[0]], global_pca)[0],
        dataset_goal_point=project_points(dataset_pos[[-1]], global_pca)[0],
        rollout_goal_point=project_points(rollout_pos[[-1]], global_pca)[0],
        nominal_point=project_points(nominal_position[None, :], global_pca)[0],
        ellipsoid_boundary=np.zeros((0, 2), dtype=np.float64),
        explained_ratio=np.asarray(global_pca["explained_variance_ratio"], dtype=np.float64),
        add_zoom_inset=False,
        focus_main_view=True,
    )
    make_plot(
        episode_dir / "pca_local.png",
        title=f"Local PCA obstacle view | episode {args.episode_idx} step {obstacle_step}",
        background=project_points(background_pos, local_pca),
        focus_background=None,
        construction_samples=project_points(construction_samples_global, local_pca),
        test_inside_samples=project_points(heldout_inside_samples, local_pca),
        test_outside_samples=project_points(heldout_outside_samples, local_pca),
        dataset_traj=project_points(dataset_pos, local_pca),
        rollout_traj=project_points(rollout_pos, local_pca),
        start_point=project_points(dataset_pos[[0]], local_pca)[0],
        dataset_goal_point=project_points(dataset_pos[[-1]], local_pca)[0],
        rollout_goal_point=project_points(rollout_pos[[-1]], local_pca)[0],
        nominal_point=project_points(nominal_position[None, :], local_pca)[0],
        ellipsoid_boundary=local_boundary,
        explained_ratio=np.asarray(local_pca["explained_variance_ratio"], dtype=np.float64),
        add_zoom_inset=False,
        focus_main_view=False,
    )
    obstacle_coords = ellipsoid_coords(all_obstacle_samples, ellipsoid)
    construction_coords = obstacle_coords[:cal_end]
    test_coords = obstacle_coords[cal_end:]
    make_ellipsoid_coords_plot(
        episode_dir / "ellipsoid_coords.png",
        construction_coords=construction_coords[:, :2],
        test_coords=test_coords[:, :2],
        test_inside_mask=heldout_inside,
        nominal_coords=ellipsoid_coords(nominal_position[None, :], ellipsoid)[0, :2],
    )
    torch.save(
        {
            "rollout_cache_path": str(rollout_cache_path),
            "background_rows": background_rows.astype(np.int64),
            "sampled_qpos": sampled_qpos.astype(np.float64),
            "obstacle_position_samples": all_obstacle_samples.astype(np.float64),
            "obstacle_nominal_qpos": center_qpos.astype(np.float64),
            "obstacle_nominal_qvel": rollout_qvel[obstacle_step].astype(np.float64),
            "obstacle_nominal_position": nominal_position.astype(np.float64),
            "ellipsoid": ellipsoid,
            "global_pca": global_pca,
            "local_pca": local_pca,
            "empirical_coverage_mask": heldout_inside.astype(bool),
            "empirical_coverage": empirical_coverage,
        },
        episode_dir / "analysis.pt",
    )
    save_json(
        episode_dir / "summary.json",
        {
            "episode_idx": int(args.episode_idx),
            "obstacle_step": int(obstacle_step),
            "rollout_cache_path": str(rollout_cache_path),
            "background_dataset_path": str(background_dataset_path),
            "test_dataset_path": str(dataset_path),
            "checkpoint_path": str(checkpoint_path),
            "joint_range": joint_ranges.tolist(),
            "obstacle_center_qpos": center_qpos.tolist(),
            "obstacle_center_qvel": rollout_qvel[obstacle_step].tolist(),
            "obstacle_center_latent_norm": float(np.linalg.norm(nominal_position)),
            "sample_counts": {
                "pca": int(args.set_pca_count),
                "norm": int(args.set_norm_count),
                "cal": int(args.set_cal_count),
                "test": int(args.set_test_count),
                "background": int(args.background_samples),
                "overlay": int(overlay_count),
            },
            "delta": float(args.delta),
            "empirical_coverage": empirical_coverage,
            "empirical_coverage_count_inside": int(np.count_nonzero(heldout_inside)),
            "empirical_coverage_count_total": int(heldout_inside.shape[0]),
            "conformal_radius": float(ellipsoid["conformal_radius"]),
            "global_pca_explained_variance_ratio": np.asarray(
                global_pca["explained_variance_ratio"], dtype=np.float64
            ).tolist(),
            "local_pca_explained_variance_ratio": np.asarray(
                local_pca["explained_variance_ratio"], dtype=np.float64
            ).tolist(),
            "global_plot_explained_variance_ratio": np.asarray(
                global_pca["explained_variance_ratio"], dtype=np.float64
            ).tolist(),
        },
    )

    log_progress("Analysis complete.")
    print(f"Analysis dir:  {episode_dir}")
    print(f"Empirical Coverage:      {empirical_coverage:.4f} ({int(np.count_nonzero(heldout_inside))}/{heldout_inside.shape[0]})")


if __name__ == "__main__":
    main()
