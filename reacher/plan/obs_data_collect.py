#!/usr/bin/env python3
"""Generate a balanced Reacher circle-obstacle image dataset.

The positive class is IK configurations whose fingertip lies inside a workspace
circle. The negative class is joint-space configurations whose fingertip lies
outside the circle with a configurable margin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_OUT_DIR = "reacher/plan/obstacle_data"
DEFAULT_SAMPLES_PER_CLASS = 4096
DIAGNOSTIC_PLOT_NAME = "workspace_samples.png"
OBSTACLE_OVERLAY_NAME = "obstacle_overlay.png"
OUTSIDE_OVERLAY_NAME = "outside_overlay.png"
OBSTACLE_DATA_NAME = "obstacle_classifier_data.pt"
DEFAULT_OVERLAY_PERTURB_ALPHA = 0.035
TRAIN_FRACTION = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--inside-sample-count", type=int, default=None)
    parser.add_argument("--outside-sample-count", type=int, default=None)
    parser.add_argument("--sampling-budget", type=int, default=200_000)
    parser.add_argument("--inside-bend-sign", "--bend-sign", dest="inside_bend_sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--circle-center-x", type=float, default=0.145)
    parser.add_argument("--circle-center-y", type=float, default=0.145)
    parser.add_argument("--circle-radius", type=float, default=0.03)
    parser.add_argument("--outside-margin", type=float, default=0.005)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--physics-freq-hz", type=float, default=100.0)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[obs_data_collect] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]


def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep


def make_render_env(
    *,
    seed: int,
    time_limit: float,
    width: int,
    height: int,
    physics_freq_hz: float,
) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name="hard",
        seed=seed,
        time_limit=time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env


def reset_env_to_state(
    env: DmControlGymEnv,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def render_qpos_batch(
    env: DmControlGymEnv,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
    progress_desc: str,
) -> np.ndarray:
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    frames: list[np.ndarray] = []
    for qpos in tqdm(qpos_batch, desc=progress_desc, unit="image"):
        frames.append(
            reset_env_to_state(
                env,
                seed=seed,
                qpos=np.asarray(qpos, dtype=np.float32),
                qvel=qvel,
                height=height,
                width=width,
            ).copy()
        )
    return np.stack(frames, axis=0)


def get_arm_geom_ids(model: Any) -> np.ndarray:
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


def make_segmentation_scene_option(model: Any) -> tuple[Any, int, int]:
    from dm_control.mujoco.wrapper import core as dm_core

    target_geom_id = int(model.name2id("target", "geom"))
    original_group = int(model.geom_group[target_geom_id])
    model.geom_group[target_geom_id] = 3
    scene_option = dm_core.MjvOption()
    scene_option.geomgroup[:] = 1
    scene_option.geomgroup[3] = 0
    return scene_option, target_geom_id, original_group


def render_masked_overlay(
    env: DmControlGymEnv,
    seed: int,
    qpos_batch: np.ndarray,
    *,
    height: int,
    width: int,
    progress_desc: str,
    perturb_alpha: float,
) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    model = physics.model
    qvel = np.zeros(qpos_batch.shape[1], dtype=np.float32)
    arm_geom_ids = get_arm_geom_ids(model)
    scene_option, target_geom_id, original_group = make_segmentation_scene_option(model)
    canvas: np.ndarray | None = None
    nominal_frame: np.ndarray | None = None
    nominal_mask: np.ndarray | None = None
    try:
        for index, qpos in enumerate(tqdm(qpos_batch, desc=progress_desc, unit="image")):
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
            mask = build_arm_mask(segmentation, arm_geom_ids)
            if index == 0:
                nominal_frame = frame.copy()
                nominal_mask = mask.copy()
                canvas = np.full_like(frame, 255, dtype=np.uint8)
            else:
                if canvas is None:
                    raise RuntimeError("Overlay canvas was not initialized.")
                canvas = alpha_composite_masked(canvas, frame, mask, alpha=float(perturb_alpha))
    finally:
        model.geom_group[target_geom_id] = original_group
    if canvas is None or nominal_frame is None or nominal_mask is None:
        raise RuntimeError("No frames were rendered for overlay generation.")
    return alpha_composite_masked(canvas, nominal_frame, nominal_mask, alpha=1.0)


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


def save_obstacle_overlay(
    *,
    seed: int,
    time_limit: float,
    width: int,
    height: int,
    physics_freq_hz: float,
    center_qpos: np.ndarray,
    sample_qpos: np.ndarray,
    out_path: Path,
    progress_desc: str,
    perturb_alpha: float = DEFAULT_OVERLAY_PERTURB_ALPHA,
) -> None:
    overlay_qpos_batch = np.concatenate((center_qpos[None, :], sample_qpos), axis=0)
    env = make_render_env(
        seed=seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    try:
        overlay = render_masked_overlay(
            env,
            seed,
            overlay_qpos_batch,
            height=height,
            width=width,
            progress_desc=progress_desc,
            perturb_alpha=float(perturb_alpha),
        )
    finally:
        env.close()
    save_rgb_image(out_path, overlay)


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def get_descendant_body_ids(model: Any, root_body_id: int) -> set[int]:
    descendants = {int(root_body_id)}
    parent_ids = np.asarray(model.body_parentid, dtype=np.int32)
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            parent_id = int(parent_ids[body_id])
            if parent_id in descendants and body_id not in descendants:
                descendants.add(body_id)
                changed = True
    return descendants


def infer_planar_arm_geometry(env: DmControlGymEnv) -> dict[str, Any]:
    physics = env._env.physics
    model = physics.model
    with physics.reset_context():
        physics.data.qpos[:2] = 0.0
        physics.data.qvel[:2] = 0.0

    arm_body_id = int(model.name2id("arm", "body"))
    hand_body_id = int(model.name2id("hand", "body"))
    base_xy = np.asarray(physics.data.xpos[arm_body_id][:2], dtype=np.float64)
    hand_xy = np.asarray(physics.data.xpos[hand_body_id][:2], dtype=np.float64)
    hand_xy_local = hand_xy - base_xy
    link1 = float(np.linalg.norm(hand_xy_local))

    hand_descendants = get_descendant_body_ids(model, hand_body_id)
    candidates: list[tuple[float, np.ndarray, str]] = []

    for site_id in range(int(model.nsite)):
        name = model.id2name(site_id, "site")
        if name is None or "target" in name or int(model.site_bodyid[site_id]) not in hand_descendants:
            continue
        xy_local = np.asarray(physics.data.site_xpos[site_id][:2], dtype=np.float64) - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"site:{name}"))

    for geom_id in range(int(model.ngeom)):
        name = model.id2name(geom_id, "geom")
        if name is None or name in {"target", "root"} or int(model.geom_bodyid[geom_id]) not in hand_descendants:
            continue
        xy_local = np.asarray(physics.data.geom_xpos[geom_id][:2], dtype=np.float64) - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"geom:{name}"))

    for body_id in hand_descendants:
        xy_local = np.asarray(physics.data.xpos[body_id][:2], dtype=np.float64) - base_xy
        candidates.append((float(np.linalg.norm(xy_local - hand_xy_local)), xy_local, f"body:{model.id2name(body_id, 'body')}"))

    if not candidates:
        raise RuntimeError("Failed to infer fingertip geometry from the DM Control Reacher model.")

    link2, tip_xy, tip_source = max(candidates, key=lambda item: item[0])
    return {
        "base_xy": base_xy,
        "link1": float(link1),
        "link2": float(link2),
        "reach_min": float(abs(link1 - link2)),
        "reach_max": float(link1 + link2),
        "tip_source": str(tip_source),
        "tip_xy_at_zero": tip_xy,
    }


def joint_limits_with_unbounded_fixed(env: DmControlGymEnv) -> tuple[np.ndarray, np.ndarray]:
    model = env._env.physics.model
    raw = np.asarray(model.jnt_range[:2], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[:2], dtype=bool)
    lower = raw[:, 0].copy()
    upper = raw[:, 1].copy()
    lower[~limited] = -np.inf
    upper[~limited] = np.inf
    return lower, upper


def resolve_circle_spec(
    geom: dict[str, Any],
    *,
    circle_center_x: float | None,
    circle_center_y: float | None,
    circle_radius: float | None,
) -> tuple[np.ndarray, float]:
    reach_min = float(geom["reach_min"])
    reach_max = float(geom["reach_max"])
    radius = max(float(circle_radius) if circle_radius is not None else min(0.03, 0.12 * (reach_max - reach_min)), 1e-4)

    default_center_radius = reach_min + 0.72 * max(reach_max - reach_min - radius, 1e-6)
    default_center_local = default_center_radius * np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0)
    center_local = np.array(
        [
            default_center_local[0] if circle_center_x is None else float(circle_center_x),
            default_center_local[1] if circle_center_y is None else float(circle_center_y),
        ],
        dtype=np.float64,
    )

    center_norm = float(np.linalg.norm(center_local))
    min_center_norm = reach_min + radius + 1e-4
    max_center_norm = reach_max - radius - 1e-4
    if center_norm < 1e-9:
        center_local = default_center_local.copy()
        center_norm = float(np.linalg.norm(center_local))
    if center_norm < min_center_norm:
        center_local *= min_center_norm / center_norm
    if center_norm > max_center_norm:
        center_local *= max_center_norm / center_norm
    if center_local[0] <= 0.0 or center_local[1] <= 0.0:
        center_local = np.abs(center_local)
        center_norm = float(np.linalg.norm(center_local))
        if center_norm > max_center_norm:
            center_local *= max_center_norm / center_norm
    return np.asarray(geom["base_xy"], dtype=np.float64) + center_local, radius


def sample_points_in_circle(rng: np.random.Generator, center_xy: np.ndarray, radius: float, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=count)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=count))
    offsets = np.stack((radial * np.cos(theta), radial * np.sin(theta)), axis=1)
    return center_xy[None, :] + offsets


def solve_two_link_ik(
    target_xy: np.ndarray,
    *,
    link1: float,
    link2: float,
    bend_sign: int,
) -> np.ndarray:
    x = float(target_xy[0])
    y = float(target_xy[1])
    radius_sq = x * x + y * y
    cos_theta2 = (radius_sq - link1 * link1 - link2 * link2) / (2.0 * link1 * link2)
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    sin_theta2 = float(bend_sign) * math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = math.atan2(y, x) - math.atan2(link2 * sin_theta2, link1 + link2 * cos_theta2)
    return np.array([wrap_angle(theta1), wrap_angle(theta2)], dtype=np.float64)


def forward_kinematics(qpos: np.ndarray, *, link1: float, link2: float) -> tuple[np.ndarray, np.ndarray]:
    theta1 = float(qpos[0])
    theta2 = float(qpos[1])
    elbow_xy = np.array([link1 * math.cos(theta1), link1 * math.sin(theta1)], dtype=np.float64)
    tip_xy = elbow_xy + np.array(
        [link2 * math.cos(theta1 + theta2), link2 * math.sin(theta1 + theta2)],
        dtype=np.float64,
    )
    return elbow_xy, tip_xy


def qpos_within_limits(qpos: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-6) -> bool:
    if np.any(np.isfinite(lower) & (qpos < lower - tol)):
        return False
    if np.any(np.isfinite(upper) & (qpos > upper + tol)):
        return False
    return True


def solve_ik_batch(
    targets_xy: np.ndarray,
    *,
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_list: list[np.ndarray] = []
    elbow_list: list[np.ndarray] = []
    tip_list: list[np.ndarray] = []
    for target_xy_world in targets_xy:
        target_xy_local = np.asarray(target_xy_world, dtype=np.float64) - base_xy
        qpos = solve_two_link_ik(target_xy_local, link1=link1, link2=link2, bend_sign=bend_sign)
        if not qpos_within_limits(qpos, lower, upper):
            continue
        elbow_xy_local, tip_xy_local = forward_kinematics(qpos, link1=link1, link2=link2)
        if np.linalg.norm(tip_xy_local - target_xy_local) > 1e-5:
            continue
        qpos_list.append(qpos)
        elbow_list.append(base_xy + elbow_xy_local)
        tip_list.append(base_xy + tip_xy_local)
    if not qpos_list:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
        )
    return np.stack(qpos_list, axis=0), np.stack(elbow_list, axis=0), np.stack(tip_list, axis=0)


def collect_valid_ik_samples(
    *,
    count: int,
    target_sampler: Callable[[int], np.ndarray],
    base_xy: np.ndarray,
    link1: float,
    link2: float,
    bend_sign: int,
    lower: np.ndarray,
    upper: np.ndarray,
    sampling_budget: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_chunks: list[np.ndarray] = []
    elbow_chunks: list[np.ndarray] = []
    tip_chunks: list[np.ndarray] = []
    remaining = int(count)
    draws = 0
    while remaining > 0 and draws < sampling_budget:
        batch_count = min(max(remaining * 2, 1024), sampling_budget - draws)
        qpos_batch, elbow_xy, tip_xy = solve_ik_batch(
            target_sampler(batch_count),
            base_xy=base_xy,
            link1=link1,
            link2=link2,
            bend_sign=bend_sign,
            lower=lower,
            upper=upper,
        )
        draws += batch_count
        if qpos_batch.shape[0] == 0:
            continue
        take = min(remaining, qpos_batch.shape[0])
        qpos_chunks.append(qpos_batch[:take])
        elbow_chunks.append(elbow_xy[:take])
        tip_chunks.append(tip_xy[:take])
        remaining -= take

    if remaining > 0:
        raise RuntimeError(f"Failed to collect {count} valid {label} IK samples after {draws} target draws; missing {remaining}.")
    return np.concatenate(qpos_chunks, axis=0), np.concatenate(elbow_chunks, axis=0), np.concatenate(tip_chunks, axis=0)


def qpos_sampling_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_lower = np.asarray(lower, dtype=np.float64).copy()
    sample_upper = np.asarray(upper, dtype=np.float64).copy()
    unbounded = ~np.isfinite(sample_lower) | ~np.isfinite(sample_upper)
    sample_lower[unbounded] = -math.pi
    sample_upper[unbounded] = math.pi
    return sample_lower, sample_upper


def sample_valid_outside_qpos(
    rng: np.random.Generator,
    *,
    count: int,
    circle_center_xy: np.ndarray,
    circle_radius: float,
    base_xy: np.ndarray,
    outside_margin: float,
    link1: float,
    link2: float,
    lower: np.ndarray,
    upper: np.ndarray,
    sampling_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_lower, sample_upper = qpos_sampling_bounds(lower, upper)
    min_tip_distance = float(circle_radius) + max(float(outside_margin), 1e-4)
    qpos_chunks: list[np.ndarray] = []
    elbow_chunks: list[np.ndarray] = []
    tip_chunks: list[np.ndarray] = []
    remaining = int(count)
    draws = 0
    while remaining > 0 and draws < sampling_budget:
        batch_count = min(max(remaining * 4, 1024), sampling_budget - draws)
        qpos_batch = rng.uniform(sample_lower, sample_upper, size=(batch_count, sample_lower.shape[0]))
        draws += batch_count
        elbow_local = np.stack((link1 * np.cos(qpos_batch[:, 0]), link1 * np.sin(qpos_batch[:, 0])), axis=1)
        tip_local = elbow_local + np.stack(
            (
                link2 * np.cos(qpos_batch[:, 0] + qpos_batch[:, 1]),
                link2 * np.sin(qpos_batch[:, 0] + qpos_batch[:, 1]),
            ),
            axis=1,
        )
        elbow_xy = base_xy[None, :] + elbow_local
        tip_xy = base_xy[None, :] + tip_local
        outside_mask = np.linalg.norm(tip_xy - circle_center_xy[None, :], axis=1) >= min_tip_distance
        if not np.any(outside_mask):
            continue
        take = min(remaining, int(np.sum(outside_mask)))
        qpos_chunks.append(qpos_batch[outside_mask][:take])
        elbow_chunks.append(elbow_xy[outside_mask][:take])
        tip_chunks.append(tip_xy[outside_mask][:take])
        remaining -= take

    if remaining > 0:
        raise RuntimeError(f"Failed to collect {count} outside-circle samples after {draws} qpos draws; missing {remaining}.")
    return np.concatenate(qpos_chunks, axis=0), np.concatenate(elbow_chunks, axis=0), np.concatenate(tip_chunks, axis=0)


def build_balanced_dataset(
    inside_qpos: np.ndarray,
    inside_elbow_xy: np.ndarray,
    inside_tip_xy: np.ndarray,
    outside_qpos: np.ndarray,
    outside_elbow_xy: np.ndarray,
    outside_tip_xy: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    qpos = np.concatenate((inside_qpos, outside_qpos), axis=0).astype(np.float32)
    elbow_xy = np.concatenate((inside_elbow_xy, outside_elbow_xy), axis=0).astype(np.float32)
    tip_xy = np.concatenate((inside_tip_xy, outside_tip_xy), axis=0).astype(np.float32)
    labels = np.concatenate(
        (
            np.ones((inside_qpos.shape[0],), dtype=np.int64),
            np.zeros((outside_qpos.shape[0],), dtype=np.int64),
        ),
        axis=0,
    )
    indices = np.arange(labels.shape[0], dtype=np.int64)
    rng.shuffle(indices)
    shuffled_labels = labels[indices]
    obstacle_train, obstacle_cal = split_indices(np.flatnonzero(shuffled_labels == 1), rng, TRAIN_FRACTION)
    outside_train, outside_cal = split_indices(np.flatnonzero(shuffled_labels == 0), rng, TRAIN_FRACTION)
    train_idx = np.concatenate((obstacle_train, outside_train), axis=0)
    calibration_idx = np.concatenate((obstacle_cal, outside_cal), axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(calibration_idx)
    return {
        "qpos": qpos[indices],
        "task_target": qpos[indices],
        "elbow_xy": elbow_xy[indices],
        "tip_xy": tip_xy[indices],
        "label": shuffled_labels,
        "train_idx": train_idx.astype(np.int64),
        "calibration_idx": calibration_idx.astype(np.int64),
    }


def save_workspace_plot(
    *,
    out_path: Path,
    geom: dict[str, Any],
    circle_center_xy: np.ndarray,
    circle_radius: float,
    center_qpos: np.ndarray,
    inside_elbow_xy: np.ndarray,
    inside_tip_xy: np.ndarray,
    outside_elbow_xy: np.ndarray,
    outside_tip_xy: np.ndarray,
    bend_sign: int,
) -> None:
    base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    reach_max = float(geom["reach_max"])
    reach_min = float(geom["reach_min"])
    ax.add_patch(plt.Circle(base_xy, reach_max, color="#d9d9d9", fill=False, linestyle="--", linewidth=1.0))
    if reach_min > 1e-4:
        ax.add_patch(plt.Circle(base_xy, reach_min, color="#ececec", fill=False, linestyle=":", linewidth=1.0))
    ax.add_patch(plt.Circle(circle_center_xy, circle_radius, color="#d55e00", fill=False, linewidth=2.0))

    max_arms = min(inside_tip_xy.shape[0], 512)
    for elbow_point, tip_point in zip(inside_elbow_xy[:max_arms], inside_tip_xy[:max_arms], strict=True):
        ax.plot([base_xy[0], elbow_point[0]], [base_xy[1], elbow_point[1]], color="#111111", linewidth=0.7, alpha=0.08)
        ax.plot([elbow_point[0], tip_point[0]], [elbow_point[1], tip_point[1]], color="#111111", linewidth=0.7, alpha=0.08)

    ax.scatter(inside_tip_xy[:, 0], inside_tip_xy[:, 1], s=8, c="#d55e00", alpha=0.45, label="obstacle tips")
    ax.scatter(outside_tip_xy[:, 0], outside_tip_xy[:, 1], s=8, c="#009e73", alpha=0.35, label="outside tips")

    center_elbow_local, center_tip_local = forward_kinematics(center_qpos, link1=float(geom["link1"]), link2=float(geom["link2"]))
    center_elbow_xy = base_xy + center_elbow_local
    center_tip_xy = base_xy + center_tip_local
    ax.plot([base_xy[0], center_elbow_xy[0]], [base_xy[1], center_elbow_xy[1]], color="#000000", linewidth=2.0)
    ax.plot([center_elbow_xy[0], center_tip_xy[0]], [center_elbow_xy[1], center_tip_xy[1]], color="#000000", linewidth=2.0)
    ax.scatter([base_xy[0], center_elbow_xy[0], center_tip_xy[0]], [base_xy[1], center_elbow_xy[1], center_tip_xy[1]], c="#000000", s=18)
    ax.scatter([circle_center_xy[0]], [circle_center_xy[1]], s=24, c="#d55e00")

    ax.set_aspect("equal", adjustable="box")
    lim = reach_max + 0.04
    ax.set_xlim(base_xy[0] - lim, base_xy[0] + lim)
    ax.set_ylim(base_xy[1] - lim, base_xy[1] + lim)
    ax.grid(alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Reacher circle obstacle samples, bend_sign={bend_sign:+d}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if args.inside_sample_count is not None and int(args.inside_sample_count) <= 0:
        raise ValueError("--inside-sample-count must be positive when provided.")
    if args.outside_sample_count is not None and int(args.outside_sample_count) <= 0:
        raise ValueError("--outside-sample-count must be positive when provided.")
    if int(args.sampling_budget) <= 0:
        raise ValueError("--sampling-budget must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if float(args.circle_radius) <= 0.0:
        raise ValueError("--circle-radius must be positive.")
    if float(args.outside_margin) < 0.0:
        raise ValueError("--outside-margin must be non-negative.")


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size == 1:
        return shuffled.copy(), np.zeros((0,), dtype=np.int64)
    train_count = int(np.floor(float(train_fraction) * float(shuffled.size)))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    return shuffled[:train_count], shuffled[train_count:]


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inside_count = int(args.inside_sample_count or args.samples_per_class)
    outside_count = int(args.outside_sample_count or args.samples_per_class)

    log_progress("Inferring arm geometry and sampling circle obstacle states.")
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        geom = infer_planar_arm_geometry(env)
        lower, upper = joint_limits_with_unbounded_fixed(env)
        circle_center_xy, circle_radius = resolve_circle_spec(
            geom,
            circle_center_x=float(args.circle_center_x),
            circle_center_y=float(args.circle_center_y),
            circle_radius=float(args.circle_radius),
        )
        base_xy = np.asarray(geom["base_xy"], dtype=np.float64)
        inside_qpos, inside_elbow_xy, inside_tip_xy = collect_valid_ik_samples(
            count=inside_count,
            target_sampler=lambda batch_count: sample_points_in_circle(rng, circle_center_xy, float(circle_radius), batch_count),
            base_xy=base_xy,
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            bend_sign=int(args.inside_bend_sign),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
            label="inside-circle",
        )
        outside_qpos, outside_elbow_xy, outside_tip_xy = sample_valid_outside_qpos(
            rng,
            count=outside_count,
            circle_center_xy=circle_center_xy,
            circle_radius=float(circle_radius),
            base_xy=base_xy,
            outside_margin=float(args.outside_margin),
            link1=float(geom["link1"]),
            link2=float(geom["link2"]),
            lower=lower,
            upper=upper,
            sampling_budget=int(args.sampling_budget),
        )
    finally:
        env.close()

    if outside_qpos.shape[0] < 2:
        raise RuntimeError("Need at least two outside configurations to define planner start and goal.")

    circle_center_local = circle_center_xy - np.asarray(geom["base_xy"], dtype=np.float64)
    center_qpos = solve_two_link_ik(
        circle_center_local,
        link1=float(geom["link1"]),
        link2=float(geom["link2"]),
        bend_sign=int(args.inside_bend_sign),
    )
    if not qpos_within_limits(center_qpos, lower, upper):
        raise RuntimeError("Circle-center IK solution violates joint limits.")

    balanced = build_balanced_dataset(
        inside_qpos,
        inside_elbow_xy,
        inside_tip_xy,
        outside_qpos,
        outside_elbow_xy,
        outside_tip_xy,
        rng,
    )

    log_progress("Rendering balanced dataset images.")
    env = make_render_env(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
    )
    try:
        pixels = render_qpos_batch(
            env,
            int(args.seed),
            np.asarray(balanced["qpos"], dtype=np.float32),
            height=int(args.height),
            width=int(args.width),
            progress_desc="Rendering obstacle dataset",
        )
    finally:
        env.close()

    save_workspace_plot(
        out_path=out_dir / DIAGNOSTIC_PLOT_NAME,
        geom=geom,
        circle_center_xy=circle_center_xy,
        circle_radius=float(circle_radius),
        center_qpos=center_qpos,
        inside_elbow_xy=inside_elbow_xy,
        inside_tip_xy=inside_tip_xy,
        outside_elbow_xy=outside_elbow_xy,
        outside_tip_xy=outside_tip_xy,
        bend_sign=int(args.inside_bend_sign),
    )
    save_obstacle_overlay(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
        center_qpos=center_qpos,
        sample_qpos=inside_qpos,
        out_path=out_dir / OBSTACLE_OVERLAY_NAME,
        progress_desc="Rendering obstacle overlay",
    )
    save_obstacle_overlay(
        seed=int(args.seed),
        time_limit=float(args.time_limit),
        width=int(args.width),
        height=int(args.height),
        physics_freq_hz=float(args.physics_freq_hz),
        center_qpos=center_qpos,
        sample_qpos=outside_qpos,
        out_path=out_dir / OUTSIDE_OVERLAY_NAME,
        progress_desc="Rendering outside overlay",
    )

    metadata = {
        "seed": int(args.seed),
        "image_width": int(args.width),
        "image_height": int(args.height),
        "time_limit": float(args.time_limit),
        "physics_freq_hz": float(args.physics_freq_hz),
        "inside_bend_sign": int(args.inside_bend_sign),
        "circle_center_xy": np.asarray(circle_center_xy, dtype=np.float32),
        "circle_radius": float(circle_radius),
        "outside_margin": float(args.outside_margin),
        "task_lower": lower.astype(np.float32),
        "task_upper": upper.astype(np.float32),
        "samples_per_class": int(args.samples_per_class),
        "inside_sample_count": int(inside_qpos.shape[0]),
        "outside_sample_count": int(outside_qpos.shape[0]),
        "train_fraction": float(TRAIN_FRACTION),
        "calibration_fraction": float(1.0 - TRAIN_FRACTION),
        "obstacle_label": 1,
        "non_obstacle_label": 0,
        "label_rule": "1 iff fingertip lies inside workspace circle; 0 iff fingertip is at least circle_radius + outside_margin from center",
        "tip_source": str(geom["tip_source"]),
        "link1": float(geom["link1"]),
        "link2": float(geom["link2"]),
        "reach_min": float(geom["reach_min"]),
        "reach_max": float(geom["reach_max"]),
    }

    torch.save(
        {
            "metadata": metadata,
            "dataset": {
                "pixels": pixels.astype(np.uint8),
                "task_target": balanced["task_target"].astype(np.float32),
                "qpos": balanced["qpos"].astype(np.float32),
                "qvel": np.zeros_like(balanced["qpos"], dtype=np.float32),
                "tip_xy": balanced["tip_xy"].astype(np.float32),
                "elbow_xy": balanced["elbow_xy"].astype(np.float32),
                "label": balanced["label"].astype(np.int64),
                "train_idx": balanced["train_idx"].astype(np.int64),
                "calibration_idx": balanced["calibration_idx"].astype(np.int64),
            },
            "obstacle_data": {
                "obstacle_center_qpos": np.asarray(center_qpos, dtype=np.float32),
                "obstacle_qpos": inside_qpos.astype(np.float32),
                "outside_qpos": outside_qpos.astype(np.float32),
                "obstacle_tip_xy": inside_tip_xy.astype(np.float32),
                "outside_tip_xy": outside_tip_xy.astype(np.float32),
                "circle_center_xy": np.asarray(circle_center_xy, dtype=np.float32),
                "circle_radius": float(circle_radius),
            },
        },
        out_dir / OBSTACLE_DATA_NAME,
    )

    summary = {
        "out_dir": str(out_dir),
        "dataset_path": str(out_dir / OBSTACLE_DATA_NAME),
        "workspace_plot_path": str(out_dir / DIAGNOSTIC_PLOT_NAME),
        "obstacle_overlay_path": str(out_dir / OBSTACLE_OVERLAY_NAME),
        "outside_overlay_path": str(out_dir / OUTSIDE_OVERLAY_NAME),
        "counts": {
            "obstacle": int(np.sum(balanced["label"] == 1)),
            "non_obstacle": int(np.sum(balanced["label"] == 0)),
            "total": int(balanced["label"].shape[0]),
            "train": int(balanced["train_idx"].shape[0]),
            "calibration": int(balanced["calibration_idx"].shape[0]),
        },
        "circle_center_xy": np.asarray(circle_center_xy, dtype=np.float64).tolist(),
        "circle_radius": float(circle_radius),
        "outside_margin": float(args.outside_margin),
        "inside_bend_sign": int(args.inside_bend_sign),
        "center_qpos": np.asarray(center_qpos, dtype=np.float64).tolist(),
        "link1": float(geom["link1"]),
        "link2": float(geom["link2"]),
        "reach_min": float(geom["reach_min"]),
        "reach_max": float(geom["reach_max"]),
        "tip_source": str(geom["tip_source"]),
    }
    save_json(out_dir / "summary.json", summary)

    print(f"Saved dataset:   {out_dir / OBSTACLE_DATA_NAME}")
    print(f"Saved workspace: {out_dir / DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved obstacle overlay: {out_dir / OBSTACLE_OVERLAY_NAME}")
    print(f"Saved outside overlay:  {out_dir / OUTSIDE_OVERLAY_NAME}")
    print(f"Saved summary:   {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
