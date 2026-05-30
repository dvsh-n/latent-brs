#!/usr/bin/env python3
"""Collect a balanced OGBench cube half-ellipsoid obstacle image dataset from synthesized grasped states."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

import gymnasium
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401
import torch
from ogbench.manipspace import lie
from tqdm.auto import tqdm

from ogbench_cube.data.ogbench_cube_data_gen import (
    DEFAULT_CONTROL_DECIMATION,
    DEFAULT_ENV_NAME,
    DEFAULT_SIM_FREQ_HZ,
    LocalCubePlanOracle,
    THETA_SAMPLING_BOUNDS,
    XY_SAMPLING_BOUNDS,
    Z_SAMPLING_BOUNDS,
    apply_xy_sampling_bounds,
    compute_env_timing,
    sample_valid_reset,
)

DEFAULT_OUT_DIR = Path("ogbench_cube/plan/obstacle_data_3d_ellipsoid_back")
DEFAULT_CAMERA = "front_pixels"
DEFAULT_SAMPLES_PER_CLASS = 8192
DEFAULT_MAX_ORACLE_STEPS = 80
DEFAULT_SETTLE_STEPS = 6
DEFAULT_HEIGHT_MARGIN = 0.005
DEFAULT_INSIDE_SAMPLE_ATTEMPTS = 1024
DEFAULT_ACCEPTANCE_POS_TOL = 0.03
DEFAULT_ACCEPTANCE_YAW_TOL = 0.25
DEFAULT_MAX_ATTEMPTS_PER_CLASS = 200_000
DEFAULT_NON_OBSTACLE_OUTSIDE_Y_PROB = 0.75
DIAGNOSTIC_PLOT_NAME = "balanced_obstacle_dataset_xy.png"

TABLE_Z = 0.02
OBSTACLE_BASE_Z = 0.0
OBSTACLE_PEAK_Z = 0.055
OBSTACLE_CENTER_X = 0.35
OBSTACLE_CENTER_Y = 0.0
OBSTACLE_RADIUS_X = 0.04
OBSTACLE_RADIUS_Y = 0.08


@dataclass
class GraspReference:
    block_pos: list[float]
    block_yaw: float
    effector_pos: list[float]
    effector_yaw: float
    gripper_opening: float
    delta_local: list[float]
    delta_yaw: float
    qpos: list[float]
    qvel: list[float]
    control: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--oracle-segment-dt", type=float, default=0.4)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=0.5)
    parser.add_argument("--max-oracle-steps", type=int, default=DEFAULT_MAX_ORACLE_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    parser.add_argument("--acceptance-pos-tol", type=float, default=DEFAULT_ACCEPTANCE_POS_TOL)
    parser.add_argument("--acceptance-yaw-tol", type=float, default=DEFAULT_ACCEPTANCE_YAW_TOL)
    parser.add_argument(
        "--require-grasped",
        action="store_true",
        help="Reject samples unless the synthesized state exceeds the grasp/contact thresholds.",
    )
    parser.add_argument(
        "--require-yaw-match",
        action="store_true",
        help="Reject samples unless the synthesized cube yaw is within --acceptance-yaw-tol of the request.",
    )
    parser.add_argument(
        "--non-obstacle-outside-y-prob",
        type=float,
        default=DEFAULT_NON_OBSTACLE_OUTSIDE_Y_PROB,
        help=(
            "Probability that a class-0 sample is drawn from outside the obstacle x/y footprint. "
            "The remaining probability is used for above-cap samples inside that footprint."
        ),
    )
    parser.add_argument("--max-attempts-per-class", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_CLASS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.samples_per_class) <= 0:
        raise ValueError("--samples-per-class must be positive.")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive.")
    if int(args.max_oracle_steps) <= 0 or int(args.settle_steps) < 0:
        raise ValueError("--max-oracle-steps must be positive and --settle-steps must be non-negative.")
    if float(args.height_margin) < 0.0:
        raise ValueError("--height-margin must be non-negative.")
    if float(args.acceptance_pos_tol) < 0.0 or float(args.acceptance_yaw_tol) < 0.0:
        raise ValueError("Acceptance tolerances must be non-negative.")
    if not 0.0 <= float(args.non_obstacle_outside_y_prob) <= 1.0:
        raise ValueError("--non-obstacle-outside-y-prob must lie in [0, 1].")
    if int(args.max_attempts_per_class) <= 0:
        raise ValueError("--max-attempts-per-class must be positive.")


def angular_distance(a: float, b: float) -> float:
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def rotation_z(yaw: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def half_ellipsoid_height(
    x_value: float | np.ndarray,
    y_value: float | np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    base_z: float,
    peak_z: float,
) -> float | np.ndarray:
    x_values = np.asarray(x_value, dtype=np.float64)
    y_values = np.asarray(y_value, dtype=np.float64)
    normalized = ((x_values - float(center_x)) / float(radius_x)) ** 2 + (
        (y_values - float(center_y)) / float(radius_y)
    ) ** 2
    profile = np.sqrt(np.clip(1.0 - normalized, 0.0, None))
    result = float(base_z) + (float(peak_z) - float(base_z)) * profile
    if np.ndim(x_value) == 0 and np.ndim(y_value) == 0:
        return float(result)
    return result


def obstacle_label_from_pose(block_pos: np.ndarray) -> int:
    x_value = float(block_pos[0])
    y_value = float(block_pos[1])
    z_value = float(block_pos[2])
    normalized = ((x_value - float(OBSTACLE_CENTER_X)) / float(OBSTACLE_RADIUS_X)) ** 2 + (
        (y_value - float(OBSTACLE_CENTER_Y)) / float(OBSTACLE_RADIUS_Y)
    ) ** 2
    if normalized > 1.0:
        return 0
    ceiling = float(
        half_ellipsoid_height(
            x_value,
            y_value,
            center_x=OBSTACLE_CENTER_X,
            center_y=OBSTACLE_CENTER_Y,
            radius_x=OBSTACLE_RADIUS_X,
            radius_y=OBSTACLE_RADIUS_Y,
            base_z=OBSTACLE_BASE_Z,
            peak_z=OBSTACLE_PEAK_Z,
        )
    )
    return int((z_value > float(TABLE_Z)) and (z_value <= ceiling))


def hide_target_cube(env: gymnasium.Env) -> None:
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0


def render_without_target_cube(env: gymnasium.Env, camera: str) -> np.ndarray:
    hide_target_cube(env)
    return np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)


def make_env(args: argparse.Namespace) -> gymnasium.Env:
    physics_timestep, control_timestep, _ = compute_env_timing(args.sim_freq_hz, args.control_decimation)
    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=False,
        max_episode_steps=args.max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=args.width,
        height=args.height,
    )
    apply_xy_sampling_bounds(env)
    return env


def capture_grasp_reference(env: gymnasium.Env, args: argparse.Namespace) -> tuple[GraspReference, dict[str, np.ndarray]]:
    oracle = LocalCubePlanOracle(
        env=env,
        segment_dt=args.oracle_segment_dt,
        noise=args.oracle_noise,
        noise_smoothing=args.oracle_noise_smoothing,
    )
    ob, info = sample_valid_reset(
        env,
        args.seed,
        min_sampling_dist=0.10,
        mujoco_module=mujoco,
        lie_module=lie,
    )
    oracle.reset(ob, info)
    current_info = info
    for _ in range(int(args.max_oracle_steps)):
        if cube_is_grasped(
            current_info,
            contact_threshold=args.grasp_contact_threshold,
            alignment_threshold=args.grasp_alignment_threshold,
        ):
            break
        action = np.asarray(oracle.select_action(ob, current_info), dtype=np.float32)
        ob, _, terminated, truncated, current_info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Oracle episode ended before a grasp reference was found.")
    else:
        raise RuntimeError("Oracle did not produce a valid grasp reference within max-oracle-steps.")

    target_block = int(current_info["privileged/target_block"])
    block_pos = np.asarray(current_info[f"privileged/block_{target_block}_pos"], dtype=np.float64)
    effector_pos = np.asarray(current_info["proprio/effector_pos"], dtype=np.float64)
    block_yaw = float(current_info[f"privileged/block_{target_block}_yaw"][0])
    effector_yaw = float(current_info["proprio/effector_yaw"][0])
    rot_world_block = rotation_z(block_yaw)
    delta_local = rot_world_block.T @ (effector_pos - block_pos)

    reference = GraspReference(
        block_pos=block_pos.tolist(),
        block_yaw=block_yaw,
        effector_pos=effector_pos.tolist(),
        effector_yaw=effector_yaw,
        gripper_opening=float(current_info["proprio/gripper_opening"][0]),
        delta_local=delta_local.tolist(),
        delta_yaw=float(np.arctan2(np.sin(effector_yaw - block_yaw), np.cos(effector_yaw - block_yaw))),
        qpos=np.asarray(current_info["qpos"], dtype=np.float64).tolist(),
        qvel=np.asarray(current_info["qvel"], dtype=np.float64).tolist(),
        control=np.asarray(current_info["control"], dtype=np.float64).tolist(),
    )
    return reference, current_info


def cube_is_grasped(
    info: dict[str, np.ndarray],
    *,
    contact_threshold: float,
    alignment_threshold: float,
) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float64)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float64)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_threshold and block_alignment <= alignment_threshold)


def sample_inside_pose(rng: np.random.Generator, height_margin: float) -> tuple[np.ndarray, float]:
    z_min = float(TABLE_Z + height_margin)
    for _ in range(DEFAULT_INSIDE_SAMPLE_ATTEMPTS):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius_scale = float(np.sqrt(rng.uniform(0.0, 1.0)))
        x = float(OBSTACLE_CENTER_X + OBSTACLE_RADIUS_X * radius_scale * np.cos(angle))
        y = float(OBSTACLE_CENTER_Y + OBSTACLE_RADIUS_Y * radius_scale * np.sin(angle))
        z_ceiling = float(
            half_ellipsoid_height(
                x,
                y,
                center_x=OBSTACLE_CENTER_X,
                center_y=OBSTACLE_CENTER_Y,
                radius_x=OBSTACLE_RADIUS_X,
                radius_y=OBSTACLE_RADIUS_Y,
                base_z=OBSTACLE_BASE_Z,
                peak_z=OBSTACLE_PEAK_Z,
            )
        )
        z_max = z_ceiling - float(height_margin)
        if z_max <= z_min:
            continue
        z = float(rng.uniform(z_min, z_max))
        yaw = float(rng.uniform(float(THETA_SAMPLING_BOUNDS[0]), float(THETA_SAMPLING_BOUNDS[1])))
        return np.array([x, y, z], dtype=np.float64), yaw
    raise RuntimeError(
        "Could not sample a feasible inside pose under the obstacle. "
        f"Try reducing --height-margin below {float(height_margin):.4f}."
    )


def sample_outside_pose(
    rng: np.random.Generator,
    height_margin: float,
    outside_y_prob: float,
) -> tuple[np.ndarray, float]:
    x_bounds = (float(XY_SAMPLING_BOUNDS[0, 0]), float(XY_SAMPLING_BOUNDS[1, 0]))
    z_bounds = (float(Z_SAMPLING_BOUNDS[0]), float(Z_SAMPLING_BOUNDS[1]))

    outside_xy_possible = True
    cap_clearance_possible = z_bounds[1] > float(TABLE_Z + height_margin)

    if not outside_xy_possible and not cap_clearance_possible:
        raise RuntimeError("No feasible non-obstacle sampling strategy is available.")

    if outside_xy_possible and cap_clearance_possible:
        strategy = "outside_xy_footprint" if float(rng.uniform()) < float(outside_y_prob) else "above_cap"
    elif outside_xy_possible:
        strategy = "outside_xy_footprint"
    else:
        strategy = "above_cap"

    if strategy == "outside_xy_footprint":
        for _ in range(DEFAULT_INSIDE_SAMPLE_ATTEMPTS):
            x = float(rng.uniform(*x_bounds))
            y = float(rng.uniform(float(XY_SAMPLING_BOUNDS[0, 1]), float(XY_SAMPLING_BOUNDS[1, 1])))
            normalized = ((x - float(OBSTACLE_CENTER_X)) / float(OBSTACLE_RADIUS_X)) ** 2 + (
                (y - float(OBSTACLE_CENTER_Y)) / float(OBSTACLE_RADIUS_Y)
            ) ** 2
            if normalized > 1.0:
                z = float(rng.uniform(*z_bounds))
                break
        else:
            raise RuntimeError("Could not sample a feasible non-obstacle pose outside the obstacle footprint.")
    else:
        for _ in range(DEFAULT_INSIDE_SAMPLE_ATTEMPTS):
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            radius_scale = float(np.sqrt(rng.uniform(0.0, 1.0)))
            x = float(OBSTACLE_CENTER_X + OBSTACLE_RADIUS_X * radius_scale * np.cos(angle))
            y = float(OBSTACLE_CENTER_Y + OBSTACLE_RADIUS_Y * radius_scale * np.sin(angle))
            z_min = float(
                half_ellipsoid_height(
                    x,
                    y,
                    center_x=OBSTACLE_CENTER_X,
                    center_y=OBSTACLE_CENTER_Y,
                    radius_x=OBSTACLE_RADIUS_X,
                    radius_y=OBSTACLE_RADIUS_Y,
                    base_z=OBSTACLE_BASE_Z,
                    peak_z=OBSTACLE_PEAK_Z,
                )
                + float(height_margin)
            )
            if z_min >= z_bounds[1]:
                continue
            z = float(rng.uniform(max(z_bounds[0], z_min), z_bounds[1]))
            break
        else:
            raise RuntimeError(
                "Could not sample a feasible non-obstacle pose above the obstacle. "
                f"Try reducing --height-margin below {float(height_margin):.4f}."
            )

    yaw = float(rng.uniform(float(THETA_SAMPLING_BOUNDS[0]), float(THETA_SAMPLING_BOUNDS[1])))
    return np.array([x, y, z], dtype=np.float64), yaw


def synthesize_grasped_state(
    env: gymnasium.Env,
    *,
    block_pos: np.ndarray,
    block_yaw: float,
    reference: GraspReference,
    settle_steps: int,
) -> dict[str, np.ndarray]:
    unwrapped = env.unwrapped
    reference_qpos = np.asarray(reference.qpos, dtype=np.float64)
    unwrapped._data.qpos[:] = reference_qpos
    unwrapped._data.qvel[:] = np.asarray(reference.qvel, dtype=np.float64)
    unwrapped._data.ctrl[:] = np.asarray(reference.control, dtype=np.float64)

    rot_world_block = rotation_z(block_yaw)
    delta_local = np.asarray(reference.delta_local, dtype=np.float64)
    effector_pos = np.asarray(block_pos, dtype=np.float64) + rot_world_block @ delta_local
    effector_yaw = float(block_yaw + reference.delta_yaw)

    effector_pose = lie.SE3.from_rotation_and_translation(
        rotation=lie.SO3.from_z_radians(effector_yaw) @ unwrapped._effector_down_rotation,
        translation=effector_pos,
    )
    attach_pose = effector_pose @ unwrapped._T_pa
    qpos_arm = unwrapped._ik.solve(
        pos=attach_pose.translation(),
        quat=attach_pose.rotation().wxyz,
        curr_qpos=reference_qpos[unwrapped._arm_joint_ids],
    )

    unwrapped._data.qpos[unwrapped._arm_joint_ids] = qpos_arm
    object_joint = unwrapped._data.joint("object_joint_0")
    object_joint.qpos[:3] = np.asarray(block_pos, dtype=np.float64)
    object_joint.qpos[3:] = np.asarray(lie.SO3.from_z_radians(block_yaw).wxyz, dtype=np.float64)
    unwrapped._data.qvel[:] = 0.0

    hide_target_cube(env)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()

    for _ in range(max(0, int(settle_steps))):
        current_info = unwrapped.get_step_info()
        action = np.zeros(5, dtype=np.float64)
        action[:3] = effector_pos - np.asarray(current_info["proprio/effector_pos"], dtype=np.float64)
        action[3] = effector_yaw - float(current_info["proprio/effector_yaw"][0])
        action[4] = float(reference.gripper_opening) - float(current_info["proprio/gripper_opening"][0])
        normalized_action = np.asarray(unwrapped.normalize_action(action), dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(normalized_action)
        if terminated or truncated:
            raise RuntimeError("Episode ended while settling synthesized grasp state.")
        hide_target_cube(env)

    return unwrapped.get_step_info()


def evaluate_sample(
    info: dict[str, np.ndarray],
    *,
    desired_block_pos: np.ndarray,
    desired_block_yaw: float,
    contact_threshold: float,
    alignment_threshold: float,
) -> dict[str, Any]:
    block_pos = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
    block_yaw = float(info["privileged/block_0_yaw"][0])
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
    gripper_contact = float(info["proprio/gripper_contact"][0])
    grasp_alignment_error = float(np.linalg.norm(block_pos - effector_pos))
    return {
        "block_pos": block_pos.astype(np.float32),
        "block_yaw": np.float32(block_yaw),
        "qpos": np.asarray(info["qpos"], dtype=np.float32),
        "qvel": np.asarray(info["qvel"], dtype=np.float32),
        "control": np.asarray(info["control"], dtype=np.float32),
        "gripper_contact": np.float32(gripper_contact),
        "grasp_alignment_error": np.float32(grasp_alignment_error),
        "block_pos_error": np.float32(np.linalg.norm(block_pos - desired_block_pos)),
        "block_yaw_error": np.float32(angular_distance(block_yaw, desired_block_yaw)),
        "actual_label": np.int64(obstacle_label_from_pose(block_pos)),
        "grasped": bool(gripper_contact >= contact_threshold and grasp_alignment_error <= alignment_threshold),
    }


def sample_balanced_dataset(
    env: gymnasium.Env,
    *,
    args: argparse.Namespace,
    reference: GraspReference,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    class_names = {1: "obstacle", 0: "non_obstacle"}
    accepted: dict[int, list[dict[str, Any]]] = {1: [], 0: []}
    attempt_counts = {1: 0, 0: 0}
    rejected_counts = {
        1: {"synthesis_failed": 0, "not_grasped": 0, "wrong_label": 0, "pose_error": 0},
        0: {"synthesis_failed": 0, "not_grasped": 0, "wrong_label": 0, "pose_error": 0},
    }

    for label in (1, 0):
        progress = tqdm(total=int(args.samples_per_class), desc=f"Sampling {class_names[label]}", unit="image")
        try:
            while len(accepted[label]) < int(args.samples_per_class):
                attempt_counts[label] += 1
                if attempt_counts[label] > int(args.max_attempts_per_class):
                    raise RuntimeError(
                        f"Exceeded max attempts while collecting {class_names[label]} samples: "
                        f"{attempt_counts[label]} attempts for {len(accepted[label])} accepted."
                    )

                env.reset(seed=args.seed + 100_000 * label + attempt_counts[label])
                desired_block_pos, desired_block_yaw = (
                    sample_inside_pose(rng, float(args.height_margin))
                    if label == 1
                    else sample_outside_pose(
                        rng,
                        float(args.height_margin),
                        float(args.non_obstacle_outside_y_prob),
                    )
                )
                try:
                    info = synthesize_grasped_state(
                        env,
                        block_pos=desired_block_pos,
                        block_yaw=desired_block_yaw,
                        reference=reference,
                        settle_steps=int(args.settle_steps),
                    )
                except RuntimeError:
                    rejected_counts[label]["synthesis_failed"] += 1
                    continue

                metrics = evaluate_sample(
                    info,
                    desired_block_pos=desired_block_pos,
                    desired_block_yaw=desired_block_yaw,
                    contact_threshold=float(args.grasp_contact_threshold),
                    alignment_threshold=float(args.grasp_alignment_threshold),
                )
                if int(metrics["actual_label"]) != int(label):
                    rejected_counts[label]["wrong_label"] += 1
                    continue
                if float(metrics["block_pos_error"]) > float(args.acceptance_pos_tol):
                    rejected_counts[label]["pose_error"] += 1
                    continue
                if args.require_yaw_match and float(metrics["block_yaw_error"]) > float(args.acceptance_yaw_tol):
                    rejected_counts[label]["pose_error"] += 1
                    continue
                if args.require_grasped and not metrics["grasped"]:
                    rejected_counts[label]["not_grasped"] += 1
                    continue

                frame = render_without_target_cube(env, str(args.camera))
                accepted[label].append(
                    {
                        "pixels": frame.astype(np.uint8),
                        "task_target": np.asarray(desired_block_pos, dtype=np.float32),
                        "yaw": np.float32(desired_block_yaw),
                        "label": np.int64(label),
                        **metrics,
                    }
                )
                progress.update(1)
        finally:
            progress.close()

    all_samples = accepted[1] + accepted[0]
    labels = np.asarray([sample["label"] for sample in all_samples], dtype=np.int64)

    dataset = {
        "pixels": np.stack([sample["pixels"] for sample in all_samples], axis=0).astype(np.uint8),
        "task_target": np.stack([sample["task_target"] for sample in all_samples], axis=0).astype(np.float32),
        "yaw": np.asarray([sample["yaw"] for sample in all_samples], dtype=np.float32),
        "label": labels.astype(np.int64),
        "block_pos": np.stack([sample["block_pos"] for sample in all_samples], axis=0).astype(np.float32),
        "block_yaw": np.asarray([sample["block_yaw"] for sample in all_samples], dtype=np.float32),
        "qpos": np.stack([sample["qpos"] for sample in all_samples], axis=0).astype(np.float32),
        "qvel": np.stack([sample["qvel"] for sample in all_samples], axis=0).astype(np.float32),
        "control": np.stack([sample["control"] for sample in all_samples], axis=0).astype(np.float32),
        "gripper_contact": np.asarray([sample["gripper_contact"] for sample in all_samples], dtype=np.float32),
        "grasp_alignment_error": np.asarray(
            [sample["grasp_alignment_error"] for sample in all_samples],
            dtype=np.float32,
        ),
        "block_pos_error": np.asarray([sample["block_pos_error"] for sample in all_samples], dtype=np.float32),
        "block_yaw_error": np.asarray([sample["block_yaw_error"] for sample in all_samples], dtype=np.float32),
    }
    stats = {
        "attempt_counts": attempt_counts,
        "rejected_counts": rejected_counts,
    }
    return dataset, stats


def save_balanced_dataset_diagnostic(
    out_path: Path,
    block_pos: np.ndarray,
    labels: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    obstacle_mask = labels == 1
    safe_mask = ~obstacle_mask

    ax.scatter(
        block_pos[safe_mask, 0],
        block_pos[safe_mask, 1],
        s=18.0,
        c="#009e73",
        alpha=0.7,
        edgecolors="none",
        label="non-obstacle",
    )
    ax.scatter(
        block_pos[obstacle_mask, 0],
        block_pos[obstacle_mask, 1],
        s=18.0,
        c="#d55e00",
        alpha=0.75,
        edgecolors="none",
        label="obstacle",
    )
    theta = np.linspace(0.0, 2.0 * np.pi, num=256, dtype=np.float64)
    boundary_x = float(OBSTACLE_CENTER_X) + float(OBSTACLE_RADIUS_X) * np.cos(theta)
    boundary_y = float(OBSTACLE_CENTER_Y) + float(OBSTACLE_RADIUS_Y) * np.sin(theta)
    ax.fill(boundary_x, boundary_y, color="#d55e00", alpha=0.08, label="half-ellipsoid footprint")
    ax.plot(boundary_x, boundary_y, color="#4d4d4d", linestyle="--", linewidth=1.2, label="footprint boundary")
    ax.set_title("Balanced obstacle dataset by cube x/y position")
    ax.set_xlabel("cube x")
    ax.set_ylabel("cube y")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    try:
        reference, reference_info = capture_grasp_reference(env, args)
        reference_frame = render_without_target_cube(env, str(args.camera))
        dataset, stats = sample_balanced_dataset(env, args=args, reference=reference, rng=rng)
    finally:
        env.close()

    imageio.imwrite(out_dir / "reference_grasp.png", reference_frame)

    save_balanced_dataset_diagnostic(
        out_dir / DIAGNOSTIC_PLOT_NAME,
        dataset["block_pos"],
        dataset["label"],
    )

    torch.save(
        {
            "metadata": {
                "seed": int(args.seed),
                "camera": str(args.camera),
                "image_width": int(args.width),
                "image_height": int(args.height),
                "table_top_z": float(TABLE_Z),
                "task_xy_bounds": np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float32),
                "task_z_bounds": np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float32),
                "task_yaw_bounds": np.asarray(THETA_SAMPLING_BOUNDS, dtype=np.float32),
                "obstacle_profile": "half_3d_ellipsoid",
                "obstacle_base_z": float(OBSTACLE_BASE_Z),
                "obstacle_peak_z": float(OBSTACLE_PEAK_Z),
                "obstacle_center_xy": np.asarray([OBSTACLE_CENTER_X, OBSTACLE_CENTER_Y], dtype=np.float32),
                "obstacle_radius_xy": np.asarray([OBSTACLE_RADIUS_X, OBSTACLE_RADIUS_Y], dtype=np.float32),
                "samples_per_class": int(args.samples_per_class),
                "settle_steps": int(args.settle_steps),
                "acceptance_pos_tol": float(args.acceptance_pos_tol),
                "acceptance_yaw_tol": float(args.acceptance_yaw_tol),
                "require_grasped": bool(args.require_grasped),
                "require_yaw_match": bool(args.require_yaw_match),
                "non_obstacle_outside_y_prob": float(args.non_obstacle_outside_y_prob),
                "height_margin": float(args.height_margin),
                "balanced_total_count": int(dataset["label"].shape[0]),
            },
            "dataset": dataset,
            "reference_grasp": jsonable(reference),
        },
        out_dir / "obstacle_classifier_data_3d_ellipsoid.pt",
    )

    save_json(
        out_dir / "summary.json",
        {
            "out_dir": out_dir,
            "camera": str(args.camera),
            "counts": {
                "balanced_obstacle_count": int(np.sum(dataset["label"] == 1)),
                "balanced_non_obstacle_count": int(np.sum(dataset["label"] == 0)),
                "balanced_total_count": int(dataset["label"].shape[0]),
            },
            "sampling": {
                "samples_per_class": int(args.samples_per_class),
                "task_xy_bounds": np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float32),
                "task_z_bounds": np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float32),
                "task_yaw_bounds": np.asarray(THETA_SAMPLING_BOUNDS, dtype=np.float32),
                "obstacle_center_xy": [float(OBSTACLE_CENTER_X), float(OBSTACLE_CENTER_Y)],
                "obstacle_radius_xy": [float(OBSTACLE_RADIUS_X), float(OBSTACLE_RADIUS_Y)],
                "obstacle_peak_z": float(OBSTACLE_PEAK_Z),
                "height_margin": float(args.height_margin),
            },
            "acceptance": {
                "grasp_contact_threshold": float(args.grasp_contact_threshold),
                "grasp_alignment_threshold": float(args.grasp_alignment_threshold),
                "position_tolerance": float(args.acceptance_pos_tol),
                "yaw_tolerance": float(args.acceptance_yaw_tol),
                "require_grasped": bool(args.require_grasped),
                "require_yaw_match": bool(args.require_yaw_match),
                "non_obstacle_outside_y_prob": float(args.non_obstacle_outside_y_prob),
            },
            "attempt_counts": stats["attempt_counts"],
            "rejected_counts": stats["rejected_counts"],
            "quality": {
                "block_pos_error_mean": float(np.mean(dataset["block_pos_error"])),
                "block_pos_error_max": float(np.max(dataset["block_pos_error"])),
                "block_yaw_error_mean": float(np.mean(dataset["block_yaw_error"])),
                "block_yaw_error_max": float(np.max(dataset["block_yaw_error"])),
                "gripper_contact_mean": float(np.mean(dataset["gripper_contact"])),
                "grasp_alignment_error_mean": float(np.mean(dataset["grasp_alignment_error"])),
            },
            "reference_grasped": cube_is_grasped(
                reference_info,
                contact_threshold=float(args.grasp_contact_threshold),
                alignment_threshold=float(args.grasp_alignment_threshold),
            ),
        },
    )

    print(f"Saved diagnostic: {out_dir / DIAGNOSTIC_PLOT_NAME}")
    print(f"Saved dataset:    {out_dir / 'obstacle_classifier_data_3d_ellipsoid.pt'}")
    print(f"Saved summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
