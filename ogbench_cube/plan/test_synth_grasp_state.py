#!/usr/bin/env python3
"""Test direct synthesis of grasped OGBench cube states with IK."""

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
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie

from ogbench_cube.data.ogbench_cube_data_gen import (
    DEFAULT_CONTROL_DECIMATION,
    DEFAULT_ENV_NAME,
    DEFAULT_SIM_FREQ_HZ,
    LocalCubePlanOracle,
    Z_SAMPLING_BOUNDS,
    apply_xy_sampling_bounds,
    compute_env_timing,
    sample_valid_reset,
)

TABLE_Z = 0.02
OBSTACLE_BASE_Z = 0.0
OBSTACLE_PEAK_Z = 0.08
OBSTACLE_Y_BOUNDS = (-0.06, 0.06)
DEFAULT_CAMERA = "front_pixels"
DEFAULT_OUT_DIR = Path("ogbench_cube/plan/synth_grasp_test")
DEFAULT_NUM_INSIDE = 16
DEFAULT_NUM_OUTSIDE = 16
DEFAULT_MAX_ORACLE_STEPS = 80
DEFAULT_SETTLE_STEPS = 6
DEFAULT_HEIGHT_MARGIN = 0.005
DEFAULT_INSIDE_SAMPLE_ATTEMPTS = 1024


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
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--oracle-segment-dt", type=float, default=0.4)
    parser.add_argument("--oracle-noise", type=float, default=0.0)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=0.5)
    parser.add_argument("--max-oracle-steps", type=int, default=DEFAULT_MAX_ORACLE_STEPS)
    parser.add_argument("--num-inside", type=int, default=DEFAULT_NUM_INSIDE)
    parser.add_argument("--num-outside", type=int, default=DEFAULT_NUM_OUTSIDE)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--grasp-contact-threshold", type=float, default=0.5)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=0.03)
    parser.add_argument("--height-margin", type=float, default=DEFAULT_HEIGHT_MARGIN)
    return parser.parse_args()


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


def half_ellipse_height(
    y_value: float,
    *,
    y_bounds: tuple[float, float],
    base_z: float,
    peak_z: float,
) -> float:
    center_y = 0.5 * (float(y_bounds[0]) + float(y_bounds[1]))
    half_width = 0.5 * (float(y_bounds[1]) - float(y_bounds[0]))
    normalized = (float(y_value) - center_y) / half_width
    profile = np.sqrt(np.clip(1.0 - normalized**2, 0.0, None))
    return float(base_z) + (float(peak_z) - float(base_z)) * float(profile)


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


def sample_inside_pose(rng: np.random.Generator, env: gymnasium.Env, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    bounds = np.asarray(env.unwrapped._object_sampling_bounds, dtype=np.float64)
    z_min = TABLE_Z + float(args.height_margin)
    for _ in range(DEFAULT_INSIDE_SAMPLE_ATTEMPTS):
        x = float(rng.uniform(bounds[0, 0], bounds[1, 0]))
        y = float(rng.uniform(OBSTACLE_Y_BOUNDS[0], OBSTACLE_Y_BOUNDS[1]))
        z_ceiling = half_ellipse_height(
            y,
            y_bounds=OBSTACLE_Y_BOUNDS,
            base_z=OBSTACLE_BASE_Z,
            peak_z=OBSTACLE_PEAK_Z,
        )
        z_max = z_ceiling - float(args.height_margin)
        if z_max <= z_min:
            continue
        z = float(rng.uniform(z_min, z_max))
        yaw = float(rng.uniform(0.0, 2.0 * np.pi))
        return np.array([x, y, z], dtype=np.float64), yaw
    raise RuntimeError(
        "Could not sample a feasible inside pose under the obstacle. "
        f"Try reducing --height-margin below {float(args.height_margin):.4f}."
    )


def sample_outside_pose(rng: np.random.Generator, env: gymnasium.Env, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    del args
    bounds = np.asarray(env.unwrapped._object_sampling_bounds, dtype=np.float64)
    x = float(rng.uniform(bounds[0, 0], bounds[1, 0]))
    y_ranges = [
        (float(bounds[0, 1]), float(OBSTACLE_Y_BOUNDS[0])),
        (float(OBSTACLE_Y_BOUNDS[1]), float(bounds[1, 1])),
    ]
    valid_ranges = [(lo, hi) for lo, hi in y_ranges if hi > lo]
    widths = np.asarray([hi - lo for lo, hi in valid_ranges], dtype=np.float64)
    idx = int(rng.choice(len(valid_ranges), p=widths / widths.sum()))
    y = float(rng.uniform(valid_ranges[idx][0], valid_ranges[idx][1]))
    z_bounds = np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float64)
    z = float(rng.uniform(float(z_bounds[0]), float(z_bounds[1])))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
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
    unwrapped._data.qpos[:] = np.asarray(reference.qpos, dtype=np.float64)
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
        curr_qpos=np.asarray(reference.qpos, dtype=np.float64)[unwrapped._arm_joint_ids],
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
    obstacle_ceiling = half_ellipse_height(
        float(block_pos[1]),
        y_bounds=OBSTACLE_Y_BOUNDS,
        base_z=OBSTACLE_BASE_Z,
        peak_z=OBSTACLE_PEAK_Z,
    )
    return {
        "block_pos_error": float(np.linalg.norm(block_pos - desired_block_pos)),
        "block_yaw_error": angular_distance(block_yaw, desired_block_yaw),
        "grasp_alignment_error": float(np.linalg.norm(block_pos - effector_pos)),
        "gripper_contact": gripper_contact,
        "grasped": bool(
            gripper_contact >= float(contact_threshold)
            and np.linalg.norm(block_pos - effector_pos) <= float(alignment_threshold)
        ),
        "under_obstacle": bool(block_pos[1] >= OBSTACLE_Y_BOUNDS[0] and block_pos[1] <= OBSTACLE_Y_BOUNDS[1] and block_pos[2] <= obstacle_ceiling),
        "obstacle_ceiling_z": obstacle_ceiling,
        "block_pos": block_pos.tolist(),
        "block_yaw": block_yaw,
        "effector_pos": effector_pos.tolist(),
        "effector_yaw": float(info["proprio/effector_yaw"][0]),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    try:
        reference, reference_info = capture_grasp_reference(env, args)
        reference_frame = render_without_target_cube(env, args.camera)
        imageio.imwrite(args.out_dir / "reference_grasp.png", reference_frame)

        rng = np.random.default_rng(args.seed + 1)
        samples: list[dict[str, Any]] = []
        sample_specs = [("inside", int(args.num_inside)), ("outside", int(args.num_outside))]

        for label, count in sample_specs:
            for index in range(count):
                env.reset(seed=args.seed + 10_000 + len(samples))
                if label == "inside":
                    block_pos, block_yaw = sample_inside_pose(rng, env, args)
                else:
                    block_pos, block_yaw = sample_outside_pose(rng, env, args)

                info = synthesize_grasped_state(
                    env,
                    block_pos=block_pos,
                    block_yaw=block_yaw,
                    reference=reference,
                    settle_steps=args.settle_steps,
                )
                frame = render_without_target_cube(env, args.camera)
                file_name = f"{label}_{index:03d}.png"
                imageio.imwrite(args.out_dir / file_name, frame)
                metrics = evaluate_sample(
                    info,
                    desired_block_pos=block_pos,
                    desired_block_yaw=block_yaw,
                    contact_threshold=args.grasp_contact_threshold,
                    alignment_threshold=args.grasp_alignment_threshold,
                )
                metrics.update(
                    {
                        "label": label,
                        "index": index,
                        "file_name": file_name,
                        "desired_block_pos": block_pos.tolist(),
                        "desired_block_yaw": block_yaw,
                    }
                )
                samples.append(metrics)

        summary = {
            "args": vars(args),
            "reference": reference,
            "reference_grasped": cube_is_grasped(
                reference_info,
                contact_threshold=args.grasp_contact_threshold,
                alignment_threshold=args.grasp_alignment_threshold,
            ),
            "samples": samples,
        }
        with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(jsonable(summary), handle, indent=2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
