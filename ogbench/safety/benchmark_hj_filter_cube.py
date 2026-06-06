#!/usr/bin/env python3
"""Closed-loop OGBench cube benchmark with a latent HJ safety filter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from tqdm.auto import tqdm

from ogbench.plan import benchmark_cube_hard as hard
from ogbench.safety.hj_filter import OGBenchCubeHJSafetyFilter, parse_hidden_sizes

CubePlanOracle = hard.CubePlanOracle

DEFAULT_DATASET_PATH = "ogbench/data/test_data/ogbench_cube_test.h5"
DEFAULT_MODEL_DIR = "ogbench/models/mlpdyn"
DEFAULT_HJ_CACHE = "ogbench/safety/cache/cube_latent_safety_classifier_train_tanh2.pt"
DEFAULT_HJ_POLICY = "ogbench/safety/runs/pyhj_cube_train_tanh2/policy_latest.pth"
DEFAULT_CLASSIFIER = "ogbench/safety/obs_net/model.pt"
DEFAULT_OUT_DIR = "ogbench/safety/runs/closed_loop_hj_filter_cube"
DEFAULT_START_GOAL_PATH = "ogbench/experiments/cube_obstacle/ogbench_cube_stuff/ogbench_cube_stuff/start_goal.pt"
DEFAULT_HEIGHT_OBS_DATA = "ogbench/experiments/cube_obstacle/obstacle_data_3d_ellipsoid_front/height_classifier_data.pt"
OBSTACLE_BASE_Z = 0.0
OBSTACLE_PEAK_Z = 0.055
OBSTACLE_CENTER_X = 0.35
OBSTACLE_CENTER_Y = 0.0
OBSTACLE_RADIUS_X = 0.04
OBSTACLE_RADIUS_Y = 0.08
TABLE_Z = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-method", choices=("ilqr", "swm_cost", "swm_action"), default="ilqr")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--start-goal-path", type=Path, default=None)
    parser.add_argument(
        "--use-start-goal-pixels",
        action="store_true",
        help="When --start-goal-path provides pixels, use those stored start/goal images for initial WM/HJ embeddings.",
    )
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=10)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument(
        "--episode-indices-file",
        type=Path,
        default=None,
        help="JSON file containing an episode_indices list for deterministic held-out evaluation.",
    )
    parser.add_argument("--eval-budget", type=int, default=120)
    parser.add_argument("--cube-success-threshold", type=float, default=hard.DEFAULT_CUBE_SUCCESS_THRESHOLD)
    parser.add_argument("--success-mode", choices=("cube_distance", "ogbench_success"), default="cube_distance")
    parser.add_argument("--video-fps", type=int, default=hard.VIDEO_FPS)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("paired", "nominal", "filtered", "lpb", "hj_lpb", "all"),
        default="paired",
    )

    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR), help="Nominal iLQR cube world model.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=hard.HORIZON)
    parser.add_argument("--q-terminal", type=float, default=hard.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=hard.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=hard.R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--lpb-bank-path", type=Path, default=None)
    parser.add_argument("--lpb-weight", type=float, default=1.0)
    parser.add_argument("--lpb-threshold-scale", type=float, default=1.0)
    parser.add_argument("--lpb-stage-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--env-max-episode-steps", type=int, default=None)
    parser.add_argument("--max-oracle-steps", type=int, default=hard.MAX_ORACLE_STEPS)
    parser.add_argument("--oracle-segment-dt", type=float, default=hard.ORACLE_SEGMENT_DT)
    parser.add_argument("--oracle-noise", type=float, default=hard.ORACLE_NOISE)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=hard.ORACLE_NOISE_SMOOTHING)
    parser.add_argument("--grasp-contact-threshold", type=float, default=hard.GRASP_CONTACT_THRESHOLD)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=hard.GRASP_ALIGNMENT_THRESHOLD)
    parser.add_argument(
        "--constant-grasp-action",
        type=float,
        default=None,
        help="If set, clamp raw action[4] to this value during policy/HJ execution after oracle grasp.",
    )

    parser.add_argument("--hj-cache-path", type=Path, default=Path(DEFAULT_HJ_CACHE))
    parser.add_argument("--hj-policy-path", type=Path, default=Path(DEFAULT_HJ_POLICY))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER))
    parser.add_argument("--classifier-threshold", default="base")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="auto")
    parser.add_argument(
        "--geometry-rule",
        choices=("ellipsoid", "height"),
        default="ellipsoid",
        help="Ground-truth safety rule used for closed-loop violation accounting.",
    )
    parser.add_argument(
        "--height-obstacle-data-path",
        type=Path,
        default=Path(DEFAULT_HEIGHT_OBS_DATA),
        help="Height obstacle payload whose metadata contains height_threshold.",
    )
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=None,
        help="Unsafe iff proprio/effector_pos[2] is greater than this value. Defaults to height obstacle metadata.",
    )
    parser.add_argument("--hj-model-dir", type=Path, default=None, help="Safety WM override. Defaults to cache metadata.")
    parser.add_argument("--hj-checkpoint", type=Path, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
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


def save_case_summary(case_dir: Path, summary: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")


def reset_ilqr_warm_start(ilqr_policy: hard.ILQRPolicyAdapter) -> None:
    ilqr_policy.solver.prev_u_guess.zero_()


def half_ellipsoid_height(x_value: np.ndarray, y_value: np.ndarray) -> np.ndarray:
    normalized = ((np.asarray(x_value) - OBSTACLE_CENTER_X) / OBSTACLE_RADIUS_X) ** 2 + (
        (np.asarray(y_value) - OBSTACLE_CENTER_Y) / OBSTACLE_RADIUS_Y
    ) ** 2
    profile = np.sqrt(np.clip(1.0 - normalized, 0.0, None))
    return OBSTACLE_BASE_Z + (OBSTACLE_PEAK_Z - OBSTACLE_BASE_Z) * profile


def block_geometry_unsafe(block_pos: np.ndarray) -> bool:
    pos = np.asarray(block_pos, dtype=np.float32).reshape(-1)
    normalized = ((float(pos[0]) - OBSTACLE_CENTER_X) / OBSTACLE_RADIUS_X) ** 2 + (
        (float(pos[1]) - OBSTACLE_CENTER_Y) / OBSTACLE_RADIUS_Y
    ) ** 2
    if normalized > 1.0:
        return False
    ceiling = float(half_ellipsoid_height(pos[0], pos[1]))
    return bool(float(pos[2]) > TABLE_Z and float(pos[2]) <= ceiling)


def load_height_threshold(path: Path) -> float:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if "height_threshold" not in metadata:
        raise KeyError(f"{path} does not contain metadata['height_threshold'].")
    return float(metadata["height_threshold"])


def height_geometry_unsafe(info: dict[str, Any], threshold: float) -> bool:
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32).reshape(-1)
    return bool(float(effector_pos[2]) > float(threshold))


def geometry_unsafe(args: argparse.Namespace, info: dict[str, Any]) -> bool:
    if args.geometry_rule == "height":
        if args.height_threshold is None:
            raise ValueError("--height-threshold must be resolved before height geometry evaluation.")
        return height_geometry_unsafe(info, float(args.height_threshold))
    return block_geometry_unsafe(info["privileged/block_0_pos"])


def load_start_goal_pairs(path: Path, num_eval: int) -> list[dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "start" not in payload or "goal" not in payload:
        raise TypeError(f"Expected {path} to contain top-level start/goal dictionaries.")
    pair_count = int(payload.get("metadata", {}).get("pair_count", 0) or next(iter(payload["start"].values())).shape[0])
    if num_eval > pair_count:
        raise ValueError(f"Requested {num_eval} pairs, but {path} only contains {pair_count}.")
    pairs = []
    for idx in range(num_eval):
        pairs.append(
            {
                "metadata": payload.get("metadata", {}),
                "start": {key: value[idx] for key, value in payload["start"].items()},
                "goal": {key: value[idx] for key, value in payload["goal"].items()},
            }
        )
    return pairs


def synthesize_qpos_qvel_from_block_pose(env: gymnasium.Env, pos: np.ndarray, yaw: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    joint_qpos = unwrapped._data.joint("object_joint_0").qpos
    joint_qpos[:3] = np.asarray(pos, dtype=np.float64)
    joint_qpos[3:] = np.asarray(hard.lie.SO3.from_z_radians(float(yaw)).wxyz, dtype=np.float64)
    unwrapped.pre_step()
    hard.mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return (
        np.asarray(unwrapped._data.qpos, dtype=np.float32).copy(),
        np.zeros_like(np.asarray(unwrapped._data.qvel, dtype=np.float32)),
    )


def pair_pixel(pair: dict[str, Any], endpoint: str) -> np.ndarray:
    value = pair[endpoint].get("pixels")
    if value is None:
        raise KeyError(f"--use-start-goal-pixels requires start_goal pair {endpoint!r} to contain 'pixels'.")
    pixels = np.asarray(value, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected {endpoint} pixels with shape [H, W, 3], got {pixels.shape}.")
    return pixels


def goal_success(args: argparse.Namespace, info: dict[str, Any], dist: float) -> bool:
    if args.success_mode == "ogbench_success":
        success = info.get("success", False)
        if isinstance(success, dict):
            return all(bool(value) for value in success.values())
        return bool(np.asarray(success).item())
    return bool(float(dist) <= float(args.cube_success_threshold))


def run_one_case(
    *,
    args: argparse.Namespace,
    case: hard.EvalCase,
    case_idx: int,
    run_dir: Path,
    label: str,
    use_filter: bool,
    use_lpb: bool,
    device: Any,
    ilqr_assets: tuple[Any, ...],
    hj_filter: OGBenchCubeHJSafetyFilter,
    start_goal_pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if start_goal_pair is None:
        episode = hard.load_dataset_episode(args.dataset_path, case.episode_idx)
        qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
        qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
        target_block_pos_np = np.asarray(episode["target_block_pos"], dtype=np.float32)
        target_block_yaw_np = np.asarray(episode["target_block_yaw"], dtype=np.float32)
        episode_seed = int(episode["episode_seed"])
        env_name = str(episode["env_name"])
        camera = str(episode["camera"])
        width = int(episode["width"])
        height = int(episode["height"])
        physics_timestep = float(episode["physics_timestep"])
        control_timestep = float(episode["control_timestep"])
        max_episode_steps = (
            int(args.env_max_episode_steps)
            if args.env_max_episode_steps is not None
            else max(int(episode["max_episode_steps"]), int(args.max_oracle_steps) + int(args.eval_budget) + 1)
        )
    else:
        metadata = start_goal_pair.get("metadata", {})
        episode_seed = int(metadata.get("episode_seed", args.seed)) + int(case_idx)
        env_name = str(metadata.get("env_name", "cube-single-v0"))
        camera = str(metadata.get("camera", "front_pixels"))
        width = int(metadata.get("image_width", 224))
        height = int(metadata.get("image_height", 224))
        physics_timestep = 1.0 / 500.0
        control_timestep = 25.0 / 500.0
        max_episode_steps = (
            int(args.env_max_episode_steps)
            if args.env_max_episode_steps is not None
            else int(args.max_oracle_steps) + int(args.eval_budget) + 8
        )

    case_dir = run_dir / label / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = hard.make_env(
        env_name=env_name,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        max_episode_steps=max_episode_steps,
        width=width,
        height=height,
    )
    oracle = CubePlanOracle(
        env=env,
        segment_dt=args.oracle_segment_dt,
        noise=args.oracle_noise,
        noise_smoothing=args.oracle_noise_smoothing,
    )

    if start_goal_pair is None:
        start_qpos = qpos_np[case.start_step]
        start_qvel = qvel_np[case.start_step]
        goal_qpos = qpos_np[case.goal_step]
        goal_qvel = qvel_np[case.goal_step]
        start_target_pos = target_block_pos_np[case.start_step]
        start_target_yaw = float(target_block_yaw_np[case.start_step, 0])
        goal_target_pos = target_block_pos_np[case.goal_step]
        goal_target_yaw = float(target_block_yaw_np[case.goal_step, 0])
    else:
        start = start_goal_pair["start"]
        goal = start_goal_pair["goal"]
        start_qpos, start_qvel = synthesize_qpos_qvel_from_block_pose(
            env, np.asarray(start["task_target"], dtype=np.float32), float(np.asarray(start["yaw"]).reshape(-1)[0]), episode_seed
        )
        goal_qpos, goal_qvel = synthesize_qpos_qvel_from_block_pose(
            env, np.asarray(goal["task_target"], dtype=np.float32), float(np.asarray(goal["yaw"]).reshape(-1)[0]), episode_seed
        )
        start_target_pos = np.asarray(goal["task_target"], dtype=np.float32)
        start_target_yaw = float(np.asarray(goal["yaw"]).reshape(-1)[0])
        goal_target_pos = np.asarray(goal["task_target"], dtype=np.float32)
        goal_target_yaw = float(np.asarray(goal["yaw"]).reshape(-1)[0])

    start_frame, _, _ = hard.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=start_qvel,
        target_block_pos=start_target_pos,
        target_block_yaw=start_target_yaw,
        camera=camera,
    )
    goal_frame, goal_info, _ = hard.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=goal_qpos,
        qvel=goal_qvel,
        target_block_pos=goal_target_pos,
        target_block_yaw=goal_target_yaw,
        camera=camera,
    )
    current_frame, current_info, current_obs = hard.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=start_qvel,
        target_block_pos=start_target_pos,
        target_block_yaw=start_target_yaw,
        camera=camera,
    )
    if start_goal_pair is not None and args.use_start_goal_pixels:
        start_frame = pair_pixel(start_goal_pair, "start")
        goal_frame = pair_pixel(start_goal_pair, "goal")
        current_frame = start_frame.copy()
    hard.save_rgb_image(case_dir / "start_image.png", start_frame)
    hard.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    goal_block_pos = np.asarray(goal_info["privileged/target_block_pos"], dtype=np.float32)
    goal_block_yaw = float(goal_info["privileged/target_block_yaw"][0])
    rollout_frames = [current_frame.copy()]

    ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
    ilqr_policy.set_lpb_enabled(use_lpb)
    reset_ilqr_warm_start(ilqr_policy)
    start_emb = hard.encode_single_frame(
        ilqr_model,
        current_frame,
        device=device,
        img_size=int(ilqr_config["img_size"]),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    goal_emb = hard.encode_single_frame(
        ilqr_model,
        goal_frame,
        device=device,
        img_size=int(ilqr_config["img_size"]),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    ilqr_policy.reset(start_embedding=start_emb, goal_embedding=goal_emb)
    hj_filter.reset(current_frame)

    initial_safety = hj_filter.evaluate_state(hj_filter.current_state(), "initial")
    cube_goal_distances = [hard.cube_distance(current_info, goal_block_pos)]
    cube_yaw_errors = [hard.cube_yaw_error(current_info, goal_block_yaw)]
    step_records: list[dict[str, Any]] = []
    block_pos_trajectory = [np.asarray(current_info["privileged/block_0_pos"], dtype=np.float32).tolist()]
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []
    overrides = 0
    learned_safety_violations = bool(initial_safety["initial_l"] <= 0.0)
    geometric_safety_violations = geometry_unsafe(args, current_info)
    oracle_steps_executed = 0
    policy_steps_executed = 0
    terminated = False
    truncated = False
    stop_reason = "eval_budget"

    oracle_grasped = hard.cube_is_grasped(
        current_info,
        contact_threshold=args.grasp_contact_threshold,
        alignment_threshold=args.grasp_alignment_threshold,
    )
    if not oracle_grasped:
        oracle.reset(None, current_info)
        for oracle_step in range(int(args.max_oracle_steps)):
            oracle_action = np.asarray(oracle.select_action(None, current_info), dtype=np.float32)
            _, _, terminated, truncated, current_info = env.step(oracle_action)
            current_obs = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
            del current_obs
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            rollout_frames.append(current_frame.copy())
            oracle_steps_executed += 1

            next_emb = hard.encode_single_frame(
                ilqr_model,
                current_frame,
                device=device,
                img_size=int(ilqr_config["img_size"]),
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            ilqr_policy.append_embedding(next_emb)
            hj_filter.append_frame(current_frame)

            post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
            learned_safety_violations = learned_safety_violations or bool(post["post_l"] <= 0.0)
            geometric_unsafe = geometry_unsafe(args, current_info)
            geometric_safety_violations = geometric_safety_violations or geometric_unsafe
            dist = hard.cube_distance(current_info, goal_block_pos)
            yaw_err = hard.cube_yaw_error(current_info, goal_block_yaw)
            cube_goal_distances.append(dist)
            cube_yaw_errors.append(yaw_err)
            block_pos_trajectory.append(np.asarray(current_info["privileged/block_0_pos"], dtype=np.float32).tolist())
            oracle_grasped = hard.cube_is_grasped(
                current_info,
                contact_threshold=args.grasp_contact_threshold,
                alignment_threshold=args.grasp_alignment_threshold,
            )
            step_records.append(
                {
                    "phase": "oracle",
                    "step": int(oracle_steps_executed),
                    "lpb_guided": bool(use_lpb),
                    "cube_goal_distance": float(dist),
                    "cube_yaw_error": float(yaw_err),
                    "oracle_grasped": bool(oracle_grasped),
                    **post,
                    "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
                    "post_geometry_unsafe": bool(geometric_unsafe),
                }
            )
            if oracle_grasped:
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break

    success = goal_success(args, current_info, float(np.min(cube_goal_distances)))
    if success:
        stop_reason = "goal_reached"

    if oracle_grasped and not (terminated or truncated) and not success:
        for policy_step in range(int(args.eval_budget)):
            nominal_raw, base_record = ilqr_policy.get_action()
            decision = hj_filter.filter_action(nominal_raw)
            record = dict(decision.record)
            would_override = bool(record["override"])
            if use_filter:
                action_raw = decision.action_raw
                action_norm = decision.action_norm
                overrides += int(would_override)
            else:
                action_raw = np.asarray(nominal_raw, dtype=np.float32)
                action_norm = hj_filter.raw_to_norm(action_raw)
                record["would_override"] = would_override
                record["override"] = False
                record["override_reason"] = "monitor_only_nominal_execution"
                record["executed_action_raw"] = action_raw.tolist()
                record["executed_action_norm"] = action_norm.tolist()
            if args.constant_grasp_action is not None:
                action_raw = np.asarray(action_raw, dtype=np.float32).copy()
                action_raw[4] = float(args.constant_grasp_action)
                action_norm = hj_filter.raw_to_norm(action_raw)
                record["constant_grasp_action"] = float(args.constant_grasp_action)
                record["executed_action_raw"] = action_raw.tolist()
                record["executed_action_norm"] = action_norm.tolist()

            _, _, terminated, truncated, current_info = env.step(action_raw)
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            rollout_frames.append(current_frame.copy())
            policy_steps_executed += 1

            next_emb = hard.encode_single_frame(
                ilqr_model,
                current_frame,
                device=device,
                img_size=int(ilqr_config["img_size"]),
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            ilqr_policy.append_embedding(next_emb)
            hj_filter.append_frame(current_frame)
            post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
            learned_safety_violations = learned_safety_violations or bool(post["post_l"] <= 0.0)
            geometric_unsafe = geometry_unsafe(args, current_info)
            geometric_safety_violations = geometric_safety_violations or geometric_unsafe

            dist = hard.cube_distance(current_info, goal_block_pos)
            yaw_err = hard.cube_yaw_error(current_info, goal_block_yaw)
            cube_goal_distances.append(dist)
            cube_yaw_errors.append(yaw_err)
            block_pos_trajectory.append(np.asarray(current_info["privileged/block_0_pos"], dtype=np.float32).tolist())
            executed_actions_raw.append(np.asarray(action_raw, dtype=np.float32).tolist())
            executed_actions_norm.append(np.asarray(action_norm, dtype=np.float32).tolist())
            success = goal_success(args, current_info, dist)
            step_records.append(
                {
                    "phase": "policy",
                    "step": int(oracle_steps_executed + policy_step + 1),
                    "policy_step": int(policy_step + 1),
                    "hj_filter_enabled": bool(use_filter),
                    "lpb_guided": bool(use_lpb),
                    "cube_goal_distance": float(dist),
                    "cube_yaw_error": float(yaw_err),
                    **base_record,
                    **record,
                    **post,
                    "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
                    "post_geometry_unsafe": bool(geometric_unsafe),
                }
            )
            if success:
                stop_reason = "goal_reached"
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break
        if policy_steps_executed >= int(args.eval_budget) and not success and not (terminated or truncated):
            stop_reason = "eval_budget"
    elif not oracle_grasped and not (terminated or truncated) and not success:
        stop_reason = "oracle_failed_to_grasp"

    video_path = None
    if not args.no_videos:
        video_path = str(hard.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
    env.close()

    min_l = min([initial_safety["initial_l"]] + [float(item["post_l"]) for item in step_records if "post_l" in item])
    min_v = min([initial_safety["initial_V"]] + [float(item["post_V"]) for item in step_records if "post_V" in item])
    min_b = min([initial_safety["initial_B"]] + [float(item["post_B"]) for item in step_records if "post_B" in item])
    summary = {
        **hard.asdict(case),
        "label": label,
        "base_method": args.base_method,
        "hj_filter_enabled": bool(use_filter),
        "lpb_guided": bool(use_lpb),
        "success": bool(success),
        "safety_violation": bool(geometric_safety_violations),
        "geometric_safety_violation": bool(geometric_safety_violations),
        "learned_safety_violation": bool(learned_safety_violations),
        "success_metric": "cube_position_l2",
        "success_mode": str(args.success_mode),
        "cube_success_threshold": float(args.cube_success_threshold),
        "geometry_rule": str(args.geometry_rule),
        "height_threshold": None if args.height_threshold is None else float(args.height_threshold),
        "oracle_grasped": bool(oracle_grasped),
        "oracle_steps_executed": int(oracle_steps_executed),
        "policy_steps_executed": int(policy_steps_executed),
        "steps_executed": int(oracle_steps_executed + policy_steps_executed),
        "stop_reason": stop_reason,
        "episode_seed": int(episode_seed),
        "env_name": env_name,
        "camera": camera,
        "override_count": int(overrides),
        "override_rate": float(overrides / max(policy_steps_executed, 1)),
        "lpb_violation_rate_mean": float(
            np.mean([item["lpb_violation_rate"] for item in step_records if "lpb_violation_rate" in item])
        )
        if any("lpb_violation_rate" in item for item in step_records)
        else None,
        "lpb_distance_max": float(
            np.max([item["lpb_distance_max"] for item in step_records if "lpb_distance_max" in item])
        )
        if any("lpb_distance_max" in item for item in step_records)
        else None,
        "initial_l": float(initial_safety["initial_l"]),
        "initial_V": float(initial_safety["initial_V"]),
        "initial_B": float(initial_safety["initial_B"]),
        "min_l": float(min_l),
        "min_V": float(min_v),
        "min_B": float(min_b),
        "initial_cube_goal_distance": float(cube_goal_distances[0]),
        "final_cube_goal_distance": float(cube_goal_distances[-1]),
        "min_cube_goal_distance": float(np.min(cube_goal_distances)),
        "initial_cube_yaw_error": float(cube_yaw_errors[0]),
        "final_cube_yaw_error": float(cube_yaw_errors[-1]),
        "min_cube_yaw_error": float(np.min(cube_yaw_errors)),
        "goal_block_pos": goal_block_pos.tolist(),
        "goal_block_yaw": float(goal_block_yaw),
        "final_block_pos": np.asarray(current_info["privileged/block_0_pos"], dtype=np.float32).tolist(),
        "final_block_yaw": float(current_info["privileged/block_0_yaw"][0]),
        "video_path": video_path,
        "cube_goal_distances": cube_goal_distances,
        "cube_yaw_errors": cube_yaw_errors,
        "block_pos_trajectory": block_pos_trajectory,
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "step_records": step_records,
    }
    save_case_summary(case_dir, summary)
    return summary


def sample_eval_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[hard.EvalCase]:
    if args.start_goal_path is not None:
        if args.episode_idx is not None or args.episode_indices_file is not None:
            raise ValueError("--start-goal-path uses the first --num-eval pairs; do not combine with episode selectors.")
        return [hard.EvalCase(int(idx), 0, 1, 2) for idx in range(int(args.num_eval))]
    if args.episode_indices_file is None:
        return hard.sample_eval_cases(args, ep_len)
    if args.episode_idx is not None:
        raise ValueError("Use either --episode-idx or --episode-indices-file, not both.")

    path = args.episode_indices_file.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_indices = payload.get("episode_indices", payload) if isinstance(payload, dict) else payload
    if not isinstance(episode_indices, list) or not episode_indices:
        raise ValueError(f"{path} must contain a non-empty episode_indices list.")

    cases: list[hard.EvalCase] = []
    for item in episode_indices:
        episode_idx = int(item)
        if episode_idx < 0 or episode_idx >= int(ep_len.shape[0]):
            raise ValueError(f"Episode index {episode_idx} is outside [0, {ep_len.shape[0] - 1}].")
        if int(ep_len[episode_idx]) < 2:
            raise ValueError(f"Episode {episode_idx} has length {int(ep_len[episode_idx])}, expected at least 2.")
        cases.append(hard.EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx])))
    args.num_eval = len(cases)
    return cases


def aggregate(label: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"label": label, "num_eval": 0}
    successes = np.asarray([case["success"] for case in cases], dtype=bool)
    violations = np.asarray([case["safety_violation"] for case in cases], dtype=bool)
    overrides = np.asarray([case["override_count"] for case in cases], dtype=np.float64)
    policy_steps = np.asarray([case["policy_steps_executed"] for case in cases], dtype=np.float64)
    return {
        "label": label,
        "num_eval": int(len(cases)),
        "success_rate": float(np.mean(successes) * 100.0),
        "safety_violation_rate": float(np.mean(violations) * 100.0),
        "mean_override_count": float(np.mean(overrides)),
        "mean_override_rate": float(np.sum(overrides) / max(np.sum(policy_steps), 1.0)),
        "mean_min_l": float(np.mean([case["min_l"] for case in cases])),
        "mean_min_V": float(np.mean([case["min_V"] for case in cases])),
        "mean_min_B": float(np.mean([case["min_B"] for case in cases])),
        "mean_final_cube_goal_distance": float(np.mean([case["final_cube_goal_distance"] for case in cases])),
        "mean_lpb_violation_rate": None
        if not any(case.get("lpb_violation_rate_mean") is not None for case in cases)
        else float(np.mean([case["lpb_violation_rate_mean"] for case in cases if case.get("lpb_violation_rate_mean") is not None])),
        "max_lpb_distance": None
        if not any(case.get("lpb_distance_max") is not None for case in cases)
        else float(np.max([case["lpb_distance_max"] for case in cases if case.get("lpb_distance_max") is not None])),
    }


def main() -> None:
    args = parse_args()
    if args.base_method != "ilqr":
        raise NotImplementedError("Cube HJ benchmark currently supports --base-method ilqr.")
    args.method = args.base_method
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if args.start_goal_path is None and not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.geometry_rule == "height" and args.height_threshold is None:
        args.height_obstacle_data_path = args.height_obstacle_data_path.expanduser().resolve()
        args.height_threshold = load_height_threshold(args.height_obstacle_data_path)
    start_goal_pairs = None
    if args.start_goal_path is not None:
        args.start_goal_path = args.start_goal_path.expanduser().resolve()
        start_goal_pairs = load_start_goal_pairs(args.start_goal_path, int(args.num_eval))

    device = hard.require_device(args.device)
    ep_len = np.full((int(args.num_eval),), 2, dtype=np.int64) if args.start_goal_path is not None else hard.load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)

    run_name = f"{int(time.time())}_{args.base_method}_hj_seed_{args.seed}"
    run_root = args.out_dir.expanduser().resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    ilqr_assets = hard.load_ilqr_assets(args, device)
    method_config = ilqr_assets[2]
    hj_filter = OGBenchCubeHJSafetyFilter(
        cache_path=args.hj_cache_path,
        policy_path=args.hj_policy_path,
        classifier_checkpoint=args.classifier_checkpoint,
        device_arg=args.device,
        model_dir=args.hj_model_dir,
        checkpoint=args.hj_checkpoint,
        classifier_threshold=str(args.classifier_threshold),
        margin_transform=str(args.margin_transform),
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        action_low=float(args.action_low),
        action_high=float(args.action_high),
        epsilon=float(args.epsilon),
    )

    labels = {
        "paired": [("nominal", False, False), ("hj_filtered", True, False)],
        "nominal": [("nominal", False, False)],
        "filtered": [("hj_filtered", True, False)],
        "lpb": [("lpb_guided", False, True)],
        "hj_lpb": [("hj_lpb", True, True)],
        "all": [
            ("nominal", False, False),
            ("hj_filtered", True, False),
            ("lpb_guided", False, True),
            ("hj_lpb", True, True),
        ],
    }[args.mode]
    if any(use_lpb for _, _, use_lpb in labels) and args.lpb_bank_path is None:
        raise ValueError(f"--mode {args.mode} requires --lpb-bank-path.")

    all_results: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in labels}
    for case_idx, case in enumerate(tqdm(cases, desc="Closed-loop HJ cube eval")):
        for label, use_filter, use_lpb in labels:
            summary = run_one_case(
                args=args,
                case=case,
                case_idx=case_idx,
                run_dir=run_root,
                label=label,
                use_filter=use_filter,
                use_lpb=use_lpb,
                device=device,
                ilqr_assets=ilqr_assets,
                hj_filter=hj_filter,
                start_goal_pair=None if start_goal_pairs is None else start_goal_pairs[case_idx],
            )
            all_results[label].append(summary)
            partial = {
                "partial": True,
                "base_method": args.base_method,
                "method_config": method_config,
                "hj_filter": {
                    "cache_path": str(args.hj_cache_path),
                    "policy_path": str(args.hj_policy_path),
                    "classifier_checkpoint": str(args.classifier_checkpoint),
                    "epsilon": float(args.epsilon),
                    "margin_transform": str(args.margin_transform),
                },
                "geometry_rule": str(args.geometry_rule),
                "height_threshold": None if args.height_threshold is None else float(args.height_threshold),
                "lpb": method_config.get("lpb"),
                "evaluated_episode_indices": [int(item.episode_idx) for item in cases],
                "aggregates": {name: aggregate(name, result) for name, result in all_results.items()},
            }
            (run_root / "metrics_partial.json").write_text(json.dumps(jsonable(partial), indent=2), encoding="utf-8")

    metrics = {
        "partial": False,
        "base_method": args.base_method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "episode_indices_file": (
            str(args.episode_indices_file.expanduser().resolve()) if args.episode_indices_file is not None else None
        ),
        "seed": int(args.seed),
        "num_eval": len(cases),
        "eval_budget": int(args.eval_budget),
        "goal_protocol": "oracle_grasp_then_start_step_0_to_final_episode_step",
        "geometry_rule": str(args.geometry_rule),
        "height_threshold": None if args.height_threshold is None else float(args.height_threshold),
        "height_obstacle_data_path": str(args.height_obstacle_data_path) if args.geometry_rule == "height" else None,
        "hj_filter": {
            "cache_path": str(args.hj_cache_path.expanduser().resolve()),
            "policy_path": str(args.hj_policy_path.expanduser().resolve()),
            "classifier_checkpoint": str(args.classifier_checkpoint.expanduser().resolve()),
            "epsilon": float(args.epsilon),
            "margin_transform": str(args.margin_transform),
            "barrier": "min(classifier_margin, critic_value)",
            "switch_rule": "execute nominal iff predicted nominal next barrier > epsilon",
        },
        "lpb": method_config.get("lpb"),
        "evaluated_episode_indices": [int(item.episode_idx) for item in cases],
        "aggregates": {label: aggregate(label, result) for label, result in all_results.items()},
        "cases": all_results,
    }
    (run_root / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(run_root), "aggregates": metrics["aggregates"]}), indent=2))


if __name__ == "__main__":
    main()
