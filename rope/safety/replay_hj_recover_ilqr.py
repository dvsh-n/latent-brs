#!/usr/bin/env python3
"""Replay recorded rope actions, recover with HJ safety, then hand off to iLQR."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.plan import benchmark_rope_hard as hard
from rope.safety.benchmark_hj_filter_rope import DEFAULT_CLASSIFIER, DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, jsonable
from rope.safety.hj_filter import RopeHJSafetyFilter, parse_hidden_sizes
from rope.shared.lab_env import LabEnv

DEFAULT_HJ_CACHE = "rope/safety/cache/rope_latent_safety_classifier_train_noshadow.pt"
DEFAULT_HJ_POLICY = "rope/safety/runs/pyhj_train_noshadow/policy_latest.pth"
DEFAULT_OUT_DIR = "rope/safety/runs/replay_hj_recover_ilqr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--hj-cache-path", type=Path, default=Path(DEFAULT_HJ_CACHE))
    parser.add_argument("--hj-policy-path", type=Path, default=Path(DEFAULT_HJ_POLICY))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="identity")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--recovery-epsilon", type=float, default=0.0)
    parser.add_argument("--safe-steps-required", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, action="append", default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--min-first-unsafe-step", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--goal-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--replay-hj-source",
        choices=("dataset", "sim"),
        default="dataset",
        help=(
            "HJ state source during the recorded-action replay phase. "
            "'dataset' uses exact HDF5 frames so unsafe-cache episodes reproduce the classifier labels; "
            "'sim' uses live LabEnv renders."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-videos", action="store_true")

    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--q-terminal", type=float, default=5.0)
    parser.add_argument("--q-stage", type=float, default=0.005)
    parser.add_argument("--r-control", type=float, default=0.001)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)

    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=parse_hidden_sizes("128 128 128"))
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=parse_hidden_sizes("128 128 128"))
    return parser.parse_args()


def select_unsafe_episodes(args: argparse.Namespace) -> list[int]:
    cache = torch.load(args.hj_cache_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    episode_idx = cache["episode_idx"].detach().cpu().numpy().astype(np.int64)
    step_idx = cache["step_idx"].detach().cpu().numpy().astype(np.int64)
    margins = cache["safety_margin"].detach().cpu().numpy().astype(np.float32)

    if args.episode_idx is not None:
        return [int(item) for item in args.episode_idx]

    candidates: list[int] = []
    for episode in np.unique(episode_idx[margins <= 0.0]).tolist():
        unsafe_steps = step_idx[(episode_idx == int(episode)) & (margins <= 0.0)]
        if unsafe_steps.size and int(np.min(unsafe_steps)) >= int(args.min_first_unsafe_step):
            candidates.append(int(episode))
    if not candidates:
        raise ValueError("No unsafe candidate episodes found in cache.")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(np.asarray(candidates, dtype=np.int64), size=min(args.num_episodes, len(candidates)), replace=False)
    return [int(item) for item in np.sort(selected)]


def ilqr_append_frame(
    *,
    ilqr_assets: tuple[Any, ...],
    frame: np.ndarray,
    device: torch.device,
) -> None:
    ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
    next_emb = hard.encode_single_frame(
        ilqr_model,
        frame,
        device=device,
        img_size=int(ilqr_config["img_size"]),
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    ilqr_policy.append_embedding(next_emb)


def reset_ilqr(
    *,
    ilqr_assets: tuple[Any, ...],
    start_frame: np.ndarray,
    goal_frame: np.ndarray,
    device: torch.device,
) -> None:
    ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
    ilqr_policy.solver.prev_u_guess.zero_()
    start_emb = hard.encode_single_frame(
        ilqr_model,
        start_frame,
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


def run_episode(
    *,
    args: argparse.Namespace,
    episode_idx: int,
    case_idx: int,
    out_root: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
    device: torch.device,
    ilqr_assets: tuple[Any, ...],
    hj_filter: RopeHJSafetyFilter,
) -> dict[str, Any]:
    episode = hard.load_dataset_episode(args.dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
    actions_np = np.asarray(episode["action"], dtype=np.float32)
    task_target_np = np.asarray(episode["task_target"], dtype=np.float32)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    control_np = np.asarray(episode["control"], dtype=np.float32)
    time_np = np.asarray(episode["time"], dtype=np.float32)
    camera = str(episode["camera"])
    camera_id = env.model.camera(camera).id
    control_decimation = int(episode["control_decimation"])
    max_steps = min(int(args.max_steps), int(actions_np.shape[0]) - 1)

    case_dir = out_root / f"case_{case_idx:04d}_episode_{episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    start_frame, _ = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        control=control_np[0],
        task_target=task_target_np[0],
        camera_id=camera_id,
        elapsed_time=float(time_np[0, 0]),
    )
    goal_frame, goal_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[-1],
        qvel=qvel_np[-1],
        control=control_np[-1],
        task_target=task_target_np[-1],
        camera_id=camera_id,
        elapsed_time=float(time_np[-1, 0]),
    )
    current_frame, current_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        control=control_np[0],
        task_target=task_target_np[0],
        camera_id=camera_id,
        elapsed_time=float(time_np[0, 0]),
    )
    hard.save_rgb_image(case_dir / "start_image.png", start_frame)
    hard.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    video_writer = None
    if not args.no_videos:
        video_writer = hard.FFMpegRolloutWriter(case_dir / "rollout.mp4", fps=args.video_fps, first_frame=current_frame)

    reset_ilqr(ilqr_assets=ilqr_assets, start_frame=current_frame, goal_frame=goal_frame, device=device)
    if args.replay_hj_source == "dataset":
        hj_filter.reset(pixels_np[0])
    else:
        hj_filter.reset(current_frame)
    goal_task_target = np.asarray(goal_info["task_target"], dtype=np.float32)
    goal_left_pos = np.asarray(goal_info["left_attachment_pos"], dtype=np.float32)
    goal_right_pos = np.asarray(goal_info["right_attachment_pos"], dtype=np.float32)
    goal_rope_length = np.asarray(goal_info["rope_length"], dtype=np.float32)

    initial = hj_filter.evaluate_state(hj_filter.current_state(), "initial")
    phase = "dataset_replay"
    safe_streak = 0
    warning_step: int | None = None
    recovery_done_step: int | None = None
    overrides = 0
    recorded_replay_violation = bool(args.replay_hj_source == "dataset" and initial["initial_l"] <= 0.0)
    live_safety_violation = bool(args.replay_hj_source == "sim" and initial["initial_l"] <= 0.0)
    recorded_violation_before_warning = bool(recorded_replay_violation)
    late_warning = False
    safety_violation = bool(recorded_replay_violation or live_safety_violation)
    success = hard.task_target_distance(current_info, goal_task_target) <= float(args.goal_tolerance)
    stop_reason = "goal_tolerance_reached" if success else "max_steps"

    step_records: list[dict[str, Any]] = []
    task_target_distances = [hard.task_target_distance(current_info, goal_task_target)]
    left_attachment_distances = [hard.left_attachment_distance(current_info, goal_left_pos)]
    right_attachment_distances = [hard.right_attachment_distance(current_info, goal_right_pos)]
    rope_length_errors = [hard.rope_length_error(current_info, goal_rope_length)]
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []

    for step in range(0 if success else max_steps):
        step_phase = phase
        hj_state_source = "dataset" if phase == "dataset_replay" and args.replay_hj_source == "dataset" else "sim"
        state = hj_filter.current_state()
        current_eval = hj_filter.evaluate_state(state, "current")
        if hj_state_source == "dataset" and current_eval["current_l"] <= 0.0:
            recorded_replay_violation = True
            if warning_step is None:
                recorded_violation_before_warning = True
        elif hj_state_source == "sim" and current_eval["current_l"] <= 0.0:
            live_safety_violation = True

        if phase == "dataset_replay":
            nominal_raw = actions_np[step].astype(np.float32)
            decision = hj_filter.filter_action(nominal_raw)
            record = dict(decision.record)
            if bool(record["override"]):
                phase = "hj_recovery"
                step_phase = "dataset_replay_warning"
                warning_step = step + 1
                late_warning = bool(record.get("current_l", 1.0) <= 0.0 or recorded_violation_before_warning)
                action_raw = decision.action_raw
                action_norm = decision.action_norm
                overrides += 1
            else:
                action_raw = nominal_raw
                action_norm = hj_filter.raw_to_norm(action_raw)
                record["executed_action_raw"] = action_raw.tolist()
                record["executed_action_norm"] = action_norm.tolist()
        elif phase == "hj_recovery":
            safe_norm = hj_filter.safety_action_norm(state)
            safe_raw = hj_filter.norm_to_raw(safe_norm)
            safe_next = hj_filter.predict_next(state, safe_norm)
            safe_next_eval = hj_filter.evaluate_state(safe_next, "safe_next")
            action_raw = safe_raw
            action_norm = safe_norm
            overrides += 1
            record = {
                **current_eval,
                **safe_next_eval,
                "override": True,
                "override_reason": "hj_recovery_until_barrier_safe",
                "nominal_action_raw": None,
                "nominal_action_norm": None,
                "safety_action_raw": safe_raw.tolist(),
                "safety_action_norm": safe_norm.tolist(),
                "executed_action_raw": safe_raw.tolist(),
                "executed_action_norm": safe_norm.tolist(),
            }
        else:
            ilqr_policy = ilqr_assets[0]
            action_raw, ilqr_record = ilqr_policy.get_action()
            action_raw = action_raw.astype(np.float32)
            action_norm = hj_filter.raw_to_norm(action_raw)
            next_state = hj_filter.predict_next(state, action_norm)
            next_eval = hj_filter.evaluate_state(next_state, "ilqr_next")
            record = {
                **current_eval,
                **next_eval,
                **ilqr_record,
                "override": False,
                "override_reason": "ilqr_goal_phase",
                "nominal_action_raw": action_raw.tolist(),
                "nominal_action_norm": action_norm.tolist(),
                "safety_action_raw": None,
                "safety_action_norm": None,
                "executed_action_raw": action_raw.tolist(),
                "executed_action_norm": action_norm.tolist(),
            }

        current_time = float(current_info["time"][0]) + float(episode["control_timestep"])
        current_frame, current_info = hard.step_env_with_action(
            env,
            renderer,
            action=action_raw,
            control_decimation=control_decimation,
            camera_id=camera_id,
            elapsed_time=current_time,
        )
        if video_writer is not None:
            video_writer.write(current_frame)

        if step_phase == "dataset_replay" and args.replay_hj_source == "dataset" and step + 1 < pixels_np.shape[0]:
            hj_filter.append_frame(pixels_np[step + 1])
            post_hj_state_source = "dataset"
        elif step_phase == "dataset_replay_warning":
            hj_filter.reset(current_frame)
            post_hj_state_source = "sim_reset_after_override"
        else:
            hj_filter.append_frame(current_frame)
            post_hj_state_source = "sim"
        ilqr_append_frame(ilqr_assets=ilqr_assets, frame=current_frame, device=device)
        post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
        if post["post_l"] <= 0.0:
            if post_hj_state_source == "dataset":
                recorded_replay_violation = True
                if warning_step is None:
                    recorded_violation_before_warning = True
            else:
                live_safety_violation = True

        if phase == "hj_recovery":
            if post["post_l"] > 0.0 and post["post_B"] > float(args.recovery_epsilon):
                safe_streak += 1
            else:
                safe_streak = 0
            if safe_streak >= int(args.safe_steps_required):
                phase = "ilqr_goal"
                recovery_done_step = step + 1

        task_dist = hard.task_target_distance(current_info, goal_task_target)
        left_dist = hard.left_attachment_distance(current_info, goal_left_pos)
        right_dist = hard.right_attachment_distance(current_info, goal_right_pos)
        length_err = hard.rope_length_error(current_info, goal_rope_length)
        task_target_distances.append(task_dist)
        left_attachment_distances.append(left_dist)
        right_attachment_distances.append(right_dist)
        rope_length_errors.append(length_err)
        executed_actions_raw.append(np.asarray(action_raw, dtype=np.float32).tolist())
        executed_actions_norm.append(np.asarray(action_norm, dtype=np.float32).tolist())
        safety_violation = bool(recorded_replay_violation or live_safety_violation)
        success = task_dist <= float(args.goal_tolerance)

        step_records.append(
            {
                "step": int(step + 1),
                "dataset_step": int(step + 1),
                "phase": (
                    "hj_recovery_to_ilqr_handoff"
                    if recovery_done_step == step + 1
                    else ("dataset_replay_warning" if step_phase == "dataset_replay_warning" else phase)
                ),
                "hj_state_source": hj_state_source,
                "post_hj_state_source": post_hj_state_source,
                "task_target_distance": float(task_dist),
                "left_attachment_distance": float(left_dist),
                "right_attachment_distance": float(right_dist),
                "rope_length_error": float(length_err),
                **record,
                **post,
                "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
            }
        )

        if success:
            stop_reason = "goal_tolerance_reached"
            break

    video_path = None
    video_error = None
    if video_writer is not None:
        video, video_error = video_writer.close()
        video_path = None if video is None else str(video)

    min_l = min([initial["initial_l"]] + [float(item["post_l"]) for item in step_records])
    min_v = min([initial["initial_V"]] + [float(item["post_V"]) for item in step_records])
    min_b = min([initial["initial_B"]] + [float(item["post_B"]) for item in step_records])
    summary = {
        "episode_idx": int(episode_idx),
        "method": "replay_hj_recover_ilqr",
        "success": bool(success),
        "safety_violation": bool(safety_violation),
        "recorded_replay_violation": bool(recorded_replay_violation),
        "recorded_violation_before_warning": bool(recorded_violation_before_warning),
        "live_safety_violation": bool(live_safety_violation),
        "late_warning": bool(late_warning),
        "goal_tolerance": float(args.goal_tolerance),
        "stop_reason": stop_reason,
        "steps_executed": int(len(step_records)),
        "warning_step": warning_step,
        "recovery_done_step": recovery_done_step,
        "override_count": int(overrides),
        "override_rate": float(overrides / max(len(step_records), 1)),
        "initial_l": float(initial["initial_l"]),
        "initial_V": float(initial["initial_V"]),
        "initial_B": float(initial["initial_B"]),
        "min_l": float(min_l),
        "min_V": float(min_v),
        "min_B": float(min_b),
        "initial_task_target_distance": float(task_target_distances[0]),
        "final_task_target_distance": float(task_target_distances[-1]),
        "video_path": video_path,
        "video_error": video_error,
        "task_target_distances": task_target_distances,
        "left_attachment_distances": left_attachment_distances,
        "right_attachment_distances": right_attachment_distances,
        "rope_length_errors": rope_length_errors,
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - pixels_np[0].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - pixels_np[-1].astype(np.float32))),
        "step_records": step_records,
    }
    (case_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    return summary


def run_nominal_ilqr_episode(
    *,
    args: argparse.Namespace,
    episode_idx: int,
    case_idx: int,
    out_root: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
    device: torch.device,
    ilqr_assets: tuple[Any, ...],
    hj_filter: RopeHJSafetyFilter,
) -> dict[str, Any]:
    episode = hard.load_dataset_episode(args.dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"], np.uint8)
    task_target_np = np.asarray(episode["task_target"], np.float32)
    qpos_np = np.asarray(episode["qpos"], np.float32)
    qvel_np = np.asarray(episode["qvel"], np.float32)
    control_np = np.asarray(episode["control"], np.float32)
    time_np = np.asarray(episode["time"], np.float32)
    camera = str(episode["camera"])
    camera_id = env.model.camera(camera).id
    control_decimation = int(episode["control_decimation"])

    case_dir = out_root / f"case_{case_idx:04d}_episode_{episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    start_frame, _ = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        control=control_np[0],
        task_target=task_target_np[0],
        camera_id=camera_id,
        elapsed_time=float(time_np[0, 0]),
    )
    goal_frame, goal_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[-1],
        qvel=qvel_np[-1],
        control=control_np[-1],
        task_target=task_target_np[-1],
        camera_id=camera_id,
        elapsed_time=float(time_np[-1, 0]),
    )
    current_frame, current_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        control=control_np[0],
        task_target=task_target_np[0],
        camera_id=camera_id,
        elapsed_time=float(time_np[0, 0]),
    )
    hard.save_rgb_image(case_dir / "start_image.png", start_frame)
    hard.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    video_writer = None
    if not args.no_videos:
        video_writer = hard.FFMpegRolloutWriter(case_dir / "rollout.mp4", fps=args.video_fps, first_frame=current_frame)

    reset_ilqr(ilqr_assets=ilqr_assets, start_frame=current_frame, goal_frame=goal_frame, device=device)
    hj_filter.reset(current_frame)
    goal_task_target = np.asarray(goal_info["task_target"], np.float32)
    goal_left_pos = np.asarray(goal_info["left_attachment_pos"], np.float32)
    goal_right_pos = np.asarray(goal_info["right_attachment_pos"], np.float32)
    goal_rope_length = np.asarray(goal_info["rope_length"], np.float32)

    initial = hj_filter.evaluate_state(hj_filter.current_state(), "initial")
    live_safety_violation = bool(initial["initial_l"] <= 0.0)
    success = hard.task_target_distance(current_info, goal_task_target) <= float(args.goal_tolerance)
    stop_reason = "goal_tolerance_reached" if success else "max_steps"

    step_records: list[dict[str, Any]] = []
    task_target_distances = [hard.task_target_distance(current_info, goal_task_target)]
    left_attachment_distances = [hard.left_attachment_distance(current_info, goal_left_pos)]
    right_attachment_distances = [hard.right_attachment_distance(current_info, goal_right_pos)]
    rope_length_errors = [hard.rope_length_error(current_info, goal_rope_length)]
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []

    for step in range(0 if success else int(args.max_steps)):
        state = hj_filter.current_state()
        current_eval = hj_filter.evaluate_state(state, "current")
        if current_eval["current_l"] <= 0.0:
            live_safety_violation = True

        ilqr_policy = ilqr_assets[0]
        action_raw, ilqr_record = ilqr_policy.get_action()
        action_raw = action_raw.astype(np.float32)
        action_norm = hj_filter.raw_to_norm(action_raw)
        predicted_next = hj_filter.predict_next(state, action_norm)
        predicted_next_eval = hj_filter.evaluate_state(predicted_next, "ilqr_next")

        current_time = float(current_info["time"][0]) + float(episode["control_timestep"])
        current_frame, current_info = hard.step_env_with_action(
            env,
            renderer,
            action=action_raw,
            control_decimation=control_decimation,
            camera_id=camera_id,
            elapsed_time=current_time,
        )
        if video_writer is not None:
            video_writer.write(current_frame)

        hj_filter.append_frame(current_frame)
        ilqr_append_frame(ilqr_assets=ilqr_assets, frame=current_frame, device=device)
        post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
        if post["post_l"] <= 0.0:
            live_safety_violation = True

        task_dist = hard.task_target_distance(current_info, goal_task_target)
        left_dist = hard.left_attachment_distance(current_info, goal_left_pos)
        right_dist = hard.right_attachment_distance(current_info, goal_right_pos)
        length_err = hard.rope_length_error(current_info, goal_rope_length)
        task_target_distances.append(task_dist)
        left_attachment_distances.append(left_dist)
        right_attachment_distances.append(right_dist)
        rope_length_errors.append(length_err)
        executed_actions_raw.append(np.asarray(action_raw, np.float32).tolist())
        executed_actions_norm.append(np.asarray(action_norm, np.float32).tolist())
        success = task_dist <= float(args.goal_tolerance)

        step_records.append(
            {
                "step": int(step + 1),
                "phase": "nominal_ilqr",
                "hj_state_source": "sim",
                "post_hj_state_source": "sim",
                "task_target_distance": float(task_dist),
                "left_attachment_distance": float(left_dist),
                "right_attachment_distance": float(right_dist),
                "rope_length_error": float(length_err),
                **current_eval,
                **predicted_next_eval,
                **ilqr_record,
                **post,
                "override": False,
                "override_reason": "nominal_ilqr_no_steering",
                "nominal_action_raw": action_raw.tolist(),
                "nominal_action_norm": action_norm.tolist(),
                "safety_action_raw": None,
                "safety_action_norm": None,
                "executed_action_raw": action_raw.tolist(),
                "executed_action_norm": action_norm.tolist(),
                "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
            }
        )

        if success:
            stop_reason = "goal_tolerance_reached"
            break

    video_path = None
    video_error = None
    if video_writer is not None:
        video, video_error = video_writer.close()
        video_path = None if video is None else str(video)

    min_l = min([initial["initial_l"]] + [float(item["post_l"]) for item in step_records])
    min_v = min([initial["initial_V"]] + [float(item["post_V"]) for item in step_records])
    min_b = min([initial["initial_B"]] + [float(item["post_B"]) for item in step_records])
    summary = {
        "episode_idx": int(episode_idx),
        "method": "nominal_ilqr",
        "success": bool(success),
        "safety_violation": bool(live_safety_violation),
        "recorded_replay_violation": False,
        "recorded_violation_before_warning": False,
        "live_safety_violation": bool(live_safety_violation),
        "late_warning": False,
        "goal_tolerance": float(args.goal_tolerance),
        "stop_reason": stop_reason,
        "steps_executed": int(len(step_records)),
        "warning_step": None,
        "recovery_done_step": None,
        "override_count": 0,
        "override_rate": 0.0,
        "initial_l": float(initial["initial_l"]),
        "initial_V": float(initial["initial_V"]),
        "initial_B": float(initial["initial_B"]),
        "min_l": float(min_l),
        "min_V": float(min_v),
        "min_B": float(min_b),
        "initial_task_target_distance": float(task_target_distances[0]),
        "final_task_target_distance": float(task_target_distances[-1]),
        "video_path": video_path,
        "video_error": video_error,
        "task_target_distances": task_target_distances,
        "left_attachment_distances": left_attachment_distances,
        "right_attachment_distances": right_attachment_distances,
        "rope_length_errors": rope_length_errors,
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - pixels_np[0].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - pixels_np[-1].astype(np.float32))),
        "step_records": step_records,
    }
    (case_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    return summary


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"num_eval": 0}
    successes = np.asarray([case["success"] for case in cases], dtype=bool)
    violations = np.asarray([case["safety_violation"] for case in cases], dtype=bool)
    recorded_violations = np.asarray([case["recorded_replay_violation"] for case in cases], dtype=bool)
    recorded_before_warning = np.asarray([case["recorded_violation_before_warning"] for case in cases], dtype=bool)
    live_violations = np.asarray([case["live_safety_violation"] for case in cases], dtype=bool)
    late_warnings = np.asarray([case["late_warning"] for case in cases], dtype=bool)
    overrides = np.asarray([case["override_count"] for case in cases], dtype=np.float64)
    steps = np.asarray([case["steps_executed"] for case in cases], dtype=np.float64)
    warned = np.asarray([case["warning_step"] is not None for case in cases], dtype=bool)
    recovered = np.asarray([case["recovery_done_step"] is not None for case in cases], dtype=bool)
    return {
        "num_eval": int(len(cases)),
        "success_rate": float(np.mean(successes) * 100.0),
        "safety_violation_rate": float(np.mean(violations) * 100.0),
        "recorded_replay_violation_rate": float(np.mean(recorded_violations) * 100.0),
        "recorded_violation_before_warning_rate": float(np.mean(recorded_before_warning) * 100.0),
        "live_safety_violation_rate": float(np.mean(live_violations) * 100.0),
        "late_warning_rate": float(np.mean(late_warnings) * 100.0),
        "warning_rate": float(np.mean(warned) * 100.0),
        "recovery_handoff_rate": float(np.mean(recovered) * 100.0),
        "mean_override_count": float(np.mean(overrides)),
        "mean_override_rate": float(np.sum(overrides) / max(np.sum(steps), 1.0)),
        "mean_min_l": float(np.mean([case["min_l"] for case in cases])),
        "mean_min_V": float(np.mean([case["min_V"] for case in cases])),
        "mean_min_B": float(np.mean([case["min_B"] for case in cases])),
        "mean_final_task_target_distance": float(np.mean([case["final_task_target_distance"] for case in cases])),
    }


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if args.safe_steps_required < 1:
        raise ValueError("--safe-steps-required must be positive.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive.")

    episodes = select_unsafe_episodes(args)
    with h5py.File(args.dataset_path, "r") as h5:
        max_ep = int(h5["ep_len"].shape[0]) - 1
    for episode in episodes:
        if episode < 0 or episode > max_ep:
            raise ValueError(f"episode {episode} outside [0, {max_ep}].")

    device = hard.require_device(args.device)
    ilqr_assets = hard.load_ilqr_assets(args, device)
    hj_filter = RopeHJSafetyFilter(
        cache_path=args.hj_cache_path,
        policy_path=args.hj_policy_path,
        classifier_checkpoint=args.classifier_checkpoint,
        device_arg=args.device,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        classifier_threshold=str(args.classifier_threshold),
        margin_transform=str(args.margin_transform),
        actor_hidden=args.actor_hidden,
        critic_hidden=args.critic_hidden,
        action_low=float(args.action_low),
        action_high=float(args.action_high),
        epsilon=float(args.epsilon),
    )

    out_root = args.out_dir.expanduser().resolve() / f"{int(time.time())}_replay_hj_recover_ilqr_seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    probe = hard.load_dataset_episode(args.dataset_path, episodes[0])
    nominal_summaries: list[dict[str, Any]] = []
    steered_summaries: list[dict[str, Any]] = []
    env = LabEnv()
    with mujoco.Renderer(env.model, height=int(probe["height"]), width=int(probe["width"])) as renderer:
        for case_idx, episode_idx in enumerate(tqdm(episodes, desc="Nominal + replay HJ recover iLQR")):
            nominal_summaries.append(
                run_nominal_ilqr_episode(
                    args=args,
                    episode_idx=episode_idx,
                    case_idx=case_idx,
                    out_root=out_root / "nominal_ilqr",
                    env=env,
                    renderer=renderer,
                    device=device,
                    ilqr_assets=ilqr_assets,
                    hj_filter=hj_filter,
                )
            )
            steered_summaries.append(
                run_episode(
                    args=args,
                    episode_idx=episode_idx,
                    case_idx=case_idx,
                    out_root=out_root / "replay_hj_recover_ilqr",
                    env=env,
                    renderer=renderer,
                    device=device,
                    ilqr_assets=ilqr_assets,
                    hj_filter=hj_filter,
                )
            )

    aggregates = {
        "nominal_ilqr": aggregate(nominal_summaries),
        "replay_hj_recover_ilqr": aggregate(steered_summaries),
    }
    metrics = {
        "dataset_path": str(args.dataset_path),
        "episodes": episodes,
        "goal_tolerance": float(args.goal_tolerance),
        "max_steps": int(args.max_steps),
        "hj_filter": {
            "cache_path": str(args.hj_cache_path.expanduser().resolve()),
            "policy_path": str(args.hj_policy_path.expanduser().resolve()),
            "classifier_checkpoint": str(args.classifier_checkpoint.expanduser().resolve()),
            "epsilon": float(args.epsilon),
            "recovery_epsilon": float(args.recovery_epsilon),
            "safe_steps_required": int(args.safe_steps_required),
            "margin_transform": str(args.margin_transform),
        },
        "aggregate": aggregates["replay_hj_recover_ilqr"],
        "aggregates": aggregates,
        "cases": {
            "nominal_ilqr": nominal_summaries,
            "replay_hj_recover_ilqr": steered_summaries,
        },
    }
    (out_root / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(out_root), "aggregates": aggregates}), indent=2))


if __name__ == "__main__":
    main()
