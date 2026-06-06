#!/usr/bin/env python3
"""Closed-loop rope benchmark with a latent HJ safety filter."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.plan import benchmark_rope_hard as hard
from rope.safety.hj_filter import RopeHJSafetyFilter, parse_hidden_sizes
from rope.shared.lab_env import LabEnv

DEFAULT_DATASET_PATH = "rope/data/train_data_noshadow.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow_ft"
DEFAULT_HJ_CACHE = "rope/safety/cache/rope_latent_safety_classifier_train_noshadow_tanh.pt"
DEFAULT_HJ_POLICY = "rope/safety/runs/pyhj_train_noshadow_tanh/policy_latest.pth"
DEFAULT_CLASSIFIER = "rope/safety/obs_net/da270d7d1050f110/model.pt"
DEFAULT_OUT_DIR = "rope/safety/runs/closed_loop_hj_filter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-method", choices=("ilqr", "swm_cost", "swm_action"), default="ilqr")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument(
        "--case-source",
        choices=("random", "sequential", "unsafe-cache"),
        default="random",
        help=(
            "Sample normal random benchmark cases, the first valid dataset episodes in order, "
            "or full episodes whose cached trajectory contains unsafe frames."
        ),
    )
    parser.add_argument(
        "--min-first-unsafe-step",
        type=int,
        default=1,
        help="For --case-source unsafe-cache, skip episodes already unsafe before this cached step.",
    )
    parser.add_argument("--eval-budget", type=int, default=120)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--mode", choices=("paired", "nominal", "filtered"), default="paired")

    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--q-terminal", type=float, default=5.0)
    parser.add_argument("--q-stage", type=float, default=0.005)
    parser.add_argument("--r-control", type=float, default=0.001)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)

    parser.add_argument("--policy", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--solver-config", type=Path, default=hard.DEFAULT_SOLVER_CONFIG)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=1)
    parser.add_argument("--swm-history-size", type=int, default=None)
    parser.add_argument("--swm-img-size", type=int, default=224)
    parser.add_argument("--process-key", action="append", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--show-solver-output", action="store_true")

    parser.add_argument("--hj-cache-path", type=Path, default=Path(DEFAULT_HJ_CACHE))
    parser.add_argument("--hj-policy-path", type=Path, default=Path(DEFAULT_HJ_POLICY))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="auto")
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


def reset_ilqr_warm_start(ilqr_policy: hard.ILQRPolicyAdapter) -> None:
    ilqr_policy.solver.prev_u_guess.zero_()


def get_nominal_action(
    *,
    args: argparse.Namespace,
    ilqr_assets: tuple[Any, ...] | None,
    swm_policy: Any | None,
    episode_history: hard.RopeEpisodeHistory | None,
    action_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.base_method == "ilqr":
        assert ilqr_assets is not None
        ilqr_policy = ilqr_assets[0]
        action_raw, record = ilqr_policy.get_action()
        action_norm = hard.action_to_standardized(action_raw, ilqr_policy.action_mean, ilqr_policy.action_std)
        return action_raw.astype(np.float32), {**record, "nominal_action_norm_from_base": action_norm.tolist()}

    assert swm_policy is not None and episode_history is not None
    if args.show_solver_output:
        action_batch = swm_policy.get_action(episode_history.info())
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            action_batch = swm_policy.get_action(episode_history.info())
    action_raw = np.asarray(action_batch, dtype=np.float32).reshape(-1, action_dim)[0]
    return action_raw, {}


def run_one_case(
    *,
    args: argparse.Namespace,
    case: hard.EvalCase,
    case_idx: int,
    run_dir: Path,
    label: str,
    use_filter: bool,
    env: LabEnv,
    renderer: mujoco.Renderer,
    device: torch.device,
    ilqr_assets: tuple[Any, ...] | None,
    swm_policy: Any | None,
    swm_history_size: int | None,
    hj_filter: RopeHJSafetyFilter,
    action_dim: int,
) -> dict[str, Any]:
    episode = hard.load_dataset_episode(args.dataset_path, case.episode_idx)
    pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
    task_target_np = np.asarray(episode["task_target"], dtype=np.float32)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    control_np = np.asarray(episode["control"], dtype=np.float32)
    time_np = np.asarray(episode["time"], dtype=np.float32)
    camera = str(episode["camera"])
    control_decimation = int(episode["control_decimation"])
    goal_tolerance = float(args.goal_tolerance) if args.goal_tolerance is not None else float(episode["goal_tolerance"])
    camera_id = env.model.camera(camera).id

    case_dir = run_dir / label / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    start_frame, _start_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        control=control_np[case.start_step],
        task_target=task_target_np[case.start_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[case.start_step, 0]),
    )
    goal_frame, goal_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[case.goal_step],
        qvel=qvel_np[case.goal_step],
        control=control_np[case.goal_step],
        task_target=task_target_np[case.goal_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[case.goal_step, 0]),
    )
    current_frame, current_info = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        control=control_np[case.start_step],
        task_target=task_target_np[case.start_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[case.start_step, 0]),
    )
    hard.save_rgb_image(case_dir / "start_image.png", start_frame)
    hard.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    video_writer = None
    if not args.no_videos:
        video_writer = hard.FFMpegRolloutWriter(case_dir / "rollout.mp4", fps=args.video_fps, first_frame=current_frame)

    goal_task_target = np.asarray(goal_info["task_target"], dtype=np.float32)
    goal_left_pos = np.asarray(goal_info["left_attachment_pos"], dtype=np.float32)
    goal_right_pos = np.asarray(goal_info["right_attachment_pos"], dtype=np.float32)
    goal_rope_length = np.asarray(goal_info["rope_length"], dtype=np.float32)

    if args.base_method == "ilqr":
        assert ilqr_assets is not None
        ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
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
        episode_history = None
    else:
        assert swm_policy is not None
        hard.reset_swm_policy_state(swm_policy)
        assert swm_history_size is not None
        episode_history = hard.RopeEpisodeHistory(
            history_size=swm_history_size,
            action_dim=action_dim,
            case_id=case_idx,
            goal_frame=goal_frame,
            goal_info=goal_info,
            goal_step=case.goal_step,
        )
        episode_history.reset(frame=current_frame, info=current_info, step_idx=case.start_step)

    hj_filter.reset(current_frame)
    initial_safety = hj_filter.evaluate_state(hj_filter.current_state(), "initial")

    task_target_distances = [hard.task_target_distance(current_info, goal_task_target)]
    left_attachment_distances = [hard.left_attachment_distance(current_info, goal_left_pos)]
    right_attachment_distances = [hard.right_attachment_distance(current_info, goal_right_pos)]
    rope_length_errors = [hard.rope_length_error(current_info, goal_rope_length)]
    step_records: list[dict[str, Any]] = []
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []
    overrides = 0
    safety_violations = bool(initial_safety["initial_l"] <= 0.0)

    success = task_target_distances[-1] <= goal_tolerance
    stop_reason = "goal_tolerance_reached" if success else "eval_budget"

    for policy_step in range(0 if success else int(args.eval_budget)):
        nominal_raw, base_record = get_nominal_action(
            args=args,
            ilqr_assets=ilqr_assets,
            swm_policy=swm_policy,
            episode_history=episode_history,
            action_dim=action_dim,
        )
        decision = hj_filter.filter_action(nominal_raw)
        record = dict(decision.record)
        would_override = bool(record["override"])
        if use_filter:
            action_raw = decision.action_raw
            action_norm = decision.action_norm
            overrides += int(would_override)
        else:
            action_raw = nominal_raw.astype(np.float32)
            action_norm = hj_filter.raw_to_norm(action_raw)
            record["would_override"] = would_override
            record["override"] = False
            record["override_reason"] = "monitor_only_nominal_execution"
            record["executed_action_raw"] = action_raw.tolist()
            record["executed_action_norm"] = action_norm.tolist()

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

        if args.base_method == "ilqr":
            assert ilqr_assets is not None
            ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
            next_emb = hard.encode_single_frame(
                ilqr_model,
                current_frame,
                device=device,
                img_size=int(ilqr_config["img_size"]),
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            ilqr_policy.append_embedding(next_emb)
        else:
            assert episode_history is not None
            episode_history.append(
                frame=current_frame,
                action=action_raw,
                info=current_info,
                step_idx=case.start_step + policy_step + 1,
            )

        hj_filter.append_frame(current_frame)
        post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
        safety_violations = safety_violations or bool(post["post_l"] <= 0.0)

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

        step_records.append(
            {
                "phase": "policy",
                "step": int(policy_step + 1),
                "hj_filter_enabled": bool(use_filter),
                "task_target_distance": float(task_dist),
                "left_attachment_distance": float(left_dist),
                "right_attachment_distance": float(right_dist),
                "rope_length_error": float(length_err),
                **base_record,
                **record,
                **post,
                "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
            }
        )

        success = task_dist <= goal_tolerance
        if success:
            stop_reason = "goal_tolerance_reached"
            break
    else:
        if not success:
            stop_reason = "eval_budget"

    video_path = None
    video_error = None
    video_frames_written = 0
    if video_writer is not None:
        video, video_error = video_writer.close()
        video_path = None if video is None else str(video)
        video_frames_written = int(video_writer.frames_written)

    final_info = current_info
    min_l = min([initial_safety["initial_l"]] + [float(item["post_l"]) for item in step_records])
    min_v = min([initial_safety["initial_V"]] + [float(item["post_V"]) for item in step_records])
    min_b = min([initial_safety["initial_B"]] + [float(item["post_B"]) for item in step_records])
    summary = {
        **hard.asdict(case),
        "label": label,
        "base_method": args.base_method,
        "hj_filter_enabled": bool(use_filter),
        "success": bool(success),
        "safety_violation": bool(safety_violations),
        "success_metric": "task_target_l2",
        "goal_tolerance": float(goal_tolerance),
        "policy_steps_executed": int(len(step_records)),
        "steps_executed": int(len(step_records)),
        "stop_reason": stop_reason,
        "episode_seed": int(episode["episode_seed"]),
        "dataset_terminated": bool(episode["terminated"]),
        "dataset_truncated": bool(episode["truncated"]),
        "mode": str(episode["mode"]),
        "camera": camera,
        "control_decimation": int(control_decimation),
        "control_timestep": float(episode["control_timestep"]),
        "override_count": int(overrides),
        "override_rate": float(overrides / max(len(step_records), 1)),
        "initial_l": float(initial_safety["initial_l"]),
        "initial_V": float(initial_safety["initial_V"]),
        "initial_B": float(initial_safety["initial_B"]),
        "min_l": float(min_l),
        "min_V": float(min_v),
        "min_B": float(min_b),
        "initial_task_target_distance": float(task_target_distances[0]),
        "final_task_target_distance": float(task_target_distances[-1]),
        "min_task_target_distance": float(np.min(task_target_distances)),
        "initial_left_attachment_distance": float(left_attachment_distances[0]),
        "final_left_attachment_distance": float(left_attachment_distances[-1]),
        "initial_right_attachment_distance": float(right_attachment_distances[0]),
        "final_right_attachment_distance": float(right_attachment_distances[-1]),
        "initial_rope_length_error": float(rope_length_errors[0]),
        "final_rope_length_error": float(rope_length_errors[-1]),
        "goal_task_target": goal_task_target.tolist(),
        "final_task_target": np.asarray(final_info["task_target"], dtype=np.float32).tolist(),
        "goal_left_attachment_pos": goal_left_pos.tolist(),
        "goal_right_attachment_pos": goal_right_pos.tolist(),
        "final_left_attachment_pos": np.asarray(final_info["left_attachment_pos"], dtype=np.float32).tolist(),
        "final_right_attachment_pos": np.asarray(final_info["right_attachment_pos"], dtype=np.float32).tolist(),
        "goal_rope_length": goal_rope_length.tolist(),
        "final_rope_length": np.asarray(final_info["rope_length"], dtype=np.float32).tolist(),
        "video_path": video_path,
        "video_error": video_error,
        "video_frames_written": int(video_frames_written),
        "task_target_distances": task_target_distances,
        "left_attachment_distances": left_attachment_distances,
        "right_attachment_distances": right_attachment_distances,
        "rope_length_errors": rope_length_errors,
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - pixels_np[case.start_step].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - pixels_np[case.goal_step].astype(np.float32))),
        "step_records": step_records,
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2)
    return summary


def aggregate(label: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"label": label, "num_eval": 0}
    successes = np.asarray([case["success"] for case in cases], dtype=bool)
    violations = np.asarray([case["safety_violation"] for case in cases], dtype=bool)
    overrides = np.asarray([case["override_count"] for case in cases], dtype=np.float64)
    steps = np.asarray([case["steps_executed"] for case in cases], dtype=np.float64)
    return {
        "label": label,
        "num_eval": int(len(cases)),
        "success_rate": float(np.mean(successes) * 100.0),
        "safety_violation_rate": float(np.mean(violations) * 100.0),
        "mean_override_count": float(np.mean(overrides)),
        "mean_override_rate": float(np.sum(overrides) / max(np.sum(steps), 1.0)),
        "mean_min_l": float(np.mean([case["min_l"] for case in cases])),
        "mean_min_V": float(np.mean([case["min_V"] for case in cases])),
        "mean_min_B": float(np.mean([case["min_B"] for case in cases])),
        "mean_final_task_target_distance": float(np.mean([case["final_task_target_distance"] for case in cases])),
    }


def sample_unsafe_cache_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[hard.EvalCase]:
    cache = torch.load(args.hj_cache_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    episode_idx = cache["episode_idx"].detach().cpu().numpy().astype(np.int64)
    step_idx = cache["step_idx"].detach().cpu().numpy().astype(np.int64)
    margins = cache["safety_margin"].detach().cpu().numpy().astype(np.float32)
    unsafe_eps = np.unique(episode_idx[margins <= 0.0])
    candidates: list[int] = []
    for episode in unsafe_eps.tolist():
        if int(episode) < 0 or int(episode) >= ep_len.shape[0] or int(ep_len[int(episode)]) < 2:
            continue
        unsafe_steps = step_idx[(episode_idx == int(episode)) & (margins <= 0.0)]
        if unsafe_steps.size == 0:
            continue
        if int(np.min(unsafe_steps)) < int(args.min_first_unsafe_step):
            continue
        candidates.append(int(episode))
    if not candidates:
        raise ValueError(
            "No unsafe-cache benchmark cases available after filtering. "
            "Lower --min-first-unsafe-step or check the cache."
        )
    if args.episode_idx is not None:
        episode_idx_arg = int(args.episode_idx)
        if episode_idx_arg not in candidates:
            raise ValueError(f"--episode-idx {episode_idx_arg} is not an unsafe-cache candidate.")
        selected = [episode_idx_arg]
    else:
        if int(args.num_eval) > len(candidates):
            raise ValueError(f"Requested {args.num_eval} unsafe-cache cases, but only {len(candidates)} are available.")
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(np.asarray(candidates, dtype=np.int64), size=int(args.num_eval), replace=False)).tolist()
    return [hard.EvalCase(int(ep), 0, int(ep_len[int(ep)]) - 1, int(ep_len[int(ep)])) for ep in selected]


def sample_sequential_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[hard.EvalCase]:
    valid = np.flatnonzero(ep_len >= 2)
    if valid.size == 0:
        raise ValueError("Need at least one episode with at least two frames.")
    if args.episode_idx is not None:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"Episode {episode_idx} has length {ep_len[episode_idx]}, expected at least 2.")
        return [hard.EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx]))]
    if int(args.num_eval) > valid.size:
        raise ValueError(f"Requested {args.num_eval} episodes, but only {valid.size} are valid.")
    selected = valid[: int(args.num_eval)]
    return [hard.EvalCase(int(ep), 0, int(ep_len[int(ep)]) - 1, int(ep_len[int(ep)])) for ep in selected]


def main() -> None:
    args = parse_args()
    args.method = args.base_method
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.base_method.startswith("swm") and not args.policy:
        raise ValueError("--policy is required for --base-method swm_cost/swm_action.")

    device = hard.require_device(args.device)
    ep_len = hard.load_episode_lengths(args.dataset_path)
    if args.case_source == "unsafe-cache":
        sampled_cases = sample_unsafe_cache_cases(args, ep_len)
    elif args.case_source == "sequential":
        sampled_cases = sample_sequential_cases(args, ep_len)
    else:
        sampled_cases = hard.sample_eval_cases(args, ep_len)
    cases = hard.slice_eval_cases(args, sampled_cases)
    action_dim = hard.read_action_dim(args.dataset_path)
    hard.validate_parent_swm_args(args, action_dim)

    run_name = f"{int(time.time())}_{args.base_method}_hj_seed_{args.seed}"
    run_root = args.out_dir.expanduser().resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    ilqr_assets = None
    swm_policy = None
    swm_history_size = None
    method_config: dict[str, Any]
    if args.base_method == "ilqr":
        ilqr_assets = hard.load_ilqr_assets(args, device)
        method_config = ilqr_assets[2]
    else:
        swm_policy, swm_history_size, method_config = hard.make_swm_policy(args, device, action_dim)

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

    labels: list[tuple[str, bool]]
    if args.mode == "paired":
        labels = [("nominal", False), ("hj_filtered", True)]
    elif args.mode == "nominal":
        labels = [("nominal", False)]
    else:
        labels = [("hj_filtered", True)]

    all_results: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in labels}
    probe_episode = hard.load_dataset_episode(args.dataset_path, cases[0].episode_idx)
    env = LabEnv()
    with mujoco.Renderer(env.model, height=int(probe_episode["height"]), width=int(probe_episode["width"])) as renderer:
        for case_idx, case in enumerate(tqdm(cases, desc="Closed-loop HJ rope eval")):
            for label, use_filter in labels:
                summary = run_one_case(
                    args=args,
                    case=case,
                    case_idx=case_idx,
                    run_dir=run_root,
                    label=label,
                    use_filter=use_filter,
                    env=env,
                    renderer=renderer,
                    device=device,
                    ilqr_assets=ilqr_assets,
                    swm_policy=swm_policy,
                    swm_history_size=swm_history_size,
                    hj_filter=hj_filter,
                    action_dim=action_dim,
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
                    "sampled_episode_indices": [int(item.episode_idx) for item in sampled_cases],
                    "evaluated_episode_indices": [int(item.episode_idx) for item in cases],
                    "aggregates": {name: aggregate(name, result) for name, result in all_results.items()},
                }
                (run_root / "metrics_partial.json").write_text(json.dumps(jsonable(partial), indent=2), encoding="utf-8")

    metrics = {
        "partial": False,
        "base_method": args.base_method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "case_source": str(args.case_source),
        "requested_num_eval": int(args.num_eval),
        "evaluated_case_count": int(len(cases)),
        "eval_budget": int(args.eval_budget),
        "goal_protocol": "start_step_0_to_final_episode_step",
        "hj_filter": {
            "cache_path": str(args.hj_cache_path.expanduser().resolve()),
            "policy_path": str(args.hj_policy_path.expanduser().resolve()),
            "classifier_checkpoint": str(args.classifier_checkpoint.expanduser().resolve()),
            "epsilon": float(args.epsilon),
            "margin_transform": str(args.margin_transform),
            "barrier": "min(classifier_margin, critic_value)",
            "switch_rule": "execute nominal iff predicted nominal next barrier > epsilon",
        },
        "sampled_episode_indices": [int(item.episode_idx) for item in sampled_cases],
        "evaluated_episode_indices": [int(item.episode_idx) for item in cases],
        "aggregates": {label: aggregate(label, result) for label, result in all_results.items()},
        "cases": all_results,
    }
    (run_root / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(run_root), "aggregates": metrics["aggregates"]}), indent=2))


if __name__ == "__main__":
    main()
