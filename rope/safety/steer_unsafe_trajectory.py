#!/usr/bin/env python3
"""Replay unsafe rope trajectories with and without the latent HJ filter."""

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
from tqdm.auto import tqdm

from rope.plan import benchmark_rope_hard as hard
from rope.safety.benchmark_hj_filter_rope import (
    DEFAULT_CLASSIFIER,
    DEFAULT_DATASET_PATH,
    DEFAULT_HJ_CACHE,
    DEFAULT_HJ_POLICY,
    DEFAULT_MODEL_DIR,
    jsonable,
)
from rope.safety.hj_filter import RopeHJSafetyFilter, parse_hidden_sizes
from rope.shared.lab_env import LabEnv

DEFAULT_OUT_DIR = "rope/safety/runs/unsafe_trajectory_steering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--hj-cache-path", type=Path, default=Path(DEFAULT_HJ_CACHE))
    parser.add_argument("--hj-policy-path", type=Path, default=Path(DEFAULT_HJ_POLICY))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="auto")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, action="append", default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--start-before-unsafe",
        type=int,
        default=10,
        help="Start this many steps before the first unsafe cached frame.",
    )
    parser.add_argument(
        "--min-first-unsafe-step",
        type=int,
        default=1,
        help="Skip cache-unsafe episodes whose first unsafe frame occurs before this step.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
    return parser.parse_args()


def find_unsafe_windows(
    cache_path: Path,
    count: int,
    seed: int,
    start_before: int,
    min_first_unsafe_step: int,
) -> list[dict[str, int]]:
    import torch

    cache = torch.load(cache_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    episode_idx = cache["episode_idx"].detach().cpu().numpy().astype(np.int64)
    step_idx = cache["step_idx"].detach().cpu().numpy().astype(np.int64)
    margins = cache["safety_margin"].detach().cpu().numpy().astype(np.float32)
    unsafe_eps = np.unique(episode_idx[margins <= 0.0])
    if unsafe_eps.size == 0:
        raise ValueError(f"No unsafe episodes found in {cache_path}.")
    windows: list[dict[str, int]] = []
    for episode in unsafe_eps:
        unsafe_steps = step_idx[(episode_idx == episode) & (margins <= 0.0)]
        first_unsafe_step = int(np.min(unsafe_steps))
        if first_unsafe_step < int(min_first_unsafe_step):
            continue
        windows.append(
            {
                "episode_idx": int(episode),
                "first_unsafe_step": first_unsafe_step,
                "start_step": max(0, first_unsafe_step - int(start_before)),
            }
        )
    if not windows:
        raise ValueError(
            "No unsafe windows remain after --min-first-unsafe-step filtering. "
            "Lower --min-first-unsafe-step."
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(windows), size=min(int(count), len(windows)), replace=False)
    return [windows[int(idx)] for idx in np.sort(selected)]


def windows_for_explicit_episodes(cache_path: Path, episodes: list[int], start_before: int) -> list[dict[str, int]]:
    import torch

    cache = torch.load(cache_path.expanduser().resolve(), map_location="cpu", weights_only=False)
    episode_idx = cache["episode_idx"].detach().cpu().numpy().astype(np.int64)
    step_idx = cache["step_idx"].detach().cpu().numpy().astype(np.int64)
    margins = cache["safety_margin"].detach().cpu().numpy().astype(np.float32)
    windows: list[dict[str, int]] = []
    for episode in episodes:
        unsafe_steps = step_idx[(episode_idx == episode) & (margins <= 0.0)]
        if unsafe_steps.size:
            first_unsafe_step = int(np.min(unsafe_steps))
            start_step = max(0, first_unsafe_step - int(start_before))
        else:
            first_unsafe_step = -1
            start_step = 0
        windows.append(
            {
                "episode_idx": int(episode),
                "first_unsafe_step": first_unsafe_step,
                "start_step": start_step,
            }
        )
    return windows


def run_replay(
    *,
    args: argparse.Namespace,
    window: dict[str, int],
    label: str,
    use_filter: bool,
    out_root: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
    hj_filter: RopeHJSafetyFilter,
) -> dict[str, Any]:
    episode_idx = int(window["episode_idx"])
    start_step = int(window["start_step"])
    first_unsafe_step = int(window.get("first_unsafe_step", -1))
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
    ep_len = int(actions_np.shape[0])
    if start_step < 0 or start_step >= ep_len - 1:
        raise ValueError(f"start_step={start_step} is outside episode {episode_idx} length {ep_len}.")
    max_available = ep_len - start_step - 1
    max_steps = max_available if args.max_steps is None else min(int(args.max_steps), max_available)

    case_dir = out_root / label / f"episode_{episode_idx:05d}_start_{start_step:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    start_frame, _ = hard.reset_env_to_state(
        env,
        renderer,
        qpos=qpos_np[start_step],
        qvel=qvel_np[start_step],
        control=control_np[start_step],
        task_target=task_target_np[start_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[start_step, 0]),
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
        qpos=qpos_np[start_step],
        qvel=qvel_np[start_step],
        control=control_np[start_step],
        task_target=task_target_np[start_step],
        camera_id=camera_id,
        elapsed_time=float(time_np[start_step, 0]),
    )
    hard.save_rgb_image(case_dir / "start_image.png", start_frame)
    hard.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    video_writer = None
    if not args.no_videos:
        video_writer = hard.FFMpegRolloutWriter(case_dir / "rollout.mp4", fps=args.video_fps, first_frame=current_frame)

    history_start = max(0, start_step - hj_filter.history_len + 1)
    hj_filter.reset(pixels_np[history_start])
    for hist_step in range(history_start + 1, start_step + 1):
        hj_filter.append_frame(pixels_np[hist_step])
    initial = hj_filter.evaluate_state(hj_filter.current_state(), "initial")
    goal_task_target = np.asarray(goal_info["task_target"], dtype=np.float32)
    step_records: list[dict[str, Any]] = []
    safety_violation = bool(initial["initial_l"] <= 0.0)
    overrides = 0
    task_target_distances = [hard.task_target_distance(current_info, goal_task_target)]
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []

    for step in range(max_steps):
        dataset_step = start_step + step
        nominal_raw = actions_np[dataset_step].astype(np.float32)
        decision = hj_filter.filter_action(nominal_raw)
        record = dict(decision.record)
        would_override = bool(record["override"])
        if use_filter:
            action_raw = decision.action_raw
            action_norm = decision.action_norm
            overrides += int(would_override)
        else:
            action_raw = nominal_raw
            action_norm = hj_filter.raw_to_norm(action_raw)
            record["would_override"] = would_override
            record["override"] = False
            record["override_reason"] = "monitor_only_replay_execution"
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
        hj_filter.append_frame(current_frame)
        post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
        safety_violation = safety_violation or bool(post["post_l"] <= 0.0)
        task_dist = hard.task_target_distance(current_info, goal_task_target)
        task_target_distances.append(task_dist)
        executed_actions_raw.append(np.asarray(action_raw, dtype=np.float32).tolist())
        executed_actions_norm.append(np.asarray(action_norm, dtype=np.float32).tolist())
        step_records.append(
            {
                "step": int(step + 1),
                "dataset_step": int(dataset_step + 1),
                "hj_filter_enabled": bool(use_filter),
                "task_target_distance": float(task_dist),
                **record,
                **post,
                "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
            }
        )

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
        "start_step": int(start_step),
        "first_unsafe_step": int(first_unsafe_step),
        "label": label,
        "hj_filter_enabled": bool(use_filter),
        "steps_executed": int(len(step_records)),
        "safety_violation": bool(safety_violation),
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
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - pixels_np[start_step].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - pixels_np[-1].astype(np.float32))),
        "step_records": step_records,
    }
    (case_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    return summary


def aggregate(label: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    violations = np.asarray([case["safety_violation"] for case in cases], dtype=bool)
    overrides = np.asarray([case["override_count"] for case in cases], dtype=np.float64)
    steps = np.asarray([case["steps_executed"] for case in cases], dtype=np.float64)
    return {
        "label": label,
        "num_episodes": int(len(cases)),
        "safety_violation_rate": float(np.mean(violations) * 100.0) if cases else 0.0,
        "mean_override_count": float(np.mean(overrides)) if cases else 0.0,
        "mean_override_rate": float(np.sum(overrides) / max(np.sum(steps), 1.0)) if cases else 0.0,
        "mean_min_l": float(np.mean([case["min_l"] for case in cases])) if cases else 0.0,
        "mean_min_V": float(np.mean([case["min_V"] for case in cases])) if cases else 0.0,
        "mean_min_B": float(np.mean([case["min_B"] for case in cases])) if cases else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if args.episode_idx is None:
        windows = find_unsafe_windows(
            args.hj_cache_path,
            args.num_episodes,
            args.seed,
            args.start_before_unsafe,
            args.min_first_unsafe_step,
        )
    else:
        windows = windows_for_explicit_episodes(
            args.hj_cache_path,
            [int(item) for item in args.episode_idx],
            args.start_before_unsafe,
        )

    with h5py.File(args.dataset_path, "r") as h5:
        max_ep = int(h5["ep_len"].shape[0]) - 1
    for window in windows:
        episode_idx = int(window["episode_idx"])
        if episode_idx < 0 or episode_idx > max_ep:
            raise ValueError(f"episode {episode_idx} outside [0, {max_ep}].")

    out_root = args.out_dir.expanduser().resolve() / f"{int(time.time())}_replay_hj_seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    device = hard.require_device(args.device)
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

    probe_episode = hard.load_dataset_episode(args.dataset_path, int(windows[0]["episode_idx"]))
    results = {"replay_nominal": [], "hj_filtered_replay": []}
    env = LabEnv()
    with mujoco.Renderer(env.model, height=int(probe_episode["height"]), width=int(probe_episode["width"])) as renderer:
        for window in tqdm(windows, desc="Unsafe trajectory steering"):
            results["replay_nominal"].append(
                run_replay(
                    args=args,
                    window=window,
                    label="replay_nominal",
                    use_filter=False,
                    out_root=out_root,
                    env=env,
                    renderer=renderer,
                    hj_filter=hj_filter,
                )
            )
            results["hj_filtered_replay"].append(
                run_replay(
                    args=args,
                    window=window,
                    label="hj_filtered_replay",
                    use_filter=True,
                    out_root=out_root,
                    env=env,
                    renderer=renderer,
                    hj_filter=hj_filter,
                )
            )

    metrics = {
        "dataset_path": str(args.dataset_path),
        "windows": windows,
        "hj_filter": {
            "cache_path": str(args.hj_cache_path.expanduser().resolve()),
            "policy_path": str(args.hj_policy_path.expanduser().resolve()),
            "classifier_checkpoint": str(args.classifier_checkpoint.expanduser().resolve()),
            "epsilon": float(args.epsilon),
            "margin_transform": str(args.margin_transform),
        },
        "aggregates": {label: aggregate(label, cases) for label, cases in results.items()},
        "cases": results,
    }
    (out_root / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(out_root), "aggregates": metrics["aggregates"]}), indent=2))


if __name__ == "__main__":
    main()
