#!/usr/bin/env python3
"""Nominal-only Haoran iLQR evaluation on Reacher start/goal pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.plan.benchmark_reacher_hard import wrapped_qpos_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
HAORAN_ILQR_PATH = REPO_ROOT / "reacher" / "experiments" / "legacy_obstacle" / "plan_ilqr_mpc.py"
DEFAULT_START_GOAL_PATH = REPO_ROOT / "reacher" / "Haoran_obs_data" / "reacher_stuff" / "reacher_stuff" / "start_goal.pt"
DEFAULT_MODEL_DIR = REPO_ROOT / "reacher" / "Haoran_obs_data" / "reacher_stuff" / "reacher_stuff" / "mlpdyn_ft_6"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "lewm_epoch_1_object.ckpt"
DEFAULT_DATASET_PATH = REPO_ROOT / "reacher" / "data" / "train_data_noisy.h5"
DEFAULT_OUT_DIR = REPO_ROOT / "reacher" / "safety" / "runs" / "haoran_planner_start_goal_first35"

GEOM_BOX_LOWER = np.asarray([0.0, -2.88], dtype=np.float32)
GEOM_BOX_UPPER = np.asarray([3.1415, -2.45], dtype=np.float32)
GEOM_INSIDE_BEND_SIGN = -1


def load_haoran_ilqr_module() -> Any:
    spec = importlib.util.spec_from_file_location("reacher_haoran_plan_ilqr_mpc_eval", HAORAN_ILQR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Haoran ILQR planner from {HAORAN_ILQR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


haoran = load_haoran_ilqr_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start-goal-path", type=Path, default=DEFAULT_START_GOAL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--stats-dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=35)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--max-mpc-steps", type=int, default=100)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--q-terminal", type=float, default=5.0)
    parser.add_argument("--q-stage", type=float, default=0.005)
    parser.add_argument("--r-control", type=float, default=0.1)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--qpos-success-threshold", type=float, default=0.1)
    parser.add_argument("--no-videos", action="store_true")
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


def load_start_goal_pairs(path: Path, num_eval: int) -> list[dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    pairs = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"Expected {path} to contain a list of start/goal pairs.")
    if num_eval > len(pairs):
        raise ValueError(f"Requested {num_eval} pairs, but {path} contains {len(pairs)}.")
    return list(pairs[:num_eval])


def qpos_geometry_unsafe(qpos: np.ndarray) -> bool:
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    inside_box = bool(np.all((qpos_arr >= GEOM_BOX_LOWER) & (qpos_arr <= GEOM_BOX_UPPER)))
    selected_ik_branch = bool(float(GEOM_INSIDE_BEND_SIGN) * np.sin(float(qpos_arr[1])) > 0.0)
    return inside_box and selected_ik_branch


def load_dataset_context(dataset_path: Path) -> dict[str, Any]:
    with h5py.File(dataset_path.expanduser().resolve(), "r") as h5:
        obs_dim = int(h5["observation"].shape[-1])
        qpos_dim = int(h5["qpos"].shape[-1])
        qvel_dim = int(h5["qvel"].shape[-1])
        height = int(h5["pixels"].shape[-3])
        width = int(h5["pixels"].shape[-2])
        episode_seed = int(np.asarray(h5["episode_seed"][:], dtype=np.int64)[0]) if "episode_seed" in h5 else 0
        physics_freq_hz = float(np.asarray(h5["physics_freq_hz"][:]).reshape(-1)[0]) if "physics_freq_hz" in h5 else 100.0
        time_limit = float(np.asarray(h5["time_limit"][:]).reshape(-1)[0]) if "time_limit" in h5 else 10.0
    return {
        "obs_dim": obs_dim,
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "height": height,
        "width": width,
        "episode_seed": episode_seed,
        "physics_freq_hz": physics_freq_hz,
        "time_limit": time_limit,
    }


def save_case_payload(case_dir: Path, payload: dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, case_dir / "nominal_rollout.pt")


def run_case(
    *,
    args: argparse.Namespace,
    case_idx: int,
    pair: dict[str, Any],
    run_dir: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    context: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    img_size = int(config.get("img_size", 224))
    qpos_dim = int(context["qpos_dim"])
    qvel_dim = int(context["qvel_dim"])
    height = int(context["height"])
    width = int(context["width"])
    episode_seed = int(args.seed) + int(case_idx)

    start = pair["start"]
    goal = pair["goal"]
    start_qpos = np.asarray(start["qpos"], dtype=np.float32)
    start_qvel = np.asarray(start.get("qvel", np.zeros((qvel_dim,), dtype=np.float32)), dtype=np.float32)
    goal_qpos = np.asarray(goal["qpos"], dtype=np.float32)
    goal_qvel = np.asarray(goal.get("qvel", np.zeros((qvel_dim,), dtype=np.float32)), dtype=np.float32)

    case_dir = run_dir / f"case_{case_idx:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    env = haoran.make_render_env(
        seed=episode_seed,
        time_limit=float(context["time_limit"]),
        width=width,
        height=height,
        physics_freq_hz=float(context["physics_freq_hz"]),
    )
    start_frame = haoran.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=start_qvel,
        height=height,
        width=width,
    )
    goal_frame = haoran.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=goal_qpos,
        qvel=goal_qvel,
        height=height,
        width=width,
    )
    haoran.save_rgb_image(case_dir / "start_image.png", start_frame)
    haoran.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    start_emb = haoran.encode_single_frame(
        model, start_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
    )
    goal_emb = haoran.encode_single_frame(
        model, goal_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
    )
    goal_state = haoran.make_markov_state(goal_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)

    dynamics = haoran.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    solver = haoran.ILQRMPCSolver(
        dynamics,
        state_cost_dim=embed_dim,
        horizon=int(args.horizon),
        q_terminal=float(args.q_terminal),
        q_stage=float(args.q_stage),
        r_control=float(args.r_control),
        max_iters=int(args.ilqr_max_iters),
        tol=float(args.ilqr_tol),
        regularization=float(args.ilqr_regularization),
        device=device,
    )

    current_frame = haoran.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=start_qvel,
        height=height,
        width=width,
    )
    current_emb = haoran.encode_single_frame(
        model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
    )
    current_state = haoran.make_markov_state(current_emb)
    goal_obs = haoran.build_goal_observation(env, goal_qpos=goal_qpos, goal_qvel=goal_qvel, obs_dim=int(context["obs_dim"]))
    current_obs = haoran.build_observation_from_env(env, obs_dim=int(context["obs_dim"]), goal_obs=goal_obs)

    rollout_frames = [current_frame.copy()]
    executed_qpos = [np.asarray(env._env.physics.data.qpos[:qpos_dim], dtype=np.float32).copy()]
    executed_qvel = [np.asarray(env._env.physics.data.qvel[:qvel_dim], dtype=np.float32).copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    nominal_state_rollouts: list[np.ndarray] = []
    nominal_action_rollouts: list[np.ndarray] = []
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    qpos_distances = [wrapped_qpos_distance(executed_qpos[-1], goal_qpos)]
    obs_distances = [haoran.compute_observation_goal_distance(current_obs, goal_obs)]
    geometry_unsafe_over_time = [qpos_geometry_unsafe(executed_qpos[-1])]
    stop_reason = "max_mpc_steps"
    success = qpos_distances[-1] <= float(args.qpos_success_threshold)

    for _ in range(0 if success else int(args.max_mpc_steps)):
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        x_plan, u_plan, solve_time, n_iters, plan_cost = solver.solve(current_state_np, goal_state_np)
        nominal_state_rollouts.append(x_plan.copy())
        nominal_action_rollouts.append(u_plan.copy())
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))
        ilqr_costs.append(float(plan_cost))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = haoran.normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = haoran.build_observation_from_env(env, obs_dim=int(context["obs_dim"]), goal_obs=goal_obs)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = haoran.encode_single_frame(
            model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
        )
        current_state = haoran.make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        qpos = np.asarray(env._env.physics.data.qpos[:qpos_dim], dtype=np.float32).copy()
        qvel = np.asarray(env._env.physics.data.qvel[:qvel_dim], dtype=np.float32).copy()
        rollout_frames.append(current_frame.copy())
        executed_qpos.append(qpos)
        executed_qvel.append(qvel)
        qpos_distances.append(wrapped_qpos_distance(qpos, goal_qpos))
        obs_distances.append(haoran.compute_observation_goal_distance(current_obs, goal_obs))
        geometry_unsafe_over_time.append(qpos_geometry_unsafe(qpos))

        success = qpos_distances[-1] <= float(args.qpos_success_threshold)
        if success:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_qpos = executed_qpos[-1]
    final_qvel = executed_qvel[-1]
    video_path = None
    if not args.no_videos:
        video_path = str(haoran.save_rollout_video(rollout_frames, case_dir, fps=int(args.video_fps)))
    env.close()

    safe = not any(geometry_unsafe_over_time)
    robust_success = bool(success and safe)
    summary = {
        "case_idx": int(case_idx),
        "episode_seed": int(episode_seed),
        "success": bool(success),
        "safe": bool(safe),
        "safety_violation": bool(not safe),
        "robust_success": robust_success,
        "success_metric": "wrapped_qpos_l2",
        "qpos_success_threshold": float(args.qpos_success_threshold),
        "stop_reason": stop_reason,
        "steps_executed": int(len(executed_actions_raw)),
        "initial_qpos_distance": float(qpos_distances[0]),
        "final_qpos_distance": float(qpos_distances[-1]),
        "min_qpos_distance": float(np.min(qpos_distances)),
        "initial_observation_distance": float(obs_distances[0]),
        "final_observation_distance": float(obs_distances[-1]),
        "min_observation_distance": float(np.min(obs_distances)),
        "start_qpos": start_qpos.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "final_qpos": final_qpos.tolist(),
        "start_qvel": start_qvel.tolist(),
        "goal_qvel": goal_qvel.tolist(),
        "final_qvel": final_qvel.tolist(),
        "video_path": video_path,
        "qpos_trajectory": np.stack(executed_qpos, axis=0).tolist(),
        "qvel_trajectory": np.stack(executed_qvel, axis=0).tolist(),
        "executed_actions_raw": np.stack(executed_actions_raw, axis=0).tolist()
        if executed_actions_raw
        else [],
        "executed_actions_norm": np.stack(executed_actions_norm, axis=0).tolist()
        if executed_actions_norm
        else [],
        "geometry_unsafe_over_time": [bool(item) for item in geometry_unsafe_over_time],
    }
    payload = {
        "summary": summary,
        "nominal_rollouts": {
            "state_plans": np.stack(nominal_state_rollouts, axis=0)
            if nominal_state_rollouts
            else np.empty((0, int(args.horizon) + 1, markov_state_dim), dtype=np.float64),
            "action_plans": np.stack(nominal_action_rollouts, axis=0)
            if nominal_action_rollouts
            else np.empty((0, int(args.horizon), action_dim), dtype=np.float64),
            "solve_times_ms": np.asarray(solve_times_ms, dtype=np.float64),
            "ilqr_iterations": np.asarray(ilqr_iterations, dtype=np.int64),
            "ilqr_costs": np.asarray(ilqr_costs, dtype=np.float64),
        },
        "executed_rollout": {
            "qpos": np.stack(executed_qpos, axis=0),
            "qvel": np.stack(executed_qvel, axis=0),
            "actions_raw": np.stack(executed_actions_raw, axis=0)
            if executed_actions_raw
            else np.empty((0, action_dim), dtype=np.float32),
            "actions_norm": np.stack(executed_actions_norm, axis=0)
            if executed_actions_norm
            else np.empty((0, action_dim), dtype=np.float32),
        },
    }
    save_case_payload(case_dir, payload)
    (case_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    return summary


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    successes = np.asarray([case["success"] for case in cases], dtype=bool)
    safe = np.asarray([case["safe"] for case in cases], dtype=bool)
    robust = np.asarray([case["robust_success"] for case in cases], dtype=bool)
    return {
        "num_eval": int(len(cases)),
        "success_rate": float(np.mean(successes) * 100.0),
        "safety_rate": float(np.mean(safe) * 100.0),
        "safety_violation_rate": float((1.0 - np.mean(safe)) * 100.0),
        "robust_success_rate": float(np.mean(robust) * 100.0),
        "mean_final_qpos_distance": float(np.mean([case["final_qpos_distance"] for case in cases])),
        "mean_min_qpos_distance": float(np.mean([case["min_qpos_distance"] for case in cases])),
        "episode_successes": successes.astype(int).tolist(),
        "episode_safe": safe.astype(int).tolist(),
        "episode_robust_successes": robust.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    device = haoran.require_device(str(args.device))
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    stats_dataset_path = args.stats_dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    run_dir = out_root / f"{int(time.time())}_haoran_nominal_seed_{int(args.seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = haoran.load_config(model_dir)
    model = haoran.load_model(checkpoint_path, device)
    if int(config.get("history_size", 1)) != 1:
        raise ValueError(f"Expected history_size=1, got {config.get('history_size')}.")
    action_dim = int(config.get("action_dim", 2))
    pixel_mean, pixel_std = haoran.imagenet_pixel_stats(device)
    action_mean, action_std = haoran.load_action_stats([stats_dataset_path], action_dim)
    context = load_dataset_context(dataset_path)
    pairs = load_start_goal_pairs(args.start_goal_path, int(args.num_eval))

    method_config = {
        "planner_source": str(HAORAN_ILQR_PATH),
        "start_goal_path": str(args.start_goal_path.expanduser().resolve()),
        "dataset_path": str(dataset_path),
        "stats_dataset_path": str(stats_dataset_path),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "seed": int(args.seed),
        "num_eval": int(args.num_eval),
        "horizon": int(args.horizon),
        "max_mpc_steps": int(args.max_mpc_steps),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "ilqr_max_iters": int(args.ilqr_max_iters),
        "ilqr_tol": float(args.ilqr_tol),
        "ilqr_regularization": float(args.ilqr_regularization),
        "qpos_success_threshold": float(args.qpos_success_threshold),
        "pixel_stats": "imagenet",
        "action_stats": "hdf5_raw_actions",
        "state_cost_dim": int(config.get("embed_dim", 18)),
        "geometric_safety": {
            "box_lower": GEOM_BOX_LOWER.tolist(),
            "box_upper": GEOM_BOX_UPPER.tolist(),
            "inside_bend_sign": int(GEOM_INSIDE_BEND_SIGN),
        },
    }

    cases: list[dict[str, Any]] = []
    for case_idx, pair in enumerate(tqdm(pairs, desc="Haoran nominal reacher eval")):
        summary = run_case(
            args=args,
            case_idx=case_idx,
            pair=pair,
            run_dir=run_dir,
            model=model,
            config=config,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            action_mean=action_mean,
            action_std=action_std,
            context=context,
            device=device,
        )
        cases.append(summary)
        partial = {
            "partial": True,
            "method_config": method_config,
            "aggregates": aggregate(cases),
            "cases": cases,
        }
        (run_dir / "metrics_partial.json").write_text(json.dumps(jsonable(partial), indent=2), encoding="utf-8")

    metrics = {
        "partial": False,
        "method_config": method_config,
        "aggregates": aggregate(cases),
        "cases": cases,
    }
    (run_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(run_dir), "aggregates": metrics["aggregates"]}), indent=2))


if __name__ == "__main__":
    main()
