#!/usr/bin/env python3
"""Planner ablation: CEM MPC over the Reacher latent MLP dynamics model.

This mirrors ``benchmark_reacher_hard.py --method ilqr``: each evaluation case
starts at episode step 0 and uses the final episode frame as the goal. The
world model, latent cost, dataset split, action normalization, and success
metric are kept fixed; only the planner is changed from iLQR to CEM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.plan import plan_ilqr_mpc as base
from reacher.train.reacher_policy_train import flatten_observation

DEFAULT_OUT_DIR = "reacher/plan/reacher_cem_ablation"
DEFAULT_NUM_EVAL = 50
DEFAULT_SEED = 42
DEFAULT_SUCCESS_THRESHOLD = 0.05
DEFAULT_QPOS_SUCCESS_THRESHOLD = 0.1


@dataclass(frozen=True)
class EvalCase:
    episode_idx: int
    start_step: int
    goal_step: int
    ep_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=Path(base.DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(base.DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-eval", type=int, default=DEFAULT_NUM_EVAL)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=base.MAX_MPC_STEPS)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--qpos-success-threshold", type=float, default=DEFAULT_QPOS_SUCCESS_THRESHOLD)
    parser.add_argument("--video-fps", type=int, default=base.VIDEO_FPS)
    parser.add_argument("--no-videos", action="store_true")

    parser.add_argument("--horizon", type=int, default=base.HORIZON)
    parser.add_argument("--receding-horizon", type=int, default=1)
    parser.add_argument("--q-terminal", type=float, default=base.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=base.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=base.R_CONTROL)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-iters", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--cem-var-scale", type=float, default=1.0)
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def wrapped_angle_diff(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return ((np.asarray(current) - np.asarray(goal) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def wrapped_qpos_distance(current: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(wrapped_angle_diff(current, goal)))


def current_qpos_qvel(env: base.DmControlGymEnv, qpos_dim: int, qvel_dim: int) -> tuple[np.ndarray, np.ndarray]:
    physics = env._env.physics
    return (
        np.asarray(physics.data.qpos[:qpos_dim], dtype=np.float32).copy(),
        np.asarray(physics.data.qvel[:qvel_dim], dtype=np.float32).copy(),
    )


def load_episode_lengths(dataset_path: Path) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        return np.asarray(h5["ep_len"][:], dtype=np.int64)


def sample_eval_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[EvalCase]:
    valid = np.flatnonzero(ep_len >= 2)
    if valid.size == 0:
        raise ValueError("Need at least one episode with at least two frames.")
    if args.episode_idx is not None:
        if int(args.num_eval) != 1:
            raise ValueError("--episode-idx pins one debug episode. Omit it for --num-eval > 1.")
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"Episode {episode_idx} has length {ep_len[episode_idx]}, expected at least 2.")
        return [EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx]))]
    if int(args.num_eval) > valid.size:
        raise ValueError(f"Requested {args.num_eval} episodes, but only {valid.size} are valid.")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(valid, size=int(args.num_eval), replace=False))
    return [EvalCase(int(ep), 0, int(ep_len[ep]) - 1, int(ep_len[ep])) for ep in selected]


class CEMMPCSolver:
    def __init__(
        self,
        dynamics: base.MarkovDynamicsTorch,
        *,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        num_samples: int,
        num_iters: int,
        topk: int,
        var_scale: float,
        seed: int,
        device: torch.device,
    ) -> None:
        self.dynamics = dynamics
        self.horizon = int(horizon)
        self.action_dim = int(dynamics.action_dim)
        self.q_terminal = float(q_terminal)
        self.q_stage = float(q_stage)
        self.r_control = float(r_control)
        self.num_samples = int(num_samples)
        self.num_iters = int(num_iters)
        self.topk = int(topk)
        self.var_scale = float(var_scale)
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(int(seed))

    def _initial_distribution(self, init_action: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if init_action is None:
            mean = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
        else:
            mean = init_action.to(device=self.device, dtype=torch.float32)
            if mean.ndim != 2 or mean.shape[-1] != self.action_dim:
                raise ValueError(f"Expected init_action shape (T, {self.action_dim}), got {tuple(mean.shape)}.")
            if mean.shape[0] < self.horizon:
                pad = torch.zeros((self.horizon - mean.shape[0], self.action_dim), dtype=torch.float32, device=self.device)
                mean = torch.cat((mean, pad), dim=0)
            else:
                mean = mean[: self.horizon]
        std = self.var_scale * torch.ones_like(mean)
        return mean, std

    @torch.inference_mode()
    def _cost(self, x0: torch.Tensor, x_goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = x0.unsqueeze(0).expand(actions.shape[0], -1)
        costs = torch.zeros((actions.shape[0],), dtype=torch.float32, device=self.device)
        for step in range(self.horizon):
            err = x - x_goal.unsqueeze(0)
            costs = costs + self.q_stage * torch.sum(err * err, dim=-1)
            costs = costs + self.r_control * torch.sum(actions[:, step] * actions[:, step], dim=-1)
            x = self.dynamics.step(x, actions[:, step])
        terminal_err = x - x_goal.unsqueeze(0)
        return costs + self.q_terminal * torch.sum(terminal_err * terminal_err, dim=-1)

    @torch.inference_mode()
    def solve(
        self,
        x0_np: np.ndarray,
        x_goal_np: np.ndarray,
        init_action: torch.Tensor | None,
    ) -> tuple[np.ndarray, float, float]:
        if self.num_samples < 2:
            raise ValueError("--cem-samples must be at least 2.")
        if not (1 <= self.topk <= self.num_samples):
            raise ValueError("--cem-topk must be in [1, --cem-samples].")
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.tensor(x_goal_np, dtype=torch.float32, device=self.device)
        mean, std = self._initial_distribution(init_action)

        base.maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()
        best_cost = float("nan")
        for _ in range(self.num_iters):
            noise = torch.randn(
                (self.num_samples, self.horizon, self.action_dim),
                generator=self.generator,
                device=self.device,
            )
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates[0] = mean
            costs = self._cost(x0, x_goal, candidates)
            elite_costs, elite_idx = torch.topk(costs, k=self.topk, largest=False)
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0)
            best_cost = float(elite_costs[0].item())
        base.maybe_cuda_synchronize(self.device)
        return mean.detach().cpu().numpy().astype(np.float32), time.perf_counter() - t0, best_cost


class CEMPolicy:
    def __init__(
        self,
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        solver: CEMMPCSolver,
        receding_horizon: int,
        warm_start: bool,
    ) -> None:
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)
        self.solver = solver
        self.receding_horizon = int(receding_horizon)
        self.warm_start = bool(warm_start)
        self.action_buffer: deque[np.ndarray] = deque(maxlen=self.receding_horizon)
        self.next_init: torch.Tensor | None = None
        self.previous_embedding: torch.Tensor | None = None
        self.goal_state: np.ndarray | None = None

    def reset(self, goal_embedding: torch.Tensor) -> None:
        self.action_buffer.clear()
        self.next_init = None
        self.previous_embedding = None
        self.goal_state = base.make_markov_state(goal_embedding).detach().cpu().numpy().astype(np.float32)

    def get_action(self, current_embedding: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if self.goal_state is None:
            raise RuntimeError("Call reset() before get_action().")
        if len(self.action_buffer) == 0:
            current_state = base.make_markov_state(current_embedding, self.previous_embedding)
            current_state_np = current_state.detach().cpu().numpy().astype(np.float32)
            u_plan, solve_time, plan_cost = self.solver.solve(current_state_np, self.goal_state, self.next_init)
            keep = min(self.receding_horizon, u_plan.shape[0])
            self.next_init = torch.tensor(u_plan[keep:], dtype=torch.float32) if self.warm_start else None
            for action in u_plan[:keep]:
                self.action_buffer.append(action.astype(np.float32))
            record = {
                "solve_time_ms": float(solve_time * 1000.0),
                "cem_cost": float(plan_cost),
                "cem_replanned": 1.0,
            }
        else:
            record = {"solve_time_ms": 0.0, "cem_cost": float("nan"), "cem_replanned": 0.0}

        action_norm = self.action_buffer.popleft().astype(np.float32)
        action_raw = base.normalized_to_raw_action(action_norm, self.action_mean, self.action_std)
        self.previous_embedding = current_embedding
        return action_raw, action_norm, record


def load_assets(args: argparse.Namespace, device: torch.device) -> tuple[CEMPolicy, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor]:
    model_dir = args.model_dir.expanduser().resolve()
    config = base.load_config(model_dir)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else base.latest_object_checkpoint(model_dir).resolve()
    model = base.load_model(checkpoint, device)

    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for the Markov MLP model, got {history_size}.")
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))

    stats_dataset_path = args.stats_dataset_path.expanduser().resolve() if args.stats_dataset_path is not None else args.dataset_path
    stats_dataset = base.LeWMReacherDataset(
        stats_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )

    dynamics = base.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    solver = CEMMPCSolver(
        dynamics,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        num_samples=args.cem_samples,
        num_iters=args.cem_iters,
        topk=args.cem_topk,
        var_scale=args.cem_var_scale,
        seed=args.seed,
        device=device,
    )
    policy = CEMPolicy(
        action_mean=stats_dataset.action_mean.astype(np.float32),
        action_std=stats_dataset.action_std.astype(np.float32),
        solver=solver,
        receding_horizon=args.receding_horizon,
        warm_start=not bool(args.no_warm_start),
    )
    method_config = {
        "planner": "cem",
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "stats_dataset_path": str(stats_dataset_path),
        "img_size": img_size,
        "action_dim": action_dim,
        "markov_state_dim": markov_state_dim,
        "horizon": int(args.horizon),
        "receding_horizon": int(args.receding_horizon),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "cem_samples": int(args.cem_samples),
        "cem_iters": int(args.cem_iters),
        "cem_topk": int(args.cem_topk),
        "cem_var_scale": float(args.cem_var_scale),
        "warm_start": not bool(args.no_warm_start),
    }
    return policy, model, method_config, stats_dataset.pixel_mean, stats_dataset.pixel_std


def summarize_values(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "count": int(finite.size),
    }


def run_case(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    device: torch.device,
    assets: tuple[CEMPolicy, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    policy, model, method_config, pixel_mean, pixel_std = assets
    img_size = int(method_config["img_size"])
    episode = base.load_dataset_episode(args.dataset_path, case.episode_idx)
    pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    obs_np = np.asarray(episode["observation"], dtype=np.float32)

    start_frame = pixels_np[case.start_step]
    goal_frame = pixels_np[case.goal_step]
    goal_obs = obs_np[case.goal_step].astype(np.float32)
    goal_qpos = qpos_np[case.goal_step].astype(np.float32)
    goal_qvel = qvel_np[case.goal_step].astype(np.float32)

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    base.save_rgb_image(case_dir / "start_image.png", start_frame)
    base.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    env = base.make_render_env(
        seed=int(episode["episode_seed"]),
        time_limit=float(episode["time_limit"]),
        width=int(episode["width"]),
        height=int(episode["height"]),
        physics_freq_hz=float(episode["physics_freq_hz"]),
    )
    current_frame = base.reset_env_to_state(
        env,
        seed=int(episode["episode_seed"]),
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        height=int(episode["height"]),
        width=int(episode["width"]),
    )
    current_obs = flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    qpos, qvel = current_qpos_qvel(env, qpos_np.shape[1], qvel_np.shape[1])

    goal_embedding = base.encode_single_frame(
        model,
        goal_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    policy.reset(goal_embedding)

    rollout_frames = [current_frame.copy()]
    obs_distances = [float(np.linalg.norm(current_obs - goal_obs))]
    qpos_distances = [wrapped_qpos_distance(qpos, goal_qpos)]
    raw_qpos_distances = [float(np.linalg.norm(qpos - goal_qpos))]
    qvel_distances = [float(np.linalg.norm(qvel - goal_qvel))]
    step_records: list[dict[str, float]] = []
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    stop_reason = "eval_budget"
    success = qpos_distances[-1] <= float(args.qpos_success_threshold)
    if success:
        stop_reason = "goal_reached"

    for step in range(int(args.eval_budget)):
        if success:
            break
        current_embedding = base.encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        action_raw, action_norm, record = policy.get_action(current_embedding)
        executed_actions_raw.append(np.asarray(action_raw, dtype=np.float32).copy())
        executed_actions_norm.append(np.asarray(action_norm, dtype=np.float32).copy())

        obs, _, terminated, truncated, _ = env.step(action_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=int(episode["height"]), width=int(episode["width"]), camera_id=0)
        qpos, qvel = current_qpos_qvel(env, qpos_np.shape[1], qvel_np.shape[1])

        obs_distance = float(np.linalg.norm(current_obs - goal_obs))
        qpos_distance = wrapped_qpos_distance(qpos, goal_qpos)
        raw_qpos_distance = float(np.linalg.norm(qpos - goal_qpos))
        qvel_distance = float(np.linalg.norm(qvel - goal_qvel))
        obs_distances.append(obs_distance)
        qpos_distances.append(qpos_distance)
        raw_qpos_distances.append(raw_qpos_distance)
        qvel_distances.append(qvel_distance)
        rollout_frames.append(current_frame.copy())
        step_records.append(
            {
                "step": float(step + 1),
                "observation_distance": obs_distance,
                "qpos_distance": qpos_distance,
                "raw_qpos_distance": raw_qpos_distance,
                "qvel_distance": qvel_distance,
                **record,
            }
        )

        success = qpos_distance <= float(args.qpos_success_threshold)
        if success:
            stop_reason = "goal_reached"
        elif terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
        if success or terminated or truncated:
            break

    video_path = None
    if not args.no_videos:
        video_path = str(base.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
    env.close()

    strict_observation_success = float(np.min(obs_distances)) <= float(args.success_threshold)
    qpos_success = float(np.min(qpos_distances)) <= float(args.qpos_success_threshold)
    summary = {
        **asdict(case),
        "success": bool(success),
        "success_metric": "wrapped_qpos_l2",
        "primary_success_threshold": float(args.qpos_success_threshold),
        "strict_observation_success": bool(strict_observation_success),
        "qpos_success": bool(qpos_success),
        "initial_observation_distance": float(obs_distances[0]),
        "final_observation_distance": float(obs_distances[-1]),
        "min_observation_distance": float(np.min(obs_distances)),
        "initial_qpos_distance": float(qpos_distances[0]),
        "final_qpos_distance": float(qpos_distances[-1]),
        "min_qpos_distance": float(np.min(qpos_distances)),
        "initial_raw_qpos_distance": float(raw_qpos_distances[0]),
        "final_raw_qpos_distance": float(raw_qpos_distances[-1]),
        "min_raw_qpos_distance": float(np.min(raw_qpos_distances)),
        "initial_qvel_distance": float(qvel_distances[0]),
        "final_qvel_distance": float(qvel_distances[-1]),
        "min_qvel_distance": float(np.min(qvel_distances)),
        "steps_executed": len(step_records),
        "cem_replans": int(sum(float(record.get("cem_replanned", 0.0)) >= 0.5 for record in step_records)),
        "total_planning_time_ms": float(sum(float(record.get("solve_time_ms", 0.0)) for record in step_records)),
        "stop_reason": stop_reason,
        "episode_seed": int(episode["episode_seed"]),
        "goal_qpos": goal_qpos.tolist(),
        "final_qpos": qpos.tolist(),
        "goal_qvel": goal_qvel.tolist(),
        "final_qvel": qvel.tolist(),
        "goal_observation": goal_obs.tolist(),
        "final_observation": current_obs.tolist(),
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "video_path": video_path,
        "step_records": step_records,
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_metrics(
    *,
    args: argparse.Namespace,
    method_config: dict[str, Any],
    cases: list[EvalCase],
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    successes = np.asarray([case["success"] for case in case_results], dtype=bool)
    strict_successes = np.asarray([case["strict_observation_success"] for case in case_results], dtype=bool)
    qpos_successes = np.asarray([case["qpos_success"] for case in case_results], dtype=bool)
    solve_step_ms = [
        float(record["solve_time_ms"])
        for case in case_results
        for record in case["step_records"]
        if "solve_time_ms" in record
    ]
    solve_replan_ms = [
        float(record["solve_time_ms"])
        for case in case_results
        for record in case["step_records"]
        if float(record.get("cem_replanned", 0.0)) >= 0.5
    ]
    return {
        "method": "cem",
        "method_config": method_config,
        "success_rate": 0.0 if successes.size == 0 else float(np.mean(successes) * 100.0),
        "strict_observation_success_rate": 0.0 if strict_successes.size == 0 else float(np.mean(strict_successes) * 100.0),
        "qpos_success_rate": 0.0 if qpos_successes.size == 0 else float(np.mean(qpos_successes) * 100.0),
        "success_metric": "wrapped_qpos_l2",
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "num_eval": len(case_results),
        "requested_num_eval": int(args.num_eval),
        "eval_budget": int(args.eval_budget),
        "success_threshold": float(args.success_threshold),
        "qpos_success_threshold": float(args.qpos_success_threshold),
        "goal_protocol": "start_step_0_to_final_episode_step",
        "episode_indices": [int(case.episode_idx) for case in cases],
        "min_qpos_distance": summarize_values([float(case["min_qpos_distance"]) for case in case_results]),
        "final_qpos_distance": summarize_values([float(case["final_qpos_distance"]) for case in case_results]),
        "steps_executed": summarize_values([float(case["steps_executed"]) for case in case_results]),
        "cem_replans_per_episode": summarize_values([float(case["cem_replans"]) for case in case_results]),
        "solve_time_ms_per_control_step": summarize_values(solve_step_ms),
        "solve_time_ms_per_replan": summarize_values(solve_replan_ms),
        "total_planning_time_ms_per_episode": summarize_values([float(case["total_planning_time_ms"]) for case in case_results]),
        "runtime_note": (
            "solve_time_ms_per_control_step includes zero-time cached receding-horizon actions; "
            "solve_time_ms_per_replan measures actual CEM optimizer calls."
        ),
        "cases": case_results,
    }


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.receding_horizon < 1 or args.receding_horizon > args.horizon:
        raise ValueError("--receding-horizon must be in [1, --horizon].")
    if args.cem_iters < 1:
        raise ValueError("--cem-iters must be positive.")

    device = base.require_device(args.device)
    ep_len = load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)
    assets = load_assets(args, device)
    method_config = assets[2]

    run_name = f"{int(time.time())}_cem_ablation_seed_{args.seed}"
    out_root = args.out_dir.expanduser().resolve() / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    for case_idx, case in enumerate(tqdm(cases, desc="Reacher CEM planner ablation")):
        case_results.append(
            run_case(
                args=args,
                case=case,
                case_idx=case_idx,
                out_root=out_root,
                device=device,
                assets=assets,
            )
        )
        partial = build_metrics(args=args, method_config=method_config, cases=cases, case_results=case_results)
        partial["partial"] = True
        with (out_root / "metrics_partial.json").open("w", encoding="utf-8") as handle:
            json.dump(partial, handle, indent=2)

    metrics = build_metrics(args=args, method_config=method_config, cases=cases, case_results=case_results)
    metrics["partial"] = False
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"qpos_success_rate: {metrics['qpos_success_rate']:.2f}")
    print(f"solve_time_ms_per_replan: {metrics['solve_time_ms_per_replan']['mean']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
