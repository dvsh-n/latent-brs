#!/usr/bin/env python3
"""Closed-loop Reacher benchmark with a latent HJ safety filter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.plan import benchmark_reacher_hard as hard
from reacher.plan import plan_ilqr_mpc as ilqr_base
from reacher.safety.hj_filter import ReacherHJSafetyFilter, parse_hidden_sizes

HAORAN_ILQR_PATH = (
    Path(__file__).resolve().parents[1]
    / "Haoran_obs_data"
    / "reacher_stuff"
    / "reacher_stuff"
    / "plan_ilqr_mpc.py"
)
DEFAULT_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/Haoran_obs_data/mlpdyn_ft_6"
DEFAULT_HJ_CACHE = "reacher/safety/cache/reacher_latent_safety_classifier_train_tanh.pt"
DEFAULT_HJ_POLICY = "reacher/safety/runs/pyhj_train_tanh/policy_latest.pth"
DEFAULT_CLASSIFIER = "reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt"
DEFAULT_OUT_DIR = "reacher/safety/runs/closed_loop_hj_filter"
DEFAULT_START_GOAL_PATH = "reacher/Haoran_obs_data/reacher_stuff/reacher_stuff/start_goal.pt"
GEOM_BOX_LOWER = np.asarray([0.0, -2.88], dtype=np.float32)
GEOM_BOX_UPPER = np.asarray([3.1415, -2.45], dtype=np.float32)
GEOM_INSIDE_BEND_SIGN = -1


def load_haoran_ilqr_module() -> Any:
    spec = importlib.util.spec_from_file_location("reacher_haoran_plan_ilqr_mpc", HAORAN_ILQR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Haoran ILQR planner from {HAORAN_ILQR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


haoran_ilqr = load_haoran_ilqr_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-method", choices=("ilqr",), default="ilqr")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--start-goal-path", type=Path, default=None)
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument(
        "--episode-indices-file",
        type=Path,
        default=None,
        help="JSON file containing an episode_indices list to evaluate exactly, e.g. a held-out unsafe set.",
    )
    parser.add_argument("--eval-budget", type=int, default=ilqr_base.MAX_MPC_STEPS)
    parser.add_argument("--success-threshold", type=float, default=hard.DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--qpos-success-threshold", type=float, default=hard.DEFAULT_QPOS_SUCCESS_THRESHOLD)
    parser.add_argument("--video-fps", type=int, default=ilqr_base.VIDEO_FPS)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--mode", choices=("paired", "nominal", "filtered", "lpb", "all"), default="paired")

    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=ilqr_base.HORIZON)
    parser.add_argument("--q-terminal", type=float, default=ilqr_base.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=ilqr_base.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=ilqr_base.R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--lpb-bank-path", type=Path, default=None)
    parser.add_argument("--lpb-weight", type=float, default=1.0)
    parser.add_argument("--lpb-threshold-scale", type=float, default=1.0)
    parser.add_argument("--lpb-stage-only", action=argparse.BooleanOptionalAction, default=True)

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


class HaoranILQRPolicyAdapter:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: dict[str, object],
        action_mean: np.ndarray,
        action_std: np.ndarray,
        device: torch.device,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        ilqr_max_iters: int,
        ilqr_tol: float,
        ilqr_regularization: float,
        lpb_barrier: Any | None = None,
        lpb_weight: float = 0.0,
        lpb_stage_only: bool = True,
    ) -> None:
        action_dim = int(config.get("action_dim", 2))
        embed_dim = int(config.get("embed_dim", 18))
        history_size = int(config.get("history_size", 1))
        if history_size != 1:
            raise ValueError(f"Haoran ILQR planner expects history_size=1, got {history_size}.")
        markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
        dynamics = haoran_ilqr.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
        self.model = model
        self.device = device
        self.action_mean = action_mean
        self.action_std = action_std
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.state_mode = "haoran_markov_embedding_cost_only"
        self.solver = haoran_ilqr.ILQRMPCSolver(
            dynamics,
            state_cost_dim=embed_dim,
            horizon=horizon,
            q_terminal=q_terminal,
            q_stage=q_stage,
            r_control=r_control,
            max_iters=ilqr_max_iters,
            tol=ilqr_tol,
            regularization=ilqr_regularization,
            device=device,
            lpb_barrier=lpb_barrier,
            lpb_weight=lpb_weight,
            lpb_stage_only=lpb_stage_only,
        )
        self.goal_state: np.ndarray | None = None
        self.previous_embedding: torch.Tensor | None = None

    def set_lpb_enabled(self, enabled: bool) -> None:
        self.solver.lpb_enabled = bool(enabled) and self.solver.lpb_barrier is not None and self.solver.lpb_weight > 0.0

    def reset(self, *, start_embedding: torch.Tensor, goal_embedding: torch.Tensor) -> None:
        self.previous_embedding = None
        goal_state = haoran_ilqr.make_markov_state(goal_embedding)
        self.goal_state = goal_state.detach().cpu().numpy().astype(np.float64)

    def get_action(self, current_embedding: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        if self.goal_state is None:
            raise RuntimeError("HaoranILQRPolicyAdapter.reset() must be called before get_action().")
        current_state = haoran_ilqr.make_markov_state(current_embedding, self.previous_embedding)
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, solve_time, n_iters, plan_cost = self.solver.solve(current_state_np, self.goal_state)
        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = haoran_ilqr.normalized_to_raw_action(u0_norm, self.action_mean, self.action_std)
        self.previous_embedding = current_embedding
        return u0_raw, {
            "solve_time_ms": float(solve_time * 1000.0),
            "ilqr_iterations": float(n_iters),
            "plan_cost": float(plan_cost),
            **self.solver.last_lpb_diagnostics,
        }


def load_haoran_ilqr_assets(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[HaoranILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    config = haoran_ilqr.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else haoran_ilqr.latest_object_checkpoint(model_dir).resolve()
    )
    model = haoran_ilqr.load_model(checkpoint_path, device)
    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected Haoran ILQR model history_size=1, got {history_size}.")
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    lpb_barrier = None
    lpb_bank_path = getattr(args, "lpb_bank_path", None)
    if lpb_bank_path is not None:
        from reacher.safety.lpb.barrier import ReacherLPBBarrier

        lpb_barrier = ReacherLPBBarrier(
            lpb_bank_path,
            device=device,
            threshold_scale=float(getattr(args, "lpb_threshold_scale", 1.0)),
        )
        if int(lpb_barrier.state_dim) != int(markov_state_dim):
            raise ValueError(
                "LPB bank Markov-state dimension does not match the Reacher HJ iLQR model: "
                f"bank={lpb_barrier.state_dim}, model={markov_state_dim}"
            )
    stats_dataset_path = (
        args.stats_dataset_path.expanduser().resolve()
        if args.stats_dataset_path is not None
        else args.dataset_path
    )
    if not stats_dataset_path.is_file():
        raise FileNotFoundError(
            f"Stats dataset not found: {stats_dataset_path}. "
            "Pass --stats-dataset-path explicitly for Haoran-style action stats."
        )
    pixel_mean, pixel_std = haoran_ilqr.imagenet_pixel_stats(device)
    action_mean, action_std = haoran_ilqr.load_action_stats([stats_dataset_path], action_dim)
    policy = HaoranILQRPolicyAdapter(
        model=model,
        config=config,
        action_mean=action_mean,
        action_std=action_std,
        device=device,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        ilqr_max_iters=args.ilqr_max_iters,
        ilqr_tol=args.ilqr_tol,
        ilqr_regularization=args.ilqr_regularization,
        lpb_barrier=lpb_barrier,
        lpb_weight=float(getattr(args, "lpb_weight", 1.0)),
        lpb_stage_only=bool(getattr(args, "lpb_stage_only", True)),
    )
    policy.set_lpb_enabled(lpb_barrier is not None)
    method_config = {
        "planner_source": str(HAORAN_ILQR_PATH),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "stats_dataset_path": str(stats_dataset_path),
        "pixel_stats": "imagenet",
        "action_stats": "hdf5_raw_actions",
        "img_size": img_size,
        "history_size": history_size,
        "embed_dim": int(config.get("embed_dim", 18)),
        "markov_state_dim": int(config.get("markov_state_dim", 2 * int(config.get("embed_dim", 18)))),
        "ilqr_state_mode": policy.state_mode,
        "state_cost_dim": int(config.get("embed_dim", 18)),
        "horizon": int(args.horizon),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "ilqr_max_iters": int(args.ilqr_max_iters),
        "lpb": None
        if lpb_barrier is None
        else {
            "bank_path": str(lpb_bank_path.expanduser().resolve()),
            "weight": float(getattr(args, "lpb_weight", 1.0)),
            "threshold": float(lpb_barrier.threshold),
            "threshold_scale": float(getattr(args, "lpb_threshold_scale", 1.0)),
            "scaled_threshold": float(lpb_barrier.scaled_threshold),
            "stage_only": bool(getattr(args, "lpb_stage_only", True)),
            "metadata": lpb_barrier.metadata,
        },
    }
    return policy, model, method_config, pixel_mean, pixel_std, checkpoint_path


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


def reset_ilqr_policy(
    *,
    ilqr_assets: tuple[Any, ...],
    start_frame: np.ndarray,
    goal_frame: np.ndarray,
    device: torch.device,
    frame_batch_size: int,
) -> HaoranILQRPolicyAdapter:
    ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
    ilqr_policy.solver.prev_u_guess.zero_()
    img_size = int(ilqr_config["img_size"])
    pixels = haoran_ilqr.preprocess_pixels(
        np.stack([start_frame, goal_frame], axis=0),
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    start_emb, goal_emb = haoran_ilqr.encode_frames(
        ilqr_model,
        pixels,
        device=device,
        frame_batch_size=frame_batch_size,
    )
    ilqr_policy.reset(start_embedding=start_emb, goal_embedding=goal_emb)
    return ilqr_policy


def load_start_goal_pairs(path: Path, num_eval: int) -> list[dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    pairs = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"Expected {path} to contain a list of start/goal pairs.")
    if num_eval > len(pairs):
        raise ValueError(f"Requested {num_eval} pairs, but {path} only contains {len(pairs)}.")
    return list(pairs[:num_eval])


def qpos_geometry_unsafe(qpos: np.ndarray) -> bool:
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    inside_box = bool(np.all((qpos_arr >= GEOM_BOX_LOWER) & (qpos_arr <= GEOM_BOX_UPPER)))
    selected_ik_branch = bool(float(GEOM_INSIDE_BEND_SIGN) * np.sin(float(qpos_arr[1])) > 0.0)
    return inside_box and selected_ik_branch


def load_eval_cases(args: argparse.Namespace, ep_len: np.ndarray | None) -> list[hard.EvalCase]:
    if args.start_goal_path is not None:
        if args.episode_idx is not None or args.episode_indices_file is not None:
            raise ValueError("--start-goal-path uses the first --num-eval pairs; do not combine with episode selectors.")
        return [hard.EvalCase(int(idx), 0, 1, 2) for idx in range(int(args.num_eval))]
    if ep_len is None:
        raise ValueError("ep_len is required when --start-goal-path is not used.")
    if args.episode_indices_file is None:
        return hard.sample_eval_cases(args, ep_len)
    if args.episode_idx is not None:
        raise ValueError("Use either --episode-idx or --episode-indices-file, not both.")

    path = args.episode_indices_file.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        episode_indices = payload.get("episode_indices")
    else:
        episode_indices = payload
    if not isinstance(episode_indices, list) or not episode_indices:
        raise ValueError(f"{path} must contain a non-empty episode_indices list.")

    cases: list[hard.EvalCase] = []
    seen: set[int] = set()
    for episode_idx_raw in episode_indices:
        episode_idx = int(episode_idx_raw)
        if episode_idx in seen:
            raise ValueError(f"Duplicate episode index in {path}: {episode_idx}")
        seen.add(episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"Episode {episode_idx} is outside [0, {ep_len.shape[0] - 1}].")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"Episode {episode_idx} has length {ep_len[episode_idx]}, expected at least 2.")
        cases.append(hard.EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx])))

    args.num_eval = len(cases)
    return cases


def run_one_case(
    *,
    args: argparse.Namespace,
    case: hard.EvalCase,
    case_idx: int,
    run_dir: Path,
    label: str,
    use_filter: bool,
    use_lpb: bool,
    device: torch.device,
    ilqr_assets: tuple[Any, ...],
    hj_filter: ReacherHJSafetyFilter,
    start_goal_pair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if start_goal_pair is None:
        episode = ilqr_base.load_dataset_episode(args.dataset_path, case.episode_idx)
        pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
        qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
        qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
        obs_np = np.asarray(episode["observation"], dtype=np.float32)
        episode_seed = int(episode["episode_seed"])
        physics_freq_hz = float(episode["physics_freq_hz"])
        time_limit = float(episode["time_limit"])
        height = int(episode["height"])
        width = int(episode["width"])
        start_frame = pixels_np[case.start_step]
        goal_frame = pixels_np[case.goal_step]
        goal_obs = obs_np[case.goal_step].astype(np.float32)
        goal_qpos = qpos_np[case.goal_step].astype(np.float32)
        goal_qvel = qvel_np[case.goal_step].astype(np.float32)
        start_qpos = qpos_np[case.start_step]
        start_qvel = qvel_np[case.start_step]
    else:
        episode_seed = int(args.seed) + int(case_idx)
        physics_freq_hz = 100.0
        time_limit = 10.0
        height = width = 224
        start = start_goal_pair["start"]
        goal = start_goal_pair["goal"]
        start_qpos = np.asarray(start["qpos"], dtype=np.float32)
        start_qvel = np.asarray(start.get("qvel", np.zeros_like(start_qpos)), dtype=np.float32)
        goal_qpos = np.asarray(goal["qpos"], dtype=np.float32)
        goal_qvel = np.asarray(goal.get("qvel", np.zeros_like(goal_qpos)), dtype=np.float32)
        obs_np = qpos_np = qvel_np = None

    case_dir = run_dir / label / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    env = ilqr_base.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    if start_goal_pair is None:
        goal_obs = goal_obs
    else:
        goal_frame = ilqr_base.reset_env_to_state(
            env,
            seed=episode_seed,
            qpos=goal_qpos,
            qvel=goal_qvel,
            height=height,
            width=width,
        )
        goal_obs = hard.flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    current_frame = ilqr_base.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=start_qvel,
        height=height,
        width=width,
    )
    if start_goal_pair is not None:
        start_frame = current_frame.copy()
    ilqr_base.save_rgb_image(case_dir / "start_image.png", start_frame)
    ilqr_base.save_rgb_image(case_dir / "goal_image.png", goal_frame)
    current_obs = hard.flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    qpos, qvel = hard.current_qpos_qvel(env, goal_qpos.shape[0], goal_qvel.shape[0])

    ilqr_policy = reset_ilqr_policy(
        ilqr_assets=ilqr_assets,
        start_frame=current_frame,
        goal_frame=goal_frame,
        device=device,
        frame_batch_size=int(args.frame_batch_size),
    )
    ilqr_policy.set_lpb_enabled(use_lpb)
    hj_filter.reset(current_frame)
    initial_safety = hj_filter.evaluate_state(hj_filter.current_state(), "initial")

    rollout_frames = [current_frame.copy()]
    obs_distances = [float(np.linalg.norm(current_obs - goal_obs))]
    qpos_distances = [hard.wrapped_qpos_distance(qpos, goal_qpos)]
    raw_qpos_distances = [float(np.linalg.norm(qpos - goal_qpos))]
    qvel_distances = [float(np.linalg.norm(qvel - goal_qvel))]
    step_records: list[dict[str, Any]] = []
    qpos_trajectory = [qpos.astype(np.float32).tolist()]
    qvel_trajectory = [qvel.astype(np.float32).tolist()]
    executed_actions_raw: list[list[float]] = []
    executed_actions_norm: list[list[float]] = []
    overrides = 0
    learned_safety_violations = bool(initial_safety["initial_l"] <= 0.0)
    geometric_safety_violations = qpos_geometry_unsafe(qpos)

    success = qpos_distances[-1] <= float(args.qpos_success_threshold)
    stop_reason = "goal_reached" if success else "eval_budget"

    for step in range(0 if success else int(args.eval_budget)):
        current_emb = haoran_ilqr.encode_single_frame(
            ilqr_assets[1],
            current_frame,
            device=device,
            img_size=int(ilqr_assets[2]["img_size"]),
            pixel_mean=ilqr_assets[3],
            pixel_std=ilqr_assets[4],
        )
        nominal_raw, base_record = ilqr_policy.get_action(current_emb)
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

        obs, _, terminated, truncated, _ = env.step(action_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        qpos, qvel = hard.current_qpos_qvel(env, goal_qpos.shape[0], goal_qvel.shape[0])
        ilqr_policy.previous_embedding = current_emb
        hj_filter.append_frame(current_frame)
        post = hj_filter.evaluate_state(hj_filter.current_state(), "post")
        learned_safety_violations = learned_safety_violations or bool(post["post_l"] <= 0.0)
        geometric_unsafe = qpos_geometry_unsafe(qpos)
        geometric_safety_violations = geometric_safety_violations or geometric_unsafe

        obs_distance = float(np.linalg.norm(current_obs - goal_obs))
        qpos_distance = hard.wrapped_qpos_distance(qpos, goal_qpos)
        raw_qpos_distance = float(np.linalg.norm(qpos - goal_qpos))
        qvel_distance = float(np.linalg.norm(qvel - goal_qvel))
        obs_distances.append(obs_distance)
        qpos_distances.append(qpos_distance)
        raw_qpos_distances.append(raw_qpos_distance)
        qvel_distances.append(qvel_distance)
        rollout_frames.append(current_frame.copy())
        executed_actions_raw.append(np.asarray(action_raw, dtype=np.float32).tolist())
        executed_actions_norm.append(np.asarray(action_norm, dtype=np.float32).tolist())
        qpos_trajectory.append(qpos.astype(np.float32).tolist())
        qvel_trajectory.append(qvel.astype(np.float32).tolist())
        step_records.append(
            {
                "step": int(step + 1),
                "hj_filter_enabled": bool(use_filter),
                "lpb_guided": bool(use_lpb),
                "observation_distance": obs_distance,
                "qpos_distance": qpos_distance,
                "raw_qpos_distance": raw_qpos_distance,
                "qvel_distance": qvel_distance,
                **base_record,
                **record,
                **post,
                "post_classifier_unsafe": bool(post["post_l"] <= 0.0),
                "post_geometry_unsafe": bool(geometric_unsafe),
            }
        )

        success = qpos_distance <= float(args.qpos_success_threshold)
        if success:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    video_path = None
    if not args.no_videos:
        video_path = str(ilqr_base.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
    env.close()

    min_l = min([initial_safety["initial_l"]] + [float(item["post_l"]) for item in step_records])
    min_v = min([initial_safety["initial_V"]] + [float(item["post_V"]) for item in step_records])
    min_b = min([initial_safety["initial_B"]] + [float(item["post_B"]) for item in step_records])
    strict_observation_success = float(np.min(obs_distances)) <= float(args.success_threshold)
    qpos_success = float(np.min(qpos_distances)) <= float(args.qpos_success_threshold)
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
        "success_metric": "wrapped_qpos_l2",
        "primary_success_threshold": float(args.qpos_success_threshold),
        "strict_observation_success": bool(strict_observation_success),
        "qpos_success": bool(qpos_success),
        "policy_steps_executed": int(len(step_records)),
        "steps_executed": int(len(step_records)),
        "stop_reason": stop_reason,
        "episode_seed": episode_seed,
        "override_count": int(overrides),
        "override_rate": float(overrides / max(len(step_records), 1)),
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
        "goal_qpos": goal_qpos.tolist(),
        "final_qpos": qpos.tolist(),
        "goal_qvel": goal_qvel.tolist(),
        "final_qvel": qvel.tolist(),
        "goal_observation": goal_obs.tolist(),
        "final_observation": current_obs.tolist(),
        "video_path": video_path,
        "executed_actions_raw": executed_actions_raw,
        "executed_actions_norm": executed_actions_norm,
        "qpos_trajectory": qpos_trajectory,
        "qvel_trajectory": qvel_trajectory,
        "step_records": step_records,
    }
    hard.save_case_summary(case_dir, jsonable(summary))
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
        "mean_final_qpos_distance": float(np.mean([case["final_qpos_distance"] for case in cases])),
        "mean_lpb_violation_rate": None
        if not any(case.get("lpb_violation_rate_mean") is not None for case in cases)
        else float(np.mean([case["lpb_violation_rate_mean"] for case in cases if case.get("lpb_violation_rate_mean") is not None])),
        "max_lpb_distance": None
        if not any(case.get("lpb_distance_max") is not None for case in cases)
        else float(np.max([case["lpb_distance_max"] for case in cases if case.get("lpb_distance_max") is not None])),
    }


def main() -> None:
    args = parse_args()
    args.method = args.base_method
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if args.start_goal_path is None and not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    start_goal_pairs = None
    if args.start_goal_path is not None:
        args.start_goal_path = args.start_goal_path.expanduser().resolve()
        start_goal_pairs = load_start_goal_pairs(args.start_goal_path, int(args.num_eval))

    device = ilqr_base.require_device(args.device)
    ep_len = None if args.start_goal_path is not None else hard.load_episode_lengths(args.dataset_path)
    cases = load_eval_cases(args, ep_len)

    run_name = f"{int(time.time())}_{args.base_method}_hj_seed_{args.seed}"
    run_root = args.out_dir.expanduser().resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    ilqr_assets = load_haoran_ilqr_assets(args, device)
    method_config = ilqr_assets[2]
    hj_filter = ReacherHJSafetyFilter(
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

    labels: list[tuple[str, bool, bool]] = {
        "paired": [("nominal", False, False), ("hj_filtered", True, False)],
        "nominal": [("nominal", False, False)],
        "filtered": [("hj_filtered", True, False)],
        "lpb": [("lpb_guided", False, True)],
        "all": [
            ("nominal", False, False),
            ("hj_filtered", True, False),
            ("lpb_guided", False, True),
        ],
    }[args.mode]
    if any(use_lpb for _, _, use_lpb in labels) and args.lpb_bank_path is None:
        raise ValueError(f"--mode {args.mode} requires --lpb-bank-path.")

    all_results: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in labels}
    for case_idx, case in enumerate(tqdm(cases, desc="Closed-loop HJ reacher eval")):
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
        "seed": int(args.seed),
        "requested_num_eval": int(args.num_eval),
        "evaluated_case_count": int(len(cases)),
        "episode_indices_file": str(args.episode_indices_file.expanduser().resolve())
        if args.episode_indices_file is not None
        else None,
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
        "lpb": method_config.get("lpb"),
        "evaluated_episode_indices": [int(item.episode_idx) for item in cases],
        "aggregates": {label: aggregate(label, result) for label, result in all_results.items()},
        "cases": all_results,
    }
    (run_root / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"saved_to": str(run_root), "aggregates": metrics["aggregates"]}), indent=2))


if __name__ == "__main__":
    main()
