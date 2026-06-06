#!/usr/bin/env python3
"""Hard Reacher benchmark for iLQR and stable-worldmodel policies.

The protocol intentionally uses the hard episode endpoints from
``plan_ilqr_mpc.py``: start at episode step 0 and use the final episode frame
as the goal. Success is pose-only wrapped qpos distance so velocity mismatch
does not turn a visually reached goal into a failure; full observation, raw
qpos, and qvel distances are still logged as diagnostics.
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
STABLE_WORLDMODEL_DIR = ROOT_DIR / "third_party" / "stable-worldmodel"
for path in (ROOT_DIR, STABLE_WORLDMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import hydra
import imageio.v2 as imageio
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

from reacher.plan import plan_ilqr_mpc as ilqr_base
from reacher.train.reacher_policy_train import flatten_observation

DEFAULT_OUT_DIR = "reacher/plan/reacher_hard_eval"
DEFAULT_SOLVER_CONFIG = (
    STABLE_WORLDMODEL_DIR / "scripts" / "plan" / "config" / "solver" / "cem.yaml"
)
DEFAULT_SUCCESS_THRESHOLD = 0.05
DEFAULT_QPOS_SUCCESS_THRESHOLD = 0.1
DEFAULT_NUM_EVAL = 50
DEFAULT_EVAL_BUDGET = ilqr_base.MAX_MPC_STEPS
DEFAULT_SEED = 42
DEFAULT_SWM_HISTORY_SIZE = 3
IMAGENET_NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True)
class EvalCase:
    episode_idx: int
    start_step: int
    goal_step: int
    ep_len: int


def wrapped_angle_diff(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return ((np.asarray(current) - np.asarray(goal) + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def wrapped_qpos_distance(current: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(wrapped_angle_diff(current, goal)))


class SingleVectorEnvAdapter:
    """Tiny vector-env facade expected by stable-worldmodel policies."""

    def __init__(self, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=action_space.low[None, ...],
            high=action_space.high[None, ...],
            dtype=action_space.dtype,
        )


class EpisodeHistory:
    def __init__(
        self,
        *,
        history_size: int,
        action_dim: int,
        case_id: int,
        goal_frame: np.ndarray,
        goal_obs: np.ndarray,
        goal_qpos: np.ndarray,
        goal_qvel: np.ndarray,
        goal_step: int,
    ) -> None:
        self.history_size = int(history_size)
        self.action_dim = int(action_dim)
        self.case_id = int(case_id)
        self.goal_frame = np.asarray(goal_frame, dtype=np.uint8)
        self.goal_obs = np.asarray(goal_obs, dtype=np.float32)
        self.goal_qpos = np.asarray(goal_qpos, dtype=np.float32)
        self.goal_qvel = np.asarray(goal_qvel, dtype=np.float32)
        self.goal_step = int(goal_step)
        self.frames: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.actions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.observations: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qpos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qvel: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.step_idx: deque[int] = deque(maxlen=self.history_size)

    def reset(
        self,
        *,
        frame: np.ndarray,
        observation: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        zero_action = np.zeros((self.action_dim,), dtype=np.float32)
        for _ in range(self.history_size):
            self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
            self.actions.append(zero_action.copy())
            self.observations.append(np.asarray(observation, dtype=np.float32).copy())
            self.qpos.append(np.asarray(qpos, dtype=np.float32).copy())
            self.qvel.append(np.asarray(qvel, dtype=np.float32).copy())
            self.step_idx.append(0)

    def append(
        self,
        *,
        frame: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        step_idx: int,
    ) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.qpos.append(np.asarray(qpos, dtype=np.float32).copy())
        self.qvel.append(np.asarray(qvel, dtype=np.float32).copy())
        self.step_idx.append(int(step_idx))

    def info(self) -> dict[str, np.ndarray]:
        frames = np.stack(list(self.frames), axis=0)
        actions = np.stack(list(self.actions), axis=0)
        observations = np.stack(list(self.observations), axis=0)
        qpos = np.stack(list(self.qpos), axis=0)
        qvel = np.stack(list(self.qvel), axis=0)
        step_idx = np.asarray(list(self.step_idx), dtype=np.int64)
        ids = np.full((self.history_size,), self.case_id, dtype=np.int64)
        goal_ids = np.asarray([self.case_id], dtype=np.int64)
        goal_step_idx = np.asarray([self.goal_step], dtype=np.int64)

        return {
            "pixels": frames[None, ...],
            "goal": self.goal_frame[None, None, ...],
            "action": actions[None, ...],
            "observation": observations[None, ...],
            "goal_observation": self.goal_obs[None, None, ...],
            "qpos": qpos[None, ...],
            "goal_qpos": self.goal_qpos[None, None, ...],
            "qvel": qvel[None, ...],
            "goal_qvel": self.goal_qvel[None, None, ...],
            "step_idx": step_idx[None, ...],
            "goal_step_idx": goal_step_idx[None, ...],
            "id": ids[None, ...],
            "goal_id": goal_ids[None, ...],
        }


class ILQRPolicyAdapter:
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
        if history_size == 1:
            markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
            dynamics = ilqr_base.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
            state_mode = "markov_embedding_delta"
        elif history_size == 2:
            dynamics = History2DynamicsTorch(model, embed_dim, action_dim, device)
            state_mode = "history2_embedding_pair"
        else:
            raise ValueError(f"Unsupported ILQR model history_size={history_size}; expected 1 or 2.")
        self.model = model
        self.device = device
        self.action_mean = action_mean
        self.action_std = action_std
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.state_mode = state_mode
        self.solver = ilqr_base.ILQRMPCSolver(
            dynamics,
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
        self.current_embedding: torch.Tensor | None = None

    def set_lpb_enabled(self, enabled: bool) -> None:
        self.solver.lpb_enabled = bool(enabled) and self.solver.lpb_barrier is not None and self.solver.lpb_weight > 0.0

    def reset(self, *, start_embedding: torch.Tensor, goal_embedding: torch.Tensor) -> None:
        self.previous_embedding = None if self.history_size == 1 else start_embedding
        self.current_embedding = start_embedding
        if self.history_size == 1:
            goal_state = ilqr_base.make_markov_state(goal_embedding)
        else:
            goal_state = torch.cat((goal_embedding, goal_embedding), dim=-1)
        self.goal_state = goal_state.detach().cpu().numpy().astype(np.float64)

    def get_action(self, current_embedding: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
        if self.goal_state is None:
            raise RuntimeError("ILQRPolicyAdapter.reset() must be called before get_action().")
        if self.history_size == 1:
            current_state = ilqr_base.make_markov_state(current_embedding, self.previous_embedding)
        else:
            previous = self.previous_embedding if self.previous_embedding is not None else current_embedding
            current_state = torch.cat((previous, current_embedding), dim=-1)
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, solve_time, n_iters, plan_cost = self.solver.solve(current_state_np, self.goal_state)
        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = ilqr_base.normalized_to_raw_action(u0_norm, self.action_mean, self.action_std)
        self.previous_embedding = current_embedding
        self.current_embedding = current_embedding
        return u0_raw, {
            "solve_time_ms": float(solve_time * 1000.0),
            "ilqr_iterations": float(n_iters),
            "plan_cost": float(plan_cost),
            **self.solver.last_lpb_diagnostics,
        }


class History2DynamicsTorch:
    """One-step dynamics for the non-Markov history-2 MLP ablation."""

    def __init__(self, model: torch.nn.Module, embed_dim: int, action_dim: int, device: torch.device) -> None:
        predictor = model.predictor
        if predictor.history_size != 2 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError(
                "History-2 iLQR expects an MLP dynamics model with "
                "history_size=2, action_history_size=1, and num_preds=1."
            )
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("This planner assumes an identity action encoder.")
        if int(predictor.embed_dim) != int(embed_dim):
            raise ValueError(f"Predictor embedding dim mismatch: expected {embed_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != int(action_dim):
            raise ValueError(f"Predictor action dim mismatch: expected {action_dim}, got {predictor.action_dim}.")

        self.predictor = predictor.to(device)
        self.state_dim = int(2 * embed_dim)
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.device = device

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        prev = x[..., : self.embed_dim]
        current = x[..., self.embed_dim :]
        emb_hist = torch.stack((prev, current), dim=-2)
        action = u.unsqueeze(-2)
        pred_next = self.predictor(emb_hist, action)[..., 0, :]
        next_state = torch.cat((current, pred_next), dim=-1)
        return next_state[0] if squeeze else next_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=("ilqr", "swm_cost", "swm_action"),
        default="ilqr",
        help="Policy family to benchmark.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(ilqr_base.DEFAULT_TEST_DATASET_PATH),
        help="Reacher test HDF5 dataset.",
    )
    parser.add_argument(
        "--stats-dataset-path",
        type=Path,
        default=None,
        help="Dataset used to fit iLQR action normalization stats. Defaults to --dataset-path.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Directory for metrics and media.")
    parser.add_argument("--device", default=ilqr_base.DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-eval", type=int, default=DEFAULT_NUM_EVAL, help="Number of episodes to sample when --episode-idx is omitted.")
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Debug override: evaluate exactly this one episode. Omit for normal multi-episode benchmarking.",
    )
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help="Diagnostic strict threshold on full flattened observation distance.",
    )
    parser.add_argument(
        "--qpos-success-threshold",
        type=float,
        default=DEFAULT_QPOS_SUCCESS_THRESHOLD,
        help="Primary success threshold on wrapped joint-position/qpos distance; ignores velocity.",
    )
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=ilqr_base.VIDEO_FPS)
    parser.add_argument("--no-videos", action="store_true", help="Skip rollout videos. Videos are saved by default.")

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(ilqr_base.DEFAULT_MODEL_DIR),
        help="Reacher iLQR dynamics model directory.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional exact iLQR checkpoint path.")
    parser.add_argument("--horizon", type=int, default=ilqr_base.HORIZON)
    parser.add_argument("--q-terminal", type=float, default=ilqr_base.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=ilqr_base.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=ilqr_base.R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--lpb-bank-path", type=Path, default=None)
    parser.add_argument("--lpb-weight", type=float, default=1.0)
    parser.add_argument("--lpb-threshold-scale", type=float, default=1.0)
    parser.add_argument("--lpb-stage-only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--policy", default=None, help="Stable-worldmodel run name, directory, or checkpoint stem.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--solver-config",
        type=Path,
        default=DEFAULT_SOLVER_CONFIG,
        help="Stable-worldmodel solver config, e.g. CEM or Adam YAML.",
    )
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--swm-history-size", type=int, default=None)
    parser.add_argument("--swm-img-size", type=int, default=224)
    parser.add_argument(
        "--process-key",
        action="append",
        default=None,
        help="Dataset column to StandardScale for stable-worldmodel policies. Defaults to action.",
    )
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def require_swm_policy(args: argparse.Namespace) -> str:
    if args.method.startswith("swm") and not args.policy:
        raise ValueError(f"--policy is required for --method {args.method}.")
    return str(args.policy)


def load_episode_lengths(dataset_path: Path) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        return np.asarray(h5["ep_len"][:], dtype=np.int64)


def sample_eval_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[EvalCase]:
    valid = np.flatnonzero(ep_len >= 2)
    if valid.size == 0:
        raise ValueError("Need at least one episode with at least two frames.")
    if args.episode_idx is not None:
        if int(args.num_eval) != 1:
            raise ValueError("--episode-idx pins a single debug episode. Omit --episode-idx for --num-eval > 1.")
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"Episode {episode_idx} has length {ep_len[episode_idx]}, expected at least 2.")
        return [EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx]))]

    if args.num_eval > valid.size:
        raise ValueError(f"Requested {args.num_eval} episodes, but only {valid.size} are valid.")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(valid, size=int(args.num_eval), replace=False))
    return [EvalCase(int(ep), 0, int(ep_len[ep]) - 1, int(ep_len[ep])) for ep in selected]


def fit_processors(dataset_path: Path, keys: list[str]) -> dict[str, preprocessing.StandardScaler]:
    processors: dict[str, preprocessing.StandardScaler] = {}
    with h5py.File(dataset_path, "r") as h5:
        for key in keys:
            if key not in h5:
                raise KeyError(f"Cannot fit processor for missing dataset column '{key}' in {dataset_path}.")
            data = np.asarray(h5[key][:], dtype=np.float32)
            if data.ndim == 1:
                data = data[:, None]
            data = data.reshape(data.shape[0], -1)
            data = data[~np.isnan(data).any(axis=1)]
            if data.size == 0:
                raise ValueError(f"Dataset column '{key}' has no finite rows for normalization.")
            processor = preprocessing.StandardScaler()
            processor.fit(data)
            processors[key] = processor
            if key != "action":
                processors[f"goal_{key}"] = processor
    return processors


def make_img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_NORMALIZE),
            transforms.Resize(size=img_size),
        ]
    )


def infer_swm_history_size(model: torch.nn.Module, fallback: int) -> int:
    if hasattr(model, "history_size"):
        return int(getattr(model, "history_size"))
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "num_frames"):
        return int(getattr(predictor, "num_frames"))
    return int(fallback)


def scan_model_for_attribute(module: torch.nn.Module, attribute_name: str) -> torch.nn.Module | None:
    if hasattr(module, attribute_name):
        module.eval()
        return module
    for child in module.children():
        result = scan_model_for_attribute(child, attribute_name)
        if result is not None:
            return result
    return None


def load_swm_model(policy_name: str, attribute_name: str, cache_dir: Path | None) -> torch.nn.Module:
    policy_path = Path(policy_name).expanduser()
    if policy_path.is_file():
        loaded = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, torch.nn.Module):
            raise TypeError(f"Checkpoint {policy_path} did not contain a torch module.")
        model = scan_model_for_attribute(loaded, attribute_name)
        if model is None:
            raise RuntimeError(f"No module with '{attribute_name}' found in {policy_path}.")
        return model

    if attribute_name == "get_cost":
        return swm.policy.AutoCostModel(policy_name, cache_dir=cache_dir)
    if attribute_name == "get_action":
        return swm.policy.AutoActionableModel(policy_name, cache_dir=cache_dir)
    raise ValueError(f"Unsupported stable-worldmodel attribute '{attribute_name}'.")


def make_swm_policy(args: argparse.Namespace, device: torch.device, action_space: spaces.Box) -> tuple[Any, int, dict[str, Any]]:
    policy_name = require_swm_policy(args)
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None
    process_keys = args.process_key if args.process_key is not None else ["action"]
    process = fit_processors(args.dataset_path, process_keys)
    transform = {
        "pixels": make_img_transform(args.swm_img_size),
        "goal": make_img_transform(args.swm_img_size),
    }

    if args.method == "swm_cost":
        model = load_swm_model(policy_name, "get_cost", cache_dir)
        if not hasattr(model, "get_cost"):
            raise TypeError(f"Policy '{policy_name}' does not expose get_cost(). Use --method swm_action instead.")
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        if hasattr(model, "interpolate_pos_encoding"):
            model.interpolate_pos_encoding = True
        solver_cfg = OmegaConf.load(args.solver_config)
        solver_cfg.device = str(device)
        solver_cfg.seed = int(args.seed)
        solver = hydra.utils.instantiate(solver_cfg, model=model)
        plan_config = swm.PlanConfig(
            horizon=int(args.plan_horizon),
            receding_horizon=int(args.receding_horizon),
            action_block=int(args.action_block),
            warm_start=not bool(args.no_warm_start),
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process=process,
            transform=transform,
        )
        method_config: dict[str, Any] = {
            "policy": policy_name,
            "solver_config": str(args.solver_config),
            "plan_config": {
                "horizon": int(args.plan_horizon),
                "receding_horizon": int(args.receding_horizon),
                "action_block": int(args.action_block),
                "warm_start": not bool(args.no_warm_start),
            },
            "process_keys": process_keys,
        }
    else:
        model = load_swm_model(policy_name, "get_action", cache_dir)
        if not hasattr(model, "get_action"):
            raise TypeError(f"Policy '{policy_name}' does not expose get_action(). Use --method swm_cost instead.")
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        policy = swm.policy.FeedForwardPolicy(
            model=model,
            process=process,
            transform=transform,
        )
        method_config = {
            "policy": policy_name,
            "process_keys": process_keys,
        }

    history_size = int(args.swm_history_size) if args.swm_history_size is not None else infer_swm_history_size(model, DEFAULT_SWM_HISTORY_SIZE)
    if history_size < 1:
        raise ValueError("--swm-history-size must be positive.")
    policy.set_env(SingleVectorEnvAdapter(action_space))
    method_config["history_size"] = history_size
    method_config["swm_img_size"] = int(args.swm_img_size)
    return policy, history_size, method_config


def load_ilqr_assets(args: argparse.Namespace, device: torch.device) -> tuple[ILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    config = ilqr_base.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else ilqr_base.latest_object_checkpoint(model_dir).resolve()
    )
    model = ilqr_base.load_model(checkpoint_path, device)
    history_size = int(config.get("history_size", 1))
    if history_size not in (1, 2):
        raise ValueError(f"Expected ILQR model history_size in {{1, 2}}, got {history_size}.")
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
                "LPB bank Markov-state dimension does not match the Reacher iLQR model: "
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
            "Pass --stats-dataset-path explicitly, or omit it to use --dataset-path."
        )
    train_stats_dataset = ilqr_base.LeWMReacherDataset(
        stats_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    policy = ILQRPolicyAdapter(
        model=model,
        config=config,
        action_mean=train_stats_dataset.action_mean.astype(np.float32),
        action_std=train_stats_dataset.action_std.astype(np.float32),
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
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "stats_dataset_path": str(stats_dataset_path),
        "img_size": img_size,
        "history_size": history_size,
        "ilqr_state_mode": policy.state_mode,
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
    return policy, model, method_config, train_stats_dataset.pixel_mean, train_stats_dataset.pixel_std, checkpoint_path


def current_qpos_qvel(env: ilqr_base.DmControlGymEnv, qpos_dim: int, qvel_dim: int) -> tuple[np.ndarray, np.ndarray]:
    physics = env._env.physics
    return (
        np.asarray(physics.data.qpos[:qpos_dim], dtype=np.float32).copy(),
        np.asarray(physics.data.qvel[:qvel_dim], dtype=np.float32).copy(),
    )


def save_case_summary(case_dir: Path, summary: dict[str, Any]) -> None:
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_case(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    device: torch.device,
    ilqr_assets: tuple[ILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path] | None,
    swm_policy: Any | None,
    swm_history_size: int | None,
) -> dict[str, Any]:
    episode = ilqr_base.load_dataset_episode(args.dataset_path, case.episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    start_frame = pixels_np[case.start_step]
    goal_frame = pixels_np[case.goal_step]
    goal_obs = obs_np[case.goal_step].astype(np.float32)
    start_obs_dataset = obs_np[case.start_step].astype(np.float32)
    goal_qpos = qpos_np[case.goal_step].astype(np.float32)
    goal_qvel = qvel_np[case.goal_step].astype(np.float32)

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ilqr_base.save_rgb_image(case_dir / "start_image.png", start_frame)
    ilqr_base.save_rgb_image(case_dir / "goal_image.png", goal_frame)

    env = ilqr_base.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    current_frame = ilqr_base.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        height=height,
        width=width,
    )
    current_obs = flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    qpos, qvel = current_qpos_qvel(env, qpos_np.shape[1], qvel_np.shape[1])

    rollout_frames = [current_frame.copy()]
    obs_distances = [float(np.linalg.norm(current_obs - goal_obs))]
    qpos_distances = [wrapped_qpos_distance(qpos, goal_qpos)]
    raw_qpos_distances = [float(np.linalg.norm(qpos - goal_qpos))]
    qvel_distances = [float(np.linalg.norm(qvel - goal_qvel))]
    step_records: list[dict[str, float]] = []
    stop_reason = "eval_budget"
    success = qpos_distances[-1] <= float(args.qpos_success_threshold)

    if args.method == "ilqr":
        assert ilqr_assets is not None
        ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
        img_size = int(ilqr_config["img_size"])
        pixels = ilqr_base.preprocess_pixels(
            np.stack([start_frame, goal_frame], axis=0),
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        start_emb, goal_emb = ilqr_base.encode_frames(
            ilqr_model,
            pixels,
            device=device,
            frame_batch_size=args.frame_batch_size,
        )
        ilqr_policy.reset(start_embedding=start_emb, goal_embedding=goal_emb)
        episode_history = None
    else:
        assert swm_policy is not None and swm_history_size is not None
        episode_history = EpisodeHistory(
            history_size=swm_history_size,
            action_dim=env.action_space.shape[0],
            case_id=case_idx,
            goal_frame=goal_frame,
            goal_obs=goal_obs,
            goal_qpos=goal_qpos,
            goal_qvel=goal_qvel,
            goal_step=case.goal_step,
        )
        episode_history.reset(
            frame=current_frame,
            observation=current_obs,
            qpos=qpos,
            qvel=qvel,
        )

    if success:
        stop_reason = "goal_reached"

    for step in range(int(args.eval_budget)):
        if success:
            break

        if args.method == "ilqr":
            assert ilqr_assets is not None
            ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
            img_size = int(ilqr_config["img_size"])
            current_emb = ilqr_base.encode_single_frame(
                ilqr_model,
                current_frame,
                device=device,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            action_raw, record = ilqr_policy.get_action(current_emb)
        else:
            assert swm_policy is not None and episode_history is not None
            action_batch = swm_policy.get_action(episode_history.info())
            action_raw = np.asarray(action_batch, dtype=np.float32).reshape(-1, env.action_space.shape[0])[0]
            record = {}

        obs, _, terminated, truncated, _ = env.step(action_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
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

        if args.method != "ilqr":
            assert episode_history is not None
            episode_history.append(
                frame=current_frame,
                action=action_raw,
                observation=current_obs,
                qpos=qpos,
                qvel=qvel,
                step_idx=step + 1,
            )

        if success or terminated or truncated:
            break

    video_path = None
    if not args.no_videos:
        video_path = str(ilqr_base.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
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
        "success_threshold": float(args.success_threshold),
        "qpos_success_threshold": float(args.qpos_success_threshold),
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
        "stop_reason": stop_reason,
        "episode_seed": episode_seed,
        "start_observation_dataset": start_obs_dataset.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "final_qpos": qpos.tolist(),
        "goal_qvel": goal_qvel.tolist(),
        "final_qvel": qvel.tolist(),
        "goal_observation": goal_obs.tolist(),
        "final_observation": current_obs.tolist(),
        "video_path": video_path,
        "step_records": step_records,
    }
    save_case_summary(case_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.success_threshold <= 0:
        raise ValueError("--success-threshold must be positive.")
    if args.qpos_success_threshold <= 0:
        raise ValueError("--qpos-success-threshold must be positive.")

    device = ilqr_base.require_device(args.device)
    run_name = f"{int(time.time())}_{args.method}_seed_{args.seed}"
    out_root = args.out_dir.expanduser().resolve() / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    ep_len = load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)

    ilqr_assets = None
    swm_policy = None
    swm_history_size = None
    method_config: dict[str, Any]
    if args.method == "ilqr":
        ilqr_assets = load_ilqr_assets(args, device)
        method_config = ilqr_assets[2]
    else:
        probe_episode = ilqr_base.load_dataset_episode(args.dataset_path, cases[0].episode_idx)
        env_for_space = ilqr_base.make_render_env(
            seed=int(probe_episode["episode_seed"]),
            time_limit=float(probe_episode["time_limit"]),
            width=int(probe_episode["width"]),
            height=int(probe_episode["height"]),
            physics_freq_hz=float(probe_episode["physics_freq_hz"]),
        )
        swm_policy, swm_history_size, method_config = make_swm_policy(args, device, env_for_space.action_space)
        env_for_space.close()

    case_results = []
    for case_idx, case in enumerate(tqdm(cases, desc="Hard Reacher eval")):
        case_results.append(
            run_case(
                args=args,
                case=case,
                case_idx=case_idx,
                out_root=out_root,
                device=device,
                ilqr_assets=ilqr_assets,
                swm_policy=swm_policy,
                swm_history_size=swm_history_size,
            )
        )

    successes = np.asarray([case["success"] for case in case_results], dtype=bool)
    strict_observation_successes = np.asarray(
        [case["strict_observation_success"] for case in case_results],
        dtype=bool,
    )
    qpos_successes = np.asarray([case["qpos_success"] for case in case_results], dtype=bool)
    metrics = {
        "success_rate": float(np.mean(successes) * 100.0),
        "success_metric": "wrapped_qpos_l2",
        "primary_success_threshold": float(args.qpos_success_threshold),
        "strict_observation_success_rate": float(np.mean(strict_observation_successes) * 100.0),
        "qpos_success_rate": float(np.mean(qpos_successes) * 100.0),
        "episode_successes": successes.astype(int).tolist(),
        "episode_strict_observation_successes": strict_observation_successes.astype(int).tolist(),
        "episode_qpos_successes": qpos_successes.astype(int).tolist(),
        "method": args.method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "num_eval": len(case_results),
        "eval_budget": int(args.eval_budget),
        "success_threshold": float(args.success_threshold),
        "qpos_success_threshold": float(args.qpos_success_threshold),
        "goal_protocol": "start_step_0_to_final_episode_step",
        "cases": case_results,
    }
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"qpos_success_rate: {metrics['qpos_success_rate']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
