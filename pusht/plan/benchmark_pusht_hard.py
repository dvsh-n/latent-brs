#!/usr/bin/env python3
"""Hard PushT benchmark for MPPI and stable-worldmodel policies.

The protocol uses hard episode endpoints: reset to episode step 0, use the
final episode frame/state as the goal, and count success with the best
block-position distance reached anywhere in the rollout.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[2]
STABLE_WORLDMODEL_DIR = ROOT_DIR / "third_party" / "stable-worldmodel"
STABLE_PRETRAINING_DIR = ROOT_DIR / "third_party" / "stable-pretraining"
for path in (ROOT_DIR, STABLE_WORLDMODEL_DIR, STABLE_PRETRAINING_DIR):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

from pusht.plan import plan_ilqr_mpc as pusht_base
from pusht.plan import plan_mppi as mppi_base
from pusht.train.mlpdyn_train import required_markov_history


DEFAULT_DATASET_PATH = "pusht/data/train_data/pusht_diffusion_train_combined.h5"
DEFAULT_MODEL_DIR = mppi_base.DEFAULT_MODEL_DIR
DEFAULT_OUT_DIR = "pusht/plan/pusht_hard_eval"
DEFAULT_SOLVER_CONFIG = STABLE_WORLDMODEL_DIR / "scripts" / "plan" / "config" / "solver" / "cem.yaml"
DEFAULT_NUM_EVAL = 50
DEFAULT_SEED = 42
DEFAULT_EVAL_BUDGET = mppi_base.MAX_MPC_STEPS
DEFAULT_POSITION_SUCCESS_THRESHOLD = 20.0
DEFAULT_YAW_SUCCESS_THRESHOLD = pusht_base.DEFAULT_BLOCK_YAW_SUCCESS_THRESHOLD
DEFAULT_REQUIRE_YAW_SUCCESS = True
DEFAULT_SWM_HISTORY_SIZE = 1
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


class SingleVectorEnvAdapter:
    """Tiny vector-env facade expected by stable-worldmodel policies."""

    def __init__(self, action_dim: int) -> None:
        single_action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        )
        self.num_envs = 1
        self.single_action_space = single_action_space
        self.action_space = spaces.Box(
            low=single_action_space.low[None, ...],
            high=single_action_space.high[None, ...],
            dtype=single_action_space.dtype,
        )


def env_state_to_dataset_state(
    env_state: np.ndarray,
    *,
    state_format: str,
    goal_pose: np.ndarray | None,
    fallback_goal_state: np.ndarray,
) -> np.ndarray:
    env_state = np.asarray(env_state, dtype=np.float32).reshape(-1)
    fallback_goal_state = np.asarray(fallback_goal_state, dtype=np.float32).reshape(-1)
    if state_format == "env_state":
        out = np.zeros_like(fallback_goal_state, dtype=np.float32)
        n = min(out.size, env_state.size)
        out[:n] = env_state[:n]
        return out
    if state_format != "privileged_goal_state":
        raise ValueError(f"Unsupported dataset state format: {state_format}")
    if goal_pose is None:
        goal_pose = fallback_goal_state[4:7]
    theta = float(env_state[4])
    return np.asarray(
        [
            env_state[2],
            env_state[3],
            np.cos(theta),
            np.sin(theta),
            goal_pose[0],
            goal_pose[1],
            goal_pose[2],
        ],
        dtype=np.float32,
    )


def env_state_to_dataset_proprio(env_state: np.ndarray, fallback_proprio: np.ndarray) -> np.ndarray:
    env_state = np.asarray(env_state, dtype=np.float32).reshape(-1)
    fallback = np.asarray(fallback_proprio, dtype=np.float32).reshape(-1)
    out = np.zeros_like(fallback, dtype=np.float32)
    if out.size >= 2:
        out[:2] = env_state[:2]
    if out.size >= 4 and env_state.size >= 7:
        out[2:4] = env_state[5:7]
    return out


class PushTEpisodeHistory:
    def __init__(
        self,
        *,
        history_size: int,
        action_dim: int,
        case_id: int,
        goal_frame: np.ndarray,
        goal_state: np.ndarray,
        goal_proprio: np.ndarray,
        goal_block_pose: np.ndarray,
        goal_agent_pos: np.ndarray,
        goal_step: int,
    ) -> None:
        self.history_size = int(history_size)
        self.action_dim = int(action_dim)
        self.case_id = int(case_id)
        self.goal_frame = np.asarray(goal_frame, dtype=np.uint8)
        self.goal_state = np.asarray(goal_state, dtype=np.float32)
        self.goal_proprio = np.asarray(goal_proprio, dtype=np.float32)
        self.goal_block_pose = np.asarray(goal_block_pose, dtype=np.float32)
        self.goal_agent_pos = np.asarray(goal_agent_pos, dtype=np.float32)
        self.goal_step = int(goal_step)
        self.frames: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.actions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.states: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.proprios: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.block_poses: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.agent_positions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.step_idx: deque[int] = deque(maxlen=self.history_size)

    def reset(
        self,
        *,
        frame: np.ndarray,
        state: np.ndarray,
        proprio: np.ndarray,
        block_pose: np.ndarray,
        agent_pos: np.ndarray,
    ) -> None:
        zero_action = np.zeros((self.action_dim,), dtype=np.float32)
        for _ in range(self.history_size):
            self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
            self.actions.append(zero_action.copy())
            self.states.append(np.asarray(state, dtype=np.float32).copy())
            self.proprios.append(np.asarray(proprio, dtype=np.float32).copy())
            self.block_poses.append(np.asarray(block_pose, dtype=np.float32).copy())
            self.agent_positions.append(np.asarray(agent_pos, dtype=np.float32).copy())
            self.step_idx.append(0)

    def append(
        self,
        *,
        frame: np.ndarray,
        action: np.ndarray,
        state: np.ndarray,
        proprio: np.ndarray,
        block_pose: np.ndarray,
        agent_pos: np.ndarray,
        step_idx: int,
    ) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.proprios.append(np.asarray(proprio, dtype=np.float32).copy())
        self.block_poses.append(np.asarray(block_pose, dtype=np.float32).copy())
        self.agent_positions.append(np.asarray(agent_pos, dtype=np.float32).copy())
        self.step_idx.append(int(step_idx))

    def info(self) -> dict[str, np.ndarray]:
        frames = np.stack(list(self.frames), axis=0)
        actions = np.stack(list(self.actions), axis=0)
        states = np.stack(list(self.states), axis=0)
        proprios = np.stack(list(self.proprios), axis=0)
        block_poses = np.stack(list(self.block_poses), axis=0)
        agent_positions = np.stack(list(self.agent_positions), axis=0)
        step_idx = np.asarray(list(self.step_idx), dtype=np.int64)
        ids = np.full((self.history_size,), self.case_id, dtype=np.int64)
        goal_ids = np.asarray([self.case_id], dtype=np.int64)
        goal_step_idx = np.asarray([self.goal_step], dtype=np.int64)
        return {
            "pixels": frames[None, ...],
            "goal": self.goal_frame[None, None, ...],
            "action": actions[None, ...],
            "state": states[None, ...],
            "goal_state": self.goal_state[None, None, ...],
            "proprio": proprios[None, ...],
            "goal_proprio": self.goal_proprio[None, None, ...],
            "block_pose": block_poses[None, ...],
            "goal_block_pose": self.goal_block_pose[None, None, ...],
            "agent_pos": agent_positions[None, ...],
            "goal_agent_pos": self.goal_agent_pos[None, None, ...],
            "step_idx": step_idx[None, ...],
            "goal_step_idx": goal_step_idx[None, ...],
            "id": ids[None, ...],
            "goal_id": goal_ids[None, ...],
        }


class MPPIPolicyAdapter:
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
        mppi_samples: int,
        mppi_update_iters: int,
        mppi_reward_weight: float,
        mppi_noise_level: float,
        mppi_noise_decay: float,
        mppi_beta_filter: float,
        seed: int,
    ) -> None:
        self.model = model
        self.config = config
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device
        self.action_dim = int(config.get("action_dim", 2))
        self.markov_deriv = int(config.get("markov_deriv", 1))
        self.markov_state_dim = int(config.get("markov_state_dim", (self.markov_deriv + 1) * int(config.get("embed_dim", 48))))
        self.history_len = required_markov_history(self.markov_deriv)
        self.current_history: list[torch.Tensor] = []
        self.goal_state_torch: torch.Tensor | None = None
        self.seed = int(seed)

        self.jax, self.jnp, _ = mppi_base.initialize_jax("gpu" if device.type == "cuda" else "cpu")
        _probe = self.jnp.zeros((1,), dtype=self.jnp.float32)
        self.jax.block_until_ready(_probe)

        jax_dynamics = mppi_base.build_batched_jax_dynamics(model.predictor.net, device, self.markov_state_dim)
        rollout_fn, eval_fn = mppi_base.make_mppi_rollout_and_eval(
            jax_dynamics,
            q_stage=q_stage,
            q_terminal=q_terminal,
            r_control=r_control,
        )
        mppi_config = {
            "planning": {
                "action_dim": self.action_dim,
                "n_sample": int(mppi_samples),
                "horizon": int(horizon),
                "n_update_iter": int(mppi_update_iters),
                "use_last": True,
                "reject_bad": False,
                "mppi": {
                    "reward_weight": float(mppi_reward_weight),
                    "noise_level": float(mppi_noise_level),
                    "noise_decay": float(mppi_noise_decay),
                    "beta_filter": float(mppi_beta_filter),
                },
            }
        }
        action_lower_lim = self.jnp.full((self.action_dim,), mppi_base.CONTROL_MIN_NORM, dtype=self.jnp.float32)
        action_upper_lim = self.jnp.full((self.action_dim,), mppi_base.CONTROL_MAX_NORM, dtype=self.jnp.float32)
        planner_cls = mppi_base.load_mppi_planner_class()
        self.planner = planner_cls(
            config=mppi_config,
            model_rollout_fn=rollout_fn,
            evaluate_traj_fn=eval_fn,
            action_lower_lim=action_lower_lim,
            action_upper_lim=action_upper_lim,
        )
        self.prev_u = self.jnp.zeros((int(horizon), self.action_dim), dtype=self.jnp.float32)
        self.goal_state_jax = None
        self.jax_key = self.jax.random.PRNGKey(self.seed)

        def trajopt(key: Any, state_cur: Any, init_action_seq: Any, goal_state_jax: Any):
            return self.planner.trajectory_optimization(
                key,
                state_cur,
                init_action_seq,
                skip=False,
                goal_state=goal_state_jax,
            )

        self.jit_trajopt = self.jax.jit(trajopt)

    def reset(self, *, start_embedding: torch.Tensor, goal_state: torch.Tensor, case_seed: int) -> None:
        self.current_history = [start_embedding] * self.history_len
        self.goal_state_torch = goal_state
        self.goal_state_jax = self.jnp.asarray(goal_state.detach().cpu().numpy().astype(np.float32))
        self.prev_u = self.jnp.zeros_like(self.prev_u)
        self.jax_key = self.jax.random.PRNGKey(int(case_seed))

    def append_embedding(self, embedding: torch.Tensor) -> None:
        self.current_history.append(embedding)
        self.current_history = self.current_history[-self.history_len :]

    def current_state(self) -> torch.Tensor:
        state = pusht_base.make_markov_state(self.current_history, self.markov_deriv)
        if int(state.numel()) != self.markov_state_dim:
            raise ValueError(f"State dimension mismatch: config says {self.markov_state_dim}, built {state.numel()}.")
        return state

    def get_action(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if self.goal_state_jax is None:
            raise RuntimeError("MPPIPolicyAdapter.reset must be called before get_action.")
        current_state_np = self.current_state().detach().cpu().numpy().astype(np.float32)
        init_action_seq = mppi_base.shift_warmstart(self.prev_u)
        self.jax_key, subkey = self.jax.random.split(self.jax_key)
        t0 = time.perf_counter()
        plan = self.jit_trajopt(
            subkey,
            self.jnp.asarray(current_state_np, dtype=self.jnp.float32),
            init_action_seq,
            self.goal_state_jax,
        )
        self.jax.block_until_ready(plan["act_seq"])
        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        u_plan = np.asarray(plan["act_seq"], dtype=np.float32)
        self.prev_u = self.jnp.asarray(u_plan, dtype=self.jnp.float32)
        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = pusht_base.normalized_to_raw_action(u0_norm, self.action_mean, self.action_std)
        return u0_raw, u0_norm, {
            "solve_time_ms": float(solve_time_ms),
            "mppi_reward": float(plan["reward"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=("mppi", "swm_cost", "swm_action"), default="mppi")
    parser.add_argument("--policy", type=str, default=None, help="Stable-worldmodel object checkpoint or checkpoint directory.")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--action-stats-dataset-path",
        type=Path,
        default=None,
        help="Dataset used to compute action mean/std for MPPI. Defaults to the model config, then the benchmark dataset.",
    )
    parser.add_argument("--num-eval", type=int, default=DEFAULT_NUM_EVAL)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=mppi_base.DEVICE)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--video-fps", type=int, default=mppi_base.VIDEO_FPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the rollout at the first successful timestep.",
    )
    parser.add_argument(
        "--position-success-threshold",
        "--success-threshold",
        dest="position_success_threshold",
        type=float,
        default=DEFAULT_POSITION_SUCCESS_THRESHOLD,
        help="Primary success threshold on the minimum block xy position distance over the rollout, in pixels.",
    )
    parser.add_argument(
        "--yaw-success-threshold",
        type=float,
        default=DEFAULT_YAW_SUCCESS_THRESHOLD,
        help="Wrapped block yaw threshold in radians.",
    )
    parser.add_argument(
        "--require-yaw-success",
        action="store_true",
        default=DEFAULT_REQUIRE_YAW_SUCCESS,
        help="Require yaw error to pass --yaw-success-threshold in addition to position distance.",
    )
    parser.add_argument(
        "--ignore-yaw-success",
        dest="require_yaw_success",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Use position-only success and keep yaw as a diagnostic.",
    )

    parser.add_argument("--horizon", type=int, default=mppi_base.HORIZON)
    parser.add_argument("--q-terminal", type=float, default=mppi_base.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=mppi_base.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=mppi_base.R_CONTROL)
    parser.add_argument("--mppi-samples", type=int, default=mppi_base.MPPI_SAMPLES)
    parser.add_argument("--mppi-update-iters", type=int, default=mppi_base.MPPI_UPDATE_ITERS)
    parser.add_argument("--mppi-reward-weight", type=float, default=mppi_base.MPPI_REWARD_WEIGHT)
    parser.add_argument("--mppi-noise-level", type=float, default=mppi_base.MPPI_NOISE_LEVEL)
    parser.add_argument("--mppi-noise-decay", type=float, default=mppi_base.MPPI_NOISE_DECAY)
    parser.add_argument("--mppi-beta-filter", type=float, default=mppi_base.MPPI_BETA_FILTER)

    parser.add_argument("--solver-config", type=Path, default=DEFAULT_SOLVER_CONFIG)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--swm-history-size", type=int, default=None)
    parser.add_argument("--swm-img-size", type=int, default=224)
    parser.add_argument("--process-key", action="append", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


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
            if key == "pixels":
                raise ValueError("Do not fit a StandardScaler on pixels; use image transforms instead.")
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


def make_swm_policy(args: argparse.Namespace, device: torch.device, action_dim: int) -> tuple[Any, int, dict[str, Any]]:
    if not args.policy:
        raise ValueError(f"--policy is required for --method {args.method}.")
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None
    process_keys = args.process_key if args.process_key is not None else ["action"]
    process = fit_processors(args.dataset_path, process_keys)
    transform = {
        "pixels": make_img_transform(args.swm_img_size),
        "goal": make_img_transform(args.swm_img_size),
    }
    if args.method == "swm_cost":
        model = load_swm_model(str(args.policy), "get_cost", cache_dir)
        if not hasattr(model, "get_cost"):
            raise TypeError(f"Policy '{args.policy}' does not expose get_cost(). Use --method swm_action instead.")
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
        policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_config, process=process, transform=transform)
        method_config: dict[str, Any] = {
            "policy": str(args.policy),
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
        model = load_swm_model(str(args.policy), "get_action", cache_dir)
        if not hasattr(model, "get_action"):
            raise TypeError(f"Policy '{args.policy}' does not expose get_action(). Use --method swm_cost instead.")
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        policy = swm.policy.FeedForwardPolicy(model=model, process=process, transform=transform)
        method_config = {"policy": str(args.policy), "process_keys": process_keys}

    history_size = int(args.swm_history_size) if args.swm_history_size is not None else infer_swm_history_size(model, DEFAULT_SWM_HISTORY_SIZE)
    if history_size < 1:
        raise ValueError("--swm-history-size must be positive.")
    policy.set_env(SingleVectorEnvAdapter(action_dim))
    method_config["history_size"] = history_size
    method_config["swm_img_size"] = int(args.swm_img_size)
    return policy, history_size, method_config


def load_mppi_policy(args: argparse.Namespace, device: torch.device) -> tuple[MPPIPolicyAdapter, dict[str, Any]]:
    model_dir = pusht_base.resolve_model_dir(args)
    config = pusht_base.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else pusht_base.latest_object_checkpoint(model_dir).resolve()
    )
    model = pusht_base.load_model(checkpoint_path, device)
    frameskip = int(config.get("frameskip", 1))
    if frameskip != 1:
        raise ValueError(f"This PushT MPPI benchmark supports frameskip=1 only, got frameskip={frameskip}.")
    action_dim = int(config.get("action_dim", 2))
    action_stats_dataset_path = (
        args.action_stats_dataset_path.expanduser().resolve() if args.action_stats_dataset_path is not None else None
    )
    train_dataset_paths = mppi_base.resolve_action_stats_dataset_paths(
        config,
        args.dataset_path,
        action_stats_dataset_path=action_stats_dataset_path,
    )
    action_mean, action_std = pusht_base.load_action_stats(train_dataset_paths, action_dim)
    policy = MPPIPolicyAdapter(
        model=model,
        config=config,
        action_mean=action_mean,
        action_std=action_std,
        device=device,
        horizon=int(args.horizon),
        q_terminal=float(args.q_terminal),
        q_stage=float(args.q_stage),
        r_control=float(args.r_control),
        mppi_samples=int(args.mppi_samples),
        mppi_update_iters=int(args.mppi_update_iters),
        mppi_reward_weight=float(args.mppi_reward_weight),
        mppi_noise_level=float(args.mppi_noise_level),
        mppi_noise_decay=float(args.mppi_noise_decay),
        mppi_beta_filter=float(args.mppi_beta_filter),
        seed=int(args.seed),
    )
    method_config = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(model_dir / "config.json"),
        "train_dataset_paths": [str(path) for path in train_dataset_paths],
        "horizon": int(args.horizon),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "mppi_samples": int(args.mppi_samples),
        "mppi_update_iters": int(args.mppi_update_iters),
        "mppi_reward_weight": float(args.mppi_reward_weight),
        "mppi_noise_level": float(args.mppi_noise_level),
        "mppi_noise_decay": float(args.mppi_noise_decay),
        "mppi_beta_filter": float(args.mppi_beta_filter),
        "action_dim": action_dim,
        "action_stats_dataset_path": str(action_stats_dataset_path) if action_stats_dataset_path is not None else None,
        "markov_deriv": int(config.get("markov_deriv", 1)),
        "markov_state_dim": policy.markov_state_dim,
    }
    return policy, method_config


def save_case_summary(case_dir: Path, summary: dict[str, Any]) -> None:
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def reset_swm_policy_episode_state(swm_policy: Any | None) -> None:
    if swm_policy is None:
        return
    action_buffer = getattr(swm_policy, "_action_buffer", None)
    if action_buffer is not None:
        action_buffer.clear()
    if hasattr(swm_policy, "action_buffer"):
        swm_policy.action_buffer.clear()
    if hasattr(swm_policy, "_next_init"):
        swm_policy._next_init = None


def run_case(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    device: torch.device,
    mppi_policy: MPPIPolicyAdapter | None,
    swm_policy: Any | None,
    swm_history_size: int | None,
) -> dict[str, Any]:
    episode = pusht_base.load_dataset_episode(args.dataset_path, case.episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    state_np = np.asarray(episode["state"], dtype=np.float32)
    proprio_np = np.asarray(episode["proprio"], dtype=np.float32)
    height = int(episode["height"])
    width = int(episode["width"])
    state_format = pusht_base.infer_dataset_state_format(state_np[case.start_step])
    env_state_np = np.stack(
        [
            pusht_base.dataset_row_to_env_state(state_row, proprio_row, state_format)
            for state_row, proprio_row in zip(state_np, proprio_np)
        ],
        axis=0,
    )
    goal_pose = pusht_base.dataset_row_to_goal_pose(state_np[case.goal_step], state_format)
    goal_block = pusht_base.dataset_row_to_block_pose(state_np[case.goal_step], env_state_np[case.goal_step], state_format)
    goal_agent = env_state_np[case.goal_step, :2].astype(np.float32)

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    pusht_base.save_rgb_image(case_dir / "start_image.png", pixels_np[case.start_step])
    pusht_base.save_rgb_image(case_dir / "goal_image.png", pixels_np[case.goal_step])

    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    true_latents = None
    goal_state = None
    if args.method == "mppi":
        assert mppi_policy is not None
        img_size = int(mppi_policy.config.get("img_size", 224))
        pixels = pusht_base.preprocess_pixels(
            pixels_np,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        true_latents = pusht_base.encode_frames(
            mppi_policy.model,
            pixels,
            device=device,
            frame_batch_size=args.frame_batch_size,
        )
        goal_history = [emb for emb in true_latents[-mppi_policy.history_len :]]
        goal_state = pusht_base.make_markov_state(goal_history, mppi_policy.markov_deriv)

    plan_env = pusht_base.make_planning_env(width=width, height=height)
    viz_env = None if args.no_videos else pusht_base.make_visualization_env(width=width, height=height)

    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    step_records: list[dict[str, Any]] = []
    solve_times_ms: list[float] = []
    mppi_rewards: list[float] = []
    latent_goal_distances: list[float] = []
    block_goal_distances: list[float] = []
    block_position_distances: list[float] = []
    block_yaw_errors: list[float] = []
    rollout_frames: list[np.ndarray] = []
    num_action_clips = 0
    stop_reason = "eval_budget"
    video_path: str | None = None

    def success_at(position_distance: float, yaw_error: float) -> bool:
        position_ok = position_distance <= float(args.position_success_threshold)
        if not bool(args.require_yaw_success):
            return bool(position_ok)
        return bool(position_ok and yaw_error <= float(args.yaw_success_threshold))

    try:
        pusht_base.set_goal_pose(plan_env, goal_pose)
        if viz_env is not None:
            pusht_base.set_goal_pose(viz_env, goal_pose)
        action_low, action_high = pusht_base.pusht_agent_action_bounds()
        hidden_start = pusht_base.reset_env_to_state(plan_env, env_state_np[case.start_step])
        if viz_env is not None:
            visible_start = pusht_base.reset_env_to_state(viz_env.unwrapped, env_state_np[case.start_step])
            rollout_frames.append(visible_start.copy())
        current_hidden_frame = hidden_start
        current_env_state = pusht_base.extract_full_state(plan_env)
        current_block = pusht_base.current_block_pose(plan_env)
        current_agent = pusht_base.current_agent_pos(plan_env)
        position_distance, yaw_error, mixed_pose_distance = pusht_base.block_pose_components(current_block, goal_block)
        block_position_distances.append(position_distance)
        block_yaw_errors.append(yaw_error)
        block_goal_distances.append(mixed_pose_distance)

        swm_history = None
        if args.method == "mppi":
            assert mppi_policy is not None and goal_state is not None
            # Match plan_mppi.py exactly: initialize the Markov history from the
            # dataset start-frame latent, not from a freshly rendered reset frame.
            assert true_latents is not None
            current_emb = true_latents[case.start_step]
            mppi_policy.reset(
                start_embedding=current_emb,
                goal_state=goal_state,
                case_seed=int(args.seed) + case_idx,
            )
            latent_goal_distances.append(float(torch.linalg.vector_norm(mppi_policy.current_state() - goal_state).item()))
        else:
            assert swm_policy is not None and swm_history_size is not None
            reset_swm_policy_episode_state(swm_policy)
            swm_history = PushTEpisodeHistory(
                history_size=swm_history_size,
                action_dim=2,
                case_id=case_idx,
                goal_frame=pixels_np[case.goal_step],
                goal_state=state_np[case.goal_step],
                goal_proprio=proprio_np[case.goal_step],
                goal_block_pose=goal_block,
                goal_agent_pos=goal_agent,
                goal_step=case.goal_step,
            )
            swm_history.reset(
                frame=current_hidden_frame,
                state=state_np[case.start_step],
                proprio=proprio_np[case.start_step],
                block_pose=current_block,
                agent_pos=current_agent,
            )

        success = success_at(block_position_distances[-1], block_yaw_errors[-1])
        if success:
            stop_reason = "initial_goal_reached"

        for step in range(int(args.eval_budget)):
            if success and bool(args.stop_on_success):
                break
            if args.method == "mppi":
                assert mppi_policy is not None
                action_raw, action_norm, diagnostics = mppi_policy.get_action()
                executed_actions_norm.append(action_norm.copy())
                solve_times_ms.append(float(diagnostics["solve_time_ms"]))
                mppi_rewards.append(float(diagnostics["mppi_reward"]))
            else:
                assert swm_policy is not None and swm_history is not None
                with torch.inference_mode():
                    action_raw = np.asarray(swm_policy.get_action(swm_history.info()), dtype=np.float32).reshape(-1)[:2]
                diagnostics = {}

            unclipped_action_env = pusht_base.raw_to_env_action(action_raw, pusht_base.current_agent_pos(plan_env))
            action_env = pusht_base.raw_to_env_action(
                action_raw,
                pusht_base.current_agent_pos(plan_env),
                action_low=action_low,
                action_high=action_high,
            )
            if not np.allclose(action_env, unclipped_action_env):
                num_action_clips += 1

            executed_actions_raw.append(action_raw.copy())
            executed_actions_env.append(action_env.copy())
            _, _, terminated, truncated, _ = plan_env.step(action_env)
            current_hidden_frame = np.asarray(plan_env._render(visualize=False), dtype=np.uint8)
            current_env_state = pusht_base.extract_full_state(plan_env)
            current_block = pusht_base.current_block_pose(plan_env)
            current_agent = pusht_base.current_agent_pos(plan_env)
            if viz_env is not None:
                synced_visible_frame = pusht_base.reset_env_to_state(viz_env.unwrapped, current_env_state)
                rollout_frames.append(synced_visible_frame.copy())

            position_distance, yaw_error, mixed_pose_distance = pusht_base.block_pose_components(current_block, goal_block)
            block_position_distances.append(position_distance)
            block_yaw_errors.append(yaw_error)
            block_goal_distances.append(mixed_pose_distance)

            if args.method == "mppi":
                assert mppi_policy is not None and goal_state is not None
                img_size = int(mppi_policy.config.get("img_size", 224))
                next_emb = pusht_base.encode_single_frame(
                    mppi_policy.model,
                    current_hidden_frame,
                    device=device,
                    img_size=img_size,
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                )
                mppi_policy.append_embedding(next_emb)
                latent_goal_distances.append(float(torch.linalg.vector_norm(mppi_policy.current_state() - goal_state).item()))
            else:
                assert swm_history is not None
                current_state = env_state_to_dataset_state(
                    current_env_state,
                    state_format=state_format,
                    goal_pose=goal_pose,
                    fallback_goal_state=state_np[case.goal_step],
                )
                current_proprio = env_state_to_dataset_proprio(current_env_state, proprio_np[case.start_step])
                swm_history.append(
                    frame=current_hidden_frame,
                    action=action_raw,
                    state=current_state,
                    proprio=current_proprio,
                    block_pose=current_block,
                    agent_pos=current_agent,
                    step_idx=step + 1,
                )

            success = success_at(position_distance, yaw_error)
            record = {
                "step": step,
                "block_goal_distance": float(mixed_pose_distance),
                "block_position_distance": float(position_distance),
                "block_yaw_error": float(yaw_error),
                "action_raw": action_raw.tolist(),
                "action_env": action_env.tolist(),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **diagnostics,
            }
            step_records.append(record)
            if success and bool(args.stop_on_success):
                stop_reason = "goal_reached"
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break

        if not args.no_videos:
            video_path = str(pusht_base.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
    finally:
        plan_env.close()
        if viz_env is not None:
            viz_env.close()
        del rollout_frames
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    min_position_distance = float(np.min(block_position_distances))
    min_yaw_error = float(np.min(block_yaw_errors))
    if bool(args.require_yaw_success):
        block_success = bool(
            np.any(
                (np.asarray(block_position_distances) <= float(args.position_success_threshold))
                & (np.asarray(block_yaw_errors) <= float(args.yaw_success_threshold))
            )
        )
        success_metric = "min_block_position_l2_and_yaw"
    else:
        block_success = bool(min_position_distance <= float(args.position_success_threshold))
        success_metric = "min_block_position_l2"
    if block_success and stop_reason == "eval_budget":
        stop_reason = "eval_budget_success_reached"
    summary = {
        **asdict(case),
        "success": bool(block_success),
        "success_metric": success_metric,
        "position_success_threshold": float(args.position_success_threshold),
        "yaw_success_threshold": float(args.yaw_success_threshold),
        "require_yaw_success": bool(args.require_yaw_success),
        "stop_on_success": bool(args.stop_on_success),
        "legacy_mixed_pose_success_threshold": float(args.position_success_threshold),
        "initial_block_goal_distance": float(block_goal_distances[0]),
        "final_block_goal_distance": float(block_goal_distances[-1]),
        "min_block_goal_distance": float(np.min(block_goal_distances)),
        "initial_block_position_distance": float(block_position_distances[0]),
        "final_block_position_distance": float(block_position_distances[-1]),
        "min_block_position_distance": min_position_distance,
        "initial_block_yaw_error": float(block_yaw_errors[0]),
        "final_block_yaw_error": float(block_yaw_errors[-1]),
        "min_block_yaw_error": min_yaw_error,
        "initial_latent_goal_distance": float(latent_goal_distances[0]) if latent_goal_distances else None,
        "final_latent_goal_distance": float(latent_goal_distances[-1]) if latent_goal_distances else None,
        "min_latent_goal_distance": float(np.min(latent_goal_distances)) if latent_goal_distances else None,
        "steps_executed": len(step_records),
        "stop_reason": stop_reason,
        "dataset_state_format": state_format,
        "start_agent_pos": env_state_np[case.start_step, :2].astype(np.float32).tolist(),
        "goal_agent_pos": goal_agent.tolist(),
        "goal_block_pose": goal_block.tolist(),
        "final_block_pose": current_block.tolist(),
        "goal_pose": goal_pose.tolist() if goal_pose is not None else None,
        "num_action_clips": int(num_action_clips),
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "executed_actions_env": [action.tolist() for action in executed_actions_env],
        "solve_times_ms": solve_times_ms,
        "mppi_rewards": mppi_rewards,
        "block_goal_distances": block_goal_distances,
        "block_position_distances": block_position_distances,
        "block_yaw_errors": block_yaw_errors,
        "latent_goal_distances": latent_goal_distances,
        "video_path": video_path,
        "step_records": step_records,
    }
    save_case_summary(case_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.position_success_threshold <= 0:
        raise ValueError("--position-success-threshold must be positive.")
    if args.yaw_success_threshold <= 0:
        raise ValueError("--yaw-success-threshold must be positive.")

    device = pusht_base.require_device(args.device)
    ep_len = load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)

    mppi_policy = None
    swm_policy = None
    swm_history_size = None
    if args.method == "mppi":
        mppi_policy, method_config = load_mppi_policy(args, device)
    else:
        swm_policy, swm_history_size, method_config = make_swm_policy(args, device, action_dim=2)

    run_name = f"{int(time.time())}_{args.method}_seed_{args.seed}"
    out_root = args.out_dir / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    case_results = []
    for case_idx, case in enumerate(tqdm(cases, desc="Hard PushT eval")):
        case_results.append(
            run_case(
                args=args,
                case=case,
                case_idx=case_idx,
                out_root=out_root,
                device=device,
                mppi_policy=mppi_policy,
                swm_policy=swm_policy,
                swm_history_size=swm_history_size,
            )
        )

    successes = np.asarray([case["success"] for case in case_results], dtype=bool)
    success_metric = "min_block_position_l2_and_yaw" if bool(args.require_yaw_success) else "min_block_position_l2"
    metrics = {
        "success_rate": float(np.mean(successes) * 100.0),
        "episode_successes": successes.astype(int).tolist(),
        "success_metric": success_metric,
        "position_success_threshold": float(args.position_success_threshold),
        "yaw_success_threshold": float(args.yaw_success_threshold),
        "require_yaw_success": bool(args.require_yaw_success),
        "legacy_mixed_pose_success_threshold": float(args.position_success_threshold),
        "method": args.method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "num_eval": int(args.num_eval),
        "eval_budget": int(args.eval_budget),
        "start_step": 0,
        "goal_protocol": "episode_final_step",
        "episode_indices": [int(case["episode_idx"]) for case in case_results],
        "goal_steps": [int(case["goal_step"]) for case in case_results],
        "ep_lens": [int(case["ep_len"]) for case in case_results],
        "initial_block_goal_distances": [case["initial_block_goal_distance"] for case in case_results],
        "final_block_goal_distances": [case["final_block_goal_distance"] for case in case_results],
        "min_block_goal_distances": [case["min_block_goal_distance"] for case in case_results],
        "initial_block_position_distances": [case["initial_block_position_distance"] for case in case_results],
        "final_block_position_distances": [case["final_block_position_distance"] for case in case_results],
        "min_block_position_distances": [case["min_block_position_distance"] for case in case_results],
        "initial_block_yaw_errors": [case["initial_block_yaw_error"] for case in case_results],
        "final_block_yaw_errors": [case["final_block_yaw_error"] for case in case_results],
        "min_block_yaw_errors": [case["min_block_yaw_error"] for case in case_results],
        "initial_latent_goal_distances": [case["initial_latent_goal_distance"] for case in case_results],
        "final_latent_goal_distances": [case["final_latent_goal_distance"] for case in case_results],
        "min_latent_goal_distances": [case["min_latent_goal_distance"] for case in case_results],
        "steps_executed": [case["steps_executed"] for case in case_results],
        "stop_reasons": [case["stop_reason"] for case in case_results],
        "cases": case_results,
    }
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
