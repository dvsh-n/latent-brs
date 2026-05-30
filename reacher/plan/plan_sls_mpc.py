#!/usr/bin/env python3
"""Plan in Reacher pixel space using conformal SLS MPC over a Markov-state world model."""

import os
import sys
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm
import pyrallis

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

# gpu_sls core modules
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints

from error_model import MGNLLPredictor
from reacher.plan.plan_ilqr_mpc import (
    latest_object_checkpoint as latest_reacher_object_checkpoint,
    load_config as load_reacher_config,
    load_dataset_episode,
    make_render_env,
    require_device,
    reset_env_to_state as reset_reacher_env_to_state,
    resolve_start_goal_qpos,
    save_rollout_video,
)

DEFAULT_DATASET_PATH = Path("reacher/data/test_data_noisy.h5")
DEFAULT_ACTION_STATS_CANDIDATES = (
    Path("reacher/data/train_data_noisy.h5"),
    DEFAULT_DATASET_PATH,
)
DEFAULT_START_QPOS = np.array([0.1, -2.0], dtype=np.float32)
DEFAULT_GOAL_QPOS = np.array([2.42, -0.95], dtype=np.float32)

@dataclass
class PlanSLSReacherConfig:
    """Configuration for conformal SLS MPC Reacher planning."""
    q_learned: float = field(default=0.0, metadata={"help": "Conformal quantile for the disturbance bound."})
    model_dir: Path = field(default=Path("reacher/models/mlpdyn_embd_5"))
    error_model_ckpt: Path = field(default=Path("reacher/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("reacher/eval/fixed_error_covariance.pt"))
    enable_obstacle: bool = field(default=True)
    obstacle_model_path: Path = field(default=Path("reacher/plan/obs_net"))
    obstacle_margin: float = field(default=0.0)
    obstacle_penalty_weight: float = field(default=1000.0)
    use_latent_ellipsoid_constraint: bool = field(default=False)
    latent_ellipsoid_path: Path = field(default=Path("reacher/eval/latent_ellipsoid/latent_ellipsoid.pt"))
    latent_ellipsoid_margin: float = field(default=0.0)
    action_stats_dataset_path: Optional[Path] = field(default=None)
    dataset_path: Path = field(default=DEFAULT_DATASET_PATH)
    out_dir: Path = field(default=Path("reacher/plan/sls_mpc_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=20)
    max_mpc_steps: int = field(default=150)
    video_fps: int = field(default=30)
    episode_idx: Optional[int] = field(default=3781)
    seed: int = field(default=42)
    q_stage: float = field(default=0.005)
    q_terminal: float = field(default=5.0)
    r_control: float = field(default=0.01)
    mppi_horizon: Optional[int] = field(default=None)
    mppi_stage_weight: float = field(default=0.005)
    mppi_terminal_weight: float = field(default=5.0)
    mppi_state_box_penalty: float = field(default=0.0)

# --- JAX / PyTorch Bridge Engines ---

class JAXObstacleMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    feature_mean: jax.Array
    feature_std: jax.Array
    threshold: jax.Array
    input_dim: int

    def __call__(self, state):
        z = state[: self.input_dim]
        x = (z - self.feature_mean) / self.feature_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jax.nn.gelu(x)
        return self.linear_layers[-1](x).squeeze(-1)

def build_jax_obstacle_from_artifact(artifact_path: Path, key: jax.Array) -> JAXObstacleMLP:
    artifact_path = artifact_path.expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Obstacle model artifact not found: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    state_dict = artifact["state_dict"]
    input_dim = int(artifact["input_dim"])
    hidden_dim = int(artifact["hidden_dim"])
    depth = int(artifact["depth"])
    dropout = float(artifact["dropout"])

    linear_layers = []
    layer_norm_scales = []
    layer_norm_biases = []
    keys = jax.random.split(key, depth)

    module_idx = 0
    current_dim = input_dim
    for i in range(depth - 1):
        linear = eqx.nn.Linear(current_dim, hidden_dim, key=keys[i])
        linear = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            linear,
            (
                jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
                jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
            ),
        )
        linear_layers.append(linear)

        ln_idx = module_idx + 1
        layer_norm_scales.append(jnp.asarray(state_dict[f"net.{ln_idx}.weight"].detach().cpu().numpy()))
        layer_norm_biases.append(jnp.asarray(state_dict[f"net.{ln_idx}.bias"].detach().cpu().numpy()))

        module_idx += 4 if dropout > 0.0 else 3
        current_dim = hidden_dim

    output_linear = eqx.nn.Linear(current_dim, 1, key=keys[-1])
    output_linear = eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        output_linear,
        (
            jnp.asarray(state_dict[f"net.{module_idx}.weight"].detach().cpu().numpy()),
            jnp.asarray(state_dict[f"net.{module_idx}.bias"].detach().cpu().numpy()),
        ),
    )
    linear_layers.append(output_linear)

    return JAXObstacleMLP(
        linear_layers=linear_layers,
        layer_norm_scales=layer_norm_scales,
        layer_norm_biases=layer_norm_biases,
        feature_mean=jnp.asarray(artifact["feature_mean"], dtype=jnp.float64),
        feature_std=jnp.maximum(jnp.asarray(artifact["feature_std"], dtype=jnp.float64), 1e-6),
        threshold=jnp.asarray(float(artifact["conformal_safe_score_threshold"]), dtype=jnp.float64),
        input_dim=input_dim,
    )

def build_jax_dynamics(torch_dynamics_net: torch.nn.Module, device: torch.device, state_dim: int, action_dim: int):
    def _fwd_fn(x_np, u_np):
        with torch.no_grad():
            x_t = torch.from_numpy(np.array(x_np)).float().to(device)
            u_t = torch.from_numpy(np.array(u_np)).float().to(device)
            inp = torch.cat((x_t, u_t), dim=-1)
            out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0) if inp.ndim == 1 else torch_dynamics_net(inp)
            return np.asarray(out.cpu().numpy(), dtype=np.float64)

    def _vjp_fn(x_np, u_np, g_np):
        x_t = torch.from_numpy(np.array(x_np)).float().to(device).requires_grad_(True)
        u_t = torch.from_numpy(np.array(u_np)).float().to(device).requires_grad_(True)
        inp = torch.cat((x_t, u_t), dim=-1)
        out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0) if inp.ndim == 1 else torch_dynamics_net(inp)
        g_t = torch.from_numpy(np.asarray(g_np)).float().to(device)
        out.backward(g_t)
        return np.asarray(x_t.grad.cpu().numpy(), dtype=np.float64), np.asarray(u_t.grad.cpu().numpy(), dtype=np.float64)

    @jax.custom_vjp
    def jax_dynamics(x, u, t, parameter):
        result_shape = jax.ShapeDtypeStruct((state_dim,), jnp.float64)
        return jax.pure_callback(_fwd_fn, result_shape, x, u, vmap_method="sequential")

    def jax_dynamics_fwd(x, u, t, parameter):
        y = jax_dynamics(x, u, t, parameter)
        return y, (x, u)

    def jax_dynamics_bwd(res, g):
        x, u = res
        jac_x_shape = jax.ShapeDtypeStruct((state_dim,), jnp.float64)
        jac_u_shape = jax.ShapeDtypeStruct((action_dim,), jnp.float64)
        vjp_x, vjp_u = jax.pure_callback(_vjp_fn, (jac_x_shape, jac_u_shape), x, u, g, vmap_method="sequential")
        return vjp_x, vjp_u, None, None

    jax_dynamics.defvjp(jax_dynamics_fwd, jax_dynamics_bwd)
    return jax_dynamics

def build_jax_disturbance(error_model: torch.nn.Module, q_learned: float, device: torch.device, state_dim: int, action_dim: int):
    def _dist_fn(X_prefix_np, U_prefix_np):
        with torch.no_grad():
            X_t = torch.from_numpy(np.array(X_prefix_np)).float().to(device)
            U_t = torch.from_numpy(np.array(U_prefix_np)).float().to(device)
            if X_t.ndim == 1:
                X_t = X_t.unsqueeze(0)
                U_t = U_t.unsqueeze(0)
            model_input = torch.cat([X_t, U_t], dim=-1)
            L = error_model(model_input) 
            return np.asarray((q_learned * L).cpu().numpy(), dtype=np.float64)

    def jax_disturbance(X_prefix, U_prefix):
        T = X_prefix.shape[0]
        result_shape = jax.ShapeDtypeStruct((T, state_dim, state_dim), jnp.float64)
        return jax.pure_callback(_dist_fn, result_shape, X_prefix, U_prefix, vmap_method="sequential")
        
    return jax_disturbance

def make_constant_jax_disturbance(calibrated_cholesky: np.ndarray, state_dim: int):
    calibrated_cholesky = jnp.asarray(calibrated_cholesky, dtype=jnp.float64)
    if calibrated_cholesky.shape != (state_dim, state_dim):
        raise ValueError(
            f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {calibrated_cholesky.shape}."
        )

    def jax_disturbance(X_prefix, U_prefix):
        seq_len = X_prefix.shape[0]
        return jnp.broadcast_to(calibrated_cholesky, (seq_len, state_dim, state_dim))

    return jax_disturbance

def load_calibrated_cholesky(path: Path) -> np.ndarray:
    payload = torch.load(path.expanduser(), map_location="cpu")
    if "calibrated_cholesky" in payload:
        matrix = payload["calibrated_cholesky"]
    elif "cholesky" in payload and "q_fixed" in payload:
        matrix = payload["cholesky"] * payload["q_fixed"]
    else:
        raise KeyError(
            f"{path} must contain either 'calibrated_cholesky' or both 'cholesky' and 'q_fixed'."
        )
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)

def load_action_stats_from_dataset(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Action-statistics dataset not found: {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        if "action" not in h5:
            raise KeyError(f"{dataset_path} does not contain an 'action' dataset.")
        if int(h5["action"].shape[-1]) != int(action_dim):
            raise ValueError(
                f"Expected action_dim={action_dim} in {dataset_path}, got {h5['action'].shape[-1]}."
            )
        actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = actions[~np.isnan(actions).any(axis=1)]
    if finite_actions.shape[0] == 0:
        raise ValueError(f"No finite actions found in {dataset_path}.")
    action_mean = finite_actions.mean(axis=0).astype(np.float64)
    action_std = np.maximum(finite_actions.std(axis=0).astype(np.float64), 1e-6)
    return action_mean, action_std

def resolve_action_stats_dataset_path(cfg: PlanSLSReacherConfig) -> Path:
    if cfg.action_stats_dataset_path is not None:
        return cfg.action_stats_dataset_path
    for candidate in DEFAULT_ACTION_STATS_CANDIDATES:
        if candidate.expanduser().is_file():
            return candidate
    if cfg.dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        return cfg.dataset_path
    return DEFAULT_ACTION_STATS_CANDIDATES[0]

def resolve_obstacle_model_artifact_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Obstacle model artifact not found: {candidate}")
    direct_model = candidate / "model.pt"
    if direct_model.is_file():
        return direct_model
    matches = sorted(candidate.glob("*/model.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No obstacle model artifact found under {candidate}")
    return max(matches, key=lambda item: item.stat().st_mtime)

def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array

def _pick_endpoint_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of keys {names}, got {sorted(mapping.keys())}.")

def _pick_optional_endpoint_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None

def _select_pair_value(value, episode_idx: int, pair_count: Optional[int]):
    if isinstance(value, dict):
        return {key: _select_pair_value(item, episode_idx, pair_count) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if pair_count is not None and len(value) == int(pair_count):
            return value[episode_idx]
        return value
    if isinstance(value, torch.Tensor):
        if pair_count is not None and value.ndim > 0 and int(value.shape[0]) == int(pair_count):
            return value[episode_idx]
        return value
    if isinstance(value, np.ndarray):
        if pair_count is not None and value.ndim > 0 and int(value.shape[0]) == int(pair_count):
            return value[episode_idx]
        return value
    return value

def _infer_pair_count(payload) -> Optional[int]:
    if isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and "pair_count" in metadata:
            return int(metadata["pair_count"])
        if "pair_count" in payload:
            return int(payload["pair_count"])
        for key in ("pairs", "episodes", "endpoint_pairs"):
            if key in payload and isinstance(payload[key], (list, tuple)):
                return len(payload[key])
        if "start" in payload and ("goal" in payload or "end" in payload):
            start = payload["start"]
            if isinstance(start, dict):
                for item in start.values():
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
                        return int(item.shape[0])
                    if isinstance(item, np.ndarray) and item.ndim > 0:
                        return int(item.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return None

def _select_endpoint_pair(payload, episode_idx: int):
    pair_count = _infer_pair_count(payload)
    if pair_count is not None:
        if episode_idx < 0 or episode_idx >= pair_count:
            raise ValueError(f"episode_idx must be in [0, {pair_count - 1}], got {episode_idx}.")

    if isinstance(payload, (list, tuple)):
        return payload[episode_idx], pair_count

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported endpoint pair payload type: {type(payload)!r}")

    for key in ("pairs", "episodes", "endpoint_pairs"):
        if key in payload:
            pairs = payload[key]
            if not isinstance(pairs, (list, tuple)):
                raise TypeError(f"Endpoint payload key '{key}' must be a list/tuple, got {type(pairs)!r}.")
            return pairs[episode_idx], len(pairs)

    if "start" in payload and ("goal" in payload or "end" in payload):
        return {
            "start": _select_pair_value(payload["start"], episode_idx, pair_count),
            "goal": _select_pair_value(_pick_endpoint_key(payload, ("goal", "end")), episode_idx, pair_count),
        }, pair_count

    raise KeyError(
        "Endpoint .pt payload must be a list of pairs, contain a 'pairs'/'episodes' list, "
        "or contain top-level 'start' and 'goal'/'end' entries."
    )

def _load_endpoint_pair_episode(dataset_path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray], int]:
    payload = torch.load(dataset_path.expanduser(), map_location="cpu", weights_only=False)
    pair_count = _infer_pair_count(payload)
    rng = np.random.default_rng(seed)
    if episode_idx is None:
        if pair_count is None:
            episode_idx = 0
        else:
            episode_idx = int(rng.integers(pair_count))

    pair, pair_count = _select_endpoint_pair(payload, int(episode_idx))
    if not isinstance(pair, dict):
        raise TypeError(f"Selected endpoint pair must be a dict, got {type(pair)!r}.")

    start = _pick_endpoint_key(pair, ("start", "initial", "source"))
    goal = _pick_endpoint_key(pair, ("goal", "end", "target"))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("Endpoint pair 'start' and 'goal'/'end' entries must both be dicts.")

    pixels_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("pixels", "pixel", "image", "rgb")), dtype=np.uint8),
            _as_numpy(_pick_endpoint_key(goal, ("pixels", "pixel", "image", "rgb")), dtype=np.uint8),
        ],
        axis=0,
    )
    task_target_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("task_target", "task_state", "target")), dtype=np.float32),
            _as_numpy(_pick_endpoint_key(goal, ("task_target", "task_state", "target")), dtype=np.float32),
        ],
        axis=0,
    )
    start_qpos = _as_numpy(_pick_endpoint_key(start, ("qpos", "q_pos")), dtype=np.float32)
    goal_qpos = _as_numpy(_pick_endpoint_key(goal, ("qpos", "q_pos")), dtype=np.float32)
    qpos_np = np.stack([start_qpos, goal_qpos], axis=0)
    start_qvel_raw = _pick_optional_endpoint_key(start, ("qvel", "q_vel"))
    goal_qvel_raw = _pick_optional_endpoint_key(goal, ("qvel", "q_vel"))
    start_qvel = np.zeros_like(start_qpos, dtype=np.float32) if start_qvel_raw is None else _as_numpy(start_qvel_raw, dtype=np.float32)
    goal_qvel = np.zeros_like(goal_qpos, dtype=np.float32) if goal_qvel_raw is None else _as_numpy(goal_qvel_raw, dtype=np.float32)
    qvel_np = np.stack(
        [
            start_qvel,
            goal_qvel,
        ],
        axis=0,
    )
    control_np = np.stack(
        [
            _as_numpy(_pick_endpoint_key(start, ("control", "ctrl")), dtype=np.float32),
            _as_numpy(_pick_endpoint_key(goal, ("control", "ctrl")), dtype=np.float32),
        ],
        axis=0,
    )
    time_np = np.asarray([[0.0], [1.0 / 30.0]], dtype=np.float32)

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    episode = {
        "pixels": pixels_np,
        "task_target": task_target_np,
        "qpos": qpos_np,
        "qvel": qvel_np,
        "control": control_np,
        "time": time_np,
        "camera_name": str(metadata.get("camera", "video_cam")) if isinstance(metadata, dict) else "video_cam",
        "control_decimation": int(metadata.get("control_decimation", 25)) if isinstance(metadata, dict) else 25,
        "disable_shadows": bool(metadata.get("disable_shadows", True)) if isinstance(metadata, dict) else True,
        "control_timestep": float(metadata.get("control_timestep", 1.0 / 30.0)) if isinstance(metadata, dict) else 1.0 / 30.0,
        "is_endpoint_pair": True,
    }
    print(
        f"Loaded endpoint pair {episode_idx}"
        + (f"/{pair_count}" if pair_count is not None else "")
        + f" from {dataset_path}"
    )
    return episode, int(episode_idx)

def _load_hdf5_episode(dataset_path: Path, episode_idx: Optional[int], horizon: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        valid_episodes = np.flatnonzero(ep_len >= horizon)
        if valid_episodes.size == 0:
            raise ValueError(f"No episodes in {dataset_path} have length >= horizon={horizon}.")
        if episode_idx is None:
            episode_idx = int(rng.choice(valid_episodes))
        elif episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"episode_idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        elif ep_len[episode_idx] < horizon:
            raise ValueError(f"episode_idx {episode_idx} has length {ep_len[episode_idx]}, below horizon={horizon}.")

        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        episode = {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "task_target": np.asarray(h5["task_target"][rows], dtype=np.float32),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "control": np.asarray(h5["control"][rows], dtype=np.float32),
            "time": np.asarray(h5["time"][rows], dtype=np.float32) if "time" in h5 else np.zeros((len(rows), 1), dtype=np.float32),
            "camera_name": str(h5.attrs.get("camera", "video_cam")),
            "control_decimation": int(h5.attrs.get("control_decimation", 25)),
            "disable_shadows": bool(h5.attrs.get("disable_shadows", True)),
            "control_timestep": float(h5.attrs.get("control_timestep", 1.0 / 30.0)),
            "is_endpoint_pair": False,
        }
    return episode, int(episode_idx)

def load_planning_episode(dataset_path: Path, episode_idx: Optional[int], horizon: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Planning dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".pt":
        return _load_endpoint_pair_episode(dataset_path, episode_idx, seed)
    return _load_hdf5_episode(dataset_path, episode_idx, horizon, seed)

# --- Cost, Context & Utilities ---

def make_goal_tracking_cost(
    r_control: float,
    horizon: int,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W, W_terminal)
        dz = z - goal_state
        total_cost = jnp.sum(active_W * dz**2) + r_control * jnp.sum(u**2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_violation = jax.nn.softplus(
                obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z)
            )
            total_cost = total_cost + obstacle_penalty_weight * obstacle_violation**2
        return total_cost
    return cost

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

def load_markov_ellipsoid_unit_precision(path: Path, state_dim: int) -> np.ndarray:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Latent ellipsoid artifact not found: {path}")
    if path.suffix == ".npz":
        payload = np.load(path)
        if "markov_unit_precision" in payload:
            matrix = payload["markov_unit_precision"]
        elif "unit_precision" in payload:
            matrix = payload["unit_precision"]
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "markov_unit_precision" in payload:
            matrix = payload["markov_unit_precision"]
        elif "unit_precision" in payload:
            matrix = payload["unit_precision"]
        else:
            raise KeyError(f"{path} must contain 'markov_unit_precision' or 'unit_precision'.")
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (state_dim, state_dim):
        raise ValueError(f"Expected ellipsoid matrix shape {(state_dim, state_dim)}, got {matrix.shape}.")
    return matrix

def make_markov_ellipsoid_constraint(unit_precision: np.ndarray, margin: float = 0.0):
    unit_precision = jnp.asarray(unit_precision, dtype=jnp.float64)
    def constraint(x, u, t):
        score = x @ unit_precision @ x
        return jnp.asarray([score - 1.0 + float(margin)])
    return constraint

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None: candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return pixel_mean, pixel_std

def preprocess_reacher_pixels(
    pixels: np.ndarray | torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)) if isinstance(pixels, np.ndarray) else pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std

@torch.no_grad()
def encode_single_frame(
    model: torch.nn.Module,
    pixel_np: np.ndarray,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = preprocess_reacher_pixels(
        pixel_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    ).to(device)
    output = model.encoder(tensor, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels_np: np.ndarray,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = preprocess_reacher_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    ).to(device)
    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32]
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float64)

def main():
    cfg = pyrallis.parse(config_class=PlanSLSReacherConfig)
    device = require_device(cfg.device)
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    init_key = jax.random.PRNGKey(cfg.seed)

    # 1. Load Configurations & Model Parameters
    model_dir = cfg.model_dir.expanduser().resolve()
    config_dict = load_reacher_config(model_dir)
    checkpoint_path = latest_reacher_object_checkpoint(model_dir).resolve()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device).eval()

    state_dim = int(config_dict.get("markov_state_dim", 10))
    action_dim = int(config_dict.get("action_dim", 2))
    img_size = int(config_dict.get("img_size", 224))
    pixel_mean, pixel_std = imagenet_pixel_stats(device)

    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    obstacle_model = None
    obstacle_constraint = None
    if cfg.enable_obstacle:
        obstacle_artifact_path = resolve_obstacle_model_artifact_path(cfg.obstacle_model_path)
        obstacle_model = build_jax_obstacle_from_artifact(obstacle_artifact_path, init_key)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(
                f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}."
            )
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {obstacle_artifact_path} "
            f"with threshold {float(obstacle_model.threshold):.6g} and margin {cfg.obstacle_margin:.6g}"
        )
    else:
        print("Obstacle avoidance disabled.")
    if cfg.use_constant_covariance:
        calibrated_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky, state_dim)
        print(f"Using fixed calibrated covariance disturbance from {cfg.constant_covariance_path}")
    else:
        error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = build_jax_disturbance(error_model, cfg.q_learned, device, state_dim, action_dim)

    # 2. Extract Normalization Parameters via LeWMRopeDataset wrapper
    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    action_mean, action_std = load_action_stats_from_dataset(action_stats_dataset_path, action_dim)
    print(f"Using action statistics from {action_stats_dataset_path}")

    dataset_path = cfg.dataset_path.expanduser().resolve()
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one reacher test trajectory with 2 or more frames.")
    rng = np.random.default_rng(cfg.seed)
    if cfg.episode_idx is None:
        episode_idx = int(rng.choice(valid_episodes))
    else:
        episode_idx = int(cfg.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"episode_idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"episode_idx {episode_idx} must have at least 2 frames, got {ep_len[episode_idx]}.")

    episode = load_dataset_episode(dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    start_qpos, goal_qpos, start_idx, goal_idx, start_goal_source = resolve_start_goal_qpos(
        dataset_qpos=qpos_np,
        swap_start_goal=False,
        default_start_qpos=DEFAULT_START_QPOS,
        default_goal_qpos=DEFAULT_GOAL_QPOS,
    )
    run_name = (
        f"{int(time.time())}_episode_custom"
        if start_goal_source == "fixed_qpos"
        else f"{int(time.time())}_episode_{episode_idx:05d}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    zero_qvel = np.zeros_like(qvel_np[0], dtype=np.float32)
    env = make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )

    # 3. Reference Extraction Sequence
    start_frame = reset_reacher_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    goal_frame = reset_reacher_env_to_state(
        env,
        seed=episode_seed,
        qpos=goal_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    start_z = encode_single_frame(model, start_frame, device, img_size, pixel_mean, pixel_std)
    goal_z = encode_single_frame(model, goal_frame, device, img_size, pixel_mean, pixel_std)
    start_state = torch.cat([start_z, torch.zeros_like(start_z)], dim=-1).cpu().numpy().astype(np.float64)
    goal_state = torch.cat([goal_z, torch.zeros_like(goal_z)], dim=-1).cpu().numpy().astype(np.float64)

    if obstacle_model is not None:
        start_score = float(obstacle_model(jnp.asarray(start_state)))
        goal_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        if start_score <= required_score or goal_score <= required_score:
            print(
                "Terminating: start and goal must both be outside the conformal obstacle set. "
                f"Required score > {required_score:.6g}; "
                f"start_score={start_score:.6g}, goal_score={goal_score:.6g}."
            )
            sys.exit(1)
        print(
            "Obstacle sanity check passed: "
            f"start_score={start_score:.6g}, goal_score={goal_score:.6g}, "
            f"required_score>{required_score:.6g}"
        )

    imageio.imwrite(run_dir / "start_reacher.png", start_frame)
    imageio.imwrite(run_dir / "goal_reacher.png", goal_frame)

    # 4. iLQR-style nominal objective: every stage and terminal state targets the final goal.
    W_state = jnp.ones((state_dim,)) * cfg.q_stage
    W_term = jnp.ones((state_dim,)) * cfg.q_terminal

    cost = make_goal_tracking_cost(
        r_control=cfg.r_control,
        horizon=cfg.horizon,
        W_terminal=W_term,
        goal_state=jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )

    # 5. Solver Parameter Building Footprint
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=True, initialize_nominal=True, warm_start=False, rti=True)
    sqp_cfg = SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=True)
    admm_cfg = ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e5, max_iterations=1200, rho_update_frequency=20, initial_rho=1.0)
    
    mpc_dt = 1.0 / float(physics_freq_hz)
    mpc_cfg = MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_state, u_ref=jnp.zeros(action_dim), dt=mpc_dt)

    u_min, u_max = -3.5 * jnp.ones(action_dim), 3.5 * jnp.ones(action_dim)
    x_min, x_max = -6.0 * jnp.ones(state_dim), 6.0 * jnp.ones(state_dim)
    # Keep the finite-difference half of the Markov state tightly bounded.
    x_min = x_min.at[x_min.shape[0]//2:].set(-0.5)
    x_max = x_max.at[x_max.shape[0]//2:].set(0.5)
    ellipsoid_unit_precision = None
    if cfg.use_latent_ellipsoid_constraint:
        ellipsoid_unit_precision = load_markov_ellipsoid_unit_precision(cfg.latent_ellipsoid_path, state_dim)
        state_constraint = make_markov_ellipsoid_constraint(
            ellipsoid_unit_precision,
            margin=cfg.latent_ellipsoid_margin,
        )
        state_constraint_count = 1
        print(
            f"Using calibrated Markov ellipsoid state constraint from {cfg.latent_ellipsoid_path} "
            f"with margin {cfg.latent_ellipsoid_margin:.6g}"
        )
        
        # Ellipsoid sanity check: ensure start and goal are within the ellipsoid
        unit_precision_np = np.asarray(ellipsoid_unit_precision, dtype=np.float64)
        start_ellipsoid_score = float(start_state @ unit_precision_np @ start_state)
        goal_ellipsoid_score = float(goal_state @ unit_precision_np @ goal_state)
        ellipsoid_margin_score = 1.0 + float(cfg.latent_ellipsoid_margin)
        if start_ellipsoid_score > ellipsoid_margin_score or goal_ellipsoid_score > ellipsoid_margin_score:
            print(
                "Terminating: start and goal must both be inside the conformal ellipsoid. "
                f"Required score <= {ellipsoid_margin_score:.6g}; "
                f"start_score={start_ellipsoid_score:.6g}, goal_score={goal_ellipsoid_score:.6g}."
            )
            sys.exit(1)
        print(
            "Ellipsoid sanity check passed: "
            f"start_score={start_ellipsoid_score:.6g}, goal_score={goal_ellipsoid_score:.6g}, "
            f"max_allowed_score<={ellipsoid_margin_score:.6g}"
        )
    else:
        state_constraint = make_state_box_constraints(x_min, x_max)
        state_constraint_count = 2 * state_dim
        print("Using hand-tuned Markov state box constraints.")
    if obstacle_constraint is not None:
        constraints_all = combine_constraints(
            state_constraint,
            obstacle_constraint,
            make_control_box_constraints(u_min, u_max),
        )
    else:
        constraints_all = combine_constraints(
            state_constraint,
            make_control_box_constraints(u_min, u_max),
        )

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg, config=mpc_cfg, dynamics=dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + state_constraint_count + (1 if obstacle_constraint is not None else 0),
        disturbance=disturbance, shift=1, X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64), U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64)
    )

    # 6. Receding Horizon Closed-Loop Execution
    current_frame = reset_reacher_env_to_state(
        env,
        seed=episode_seed,
        qpos=start_qpos,
        qvel=zero_qvel,
        height=height,
        width=width,
    )
    current_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []
    X_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_u0 = np.zeros(action_dim, dtype=np.float32)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="Reacher conformal SLS execution loop")
    for step_idx in pbar:
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_ref, parameter=mpc_dt)
            solver_status = "sls_mpc"
        except Exception:
            u0, X_pred, U_pred = None, None, None
            solver_status = "exception_fallback"

        if u0 is None or X_pred is None or not np.all(np.isfinite(np.asarray(X_pred))):
            u0 = prev_u0
            solver_status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)

        u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        rollout_frames.append(current_frame.copy())
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u0_raw.astype(np.float32))

        next_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        current_qpos = np.asarray(env._env.physics.data.qpos[: qpos_np.shape[1]], dtype=np.float64)
        latent_err = float(np.linalg.norm(current_state - goal_state))
        task_err = float(np.linalg.norm(current_qpos - goal_qpos.astype(np.float64)))
        if obstacle_model is not None:
            obstacle_score = float(obstacle_model(jnp.asarray(current_state)))
            obs_free = obstacle_score > 0.0
        else:
            obs_free = "n/a"
        if ellipsoid_unit_precision is not None:
            ellipsoid_score = float(current_state @ ellipsoid_unit_precision @ current_state)
            ellip_in = ellipsoid_score <= 1.0 - float(cfg.latent_ellipsoid_margin)
        else:
            ellip_in = "n/a"
        pbar.set_postfix(
            latent_error=f"{latent_err:.4f}",
            task_error=f"{task_err:.4f}",
            obs_free=obs_free,
            ellip_in=ellip_in,
            status=solver_status,
        )

        if latent_err <= 0.2 or task_err <= 0.05 or terminated or truncated:
            break

    save_rollout_video(rollout_frames, run_dir, fps=cfg.video_fps)
    env.close()
    np.savez(
        run_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    print(f"Reacher SLS MPC planning sequence logged inside: {run_dir}")

if __name__ == "__main__":
    main()
