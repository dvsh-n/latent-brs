#!/usr/bin/env python3
"""Plan in OGBench cube pixel space using Conformal SLS MPC following an Oracle grasp phase."""

import os
import sys
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

mpl_config_dir = Path(__file__).resolve().parent / ".mplconfig"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium
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

# OGBench environment & data tracking utilities
import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie
from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset
from error_model import MGNLLPredictor

@dataclass
class PlanSLSOGBenchConfig:
    """Configuration for Conformal SLS MPC OGBench Cube Planning"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("ogbench_cube/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("ogbench_cube/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("ogbench_cube/eval/fixed_error_covariance.pt"))
    enable_obstacle: bool = field(default=False)
    obstacle_model_path: Path = field(default=Path("ogbench_cube/models/obs_net/model.pt"))
    obstacle_margin: float = field(default=0.0)
    obstacle_penalty_weight: float = field(default=1000.0)
    dataset_path: Path = field(default=Path("ogbench_cube/data/test_data/ogbench_cube_test.h5"))
    action_stats_dataset_path: Optional[Path] = field(default=None)
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mpc_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=15)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    visualize_success_colors: bool = field(default=False)
    terminate_on_ogbench_success: bool = field(default=True)
    debug_solver: bool = field(default=False)
    max_normalized_action: float = field(default=10.0)
    
    grasp_contact_threshold: float = 0.5
    grasp_alignment_threshold: float = 0.03
    q_stage: float = field(default=0.005)
    q_terminal: float = field(default=5.0)
    r_control: float = field(default=0.1)
    r_control_u4: float = field(default=0.1)

# --- Helper Utilities ---

class JAXObstacleMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    feature_mean: jax.Array
    feature_std: jax.Array
    threshold: jax.Array
    input_dim: int
    activation: str = eqx.field(static=True)

    def __call__(self, state):
        z = state[: self.input_dim]
        x = (z - self.feature_mean) / self.feature_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jnp.tanh(x) if self.activation == "tanh" else jax.nn.gelu(x)
        return self.linear_layers[-1](x).squeeze(-1)

def resolve_obstacle_activation(artifact: dict, artifact_path: Path) -> str:
    activation = artifact.get("activation") or artifact.get("cache_config", {}).get("activation")
    if activation is not None:
        return str(activation).lower()
    return "tanh" if int(artifact["hidden_dim"]) == 6 or "obs_net_small" in str(artifact_path) else "gelu"

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
    activation = resolve_obstacle_activation(artifact, artifact_path)

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
        activation=activation,
    )

def load_calibrated_cholesky(path: Path) -> np.ndarray:
    payload = torch.load(path.expanduser(), map_location="cpu")
    if "calibrated_cholesky" in payload:
        matrix = payload["calibrated_cholesky"]
    elif "cholesky" in payload and "q_fixed" in payload:
        matrix = payload["cholesky"] * payload["q_fixed"]
    else:
        raise KeyError(f"{path} must contain either 'calibrated_cholesky' or both 'cholesky' and 'q_fixed'.")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)

def make_constant_jax_disturbance(calibrated_cholesky: np.ndarray, state_dim: int):
    calibrated_cholesky = jnp.asarray(calibrated_cholesky, dtype=jnp.float64)
    if calibrated_cholesky.shape != (state_dim, state_dim):
        raise ValueError(f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {calibrated_cholesky.shape}.")

    def jax_disturbance(X_prefix, U_prefix):
        return jnp.broadcast_to(calibrated_cholesky, (X_prefix.shape[0], state_dim, state_dim))

    return jax_disturbance

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    return lambda x, u, t: jnp.concatenate([x - x_max, x_min - x], axis=0)

def combine_constraints(*constraints):
    return lambda x, u, t: jnp.concatenate([constraint(x, u, t) for constraint in constraints], axis=0)

def make_action_weights(action_dim: int, r_control: float, r_control_u4: float, grip_idx: int = 4) -> jnp.ndarray:
    weights = jnp.ones((int(action_dim),), dtype=jnp.float64) * float(r_control)
    if 0 <= int(grip_idx) < int(action_dim):
        weights = weights.at[int(grip_idx)].set(float(r_control_u4))
    return weights

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint

def obstacle_set_violation(obstacle_model: JAXObstacleMLP, state: jax.Array, margin: float) -> jax.Array:
    required_score = obstacle_model.threshold + float(margin)
    return jnp.maximum(required_score - obstacle_model(state), 0.0)

def obstacle_linearization_stats(
    obstacle_model: JAXObstacleMLP,
    states: np.ndarray,
    *,
    margin: float,
) -> dict[str, float | bool]:
    states_jax = jnp.asarray(states, dtype=jnp.float64)
    required_score = obstacle_model.threshold + float(margin)

    def constraint_value(state):
        return required_score - obstacle_model(state)

    scores = jax.vmap(obstacle_model)(states_jax)
    values = jax.vmap(constraint_value)(states_jax)
    grads = jax.vmap(jax.grad(constraint_value))(states_jax)
    grad_norms = jnp.linalg.norm(grads, axis=-1)
    return {
        "score_min": float(jnp.min(scores)),
        "score_max": float(jnp.max(scores)),
        "g_min": float(jnp.min(values)),
        "g_max": float(jnp.max(values)),
        "grad_norm_min": float(jnp.min(grad_norms)),
        "grad_norm_max": float(jnp.max(grad_norms)),
        "grad_abs_max": float(jnp.max(jnp.abs(grads))),
        "finite": bool(jnp.all(jnp.isfinite(scores)) & jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(grads))),
    }

def print_obstacle_debug(label: str, obstacle_model: JAXObstacleMLP | None, states: np.ndarray, margin: float) -> None:
    if obstacle_model is None:
        return
    stats = obstacle_linearization_stats(obstacle_model, states, margin=margin)
    print(
        f"[obs-debug:{label}] "
        f"score=[{stats['score_min']:.6g},{stats['score_max']:.6g}] "
        f"g=[{stats['g_min']:.6g},{stats['g_max']:.6g}] "
        f"|grad|=[{stats['grad_norm_min']:.6g},{stats['grad_norm_max']:.6g}] "
        f"grad_abs_max={stats['grad_abs_max']:.6g} finite={stats['finite']}"
    )

def reset_controller_primal_warm_start(controller: GenericMPC, current_state: np.ndarray) -> None:
    controller.U0 = jnp.zeros((controller.config.N, controller.config.nu), dtype=jnp.float64)
    controller.X0 = jnp.tile(jnp.asarray(current_state, dtype=jnp.float64)[None, :], (controller.config.N + 1, 1))
    controller.V0 = jnp.zeros((controller.config.N + 1, controller.config.n), dtype=jnp.float64)

def sanitize_controller_warm_start(
    controller: GenericMPC,
    current_state: np.ndarray,
    *,
    obstacle_model: JAXObstacleMLP | None,
    obstacle_margin: float,
    max_abs: float,
    debug: bool,
) -> bool:
    X0_np = np.asarray(controller.X0, dtype=np.float64)
    U0_np = np.asarray(controller.U0, dtype=np.float64)
    reason = None
    if not np.all(np.isfinite(X0_np)) or not np.all(np.isfinite(U0_np)):
        reason = "nonfinite"
    elif np.max(np.abs(X0_np)) > max_abs or np.max(np.abs(U0_np)) > max_abs:
        reason = "huge"
    elif obstacle_model is not None:
        stats = obstacle_linearization_stats(obstacle_model, X0_np, margin=obstacle_margin)
        if (not stats["finite"]) or stats["g_max"] > 0.0:
            reason = f"obstacle_violation_gmax={stats['g_max']:.6g}"
    if reason is None:
        return False
    reset_controller_primal_warm_start(controller, current_state)
    if debug:
        print(f"[solver-debug] reset primal warm start before solve: {reason}")
    return True

def make_tracking_cost(
    action_weights: jnp.ndarray,
    horizon: int,
    W_term: jnp.ndarray,
    goal_state: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    action_weights = jnp.asarray(action_weights, dtype=jnp.float64)
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        dz = z - jnp.where(is_not_terminal, reference[t], goal_state)
        total_cost = jnp.sum(jnp.where(is_not_terminal, W, W_term) * dz**2) + jnp.sum(action_weights * u**2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_violation = obstacle_set_violation(obstacle_model, z, obstacle_margin)
            total_cost = total_cost + obstacle_penalty_weight * obstacle_violation**2
        return total_cost
    return cost

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def resolve_action_stats_dataset_path(cfg: PlanSLSOGBenchConfig) -> Path:
    return cfg.action_stats_dataset_path if cfg.action_stats_dataset_path is not None else cfg.dataset_path

def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array

def _pick_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of keys {names}, got {sorted(mapping.keys())}.")

def _pick_optional_key(mapping: dict, names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None

def _pick_optional_or(mapping: dict, names: tuple[str, ...], default):
    value = _pick_optional_key(mapping, names)
    return default if value is None else value

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
        if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
            start = payload["start"]
            if isinstance(start, dict):
                for item in start.values():
                    if isinstance(item, (torch.Tensor, np.ndarray)) and item.ndim > 0:
                        return int(item.shape[0])
    if isinstance(payload, (list, tuple)):
        return len(payload)
    return None

def _select_pair_value(value, episode_idx: int, pair_count: Optional[int]):
    if isinstance(value, dict):
        return {key: _select_pair_value(item, episode_idx, pair_count) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and pair_count is not None and len(value) == pair_count:
        return value[episode_idx]
    if isinstance(value, torch.Tensor) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    if isinstance(value, np.ndarray) and pair_count is not None and value.ndim > 0 and int(value.shape[0]) == pair_count:
        return value[episode_idx]
    return value

def _select_endpoint_pair(payload, episode_idx: int):
    pair_count = _infer_pair_count(payload)
    if pair_count is not None and not 0 <= episode_idx < pair_count:
        raise ValueError(f"episode_idx must be in [0, {pair_count - 1}], got {episode_idx}.")
    if isinstance(payload, (list, tuple)):
        return payload[episode_idx], pair_count
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported endpoint payload type: {type(payload)!r}.")
    for key in ("pairs", "episodes", "endpoint_pairs"):
        if key in payload:
            pairs = payload[key]
            if not isinstance(pairs, (list, tuple)):
                raise TypeError(f"Endpoint payload key '{key}' must be a list/tuple, got {type(pairs)!r}.")
            return pairs[episode_idx], len(pairs)
    if "start" in payload and ("goal" in payload or "end" in payload or "target" in payload):
        return {
            "start": _select_pair_value(payload["start"], episode_idx, pair_count),
            "goal": _select_pair_value(_pick_key(payload, ("goal", "end", "target")), episode_idx, pair_count),
        }, pair_count
    raise KeyError(
        "Endpoint .pt payload must be a list of pairs, contain a 'pairs'/'episodes' list, "
        "or contain top-level 'start' and 'goal'/'end'/'target' entries."
    )

def _load_endpoint_pair(path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    pair_count = _infer_pair_count(payload)
    rng = np.random.default_rng(seed)
    if episode_idx is None:
        episode_idx = 0 if pair_count is None else int(rng.integers(pair_count))
    pair, pair_count = _select_endpoint_pair(payload, int(episode_idx))
    if not isinstance(pair, dict):
        raise TypeError(f"Selected endpoint pair must be a dict, got {type(pair)!r}.")

    start = _pick_key(pair, ("start", "initial", "source"))
    goal = _pick_key(pair, ("goal", "end", "target"))
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise TypeError("Endpoint pair 'start' and 'goal'/'end' entries must both be dicts.")

    start_qpos_raw = _pick_optional_key(start, ("qpos", "q_pos"))
    goal_qpos_raw = _pick_optional_key(goal, ("qpos", "q_pos"))
    start_block_pos = _as_numpy(_pick_key(start, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    goal_block_pos = _as_numpy(_pick_key(goal, ("task_target", "block_pos", "object_pos", "pos")), dtype=np.float32)
    start_block_yaw = float(_as_numpy(_pick_key(start, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    goal_block_yaw = float(_as_numpy(_pick_key(goal, ("yaw", "block_yaw", "object_yaw"))).reshape(-1)[0])
    start_qpos = None if start_qpos_raw is None else _as_numpy(start_qpos_raw, dtype=np.float32)
    goal_qpos = None if goal_qpos_raw is None else _as_numpy(goal_qpos_raw, dtype=np.float32)
    start_qvel_raw = _pick_optional_key(start, ("qvel", "q_vel"))
    goal_qvel_raw = _pick_optional_key(goal, ("qvel", "q_vel"))
    start_pixels_raw = _pick_optional_key(start, ("pixels", "image", "rgb"))
    goal_pixels_raw = _pick_optional_key(goal, ("pixels", "image", "rgb"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    episode = {
        "needs_qpos_synthesis": start_qpos is None or goal_qpos is None,
        "block_pos_init": start_block_pos,
        "block_yaw_init": start_block_yaw,
        "block_pos_goal": goal_block_pos,
        "block_yaw_goal": goal_block_yaw,
        "qpos_init": start_qpos,
        "qvel_init": None if start_qpos is None else (np.zeros_like(start_qpos, dtype=np.float32) if start_qvel_raw is None else _as_numpy(start_qvel_raw, dtype=np.float32)),
        "qpos_goal": goal_qpos,
        "qvel_goal": None if goal_qpos is None else (np.zeros_like(goal_qpos, dtype=np.float32) if goal_qvel_raw is None else _as_numpy(goal_qvel_raw, dtype=np.float32)),
        "target_block_pos_init": _as_numpy(_pick_optional_or(start, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos), dtype=np.float32),
        "target_block_yaw_init": float(_as_numpy(_pick_optional_or(start, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)).reshape(-1)[0]),
        "target_block_pos_goal": _as_numpy(_pick_optional_or(goal, ("target_block_pos", "privileged/target_block_pos", "target_pos", "goal_pos"), goal_block_pos), dtype=np.float32),
        "target_block_yaw_goal": float(_as_numpy(_pick_optional_or(goal, ("target_block_yaw", "privileged/target_block_yaw", "target_yaw", "goal_yaw"), goal_block_yaw)).reshape(-1)[0]),
        "start_pixels": None if start_pixels_raw is None else _as_numpy(start_pixels_raw, dtype=np.uint8),
        "goal_pixels": None if goal_pixels_raw is None else _as_numpy(goal_pixels_raw, dtype=np.uint8),
        "episode_seed": int(metadata.get("episode_seed", seed)) if isinstance(metadata, dict) else int(seed),
        "env_name": str(metadata.get("env_name", "cube-single-v0")) if isinstance(metadata, dict) else "cube-single-v0",
        "camera": str(metadata.get("camera", "front_pixels")) if isinstance(metadata, dict) else "front_pixels",
    }
    print(
        f"Loaded endpoint pair {episode_idx}"
        + (f"/{pair_count}" if pair_count is not None else "")
        + f" from {path}"
    )
    return episode, int(episode_idx)

def synthesize_qpos_qvel_from_block_pose(env: gymnasium.Env, pos: np.ndarray, yaw: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    joint_qpos = unwrapped._data.joint("object_joint_0").qpos
    joint_qpos[:3] = np.asarray(pos, dtype=np.float64)
    joint_qpos[3:] = np.asarray(lie.SO3.from_z_radians(float(yaw)).wxyz, dtype=np.float64)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return (
        np.asarray(unwrapped._data.qpos, dtype=np.float32).copy(),
        np.zeros_like(np.asarray(unwrapped._data.qvel, dtype=np.float32)),
    )

def load_planning_episode(path: Path, episode_idx: Optional[int], seed: int) -> tuple[dict[str, np.ndarray | int | str], int]:
    path = path.expanduser()
    if path.suffix.lower() == ".pt":
        return _load_endpoint_pair(path, episode_idx, seed)
    with h5py.File(path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        selected_idx = episode_idx if episode_idx is not None else int(np.random.choice(np.flatnonzero(ep_len >= 2)))
        rows = np.arange(int(h5["ep_offset"][selected_idx]), int(h5["ep_offset"][selected_idx]) + int(h5["ep_len"][selected_idx]))
        episode = {
            "qpos_init": np.asarray(h5["qpos"][rows[0]], dtype=np.float32),
            "qvel_init": np.asarray(h5["qvel"][rows[0]], dtype=np.float32),
            "qpos_goal": np.asarray(h5["qpos"][rows[-1]], dtype=np.float32),
            "qvel_goal": np.asarray(h5["qvel"][rows[-1]], dtype=np.float32),
            "start_pixels": np.asarray(h5["pixels"][rows[0]], dtype=np.uint8),
            "goal_pixels": np.asarray(h5["pixels"][rows[-1]], dtype=np.uint8),
            "target_block_pos_init": np.asarray(h5["target_block_pos"][rows[0]], dtype=np.float32),
            "target_block_yaw_init": float(h5["target_block_yaw"][rows[0], 0]),
            "target_block_pos_goal": np.asarray(h5["target_block_pos"][rows[-1]], dtype=np.float32),
            "target_block_yaw_goal": float(h5["target_block_yaw"][rows[-1], 0]),
            "episode_seed": int(h5["episode_seed"][selected_idx]) if "episode_seed" in h5 else int(seed),
            "env_name": str(h5.attrs.get("env_name", "cube-single-v0")),
            "camera": str(h5.attrs.get("camera", "front_pixels")),
        }
    return episode, int(selected_idx)

def cube_is_grasped(info, contact_thresh, align_thresh) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_thresh and block_alignment <= align_thresh)

def ogbench_success(info: dict) -> bool:
    success = info.get("success", False)
    if isinstance(success, dict):
        return all(bool(value) for value in success.values())
    return bool(np.asarray(success).item())

def restore_target_pose(env: gymnasium.Env, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    hide_target_cube(env)

def hide_target_cube(env: gymnasium.Env) -> None:
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

def render_without_target_cube(env: gymnasium.Env, camera: str) -> np.ndarray:
    hide_target_cube(env)
    return np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)

def reset_env_to_state(
    env: gymnasium.Env,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_block_pos: np.ndarray,
    target_block_yaw: float,
    camera: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[:qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[:qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(env, target_block_pos, target_block_yaw)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    return render_without_target_cube(env, camera), unwrapped.get_step_info()

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch[_=](\d+).*\.ckpt$")
    candidates = [(int(m.group(1)), p) for p in model_dir.glob("*.ckpt") for m in [pattern.match(p.name)] if m]
    if not candidates: raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    # FIX: Pushed tensor to device immediately before normalization arithmetic
    tensor = torch.from_numpy(pixel_np.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean) / pixel_std
    return model.projector(model.encoder(tensor, interpolate_pos_encoding=True).last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    # FIX: Pushed tensor to device immediately before normalization arithmetic
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean) / pixel_std
    
    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32]
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

# --- JAX Black-box Wrappers ---

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
        return jax.pure_callback(_fwd_fn, jax.ShapeDtypeStruct((state_dim,), jnp.float64), x, u, vmap_method="sequential")

    def jax_dynamics_fwd(x, u, t, parameter):
        return jax_dynamics(x, u, t, parameter), (x, u)

    def jax_dynamics_bwd(res, g):
        x, u = res
        vjp_x, vjp_u = jax.pure_callback(_vjp_fn, (jax.ShapeDtypeStruct((state_dim,), jnp.float64), jax.ShapeDtypeStruct((action_dim,), jnp.float64)), x, u, g, vmap_method="sequential")
        return vjp_x, vjp_u, None, None

    jax_dynamics.defvjp(jax_dynamics_fwd, jax_dynamics_bwd)
    return jax_dynamics

def build_jax_disturbance(error_model: torch.nn.Module, q_learned: float, device: torch.device, state_dim: int):
    def _dist_fn(X_prefix_np, U_prefix_np):
        with torch.no_grad():
            X_t = torch.from_numpy(np.array(X_prefix_np)).float().to(device)
            U_t = torch.from_numpy(np.array(U_prefix_np)).float().to(device)
            if X_t.ndim == 1:
                X_t, U_t = X_t.unsqueeze(0), U_t.unsqueeze(0)
            L = error_model(torch.cat([X_t, U_t], dim=-1)) 
            return np.asarray((q_learned * L).cpu().numpy(), dtype=np.float64)
    return lambda X_p, U_p: jax.pure_callback(_dist_fn, jax.ShapeDtypeStruct((X_p.shape[0], state_dim, state_dim), jnp.float64), X_p, U_p, vmap_method="sequential")

def main():
    cfg = pyrallis.parse(config_class=PlanSLSOGBenchConfig)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve() / f"{int(time.time())}_sls_cube"
    out_dir.mkdir(parents=True, exist_ok=True)
    init_key = jax.random.PRNGKey(cfg.seed)

    # 1. Environment & Config Loading
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json") as f: config_dict = json.load(f)
    
    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()

    state_dim = config_dict.get("markov_state_dim", 48)
    action_dim = config_dict.get("action_dim", 5)
    img_size = config_dict.get("img_size", 224)

    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    obstacle_model = None
    obstacle_constraint = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, init_key)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}.")
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {cfg.obstacle_model_path} "
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
        disturbance = build_jax_disturbance(error_model, cfg.q_learned, device, state_dim)

    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    train_stats = LeWMOGBenchCubeDataset(
        str(action_stats_dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean, action_std = train_stats.action_mean.astype(np.float32), train_stats.action_std.astype(np.float32)
    print(f"Using action statistics from {action_stats_dataset_path}")
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    episode, episode_idx = load_planning_episode(cfg.dataset_path, cfg.episode_idx, cfg.seed)
    qpos_init = episode["qpos_init"]
    qvel_init = episode["qvel_init"]
    qpos_goal = episode["qpos_goal"]
    qvel_goal = episode["qvel_goal"]
    target_block_pos_init = episode["target_block_pos_init"]
    target_block_yaw_init = float(episode["target_block_yaw_init"])
    target_block_pos_goal = episode["target_block_pos_goal"]
    target_block_yaw_goal = float(episode["target_block_yaw_goal"])
    start_pixels = episode.get("start_pixels")
    goal_pixels = episode.get("goal_pixels")
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])

    env = gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=cfg.visualize_success_colors,
        width=config_dict.get("width", 256),
        height=config_dict.get("height", 256),
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0, noise_smoothing=0.5)

    if bool(episode.get("needs_qpos_synthesis", False)):
        qpos_init, qvel_init = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_init"], dtype=np.float32),
            float(episode["block_yaw_init"]),
            episode_seed,
        )
        qpos_goal, qvel_goal = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_goal"], dtype=np.float32),
            float(episode["block_yaw_goal"]),
            episode_seed,
        )

    goal_frame, _ = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_goal,
        qvel=qvel_goal,
        target_block_pos=target_block_pos_goal,
        target_block_yaw=target_block_yaw_goal,
        camera=camera,
    )

    current_frame, current_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_init,
        qvel=qvel_init,
        target_block_pos=target_block_pos_init,
        target_block_yaw=target_block_yaw_init,
        camera=camera,
    )
    start_image = np.asarray(start_pixels, dtype=np.uint8).copy() if start_pixels is not None else current_frame.copy()
    goal_image = np.asarray(goal_pixels, dtype=np.uint8).copy() if goal_pixels is not None else goal_frame.copy()

    save_rgb_image(out_dir / "start_image.png", start_image)
    save_rgb_image(out_dir / "goal_image.png", goal_image)

    goal_emb = encode_single_frame(model, goal_image, device, img_size, pixel_mean, pixel_std)
    goal_state = torch.cat([goal_emb, torch.zeros_like(goal_emb)], dim=-1).cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    oracle_pbar = tqdm(range(cfg.max_oracle_steps), desc="Oracle Reaching Phase")
    grasped = False

    # Initialize geometry logic tracking bounds
    oracle.reset(None, current_info)

    # --- Stage 1: Execute Oracle Tracking Loop until contact registers ---
    for _ in oracle_pbar:
        grasped = cube_is_grasped(current_info, cfg.grasp_contact_threshold, cfg.grasp_alignment_threshold)
        if grasped: break
        
        raw_action = np.asarray(oracle.select_action(None, current_info), dtype=np.float32)
        _, _, _, _, current_info = env.step(raw_action)
        current_frame = render_without_target_cube(env, camera)
        rollout_frames.append(current_frame.copy())

    if not grasped:
        print("[ABORT] Oracle failed to isolate safe initial alignment grasp margins.")
        env.close()
        return

    # --- Stage 2: Conformal SLS Latent Space Receding Horizon Loop ---
    print("\n[HANDOFF] Handoff conditions met. Initializing SLS Engine Core...")
    # W_stage = jnp.ones((state_dim,)) * 100
    # W_stage = W_stage.at[:state_dim // 2].set(1.0)
    # W_term = jnp.ones((state_dim,))*0.1
    # W_term = W_term.at[:state_dim // 2].set(100.0)
    W_stage = jnp.ones((state_dim,)) * cfg.q_stage
    # W_stage = W_stage.at[:state_dim // 2].set(1.0)
    W_term = jnp.ones((state_dim,))*cfg.q_terminal
    # W_term = W_term.at[:state_dim // 2].set(100.0)
    action_weights = make_action_weights(action_dim, cfg.r_control, cfg.r_control_u4)
    cost = make_tracking_cost(
        action_weights,
        cfg.horizon,
        W_term,
        jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )

    
    min_state, max_state = -5.0*jnp.ones(state_dim), 5.0*jnp.ones(state_dim)
    min_state, max_state = min_state.at[:state_dim // 2].set(-3.0), max_state.at[:state_dim // 2].set(3.0)
    state_box = make_state_box_constraints(min_state, max_state)

    constraints_all = combine_constraints(state_box,
        make_control_box_constraints(-2.0*jnp.ones(action_dim), 2.0*jnp.ones(action_dim)),
        *(() if obstacle_constraint is None else (obstacle_constraint,)),
    )

    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=True, initialize_nominal=True, warm_start=False, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=1200, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=jnp.zeros(action_dim), dt=1.0/20.0),
        dynamics=dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim + (1 if obstacle_constraint is not None else 0), disturbance=disturbance, shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim), dtype=jnp.float64), U_in=jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    )

    current_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    if obstacle_model is not None:
        start_score = float(obstacle_model(jnp.asarray(current_state)))
        goal_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        if start_score <= required_score or goal_score <= required_score:
            print(
                "Terminating: start and goal must both be outside the conformal obstacle set. "
                f"Required score > {required_score:.6g}; start_score={start_score:.6g}, goal_score={goal_score:.6g}."
            )
            env.close()
            return
        print(
            "Obstacle sanity check passed: "
            f"start_score={start_score:.6g}, goal_score={goal_score:.6g}, required_score>{required_score:.6g}"
        )
    X_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_u0 = np.zeros(action_dim, dtype=np.float32)
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="SLS Local Track Core Loop")
    for step_idx in mpc_pbar:
        sanitize_controller_warm_start(
            controller,
            current_state,
            obstacle_model=obstacle_model,
            obstacle_margin=cfg.obstacle_margin,
            max_abs=float(cfg.max_normalized_action),
            debug=cfg.debug_solver,
        )
        if cfg.debug_solver and obstacle_model is not None:
            print_obstacle_debug("current", obstacle_model, current_state[None, :], cfg.obstacle_margin)
            print_obstacle_debug("warm_start", obstacle_model, np.asarray(controller.X0), cfg.obstacle_margin)
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_ref, parameter=1.0/20.0)
            status = "sls_mpc"
        except Exception as e:
            import traceback
            traceback.print_exc()
            u0, X_pred = None, None
            status = "exception"

        solver_output_valid = False
        if u0 is not None and X_pred is not None and U_pred is not None:
            u0_np = np.asarray(u0, dtype=np.float64).reshape(-1)
            X_pred_np = np.asarray(X_pred, dtype=np.float64)
            U_pred_np = np.asarray(U_pred, dtype=np.float64)
            solver_output_valid = bool(
                np.all(np.isfinite(u0_np))
                and np.all(np.isfinite(X_pred_np))
                and np.all(np.isfinite(U_pred_np))
                and np.max(np.abs(u0_np)) <= float(cfg.max_normalized_action)
                and np.max(np.abs(U_pred_np)) <= float(cfg.max_normalized_action)
            )
            if cfg.debug_solver:
                print(
                    f"[solver-debug:{step_idx}] "
                    f"u0_abs_max={np.max(np.abs(u0_np)):.6g} "
                    f"U_abs_max={np.max(np.abs(U_pred_np)):.6g} "
                    f"X_abs_max={np.max(np.abs(X_pred_np)):.6g} "
                    f"finite={np.all(np.isfinite(u0_np)) and np.all(np.isfinite(X_pred_np)) and np.all(np.isfinite(U_pred_np))} "
                    f"valid={solver_output_valid}"
                )
                if obstacle_model is not None:
                    print_obstacle_debug("pred", obstacle_model, X_pred_np, cfg.obstacle_margin)

        if not solver_output_valid:
            u0 = prev_u0
            status = "frozen_fallback"
            reset_controller_primal_warm_start(controller, current_state)
            controller.w = jnp.zeros_like(controller.w)
            controller.y = jnp.zeros_like(controller.y)
            controller.rho = jnp.asarray(controller.admm_config.initial_rho, dtype=controller.rho.dtype)
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)

        u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        _, _, _, _, current_info = env.step(u_raw)
        reached_ogbench_success = ogbench_success(current_info)
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u_raw.astype(np.float32))
        
        current_frame = render_without_target_cube(env, camera)
        rollout_frames.append(current_frame.copy())

        next_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        lat_err = float(np.linalg.norm(current_state - goal_state))
        if reached_ogbench_success:
            status = "ogbench_success"
        obs_free = "n/a"
        if obstacle_model is not None:
            obstacle_score = float(obstacle_model(jnp.asarray(current_state)))
            obs_free = obstacle_score > float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        mpc_pbar.set_postfix(latent_err=f"{lat_err:.4f}", obs_free=obs_free, status=status)
        if cfg.terminate_on_ogbench_success and reached_ogbench_success: break
        if lat_err <= 0.05: break

    imageio.mimwrite(out_dir / "cube_sls_rollout.mp4", rollout_frames, fps=cfg.video_fps)
    np.savez(
        out_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    env.close()

if __name__ == "__main__":
    main()
