#!/usr/bin/env python3
"""Plan in OGBench space using Conformal SLS MPC warmstarted by MPPI over an Equinox-wrapped world model."""

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
import numpy as np
import torch
import mujoco
from tqdm.auto import tqdm
import pyrallis

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config, lax
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.mppi_planner import MPPIPlanner

import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie
from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiCubeConfig:
    """Configuration for Warmstarted Conformal SLS MPC on OGBench Cubes"""
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
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mppi_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=16)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    visualize_success_colors: bool = field(default=False)
    terminate_on_ogbench_success: bool = field(default=True)

    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 30.0
    mppi_noise_level: float = 0.25
    mppi_beta_filter: float = 0.6
    mppi_q_stage: float = 100.0
    mppi_q_terminal: float = 100.0
    mppi_state_box_penalty: float = 0.0
    mppi_r_control: float = 0.01

    grasp_contact_threshold: float = 0.5
    grasp_alignment_threshold: float = 0.03
    q_stage: float = field(default=10.0)
    q_terminal: float = field(default=100.0)
    r_control: float = field(default=0.01)

# --- Layer Weight Ingestion ---
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

def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array) -> eqx.Module:
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    layers = []
    keys = jax.random.split(key, len(pt_linears))
    for i, pt_layer in enumerate(pt_linears):
        out_f, in_f = pt_layer.weight.shape
        eqx_linear = eqx.nn.Linear(in_f, out_f, key=keys[i])
        w, b = jnp.array(pt_layer.weight.detach().cpu().numpy()), jnp.array(pt_layer.bias.detach().cpu().numpy())
        layers.append(eqx.tree_at(lambda l: (l.weight, l.bias), eqx_linear, (w, b)))
        if i < len(pt_linears) - 1: layers.append(jax.nn.gelu)

    class JAXMLP(eqx.Module):
        layers: list
        def __call__(self, x):
            for layer in self.layers: x = layer(x)
            return x
    return JAXMLP(layers)

def make_jax_disturbance(eqx_error_model, q_learned, state_dim, diagonal):
    def _mgnll_forward(raw):
        if diagonal: return jnp.diag(jnp.exp(raw) + 1e-4)
        L = jnp.zeros((state_dim, state_dim))
        L = L.at[jnp.tril_indices(state_dim)].set(raw)
        return L.at[jnp.arange(state_dim), jnp.arange(state_dim)].set(jnp.exp(jnp.diag(L)) + 1e-4)
    return lambda X, U: q_learned * jax.vmap(_mgnll_forward)(jax.vmap(eqx_error_model)(jnp.concatenate([X, U], axis=-1)))

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

def make_mppi_rollout_and_eval(
    dynamics_fn,
    W_mppi_stage,
    W_mppi_terminal,
    goal_state,
    *,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
    box_min: jnp.ndarray | None = None,
    box_max: jnp.ndarray | None = None,
    box_penalty_weight: float = 0.0,
    r_control: float = 0.01,
    action_ref: jnp.ndarray | None = None,
):
    if box_min is not None:
        box_min = jnp.asarray(box_min, dtype=jnp.float64)
    if box_max is not None:
        box_max = jnp.asarray(box_max, dtype=jnp.float64)
    if action_ref is None:
        raise ValueError("action_ref must be provided for MPPI action regularization.")
    action_ref = jnp.asarray(action_ref, dtype=jnp.float64)

    def rollout(state_cur, act_seqs, reach_config=None):
        def step_fn(s, u):
            nxt = dynamics_fn(s, u)
            return nxt, nxt
        return jax.vmap(lambda actions: lax.scan(step_fn, state_cur, actions)[1])(act_seqs), {}

    def eval_fn(states, acts, reach_config=None, aux=None, *args, **kwargs):
        delta = states - goal_state[None, None, :]
        stage_costs = jnp.sum(W_mppi_stage[None, None, :] * delta**2, axis=-1)
        terminal_costs = jnp.sum(W_mppi_terminal[None, :] * delta[:, -1, :] ** 2, axis=-1)
        action_delta = acts - action_ref[None, None, :]
        action_costs = r_control * jnp.sum(action_delta ** 2, axis=-1)
        if box_min is not None and box_max is not None and box_penalty_weight > 0.0:
            lower_violation = jnp.maximum(box_min[None, None, :] - states, 0.0)
            upper_violation = jnp.maximum(states - box_max[None, None, :], 0.0)
            box_costs = box_penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2, axis=-1)
        else:
            box_costs = jnp.zeros_like(stage_costs)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            flat_states = states.reshape((-1, states.shape[-1]))
            obstacle_violation = jax.vmap(
                lambda z: jax.nn.softplus(obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z))
            )(flat_states).reshape(states.shape[:-1])
            obstacle_costs = obstacle_penalty_weight * obstacle_violation**2
        else:
            obstacle_costs = jnp.zeros_like(stage_costs)
        total_cost = jnp.sum(stage_costs + action_costs + box_costs + obstacle_costs, axis=-1) + terminal_costs
        return {"rewards": -total_cost}
    return rollout, eval_fn

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    return lambda x, u, t: jnp.concatenate([x - x_max, x_min - x], axis=0)

def combine_constraints(*constraints):
    return lambda x, u, t: jnp.concatenate([constraint(x, u, t) for constraint in constraints], axis=0)

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint

def make_tracking_cost(
    action_weight: float,
    horizon: int,
    W_stage: jnp.ndarray,
    W_terminal: jnp.ndarray,
    goal_state: jnp.ndarray,
    action_ref: jnp.ndarray,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    action_ref = jnp.asarray(action_ref, dtype=jnp.float64)
    def cost(W_ignored, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W_stage, W_terminal)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state)
        state_error = z - active_ref
        action_error = u - action_ref
        total_cost = jnp.sum(active_W * (state_error ** 2)) + action_weight * jnp.sum(action_error ** 2)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_violation = jax.nn.softplus(
                obstacle_model.threshold + float(obstacle_margin) - obstacle_model(z)
            )
            total_cost = total_cost + obstacle_penalty_weight * obstacle_violation**2
        return total_cost
    return cost

def make_action_reference(action_dim: int, u_min: jnp.ndarray, u_max: jnp.ndarray, grip_idx: int = 4) -> jnp.ndarray:
    action_ref = jnp.zeros((action_dim,), dtype=jnp.float64)
    if not 0 <= grip_idx < action_dim:
        return action_ref
    grip_midpoint = 0.5 * (u_min[grip_idx] + u_max[grip_idx])
    return action_ref.at[grip_idx].set(grip_midpoint)

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)

def resolve_action_stats_dataset_path(cfg: PlanSLSMoppiCubeConfig) -> Path:
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

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch[_=](\d+).*\.ckpt$")
    candidates = []
    for path in model_dir.glob("*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No valid checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

# --- Target Cube Sync & Hiding Utilities ---
def hide_target_cube(env: gymnasium.Env) -> None:
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

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

def render_without_target_cube(env: gymnasium.Env, camera: str) -> np.ndarray:
    hide_target_cube(env)
    return np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)

def reset_env_to_state(env: gymnasium.Env, seed: int, qpos: np.ndarray, qvel: np.ndarray, target_block_pos: np.ndarray, target_block_yaw: float, camera: str) -> tuple[np.ndarray, dict]:
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(env, target_block_pos=target_block_pos, target_block_yaw=target_block_yaw)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    frame = render_without_target_cube(env, camera)
    info = unwrapped.get_step_info()
    return frame, info

# --- Vision Frame Encoding Utilities ---
@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(pixel_np.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean.to(tensor.device)) / pixel_std.to(tensor.device)
    output = model.encoder(tensor.to(device), interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - pixel_mean.to(tensor.device)) / pixel_std.to(tensor.device)

    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

def main():
    cfg = pyrallis.parse(config_class=PlanSLSMoppiCubeConfig)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve() / f"{int(time.time())}_mppi_sls_cube"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json") as f: config_dict = json.load(f)

    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()

    state_dim, action_dim = config_dict.get("markov_state_dim", 48), config_dict.get("action_dim", 5)
    img_size = config_dict.get("img_size", 224)

    init_key = jax.random.PRNGKey(cfg.seed)
    k1, k2, k3 = jax.random.split(init_key, 3)
    eqx_dyn = build_equinox_mlp_from_pytorch(model.predictor.net, k1)
    dynamics = lambda x, u, t=0.0, parameter=1.0: eqx_dyn(jnp.concatenate([x, u], axis=-1))
    obstacle_model = None
    obstacle_constraint = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, k3)
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
        disturbance = make_jax_disturbance(build_equinox_mlp_from_pytorch(error_model.net, k2), cfg.q_learned, state_dim, error_model.diagonal)

    episode, episode_idx = load_planning_episode(cfg.dataset_path, cfg.episode_idx, cfg.seed)
    qpos_init = episode["qpos_init"]
    qvel_init = episode["qvel_init"]
    qpos_goal = episode["qpos_goal"]
    qvel_goal = episode["qvel_goal"]
    target_block_pos_init = episode["target_block_pos_init"]
    target_block_yaw_init = float(episode["target_block_yaw_init"])
    target_block_pos_goal = episode["target_block_pos_goal"]
    target_block_yaw_goal = float(episode["target_block_yaw_goal"])
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])

    env = gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=cfg.visualize_success_colors,
        width=256,
        height=256,
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

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

    save_rgb_image(out_dir / "start_image.png", current_frame)
    save_rgb_image(out_dir / "goal_image.png", goal_frame)

    pixel_mean, pixel_std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    goal_emb = encode_single_frame(model, goal_frame, device, img_size, pixel_mean, pixel_std)
    goal_state = torch.cat([goal_emb, torch.zeros_like(goal_emb)], dim=-1).cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    grasped = False

    oracle.reset(None, current_info)

    # Run analytical planner tracking stages until containment validation triggers
    for _ in range(cfg.max_oracle_steps):
        if cube_is_grasped(current_info, cfg.grasp_contact_threshold, cfg.grasp_alignment_threshold): grasped = True; break
        current_info = env.step(np.asarray(oracle.select_action(None, current_info), dtype=np.float32))[4]
        rollout_frames.append(render_without_target_cube(env, camera))

    if not grasped: return env.close()

    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    train_stats = LeWMOGBenchCubeDataset(
        str(action_stats_dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim
    )
    action_mean = train_stats.action_mean.astype(np.float32)
    action_std = train_stats.action_std.astype(np.float32)
    print(f"Using action statistics from {action_stats_dataset_path}")

    W_mppi_stage = jnp.ones((state_dim,)) * cfg.mppi_q_stage
    W_mppi_stage = W_mppi_stage.at[state_dim // 2:].set(1.0)
    W_mppi_terminal = jnp.ones((state_dim,)) * cfg.mppi_q_terminal
    W_mppi_terminal = W_mppi_terminal.at[state_dim // 2:].set(1.0)
    W_stage_scaled = jnp.ones((state_dim,)) * cfg.q_stage
    W_stage_scaled = W_stage_scaled.at[state_dim // 2:].set(1.0)
    W_terminal_scaled = jnp.ones((state_dim,)) * cfg.q_terminal
    W_terminal_scaled = W_terminal_scaled.at[state_dim // 2:].set(1.0)

    x_min, x_max = -100.0 * jnp.ones(state_dim), 100.0 * jnp.ones(state_dim)
    u_min, u_max = -2.0 * jnp.ones(action_dim), 2.0 * jnp.ones(action_dim)
    u_max = u_max.at[4].set(-2.0)
    u_min = u_min.at[4].set(-10.0)
    action_ref = make_action_reference(action_dim, u_min, u_max)
    mppi_roll, mppi_ev = make_mppi_rollout_and_eval(
        dynamics,
        W_mppi_stage,
        W_mppi_terminal,
        jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
        box_min=x_min,
        box_max=x_max,
        box_penalty_weight=cfg.mppi_state_box_penalty,
        r_control=cfg.mppi_r_control,
        action_ref=action_ref,
    )

    mppi_planner = MPPIPlanner(
        config={"planning": {"action_dim": action_dim, "n_sample": cfg.mppi_samples, "horizon": cfg.horizon, "n_update_iter": cfg.mppi_update_iter, "use_last": True, "reject_bad": False, "mppi": {"reward_weight": cfg.mppi_reward_weight, "noise_level": cfg.mppi_noise_level, "noise_decay": 1.0, "beta_filter": cfg.mppi_beta_filter}}},
        model_rollout_fn=mppi_roll, evaluate_traj_fn=mppi_ev, action_lower_lim=u_min, action_upper_lim=u_max
    )

    @eqx.filter_jit
    def run_mppi_opt(key_arg, state_arg, actions_arg):
        return mppi_planner.trajectory_optimization(key_arg, state_arg, actions_arg, skip=False)

    cost = make_tracking_cost(
        action_weight=cfg.r_control,
        horizon=cfg.horizon,
        W_stage=W_stage_scaled,
        W_terminal=W_terminal_scaled,
        goal_state=jnp.asarray(goal_state),
        action_ref=action_ref,
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )

    constraints_all = combine_constraints(
        make_state_box_constraints(x_min, x_max),
        make_control_box_constraints(u_min, u_max),
        *(() if obstacle_constraint is None else (obstacle_constraint,)),
    )

    u_init = jnp.zeros((cfg.horizon, action_dim))
    u_init = u_init.at[:, 4].set(action_ref[4])

    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=False, initialize_nominal=True, warm_start=True, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=300, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage_scaled, u_ref=action_ref, dt=1.0/20.0),
        dynamics=dynamics, constraints=constraints_all, obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim + (1 if obstacle_constraint is not None else 0),
        disturbance=disturbance, shift=1, X_in=jnp.zeros((cfg.horizon + 1, state_dim)), U_in=u_init
    )

    current_emb = encode_single_frame(model, rollout_frames[-1], device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    prev_U = jnp.zeros((cfg.horizon, action_dim))
    prev_U = prev_U.at[:, 4].set(action_ref[4])
    jax_seed = jax.random.PRNGKey(cfg.seed)

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

    prev_u0 = np.zeros(action_dim, dtype=np.float32)
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="Refined MPPI + SLS tracking loops")
    for _ in mpc_pbar:
        jax_seed, subkey = jax.random.split(jax_seed)
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)

        mppi_ok = False
        X_ws = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
        U_ws = init_act_seq
        try:
            mppi_res = run_mppi_opt(subkey, jnp.asarray(current_state), init_act_seq)
            X_mppi = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
            U_mppi = jnp.asarray(mppi_res["act_seq"])
            if np.all(np.isfinite(np.asarray(X_mppi))) and np.all(np.isfinite(np.asarray(U_mppi))):
                X_ws = X_mppi
                U_ws = U_mppi
                mppi_ok = True
        except Exception:
            pass

        controller.X_in, controller.U_in = X_ws, U_ws

        try:
            u0, X_pred, U_pred, *_ = controller.run(x0=current_state, reference=X_ws, parameter=1.0/20.0)
            status = "sls_refined" if mppi_ok else "sls_mpc"
        except Exception:
            u0, X_pred, U_pred = None, None, None
            status = "exception_fallback"

        if u0 is None or X_pred is None or U_pred is None:
            u0, X_pred, U_pred = None, None, None
        elif not (
            np.all(np.isfinite(np.asarray(u0)))
            and np.all(np.isfinite(np.asarray(X_pred)))
            and np.all(np.isfinite(np.asarray(U_pred)))
        ):
            u0, X_pred, U_pred = None, None, None
            status = "nonfinite_fallback"

        if u0 is None:
            u0 = prev_u0
            status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            prev_U = U_pred

        u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        current_info = env.step(u_raw)[4]
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u_raw.astype(np.float32))
        reached_ogbench_success = ogbench_success(current_info)

        frame = render_without_target_cube(env, camera)
        rollout_frames.append(frame)

        next_emb = encode_single_frame(model, frame, device, img_size, pixel_mean, pixel_std)
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

    imageio.mimwrite(out_dir / "cube_mppi_sls.mp4", rollout_frames, fps=cfg.video_fps)
    np.savez(
        out_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    env.close()

if __name__ == "__main__":
    main()
