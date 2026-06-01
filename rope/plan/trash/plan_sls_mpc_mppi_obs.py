#!/usr/bin/env python3
"""Plan in Rope pixel space using Conformal SLS MPC warmstarted by MPPI with conformal obstacle avoidance."""

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
from jax import config, lax
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints
from gpu_sls.mppi_planner import MPPIPlanner

from rope.train.mlpdyn_train import LeWMRopeDataset, preprocess_pixels
from rope.shared.lab_env import LabEnv, TaskState
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiRopeConfig:
    """Configuration for Warmstarted Conformal SLS MPC on Rope Lines with obstacle avoidance."""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("rope/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("rope/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("rope/eval/fixed_error_covariance.pt"))
    enable_obstacle: bool = True
    obstacle_model_path: Path = field(default=Path("rope/models/obs_net/da270d7d1050f110/model.pt"))
    obstacle_margin: float = 0.0
    obstacle_penalty_weight: float = 1000.0
    dataset_path: Path = field(default=Path("rope/data/expert_data/rope_random_cubic_spline.h5"))
    action_stats_dataset_path: Path = field(default=Path("rope/data/test_data_noshadow/rope_random_cubic_spline.h5"))
    out_dir: Path = field(default=Path("rope/plan/sls_mppi_conformal_obs"))
    device: str = field(default="auto")
    horizon: int = field(default=24)
    max_mpc_steps: int = field(default=150)
    video_fps: int = field(default=30)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    
    mppi_samples: int = 512
    mppi_update_iter: int = 6
    mppi_reward_weight: float = 25.0
    mppi_noise_level: float = 0.2
    mppi_beta_filter: float = 0.65
    mppi_state_box_penalty: float = 1000.0

# --- PyTorch weight ingestion to Equinox Objects ---

def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array, activation=jax.nn.gelu) -> eqx.Module:
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    layers = []
    keys = jax.random.split(key, len(pt_linears))
    for i, pt_layer in enumerate(pt_linears):
        out_features, in_features = pt_layer.weight.shape
        eqx_linear = eqx.nn.Linear(in_features, out_features, key=keys[i])
        w = jnp.array(pt_layer.weight.detach().cpu().numpy())
        b = jnp.array(pt_layer.bias.detach().cpu().numpy()) if pt_layer.bias is not None else jnp.zeros(out_features)
        eqx_linear = eqx.tree_at(lambda l: (l.weight, l.bias), eqx_linear, (w, b))
        layers.append(eqx_linear)
        if i < len(pt_linears) - 1:
            layers.append(activation)
            
    class JAXMLP(eqx.Module):
        layers: list
        def __call__(self, x):
            for layer in self.layers: x = layer(x)
            return x
    return JAXMLP(layers)

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
    if not artifact_path.expanduser().is_file():
        raise FileNotFoundError(f"Obstacle model artifact not found: {artifact_path}")
    artifact = torch.load(artifact_path.expanduser(), map_location="cpu", weights_only=False)
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

def make_obstacle_constraint(obstacle_model: JAXObstacleMLP, margin: float):
    def constraint(x, u, t):
        return jnp.asarray([obstacle_model.threshold + float(margin) - obstacle_model(x)])
    return constraint

def make_jax_dynamics(eqx_dyn_model):
    def jax_dynamics(x, u, t=0.0, parameter=1.0):
        return eqx_dyn_model(jnp.concatenate([x, u], axis=-1))
    return jax_dynamics

def make_jax_disturbance(eqx_error_model, q_learned, state_dim, diagonal):
    def _mgnll_forward(raw):
        if diagonal: return jnp.diag(jnp.exp(raw) + 1e-4)
        L = jnp.zeros((state_dim, state_dim))
        L = L.at[jnp.tril_indices(state_dim)].set(raw)
        diag_idx = jnp.arange(state_dim)
        return L.at[diag_idx, diag_idx].set(jnp.exp(L[diag_idx, diag_idx]) + 1e-4)

    def dist_fn(X_seq, U_seq):
        inp = jnp.concatenate([X_seq, U_seq], axis=-1)
        raw_preds = jax.vmap(eqx_error_model)(inp)
        return q_learned * jax.vmap(_mgnll_forward)(raw_preds)
    return dist_fn

def make_constant_jax_disturbance(calibrated_cholesky: np.ndarray, state_dim: int):
    calibrated_cholesky = jnp.asarray(calibrated_cholesky, dtype=jnp.float64)
    if calibrated_cholesky.shape != (state_dim, state_dim):
        raise ValueError(
            f"Expected calibrated Cholesky shape {(state_dim, state_dim)}, got {calibrated_cholesky.shape}."
        )

    def dist_fn(X_seq, U_seq):
        seq_len = X_seq.shape[0]
        matrices = jnp.broadcast_to(calibrated_cholesky, (seq_len, state_dim, state_dim))
        return matrices
        # active = jnp.arange(seq_len) < ((seq_len + 1) // 2)
        # return matrices * active[:, None, None]

    return dist_fn

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
    return np.asarray(matrix.detach().cpu().numpy(), dtype=np.float64)

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

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float64)

def render_rgb_frame(
    renderer: mujoco.Renderer,
    env: LabEnv,
    camera_id: int,
    *,
    disable_shadows: bool,
) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()

def extract_step_info(env: LabEnv, *, elapsed_time: float) -> dict[str, np.ndarray]:
    return {
        "task_target": env.task_controller.desired_state.as_array().astype(np.float32),
        "qpos": env.data.qpos.copy().astype(np.float32),
        "qvel": env.data.qvel.copy().astype(np.float32),
        "control": env.data.ctrl.copy().astype(np.float32),
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }

def reset_env_to_state(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
    camera_id: int,
    elapsed_time: float,
    disable_shadows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)
    return frame, extract_step_info(env, elapsed_time=elapsed_time)

def step_env_with_action(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    action: np.ndarray,
    control_decimation: int,
    camera_id: int,
    elapsed_time: float,
    disable_shadows: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.apply_task_delta(np.asarray(action, dtype=np.float64))
    env.step(int(control_decimation))
    frame = render_rgb_frame(renderer, env, camera_id, disable_shadows=disable_shadows)
    return frame, extract_step_info(env, elapsed_time=elapsed_time)

def make_mppi_rollout_and_eval(
    jax_dynamics_fn,
    state_dim,
    action_dim,
    horizon,
    W_state,
    goal_state,
    action_ref,
    box_min=None,
    box_max=None,
    box_penalty_weight: float = 0.0,
    obstacle_model: JAXObstacleMLP | None = None,
    obstacle_margin: float = 0.0,
    obstacle_penalty_weight: float = 0.0,
):
    if box_min is not None:
        box_min = jnp.asarray(box_min)
    if box_max is not None:
        box_max = jnp.asarray(box_max)
    action_ref = jnp.asarray(action_ref, dtype=jnp.float64)

    def mppi_rollout_fn(state_cur, act_seqs, reach_config=None):
        def single_sample_rollout(actions):
            def step(state, u):
                next_state = jax_dynamics_fn(state, u, 0.0, 1.0)
                return next_state, next_state
            _, states = lax.scan(step, state_cur, actions)
            return states
        return jax.vmap(single_sample_rollout)(act_seqs), {}

    def mppi_eval_fn(state_seqs, act_seqs, reach_config=None, aux=None, *args, **kwargs):
        delta = state_seqs - goal_state[None, None, :]
        stage_costs = jnp.sum(W_state[None, None, :] * (delta ** 2), axis=-1)
        action_costs = 1.0 * jnp.sum((act_seqs - action_ref[None, None, :]) ** 2, axis=-1)
        if box_min is not None and box_max is not None and box_penalty_weight > 0.0:
            lower_violation = jnp.maximum(box_min[None, None, :] - state_seqs, 0.0)
            upper_violation = jnp.maximum(state_seqs - box_max[None, None, :], 0.0)
            box_costs = box_penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2, axis=-1)
        else:
            box_costs = jnp.zeros_like(stage_costs)
        if obstacle_model is not None and obstacle_penalty_weight > 0.0:
            obstacle_scores = jax.vmap(jax.vmap(obstacle_model))(state_seqs)
            obstacle_violation = jnp.maximum(
                obstacle_model.threshold + float(obstacle_margin) - obstacle_scores,
                0.0,
            )
            obstacle_costs = obstacle_penalty_weight * obstacle_violation**2
        else:
            obstacle_costs = jnp.zeros_like(stage_costs)
        return {"rewards": -jnp.sum(stage_costs + action_costs + box_costs + obstacle_costs, axis=-1)}

    return mppi_rollout_fn, mppi_eval_fn

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    def constraints(x, u, t):
        # Maps boundaries to: [x - x_max <= 0, x_min - x <= 0]
        return jnp.concatenate([x - x_max, x_min - x], axis=0)
    return constraints

def make_tracking_cost(action_weight: float, horizon: int, W_term: jnp.ndarray, goal_state: jnp.ndarray, action_ref: jnp.ndarray):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        dz = z - jnp.where(is_not_terminal, reference[t], goal_state)
        return jnp.sum(jnp.where(is_not_terminal, W, W_term) * dz**2) + action_weight * jnp.sum((u - action_ref)**2)
    return cost

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = [ (int(m.group(1)), p) for p in model_dir.glob("*_epoch_*_object.ckpt") for m in [pattern.match(p.name)] if m ]
    if not candidates: raise FileNotFoundError(f"No object checkpoints in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

# @torch.no_grad()
# def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
#     tensor = preprocess_pixels(torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous(), img_size).to(device)
#     latents = []
#     for start in range(0, tensor.shape[0], 32):
#         latents.append(model.projector(model.encoder(tensor[start : start + 32], interpolate_pos_encoding=True).last_hidden_state[:, 0]))
#     return torch.cat(latents, dim=0)

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    # 1. Convert to torch and match channels-first format (B, C, H, W)
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous()
    
    # 2. Preprocess using rope's pipeline
    tensor = preprocess_pixels(tensor, img_size).to(device)
    
    # CRITICAL FIX: If your preprocessing or array pipeline wraps it in an extra batch dimension
    # resulting in (1, B, C, H, W), squeeze it down to 4D (B, C, H, W).
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0) # Drops the redundant leading dimension

    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32]
        
        # Ensure chunk is exactly 4D before passing it to ViT
        if chunk.ndim == 5:
            chunk = chunk.squeeze(0)
            
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
        
    return torch.cat(latents, dim=0)

def main():
    cfg = pyrallis.parse(config_class=PlanSLSMoppiRopeConfig)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r") as f: config_dict = json.load(f)
    
    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()
    
    state_dim = config_dict.get("markov_state_dim", 36)
    action_dim = config_dict.get("action_dim", 5) # 5D gripper actions
    img_size = config_dict.get("img_size", 224)
    action_mean_raw, action_std_raw = load_action_stats_from_dataset(cfg.action_stats_dataset_path, action_dim)
    average_action = np.zeros((action_dim,), dtype=np.float64)
    print(
        f"Using large-dataset action statistics from {cfg.action_stats_dataset_path}: "
        f"raw_mean={action_mean_raw.tolist()}, raw_std={action_std_raw.tolist()}, "
        f"normalized_average={average_action.tolist()}"
    )

    init_key = jax.random.PRNGKey(cfg.seed)
    key_dyn, key_err, key_obs = jax.random.split(init_key, 3)
    dynamics = make_jax_dynamics(build_equinox_mlp_from_pytorch(model.predictor.net, key_dyn))
    obstacle_model = None
    obstacle_constraint = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, key_obs)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(
                f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}."
            )
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {cfg.obstacle_model_path} "
            f"with threshold {float(obstacle_model.threshold):.6g} and margin {cfg.obstacle_margin:.6g}"
        )
    else:
        print("Obstacle avoidance disabled; skipping obstacle classifier load and start/goal obstacle sanity check.")
    if cfg.use_constant_covariance:
        calibrated_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky, state_dim)
        print(f"Using fixed calibrated covariance disturbance from {cfg.constant_covariance_path}")
    else:
        error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = make_jax_disturbance(
            build_equinox_mlp_from_pytorch(error_model.net, key_err),
            cfg.q_learned,
            state_dim,
            error_model.diagonal,
        )

    with h5py.File(cfg.dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(np.random.choice(np.flatnonzero(ep_len >= cfg.horizon)))
        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        task_target_np = np.asarray(h5["task_target"][rows], dtype=np.float32)
        qpos_np = np.asarray(h5["qpos"][rows], dtype=np.float32)
        qvel_np = np.asarray(h5["qvel"][rows], dtype=np.float32)
        control_np = np.asarray(h5["control"][rows], dtype=np.float32)
        time_np = np.asarray(h5["time"][rows], dtype=np.float32) if "time" in h5 else np.zeros((len(rows), 1), dtype=np.float32)
        camera_name = str(h5.attrs.get("camera", "video_cam"))
        control_decimation = int(h5.attrs.get("control_decimation", 25))
        disable_shadows = bool(h5.attrs.get("disable_shadows", True))
        control_timestep = float(h5.attrs.get("control_timestep", 1.0 / 30.0))

    true_latents = encode_frames(model, pixels_np, device, img_size)
    goal_state = torch.cat([true_latents[-1], true_latents[-1] - true_latents[-2]], dim=-1).cpu().numpy().astype(np.float64)
    start_state = torch.cat([true_latents[0], torch.zeros_like(true_latents[0])], dim=-1).cpu().numpy().astype(np.float64)

    if obstacle_model is not None:
        start_obstacle_score = float(obstacle_model(jnp.asarray(start_state)))
        goal_obstacle_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_obstacle_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        if start_obstacle_score <= required_obstacle_score or goal_obstacle_score <= required_obstacle_score:
            print(
                "Terminating: start and goal must both be outside the conformal obstacle set. "
                f"Required score > {required_obstacle_score:.6g}; "
                f"start_score={start_obstacle_score:.6g}, goal_score={goal_obstacle_score:.6g}."
            )
            sys.exit(1)
        print(
            "Obstacle sanity check passed: "
            f"start_score={start_obstacle_score:.6g}, goal_score={goal_obstacle_score:.6g}, "
            f"required_score>{required_obstacle_score:.6g}"
        )

    run_dir = out_dir / f"{int(time.time())}_mppi_sls_rope_obs_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    W_mppi = jnp.ones((state_dim,)) * 100
    W_mppi = W_mppi.at[state_dim // 2:].set(1.0)
    W_stage = jnp.ones((state_dim,)) * 10.0
    W_stage = W_stage.at[state_dim // 2:].set(1.0)
    W_term = jnp.ones((state_dim,)) * 10.0
    W_term = W_term.at[state_dim // 2:].set(1.0)

    action_ref = jnp.asarray(average_action, dtype=jnp.float64)
    cost = make_tracking_cost(1.0, cfg.horizon, W_term, jnp.asarray(goal_state), action_ref)

    box_min, box_max = -3.0 * jnp.ones(state_dim), 3.0 * jnp.ones(state_dim)
    # second half (velocity) should stay around +/- 0.5
    box_min = box_min.at[state_dim // 2:].set(-0.5)
    box_max = box_max.at[state_dim // 2:].set(0.5)

    mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
        dynamics,
        state_dim,
        action_dim,
        cfg.horizon,
        W_mppi,
        jnp.asarray(goal_state),
        action_ref,
        box_min=box_min,
        box_max=box_max,
        box_penalty_weight=cfg.mppi_state_box_penalty,
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=cfg.obstacle_penalty_weight,
    )

    mppi_planner = MPPIPlanner(
        config={"planning": {"action_dim": action_dim, "n_sample": cfg.mppi_samples, "horizon": cfg.horizon, "n_update_iter": cfg.mppi_update_iter, "use_last": True, "reject_bad": False, "mppi": {"reward_weight": cfg.mppi_reward_weight, "noise_level": cfg.mppi_noise_level, "noise_decay": 1.0, "beta_filter": cfg.mppi_beta_filter}}},
        model_rollout_fn=mppi_rollout,
        evaluate_traj_fn=mppi_eval,
        action_lower_lim=-2.0 * jnp.ones(action_dim), action_upper_lim=2.0 * jnp.ones(action_dim)
    )
    jit_mppi_trajopt = jax.jit(lambda k, s, a: mppi_planner.trajectory_optimization(k, s, a, skip=False))

    # SLS Setup footprint
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=True, initialize_nominal=True, warm_start=False, rti=False)
    controller = GenericMPC(
        sls_cfg, SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=True),
        ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e4, max_iterations=400, rho_update_frequency=20, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=action_ref, dt=1.0/30.0),
        dynamics=dynamics,
        constraints=(
            combine_constraints(
                make_state_box_constraints(box_min, box_max),
                obstacle_constraint,
                make_control_box_constraints(-5.0*jnp.ones(action_dim), 5.0*jnp.ones(action_dim)),
            )
            if obstacle_constraint is not None
            else combine_constraints(
                make_state_box_constraints(box_min, box_max),
                make_control_box_constraints(-5.0*jnp.ones(action_dim), 5.0*jnp.ones(action_dim)),
            )
        ),
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim + (1 if obstacle_constraint is not None else 0), disturbance=disturbance, shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim), dtype=jnp.float64), U_in=jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    )

    env = LabEnv()
    camera_id = env.model.camera(camera_name).id
    with mujoco.Renderer(env.model, height=int(pixels_np.shape[1]), width=int(pixels_np.shape[2])) as renderer:
        current_frame, current_info = reset_env_to_state(
            env,
            renderer,
            qpos=qpos_np[0],
            qvel=qvel_np[0],
            control=control_np[0],
            task_target=task_target_np[0],
            camera_id=camera_id,
            elapsed_time=float(time_np[0, 0]),
            disable_shadows=disable_shadows,
        )
        current_latent = encode_frames(model, current_frame[None], device, img_size)[0]
        previous_latent = current_latent.clone()
        current_state = torch.cat([current_latent, torch.zeros_like(current_latent)], dim=-1).cpu().numpy().astype(np.float64)
        rollout_frames = [current_frame.copy()]
        executed_actions_norm: list[np.ndarray] = []
        executed_actions_raw: list[np.ndarray] = []
        prev_U = jnp.broadcast_to(action_ref, (cfg.horizon, action_dim)).astype(jnp.float64)
        jax_seed_key = jax.random.PRNGKey(cfg.seed)

        pbar = tqdm(range(cfg.max_mpc_steps), desc="Receding Horizon MPPI + SLS Sequence Loops")
        for step_idx in pbar:
            jax_seed_key, subkey = jax.random.split(jax_seed_key)
            init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
            
            mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state), init_act_seq)
            X_warmstart = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
            
            controller.X_in = X_warmstart
            controller.U_in = jnp.asarray(mppi_res["act_seq"])

            try:
                u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_warmstart, parameter=1.0/30.0)
                solver_status = "sls_refined"
            except Exception:
                u0, X_pred, U_pred = mppi_res["act_seq"][0], X_warmstart, mppi_res["act_seq"]
                solver_status = "mppi_fallback"

            u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
            u0_raw = normalized_to_raw_action(u0_norm, action_mean_raw, action_std_raw)
            prev_U = jnp.concatenate([U_pred[1:], U_pred[-1:]], axis=0)

            current_frame, current_info = step_env_with_action(
                env,
                renderer,
                action=u0_raw,
                control_decimation=control_decimation,
                camera_id=camera_id,
                elapsed_time=(step_idx + 1) * control_timestep,
                disable_shadows=disable_shadows,
            )
            rollout_frames.append(current_frame.copy())
            executed_actions_norm.append(u0_norm.astype(np.float32))
            executed_actions_raw.append(u0_raw.astype(np.float32))

            next_latent = encode_frames(model, current_frame[None], device, img_size)[0]
            current_state = torch.cat([next_latent, next_latent - previous_latent], dim=-1).cpu().numpy().astype(np.float64)
            previous_latent = next_latent

            latent_err = float(np.linalg.norm(current_state - goal_state))
            task_err = float(np.linalg.norm(np.asarray(current_info["task_target"], dtype=np.float64) - task_target_np[-1].astype(np.float64)))
            pbar.set_postfix(lat_err=f"{latent_err:.3f}", task_err=f"{task_err:.3f}", status=solver_status)
            if latent_err <= 0.05 or task_err <= 1e-3:
                break

    imageio.mimwrite(run_dir / "mppi_sls_rope.mp4", rollout_frames, fps=cfg.video_fps)
    np.savez(
        run_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    print(f"Rollout successfully complete. Artifacts written to {run_dir}")

if __name__ == "__main__":
    main()
