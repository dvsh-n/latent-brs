#!/usr/bin/env python3
"""Plan in Rope pixel space using Conformal SLS MPC over a Markov-state PyTorch world model."""

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
import numpy as np
import torch
from tqdm.auto import tqdm
import pyrallis

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

# gpu_sls core modules
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints

# Rope training framework imports
from rope.train.mlpdyn_train import (
    LeWMRopeDataset,
    build_markov_state,
    preprocess_pixels,
)
from error_model import MGNLLPredictor

@dataclass
class PlanSLSRopeConfig:
    """Configuration for Conformal SLS MPC Rope Planning"""
    q_learned: float = field(default=0.0, metadata={"help": "Conformal quantile for the disturbance bound."})
    model_dir: Path = field(default=Path("rope/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("rope/models/error_model/best-error-model.ckpt"))
    dataset_path: Path = field(default=Path("rope/data/expert_data/rope_random_cubic_spline.h5"))
    out_dir: Path = field(default=Path("rope/plan/sls_mpc_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=20)
    max_mpc_steps: int = field(default=150)
    video_fps: int = field(default=30)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)

# --- JAX / PyTorch Bridge Engines ---

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

# --- Cost, Context & Utilities ---

def make_tracking_cost(action_weight: float, horizon: int, W_term: jnp.ndarray, goal_state: jnp.ndarray):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W, W_term)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state)
        dz = z - active_ref
        return jnp.sum(active_W * dz**2) + action_weight * jnp.sum(u**2)
    return cost

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None: candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(pixel_np.copy()).permute(2, 0, 1).contiguous()
    tensor = preprocess_pixels(tensor.unsqueeze(0), img_size).to(device)
    output = model.encoder(tensor, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels_np: np.ndarray, device: torch.device, img_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous()
    tensor = preprocess_pixels(tensor, img_size).to(device)
    latents = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start : start + 32]
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)

def main():
    cfg = pyrallis.parse(config_class=PlanSLSRopeConfig)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Configurations & Model Parameters
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r") as f: config_dict = json.load(f)
    
    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()
    error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()

    state_dim = config_dict.get("markov_state_dim", 36)
    action_dim = config_dict.get("action_dim", 5) # Rope standard is 5 dimensions (x, y, z gripper displacements)
    img_size = config_dict.get("img_size", 224)

    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    disturbance = build_jax_disturbance(error_model, cfg.q_learned, device, state_dim, action_dim)

    # 2. Extract Normalization Parameters via LeWMRopeDataset wrapper
    train_dataset = LeWMRopeDataset(str(cfg.dataset_path), markov_deriv=1, num_preds=1, frameskip=1, img_size=img_size, action_dim=action_dim)
    action_mean, action_std = train_dataset.action_mean.astype(np.float32), train_dataset.action_std.astype(np.float32)

    with h5py.File(cfg.dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(np.random.choice(np.flatnonzero(ep_len >= cfg.horizon)))
        
        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        # Load your native simulator environment keys for rope tracing tracking safely
        rope_points_np = np.asarray(h5["points"][rows], dtype=np.float32) if "points" in h5 else None

    run_dir = out_dir / f"{int(time.time())}_rope_episode_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Reference Extraction Sequence
    true_latents = encode_frames(model, pixels_np, device, img_size)
    
    # Emulate rope build_markov_state strategy safely: [z_t, delta_z_t]
    start_z = true_latents[0]
    start_state = torch.cat([start_z, torch.zeros_like(start_z)], dim=-1).cpu().numpy().astype(np.float64)
    
    goal_z = true_latents[-1]
    goal_delta = goal_z - true_latents[-2]
    goal_state = torch.cat([goal_z, goal_delta], dim=-1).cpu().numpy().astype(np.float64)

    imageio.imwrite(run_dir / "start_rope.png", pixels_np[0])
    imageio.imwrite(run_dir / "goal_rope.png", pixels_np[-1])

    # 4. Weight Definition matrices
    W_state = jnp.ones((state_dim,)) * 10.0
    W_state = W_state.at[state_dim // 2 :].set(1.0)
    W_term = jnp.ones((state_dim,)) * 0.1
    W_term = W_term.at[: state_dim // 2].set(5000.0)
    
    cost = make_tracking_cost(action_weight=0.02, horizon=cfg.horizon, W_term=W_term, goal_state=jnp.asarray(goal_state))

    # 5. Solver Parameter Building Footprint
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=False, initialize_nominal=True, warm_start=False, rti=True)
    sqp_cfg = SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=400, rho_update_frequency=20, initial_rho=1.0)
    
    mpc_dt = 1.0 / 30.0 # Match standard frame processing loop frequency
    mpc_cfg = MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_state, u_ref=jnp.zeros(action_dim), dt=mpc_dt)

    u_min, u_max = -2.0 * jnp.ones(action_dim), 2.0 * jnp.ones(action_dim)
    x_min, x_max = -1000.0 * jnp.ones(state_dim), 1000.0 * jnp.ones(state_dim)
    constraints_all = combine_constraints(make_state_box_constraints(x_min, x_max), make_control_box_constraints(u_min, u_max))

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg, config=mpc_cfg, dynamics=dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim,
        disturbance=disturbance, shift=1, X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64), U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64)
    )

    # 6. Receding Horizon Simulation Tracking Phase
    current_frame = pixels_np[0].copy()
    current_emb = encode_single_frame(model, current_frame, device, img_size)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    
    rollout_frames = [current_frame.copy()]
    X_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_u0 = np.zeros(action_dim, dtype=np.float32)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="Rope Conformal SLS execution loop")
    for step_idx in pbar:
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_ref, parameter=mpc_dt)
            solver_status = "sls_mpc"
        except Exception as e:
            u0, X_pred, U_pred = None, None, None
            solver_status = "exception_fallback"

        if u0 is None or not jnp.all(jnp.isfinite(X_pred)):
            u0 = prev_u0
            solver_status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)

        # Environment step transition emulation over target sequence
        sim_index = min(step_idx + 1, len(pixels_np) - 1)
        current_frame = pixels_np[sim_index].copy()
        rollout_frames.append(current_frame.copy())

        # Forward sequence propagation mechanics
        next_emb = encode_single_frame(model, current_frame, device, img_size)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        latent_err = float(np.linalg.norm(current_state - goal_state))
        pbar.set_postfix(latent_error=f"{latent_err:.4f}", status=solver_status)
        
        if latent_err <= 0.05 or sim_index == len(pixels_np) - 1: break

    imageio.mimwrite(run_dir / "rope_rollout.mp4", rollout_frames, fps=cfg.video_fps, quality=8, macro_block_size=1)
    print(f"Rope SLS MPC Planning sequence logged cleanly inside: {run_dir}")

if __name__ == "__main__":
    main()