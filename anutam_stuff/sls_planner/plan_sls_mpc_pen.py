#!/usr/bin/env python3
"""Plan in Reacher pixel space using Conformal SLS MPC over a Markov-state PyTorch world model."""

import os
import sys
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# If macOS, use glfw, otherwise default to egl (Linux/headless)
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
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import config
# Adopted from mpc_neural.py precision settings
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

# gpu_sls imports
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints

# Local reacher imports
from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.mlpdyn_train import LeWMReacherDataset
from reacher.train.reacher_policy_train import DmControlGymEnv
from error_model import MGNLLPredictor

# --- Configuration Dataclass ---
@dataclass
class PlanSLSConfig:
    """Configuration for Conformal SLS MPC Planning"""
    q_learned: float = field(default=0.99, metadata={"help": "Conformal quantile for the disturbance bound (required)."})
    dist_cost_weight: float = field(default=3.0, metadata={"help": "Weight for the disturbance minimization cost."})
    model_dir: Path = field(default=Path("reacher/models/mlpdyn_ft_1"))
    error_model_ckpt: Path = field(default=Path("reacher/models/error_model/best-error-model.ckpt"))
    dataset_path: Path = field(default=Path("reacher/data/test_data_50hz/reacher_test.h5"))
    out_dir: Path = field(default=Path("reacher/plan/sls_mpc_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=35)
    max_mpc_steps: int = field(default=120)
    video_fps: int = field(default=60)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)

# --- JAX / PyTorch Bridge ---

def build_jax_dynamics(torch_dynamics_net: torch.nn.Module, device: torch.device, state_dim: int, action_dim: int):
    """Wraps a PyTorch MLP into a JAX-differentiable function using custom_vjp."""
    
    def _fwd_fn(x_np, u_np):
        with torch.no_grad():
            x_t = torch.from_numpy(np.array(x_np)).float().to(device)
            u_t = torch.from_numpy(np.array(u_np)).float().to(device)
            inp = torch.cat((x_t, u_t), dim=-1)
            if inp.ndim == 1:
                out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0)
            else:
                out = torch_dynamics_net(inp)
            return np.asarray(out.cpu().numpy(), dtype=np.float64)

    def _vjp_fn(x_np, u_np, g_np):
        x_t = torch.from_numpy(np.array(x_np)).float().to(device).requires_grad_(True)
        u_t = torch.from_numpy(np.array(u_np)).float().to(device).requires_grad_(True)
        inp = torch.cat((x_t, u_t), dim=-1)
        
        if inp.ndim == 1:
            out = torch_dynamics_net(inp.unsqueeze(0)).squeeze(0)
        else:
            out = torch_dynamics_net(inp)
            
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

def build_jax_disturbance_pointwise(error_model: torch.nn.Module, q_learned: float, device: torch.device, state_dim: int, action_dim: int):
    """Wraps the Conformal MGNLL Error model into a differentiable JAX point-wise function."""
    
    def _fwd_fn(x_np, u_np):
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(x_np)).float().to(device)
            u_t = torch.from_numpy(np.asarray(u_np)).float().to(device)
            inp = torch.cat([x_t, u_t], dim=-1)
            
            L = error_model(inp) 
            E_vec = q_learned * L.squeeze(0)
            E_mat = torch.diag(E_vec)
            return np.asarray(E_mat.cpu().numpy(), dtype=np.float64)

    def _vjp_fn(x_np, u_np, g_np):
        # We need the gradient of the diagonal matrix w.r.t the state and action
        x_t = torch.from_numpy(np.asarray(x_np)).float().to(device).requires_grad_(True)
        u_t = torch.from_numpy(np.asarray(u_np)).float().to(device).requires_grad_(True)
        inp = torch.cat([x_t, u_t], dim=-1)
        
        L = error_model(inp)
        E_vec = q_learned * L.squeeze(0)
        E_mat = torch.diag(E_vec)
        
        g_t = torch.from_numpy(np.asarray(g_np)).float().to(device)
        E_mat.backward(g_t)
        
        grad_x = x_t.grad if x_t.grad is not None else torch.zeros_like(x_t)
        grad_u = u_t.grad if u_t.grad is not None else torch.zeros_like(u_t)
        
        return np.asarray(grad_x.cpu().numpy(), dtype=np.float64), np.asarray(grad_u.cpu().numpy(), dtype=np.float64)

    @jax.custom_vjp
    def jax_disturbance(x, u):
        result_shape = jax.ShapeDtypeStruct((state_dim, state_dim), jnp.float64)
        return jax.pure_callback(_fwd_fn, result_shape, x, u)

    def jax_disturbance_fwd(x, u):
        return jax_disturbance(x, u), (x, u)

    def jax_disturbance_bwd(res, g):
        x, u = res
        jac_x_shape = jax.ShapeDtypeStruct((state_dim,), jnp.float64)
        jac_u_shape = jax.ShapeDtypeStruct((action_dim,), jnp.float64)
        vjp_x, vjp_u = jax.pure_callback(_vjp_fn, (jac_x_shape, jac_u_shape), x, u, g)
        return vjp_x, vjp_u

    jax_disturbance.defvjp(jax_disturbance_fwd, jax_disturbance_bwd)
    return jax_disturbance

# --- Cost and Constraints ---

def make_tracking_cost(dist_fn, action_weight: float = 0.1, dist_weight: float = 1.0, horizon: int = 35, W_term: Optional[jnp.ndarray] = None, goal_state: Optional[jnp.ndarray] = None):
    """Quadratic tracking cost that also penalizes the magnitude of the disturbance."""
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        
        # State tracking cost
        active_W = jnp.where(is_not_terminal, W, W_term)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state)
        dz = z - active_ref
        track_cost = jnp.sum(active_W * dz**2) + action_weight * jnp.sum(u**2)
        
        # Disturbance minimization cost (Frobenius norm squared of E)
        E = dist_fn(z, u)
        dist_cost = jnp.sum(E**2)
        
        return track_cost + dist_weight * dist_cost
    return cost

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
    return constraints

# --- Setup Utilities ---

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)

def hide_target(env: DmControlGymEnv) -> None:
    target_geom_id = env._env.physics.model.name2id("target", "geom")
    env._env.physics.model.geom_rgba[target_geom_id] = [0, 0, 0, 0]

def configure_dm_control_timing(env: DmControlGymEnv, *, physics_timestep: float, time_limit: float) -> None:
    dm_env = env._env
    dm_env.physics.model.opt.timestep = physics_timestep
    dm_env._n_sub_steps = 1
    dm_env._step_limit = float("inf") if time_limit == float("inf") else time_limit / physics_timestep

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def save_rollout_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4_path = out_dir / "rollout.mp4"
    gif_path = out_dir / "rollout.gif"
    try:
        imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4_path
    except Exception:
        imageio.mimwrite(gif_path, frames, fps=fps)
        return gif_path

def preprocess_pixels(pixels: np.ndarray, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (tensor - pixel_mean) / pixel_std

@torch.no_grad()
def encode_frames(model: torch.nn.Module, pixels: torch.Tensor, device: torch.device, frame_batch_size: int) -> torch.Tensor:
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    return torch.cat(latents, dim=0)

@torch.no_grad()
def encode_single_frame(model: torch.nn.Module, pixel: np.ndarray, device: torch.device, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std).to(device)
    output = model.encoder(batch, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]

def make_markov_state(embedding: torch.Tensor, previous_embedding: torch.Tensor | None = None) -> torch.Tensor:
    delta = torch.zeros_like(embedding) if previous_embedding is None else embedding - previous_embedding
    return torch.cat((embedding, delta), dim=-1)

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)

def make_render_env(*, seed: int, time_limit: float, width: int, height: int, physics_freq_hz: float) -> DmControlGymEnv:
    env = DmControlGymEnv(domain_name="reacher", task_name="hard", seed=seed, time_limit=time_limit, action_cost_weight=0.0, action_rate_cost_weight=0.0, velocity_cost_weight=0.0)
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env

def reset_env_to_state(env: DmControlGymEnv, *, seed: int, qpos: np.ndarray, qvel: np.ndarray, height: int, width: int) -> np.ndarray:
    env.reset(seed=seed)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)

# --- Main Pipeline ---

def main():
    cfg = pyrallis.parse(config_class=PlanSLSConfig)
    if cfg.q_learned == 0.0:
        print("WARNING: q_learned is 0.0. Ensure you pass the correct value via yaml or CLI.")
        
    device = require_device(cfg.device)
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Visual and Dynamics Model
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    
    checkpoint_path = latest_object_checkpoint(model_dir).resolve()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device).eval()
    
    state_dim = config_dict.get("markov_state_dim", 2 * config_dict.get("embed_dim", 18))
    action_dim = config_dict.get("action_dim", 2)
    img_size = config_dict.get("img_size", 224)

    # 2. Load Error Model
    error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()

    # 3. Build JAX Callables (Using point-wise disturbance & vmap for the trajectory wrapper)
    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    dist_pointwise = build_jax_disturbance_pointwise(error_model, cfg.q_learned, device, state_dim, action_dim)
    
    # GenericMPC evaluates disturbances over trajectory states TxN. vmap handles this seamlessly.
    disturbance_traj = jax.vmap(dist_pointwise)

    # 4. Data / Normalization Setup
    train_dataset_path = str(cfg.dataset_path.expanduser().resolve())
    train_stats_dataset = LeWMReacherDataset(
        train_dataset_path, history_size=1, num_preds=1, frameskip=int(config_dict.get("frameskip", 1)),
        img_size=img_size, action_dim=action_dim,
    )
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    action_mean, action_std = train_stats_dataset.action_mean.astype(np.float32), train_stats_dataset.action_std.astype(np.float32)

    with h5py.File(cfg.dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    
    rng = np.random.default_rng(cfg.seed)
    episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(rng.choice(np.flatnonzero(ep_len >= 2)))

    with h5py.File(cfg.dataset_path, "r") as h5:
        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        qpos_np = np.asarray(h5["qpos"][rows], dtype=np.float32)
        qvel_np = np.asarray(h5["qvel"][rows], dtype=np.float32)
        obs_np = np.asarray(h5["observation"][rows], dtype=np.float32)
        episode_seed = int(h5["episode_seed"][episode_idx])
        physics_freq_hz = float(h5.attrs.get("physics_freq_hz", 100.0))
        time_limit = float(h5.attrs.get("time_limit", 10.0))
        height, width = int(pixels_np.shape[1]), int(pixels_np.shape[2])

    run_name = f"{int(time.time())}_episode_{episode_idx:05d}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tube_plots_dir = run_dir / "tube_plots"
    tube_plots_dir.mkdir(parents=True, exist_ok=True)

    # 5. Goal State Extraction
    pixels_t = preprocess_pixels(pixels_np, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    true_latents = encode_frames(model, pixels_t, device=device, frame_batch_size=32)
    start_state = make_markov_state(true_latents[0]).detach().cpu().numpy().astype(np.float64)
    goal_state = make_markov_state(true_latents[-1]).detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)

    # State tracking weights
    W_state = jnp.ones((state_dim,))* 0.002
    W_state = W_state.at[state_dim // 2 :].set(0.2)
    W_term = jnp.ones((state_dim,))*0.10
    W_term = W_term.at[:state_dim // 2].set(1000.0)
    
    # 6. Pass the differentiable dist_pointwise function down to trajax via the cost function
    cost = make_tracking_cost(
        dist_fn=dist_pointwise, 
        action_weight=0.01, 
        dist_weight=cfg.dist_cost_weight, 
        horizon=cfg.horizon, 
        W_term=W_term, 
        goal_state=goal_state
    )

    save_rgb_image(run_dir / "start_image.png", pixels_np[0])
    save_rgb_image(run_dir / "goal_image.png", pixels_np[-1])

    # 7. SLS Configuration
    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,           
        max_initial_sqp_iterations=0,
        initialize_nominal=True,
        warm_start=True,
        rti=False,
        R_bar = None,
        Q_bar = None,
    )
    
    sqp_cfg = SQPConfig(
        max_sqp_iterations=3, 
        warm_start=False,
        feas_tol=1e-2, 
        step_tol=1e-4,
        line_search=False
    )
    
    admm_cfg = ADMMConfig(
        eps_abs=1e-2, 
        eps_rel=0,
        rho_max=1e3, 
        max_iterations=400,
        rho_update_frequency=20,
        initial_rho=1.0,
    )
    
    mpc_dt = 1.0 / physics_freq_hz
    
    mpc_cfg = MPCConfig(
        n=state_dim,
        nu=action_dim,
        N=cfg.horizon,
        W=W_state,
        u_ref=jnp.zeros(action_dim),
        dt=mpc_dt,
    )

    u_min, u_max = -500.0 * jnp.ones(action_dim), 500.0 * jnp.ones(action_dim)
    x_min, x_max = -1000.0 * jnp.ones(state_dim), 1000.0 * jnp.ones(state_dim)
    constraints_all = combine_constraints(
        make_state_box_constraints(x_min, x_max), 
        make_control_box_constraints(u_min, u_max)
    )
    
    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg,
        config=mpc_cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), 
        cost=cost,
        num_constraints=2 * action_dim + 2 * state_dim,
        disturbance=disturbance_traj, # <--- The vmap'd trajectory operator 
        shift=1,
        X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64),
    )

    # 8. Env Setup & MPC Loop
    env = make_render_env(seed=episode_seed, time_limit=time_limit, width=width, height=height, physics_freq_hz=physics_freq_hz)
    current_frame = reset_env_to_state(env, seed=episode_seed, qpos=qpos_np[0], qvel=qvel_np[0], height=height, width=width)
    
    current_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    current_state = make_markov_state(current_emb).detach().cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    X_ref = jnp.tile(goal_state[None, :], (cfg.horizon + 1, 1))

    prev_u0 = np.zeros(action_dim, dtype=np.float32)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="SLS MPC Steps")
    for mpc_step in pbar:
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(
                x0=current_state, reference=X_ref, parameter=mpc_dt
            )
            solver_status = "genericmpc"
        except Exception as e:
            print(f"\n[WARN] GenericMPC solve raised exception: {e}")
            u0, X_pred, U_pred = None, None, None
            solver_info = []

        if u0 is None or not jnp.all(jnp.isfinite(X_pred)) or not jnp.all(jnp.isfinite(U_pred)):
            print("\n[WARN] SLS Solver failed or returned NaN. Using fallback (previous action).")
            u0 = prev_u0
            solver_status = "fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            
            if len(solver_info) >= 3:
                Phi_x = solver_info[2]
                tube = np.asarray(jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1))
                
                np.save(tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.npy", tube)
                
                n_cols = 6
                n_rows = int(np.ceil(state_dim / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=True)
                axes = np.atleast_1d(axes).flatten()
                horizon_axis = np.arange(cfg.horizon + 1)
                
                for dim_idx in range(state_dim):
                    ax = axes[dim_idx]
                    ax.plot(horizon_axis, tube[:, dim_idx], color="tab:blue", linewidth=1.5)
                    ax.set_title(f"Dim {dim_idx}", fontsize=8)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.tick_params(axis="both", which="major", labelsize=8)
                
                for dim_idx in range(state_dim, len(axes)):
                    axes[dim_idx].axis("off")
                
                fig.suptitle(f"Projected Tube Widths Across Horizon (MPC Step {mpc_step})", fontsize=14)
                plt.tight_layout()
                plt.savefig(tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.png", dpi=120)
                plt.close()
        
        # Apply Action
        u0_norm = np.asarray(u0, dtype=np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        obs, _, terminated, truncated, _ = env.step(u0_raw)
        
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        
        # Next State encoding
        next_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        current_state = make_markov_state(next_emb, current_emb).detach().cpu().numpy().astype(np.float64)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        
        # Telemetry
        obs_err = float(np.linalg.norm(current_obs - goal_obs))
        latent_err = float(np.linalg.norm(current_state - goal_state))
        pbar.set_postfix(obs_err=f"{obs_err:.3f}", lat_err=f"{latent_err:.3f}", status=solver_status)

        if obs_err <= 0.05 or terminated or truncated:
            break

    if rollout_frames:
        save_rollout_video(rollout_frames, run_dir, fps=cfg.video_fps)
    env.close()
    print(f"Saved run data to {run_dir}")

if __name__ == "__main__":
    main()