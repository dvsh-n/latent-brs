#!/usr/bin/env python3
"""Plan in Reacher pixel space using Conformal SLS MPC warmstarted by MPPI over a Markov-state world model."""

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
import equinox as eqx
from jax import config, lax
config.update("jax_default_matmul_precision", "highest")
config.update("jax_enable_x64", True)

# gpu_sls imports
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints

# MPPI planner imports
from gpu_sls.mppi_planner import MPPIPlanner

# Local reacher imports
from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.mlpdyn_train import LeWMReacherDataset
from reacher.train.reacher_policy_train import DmControlGymEnv
from error_model import MGNLLPredictor

# --- Configuration Dataclass ---
@dataclass
class PlanSLSConfig:
    """Configuration for Conformal SLS MPC Planning with MPPI Warmstart"""
    q_learned: float = field(default=0.0, metadata={"help": "Conformal quantile for the disturbance bound."})
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
    
    # MPPI configuration fields
    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 20.0
    mppi_noise_level: float = 0.15
    mppi_beta_filter: float = 0.7


# --- PyTorch to Native JAX / Equinox Bridge ---

def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array, activation=jax.nn.gelu) -> eqx.Module:
    """
    Dynamically creates a native JAX Equinox MLP matching the PyTorch model's architecture.
    Extracts weights (handling spectral norm) for zero-copy execution on the GPU.
    """
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    
    layers = []
    keys = jax.random.split(key, len(pt_linears))
    for i, pt_layer in enumerate(pt_linears):
        out_features, in_features = pt_layer.weight.shape
        eqx_linear = eqx.nn.Linear(in_features, out_features, key=keys[i])
        
        # Transfer computed weights (this captures spectral_norm scaling)
        w = jnp.array(pt_layer.weight.detach().cpu().numpy())
        if pt_layer.bias is not None:
            b = jnp.array(pt_layer.bias.detach().cpu().numpy())
        else:
            b = jnp.zeros(out_features)
            
        eqx_linear = eqx.tree_at(lambda l: (l.weight, l.bias), eqx_linear, (w, b))
        layers.append(eqx_linear)
        
        # Add activation for all but the last layer
        if i < len(pt_linears) - 1:
            layers.append(activation)
            
    class JAXMLP(eqx.Module):
        layers: list
        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
            
    return JAXMLP(layers)

def make_jax_dynamics(eqx_dyn_model):
    """Wraps the Equinox MLP into the SLS expected (x, u, t, param) signature."""
    def jax_dynamics(x, u, t=0.0, parameter=1.0):
        inp = jnp.concatenate([x, u], axis=-1)
        return eqx_dyn_model(inp)
    return jax_dynamics

def make_jax_disturbance(eqx_error_model, q_learned, state_dim, diagonal):
    """Wraps the Equinox Error model and handles the MGNLL matrix transformation natively in JAX."""
    def _mgnll_forward(raw):
        if diagonal:
            return jnp.diag(jnp.exp(raw) + 1e-4)
            
        L = jnp.zeros((state_dim, state_dim))
        tril_indices = jnp.tril_indices(state_dim)
        L = L.at[tril_indices].set(raw)
        
        diag_idx = jnp.arange(state_dim)
        L = L.at[diag_idx, diag_idx].set(jnp.exp(L[diag_idx, diag_idx]) + 1e-4)
        return L

    def dist_fn(X_seq, U_seq):
        # Concatenate state and action sequences along the feature dimension
        inp = jnp.concatenate([X_seq, U_seq], axis=-1)
        # Vectorize the model and matrix transformation across the sequence length (Horizon)
        raw_preds = jax.vmap(eqx_error_model)(inp)
        L_mats = jax.vmap(_mgnll_forward)(raw_preds)
        return q_learned * L_mats
        
    return dist_fn


# --- MPPI Rollout Utils ---

def make_mppi_rollout_and_eval(jax_dynamics_fn, state_dim, action_dim, horizon, W_state, goal_state):
    def mppi_rollout_fn(state_cur, act_seqs, reach_config=None):
        def single_sample_rollout(actions):
            def step(state, u):
                next_state = jax_dynamics_fn(state, u, 0.0, 1.0)
                return next_state, next_state
            _, states = lax.scan(step, state_cur, actions)
            return states

        # Native JAX batching over samples
        state_seqs = jax.vmap(single_sample_rollout)(act_seqs)
        return state_seqs, {}

    def mppi_eval_fn(state_seqs, act_seqs, reach_config=None, aux=None, *args, **kwargs):
        delta = state_seqs - goal_state[None, None, :]
        stage_costs = jnp.sum(W_state[None, None, :] * (delta ** 2), axis=-1)
        action_costs = 0.1 * jnp.sum(act_seqs ** 2, axis=-1)
        total_costs = jnp.sum(stage_costs + action_costs, axis=-1)
        return {"rewards": -total_costs}
        
    # def mppi_eval_fn(state_seqs, act_seqs, reach_config=None, aux=None, *args, **kwargs):
    #     # state_seqs shape: (Batch, Long_Horizon, StateDim)
    #     B, H_long, _ = state_seqs.shape
    #     delta = state_seqs - goal_state[None, None, :]
        
    #     # 1. Create a time-discounting vector gamma^t (e.g., gamma = 0.95)
    #     gamma = 0.96
    #     discount_factors = jnp.power(gamma, jnp.arange(H_long - 1)) # Shape: (H_long - 1,)
        
    #     # 2. Compute stage errors per step
    #     per_step_stage_costs = jnp.sum(W_mppi_stage[None, None, :] * (delta[:, :-1, :] ** 2), axis=-1)
    #     action_costs = 0.01 * jnp.sum(act_seqs ** 2, axis=-1)
        
    #     # 3. Apply discounting to stage costs and action costs
    #     discounted_stage = jnp.sum(per_step_stage_costs * discount_factors[None, :], axis=-1)
    #     discounted_actions = jnp.sum(action_costs[:, :-1] * discount_factors[None, :], axis=-1)
        
    #     # 4. Terminal cost at the very end of the 140 steps remains undiscounted
    #     term_cost = jnp.sum(W_mppi_term[None, :] * (delta[:, -1, :] ** 2), axis=-1)
        
    #     total_costs = discounted_stage + discounted_actions + term_cost + action_costs[:, -1]
    #     return {"rewards": -total_costs}

    return mppi_rollout_fn, mppi_eval_fn

# --- Cost and Constraints ---

def make_tracking_cost(action_weight: float = 0.1, horizon: int = 35, W_term: Optional[jnp.ndarray] = None, goal_state: Optional[jnp.ndarray] = None):
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

# --- Setup Helpers ---

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

def shift_warmstart(X: jnp.ndarray, U: jnp.ndarray):
    X_shift = jnp.concatenate([X[1:], X[-1:]], axis=0)
    U_shift = jnp.concatenate([U[1:], U[-1:]], axis=0)
    return X_shift, U_shift


# --- Main Pipeline ---

def main():
    cfg = pyrallis.parse(config_class=PlanSLSConfig)
    if cfg.q_learned == 0.0:
        print("WARNING: q_learned is 0.0. Ensure you pass the correct value via yaml or CLI.")
        
    device = require_device(cfg.device)
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch Visual and Dynamics Model
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    
    checkpoint_path = latest_object_checkpoint(model_dir).resolve()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device).eval()
    
    state_dim = config_dict.get("markov_state_dim", 2 * config_dict.get("embed_dim", 18))
    action_dim = config_dict.get("action_dim", 2)
    img_size = config_dict.get("img_size", 224)

    # 2. Load PyTorch Error Model
    error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()

    # 3. Create Native JAX Modules and callables
    init_key = jax.random.PRNGKey(cfg.seed)
    key_dyn, key_err = jax.random.split(init_key)
    
    eqx_dyn = build_equinox_mlp_from_pytorch(model.predictor.net, key_dyn)
    eqx_err = build_equinox_mlp_from_pytorch(error_model.net, key_err)
    
    dynamics = make_jax_dynamics(eqx_dyn)
    disturbance = make_jax_disturbance(eqx_err, cfg.q_learned, state_dim, error_model.diagonal)

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

    # 5. Goal State Extraction & Cost Weights Setup
    pixels_t = preprocess_pixels(pixels_np, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    true_latents = encode_frames(model, pixels_t, device=device, frame_batch_size=32)
    start_state = make_markov_state(true_latents[0]).detach().cpu().numpy().astype(np.float64)
    goal_state = make_markov_state(true_latents[-1]).detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)

    W_state_mppi = jnp.ones((state_dim,)) * 50.0
    W_state_mppi = W_state_mppi.at[state_dim // 2 :].set(10.0)
    W_state = jnp.ones((state_dim,)) * 10.0
    W_state = W_state.at[state_dim // 2 :].set(1.0)
    W_term = jnp.ones((state_dim,)) * 0.1
    W_term = W_term.at[:state_dim // 2].set(4000.0)
    cost = make_tracking_cost(action_weight=0.1, horizon=cfg.horizon, W_term=W_term, goal_state=goal_state)

    save_rgb_image(run_dir / "start_image.png", pixels_np[0])
    save_rgb_image(run_dir / "goal_image.png", pixels_np[-1])

    # 6. MPPI Planner Configuration & Setup
    mppi_config_dict = {
        "planning": {
            "action_dim": action_dim,
            "n_sample": cfg.mppi_samples,
            "horizon": cfg.horizon,
            "n_update_iter": cfg.mppi_update_iter,
            "use_last": True,
            "reject_bad": False,
            "mppi": {
                "reward_weight": cfg.mppi_reward_weight,
                "noise_level": cfg.mppi_noise_level,
                "noise_decay": 1.0,
                "beta_filter": cfg.mppi_beta_filter
            }
        }
    }
    
    mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
        dynamics, state_dim, action_dim, cfg.horizon, W_state_mppi, goal_state
    )
    
    u_min, u_max = -500.0 * jnp.ones(action_dim), 500.0 * jnp.ones(action_dim)
    mppi_planner = MPPIPlanner(
        config=mppi_config_dict,
        model_rollout_fn=mppi_rollout,
        evaluate_traj_fn=mppi_eval,
        action_lower_lim=u_min,
        action_upper_lim=u_max
    )

    def standard_mppi_jit_wrapper(key, state, init_act):
        return mppi_planner.trajectory_optimization(key, state, init_act, skip=False)

    jit_mppi_trajopt = jax.jit(standard_mppi_jit_wrapper)

    # 7. SLS Solver Configuration
    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        warm_start=True,
        rti=True,
        R_bar=None,
        Q_bar=None,
    )
    sqp_cfg = SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=400, rho_update_frequency=20, initial_rho=1.0)
    
    mpc_dt = 1.0 / physics_freq_hz
    mpc_cfg = MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_state, u_ref=jnp.zeros(action_dim), dt=mpc_dt)

    x_min, x_max = -5.0 * jnp.ones(state_dim), 5.0 * jnp.ones(state_dim)
    constraints_all = combine_constraints(make_state_box_constraints(x_min, x_max), make_control_box_constraints(u_min, u_max))

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg,
        config=mpc_cfg, dynamics=dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim,
        disturbance=disturbance, shift=1,
        X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64),
        U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64),
    )

    # 8. Env Setup & Receding Horizon MPC Loop
    env = make_render_env(seed=episode_seed, time_limit=time_limit, width=width, height=height, physics_freq_hz=physics_freq_hz)
    current_frame = reset_env_to_state(env, seed=episode_seed, qpos=qpos_np[0], qvel=qvel_np[0], height=height, width=width)
    
    current_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    current_state = make_markov_state(current_emb).detach().cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    
    prev_U = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    jax_seed_key = jax.random.PRNGKey(cfg.seed)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="MPPI + SLS Steps")
    for _ in pbar:
        jax_seed_key, subkey = jax.random.split(jax_seed_key)
        
        # 8a. Query MPPI Trajectory Optimization for Warmstart Sequence
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
        mppi_res = jit_mppi_trajopt(
            subkey, 
            jnp.asarray(current_state), 
            init_act_seq
        )
        
        X_mppi = jnp.asarray(mppi_res["state_seq"])
        U_mppi = jnp.asarray(mppi_res["act_seq"])
        
        X_warmstart = jnp.concatenate([jnp.asarray(current_state)[None, :], X_mppi], axis=0) 
        X_ref = X_warmstart

        controller.X_in = X_warmstart
        controller.U_in = U_mppi

        # 8b. Run SLS Refinement
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(
                x0=current_state, reference=X_ref, parameter=mpc_dt
            )
            solver_status = "sls_refined"
        except Exception as e:
            print(f"\n[WARN] GenericMPC solve raised exception: {e}")
            u0, X_pred, U_pred = None, None, None

        if u0 is None or not jnp.all(jnp.isfinite(X_pred)) or not jnp.all(jnp.isfinite(U_pred)):
            print("\n[WARN] SLS Solver failed. Falling back directly to MPPI nominal candidates.")
            X_pred = X_warmstart
            U_pred = U_mppi
            u0 = U_pred[0]
            solver_status = "mppi_fallback"
            
        prev_X, prev_U = shift_warmstart(X_pred, U_pred)
        
        # 8c. Apply Action Step
        u0_norm = np.asarray(u0, dtype=np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        obs, _, terminated, truncated, _ = env.step(u0_raw)
        
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        
        # 8d. State Update
        next_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        current_state = make_markov_state(next_emb, current_emb).detach().cpu().numpy().astype(np.float64)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        
        obs_err = float(np.linalg.norm(current_obs - goal_obs))
        latent_err = float(np.linalg.norm(current_state - goal_state))
        pbar.set_postfix(obs_err=f"{obs_err:.3f}", lat_err=f"{latent_err:.3f}", status=solver_status)

        if latent_err <= 0.05 or terminated or truncated:
            break

    if rollout_frames:
        save_rollout_video(rollout_frames, run_dir, fps=cfg.video_fps)
    env.close()
    print(f"Saved run data to {run_dir}")

if __name__ == "__main__":
    main()