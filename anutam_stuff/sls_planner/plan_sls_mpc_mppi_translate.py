#!/usr/bin/env python3
"""Plan in Reacher pixel space using Cross-Latent Trajectory Translation with MPPI and Conformal SLS MPC."""

import os
import sys
import re
import time
import json
import concurrent.futures
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
import matplotlib.pyplot as plt

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
from reacher.train.latent_translate_mlp import LatentTranslator
from error_model import MGNLLPredictor

# --- Configuration Dataclass ---
@dataclass
class PlanSLSConfig:
    """Configuration for Cross-Latent Translated SLS MPC with MPPI Warmstart"""
    q_learned: float = field(default=0.0)
    source_model_dir: Path = field(default=Path("reacher/models/mlpdyn_ft_4"))
    target_model_dir: Path = field(default=Path("reacher/models/mlpdyn_ft_7"))
    translation_model_ckpt: Path = field(default=Path("reacher/models/translate_4_to_7/last.ckpt"))
    target_error_model_ckpt: Path = field(default=Path("reacher/models/error_model/target_best-error-model.ckpt"))
    
    dataset_path: Path = field(default=Path("reacher/data/test_data_50hz/reacher_test.h5"))
    out_dir: Path = field(default=Path("reacher/plan/sls_mpc_cross_latent"))
    device: str = field(default="auto")
    horizon: int = field(default=35)
    max_mpc_steps: int = field(default=120)
    video_fps: int = field(default=60)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    
    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 20.0
    mppi_noise_level: float = 0.15
    mppi_beta_filter: float = 0.7


# --- PyTorch to Equinox Weights Ingestion Engine ---

def build_equinox_mlp_from_pytorch(pt_model: torch.nn.Module, key: jax.Array, activation=jax.nn.gelu) -> eqx.Module:
    """Extracts weights directly from PyTorch architectures into native, compile-safe Equinox modules."""
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
            for layer in self.layers:
                x = layer(x)
            return x
    return JAXMLP(layers)

def make_jax_dynamics(eqx_dyn_model):
    def jax_dynamics(x, u, t=0.0, parameter=1.0):
        inp = jnp.concatenate([x, u], axis=-1)
        return eqx_dyn_model(inp)
    return jax_dynamics

def make_jax_disturbance(eqx_error_model, q_learned, state_dim, diagonal):
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
        inp = jnp.concatenate([X_seq, U_seq], axis=-1)
        raw_preds = jax.vmap(eqx_error_model)(inp)
        L_mats = jax.vmap(_mgnll_forward)(raw_preds)
        return q_learned * L_mats
    return dist_fn


# --- Asynchronous Threaded Visualization Plotter ---

def asynchronous_tube_plot(mpc_step, tube_data, state_dim, save_path):
    """Draws tracking tube widths inside a separate worker thread to maximize GPU math throughput."""
    n_cols = 6
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=True)
    axes = axes.flatten()
    horizon_axis = np.arange(tube_data.shape[0])
    
    for dim_idx in range(state_dim):
        ax = axes[dim_idx]
        ax.plot(horizon_axis, tube_data[:, dim_idx], color="tab:blue", linewidth=1.5)
        ax.set_title(f"Dim {dim_idx}", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)
        
    for j in range(state_dim, len(axes)):
        axes[j].axis("off")
        
    fig.suptitle(f"Projected Target Tube Widths (MPC Step {mpc_step})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# --- Custom MPPI Rollout (Source Space Objective Mapping) ---

def make_mppi_rollout_and_eval(jax_source_dynamics_fn, state_dim, action_dim, W_mppi_stage, W_mppi_term, goal_state_source):
    def mppi_rollout_fn(state_cur, act_seqs, reach_config=None):
        def single_sample_rollout(actions):
            def step(state, u):
                next_state = jax_source_dynamics_fn(state, u, 0.0, 1.0)
                return next_state, next_state
            _, states = lax.scan(step, state_cur, actions)
            return states
        state_seqs = jax.vmap(single_sample_rollout)(act_seqs)
        return state_seqs, {}

    def mppi_eval_fn(state_seqs, act_seqs, reach_config=None, aux=None, *args, **kwargs):
        delta = state_seqs - goal_state_source[None, None, :]
        H_long = state_seqs.shape[1]
        
        # Exponential time decay factor to incentivize near-term execution rate
        gamma = 1.0 # 0.96
        discount_factors = jnp.power(gamma, jnp.arange(H_long - 1))
        
        per_step_stage_costs = jnp.sum(W_mppi_stage[None, None, :] * (delta[:, :-1, :] ** 2), axis=-1)
        action_costs = 0.01 * jnp.sum(act_seqs ** 2, axis=-1)
        
        discounted_stage = jnp.sum(per_step_stage_costs * discount_factors[None, :], axis=-1)
        discounted_actions = jnp.sum(action_costs[:, :-1] * discount_factors[None, :], axis=-1)
        
        term_cost = jnp.sum(W_mppi_term[None, :] * (delta[:, -1, :] ** 2), axis=-1)
        total_costs = discounted_stage + discounted_actions + term_cost + action_costs[:, -1]
        return {"rewards": -total_costs}

    return mppi_rollout_fn, mppi_eval_fn


# --- Tracking Cost Definition ---

def make_tracking_cost(action_weight: float, horizon: int, W_term: jnp.ndarray, goal_state_target: jnp.ndarray, W_track: jnp.ndarray):
    def cost(W_ignored, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W_track, W_term)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state_target)
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
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None: candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No object checkpoints in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def require_device(device_arg: str) -> torch.device:
    if device_arg in {"auto", "gpu"}: device_arg = "cuda" if torch.cuda.is_available() else "cpu"
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
    imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
    return mp4_path

def preprocess_pixels(pixels: np.ndarray, img_size: int, pixel_mean: torch.Tensor, pixel_std: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    if tensor.ndim == 3: tensor = tensor.unsqueeze(0)
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


# --- Main Executive Pipeline ---

def main():
    cfg = pyrallis.parse(config_class=PlanSLSConfig)
    device = require_device(cfg.device)
    out_dir = cfg.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vis_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    # 1. Load PyTorch Base Objects (Source, Target, Translation, Error)
    src_dir = cfg.source_model_dir.expanduser().resolve()
    tgt_dir = cfg.target_model_dir.expanduser().resolve()
    
    with open(src_dir / "config.json", "r") as f: src_config = json.load(f)
    with open(tgt_dir / "config.json", "r") as f: tgt_config = json.load(f)
        
    src_ckpt = latest_object_checkpoint(src_dir).resolve()
    tgt_ckpt = latest_object_checkpoint(tgt_dir).resolve()
    
    src_model = torch.load(src_ckpt, map_location=device, weights_only=False).to(device).eval()
    tgt_model = torch.load(tgt_ckpt, map_location=device, weights_only=False).to(device).eval()
    
    # Unpack custom dictionary mapping checkpoint parameters safely
    ckpt_payload = torch.load(cfg.translation_model_ckpt, map_location=device, weights_only=False)
    checkpoint_src_dim = ckpt_payload["model_state_dict"]["source_mean"].shape[1]
    checkpoint_tgt_dim = ckpt_payload["model_state_dict"]["target_mean"].shape[1]

    trans_model = LatentTranslator(
        source_dim=checkpoint_src_dim, target_dim=checkpoint_tgt_dim,
        hidden_dim=128, depth=2, dropout=0.0,
        source_mean=torch.zeros(checkpoint_src_dim), source_std=torch.ones(checkpoint_src_dim),
        target_mean=torch.zeros(checkpoint_tgt_dim), target_std=torch.ones(checkpoint_tgt_dim)
    ).to(device)
    trans_model.load_state_dict(ckpt_payload["model_state_dict"])
    trans_model.eval()

    error_model = MGNLLPredictor.load_from_checkpoint(cfg.target_error_model_ckpt).to(device).eval()
    
    src_state_dim = src_config.get("markov_state_dim", 36)
    tgt_state_dim = tgt_config.get("markov_state_dim", 36)
    action_dim = src_config.get("action_dim", 2)
    img_size = src_config.get("img_size", 224)

    # 2. Build Unified JAX / Equinox Modules
    init_key = jax.random.PRNGKey(cfg.seed)
    keys = jax.random.split(init_key, 4)
    
    eqx_src_dyn = build_equinox_mlp_from_pytorch(src_model.predictor.net, keys[0])
    eqx_tgt_dyn = build_equinox_mlp_from_pytorch(tgt_model.predictor.net, keys[1])
    eqx_trans   = build_equinox_mlp_from_pytorch(trans_model.net, keys[2])
    eqx_err     = build_equinox_mlp_from_pytorch(error_model.net, keys[3])
    
    source_dynamics = make_jax_dynamics(eqx_src_dyn)
    target_dynamics = make_jax_dynamics(eqx_tgt_dyn)
    target_disturbance = make_jax_disturbance(eqx_err, cfg.q_learned, tgt_state_dim, error_model.diagonal)

    # 3. Dynamic Structural Trajectory Translation & Reconstruction Wrapper
    def translate_and_reconstruct_trajectory(src_X_seq, last_tgt_pos):
        """
        Inputs:
            src_X_seq: (H, 24) short-horizon source Markov states
            last_tgt_pos: (5,) the true target position embedding from the previous tracking step
        Outputs:
            tgt_X_seq: (H, 10) fully reconstructed target Markov tracking reference
        """
        src_positions = src_X_seq[:, :12]
        
        # Translate position space mapping components (12 -> 5 dims)
        tgt_positions = jax.vmap(eqx_trans)(src_positions)
        
        # Shift target positions down to calculate historical delta vectors
        prev_tgt_positions = jnp.concatenate([last_tgt_pos[None, :], tgt_positions[:-1]], axis=0)
        tgt_velocities = tgt_positions - prev_tgt_positions
        
        return jnp.concatenate([tgt_positions, tgt_velocities], axis=-1)

    jax_translate_trajectory = jax.jit(translate_and_reconstruct_trajectory)

    # 4. Environment Data Parsing & Goal Generation
    train_dataset_path = str(cfg.dataset_path.expanduser().resolve())
    train_stats_dataset = LeWMReacherDataset(train_dataset_path, history_size=1, num_preds=1, frameskip=1, img_size=img_size, action_dim=action_dim)
    pixel_mean, pixel_std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    action_mean, action_std = train_stats_dataset.action_mean.astype(np.float32), train_stats_dataset.action_std.astype(np.float32)

    with h5py.File(cfg.dataset_path, "r") as h5: ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)
    episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(rng.choice(np.flatnonzero(ep_len >= 2)))

    with h5py.File(cfg.dataset_path, "r") as h5:
        offset = int(h5["ep_offset"][episode_idx])
        length = int(h5["ep_len"][episode_idx])
        rows = np.arange(offset, offset + length, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        obs_np = np.asarray(h5["observation"][rows], dtype=np.float32)
        episode_seed = int(h5["episode_seed"][episode_idx])
        physics_freq_hz = float(h5.attrs.get("physics_freq_hz", 100.0))
        time_limit = float(h5.attrs.get("time_limit", 10.0))
        height, width = int(pixels_np.shape[1]), int(pixels_np.shape[2])

    run_dir = out_dir / f"{int(time.time())}_episode_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tube_plots_dir = run_dir / "tube_plots"
    tube_plots_dir.mkdir(parents=True, exist_ok=True)

    pixels_t = preprocess_pixels(pixels_np, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    src_latents = encode_frames(src_model, pixels_t, device=device, frame_batch_size=32)
    tgt_latents = encode_frames(tgt_model, pixels_t, device=device, frame_batch_size=32)
    
    start_state_src = make_markov_state(src_latents[0]).detach().cpu().numpy().astype(np.float64)
    goal_state_src  = make_markov_state(src_latents[-1]).detach().cpu().numpy().astype(np.float64)
    goal_state_tgt  = make_markov_state(tgt_latents[-1]).detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)

    # --- Cost Setup ---
    W_mppi_stage = jnp.ones((src_state_dim,)) * 1.0
    W_mppi_stage = W_mppi_stage.at[src_state_dim // 2 :].set(0.01)  # Lighter velocity penalty during MPPI warmup

    W_mppi_term  = jnp.ones((src_state_dim,)) * 100.0
    W_mppi_term = W_mppi_term.at[src_state_dim // 2 :].set(0.01)  # Lighter velocity penalty during MPPI warmup

    # SLS Local Tracking Weights (Target Space - 10D)
    W_sls_track  = jnp.ones((tgt_state_dim,)) * 100.0
    W_sls_track  = W_sls_track.at[tgt_state_dim // 2 :].set(0.01)

    # Dedicated SLS Terminal Tracking Weights (Target Space - 10D)
    W_sls_term   = jnp.ones((tgt_state_dim,)) * 0.00
    W_sls_term   = W_sls_term.at[:tgt_state_dim // 2].set(0.00)  # Heavy landing penalty on target positions

    # Feed ONLY target-dimension arrays into the tracking cost closure
    cost = make_tracking_cost(
        action_weight=0.01, 
        horizon=cfg.horizon, 
        W_term=W_sls_term,          # Fixed: Length 10
        goal_state_target=goal_state_tgt, # Length 10
        W_track=W_sls_track         # Length 10
    )

    save_rgb_image(run_dir / "start_image.png", pixels_np[0])
    save_rgb_image(run_dir / "goal_image.png", pixels_np[-1])

    # 5. Long-Horizon MPPI Configuration (Horizon multiplied by 4)
    mppi_horizon_long = cfg.horizon * 4
    mppi_config_dict = {
        "planning": {
            "action_dim": action_dim,
            "n_sample": cfg.mppi_samples,
            "horizon": mppi_horizon_long,
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
    
    mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(source_dynamics, src_state_dim, action_dim, W_mppi_stage, W_mppi_term, goal_state_src)
    u_min, u_max = -500.0 * jnp.ones(action_dim), 500.0 * jnp.ones(action_dim)
    
    mppi_planner = MPPIPlanner(config=mppi_config_dict, model_rollout_fn=mppi_rollout, evaluate_traj_fn=mppi_eval, action_lower_lim=u_min, action_upper_lim=u_max)
    jit_mppi_trajopt = jax.jit(lambda k, s, a: mppi_planner.trajectory_optimization(k, s, a, skip=False))

    # 6. SLS Target Solver Construction
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=True, max_initial_sqp_iterations=0, initialize_nominal=True, warm_start=True, rti=False)
    sqp_cfg = SQPConfig(max_sqp_iterations=3, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False)
    admm_cfg = ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=200, rho_update_frequency=20, initial_rho=1.0)
    
    mpc_dt = 1.0 / physics_freq_hz
    mpc_cfg = MPCConfig(n=tgt_state_dim, nu=action_dim, N=cfg.horizon, W=W_sls_track, u_ref=jnp.zeros(action_dim), dt=mpc_dt)

    x_min, x_max = -1000.0 * jnp.ones(tgt_state_dim), 1000.0 * jnp.ones(tgt_state_dim)
    constraints_all = combine_constraints(make_state_box_constraints(x_min, x_max), make_control_box_constraints(u_min, u_max))

    controller = GenericMPC(
        sls_cfg, sqp_cfg, admm_cfg, config=mpc_cfg, dynamics=target_dynamics, constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * tgt_state_dim,
        disturbance=target_disturbance, shift=1, X_in=jnp.zeros((mpc_cfg.N + 1, mpc_cfg.n), dtype=jnp.float64), U_in=jnp.zeros((mpc_cfg.N, mpc_cfg.nu), dtype=jnp.float64),
    )

    # 7. Receding Horizon Execution Loop
    env = make_render_env(seed=episode_seed, time_limit=time_limit, width=width, height=height, physics_freq_hz=physics_freq_hz)
    current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
    
    current_emb_src = encode_single_frame(src_model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    current_state_src = make_markov_state(current_emb_src).detach().cpu().numpy().astype(np.float64)
    
    current_emb_tgt = encode_single_frame(tgt_model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
    current_state_tgt = make_markov_state(current_emb_tgt).detach().cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    prev_U = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    prev_u0 = np.zeros(action_dim, dtype=np.float32)
    jax_seed_key = jax.random.PRNGKey(cfg.seed)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="Cross-Latent SLS Loop")
    for mpc_step in pbar:
        jax_seed_key, subkey = jax.random.split(jax_seed_key)
        
        # 7a. Pad short-horizon memory to pass down to long-horizon MPPI
        nominal_warmstart = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0) 
        padding_actions = jnp.tile(nominal_warmstart[-1:], (mppi_horizon_long - cfg.horizon, 1))
        init_act_seq_long = jnp.concatenate([nominal_warmstart, padding_actions], axis=0)

        # 7b. Optimize inside Source Space via MPPI
        mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state_src), init_act_seq_long)
        X_mppi_src_long = jnp.asarray(mppi_res["state_seq"])
        U_mppi_long = jnp.asarray(mppi_res["act_seq"])
        
        # 7c. Slicing and cross-latent translation to target space reference
        X_mppi_src = X_mppi_src_long[:cfg.horizon]
        U_mppi = U_mppi_long[:cfg.horizon]
        
        # Extract previous target position embedding to construct true initial velocity parameters
        last_tgt_pos_anchor = jnp.asarray(current_state_tgt[:checkpoint_tgt_dim])
        
        # Map cut-down nominal source states through translation and reconstruction module
        X_mppi_tgt = jax_translate_trajectory(X_mppi_src, last_tgt_pos_anchor)
        
        X_warmstart_tgt = jnp.concatenate([jnp.asarray(current_state_tgt)[None, :], X_mppi_tgt], axis=0)
        X_ref_tgt = X_warmstart_tgt

        controller.X_in = X_warmstart_tgt
        controller.U_in = U_mppi

        # 7d. Execute SLS Local Robust Optimization inside Target Space
        try:
            u0, X_pred_tgt, U_pred, *solver_info = controller.run(x0=current_state_tgt, reference=X_ref_tgt, parameter=mpc_dt)
            solver_status = "genericmpc_tgt"
        except Exception as e:
            print(f"\n[WARN] GenericMPC target solve exception: {e}")
            u0, X_pred_tgt, U_pred = None, None, None
            solver_info = []

        if u0 is None or not jnp.all(jnp.isfinite(X_pred_tgt)) or not jnp.all(jnp.isfinite(U_pred)):
            print("\n[WARN] Target SLS Solver failure. Falling back to nominal translated MPPI path.")
            u0 = U_mppi[0]
            solver_status = "fallback_nominal"
            prev_X_tgt, prev_U = shift_warmstart(X_warmstart_tgt, U_mppi)
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            prev_X_tgt, prev_U = shift_warmstart(X_pred_tgt, U_pred)
            
            # Tube width visualization logging (Non-blocking background thread)
            if len(solver_info) >= 3:
                Phi_x = solver_info[2]
                tube = np.asarray(jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1))
                np.save(tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.npy", tube)
                
                plot_path = tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.png"
                vis_executor.submit(asynchronous_tube_plot, mpc_step, tube, tgt_state_dim, plot_path)
        
        # 7e. Step Physical Reacher Environment
        u0_norm = np.asarray(u0, dtype=np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        obs, _, terminated, truncated, _ = env.step(u0_raw)
        
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        
        # 7f. Multilateral State Ingestion Update
        next_emb_src = encode_single_frame(src_model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        current_state_src = make_markov_state(next_emb_src, current_emb_src).detach().cpu().numpy().astype(np.float64)
        current_emb_src = next_emb_src
        
        next_emb_tgt = encode_single_frame(tgt_model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        current_state_tgt = make_markov_state(next_emb_tgt, current_emb_tgt).detach().cpu().numpy().astype(np.float64)
        current_emb_tgt = next_emb_tgt

        rollout_frames.append(current_frame.copy())
        
        obs_err = float(np.linalg.norm(current_obs - goal_obs))
        latent_err = float(np.linalg.norm(current_state_tgt - goal_state_tgt))
        pbar.set_postfix(obs_err=f"{obs_err:.3f}", tgt_lat_err=f"{latent_err:.3f}", status=solver_status)

        if obs_err <= 0.05 or terminated or truncated: break

    vis_executor.shutdown(wait=False)
    if rollout_frames: save_rollout_video(rollout_frames, run_dir, fps=cfg.video_fps)
    env.close()
    print(f"Process complete. Saved run data to: {run_dir}")

if __name__ == "__main__":
    main()