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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

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
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset, build_markov_state
from error_model import MGNLLPredictor

@dataclass
class PlanSLSOGBenchConfig:
    """Configuration for Conformal SLS MPC OGBench Cube Planning"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("ogbench_cube/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("ogbench_cube/models/error_model/best-error-model.ckpt"))
    dataset_path: Path = field(default=Path("ogbench_cube/data/test_data/ogbench_cube_test.h5"))
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mpc_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=15)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    
    grasp_contact_threshold: float = 0.5
    grasp_alignment_threshold: float = 0.03

# --- Helper Utilities ---

def hide_target_cube(env) -> None:
    """Zeroes out the alpha transparency channel for the target block visual geoms."""
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    return lambda x, u, t: jnp.concatenate([x - x_max, x_min - x], axis=0)

def combine_constraints(c1, c2):
    return lambda x, u, t: jnp.concatenate([c1(x, u, t), c2(x, u, t)], axis=0)

def make_tracking_cost(action_weight: float, horizon: int, W_term: jnp.ndarray, goal_state: jnp.ndarray):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        dz = z - jnp.where(is_not_terminal, reference[t], goal_state)
        return jnp.sum(jnp.where(is_not_terminal, W, W_term) * dz**2) + action_weight * jnp.sum(u**2)
    return cost

def cube_is_grasped(info, contact_thresh, align_thresh) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_thresh and block_alignment <= align_thresh)

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

    # 1. Environment & Config Loading
    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json") as f: config_dict = json.load(f)
    
    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()
    error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()

    state_dim = config_dict.get("markov_state_dim", 48)
    action_dim = config_dict.get("action_dim", 5)
    img_size = config_dict.get("img_size", 224)

    dynamics = build_jax_dynamics(model.predictor.net, device, state_dim, action_dim)
    disturbance = build_jax_disturbance(error_model, cfg.q_learned, device, state_dim)

    train_stats = LeWMOGBenchCubeDataset(str(cfg.dataset_path), markov_deriv=1, num_preds=1, frameskip=1, img_size=img_size, action_dim=action_dim)
    action_mean, action_std = train_stats.action_mean.astype(np.float32), train_stats.action_std.astype(np.float32)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with h5py.File(cfg.dataset_path, "r") as h5:
        episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(np.random.choice(np.flatnonzero(h5["ep_len"][:] >= 2)))
        rows = np.arange(int(h5["ep_offset"][episode_idx]), int(h5["ep_offset"][episode_idx]) + int(h5["ep_len"][episode_idx]))
        qpos_init, qvel_init = h5["qpos"][rows[0]], h5["qvel"][rows[0]]
        qpos_goal, qvel_goal = h5["qpos"][rows[-1]], h5["qvel"][rows[-1]]
        
        target_pos_goal = np.asarray(h5["target_block_pos"][rows[-1]], dtype=np.float32)
        target_yaw_goal = float(np.asarray(h5["target_block_yaw"][rows[-1]]).reshape(-1)[0])

    env = gymnasium.make("cube-single-v0", terminate_at_goal=False, mode="data_collection", width=config_dict.get("width", 256), height=config_dict.get("height", 256))
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0, noise_smoothing=0.5)

    # --- Fetch goal image sequence & propagate physics before capturing frame ---
    env.reset(seed=cfg.seed)
    hide_target_cube(env)
    
    # Sync target body orientations before evaluating visual render boundaries
    env.unwrapped._target_block = 0
    target_mocap_id = env.unwrapped._cube_target_mocap_ids[0]
    env.unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_pos_goal, dtype=np.float64)
    env.unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(lie.SO3.from_z_radians(float(target_yaw_goal)).wxyz, dtype=np.float64)
    
    env.unwrapped._data.qpos[:qpos_goal.shape[0]] = qpos_goal.astype(np.float64)
    env.unwrapped._data.qvel[:qvel_goal.shape[0]] = qvel_goal.astype(np.float64)
    env.unwrapped.pre_step()
    mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)  # Force visual state sync
    env.unwrapped.post_step()
    goal_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)

    # --- Fetch initial image sequence & propagate physics before capturing frame ---
    env.reset(seed=cfg.seed)
    hide_target_cube(env)
    env.unwrapped._data.qpos[:qpos_init.shape[0]] = qpos_init.astype(np.float64)
    env.unwrapped._data.qvel[:qvel_init.shape[0]] = qvel_init.astype(np.float64)
    env.unwrapped.pre_step()
    mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)  # Force visual state sync
    env.unwrapped.post_step()
    current_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)
    current_info = env.unwrapped.get_step_info()

    goal_emb = encode_single_frame(model, goal_frame, device, img_size, pixel_mean, pixel_std)
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
        current_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)
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
    W_stage = jnp.ones((state_dim,)) * 0.005
    # W_stage = W_stage.at[:state_dim // 2].set(1.0)
    W_term = jnp.ones((state_dim,))*5.0
    # W_term = W_term.at[:state_dim // 2].set(100.0)
    cost = make_tracking_cost(0.1, cfg.horizon, W_term, jnp.asarray(goal_state))

    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=False, initialize_nominal=True, warm_start=False, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=300, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=jnp.zeros(action_dim), dt=1.0/20.0),
        dynamics=dynamics, constraints=combine_constraints(make_state_box_constraints(-100.0*jnp.ones(state_dim), 100.0*jnp.ones(state_dim)), make_control_box_constraints(-10.0*jnp.ones(action_dim), 10.0*jnp.ones(action_dim))),
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim, disturbance=disturbance, shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim), dtype=jnp.float64), U_in=jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    )

    current_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    X_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_u0 = np.zeros(action_dim, dtype=np.float32)

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="SLS Local Track Core Loop")
    for _ in mpc_pbar:
        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_ref, parameter=1.0/20.0)
            status = "sls_mpc"
        except Exception as e:
            import traceback
            traceback.print_exc()
            u0, X_pred = None, None
            status = "exception"

        if u0 is None or not jnp.all(jnp.isfinite(X_pred)):
            u0 = prev_u0
            status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)

        # Map standardized control variables back to raw gym dimensions
        u_raw = ((np.asarray(u0, dtype=np.float32) * action_std.flatten()) + action_mean.flatten()).astype(np.float32)
        _, _, _, _, current_info = env.step(u_raw)
        
        current_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)
        rollout_frames.append(current_frame.copy())

        next_emb = encode_single_frame(model, current_frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        lat_err = float(np.linalg.norm(current_state - goal_state))
        mpc_pbar.set_postfix(latent_err=f"{lat_err:.4f}", status=status)
        if lat_err <= 0.05: break

    imageio.mimwrite(out_dir / "cube_sls_rollout.mp4", rollout_frames, fps=cfg.video_fps)
    env.close()

if __name__ == "__main__":
    main()