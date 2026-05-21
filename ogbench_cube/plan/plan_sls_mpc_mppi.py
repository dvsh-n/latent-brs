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

os.environ.setdefault("MUJOCO_GL", "egl")

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
from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiCubeConfig:
    """Configuration for Warmstarted Conformal SLS MPC on OGBench Cubes"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("ogbench_cube/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("ogbench_cube/models/error_model/best-error-model.ckpt"))
    dataset_path: Path = field(default=Path("ogbench_cube/data/test_data/ogbench_cube_test.h5"))
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mppi_conformal"))
    device: str = field(default="auto")
    horizon: int = field(default=16)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    
    mppi_samples: int = 512
    mppi_update_iter: int = 5
    mppi_reward_weight: float = 30.0
    mppi_noise_level: float = 0.25
    mppi_beta_filter: float = 0.6
    
    grasp_contact_threshold: float = 0.5
    grasp_alignment_threshold: float = 0.03

# --- Layer Weight Ingestion ---
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

def make_mppi_rollout_and_eval(dynamics_fn, W_mppi, goal_state):
    def rollout(state_cur, act_seqs, reach_config=None):
        def step_fn(s, u):
            nxt = dynamics_fn(s, u)
            return nxt, nxt
        return jax.vmap(lambda actions: lax.scan(step_fn, state_cur, actions)[1])(act_seqs), {}
        
    def eval_fn(states, acts, reach_config=None, aux=None, *args, **kwargs):
        costs = jnp.sum(W_mppi[None, None, :] * ((states - goal_state[None, None, :]) ** 2), axis=-1)
        return {"rewards": -jnp.sum(costs + 0.01 * jnp.sum(acts ** 2, axis=-1), axis=-1)}
    return rollout, eval_fn

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    return lambda x, u, t: jnp.concatenate([u - u_max, u_min - u], axis=0)

def make_state_box_constraints(x_min, x_max):
    x_min, x_max = jnp.asarray(x_min), jnp.asarray(x_max)
    return lambda x, u, t: jnp.concatenate([x - x_max, x_min - x], axis=0)

def combine_constraints(c1, c2):
    return lambda x, u, t: jnp.concatenate([c1(x, u, t), c2(x, u, t)], axis=0)

def make_tracking_cost(action_weight: float, horizon: int, W_stage: jnp.ndarray, W_terminal: jnp.ndarray, goal_state: jnp.ndarray):
    def cost(W_ignored, reference, z, u, t):
        is_not_terminal = (t < horizon)
        active_W = jnp.where(is_not_terminal, W_stage, W_terminal)
        active_ref = jnp.where(is_not_terminal, reference[t], goal_state)
        state_error = z - active_ref
        return jnp.sum(active_W * (state_error ** 2)) + action_weight * jnp.sum(u ** 2)
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
    candidates = []
    for path in model_dir.glob("*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates: raise FileNotFoundError(f"No valid checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def hide_target_cube(env) -> None:
    """Zeroes out the alpha transparency channel for the target block visual geoms."""
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

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
    error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
    
    state_dim, action_dim = config_dict.get("markov_state_dim", 48), config_dict.get("action_dim", 5)
    img_size = config_dict.get("img_size", 224)

    init_key = jax.random.PRNGKey(cfg.seed)
    k1, k2 = jax.random.split(init_key)
    eqx_dyn = build_equinox_mlp_from_pytorch(model.predictor.net, k1)
    dynamics = lambda x, u, t=0.0, parameter=1.0: eqx_dyn(jnp.concatenate([x, u], axis=-1))
    disturbance = make_jax_disturbance(build_equinox_mlp_from_pytorch(error_model.net, k2), cfg.q_learned, state_dim, error_model.diagonal)

    with h5py.File(cfg.dataset_path, "r") as h5:
        episode_idx = cfg.episode_idx if cfg.episode_idx is not None else int(np.random.choice(np.flatnonzero(h5["ep_len"][:] >= 2)))
        rows = np.arange(int(h5["ep_offset"][episode_idx]), int(h5["ep_offset"][episode_idx]) + int(h5["ep_len"][episode_idx]))
        qpos_init, qpos_goal = h5["qpos"][rows[0]], h5["qpos"][rows[-1]]
        qvel_init, qvel_goal = h5["qvel"][rows[0]], h5["qvel"][rows[-1]]

    env = gymnasium.make("cube-single-v0", terminate_at_goal=False, mode="data_collection", width=256, height=256)
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

    # --- 1. Fetch goal image sequence & propagate physics before capturing frame ---
    env.reset(seed=cfg.seed)
    hide_target_cube(env)
    env.unwrapped._data.qpos[:qpos_goal.shape[0]] = qpos_goal.astype(np.float64)
    env.unwrapped._data.qvel[:qvel_goal.shape[0]] = qvel_goal.astype(np.float64)
    env.unwrapped.pre_step()
    mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)  # Force visual state sync
    env.unwrapped.post_step()
    goal_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)

    # --- 2. Fetch initial image sequence & propagate physics before capturing frame ---
    env.reset(seed=cfg.seed)
    hide_target_cube(env)
    env.unwrapped._data.qpos[:qpos_init.shape[0]] = qpos_init.astype(np.float64)
    env.unwrapped._data.qvel[:qvel_init.shape[0]] = qvel_init.astype(np.float64)
    env.unwrapped.pre_step()
    mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)  # Force visual state sync
    env.unwrapped.post_step()
    current_frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)
    current_info = env.unwrapped.get_step_info()

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
        rollout_frames.append(np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8))

    if not grasped: return env.close()

    W_mppi = jnp.ones((state_dim,)) * 100
    W_mppi = W_mppi.at[state_dim // 2:].set(1.0)
    W_stage_scaled = jnp.ones((state_dim,)) * 10.0
    W_stage_scaled = W_stage_scaled.at[state_dim // 2:].set(1.0)
    W_terminal_scaled = jnp.ones((state_dim,)) * 5000.0
    W_terminal_scaled = W_terminal_scaled.at[state_dim // 2:].set(1.0)

    mppi_roll, mppi_ev = make_mppi_rollout_and_eval(dynamics, W_mppi, jnp.asarray(goal_state))
    
    mppi_planner = MPPIPlanner(
        config={"planning": {"action_dim": action_dim, "n_sample": cfg.mppi_samples, "horizon": cfg.horizon, "n_update_iter": cfg.mppi_update_iter, "use_last": True, "reject_bad": False, "mppi": {"reward_weight": cfg.mppi_reward_weight, "noise_level": cfg.mppi_noise_level, "noise_decay": 1.0, "beta_filter": cfg.mppi_beta_filter}}},
        model_rollout_fn=mppi_roll, evaluate_traj_fn=mppi_ev, action_lower_lim=-2.0*jnp.ones(action_dim), action_upper_lim=2.0*jnp.ones(action_dim)
    )

    @eqx.filter_jit
    def run_mppi_opt(key_arg, state_arg, actions_arg):
        return mppi_planner.trajectory_optimization(key_arg, state_arg, actions_arg, skip=False)

    cost = make_tracking_cost(
        action_weight=0.01, 
        horizon=cfg.horizon, 
        W_stage=W_stage_scaled,
        W_terminal=W_terminal_scaled,
        goal_state=jnp.asarray(goal_state),
    )

    x_min, x_max = -100.0 * jnp.ones(state_dim), 100.0 * jnp.ones(state_dim)
    u_min, u_max = -100.0 * jnp.ones(action_dim), 100.0 * jnp.ones(action_dim)
    constraints_all = combine_constraints(make_state_box_constraints(x_min, x_max), make_control_box_constraints(u_min, u_max))

    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=False, initialize_nominal=True, warm_start=True, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=300, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage_scaled, u_ref=jnp.zeros(action_dim), dt=1.0/20.0),
        dynamics=dynamics, constraints=constraints_all, obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim,
        disturbance=disturbance, shift=1, X_in=jnp.zeros((cfg.horizon + 1, state_dim)), U_in=jnp.zeros((cfg.horizon, action_dim))
    )

    current_emb = encode_single_frame(model, rollout_frames[-1], device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    prev_U, jax_seed = jnp.zeros((cfg.horizon, action_dim)), jax.random.PRNGKey(cfg.seed)
    train_stats = LeWMOGBenchCubeDataset(
        str(cfg.dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim
    )

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="Refined MPPI + SLS tracking loops")
    for _ in mpc_pbar:
        jax_seed, subkey = jax.random.split(jax_seed)
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
        
        mppi_res = run_mppi_opt(subkey, jnp.asarray(current_state), init_act_seq)
        X_ws = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
        
        controller.X_in, controller.U_in = X_ws, jnp.asarray(mppi_res["act_seq"])
        try:
            u0, X_pred, U_pred, *_ = controller.run(x0=current_state, reference=X_ws, parameter=1.0/20.0)
            status = "sls_refined"
        except Exception:
            u0, U_pred = mppi_res["act_seq"][0], mppi_res["act_seq"]
            status = "fallback"

        prev_U = U_pred
        u_raw = ((np.asarray(u0, dtype=np.float32) * train_stats.action_std.flatten()) + train_stats.action_mean.flatten()).astype(np.float32)
        current_info = env.step(u_raw)[4]
        
        frame = np.asarray(env.unwrapped.render(camera="front_pixels"), dtype=np.uint8)
        rollout_frames.append(frame)

        next_emb = encode_single_frame(model, frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        lat_err = float(np.linalg.norm(current_state - goal_state))
        mpc_pbar.set_postfix(latent_err=f"{lat_err:.4f}", status=status)
        if lat_err <= 0.05: break

    imageio.mimwrite(out_dir / "cube_mppi_sls.mp4", rollout_frames, fps=cfg.video_fps)
    env.close()

if __name__ == "__main__":
    main()