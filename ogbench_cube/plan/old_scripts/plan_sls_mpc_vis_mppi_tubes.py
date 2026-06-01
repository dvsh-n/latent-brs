#!/usr/bin/env python3
"""Plan in OGBench space using non-RTI Conformal SLS MPC warmstarted by MPPI, with tube visualizations."""

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
# if "MPLCONFIGDIR" not in os.environ:
#     mpl_config_dir = Path("/tmp/codex_mplconfig")
#     mpl_config_dir.mkdir(parents=True, exist_ok=True)
#     os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gymnasium
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import mujoco
from tqdm.auto import tqdm
import pyrallis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    """Configuration for Warmstarted non-RTI Conformal SLS MPC tube visualization on OGBench Cubes"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("ogbench_cube/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("ogbench_cube/models/error_model/best-error-model.ckpt"))
    dataset_path: Path = field(default=Path("ogbench_cube/data/test_data/ogbench_cube_test.h5"))
    out_dir: Path = field(default=Path("ogbench_cube/plan/sls_mppi_conformal_tube_vis"))
    device: str = field(default="auto")
    horizon: int = field(default=16)
    max_mpc_steps: int = field(default=120)
    max_oracle_steps: int = field(default=80)
    video_fps: int = field(default=20)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    visualize_success_colors: bool = field(default=False)
    terminate_on_ogbench_success: bool = field(default=True)
    vis_every: int = field(default=1)
    
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

def ogbench_success(info: dict) -> bool:
    success = info.get("success", False)
    if isinstance(success, dict):
        return all(bool(value) for value in success.values())
    return bool(np.asarray(success).item())

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def save_rollout_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4_path = out_dir / "cube_mppi_sls.mp4"
    gif_path = out_dir / "cube_mppi_sls.gif"
    try:
        imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4_path
    except Exception:
        imageio.mimwrite(gif_path, frames, fps=fps)
        return gif_path

def plot_planner_diagnostics(
    path: Path,
    *,
    step_idx: int,
    mppi_states: np.ndarray,
    sls_states: Optional[np.ndarray],
    mppi_actions: np.ndarray,
    sls_actions: Optional[np.ndarray],
    goal_state: np.ndarray,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mppi_err = np.linalg.norm(mppi_states - goal_state[None, :], axis=-1)
    sls_err = None if sls_states is None else np.linalg.norm(sls_states - goal_state[None, :], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].plot(np.arange(len(mppi_err)), mppi_err, label="MPPI warmstart", color="tab:orange", linewidth=2.0)
    if sls_err is not None:
        axes[0].plot(np.arange(len(sls_err)), sls_err, label="SLS non-RTI prediction", color="tab:blue", linewidth=2.0)
    axes[0].set_title(f"Latent distance to goal, step {step_idx:03d}")
    axes[0].set_xlabel("Horizon index")
    axes[0].set_ylabel("L2 distance")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    action_horizon = np.arange(mppi_actions.shape[0])
    for action_idx in range(mppi_actions.shape[1]):
        axes[1].plot(action_horizon, mppi_actions[:, action_idx], color="tab:orange", alpha=0.35, linewidth=1.2)
    if sls_actions is not None:
        for action_idx in range(sls_actions.shape[1]):
            axes[1].plot(np.arange(sls_actions.shape[0]), sls_actions[:, action_idx], color="tab:blue", alpha=0.45, linewidth=1.2)
    axes[1].set_title(f"Normalized actions ({status})")
    axes[1].set_xlabel("Horizon index")
    axes[1].set_ylabel("Action")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def save_frame_panel(path: Path, start_frame: np.ndarray, current_frame: np.ndarray, goal_frame: np.ndarray, step_idx: int, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    panels = [("Start", start_frame), (f"Step {step_idx:03d} ({status})", current_frame), ("Goal", goal_frame)]
    for ax, (title, frame) in zip(axes, panels):
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def save_tube_width_plot(path: Path, tube: np.ndarray, step_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dim = tube.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    horizon_axis = np.arange(tube.shape[0])

    for dim_idx in range(state_dim):
        ax = axes[dim_idx]
        ax.plot(horizon_axis, tube[:, dim_idx], color="tab:blue", linewidth=1.5)
        ax.set_title(f"Dim {dim_idx}", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.tick_params(axis="both", which="major", labelsize=8)

    for dim_idx in range(state_dim, len(axes)):
        axes[dim_idx].axis("off")

    fig.suptitle(f"Projected Tube Widths Across Horizon (MPC Step {step_idx})", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

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
    out_dir = cfg.out_dir.expanduser().resolve() / f"{int(time.time())}_mppi_sls_cube_tube_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = out_dir / "planner_diagnostics"
    frame_panels_dir = out_dir / "frame_panels"
    tube_plots_dir = out_dir / "tube_plots"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    frame_panels_dir.mkdir(parents=True, exist_ok=True)
    tube_plots_dir.mkdir(parents=True, exist_ok=True)

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
        target_block_pos_init = h5["target_block_pos"][rows[0]]
        target_block_yaw_init = float(h5["target_block_yaw"][rows[0], 0])
        target_block_pos_goal = h5["target_block_pos"][rows[-1]]
        target_block_yaw_goal = float(h5["target_block_yaw"][rows[-1], 0])

    env = gymnasium.make(
        "cube-single-v0",
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=cfg.visualize_success_colors,
        width=256,
        height=256,
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

    goal_frame, _ = reset_env_to_state(
        env,
        seed=cfg.seed,
        qpos=qpos_goal,
        qvel=qvel_goal,
        target_block_pos=target_block_pos_goal,
        target_block_yaw=target_block_yaw_goal,
        camera="front_pixels",
    )

    current_frame, current_info = reset_env_to_state(
        env,
        seed=cfg.seed,
        qpos=qpos_init,
        qvel=qvel_init,
        target_block_pos=target_block_pos_init,
        target_block_yaw=target_block_yaw_init,
        camera="front_pixels",
    )

    pixel_mean, pixel_std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    goal_emb = encode_single_frame(model, goal_frame, device, img_size, pixel_mean, pixel_std)
    goal_state = torch.cat([goal_emb, torch.zeros_like(goal_emb)], dim=-1).cpu().numpy().astype(np.float64)

    save_rgb_image(out_dir / "start_image.png", current_frame)
    save_rgb_image(out_dir / "goal_image.png", goal_frame)

    rollout_frames = [current_frame.copy()]
    grasped = False

    oracle.reset(None, current_info)
    
    # Run analytical planner tracking stages until containment validation triggers
    for _ in range(cfg.max_oracle_steps):
        if cube_is_grasped(current_info, cfg.grasp_contact_threshold, cfg.grasp_alignment_threshold): grasped = True; break
        current_info = env.step(np.asarray(oracle.select_action(None, current_info), dtype=np.float32))[4]
        rollout_frames.append(render_without_target_cube(env, "front_pixels"))

    if not grasped:
        save_rollout_video(rollout_frames, out_dir, fps=cfg.video_fps)
        env.close()
        return

    W_mppi = jnp.ones((state_dim,)) * 100
    W_mppi = W_mppi.at[state_dim // 2:].set(1.0)
    W_stage_scaled = jnp.ones((state_dim,)) * 10.0
    W_stage_scaled = W_stage_scaled.at[state_dim // 2:].set(10.0)
    W_terminal_scaled = jnp.ones((state_dim,)) * 1.0
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
    u_min, u_max = -10.0 * jnp.ones(action_dim), 10.0 * jnp.ones(action_dim)
    constraints_all = combine_constraints(make_state_box_constraints(x_min, x_max), make_control_box_constraints(u_min, u_max))

    controller = GenericMPC(
        SLSConfig(
            max_sls_iterations=1,
            sls_primal_tol=1e-2,
            enable_fastsls=True,
            max_initial_sqp_iterations=0,
            initialize_nominal=True,
            warm_start=True,
            rti=False,
            R_bar=None,
            Q_bar=None,
        ),
        SQPConfig(max_sqp_iterations=3, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=200, rho_update_frequency=20, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage_scaled, u_ref=jnp.zeros(action_dim), dt=1.0/20.0),
        dynamics=dynamics, constraints=constraints_all, obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim,
        disturbance=disturbance, shift=1, X_in=jnp.zeros((cfg.horizon + 1, state_dim)), U_in=jnp.zeros((cfg.horizon, action_dim))
    )

    current_emb = encode_single_frame(model, rollout_frames[-1], device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)
    prev_U, jax_seed = jnp.zeros((cfg.horizon, action_dim)), jax.random.PRNGKey(cfg.seed)
    trace_rows = []
    train_stats = LeWMOGBenchCubeDataset(
        str(cfg.dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim
    )

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="Refined MPPI + SLS tracking loops")
    for mpc_step in mpc_pbar:
        jax_seed, subkey = jax.random.split(jax_seed)
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
        
        mppi_res = run_mppi_opt(subkey, jnp.asarray(current_state), init_act_seq)
        X_mppi = jnp.asarray(mppi_res["state_seq"])
        U_mppi = jnp.asarray(mppi_res["act_seq"])
        X_ws = jnp.concatenate([jnp.asarray(current_state)[None, :], X_mppi], axis=0)
        X_ws_np = np.asarray(X_ws, dtype=np.float64)
        
        controller.X_in, controller.U_in = X_ws, U_mppi

        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_ws, parameter=1.0/20.0)
            if u0 is None or not jnp.all(jnp.isfinite(X_pred)) or not jnp.all(jnp.isfinite(U_pred)):
                print("\n[WARN] Non-RTI SLS returned NaN/non-finite values. Using MPPI fallback.")
                u0, U_pred = U_mppi[0], U_mppi
                status = "mppi_fallback"
                sls_states_np = None
                sls_actions_np = None
                solver_info = []
            else:
                status = "sls_non_rti"
                sls_states_np = np.asarray(X_pred, dtype=np.float64)
                sls_actions_np = np.asarray(U_pred, dtype=np.float64)
        except Exception as e:
            print(f"\n[WARN] GenericMPC solve raised exception: {e}")
            u0, U_pred = U_mppi[0], U_mppi
            status = "mppi_fallback"
            sls_states_np = None
            sls_actions_np = None
            solver_info = []

        tube_mean = np.nan
        tube_max = np.nan
        if sls_states_np is not None and len(solver_info) >= 3:
            Phi_x = solver_info[2]
            tube = np.asarray(jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1), dtype=np.float64)
            tube_mean = float(np.mean(tube))
            tube_max = float(np.max(tube))
            if cfg.vis_every > 0 and (mpc_step % cfg.vis_every == 0):
                np.save(tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.npy", tube)
                save_tube_width_plot(tube_plots_dir / f"tube_widths_step_{mpc_step:03d}.png", tube, mpc_step)

        prev_U = U_pred
        u_raw = ((np.asarray(u0, dtype=np.float32) * train_stats.action_std.flatten()) + train_stats.action_mean.flatten()).astype(np.float32)
        current_info = env.step(u_raw)[4]
        reached_ogbench_success = ogbench_success(current_info)
        
        frame = render_without_target_cube(env, "front_pixels")
        rollout_frames.append(frame)

        next_emb = encode_single_frame(model, frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        lat_err = float(np.linalg.norm(current_state - goal_state))
        if reached_ogbench_success:
            status = "ogbench_success"
        trace_rows.append(
            {
                "step": mpc_step,
                "latent_error": lat_err,
                "status": status,
                "tube_mean": tube_mean,
                "tube_max": tube_max,
                "u0_norm": np.asarray(u0, dtype=np.float32),
                "u0_raw": u_raw,
            }
        )
        if cfg.vis_every > 0 and (mpc_step % cfg.vis_every == 0):
            plot_planner_diagnostics(
                diagnostics_dir / f"planner_step_{mpc_step:03d}.png",
                step_idx=mpc_step,
                mppi_states=X_ws_np,
                sls_states=sls_states_np,
                mppi_actions=np.asarray(U_mppi, dtype=np.float64),
                sls_actions=sls_actions_np,
                goal_state=goal_state,
                status=status,
            )
            save_frame_panel(
                frame_panels_dir / f"frames_step_{mpc_step:03d}.png",
                start_frame=rollout_frames[0],
                current_frame=frame,
                goal_frame=goal_frame,
                step_idx=mpc_step,
                status=status,
            )
        mpc_pbar.set_postfix(latent_err=f"{lat_err:.4f}", status=status)
        if cfg.terminate_on_ogbench_success and reached_ogbench_success: break
        if lat_err <= 0.05: break

    save_rollout_video(rollout_frames, out_dir, fps=cfg.video_fps)
    if trace_rows:
        np.savez(
            out_dir / "planner_trace.npz",
            step=np.asarray([row["step"] for row in trace_rows], dtype=np.int64),
            latent_error=np.asarray([row["latent_error"] for row in trace_rows], dtype=np.float64),
            status=np.asarray([row["status"] for row in trace_rows]),
            tube_mean=np.asarray([row["tube_mean"] for row in trace_rows], dtype=np.float64),
            tube_max=np.asarray([row["tube_max"] for row in trace_rows], dtype=np.float64),
            u0_norm=np.stack([row["u0_norm"] for row in trace_rows]),
            u0_raw=np.stack([row["u0_raw"] for row in trace_rows]),
        )
    env.close()

if __name__ == "__main__":
    main()
