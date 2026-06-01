#!/usr/bin/env python3
"""Plan in Rope pixel space using RTI Conformal SLS MPC warmstarted by MPPI, with tube visualizations."""

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
from gpu_sls.utils.constraint_utils import combine_constraints, make_state_box_constraints
from gpu_sls.mppi_planner import MPPIPlanner

from rope.train.mlpdyn_train import LeWMRopeDataset, build_markov_state, preprocess_pixels, required_markov_history
from rope.shared.lab_env import LabEnv, TaskState
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiRopeConfig:
    """Configuration for Warmstarted RTI Conformal SLS MPC tube visualization on Rope Lines"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("rope/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("rope/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("rope/eval/fixed_error_covariance.pt"))
    track_fixed_covariance_coverage: bool = field(default=True)
    dataset_path: Path = field(default=Path("rope/data/expert_data/rope_random_cubic_spline.h5"))
    out_dir: Path = field(default=Path("rope/plan/sls_mppi_conformal_rti_tube_vis"))
    device: str = field(default="auto")
    horizon: int = field(default=24)
    max_mpc_steps: int = field(default=150)
    video_fps: int = field(default=30)
    episode_idx: Optional[int] = field(default=None)
    seed: int = field(default=42)
    vis_every: int = field(default=1)
    
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

def fixed_ellipsoid_score(calibrated_cholesky: np.ndarray, error: np.ndarray) -> float:
    whitened = np.linalg.solve(calibrated_cholesky, np.asarray(error, dtype=np.float64))
    return float(np.linalg.norm(whitened, ord=2))

def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (np.asarray(action_norm, dtype=np.float64) * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float64)

def render_rgb_frame(renderer: mujoco.Renderer, env: LabEnv, camera_id: int, *, disable_shadows: bool) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    if disable_shadows:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    return np.asarray(renderer.render(), dtype=np.uint8).copy()

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
    return frame, {
        "task_target": env.task_controller.desired_state.as_array().astype(np.float32),
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }

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
    return frame, {
        "task_target": env.task_controller.desired_state.as_array().astype(np.float32),
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }

def markov_state_at(latents: torch.Tensor, index: int, markov_deriv: int) -> np.ndarray:
    history_len = required_markov_history(markov_deriv)
    start_idx = max(0, index - history_len + 1)
    history = latents[start_idx : index + 1]
    if history.shape[0] < history_len:
        padding_amt = history_len - history.shape[0]
        history = torch.cat((history[:1].repeat(padding_amt, 1), history), dim=0)
    return build_markov_state(history.unsqueeze(0), markov_deriv)[0].cpu().numpy().astype(np.float64)

def markov_state_from_latent_history(latents: list[torch.Tensor], markov_deriv: int) -> np.ndarray:
    history_len = required_markov_history(markov_deriv)
    history = torch.stack(latents[-history_len:], dim=0)
    if history.shape[0] < history_len:
        padding_amt = history_len - history.shape[0]
        history = torch.cat((history[:1].repeat(padding_amt, 1), history), dim=0)
    return build_markov_state(history.unsqueeze(0), markov_deriv)[0].cpu().numpy().astype(np.float64)

def make_mppi_rollout_and_eval(
    jax_dynamics_fn,
    state_dim,
    action_dim,
    horizon,
    W_state,
    goal_state,
    box_min=None,
    box_max=None,
    box_penalty_weight: float = 0.0,
):
    if box_min is not None:
        box_min = jnp.asarray(box_min)
    if box_max is not None:
        box_max = jnp.asarray(box_max)

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
        action_costs = 1.0 * jnp.sum(act_seqs ** 2, axis=-1)
        if box_min is not None and box_max is not None and box_penalty_weight > 0.0:
            lower_violation = jnp.maximum(box_min[None, None, :] - state_seqs, 0.0)
            upper_violation = jnp.maximum(state_seqs - box_max[None, None, :], 0.0)
            box_costs = box_penalty_weight * jnp.sum(lower_violation**2 + upper_violation**2, axis=-1)
        else:
            box_costs = jnp.zeros_like(stage_costs)
        return {"rewards": -jnp.sum(stage_costs + action_costs + box_costs, axis=-1)}

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

def make_tracking_cost(action_weight: float, horizon: int, W_term: jnp.ndarray, goal_state: jnp.ndarray):
    def cost(W, reference, z, u, t):
        is_not_terminal = (t < horizon)
        dz = z - jnp.where(is_not_terminal, reference[t], goal_state)
        return jnp.sum(jnp.where(is_not_terminal, W, W_term) * dz**2) + action_weight * jnp.sum(u**2)
    return cost

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = [ (int(m.group(1)), p) for p in model_dir.glob("*_epoch_*_object.ckpt") for m in [pattern.match(p.name)] if m ]
    if not candidates: raise FileNotFoundError(f"No object checkpoints in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def resolve_action_stats_dataset(config_dict: dict, fallback_dataset_path: Path) -> Path:
    candidates = [
        Path(str(config_dict.get("dataset_path", ""))).expanduser(),
        fallback_dataset_path.expanduser(),
        Path("rope/data/test_data_noshadow/rope_random_cubic_spline.h5"),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find an action-statistics dataset. Tried: "
        + ", ".join(str(candidate) for candidate in candidates if str(candidate))
    )

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))

def save_rollout_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4_path = out_dir / "mppi_sls_rope.mp4"
    gif_path = out_dir / "mppi_sls_rope.gif"
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
        axes[0].plot(np.arange(len(sls_err)), sls_err, label="SLS RTI prediction", color="tab:blue", linewidth=2.0)
    axes[0].set_title(f"Latent distance to goal, step {step_idx:03d}")
    axes[0].set_xlabel("Horizon index")
    axes[0].set_ylabel("L2 distance")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    for action_idx in range(mppi_actions.shape[1]):
        axes[1].plot(np.arange(mppi_actions.shape[0]), mppi_actions[:, action_idx], color="tab:orange", alpha=0.35, linewidth=1.2)
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

def save_tube_bounds_plot(path: Path, tube: np.ndarray, nominal_states: np.ndarray, step_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    horizon_len = min(tube.shape[0], nominal_states.shape[0])
    tube = tube[:horizon_len]
    nominal_states = nominal_states[:horizon_len]
    state_dim = tube.shape[1]
    n_cols = 6
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    horizon_axis = np.arange(horizon_len)

    for dim_idx in range(state_dim):
        ax = axes[dim_idx]
        nominal = nominal_states[:, dim_idx]
        width = tube[:, dim_idx]
        ax.fill_between(horizon_axis, nominal - width, nominal + width, color="tab:blue", alpha=0.22, linewidth=0.0)
        ax.plot(horizon_axis, nominal, color="tab:blue", linewidth=1.5)
        ax.set_title(f"Dim {dim_idx}", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.tick_params(axis="both", which="major", labelsize=8)

    for dim_idx in range(state_dim, len(axes)):
        axes[dim_idx].axis("off")

    fig.suptitle(f"RTI Nominal Latent Plan With Projected Tube Bounds (MPC Step {step_idx})", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

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
    action_dim = config_dict.get("action_dim", 5)
    img_size = config_dict.get("img_size", 224)
    markov_deriv = int(config_dict.get("markov_deriv", 1))
    frameskip = int(config_dict.get("frameskip", 1))

    init_key = jax.random.PRNGKey(cfg.seed)
    key_dyn, key_err = jax.random.split(init_key)
    dynamics = make_jax_dynamics(build_equinox_mlp_from_pytorch(model.predictor.net, key_dyn))
    fixed_coverage_cholesky = None
    if cfg.use_constant_covariance or cfg.track_fixed_covariance_coverage:
        fixed_coverage_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)

    if cfg.use_constant_covariance:
        calibrated_cholesky = fixed_coverage_cholesky
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

    action_stats_dataset_path = resolve_action_stats_dataset(config_dict, cfg.dataset_path)
    print(f"Using action statistics from {action_stats_dataset_path}")
    action_stats_dataset = LeWMRopeDataset(
        action_stats_dataset_path,
        markov_deriv=markov_deriv,
        num_preds=1,
        frameskip=frameskip,
        img_size=img_size,
        action_dim=action_dim,
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
    action_mean = action_stats_dataset.action_mean.astype(np.float64)
    action_std = action_stats_dataset.action_std.astype(np.float64)

    run_dir = out_dir / f"{int(time.time())}_mppi_sls_rope_rti_tube_vis_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = run_dir / "planner_diagnostics"
    frame_panels_dir = run_dir / "frame_panels"
    tube_plots_dir = run_dir / "tube_plots"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    frame_panels_dir.mkdir(parents=True, exist_ok=True)
    tube_plots_dir.mkdir(parents=True, exist_ok=True)

    true_latents = encode_frames(model, pixels_np, device, img_size)
    goal_state = markov_state_at(true_latents, len(true_latents) - 1, markov_deriv)
    save_rgb_image(run_dir / "start_image.png", pixels_np[0])
    save_rgb_image(run_dir / "goal_image.png", pixels_np[-1])

    W_mppi = jnp.ones((state_dim,)) * 100
    W_mppi = W_mppi.at[state_dim // 2:].set(1.0)
    W_stage = jnp.ones((state_dim,)) * 10.0
    W_stage = W_stage.at[state_dim // 2:].set(1.0)
    W_term = jnp.ones((state_dim,)) * 10.0
    W_term = W_term.at[state_dim // 2:].set(1.0)

    cost = make_tracking_cost(1.0, cfg.horizon, W_term, jnp.asarray(goal_state))

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
        box_min=box_min,
        box_max=box_max,
        box_penalty_weight=cfg.mppi_state_box_penalty,
    )

    mppi_planner = MPPIPlanner(
        config={"planning": {"action_dim": action_dim, "n_sample": cfg.mppi_samples, "horizon": cfg.horizon, "n_update_iter": cfg.mppi_update_iter, "use_last": True, "reject_bad": False, "mppi": {"reward_weight": cfg.mppi_reward_weight, "noise_level": cfg.mppi_noise_level, "noise_decay": 1.0, "beta_filter": cfg.mppi_beta_filter}}},
        model_rollout_fn=mppi_rollout,
        evaluate_traj_fn=mppi_eval,
        action_lower_lim=-2.0 * jnp.ones(action_dim), action_upper_lim=2.0 * jnp.ones(action_dim)
    )
    jit_mppi_trajopt = jax.jit(lambda k, s, a: mppi_planner.trajectory_optimization(k, s, a, skip=False))

    # SLS Setup footprint
    sls_cfg = SLSConfig(max_sls_iterations=1, sls_primal_tol=1e-2, enable_fastsls=True, initialize_nominal=True, warm_start=False, rti=True)
    controller = GenericMPC(
        sls_cfg, SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=True),
        ADMMConfig(eps_abs=5e-2, eps_rel=1e-4, rho_max=1e4, max_iterations=400, rho_update_frequency=20, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=jnp.zeros(action_dim), dt=1.0/30.0),
        dynamics=dynamics, constraints=combine_constraints(make_state_box_constraints(box_min, box_max), make_control_box_constraints(-5.0*jnp.ones(action_dim), 5.0*jnp.ones(action_dim))),
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 2 * state_dim, disturbance=disturbance, shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim), dtype=jnp.float64), U_in=jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    )

    rollout_frames = []
    prev_U = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    jax_seed_key = jax.random.PRNGKey(cfg.seed)
    trace_rows = []
    fixed_coverage_total = 0
    fixed_coverage_covered = 0
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
        latent_history = [current_latent]
        current_state = markov_state_from_latent_history(latent_history, markov_deriv)
        rollout_frames.append(current_frame.copy())

        pbar = tqdm(range(cfg.max_mpc_steps), desc="Receding Horizon MPPI + SLS Sequence Loops")
        for step_idx in pbar:
            jax_seed_key, subkey = jax.random.split(jax_seed_key)
            init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
            
            mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state), init_act_seq)
            X_mppi = jnp.asarray(mppi_res["state_seq"])
            U_mppi = jnp.asarray(mppi_res["act_seq"])
            X_warmstart = jnp.concatenate([jnp.asarray(current_state)[None, :], X_mppi], axis=0)
            X_warmstart_np = np.asarray(X_warmstart, dtype=np.float64)
            
            controller.X_in = X_warmstart
            controller.U_in = U_mppi

            try:
                u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_warmstart, parameter=1.0/30.0)
                if u0 is None or not jnp.all(jnp.isfinite(X_pred)) or not jnp.all(jnp.isfinite(U_pred)):
                    print("\n[WARN] RTI SLS returned NaN/non-finite values. Using MPPI fallback.")
                    u0, X_pred, U_pred = U_mppi[0], X_warmstart, U_mppi
                    solver_status = "mppi_fallback"
                    sls_states_np = None
                    sls_actions_np = None
                    solver_info = []
                else:
                    solver_status = "sls_rti"
                    sls_states_np = np.asarray(X_pred, dtype=np.float64)
                    sls_actions_np = np.asarray(U_pred, dtype=np.float64)
            except Exception as e:
                print(f"\n[WARN] GenericMPC solve raised exception: {e}")
                u0, X_pred, U_pred = U_mppi[0], X_warmstart, U_mppi
                solver_status = "mppi_fallback"
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
                if cfg.vis_every > 0 and (step_idx % cfg.vis_every == 0):
                    np.save(tube_plots_dir / f"tube_widths_step_{step_idx:03d}.npy", tube)
                    save_tube_bounds_plot(tube_plots_dir / f"tube_bounds_step_{step_idx:03d}.png", tube, sls_states_np, step_idx)

            predicted_next_state_planned = np.asarray(
                dynamics(jnp.asarray(current_state), jnp.asarray(u0), 0.0, 1.0),
                dtype=np.float64,
            )
            u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
            u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
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

            next_latent = encode_frames(model, current_frame[None], device, img_size)[0]
            latent_history.append(next_latent)
            next_state_actual = markov_state_from_latent_history(latent_history, markov_deriv)
            fixed_score = np.nan
            fixed_planned_action_score = np.nan
            fixed_covered = False
            if fixed_coverage_cholesky is not None:
                planned_action_error = next_state_actual - predicted_next_state_planned
                fixed_planned_action_score = fixed_ellipsoid_score(fixed_coverage_cholesky, planned_action_error)
                fixed_score = fixed_planned_action_score
                fixed_covered = bool(fixed_score <= 1.0)
                fixed_coverage_total += 1
                fixed_coverage_covered += int(fixed_covered)

            current_state = next_state_actual

            latent_err = float(np.linalg.norm(current_state - goal_state))
            task_err = float(np.linalg.norm(np.asarray(current_info["task_target"], dtype=np.float64) - task_target_np[-1].astype(np.float64)))
            trace_rows.append(
                {
                    "step": step_idx,
                    "latent_error": latent_err,
                    "task_error": task_err,
                    "status": solver_status,
                    "tube_mean": tube_mean,
                    "tube_max": tube_max,
                    "u0_norm": u0_norm.astype(np.float32),
                    "u0_raw": u0_raw.astype(np.float32),
                    "fixed_ellipsoid_score": fixed_score,
                    "fixed_ellipsoid_planned_action_score": fixed_planned_action_score,
                    "fixed_ellipsoid_covered": fixed_covered,
                }
            )
            if cfg.vis_every > 0 and (step_idx % cfg.vis_every == 0):
                plot_planner_diagnostics(
                    diagnostics_dir / f"planner_step_{step_idx:03d}.png",
                    step_idx=step_idx,
                    mppi_states=X_warmstart_np,
                    sls_states=sls_states_np,
                    mppi_actions=np.asarray(U_mppi, dtype=np.float64),
                    sls_actions=sls_actions_np,
                    goal_state=goal_state,
                    status=solver_status,
                )
                save_frame_panel(
                    frame_panels_dir / f"frames_step_{step_idx:03d}.png",
                    start_frame=rollout_frames[0],
                    current_frame=current_frame,
                    goal_frame=pixels_np[-1],
                    step_idx=step_idx,
                    status=solver_status,
                )
            pbar.set_postfix(lat_err=f"{latent_err:.3f}", task_err=f"{task_err:.3f}", status=solver_status, fixed_cov=bool(fixed_covered))
            prev_U = jnp.concatenate([U_pred[1:], U_pred[-1:]], axis=0)
            if latent_err <= 0.05 or task_err <= 1e-3:
                break

    save_rollout_video(rollout_frames, run_dir, fps=cfg.video_fps)
    fixed_coverage_percent = (
        100.0 * fixed_coverage_covered / fixed_coverage_total
        if fixed_coverage_total > 0
        else float("nan")
    )
    if trace_rows:
        np.savez(
            run_dir / "planner_trace.npz",
            step=np.asarray([row["step"] for row in trace_rows], dtype=np.int64),
            latent_error=np.asarray([row["latent_error"] for row in trace_rows], dtype=np.float64),
            task_error=np.asarray([row["task_error"] for row in trace_rows], dtype=np.float64),
            status=np.asarray([row["status"] for row in trace_rows]),
            tube_mean=np.asarray([row["tube_mean"] for row in trace_rows], dtype=np.float64),
            tube_max=np.asarray([row["tube_max"] for row in trace_rows], dtype=np.float64),
            u0_norm=np.stack([row["u0_norm"] for row in trace_rows]),
            u0_raw=np.stack([row["u0_raw"] for row in trace_rows]),
            fixed_ellipsoid_score=np.asarray([row["fixed_ellipsoid_score"] for row in trace_rows], dtype=np.float64),
            fixed_ellipsoid_planned_action_score=np.asarray([row["fixed_ellipsoid_planned_action_score"] for row in trace_rows], dtype=np.float64),
            fixed_ellipsoid_covered=np.asarray([row["fixed_ellipsoid_covered"] for row in trace_rows], dtype=bool),
        )
    coverage_summary = {
        "constant_covariance_path": str(cfg.constant_covariance_path),
        "tracked": bool(fixed_coverage_cholesky is not None),
        "covered": int(fixed_coverage_covered),
        "total": int(fixed_coverage_total),
        "coverage_percent": fixed_coverage_percent,
        "coverage_action_source": "executed normalized planner action",
        "planned_action_score_field": "fixed_ellipsoid_planned_action_score",
    }
    with (run_dir / "fixed_ellipsoid_coverage.json").open("w") as f:
        json.dump(coverage_summary, f, indent=2)
    print(
        f"Fixed ellipsoid one-step coverage: "
        f"{fixed_coverage_covered}/{fixed_coverage_total} ({fixed_coverage_percent:.2f}%)"
    )
    print(f"Rollout successfully complete. Artifacts written to {run_dir}")

if __name__ == "__main__":
    main()
