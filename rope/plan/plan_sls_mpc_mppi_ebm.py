#!/usr/bin/env python3
"""Plan in Rope pixel space using Conformal SLS MPC warmstarted by MPPI with an EBM domain constraint."""

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

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints
from gpu_sls.mppi_planner import MPPIPlanner

from rope.train.mlpdyn_train import LeWMRopeDataset, preprocess_pixels
from error_model import MGNLLPredictor

@dataclass
class PlanSLSMoppiRopeConfig:
    """Configuration for Warmstarted Conformal SLS MPC on Rope Lines with an EBM domain constraint"""
    q_learned: float = field(default=0.0)
    model_dir: Path = field(default=Path("rope/models/mlpdyn"))
    error_model_ckpt: Path = field(default=Path("rope/models/error_model/best-error-model.ckpt"))
    use_constant_covariance: bool = field(default=False)
    constant_covariance_path: Path = field(default=Path("rope/eval/fixed_error_covariance.pt"))
    ebm_artifact_path: Path = field(default=Path("rope/models/latent_ebm/model.pt"))
    ebm_penalty_weight: float = 1000.0
    ebm_threshold_scale: float = 1.0
    dataset_path: Path = field(default=Path("rope/data/expert_data/rope_random_cubic_spline.h5"))
    out_dir: Path = field(default=Path("rope/plan/sls_mppi_conformal_ebm"))
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

class JAXEnergyMLP(eqx.Module):
    linear_layers: list
    layer_norm_scales: list
    layer_norm_biases: list
    pair_mean: jax.Array
    pair_std: jax.Array
    energy_threshold: jax.Array

    def __call__(self, pair):
        x = (pair - self.pair_mean) / self.pair_std
        for i, linear in enumerate(self.linear_layers[:-1]):
            x = linear(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
            x = x * self.layer_norm_scales[i] + self.layer_norm_biases[i]
            x = jax.nn.silu(x)
        return self.linear_layers[-1](x).squeeze(-1)

def build_jax_ebm_from_artifact(artifact_path: Path, key: jax.Array, threshold_scale: float = 1.0) -> JAXEnergyMLP:
    artifact = torch.load(artifact_path.expanduser(), map_location="cpu", weights_only=False)
    state_dict = artifact["state_dict"]
    input_dim = int(artifact["input_dim"])
    hidden_dim = int(artifact["hidden_dim"])
    depth = int(artifact["depth"])

    linear_layers = []
    layer_norm_scales = []
    layer_norm_biases = []
    keys = jax.random.split(key, depth + 1)

    module_idx = 0
    current_dim = input_dim
    for i in range(depth):
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

        module_idx += 3
        if f"net.{module_idx}.weight" not in state_dict:
            module_idx += 1
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

    return JAXEnergyMLP(
        linear_layers=linear_layers,
        layer_norm_scales=layer_norm_scales,
        layer_norm_biases=layer_norm_biases,
        pair_mean=jnp.asarray(artifact["pair_mean"], dtype=jnp.float64),
        pair_std=jnp.asarray(artifact["pair_std"], dtype=jnp.float64),
        energy_threshold=jnp.asarray(float(artifact["energy_threshold"]) * float(threshold_scale), dtype=jnp.float64),
    )

def make_ebm_domain_constraint(ebm_model: JAXEnergyMLP):
    def constraint(x, u, t):
        pair = jnp.concatenate([x, u], axis=-1)
        return jnp.asarray([ebm_model(pair) - ebm_model.energy_threshold])
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

def make_mppi_rollout_and_eval(
    jax_dynamics_fn,
    state_dim,
    action_dim,
    horizon,
    W_state,
    goal_state,
    ebm_model=None,
    ebm_penalty_weight: float = 0.0,
):
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
        if ebm_model is not None and ebm_penalty_weight > 0.0:
            pairs = jnp.concatenate([state_seqs, act_seqs], axis=-1)
            energies = jax.vmap(jax.vmap(ebm_model))(pairs)
            ebm_violation = jnp.maximum(energies - ebm_model.energy_threshold, 0.0)
            ebm_costs = ebm_penalty_weight * ebm_violation**2
        else:
            ebm_costs = jnp.zeros_like(stage_costs)
        return {"rewards": -jnp.sum(stage_costs + action_costs + ebm_costs, axis=-1)}

    return mppi_rollout_fn, mppi_eval_fn

def make_control_box_constraints(u_min, u_max):
    u_min, u_max = jnp.asarray(u_min), jnp.asarray(u_max)
    def constraints(x, u, t):
        return jnp.concatenate([u - u_max, u_min - u], axis=0)
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

    init_key = jax.random.PRNGKey(cfg.seed)
    key_dyn, key_err, key_ebm = jax.random.split(init_key, 3)
    dynamics = make_jax_dynamics(build_equinox_mlp_from_pytorch(model.predictor.net, key_dyn))
    ebm_model = build_jax_ebm_from_artifact(cfg.ebm_artifact_path, key_ebm, threshold_scale=cfg.ebm_threshold_scale)
    expected_ebm_dim = state_dim + action_dim
    if ebm_model.pair_mean.shape[0] != expected_ebm_dim:
        raise ValueError(
            f"EBM input_dim={ebm_model.pair_mean.shape[0]} but planner expected state_dim+action_dim={expected_ebm_dim}."
        )
    ebm_constraint = make_ebm_domain_constraint(ebm_model)
    print(
        f"Using EBM domain constraint from {cfg.ebm_artifact_path} "
        f"with threshold {float(ebm_model.energy_threshold):.6g}"
    )
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

    run_dir = out_dir / f"{int(time.time())}_mppi_sls_rope_{episode_idx:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    true_latents = encode_frames(model, pixels_np, device, img_size)
    goal_state = torch.cat([true_latents[-1], true_latents[-1] - true_latents[-2]], dim=-1).cpu().numpy().astype(np.float64)

    W_mppi = jnp.ones((state_dim,)) * 100
    W_mppi = W_mppi.at[state_dim // 2:].set(1.0)
    W_stage = jnp.ones((state_dim,)) * 10.0
    W_stage = W_stage.at[state_dim // 2:].set(1.0)
    W_term = jnp.ones((state_dim,)) * 10.0
    W_term = W_term.at[state_dim // 2:].set(1.0)

    cost = make_tracking_cost(1.0, cfg.horizon, W_term, jnp.asarray(goal_state))

    mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
        dynamics,
        state_dim,
        action_dim,
        cfg.horizon,
        W_mppi,
        jnp.asarray(goal_state),
        ebm_model=ebm_model,
        ebm_penalty_weight=cfg.ebm_penalty_weight,
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
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage, u_ref=jnp.zeros(action_dim), dt=1.0/30.0),
        dynamics=dynamics,
        constraints=combine_constraints(ebm_constraint, make_control_box_constraints(-5.0*jnp.ones(action_dim), 5.0*jnp.ones(action_dim))),
        obstacles=jnp.zeros((0, 3)), cost=cost, num_constraints=2 * action_dim + 1, disturbance=disturbance, shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim), dtype=jnp.float64), U_in=jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    )

    current_frame = pixels_np[0].copy()
    current_state = torch.cat([true_latents[0], torch.zeros_like(true_latents[0])], dim=-1).cpu().numpy().astype(np.float64)
    rollout_frames = [current_frame.copy()]
    prev_U = jnp.zeros((cfg.horizon, action_dim), dtype=jnp.float64)
    jax_seed_key = jax.random.PRNGKey(cfg.seed)

    pbar = tqdm(range(cfg.max_mpc_steps), desc="Receding Horizon MPPI + SLS Sequence Loops")
    for step_idx in pbar:
        jax_seed_key, subkey = jax.random.split(jax_seed_key)
        init_act_seq = jnp.concatenate([prev_U[1:], prev_U[-1:]], axis=0)
        
        mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state), init_act_seq)
        X_warmstart = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
        
        controller.X_in = X_warmstart
        controller.U_in = jnp.asarray(mppi_res["act_seq"])

        current_pair_energy = ebm_model(jnp.concatenate([jnp.asarray(current_state), jnp.zeros(action_dim)]))
        warmstart_pairs = jnp.concatenate([X_warmstart[:-1], controller.U_in], axis=-1)
        warmstart_energies = jax.vmap(ebm_model)(warmstart_pairs)
        print(
            f"EBM energy step={step_idx:03d}: "
            f"current_zero_u={float(current_pair_energy):.3f}, "
            f"warmstart_min={float(jnp.min(warmstart_energies)):.3f}, "
            f"warmstart_max={float(jnp.max(warmstart_energies)):.3f}, "
            f"threshold={float(ebm_model.energy_threshold):.3f}"
        )

        try:
            u0, X_pred, U_pred, *solver_info = controller.run(x0=current_state, reference=X_warmstart, parameter=1.0/30.0)
            solver_status = "sls_refined"
        except Exception:
            u0, X_pred, U_pred = mppi_res["act_seq"][0], X_warmstart, mppi_res["act_seq"]
            solver_status = "mppi_fallback"

        prev_U = jnp.concatenate([U_pred[1:], U_pred[-1:]], axis=0)
        
        sim_index = min(step_idx + 1, len(pixels_np) - 1)
        current_frame = pixels_np[sim_index].copy()
        rollout_frames.append(current_frame.copy())

        current_state = torch.cat([true_latents[sim_index], true_latents[sim_index] - true_latents[sim_index-1]], dim=-1).cpu().numpy().astype(np.float64)

        latent_err = float(np.linalg.norm(current_state - goal_state))
        pbar.set_postfix(lat_err=f"{latent_err:.3f}", status=solver_status)
        if sim_index == len(pixels_np) - 1: break

    imageio.mimwrite(run_dir / "mppi_sls_rope.mp4", rollout_frames, fps=cfg.video_fps)
    print(f"Rollout successfully complete. Artifacts written to {run_dir}")

if __name__ == "__main__":
    main()
