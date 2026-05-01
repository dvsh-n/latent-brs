#!/usr/bin/env python3
"""Plan in Reacher pixel space with ADMM MPC over a Markov Koopman world model."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.waste.koopdyn_lindec_train import LeWMReacherDataset
from reacher.train.reacher_policy_train import DmControlGymEnv, flatten_observation

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_100hz/reacher_expert_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/lewm_reacher_koop_lindec_markov_100hz_ms20"
DEFAULT_OUT_DIR = "reacher/plan/admm_mpc_koopwm"

DEVICE = "cuda"
GOAL_QPOS_TOL = 1
HORIZON = 25
MAX_MPC_STEPS = 500

Q_TERMINAL = 2.0
R_CONTROL = 0.2

RHO = 10.0
MAX_ADMM_STEPS = 1
ENABLE_WARM_START = True
ENABLE_EARLY_STOP = False
ADMM_PRIMAL_TOL = 1e-4
ADMM_DUAL_TOL = 1e-2

ENABLE_CONSOLE_METRICS = False
ENABLE_PLOTS = True
ENABLE_ROLLOUT_VIDEO = True
ENABLE_START_GOAL_IMAGES = True
SHOW_PLOTS = False

VIDEO_FPS = 60
PLOT_DPI = 160
EPISODE_IDX = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--episode-idx", type=int, default=EPISODE_IDX)
    return parser.parse_args()


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


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


def preprocess_pixels(
    pixels: np.ndarray | torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        tensor = pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    return (tensor - pixel_mean) / pixel_std


@torch.no_grad()
def encode_frames(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    return torch.cat(latents, dim=0)


@torch.no_grad()
def encode_single_frame(
    model: torch.nn.Module,
    pixel: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std).to(device)
    output = model.encoder(batch, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])[0]


def build_markov_state(current_emb: torch.Tensor, previous_emb: torch.Tensor) -> torch.Tensor:
    return torch.cat((current_emb, current_emb - previous_emb), dim=-1)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def raw_to_normalized_bounds(
    low: np.ndarray,
    high: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low_norm = (low - action_mean.reshape(-1)) / action_std.reshape(-1)
    high_norm = (high - action_mean.reshape(-1)) / action_std.reshape(-1)
    return low_norm.astype(np.float32), high_norm.astype(np.float32)


def goal_reached(
    env: DmControlGymEnv,
    goal_qpos: np.ndarray,
) -> tuple[bool, float]:
    physics = env._env.physics
    current_qpos = np.asarray(physics.data.qpos[: goal_qpos.shape[0]], dtype=np.float32)
    qpos_err = float(np.linalg.norm(current_qpos - goal_qpos))
    reached = qpos_err <= GOAL_QPOS_TOL
    return reached, qpos_err


class GPUADMM_MPCSolver:
    def __init__(
        self,
        A_np: np.ndarray,
        B_np: np.ndarray,
        *,
        terminal_output_matrix: np.ndarray,
        terminal_target: np.ndarray,
        q_terminal: float,
        r_control: float,
        u_min: np.ndarray,
        u_max: np.ndarray,
        horizon: int,
        rho: float,
        n_admm_steps: int,
        device: torch.device,
    ) -> None:
        self.H = int(horizon)
        self.nx = int(A_np.shape[0])
        self.nu = int(B_np.shape[1])
        self.rho = float(rho)
        self.steps = int(n_admm_steps)
        self.device = device

        self.A = torch.tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)
        self.C_terminal = torch.tensor(terminal_output_matrix, dtype=torch.float32, device=device)
        self.goal_terminal = torch.tensor(terminal_target, dtype=torch.float32, device=device)
        self.ny_terminal = int(self.C_terminal.shape[0])
        self.Q_terminal = torch.eye(self.ny_terminal, device=device) * float(q_terminal)
        self.R = torch.eye(self.nu, device=device) * (float(r_control) if r_control > 0 else 1.0)
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=device)

        self.nz_total = (self.H + 1) * self.nx
        self.nu_total = self.H * self.nu
        self.n_vars = self.nz_total + self.nu_total
        self.n_eq = (self.H + 1) * self.nx
        self.u_slice = slice(0, self.nu_total)
        self.xN_idx = self.H * self.nu + self.H * self.nx
        self.u_min_tiled = self.u_min.repeat(self.H)
        self.u_max_tiled = self.u_max.repeat(self.H)

        self._warm_sol = torch.zeros(self.n_vars, device=self.device)
        self._warm_Z_u = torch.zeros(self.nu_total, device=self.device)
        self._warm_Y_u = torch.zeros(self.nu_total, device=self.device)

        self._sol = torch.zeros(self.n_vars, device=self.device)
        self._Z_u = torch.zeros(self.nu_total, device=self.device)
        self._Y_u = torch.zeros(self.nu_total, device=self.device)
        self._q_base = torch.zeros(self.n_vars, device=self.device)
        self._q_nominal = torch.empty(self.n_vars, device=self.device)
        self._current_q = torch.empty(self.n_vars, device=self.device)
        self._rhs = torch.empty(self.n_vars + self.n_eq, device=self.device)
        self._z_prev = torch.empty(self.nu_total, device=self.device)
        self._b_eq = torch.zeros(self.n_eq, device=self.device)
        self._u_shift = torch.empty((self.H, self.nu), device=self.device)
        self._z_shift = torch.empty((self.H + 1, self.nx), device=self.device)
        self._zu_shift = torch.empty((self.H, self.nu), device=self.device)
        self._yu_shift = torch.empty((self.H, self.nu), device=self.device)

        self._build_kkt_inverse()

    def _build_kkt_inverse(self) -> None:
        nx, nu, H = self.nx, self.nu, self.H
        p_blocks = []
        r_prox = self.R + self.rho * torch.eye(nu, device=self.device)
        for _ in range(H):
            p_blocks.append(r_prox)

        eye_nx = torch.eye(nx, device=self.device)
        for _ in range(H):
            p_blocks.append(1e-3 * eye_nx)
        terminal_state_cost = self.C_terminal.T @ self.Q_terminal @ self.C_terminal
        p_blocks.append(1e-3 * eye_nx + terminal_state_cost)
        self.P_mat = torch.block_diag(*p_blocks)

        a_eq = torch.zeros((self.n_eq, self.n_vars), device=self.device)
        x_start = H * nu
        a_eq[0:nx, x_start : x_start + nx] = torch.eye(nx, device=self.device)

        for k in range(H):
            row = (k + 1) * nx
            col_u = k * nu
            col_x = x_start + k * nx
            col_x_next = x_start + (k + 1) * nx
            a_eq[row : row + nx, col_u : col_u + nu] = -self.B
            a_eq[row : row + nx, col_x : col_x + nx] = -self.A
            a_eq[row : row + nx, col_x_next : col_x_next + nx] = torch.eye(nx, device=self.device)

        top = torch.cat([self.P_mat, a_eq.T], dim=1)
        bottom = torch.cat([a_eq, torch.zeros((self.n_eq, self.n_eq), device=self.device)], dim=1)
        kkt = torch.cat([top, bottom], dim=0)
        self.KKT_inv = torch.linalg.inv(kkt)

    def _shift_warm_start(self, z0: torch.Tensor) -> None:
        if self.H <= 1:
            return

        u_prev = self._warm_sol[self.u_slice].view(self.H, self.nu)
        z_prev = self._warm_sol[self.nu_total :].view(self.H + 1, self.nx)
        zu_prev = self._warm_Z_u.view(self.H, self.nu)
        yu_prev = self._warm_Y_u.view(self.H, self.nu)

        self._u_shift[:-1] = u_prev[1:]
        self._u_shift[-1] = u_prev[-1]
        self._zu_shift[:-1] = zu_prev[1:]
        self._zu_shift[-1] = zu_prev[-1]
        self._yu_shift[:-1] = yu_prev[1:]
        self._yu_shift[-1] = yu_prev[-1]
        self._z_shift[0] = z0
        self._z_shift[1:-1] = z_prev[2:]
        self._z_shift[-1] = z_prev[-1]

        self._sol[self.u_slice] = self._u_shift.reshape(-1)
        self._sol[self.nu_total :] = self._z_shift.reshape(-1)
        self._Z_u.copy_(self._zu_shift.reshape(-1))
        self._Y_u.copy_(self._yu_shift.reshape(-1))

    def solve_torch(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        if ENABLE_WARM_START:
            self._sol.copy_(self._warm_sol)
            self._Z_u.copy_(self._warm_Z_u)
            self._Y_u.copy_(self._warm_Y_u)
            self._shift_warm_start(z0)
        else:
            self._sol.zero_()
            self._Z_u.zero_()
            self._Y_u.zero_()

        sol = self._sol
        z_u = self._Z_u
        y_u = self._Y_u
        q_nominal = self._q_nominal
        current_q = self._current_q
        rhs = self._rhs
        z_prev = self._z_prev
        b_eq = self._b_eq

        q_nominal.copy_(self._q_base)
        q_nominal[self.xN_idx :] += self.C_terminal.T @ (self.Q_terminal @ (-self.goal_terminal))

        b_eq.zero_()
        b_eq[: self.nx] = z0

        for _ in range(self.steps):
            current_q.copy_(q_nominal)
            current_q[self.u_slice] -= self.rho * (z_u - y_u)
            rhs[: self.n_vars] = -current_q
            rhs[self.n_vars :] = b_eq
            sol_augmented = self.KKT_inv @ rhs

            sol = sol_augmented[: self.n_vars]
            u_sol = sol[self.u_slice]

            z_prev.copy_(z_u)
            torch.clamp(u_sol + y_u, min=self.u_min_tiled, max=self.u_max_tiled, out=z_u)
            y_u.add_(u_sol - z_u)

            if ENABLE_EARLY_STOP:
                primal_res = torch.linalg.vector_norm(u_sol - z_u)
                dual_res = torch.linalg.vector_norm(self.rho * (z_u - z_prev))
                if primal_res <= ADMM_PRIMAL_TOL and dual_res <= ADMM_DUAL_TOL:
                    break

        if ENABLE_WARM_START:
            self._warm_sol.copy_(sol)
            self._warm_Z_u.copy_(z_u)
            self._warm_Y_u.copy_(y_u)

        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0
        z_traj = sol[self.nu_total :].reshape(self.H + 1, self.nx)
        u_traj = sol[: self.nu_total].reshape(self.H, self.nu)
        return z_traj, u_traj, solve_time


def plot_action_traces(actions_raw: np.ndarray, out_path: Path) -> Path:
    fig, axes = plt.subplots(actions_raw.shape[1], 1, figsize=(10, 2.5 * actions_raw.shape[1]), sharex=True)
    if actions_raw.shape[1] == 1:
        axes = [axes]
    steps = np.arange(actions_raw.shape[0])
    for dim, axis in enumerate(axes):
        axis.plot(steps, actions_raw[:, dim], linewidth=1.6)
        axis.set_ylabel(f"u{dim}")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("executed step")
    fig.suptitle("Executed raw actions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def plot_goal_distance_curve(
    latent_distances: np.ndarray,
    embedding_distances: np.ndarray,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = np.arange(latent_distances.shape[0])

    axes[0].plot(steps, latent_distances, linewidth=1.7)
    axes[0].set_ylabel(r"$||z_t - z_g||_2$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, embedding_distances, linewidth=1.7)
    axes[1].set_ylabel(r"$||\hat e_t - e_g||_2$")
    axes[1].set_xlabel("executed step")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Goal-distance curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def plot_terminal_predictions(
    plan_terminal_decoded: np.ndarray,
    goal_decoded: np.ndarray,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = np.arange(plan_terminal_decoded.shape[0])
    state_dim = goal_decoded.shape[0]
    midpoint = state_dim // 2

    for dim in range(midpoint):
        axes[0].plot(steps, plan_terminal_decoded[:, dim], linewidth=1.1, alpha=0.9)
        axes[0].axhline(goal_decoded[dim], color="black", linewidth=0.8, alpha=0.2)
    axes[0].set_ylabel("pred emb dims")
    axes[0].grid(True, alpha=0.25)

    for dim in range(midpoint, state_dim):
        axes[1].plot(steps, plan_terminal_decoded[:, dim], linewidth=1.1, alpha=0.9)
        axes[1].axhline(goal_decoded[dim], color="black", linewidth=0.8, alpha=0.2)
    axes[1].set_ylabel("pred delta dims")
    axes[1].set_xlabel("executed step")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Predicted terminal decoded Markov state vs goal decoded state")
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def save_rollout_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path:
    mp4_path = out_dir / "rollout.mp4"
    gif_path = out_dir / "rollout.gif"
    try:
        imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4_path
    except Exception:
        imageio.mimwrite(gif_path, frames, fps=fps)
        return gif_path


def load_dataset_episode(
    dataset_path: Path,
    episode_idx: int,
) -> dict[str, np.ndarray | int | float]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        return {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "action": np.asarray(h5["action"][rows], dtype=np.float32),
            "observation": np.asarray(h5["observation"][rows], dtype=np.float32),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]),
            "physics_freq_hz": float(h5.attrs.get("physics_freq_hz", 100.0)),
            "time_limit": float(h5.attrs.get("time_limit", 10.0)),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
        }


def make_render_env(
    *,
    seed: int,
    time_limit: float,
    width: int,
    height: int,
    physics_freq_hz: float,
) -> DmControlGymEnv:
    env = DmControlGymEnv(
        domain_name="reacher",
        task_name="hard",
        seed=seed,
        time_limit=time_limit,
        action_cost_weight=0.0,
        action_rate_cost_weight=0.0,
        velocity_cost_weight=0.0,
    )
    env.reset(seed=seed)
    configure_dm_control_timing(env, physics_timestep=1.0 / physics_freq_hz, time_limit=time_limit)
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    return env


def reset_env_to_state(
    env: DmControlGymEnv,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    env.reset(seed=seed)
    # Seeded reset rebuilds the dm_control env, so visual tweaks must be re-applied
    # before the first rendered frame is fed to the encoder.
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    model = load_model(checkpoint_path, device)

    history_size = int(config.get("history_size", 2))
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    if history_size != 2:
        raise ValueError(f"Expected history_size=2 for this planner, got {history_size}.")

    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = LeWMReacherDataset(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    pixel_mean = train_stats_dataset.pixel_mean
    pixel_std = train_stats_dataset.pixel_std
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 3)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one test trajectory with 3 or more frames.")

    if args.episode_idx is None:
        rng = np.random.default_rng()
        episode_idx = int(rng.choice(valid_episodes))
    else:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 3:
            raise ValueError(
                f"--episode-idx {episode_idx} is invalid for planning: need at least 3 frames, got {ep_len[episode_idx]}."
            )
    episode = load_dataset_episode(dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    run_name = f"episode_{episode_idx:05d}_{int(time.time())}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    true_latents = encode_frames(
        model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_emb_prev = true_latents[0]
    start_emb_curr = true_latents[1]
    goal_emb_prev = true_latents[-2]
    goal_emb_curr = true_latents[-1]

    start_markov = build_markov_state(start_emb_curr, start_emb_prev)
    goal_markov = build_markov_state(goal_emb_curr, goal_emb_prev)
    with torch.inference_mode():
        z_goal = model.predictor.lift_state(goal_markov.unsqueeze(0)).squeeze(0)
        decoded_goal = model.predictor.decode_state(z_goal.unsqueeze(0)).squeeze(0)
    embed_dim = int(goal_emb_curr.numel())
    c_weight = model.predictor.C.weight.detach().cpu().numpy().astype(np.float32)
    c_bias = model.predictor.C.bias.detach().cpu().numpy().astype(np.float32)
    terminal_output_matrix = c_weight[:embed_dim]
    terminal_target = goal_emb_curr.detach().cpu().numpy().astype(np.float32) - c_bias[:embed_dim]

    env = make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )

    render_start = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[1],
        qvel=qvel_np[1],
        height=height,
        width=width,
    )
    start_obs = flatten_observation(env._env.task.get_observation(env._env.physics))
    start_raw_emb = encode_single_frame(
        model,
        render_start,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )

    if ENABLE_START_GOAL_IMAGES:
        save_rgb_image(out_dir / "start_image.png", pixels_np[1])
        save_rgb_image(out_dir / "goal_image.png", pixels_np[-1])
        save_rgb_image(out_dir / "sim_start_image.png", render_start)

    action_low_norm, action_high_norm = raw_to_normalized_bounds(
        env.action_space.low.astype(np.float32),
        env.action_space.high.astype(np.float32),
        action_mean,
        action_std,
    )

    A_np = model.predictor.A.weight.detach().cpu().numpy()
    B_np = model.predictor.B.weight.detach().cpu().numpy()
    mpc_solver = GPUADMM_MPCSolver(
        A_np,
        B_np,
        terminal_output_matrix=terminal_output_matrix,
        terminal_target=terminal_target,
        q_terminal=Q_TERMINAL,
        r_control=R_CONTROL,
        u_min=action_low_norm,
        u_max=action_high_norm,
        horizon=args.horizon,
        rho=RHO,
        n_admm_steps=MAX_ADMM_STEPS,
        device=device,
    )

    current_prev_emb = start_emb_prev.detach().clone()
    current_emb = start_emb_curr.detach().clone()
    with torch.inference_mode():
        initial_markov = build_markov_state(current_emb, current_prev_emb)
        initial_z = model.predictor.lift_state(initial_markov.unsqueeze(0)).squeeze(0)
        initial_decoded = model.predictor.decode_state(initial_z.unsqueeze(0)).squeeze(0)

    rollout_frames = [render_start.copy()]
    solve_times = []
    sim_observations = [start_obs.copy()]
    num_executed_steps = 0

    timing_sums = {
        "encode": 0.0,
        "solve": 0.0,
        "decode": 0.0,
        "step": 0.0,
        "total": 0.0,
    }

    goal_obs = obs_np[-1]
    goal_qpos = qpos_np[-1]
    goal_qvel = qvel_np[-1]
    stop_reason = "max_mpc_steps"
    initial_goal_latent_distance = float(torch.linalg.vector_norm(initial_z - z_goal).item())
    initial_goal_decoded_embedding_distance = float(torch.linalg.vector_norm(initial_decoded[:embed_dim] - goal_emb_curr).item())
    initial_goal_observation_distance = float(np.linalg.norm(start_obs - goal_obs))

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        reached_goal, qpos_err = goal_reached(env, goal_qpos)
        if reached_goal:
            stop_reason = "goal_reached"
            break

        step_t0 = time.perf_counter()

        encode_t0 = time.perf_counter()
        with torch.inference_mode():
            current_markov = build_markov_state(current_emb, current_prev_emb)
            z_current = model.predictor.lift_state(current_markov.unsqueeze(0)).squeeze(0)
            decoded_current = model.predictor.decode_state(z_current.unsqueeze(0)).squeeze(0)
        maybe_cuda_synchronize(device)
        timing_sums["encode"] += time.perf_counter() - encode_t0

        current_decoded_embedding_goal_distance = float(torch.linalg.vector_norm(decoded_current[:embed_dim] - goal_emb_curr).item())

        solve_t0 = time.perf_counter()
        z_traj_t, u_traj_t, solve_time = mpc_solver.solve_torch(z_current)
        timing_sums["solve"] += time.perf_counter() - solve_t0
        solve_times.append(solve_time)

        decode_t0 = time.perf_counter()
        with torch.inference_mode():
            decoded_terminal = model.predictor.decode_state(z_traj_t[-1].unsqueeze(0)).squeeze(0)
        maybe_cuda_synchronize(device)
        timing_sums["decode"] += time.perf_counter() - decode_t0

        u0_norm = u_traj_t[0].detach().cpu().numpy()
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        num_executed_steps += 1

        step_env_t0 = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(u0_raw)
        frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = encode_single_frame(
            model,
            frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        maybe_cuda_synchronize(device)
        timing_sums["step"] += time.perf_counter() - step_env_t0

        rollout_frames.append(frame.copy())
        sim_observations.append(np.asarray(obs, dtype=np.float32).copy())

        current_prev_emb = current_emb
        current_emb = next_emb

        timing_sums["total"] += time.perf_counter() - step_t0

        reached_goal, qpos_err = goal_reached(env, goal_qpos)
        pbar.set_postfix(
            solve_ms=f"{solve_time * 1000.0:.2f}",
            emb_goal=f"{current_decoded_embedding_goal_distance:.3f}",
            qpos_goal=f"{qpos_err:.3f}",
        )

        if reached_goal:
            stop_reason = "goal_reached"
            break

        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_qpos = np.asarray(env._env.physics.data.qpos[: goal_qpos.shape[0]], dtype=np.float32)
    final_qvel = np.asarray(env._env.physics.data.qvel[: goal_qvel.shape[0]], dtype=np.float32)
    final_emb = current_emb
    _, final_goal_qpos_distance = goal_reached(env, goal_qpos)
    with torch.inference_mode():
        final_markov = build_markov_state(final_emb, current_prev_emb)
        z_final = model.predictor.lift_state(final_markov.unsqueeze(0)).squeeze(0)
        decoded_final = model.predictor.decode_state(z_final.unsqueeze(0)).squeeze(0)

    metrics = {
        "episode_idx": episode_idx,
        "episode_seed": episode_seed,
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "train_dataset_path": str(train_dataset_path),
        "goal_qpos_tolerance": GOAL_QPOS_TOL,
        "history_size": history_size,
        "horizon": args.horizon,
        "max_mpc_steps": args.max_mpc_steps,
        "num_executed_steps": int(num_executed_steps),
        "stop_reason": stop_reason,
        "goal_reached": stop_reason == "goal_reached",
        "final_goal_qpos_distance": float(final_goal_qpos_distance),
        "physics_freq_hz": physics_freq_hz,
        "initial_goal_latent_distance": initial_goal_latent_distance,
        "final_goal_latent_distance": float(torch.linalg.vector_norm(z_final - z_goal).item()),
        "initial_goal_decoded_embedding_distance": initial_goal_decoded_embedding_distance,
        "final_goal_decoded_embedding_distance": float(torch.linalg.vector_norm(decoded_final[:embed_dim] - goal_emb_curr).item()),
        "initial_goal_decoded_state_distance": float(torch.linalg.vector_norm(initial_decoded - decoded_goal).item()),
        "final_goal_decoded_state_distance": float(torch.linalg.vector_norm(decoded_final - decoded_goal).item()),
        "initial_goal_observation_distance": initial_goal_observation_distance,
        "final_goal_observation_distance": float(np.linalg.norm(sim_observations[-1] - goal_obs)),
        "final_goal_qvel_distance": float(np.linalg.norm(final_qvel - goal_qvel)),
        "initial_render_vs_dataset_start_embedding_distance": float(torch.linalg.vector_norm(start_raw_emb - start_emb_curr).item()),
        "mean_solve_time_ms": float(np.mean(solve_times) * 1000.0) if solve_times else 0.0,
        "max_solve_time_ms": float(np.max(solve_times) * 1000.0) if solve_times else 0.0,
        "timing_breakdown_ms_per_step": {
            key: (value / max(len(solve_times), 1)) * 1000.0 for key, value in timing_sums.items()
        },
        "goal_qpos": goal_qpos.tolist(),
        "goal_qvel": goal_qvel.tolist(),
        "final_qpos": final_qpos.tolist(),
        "final_qvel": final_qvel.tolist(),
        "terminal_cost_target": "decoded_embedding_only",
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_paths: list[str] = []

    video_path = None
    if ENABLE_ROLLOUT_VIDEO and rollout_frames:
        video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps))

    env.close()

    if ENABLE_CONSOLE_METRICS:
        print(
            json.dumps(
                {
                    "episode_idx": episode_idx,
                    "episode_seed": episode_seed,
                    "num_executed_steps": metrics["num_executed_steps"],
                    "initial_goal_latent_distance": metrics["initial_goal_latent_distance"],
                    "final_goal_latent_distance": metrics["final_goal_latent_distance"],
                    "initial_goal_decoded_embedding_distance": metrics["initial_goal_decoded_embedding_distance"],
                    "final_goal_decoded_embedding_distance": metrics["final_goal_decoded_embedding_distance"],
                    "final_goal_decoded_state_distance": metrics["final_goal_decoded_state_distance"],
                    "final_goal_observation_distance": metrics["final_goal_observation_distance"],
                    "final_goal_qpos_distance": metrics["final_goal_qpos_distance"],
                    "final_goal_qvel_distance": metrics["final_goal_qvel_distance"],
                    "initial_render_vs_dataset_start_embedding_distance": metrics[
                        "initial_render_vs_dataset_start_embedding_distance"
                    ],
                    "mean_solve_time_ms": metrics["mean_solve_time_ms"],
                    "metrics_path": str(metrics_path),
                    "plot_paths": plot_paths,
                    "video_path": video_path,
                },
                indent=2,
            )
        )

    print(f"Saved to: {out_dir}")

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
