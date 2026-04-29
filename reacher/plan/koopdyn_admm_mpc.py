#!/usr/bin/env python3
"""Plan in Reacher pixel space with unconstrained ADMM-style MPC over a Koopman model."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm
import json

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.shared.models import DeepKoopmanLinDec
from reacher.train.reacher_policy_train import DmControlGymEnv, flatten_observation

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data/reacher_test.h5"
DEFAULT_KOOPMAN_PATH = "reacher/models/koopdyn_lindec/koopman_lindec_1.pt"
DEFAULT_OUT_DIR = "reacher/plan/koopdyn_admm_mpc"

DEVICE = "auto"
HORIZON = 25
MAX_MPC_STEPS = 250
Q_TERMINAL = 10.0
Q_STAGE = 0.005
R_CONTROL = 0.1
VIDEO_FPS = 60
EPISODE_IDX = None
ADMM_RHO = 1.0
ADMM_STEPS = 1
GOAL_OBS_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-save-path", type=Path, default=Path(DEFAULT_KOOPMAN_PATH))
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--visual-model-dir", type=Path, default=None)
    parser.add_argument("--visual-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--episode-idx", type=int, default=EPISODE_IDX)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--admm-rho", type=float, default=ADMM_RHO)
    parser.add_argument("--admm-steps", type=int, default=ADMM_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_torch_object(path: Path, device: torch.device) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


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


def load_visual_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_visual_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = load_torch_object(checkpoint_path, device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Visual checkpoint did not contain a torch module: {checkpoint_path}")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_koopman_model(checkpoint_path: Path, device: torch.device) -> tuple[DeepKoopmanLinDec, dict[str, object]]:
    payload = load_torch_object(checkpoint_path, device)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected Koopman checkpoint payload dict at {checkpoint_path}")
    model_config = payload["model_config"]
    model = DeepKoopmanLinDec(**model_config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, payload


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


def make_markov_state(embedding: torch.Tensor, previous_embedding: torch.Tensor | None = None) -> torch.Tensor:
    if previous_embedding is None:
        delta = torch.zeros_like(embedding)
    else:
        delta = embedding - previous_embedding
    return torch.cat((embedding, delta), dim=-1)


class MinMaxNormalizer:
    def __init__(self, stats: dict[str, torch.Tensor]) -> None:
        self.min = torch.as_tensor(stats["min"], dtype=torch.float32)
        self.range = torch.as_tensor(stats["range"], dtype=torch.float32)
        self.range[self.range == 0] = 1e-6

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.min.to(x.device)) / self.range.to(x.device) - 1.0


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def preprocess_markov_state(
    state_raw: torch.Tensor,
    *,
    normalizer: MinMaxNormalizer,
    enable_normalization: bool,
) -> torch.Tensor:
    if enable_normalization:
        return normalizer.normalize(state_raw)
    return state_raw


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
    hide_target(env)
    configure_offscreen_framebuffer(env, width, height)
    physics = env._env.physics
    with physics.reset_context():
        physics.data.qpos[: qpos.shape[0]] = qpos
        physics.data.qvel[: qvel.shape[0]] = qvel
    env._last_action = np.zeros_like(env.action_space.low, dtype=np.float32)
    return physics.render(height=height, width=width, camera_id=0)


def goal_reached(current_obs: np.ndarray, goal_obs: np.ndarray, threshold: float = GOAL_OBS_THRESHOLD) -> tuple[bool, float]:
    obs_err = float(np.linalg.norm(current_obs - goal_obs))
    return obs_err <= threshold, obs_err


class KoopmanADMMPlanner:
    def __init__(
        self,
        model: DeepKoopmanLinDec,
        *,
        goal_state_proc: torch.Tensor,
        horizon: int,
        q_stage: float,
        q_terminal: float,
        r_control: float,
        rho: float,
        admm_steps: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.state_dim = int(model.state_dim)
        self.control_dim = int(model.control_dim)
        self.latent_dim = int(model.latent_dim)
        self.q_stage = float(q_stage)
        self.q_terminal = float(q_terminal)
        self.r_control = float(r_control)
        self.rho = float(rho)
        self.admm_steps = max(1, int(admm_steps))
        self.device = device

        with torch.inference_mode():
            z_goal = model.lift_state(goal_state_proc.unsqueeze(0).to(device))[0]
        self.z_goal = z_goal

        self.A = model.A.weight.detach().to(device=device, dtype=torch.float32)
        self.B = model.B.weight.detach().to(device=device, dtype=torch.float32)
        self.C = model.C.weight.detach().to(device=device, dtype=torch.float32)
        self.c_bias = model.C.bias.detach().to(device=device, dtype=torch.float32)
        self.x_goal = goal_state_proc.to(device=device, dtype=torch.float32)

        self.nz_total = (self.horizon + 1) * self.latent_dim
        self.nu_total = self.horizon * self.control_dim
        self.n_vars = self.nu_total + self.nz_total
        self.n_eq = (self.horizon + 1) * self.latent_dim
        self.u_slice = slice(0, self.nu_total)
        self.z_slice = slice(self.nu_total, self.n_vars)

        self._warm_sol = torch.zeros(self.n_vars, dtype=torch.float32, device=device)
        self._sol = torch.zeros(self.n_vars, dtype=torch.float32, device=device)
        self._rhs = torch.zeros(self.n_vars + self.n_eq, dtype=torch.float32, device=device)
        self._b_eq = torch.zeros(self.n_eq, dtype=torch.float32, device=device)
        self._u_shift = torch.empty((self.horizon, self.control_dim), dtype=torch.float32, device=device)
        self._z_shift = torch.empty((self.horizon + 1, self.latent_dim), dtype=torch.float32, device=device)

        self._build_quadratic_terms()
        self._build_kkt_inverse()

    def _state_quadratic_terms(self, weight: float) -> tuple[torch.Tensor, torch.Tensor]:
        if weight <= 0.0:
            return (
                torch.zeros((self.latent_dim, self.latent_dim), dtype=torch.float32, device=self.device),
                torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device),
            )
        residual_offset = self.c_bias - self.x_goal
        p = 2.0 * weight * (self.C.T @ self.C)
        q = 2.0 * weight * (self.C.T @ residual_offset)
        return p, q

    def _build_quadratic_terms(self) -> None:
        p_blocks = []
        q_blocks = []

        control_block = 2.0 * self.r_control * torch.eye(self.control_dim, device=self.device)
        for _ in range(self.horizon):
            p_blocks.append(control_block)
            q_blocks.append(torch.zeros(self.control_dim, dtype=torch.float32, device=self.device))

        p_stage, q_stage = self._state_quadratic_terms(self.q_stage)
        p_terminal, q_terminal = self._state_quadratic_terms(self.q_terminal)
        for _ in range(self.horizon):
            p_blocks.append(p_stage)
            q_blocks.append(q_stage)
        p_blocks.append(p_terminal)
        q_blocks.append(q_terminal)

        self.P = torch.block_diag(*p_blocks)
        self.q = torch.cat(q_blocks, dim=0)

    def _build_kkt_inverse(self) -> None:
        a_eq = torch.zeros((self.n_eq, self.n_vars), dtype=torch.float32, device=self.device)
        z_start = self.nu_total
        a_eq[0 : self.latent_dim, z_start : z_start + self.latent_dim] = torch.eye(
            self.latent_dim,
            dtype=torch.float32,
            device=self.device,
        )

        for step in range(self.horizon):
            row = (step + 1) * self.latent_dim
            col_u = step * self.control_dim
            col_z = z_start + step * self.latent_dim
            col_z_next = z_start + (step + 1) * self.latent_dim
            a_eq[row : row + self.latent_dim, col_u : col_u + self.control_dim] = -self.B
            a_eq[row : row + self.latent_dim, col_z : col_z + self.latent_dim] = -self.A
            a_eq[row : row + self.latent_dim, col_z_next : col_z_next + self.latent_dim] = torch.eye(
                self.latent_dim,
                dtype=torch.float32,
                device=self.device,
            )

        top = torch.cat([self.P, a_eq.T], dim=1)
        bottom = torch.cat(
            [a_eq, torch.zeros((self.n_eq, self.n_eq), dtype=torch.float32, device=self.device)],
            dim=1,
        )
        self.kkt_inv = torch.linalg.inv(torch.cat([top, bottom], dim=0))

    def _shift_warm_start(self, z0: torch.Tensor) -> None:
        if self.horizon <= 1:
            self._sol.zero_()
            self._sol[self.nu_total : self.nu_total + self.latent_dim] = z0
            return

        u_prev = self._warm_sol[self.u_slice].view(self.horizon, self.control_dim)
        z_prev = self._warm_sol[self.z_slice].view(self.horizon + 1, self.latent_dim)
        self._u_shift[:-1] = u_prev[1:]
        self._u_shift[-1] = u_prev[-1]
        self._z_shift[0] = z0
        self._z_shift[1:-1] = z_prev[2:]
        self._z_shift[-1] = z_prev[-1]
        self._sol[self.u_slice] = self._u_shift.reshape(-1)
        self._sol[self.z_slice] = self._z_shift.reshape(-1)

    def solve(self, current_state_proc: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        with torch.inference_mode():
            z0 = self.model.lift_state(current_state_proc.unsqueeze(0).to(self.device))[0]

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        self._shift_warm_start(z0)
        self._b_eq.zero_()
        self._b_eq[: self.latent_dim] = z0

        self._rhs[: self.n_vars] = -self.q
        self._rhs[self.n_vars :] = self._b_eq
        sol_augmented = self.kkt_inv @ self._rhs
        self._sol.copy_(sol_augmented[: self.n_vars])

        # With unconstrained controls, the ADMM projection step is the identity.
        # Re-solving is redundant, but we keep the interface aligned with the
        # constrained planner family.
        _ = self.rho
        _ = self.admm_steps

        self._warm_sol.copy_(self._sol)
        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0

        u_traj = self._sol[self.u_slice].view(self.horizon, self.control_dim)
        z_traj = self._sol[self.z_slice].view(self.horizon + 1, self.latent_dim)
        return (
            z_traj.detach().cpu().numpy().astype(np.float64),
            u_traj.detach().cpu().numpy().astype(np.float64),
            solve_time,
        )


def resolve_visual_assets(
    source_metadata: dict[str, object],
    *,
    visual_model_dir_arg: Path | None,
    visual_checkpoint_arg: Path | None,
) -> tuple[Path, Path]:
    if visual_model_dir_arg is not None:
        visual_model_dir = visual_model_dir_arg.expanduser().resolve()
    else:
        model_dir_str = source_metadata.get("model_dir")
        if not model_dir_str:
            raise ValueError("Koopman checkpoint metadata is missing source visual model_dir.")
        visual_model_dir = Path(str(model_dir_str)).expanduser().resolve()

    if visual_checkpoint_arg is not None:
        visual_checkpoint = visual_checkpoint_arg.expanduser().resolve()
    else:
        checkpoint_str = source_metadata.get("checkpoint")
        visual_checkpoint = Path(str(checkpoint_str)).expanduser().resolve() if checkpoint_str else None
        if visual_checkpoint is None or not visual_checkpoint.is_file():
            visual_checkpoint = latest_object_checkpoint(visual_model_dir).resolve()

    return visual_model_dir, visual_checkpoint


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    model_save_path = args.model_save_path.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    koopman_model, koopman_payload = load_koopman_model(model_save_path, device)
    source_metadata = koopman_payload.get("source_metadata", {})
    if not source_metadata:
        raise ValueError("Koopman checkpoint is missing source_metadata needed for planning.")

    visual_model_dir, visual_checkpoint = resolve_visual_assets(
        source_metadata,
        visual_model_dir_arg=args.visual_model_dir,
        visual_checkpoint_arg=args.visual_checkpoint,
    )
    visual_config = load_visual_config(visual_model_dir)
    visual_model = load_visual_model(visual_checkpoint, device)

    img_size = int(source_metadata.get("img_size", visual_config.get("img_size", 224)))
    embed_dim = int(source_metadata.get("embed_dim", visual_config.get("embed_dim", 18)))
    markov_state_dim = int(source_metadata.get("state_dim", 2 * embed_dim))
    action_dim = int(source_metadata.get("action_dim", 2))

    if int(koopman_model.state_dim) != markov_state_dim:
        raise ValueError(
            f"Koopman checkpoint state_dim={koopman_model.state_dim} does not match source metadata {markov_state_dim}."
        )
    if int(koopman_model.control_dim) != action_dim:
        raise ValueError(
            f"Koopman checkpoint control_dim={koopman_model.control_dim} does not match source metadata {action_dim}."
        )

    training_config = koopman_payload.get("training_config", {})
    enable_normalization = bool(training_config.get("enable_normalization", False))
    normalizer = MinMaxNormalizer(koopman_payload["normalization_stats"])

    action_mean = torch.as_tensor(source_metadata["action_mean"], dtype=torch.float32).cpu().numpy().reshape(1, -1)
    action_std = torch.as_tensor(source_metadata["action_std"], dtype=torch.float32).cpu().numpy().reshape(1, -1)
    if action_mean.shape[1] != action_dim or action_std.shape[1] != action_dim:
        raise ValueError("Action normalization stats do not match Koopman control dimension.")

    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one test trajectory with 2 or more frames.")

    rng = np.random.default_rng(args.seed)
    if args.episode_idx is None:
        episode_idx = int(rng.choice(valid_episodes))
    else:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"--episode-idx {episode_idx} must have at least 2 frames, got {ep_len[episode_idx]}.")

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

    run_name = f"{int(time.time())}_episode_{episode_idx:05d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    true_latents = encode_frames(
        visual_model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_emb = true_latents[0]
    goal_emb = true_latents[-1]
    start_state_raw = make_markov_state(start_emb)
    goal_state_raw = make_markov_state(goal_emb)
    if int(start_state_raw.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: expected {markov_state_dim}, got {start_state_raw.numel()}.")

    start_state_proc = preprocess_markov_state(
        start_state_raw,
        normalizer=normalizer,
        enable_normalization=enable_normalization,
    )
    goal_state_proc = preprocess_markov_state(
        goal_state_raw,
        normalizer=normalizer,
        enable_normalization=enable_normalization,
    )

    save_rgb_image(out_dir / "start_image.png", pixels_np[0])
    save_rgb_image(out_dir / "goal_image.png", pixels_np[-1])

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
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        height=height,
        width=width,
    )

    planner = KoopmanADMMPlanner(
        koopman_model,
        goal_state_proc=goal_state_proc,
        horizon=args.horizon,
        q_stage=args.q_stage,
        q_terminal=args.q_terminal,
        r_control=args.r_control,
        rho=args.admm_rho,
        admm_steps=args.admm_steps,
        device=device,
    )

    current_frame = render_start
    current_emb = encode_single_frame(
        visual_model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    previous_emb: torch.Tensor | None = None
    current_state_raw = make_markov_state(current_emb, previous_emb)
    current_state_proc = preprocess_markov_state(
        current_state_raw,
        normalizer=normalizer,
        enable_normalization=enable_normalization,
    )
    goal_obs = obs_np[-1].astype(np.float32)
    current_obs = flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)

    rollout_frames = [current_frame.copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    solve_times_ms: list[float] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state_proc - goal_state_proc).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    observation_goal_distances = [float(np.linalg.norm(current_obs - goal_obs))]
    stop_reason = "max_mpc_steps"

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        _, u_plan, solve_time = planner.solve(current_state_proc)
        solve_times_ms.append(solve_time * 1000.0)

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        obs, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = encode_single_frame(
            visual_model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state_raw = make_markov_state(next_emb, current_emb)
        current_state_proc = preprocess_markov_state(
            current_state_raw,
            normalizer=normalizer,
            enable_normalization=enable_normalization,
        )
        previous_emb = current_emb
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        latent_goal_distance = float(torch.linalg.vector_norm(current_state_proc - goal_state_proc).item())
        embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
        observation_goal_distance = float(np.linalg.norm(current_obs - goal_obs))
        latent_goal_distances.append(latent_goal_distance)
        embedding_goal_distances.append(embedding_goal_distance)
        observation_goal_distances.append(observation_goal_distance)

        pbar.set_postfix(
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            latent_goal=f"{latent_goal_distance:.3f}",
            obs_goal=f"{observation_goal_distance:.3f}",
        )

        reached_goal, _ = goal_reached(current_obs, goal_obs)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_qpos = np.asarray(env._env.physics.data.qpos[: qpos_np.shape[1]], dtype=np.float32)
    final_qvel = np.asarray(env._env.physics.data.qvel[: qvel_np.shape[1]], dtype=np.float32)
    final_obs = flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)
    video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    env.close()

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
