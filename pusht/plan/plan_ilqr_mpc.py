#!/usr/bin/env python3
"""Plan in PushT pixel space with nominal iLQR MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from pusht.data.pusht_preproc import make_no_target_env, set_render_state
from pusht.train.mlpdyn_train import LeWMPushTDataset

DEFAULT_DATASET_PATH = "pusht/data/pusht_expert_train_preproc.h5"
DEFAULT_MODEL_DIR = "pusht/models/mlpdyn"
DEFAULT_OUT_DIR = "pusht/plan/ilqr_mpc_mlpdyn"

DEVICE = "auto"
HORIZON = 40
MAX_MPC_STEPS = 300
Q_TERMINAL = 5.0
Q_STAGE = 0.005
R_CONTROL = 0.1
VIDEO_FPS = 10
EPISODE_IDX = 0
ENV_ACTION_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--episode-idx", type=int, default=EPISODE_IDX)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
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


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def raw_to_env_action(raw_action: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    return (agent_pos.astype(np.float32) + ENV_ACTION_SCALE * raw_action.astype(np.float32)).astype(np.float32)


def angle_diff(angle: float, target: float) -> float:
    return float((angle - target + np.pi) % (2.0 * np.pi) - np.pi)


def block_pose_distance(current_block_pose: np.ndarray, goal_block_pose: np.ndarray) -> float:
    pose_err = current_block_pose.astype(np.float64) - goal_block_pose.astype(np.float64)
    pose_err[2] = angle_diff(float(current_block_pose[2]), float(goal_block_pose[2]))
    return float(np.linalg.norm(pose_err))


def goal_reached(current_block_pose: np.ndarray, goal_block_pose: np.ndarray, threshold: float = 5.0) -> tuple[bool, float]:
    pose_dist = block_pose_distance(current_block_pose, goal_block_pose)
    return pose_dist <= threshold, pose_dist


def load_dataset_episode(
    dataset_path: Path,
    episode_idx: int,
) -> dict[str, np.ndarray | int]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        return {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "action": np.asarray(h5["action"][rows], dtype=np.float32),
            "state": np.asarray(h5["state"][rows], dtype=np.float32),
            "proprio": np.asarray(h5["proprio"][rows], dtype=np.float32),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
        }


def make_render_env(*, width: int, height: int) -> Any:
    return make_no_target_env(height=height, width=width)


def reset_env_to_state(env: Any, state: np.ndarray) -> np.ndarray:
    set_render_state(env, np.asarray(state, dtype=np.float64))
    return np.asarray(env._render(visualize=False), dtype=np.uint8)


def current_block_pose(env: Any) -> np.ndarray:
    return np.asarray([env.block.position.x, env.block.position.y, env.block.angle], dtype=np.float32)


def current_agent_pos(env: Any) -> np.ndarray:
    return np.asarray([env.agent.position.x, env.agent.position.y], dtype=np.float32)


class MarkovDynamicsTorch:
    def __init__(self, model: torch.nn.Module, state_dim: int, action_dim: int, device: torch.device) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError(
                "This planner expects a one-step Markov MLP dynamics model with "
                "history_size=1, action_history_size=1, and num_preds=1."
            )
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("This planner assumes an identity action encoder.")
        if int(predictor.embed_dim) != state_dim:
            raise ValueError(f"Predictor state dim mismatch: expected {state_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != action_dim:
            raise ValueError(f"Predictor action dim mismatch: expected {action_dim}, got {predictor.action_dim}.")

        self.net = predictor.net.to(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, u), dim=-1))


class ILQRMPCSolver:
    def __init__(
        self,
        dynamics: MarkovDynamicsTorch,
        *,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        max_iters: int,
        tol: float,
        regularization: float,
        device: torch.device,
    ) -> None:
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim
        self.action_dim = dynamics.action_dim
        self.horizon = int(horizon)
        self.q_terminal = float(q_terminal)
        self.q_stage = float(q_stage)
        self.r_control = float(r_control)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.regularization = float(regularization)
        self.device = device
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _make_initial_action_guess(self) -> torch.Tensor:
        if self.horizon <= 1:
            return self.prev_u_guess.clone()
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return guess

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            x_curr = self.dynamics.step(x_curr, u_seq[step])
            x_traj[step + 1] = x_curr
        return x_traj

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_goal
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
        cost = cost + self.q_terminal * torch.dot(terminal_err, terminal_err)
        return cost

    def _linearize_dynamics(self, x_traj: torch.Tensor, u_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_list = []
        b_list = []

        def dyn_cat(inp: torch.Tensor) -> torch.Tensor:
            x = inp[: self.state_dim]
            u = inp[self.state_dim :]
            return self.dynamics.step(x, u)

        for step in range(self.horizon):
            xu = torch.cat((x_traj[step], u_seq[step]), dim=0).detach().requires_grad_(True)
            jac = torch.autograd.functional.jacobian(dyn_cat, xu, vectorize=True)
            a_list.append(jac[:, : self.state_dim].detach())
            b_list.append(jac[:, self.state_dim :].detach())

        return torch.stack(a_list, dim=0), torch.stack(b_list, dim=0)

    def solve(self, x0_np: np.ndarray, x_goal_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.tensor(x_goal_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_goal).item())
        iterations = 0
        reg = self.regularization

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_seq)
            k_seq = torch.empty((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            kk_seq = torch.empty((self.horizon, self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)

            terminal_err = x_traj[self.horizon] - x_goal
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_goal
                u = u_seq[step]
                a = a_seq[step]
                b = b_seq[step]

                l_x = 2.0 * self.q_stage * x_err
                l_u = 2.0 * self.r_control * u
                l_xx = 2.0 * self.q_stage * self.eye_x
                l_uu = 2.0 * self.r_control * self.eye_u

                q_x = l_x + a.T @ v_x
                q_u = l_u + b.T @ v_x
                q_xx = l_xx + a.T @ v_xx @ a
                q_ux = b.T @ v_xx @ a
                q_uu = l_uu + b.T @ v_xx @ b + reg * self.eye_u
                q_uu = 0.5 * (q_uu + q_uu.T)

                try:
                    q_uu_inv = torch.linalg.inv(q_uu)
                except RuntimeError:
                    backward_ok = False
                    break

                k = -q_uu_inv @ q_u
                kk = -q_uu_inv @ q_ux
                k_seq[step] = k
                kk_seq[step] = kk

                v_x = q_x + kk.T @ q_uu @ k + kk.T @ q_u + q_ux.T @ k
                v_xx = q_xx + kk.T @ q_uu @ kk + kk.T @ q_ux + q_ux.T @ kk
                v_xx = 0.5 * (v_xx + v_xx.T)

            if not backward_ok:
                reg = min(reg * 10.0, 1e6)
                continue

            accepted = False
            candidate_best = None
            for alpha in self.line_search_alphas:
                x_new = torch.empty_like(x_traj)
                u_new = torch.empty_like(u_seq)
                x_new[0] = x0
                for step in range(self.horizon):
                    dx = x_new[step] - x_traj[step]
                    u_new[step] = u_seq[step] + alpha * k_seq[step] + kk_seq[step] @ dx
                    x_new[step + 1] = self.dynamics.step(x_new[step], u_new[step])
                new_cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    candidate_best = (x_new, u_new, new_cost, alpha)
                    accepted = True
                    break

            if not accepted:
                reg = min(reg * 10.0, 1e6)
                if reg >= 1e6:
                    break
                continue

            x_traj, u_seq, new_cost, alpha = candidate_best
            max_du = float(torch.max(torch.abs(alpha * k_seq)).item())
            cost_improvement = current_cost - new_cost
            current_cost = new_cost
            reg = max(self.regularization, reg * 0.5)

            if cost_improvement <= self.tol or max_du <= self.tol:
                break

        self.prev_u_guess = u_seq.detach().clone()

        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0
        return (
            x_traj.detach().cpu().numpy().astype(np.float64),
            u_seq.detach().cpu().numpy().astype(np.float64),
            solve_time,
            iterations,
            current_cost,
        )


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

    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for the MLP model, got {history_size}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 32))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))

    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = LeWMPushTDataset(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one trajectory with 2 or more frames.")

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
    state_np = np.asarray(episode["state"])
    proprio_np = np.asarray(episode["proprio"])
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
        model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_emb = true_latents[0]
    goal_emb = true_latents[-1]
    goal_prev_emb = true_latents[-2]
    start_state = make_markov_state(start_emb)
    goal_state = make_markov_state(goal_emb, goal_prev_emb)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    save_rgb_image(out_dir / "start_image.png", pixels_np[0])
    save_rgb_image(out_dir / "goal_image.png", pixels_np[-1])

    env = make_render_env(width=width, height=height)
    render_start = reset_env_to_state(env, state_np[0])
    dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    mpc_solver = ILQRMPCSolver(
        dynamics,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        device=device,
    )

    current_frame = render_start
    current_emb = encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = make_markov_state(current_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_block = state_np[-1, 2:5].astype(np.float32)
    current_block = current_block_pose(env)

    rollout_frames = [current_frame.copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    block_goal_distances = [block_pose_distance(current_block, goal_block)]
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    stop_reason = "max_mpc_steps"

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np, goal_state_np)
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))
        ilqr_costs.append(float(plan_cost))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        u0_env = raw_to_env_action(u0_raw, current_agent_pos(env))
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())
        executed_actions_env.append(u0_env.copy())

        _, _, terminated, truncated, _ = env.step(u0_env)
        current_frame = np.asarray(env._render(visualize=False), dtype=np.uint8)
        next_emb = encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = make_markov_state(next_emb, current_emb)
        current_emb = next_emb
        current_block = current_block_pose(env)

        rollout_frames.append(current_frame.copy())
        latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
        block_goal_distance = block_pose_distance(current_block, goal_block)
        latent_goal_distances.append(latent_goal_distance)
        block_goal_distances.append(block_goal_distance)

        pbar.set_postfix(
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            iters=f"{ilqr_iterations[-1]}",
            latent_goal=f"{latent_goal_distance:.3f}",
            block_goal=f"{block_goal_distance:.3f}",
        )

        reached_goal, _ = goal_reached(current_block, goal_block)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_block = current_block_pose(env)
    final_agent = current_agent_pos(env)
    video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    env.close()

    metrics = {
        "episode_idx": episode_idx,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(model_dir / "config.json"),
        "dataset_path": str(dataset_path),
        "history_size": history_size,
        "action_history_size": 1,
        "state_space": "latent_plus_latent_delta",
        "markov_state_dim": markov_state_dim,
        "horizon": args.horizon,
        "max_mpc_steps": args.max_mpc_steps,
        "stop_reason": stop_reason,
        "num_mpc_steps": len(executed_actions_norm),
        "action_space": "normalized_dataset_action_then_raw_then_absolute_env_target",
        "env_action_scale": ENV_ACTION_SCALE,
        "start_agent_pos": state_np[0, :2].tolist(),
        "goal_block_pose": goal_block.tolist(),
        "final_agent_pos": final_agent.tolist(),
        "final_block_pose": final_block.tolist(),
        "goal_proprio": proprio_np[-1].tolist(),
        "latent_goal_distance_initial": latent_goal_distances[0],
        "latent_goal_distance_final": latent_goal_distances[-1],
        "block_goal_distance_initial": block_goal_distances[0],
        "block_goal_distance_final": block_goal_distances[-1],
        "latent_goal_distances": latent_goal_distances,
        "block_goal_distances": block_goal_distances,
        "solve_times_ms": solve_times_ms,
        "ilqr_iterations": ilqr_iterations,
        "ilqr_costs": ilqr_costs,
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_env": [action.tolist() for action in executed_actions_env],
        "video_path": video_path,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        json.dumps(
            {
                "episode_idx": episode_idx,
                "latent_goal_distance_final": metrics["latent_goal_distance_final"],
                "block_goal_distance_final": metrics["block_goal_distance_final"],
                "stop_reason": stop_reason,
                "metrics_path": str(metrics_path),
                "video_path": video_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
