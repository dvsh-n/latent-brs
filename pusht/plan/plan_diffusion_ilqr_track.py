#!/usr/bin/env python3
"""Track a diffusion-planned PushT image trajectory with latent-space iLQR MPC."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from pusht.shared.pusht_env import get_pusht_agent_pos, get_pusht_block_pose, make_no_target_env, make_pusht_env
from pusht.shared.utils import load_expert_policy_bundle, select_expert_action
from pusht.train.mlpdyn_train import LeWMPushTDataset, build_markov_state, required_markov_history

DEFAULT_DIFFUSION_MODEL_DIR = Path("pusht/models")
DEFAULT_DYNAMICS_MODEL_DIR = Path("pusht/models/mlpdyn")
DEFAULT_OUT_DIR = Path("pusht/plan/diffusion_ilqr_track")
ENV_ID = "gym_pusht/PushT-v0"
ENV_ACTION_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-model-dir", type=Path, default=DEFAULT_DIFFUSION_MODEL_DIR)
    parser.add_argument("--dynamics-model-dir", type=Path, default=DEFAULT_DYNAMICS_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-mpc-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--q-terminal", type=float, default=5.0)
    parser.add_argument("--q-stage", type=float, default=0.02)
    parser.add_argument("--r-control", type=float, default=0.1)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--action-mode", default="auto", choices=["auto", "absolute", "relative"])
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_arg = "cuda"
        else:
            device_arg = "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


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


def make_markov_state(history: list[torch.Tensor], markov_deriv: int) -> torch.Tensor:
    context_len = required_markov_history(markov_deriv)
    history_tensor = torch.stack(history[-context_len:], dim=0)
    if history_tensor.shape[0] < context_len:
        pad = history_tensor[:1].repeat(context_len - history_tensor.shape[0], 1)
        history_tensor = torch.cat((pad, history_tensor), dim=0)
    return build_markov_state(history_tensor, markov_deriv)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def raw_to_env_action(raw_action: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    return (agent_pos.astype(np.float32) + ENV_ACTION_SCALE * raw_action.astype(np.float32)).astype(np.float32)


def angle_diff(angle: float, target: float) -> float:
    return float((angle - target + np.pi) % (2.0 * np.pi) - np.pi)


def block_pose_distance(current_block_pose: np.ndarray, target_block_pose: np.ndarray) -> float:
    pose_err = current_block_pose.astype(np.float64) - target_block_pose.astype(np.float64)
    pose_err[2] = angle_diff(float(current_block_pose[2]), float(target_block_pose[2]))
    return float(np.linalg.norm(pose_err))


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


def save_rollout_video(frames: list[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
    return out_path


def extract_full_state(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "get_state"):
        return np.asarray(base_env.get_state(), dtype=np.float64).reshape(-1)

    if not (
        hasattr(base_env, "agent")
        and hasattr(base_env.agent, "position")
        and hasattr(base_env.agent, "velocity")
        and hasattr(base_env, "block")
        and hasattr(base_env.block, "position")
        and hasattr(base_env.block, "angle")
    ):
        raise AttributeError("PushT env does not expose get_state() and is missing agent/block bodies.")

    return np.asarray(
        [
            float(base_env.agent.position.x),
            float(base_env.agent.position.y),
            float(base_env.block.position.x),
            float(base_env.block.position.y),
            float(base_env.block.angle),
            float(base_env.agent.velocity.x),
            float(base_env.agent.velocity.y),
        ],
        dtype=np.float64,
    )


def reset_env_to_state(env: Any, state: np.ndarray) -> np.ndarray:
    env.agent.velocity = [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._set_state(np.asarray(state[:5], dtype=np.float64))
    env.agent.velocity = [float(state[5]), float(state[6])] if state.shape[0] >= 7 else [0.0, 0.0]
    env.block.velocity = [0.0, 0.0]
    env.block.angular_velocity = 0.0
    env._last_action = None
    return np.asarray(env._render(visualize=False), dtype=np.uint8)


def current_block_pose(env: Any) -> np.ndarray:
    return get_pusht_block_pose(env)


def current_agent_pos(env: Any) -> np.ndarray:
    return get_pusht_agent_pos(env)


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


class ILQRTrajectoryTracker:
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

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_ref[step]
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_ref[self.horizon]
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

    def solve(self, x0_np: np.ndarray, x_ref_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        if x_ref_np.shape != (self.horizon + 1, self.state_dim):
            raise ValueError(
                f"Expected x_ref_np with shape {(self.horizon + 1, self.state_dim)}, got {x_ref_np.shape}."
            )

        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_ref = torch.tensor(x_ref_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_ref).item())
        iterations = 0
        reg = self.regularization

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_seq)
            k_seq = torch.empty((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            kk_seq = torch.empty((self.horizon, self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)

            terminal_err = x_traj[self.horizon] - x_ref[self.horizon]
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_ref[step]
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

            candidate_best = None
            for alpha in self.line_search_alphas:
                x_new = torch.empty_like(x_traj)
                u_new = torch.empty_like(u_seq)
                x_new[0] = x0
                for step in range(self.horizon):
                    dx = x_new[step] - x_traj[step]
                    u_new[step] = u_seq[step] + alpha * k_seq[step] + kk_seq[step] @ dx
                    x_new[step + 1] = self.dynamics.step(x_new[step], u_new[step])
                new_cost = float(self._trajectory_cost(x_new, u_new, x_ref).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    candidate_best = (x_new, u_new, new_cost, alpha)
                    break

            if candidate_best is None:
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


def build_reference_markov_states(
    *,
    model: torch.nn.Module,
    reference_frames: list[np.ndarray],
    device: torch.device,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    frame_batch_size: int,
    markov_deriv: int,
) -> torch.Tensor:
    pixels = preprocess_pixels(
        np.stack(reference_frames, axis=0),
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    latents = encode_frames(model, pixels, device=device, frame_batch_size=frame_batch_size)
    history_len = required_markov_history(markov_deriv)
    markov_states = []
    for idx in range(latents.shape[0]):
        start = max(0, idx - history_len + 1)
        history = [latents[j] for j in range(start, idx + 1)]
        if len(history) < history_len:
            history = [history[0]] * (history_len - len(history)) + history
        markov_states.append(make_markov_state(history, markov_deriv))
    return torch.stack(markov_states, dim=0)


def rollout_diffusion_mpc_plan(
    *,
    bundle: Any,
    start_state: np.ndarray,
    horizon: int,
    action_mode: str,
    render_width: int,
    render_height: int,
) -> dict[str, list[np.ndarray]]:
    plan_env = make_pusht_env(
        ENV_ID,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max(horizon, 1),
        observation_width=render_width,
        observation_height=render_height,
        visualization_width=render_width,
        visualization_height=render_height,
    )
    no_target_render_env = make_no_target_env(height=render_height, width=render_width, max_episode_steps=max(horizon, 1))
    try:
        plan_env.reset(seed=0)
        initial_visible_frame = reset_env_to_state(plan_env.unwrapped, start_state).copy()
        initial_hidden_frame = reset_env_to_state(no_target_render_env, start_state).copy()
        bundle.policy.reset()
        observation = {
            "pixels": initial_visible_frame,
            "agent_pos": current_agent_pos(plan_env).copy(),
        }

        visible_frames = [initial_visible_frame]
        hidden_frames = [initial_hidden_frame]
        states = [np.asarray(start_state, dtype=np.float64).copy()]
        block_poses = [current_block_pose(plan_env).copy()]
        agent_positions = [current_agent_pos(plan_env).copy()]
        actions_env = []

        for _ in range(horizon):
            action_env = select_expert_action(bundle, observation, env=plan_env, action_mode=action_mode)
            actions_env.append(np.asarray(action_env, dtype=np.float32).reshape(2))
            observation, _, terminated, truncated, _ = plan_env.step(action_env)
            next_state = extract_full_state(plan_env)
            visible_frames.append(np.asarray(plan_env.render(), dtype=np.uint8))
            hidden_frames.append(reset_env_to_state(no_target_render_env, next_state).copy())
            states.append(next_state)
            block_poses.append(current_block_pose(plan_env).copy())
            agent_positions.append(current_agent_pos(plan_env).copy())
            if terminated or truncated:
                break
    finally:
        no_target_render_env.close()
        plan_env.close()

    return {
        "states": states,
        "actions_env": actions_env,
        "visible_frames": visible_frames,
        "hidden_frames": hidden_frames,
        "block_poses": block_poses,
        "agent_positions": agent_positions,
    }


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    diffusion_model_dir = args.diffusion_model_dir.expanduser().resolve()
    dynamics_model_dir = args.dynamics_model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(dynamics_model_dir).resolve()
    )

    config = load_config(dynamics_model_dir)
    model = load_model(checkpoint_path, device)
    diffusion_bundle = load_expert_policy_bundle(diffusion_model_dir, device=args.device)

    markov_deriv = int(config.get("markov_deriv", 1))
    img_size = int(config.get("img_size", 224))
    frameskip = int(config.get("frameskip", 1))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 48))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    if frameskip != 1:
        raise ValueError(f"This planner currently supports frameskip=1 only, got frameskip={frameskip}.")

    train_dataset_path = Path(str(config.get("dataset_path", ""))).expanduser().resolve()
    if not train_dataset_path.is_file():
        raise FileNotFoundError(
            "Dynamics config dataset_path is required to recover action normalization stats "
            f"and was not found: {train_dataset_path}"
        )
    train_stats_dataset = LeWMPushTDataset(
        train_dataset_path,
        markov_deriv=markov_deriv,
        num_preds=1,
        frameskip=frameskip,
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    run_name = f"{int(time.time())}_seed_{args.seed:05d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    init_env = make_pusht_env(
        ENV_ID,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=max(args.max_mpc_steps, 1),
        observation_width=img_size,
        observation_height=img_size,
        visualization_width=img_size,
        visualization_height=img_size,
    )
    try:
        init_env.reset(seed=args.seed)
        start_state = extract_full_state(init_env)
        start_visible_frame = np.asarray(init_env.render(), dtype=np.uint8)
    finally:
        init_env.close()

    track_env = make_no_target_env(height=img_size, width=img_size, max_episode_steps=args.max_mpc_steps)
    current_frame = reset_env_to_state(track_env, start_state)
    initial_block_pose = current_block_pose(track_env).copy()
    initial_agent_pos = current_agent_pos(track_env).copy()
    save_rgb_image(out_dir / "start_visible.png", start_visible_frame)
    save_rgb_image(out_dir / "start_tracking_frame.png", current_frame)

    dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    tracker = ILQRTrajectoryTracker(
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

    history_len = required_markov_history(markov_deriv)
    current_emb = encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_history = [current_emb] * history_len
    current_state = make_markov_state(current_history, markov_deriv)

    tracking_frames = [current_frame.copy()]
    diffusion_plan_videos: list[str] = []
    hidden_plan_videos: list[str] = []
    first_reference_frames: list[np.ndarray] = [current_frame.copy()]
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    diffusion_plan_actions_env: list[list[list[float]]] = []
    diffusion_plan_states_raw: list[list[list[float]]] = []
    diffusion_plan_block_poses: list[list[list[float]]] = []
    diffusion_plan_agent_positions: list[list[list[float]]] = []
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    latent_track_errors: list[float] = []
    block_track_errors: list[float] = []
    stop_reason = "max_mpc_steps"

    try:
        pbar = tqdm(range(args.max_mpc_steps), desc="Tracking MPC")
        for step_idx in pbar:
            current_raw_state = extract_full_state(track_env)
            diffusion_plan = rollout_diffusion_mpc_plan(
                bundle=diffusion_bundle,
                start_state=current_raw_state,
                horizon=args.horizon,
                action_mode=args.action_mode,
                render_width=img_size,
                render_height=img_size,
            )
            reference_markov = build_reference_markov_states(
                model=model,
                reference_frames=diffusion_plan["hidden_frames"],
                device=device,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                frame_batch_size=args.frame_batch_size,
                markov_deriv=markov_deriv,
            )
            if int(reference_markov.shape[-1]) != markov_state_dim:
                raise ValueError(
                    f"State dimension mismatch: config says {markov_state_dim}, built {reference_markov.shape[-1]}."
                )

            diffusion_plan_actions_env.append(
                [np.asarray(action, dtype=np.float32).tolist() for action in diffusion_plan["actions_env"]]
            )
            diffusion_plan_states_raw.append([np.asarray(state, dtype=np.float64).tolist() for state in diffusion_plan["states"]])
            diffusion_plan_block_poses.append(
                [np.asarray(block_pose, dtype=np.float32).tolist() for block_pose in diffusion_plan["block_poses"]]
            )
            diffusion_plan_agent_positions.append(
                [np.asarray(agent_pos, dtype=np.float32).tolist() for agent_pos in diffusion_plan["agent_positions"]]
            )

            if len(diffusion_plan["visible_frames"]) >= 2:
                first_reference_frames.append(diffusion_plan["hidden_frames"][1].copy())
            else:
                first_reference_frames.append(diffusion_plan["hidden_frames"][0].copy())

            if step_idx < 5:
                diffusion_plan_videos.append(
                    str(save_rollout_video(diffusion_plan["visible_frames"], out_dir / f"diffusion_plan_{step_idx:03d}.mp4", args.video_fps))
                )
                hidden_plan_videos.append(
                    str(save_rollout_video(diffusion_plan["hidden_frames"], out_dir / f"hidden_plan_{step_idx:03d}.mp4", args.video_fps))
                )

            if reference_markov.shape[0] < 2:
                stop_reason = "diffusion_plan_too_short"
                break

            x0_np = current_state.detach().cpu().numpy().astype(np.float64)
            x_ref_np = reference_markov.detach().cpu().numpy().astype(np.float64)
            _, u_plan, solve_time, n_iters, plan_cost = tracker.solve(x0_np, x_ref_np)
            solve_times_ms.append(solve_time * 1000.0)
            ilqr_iterations.append(int(n_iters))
            ilqr_costs.append(float(plan_cost))

            u0_norm = u_plan[0].astype(np.float32)
            u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
            u0_env = raw_to_env_action(u0_raw, current_agent_pos(track_env))
            executed_actions_norm.append(u0_norm.copy())
            executed_actions_raw.append(u0_raw.copy())
            executed_actions_env.append(u0_env.copy())

            _, _, terminated, truncated, _ = track_env.step(u0_env)
            current_frame = np.asarray(track_env._render(visualize=False), dtype=np.uint8)
            next_emb = encode_single_frame(
                model,
                current_frame,
                device=device,
                img_size=img_size,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
            )
            current_history.append(next_emb)
            current_history = current_history[-history_len:]
            current_state = make_markov_state(current_history, markov_deriv)

            ref_idx = min(1, reference_markov.shape[0] - 1)
            latent_error = float(torch.linalg.vector_norm(current_state - reference_markov[ref_idx]).item())
            block_error = block_pose_distance(current_block_pose(track_env), diffusion_plan["block_poses"][ref_idx])
            latent_track_errors.append(latent_error)
            block_track_errors.append(block_error)
            tracking_frames.append(current_frame.copy())

            pbar.set_postfix(
                solve_ms=f"{solve_times_ms[-1]:.1f}",
                iters=f"{ilqr_iterations[-1]}",
                latent_err=f"{latent_error:.3f}",
                block_err=f"{block_error:.3f}",
            )

            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break

        final_block_pose = current_block_pose(track_env)
        final_agent_pos = current_agent_pos(track_env)
    finally:
        track_env.close()

    tracking_video_path = save_rollout_video(tracking_frames, out_dir / "tracking_rollout.mp4", args.video_fps)
    reference_preview_path = save_rollout_video(first_reference_frames, out_dir / "reference_first_step_preview.mp4", args.video_fps)

    metrics = {
        "seed": args.seed,
        "diffusion_model_dir": str(diffusion_model_dir),
        "dynamics_model_dir": str(dynamics_model_dir),
        "checkpoint": str(checkpoint_path),
        "config_path": str(dynamics_model_dir / "config.json"),
        "markov_deriv": markov_deriv,
        "markov_state_dim": markov_state_dim,
        "horizon": args.horizon,
        "max_mpc_steps": args.max_mpc_steps,
        "stop_reason": stop_reason,
        "num_tracking_steps": len(executed_actions_env),
        "env_action_scale": ENV_ACTION_SCALE,
        "initial_block_pose": initial_block_pose.tolist(),
        "initial_agent_pos": initial_agent_pos.tolist(),
        "start_state_raw": np.asarray(start_state, dtype=np.float64).tolist(),
        "final_block_pose": final_block_pose.tolist(),
        "final_agent_pos": final_agent_pos.tolist(),
        "latent_track_errors": latent_track_errors,
        "block_track_errors": block_track_errors,
        "solve_times_ms": solve_times_ms,
        "ilqr_iterations": ilqr_iterations,
        "ilqr_costs": ilqr_costs,
        "diffusion_plan_actions_env": diffusion_plan_actions_env,
        "diffusion_plan_states_raw": diffusion_plan_states_raw,
        "diffusion_plan_block_poses": diffusion_plan_block_poses,
        "diffusion_plan_agent_positions": diffusion_plan_agent_positions,
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_env": [action.tolist() for action in executed_actions_env],
        "diffusion_plan_videos": diffusion_plan_videos,
        "hidden_plan_videos": hidden_plan_videos,
        "reference_preview_path": str(reference_preview_path),
        "tracking_video_path": str(tracking_video_path),
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        json.dumps(
            {
                "seed": args.seed,
                "stop_reason": stop_reason,
                "num_tracking_steps": len(executed_actions_env),
                "latent_track_error_final": latent_track_errors[-1] if latent_track_errors else None,
                "block_track_error_final": block_track_errors[-1] if block_track_errors else None,
                "metrics_path": str(metrics_path),
                "tracking_video_path": str(tracking_video_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
