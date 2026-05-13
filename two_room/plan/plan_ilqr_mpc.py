#!/usr/bin/env python3
"""Plan in Two Room pixel space with nominal iLQR MPC over a Markov-state MLP world model."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_room.shared import TwoRoomLayout, make_two_room_env, reset_two_room_env_to_state


DEFAULT_MODEL_DIR = "two_room/models/mlpdyn"
DEFAULT_OUT_DIR = "two_room/plan/ilqr_mpc_mlpdyn"

DEVICE = "auto"
SAME_ROOM = True
HORIZON = 25
MAX_MPC_STEPS = 120
Q_TERMINAL = 5.0
Q_STAGE = 0.005
R_CONTROL = 0.1
DELTA_Q_SCALE = 1e-3
VIDEO_FPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=50)
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


def preprocess_pixels(pixels: np.ndarray | torch.Tensor, *, img_size: int) -> torch.Tensor:
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        tensor = pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = F.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    pixel_mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    pixel_std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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
) -> torch.Tensor:
    batch = preprocess_pixels(pixel, img_size=img_size).to(device)
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


def load_action_stats(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]}.")
        finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
    finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
    return action_mean, np.maximum(action_std, 1e-6)


def room_side(state: np.ndarray, wall_axis: int) -> int:
    room_idx = 0 if wall_axis == 1 else 1
    return int(float(state[room_idx]) >= 112.0)


def sample_start_and_goal(
    env,
    *,
    same_room: bool,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, info = env.reset(seed=seed)
    start_state = np.asarray(info["state"], dtype=np.float32)
    goal_state = np.asarray(info["goal_state"], dtype=np.float32)

    if same_room:
        start_side = room_side(start_state, env.wall_axis)
        for _ in range(256):
            candidate = env._sample_position_in_room(start_side)
            if float(np.linalg.norm(candidate - start_state)) > max(env.SUCCESS_DISTANCE * 1.5, env.agent_speed * 2.0):
                goal_state = np.asarray(candidate, dtype=np.float32)
                break
        else:
            goal_state = np.asarray(env._sample_position_in_room(start_side), dtype=np.float32)
        env.set_goal_state(goal_state)

    frame = reset_two_room_env_to_state(env, start_state, goal_state=goal_state)
    return start_state, goal_state, frame


def goal_reached(current_state: np.ndarray, goal_state: np.ndarray, threshold: float) -> tuple[bool, float]:
    state_err = float(np.linalg.norm(current_state - goal_state))
    return state_err <= threshold, state_err


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
        half_state_dim = self.state_dim // 2
        if 2 * half_state_dim != self.state_dim:
            raise ValueError(f"Expected an even Markov state dimension, got {self.state_dim}.")
        q_diag = torch.ones(self.state_dim, dtype=torch.float32, device=device)
        q_diag[half_state_dim:] = DELTA_Q_SCALE
        self.q_state = torch.diag(q_diag)
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
            cost = cost + self.q_stage * (state_err @ self.q_state @ state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
        cost = cost + self.q_terminal * (terminal_err @ self.q_state @ terminal_err)
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
            v_x = 2.0 * self.q_terminal * (self.q_state @ terminal_err)
            v_xx = 2.0 * self.q_terminal * self.q_state
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_goal
                u = u_seq[step]
                a = a_seq[step]
                b = b_seq[step]

                l_x = 2.0 * self.q_stage * (self.q_state @ x_err)
                l_u = 2.0 * self.r_control * u
                l_xx = 2.0 * self.q_stage * self.q_state
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
        raise ValueError(f"Expected history_size=1 for the finetuned MLP model, got {history_size}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 24))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))

    train_dataset_path = Path(str(config["dataset_path"])).expanduser().resolve()
    action_mean, action_std = load_action_stats(train_dataset_path, action_dim)
    sample_seed = args.seed
    layout = TwoRoomLayout()
    probe_env = make_two_room_env(render_mode="rgb_array", render_target=False, layout=layout)
    height = int(probe_env.IMG_SIZE)
    width = int(probe_env.IMG_SIZE)
    success_distance = float(probe_env.SUCCESS_DISTANCE)
    probe_env.close()

    run_name = f"{int(time.time())}_sampled"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_two_room_env(
        render_mode="rgb_array",
        render_target=False,
        agent_speed=5.0,
        layout=layout,
    )
    start_state, goal_state, start_frame = sample_start_and_goal(env, same_room=SAME_ROOM, seed=sample_seed)
    goal_frame = reset_two_room_env_to_state(env, goal_state, goal_state=goal_state)
    current_frame = reset_two_room_env_to_state(env, start_state, goal_state=goal_state)

    save_rgb_image(out_dir / "start_image.png", start_frame)
    save_rgb_image(out_dir / "goal_image.png", goal_frame)

    encoded = encode_frames(
        model,
        preprocess_pixels(np.stack([start_frame, goal_frame], axis=0), img_size=img_size),
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_emb = encoded[0]
    goal_emb = encoded[1]
    current_emb = start_emb
    current_latent_state = make_markov_state(current_emb)
    goal_latent_state = make_markov_state(goal_emb)
    if int(current_latent_state.numel()) != markov_state_dim:
        raise ValueError(
            f"State dimension mismatch: config says {markov_state_dim}, built {current_latent_state.numel()}."
        )

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

    rollout_frames = [current_frame.copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_latent_state - goal_latent_state).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    state_goal_distances = [float(np.linalg.norm(start_state - goal_state))]
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    stop_reason = "max_mpc_steps"
    current_state = start_state.copy()

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        current_state_np64 = current_latent_state.detach().cpu().numpy().astype(np.float64)
        goal_state_np64 = goal_latent_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np64, goal_state_np64)
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))
        ilqr_costs.append(float(plan_cost))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, info = env.step(u0_raw)
        current_state = np.asarray(info["state"], dtype=np.float32)
        current_frame = np.asarray(env.render(), dtype=np.uint8)
        next_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size)
        current_latent_state = make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        latent_goal_distance = float(torch.linalg.vector_norm(current_latent_state - goal_latent_state).item())
        embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
        state_goal_distance = float(np.linalg.norm(current_state - goal_state))
        latent_goal_distances.append(latent_goal_distance)
        embedding_goal_distances.append(embedding_goal_distance)
        state_goal_distances.append(state_goal_distance)

        pbar.set_postfix(
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            iters=f"{ilqr_iterations[-1]}",
            latent_goal=f"{latent_goal_distance:.3f}",
            state_goal=f"{state_goal_distance:.3f}",
        )

        reached_goal, _ = goal_reached(current_state, goal_state, success_distance)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    metrics = {
        "episode_idx": None,
        "sample_seed": sample_seed,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(train_dataset_path),
        "same_room": bool(SAME_ROOM),
        "start_state": start_state.tolist(),
        "goal_state": goal_state.tolist(),
        "final_state": current_state.tolist(),
        "success_distance": success_distance,
        "stop_reason": stop_reason,
        "num_mpc_steps": len(executed_actions_raw),
        "final_state_goal_distance": state_goal_distances[-1],
        "final_embedding_goal_distance": embedding_goal_distances[-1],
        "final_latent_goal_distance": latent_goal_distances[-1],
        "mean_solve_time_ms": float(np.mean(solve_times_ms)) if solve_times_ms else 0.0,
        "max_solve_time_ms": float(np.max(solve_times_ms)) if solve_times_ms else 0.0,
        "mean_ilqr_iterations": float(np.mean(ilqr_iterations)) if ilqr_iterations else 0.0,
        "video_path": video_path,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    env.close()
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
