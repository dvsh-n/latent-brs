#!/usr/bin/env python3
"""Plan in Reacher pixel space with IPOPT MPC over a Markov-state MLP world model."""

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
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import casadi as ca
except ImportError as exc:
    raise RuntimeError(
        "casadi is required for this planner. Use the project venv, e.g. "
        "`/home/devesh/latent-brs/latent_brs_venv/bin/python reacher/plan/plan_ipopt_mpc.py`."
    ) from exc

from reacher.eval.reacher_policy_viz import configure_offscreen_framebuffer
from reacher.train.mlpdyn_train import LeWMReacherDataset
from reacher.train.reacher_policy_train import DmControlGymEnv, flatten_observation

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft"
DEFAULT_OUT_DIR = "reacher/plan/ipopt_mpc_mlpdyn"

DEVICE = "auto"
HORIZON = 25
MAX_MPC_STEPS = 500
Q_TERMINAL = 1.0
Q_STAGE = 0.005
R_CONTROL = 0.1
VIDEO_FPS = 60 
EPISODE_IDX = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
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
    parser.add_argument("--ipopt-max-iter", type=int, default=100)
    parser.add_argument("--ipopt-tol", type=float, default=1e-4)
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


def goal_reached(current_obs: np.ndarray, goal_obs: np.ndarray, threshold: float = 0.05) -> tuple[bool, float]:
    obs_err = float(np.linalg.norm(current_obs - goal_obs))
    return obs_err <= threshold, obs_err


def gelu_casadi(x: ca.MX) -> ca.MX:
    return 0.5 * x * (1.0 + ca.erf(x / np.sqrt(2.0)))


def linear_casadi(x: ca.MX, weight: np.ndarray, bias: np.ndarray) -> ca.MX:
    return ca.mtimes(weight, x) + bias


def export_mlp_dynamics(model: torch.nn.Module) -> tuple[ca.Function, int, int]:
    predictor = model.predictor
    if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
        raise ValueError(
            "This planner expects a one-step Markov MLP dynamics model with "
            "history_size=1, action_history_size=1, and num_preds=1."
        )
    if type(model.action_encoder).__name__ != "Identity":
        raise ValueError("This planner assumes an identity action encoder.")

    state_dim = int(predictor.embed_dim)
    action_dim = int(predictor.action_dim)
    layers = [module for module in predictor.net if isinstance(module, torch.nn.Linear)]
    if len(layers) < 2:
        raise ValueError("Predictor net does not match the expected MLP structure.")

    weights = [layer.weight.detach().cpu().numpy().astype(np.float64) for layer in layers]
    biases = [layer.bias.detach().cpu().numpy().astype(np.float64).reshape(-1, 1) for layer in layers]

    x = ca.MX.sym("x", state_dim)
    u = ca.MX.sym("u", action_dim)
    h = ca.vertcat(x, u)
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        h = linear_casadi(h, weight, bias)
        if idx < len(weights) - 1:
            h = gelu_casadi(h)
    return ca.Function("f_dyn", [x, u], [h]), state_dim, action_dim


class IPOPTMPCSolver:
    def __init__(
        self,
        dynamics_fn: ca.Function,
        *,
        state_dim: int,
        action_dim: int,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        ipopt_max_iter: int,
        ipopt_tol: float,
    ) -> None:
        self.dynamics_fn = dynamics_fn
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.prev_x_guess: np.ndarray | None = None
        self.prev_u_guess: np.ndarray | None = None

        opti = ca.Opti()
        x_var = opti.variable(self.state_dim, self.horizon + 1)
        u_var = opti.variable(self.action_dim, self.horizon)
        x0_param = opti.parameter(self.state_dim)
        xg_param = opti.parameter(self.state_dim)

        opti.subject_to(x_var[:, 0] == x0_param)
        for step in range(self.horizon):
            x_next = self.dynamics_fn(x_var[:, step], u_var[:, step])
            opti.subject_to(x_var[:, step + 1] == x_next)

        objective = 0
        for step in range(self.horizon):
            state_err = x_var[:, step] - xg_param
            control = u_var[:, step]
            objective += float(q_stage) * ca.sumsqr(state_err)
            objective += float(r_control) * ca.sumsqr(control)
        objective += float(q_terminal) * ca.sumsqr(x_var[:, self.horizon] - xg_param)
        opti.minimize(objective)

        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": int(ipopt_max_iter),
                "tol": float(ipopt_tol),
                "sb": "yes",
                "warm_start_init_point": "yes",
            },
            "print_time": False,
            "expand": False,
        }
        opti.solver("ipopt", opts)

        self.opti = opti
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.xg_param = xg_param

    def _make_initial_state_guess(self, x0: np.ndarray) -> np.ndarray:
        if self.prev_x_guess is None:
            return np.repeat(x0.reshape(-1, 1), self.horizon + 1, axis=1)
        guess = np.empty_like(self.prev_x_guess)
        guess[:, :-1] = self.prev_x_guess[:, 1:]
        guess[:, -1] = self.prev_x_guess[:, -1]
        guess[:, 0] = x0
        return guess

    def _make_initial_action_guess(self) -> np.ndarray:
        if self.prev_u_guess is None:
            return np.zeros((self.action_dim, self.horizon), dtype=np.float64)
        guess = np.empty_like(self.prev_u_guess)
        guess[:, :-1] = self.prev_u_guess[:, 1:]
        guess[:, -1] = self.prev_u_guess[:, -1]
        return guess

    def solve(self, x0: np.ndarray, x_goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.xg_param, x_goal)
        self.opti.set_initial(self.x_var, self._make_initial_state_guess(x0))
        self.opti.set_initial(self.u_var, self._make_initial_action_guess())

        t0 = time.perf_counter()
        solution = self.opti.solve()
        solve_time = time.perf_counter() - t0

        x_sol = np.asarray(solution.value(self.x_var), dtype=np.float64)
        u_sol = np.asarray(solution.value(self.u_var), dtype=np.float64)
        self.prev_x_guess = x_sol
        self.prev_u_guess = u_sol
        return x_sol, u_sol, solve_time


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
        raise ValueError(f"Expected history_size=1 for the finetuned MLP model, got {history_size}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))

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
    start_emb = true_latents[0]
    goal_emb = true_latents[-1]
    start_state = make_markov_state(start_emb)
    goal_state = make_markov_state(goal_emb)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

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
    save_rgb_image(out_dir / "sim_start_image.png", render_start)

    dynamics_fn, state_dim, dynamics_action_dim = export_mlp_dynamics(model)
    if state_dim != markov_state_dim:
        raise ValueError(f"Dynamics state dim {state_dim} does not match config state dim {markov_state_dim}.")
    if dynamics_action_dim != action_dim:
        raise ValueError(f"Dynamics action dim {dynamics_action_dim} does not match config action dim {action_dim}.")

    mpc_solver = IPOPTMPCSolver(
        dynamics_fn,
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_tol=args.ipopt_tol,
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
    prev_emb = None
    current_state = make_markov_state(current_emb, prev_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)
    current_obs = flatten_observation(env._env.task.get_observation(env._env.physics)).astype(np.float32)

    rollout_frames = [current_frame.copy()]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    observation_goal_distances = [float(np.linalg.norm(current_obs - goal_obs))]
    solve_times_ms: list[float] = []
    stop_reason = "max_mpc_steps"

    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        x_plan, u_plan, solve_time = mpc_solver.solve(current_state_np, goal_state_np)
        solve_times_ms.append(solve_time * 1000.0)

        u0_norm = u_plan[:, 0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        obs, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = make_markov_state(next_emb, current_emb)
        prev_emb = current_emb
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
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

    metrics = {
        "episode_idx": episode_idx,
        "episode_seed": episode_seed,
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "train_dataset_path": str(train_dataset_path),
        "horizon": int(args.horizon),
        "max_mpc_steps": int(args.max_mpc_steps),
        "num_executed_steps": int(len(executed_actions_raw)),
        "stop_reason": stop_reason,
        "goal_reached": stop_reason == "goal_reached",
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "history_size": history_size,
        "embed_dim": embed_dim,
        "markov_state_dim": markov_state_dim,
        "measurement_definition": "state = concat(projector(encoder(image)), zeros_like(projector(encoder(image))))",
        "start_state_velocity_zero": True,
        "goal_state_velocity_zero": True,
        "initial_goal_latent_distance": latent_goal_distances[0],
        "final_goal_latent_distance": latent_goal_distances[-1],
        "initial_goal_embedding_distance": embedding_goal_distances[0],
        "final_goal_embedding_distance": embedding_goal_distances[-1],
        "initial_goal_observation_distance": observation_goal_distances[0],
        "final_goal_observation_distance": observation_goal_distances[-1],
        "mean_solve_time_ms": float(np.mean(solve_times_ms)) if solve_times_ms else 0.0,
        "max_solve_time_ms": float(np.max(solve_times_ms)) if solve_times_ms else 0.0,
        "goal_qpos": qpos_np[-1].tolist(),
        "goal_qvel": qvel_np[-1].tolist(),
        "final_qpos": final_qpos.tolist(),
        "final_qvel": final_qvel.tolist(),
        "final_obs": final_obs.tolist(),
        "video_path": video_path,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if executed_actions_raw:
        np.save(out_dir / "actions_raw.npy", np.stack(executed_actions_raw, axis=0))
        np.save(out_dir / "actions_normalized.npy", np.stack(executed_actions_norm, axis=0))
    np.save(out_dir / "latent_goal_distances.npy", np.asarray(latent_goal_distances, dtype=np.float32))
    np.save(out_dir / "embedding_goal_distances.npy", np.asarray(embedding_goal_distances, dtype=np.float32))
    np.save(out_dir / "observation_goal_distances.npy", np.asarray(observation_goal_distances, dtype=np.float32))

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
