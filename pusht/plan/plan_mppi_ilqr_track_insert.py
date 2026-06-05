#!/usr/bin/env python3
"""Plan PushT insertion in pixel space with MPPI warm starts tracked by iLQR."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import traceback
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_SLS_SRC = REPO_ROOT / "third_party" / "gpu_sls" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if GPU_SLS_SRC.is_dir() and str(GPU_SLS_SRC) not in sys.path:
    sys.path.insert(0, str(GPU_SLS_SRC))

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax

from gpu_sls.mppi_planner import MPPIPlanner
from pusht.plan.plan_ilqr_mpc import (
    CONTROL_MAX_NORM,
    CONTROL_MIN_NORM,
    ENV_ACTION_SCALE,
    block_pose_distance,
    current_agent_pos,
    current_block_pose,
    encode_frames,
    encode_single_frame,
    extract_full_state,
    latest_object_checkpoint,
    load_action_stats,
    load_config,
    load_model,
    make_markov_state,
    maybe_cuda_synchronize,
    MarkovDynamicsTorch,
    normalized_to_raw_action,
    preprocess_pixels,
    pusht_agent_action_bounds,
    raw_to_env_action,
    require_device,
    resolve_dataset_paths,
    resolve_model_dir,
    reset_env_to_state,
    save_rgb_image,
    save_rollout_video,
    set_goal_pose,
)
from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    get_pusht_goal_pose,
    make_obstacle_pusht_init_state,
    make_pusht_env,
)
from pusht.train.mlpdyn_train import required_markov_history

DEFAULT_MODEL_DIR = "pusht/models/mlpdyn_embd_8_insert"
DEFAULT_DATASET_PATH = "pusht/data/pusht_diffusion_insertion.h5"
DEFAULT_OUT_DIR = "pusht/plan/mppi_ilqr_track_insert_mlpdyn"
DEVICE = "auto"
MPPI_HORIZON = 65
ILQR_HORIZON = 15
MAX_MPC_STEPS = 400
MPPI_Q_TERMINAL = 10.0
MPPI_Q_STAGE = 0.05
MPPI_R_CONTROL = 0.01
ILQR_Q_TERMINAL = 10.0
ILQR_Q_STAGE = 1.0
ILQR_R_CONTROL = 0.01
VIDEO_FPS = 10
MPPI_SAMPLES = 2048
MPPI_UPDATE_ITERS = 5
MPPI_REWARD_WEIGHT = 20.0
MPPI_NOISE_LEVEL = 0.35
MPPI_NOISE_DECAY = 1.0
MPPI_BETA_FILTER = 0.7
JAX_PLATFORM = "auto"
JAX_FALLBACK_ENV = "PUSHT_MPPI_JAX_FALLBACK"
INSERTION_INIT_BLOCK_OFFSET = (110.0, 120.0)
INSERTION_INIT_MAX_TILT_DEG = 10.0
INSERTION_INIT_AXIS_THRESHOLD = 10.0
INSERTION_INIT_PUSHER_FACE_OFFSET = 15.0
INSERTION_PUSHER_TOP_EDGE_MARGIN = 5.0
INSERTION_PUSHER_TOP_EDGE_FAIL_STEPS = 5
PUSHT_TEE_CAP_TOP_Y = 0.0
PUSHT_CANVAS_SIZE = 512.0
PUSHT_TEE_SCALE = 30.0
PUSHT_TEE_LENGTH = 4.0
INSERTION_VISUAL_BUFFER = 12.0
INSERTION_VISUAL_THICKNESS = 80.0
INSERTION_COLOR_RGBA = (255, 140, 0, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--mppi-horizon", type=int, default=MPPI_HORIZON)
    parser.add_argument("--ilqr-horizon", type=int, default=ILQR_HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--mppi-q-terminal", type=float, default=MPPI_Q_TERMINAL)
    parser.add_argument("--mppi-q-stage", type=float, default=MPPI_Q_STAGE)
    parser.add_argument("--mppi-r-control", type=float, default=MPPI_R_CONTROL)
    parser.add_argument("--ilqr-q-terminal", type=float, default=ILQR_Q_TERMINAL)
    parser.add_argument("--ilqr-q-stage", type=float, default=ILQR_Q_STAGE)
    parser.add_argument("--ilqr-r-control", type=float, default=ILQR_R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=50)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--mppi-samples", type=int, default=MPPI_SAMPLES)
    parser.add_argument("--mppi-update-iters", type=int, default=MPPI_UPDATE_ITERS)
    parser.add_argument("--mppi-reward-weight", type=float, default=MPPI_REWARD_WEIGHT)
    parser.add_argument("--mppi-noise-level", type=float, default=MPPI_NOISE_LEVEL)
    parser.add_argument("--mppi-noise-decay", type=float, default=MPPI_NOISE_DECAY)
    parser.add_argument("--mppi-beta-filter", type=float, default=MPPI_BETA_FILTER)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu"), default=JAX_PLATFORM)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. Any unset specific seed is deterministically derived from this.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Seed for insertion start offset/tilt sampling. If unset, one is selected and recorded.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Seed for PushT env resets. If unset, one is selected and recorded.",
    )
    parser.add_argument(
        "--mppi-seed",
        type=int,
        default=None,
        help="Seed for MPPI sampling. If unset, one is selected and recorded.",
    )
    parser.add_argument(
        "--insertion-init-block-offset",
        type=float,
        nargs=2,
        default=INSERTION_INIT_BLOCK_OFFSET,
        metavar=("MIN", "MAX"),
        help="Range to sample the insertion start block offset from the green goal.",
    )
    parser.add_argument(
        "--insertion-init-max-tilt-deg",
        type=float,
        default=INSERTION_INIT_MAX_TILT_DEG,
        help="Maximum absolute tilt sampled for the insertion start.",
    )
    parser.add_argument(
        "--insertion-init-axis-threshold",
        type=float,
        default=INSERTION_INIT_AXIS_THRESHOLD,
        help="Diffusion-plan insertion success threshold along the sampled insertion axis.",
    )
    parser.add_argument(
        "--insertion-init-pusher-face-offset",
        type=float,
        default=INSERTION_INIT_PUSHER_FACE_OFFSET,
        help="Distance from the T top face center to the pusher center.",
    )
    parser.add_argument(
        "--insertion-pusher-top-edge-margin",
        type=float,
        default=INSERTION_PUSHER_TOP_EDGE_MARGIN,
        help="Failure margin when the pusher crosses below the T cap top edge.",
    )
    parser.add_argument(
        "--insertion-pusher-top-edge-fail-steps",
        type=int,
        default=INSERTION_PUSHER_TOP_EDGE_FAIL_STEPS,
        help="Consecutive steps below the T cap top edge before terminating as failure.",
    )
    parser.add_argument(
        "--insertion-visual-buffer",
        type=float,
        default=INSERTION_VISUAL_BUFFER,
        help="Pixel gap between the visual-only orange insertion guide and the goal T slot.",
    )
    parser.add_argument(
        "--insertion-visual-thickness",
        type=float,
        default=INSERTION_VISUAL_THICKNESS,
        help="Pixel thickness of the visual-only orange insertion guide walls.",
    )
    return parser.parse_args()


def configure_jax_platform(device: torch.device, requested_platform: str) -> str:
    if requested_platform == "auto":
        return "gpu" if device.type == "cuda" else "cpu"
    return requested_platform


def initialize_jax(platform: str) -> tuple[Any, Any, Any]:
    os.environ["JAX_PLATFORMS"] = platform

    import jax
    import jax.numpy as jnp
    from jax import config, lax

    config.update("jax_default_matmul_precision", "highest")
    config.update("jax_enable_x64", False)
    return jax, jnp, lax


def build_batched_jax_dynamics(
    torch_dynamics_net: torch.nn.Module,
    device: torch.device,
    state_dim: int,
) -> Callable[[Any, Any, float, float], Any]:
    import jax
    import jax.numpy as jnp

    def _fwd_fn(x_np: np.ndarray, u_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32).copy()).to(device)
            u_t = torch.from_numpy(np.asarray(u_np, dtype=np.float32).copy()).to(device)
            inp = torch.cat((x_t, u_t), dim=-1)
            out = torch_dynamics_net(inp)
            return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)

    def jax_dynamics(x: jnp.ndarray, u: jnp.ndarray, t: float, parameter: float) -> jnp.ndarray:
        del t, parameter
        result_shape = jax.ShapeDtypeStruct(x.shape[:-1] + (state_dim,), jnp.float32)
        return jax.pure_callback(_fwd_fn, result_shape, x, u)

    return jax_dynamics


def is_wrapped_keyboard_interrupt(exc: BaseException) -> bool:
    text = "".join(traceback.format_exception_only(type(exc), exc))
    return "KeyboardInterrupt" in text and (
        "CpuCallback error calling callback" in text or "jax.pure_callback failed" in text
    )


def make_mppi_rollout_and_eval(
    jax_dynamics_fn: Callable[[Any, Any, float, float], Any],
    *,
    q_stage: float,
    q_terminal: float,
    r_control: float,
):
    import jax.numpy as jnp
    from jax import lax

    def mppi_rollout_fn(state_cur: Any, act_seqs: Any, reach_config: dict | None = None):
        del reach_config
        batch_size = act_seqs.shape[0]
        state0 = jnp.broadcast_to(state_cur, (batch_size, state_cur.shape[0]))

        def step(state_batch: Any, u_batch: Any):
            next_state_batch = jax_dynamics_fn(state_batch, u_batch, 0.0, 1.0)
            return next_state_batch, next_state_batch

        _, state_seqs = lax.scan(step, state0, act_seqs.swapaxes(0, 1))
        state_seqs = state_seqs.swapaxes(0, 1)
        return state_seqs, {}

    def mppi_eval_fn(
        state_seqs: Any,
        act_seqs: Any,
        reach_config: dict | None = None,
        aux: dict | None = None,
        *,
        goal_state: Any,
    ):
        del reach_config, aux
        if state_seqs.shape[1] > 1:
            stage_delta = state_seqs[:, :-1, :] - goal_state[None, None, :]
            stage_costs = q_stage * jnp.sum(stage_delta**2, axis=-1).sum(axis=-1)
        else:
            stage_costs = jnp.zeros((state_seqs.shape[0],), dtype=state_seqs.dtype)
        terminal_delta = state_seqs[:, -1, :] - goal_state[None, :]
        terminal_costs = q_terminal * jnp.sum(terminal_delta**2, axis=-1)
        action_costs = r_control * jnp.sum(act_seqs**2, axis=(-1, -2))
        total_costs = stage_costs + terminal_costs + action_costs
        return {"rewards": -total_costs}

    return mppi_rollout_fn, mppi_eval_fn


def shift_warmstart(U: Any) -> Any:
    import jax.numpy as jnp

    return jnp.concatenate([U[1:], U[-1:]], axis=0)


def hard_pusht_goal_distances(current_block_pose: np.ndarray, goal_block_pose: np.ndarray) -> tuple[float, float, float]:
    position_distance = float(np.linalg.norm(current_block_pose[:2].astype(np.float64) - goal_block_pose[:2].astype(np.float64)))
    yaw_distance = abs(float((current_block_pose[2] - goal_block_pose[2] + np.pi) % (2.0 * np.pi) - np.pi))
    pose_distance = block_pose_distance(current_block_pose, goal_block_pose)
    return position_distance, yaw_distance, pose_distance


def hard_pusht_goal_success(
    current_block_pose: np.ndarray,
    goal_block_pose: np.ndarray,
    *,
    position_success_threshold: float,
    yaw_success_threshold: float,
    require_yaw_success: bool,
) -> tuple[bool, dict[str, float | bool]]:
    position_distance, yaw_distance, pose_distance = hard_pusht_goal_distances(current_block_pose, goal_block_pose)
    position_success = position_distance <= position_success_threshold
    yaw_success = yaw_distance <= yaw_success_threshold
    success = bool(position_success and (yaw_success or not require_yaw_success))
    return success, {
        "position_goal_distance": position_distance,
        "yaw_goal_distance": yaw_distance,
        "block_goal_distance": pose_distance,
        "position_success": bool(position_success),
        "yaw_success": bool(yaw_success),
        "success": success,
    }


def make_visible_pixel_env(env_id: str, *, width: int, height: int, max_episode_steps: int) -> Any:
    env = make_pusht_env(
        env_id,
        obs_type="pixels",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
        hide_target=False,
    )
    env.reset(seed=0)
    return env


def save_rgb_video(path: Path, frames: list[np.ndarray], *, fps: int) -> Path:
    if not frames:
        raise ValueError("Cannot save an empty video.")
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to save rollout videos.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = path.with_suffix(".mp4")
    gif_path = path.with_suffix(".gif")
    try:
        imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        return mp4_path
    except Exception:
        imageio.mimwrite(gif_path, frames, fps=fps)
        return gif_path


def resolve_run_seeds(args: argparse.Namespace) -> dict[str, int | None]:
    requested = {
        "base": args.seed,
        "init": args.init_seed,
        "env": args.env_seed,
        "mppi": args.mppi_seed,
    }
    names = ("init", "env", "mppi")
    resolved: dict[str, int | None] = dict(requested)
    missing = [name for name in names if resolved[name] is None]
    if missing:
        if args.seed is None:
            generated = np.random.SeedSequence().generate_state(len(missing), dtype=np.uint32)
            for name, seed_value in zip(missing, generated, strict=True):
                resolved[name] = int(seed_value)
        else:
            generated = np.random.SeedSequence(int(args.seed)).generate_state(len(names), dtype=np.uint32)
            derived = {name: int(seed_value) for name, seed_value in zip(names, generated, strict=True)}
            for name in missing:
                resolved[name] = derived[name]
    return {
        "requested_base_seed": requested["base"],
        "requested_init_seed": requested["init"],
        "requested_env_seed": requested["env"],
        "requested_mppi_seed": requested["mppi"],
        "init_seed": int(resolved["init"]),
        "env_seed": int(resolved["env"]),
        "mppi_seed": int(resolved["mppi"]),
    }


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _local_rect_to_frame_polygon(
    goal_pose: np.ndarray,
    rect: tuple[float, float, float, float],
    frame_shape: tuple[int, ...],
) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = rect
    corners = np.asarray(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float64,
    )
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    world = goal_pose[:2] + corners @ _rotation_matrix(float(goal_pose[2])).T
    height, width = frame_shape[:2]
    scale = np.asarray([float(width) / PUSHT_CANVAS_SIZE, float(height) / PUSHT_CANVAS_SIZE], dtype=np.float64)
    pixels = world * scale
    return [(float(x), float(y)) for x, y in pixels]


def insertion_visual_rects(*, buffer: float, thickness: float) -> list[tuple[float, float, float, float]]:
    buffer = float(buffer)
    thickness = float(thickness)
    stem_xmin = -0.5 * PUSHT_TEE_SCALE
    stem_xmax = 0.5 * PUSHT_TEE_SCALE
    cap_ymax = PUSHT_TEE_SCALE
    stem_ymax = PUSHT_TEE_LENGTH * PUSHT_TEE_SCALE

    inner_xmin = stem_xmin - buffer
    inner_xmax = stem_xmax + buffer
    inner_ymin = cap_ymax + buffer
    inner_ymax = stem_ymax + buffer
    outer_xmin = inner_xmin - thickness
    outer_xmax = inner_xmax + thickness
    outer_ymax = inner_ymax + thickness

    return [
        (outer_xmin, inner_xmin, inner_ymin, outer_ymax),
        (inner_xmax, outer_xmax, inner_ymin, outer_ymax),
        (outer_xmin, outer_xmax, inner_ymax, outer_ymax),
    ]


def render_insertion_obstacle_frame(
    frame: np.ndarray,
    goal_pose: np.ndarray,
    *,
    buffer: float,
    thickness: float,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for rect in insertion_visual_rects(buffer=buffer, thickness=thickness):
        polygon = _local_rect_to_frame_polygon(np.asarray(goal_pose, dtype=np.float64), rect, frame.shape)
        draw.polygon(polygon, fill=INSERTION_COLOR_RGBA)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def insertion_init_axis(block_theta: float) -> np.ndarray:
    return np.asarray([-np.sin(block_theta), np.cos(block_theta)], dtype=np.float64)


def insertion_axis_distance(block_pose: np.ndarray, goal_pose: np.ndarray, axis: np.ndarray) -> float:
    block_xy = np.asarray(block_pose, dtype=np.float64).reshape(-1)[:2]
    goal_xy = np.asarray(goal_pose, dtype=np.float64).reshape(-1)[:2]
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("Insertion axis must have nonzero norm.")
    return abs(float(np.dot(block_xy - goal_xy, axis / norm)))


def pusher_local_y(block_pose: np.ndarray, agent_pos: np.ndarray) -> float:
    block_pose = np.asarray(block_pose, dtype=np.float64).reshape(-1)
    agent_pos = np.asarray(agent_pos, dtype=np.float64).reshape(-1)[:2]
    theta = float(block_pose[2])
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    local_xy = rotation.T @ (agent_pos - block_pose[:2])
    return float(local_xy[1])


def diffusion_insertion_success(
    block_pose: np.ndarray,
    goal_pose: np.ndarray,
    axis: np.ndarray,
    *,
    threshold: float,
) -> tuple[bool, float]:
    distance = insertion_axis_distance(block_pose, goal_pose, axis)
    return bool(distance <= float(threshold)), distance


class BoundedILQRTrajectoryTracker:
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
        control_min: float,
        control_max: float,
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
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.u_min = torch.full((self.action_dim,), float(control_min), dtype=torch.float32, device=device)
        self.u_max = torch.full((self.action_dim,), float(control_max), dtype=torch.float32, device=device)
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _project_control(self, u: torch.Tensor) -> torch.Tensor:
        return torch.clamp(u, min=self.u_min, max=self.u_max)

    def _make_initial_action_guess(self, warmstart_np: np.ndarray | None) -> torch.Tensor:
        if warmstart_np is not None:
            if warmstart_np.shape != (self.horizon, self.action_dim):
                raise ValueError(
                    f"Expected warmstart shape {(self.horizon, self.action_dim)}, got {warmstart_np.shape}."
                )
            return self._project_control(torch.tensor(warmstart_np, dtype=torch.float32, device=self.device))
        if self.horizon <= 1:
            return self._project_control(self.prev_u_guess.clone())
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return self._project_control(guess)

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            u_step = self._project_control(u_seq[step])
            x_curr = self.dynamics.step(x_curr, u_step)
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

    def _solve_box_qp(
        self,
        q_uu: torch.Tensor,
        q_u: torch.Tensor,
        q_ux: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_low = self.u_min - u
        delta_high = self.u_max - u
        best_obj: float | None = None
        best_k: torch.Tensor | None = None
        best_kk: torch.Tensor | None = None
        grad_tol = 1e-5
        bound_tol = 1e-6

        for status in itertools.product((-1, 0, 1), repeat=self.action_dim):
            k = torch.zeros((self.action_dim,), dtype=torch.float32, device=self.device)
            kk = torch.zeros((self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)
            free_idx = [idx for idx, tag in enumerate(status) if tag == 0]
            clamped_idx = [idx for idx, tag in enumerate(status) if tag != 0]

            for idx in clamped_idx:
                k[idx] = delta_low[idx] if status[idx] < 0 else delta_high[idx]

            if free_idx:
                free = torch.tensor(free_idx, dtype=torch.long, device=self.device)
                q_ff = q_uu.index_select(0, free).index_select(1, free)
                rhs_k = q_u.index_select(0, free)
                if clamped_idx:
                    clamped = torch.tensor(clamped_idx, dtype=torch.long, device=self.device)
                    q_fc = q_uu.index_select(0, free).index_select(1, clamped)
                    rhs_k = rhs_k + q_fc @ k.index_select(0, clamped)
                try:
                    k_free = -torch.linalg.solve(q_ff, rhs_k.unsqueeze(1)).squeeze(1)
                    kk_free = -torch.linalg.solve(q_ff, q_ux.index_select(0, free))
                except RuntimeError:
                    continue
                k[free] = k_free
                kk[free] = kk_free

            feasible = bool(torch.all(k >= delta_low - bound_tol) and torch.all(k <= delta_high + bound_tol))
            if not feasible:
                continue

            grad = q_u + q_uu @ k
            kkt_ok = True
            for idx, tag in enumerate(status):
                grad_i = float(grad[idx].item())
                if tag == 0 and abs(grad_i) > grad_tol:
                    kkt_ok = False
                    break
                if tag < 0 and grad_i < -grad_tol:
                    kkt_ok = False
                    break
                if tag > 0 and grad_i > grad_tol:
                    kkt_ok = False
                    break
            if not kkt_ok:
                continue

            obj = float((0.5 * torch.dot(k, q_uu @ k) + torch.dot(q_u, k)).item())
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_k = k
                best_kk = kk

        if best_k is None or best_kk is None:
            raise RuntimeError("Box-constrained control subproblem failed.")
        return best_k, best_kk

    def solve(
        self,
        x0_np: np.ndarray,
        x_ref_np: np.ndarray,
        warmstart_np: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        if x_ref_np.shape != (self.horizon + 1, self.state_dim):
            raise ValueError(
                f"Expected x_ref_np with shape {(self.horizon + 1, self.state_dim)}, got {x_ref_np.shape}."
            )

        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_ref = torch.tensor(x_ref_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess(warmstart_np)

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
                    k, kk = self._solve_box_qp(q_uu, q_u, q_ux, u)
                except RuntimeError:
                    backward_ok = False
                    break

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
                    u_candidate = u_seq[step] + alpha * k_seq[step] + kk_seq[step] @ dx
                    u_new[step] = self._project_control(u_candidate)
                    x_new[step + 1] = self.dynamics.step(x_new[step], u_new[step])
                new_cost = float(self._trajectory_cost(x_new, u_new, x_ref).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    candidate_best = (x_new, u_new, new_cost)
                    accepted = True
                    break

            if not accepted:
                reg = min(reg * 10.0, 1e6)
                if reg >= 1e6:
                    break
                continue

            prev_u_seq = u_seq
            x_traj, u_seq, new_cost = candidate_best
            max_du = float(torch.max(torch.abs(u_seq - prev_u_seq)).item())
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
    if args.ilqr_horizon < 1:
        raise ValueError(f"--ilqr-horizon must be positive, got {args.ilqr_horizon}.")
    if args.mppi_horizon < args.ilqr_horizon:
        raise ValueError(
            f"--mppi-horizon must be >= --ilqr-horizon, got {args.mppi_horizon} and {args.ilqr_horizon}."
        )
    insertion_init_block_offset_range = np.asarray(args.insertion_init_block_offset, dtype=np.float64).reshape(-1)
    if insertion_init_block_offset_range.shape != (2,):
        raise ValueError("--insertion-init-block-offset expects exactly two values: MIN MAX.")
    if np.any(insertion_init_block_offset_range < 0.0):
        raise ValueError("--insertion-init-block-offset values must be >= 0.")
    if insertion_init_block_offset_range[1] < insertion_init_block_offset_range[0]:
        raise ValueError("--insertion-init-block-offset MAX must be >= MIN.")
    if args.insertion_init_max_tilt_deg < 0.0:
        raise ValueError("--insertion-init-max-tilt-deg must be >= 0.")
    if args.insertion_init_axis_threshold < 0.0:
        raise ValueError("--insertion-init-axis-threshold must be >= 0.")
    if args.insertion_init_pusher_face_offset < 0.0:
        raise ValueError("--insertion-init-pusher-face-offset must be >= 0.")
    if args.insertion_pusher_top_edge_margin < 0.0:
        raise ValueError("--insertion-pusher-top-edge-margin must be >= 0.")
    if args.insertion_pusher_top_edge_fail_steps < 1:
        raise ValueError("--insertion-pusher-top-edge-fail-steps must be >= 1.")
    if args.insertion_visual_buffer < 0.0:
        raise ValueError("--insertion-visual-buffer must be >= 0.")
    if args.insertion_visual_thickness < 0.0:
        raise ValueError("--insertion-visual-thickness must be >= 0.")
    device = require_device(args.device)
    requested_jax_platform = configure_jax_platform(device, args.jax_platform)
    fallback_enabled = args.jax_platform == "auto" and os.environ.get(JAX_FALLBACK_ENV) != "1"
    model_dir = resolve_model_dir(args)
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_seeds = resolve_run_seeds(args)
    init_seed = int(run_seeds["init_seed"])
    env_seed = int(run_seeds["env_seed"])
    mppi_seed = int(run_seeds["mppi_seed"])

    model_config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    model = load_model(checkpoint_path, device)

    markov_deriv = int(model_config.get("markov_deriv", 1))
    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv for the MLP model, got {markov_deriv}.")

    img_size = int(model_config.get("img_size", 224))
    frameskip = int(model_config.get("frameskip", 1))
    action_dim = int(model_config.get("action_dim", 2))
    embed_dim = int(model_config.get("embed_dim", 48))
    markov_state_dim = int(model_config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    if frameskip != 1:
        raise ValueError(
            f"This PushT MPC planner currently supports frameskip=1 only, but the model config has frameskip={frameskip}."
        )

    train_dataset_paths = resolve_dataset_paths(model_config.get("dataset_path"), dataset_path)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    action_mean, action_std = load_action_stats(train_dataset_paths, action_dim)

    rng = np.random.default_rng(init_seed)
    insertion_init_sampled_block_offset = float(
        rng.uniform(insertion_init_block_offset_range[0], insertion_init_block_offset_range[1])
    )
    insertion_init_sampled_tilt_deg = float(
        rng.uniform(-float(args.insertion_init_max_tilt_deg), float(args.insertion_init_max_tilt_deg))
    )
    insertion_init_horizontal_displacement = float(
        insertion_init_sampled_block_offset * np.tan(np.deg2rad(insertion_init_sampled_tilt_deg))
    )

    height = img_size
    width = img_size
    setup_env = make_visible_pixel_env(args.env_id, width=width, height=height, max_episode_steps=args.max_mpc_steps)
    try:
        setup_env.reset(seed=env_seed)
        goal_pose = get_pusht_goal_pose(setup_env)
        start_env_state = make_obstacle_pusht_init_state(
            goal_pose,
            block_offset=insertion_init_sampled_block_offset,
            tilt_deg=insertion_init_sampled_tilt_deg,
            pusher_face_offset=float(args.insertion_init_pusher_face_offset),
        )
        goal_env_state = make_obstacle_pusht_init_state(
            goal_pose,
            block_offset=0.0,
            tilt_deg=0.0,
            pusher_face_offset=float(args.insertion_init_pusher_face_offset),
        )
        start_image = reset_env_to_state(setup_env.unwrapped, start_env_state)
        goal_image = reset_env_to_state(setup_env.unwrapped, goal_env_state)
    finally:
        setup_env.close()

    start_block_pose = start_env_state[2:5].astype(np.float32, copy=True)
    goal_block = goal_pose.astype(np.float32, copy=True)
    insertion_axis = insertion_init_axis(float(start_block_pose[2]))

    run_name = f"{int(time.time())}_insert_init_{init_seed:010d}_mppi_{mppi_seed:010d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = preprocess_pixels(
        np.stack([start_image, goal_image], axis=0),
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    target_latents = encode_frames(
        model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    history_len = required_markov_history(markov_deriv)
    start_emb = target_latents[0]
    goal_emb = target_latents[1]
    start_history = [start_emb] * history_len
    goal_history = [goal_emb] * history_len
    start_state = make_markov_state(start_history, markov_deriv)
    goal_state = make_markov_state(goal_history, markov_deriv)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    save_rgb_image(out_dir / "start_image.png", start_image)
    save_rgb_image(out_dir / "goal_image.png", goal_image)
    save_rgb_image(
        out_dir / "start_image_insertion_obstacle.png",
        render_insertion_obstacle_frame(
            start_image,
            goal_pose,
            buffer=float(args.insertion_visual_buffer),
            thickness=float(args.insertion_visual_thickness),
        ),
    )
    save_rgb_image(
        out_dir / "goal_image_insertion_obstacle.png",
        render_insertion_obstacle_frame(
            goal_image,
            goal_pose,
            buffer=float(args.insertion_visual_buffer),
            thickness=float(args.insertion_visual_thickness),
        ),
    )

    try:
        jax, jnp, _lax = initialize_jax(requested_jax_platform)
        # Force backend initialization here so a broken GPU setup fails before planning starts.
        _ = jnp.zeros((1,), dtype=jnp.float32)
        jax.block_until_ready(_)
        jax_platform = requested_jax_platform
    except Exception:
        if not fallback_enabled or requested_jax_platform != "gpu":
            raise
        traceback.print_exc = lambda *args, **kwargs: None
        sys.stderr.write("JAX GPU init failed, restarting with CPU backend.\n")
        sys.stderr.flush()
        restart_env = os.environ.copy()
        restart_env["JAX_PLATFORMS"] = "cpu"
        restart_env[JAX_FALLBACK_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], restart_env)

    jax_dynamics = build_batched_jax_dynamics(model.predictor.net, device, markov_state_dim)
    mppi_rollout_fn, mppi_eval_fn = make_mppi_rollout_and_eval(
        jax_dynamics,
        q_stage=args.mppi_q_stage,
        q_terminal=args.mppi_q_terminal,
        r_control=args.mppi_r_control,
    )
    mppi_config = {
        "planning": {
            "action_dim": action_dim,
            "n_sample": args.mppi_samples,
            "horizon": args.mppi_horizon,
            "n_update_iter": args.mppi_update_iters,
            "use_last": True,
            "reject_bad": False,
            "mppi": {
                "reward_weight": args.mppi_reward_weight,
                "noise_level": args.mppi_noise_level,
                "noise_decay": args.mppi_noise_decay,
                "beta_filter": args.mppi_beta_filter,
            },
        }
    }
    action_lower_lim = jnp.full((action_dim,), CONTROL_MIN_NORM, dtype=jnp.float32)
    action_upper_lim = jnp.full((action_dim,), CONTROL_MAX_NORM, dtype=jnp.float32)
    planner = MPPIPlanner(
        config=mppi_config,
        model_rollout_fn=mppi_rollout_fn,
        evaluate_traj_fn=mppi_eval_fn,
        action_lower_lim=action_lower_lim,
        action_upper_lim=action_upper_lim,
    )
    tracker_dynamics = MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    ilqr_tracker = BoundedILQRTrajectoryTracker(
        tracker_dynamics,
        horizon=args.ilqr_horizon,
        q_terminal=args.ilqr_q_terminal,
        q_stage=args.ilqr_q_stage,
        r_control=args.ilqr_r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        control_min=CONTROL_MIN_NORM,
        control_max=CONTROL_MAX_NORM,
        device=device,
    )

    def mppi_trajopt(
        key: jax.Array,
        state_cur: jnp.ndarray,
        init_action_seq: jnp.ndarray,
        goal_state_jax: jnp.ndarray,
    ):
        return planner.trajectory_optimization(
            key,
            state_cur,
            init_action_seq,
            skip=False,
            goal_state=goal_state_jax,
        )

    jit_mppi_trajopt = jax.jit(mppi_trajopt)

    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_env: list[np.ndarray] = []
    solve_times_ms: list[float] = []
    mppi_solve_times_ms: list[float] = []
    ilqr_solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    ilqr_costs: list[float] = []
    mppi_plan_rewards: list[float] = []
    latent_track_errors: list[float] = []
    stop_reason = "max_mpc_steps"
    video_path: str | None = None
    insertion_obstacle_video_path: str | None = None
    metrics_path = out_dir / "metrics.json"
    final_block = start_block_pose.copy()
    final_agent = start_env_state[:2].astype(np.float32, copy=True)
    rollout_frames: list[np.ndarray] = []
    insertion_obstacle_rollout_frames: list[np.ndarray] = []
    latent_goal_distances: list[float] = []
    block_goal_distances: list[float] = []
    position_goal_distances: list[float] = []
    yaw_goal_distances: list[float] = []
    insertion_axis_distances: list[float] = []
    pusher_local_ys: list[float] = []
    success_log: list[bool] = []
    success = False
    pusher_top_edge_violation_steps = 0
    num_action_clips = 0
    prev_u = jnp.zeros((args.mppi_horizon, action_dim), dtype=jnp.float32)
    goal_state_jax = jnp.asarray(goal_state.detach().cpu().numpy().astype(np.float32))
    jax_key = jax.random.PRNGKey(mppi_seed)
    action_low, action_high = pusht_agent_action_bounds()

    def scalar_or_nan(values: list[float], index: int) -> float:
        return values[index] if values else float("nan")

    def save_outputs() -> dict[str, Any]:
        nonlocal video_path, insertion_obstacle_video_path
        metrics = {
            **run_seeds,
            "planner": "mppi_ilqr_track_insert",
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint_path),
            "config_path": str(model_dir / "config.json"),
            "dataset_path": str(dataset_path),
            "train_dataset_paths": [str(path) for path in train_dataset_paths],
            "markov_deriv": markov_deriv,
            "action_history_size": 1,
            "state_space": "latent_plus_finite_differences",
            "markov_state_dim": markov_state_dim,
            "frameskip": frameskip,
            "mppi_horizon": args.mppi_horizon,
            "ilqr_horizon": args.ilqr_horizon,
            "max_mpc_steps": args.max_mpc_steps,
            "stop_reason": stop_reason,
            "success": bool(success),
            "num_mpc_steps": len(executed_actions_norm),
            "action_space": "normalized_dataset_action_then_raw_then_absolute_env_target",
            "control_bound_low_norm": CONTROL_MIN_NORM,
            "control_bound_high_norm": CONTROL_MAX_NORM,
            "env_action_scale": ENV_ACTION_SCALE,
            "action_target_low": action_low.tolist(),
            "action_target_high": action_high.tolist(),
            "num_action_clips": num_action_clips,
            "start_agent_pos": start_env_state[:2].tolist(),
            "start_block_pose": start_block_pose.tolist(),
            "goal_block_pose": goal_block.tolist(),
            "goal_pose": goal_pose.tolist() if goal_pose is not None else None,
            "goal_agent_pos": goal_env_state[:2].tolist(),
            "final_agent_pos": final_agent.tolist(),
            "final_block_pose": final_block.tolist(),
            "insertion_init_block_offset": insertion_init_sampled_block_offset,
            "insertion_init_block_offset_range": insertion_init_block_offset_range.astype(float).tolist(),
            "insertion_init_max_tilt_deg": float(args.insertion_init_max_tilt_deg),
            "insertion_init_sampled_tilt_deg": insertion_init_sampled_tilt_deg,
            "insertion_init_horizontal_displacement": insertion_init_horizontal_displacement,
            "insertion_init_pusher_face_offset": float(args.insertion_init_pusher_face_offset),
            "insertion_init_axis": insertion_axis.astype(float).tolist(),
            "insertion_init_axis_threshold": float(args.insertion_init_axis_threshold),
            "insertion_pusher_top_edge_margin": float(args.insertion_pusher_top_edge_margin),
            "insertion_pusher_top_edge_fail_steps": int(args.insertion_pusher_top_edge_fail_steps),
            "insertion_pusher_top_edge_violation_steps": int(pusher_top_edge_violation_steps),
            "insertion_visual_buffer": float(args.insertion_visual_buffer),
            "insertion_visual_thickness": float(args.insertion_visual_thickness),
            "latent_goal_distance_initial": scalar_or_nan(latent_goal_distances, 0),
            "latent_goal_distance_final": scalar_or_nan(latent_goal_distances, -1),
            "block_goal_distance_initial": scalar_or_nan(block_goal_distances, 0),
            "block_goal_distance_final": scalar_or_nan(block_goal_distances, -1),
            "block_goal_distance_min": min(block_goal_distances) if block_goal_distances else float("nan"),
            "position_goal_distance_initial": scalar_or_nan(position_goal_distances, 0),
            "position_goal_distance_final": scalar_or_nan(position_goal_distances, -1),
            "position_goal_distance_min": min(position_goal_distances) if position_goal_distances else float("nan"),
            "yaw_goal_distance_initial": scalar_or_nan(yaw_goal_distances, 0),
            "yaw_goal_distance_final": scalar_or_nan(yaw_goal_distances, -1),
            "yaw_goal_distance_min": min(yaw_goal_distances) if yaw_goal_distances else float("nan"),
            "insertion_axis_distance_initial": scalar_or_nan(insertion_axis_distances, 0),
            "insertion_axis_distance_final": scalar_or_nan(insertion_axis_distances, -1),
            "insertion_axis_distance_min": min(insertion_axis_distances) if insertion_axis_distances else float("nan"),
            "latent_goal_distances": latent_goal_distances,
            "block_goal_distances": block_goal_distances,
            "position_goal_distances": position_goal_distances,
            "yaw_goal_distances": yaw_goal_distances,
            "insertion_axis_distances": insertion_axis_distances,
            "pusher_local_ys": pusher_local_ys,
            "success_log": success_log,
            "solve_times_ms": solve_times_ms,
            "mppi_solve_times_ms": mppi_solve_times_ms,
            "ilqr_solve_times_ms": ilqr_solve_times_ms,
            "ilqr_iterations": ilqr_iterations,
            "ilqr_costs": ilqr_costs,
            "latent_track_errors": latent_track_errors,
            "mppi_samples": args.mppi_samples,
            "mppi_update_iters": args.mppi_update_iters,
            "mppi_reward_weight": args.mppi_reward_weight,
            "mppi_noise_level": args.mppi_noise_level,
            "mppi_noise_decay": args.mppi_noise_decay,
            "mppi_beta_filter": args.mppi_beta_filter,
            "jax_platform": jax_platform,
            "mppi_q_stage": args.mppi_q_stage,
            "mppi_q_terminal": args.mppi_q_terminal,
            "mppi_r_control": args.mppi_r_control,
            "ilqr_q_stage": args.ilqr_q_stage,
            "ilqr_q_terminal": args.ilqr_q_terminal,
            "ilqr_r_control": args.ilqr_r_control,
            "mppi_plan_rewards": mppi_plan_rewards,
            "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
            "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
            "executed_actions_env": [action.tolist() for action in executed_actions_env],
            "video_path": video_path,
            "insertion_obstacle_video_path": insertion_obstacle_video_path,
        }
        if rollout_frames and video_path is None:
            video_path = str(save_rollout_video(rollout_frames, out_dir, fps=args.video_fps))
            metrics["video_path"] = video_path
        if insertion_obstacle_rollout_frames and insertion_obstacle_video_path is None:
            insertion_obstacle_video_path = str(
                save_rgb_video(
                    out_dir / "rollout_insertion_obstacle.mp4",
                    insertion_obstacle_rollout_frames,
                    fps=args.video_fps,
                )
            )
            metrics["insertion_obstacle_video_path"] = insertion_obstacle_video_path
        metrics["video_path"] = video_path
        metrics["insertion_obstacle_video_path"] = insertion_obstacle_video_path
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        return metrics

    plan_env = make_visible_pixel_env(args.env_id, width=width, height=height, max_episode_steps=args.max_mpc_steps)
    viz_env = make_visible_pixel_env(args.env_id, width=width, height=height, max_episode_steps=args.max_mpc_steps)
    try:
        plan_env.reset(seed=env_seed)
        viz_env.reset(seed=env_seed)
        set_goal_pose(plan_env, goal_pose)
        set_goal_pose(viz_env, goal_pose)
        hidden_start = reset_env_to_state(plan_env.unwrapped, start_env_state)
        visible_start = reset_env_to_state(viz_env.unwrapped, start_env_state)

        current_hidden_frame = hidden_start
        current_emb = encode_single_frame(
            model,
            current_hidden_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_history = [current_emb] * history_len
        current_state = make_markov_state(current_history, markov_deriv)
        current_block = current_block_pose(plan_env)

        rollout_frames = [visible_start.copy()]
        insertion_obstacle_rollout_frames = [
            render_insertion_obstacle_frame(
                visible_start,
                goal_pose,
                buffer=float(args.insertion_visual_buffer),
                thickness=float(args.insertion_visual_thickness),
            )
        ]
        latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
        initial_success, initial_axis_distance = diffusion_insertion_success(
            current_block,
            goal_block,
            insertion_axis,
            threshold=float(args.insertion_init_axis_threshold),
        )
        initial_position_distance, initial_yaw_distance, initial_pose_distance = hard_pusht_goal_distances(
            current_block,
            goal_block,
        )
        block_goal_distances = [initial_pose_distance]
        position_goal_distances = [initial_position_distance]
        yaw_goal_distances = [initial_yaw_distance]
        insertion_axis_distances = [initial_axis_distance]
        pusher_local_ys = [pusher_local_y(current_block, current_agent_pos(plan_env))]
        success_log = [bool(initial_success)]
        success = bool(initial_success)

        pbar = tqdm(range(args.max_mpc_steps), desc="MPPI+iLQR Steps")
        try:
            if success:
                stop_reason = "success"
            for _ in pbar:
                if success:
                    break
                current_state_np = current_state.detach().cpu().numpy().astype(np.float32)
                init_action_seq = shift_warmstart(prev_u)
                jax_key, subkey = jax.random.split(jax_key)

                t0 = time.perf_counter()
                plan = jit_mppi_trajopt(
                    subkey,
                    jnp.asarray(current_state_np, dtype=jnp.float32),
                    init_action_seq,
                    goal_state_jax,
                )
                jax.block_until_ready(plan["state_seq"])
                mppi_ms = (time.perf_counter() - t0) * 1000.0
                mppi_solve_times_ms.append(mppi_ms)

                mppi_u_plan = np.asarray(plan["act_seq"], dtype=np.float32)
                mppi_state_seq = np.asarray(plan["state_seq"], dtype=np.float32)
                prev_u = jnp.asarray(mppi_u_plan, dtype=jnp.float32)
                mppi_plan_rewards.append(float(plan["reward"]))

                x_ref_np = np.concatenate(
                    [current_state_np[None, :], mppi_state_seq[: args.ilqr_horizon]],
                    axis=0,
                ).astype(np.float64)
                ilqr_warmstart_np = mppi_u_plan[: args.ilqr_horizon].astype(np.float64)
                _x_track, ilqr_u_plan, ilqr_solve_time, n_iters, plan_cost = ilqr_tracker.solve(
                    current_state_np.astype(np.float64),
                    x_ref_np,
                    ilqr_warmstart_np,
                )
                ilqr_ms = ilqr_solve_time * 1000.0
                ilqr_solve_times_ms.append(ilqr_ms)
                solve_times_ms.append(mppi_ms + ilqr_ms)
                ilqr_iterations.append(int(n_iters))
                ilqr_costs.append(float(plan_cost))

                u0_norm = ilqr_u_plan[0].astype(np.float32)
                u0_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
                unclipped_u0_env = raw_to_env_action(u0_raw, current_agent_pos(plan_env))
                u0_env = raw_to_env_action(
                    u0_raw,
                    current_agent_pos(plan_env),
                    action_low=action_low,
                    action_high=action_high,
                )
                if not np.allclose(u0_env, unclipped_u0_env):
                    num_action_clips += 1
                executed_actions_norm.append(u0_norm.copy())
                executed_actions_raw.append(u0_raw.copy())
                executed_actions_env.append(u0_env.copy())

                _, _, _terminated, truncated, _ = plan_env.step(u0_env)
                current_hidden_frame = np.asarray(plan_env.unwrapped._render(visualize=False), dtype=np.uint8)
                next_emb = encode_single_frame(
                    model,
                    current_hidden_frame,
                    device=device,
                    img_size=img_size,
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                )
                current_history.append(next_emb)
                current_history = current_history[-history_len:]
                current_state = make_markov_state(current_history, markov_deriv)
                current_block = current_block_pose(plan_env)

                synced_state = extract_full_state(plan_env)
                visible_frame = reset_env_to_state(viz_env.unwrapped, synced_state)

                rollout_frames.append(visible_frame.copy())
                insertion_obstacle_rollout_frames.append(
                    render_insertion_obstacle_frame(
                        visible_frame,
                        goal_pose,
                        buffer=float(args.insertion_visual_buffer),
                        thickness=float(args.insertion_visual_thickness),
                    )
                )
                latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
                step_success, axis_distance = diffusion_insertion_success(
                    current_block,
                    goal_block,
                    insertion_axis,
                    threshold=float(args.insertion_init_axis_threshold),
                )
                position_goal_distance, yaw_goal_distance, block_goal_distance = hard_pusht_goal_distances(
                    current_block,
                    goal_block,
                )
                current_pusher_local_y = pusher_local_y(current_block, current_agent_pos(plan_env))
                pusher_top_edge_violated = (
                    current_pusher_local_y
                    > PUSHT_TEE_CAP_TOP_Y + float(args.insertion_pusher_top_edge_margin)
                )
                if pusher_top_edge_violated:
                    pusher_top_edge_violation_steps += 1
                else:
                    pusher_top_edge_violation_steps = 0
                pusher_crossed_top_edge = (
                    pusher_top_edge_violation_steps >= int(args.insertion_pusher_top_edge_fail_steps)
                )
                latent_track_error = float(torch.linalg.vector_norm(current_state - torch.from_numpy(x_ref_np[1]).to(current_state)).item())
                latent_goal_distances.append(latent_goal_distance)
                block_goal_distances.append(block_goal_distance)
                position_goal_distances.append(position_goal_distance)
                yaw_goal_distances.append(yaw_goal_distance)
                insertion_axis_distances.append(float(axis_distance))
                pusher_local_ys.append(float(current_pusher_local_y))
                success_log.append(bool(step_success))
                latent_track_errors.append(latent_track_error)
                success = bool(step_success)

                pbar.set_postfix(
                    solve_ms=f"{solve_times_ms[-1]:.1f}",
                    mppi=f"{mppi_ms:.1f}",
                    ilqr=f"{ilqr_ms:.1f}",
                    iters=f"{ilqr_iterations[-1]}",
                    reward=f"{mppi_plan_rewards[-1]:.3f}",
                    track=f"{latent_track_error:.3f}",
                    latent_goal=f"{latent_goal_distance:.3f}",
                    axis_goal=f"{axis_distance:.2f}",
                    pos_goal=f"{position_goal_distance:.2f}",
                    yaw_goal=f"{yaw_goal_distance:.3f}",
                    success=int(success),
                )

                if success:
                    stop_reason = "success"
                    break
                if pusher_crossed_top_edge:
                    stop_reason = "pusher_crossed_top_edge"
                    break
                if truncated:
                    stop_reason = "truncated"
                    break
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        except Exception as exc:
            if not is_wrapped_keyboard_interrupt(exc):
                raise
            stop_reason = "keyboard_interrupt"
        finally:
            pbar.close()

        final_block = current_block_pose(plan_env)
        final_agent = current_agent_pos(plan_env)
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        plan_env.close()
        viz_env.close()

    metrics = save_outputs()

    print(
        json.dumps(
            {
                "success": metrics["success"],
                "init_seed": metrics["init_seed"],
                "env_seed": metrics["env_seed"],
                "mppi_seed": metrics["mppi_seed"],
                "insertion_axis_distance_min": metrics["insertion_axis_distance_min"],
                "insertion_axis_distance_final": metrics["insertion_axis_distance_final"],
                "position_goal_distance_min": metrics["position_goal_distance_min"],
                "latent_goal_distance_final": metrics["latent_goal_distance_final"],
                "block_goal_distance_final": metrics["block_goal_distance_final"],
                "position_goal_distance_final": metrics["position_goal_distance_final"],
                "yaw_goal_distance_final": metrics["yaw_goal_distance_final"],
                "stop_reason": stop_reason,
                "metrics_path": str(metrics_path),
                "video_path": video_path,
                "insertion_obstacle_video_path": insertion_obstacle_video_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
