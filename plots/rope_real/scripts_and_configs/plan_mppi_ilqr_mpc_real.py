#!/usr/bin/env python3
"""Run real-rope latent MPPI+iLQR MPC on real hardware with safety checks."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GPU_SLS_SRC = ROOT / "third_party" / "gpu_sls" / "src"
if GPU_SLS_SRC.exists() and str(GPU_SLS_SRC) not in sys.path:
    sys.path.insert(0, str(GPU_SLS_SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_default_matmul_precision", "highest")
jax_config.update("jax_enable_x64", True)

from rope.data_real.collision_guard import MujocoArmCollisionGuard  # noqa: E402
from rope.data_real.rope_real_data_gen import (  # noqa: E402
    HARDWARE_HOME_Q0_DEG,
    HARDWARE_HOME_Q1_DEG,
    TASK_HEIGHT_BOUNDS,
    TASK_REACH_BOUNDS,
    TASK_WIDTH_BOUNDS,
    HomeAnchoredLinePlanner,
    action_durations,
    capture_fresh_frame,
    make_camera,
    make_robot,
    move_robot_to_planner_home,
    wait_until_settled,
)
from rope.plan.plan_ilqr_mpc_sim import (  # noqa: E402
    ILQRMPCSolver,
    encode_single_frame,
    latest_object_checkpoint,
    load_config,
    load_model,
    make_markov_state,
    maybe_cuda_synchronize,
    require_device,
    required_markov_history,
    save_rgb_image,
)
from rope.shared.lab_env import LabEnv  # noqa: E402
from gpu_sls.mppi_planner import MPPIPlanner  # noqa: E402


DEFAULT_CONFIG = ROOT / "rope" / "plan" / "plan_mppi_ilqr_mpc_real.yaml"
DEFAULT_MODEL_DIR = ROOT / "rope" / "models" / "mlpdyn_ft_real"
DEFAULT_OUT_DIR = ROOT / "rope" / "plan" / "mppi_ilqr_mpc_real"
DEFAULT_STATS_GLOB = str(ROOT / "rope" / "data_real" / "real_data" / "rope_real_shard*.h5")
DEFAULT_GOAL_DATASET_GLOB = DEFAULT_STATS_GLOB
DEFAULT_SHARD = None
DEFAULT_WAYPOINT_IDX = None
WRITE_HDF5_TRAJECTORY = True
OPTIMIZE_3D_TASK_ACTIONS = True
DEFAULT_TASK_ACTION_MIN_M = np.array([-0.01010218, -0.00391459, -0.01059198], dtype=np.float64)
DEFAULT_TASK_ACTION_MAX_M = np.array([0.00973974, 0.00397003, 0.01060879], dtype=np.float64)
SHARD_RE = re.compile(r"_shard(\d+)\.h5$")


def planner_defaults() -> dict[str, object]:
    return {
        "model_dir": DEFAULT_MODEL_DIR,
        "checkpoint": None,
        "stats_dataset_glob": DEFAULT_STATS_GLOB,
        "goal_dataset_glob": DEFAULT_GOAL_DATASET_GLOB,
        "action_key": None,
        "shard": DEFAULT_SHARD,
        "waypoint_idx": DEFAULT_WAYPOINT_IDX,
        "out_dir": DEFAULT_OUT_DIR,
        "device": "auto",
        "horizon": 12,
        "max_mpc_steps": 40,
        "execute_steps_per_plan": 1,
        "task_action_min_m": DEFAULT_TASK_ACTION_MIN_M.tolist(),
        "task_action_max_m": DEFAULT_TASK_ACTION_MAX_M.tolist(),
        "max_normalized_control": 3.0,
        "q_terminal": 5.0,
        "q_stage": 0.005,
        "r_control": 0.05,
        "ilqr_max_iters": 12,
        "ilqr_tol": 1e-4,
        "ilqr_regularization": 1e-3,
        "mppi_horizon": None,
        "mppi_samples": 2048,
        "mppi_update_iter": 6,
        "mppi_reward_weight": 25.0,
        "mppi_noise_level": 0.2,
        "mppi_beta_filter": 0.65,
        "mppi_q_term": 5.0,
        "mppi_q": 0.005,
        "mppi_r": 0.01,
        "mppi_min_task_delta_m": 0.0,
        "mppi_min_task_delta_penalty": 0.0,
        "video_fps": 5,
        "write_hdf5_trajectory": WRITE_HDF5_TRAJECTORY,
        "optimize_3d_task_actions": OPTIMIZE_3D_TASK_ACTIONS,
        "pre_command_sleep_s": 1.0,
        "dry_run": False,
        "compression": "lzf",
    }


def load_yaml_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(config).__name__}.")
    return {str(key).replace("-", "_"): value for key, value in config.items()}


def normalize_mppi_min_task_delta_m(value: object) -> list[float]:
    values = np.asarray(value if isinstance(value, (list, tuple, np.ndarray)) else [value], dtype=np.float64).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 3)
    elif values.size != 3:
        raise ValueError("--mppi-min-task-delta-m must provide either 1 scalar or 3 per-dimension values.")
    if np.any(values < 0.0):
        raise ValueError("--mppi-min-task-delta-m cannot contain negative values.")
    return [float(item) for item in values]


def real_defaults() -> dict[str, object]:
    return {
        "task_reach_bounds": list(TASK_REACH_BOUNDS),
        "task_height_bounds": list(TASK_HEIGHT_BOUNDS),
        "task_width_bounds": list(TASK_WIDTH_BOUNDS),
        "home_q0_deg": HARDWARE_HOME_Q0_DEG.tolist(),
        "home_q1_deg": HARDWARE_HOME_Q1_DEG.tolist(),
        "robot_backend": "drake-lcm",
        "arm_mapping": "robot0-left",
        "drake_publish_period": 0.005,
        "home_move_duration": 12.0,
        "setup_move_duration": 12.0,
        "trajectory_hold_duration": 0.25,
        "trajectory_start_blend_duration": 4.0,
        "trajectory_start_settle_duration": 0.25,
        "trajectory_start_ready_tolerance_deg": 0.15,
        "trajectory_start_ready_velocity_deg_s": 1.0,
        "trajectory_start_ready_timeout": 15.0,
        "status_timeout": 5.0,
        "max_home_x_gap_m": 0.008,
        "max_home_z_gap_m": 0.004,
        "max_control_joint_step_deg": 9.0,
        "max_reset_joint_move_deg": 90.0,
        "max_command_measured_gap_deg": 0.75,
        "max_real_joint_speed_deg_s": 15.0,
        "max_task_speed_m_s": 0.03,
        "min_action_duration": 0.35,
        "settle_tolerance_deg": 0.50,
        "settle_velocity_deg_s": 1.0,
        "settle_timeout": 8.0,
        "settle_poll_period": 0.05,
        "ik_position_tol": 0.005,
        "ik_max_joint_step_deg": 8.0,
        "ik_joint7_min_deg": -140.0,
        "ik_joint7_max_deg": 140.0,
        "disable_collision_guard": False,
        "arm_arm_min_distance": 0.06,
        "collision_control_samples": 5,
        "collision_reset_samples": 25,
        "enable_camera": True,
        "camera_index": 0,
        "camera_device": None,
        "camera_width": 224,
        "camera_height": 224,
        "camera_crop_center_x": 0.525,
        "camera_crop_center_y": 0.5,
        "camera_crop_zoom": 0.9,
        "camera_capture_width": None,
        "camera_capture_height": None,
        "camera_warmup_frames": 10,
        "camera_drop_frames_after_motion": 3,
        "camera_settle_delay": 0.05,
    }


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = planner_defaults()
    defaults.update(real_defaults())
    config = load_yaml_config(pre_args.config)
    defaults.update({key: value for key, value in config.items() if key in defaults})

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--model-dir", type=Path, default=defaults["model_dir"])
    parser.add_argument("--checkpoint", type=Path, default=defaults["checkpoint"])
    parser.add_argument("--stats-dataset-glob", default=defaults["stats_dataset_glob"])
    parser.add_argument("--goal-dataset-glob", default=defaults["goal_dataset_glob"])
    parser.add_argument("--action-key", default=defaults["action_key"])
    parser.add_argument("--shard", type=int, default=defaults["shard"], help="Real-data shard id to draw the start/goal from. Defaults to random.")
    parser.add_argument("--waypoint-idx", type=int, default=defaults["waypoint_idx"], help="Adjacent waypoint-pair index within the shard. Defaults to random.")
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--horizon", type=int, default=defaults["horizon"])
    parser.add_argument("--max-mpc-steps", type=int, default=defaults["max_mpc_steps"])
    parser.add_argument("--execute-steps-per-plan", type=int, default=defaults["execute_steps_per_plan"])
    parser.add_argument("--task-action-min-m", type=float, nargs=3, default=defaults["task_action_min_m"])
    parser.add_argument("--task-action-max-m", type=float, nargs=3, default=defaults["task_action_max_m"])
    parser.add_argument("--max-normalized-control", type=float, default=defaults["max_normalized_control"])
    parser.add_argument("--q-terminal", type=float, default=defaults["q_terminal"])
    parser.add_argument("--q-stage", type=float, default=defaults["q_stage"])
    parser.add_argument("--r-control", type=float, default=defaults["r_control"])
    parser.add_argument("--ilqr-max-iters", type=int, default=defaults["ilqr_max_iters"])
    parser.add_argument("--ilqr-tol", type=float, default=defaults["ilqr_tol"])
    parser.add_argument("--ilqr-regularization", type=float, default=defaults["ilqr_regularization"])
    parser.add_argument("--mppi-horizon", type=int, default=defaults["mppi_horizon"], help="MPPI nominal-planning horizon. Defaults to --horizon.")
    parser.add_argument("--mppi-samples", type=int, default=defaults["mppi_samples"])
    parser.add_argument("--mppi-update-iter", type=int, default=defaults["mppi_update_iter"])
    parser.add_argument("--mppi-reward-weight", type=float, default=defaults["mppi_reward_weight"])
    parser.add_argument("--mppi-noise-level", type=float, default=defaults["mppi_noise_level"])
    parser.add_argument("--mppi-beta-filter", type=float, default=defaults["mppi_beta_filter"])
    parser.add_argument("--mppi-q-term", type=float, default=defaults["mppi_q_term"])
    parser.add_argument("--mppi-q", type=float, default=defaults["mppi_q"])
    parser.add_argument("--mppi-r", type=float, default=defaults["mppi_r"])
    parser.add_argument("--mppi-min-task-delta-m", type=float, nargs="+", default=defaults["mppi_min_task_delta_m"])
    parser.add_argument("--mppi-min-task-delta-penalty", type=float, default=defaults["mppi_min_task_delta_penalty"])
    parser.add_argument("--video-fps", type=int, default=defaults["video_fps"])
    parser.add_argument("--write-hdf5-trajectory", action=argparse.BooleanOptionalAction, default=defaults["write_hdf5_trajectory"])
    parser.add_argument("--pre-command-sleep-s", type=float, default=defaults["pre_command_sleep_s"])
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default=defaults["compression"])
    parser.add_argument(
        "--optimize-3d-task-actions",
        action=argparse.BooleanOptionalAction,
        default=defaults["optimize_3d_task_actions"],
        help="Optimize 3D task deltas and map them to hardware targets with IK. Disable for native 14D control models.",
    )
    parser.add_argument("--dry-run", action="store_true", default=defaults["dry_run"], help="Plan, safety-check, and log without commanding hardware.")
    parser.add_argument("--i-understand-this-moves-real-robots", action="store_true")

    for key, value in real_defaults().items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=defaults[key])
        elif isinstance(value, list):
            parser.add_argument(option, type=float, nargs=len(value), default=defaults[key])
        elif isinstance(value, int):
            parser.add_argument(option, type=int, default=defaults[key])
        elif isinstance(value, float):
            parser.add_argument(option, type=float, default=defaults[key])
        else:
            parser.add_argument(option, default=defaults[key])

    args = parser.parse_args()
    for key in ("model_dir", "checkpoint", "out_dir"):
        value = getattr(args, key)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))
    if args.shard is not None and args.shard < 0:
        raise ValueError("--shard must be non-negative when provided.")
    if args.waypoint_idx is not None and args.waypoint_idx < 0:
        raise ValueError("--waypoint-idx must be non-negative when provided.")
    if len(args.task_action_min_m) != 3 or len(args.task_action_max_m) != 3:
        raise ValueError("--task-action-min-m and --task-action-max-m must each provide 3 values.")
    if np.any(np.asarray(args.task_action_min_m, dtype=np.float64) > np.asarray(args.task_action_max_m, dtype=np.float64)):
        raise ValueError("--task-action-min-m must be <= --task-action-max-m elementwise.")
    if args.pre_command_sleep_s < 0.0:
        raise ValueError("--pre-command-sleep-s cannot be negative.")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive.")
    if args.execute_steps_per_plan < 1:
        raise ValueError("--execute-steps-per-plan must be positive.")
    if args.execute_steps_per_plan > args.horizon:
        raise ValueError(
            f"--execute-steps-per-plan must be <= --horizon. Got execute_steps_per_plan={args.execute_steps_per_plan}, horizon={args.horizon}."
        )
    if args.mppi_horizon is None:
        args.mppi_horizon = int(args.horizon)
    if args.mppi_horizon < args.horizon:
        raise ValueError(f"--mppi-horizon must be >= --horizon. Got mppi_horizon={args.mppi_horizon}, horizon={args.horizon}.")
    if args.mppi_samples < 1:
        raise ValueError("--mppi-samples must be positive.")
    if args.mppi_update_iter < 1:
        raise ValueError("--mppi-update-iter must be positive.")
    if args.mppi_noise_level < 0.0:
        raise ValueError("--mppi-noise-level cannot be negative.")
    args.mppi_min_task_delta_m = normalize_mppi_min_task_delta_m(args.mppi_min_task_delta_m)
    if args.mppi_min_task_delta_penalty < 0.0:
        raise ValueError("--mppi-min-task-delta-penalty cannot be negative.")
    if not 0.0 <= args.mppi_beta_filter <= 1.0:
        raise ValueError("--mppi-beta-filter must be in [0, 1].")
    if not args.dry_run and not args.i_understand_this_moves_real_robots:
        raise RuntimeError("Refusing to move hardware without --i-understand-this-moves-real-robots.")
    return args


def shard_id_from_path(path: Path) -> int | None:
    match = SHARD_RE.search(path.name)
    return int(match.group(1)) if match is not None else None


def discover_goal_shards(pattern: str) -> list[tuple[int, Path]]:
    paths = sorted(Path(p).resolve() for p in glob.glob(str(Path(pattern).expanduser())))
    shards: list[tuple[int, Path]] = []
    for path in paths:
        shard = shard_id_from_path(path)
        if shard is not None:
            shards.append((shard, path))
    if not shards:
        raise FileNotFoundError(f"No goal shards matching '*_shardNNNN.h5' found for: {pattern}")
    return shards


def shard_waypoint_count(h5: object) -> int:
    attrs = h5.attrs  # type: ignore[attr-defined]
    if "samples_per_waypoint" not in attrs:
        raise ValueError("Goal shard is missing samples_per_waypoint attr.")
    samples_per_waypoint = int(attrs["samples_per_waypoint"])
    if samples_per_waypoint < 1:
        raise ValueError(f"Invalid samples_per_waypoint={samples_per_waypoint}.")
    total_transitions = int(h5["task_before"].shape[0])  # type: ignore[index]
    attr_waypoints = int(attrs.get("num_waypoints", total_transitions // samples_per_waypoint))
    return min(attr_waypoints, total_transitions // samples_per_waypoint)


def load_real_waypoint_pair(
    pattern: str,
    *,
    shard: int | None,
    waypoint_idx: int | None,
) -> dict[str, object]:
    import h5py

    shards = discover_goal_shards(pattern)
    if shard is not None:
        matches = [(candidate_shard, path) for candidate_shard, path in shards if candidate_shard == shard]
        if not matches:
            available = ", ".join(str(candidate_shard) for candidate_shard, _ in shards[:20])
            suffix = "..." if len(shards) > 20 else ""
            raise ValueError(f"--shard {shard} not found in {pattern}. Available shard ids: {available}{suffix}")
        shards = matches

    rng = np.random.default_rng()
    if shard is None:
        order = rng.permutation(len(shards)).tolist()
    else:
        order = [0]

    last_error: Exception | None = None
    for index in order:
        selected_shard, path = shards[index]
        try:
            with h5py.File(path, "r") as h5:
                required = ("task_before", "task_after", "q_before", "q_after", "command_duration")
                missing = [name for name in required if name not in h5]
                if missing:
                    raise ValueError(f"Goal shard {path} is missing datasets: {missing}")
                samples_per_waypoint = int(h5.attrs["samples_per_waypoint"])
                num_waypoint_pairs = shard_waypoint_count(h5)
                if num_waypoint_pairs < 1:
                    raise ValueError(f"Goal shard {path} has no complete waypoint pairs.")
                if waypoint_idx is None:
                    selected_waypoint = int(rng.integers(0, num_waypoint_pairs))
                else:
                    selected_waypoint = int(waypoint_idx)
                    if selected_waypoint >= num_waypoint_pairs:
                        raise ValueError(
                            f"--waypoint-idx must be in [0, {num_waypoint_pairs - 1}] for shard {selected_shard}, "
                            f"got {selected_waypoint}."
                        )

                row_start = selected_waypoint * samples_per_waypoint
                row_end = row_start + samples_per_waypoint - 1
                durations = np.asarray(h5["command_duration"][row_start : row_end + 1], dtype=np.float64).reshape(-1)
                return {
                    "shard": int(selected_shard),
                    "shard_path": str(path),
                    "waypoint_idx": int(selected_waypoint),
                    "row_start": int(row_start),
                    "row_end": int(row_end),
                    "samples_per_waypoint": int(samples_per_waypoint),
                    "num_waypoint_pairs": int(num_waypoint_pairs),
                    "task_start": np.asarray(h5["task_before"][row_start], dtype=np.float64).reshape(3),
                    "task_goal": np.asarray(h5["task_after"][row_end], dtype=np.float64).reshape(3),
                    "q_start": np.asarray(h5["q_before"][row_start], dtype=np.float64).reshape(14),
                    "q_goal": np.asarray(h5["q_after"][row_end], dtype=np.float64).reshape(14),
                    "expert_duration_s": float(np.sum(durations)),
                }
        except Exception as error:
            if shard is not None or waypoint_idx is not None:
                raise
            last_error = error
            continue
    raise RuntimeError(f"Could not load a valid waypoint pair from {pattern}.") from last_error


def load_action_stats(pattern: str, *, action_key: str, action_dim: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import h5py

    paths = sorted(Path(p).resolve() for p in glob.glob(str(Path(pattern).expanduser())))
    if not paths:
        raise FileNotFoundError(f"No stats datasets match: {pattern}")
    count = 0
    total = np.zeros((action_dim,), dtype=np.float64)
    total_sq = np.zeros((action_dim,), dtype=np.float64)
    used: list[str] = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            if action_key not in h5:
                continue
            dataset = h5[action_key]
            if int(dataset.shape[-1]) != action_dim:
                continue
            values = np.asarray(dataset[:], dtype=np.float64).reshape(-1, action_dim)
            values = values[np.isfinite(values).all(axis=1)]
            if values.size == 0:
                continue
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += int(values.shape[0])
            used.append(str(path))
    if count == 0:
        raise ValueError(f"No finite {action_dim}D rows found for action key {action_key!r} in {pattern}.")
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 1e-12)
    std = np.maximum(np.sqrt(var), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), used


def save_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> tuple[str | None, str | None]:
    if not frames:
        return None, None
    path = out_dir / "stop_motion.mp4"
    try:
        imageio.mimwrite(str(path), frames, fps=fps, quality=8, macro_block_size=1, format="FFMPEG")
        return str(path), None
    except Exception as mp4_error:
        gif_path = out_dir / "stop_motion.gif"
        try:
            imageio.mimwrite(str(gif_path), frames, duration=1.0 / max(int(fps), 1))
            return str(gif_path), f"mp4 export failed: {type(mp4_error).__name__}: {mp4_error}"
        except Exception as gif_error:
            return (
                None,
                "video export failed: "
                f"mp4 {type(mp4_error).__name__}: {mp4_error}; "
                f"gif {type(gif_error).__name__}: {gif_error}",
            )


def make_info_from_qpos(planner: HomeAnchoredLinePlanner, qpos: np.ndarray, task_target: np.ndarray) -> dict[str, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(14)
    qvel = np.zeros(14, dtype=np.float32)
    control = qpos.astype(np.float32)
    p0 = planner.ik0.fk_position(qpos[:7]).astype(np.float32)
    p1 = planner.ik1.fk_position(qpos[7:]).astype(np.float32)
    rope_length = np.asarray([np.linalg.norm(p0 - p1)], dtype=np.float32)
    target = np.asarray(task_target, dtype=np.float32).reshape(3)
    measured_task_target = planner.measured_task_state_from_attachment_positions(p0, p1).astype(np.float32)
    observation = np.concatenate([target, qpos.astype(np.float32), qvel, control, p0, p1, rope_length], axis=0).astype(np.float32)
    return {
        "observation": observation,
        "task_target": target,
        "measured_task_target": measured_task_target,
        "qpos": qpos.astype(np.float32),
        "qvel": qvel,
        "control": control,
        "left_attachment_pos": p0,
        "right_attachment_pos": p1,
        "rope_length": rope_length,
    }


def measured_task_state(planner: HomeAnchoredLinePlanner, info: dict[str, np.ndarray]) -> np.ndarray:
    if "measured_task_target" in info:
        return np.asarray(info["measured_task_target"], dtype=np.float64).reshape(3)
    return planner.measured_task_state_from_attachment_positions(
        np.asarray(info["left_attachment_pos"]),
        np.asarray(info["right_attachment_pos"]),
    )


class RealControlDynamicsTorch:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        state_dim: int,
        control_dim: int,
        max_normalized_control: float,
        device: torch.device,
    ) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError("Expected one-step MLP dynamics with history_size=1 and action_history_size=1.")
        if int(predictor.embed_dim) != state_dim:
            raise ValueError(f"Predictor state dim mismatch: expected {state_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != control_dim:
            raise ValueError(f"Predictor action dim mismatch: expected {control_dim}, got {predictor.action_dim}.")
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("This planner assumes an identity action encoder.")
        self.net = predictor.net.to(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(control_dim)
        self.control_dim = int(control_dim)
        self.device = device
        self.action_mean = torch.tensor(action_mean.reshape(-1), dtype=torch.float32, device=device)
        self.action_std = torch.tensor(action_std.reshape(-1), dtype=torch.float32, device=device)
        self.max_normalized_control = float(max_normalized_control)

    def step(self, x: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u_norm, -self.max_normalized_control, self.max_normalized_control)
        return self.net(torch.cat((x, u), dim=-1))

    def normalized_to_raw(self, u_norm: np.ndarray) -> np.ndarray:
        u = np.clip(
            np.asarray(u_norm, dtype=np.float64).reshape(self.control_dim),
            -self.max_normalized_control,
            self.max_normalized_control,
        )
        return (u * self.action_std.detach().cpu().numpy() + self.action_mean.detach().cpu().numpy()).astype(np.float64)


class RealTaskDynamicsTorch:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        state_dim: int,
        action_dim: int,
        max_normalized_action: float,
        device: torch.device,
    ) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError("Expected one-step MLP dynamics with history_size=1 and action_history_size=1.")
        if int(predictor.embed_dim) != state_dim:
            raise ValueError(f"Predictor state dim mismatch: expected {state_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != action_dim:
            raise ValueError(f"Predictor action dim mismatch: expected {action_dim}, got {predictor.action_dim}.")
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("This planner assumes an identity action encoder.")
        self.net = predictor.net.to(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.action_mean = torch.tensor(action_mean.reshape(-1), dtype=torch.float32, device=device)
        self.action_std = torch.tensor(action_std.reshape(-1), dtype=torch.float32, device=device)
        self.max_normalized_action = float(max_normalized_action)

    def step(self, x: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u_norm, -self.max_normalized_action, self.max_normalized_action)
        return self.net(torch.cat((x, u), dim=-1))

    def normalized_to_raw(self, u_norm: np.ndarray) -> np.ndarray:
        u = np.clip(
            np.asarray(u_norm, dtype=np.float64).reshape(self.action_dim),
            -self.max_normalized_action,
            self.max_normalized_action,
        )
        return (u * self.action_std.detach().cpu().numpy() + self.action_mean.detach().cpu().numpy()).astype(np.float64)


def make_mppi_rollout_and_eval(
    dynamics: RealControlDynamicsTorch | RealTaskDynamicsTorch,
    *,
    q_stage: float,
    q_terminal: float,
    r_control: float,
    min_task_delta_m: object = 0.0,
    min_task_delta_penalty: float = 0.0,
    goal_state: np.ndarray,
) -> tuple[object, object]:
    state_dim = int(dynamics.state_dim)
    goal = jnp.asarray(goal_state, dtype=jnp.float64).reshape((state_dim,))
    q_stage = float(q_stage)
    q_terminal = float(q_terminal)
    r_control = float(r_control)
    min_task_delta_values = normalize_mppi_min_task_delta_m(min_task_delta_m)
    min_task_delta_m = jnp.asarray(min_task_delta_values, dtype=jnp.float64)
    min_task_delta_penalty = float(min_task_delta_penalty)
    use_min_task_delta_penalty = (
        isinstance(dynamics, RealTaskDynamicsTorch)
        and any(value > 0.0 for value in min_task_delta_values)
        and min_task_delta_penalty > 0.0
    )
    action_mean = jnp.asarray(dynamics.action_mean.detach().cpu().numpy(), dtype=jnp.float64)
    action_std = jnp.asarray(dynamics.action_std.detach().cpu().numpy(), dtype=jnp.float64)

    def _torch_rollout_fn(state_cur_np: np.ndarray, act_seqs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_cur_t = torch.from_numpy(np.asarray(state_cur_np)).float().to(dynamics.device)
            act_seqs_t = torch.from_numpy(np.asarray(act_seqs_np)).float().to(dynamics.device)
            n_sample, horizon, _ = act_seqs_t.shape
            states = state_cur_t.unsqueeze(0).expand(n_sample, -1)
            rolled_states = []
            for step in range(horizon):
                states = dynamics.step(states, act_seqs_t[:, step, :])
                rolled_states.append(states)
            return np.asarray(torch.stack(rolled_states, dim=1).cpu().numpy(), dtype=np.float64)

    def rollout_fn(state_cur: jnp.ndarray, act_seqs: jnp.ndarray, reach_config: object = None) -> tuple[jnp.ndarray, dict[str, object]]:
        result_shape = jax.ShapeDtypeStruct((act_seqs.shape[0], act_seqs.shape[1], state_dim), jnp.float64)
        states = jax.pure_callback(
            _torch_rollout_fn,
            result_shape,
            state_cur,
            act_seqs,
            vmap_method="sequential",
        )
        return states, {}

    def eval_fn(
        state_seqs: jnp.ndarray,
        act_seqs: jnp.ndarray,
        reach_config: object = None,
        aux: object = None,
        *args: object,
        **kwargs: object,
    ) -> dict[str, jnp.ndarray]:
        delta = state_seqs - goal[None, None, :]
        stage_costs = q_stage * jnp.sum(delta**2, axis=-1)
        terminal_costs = q_terminal * jnp.sum(delta[:, -1, :] ** 2, axis=-1)
        action_costs = r_control * jnp.sum(act_seqs**2, axis=-1)
        if use_min_task_delta_penalty:
            raw_task_actions = act_seqs * action_std[None, None, :] + action_mean[None, None, :]
            safe_min_task_delta_m = jnp.maximum(min_task_delta_m, 1e-12)
            normalized_task_delta = jnp.where(
                min_task_delta_m[None, None, :] > 0.0,
                jnp.abs(raw_task_actions) / safe_min_task_delta_m[None, None, :],
                0.0,
            )
            max_normalized_task_delta = jnp.max(normalized_task_delta, axis=-1)
            normalized_hinge = jnp.maximum(0.0, 1.0 - max_normalized_task_delta)
            action_costs = action_costs + min_task_delta_penalty * normalized_hinge**2
        total_cost = jnp.sum(stage_costs + action_costs, axis=-1) + terminal_costs
        return {"rewards": -total_cost}

    return rollout_fn, eval_fn


class TrackingILQRMPCSolver(ILQRMPCSolver):
    def _trajectory_tracking_cost(
        self,
        x_traj: torch.Tensor,
        u_seq: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_ref[step]
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_ref[self.horizon]
        cost = cost + self.q_terminal * torch.dot(terminal_err, terminal_err)
        return cost

    def solve_tracking(
        self,
        x0_np: np.ndarray,
        x_ref_np: np.ndarray,
        u_init_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_ref = torch.tensor(x_ref_np, dtype=torch.float32, device=self.device)
        u_init = torch.tensor(u_init_np, dtype=torch.float32, device=self.device)
        if tuple(x_ref.shape) != (self.horizon + 1, self.state_dim):
            raise ValueError(f"Expected x_ref shape {(self.horizon + 1, self.state_dim)}, got {tuple(x_ref.shape)}.")
        if tuple(u_init.shape) != (self.horizon, self.action_dim):
            raise ValueError(f"Expected u_init shape {(self.horizon, self.action_dim)}, got {tuple(u_init.shape)}.")

        u_seq = u_init.detach().clone()
        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_tracking_cost(x_traj, u_seq, x_ref).item())
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
                new_cost = float(self._trajectory_tracking_cost(x_new, u_new, x_ref).item())
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


def measured_ee_goal_distance(info: dict[str, np.ndarray], goal_info: dict[str, np.ndarray]) -> float:
    left = float(np.linalg.norm(np.asarray(info["left_attachment_pos"]) - np.asarray(goal_info["left_attachment_pos"])))
    right = float(np.linalg.norm(np.asarray(info["right_attachment_pos"]) - np.asarray(goal_info["right_attachment_pos"])))
    return max(left, right)


def validate_collision_step(start_qpos: np.ndarray, target_qpos: np.ndarray, duration: float, args: argparse.Namespace, label: str) -> None:
    if args.disable_collision_guard:
        return
    guard = MujocoArmCollisionGuard(
        LabEnv(),
        min_arm_arm_distance=args.arm_arm_min_distance,
        control_path_samples=args.collision_control_samples,
        reset_path_samples=args.collision_reset_samples,
    )
    guard.validate_path(start_qpos, target_qpos, duration=duration, label=label)


def control_step_duration(current_qpos: np.ndarray, target_qpos: np.ndarray, args: argparse.Namespace) -> float:
    current = np.asarray(current_qpos, dtype=np.float64).reshape(14)
    target = np.asarray(target_qpos, dtype=np.float64).reshape(14)
    max_joint_speed = np.deg2rad(args.max_real_joint_speed_deg_s)
    duration = float(np.max(np.abs(target - current)) / max(max_joint_speed, 1e-9))
    return max(duration, float(args.min_action_duration), float(args.drake_publish_period))


def plan_control_step(
    dynamics: RealControlDynamicsTorch,
    current_qpos: np.ndarray,
    action_norm: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    current = np.asarray(current_qpos, dtype=np.float64).reshape(14)
    raw_target = dynamics.normalized_to_raw(action_norm)
    max_step = np.deg2rad(args.max_control_joint_step_deg)
    q_target = np.clip(raw_target, current - max_step, current + max_step)
    duration = control_step_duration(current, q_target, args)
    validate_collision_step(current, q_target, duration, args, label="mpc 14d control step")
    return raw_target, q_target, duration


def plan_task_step(
    planner: HomeAnchoredLinePlanner,
    current_task: np.ndarray,
    current_qpos: np.ndarray,
    action_task: np.ndarray,
    args: argparse.Namespace,
    *,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if action_min is None:
        action_min = np.asarray(args.task_action_min_m, dtype=np.float64).reshape(3)
    else:
        action_min = np.asarray(action_min, dtype=np.float64).reshape(3)
    if action_max is None:
        action_max = np.asarray(args.task_action_max_m, dtype=np.float64).reshape(3)
    else:
        action_max = np.asarray(action_max, dtype=np.float64).reshape(3)
    clipped_action = np.clip(np.asarray(action_task, dtype=np.float64).reshape(3), action_min, action_max)
    target_task = planner.task_bounds.clip(np.asarray(current_task, dtype=np.float64).reshape(3) + clipped_action)
    effective_action = target_task - np.asarray(current_task, dtype=np.float64).reshape(3)
    task_path = np.stack([current_task, target_task], axis=0)
    q_target = planner.plan_joint_step_from_current(current_qpos, target_task)
    duration = float(action_durations(np.stack([current_qpos, q_target], axis=0), task_path, args)[0])
    validate_collision_step(current_qpos, q_target, duration, args, label="mpc task step")
    return clipped_action, effective_action, q_target, duration


def append_camera_packet(
    packets: dict[str, list[np.ndarray]],
    packet: dict[str, np.ndarray | float | int],
    *,
    capture_time: float,
) -> None:
    packets["camera_capture_monotonic_time"].append(np.asarray([capture_time], dtype=np.float64))
    packets["camera_frame_time"].append(np.asarray([float(packet["frame_time"])], dtype=np.float64))
    packets["camera_receive_time"].append(np.asarray([float(packet["receive_time"])], dtype=np.float64))
    packets["camera_header_time"].append(np.asarray([float(packet["header_time"])], dtype=np.float64))
    packets["camera_frame_index"].append(np.asarray([int(packet["frame_index"])], dtype=np.int64))


def stack_rows(rows: list[dict[str, np.ndarray]], key: str, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=dtype)


def save_hdf5_rollout(
    path: Path,
    *,
    rows: list[dict[str, np.ndarray]],
    frames: list[np.ndarray],
    camera_packets: dict[str, list[np.ndarray]],
    args: argparse.Namespace,
    planner: HomeAnchoredLinePlanner,
    metrics: dict[str, object],
) -> None:
    import h5py

    compression = None if args.compression == "none" else args.compression
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "rope/plan/plan_mppi_ilqr_mpc_real.py"
        h5.attrs["hardware"] = not bool(args.dry_run)
        h5.attrs["trajectory_type"] = "latent_mppi_ilqr_mpc_stepwise"
        h5.attrs["row_semantics"] = "actions are transitions; pixels are state frames with one initial frame plus one post-settle frame per action"
        h5.attrs["action_semantics"] = "clipped_planned_task_delta_from_mppi_warmstarted_ilqr_tracking_mpc"
        h5.attrs["measured_action_semantics"] = "measured_task_after_minus_measured_task_before"
        h5.attrs["action_dim"] = 3
        h5.attrs["qpos_dim"] = 14
        h5.attrs["qvel_dim"] = 14
        h5.attrs["control_dim"] = 14
        h5.attrs["observation_dim"] = 52
        h5.attrs["num_episodes"] = 1
        h5.attrs["total_frames"] = len(frames)
        h5.attrs["total_transitions"] = len(rows)
        h5.attrs["compression"] = args.compression
        h5.attrs["robot_backend"] = args.robot_backend
        h5.attrs["arm_mapping"] = "kuka-arm0-arm1"
        h5.attrs["drake_publish_period"] = args.drake_publish_period
        h5.attrs["max_real_joint_speed_deg_s"] = args.max_real_joint_speed_deg_s
        h5.attrs["max_task_speed_m_s"] = args.max_task_speed_m_s
        h5.attrs["min_action_duration"] = args.min_action_duration
        h5.attrs["settle_tolerance_deg"] = args.settle_tolerance_deg
        h5.attrs["settle_velocity_deg_s"] = args.settle_velocity_deg_s
        h5.attrs["settle_timeout"] = args.settle_timeout
        h5.attrs["collision_guard_enabled"] = not args.disable_collision_guard
        h5.attrs["arm_arm_min_distance"] = args.arm_arm_min_distance
        h5.attrs["camera_enabled"] = args.enable_camera
        h5.attrs["camera_backend"] = "opencv"
        h5.attrs["camera_index"] = args.camera_index
        h5.attrs["camera_device"] = "" if args.camera_device is None else args.camera_device
        h5.attrs["camera_crop_center_x"] = args.camera_crop_center_x
        h5.attrs["camera_crop_center_y"] = args.camera_crop_center_y
        h5.attrs["camera_crop_zoom"] = args.camera_crop_zoom
        h5.attrs["camera_drop_frames_after_motion"] = args.camera_drop_frames_after_motion
        h5.attrs["camera_settle_delay"] = args.camera_settle_delay
        h5.attrs["home_q0_deg"] = json.dumps([float(value) for value in args.home_q0_deg])
        h5.attrs["home_q1_deg"] = json.dumps([float(value) for value in args.home_q1_deg])
        h5.attrs["home_task_state"] = json.dumps(planner.home_task.tolist())
        h5.attrs["hardware_home_qpos"] = json.dumps(planner.hardware_home_qpos.tolist())
        h5.attrs["initial_dataset_home_qpos"] = json.dumps(planner.dataset_home_qpos.tolist())
        h5.attrs["task_bounds"] = json.dumps(
            {
                "reach": list(planner.task_bounds.reach),
                "height": list(planner.task_bounds.height),
                "width": list(planner.task_bounds.width),
            }
        )
        h5.attrs["mpc_metrics_json"] = "metrics.json"
        h5.attrs["mpc_stop_reason"] = str(metrics.get("stop_reason", ""))
        h5.attrs["mpc_goal_shard"] = int(metrics.get("goal_shard", -1))
        h5.attrs["mpc_goal_waypoint_idx"] = int(metrics.get("goal_waypoint_idx", -1))
        h5.attrs["mpc_goal_row_start"] = int(metrics.get("goal_row_start", -1))
        h5.attrs["mpc_goal_row_end"] = int(metrics.get("goal_row_end", -1))

        h5.create_dataset("ep_len", data=np.asarray([len(frames)], dtype=np.int64))
        h5.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("reward", data=np.asarray([0.0], dtype=np.float32))
        h5.create_dataset("episode_seed", data=np.asarray([int(metrics.get("goal_shard", 0))], dtype=np.int64))
        h5.create_dataset("selected_episode_seed", data=np.asarray([int(metrics.get("goal_waypoint_idx", 0))], dtype=np.int64))
        h5.create_dataset("terminated", data=np.asarray([False], dtype=np.bool_))
        h5.create_dataset("truncated", data=np.asarray([metrics.get("stop_reason") != "max_mpc_steps"], dtype=np.bool_))
        h5.create_dataset("pixels", data=np.stack(frames, axis=0).astype(np.uint8), compression=compression)

        if not rows:
            h5.create_dataset("episode_idx", data=np.zeros((0,), dtype=np.int64))
            h5.create_dataset("step_idx", data=np.zeros((0,), dtype=np.int64))
            for name, values in camera_packets.items():
                if values:
                    h5.create_dataset(name, data=np.stack(values, axis=0))
            return

        h5.create_dataset("episode_idx", data=np.zeros((len(rows),), dtype=np.int64))
        h5.create_dataset("step_idx", data=np.arange(len(rows), dtype=np.int64))
        for name in (
            "action",
            "planned_action",
            "measured_action",
            "observation",
            "task_before",
            "task_after",
            "task_target",
            "measured_task_before",
            "measured_task_after",
            "q_before",
            "q_after",
            "q_cmd",
            "qpos",
            "qvel",
            "control",
            "left_attachment_pos",
            "right_attachment_pos",
            "rope_length",
            "command_duration",
            "settle_duration",
            "settle_position_error_deg",
            "settle_velocity_deg_s",
            "success",
            "state_monotonic_time",
            "action_start_monotonic_time",
            "action_end_monotonic_time",
            "u_plan",
            "raw_task_action",
            "clipped_task_action",
            "latent_goal_distance",
            "ee_goal_distance",
            "task_goal_distance",
            "solve_time_ms",
            "mppi_solve_time_ms",
            "mppi_reward",
            "ilqr_iterations",
            "ilqr_cost",
        ):
            dtype = np.bool_ if name == "success" else np.float32
            if name in {"action_start_monotonic_time", "action_end_monotonic_time", "state_monotonic_time"}:
                dtype = np.float64
            if name == "ilqr_iterations":
                dtype = np.int64
            h5.create_dataset(name, data=stack_rows(rows, name, dtype), compression=None)

        for name, values in camera_packets.items():
            if values:
                h5.create_dataset(name, data=np.stack(values, axis=0))


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    config = load_config(model_dir)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else latest_object_checkpoint(model_dir)
    model = load_model(checkpoint, device)
    img_size = int(config.get("img_size", 224))
    embed_dim = int(config.get("embed_dim", 32))
    markov_deriv = int(config.get("markov_deriv", 1))
    markov_state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * embed_dim))
    model_action_dim = int(config.get("action_dim", 3))
    action_key = str(args.action_key or config.get("action_key", "control"))
    action_mean, action_std, stats_paths = load_action_stats(args.stats_dataset_glob, action_key=action_key, action_dim=model_action_dim)

    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    history_len = required_markov_history(markov_deriv)

    pair = load_real_waypoint_pair(args.goal_dataset_glob, shard=args.shard, waypoint_idx=args.waypoint_idx)
    goal_shard = int(pair["shard"])
    goal_waypoint_idx = int(pair["waypoint_idx"])
    out_root = args.out_dir.expanduser().resolve()
    out_dir = out_root / f"{int(time.time())}_shard_{goal_shard:04d}_waypoint_{goal_waypoint_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    start_task = np.asarray(pair["task_start"], dtype=np.float64)
    goal_task = np.asarray(pair["task_goal"], dtype=np.float64)
    q_start = np.asarray(pair["q_start"], dtype=np.float64)
    q_goal = np.asarray(pair["q_goal"], dtype=np.float64)
    expert_duration = float(pair["expert_duration_s"])

    planner = HomeAnchoredLinePlanner(args)
    setup_move_duration = float(args.setup_move_duration)
    goal_snapshot_duration = max(control_step_duration(planner.hardware_home_qpos, q_goal, args), setup_move_duration)
    start_return_duration = max(control_step_duration(q_goal, q_start, args), setup_move_duration)
    print(
        "Using real-data waypoint pair: "
        f"shard={goal_shard:04d}, "
        f"waypoint_idx={goal_waypoint_idx}, "
        f"rows={int(pair['row_start'])}-{int(pair['row_end'])}, "
        f"segment_duration={expert_duration:.2f}s, "
        f"home_to_goal_duration={goal_snapshot_duration:.2f}s, "
        f"goal_to_start_duration={start_return_duration:.2f}s, "
        f"path={pair['shard_path']}"
    )
    robot = make_robot(args)
    camera = make_camera(args)
    if camera is None:
        raise RuntimeError("Real MPC requires a camera; pass --enable-camera.")

    rollout_frames: list[np.ndarray] = []
    camera_packets: dict[str, list[np.ndarray]] = {
        "camera_capture_monotonic_time": [],
        "camera_frame_time": [],
        "camera_receive_time": [],
        "camera_header_time": [],
        "camera_frame_index": [],
    }
    rows: list[dict[str, np.ndarray]] = []
    metrics: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "model_dir": str(model_dir),
        "goal_dataset_glob": str(args.goal_dataset_glob),
        "goal_shard": goal_shard,
        "goal_shard_path": str(pair["shard_path"]),
        "goal_waypoint_idx": goal_waypoint_idx,
        "goal_row_start": int(pair["row_start"]),
        "goal_row_end": int(pair["row_end"]),
        "goal_samples_per_waypoint": int(pair["samples_per_waypoint"]),
        "goal_num_waypoint_pairs": int(pair["num_waypoint_pairs"]),
        "goal_expert_duration_s": float(expert_duration),
        "goal_snapshot_duration_s": float(goal_snapshot_duration),
        "start_return_duration_s": float(start_return_duration),
        "action_key": action_key,
        "model_action_dim": int(model_action_dim),
        "stats_dataset_count": len(stats_paths),
        "task_start": start_task.tolist(),
        "task_goal": goal_task.tolist(),
        "task_action_min_m": [float(value) for value in args.task_action_min_m],
        "task_action_max_m": [float(value) for value in args.task_action_max_m],
        "max_normalized_control": float(args.max_normalized_control),
        "execute_steps_per_plan": int(args.execute_steps_per_plan),
        "action_mode": "task_3d_mapped" if args.optimize_3d_task_actions else "control_14d",
        "mppi_horizon": int(args.mppi_horizon),
        "mppi_samples": int(args.mppi_samples),
        "mppi_update_iter": int(args.mppi_update_iter),
        "mppi_reward_weight": float(args.mppi_reward_weight),
        "mppi_noise_level": float(args.mppi_noise_level),
        "mppi_beta_filter": float(args.mppi_beta_filter),
        "mppi_q_term": float(args.mppi_q_term),
        "mppi_q": float(args.mppi_q),
        "mppi_r": float(args.mppi_r),
        "mppi_min_task_delta_m": [float(value) for value in args.mppi_min_task_delta_m],
        "mppi_min_task_delta_penalty": float(args.mppi_min_task_delta_penalty),
        "pre_command_sleep_s": float(args.pre_command_sleep_s),
        "dry_run": bool(args.dry_run),
    }

    interrupted = False
    runtime_error: str | None = None
    try:
        if not args.dry_run:
            robot.connect()
            camera.connect()
            move_robot_to_planner_home(robot, planner, args)
            print(f"Moving to shard goal for goal snapshot: duration={goal_snapshot_duration:.2f}s")
            robot.command_joint_positions(q_goal, duration=goal_snapshot_duration, blocking=True)
            wait_until_settled(robot, q_goal, args)
        else:
            camera.connect()

        goal_info = planner.step_info(robot, goal_task) if not args.dry_run else make_info_from_qpos(planner, q_goal, goal_task)
        goal_frame_packet = capture_fresh_frame(camera, args)
        goal_frame = np.asarray(goal_frame_packet["pixels"], dtype=np.uint8)
        save_rgb_image(out_dir / "goal_image.png", goal_frame)
        goal_emb = encode_single_frame(model, goal_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        goal_state = make_markov_state([goal_emb] * history_len, markov_deriv)

        if not args.dry_run:
            print(f"Returning to shard start: duration={start_return_duration:.2f}s")
            robot.command_joint_positions(q_start, duration=start_return_duration, blocking=True)
            wait_until_settled(robot, q_start, args)

        start_info = planner.step_info(robot, start_task) if not args.dry_run else make_info_from_qpos(planner, q_start, start_task)
        current_info = start_info
        start_frame_packet = capture_fresh_frame(camera, args)
        start_camera_time = time.monotonic()
        current_frame = np.asarray(start_frame_packet["pixels"], dtype=np.uint8)
        save_rgb_image(out_dir / "start_image.png", current_frame)
        rollout_frames.append(current_frame.copy())
        append_camera_packet(camera_packets, start_frame_packet, capture_time=start_camera_time)
        current_emb = encode_single_frame(model, current_frame, device=device, img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        current_history = [current_emb] * history_len
        current_state = make_markov_state(current_history, markov_deriv)
        if int(current_state.numel()) != markov_state_dim:
            raise ValueError(f"State dimension mismatch: expected {markov_state_dim}, got {current_state.numel()}.")

        current_task = measured_task_state(planner, current_info)
        if args.optimize_3d_task_actions:
            if model_action_dim != 3:
                raise ValueError(f"3D task-action mode requires a 3D model, got action_dim={model_action_dim}.")
            dynamics = RealTaskDynamicsTorch(
                model,
                action_mean=action_mean,
                action_std=action_std,
                state_dim=markov_state_dim,
                action_dim=model_action_dim,
                max_normalized_action=args.max_normalized_control,
                device=device,
            )
        else:
            if model_action_dim != 14:
                raise ValueError(f"14D control mode requires a 14D model, got action_dim={model_action_dim}.")
            dynamics = RealControlDynamicsTorch(
                model,
                action_mean=action_mean,
                action_std=action_std,
                state_dim=markov_state_dim,
                control_dim=model_action_dim,
                max_normalized_control=args.max_normalized_control,
                device=device,
            )
        mppi_rollout, mppi_eval = make_mppi_rollout_and_eval(
            dynamics,
            q_stage=args.mppi_q,
            q_terminal=args.mppi_q_term,
            r_control=args.mppi_r,
            min_task_delta_m=args.mppi_min_task_delta_m,
            min_task_delta_penalty=args.mppi_min_task_delta_penalty,
            goal_state=goal_state.detach().cpu().numpy().astype(np.float64),
        )
        u_min = -float(args.max_normalized_control) * jnp.ones(model_action_dim, dtype=jnp.float64)
        u_max = float(args.max_normalized_control) * jnp.ones(model_action_dim, dtype=jnp.float64)
        mppi_planner = MPPIPlanner(
            config={
                "planning": {
                    "action_dim": model_action_dim,
                    "n_sample": args.mppi_samples,
                    "horizon": args.mppi_horizon,
                    "n_update_iter": args.mppi_update_iter,
                    "use_last": True,
                    "reject_bad": False,
                    "mppi": {
                        "reward_weight": args.mppi_reward_weight,
                        "noise_level": args.mppi_noise_level,
                        "noise_decay": 1.0,
                        "beta_filter": args.mppi_beta_filter,
                    },
                }
            },
            model_rollout_fn=mppi_rollout,
            evaluate_traj_fn=mppi_eval,
            action_lower_lim=u_min,
            action_upper_lim=u_max,
        )
        jit_mppi_trajopt = jax.jit(
            lambda key, state, act_seq: mppi_planner.trajectory_optimization(key, state, act_seq, skip=False)
        )
        print(f"Using MPPI horizon {args.mppi_horizon}; iLQR tracking horizon {args.horizon}.")

        solver = TrackingILQRMPCSolver(
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
        prev_mppi_u = jnp.zeros((args.mppi_horizon, model_action_dim), dtype=jnp.float64)
        jax_seed_key = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)

        stop_reason = "max_mpc_steps"
        success = False
        goal_distances = [measured_ee_goal_distance(current_info, goal_info)]
        task_goal_distances = [float(np.linalg.norm(measured_task_state(planner, current_info) - goal_task))]
        latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
        attempted_mpc_step: dict[str, object] | None = None

        try:
            with tqdm(total=args.max_mpc_steps, desc="Real MPC steps", unit="step") as progress:
                step_idx = 0
                while step_idx < args.max_mpc_steps:
                    qpos = np.asarray(current_info["qpos"], dtype=np.float64)
                    info_before = current_info
                    q_before = qpos.astype(np.float32)
                    measured_before = measured_task_state(planner, info_before).astype(np.float32)
                    current_task = measured_before.astype(np.float64)
                    if args.optimize_3d_task_actions:
                        assert isinstance(dynamics, RealTaskDynamicsTorch)
                    current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
                    jax_seed_key, subkey = jax.random.split(jax_seed_key)
                    init_mppi_u = jnp.concatenate([prev_mppi_u[1:], prev_mppi_u[-1:]], axis=0)
                    mppi_start = time.perf_counter()
                    mppi_res = jit_mppi_trajopt(subkey, jnp.asarray(current_state_np), init_mppi_u)
                    mppi_solve_time = time.perf_counter() - mppi_start
                    mppi_u_full = np.asarray(mppi_res["act_seq"], dtype=np.float64)
                    mppi_x_rollout = np.asarray(mppi_res["state_seq"], dtype=np.float64)
                    mppi_reward = float(np.asarray(mppi_res["reward"], dtype=np.float64))
                    if mppi_u_full.shape != (args.mppi_horizon, model_action_dim):
                        raise RuntimeError(f"MPPI returned act_seq shape {mppi_u_full.shape}, expected {(args.mppi_horizon, model_action_dim)}.")
                    if mppi_x_rollout.shape != (args.mppi_horizon, markov_state_dim):
                        raise RuntimeError(
                            f"MPPI returned state_seq shape {mppi_x_rollout.shape}, expected {(args.mppi_horizon, markov_state_dim)}."
                        )
                    if not np.all(np.isfinite(mppi_u_full)) or not np.all(np.isfinite(mppi_x_rollout)):
                        raise RuntimeError("MPPI returned non-finite nominal trajectory.")
                    mppi_x_ref = np.concatenate([current_state_np[None, :], mppi_x_rollout[: args.horizon]], axis=0)
                    mppi_u_init = mppi_u_full[: args.horizon]
                    _, u_plan, ilqr_solve_time, n_iters, plan_cost = solver.solve_tracking(
                        current_state_np,
                        mppi_x_ref,
                        mppi_u_init,
                    )
                    solve_time = mppi_solve_time + ilqr_solve_time
                    attempted_mpc_step = {
                        "step_idx": int(step_idx),
                        "action_mode": "task_3d_mapped" if args.optimize_3d_task_actions else "control_14d",
                        "current_task": current_task.astype(float).tolist(),
                        "task_goal": goal_task.astype(float).tolist(),
                        "q_before": q_before.astype(float).tolist(),
                        "mppi_u_first": mppi_u_full[0].astype(float).tolist(),
                        "mppi_reward": float(mppi_reward),
                        "mppi_solve_time_ms": float(mppi_solve_time * 1000.0),
                        "u_plan_first": u_plan[0].astype(float).tolist(),
                        "solve_time_ms": float(solve_time * 1000.0),
                        "ilqr_solve_time_ms": float(ilqr_solve_time * 1000.0),
                        "ilqr_iterations": int(n_iters),
                        "ilqr_cost": float(plan_cost),
                        "execute_steps_per_plan": int(args.execute_steps_per_plan),
                    }
                    block_steps = min(args.execute_steps_per_plan, args.max_mpc_steps - step_idx, args.horizon)
                    attempted_mpc_step["planned_block_steps"] = int(block_steps)
                    executed_plan_steps = block_steps if args.optimize_3d_task_actions else 1
                    attempted_mpc_step["executed_plan_steps"] = int(executed_plan_steps)

                    qpos = np.asarray(current_info["qpos"], dtype=np.float64)
                    info_before = current_info
                    q_before = qpos.astype(np.float32)
                    measured_before = measured_task_state(planner, info_before).astype(np.float32)
                    current_task = measured_before.astype(np.float64)

                    raw_control = None
                    action_task = None
                    planned_u = u_plan[:executed_plan_steps].sum(axis=0) if args.optimize_3d_task_actions else u_plan[0]
                    if args.optimize_3d_task_actions:
                        assert isinstance(dynamics, RealTaskDynamicsTorch)
                        planned_raw_task_actions = np.stack(
                            [dynamics.normalized_to_raw(u_plan[idx]) for idx in range(executed_plan_steps)],
                            axis=0,
                        )
                        raw_task_action = planned_raw_task_actions.sum(axis=0)
                        aggregate_action_min = executed_plan_steps * np.asarray(args.task_action_min_m, dtype=np.float64).reshape(3)
                        aggregate_action_max = executed_plan_steps * np.asarray(args.task_action_max_m, dtype=np.float64).reshape(3)
                        debug_clipped_action = np.clip(
                            raw_task_action,
                            aggregate_action_min,
                            aggregate_action_max,
                        )
                        debug_target_task = planner.task_bounds.clip(current_task + debug_clipped_action)
                        debug_effective_action = debug_target_task - current_task
                        attempted_mpc_step.update(
                            {
                                "raw_task_action": raw_task_action.astype(float).tolist(),
                                "raw_task_actions_summed": planned_raw_task_actions.astype(float).tolist(),
                                "clipped_task_action": debug_clipped_action.astype(float).tolist(),
                                "effective_task_action": debug_effective_action.astype(float).tolist(),
                                "target_task": debug_target_task.astype(float).tolist(),
                                "raw_task_action_norm_m": float(np.linalg.norm(raw_task_action)),
                                "clipped_task_action_norm_m": float(np.linalg.norm(debug_clipped_action)),
                                "effective_task_action_norm_m": float(np.linalg.norm(debug_effective_action)),
                            }
                        )
                        clipped_task_action, action_task, q_target, duration = plan_task_step(
                            planner,
                            current_task,
                            qpos,
                            raw_task_action,
                            args,
                            action_min=aggregate_action_min,
                            action_max=aggregate_action_max,
                        )
                    else:
                        assert isinstance(dynamics, RealControlDynamicsTorch)
                        raw_task_action = np.full(3, np.nan, dtype=np.float64)
                        clipped_task_action = np.full(3, np.nan, dtype=np.float64)
                        raw_control = dynamics.normalized_to_raw(planned_u)
                        attempted_mpc_step.update(
                            {
                                "raw_control": raw_control.astype(float).tolist(),
                                "raw_control_norm_rad": float(np.linalg.norm(raw_control)),
                            }
                        )
                        raw_control, q_target, duration = plan_control_step(dynamics, qpos, planned_u, args)
                        action_task = np.zeros(3, dtype=np.float64)
                    attempted_mpc_step.update(
                        {
                            "q_cmd": q_target.astype(float).tolist(),
                            "command_duration_s": float(duration),
                        }
                    )
                    target_task = (current_task + action_task).astype(np.float64) if action_task is not None else current_task
                    print(
                        "Commanded task delta "
                        f"raw={np.round(raw_task_action, 6).tolist()} "
                        f"clipped={np.round(clipped_task_action, 6).tolist()} "
                        f"effective={np.round(action_task, 6).tolist()} "
                        f"duration={duration:.3f}s "
                        f"(summed_plan_steps={executed_plan_steps})"
                    )
                    if args.pre_command_sleep_s > 0.0:
                        time.sleep(float(args.pre_command_sleep_s))
                    action_start = time.monotonic()
                    if not args.dry_run:
                        robot.command_joint_positions(q_target, duration=duration, blocking=True)
                        settled, settle_duration, settle_err, settle_speed = wait_until_settled(robot, q_target, args)
                        if not settled:
                            stop_reason = "settle_timeout"
                        else:
                            current_info = planner.step_info(robot, target_task)
                    else:
                        settle_duration = 0.0
                        settle_err = 0.0
                        settle_speed = 0.0
                        current_info = make_info_from_qpos(planner, q_target, target_task)
                    if stop_reason != "max_mpc_steps":
                        break
                    action_end = time.monotonic()
                    state_time = time.monotonic()
                    frame_packet = capture_fresh_frame(camera, args)
                    camera_time = time.monotonic()
                    current_frame = np.asarray(frame_packet["pixels"], dtype=np.uint8)
                    rollout_frames.append(current_frame.copy())
                    append_camera_packet(camera_packets, frame_packet, capture_time=camera_time)
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

                    ee_distance = measured_ee_goal_distance(current_info, goal_info)
                    measured_after = measured_task_state(planner, current_info).astype(np.float32)
                    task_goal_distance = float(np.linalg.norm(measured_after.astype(np.float64) - goal_task))
                    goal_distances.append(ee_distance)
                    task_goal_distances.append(task_goal_distance)
                    latent_goal_distances.append(float(torch.linalg.vector_norm(current_state - goal_state).item()))
                    rows.append(
                        {
                            "action": action_task.astype(np.float32),
                            "planned_action": action_task.astype(np.float32),
                            "measured_action": (measured_after - measured_before).astype(np.float32),
                            "observation": np.asarray(current_info["observation"], dtype=np.float32),
                            "task_before": measured_before.astype(np.float32),
                            "task_after": np.asarray(target_task, dtype=np.float32),
                            "task_target": np.asarray(target_task, dtype=np.float32),
                            "measured_task_before": measured_before.astype(np.float32),
                            "measured_task_after": measured_after.astype(np.float32),
                            "q_before": q_before.astype(np.float32),
                            "q_after": np.asarray(current_info["qpos"], dtype=np.float32),
                            "q_cmd": q_target.astype(np.float32),
                            "qpos": np.asarray(current_info["qpos"], dtype=np.float32),
                            "qvel": np.asarray(current_info["qvel"], dtype=np.float32),
                            "control": np.asarray(current_info["control"], dtype=np.float32),
                            "left_attachment_pos": np.asarray(current_info["left_attachment_pos"], dtype=np.float32),
                            "right_attachment_pos": np.asarray(current_info["right_attachment_pos"], dtype=np.float32),
                            "rope_length": np.asarray(current_info["rope_length"], dtype=np.float32),
                            "command_duration": np.asarray([duration], dtype=np.float32),
                            "settle_duration": np.asarray([settle_duration], dtype=np.float32),
                            "settle_position_error_deg": np.asarray([np.rad2deg(settle_err)], dtype=np.float32),
                            "settle_velocity_deg_s": np.asarray([np.rad2deg(settle_speed)], dtype=np.float32),
                            "success": np.asarray([True], dtype=np.bool_),
                            "action_start_monotonic_time": np.asarray([action_start], dtype=np.float64),
                            "action_end_monotonic_time": np.asarray([action_end], dtype=np.float64),
                            "state_monotonic_time": np.asarray([state_time], dtype=np.float64),
                            "u_plan": planned_u.astype(np.float32),
                            "raw_task_action": raw_task_action.astype(np.float32),
                            "clipped_task_action": clipped_task_action.astype(np.float32),
                            "latent_goal_distance": np.asarray([latent_goal_distances[-1]], dtype=np.float32),
                            "ee_goal_distance": np.asarray([ee_distance], dtype=np.float32),
                            "task_goal_distance": np.asarray([task_goal_distance], dtype=np.float32),
                            "solve_time_ms": np.asarray([solve_time * 1000.0], dtype=np.float32),
                            "mppi_solve_time_ms": np.asarray([mppi_solve_time * 1000.0], dtype=np.float32),
                            "mppi_reward": np.asarray([mppi_reward], dtype=np.float32),
                            "ilqr_iterations": np.asarray([n_iters], dtype=np.int64),
                            "ilqr_cost": np.asarray([plan_cost], dtype=np.float32),
                        }
                    )
                    step_idx += executed_plan_steps
                    progress.update(executed_plan_steps)
                    progress.set_postfix(
                        ee_goal=f"{ee_distance:.3f}",
                        task_goal=f"{task_goal_distance:.3f}",
                        reach=f"{float(measured_after[0]):.3f}",
                        height=f"{float(measured_after[1]):.3f}",
                        width=f"{float(measured_after[2]):.3f}",
                        latent=f"{latent_goal_distances[-1]:.3f}",
                        solve_ms=f"{solve_time * 1000.0:.1f}",
                        mppi_ms=f"{mppi_solve_time * 1000.0:.1f}",
                    )

                    shifted_mppi = mppi_u_full[executed_plan_steps:]
                    if shifted_mppi.shape[0] == 0:
                        shifted_mppi = mppi_u_full[-1:]
                    pad_len = args.mppi_horizon - shifted_mppi.shape[0]
                    if pad_len > 0:
                        shifted_mppi = np.concatenate([shifted_mppi, np.repeat(shifted_mppi[-1:], pad_len, axis=0)], axis=0)
                    prev_mppi_u = jnp.asarray(shifted_mppi[: args.mppi_horizon], dtype=jnp.float64)
                    if stop_reason != "max_mpc_steps":
                        break
        except KeyboardInterrupt:
            interrupted = True
            stop_reason = "keyboard_interrupt"
        except RuntimeError as error:
            runtime_error = str(error)
            stop_reason = "runtime_error"

        metrics.update(
            {
                "success": bool(success),
                "stop_reason": stop_reason,
                "interrupted": bool(interrupted),
                "runtime_error": runtime_error,
                "num_executed_steps": len(rows),
                "ee_goal_distance_initial_m": float(goal_distances[0]),
                "ee_goal_distance_final_m": float(goal_distances[-1]),
                "task_goal_distance_initial_m": float(task_goal_distances[0]),
                "task_goal_distance_final_m": float(task_goal_distances[-1]),
                "latent_goal_distance_initial": float(latent_goal_distances[0]),
                "latent_goal_distance_final": float(latent_goal_distances[-1]),
                "goal_distances_m": goal_distances,
                "task_goal_distances_m": task_goal_distances,
                "latent_goal_distances": latent_goal_distances,
                "attempted_mpc_step": attempted_mpc_step,
                "steps": [
                    {
                        "step_idx": int(index),
                        "u_plan": row["u_plan"].astype(float).tolist(),
                        "raw_task_action": row["raw_task_action"].astype(float).tolist(),
                        "clipped_task_action": row["clipped_task_action"].astype(float).tolist(),
                        "action": row["action"].astype(float).tolist(),
                        "measured_action": row["measured_action"].astype(float).tolist(),
                        "q_cmd": row["q_cmd"].astype(float).tolist(),
                        "command_duration_s": float(row["command_duration"][0]),
                        "settle_duration_s": float(row["settle_duration"][0]),
                        "settle_position_error_deg": float(row["settle_position_error_deg"][0]),
                        "settle_velocity_deg_s": float(row["settle_velocity_deg_s"][0]),
                        "ee_goal_distance_m": float(row["ee_goal_distance"][0]),
                        "task_goal_distance_m": float(row["task_goal_distance"][0]),
                        "latent_goal_distance": float(row["latent_goal_distance"][0]),
                        "solve_time_ms": float(row["solve_time_ms"][0]),
                        "mppi_solve_time_ms": float(row["mppi_solve_time_ms"][0]),
                        "mppi_reward": float(row["mppi_reward"][0]),
                        "ilqr_iterations": int(row["ilqr_iterations"][0]),
                        "ilqr_cost": float(row["ilqr_cost"][0]),
                    }
                    for index, row in enumerate(rows)
                ],
                "final_qpos": np.asarray(current_info["qpos"], dtype=np.float32).tolist(),
            }
        )
        video_path, video_error = save_video(rollout_frames, out_dir, args.video_fps)
        if video_path is not None:
            metrics["stop_motion_video_path"] = video_path
        if video_error is not None:
            metrics["stop_motion_video_error"] = video_error
            print(f"Warning: {video_error}")

        if args.write_hdf5_trajectory:
            h5_path = out_dir / "trajectory.h5"
            save_hdf5_rollout(
                h5_path,
                rows=rows,
                frames=rollout_frames,
                camera_packets=camera_packets,
                args=args,
                planner=planner,
                metrics=metrics,
            )
            metrics["hdf5_path"] = str(h5_path)
    finally:
        if camera is not None:
            camera.close()
        robot.stop()
        robot.close()

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved run to: {out_dir}")
    print(f"Metrics: {metrics_path}")
    print(
        "Final distances: "
        f"ee={metrics.get('ee_goal_distance_final_m', float('nan')):.4f}m, "
        f"task={metrics.get('task_goal_distance_final_m', float('nan')):.4f}m"
    )


if __name__ == "__main__":
    main()
