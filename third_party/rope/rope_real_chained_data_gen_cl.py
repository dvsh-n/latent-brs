#!/usr/bin/env python3
"""Collect real-hardware rope data with task-space closed-loop spline tracking."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTDIR = "rope/data/real_data"
DEFAULT_OUTPUT_NAME = "rope_real_chained_splines.h5"
CONFIG_FORBIDDEN_KEYS = {"i_understand_this_moves_real_robots"}
ACTION_DIM = 3
TASK_REACH_BOUNDS = (-0.1, 0.3)
TASK_HEIGHT_BOUNDS = (1.16, 1.35)
TASK_WIDTH_BOUNDS = (0.2, 0.65)
HARDWARE_HOME_Q0_DEG = np.array([-65.0, 3.0, -167.0, 113.0, 8.0, 5.0, 0.0], dtype=np.float64)
HARDWARE_HOME_Q1_DEG = np.array([65.0, 3.0, -13.0, -113.0, -8.0, -5.0, 0.0], dtype=np.float64)
ARM1_EE_Z_ROTATION_OFFSET_RAD = 0.0
ARM1_EE_POSITION_OFFSET_B = np.array([0.0, 0.0, 0.03], dtype=np.float64)
SHARD_SUFFIX_RE = re.compile(r"_shard(\d+)$")


@dataclass(frozen=True)
class TaskBounds:
    reach: tuple[float, float] = TASK_REACH_BOUNDS
    height: tuple[float, float] = TASK_HEIGHT_BOUNDS
    width: tuple[float, float] = TASK_WIDTH_BOUNDS

    def clip(self, values: np.ndarray | list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        lower = np.array([self.reach[0], self.height[0], self.width[0]], dtype=np.float64)
        upper = np.array([self.reach[1], self.height[1], self.width[1]], dtype=np.float64)
        return np.clip(array, lower, upper)

    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.reach[0] + self.reach[1]),
                0.5 * (self.height[0] + self.height[1]),
                0.5 * (self.width[0] + self.width[1]),
            ],
            dtype=np.float64,
        )


def create_resizable_dataset(
    h5: h5py.File,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = True,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def append_rows(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def should_continue(args: argparse.Namespace, num_trajectories: int, total_transitions: int) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def resolve_shard_paths(
    outdir: Path,
    output_name: str,
    *,
    shard_id: int | None = None,
) -> tuple[int, Path, Path, Path, Path]:
    requested = Path(output_name).name
    stem = Path(requested).stem if requested.endswith(".h5") else requested
    stem = SHARD_SUFFIX_RE.sub("", stem)
    if shard_id is None:
        existing_shards: list[int] = []
        for candidate in outdir.glob(f"{stem}_shard*.h5"):
            match = SHARD_SUFFIX_RE.search(candidate.stem)
            if match is not None:
                existing_shards.append(int(match.group(1)))
        shard = max(existing_shards, default=-1) + 1
    else:
        shard = shard_id
    shard_prefix = f"{stem}_shard{shard:04d}"
    output_path = outdir / f"{shard_prefix}.h5"
    async_video_path = outdir / f"{shard_prefix}_async_camera.mp4"
    async_metadata_path = outdir / f"{shard_prefix}_async_camera.csv"
    async_h5_path = outdir / f"{shard_prefix}_async_camera.h5"
    return shard, output_path, async_video_path, async_metadata_path, async_h5_path


def load_yaml_config(path: Path) -> dict[str, object]:
    with path.expanduser().open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to contain a mapping, got {type(config).__name__}.")
    return {str(key).replace("-", "_"): value for key, value in config.items()}


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, object], config_path: Path) -> None:
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown key(s) in {config_path}: {unknown}")
    forbidden = sorted(set(config) & CONFIG_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(
            f"Do not put hardware safety acknowledgement in {config_path}: {forbidden}. "
            "Pass --i-understand-this-moves-real-robots explicitly on the command line."
        )
    parser.set_defaults(**config)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-transitions", type=int, default=None)
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--control-timestep", type=float, default=0.02)
    parser.add_argument("--num-splines", type=int, default=8)
    parser.add_argument("--segment-duration", type=float, default=8.0)
    parser.add_argument("--midpoint-inflation-scale", type=float, default=0.0)
    parser.add_argument("--min-task-knot-distance", type=float, default=0.035)
    parser.add_argument("--task-reach-bounds", type=float, nargs=2, default=TASK_REACH_BOUNDS)
    parser.add_argument("--task-height-bounds", type=float, nargs=2, default=TASK_HEIGHT_BOUNDS)
    parser.add_argument("--task-width-bounds", type=float, nargs=2, default=TASK_WIDTH_BOUNDS)
    parser.add_argument("--home-q0-deg", type=float, nargs=7, default=HARDWARE_HOME_Q0_DEG.tolist())
    parser.add_argument("--home-q1-deg", type=float, nargs=7, default=HARDWARE_HOME_Q1_DEG.tolist())
    parser.add_argument("--plan-retry-attempts", type=int, default=20)
    parser.add_argument("--robot-backend", choices=("drake-lcm",), default="drake-lcm")
    parser.add_argument("--arm-mapping", choices=("robot0-left", "robot1-left"), default="robot0-left")
    parser.add_argument("--drake-publish-period", type=float, default=0.005)
    parser.add_argument("--task-cl-kp", type=float, nargs=3, default=[0.2, 0.2, 0.15])
    parser.add_argument("--task-cl-max-correction-m", type=float, nargs=3, default=[0.002, 0.002, 0.002])
    parser.add_argument("--home-move-duration", type=float, default=4.0)
    parser.add_argument("--max-home-x-gap-m", type=float, default=0.008)
    parser.add_argument("--max-home-z-gap-m", type=float, default=0.004)
    parser.add_argument("--trajectory-hold-duration", type=float, default=1.0)
    parser.add_argument("--trajectory-start-blend-duration", type=float, default=4.0)
    parser.add_argument("--trajectory-start-settle-duration", type=float, default=0.25)
    parser.add_argument("--trajectory-start-ready-tolerance-deg", type=float, default=0.25)
    parser.add_argument("--trajectory-start-ready-velocity-deg-s", type=float, default=1.0)
    parser.add_argument("--trajectory-start-ready-timeout", type=float, default=5.0)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-control-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--max-reset-joint-move-deg", type=float, default=90.0)
    parser.add_argument("--max-command-measured-gap-deg", type=float, default=0.25)
    parser.add_argument("--max-real-joint-speed-deg-s", type=float, default=8.0)
    parser.add_argument("--max-real-joint-accel-deg-s2", type=float, default=20.0)
    parser.add_argument("--ik-position-tol", type=float, default=0.005)
    parser.add_argument("--ik-max-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--disable-collision-guard", action="store_true")
    parser.add_argument("--arm-arm-min-distance", type=float, default=0.06)
    parser.add_argument("--collision-control-samples", type=int, default=5)
    parser.add_argument("--collision-reset-samples", type=int, default=25)
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--camera-backend", choices=("opencv", "ros2-topic"), default="opencv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-device", default=None)
    parser.add_argument("--camera-topic", default=None)
    parser.add_argument("--camera-transport", choices=("raw", "compressed"), default="raw")
    parser.add_argument("--camera-topic-timeout", type=float, default=5.0)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-crop-center-x", type=float, default=0.5)
    parser.add_argument("--camera-crop-center-y", type=float, default=0.5)
    parser.add_argument("--camera-crop-zoom", type=float, default=1.0)
    parser.add_argument("--camera-capture-width", type=int, default=None)
    parser.add_argument("--camera-capture-height", type=int, default=None)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--i-understand-this-moves-real-robots", action="store_true")
    if pre_args.config is not None:
        apply_config_defaults(parser, load_yaml_config(pre_args.config), pre_args.config)
    args = parser.parse_args()
    if not isinstance(args.outdir, Path):
        args.outdir = Path(args.outdir)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.plan_only and not args.i_understand_this_moves_real_robots:
        raise RuntimeError("Refusing to move hardware without --i-understand-this-moves-real-robots.")
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.shard_id is not None and args.shard_id < 0:
        raise ValueError("--shard-id must be non-negative when provided.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.control_timestep <= 0.0:
        raise ValueError("--control-timestep must be positive.")
    if args.num_splines < 1:
        raise ValueError("--num-splines must be positive.")
    if args.segment_duration <= 0.0:
        raise ValueError("--segment-duration must be positive.")
    if args.midpoint_inflation_scale < 0.0:
        raise ValueError("--midpoint-inflation-scale cannot be negative.")
    if args.min_task_knot_distance < 0.0:
        raise ValueError("--min-task-knot-distance cannot be negative.")
    if len(args.task_reach_bounds) != 2 or args.task_reach_bounds[0] >= args.task_reach_bounds[1]:
        raise ValueError("--task-reach-bounds must provide two increasing values.")
    if len(args.task_height_bounds) != 2 or args.task_height_bounds[0] >= args.task_height_bounds[1]:
        raise ValueError("--task-height-bounds must provide two increasing values.")
    if len(args.task_width_bounds) != 2 or args.task_width_bounds[0] >= args.task_width_bounds[1]:
        raise ValueError("--task-width-bounds must provide two increasing values.")
    if args.home_move_duration <= 10.0:
        raise ValueError("--home-move-duration must be greater than 10.0s so it uses reset-motion safety limits.")
    if args.max_home_x_gap_m < 0.0:
        raise ValueError("--max-home-x-gap-m cannot be negative.")
    if args.max_home_z_gap_m < 0.0:
        raise ValueError("--max-home-z-gap-m cannot be negative.")
    if args.plan_retry_attempts < 1:
        raise ValueError("--plan-retry-attempts must be positive.")
    if args.arm_mapping != "robot0-left":
        raise ValueError("This collector follows the KUKA arm0/arm1 convention; use --arm-mapping robot0-left.")
    if args.drake_publish_period <= 0.0:
        raise ValueError("--drake-publish-period must be positive.")
    if len(args.task_cl_kp) != 3 or any(value < 0.0 for value in args.task_cl_kp):
        raise ValueError("--task-cl-kp must provide 3 non-negative gains.")
    if len(args.task_cl_max_correction_m) != 3 or any(value <= 0.0 for value in args.task_cl_max_correction_m):
        raise ValueError("--task-cl-max-correction-m must provide 3 positive correction limits.")
    if args.max_control_joint_step_deg <= 0.0:
        raise ValueError("--max-control-joint-step-deg must be positive.")
    if args.max_reset_joint_move_deg <= 0.0:
        raise ValueError("--max-reset-joint-move-deg must be positive.")
    if args.max_command_measured_gap_deg <= 0.0:
        raise ValueError("--max-command-measured-gap-deg must be positive.")
    if args.max_real_joint_speed_deg_s <= 0.0:
        raise ValueError("--max-real-joint-speed-deg-s must be positive.")
    if args.max_real_joint_accel_deg_s2 <= 0.0:
        raise ValueError("--max-real-joint-accel-deg-s2 must be positive.")
    if args.ik_position_tol <= 0.0:
        raise ValueError("--ik-position-tol must be positive.")
    if args.ik_max_joint_step_deg <= 0.0:
        raise ValueError("--ik-max-joint-step-deg must be positive.")
    if args.arm_arm_min_distance < 0.0:
        raise ValueError("--arm-arm-min-distance cannot be negative.")
    if args.collision_control_samples < 1:
        raise ValueError("--collision-control-samples must be positive.")
    if args.collision_reset_samples < 1:
        raise ValueError("--collision-reset-samples must be positive.")
    if args.camera_width < 1:
        raise ValueError("--camera-width must be positive.")
    if args.camera_height < 1:
        raise ValueError("--camera-height must be positive.")
    if not 0.0 <= args.camera_crop_center_x <= 1.0:
        raise ValueError("--camera-crop-center-x must be between 0.0 and 1.0.")
    if not 0.0 <= args.camera_crop_center_y <= 1.0:
        raise ValueError("--camera-crop-center-y must be between 0.0 and 1.0.")
    if args.camera_crop_zoom <= 0.0:
        raise ValueError("--camera-crop-zoom must be positive.")
    if args.camera_topic_timeout <= 0.0:
        raise ValueError("--camera-topic-timeout must be positive.")
    if args.camera_capture_width is not None and args.camera_capture_width < 1:
        raise ValueError("--camera-capture-width must be positive when provided.")
    if args.camera_capture_height is not None and args.camera_capture_height < 1:
        raise ValueError("--camera-capture-height must be positive when provided.")
    if args.camera_warmup_frames < 0:
        raise ValueError("--camera-warmup-frames cannot be negative.")
    if args.camera_backend == "ros2-topic" and not args.camera_topic:
        raise ValueError("--camera-topic is required when --camera-backend ros2-topic is selected.")
    if args.camera_backend == "ros2-topic":
        topic = str(args.camera_topic)
        if topic.endswith("/compressed") and args.camera_transport != "compressed":
            raise ValueError(
                "camera_topic ends with /compressed but camera_transport is not compressed. "
                "Use --camera-transport compressed or point camera_topic at a raw image topic."
            )
        if not topic.endswith("/compressed") and args.camera_transport == "compressed":
            raise ValueError(
                "camera_transport is compressed but camera_topic does not end with /compressed. "
                "Use a compressed image topic or set --camera-transport raw."
            )
    if len(args.home_q0_deg) != 7:
        raise ValueError("--home-q0-deg must provide 7 joint values.")
    if len(args.home_q1_deg) != 7:
        raise ValueError("--home-q1-deg must provide 7 joint values.")


def make_robot(args: argparse.Namespace):
    from rope.real.drake_lcm_backend import DrakeLCMBimanualRobotBackend

    return DrakeLCMBimanualRobotBackend(
        arm_mapping=args.arm_mapping,
        publish_period=args.drake_publish_period,
        status_timeout=args.status_timeout,
        max_control_joint_step=np.deg2rad(args.max_control_joint_step_deg),
        max_reset_joint_move=np.deg2rad(args.max_reset_joint_move_deg),
        max_command_measured_gap=np.deg2rad(args.max_command_measured_gap_deg),
        hold_duration=args.trajectory_hold_duration,
        start_blend_duration=args.trajectory_start_blend_duration,
        start_settle_duration=args.trajectory_start_settle_duration,
        start_ready_tolerance=np.deg2rad(args.trajectory_start_ready_tolerance_deg),
        start_ready_velocity_tolerance=np.deg2rad(args.trajectory_start_ready_velocity_deg_s),
        start_ready_timeout=args.trajectory_start_ready_timeout,
    )


def make_camera(args: argparse.Namespace):
    if not args.enable_camera:
        return None
    from rope.real.camera_backend import OpenCVCamera, Ros2TopicCamera

    if args.camera_backend == "ros2-topic":
        return Ros2TopicCamera(
            topic=args.camera_topic,
            transport=args.camera_transport,
            output_width=args.camera_width,
            output_height=args.camera_height,
            crop_center_x=args.camera_crop_center_x,
            crop_center_y=args.camera_crop_center_y,
            crop_zoom=args.camera_crop_zoom,
            timeout_sec=args.camera_topic_timeout,
        )

    return OpenCVCamera(
        index=args.camera_index,
        device_path=args.camera_device,
        output_width=args.camera_width,
        output_height=args.camera_height,
        crop_center_x=args.camera_crop_center_x,
        crop_center_y=args.camera_crop_center_y,
        crop_zoom=args.camera_crop_zoom,
        capture_width=args.camera_capture_width,
        capture_height=args.camera_capture_height,
        warmup_frames=args.camera_warmup_frames,
    )


class HomeAnchoredTaskPlanner:
    def __init__(self, args: argparse.Namespace) -> None:
        from rope.data.iiwa_cartesian_ik import SingleIiwaPositionIK

        self.args = args
        self.task_bounds = TaskBounds(
            reach=tuple(float(value) for value in args.task_reach_bounds),
            height=tuple(float(value) for value in args.task_height_bounds),
            width=tuple(float(value) for value in args.task_width_bounds),
        )
        self.home_task = self.task_bounds.center()
        self.ik0 = SingleIiwaPositionIK()
        self.ik1 = SingleIiwaPositionIK(
            ee_z_rotation_offset_rad=ARM1_EE_Z_ROTATION_OFFSET_RAD,
            ee_position_offset_B=ARM1_EE_POSITION_OFFSET_B,
        )
        self.ik_position_tol = float(args.ik_position_tol)
        self.ik_max_joint_step = np.deg2rad(args.ik_max_joint_step_deg)
        q0_home, q1_home = self._hardware_home_goal()
        self.hardware_home_qpos = np.concatenate([q0_home, q1_home]).astype(np.float64)
        self.dataset_home_qpos = self.hardware_home_qpos.copy()
        self.p0_home = self.ik0.fk_position(q0_home)
        self.p1_home = self.ik1.fk_position(q1_home)
        self.nominal_home_attachment_delta_xz = self._attachment_delta_xz(self.hardware_home_qpos)

    def clip_task(self, values: np.ndarray) -> np.ndarray:
        return self.task_bounds.clip(values).astype(np.float64)

    def task_to_qpos(self, task_state: np.ndarray, q_seed: np.ndarray, *, label: str) -> np.ndarray:
        task_state = self.clip_task(task_state)
        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(14)
        delta = task_state - self.home_task
        p0_goal = self.p0_home + np.array([delta[0], 0.5 * delta[2], delta[1]], dtype=np.float64)
        p1_goal = self.p1_home + np.array([delta[0], -0.5 * delta[2], delta[1]], dtype=np.float64)
        q0_cmd = self._solve_arm(self.ik0, p0_goal, q_seed[:7], f"arm0 {label}")
        q1_cmd = self._solve_arm(self.ik1, p1_goal, q_seed[7:], f"arm1 {label}")
        return np.concatenate([q0_cmd, q1_cmd])

    def task_closed_loop_qpos(
        self,
        task_ref: np.ndarray,
        q_meas: np.ndarray,
        *,
        q_seed: np.ndarray | None = None,
        kp: np.ndarray,
        max_correction: np.ndarray,
    ) -> np.ndarray:
        q_meas = np.asarray(q_meas, dtype=np.float64).reshape(14)
        if q_seed is None:
            q_seed = q_meas
        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(14)
        p0 = self.ik0.fk_position(q_meas[:7])
        p1 = self.ik1.fk_position(q_meas[7:])
        task_meas = self.measured_task_state_from_attachment_positions(p0, p1)
        task_err = self.clip_task(task_ref) - task_meas
        correction = np.clip(kp * task_err, -max_correction, max_correction)
        task_cmd = self.clip_task(task_ref + correction)
        return self.task_to_qpos(task_cmd, q_seed, label="closed-loop servo")

    def plan_joint_path(self, task_path: np.ndarray) -> np.ndarray:
        task_path = np.asarray(task_path, dtype=np.float64)
        if task_path.ndim != 2 or task_path.shape[1] != 3:
            raise ValueError(f"Expected task_path with shape (T, 3), got {task_path.shape}.")
        q0_prev = self.dataset_home_qpos[:7].copy()
        q1_prev = self.dataset_home_qpos[7:].copy()
        q_path = [self.dataset_home_qpos.copy()]
        for index, task_state in enumerate(task_path[1:], start=1):
            delta = task_state - self.home_task
            p0_goal = self.p0_home + np.array([delta[0], 0.5 * delta[2], delta[1]], dtype=np.float64)
            p1_goal = self.p1_home + np.array([delta[0], -0.5 * delta[2], delta[1]], dtype=np.float64)
            q0_prev = self._solve_arm(self.ik0, p0_goal, q0_prev, f"arm0 waypoint {index}")
            q1_prev = self._solve_arm(self.ik1, p1_goal, q1_prev, f"arm1 waypoint {index}")
            q_path.append(np.concatenate([q0_prev, q1_prev]))
        q_path[-1] = self.dataset_home_qpos.copy()
        return np.stack(q_path, axis=0)

    def step_info(
        self,
        robot,
        task_target: np.ndarray,
        elapsed_time: float,
    ) -> dict[str, np.ndarray]:
        qpos = robot.read_qpos_14().astype(np.float32)
        qvel = robot.read_qvel_14().astype(np.float32)
        control = qpos.copy() if robot.last_commanded_qpos is None else robot.last_commanded_qpos.astype(np.float32)
        p0 = self.ik0.fk_position(qpos[:7]).astype(np.float32)
        p1 = self.ik1.fk_position(qpos[7:]).astype(np.float32)
        rope_length = np.asarray([np.linalg.norm(p0 - p1)], dtype=np.float32)
        target = np.asarray(task_target, dtype=np.float32)
        measured_task_target = self.measured_task_state_from_attachment_positions(p0, p1).astype(np.float32)
        observation = np.concatenate([target, qpos, qvel, control, p0, p1, rope_length], axis=0).astype(np.float32)
        return {
            "observation": observation,
            "task_target": target,
            "measured_task_target": measured_task_target,
            "qpos": qpos,
            "qvel": qvel,
            "control": control,
            "left_attachment_pos": p0,
            "right_attachment_pos": p1,
            "rope_length": rope_length,
            "time": np.asarray([elapsed_time], dtype=np.float32),
        }

    def _solve_arm(
        self,
        ik: SingleIiwaPositionIK,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        label: str,
    ) -> np.ndarray:
        q_sol, info = ik.solve_position_ik(
            target_pos,
            q_seed,
            position_tol=self.ik_position_tol,
            max_joint_move_from_seed=self.ik_max_joint_step,
        )
        if q_sol is None:
            raise RuntimeError(f"{label} IK failed for target {np.round(target_pos, 4)}: {info}")
        return q_sol

    def _attachment_delta_xz(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(14)
        p0 = self.ik0.fk_position(qpos[:7])
        p1 = self.ik1.fk_position(qpos[7:])
        return np.array([p0[0] - p1[0], p0[2] - p1[2]], dtype=np.float64)

    def _hardware_home_goal(self) -> tuple[np.ndarray, np.ndarray]:
        # Solve the configured hardware home once with the constrained link7
        # orientation used by the IK helper so planning and pre-homing agree.
        q0_nominal = np.deg2rad(np.asarray(self.args.home_q0_deg, dtype=np.float64))
        q1_nominal = np.deg2rad(np.asarray(self.args.home_q1_deg, dtype=np.float64))
        p0_home = self.ik0.fk_position(q0_nominal)
        p1_home = self.ik1.fk_position(q1_nominal)
        q0_home, info0 = self.ik0.solve_position_ik(
            p0_home,
            q0_nominal,
            position_tol=0.002,
            max_joint_move_from_seed=None,
        )
        q1_home, info1 = self.ik1.solve_position_ik(
            p1_home,
            q1_nominal,
            position_tol=0.002,
            max_joint_move_from_seed=None,
        )
        if q0_home is None or q1_home is None:
            raise RuntimeError(f"Could not solve oriented hardware home: arm0={info0}, arm1={info1}")
        print(
            "oriented home IK errors: "
            f"arm0 pos={info0['pos_err']:.5f}m orient={np.rad2deg(info0['orientation_err']):.2f}deg; "
            f"arm1 pos={info1['pos_err']:.5f}m orient={np.rad2deg(info1['orientation_err']):.2f}deg"
        )
        return q0_home, q1_home

    def anchor_dataset_home(self, measured_qpos: np.ndarray) -> None:
        measured_qpos = np.asarray(measured_qpos, dtype=np.float64).reshape(14)
        self.dataset_home_qpos = measured_qpos.copy()
        self.p0_home = self.ik0.fk_position(measured_qpos[:7])
        self.p1_home = self.ik1.fk_position(measured_qpos[7:])
        hardware_delta_deg = np.rad2deg(np.abs(measured_qpos - self.hardware_home_qpos))
        measured_delta_xz = self._attachment_delta_xz(measured_qpos)
        home_error_xz = np.abs(measured_delta_xz - self.nominal_home_attachment_delta_xz)
        print(
            "Anchored dataset home to measured pose: "
            f"max_joint_offset_from_hardware_home={np.max(hardware_delta_deg):.3f}deg, "
            f"nominal_attachment_dx={self.nominal_home_attachment_delta_xz[0]:.4f}m, "
            f"nominal_attachment_dz={self.nominal_home_attachment_delta_xz[1]:.4f}m, "
            f"measured_attachment_dx={measured_delta_xz[0]:.4f}m, "
            f"measured_attachment_dz={measured_delta_xz[1]:.4f}m, "
            f"home_x_error={home_error_xz[0]:.4f}m, home_z_error={home_error_xz[1]:.4f}m"
        )
        if (
            home_error_xz[0] > self.args.max_home_x_gap_m
            or home_error_xz[1] > self.args.max_home_z_gap_m
        ):
            raise RuntimeError(
                "Measured home pose does not match the configured collection home closely enough: "
                f"home_x_error={home_error_xz[0]:.4f}m (limit {self.args.max_home_x_gap_m:.4f}m), "
                f"home_z_error={home_error_xz[1]:.4f}m (limit {self.args.max_home_z_gap_m:.4f}m). "
                "Re-home or fix the physical alignment before collecting."
            )

    def measured_task_state_from_attachment_positions(
        self,
        left_attachment_pos: np.ndarray,
        right_attachment_pos: np.ndarray,
    ) -> np.ndarray:
        left = np.asarray(left_attachment_pos, dtype=np.float64)
        right = np.asarray(right_attachment_pos, dtype=np.float64)
        left_delta = left - self.p0_home
        right_delta = right - self.p1_home
        measured = self.home_task.copy()
        measured[0] += 0.5 * (left_delta[0] + right_delta[0])
        measured[1] += 0.5 * (left_delta[2] + right_delta[2])
        measured[2] += left_delta[1] - right_delta[1]
        return self.clip_task(measured)


def move_robot_to_planner_home(
    robot,
    planner: HomeAnchoredTaskPlanner,
    args: argparse.Namespace,
) -> None:
    measured = robot.read_qpos_14()
    max_delta_deg = float(np.rad2deg(np.max(np.abs(planner.hardware_home_qpos - measured))))
    if max_delta_deg <= args.trajectory_start_ready_tolerance_deg:
        print(f"Robot already at configured chained home: max_delta={max_delta_deg:.2f}deg")
        robot.prepare_joint_path_start(planner.hardware_home_qpos)
    else:
        print(
            "Moving robot to configured chained home: "
            f"duration={args.home_move_duration:.2f}s, max_delta={max_delta_deg:.2f}deg"
        )
        robot.command_joint_positions(
            planner.hardware_home_qpos,
            duration=args.home_move_duration,
            blocking=True,
        )
        robot.prepare_joint_path_start(planner.hardware_home_qpos)
    planner.anchor_dataset_home(robot.read_qpos_14())


def sample_task_point(rng: np.random.Generator, bounds: TaskBounds) -> np.ndarray:
    lower = np.array([bounds.reach[0], bounds.height[0], bounds.width[0]], dtype=np.float64)
    upper = np.array([bounds.reach[1], bounds.height[1], bounds.width[1]], dtype=np.float64)
    return rng.uniform(lower, upper)


def sample_nontrivial_task_point(
    rng: np.random.Generator,
    bounds: TaskBounds,
    previous: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    for _ in range(100):
        point = sample_task_point(rng, bounds)
        if float(np.max(np.abs(point - previous))) >= min_distance:
            return point
    return sample_task_point(rng, bounds)


def sample_chained_task_path(
    planner: HomeAnchoredTaskPlanner,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bounds = planner.task_bounds
    total_duration = float(args.num_splines) * float(args.segment_duration)
    sample_times = np.arange(0.0, total_duration, args.control_timestep, dtype=np.float64)
    sample_times = np.append(sample_times, total_duration)

    knot_times = [0.0]
    knots = [planner.home_task.copy()]
    previous_goal = planner.home_task.copy()
    lower = np.array([bounds.reach[0], bounds.height[0], bounds.width[0]], dtype=np.float64)
    upper = np.array([bounds.reach[1], bounds.height[1], bounds.width[1]], dtype=np.float64)
    extent = upper - lower
    inflated_lower = lower - float(args.midpoint_inflation_scale) * extent
    inflated_upper = upper + float(args.midpoint_inflation_scale) * extent

    for spline_index in range(args.num_splines):
        t0 = spline_index * args.segment_duration
        midpoint = rng.uniform(inflated_lower, inflated_upper)
        midpoint = planner.clip_task(midpoint)
        if spline_index == args.num_splines - 1:
            goal = planner.home_task.copy()
        else:
            goal = sample_nontrivial_task_point(rng, bounds, previous_goal, args.min_task_knot_distance)
        knot_times.extend([t0 + 0.5 * args.segment_duration, t0 + args.segment_duration])
        knots.extend([midpoint, goal])
        previous_goal = goal

    knot_times_array = np.asarray(knot_times, dtype=np.float64)
    knots_array = np.stack(knots, axis=0)
    # PCHIP is C1 smooth and shape-preserving in each task dimension. That keeps
    # the path inside the task bounds when all knots are inside the task bounds,
    # while avoiding stop-and-go velocity resets at internal spline goals.
    spline = PchipInterpolator(knot_times_array, knots_array, axis=0)
    task_path = np.asarray(spline(sample_times), dtype=np.float64)
    task_path[0] = planner.home_task.copy()
    task_path[-1] = planner.home_task.copy()
    return task_path, sample_times, knots_array


def joint_path_speed_accel(q_path: np.ndarray, sample_times: np.ndarray) -> tuple[float, float, float, float]:
    q_path = np.asarray(q_path, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if q_path.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0
    dt = np.diff(sample_times)
    dq = np.diff(q_path, axis=0)
    max_step = float(np.max(np.abs(dq))) if dq.size else 0.0
    velocity = dq / np.maximum(dt[:, None], 1e-9)
    max_speed = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    segment_peak_speeds = np.max(np.abs(velocity), axis=1) if velocity.size else np.zeros((0,), dtype=np.float64)
    avg_speed = float(np.sum(segment_peak_speeds * dt) / np.maximum(np.sum(dt), 1e-9)) if segment_peak_speeds.size else 0.0
    if velocity.shape[0] < 2:
        return max_step, max_speed, avg_speed, 0.0
    accel_dt = 0.5 * (dt[1:] + dt[:-1])
    max_accel = float(np.max(np.abs(np.diff(velocity, axis=0)) / np.maximum(accel_dt[:, None], 1e-9)))
    return max_step, max_speed, avg_speed, max_accel


def densify_for_joint_step_limit(
    q_path: np.ndarray,
    task_path: np.ndarray,
    sample_times: np.ndarray,
    *,
    max_joint_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_path = np.asarray(q_path, dtype=np.float64)
    task_path = np.asarray(task_path, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    dense_q = [q_path[0]]
    dense_task = [task_path[0]]
    dense_times = [sample_times[0]]
    for index in range(1, q_path.shape[0]):
        max_delta = float(np.max(np.abs(q_path[index] - q_path[index - 1])))
        pieces = max(1, int(np.ceil(max_delta / max(max_joint_step, 1e-12))))
        for piece in range(1, pieces + 1):
            alpha = piece / pieces
            dense_q.append((1.0 - alpha) * q_path[index - 1] + alpha * q_path[index])
            dense_task.append((1.0 - alpha) * task_path[index - 1] + alpha * task_path[index])
            dense_times.append((1.0 - alpha) * sample_times[index - 1] + alpha * sample_times[index])
    return np.stack(dense_q, axis=0), np.stack(dense_task, axis=0), np.asarray(dense_times, dtype=np.float64)


def retime_for_joint_limits(
    q_path: np.ndarray,
    sample_times: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    q_path = np.asarray(q_path, dtype=np.float64)
    base_times = np.asarray(sample_times, dtype=np.float64)
    if q_path.shape[0] != base_times.shape[0]:
        raise ValueError(f"Expected q_path and sample_times to align, got {q_path.shape[0]} and {base_times.shape[0]}.")
    if q_path.shape[0] < 2:
        return base_times.copy(), 1.0

    max_speed_limit = np.deg2rad(args.max_real_joint_speed_deg_s)
    max_accel_limit = np.deg2rad(args.max_real_joint_accel_deg_s2)
    nominal_dt = np.diff(base_times)
    dq = np.diff(q_path, axis=0)
    peak_joint_delta = np.max(np.abs(dq), axis=1) if dq.size else np.zeros((0,), dtype=np.float64)
    speed_limited_dt = peak_joint_delta / max(max_speed_limit, 1e-9)

    # Start from the nominal schedule, but give each interval enough time for
    # its own joint displacement instead of stretching the whole trajectory.
    dt = np.maximum(nominal_dt, speed_limited_dt) * 1.02

    for _ in range(32):
        times = np.concatenate([[base_times[0]], base_times[0] + np.cumsum(dt)])
        _max_step, max_speed, avg_speed, max_accel = joint_path_speed_accel(q_path, times)
        speed_ratio = max_speed / max_speed_limit
        accel_ratio = max_accel / max_accel_limit
        if speed_ratio <= 1.0 and accel_ratio <= 1.0:
            total_scale = float(times[-1] / max(base_times[-1], 1e-9))
            print(
                "Final timing: "
                f"duration={times[-1]:.2f}s, scale={total_scale:.3f}, "
                f"max_speed={np.rad2deg(max_speed):.2f}deg/s, "
                f"avg_speed={np.rad2deg(avg_speed):.2f}deg/s, "
                f"max_accel={np.rad2deg(max_accel):.2f}deg/s^2"
            )
            return times, total_scale

        dt *= max(1.0, 1.02 * speed_ratio)
        if dt.shape[0] >= 2:
            velocity = dq / np.maximum(dt[:, None], 1e-9)
            accel_dt = 0.5 * (dt[1:] + dt[:-1])
            accel = np.abs(np.diff(velocity, axis=0)) / np.maximum(accel_dt[:, None], 1e-9)
            transition_ratio = np.max(accel, axis=1) / max(max_accel_limit, 1e-9)
            if transition_ratio.size:
                growth = np.ones_like(dt)
                violating = np.nonzero(transition_ratio > 1.0)[0]
                for index in violating:
                    scale = 1.02 * np.sqrt(transition_ratio[index])
                    growth[index] = max(growth[index], scale)
                    growth[index + 1] = max(growth[index + 1], scale)
                dt *= growth
    raise RuntimeError(
        "Could not retime chained trajectory within speed/acceleration limits: "
        f"speed={np.rad2deg(max_speed):.2f}deg/s, accel={np.rad2deg(max_accel):.2f}deg/s^2."
    )


def validate_collision(q_path: np.ndarray, sample_times: np.ndarray, args: argparse.Namespace, *, label: str) -> None:
    if args.disable_collision_guard:
        print("Collision guard skipped because --disable-collision-guard is set.")
        return
    try:
        from rope.real.collision_guard import MujocoArmCollisionGuard
    except ModuleNotFoundError as error:
        if error.name == "mujoco":
            raise RuntimeError(
                "Collision guard requires the `mujoco` Python package in the active interpreter. "
                "Install `mujoco==3.6.0` for this Python environment, or explicitly pass "
                "--disable-collision-guard if you intend to run without that safety check."
            ) from error
        raise
    from rope.shared.lab_env import LabEnv

    guard = MujocoArmCollisionGuard(
        LabEnv(),
        min_arm_arm_distance=args.arm_arm_min_distance,
        control_path_samples=args.collision_control_samples,
        reset_path_samples=args.collision_reset_samples,
    )
    for index in range(1, q_path.shape[0]):
        guard.validate_path(
            q_path[index - 1],
            q_path[index],
            duration=float(sample_times[index] - sample_times[index - 1]),
            label=f"{label} segment {index - 1}->{index}",
        )
    print(f"Collision guard passed for {q_path.shape[0]} chained waypoints.")


def make_task_interpolator(task_path: np.ndarray, sample_times: np.ndarray):
    task_path = np.asarray(task_path, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.float64)
    if task_path.shape != (times.shape[0], ACTION_DIM):
        raise ValueError(f"Expected task_path shape ({times.shape[0]}, {ACTION_DIM}), got {task_path.shape}.")
    if times.shape[0] < 2:
        raise ValueError("Need at least two task samples for closed-loop interpolation.")
    first_dt = max(float(times[1] - times[0]), 1e-9)
    last_dt = max(float(times[-1] - times[-2]), 1e-9)
    interp_times = np.concatenate([[times[0] - first_dt], times, [times[-1] + last_dt]])
    interp_path = np.concatenate([task_path[:1], task_path, task_path[-1:]], axis=0)
    interpolator = PchipInterpolator(interp_times, interp_path, axis=0)

    def interpolate(sample_time: float) -> np.ndarray:
        if sample_time <= times[0]:
            return task_path[0].copy()
        if sample_time >= times[-1]:
            return task_path[-1].copy()
        return np.asarray(interpolator(sample_time), dtype=np.float64).reshape(ACTION_DIM)

    return interpolate


def make_qpos_interpolator(q_path: np.ndarray, sample_times: np.ndarray):
    q_path = np.asarray(q_path, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.float64)
    if q_path.shape != (times.shape[0], 14):
        raise ValueError(f"Expected q_path shape ({times.shape[0]}, 14), got {q_path.shape}.")
    if times.shape[0] < 2:
        raise ValueError("Need at least two qpos samples for closed-loop fallback interpolation.")
    first_dt = max(float(times[1] - times[0]), 1e-9)
    last_dt = max(float(times[-1] - times[-2]), 1e-9)
    interp_times = np.concatenate([[times[0] - first_dt], times, [times[-1] + last_dt]])
    interp_path = np.concatenate([q_path[:1], q_path, q_path[-1:]], axis=0)
    interpolator = PchipInterpolator(interp_times, interp_path, axis=0)

    def interpolate(sample_time: float) -> np.ndarray:
        if sample_time <= times[0]:
            return q_path[0].copy()
        if sample_time >= times[-1]:
            return q_path[-1].copy()
        return np.asarray(interpolator(sample_time), dtype=np.float64).reshape(14)

    return interpolate


def limit_qpos_step(q_target: np.ndarray, q_start: np.ndarray, *, max_step: float) -> np.ndarray:
    q_target = np.asarray(q_target, dtype=np.float64).reshape(14)
    q_start = np.asarray(q_start, dtype=np.float64).reshape(14)
    delta = q_target - q_start
    max_delta = float(np.max(np.abs(delta)))
    if max_delta <= max_step:
        return q_target
    return q_start + delta * (max_step / max(max_delta, 1e-12))


def limit_qpos_motion(
    q_target: np.ndarray,
    q_start: np.ndarray,
    previous_velocity: np.ndarray,
    *,
    timestep: float,
    max_step: float,
    max_speed: float,
    max_accel: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_target = np.asarray(q_target, dtype=np.float64).reshape(14)
    q_start = np.asarray(q_start, dtype=np.float64).reshape(14)
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64).reshape(14)
    dt = max(float(timestep), 1e-9)

    desired_velocity = (q_target - q_start) / dt
    desired_velocity = np.clip(desired_velocity, -max_speed, max_speed)
    desired_velocity = np.clip(
        desired_velocity,
        previous_velocity - max_accel * dt,
        previous_velocity + max_accel * dt,
    )
    q_limited = q_start + desired_velocity * dt
    q_limited = limit_qpos_step(q_limited, q_start, max_step=max_step)
    return q_limited, (q_limited - q_start) / dt


def command_task_path_closed_loop(
    robot,
    planner: HomeAnchoredTaskPlanner,
    q_path: np.ndarray,
    task_path: np.ndarray,
    sample_times: np.ndarray,
    args: argparse.Namespace,
    *,
    sample_callback=None,
) -> None:
    times = np.asarray(sample_times, dtype=np.float64)
    if not np.isclose(times[0], 0.0):
        raise ValueError("sample_times must start at 0.0.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times must be strictly increasing.")

    task_ref_at = make_task_interpolator(task_path, times)
    q_ref_at = make_qpos_interpolator(q_path, times)
    kp = np.asarray(args.task_cl_kp, dtype=np.float64).reshape(ACTION_DIM)
    max_correction = np.asarray(args.task_cl_max_correction_m, dtype=np.float64).reshape(ACTION_DIM)

    duration = float(times[-1])
    period = float(args.drake_publish_period)
    current = 0.0
    sample_index = 1
    q_cmd = robot.last_commanded_qpos.copy() if robot.last_commanded_qpos is not None else robot.read_qpos_14()
    q_cmd_velocity = np.zeros(14, dtype=np.float64)
    fallback_count = 0
    eps = 1e-9

    while current < duration - eps:
        next_time = min(current + period, duration)
        timestep = next_time - current
        q_meas = robot.read_qpos_14()
        task_ref = task_ref_at(next_time)
        q_ref = q_ref_at(next_time)
        for correction_scale in (1.0, 0.5, 0.25, 0.0):
            try:
                q_cmd = planner.task_closed_loop_qpos(
                    task_ref,
                    q_meas,
                    q_seed=q_ref,
                    kp=kp * correction_scale,
                    max_correction=max_correction * correction_scale,
                )
                break
            except RuntimeError as error:
                last_error = error
        else:
            fallback_count += 1
            if fallback_count <= 5 or fallback_count % 100 == 0:
                print(
                    "Closed-loop IK fallback to nominal joint reference: "
                    f"t={next_time:.3f}s, count={fallback_count}, error={last_error}"
                )
            q_cmd = q_ref
        q_start = robot.last_commanded_qpos.copy() if robot.last_commanded_qpos is not None else q_meas
        q_cmd, q_cmd_velocity = limit_qpos_motion(
            q_cmd,
            q_start,
            q_cmd_velocity,
            timestep=timestep,
            max_step=0.95 * np.deg2rad(args.max_control_joint_step_deg),
            max_speed=np.deg2rad(args.max_real_joint_speed_deg_s),
            max_accel=np.deg2rad(args.max_real_joint_accel_deg_s2),
        )
        robot.command_joint_path_step(q_cmd, timestep=timestep)
        current = next_time

        while sample_index < len(times) and times[sample_index] <= current + eps:
            if sample_callback is not None:
                sample_callback(sample_index, float(times[sample_index]), q_cmd.copy())
            sample_index += 1

    if fallback_count:
        print(f"Closed-loop IK used nominal joint fallback for {fallback_count} publish ticks.")

    if robot.last_commanded_qpos is not None:
        robot.command_joint_positions(robot.last_commanded_qpos, duration=args.trajectory_hold_duration, blocking=True)


def prepare_plan(
    planner: HomeAnchoredTaskPlanner,
    args: argparse.Namespace,
    *,
    trajectory_seed: int,
) -> dict[str, np.ndarray | float | int]:
    last_error: Exception | None = None
    for attempt in range(args.plan_retry_attempts):
        seed = trajectory_seed + attempt
        try:
            task_path, sample_times, task_knots = sample_chained_task_path(planner, args, seed=seed)
            q_path = planner.plan_joint_path(task_path)
            q_path, task_path, sample_times = densify_for_joint_step_limit(
                q_path,
                task_path,
                sample_times,
                max_joint_step=np.deg2rad(args.max_control_joint_step_deg),
            )
            sample_times, timing_scale = retime_for_joint_limits(q_path, sample_times, args)
            validate_collision(q_path, sample_times, args, label=f"trajectory seed {seed}")
            print_plan_summary(q_path, task_path, sample_times, task_knots, timing_scale, seed)
            return {
                "q_path": q_path,
                "task_path": task_path,
                "sample_times": sample_times,
                "task_knots": task_knots,
                "selected_seed": int(seed),
                "timing_scale": float(timing_scale),
            }
        except RuntimeError as error:
            last_error = error
            print(f"Rejected chained trajectory seed {seed}: {error}")
    raise RuntimeError(f"Could not plan a feasible chained trajectory after {args.plan_retry_attempts} attempts.") from last_error


def print_plan_summary(
    q_path: np.ndarray,
    task_path: np.ndarray,
    sample_times: np.ndarray,
    task_knots: np.ndarray,
    timing_scale: float,
    seed: int,
) -> None:
    max_step, max_speed, avg_speed, max_accel = joint_path_speed_accel(q_path, sample_times)
    delta_deg = np.rad2deg(np.max(np.abs(q_path - q_path[0]), axis=0))
    task_delta = task_path - task_path[0]
    summary = {
        "selected_seed": int(seed),
        "duration_s": float(sample_times[-1]),
        "samples": int(sample_times.shape[0]),
        "timing_scale": float(timing_scale),
        "max_joint_step_deg": float(np.rad2deg(max_step)),
        "max_joint_speed_deg_s": float(np.rad2deg(max_speed)),
        "avg_joint_speed_deg_s": float(np.rad2deg(avg_speed)),
        "max_joint_accel_deg_s2": float(np.rad2deg(max_accel)),
        "arm0_max_total_joint_delta_deg": float(np.max(delta_deg[:7])),
        "arm1_max_total_joint_delta_deg": float(np.max(delta_deg[7:])),
        "task_delta_min": np.min(task_delta, axis=0).round(5).tolist(),
        "task_delta_max": np.max(task_delta, axis=0).round(5).tolist(),
        "task_start": task_path[0].round(5).tolist(),
        "task_end": task_path[-1].round(5).tolist(),
        "num_task_knots": int(task_knots.shape[0]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def collect_trajectory(
    robot,
    planner: HomeAnchoredTaskPlanner,
    args: argparse.Namespace,
    *,
    trajectory_seed: int,
    camera=None,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    plan = prepare_plan(planner, args, trajectory_seed=trajectory_seed)
    q_path = np.asarray(plan["q_path"], dtype=np.float64)
    task_path = np.asarray(plan["task_path"], dtype=np.float64)
    sample_times = np.asarray(plan["sample_times"], dtype=np.float64)

    robot.prepare_joint_path_start(q_path[0])
    print(f"Starting chained trajectory: duration={sample_times[-1]:.2f}s, samples={sample_times.shape[0]}")
    first_sample_time = time.monotonic()
    first = planner.step_info(robot, task_path[0], elapsed_time=0.0)
    first_state_time = time.monotonic()
    first["sample_monotonic_time"] = np.asarray([first_sample_time], dtype=np.float64)
    first["state_monotonic_time"] = np.asarray([first_state_time], dtype=np.float64)
    if camera is not None:
        frame = camera.read_frame()
        first["pixels"] = np.asarray(frame["pixels"], dtype=np.uint8)
        first["camera_frame_time"] = np.asarray([float(frame["frame_time"])], dtype=np.float64)
        first["camera_receive_time"] = np.asarray([float(frame["receive_time"])], dtype=np.float64)
        first["camera_header_time"] = np.asarray([float(frame["header_time"])], dtype=np.float64)
        first["camera_frame_index"] = np.asarray([int(frame["frame_index"])], dtype=np.int64)
    lists: dict[str, list[np.ndarray]] = {key: [value] for key, value in first.items()}

    def sample(sample_index: int, elapsed_time: float, _command: np.ndarray) -> None:
        sample_monotonic_time = time.monotonic()
        info = planner.step_info(robot, task_path[sample_index], elapsed_time=elapsed_time)
        state_monotonic_time = time.monotonic()
        info["sample_monotonic_time"] = np.asarray([sample_monotonic_time], dtype=np.float64)
        info["state_monotonic_time"] = np.asarray([state_monotonic_time], dtype=np.float64)
        if camera is not None:
            frame = camera.read_frame()
            info["pixels"] = np.asarray(frame["pixels"], dtype=np.uint8)
            info["camera_frame_time"] = np.asarray([float(frame["frame_time"])], dtype=np.float64)
            info["camera_receive_time"] = np.asarray([float(frame["receive_time"])], dtype=np.float64)
            info["camera_header_time"] = np.asarray([float(frame["header_time"])], dtype=np.float64)
            info["camera_frame_index"] = np.asarray([int(frame["frame_index"])], dtype=np.int64)
        for key, value in info.items():
            lists[key].append(value)

    command_task_path_closed_loop(
        robot,
        planner,
        q_path,
        task_path,
        sample_times,
        args,
        sample_callback=sample,
    )
    trajectory = {key: np.stack(values, axis=0) for key, values in lists.items()}
    planned_actions = np.diff(task_path, axis=0).astype(np.float32)
    measured_actions = np.diff(np.asarray(trajectory["measured_task_target"], dtype=np.float32), axis=0).astype(np.float32)
    trajectory["action"] = planned_actions if planned_actions.size else np.zeros((0, ACTION_DIM), dtype=np.float32)
    trajectory["planned_action"] = trajectory["action"].copy()
    trajectory["measured_action"] = measured_actions if measured_actions.size else np.zeros((0, ACTION_DIM), dtype=np.float32)
    trajectory["selected_seed"] = np.asarray([int(plan["selected_seed"])], dtype=np.int64)
    trajectory["timing_scale"] = np.asarray([float(plan["timing_scale"])], dtype=np.float32)
    return trajectory, True, False


def run_plan_only(args: argparse.Namespace) -> None:
    planner = HomeAnchoredTaskPlanner(args)
    prepare_plan(planner, args, trajectory_seed=args.seed)
    print("Plan-only preflight passed. No robot command was sent and no HDF5 file was written.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.plan_only:
        run_plan_only(args)
        return

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    shard, output_path, async_video_path, async_metadata_path, async_h5_path = resolve_shard_paths(
        outdir,
        args.output_name,
        shard_id=args.shard_id,
    )
    if not args.overwrite:
        existing_outputs = [
            path
            for path in (output_path, async_video_path, async_metadata_path, async_h5_path)
            if path.exists()
        ]
        if existing_outputs:
            raise FileExistsError(
                "Refusing to overwrite existing shard file(s): "
                + ", ".join(str(path) for path in existing_outputs)
                + ". Pass --overwrite to replace them."
            )

    compression = None if args.compression == "none" else args.compression
    planner = HomeAnchoredTaskPlanner(args)
    robot = make_robot(args)
    camera = make_camera(args)
    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0
    async_summary = None

    try:
        robot.connect()
        if camera is not None:
            camera.connect()
            camera.start_async_recording(
                str(async_video_path),
                str(async_metadata_path),
                str(async_h5_path),
                "lzf",
            )
        move_robot_to_planner_home(robot, planner, args)
        print("Robot moved to home position.")
        return
        sample_step_info = planner.step_info(robot, planner.home_task, elapsed_time=0.0)
        sample_frame_packet = camera.read_frame() if camera is not None else None
        sample_frame = None if sample_frame_packet is None else np.asarray(sample_frame_packet["pixels"], dtype=np.uint8)
        obs_dim = int(sample_step_info["observation"].shape[0])
        qpos_dim = int(sample_step_info["qpos"].shape[0])
        qvel_dim = int(sample_step_info["qvel"].shape[0])
        control_dim = int(sample_step_info["control"].shape[0])

        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "stable_worldmodel_hdf5"
            h5.attrs["source"] = "rope/data/rope_real_chained_data_gen_cl.py"
            h5.attrs["hardware"] = True
            h5.attrs["shard"] = shard
            h5.attrs["robot_backend"] = args.robot_backend
            h5.attrs["arm_mapping"] = "kuka-arm0-arm1"
            h5.attrs["trajectory_type"] = "home_anchored_chained_cubic_splines"
            h5.attrs["seed"] = args.seed
            h5.attrs["num_splines"] = args.num_splines
            h5.attrs["requested_segment_duration"] = args.segment_duration
            h5.attrs["control_timestep"] = args.control_timestep
            h5.attrs["home_move_duration"] = args.home_move_duration
            h5.attrs["midpoint_inflation_scale"] = args.midpoint_inflation_scale
            h5.attrs["min_task_knot_distance"] = args.min_task_knot_distance
            h5.attrs["drake_publish_period"] = args.drake_publish_period
            h5.attrs["controller"] = "task_space_closed_loop_ik"
            h5.attrs["task_cl_kp"] = json.dumps([float(value) for value in args.task_cl_kp])
            h5.attrs["task_cl_max_correction_m"] = json.dumps(
                [float(value) for value in args.task_cl_max_correction_m]
            )
            h5.attrs["max_real_joint_speed_deg_s"] = args.max_real_joint_speed_deg_s
            h5.attrs["max_real_joint_accel_deg_s2"] = args.max_real_joint_accel_deg_s2
            h5.attrs["collision_guard_enabled"] = not args.disable_collision_guard
            h5.attrs["arm_arm_min_distance"] = args.arm_arm_min_distance
            h5.attrs["camera_enabled"] = args.enable_camera
            h5.attrs["camera_backend"] = args.camera_backend
            h5.attrs["camera_index"] = args.camera_index
            h5.attrs["camera_device"] = "" if args.camera_device is None else args.camera_device
            h5.attrs["camera_topic"] = "" if args.camera_topic is None else args.camera_topic
            h5.attrs["camera_transport"] = args.camera_transport
            h5.attrs["async_camera_backup_video"] = str(async_video_path) if camera is not None else ""
            h5.attrs["async_camera_backup_metadata"] = str(async_metadata_path) if camera is not None else ""
            h5.attrs["async_camera_backup_h5"] = str(async_h5_path) if camera is not None else ""
            h5.attrs["camera_resolution"] = (
                json.dumps([args.camera_height, args.camera_width]) if args.enable_camera else json.dumps([])
            )
            h5.attrs["camera_crop_center_x"] = args.camera_crop_center_x
            h5.attrs["camera_crop_center_y"] = args.camera_crop_center_y
            h5.attrs["camera_crop_zoom"] = args.camera_crop_zoom
            h5.attrs["camera_capture_resolution"] = json.dumps(
                [args.camera_capture_height, args.camera_capture_width]
            )
            h5.attrs["compression"] = args.compression
            h5.attrs["observation_dim"] = obs_dim
            h5.attrs["action_dim"] = ACTION_DIM
            h5.attrs["qpos_dim"] = qpos_dim
            h5.attrs["qvel_dim"] = qvel_dim
            h5.attrs["control_dim"] = control_dim
            h5.attrs["time_semantics"] = "planned_sample_time_seconds_from_trajectory_start"
            h5.attrs["sample_monotonic_time_semantics"] = "collector callback entry time from time.monotonic()"
            h5.attrs["state_monotonic_time_semantics"] = "state snapshot completion time from time.monotonic()"
            h5.attrs["camera_frame_time_semantics"] = "camera source timestamp when available, else frame grab time"
            h5.attrs["camera_receive_time_semantics"] = "camera callback or read receipt time from time.monotonic()"
            h5.attrs["camera_header_time_semantics"] = "camera message header stamp in seconds when available"
            h5.attrs["camera_frame_index_semantics"] = "camera backend frame counter"
            h5.attrs["action_semantics"] = "planned_task_delta"
            h5.attrs["planned_action_semantics"] = "task_path_diff"
            h5.attrs["measured_action_semantics"] = "measured_task_target_diff"
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

            ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
            ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
            reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
            seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
            selected_seed_ds = create_resizable_dataset(h5, "selected_episode_seed", (), np.int64, chunks=True)
            terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
            truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
            timing_scale_ds = create_resizable_dataset(h5, "timing_scale", (), np.float32, chunks=True)
            pixels_ds = (
                create_resizable_dataset(
                    h5,
                    "pixels",
                    sample_frame.shape,
                    np.uint8,
                    compression=compression,
                    chunks=(1, *sample_frame.shape),
                )
                if sample_frame is not None
                else None
            )
            action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
            planned_action_ds = create_resizable_dataset(h5, "planned_action", (ACTION_DIM,), np.float32, chunks=True)
            measured_action_ds = create_resizable_dataset(h5, "measured_action", (ACTION_DIM,), np.float32, chunks=True)
            obs_ds = create_resizable_dataset(h5, "observation", (obs_dim,), np.float32, chunks=True)
            task_target_ds = create_resizable_dataset(h5, "task_target", (3,), np.float32, chunks=True)
            measured_task_target_ds = create_resizable_dataset(h5, "measured_task_target", (3,), np.float32, chunks=True)
            qpos_ds = create_resizable_dataset(h5, "qpos", (qpos_dim,), np.float32, chunks=True)
            qvel_ds = create_resizable_dataset(h5, "qvel", (qvel_dim,), np.float32, chunks=True)
            control_ds = create_resizable_dataset(h5, "control", (control_dim,), np.float32, chunks=True)
            left_attachment_pos_ds = create_resizable_dataset(h5, "left_attachment_pos", (3,), np.float32, chunks=True)
            right_attachment_pos_ds = create_resizable_dataset(h5, "right_attachment_pos", (3,), np.float32, chunks=True)
            rope_length_ds = create_resizable_dataset(h5, "rope_length", (1,), np.float32, chunks=True)
            time_ds = create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
            sample_monotonic_time_ds = create_resizable_dataset(h5, "sample_monotonic_time", (1,), np.float64, chunks=True)
            state_monotonic_time_ds = create_resizable_dataset(h5, "state_monotonic_time", (1,), np.float64, chunks=True)
            camera_frame_time_ds = (
                create_resizable_dataset(h5, "camera_frame_time", (1,), np.float64, chunks=True)
                if sample_frame is not None
                else None
            )
            camera_receive_time_ds = (
                create_resizable_dataset(h5, "camera_receive_time", (1,), np.float64, chunks=True)
                if sample_frame is not None
                else None
            )
            camera_header_time_ds = (
                create_resizable_dataset(h5, "camera_header_time", (1,), np.float64, chunks=True)
                if sample_frame is not None
                else None
            )
            camera_frame_index_ds = (
                create_resizable_dataset(h5, "camera_frame_index", (1,), np.int64, chunks=True)
                if sample_frame is not None
                else None
            )
            episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
            step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

            progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
            progress_desc = "Collecting chained transitions" if args.target_transitions is not None else "Collecting chained trajectories"
            progress_unit = "step" if args.target_transitions is not None else "traj"

            with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
                while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                    move_robot_to_planner_home(robot, planner, args)
                    trajectory_seed = args.seed + seed_offset
                    seed_offset += args.plan_retry_attempts
                    trajectory, terminated, truncated = collect_trajectory(
                        robot,
                        planner,
                        args,
                        trajectory_seed=trajectory_seed,
                        camera=camera,
                    )
                    num_actions = int(trajectory["action"].shape[0])
                    if num_actions < args.min_steps:
                        skipped_short += 1
                        continue

                    episode_idx = len(step_counts)
                    padded_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                    padded_actions[:-1] = trajectory["action"]
                    padded_actions[-1] = np.nan
                    padded_planned_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                    padded_planned_actions[:-1] = trajectory["planned_action"]
                    padded_planned_actions[-1] = np.nan
                    padded_measured_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                    padded_measured_actions[:-1] = trajectory["measured_action"]
                    padded_measured_actions[-1] = np.nan

                    if pixels_ds is not None:
                        offset, _ = append_rows(pixels_ds, trajectory["pixels"])
                    else:
                        offset = int(obs_ds.shape[0])
                    append_rows(obs_ds, trajectory["observation"])
                    append_rows(action_ds, padded_actions)
                    append_rows(planned_action_ds, padded_planned_actions)
                    append_rows(measured_action_ds, padded_measured_actions)
                    append_rows(task_target_ds, trajectory["task_target"])
                    append_rows(measured_task_target_ds, trajectory["measured_task_target"])
                    append_rows(qpos_ds, trajectory["qpos"])
                    append_rows(qvel_ds, trajectory["qvel"])
                    append_rows(control_ds, trajectory["control"])
                    append_rows(left_attachment_pos_ds, trajectory["left_attachment_pos"])
                    append_rows(right_attachment_pos_ds, trajectory["right_attachment_pos"])
                    append_rows(rope_length_ds, trajectory["rope_length"])
                    append_rows(time_ds, trajectory["time"])
                    append_rows(sample_monotonic_time_ds, trajectory["sample_monotonic_time"])
                    append_rows(state_monotonic_time_ds, trajectory["state_monotonic_time"])
                    if camera_frame_time_ds is not None:
                        append_rows(camera_frame_time_ds, trajectory["camera_frame_time"])
                        append_rows(camera_receive_time_ds, trajectory["camera_receive_time"])
                        append_rows(camera_header_time_ds, trajectory["camera_header_time"])
                        append_rows(camera_frame_index_ds, trajectory["camera_frame_index"])
                    append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
                    append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
                    append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
                    append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                    append_rows(reward_ds, np.asarray([0.0], dtype=np.float32))
                    append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                    append_rows(selected_seed_ds, trajectory["selected_seed"])
                    append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                    append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))
                    append_rows(timing_scale_ds, trajectory["timing_scale"])

                    rewards.append(0.0)
                    step_counts.append(num_actions)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    progress.update(num_actions if args.target_transitions is not None else 1)
                    progress.set_postfix(
                        episodes=len(step_counts),
                        transitions=int(np.sum(step_counts, dtype=np.int64)),
                    )

            h5.attrs["num_episodes"] = len(step_counts)
            h5.attrs["total_frames"] = int(pixels_ds.shape[0] if pixels_ds is not None else obs_ds.shape[0])
            h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
            h5.attrs["skipped_short_episodes"] = skipped_short
            h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
            h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
            h5.attrs["terminated_fraction"] = float(np.mean(terminated_flags)) if terminated_flags else 0.0
            h5.attrs["truncated_fraction"] = float(np.mean(truncated_flags)) if truncated_flags else 0.0
            async_summary = None if camera is None else camera.stop_async_recording()
            h5.attrs["async_camera_full_resolution_frames"] = 0
    finally:
        if camera is not None and async_summary is None:
            async_summary = camera.stop_async_recording()
        robot.stop()
        if camera is not None:
            camera.close()
        robot.close()

    summary = {
        "output_path": str(output_path),
        "shard": shard,
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "control_timestep": args.control_timestep,
        "num_splines": args.num_splines,
        "requested_segment_duration": args.segment_duration,
    }
    if async_summary is not None:
        summary["async_camera_backup"] = async_summary
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
