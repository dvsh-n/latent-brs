#!/usr/bin/env python3
"""Collect real-rope data with synchronous line-segment actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from tqdm.auto import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rope.data_real.iiwa_cartesian_ik import SingleIiwaPositionIK
from rope.data_real.collision_guard import MujocoArmCollisionGuard
from rope.real.camera_backend import OpenCVCamera
from rope.real.drake_lcm_backend import DrakeLCMBimanualRobotBackend
from rope.shared.lab_env import LabEnv


DEFAULT_OUTDIR = "rope/data_real/real_data"
DEFAULT_OUTPUT_NAME = "rope_real.h5"
CONFIG_FORBIDDEN_KEYS = {"i_understand_this_moves_real_robots"}
CONFIG_KEY_ALIASES = {
    "num_trajectories": "num_waypoints",
    "samples_per_segment": "samples_per_waypoint",
}
SHARD_SUFFIX_RE = re.compile(r"_shard(\d+)$")

ACTION_DIM = 3
TASK_REACH_BOUNDS = (-0.08, 0.28)
TASK_HEIGHT_BOUNDS = (1.18, 1.30)
TASK_WIDTH_BOUNDS = (0.22, 0.55)
HARDWARE_HOME_Q0_DEG = np.array([-65.0, 3.0, -167.0, 113.0, 8.0, 5.0, 0.0], dtype=np.float64)
HARDWARE_HOME_Q1_DEG = np.array([65.0, 3.0, -13.0, -113.0, -8.0, -5.0, 0.0], dtype=np.float64)
ARM1_EE_Z_ROTATION_OFFSET_RAD = 0.0
ARM1_EE_POSITION_OFFSET_B = np.array([0.0, 0.0, 0.03], dtype=np.float64)


@dataclass(frozen=True)
class TaskBounds:
    reach: tuple[float, float] = TASK_REACH_BOUNDS
    height: tuple[float, float] = TASK_HEIGHT_BOUNDS
    width: tuple[float, float] = TASK_WIDTH_BOUNDS

    def lower(self) -> np.ndarray:
        return np.array([self.reach[0], self.height[0], self.width[0]], dtype=np.float64)

    def upper(self) -> np.ndarray:
        return np.array([self.reach[1], self.height[1], self.width[1]], dtype=np.float64)

    def center(self) -> np.ndarray:
        return 0.5 * (self.lower() + self.upper())

    def clip(self, values: np.ndarray | list[float]) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=np.float64), self.lower(), self.upper())


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
    values = np.asarray(values)
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def load_yaml_config(path: Path) -> dict[str, object]:
    with path.expanduser().open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to contain a mapping, got {type(config).__name__}.")
    normalized: dict[str, object] = {}
    for key, value in config.items():
        normalized_key = str(key).replace("-", "_")
        normalized[CONFIG_KEY_ALIASES.get(normalized_key, normalized_key)] = value
    return normalized


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
    parser.add_argument("--num-waypoints", "--num-trajectories", dest="num_waypoints", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument(
        "--samples-per-waypoint",
        "--samples-per-segment",
        dest="samples_per_waypoint",
        type=int,
        default=8,
    )
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
    parser.add_argument("--home-move-duration", type=float, default=12.0)
    parser.add_argument("--start-move-duration", type=float, default=None)
    parser.add_argument("--max-home-x-gap-m", type=float, default=0.008)
    parser.add_argument("--max-home-z-gap-m", type=float, default=0.004)
    parser.add_argument("--trajectory-hold-duration", type=float, default=0.25)
    parser.add_argument("--trajectory-start-blend-duration", type=float, default=4.0)
    parser.add_argument("--trajectory-start-settle-duration", type=float, default=0.25)
    parser.add_argument("--trajectory-start-ready-tolerance-deg", type=float, default=0.15)
    parser.add_argument("--trajectory-start-ready-velocity-deg-s", type=float, default=1.0)
    parser.add_argument("--trajectory-start-ready-timeout", type=float, default=15.0)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-control-joint-step-deg", type=float, default=9.0)
    parser.add_argument("--max-reset-joint-move-deg", type=float, default=90.0)
    parser.add_argument("--max-command-measured-gap-deg", type=float, default=0.15)
    parser.add_argument("--max-real-joint-speed-deg-s", type=float, default=15.0)
    parser.add_argument("--max-task-speed-m-s", type=float, default=0.04)
    parser.add_argument("--min-action-duration", type=float, default=0.0)
    parser.add_argument("--settle-tolerance-deg", type=float, default=0.25)
    parser.add_argument("--settle-velocity-deg-s", type=float, default=1.0)
    parser.add_argument("--settle-timeout", type=float, default=5.0)
    parser.add_argument("--settle-poll-period", type=float, default=0.05)
    parser.add_argument("--ik-position-tol", type=float, default=0.001)
    parser.add_argument("--ik-max-joint-step-deg", type=float, default=2.0)
    parser.add_argument("--ik-joint7-min-deg", type=float, default=None)
    parser.add_argument("--ik-joint7-max-deg", type=float, default=None)
    parser.add_argument("--disable-collision-guard", action="store_true")
    parser.add_argument("--arm-arm-min-distance", type=float, default=0.06)
    parser.add_argument("--collision-control-samples", type=int, default=5)
    parser.add_argument("--collision-reset-samples", type=int, default=25)
    parser.add_argument("--enable-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-device", default=None)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-crop-center-x", type=float, default=0.525)
    parser.add_argument("--camera-crop-center-y", type=float, default=0.5)
    parser.add_argument("--camera-crop-zoom", type=float, default=0.9)
    parser.add_argument("--camera-capture-width", type=int, default=None)
    parser.add_argument("--camera-capture-height", type=int, default=None)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--camera-drop-frames-after-motion", type=int, default=3)
    parser.add_argument("--camera-settle-delay", type=float, default=0.05)
    parser.add_argument("--async-backup", action="store_true")
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
    if args.num_waypoints < 1:
        raise ValueError("--num-waypoints must be positive.")
    if args.shard_id is not None and args.shard_id < 0:
        raise ValueError("--shard-id must be non-negative.")
    if args.samples_per_waypoint < 1:
        raise ValueError("--samples-per-waypoint must be positive.")
    if args.min_task_knot_distance < 0.0:
        raise ValueError("--min-task-knot-distance cannot be negative.")
    if len(args.task_reach_bounds) != 2 or args.task_reach_bounds[0] >= args.task_reach_bounds[1]:
        raise ValueError("--task-reach-bounds must provide two increasing values.")
    if len(args.task_height_bounds) != 2 or args.task_height_bounds[0] >= args.task_height_bounds[1]:
        raise ValueError("--task-height-bounds must provide two increasing values.")
    if len(args.task_width_bounds) != 2 or args.task_width_bounds[0] >= args.task_width_bounds[1]:
        raise ValueError("--task-width-bounds must provide two increasing values.")
    if args.home_move_duration <= 0.5:
        raise ValueError("--home-move-duration must be greater than 0.5s.")
    if args.start_move_duration is not None and args.start_move_duration <= 0.0:
        raise ValueError("--start-move-duration must be positive when provided.")
    if args.max_real_joint_speed_deg_s <= 0.0:
        raise ValueError("--max-real-joint-speed-deg-s must be positive.")
    if args.max_task_speed_m_s <= 0.0:
        raise ValueError("--max-task-speed-m-s must be positive.")
    if args.min_action_duration < 0.0:
        raise ValueError("--min-action-duration cannot be negative.")
    if args.settle_tolerance_deg < 0.0:
        raise ValueError("--settle-tolerance-deg cannot be negative.")
    if args.settle_velocity_deg_s < 0.0:
        raise ValueError("--settle-velocity-deg-s cannot be negative.")
    if args.settle_timeout < 0.0:
        raise ValueError("--settle-timeout cannot be negative.")
    if args.settle_poll_period <= 0.0:
        raise ValueError("--settle-poll-period must be positive.")
    if args.ik_position_tol <= 0.0:
        raise ValueError("--ik-position-tol must be positive.")
    if args.ik_max_joint_step_deg <= 0.0:
        raise ValueError("--ik-max-joint-step-deg must be positive.")
    if (args.ik_joint7_min_deg is None) != (args.ik_joint7_max_deg is None):
        raise ValueError("--ik-joint7-min-deg and --ik-joint7-max-deg must be provided together.")
    if (
        args.ik_joint7_min_deg is not None
        and args.ik_joint7_max_deg is not None
        and args.ik_joint7_min_deg >= args.ik_joint7_max_deg
    ):
        raise ValueError("--ik-joint7-min-deg must be less than --ik-joint7-max-deg.")
    if args.arm_mapping != "robot0-left":
        raise ValueError("This collector follows the KUKA arm0/arm1 convention; use --arm-mapping robot0-left.")
    if args.drake_publish_period <= 0.0:
        raise ValueError("--drake-publish-period must be positive.")
    if len(args.home_q0_deg) != 7 or len(args.home_q1_deg) != 7:
        raise ValueError("--home-q0-deg and --home-q1-deg must each provide 7 joint values.")
    if args.camera_drop_frames_after_motion < 0:
        raise ValueError("--camera-drop-frames-after-motion cannot be negative.")
    if args.camera_settle_delay < 0.0:
        raise ValueError("--camera-settle-delay cannot be negative.")


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
    return (
        shard,
        outdir / f"{shard_prefix}.h5",
        outdir / f"{shard_prefix}_async_camera.mp4",
        outdir / f"{shard_prefix}_async_camera.csv",
        outdir / f"{shard_prefix}_async_camera.h5",
    )


def requested_transitions(args: argparse.Namespace) -> int:
    return int(args.num_waypoints) * int(args.samples_per_waypoint)


def make_robot(args: argparse.Namespace) -> DrakeLCMBimanualRobotBackend:
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


def make_camera(args: argparse.Namespace) -> OpenCVCamera | None:
    if not args.enable_camera:
        return None
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


class HomeAnchoredLinePlanner:
    def __init__(self, args: argparse.Namespace) -> None:
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
        self.ik_joint_lower_bounds = None
        self.ik_joint_upper_bounds = None
        if args.ik_joint7_min_deg is not None and args.ik_joint7_max_deg is not None:
            self.ik_joint_lower_bounds = np.full(7, -np.inf, dtype=np.float64)
            self.ik_joint_upper_bounds = np.full(7, np.inf, dtype=np.float64)
            self.ik_joint_lower_bounds[6] = np.deg2rad(args.ik_joint7_min_deg)
            self.ik_joint_upper_bounds[6] = np.deg2rad(args.ik_joint7_max_deg)
        q0_home, q1_home = self._hardware_home_goal()
        self.hardware_home_qpos = np.concatenate([q0_home, q1_home]).astype(np.float64)
        self.dataset_home_qpos = self.hardware_home_qpos.copy()
        self.p0_home = self.ik0.fk_position(q0_home)
        self.p1_home = self.ik1.fk_position(q1_home)
        self.nominal_home_attachment_delta_xz = self._attachment_delta_xz(self.hardware_home_qpos)

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
            f"home_x_error={home_error_xz[0]:.4f}m, home_z_error={home_error_xz[1]:.4f}m"
        )
        if (
            home_error_xz[0] > self.args.max_home_x_gap_m
            or home_error_xz[1] > self.args.max_home_z_gap_m
        ):
            raise RuntimeError(
                "Measured home pose does not match the configured collection home closely enough: "
                f"home_x_error={home_error_xz[0]:.4f}m (limit {self.args.max_home_x_gap_m:.4f}m), "
                f"home_z_error={home_error_xz[1]:.4f}m (limit {self.args.max_home_z_gap_m:.4f}m)."
            )

    def plan_joint_path(self, task_path: np.ndarray, *, progress: tqdm | None = None) -> np.ndarray:
        task_path = np.asarray(task_path, dtype=np.float64)
        if task_path.ndim != 2 or task_path.shape[1] != 3:
            raise ValueError(f"Expected task_path with shape (T, 3), got {task_path.shape}.")
        q0_prev = self.dataset_home_qpos[:7].copy()
        q1_prev = self.dataset_home_qpos[7:].copy()
        q_path = []
        for index, task_state in enumerate(task_path):
            p0_goal, p1_goal = self.attachment_goals_for_task(task_state)
            q0_prev = self._solve_arm(self.ik0, p0_goal, q0_prev, f"arm0 waypoint {index}")
            q1_prev = self._solve_arm(self.ik1, p1_goal, q1_prev, f"arm1 waypoint {index}")
            q_path.append(np.concatenate([q0_prev, q1_prev]))
            if progress is not None:
                progress.update(1)
                progress.set_postfix(stage="ik", waypoint=index)
        return np.stack(q_path, axis=0)

    def plan_joint_step_from_current(self, current_qpos: np.ndarray, target_task: np.ndarray) -> np.ndarray:
        current_qpos = np.asarray(current_qpos, dtype=np.float64).reshape(14)
        p0_goal, p1_goal = self.attachment_goals_for_task(target_task)
        q0 = self._solve_arm(self.ik0, p0_goal, current_qpos[:7], "arm0 mpc target")
        q1 = self._solve_arm(self.ik1, p1_goal, current_qpos[7:], "arm1 mpc target")
        return np.concatenate([q0, q1]).astype(np.float64)

    def attachment_goals_for_task(self, task_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        task_state = np.asarray(task_state, dtype=np.float64).reshape(3)
        delta = task_state - self.home_task
        p0_goal = self.p0_home + np.array([delta[0], 0.5 * delta[2], delta[1]], dtype=np.float64)
        p1_goal = self.p1_home + np.array([delta[0], -0.5 * delta[2], delta[1]], dtype=np.float64)
        return p0_goal, p1_goal

    def step_info(self, robot: DrakeLCMBimanualRobotBackend, task_target: np.ndarray) -> dict[str, np.ndarray]:
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
        }

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
        return self.task_bounds.clip(measured)

    def _hardware_home_goal(self) -> tuple[np.ndarray, np.ndarray]:
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
            joint_lower_bounds=self.ik_joint_lower_bounds,
            joint_upper_bounds=self.ik_joint_upper_bounds,
        )
        if q_sol is None:
            raise RuntimeError(f"{label} IK failed for target {np.round(target_pos, 4)}: {info}")
        return q_sol

    def _attachment_delta_xz(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(14)
        p0 = self.ik0.fk_position(qpos[:7])
        p1 = self.ik1.fk_position(qpos[7:])
        return np.array([p0[0] - p1[0], p0[2] - p1[2]], dtype=np.float64)


def sample_task_point(rng: np.random.Generator, bounds: TaskBounds) -> np.ndarray:
    return rng.uniform(bounds.lower(), bounds.upper())


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


def line_targets(start: np.ndarray, goal: np.ndarray, num_actions: int) -> list[np.ndarray]:
    return [
        (1.0 - alpha) * start + alpha * goal
        for alpha in np.linspace(1.0 / num_actions, 1.0, num_actions, dtype=np.float64)
    ]


def sample_chained_task_path(
    planner: HomeAnchoredLinePlanner,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    previous = planner.home_task.copy()
    knots = [previous]
    points = [previous]
    for _ in range(args.num_waypoints):
        waypoint = sample_nontrivial_task_point(
            rng,
            planner.task_bounds,
            previous,
            args.min_task_knot_distance,
        )
        points.extend(line_targets(previous, waypoint, args.samples_per_waypoint))
        knots.append(waypoint)
        previous = waypoint
    return np.stack(points, axis=0), np.stack(knots, axis=0)


def action_durations(q_path: np.ndarray, task_path: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    q_path = np.asarray(q_path, dtype=np.float64)
    task_path = np.asarray(task_path, dtype=np.float64)
    max_joint_speed = np.deg2rad(args.max_real_joint_speed_deg_s)
    dq = np.max(np.abs(np.diff(q_path, axis=0)), axis=1)
    dx = np.linalg.norm(np.diff(task_path, axis=0), axis=1)
    durations = np.maximum(dq / max(max_joint_speed, 1e-9), dx / max(args.max_task_speed_m_s, 1e-9))
    durations = np.maximum(durations, float(args.min_action_duration))
    durations = np.maximum(durations, float(args.drake_publish_period))
    return durations.astype(np.float64)


def validate_collision(
    q_path: np.ndarray,
    durations: np.ndarray,
    args: argparse.Namespace,
    *,
    label: str,
    progress: tqdm | None = None,
) -> None:
    if args.disable_collision_guard:
        print("Collision guard skipped because --disable-collision-guard is set.")
        if progress is not None:
            progress.update(len(durations))
            progress.set_postfix(stage="collision skipped")
        return
    guard = MujocoArmCollisionGuard(
        LabEnv(),
        min_arm_arm_distance=args.arm_arm_min_distance,
        control_path_samples=args.collision_control_samples,
        reset_path_samples=args.collision_reset_samples,
    )
    for index, duration in enumerate(durations, start=1):
        guard.validate_path(
            q_path[index - 1],
            q_path[index],
            duration=float(duration),
            label=f"{label} action {index - 1}->{index}",
        )
        if progress is not None:
            progress.update(1)
            progress.set_postfix(stage="collision", action=f"{index}/{len(durations)}")
    print(f"Collision guard passed for {q_path.shape[0]} waypoints.")


def start_reset_duration(planner: HomeAnchoredLinePlanner, q_start: np.ndarray, args: argparse.Namespace) -> float:
    if args.start_move_duration is not None:
        return float(args.start_move_duration)
    home_to_start = float(np.max(np.abs(np.asarray(q_start, dtype=np.float64).reshape(14) - planner.dataset_home_qpos)))
    return max(home_to_start / np.deg2rad(args.max_real_joint_speed_deg_s), args.drake_publish_period)


def validate_start_reset_collision(
    planner: HomeAnchoredLinePlanner,
    q_start: np.ndarray,
    args: argparse.Namespace,
    *,
    label: str,
) -> None:
    if args.disable_collision_guard:
        return
    duration = start_reset_duration(planner, q_start, args)
    guard = MujocoArmCollisionGuard(
        LabEnv(),
        min_arm_arm_distance=args.arm_arm_min_distance,
        control_path_samples=args.collision_control_samples,
        reset_path_samples=args.collision_reset_samples,
    )
    guard.validate_path(planner.dataset_home_qpos, q_start, duration=duration, label=f"{label} home->sampled start")


def print_plan_summary(
    q_path: np.ndarray,
    task_path: np.ndarray,
    task_knots: np.ndarray,
    durations: np.ndarray,
    seed: int,
) -> None:
    dq = np.max(np.abs(np.diff(q_path, axis=0)), axis=1)
    speeds = dq / np.maximum(durations, 1e-9)
    summary = {
        "selected_seed": int(seed),
        "actions": int(durations.shape[0]),
        "num_waypoints": int(task_knots.shape[0] - 1),
        "samples_per_waypoint": int(durations.shape[0] // max(task_knots.shape[0] - 1, 1)),
        "duration_s": float(np.sum(durations)),
        "max_action_duration_s": float(np.max(durations)),
        "max_joint_step_deg": float(np.rad2deg(np.max(dq))),
        "max_joint_speed_deg_s": float(np.rad2deg(np.max(speeds))),
        "task_start": task_path[0].round(5).tolist(),
        "task_goal": task_path[-1].round(5).tolist(),
        "action_delta_min": np.min(np.diff(task_path, axis=0), axis=0).round(5).tolist(),
        "action_delta_max": np.max(np.diff(task_path, axis=0), axis=0).round(5).tolist(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def prepare_plan(
    planner: HomeAnchoredLinePlanner,
    args: argparse.Namespace,
    *,
    trajectory_seed: int,
) -> dict[str, np.ndarray | int]:
    last_error: Exception | None = None
    for attempt in range(args.plan_retry_attempts):
        seed = trajectory_seed + attempt
        try:
            task_path, task_knots = sample_chained_task_path(planner, args, seed=seed)
            planning_total = int(task_path.shape[0] + max(task_path.shape[0] - 1, 0))
            with tqdm(
                total=planning_total,
                desc=f"Planning trajectory attempt {attempt + 1}/{args.plan_retry_attempts}",
                unit="check",
                leave=False,
            ) as planning_progress:
                q_path = planner.plan_joint_path(task_path, progress=planning_progress)
                durations = action_durations(q_path, task_path, args)
                validate_start_reset_collision(planner, q_path[0], args, label=f"trajectory seed {seed}")
                validate_collision(q_path, durations, args, label=f"trajectory seed {seed}", progress=planning_progress)
            print_plan_summary(q_path, task_path, task_knots, durations, seed)
            return {
                "q_path": q_path,
                "task_path": task_path,
                "task_knots": task_knots,
                "durations": durations,
                "selected_seed": int(seed),
            }
        except RuntimeError as error:
            last_error = error
            print(f"Rejected trajectory seed {seed}: {error}")
    raise RuntimeError(f"Could not plan a feasible trajectory after {args.plan_retry_attempts} attempts.") from last_error


def move_robot_to_planner_home(
    robot: DrakeLCMBimanualRobotBackend,
    planner: HomeAnchoredLinePlanner,
    args: argparse.Namespace,
) -> None:
    measured = robot.read_qpos_14()
    max_delta_deg = float(np.rad2deg(np.max(np.abs(planner.hardware_home_qpos - measured))))
    if max_delta_deg <= args.trajectory_start_ready_tolerance_deg:
        print(f"Robot already at configured home: max_delta={max_delta_deg:.2f}deg")
        robot.prepare_joint_path_start(planner.hardware_home_qpos)
    else:
        max_segment_delta = 0.95 * np.deg2rad(args.max_reset_joint_move_deg)
        num_segments = max(1, int(np.ceil(np.max(np.abs(planner.hardware_home_qpos - measured)) / max_segment_delta)))
        segment_duration = args.home_move_duration / num_segments
        if num_segments == 1:
            print(
                f"Moving robot to configured home: duration={args.home_move_duration:.2f}s, "
                f"max_delta={max_delta_deg:.2f}deg"
            )
        else:
            print(
                "Moving robot to configured home in segments: "
                f"segments={num_segments}, total_duration={args.home_move_duration:.2f}s, "
                f"max_delta={max_delta_deg:.2f}deg, "
                f"max_segment_delta<={0.95 * args.max_reset_joint_move_deg:.2f}deg"
            )
        for segment in range(1, num_segments + 1):
            alpha = segment / num_segments
            q_segment = measured + alpha * (planner.hardware_home_qpos - measured)
            robot.command_joint_positions(q_segment, duration=segment_duration, blocking=True)
        robot.prepare_joint_path_start(planner.hardware_home_qpos)
    planner.anchor_dataset_home(robot.read_qpos_14())


def wait_until_settled(
    robot: DrakeLCMBimanualRobotBackend,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, float, float, float]:
    target = np.asarray(target, dtype=np.float64).reshape(14)
    position_tol = np.deg2rad(args.settle_tolerance_deg)
    velocity_tol = np.deg2rad(args.settle_velocity_deg_s)
    start = time.monotonic()
    last_err = float("inf")
    last_speed = float("inf")
    while time.monotonic() - start <= args.settle_timeout:
        qpos = robot.read_qpos_14()
        qvel = robot.read_qvel_14()
        last_err = float(np.max(np.abs(qpos - target)))
        last_speed = float(np.max(np.abs(qvel)))
        if last_err <= position_tol and last_speed <= velocity_tol:
            return True, time.monotonic() - start, last_err, last_speed
        robot.command_joint_positions(target, duration=args.settle_poll_period, blocking=True)
    return False, time.monotonic() - start, last_err, last_speed


def capture_fresh_frame(camera: OpenCVCamera | None, args: argparse.Namespace) -> dict[str, np.ndarray | float | int] | None:
    if camera is None:
        return None
    if args.camera_settle_delay > 0.0:
        time.sleep(args.camera_settle_delay)
    frame = None
    for _ in range(args.camera_drop_frames_after_motion + 1):
        frame = camera.read_frame()
    assert frame is not None
    return frame


def collect_trajectory(
    robot: DrakeLCMBimanualRobotBackend,
    planner: HomeAnchoredLinePlanner,
    args: argparse.Namespace,
    *,
    trajectory_seed: int,
    camera: OpenCVCamera | None,
    progress: tqdm | None = None,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    plan = prepare_plan(planner, args, trajectory_seed=trajectory_seed)
    q_path = np.asarray(plan["q_path"], dtype=np.float64)
    task_path = np.asarray(plan["task_path"], dtype=np.float64)
    durations = np.asarray(plan["durations"], dtype=np.float64)
    task_knots = np.asarray(plan["task_knots"], dtype=np.float64)

    start_duration = start_reset_duration(planner, q_path[0], args)

    print(f"Moving to sampled start without recording: duration={start_duration:.2f}s")
    robot.command_joint_positions(q_path[0], duration=float(start_duration), blocking=True)
    robot.prepare_joint_path_start(q_path[0])
    initial_frame = capture_fresh_frame(camera, args)
    initial_camera_time = time.monotonic()

    rows: dict[str, list[np.ndarray]] = {
        "action": [],
        "planned_action": [],
        "measured_action": [],
        "observation": [],
        "task_before": [],
        "task_after": [],
        "task_target": [],
        "measured_task_before": [],
        "measured_task_after": [],
        "q_before": [],
        "q_after": [],
        "q_cmd": [],
        "qpos": [],
        "qvel": [],
        "control": [],
        "left_attachment_pos": [],
        "right_attachment_pos": [],
        "rope_length": [],
        "command_duration": [],
        "settle_duration": [],
        "settle_position_error_deg": [],
        "settle_velocity_deg_s": [],
        "success": [],
        "action_start_monotonic_time": [],
        "action_end_monotonic_time": [],
        "state_monotonic_time": [],
        "camera_capture_monotonic_time": [],
        "camera_frame_time": [],
        "camera_receive_time": [],
        "camera_header_time": [],
        "camera_frame_index": [],
        "pixels": [],
    }
    if initial_frame is not None:
        rows["pixels"].append(np.asarray(initial_frame["pixels"], dtype=np.uint8))
        rows["camera_capture_monotonic_time"].append(np.asarray([initial_camera_time], dtype=np.float64))
        rows["camera_frame_time"].append(np.asarray([float(initial_frame["frame_time"])], dtype=np.float64))
        rows["camera_receive_time"].append(np.asarray([float(initial_frame["receive_time"])], dtype=np.float64))
        rows["camera_header_time"].append(np.asarray([float(initial_frame["header_time"])], dtype=np.float64))
        rows["camera_frame_index"].append(np.asarray([int(initial_frame["frame_index"])], dtype=np.int64))

    for index, duration in enumerate(durations, start=1):
        task_before = task_path[index - 1].astype(np.float32)
        task_after = task_path[index].astype(np.float32)
        q_before = robot.read_qpos_14().astype(np.float32)
        info_before = planner.step_info(robot, task_before)
        action_start = time.monotonic()
        robot.command_joint_positions(q_path[index], duration=float(duration), blocking=True)
        action_end = time.monotonic()
        settled, settle_duration, settle_err, settle_speed = wait_until_settled(robot, q_path[index], args)
        state_time = time.monotonic()
        info_after = planner.step_info(robot, task_after)
        frame = capture_fresh_frame(camera, args)
        camera_time = time.monotonic()

        planned_action = (task_after - task_before).astype(np.float32)
        measured_action = (
            info_after["measured_task_target"] - info_before["measured_task_target"]
        ).astype(np.float32)

        rows["action"].append(planned_action)
        rows["planned_action"].append(planned_action.copy())
        rows["measured_action"].append(measured_action)
        rows["observation"].append(info_after["observation"])
        rows["task_before"].append(task_before)
        rows["task_after"].append(task_after)
        rows["task_target"].append(task_after)
        rows["measured_task_before"].append(info_before["measured_task_target"])
        rows["measured_task_after"].append(info_after["measured_task_target"])
        rows["q_before"].append(q_before)
        rows["q_after"].append(info_after["qpos"])
        rows["q_cmd"].append(q_path[index].astype(np.float32))
        rows["qpos"].append(info_after["qpos"])
        rows["qvel"].append(info_after["qvel"])
        rows["control"].append(info_after["control"])
        rows["left_attachment_pos"].append(info_after["left_attachment_pos"])
        rows["right_attachment_pos"].append(info_after["right_attachment_pos"])
        rows["rope_length"].append(info_after["rope_length"])
        rows["command_duration"].append(np.asarray([duration], dtype=np.float32))
        rows["settle_duration"].append(np.asarray([settle_duration], dtype=np.float32))
        rows["settle_position_error_deg"].append(np.asarray([np.rad2deg(settle_err)], dtype=np.float32))
        rows["settle_velocity_deg_s"].append(np.asarray([np.rad2deg(settle_speed)], dtype=np.float32))
        rows["success"].append(np.asarray([settled], dtype=np.bool_))
        rows["action_start_monotonic_time"].append(np.asarray([action_start], dtype=np.float64))
        rows["action_end_monotonic_time"].append(np.asarray([action_end], dtype=np.float64))
        rows["state_monotonic_time"].append(np.asarray([state_time], dtype=np.float64))
        if frame is not None:
            rows["pixels"].append(np.asarray(frame["pixels"], dtype=np.uint8))
            rows["camera_capture_monotonic_time"].append(np.asarray([camera_time], dtype=np.float64))
            rows["camera_frame_time"].append(np.asarray([float(frame["frame_time"])], dtype=np.float64))
            rows["camera_receive_time"].append(np.asarray([float(frame["receive_time"])], dtype=np.float64))
            rows["camera_header_time"].append(np.asarray([float(frame["header_time"])], dtype=np.float64))
            rows["camera_frame_index"].append(np.asarray([int(frame["frame_index"])], dtype=np.int64))

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                step=f"{index}/{len(durations)}",
                settled=settled,
                err_deg=f"{np.rad2deg(settle_err):.3f}",
            )
        if not settled:
            raise RuntimeError(
                "Robot did not settle after action: "
                f"step={index}, err={np.rad2deg(settle_err):.3f}deg, "
                f"speed={np.rad2deg(settle_speed):.3f}deg/s."
            )

    trajectory = {key: np.stack(value, axis=0) for key, value in rows.items() if value}
    trajectory["selected_seed"] = np.asarray([int(plan["selected_seed"])], dtype=np.int64)
    trajectory["task_knots"] = task_knots.astype(np.float32)
    return trajectory, True, False


def run_plan_only(args: argparse.Namespace) -> None:
    planner = HomeAnchoredLinePlanner(args)
    prepare_plan(planner, args, trajectory_seed=args.seed)
    print("Plan-only preflight passed. No robot command was sent and no HDF5 file was written.")


def create_datasets(
    h5: h5py.File,
    *,
    sample: dict[str, np.ndarray],
    sample_frame: np.ndarray | None,
    compression: str | None,
) -> dict[str, h5py.Dataset]:
    datasets: dict[str, h5py.Dataset] = {}
    datasets["ep_len"] = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
    datasets["ep_offset"] = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
    datasets["reward"] = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
    datasets["episode_seed"] = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
    datasets["selected_episode_seed"] = create_resizable_dataset(h5, "selected_episode_seed", (), np.int64, chunks=True)
    datasets["terminated"] = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
    datasets["truncated"] = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
    datasets["episode_idx"] = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
    datasets["step_idx"] = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)
    for name, shape_tail, dtype in (
        ("action", (ACTION_DIM,), np.float32),
        ("planned_action", (ACTION_DIM,), np.float32),
        ("measured_action", (ACTION_DIM,), np.float32),
        ("observation", sample["observation"].shape, np.float32),
        ("task_before", (3,), np.float32),
        ("task_after", (3,), np.float32),
        ("task_target", (3,), np.float32),
        ("measured_task_before", (3,), np.float32),
        ("measured_task_after", (3,), np.float32),
        ("q_before", (14,), np.float32),
        ("q_after", (14,), np.float32),
        ("q_cmd", (14,), np.float32),
        ("qpos", (14,), np.float32),
        ("qvel", (14,), np.float32),
        ("control", (14,), np.float32),
        ("left_attachment_pos", (3,), np.float32),
        ("right_attachment_pos", (3,), np.float32),
        ("rope_length", (1,), np.float32),
        ("command_duration", (1,), np.float32),
        ("settle_duration", (1,), np.float32),
        ("settle_position_error_deg", (1,), np.float32),
        ("settle_velocity_deg_s", (1,), np.float32),
        ("success", (1,), np.bool_),
        ("action_start_monotonic_time", (1,), np.float64),
        ("action_end_monotonic_time", (1,), np.float64),
        ("state_monotonic_time", (1,), np.float64),
        ("camera_capture_monotonic_time", (1,), np.float64),
    ):
        datasets[name] = create_resizable_dataset(h5, name, shape_tail, dtype, chunks=True)
    if sample_frame is not None:
        for name, dtype in (
            ("camera_frame_time", np.float64),
            ("camera_receive_time", np.float64),
            ("camera_header_time", np.float64),
            ("camera_frame_index", np.int64),
        ):
            datasets[name] = create_resizable_dataset(h5, name, (1,), dtype, chunks=True)
    if sample_frame is not None:
        datasets["pixels"] = create_resizable_dataset(
            h5,
            "pixels",
            sample_frame.shape,
            np.uint8,
            compression=compression,
            chunks=(1, *sample_frame.shape),
        )
    return datasets


def write_h5_attrs(
    h5: h5py.File,
    args: argparse.Namespace,
    planner: HomeAnchoredLinePlanner,
    *,
    shard: int,
    async_video_path: Path,
    async_metadata_path: Path,
    async_h5_path: Path,
    sample: dict[str, np.ndarray],
    sample_frame: np.ndarray | None,
) -> None:
    h5.attrs["format"] = "stable_worldmodel_hdf5"
    h5.attrs["source"] = "rope/data_real/rope_real_data_gen.py"
    h5.attrs["hardware"] = True
    h5.attrs["shard"] = shard
    h5.attrs["robot_backend"] = args.robot_backend
    h5.attrs["arm_mapping"] = "kuka-arm0-arm1"
    h5.attrs["trajectory_type"] = "synchronous_chained_lines"
    h5.attrs["seed"] = args.seed
    h5.attrs["num_waypoints"] = args.num_waypoints
    h5.attrs["samples_per_waypoint"] = args.samples_per_waypoint
    h5.attrs["requested_total_transitions"] = requested_transitions(args)
    h5.attrs["min_task_knot_distance"] = args.min_task_knot_distance
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
    h5.attrs["camera_resolution"] = json.dumps(list(sample_frame.shape[:2]) if sample_frame is not None else [])
    h5.attrs["camera_crop_center_x"] = args.camera_crop_center_x
    h5.attrs["camera_crop_center_y"] = args.camera_crop_center_y
    h5.attrs["camera_crop_zoom"] = args.camera_crop_zoom
    h5.attrs["camera_drop_frames_after_motion"] = args.camera_drop_frames_after_motion
    h5.attrs["camera_settle_delay"] = args.camera_settle_delay
    h5.attrs["async_backup_enabled"] = args.async_backup
    h5.attrs["async_camera_backup_video"] = str(async_video_path) if args.async_backup else ""
    h5.attrs["async_camera_backup_metadata"] = str(async_metadata_path) if args.async_backup else ""
    h5.attrs["async_camera_backup_h5"] = str(async_h5_path) if args.async_backup else ""
    h5.attrs["compression"] = args.compression
    h5.attrs["observation_dim"] = int(sample["observation"].shape[0])
    h5.attrs["action_dim"] = ACTION_DIM
    h5.attrs["qpos_dim"] = 14
    h5.attrs["qvel_dim"] = 14
    h5.attrs["control_dim"] = 14
    h5.attrs["action_semantics"] = "planned_task_delta_between_consecutive_line_waypoints"
    h5.attrs["measured_action_semantics"] = "measured_task_after_minus_measured_task_before"
    h5.attrs["row_semantics"] = "actions are transitions; pixels are state frames with one initial frame plus one post-settle frame per action"
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


def append_trajectory(
    datasets: dict[str, h5py.Dataset],
    trajectory: dict[str, np.ndarray],
    *,
    episode_idx: int,
    episode_seed: int,
    terminated: bool,
    truncated: bool,
) -> int:
    num_rows = int(trajectory["action"].shape[0])
    offset = int(datasets["pixels"].shape[0] if "pixels" in datasets and "pixels" in trajectory else datasets["action"].shape[0])
    frame_rows = int(trajectory["pixels"].shape[0]) if "pixels" in trajectory else num_rows
    for key, values in trajectory.items():
        if key in {"selected_seed", "task_knots"}:
            continue
        if key == "pixels" and key not in datasets:
            continue
        append_rows(datasets[key], values)
    append_rows(datasets["episode_idx"], np.full((num_rows,), episode_idx, dtype=np.int64))
    append_rows(datasets["step_idx"], np.arange(num_rows, dtype=np.int64))
    append_rows(datasets["ep_len"], np.asarray([frame_rows], dtype=np.int64))
    append_rows(datasets["ep_offset"], np.asarray([offset], dtype=np.int64))
    append_rows(datasets["reward"], np.asarray([0.0], dtype=np.float32))
    append_rows(datasets["episode_seed"], np.asarray([episode_seed], dtype=np.int64))
    append_rows(datasets["selected_episode_seed"], trajectory["selected_seed"])
    append_rows(datasets["terminated"], np.asarray([terminated], dtype=np.bool_))
    append_rows(datasets["truncated"], np.asarray([truncated], dtype=np.bool_))
    return num_rows


def main() -> None:
    global h5py

    args = parse_args()
    validate_args(args)
    if args.plan_only:
        run_plan_only(args)
        return

    import h5py

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    shard, output_path, async_video_path, async_metadata_path, async_h5_path = resolve_shard_paths(
        outdir,
        args.output_name,
        shard_id=args.shard_id,
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output shard already exists: {output_path}. Pass --overwrite or choose a new shard.")

    compression = None if args.compression == "none" else args.compression
    planner = HomeAnchoredLinePlanner(args)
    robot = make_robot(args)
    camera = make_camera(args)
    async_summary = None
    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []

    try:
        robot.connect()
        if camera is not None:
            camera.connect()
            if args.async_backup:
                camera.start_async_recording(
                    str(async_video_path),
                    str(async_metadata_path),
                    str(async_h5_path),
                    "lzf",
                )
        move_robot_to_planner_home(robot, planner, args)
        sample_info = planner.step_info(robot, planner.home_task)
        sample_frame_packet = capture_fresh_frame(camera, args)
        sample_frame = None if sample_frame_packet is None else np.asarray(sample_frame_packet["pixels"], dtype=np.uint8)

        with h5py.File(output_path, "w") as h5:
            write_h5_attrs(
                h5,
                args,
                planner,
                shard=shard,
                async_video_path=async_video_path,
                async_metadata_path=async_metadata_path,
                async_h5_path=async_h5_path,
                sample=sample_info,
                sample_frame=sample_frame,
            )
            datasets = create_datasets(h5, sample=sample_info, sample_frame=sample_frame, compression=compression)

            progress_total = requested_transitions(args)

            with tqdm(total=progress_total, desc="Collecting transitions", unit="transition") as progress:
                move_robot_to_planner_home(robot, planner, args)
                trajectory, terminated, truncated = collect_trajectory(
                    robot,
                    planner,
                    args,
                    trajectory_seed=args.seed,
                    camera=camera,
                    progress=progress,
                )
                num_actions = int(trajectory["action"].shape[0])
                append_trajectory(
                    datasets,
                    trajectory,
                    episode_idx=0,
                    episode_seed=args.seed,
                    terminated=terminated,
                    truncated=truncated,
                )
                rewards.append(0.0)
                step_counts.append(num_actions)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                progress.set_postfix(
                    waypoints=args.num_waypoints,
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                )

            h5.attrs["num_episodes"] = len(step_counts)
            h5.attrs["total_frames"] = int(datasets["pixels"].shape[0] if "pixels" in datasets else datasets["action"].shape[0])
            h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
            h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
            h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
            h5.attrs["terminated_fraction"] = float(np.mean(terminated_flags)) if terminated_flags else 0.0
            h5.attrs["truncated_fraction"] = float(np.mean(truncated_flags)) if truncated_flags else 0.0
            if args.async_backup and camera is not None:
                async_summary = camera.stop_async_recording()
                h5.attrs["async_camera_full_resolution_frames"] = (
                    int(async_summary.get("h5_written_frames", 0)) if async_summary is not None else 0
                )
            else:
                h5.attrs["async_camera_full_resolution_frames"] = 0
    finally:
        if camera is not None:
            if args.async_backup and async_summary is None:
                async_summary = camera.stop_async_recording()
            camera.close()
        robot.stop()
        robot.close()

    summary = {
        "output_path": str(output_path),
        "shard": shard,
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "num_waypoints": args.num_waypoints,
        "samples_per_waypoint": args.samples_per_waypoint,
        "async_backup": args.async_backup,
    }
    if async_summary is not None:
        summary["async_camera_backup"] = async_summary
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
