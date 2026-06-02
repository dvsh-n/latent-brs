#!/usr/bin/env python3
"""Collect real-hardware rope data by moving only IIWA_STATUS / arm 0."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rope.data import rope_real_chained_data_gen_cl as chained


DEFAULT_OUTDIR = "rope/data/real_data"
DEFAULT_OUTPUT_NAME = "rope_real_chained_status1_splines.h5"
CONFIG_FORBIDDEN_KEYS = {"i_understand_this_moves_real_robots"}
ACTION_DIM = 3
TASK_REACH_BOUNDS = chained.TASK_REACH_BOUNDS
TASK_HEIGHT_BOUNDS = chained.TASK_HEIGHT_BOUNDS
TASK_WIDTH_BOUNDS = chained.TASK_WIDTH_BOUNDS
HARDWARE_HOME_Q0_DEG = chained.HARDWARE_HOME_Q0_DEG


def _as_qpos7(values: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (7,):
        raise ValueError(f"Expected {name} with shape (7,), got {array.shape}.")
    return array


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
    parser.add_argument("--trajectory-mode", choices=("random-chained", "fixed-point-to-point"), default="random-chained")
    parser.add_argument("--fixed-execution-controller", choices=("closed-loop", "open-loop"), default="closed-loop")
    parser.add_argument("--fixed-task-start", type=float, nargs=3, default=None)
    parser.add_argument("--fixed-task-goal", type=float, nargs=3, default=None)
    parser.add_argument("--fixed-task-epsilon", type=float, default=0.0)
    parser.add_argument("--midpoint-inflation-scale", type=float, default=0.0)
    parser.add_argument("--min-task-knot-distance", type=float, default=0.035)
    parser.add_argument("--task-reach-bounds", type=float, nargs=2, default=TASK_REACH_BOUNDS)
    parser.add_argument("--task-height-bounds", type=float, nargs=2, default=TASK_HEIGHT_BOUNDS)
    parser.add_argument("--task-width-bounds", type=float, nargs=2, default=TASK_WIDTH_BOUNDS)
    parser.add_argument("--home-q0-deg", type=float, nargs=7, default=HARDWARE_HOME_Q0_DEG.tolist())
    parser.add_argument("--plan-retry-attempts", type=int, default=20)
    parser.add_argument("--robot-backend", choices=("drake-lcm",), default="drake-lcm")
    parser.add_argument("--drake-publish-period", type=float, default=0.005)
    parser.add_argument("--task-cl-kp", type=float, nargs=3, default=[0.2, 0.2, 0.15])
    parser.add_argument("--task-cl-max-correction-m", type=float, nargs=3, default=[0.002, 0.002, 0.002])
    parser.add_argument("--home-move-duration", type=float, default=4.0)
    parser.add_argument("--trajectory-hold-duration", type=float, default=1.0)
    parser.add_argument("--trajectory-start-blend-duration", type=float, default=4.0)
    parser.add_argument("--trajectory-start-settle-duration", type=float, default=0.25)
    parser.add_argument("--trajectory-start-ready-tolerance-deg", type=float, default=0.25)
    parser.add_argument("--trajectory-start-ready-velocity-deg-s", type=float, default=1.0)
    parser.add_argument("--trajectory-start-ready-timeout", type=float, default=5.0)
    parser.add_argument("--fixed-start-ready-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-control-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--max-reset-joint-move-deg", type=float, default=90.0)
    parser.add_argument("--max-command-measured-gap-deg", type=float, default=0.25)
    parser.add_argument("--max-real-joint-speed-deg-s", type=float, default=8.0)
    parser.add_argument("--max-real-joint-accel-deg-s2", type=float, default=20.0)
    parser.add_argument("--ik-position-tol", type=float, default=0.005)
    parser.add_argument("--ik-max-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--ik-random-restarts", type=int, default=8)
    parser.add_argument("--ik-random-seed-noise-deg", type=float, default=45.0)
    parser.add_argument("--orientation-theta-bound-deg", type=float, default=3.0)
    parser.add_argument("--orientation-constraint-mode", choices=("z_axis_down", "full"), default="z_axis_down")
    parser.add_argument("--fixed-a7-deg", type=float, default=None)
    parser.add_argument("--fixed-a7-tolerance-deg", type=float, default=0.05)
    parser.add_argument("--print-ik-checks", action="store_true")
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
    if args.num_trajectories < 1 or args.min_steps < 1 or args.num_splines < 1:
        raise ValueError("--num-trajectories, --min-steps, and --num-splines must be positive.")
    if args.shard_id is not None and args.shard_id < 0:
        raise ValueError("--shard-id must be non-negative when provided.")
    if args.control_timestep <= 0.0 or args.segment_duration <= 0.0 or args.drake_publish_period <= 0.0:
        raise ValueError("Timing arguments must be positive.")
    if args.trajectory_mode == "fixed-point-to-point":
        if args.fixed_task_start is None or args.fixed_task_goal is None:
            raise ValueError("--fixed-task-start and --fixed-task-goal are required for fixed-point-to-point mode.")
        if len(args.fixed_task_start) != 3 or len(args.fixed_task_goal) != 3:
            raise ValueError("--fixed-task-start and --fixed-task-goal must each provide 3 values.")
    if args.fixed_task_epsilon < 0.0:
        raise ValueError("--fixed-task-epsilon cannot be negative.")
    if args.home_move_duration <= 0.5:
        raise ValueError("--home-move-duration must be greater than 0.5s so it uses reset-motion safety limits.")
    if args.fixed_start_ready_tolerance_deg < 0.0:
        raise ValueError("--fixed-start-ready-tolerance-deg cannot be negative.")
    if args.plan_retry_attempts < 1:
        raise ValueError("--plan-retry-attempts must be positive.")
    if len(args.home_q0_deg) != 7:
        raise ValueError("--home-q0-deg must provide 7 joint values.")
    for name in ("task_reach_bounds", "task_height_bounds", "task_width_bounds"):
        values = getattr(args, name)
        if len(values) != 2 or values[0] >= values[1]:
            raise ValueError(f"--{name.replace('_', '-')} must provide two increasing values.")
    if len(args.task_cl_kp) != 3 or any(value < 0.0 for value in args.task_cl_kp):
        raise ValueError("--task-cl-kp must provide 3 non-negative gains.")
    if len(args.task_cl_max_correction_m) != 3 or any(value <= 0.0 for value in args.task_cl_max_correction_m):
        raise ValueError("--task-cl-max-correction-m must provide 3 positive correction limits.")
    if args.max_control_joint_step_deg <= 0.0 or args.max_reset_joint_move_deg <= 0.0:
        raise ValueError("Joint step limits must be positive.")
    if args.max_command_measured_gap_deg <= 0.0:
        raise ValueError("--max-command-measured-gap-deg must be positive.")
    if args.max_real_joint_speed_deg_s <= 0.0 or args.max_real_joint_accel_deg_s2 <= 0.0:
        raise ValueError("Real joint speed and acceleration limits must be positive.")
    if args.ik_position_tol <= 0.0 or args.ik_max_joint_step_deg <= 0.0:
        raise ValueError("IK tolerances must be positive.")
    if args.ik_random_restarts < 0:
        raise ValueError("--ik-random-restarts cannot be negative.")
    if args.ik_random_seed_noise_deg < 0.0:
        raise ValueError("--ik-random-seed-noise-deg cannot be negative.")
    if args.orientation_theta_bound_deg < 0.0:
        raise ValueError("--orientation-theta-bound-deg cannot be negative.")
    if args.fixed_a7_tolerance_deg < 0.0:
        raise ValueError("--fixed-a7-tolerance-deg cannot be negative.")
    if args.camera_backend == "ros2-topic" and not args.camera_topic:
        raise ValueError("--camera-topic is required when --camera-backend ros2-topic is selected.")
    if args.camera_backend == "ros2-topic":
        topic = str(args.camera_topic)
        if topic.endswith("/compressed") and args.camera_transport != "compressed":
            raise ValueError("camera_topic ends with /compressed but camera_transport is not compressed.")
        if not topic.endswith("/compressed") and args.camera_transport == "compressed":
            raise ValueError("camera_transport is compressed but camera_topic does not end with /compressed.")


@dataclass
class DrakeLCMSingleStatus1RobotBackend:
    publish_period: float = 0.005
    status_timeout: float = 5.0
    max_control_joint_step: float = np.deg2rad(5.0)
    max_reset_joint_move: float = np.deg2rad(90.0)
    max_command_measured_gap: float = np.deg2rad(0.35)
    hold_duration: float = 0.1
    start_blend_duration: float = 0.75
    start_settle_duration: float = 0.15
    start_ready_tolerance: float = np.deg2rad(0.25)
    start_ready_velocity_tolerance: float = np.deg2rad(2.0)
    start_ready_timeout: float = 3.0
    last_commanded_qpos: np.ndarray | None = None

    _diagram: object | None = field(default=None, init=False)
    _context: object | None = field(default=None, init=False)
    _simulator: object | None = field(default=None, init=False)
    _command_state: object | None = field(default=None, init=False)
    _outputs: dict[str, object] = field(default_factory=dict, init=False)

    def connect(self) -> None:
        if self._simulator is not None:
            return
        measured = self._read_initial_status()
        self._build_station(measured)
        self.last_commanded_qpos = measured.copy()
        self._publish_path(measured, measured, duration=self.hold_duration)

    def close(self) -> None:
        self._diagram = None
        self._context = None
        self._simulator = None
        self._command_state = None
        self._outputs.clear()

    def stop(self) -> None:
        if self._simulator is None:
            return
        try:
            measured = self.read_qpos_7()
            self._publish_path(measured, measured, duration=self.hold_duration)
            self.last_commanded_qpos = measured.copy()
        except Exception:
            pass

    def read_qpos_7(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._outputs["position_measured"].Eval(self._context), dtype=np.float64).reshape(7)

    def read_qvel_7(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._outputs["velocity_estimated"].Eval(self._context), dtype=np.float64).reshape(7)

    def read_torque_commanded_7(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._outputs["torque_commanded"].Eval(self._context), dtype=np.float64).reshape(7)

    def read_torque_measured_7(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._outputs["torque_measured"].Eval(self._context), dtype=np.float64).reshape(7)

    def read_torque_external_7(self) -> np.ndarray:
        self._require_connected()
        return np.asarray(self._outputs["torque_external"].Eval(self._context), dtype=np.float64).reshape(7)

    def command_joint_positions(self, qpos_7: np.ndarray, *, duration: float, blocking: bool) -> None:
        del blocking
        target = _as_qpos7(qpos_7, name="qpos_7")
        start = self.last_commanded_qpos.copy() if self.last_commanded_qpos is not None else self.read_qpos_7()
        self._check_joint_delta(start, target, duration=duration)
        self._publish_path(start, target, duration=duration)
        self.last_commanded_qpos = target.copy()

    def command_joint_path_step(self, qpos_7: np.ndarray, *, timestep: float) -> None:
        target = _as_qpos7(qpos_7, name="qpos_7")
        start = self.last_commanded_qpos.copy() if self.last_commanded_qpos is not None else self.read_qpos_7()
        self._check_joint_delta(start, target, duration=0.0)
        self._set_hold_command(target)
        self._simulator.AdvanceTo(self._context.get_time() + float(timestep))
        self.last_commanded_qpos = target.copy()

    def prepare_joint_path_start(self, qpos_7: np.ndarray) -> None:
        target = _as_qpos7(qpos_7, name="qpos_7")
        measured = self.read_qpos_7()
        self._check_joint_delta(measured, target, duration=0.0)
        max_delta = float(np.max(np.abs(target - measured)))
        if max_delta > self._effective_start_ready_tolerance() and self.start_blend_duration > 0.0:
            print(
                "Blending measured arm0 start to first waypoint: "
                f"max_delta={np.rad2deg(max_delta):.3f}deg over {self.start_blend_duration:.2f}s"
            )
            self._publish_path(measured, target, duration=self.start_blend_duration)
        else:
            self._set_hold_command(target)
            if max_delta > 1e-7:
                self._simulator.AdvanceTo(self._context.get_time() + self.publish_period)
        self.last_commanded_qpos = target.copy()
        if self.start_settle_duration > 0.0:
            self._publish_path(target, target, duration=self.start_settle_duration)
            self.last_commanded_qpos = target.copy()
        self._wait_until_start_ready(target)

    def _build_station(self, initial_qpos: np.ndarray) -> None:
        from pydrake.all import DiagramBuilder, IiwaControlMode, LeafSystem, Simulator
        from rope.data.iiwa_hardware import _MakeIiwaRobot, _MakeLcm

        class CommandState:
            def __init__(self, qpos: np.ndarray) -> None:
                self.hold = qpos.copy()

            def set_hold(self, qpos: np.ndarray) -> None:
                self.hold = qpos.copy()

        backend = self
        self._command_state = CommandState(initial_qpos)

        class Status1CommandSource(LeafSystem):
            def __init__(self) -> None:
                super().__init__()
                self.DeclareVectorInputPort("q_measured", 7)
                self.DeclareVectorOutputPort("q_cmd", 7, self.CalcQ)

            def CalcQ(self, context, output) -> None:
                measured = self.get_input_port().Eval(context)
                desired = backend._command_state.hold
                output.SetFromVector(
                    measured
                    + np.clip(
                        desired - measured,
                        -backend.max_command_measured_gap,
                        backend.max_command_measured_gap,
                    )
                )

        builder = DiagramBuilder()
        lcm = _MakeLcm(builder)
        iiwa = builder.AddSystem(
            _MakeIiwaRobot(
                lcm=lcm,
                control_mode=IiwaControlMode.kPositionOnly,
                lcm_channel_suffix="",
            )
        )
        source = builder.AddSystem(Status1CommandSource())
        builder.Connect(iiwa.GetOutputPort("position_measured"), source.GetInputPort("q_measured"))
        builder.Connect(source.GetOutputPort("q_cmd"), iiwa.GetInputPort("position"))
        for name in (
            "position_measured",
            "position_commanded",
            "velocity_estimated",
            "torque_commanded",
            "torque_measured",
            "torque_external",
        ):
            builder.ExportOutput(iiwa.GetOutputPort(name), name)
        self._diagram = builder.Build()
        self._context = self._diagram.CreateDefaultContext()
        self._simulator = Simulator(self._diagram, self._context)
        self._simulator.set_target_realtime_rate(1.0)
        self._outputs = {name: self._diagram.GetOutputPort(name) for name in (
            "position_measured",
            "position_commanded",
            "velocity_estimated",
            "torque_commanded",
            "torque_measured",
            "torque_external",
        )}

    def _read_initial_status(self) -> np.ndarray:
        from drake import lcmt_iiwa_status
        from pydrake.all import DrakeLcm

        lcm = DrakeLcm()
        q0 = None

        def handler(data: bytes) -> None:
            nonlocal q0
            msg = lcmt_iiwa_status.decode(data)
            q0 = np.asarray(msg.joint_position_measured, dtype=np.float64).reshape(7)

        lcm.Subscribe("IIWA_STATUS", handler)
        start = time.time()
        while time.time() - start < self.status_timeout:
            lcm.HandleSubscriptions(100)
            if q0 is not None:
                return q0
        raise RuntimeError(f"Failed to receive IIWA_STATUS within {self.status_timeout} s.")

    def _publish_path(self, start: np.ndarray, target: np.ndarray, *, duration: float) -> None:
        start = _as_qpos7(start, name="start")
        target = _as_qpos7(target, name="target")
        duration = max(float(duration), 0.0)
        steps = max(1, int(np.ceil(duration / max(self.publish_period, 1e-9))))
        start_time = self._context.get_time()
        for step in range(1, steps + 1):
            s = float(np.clip(step / steps, 0.0, 1.0))
            alpha = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            self._set_hold_command(start + alpha * (target - start))
            self._simulator.AdvanceTo(start_time + duration * step / steps)
        self._set_hold_command(target)

    def _set_hold_command(self, qpos_7: np.ndarray) -> None:
        self._require_connected()
        self._command_state.set_hold(_as_qpos7(qpos_7, name="qpos_7"))
        self.last_commanded_qpos = _as_qpos7(qpos_7, name="qpos_7").copy()

    def wait_until_joint_target_ready(self, target: np.ndarray, *, position_tolerance: float | None = None) -> None:
        self._wait_until_start_ready(target, position_tolerance=position_tolerance)

    def _wait_until_start_ready(self, target: np.ndarray, *, position_tolerance: float | None = None) -> None:
        if position_tolerance is None:
            position_tolerance = self._effective_start_ready_tolerance()
        else:
            position_tolerance = float(position_tolerance)
        start_time = float(self._context.get_time())
        last_position_err = np.inf
        last_speed = np.inf
        while float(self._context.get_time()) - start_time <= self.start_ready_timeout:
            measured = self.read_qpos_7()
            velocity = self.read_qvel_7()
            last_position_err = float(np.max(np.abs(measured - target)))
            last_speed = float(np.max(np.abs(velocity)))
            if last_position_err <= position_tolerance and last_speed <= self.start_ready_velocity_tolerance:
                print(
                    "Start ready: "
                    f"max_position_err={np.rad2deg(last_position_err):.3f}deg, "
                    f"max_speed={np.rad2deg(last_speed):.3f}deg/s"
                )
                return
            self._set_hold_command(target)
            self._simulator.AdvanceTo(self._context.get_time() + self.publish_period)
        raise RuntimeError(
            "Robot did not settle at trajectory start before timeout: "
            f"max_position_err={np.rad2deg(last_position_err):.3f}deg, "
            f"max_speed={np.rad2deg(last_speed):.3f}deg/s "
            f"(position_tol={np.rad2deg(position_tolerance):.3f}deg)."
        )

    def _effective_start_ready_tolerance(self) -> float:
        return max(self.start_ready_tolerance, self.max_command_measured_gap)

    def _check_joint_delta(self, start: np.ndarray, target: np.ndarray, *, duration: float) -> None:
        max_delta = float(np.max(np.abs(target - start)))
        limit = self.max_reset_joint_move if duration > 0.5 else self.max_control_joint_step
        if max_delta > limit:
            raise RuntimeError(
                "Refusing unsafe arm0 joint command: "
                f"max_delta={np.rad2deg(max_delta):.2f} deg exceeds limit={np.rad2deg(limit):.2f} deg."
            )

    def _require_connected(self) -> None:
        if self._simulator is None or self._context is None or self._diagram is None:
            raise RuntimeError("DrakeLCMSingleStatus1RobotBackend is not connected.")


def make_robot(args: argparse.Namespace) -> DrakeLCMSingleStatus1RobotBackend:
    return DrakeLCMSingleStatus1RobotBackend(
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


class Status1TaskPlanner:
    def __init__(self, args: argparse.Namespace) -> None:
        from rope.data.iiwa_cartesian_ik import SingleIiwaPositionIK

        self.args = args
        self.task_bounds = TaskBounds(
            reach=tuple(float(value) for value in args.task_reach_bounds),
            height=tuple(float(value) for value in args.task_height_bounds),
            width=tuple(float(value) for value in args.task_width_bounds),
        )
        self.home_task = self.task_bounds.center()
        self.ik0 = SingleIiwaPositionIK(
            orientation_theta_bound=np.deg2rad(args.orientation_theta_bound_deg),
            orientation_constraint_mode=args.orientation_constraint_mode,
        )
        self.ik_position_tol = float(args.ik_position_tol)
        self.ik_max_joint_step = np.deg2rad(args.ik_max_joint_step_deg)
        self.fixed_a7_rad = None if args.fixed_a7_deg is None else float(np.deg2rad(args.fixed_a7_deg))
        self.fixed_a7_tolerance_rad = float(np.deg2rad(args.fixed_a7_tolerance_deg))
        self.print_ik_checks = bool(args.print_ik_checks)
        self.ik_rng = np.random.default_rng(0)
        self.hardware_home_qpos = self._hardware_home_goal()
        self.dataset_home_qpos = self.hardware_home_qpos.copy()
        self.p0_home = self.ik0.fk_position(self.hardware_home_qpos)

    def set_ik_retry_seed(self, seed: int) -> None:
        self.ik_rng = np.random.default_rng(int(seed))

    def clip_task(self, values: np.ndarray) -> np.ndarray:
        return self.task_bounds.clip(values).astype(np.float64)

    def task_to_qpos(
        self,
        task_state: np.ndarray,
        q_seed: np.ndarray,
        *,
        label: str,
        max_joint_move_from_seed: float | None = None,
        unrestricted_seed_move: bool = False,
    ) -> np.ndarray:
        task_state = self.clip_task(task_state)
        q_seed = np.asarray(q_seed, dtype=np.float64).reshape(7)
        p_goal = self.task_to_cartesian_position(task_state)
        if max_joint_move_from_seed is None and not unrestricted_seed_move:
            max_joint_move_from_seed = self.ik_max_joint_step
        return self._solve_arm(p_goal, q_seed, label, max_joint_move_from_seed=max_joint_move_from_seed)

    def task_to_cartesian_position(self, task_state: np.ndarray) -> np.ndarray:
        task_state = self.clip_task(task_state)
        delta = task_state - self.home_task
        return self.p0_home + np.array([delta[0], delta[2], delta[1]], dtype=np.float64)

    def task_closed_loop_qpos(
        self,
        task_ref: np.ndarray,
        q_meas: np.ndarray,
        *,
        q_seed: np.ndarray | None,
        kp: np.ndarray,
        max_correction: np.ndarray,
        max_joint_move_from_seed: float | None = None,
    ) -> np.ndarray:
        q_meas = np.asarray(q_meas, dtype=np.float64).reshape(7)
        q_seed = q_meas if q_seed is None else np.asarray(q_seed, dtype=np.float64).reshape(7)
        task_meas = self.measured_task_state_from_attachment_position(self.ik0.fk_position(q_meas))
        task_err = self.clip_task(task_ref) - task_meas
        correction = np.clip(kp * task_err, -max_correction, max_correction)
        return self.task_to_qpos(
            self.clip_task(task_ref + correction),
            q_seed,
            label="closed-loop servo",
            max_joint_move_from_seed=max_joint_move_from_seed,
        )

    def plan_joint_path(self, task_path: np.ndarray) -> np.ndarray:
        task_path = np.asarray(task_path, dtype=np.float64)
        start_max_joint_move = None if self.args.trajectory_mode == "fixed-point-to-point" else self.ik_max_joint_step
        q_prev = self.task_to_qpos(
            task_path[0],
            self.dataset_home_qpos,
            label="arm0 waypoint 0",
            max_joint_move_from_seed=start_max_joint_move,
            unrestricted_seed_move=self.args.trajectory_mode == "fixed-point-to-point",
        )
        q_path = [q_prev.copy()]
        for index, task_state in enumerate(task_path[1:], start=1):
            q_prev = self.task_to_qpos(task_state, q_prev, label=f"arm0 waypoint {index}")
            q_path.append(q_prev.copy())
        if self.args.trajectory_mode == "random-chained":
            q_path[-1] = self.dataset_home_qpos.copy()
        return np.stack(q_path, axis=0)

    def step_info(self, robot: DrakeLCMSingleStatus1RobotBackend, task_target: np.ndarray, elapsed_time: float) -> dict[str, np.ndarray]:
        qpos = robot.read_qpos_7().astype(np.float32)
        qvel = robot.read_qvel_7().astype(np.float32)
        control = qpos.copy() if robot.last_commanded_qpos is None else robot.last_commanded_qpos.astype(np.float32)
        torque_control = robot.read_torque_commanded_7().astype(np.float32)
        torque_measured = robot.read_torque_measured_7().astype(np.float32)
        torque_external = robot.read_torque_external_7().astype(np.float32)
        p0 = self.ik0.fk_position(qpos).astype(np.float32)
        target = np.asarray(task_target, dtype=np.float32)
        measured_task_target = self.measured_task_state_from_attachment_position(p0).astype(np.float32)
        observation = np.concatenate(
            [target, qpos, qvel, control, p0, torque_control, torque_measured, torque_external],
            axis=0,
        ).astype(np.float32)
        return {
            "observation": observation,
            "task_target": target,
            "measured_task_target": measured_task_target,
            "qpos": qpos,
            "qvel": qvel,
            "control": control,
            "torque_control": torque_control,
            "torque_measured": torque_measured,
            "torque_external": torque_external,
            "left_attachment_pos": p0,
            "time": np.asarray([elapsed_time], dtype=np.float32),
        }

    def anchor_dataset_home(self, measured_qpos: np.ndarray) -> None:
        measured_qpos = np.asarray(measured_qpos, dtype=np.float64).reshape(7)
        self.dataset_home_qpos = measured_qpos.copy()
        self.p0_home = self.ik0.fk_position(measured_qpos)
        hardware_delta_deg = np.rad2deg(np.abs(measured_qpos - self.hardware_home_qpos))
        print(
            "Anchored status1 dataset home to measured pose: "
            f"max_joint_offset_from_hardware_home={np.max(hardware_delta_deg):.3f}deg"
        )

    def measured_task_state_from_attachment_position(self, attachment_pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(attachment_pos, dtype=np.float64)
        delta = pos - self.p0_home
        measured = self.home_task.copy()
        measured[0] += delta[0]
        measured[1] += delta[2]
        measured[2] += delta[1]
        return self.clip_task(measured)

    def _solve_arm(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        label: str,
        *,
        max_joint_move_from_seed: float | None,
    ) -> np.ndarray:
        joint_lower_bounds = None
        joint_upper_bounds = None
        if self.fixed_a7_rad is not None:
            joint_lower_bounds = np.full(7, -np.inf, dtype=np.float64)
            joint_upper_bounds = np.full(7, np.inf, dtype=np.float64)
            joint_lower_bounds[6] = self.fixed_a7_rad - self.fixed_a7_tolerance_rad
            joint_upper_bounds[6] = self.fixed_a7_rad + self.fixed_a7_tolerance_rad
        seed_candidates = [np.asarray(q_seed, dtype=np.float64).reshape(7)]
        if self.args.ik_random_restarts > 0 and self.args.ik_random_seed_noise_deg > 0.0:
            noise = np.deg2rad(self.args.ik_random_seed_noise_deg)
            for _ in range(self.args.ik_random_restarts):
                candidate = seed_candidates[0] + self.ik_rng.uniform(-noise, noise, size=7)
                if self.fixed_a7_rad is not None:
                    candidate[6] = self.fixed_a7_rad
                seed_candidates.append(candidate)

        last_info = None
        for restart_index, candidate_seed in enumerate(seed_candidates):
            q_sol, info = self.ik0.solve_position_ik(
                target_pos,
                candidate_seed,
                position_tol=self.ik_position_tol,
                max_joint_move_from_seed=max_joint_move_from_seed,
                joint_lower_bounds=joint_lower_bounds,
                joint_upper_bounds=joint_upper_bounds,
            )
            last_info = info
            if q_sol is not None:
                suffix = "" if restart_index == 0 else f" restart={restart_index}"
                self._print_ik_check(f"{label}{suffix}", q_sol, info)
                return q_sol
        raise RuntimeError(f"{label} IK failed for target {np.round(target_pos, 4)} after {len(seed_candidates)} seed(s): {last_info}")

    def _hardware_home_goal(self) -> np.ndarray:
        q_nominal = np.deg2rad(np.asarray(self.args.home_q0_deg, dtype=np.float64))
        if self.fixed_a7_rad is not None:
            q_nominal[6] = self.fixed_a7_rad
        p_home = self.ik0.fk_position(q_nominal)
        joint_lower_bounds = None
        joint_upper_bounds = None
        if self.fixed_a7_rad is not None:
            joint_lower_bounds = np.full(7, -np.inf, dtype=np.float64)
            joint_upper_bounds = np.full(7, np.inf, dtype=np.float64)
            joint_lower_bounds[6] = self.fixed_a7_rad - self.fixed_a7_tolerance_rad
            joint_upper_bounds[6] = self.fixed_a7_rad + self.fixed_a7_tolerance_rad
        q_home, info = self.ik0.solve_position_ik(
            p_home,
            q_nominal,
            position_tol=0.002,
            max_joint_move_from_seed=None,
            joint_lower_bounds=joint_lower_bounds,
            joint_upper_bounds=joint_upper_bounds,
        )
        if q_home is None:
            raise RuntimeError(f"Could not solve oriented status1 hardware home: {info}")
        self._print_ik_check("status1 hardware home", q_home, info)
        print(
            "oriented status1 home IK errors: "
            f"pos={info['pos_err']:.5f}m "
            f"link7_down={np.rad2deg(info['z_axis_down_err']):.2f}deg "
            f"a7={np.rad2deg(q_home[6]):.2f}deg"
        )
        return q_home

    def _print_ik_check(self, label: str, q_sol: np.ndarray, info: dict[str, object]) -> None:
        if not self.print_ik_checks:
            return
        if label == "closed-loop servo" or label.startswith("arm0 waypoint"):
            return
        z_axis_down_err = float(info.get("z_axis_down_err", self.ik0.z_axis_down_error(q_sol)))
        a7_deg = float(np.rad2deg(q_sol[6]))
        if self.fixed_a7_rad is None:
            a7_status = "free"
        else:
            a7_err_deg = float(np.rad2deg(q_sol[6] - self.fixed_a7_rad))
            a7_status = f"target={np.rad2deg(self.fixed_a7_rad):.2f}deg err={a7_err_deg:+.4f}deg"
        print(
            "[status1 IK] "
            f"{label}: link7_down_err={np.rad2deg(z_axis_down_err):.4f}deg "
            f"bound={self.args.orientation_theta_bound_deg:.4f}deg "
            f"a7={a7_deg:.4f}deg {a7_status} "
            f"pos_err={float(info['pos_err']):.6f}m"
        )


def move_robot_to_planner_home(robot: DrakeLCMSingleStatus1RobotBackend, planner: Status1TaskPlanner, args: argparse.Namespace) -> None:
    measured = robot.read_qpos_7()
    max_delta_deg = float(np.rad2deg(np.max(np.abs(planner.hardware_home_qpos - measured))))
    if max_delta_deg <= args.trajectory_start_ready_tolerance_deg:
        print(f"Status1 already at configured chained home: max_delta={max_delta_deg:.2f}deg")
        robot.prepare_joint_path_start(planner.hardware_home_qpos)
    else:
        print(
            "Moving status1 to configured chained home: "
            f"duration={args.home_move_duration:.2f}s, max_delta={max_delta_deg:.2f}deg"
        )
        robot.command_joint_positions(planner.hardware_home_qpos, duration=args.home_move_duration, blocking=True)
        robot.wait_until_joint_target_ready(planner.hardware_home_qpos)
    planner.anchor_dataset_home(robot.read_qpos_7())


def make_qpos_interpolator(q_path: np.ndarray, sample_times: np.ndarray):
    q_path = np.asarray(q_path, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.float64)
    if q_path.shape != (times.shape[0], 7):
        raise ValueError(f"Expected q_path shape ({times.shape[0]}, 7), got {q_path.shape}.")
    first_dt = max(float(times[1] - times[0]), 1e-9)
    last_dt = max(float(times[-1] - times[-2]), 1e-9)
    interpolator = PchipInterpolator(
        np.concatenate([[times[0] - first_dt], times, [times[-1] + last_dt]]),
        np.concatenate([q_path[:1], q_path, q_path[-1:]], axis=0),
        axis=0,
    )

    def interpolate(sample_time: float) -> np.ndarray:
        if sample_time <= times[0]:
            return q_path[0].copy()
        if sample_time >= times[-1]:
            return q_path[-1].copy()
        return np.asarray(interpolator(sample_time), dtype=np.float64).reshape(7)

    return interpolate


def sample_task_point_in_epsilon_ball(
    center: np.ndarray,
    epsilon: float,
    planner: Status1TaskPlanner,
    rng: np.random.Generator,
    *,
    label: str,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64).reshape(ACTION_DIM)
    clipped_center = planner.clip_task(center)
    if not np.allclose(clipped_center, center):
        raise ValueError(f"{label} is outside task bounds and would clip to {clipped_center.tolist()}.")
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        return center.copy()

    for _ in range(10_000):
        direction = rng.normal(size=ACTION_DIM)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        radius = epsilon * float(rng.random() ** (1.0 / ACTION_DIM))
        candidate = center + radius * direction / norm
        if np.allclose(planner.clip_task(candidate), candidate):
            return candidate.astype(np.float64)
    raise RuntimeError(
        f"Could not sample {label} within epsilon ball and task bounds after 10000 attempts "
        f"(center={center.tolist()}, epsilon={epsilon})."
    )


def sample_fixed_point_to_point_task_path(
    planner: Status1TaskPlanner,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fixed point-to-point mode treats each shell-loop run as one start->goal
    # rollout. Do not multiply by num_splines here; that belongs to random
    # chained spline collection and would make fixed rollouts unexpectedly long.
    total_duration = float(args.segment_duration)
    sample_times = np.arange(0.0, total_duration, args.control_timestep, dtype=np.float64)
    sample_times = np.append(sample_times, total_duration)
    start = sample_task_point_in_epsilon_ball(
        np.asarray(args.fixed_task_start, dtype=np.float64),
        args.fixed_task_epsilon,
        planner,
        rng,
        label="fixed_task_start",
    )
    goal = sample_task_point_in_epsilon_ball(
        np.asarray(args.fixed_task_goal, dtype=np.float64),
        args.fixed_task_epsilon,
        planner,
        rng,
        label="fixed_task_goal",
    )
    print(
        "Fixed trajectory task endpoints: "
        f"start={np.round(start, 5).tolist()}, "
        f"goal={np.round(goal, 5).tolist()}, "
        f"epsilon={float(args.fixed_task_epsilon):.5f}"
    )
    s = sample_times / max(total_duration, 1e-9)
    alpha = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    task_path = start[None, :] + alpha[:, None] * (goal - start)[None, :]
    task_knots = np.stack([start, goal], axis=0)
    return task_path.astype(np.float64), sample_times, task_knots


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
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64).reshape(7)
    dt = max(float(timestep), 1e-9)
    desired_velocity = (q_target - q_start) / dt
    desired_velocity = np.clip(desired_velocity, -max_speed, max_speed)
    desired_velocity = np.clip(desired_velocity, previous_velocity - max_accel * dt, previous_velocity + max_accel * dt)
    q_limited = q_start + desired_velocity * dt
    delta = q_limited - q_start
    max_delta = float(np.max(np.abs(delta)))
    if max_delta > max_step:
        q_limited = q_start + delta * (max_step / max(max_delta, 1e-12))
    return q_limited, (q_limited - q_start) / dt


def command_task_path_closed_loop(
    robot: DrakeLCMSingleStatus1RobotBackend,
    planner: Status1TaskPlanner,
    q_path: np.ndarray,
    task_path: np.ndarray,
    sample_times: np.ndarray,
    args: argparse.Namespace,
    *,
    sample_callback=None,
) -> None:
    times = np.asarray(sample_times, dtype=np.float64)
    task_ref_at = chained.make_task_interpolator(task_path, times)
    q_ref_at = make_qpos_interpolator(q_path, times)
    kp = np.asarray(args.task_cl_kp, dtype=np.float64).reshape(ACTION_DIM)
    max_correction = np.asarray(args.task_cl_max_correction_m, dtype=np.float64).reshape(ACTION_DIM)
    duration = float(times[-1])
    period = float(args.drake_publish_period)
    current = 0.0
    sample_index = 1
    q_cmd_velocity = np.zeros(7, dtype=np.float64)
    fallback_count = 0
    first_fallback_error = None
    last_fallback_error = None
    servo_max_joint_moves = (
        np.deg2rad(max(args.max_control_joint_step_deg, 4.0 * args.ik_max_joint_step_deg)),
        np.deg2rad(args.max_reset_joint_move_deg),
    )
    eps = 1e-9
    while current < duration - eps:
        next_time = min(current + period, duration)
        timestep = next_time - current
        q_meas = robot.read_qpos_7()
        task_ref = task_ref_at(next_time)
        q_ref = q_ref_at(next_time)
        q_start = robot.last_commanded_qpos.copy() if robot.last_commanded_qpos is not None else q_meas
        solved = False
        for correction_scale in (1.0, 0.5, 0.25, 0.0):
            for servo_max_joint_move in servo_max_joint_moves:
                try:
                    q_cmd = planner.task_closed_loop_qpos(
                        task_ref,
                        q_meas,
                        q_seed=q_ref,
                        kp=kp * correction_scale,
                        max_correction=max_correction * correction_scale,
                        max_joint_move_from_seed=servo_max_joint_move,
                    )
                    solved = True
                    break
                except RuntimeError as error:
                    last_error = error
            if solved:
                break
        if not solved:
            fallback_count += 1
            if first_fallback_error is None:
                first_fallback_error = str(last_error)
            last_fallback_error = str(last_error)
            q_cmd = q_ref
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
        print(
            "Status1 closed-loop IK used nominal joint fallback: "
            f"count={fallback_count}, first_error={first_fallback_error}, last_error={last_fallback_error}"
        )
    if robot.last_commanded_qpos is not None:
        robot.command_joint_positions(robot.last_commanded_qpos, duration=args.trajectory_hold_duration, blocking=True)


def command_joint_path_open_loop(
    robot: DrakeLCMSingleStatus1RobotBackend,
    q_path: np.ndarray,
    sample_times: np.ndarray,
    args: argparse.Namespace,
    *,
    sample_callback=None,
) -> None:
    times = np.asarray(sample_times, dtype=np.float64)
    q_ref_at = make_qpos_interpolator(q_path, times)
    duration = float(times[-1])
    period = float(args.drake_publish_period)
    current = 0.0
    sample_index = 1
    q_cmd_velocity = np.zeros(7, dtype=np.float64)
    eps = 1e-9
    while current < duration - eps:
        next_time = min(current + period, duration)
        timestep = next_time - current
        q_ref = q_ref_at(next_time)
        q_meas = robot.read_qpos_7()
        q_start = robot.last_commanded_qpos.copy() if robot.last_commanded_qpos is not None else q_meas
        q_cmd, q_cmd_velocity = limit_qpos_motion(
            q_ref,
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


def move_robot_to_trajectory_start_closed_loop(
    robot: DrakeLCMSingleStatus1RobotBackend,
    planner: Status1TaskPlanner,
    q_start: np.ndarray,
    task_start: np.ndarray,
    args: argparse.Namespace,
) -> None:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    task_start = planner.clip_task(np.asarray(task_start, dtype=np.float64).reshape(ACTION_DIM))
    measured_q = robot.read_qpos_7()
    measured_task = planner.measured_task_state_from_attachment_position(planner.ik0.fk_position(measured_q))
    task_delta = float(np.linalg.norm(task_start - measured_task))
    joint_delta_deg = float(np.rad2deg(np.max(np.abs(q_start - measured_q))))
    print(
        "Closed-loop moving status1 EE to trajectory start: "
        f"duration={args.home_move_duration:.2f}s, task_delta={task_delta:.4f}m, "
        f"joint_delta={joint_delta_deg:.2f}deg"
    )
    duration = float(args.home_move_duration)
    sample_times = np.arange(0.0, duration, args.control_timestep, dtype=np.float64)
    sample_times = np.append(sample_times, duration)
    s = sample_times / max(duration, 1e-9)
    alpha = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    start_task_path = measured_task[None, :] + alpha[:, None] * (task_start - measured_task)[None, :]
    start_q_path = measured_q[None, :] + alpha[:, None] * (q_start - measured_q)[None, :]
    command_task_path_closed_loop(
        robot,
        planner,
        start_q_path,
        start_task_path,
        sample_times,
        args,
        sample_callback=None,
    )
    wait_until_task_start_ready_closed_loop(robot, planner, q_start, task_start, args)


def wait_until_task_start_ready_closed_loop(
    robot: DrakeLCMSingleStatus1RobotBackend,
    planner: Status1TaskPlanner,
    q_start: np.ndarray,
    task_start: np.ndarray,
    args: argparse.Namespace,
) -> None:
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    task_start = planner.clip_task(np.asarray(task_start, dtype=np.float64).reshape(ACTION_DIM))
    task_tolerance = max(3.0 * float(args.ik_position_tol), 0.004)
    warning_tolerance = max(task_tolerance, 0.02)
    settle_duration = 0.25
    required_settle_ticks = max(1, int(np.ceil(settle_duration / max(robot.publish_period, 1e-9))))
    settled_ticks = 0
    hold_q = robot.last_commanded_qpos.copy() if robot.last_commanded_qpos is not None else q_start.copy()
    start_time = float(robot._context.get_time())
    last_task_err = np.inf
    last_speed = np.inf
    while float(robot._context.get_time()) - start_time <= robot.start_ready_timeout:
        measured_q = robot.read_qpos_7()
        measured_task = planner.measured_task_state_from_attachment_position(planner.ik0.fk_position(measured_q))
        last_task_err = float(np.max(np.abs(task_start - measured_task)))
        last_speed = float(np.max(np.abs(robot.read_qvel_7())))
        if last_task_err <= task_tolerance and last_speed <= robot.start_ready_velocity_tolerance:
            settled_ticks += 1
            if settled_ticks >= required_settle_ticks:
                print(
                    "Task start ready: "
                    f"max_task_err={last_task_err:.5f}m, "
                    f"max_speed={np.rad2deg(last_speed):.3f}deg/s"
                )
                return
        elif last_task_err <= warning_tolerance and last_speed <= robot.start_ready_velocity_tolerance:
            settled_ticks += 1
            if settled_ticks >= required_settle_ticks:
                print(
                    "Task start settle warning: continuing with residual error "
                    f"max_task_err={last_task_err:.5f}m, "
                    f"max_speed={np.rad2deg(last_speed):.3f}deg/s "
                    f"(task_tol={task_tolerance:.5f}m)."
                )
                return
        else:
            settled_ticks = 0
        robot.command_joint_path_step(hold_q, timestep=robot.publish_period)
    raise RuntimeError(
        "Robot did not settle at trajectory task start before timeout: "
        f"max_task_err={last_task_err:.5f}m, "
        f"max_speed={np.rad2deg(last_speed):.3f}deg/s "
        f"(task_tol={task_tolerance:.5f}m, warning_tol={warning_tolerance:.5f}m)."
    )


def prepare_plan(planner: Status1TaskPlanner, args: argparse.Namespace, *, trajectory_seed: int) -> dict[str, np.ndarray | float | int]:
    last_error: Exception | None = None
    attempts = args.plan_retry_attempts
    for attempt in range(attempts):
        seed = trajectory_seed + attempt
        planner.set_ik_retry_seed(seed)
        try:
            if args.trajectory_mode == "fixed-point-to-point":
                rng = np.random.default_rng(seed)
                task_path, sample_times, task_knots = sample_fixed_point_to_point_task_path(planner, args, rng)
            else:
                task_path, sample_times, task_knots = chained.sample_chained_task_path(planner, args, seed=seed)
            q_path = planner.plan_joint_path(task_path)
            q_path, task_path, sample_times = chained.densify_for_joint_step_limit(
                q_path,
                task_path,
                sample_times,
                max_joint_step=np.deg2rad(args.max_control_joint_step_deg),
            )
            sample_times, timing_scale = chained.retime_for_joint_limits(q_path, sample_times, args)
            print_plan_summary(planner, q_path, task_path, sample_times, task_knots, timing_scale, seed)
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
            print(f"Rejected status1 chained trajectory seed {seed}: {error}")
    if args.trajectory_mode == "fixed-point-to-point":
        raise RuntimeError(
            "Could not plan the fixed status1 trajectory. The fixed start/goal are inside task bounds, "
            f"but no IK solution was found after {attempts} seeded attempt(s) with the active constraints "
            f"(orientation_mode={args.orientation_constraint_mode}, "
            f"orientation_bound={args.orientation_theta_bound_deg:.3f}deg, "
            f"fixed_a7={args.fixed_a7_deg}deg +/- {args.fixed_a7_tolerance_deg:.3f}deg). "
            "Try points closer to the home task state, relax the constraints, or run --plan-only while tuning."
        ) from last_error
    raise RuntimeError(f"Could not plan a feasible status1 trajectory after {args.plan_retry_attempts} attempts.") from last_error


def print_plan_summary(
    planner: Status1TaskPlanner,
    q_path: np.ndarray,
    task_path: np.ndarray,
    sample_times: np.ndarray,
    task_knots: np.ndarray,
    timing_scale: float,
    seed: int,
) -> None:
    max_step, max_speed, avg_speed, max_accel = chained.joint_path_speed_accel(q_path, sample_times)
    delta_deg = np.rad2deg(np.max(np.abs(q_path - q_path[0]), axis=0))
    task_delta = task_path - task_path[0]
    cartesian_start = planner.task_to_cartesian_position(task_path[0])
    cartesian_end = planner.task_to_cartesian_position(task_path[-1])
    cartesian_delta = cartesian_end - cartesian_start
    print(
        json.dumps(
            {
                "selected_seed": int(seed),
                "duration_s": float(sample_times[-1]),
                "samples": int(sample_times.shape[0]),
                "timing_scale": float(timing_scale),
                "max_joint_step_deg": float(np.rad2deg(max_step)),
                "max_joint_speed_deg_s": float(np.rad2deg(max_speed)),
                "avg_joint_speed_deg_s": float(np.rad2deg(avg_speed)),
                "max_joint_accel_deg_s2": float(np.rad2deg(max_accel)),
                "arm0_max_total_joint_delta_deg": float(np.max(delta_deg)),
                "task_delta_min": np.min(task_delta, axis=0).round(5).tolist(),
                "task_delta_max": np.max(task_delta, axis=0).round(5).tolist(),
                "task_start": task_path[0].round(5).tolist(),
                "task_end": task_path[-1].round(5).tolist(),
                "ee_cartesian_start_xyz": cartesian_start.round(5).tolist(),
                "ee_cartesian_end_xyz": cartesian_end.round(5).tolist(),
                "ee_cartesian_delta_xyz": cartesian_delta.round(5).tolist(),
                "num_task_knots": int(task_knots.shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def collect_trajectory(
    robot: DrakeLCMSingleStatus1RobotBackend,
    planner: Status1TaskPlanner,
    args: argparse.Namespace,
    *,
    trajectory_seed: int,
    camera=None,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    plan = prepare_plan(planner, args, trajectory_seed=trajectory_seed)
    q_path = np.asarray(plan["q_path"], dtype=np.float64)
    task_path = np.asarray(plan["task_path"], dtype=np.float64)
    sample_times = np.asarray(plan["sample_times"], dtype=np.float64)
    move_robot_to_trajectory_start_closed_loop(robot, planner, q_path[0], task_path[0], args)
    print(f"Starting status1 chained trajectory: duration={sample_times[-1]:.2f}s, samples={sample_times.shape[0]}")
    first_sample_time = time.monotonic()
    first = planner.step_info(robot, task_path[0], elapsed_time=0.0)
    first["sample_monotonic_time"] = np.asarray([first_sample_time], dtype=np.float64)
    first["state_monotonic_time"] = np.asarray([time.monotonic()], dtype=np.float64)
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
        info["sample_monotonic_time"] = np.asarray([sample_monotonic_time], dtype=np.float64)
        info["state_monotonic_time"] = np.asarray([time.monotonic()], dtype=np.float64)
        if camera is not None:
            frame = camera.read_frame()
            info["pixels"] = np.asarray(frame["pixels"], dtype=np.uint8)
            info["camera_frame_time"] = np.asarray([float(frame["frame_time"])], dtype=np.float64)
            info["camera_receive_time"] = np.asarray([float(frame["receive_time"])], dtype=np.float64)
            info["camera_header_time"] = np.asarray([float(frame["header_time"])], dtype=np.float64)
            info["camera_frame_index"] = np.asarray([int(frame["frame_index"])], dtype=np.int64)
        for key, value in info.items():
            lists[key].append(value)

    if args.trajectory_mode == "fixed-point-to-point" and args.fixed_execution_controller == "open-loop":
        command_joint_path_open_loop(robot, q_path, sample_times, args, sample_callback=sample)
    else:
        command_task_path_closed_loop(robot, planner, q_path, task_path, sample_times, args, sample_callback=sample)
    trajectory = {key: np.stack(values, axis=0) for key, values in lists.items()}
    planned_actions = np.diff(task_path, axis=0).astype(np.float32)
    measured_actions = np.diff(np.asarray(trajectory["measured_task_target"], dtype=np.float32), axis=0).astype(np.float32)
    trajectory["action"] = planned_actions if planned_actions.size else np.zeros((0, ACTION_DIM), dtype=np.float32)
    trajectory["planned_action"] = trajectory["action"].copy()
    trajectory["measured_action"] = measured_actions if measured_actions.size else np.zeros((0, ACTION_DIM), dtype=np.float32)
    torque_delta = np.empty_like(trajectory["torque_control"], dtype=np.float32)
    torque_delta[0] = np.nan
    torque_delta[1:] = np.diff(trajectory["torque_control"], axis=0)
    trajectory["torque_control_delta"] = torque_delta
    trajectory["selected_seed"] = np.asarray([int(plan["selected_seed"])], dtype=np.int64)
    trajectory["timing_scale"] = np.asarray([float(plan["timing_scale"])], dtype=np.float32)
    return trajectory, True, False


def run_plan_only(args: argparse.Namespace) -> None:
    planner = Status1TaskPlanner(args)
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
    shard, output_path, async_video_path, async_metadata_path, async_h5_path = chained.resolve_shard_paths(
        outdir,
        args.output_name,
        shard_id=args.shard_id,
    )
    if not args.overwrite:
        existing_outputs = [path for path in (output_path, async_video_path, async_metadata_path, async_h5_path) if path.exists()]
        if existing_outputs:
            raise FileExistsError(
                "Refusing to overwrite existing shard file(s): "
                + ", ".join(str(path) for path in existing_outputs)
                + ". Pass --overwrite to replace them."
            )

    compression = None if args.compression == "none" else args.compression
    planner = Status1TaskPlanner(args)
    robot = make_robot(args)
    camera = chained.make_camera(args)
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
            camera.start_async_recording(str(async_video_path), str(async_metadata_path), str(async_h5_path), "lzf")
        move_robot_to_planner_home(robot, planner, args)
        sample_step_info = planner.step_info(robot, planner.home_task, elapsed_time=0.0)
        sample_frame_packet = camera.read_frame() if camera is not None else None
        sample_frame = None if sample_frame_packet is None else np.asarray(sample_frame_packet["pixels"], dtype=np.uint8)
        obs_dim = int(sample_step_info["observation"].shape[0])
        qpos_dim = int(sample_step_info["qpos"].shape[0])
        control_dim = int(sample_step_info["control"].shape[0])

        with h5py.File(output_path, "w") as h5:
            h5.attrs["format"] = "stable_worldmodel_hdf5"
            h5.attrs["source"] = "rope/data/rope_real_chained_status1_data_gen_cl.py"
            h5.attrs["hardware"] = True
            h5.attrs["shard"] = shard
            h5.attrs["robot_backend"] = args.robot_backend
            h5.attrs["arm_mapping"] = "kuka-arm0-status1-only"
            h5.attrs["trajectory_type"] = "status1_home_anchored_chained_cubic_splines"
            h5.attrs["trajectory_mode"] = args.trajectory_mode
            h5.attrs["fixed_execution_controller"] = args.fixed_execution_controller
            h5.attrs["fixed_task_start"] = json.dumps(
                [] if args.fixed_task_start is None else [float(value) for value in args.fixed_task_start]
            )
            h5.attrs["fixed_task_goal"] = json.dumps(
                [] if args.fixed_task_goal is None else [float(value) for value in args.fixed_task_goal]
            )
            h5.attrs["fixed_task_epsilon"] = args.fixed_task_epsilon
            h5.attrs["seed"] = args.seed
            h5.attrs["num_splines"] = args.num_splines
            h5.attrs["requested_segment_duration"] = args.segment_duration
            h5.attrs["control_timestep"] = args.control_timestep
            h5.attrs["home_move_duration"] = args.home_move_duration
            h5.attrs["midpoint_inflation_scale"] = args.midpoint_inflation_scale
            h5.attrs["min_task_knot_distance"] = args.min_task_knot_distance
            h5.attrs["drake_publish_period"] = args.drake_publish_period
            h5.attrs["controller"] = (
                "status1_fixed_open_loop_joint_path"
                if args.trajectory_mode == "fixed-point-to-point" and args.fixed_execution_controller == "open-loop"
                else "status1_task_space_closed_loop_ik"
            )
            h5.attrs["task_cl_kp"] = json.dumps([float(value) for value in args.task_cl_kp])
            h5.attrs["task_cl_max_correction_m"] = json.dumps([float(value) for value in args.task_cl_max_correction_m])
            h5.attrs["max_real_joint_speed_deg_s"] = args.max_real_joint_speed_deg_s
            h5.attrs["max_real_joint_accel_deg_s2"] = args.max_real_joint_accel_deg_s2
            h5.attrs["orientation_constraint_mode"] = args.orientation_constraint_mode
            h5.attrs["orientation_theta_bound_deg"] = args.orientation_theta_bound_deg
            h5.attrs["fixed_a7_deg"] = np.nan if args.fixed_a7_deg is None else args.fixed_a7_deg
            h5.attrs["fixed_a7_tolerance_deg"] = args.fixed_a7_tolerance_deg
            h5.attrs["fixed_start_ready_tolerance_deg"] = args.fixed_start_ready_tolerance_deg
            h5.attrs["print_ik_checks"] = args.print_ik_checks
            h5.attrs["camera_enabled"] = args.enable_camera
            h5.attrs["camera_backend"] = args.camera_backend
            h5.attrs["camera_index"] = args.camera_index
            h5.attrs["camera_device"] = "" if args.camera_device is None else args.camera_device
            h5.attrs["camera_topic"] = "" if args.camera_topic is None else args.camera_topic
            h5.attrs["camera_transport"] = args.camera_transport
            h5.attrs["async_camera_backup_video"] = str(async_video_path) if camera is not None else ""
            h5.attrs["async_camera_backup_metadata"] = str(async_metadata_path) if camera is not None else ""
            h5.attrs["async_camera_backup_h5"] = str(async_h5_path) if camera is not None else ""
            h5.attrs["compression"] = args.compression
            h5.attrs["observation_dim"] = obs_dim
            h5.attrs["action_dim"] = ACTION_DIM
            h5.attrs["qpos_dim"] = qpos_dim
            h5.attrs["qvel_dim"] = qpos_dim
            h5.attrs["control_dim"] = control_dim
            h5.attrs["torque_control_dim"] = 7
            h5.attrs["time_semantics"] = "planned_sample_time_seconds_from_trajectory_start"
            h5.attrs["action_semantics"] = "planned_task_delta"
            h5.attrs["planned_action_semantics"] = "task_path_diff"
            h5.attrs["measured_action_semantics"] = "measured_task_target_diff"
            h5.attrs["torque_control_semantics"] = "IIWA_STATUS torque_commanded for IIWA_STATUS / arm0, sign convention from iiwa_hardware"
            h5.attrs["torque_control_delta_semantics"] = "sample-to-sample diff of torque_control; first row is NaN because no previous sample exists"
            h5.attrs["home_q0_deg"] = json.dumps([float(value) for value in args.home_q0_deg])
            h5.attrs["home_task_state"] = json.dumps(planner.home_task.tolist())
            h5.attrs["hardware_home_qpos"] = json.dumps(planner.hardware_home_qpos.tolist())
            h5.attrs["initial_dataset_home_qpos"] = json.dumps(planner.dataset_home_qpos.tolist())
            h5.attrs["task_bounds"] = json.dumps(
                {"reach": list(planner.task_bounds.reach), "height": list(planner.task_bounds.height), "width": list(planner.task_bounds.width)}
            )

            ep_len_ds = chained.create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
            ep_offset_ds = chained.create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
            reward_ds = chained.create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
            seed_ds = chained.create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
            selected_seed_ds = chained.create_resizable_dataset(h5, "selected_episode_seed", (), np.int64, chunks=True)
            terminated_ds = chained.create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
            truncated_ds = chained.create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
            timing_scale_ds = chained.create_resizable_dataset(h5, "timing_scale", (), np.float32, chunks=True)
            pixels_ds = (
                chained.create_resizable_dataset(h5, "pixels", sample_frame.shape, np.uint8, compression=compression, chunks=(1, *sample_frame.shape))
                if sample_frame is not None
                else None
            )
            action_ds = chained.create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
            planned_action_ds = chained.create_resizable_dataset(h5, "planned_action", (ACTION_DIM,), np.float32, chunks=True)
            measured_action_ds = chained.create_resizable_dataset(h5, "measured_action", (ACTION_DIM,), np.float32, chunks=True)
            obs_ds = chained.create_resizable_dataset(h5, "observation", (obs_dim,), np.float32, chunks=True)
            task_target_ds = chained.create_resizable_dataset(h5, "task_target", (3,), np.float32, chunks=True)
            measured_task_target_ds = chained.create_resizable_dataset(h5, "measured_task_target", (3,), np.float32, chunks=True)
            qpos_ds = chained.create_resizable_dataset(h5, "qpos", (qpos_dim,), np.float32, chunks=True)
            qvel_ds = chained.create_resizable_dataset(h5, "qvel", (qpos_dim,), np.float32, chunks=True)
            control_ds = chained.create_resizable_dataset(h5, "control", (control_dim,), np.float32, chunks=True)
            torque_control_ds = chained.create_resizable_dataset(h5, "torque_control", (7,), np.float32, chunks=True)
            torque_control_delta_ds = chained.create_resizable_dataset(h5, "torque_control_delta", (7,), np.float32, chunks=True)
            torque_measured_ds = chained.create_resizable_dataset(h5, "torque_measured", (7,), np.float32, chunks=True)
            torque_external_ds = chained.create_resizable_dataset(h5, "torque_external", (7,), np.float32, chunks=True)
            left_attachment_pos_ds = chained.create_resizable_dataset(h5, "left_attachment_pos", (3,), np.float32, chunks=True)
            time_ds = chained.create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
            sample_monotonic_time_ds = chained.create_resizable_dataset(h5, "sample_monotonic_time", (1,), np.float64, chunks=True)
            state_monotonic_time_ds = chained.create_resizable_dataset(h5, "state_monotonic_time", (1,), np.float64, chunks=True)
            camera_frame_time_ds = chained.create_resizable_dataset(h5, "camera_frame_time", (1,), np.float64, chunks=True) if sample_frame is not None else None
            camera_receive_time_ds = chained.create_resizable_dataset(h5, "camera_receive_time", (1,), np.float64, chunks=True) if sample_frame is not None else None
            camera_header_time_ds = chained.create_resizable_dataset(h5, "camera_header_time", (1,), np.float64, chunks=True) if sample_frame is not None else None
            camera_frame_index_ds = chained.create_resizable_dataset(h5, "camera_frame_index", (1,), np.int64, chunks=True) if sample_frame is not None else None
            episode_idx_ds = chained.create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
            step_idx_ds = chained.create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

            progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
            progress_desc = "Collecting status1 transitions" if args.target_transitions is not None else "Collecting status1 trajectories"
            progress_unit = "step" if args.target_transitions is not None else "traj"
            with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
                while chained.should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
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
                    frame_count = trajectory["observation"].shape[0]
                    padded_actions = np.empty((frame_count, ACTION_DIM), dtype=np.float32)
                    padded_actions[:-1] = trajectory["action"]
                    padded_actions[-1] = np.nan
                    padded_planned_actions = np.empty((frame_count, ACTION_DIM), dtype=np.float32)
                    padded_planned_actions[:-1] = trajectory["planned_action"]
                    padded_planned_actions[-1] = np.nan
                    padded_measured_actions = np.empty((frame_count, ACTION_DIM), dtype=np.float32)
                    padded_measured_actions[:-1] = trajectory["measured_action"]
                    padded_measured_actions[-1] = np.nan
                    offset = chained.append_rows(pixels_ds, trajectory["pixels"])[0] if pixels_ds is not None else int(obs_ds.shape[0])
                    chained.append_rows(obs_ds, trajectory["observation"])
                    chained.append_rows(action_ds, padded_actions)
                    chained.append_rows(planned_action_ds, padded_planned_actions)
                    chained.append_rows(measured_action_ds, padded_measured_actions)
                    chained.append_rows(task_target_ds, trajectory["task_target"])
                    chained.append_rows(measured_task_target_ds, trajectory["measured_task_target"])
                    chained.append_rows(qpos_ds, trajectory["qpos"])
                    chained.append_rows(qvel_ds, trajectory["qvel"])
                    chained.append_rows(control_ds, trajectory["control"])
                    chained.append_rows(torque_control_ds, trajectory["torque_control"])
                    chained.append_rows(torque_control_delta_ds, trajectory["torque_control_delta"])
                    chained.append_rows(torque_measured_ds, trajectory["torque_measured"])
                    chained.append_rows(torque_external_ds, trajectory["torque_external"])
                    chained.append_rows(left_attachment_pos_ds, trajectory["left_attachment_pos"])
                    chained.append_rows(time_ds, trajectory["time"])
                    chained.append_rows(sample_monotonic_time_ds, trajectory["sample_monotonic_time"])
                    chained.append_rows(state_monotonic_time_ds, trajectory["state_monotonic_time"])
                    if camera_frame_time_ds is not None:
                        chained.append_rows(camera_frame_time_ds, trajectory["camera_frame_time"])
                        chained.append_rows(camera_receive_time_ds, trajectory["camera_receive_time"])
                        chained.append_rows(camera_header_time_ds, trajectory["camera_header_time"])
                        chained.append_rows(camera_frame_index_ds, trajectory["camera_frame_index"])
                    chained.append_rows(episode_idx_ds, np.full((frame_count,), episode_idx, dtype=np.int64))
                    chained.append_rows(step_idx_ds, np.arange(frame_count, dtype=np.int64))
                    chained.append_rows(ep_len_ds, np.asarray([frame_count], dtype=np.int64))
                    chained.append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                    chained.append_rows(reward_ds, np.asarray([0.0], dtype=np.float32))
                    chained.append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                    chained.append_rows(selected_seed_ds, trajectory["selected_seed"])
                    chained.append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                    chained.append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))
                    chained.append_rows(timing_scale_ds, trajectory["timing_scale"])
                    rewards.append(0.0)
                    step_counts.append(num_actions)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    progress.update(num_actions if args.target_transitions is not None else 1)
                    progress.set_postfix(episodes=len(step_counts), transitions=int(np.sum(step_counts, dtype=np.int64)))

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
