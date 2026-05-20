from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
import time
from pathlib import Path

import numpy as np


def _as_qpos14(values: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (14,):
        raise ValueError(f"Expected {name} with shape (14,), got {array.shape}.")
    return array


def _smoothstep5(alpha: float) -> float:
    s = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


@dataclass
class DrakeLCMBimanualRobotBackend:
    """Bimanual KUKA iiwa backend through Drake's IIWA LCM station.

    The collector's 14D convention is [left_q(7), right_q(7)]. The station's
    physical convention is robot0 on IIWA_COMMAND/STATUS and robot1 on the _2
    channels. With the default mapping, robot0 is left and robot1 is right.
    """

    arm_mapping: str = "robot0-left"
    publish_period: float = 0.005
    status_timeout: float = 5.0
    max_control_joint_step: float = math.radians(5.0)
    max_reset_joint_move: float = math.radians(90.0)
    hold_duration: float = 0.1
    last_commanded_qpos: np.ndarray | None = None

    _diagram: object | None = field(default=None, init=False)
    _context: object | None = field(default=None, init=False)
    _simulator: object | None = field(default=None, init=False)
    _position_inputs: list[object] = field(default_factory=list, init=False)
    _position_outputs: list[object] = field(default_factory=list, init=False)
    _velocity_outputs: list[object] = field(default_factory=list, init=False)

    def connect(self) -> None:
        if self.arm_mapping not in {"robot0-left", "robot1-left"}:
            raise ValueError(f"Unsupported arm mapping: {self.arm_mapping!r}.")
        if self.publish_period <= 0.0:
            raise ValueError("publish_period must be positive.")
        measured = self._read_initial_status()
        self._build_station()
        self.last_commanded_qpos = measured.copy()
        self._publish_path(measured, measured, duration=self.hold_duration)

    def close(self) -> None:
        self._diagram = None
        self._context = None
        self._simulator = None
        self._position_inputs.clear()
        self._position_outputs.clear()
        self._velocity_outputs.clear()

    def read_qpos_14(self) -> np.ndarray:
        self._require_connected()
        q0 = np.asarray(self._position_outputs[0].Eval(self._context), dtype=np.float64).reshape(7)
        q1 = np.asarray(self._position_outputs[1].Eval(self._context), dtype=np.float64).reshape(7)
        return self._from_station_order(np.concatenate([q0, q1]))

    def read_qvel_14(self) -> np.ndarray:
        self._require_connected()
        v0 = np.asarray(self._velocity_outputs[0].Eval(self._context), dtype=np.float64).reshape(7)
        v1 = np.asarray(self._velocity_outputs[1].Eval(self._context), dtype=np.float64).reshape(7)
        return self._from_station_order(np.concatenate([v0, v1]))

    def command_joint_positions(self, qpos_14: np.ndarray, *, duration: float, blocking: bool) -> None:
        del blocking  # Drake LCM commands are published by advancing the simulator.
        target = _as_qpos14(qpos_14, name="qpos_14")
        start = self.last_commanded_qpos.copy() if self.last_commanded_qpos is not None else self.read_qpos_14()
        self._check_joint_delta(start, target, duration=duration)
        self._publish_path(start, target, duration=duration)
        self.last_commanded_qpos = target.copy()

    def stop(self) -> None:
        if self._simulator is None:
            return
        try:
            measured = self.read_qpos_14()
            self._publish_path(measured, measured, duration=self.hold_duration)
            self.last_commanded_qpos = measured.copy()
        except Exception:
            pass

    def _build_station(self) -> None:
        self._ensure_data_dir_on_path()
        from pydrake.all import IiwaControlMode, Simulator
        from rope.data.iiwa_hardware import MakeFullBimanualStation

        self._diagram = MakeFullBimanualStation(control_mode=IiwaControlMode.kPositionOnly)
        self._context = self._diagram.CreateDefaultContext()
        self._simulator = Simulator(self._diagram, self._context)
        self._simulator.set_target_realtime_rate(1.0)
        self._position_inputs = [
            self._diagram.GetInputPort("position_0"),
            self._diagram.GetInputPort("position_1"),
        ]
        self._position_outputs = [
            self._diagram.GetOutputPort("position_measured_0"),
            self._diagram.GetOutputPort("position_measured_1"),
        ]
        self._velocity_outputs = [
            self._diagram.GetOutputPort("velocity_estimated_0"),
            self._diagram.GetOutputPort("velocity_estimated_1"),
        ]

    def _read_initial_status(self) -> np.ndarray:
        from drake import lcmt_iiwa_status
        from pydrake.all import DrakeLcm

        lcm = DrakeLcm()
        q: dict[str, np.ndarray | None] = {"": None, "_2": None}

        def make_handler(suffix: str):
            def handler(data):
                msg = lcmt_iiwa_status.decode(data)
                q[suffix] = np.asarray(msg.joint_position_measured, dtype=np.float64).reshape(7)

            return handler

        lcm.Subscribe("IIWA_STATUS", make_handler(""))
        lcm.Subscribe("IIWA_STATUS_2", make_handler("_2"))

        start = time.time()
        while time.time() - start < self.status_timeout:
            lcm.HandleSubscriptions(100)
            if q[""] is not None and q["_2"] is not None:
                return self._from_station_order(np.concatenate([q[""], q["_2"]]))
        raise RuntimeError(f"Failed to receive IIWA_STATUS and IIWA_STATUS_2 within {self.status_timeout} s.")

    def _publish_path(self, start: np.ndarray, target: np.ndarray, *, duration: float) -> None:
        self._require_connected()
        start = _as_qpos14(start, name="start")
        target = _as_qpos14(target, name="target")
        duration = max(float(duration), 0.0)
        steps = max(1, int(math.ceil(duration / self.publish_period)))
        step_dt = duration / steps if duration > 0.0 else self.publish_period
        for step in range(1, steps + 1):
            alpha = _smoothstep5(step / steps)
            command = start + alpha * (target - start)
            self._fix_position_inputs(command)
            self._simulator.AdvanceTo(self._context.get_time() + step_dt)

    def _fix_position_inputs(self, collector_qpos: np.ndarray) -> None:
        station_qpos = self._to_station_order(collector_qpos)
        self._position_inputs[0].FixValue(self._context, station_qpos[:7])
        self._position_inputs[1].FixValue(self._context, station_qpos[7:])

    def _check_joint_delta(self, start: np.ndarray, target: np.ndarray, *, duration: float) -> None:
        max_delta = float(np.max(np.abs(target - start)))
        limit = self.max_reset_joint_move if duration > 0.5 else self.max_control_joint_step
        if max_delta > limit:
            raise RuntimeError(
                "Refusing unsafe joint command: "
                f"max_delta={math.degrees(max_delta):.2f} deg exceeds "
                f"limit={math.degrees(limit):.2f} deg."
            )

    def _to_station_order(self, collector_qpos: np.ndarray) -> np.ndarray:
        qpos = _as_qpos14(collector_qpos, name="collector_qpos")
        if self.arm_mapping == "robot0-left":
            return qpos.copy()
        return np.concatenate([qpos[7:], qpos[:7]])

    def _from_station_order(self, station_qpos: np.ndarray) -> np.ndarray:
        qpos = _as_qpos14(station_qpos, name="station_qpos")
        if self.arm_mapping == "robot0-left":
            return qpos.copy()
        return np.concatenate([qpos[7:], qpos[:7]])

    def _require_connected(self) -> None:
        if self._simulator is None or self._context is None or self._diagram is None:
            raise RuntimeError("DrakeLCMBimanualRobotBackend is not connected.")

    @staticmethod
    def _ensure_data_dir_on_path() -> None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))

