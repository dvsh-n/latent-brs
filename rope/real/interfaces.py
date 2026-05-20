from __future__ import annotations

from typing import Protocol

import numpy as np


class RobotBackend(Protocol):
    last_commanded_qpos: np.ndarray | None

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def read_qpos_14(self) -> np.ndarray:
        ...

    def read_qvel_14(self) -> np.ndarray:
        ...

    def command_joint_positions(self, qpos_14: np.ndarray, *, duration: float, blocking: bool) -> None:
        ...

    def stop(self) -> None:
        ...


class CameraBackend(Protocol):
    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def read_rgb_224(self) -> np.ndarray:
        ...

