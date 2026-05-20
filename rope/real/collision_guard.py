from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from rope.shared.lab_env import LabEnv


def _smoothstep5(alpha: float) -> float:
    s = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


@dataclass
class MujocoArmCollisionGuard:
    env: LabEnv
    min_arm_arm_distance: float = 0.06
    control_path_samples: int = 5
    reset_path_samples: int = 25
    _arm1_geoms: list[int] = field(default_factory=list, init=False)
    _arm2_geoms: list[int] = field(default_factory=list, init=False)
    _fromto: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64), init=False)

    def __post_init__(self) -> None:
        if self.min_arm_arm_distance < 0.0:
            raise ValueError("min_arm_arm_distance cannot be negative.")
        if self.control_path_samples < 1:
            raise ValueError("control_path_samples must be positive.")
        if self.reset_path_samples < 1:
            raise ValueError("reset_path_samples must be positive.")
        self._arm1_geoms = self._collision_geoms("arm1_")
        self._arm2_geoms = self._collision_geoms("arm2_")
        if not self._arm1_geoms or not self._arm2_geoms:
            raise RuntimeError("Could not find collision geoms for both arms.")

    def validate_path(self, start_qpos: np.ndarray, target_qpos: np.ndarray, *, duration: float, label: str) -> None:
        start = self._as_qpos(start_qpos, name="start_qpos")
        target = self._as_qpos(target_qpos, name="target_qpos")
        samples = self.reset_path_samples if duration > 0.5 else self.control_path_samples
        for index in range(samples + 1):
            alpha = _smoothstep5(index / max(samples, 1))
            qpos = start + alpha * (target - start)
            self.validate_qpos(qpos, label=f"{label} sample {index}/{samples}")

    def validate_sequence(self, qpos_sequence: np.ndarray, *, segment_duration: float, label: str) -> None:
        path = np.asarray(qpos_sequence, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 14:
            raise ValueError(f"Expected {label} path shape (N, 14), got {path.shape}.")
        if path.shape[0] < 1:
            raise ValueError(f"Expected {label} path to contain at least one waypoint.")
        self.validate_qpos(path[0], label=f"{label} waypoint 0")
        for index in range(1, path.shape[0]):
            self.validate_path(
                path[index - 1],
                path[index],
                duration=segment_duration,
                label=f"{label} segment {index - 1}->{index}",
            )

    def validate_qpos(self, qpos: np.ndarray, *, label: str) -> None:
        self.env.set_arm_joint_positions(self._as_qpos(qpos, name="qpos"))
        self._reject_cross_arm_contacts(label)
        min_distance, geom1, geom2 = self._min_cross_arm_distance()
        if min_distance < self.min_arm_arm_distance:
            name1 = self._geom_label(geom1)
            name2 = self._geom_label(geom2)
            raise RuntimeError(
                "Refusing unsafe arm-arm proximity: "
                f"{label}, min_distance={min_distance:.3f} m below "
                f"limit={self.min_arm_arm_distance:.3f} m between {name1} and {name2}."
            )

    def _reject_cross_arm_contacts(self, label: str) -> None:
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            body1 = self.env.model.body(self.env.model.geom_bodyid[contact.geom1]).name
            body2 = self.env.model.body(self.env.model.geom_bodyid[contact.geom2]).name
            if (body1.startswith("arm1_") and body2.startswith("arm2_")) or (
                body1.startswith("arm2_") and body2.startswith("arm1_")
            ):
                raise RuntimeError(
                    "Refusing arm-arm contact: "
                    f"{label}, {self._geom_label(contact.geom1)} vs {self._geom_label(contact.geom2)}, "
                    f"dist={contact.dist:.4f} m."
                )

    def _min_cross_arm_distance(self) -> tuple[float, int, int]:
        best_distance = math.inf
        best_pair = (-1, -1)
        for geom1 in self._arm1_geoms:
            for geom2 in self._arm2_geoms:
                distance = float(mujoco.mj_geomDistance(self.env.model, self.env.data, geom1, geom2, 10.0, self._fromto))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (geom1, geom2)
        return best_distance, best_pair[0], best_pair[1]

    def _collision_geoms(self, body_prefix: str) -> list[int]:
        geoms = []
        for geom_id in range(self.env.model.ngeom):
            body_name = self.env.model.body(self.env.model.geom_bodyid[geom_id]).name
            if not body_name.startswith(body_prefix):
                continue
            if int(self.env.model.geom_contype[geom_id]) == 0 and int(self.env.model.geom_conaffinity[geom_id]) == 0:
                continue
            geoms.append(geom_id)
        return geoms

    def _geom_label(self, geom_id: int) -> str:
        geom_name = self.env.model.geom(geom_id).name or f"geom{geom_id}"
        body_name = self.env.model.body(self.env.model.geom_bodyid[geom_id]).name
        return f"{body_name}/{geom_name}"

    @staticmethod
    def _as_qpos(values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (14,):
            raise ValueError(f"Expected {name} shape (14,), got {array.shape}.")
        return array
