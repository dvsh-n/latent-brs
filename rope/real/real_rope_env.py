from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from rope.real.collision_guard import MujocoArmCollisionGuard
from rope.real.interfaces import CameraBackend, RobotBackend
from rope.shared.lab_env import LabEnv, TaskState


@dataclass
class RealRopeEnv:
    robot: RobotBackend
    camera: CameraBackend
    command_duration: float = 0.05
    reset_duration: float = 3.0
    enable_collision_guard: bool = True
    arm_arm_min_distance: float = 0.06
    collision_control_samples: int = 5
    collision_reset_samples: int = 25

    def __post_init__(self) -> None:
        self.ik_env = LabEnv()
        self.collision_guard = (
            MujocoArmCollisionGuard(
                self.ik_env,
                min_arm_arm_distance=self.arm_arm_min_distance,
                control_path_samples=self.collision_control_samples,
                reset_path_samples=self.collision_reset_samples,
            )
            if self.enable_collision_guard
            else None
        )
        self.current_task_target = self.ik_env.nominal_state.as_array().astype(np.float64)

    @property
    def task_bounds(self):
        return self.ik_env.task_bounds

    @property
    def nominal_state(self):
        return self.ik_env.nominal_state

    def connect(self) -> None:
        self.robot.connect()
        self.camera.connect()
        self.current_task_target = self.ik_env.nominal_state.as_array().astype(np.float64)

    def close(self) -> None:
        self.camera.close()
        self.robot.close()

    def stop(self) -> None:
        self.robot.stop()

    def clip_task_target(self, task_state: TaskState | np.ndarray | list[float]) -> np.ndarray:
        return self.task_bounds.clip(task_state).as_array().astype(np.float64)

    def solve_task_to_joints(self, task_state: TaskState | np.ndarray | list[float]) -> np.ndarray:
        measured = self.robot.read_qpos_14()
        self.ik_env.set_arm_joint_positions(measured)
        self.ik_env.set_task_target(task_state)
        return self.ik_env.joint_controller.target.copy()

    def reset(self, task_state: TaskState | np.ndarray | list[float] | None = None) -> None:
        desired = self.nominal_state.as_array() if task_state is None else self.clip_task_target(task_state)
        q_cmd = self.solve_task_to_joints(desired)
        self.validate_command_path(q_cmd, duration=self.reset_duration, label="reset")
        self.robot.command_joint_positions(q_cmd, duration=self.reset_duration, blocking=True)
        self.current_task_target = desired.astype(np.float64)
        time.sleep(0.1)

    def apply_task_delta(self, delta: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
        delta_array = np.asarray(delta, dtype=np.float64)
        if delta_array.shape != (3,):
            raise ValueError(f"Expected 3D task delta, got shape {delta_array.shape}.")
        previous = self.current_task_target.copy()
        desired = self.clip_task_target(previous + delta_array)
        q_cmd = self.solve_task_to_joints(desired)
        self.validate_command_path(q_cmd, duration=self.command_duration, label="control")
        self.robot.command_joint_positions(q_cmd, duration=self.command_duration, blocking=True)
        self.current_task_target = desired
        return (desired - previous).astype(np.float32)

    def validate_command_path(self, q_cmd: np.ndarray, *, duration: float, label: str) -> None:
        if self.collision_guard is None:
            return
        measured = self.robot.read_qpos_14()
        self.collision_guard.validate_path(measured, q_cmd, duration=duration, label=f"{label}: measured-to-target")
        if self.robot.last_commanded_qpos is not None and not np.allclose(self.robot.last_commanded_qpos, measured):
            self.collision_guard.validate_path(
                self.robot.last_commanded_qpos,
                q_cmd,
                duration=duration,
                label=f"{label}: commanded-to-target",
            )

    def get_attachment_positions(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.ik_env.set_arm_joint_positions(qpos)
        left = self.ik_env.data.site_xpos[self.ik_env.arm1_site_id].copy()
        right = self.ik_env.data.site_xpos[self.ik_env.arm2_site_id].copy()
        return left.astype(np.float32), right.astype(np.float32)

    def get_step_info(self, *, elapsed_time: float) -> dict[str, np.ndarray]:
        qpos = self.robot.read_qpos_14().astype(np.float32)
        qvel = self.robot.read_qvel_14().astype(np.float32)
        if self.robot.last_commanded_qpos is None:
            control = qpos.copy()
        else:
            control = self.robot.last_commanded_qpos.astype(np.float32)
        left_pos, right_pos = self.get_attachment_positions(qpos)
        rope_length = np.asarray([np.linalg.norm(left_pos - right_pos)], dtype=np.float32)
        target = self.current_task_target.astype(np.float32)
        observation = np.concatenate(
            [
                target,
                qpos,
                qvel,
                control,
                left_pos,
                right_pos,
                rope_length,
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "observation": observation,
            "task_target": target,
            "qpos": qpos,
            "qvel": qvel,
            "control": control,
            "left_attachment_pos": left_pos,
            "right_attachment_pos": right_pos,
            "rope_length": rope_length,
            "time": np.asarray([elapsed_time], dtype=np.float32),
        }

    def get_rgb_frame(self) -> np.ndarray:
        return self.camera.read_rgb_224()
