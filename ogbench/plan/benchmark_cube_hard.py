#!/usr/bin/env python3
"""Hard OGBench cube benchmark for iLQR and stable-worldmodel policies.

Every method starts from episode step 0, uses the final episode state/frame as
the goal, receives the same oracle grasp phase, and is then evaluated on cube
position success. Yaw and latent distances are logged as diagnostics only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import site
import sys
import time
import types
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    message=r".*Box low's precision lowered by casting to float32.*",
    category=UserWarning,
    module=r"gymnasium\.spaces\.box",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Box high's precision lowered by casting to float32.*",
    category=UserWarning,
    module=r"gymnasium\.spaces\.box",
)

ROOT_DIR = Path(__file__).resolve().parents[2]
STABLE_WORLDMODEL_DIR = ROOT_DIR / "third_party" / "stable-worldmodel"
for path in (ROOT_DIR, STABLE_WORLDMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm


def ensure_ogbench_manipspace_importable() -> None:
    """Let local `ogbench` coexist with the installed OGBench package."""

    import ogbench as local_ogbench

    for base in site.getsitepackages() + [site.getusersitepackages()]:
        package_dir = Path(base) / "ogbench"
        if package_dir.is_dir() and str(package_dir) not in local_ogbench.__path__:
            local_ogbench.__path__.append(str(package_dir))


ensure_ogbench_manipspace_importable()


def load_local_cube_training_module() -> Any:
    module_path = ROOT_DIR / "ogbench" / "train" / "lewm_train_mlp_markov.py"
    module_name = "_latent_brs_ogbench_lewm_train_mlp_markov"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local cube training module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    # Older cube checkpoints were pickled under the pre-merge `ogbench_cube`
    # package name. Register a narrow alias so torch.load can resolve them.
    import reacher.shared.models as shared_models

    ogbench_cube_pkg = sys.modules.setdefault("ogbench_cube", types.ModuleType("ogbench_cube"))
    ogbench_cube_pkg.__path__ = []  # type: ignore[attr-defined]
    ogbench_cube_train_pkg = sys.modules.setdefault("ogbench_cube.train", types.ModuleType("ogbench_cube.train"))
    ogbench_cube_train_pkg.__path__ = []  # type: ignore[attr-defined]
    ogbench_cube_shared_pkg = sys.modules.setdefault("ogbench_cube.shared", types.ModuleType("ogbench_cube.shared"))
    ogbench_cube_shared_pkg.__path__ = []  # type: ignore[attr-defined]
    setattr(ogbench_cube_pkg, "train", ogbench_cube_train_pkg)
    setattr(ogbench_cube_pkg, "shared", ogbench_cube_shared_pkg)
    setattr(ogbench_cube_train_pkg, "mlpdyn_train", module)
    setattr(ogbench_cube_shared_pkg, "models", shared_models)
    sys.modules.setdefault("ogbench_cube.train.mlpdyn_train", module)
    sys.modules.setdefault("ogbench_cube.shared.models", shared_models)
    if hasattr(module, "LeWMOGBenchDataset") and not hasattr(module, "LeWMOGBenchCubeDataset"):
        module.LeWMOGBenchCubeDataset = module.LeWMOGBenchDataset
    return module


_LOCAL_CUBE_TRAINING = load_local_cube_training_module()
LeWMOGBenchDataset = _LOCAL_CUBE_TRAINING.LeWMOGBenchDataset

import gymnasium
import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle


DEFAULT_DATASET_PATH = "ogbench/data/expert_data/ogbench_cube_expert.h5"
DEFAULT_MODEL_DIR = "ogbench/models/mlpdyn"
DEFAULT_OUT_DIR = "ogbench/plan/ogbench_cube_hard_eval"
DEFAULT_SOLVER_CONFIG = (
    STABLE_WORLDMODEL_DIR / "scripts" / "plan" / "config" / "solver" / "cem.yaml"
)
DEFAULT_NUM_EVAL = 50
DEFAULT_EVAL_BUDGET = 120
DEFAULT_SEED = 42
DEFAULT_CUBE_SUCCESS_THRESHOLD = 0.04
DEFAULT_SWM_HISTORY_SIZE = 3

DEVICE = "auto"
HORIZON = 15
Q_TERMINAL = 5.0
Q_STAGE = 0.005
R_CONTROL = 0.1
VIDEO_FPS = 20
MAX_ORACLE_STEPS = 80
ORACLE_SEGMENT_DT = 0.4
ORACLE_NOISE = 0.0
ORACLE_NOISE_SMOOTHING = 0.5
GRASP_CONTACT_THRESHOLD = 0.5
GRASP_ALIGNMENT_THRESHOLD = 0.03


@dataclass(frozen=True)
class EvalCase:
    episode_idx: int
    start_step: int
    goal_step: int
    ep_len: int


class SingleVectorEnvAdapter:
    def __init__(self, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=action_space.low[None, ...],
            high=action_space.high[None, ...],
            dtype=action_space.dtype,
        )


class CubeEpisodeHistory:
    def __init__(
        self,
        *,
        history_size: int,
        action_dim: int,
        case_id: int,
        goal_frame: np.ndarray,
        goal_obs: np.ndarray,
        goal_qpos: np.ndarray,
        goal_qvel: np.ndarray,
        goal_block_pos: np.ndarray,
        goal_block_yaw: np.ndarray,
        goal_step: int,
    ) -> None:
        self.history_size = int(history_size)
        self.action_dim = int(action_dim)
        self.case_id = int(case_id)
        self.goal_frame = np.asarray(goal_frame, dtype=np.uint8)
        self.goal_obs = np.asarray(goal_obs, dtype=np.float32)
        self.goal_qpos = np.asarray(goal_qpos, dtype=np.float32)
        self.goal_qvel = np.asarray(goal_qvel, dtype=np.float32)
        self.goal_block_pos = np.asarray(goal_block_pos, dtype=np.float32)
        self.goal_block_yaw = np.asarray(goal_block_yaw, dtype=np.float32).reshape(-1)
        self.goal_step = int(goal_step)
        self.frames: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.actions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.observations: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qpos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qvel: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.block_pos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.block_yaw: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.step_idx: deque[int] = deque(maxlen=self.history_size)

    def reset(self, *, frame: np.ndarray, info: dict[str, Any], observation: np.ndarray, step_idx: int) -> None:
        zero_action = np.zeros((self.action_dim,), dtype=np.float32)
        for _ in range(self.history_size):
            self.append(
                frame=frame,
                action=zero_action,
                info=info,
                observation=observation,
                step_idx=step_idx,
            )

    def append(
        self,
        *,
        frame: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any],
        observation: np.ndarray,
        step_idx: int,
    ) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.qpos.append(np.asarray(info["qpos"], dtype=np.float32).copy())
        self.qvel.append(np.asarray(info["qvel"], dtype=np.float32).copy())
        self.block_pos.append(np.asarray(info["privileged/block_0_pos"], dtype=np.float32).copy())
        self.block_yaw.append(np.asarray(info["privileged/block_0_yaw"], dtype=np.float32).reshape(-1).copy())
        self.step_idx.append(int(step_idx))

    def info(self) -> dict[str, np.ndarray]:
        frames = np.stack(list(self.frames), axis=0)
        actions = np.stack(list(self.actions), axis=0)
        observations = np.stack(list(self.observations), axis=0)
        qpos = np.stack(list(self.qpos), axis=0)
        qvel = np.stack(list(self.qvel), axis=0)
        block_pos = np.stack(list(self.block_pos), axis=0)
        block_yaw = np.stack(list(self.block_yaw), axis=0)
        step_idx = np.asarray(list(self.step_idx), dtype=np.int64)
        ids = np.full((self.history_size,), self.case_id, dtype=np.int64)
        goal_step_idx = np.full((self.history_size,), self.goal_step, dtype=np.int64)

        return {
            "pixels": frames[None, ...],
            "goal": np.repeat(self.goal_frame[None, ...], self.history_size, axis=0)[None, ...],
            "action": actions[None, ...],
            "observation": observations[None, ...],
            "goal_observation": np.repeat(self.goal_obs[None, ...], self.history_size, axis=0)[None, ...],
            "qpos": qpos[None, ...],
            "goal_qpos": np.repeat(self.goal_qpos[None, ...], self.history_size, axis=0)[None, ...],
            "qvel": qvel[None, ...],
            "goal_qvel": np.repeat(self.goal_qvel[None, ...], self.history_size, axis=0)[None, ...],
            "block_pos": block_pos[None, ...],
            "goal_block_pos": np.repeat(self.goal_block_pos[None, ...], self.history_size, axis=0)[None, ...],
            "block_yaw": block_yaw[None, ...],
            "goal_block_yaw": np.repeat(self.goal_block_yaw[None, ...], self.history_size, axis=0)[None, ...],
            "step_idx": step_idx[None, ...],
            "goal_step_idx": goal_step_idx[None, ...],
            "id": ids[None, ...],
            "goal_id": ids[None, ...],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("ilqr", "swm_cost", "swm_action"), default="ilqr")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--stats-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-eval", type=int, default=DEFAULT_NUM_EVAL)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument("--cube-success-threshold", type=float, default=DEFAULT_CUBE_SUCCESS_THRESHOLD)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--no-videos", action="store_true")

    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)

    parser.add_argument("--env-max-episode-steps", type=int, default=None)
    parser.add_argument("--max-oracle-steps", type=int, default=MAX_ORACLE_STEPS)
    parser.add_argument("--oracle-segment-dt", type=float, default=ORACLE_SEGMENT_DT)
    parser.add_argument("--oracle-noise", type=float, default=ORACLE_NOISE)
    parser.add_argument("--oracle-noise-smoothing", type=float, default=ORACLE_NOISE_SMOOTHING)
    parser.add_argument("--grasp-contact-threshold", type=float, default=GRASP_CONTACT_THRESHOLD)
    parser.add_argument("--grasp-alignment-threshold", type=float, default=GRASP_ALIGNMENT_THRESHOLD)

    parser.add_argument("--policy", default=None, help="Stable-worldmodel run name, directory, or object checkpoint.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--solver-config", type=Path, default=DEFAULT_SOLVER_CONFIG)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--swm-history-size", type=int, default=None)
    parser.add_argument("--swm-img-size", type=int, default=224)
    parser.add_argument("--process-key", action="append", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


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


def required_markov_history(markov_deriv: int) -> int:
    return int(markov_deriv) + 1


def build_markov_state(history_tensor: torch.Tensor, markov_deriv: int) -> torch.Tensor:
    states = [history_tensor[-1]]
    diffs = history_tensor
    for _ in range(int(markov_deriv)):
        diffs = diffs[1:] - diffs[:-1]
        states.append(diffs[-1])
    return torch.cat(states, dim=-1)


def make_markov_state(history: list[torch.Tensor], markov_deriv: int) -> torch.Tensor:
    context_len = required_markov_history(markov_deriv)
    if not history:
        raise ValueError("At least one embedding is required to build the Markov state.")
    history_tensor = torch.stack(history[-context_len:], dim=0)
    if history_tensor.shape[0] < context_len:
        pad = history_tensor[:1].repeat(context_len - history_tensor.shape[0], 1)
        history_tensor = torch.cat((pad, history_tensor), dim=0)
    return build_markov_state(history_tensor, markov_deriv)


def normalized_to_raw_action(action_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (action_norm * action_std.reshape(-1) + action_mean.reshape(-1)).astype(np.float32)


def action_to_standardized(action: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return ((action.astype(np.float32) - action_mean.reshape(-1)) / action_std.reshape(-1)).astype(np.float32)


def angular_distance(a: float, b: float) -> float:
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def cube_is_grasped(info: dict[str, Any], *, contact_threshold: float, alignment_threshold: float) -> bool:
    target_block = int(info["privileged/target_block"])
    block_pos = np.asarray(info[f"privileged/block_{target_block}_pos"], dtype=np.float32)
    effector_pos = np.asarray(info["proprio/effector_pos"], dtype=np.float32)
    gripper_contact = float(np.asarray(info["proprio/gripper_contact"], dtype=np.float32)[0])
    block_alignment = float(np.linalg.norm(block_pos - effector_pos))
    return bool(gripper_contact >= contact_threshold and block_alignment <= alignment_threshold)


def cube_distance(info: dict[str, Any], goal_block_pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(info["privileged/block_0_pos"], dtype=np.float32) - goal_block_pos))


def cube_yaw_error(info: dict[str, Any], goal_block_yaw: float) -> float:
    return angular_distance(float(info["privileged/block_0_yaw"][0]), goal_block_yaw)


def load_dataset_episode(dataset_path: Path, episode_idx: int) -> dict[str, Any]:
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
            "target_block_pos": np.asarray(h5["target_block_pos"][rows], dtype=np.float32),
            "target_block_yaw": np.asarray(h5["target_block_yaw"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]),
            "env_name": str(h5.attrs.get("env_name", "cube-single-v0")),
            "camera": str(h5.attrs.get("camera", "front_pixels")),
            "width": int(h5["pixels"].shape[2]),
            "height": int(h5["pixels"].shape[1]),
            "physics_timestep": float(h5.attrs.get("physics_timestep", 1.0 / 500.0)),
            "control_timestep": float(h5.attrs.get("control_timestep", 25.0 / 500.0)),
            "max_episode_steps": int(h5.attrs.get("max_episode_steps", ep_len)),
            "video_fps": float(h5.attrs.get("video_fps", 20.0)),
        }


def make_env(
    *,
    env_name: str,
    physics_timestep: float,
    control_timestep: float,
    max_episode_steps: int,
    width: int,
    height: int,
) -> gymnasium.Env:
    return gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=False,
        max_episode_steps=max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=width,
        height=height,
    )


def restore_target_pose(env: gymnasium.Env, *, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    for geom_ids in unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            unwrapped._model.geom(gid).rgba[3] = 0.0


def reset_env_to_state(
    env: gymnasium.Env,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_block_pos: np.ndarray,
    target_block_yaw: float,
    camera: str,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    unwrapped._data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    unwrapped._data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    restore_target_pose(env, target_block_pos=target_block_pos, target_block_yaw=target_block_yaw)
    unwrapped.pre_step()
    mujoco.mj_forward(unwrapped._model, unwrapped._data)
    unwrapped.post_step()
    frame = np.asarray(unwrapped.render(camera=camera), dtype=np.uint8)
    info = unwrapped.get_step_info()
    obs = np.asarray(unwrapped.compute_observation(), dtype=np.float32)
    return frame, info, obs


class MarkovDynamicsTorch:
    def __init__(self, model: torch.nn.Module, state_dim: int, action_dim: int, device: torch.device) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError("Expected one-step Markov MLP dynamics predictor.")
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("Expected identity action encoder.")
        if int(predictor.embed_dim) != state_dim:
            raise ValueError(f"Predictor state dim mismatch: expected {state_dim}, got {predictor.embed_dim}.")
        if int(predictor.action_dim) != action_dim:
            raise ValueError(f"Predictor action dim mismatch: expected {action_dim}, got {predictor.action_dim}.")
        self.net = predictor.net.to(device)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, u), dim=-1))


class ILQRMPCSolver:
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
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)
        self.eye_x = torch.eye(self.state_dim, dtype=torch.float32, device=device)
        self.eye_u = torch.eye(self.action_dim, dtype=torch.float32, device=device)
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)

    def _make_initial_action_guess(self) -> torch.Tensor:
        if self.horizon <= 1:
            return self.prev_u_guess.clone()
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return guess

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            x_curr = self.dynamics.step(x_curr, u_seq[step])
            x_traj[step + 1] = x_curr
        return x_traj

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            state_err = x_traj[step] - x_goal
            cost = cost + self.q_stage * torch.dot(state_err, state_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
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

    def solve(self, x0_np: np.ndarray, x_goal_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.tensor(x_goal_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()
        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()
        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_goal).item())
        iterations = 0
        reg = self.regularization

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_seq)
            k_seq = torch.empty((self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            kk_seq = torch.empty((self.horizon, self.action_dim, self.state_dim), dtype=torch.float32, device=self.device)
            terminal_err = x_traj[self.horizon] - x_goal
            v_x = 2.0 * self.q_terminal * terminal_err
            v_xx = 2.0 * self.q_terminal * self.eye_x
            backward_ok = True

            for step in range(self.horizon - 1, -1, -1):
                x_err = x_traj[step] - x_goal
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
                new_cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
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
        return (
            x_traj.detach().cpu().numpy().astype(np.float64),
            u_seq.detach().cpu().numpy().astype(np.float64),
            time.perf_counter() - t0,
            iterations,
            current_cost,
        )


class ILQRPolicyAdapter:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: dict[str, object],
        action_mean: np.ndarray,
        action_std: np.ndarray,
        device: torch.device,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        ilqr_max_iters: int,
        ilqr_tol: float,
        ilqr_regularization: float,
    ) -> None:
        self.model = model
        self.config = config
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device
        self.markov_deriv = int(config.get("markov_deriv", 1))
        embed_dim = int(config.get("embed_dim", 64))
        self.markov_state_dim = int(config.get("markov_state_dim", (self.markov_deriv + 1) * embed_dim))
        self.action_dim = int(config.get("action_dim", 5))
        dynamics = MarkovDynamicsTorch(model, self.markov_state_dim, self.action_dim, device)
        self.solver = ILQRMPCSolver(
            dynamics,
            horizon=horizon,
            q_terminal=q_terminal,
            q_stage=q_stage,
            r_control=r_control,
            max_iters=ilqr_max_iters,
            tol=ilqr_tol,
            regularization=ilqr_regularization,
            device=device,
        )
        self.history_len = required_markov_history(self.markov_deriv)
        self.current_history: list[torch.Tensor] = []
        self.goal_state: np.ndarray | None = None

    def reset(self, *, start_embedding: torch.Tensor, goal_embedding: torch.Tensor) -> None:
        self.current_history = [start_embedding] * self.history_len
        goal_history = [goal_embedding] * self.history_len
        goal_state = make_markov_state(goal_history, self.markov_deriv)
        self.goal_state = goal_state.detach().cpu().numpy().astype(np.float64)

    def append_embedding(self, embedding: torch.Tensor) -> None:
        self.current_history.append(embedding)
        self.current_history = self.current_history[-self.history_len :]

    def current_state_np(self) -> np.ndarray:
        state = make_markov_state(self.current_history, self.markov_deriv)
        if int(state.numel()) != self.markov_state_dim:
            raise ValueError(f"State dimension mismatch: config says {self.markov_state_dim}, built {state.numel()}.")
        return state.detach().cpu().numpy().astype(np.float64)

    def get_action(self) -> tuple[np.ndarray, dict[str, float]]:
        if self.goal_state is None:
            raise RuntimeError("ILQRPolicyAdapter.reset must be called before get_action.")
        _, u_plan, solve_time, n_iters, plan_cost = self.solver.solve(self.current_state_np(), self.goal_state)
        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = normalized_to_raw_action(u0_norm, self.action_mean, self.action_std)
        return u0_raw, {
            "solve_time_ms": float(solve_time * 1000.0),
            "ilqr_iterations": float(n_iters),
            "ilqr_cost": float(plan_cost),
        }


def load_episode_lengths(dataset_path: Path) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        return np.asarray(h5["ep_len"][:], dtype=np.int64)


def sample_eval_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[EvalCase]:
    valid = np.flatnonzero(ep_len >= 2)
    if valid.size == 0:
        raise ValueError("Need at least one episode with at least two frames.")
    if args.episode_idx is not None:
        if int(args.num_eval) != 1:
            raise ValueError("--episode-idx pins a single debug episode. Omit --episode-idx for --num-eval > 1.")
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"Episode {episode_idx} has length {ep_len[episode_idx]}, expected at least 2.")
        return [EvalCase(episode_idx, 0, int(ep_len[episode_idx]) - 1, int(ep_len[episode_idx]))]
    if args.num_eval > valid.size:
        raise ValueError(f"Requested {args.num_eval} episodes, but only {valid.size} are valid.")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(valid, size=int(args.num_eval), replace=False))
    return [EvalCase(int(ep), 0, int(ep_len[ep]) - 1, int(ep_len[ep])) for ep in selected]


def fit_processors(dataset_path: Path, keys: list[str]) -> dict[str, preprocessing.StandardScaler]:
    processors: dict[str, preprocessing.StandardScaler] = {}
    with h5py.File(dataset_path, "r") as h5:
        for key in keys:
            if key not in h5:
                raise KeyError(f"Cannot fit processor for missing dataset column '{key}' in {dataset_path}.")
            data = np.asarray(h5[key][:], dtype=np.float32)
            if data.ndim == 1:
                data = data[:, None]
            data = data.reshape(data.shape[0], -1)
            data = data[~np.isnan(data).any(axis=1)]
            if data.size == 0:
                raise ValueError(f"Dataset column '{key}' has no finite rows for normalization.")
            processor = preprocessing.StandardScaler()
            processor.fit(data)
            processors[key] = processor
            if key != "action":
                processors[f"goal_{key}"] = processor
    return processors


def make_img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def require_swm_policy(args: argparse.Namespace) -> str:
    if args.method.startswith("swm") and not args.policy:
        raise ValueError(f"--policy is required for --method {args.method}.")
    return str(args.policy)


def infer_swm_history_size(model: torch.nn.Module, fallback: int) -> int:
    if hasattr(model, "history_size"):
        return int(getattr(model, "history_size"))
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "num_frames"):
        return int(getattr(predictor, "num_frames"))
    return int(fallback)


def scan_model_for_attribute(module: torch.nn.Module, attribute_name: str) -> torch.nn.Module | None:
    if hasattr(module, attribute_name):
        module.eval()
        return module
    for child in module.children():
        result = scan_model_for_attribute(child, attribute_name)
        if result is not None:
            return result
    return None


def load_swm_model(policy_name: str, attribute_name: str, cache_dir: Path | None) -> torch.nn.Module:
    policy_path = Path(policy_name).expanduser()
    if policy_path.is_file():
        loaded = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, torch.nn.Module):
            raise TypeError(f"Checkpoint {policy_path} did not contain a torch module. Convert weights files first.")
        model = scan_model_for_attribute(loaded, attribute_name)
        if model is None:
            raise RuntimeError(f"No module with '{attribute_name}' found in {policy_path}.")
        return model
    if attribute_name == "get_cost":
        return swm.policy.AutoCostModel(policy_name, cache_dir=cache_dir)
    if attribute_name == "get_action":
        return swm.policy.AutoActionableModel(policy_name, cache_dir=cache_dir)
    raise ValueError(f"Unsupported stable-worldmodel attribute '{attribute_name}'.")


def make_swm_policy(args: argparse.Namespace, device: torch.device, action_space: spaces.Box) -> tuple[Any, int, dict[str, Any]]:
    policy_name = require_swm_policy(args)
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None
    process_keys = args.process_key if args.process_key is not None else ["action"]
    process = fit_processors(args.dataset_path, process_keys)
    transform = {
        "pixels": make_img_transform(args.swm_img_size),
        "goal": make_img_transform(args.swm_img_size),
    }
    if args.method == "swm_cost":
        model = load_swm_model(policy_name, "get_cost", cache_dir)
        if not hasattr(model, "get_cost"):
            raise TypeError(f"Policy '{policy_name}' does not expose get_cost(). Use --method swm_action instead.")
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        if hasattr(model, "interpolate_pos_encoding"):
            model.interpolate_pos_encoding = True
        solver_cfg = OmegaConf.load(args.solver_config)
        solver_cfg.device = str(device)
        solver_cfg.seed = int(args.seed)
        solver = hydra.utils.instantiate(solver_cfg, model=model)
        plan_config = swm.PlanConfig(
            horizon=int(args.plan_horizon),
            receding_horizon=int(args.receding_horizon),
            action_block=int(args.action_block),
            warm_start=not bool(args.no_warm_start),
        )
        policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_config, process=process, transform=transform)
        method_config: dict[str, Any] = {
            "policy": policy_name,
            "solver_config": str(args.solver_config),
            "plan_config": {
                "horizon": int(args.plan_horizon),
                "receding_horizon": int(args.receding_horizon),
                "action_block": int(args.action_block),
                "warm_start": not bool(args.no_warm_start),
            },
            "process_keys": process_keys,
        }
    else:
        model = load_swm_model(policy_name, "get_action", cache_dir)
        if not hasattr(model, "get_action"):
            raise TypeError(f"Policy '{policy_name}' does not expose get_action(). Use --method swm_cost instead.")
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        policy = swm.policy.FeedForwardPolicy(model=model, process=process, transform=transform)
        method_config = {"policy": policy_name, "process_keys": process_keys}

    history_size = int(args.swm_history_size) if args.swm_history_size is not None else infer_swm_history_size(model, DEFAULT_SWM_HISTORY_SIZE)
    if history_size < 1:
        raise ValueError("--swm-history-size must be positive.")
    policy.set_env(SingleVectorEnvAdapter(action_space))
    method_config["history_size"] = history_size
    method_config["swm_img_size"] = int(args.swm_img_size)
    return policy, history_size, method_config


def load_ilqr_assets(args: argparse.Namespace, device: torch.device) -> tuple[ILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path]:
    model_dir = args.model_dir.expanduser().resolve()
    config = load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    model = load_model(checkpoint_path, device)
    markov_deriv = int(config.get("markov_deriv", 1))
    if markov_deriv < 0:
        raise ValueError(f"Expected non-negative markov_deriv, got {markov_deriv}.")
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 5))
    stats_dataset_path = (
        args.stats_dataset_path.expanduser().resolve()
        if args.stats_dataset_path is not None
        else args.dataset_path
    )
    train_stats_dataset = LeWMOGBenchDataset(
        stats_dataset_path,
        history_size=int(config.get("history_size", 1)),
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    policy = ILQRPolicyAdapter(
        model=model,
        config=config,
        action_mean=train_stats_dataset.action_mean.astype(np.float32),
        action_std=train_stats_dataset.action_std.astype(np.float32),
        device=device,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        ilqr_max_iters=args.ilqr_max_iters,
        ilqr_tol=args.ilqr_tol,
        ilqr_regularization=args.ilqr_regularization,
    )
    method_config = {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "stats_dataset_path": str(stats_dataset_path),
        "img_size": img_size,
        "horizon": int(args.horizon),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "ilqr_max_iters": int(args.ilqr_max_iters),
    }
    return policy, model, method_config, pixel_mean, pixel_std, checkpoint_path


def save_case_summary(case_dir: Path, summary: dict[str, Any]) -> None:
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_case(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    device: torch.device,
    ilqr_assets: tuple[ILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path] | None,
    swm_policy: Any | None,
    swm_history_size: int | None,
) -> dict[str, Any]:
    episode = load_dataset_episode(args.dataset_path, case.episode_idx)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    target_block_pos_np = np.asarray(episode["target_block_pos"], dtype=np.float32)
    target_block_yaw_np = np.asarray(episode["target_block_yaw"], dtype=np.float32)
    pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
    obs_np = np.asarray(episode["observation"], dtype=np.float32)
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])
    width = int(episode["width"])
    height = int(episode["height"])
    physics_timestep = float(episode["physics_timestep"])
    control_timestep = float(episode["control_timestep"])
    dataset_max_episode_steps = int(episode["max_episode_steps"])
    max_episode_steps = (
        int(args.env_max_episode_steps)
        if args.env_max_episode_steps is not None
        else max(dataset_max_episode_steps, int(args.max_oracle_steps) + int(args.eval_budget) + 1)
    )

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_name=env_name,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        max_episode_steps=max_episode_steps,
        width=width,
        height=height,
    )
    oracle = CubePlanOracle(
        env=env,
        segment_dt=args.oracle_segment_dt,
        noise=args.oracle_noise,
        noise_smoothing=args.oracle_noise_smoothing,
    )

    start_frame, start_info, start_obs = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        target_block_pos=target_block_pos_np[case.start_step],
        target_block_yaw=float(target_block_yaw_np[case.start_step, 0]),
        camera=camera,
    )
    goal_frame, goal_info, goal_obs = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[case.goal_step],
        qvel=qvel_np[case.goal_step],
        target_block_pos=target_block_pos_np[case.goal_step],
        target_block_yaw=float(target_block_yaw_np[case.goal_step, 0]),
        camera=camera,
    )
    current_frame, current_info, current_obs = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        target_block_pos=target_block_pos_np[case.start_step],
        target_block_yaw=float(target_block_yaw_np[case.start_step, 0]),
        camera=camera,
    )
    save_rgb_image(case_dir / "start_image.png", start_frame)
    save_rgb_image(case_dir / "goal_image.png", goal_frame)

    goal_block_pos = np.asarray(goal_info["privileged/target_block_pos"], dtype=np.float32)
    goal_block_yaw = float(goal_info["privileged/target_block_yaw"][0])
    rollout_frames = [current_frame.copy()]
    cube_goal_distances = [cube_distance(current_info, goal_block_pos)]
    cube_yaw_errors = [cube_yaw_error(current_info, goal_block_yaw)]
    step_records: list[dict[str, Any]] = []
    executed_actions_raw: list[np.ndarray] = []
    used_oracle_before_policy = True
    oracle_steps_executed = 0
    policy_steps_executed = 0
    handoff_step = 0
    handoff_info = current_info
    stop_reason = "eval_budget"
    terminated = False
    truncated = False

    if args.method == "ilqr":
        assert ilqr_assets is not None
        ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
        img_size = int(ilqr_config["img_size"])
        start_emb = encode_single_frame(
            ilqr_model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        goal_emb = encode_single_frame(
            ilqr_model,
            goal_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        ilqr_policy.reset(start_embedding=start_emb, goal_embedding=goal_emb)
        episode_history = None
    else:
        assert swm_history_size is not None
        episode_history = CubeEpisodeHistory(
            history_size=swm_history_size,
            action_dim=env.action_space.shape[0],
            case_id=case_idx,
            goal_frame=goal_frame,
            goal_obs=goal_obs,
            goal_qpos=np.asarray(goal_info["qpos"], dtype=np.float32),
            goal_qvel=np.asarray(goal_info["qvel"], dtype=np.float32),
            goal_block_pos=goal_block_pos,
            goal_block_yaw=np.asarray(goal_info["privileged/target_block_yaw"], dtype=np.float32),
            goal_step=case.goal_step,
        )
        episode_history.reset(frame=current_frame, info=current_info, observation=current_obs, step_idx=case.start_step)

    oracle_grasped = cube_is_grasped(
        current_info,
        contact_threshold=args.grasp_contact_threshold,
        alignment_threshold=args.grasp_alignment_threshold,
    )

    if not oracle_grasped:
        oracle.reset(None, current_info)
        for oracle_step in range(int(args.max_oracle_steps)):
            oracle_action = np.asarray(oracle.select_action(None, current_info), dtype=np.float32)
            executed_actions_raw.append(oracle_action.copy())
            _, _, terminated, truncated, step_info = env.step(oracle_action)
            current_info = step_info
            current_obs = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            oracle_steps_executed += 1
            rollout_frames.append(current_frame.copy())

            if args.method == "ilqr":
                assert ilqr_assets is not None
                ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
                next_emb = encode_single_frame(
                    ilqr_model,
                    current_frame,
                    device=device,
                    img_size=int(ilqr_config["img_size"]),
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                )
                ilqr_policy.append_embedding(next_emb)
            else:
                assert episode_history is not None
                episode_history.append(
                    frame=current_frame,
                    action=oracle_action,
                    info=current_info,
                    observation=current_obs,
                    step_idx=case.start_step + oracle_step + 1,
                )

            dist = cube_distance(current_info, goal_block_pos)
            yaw_err = cube_yaw_error(current_info, goal_block_yaw)
            oracle_grasped = cube_is_grasped(
                current_info,
                contact_threshold=args.grasp_contact_threshold,
                alignment_threshold=args.grasp_alignment_threshold,
            )
            cube_goal_distances.append(dist)
            cube_yaw_errors.append(yaw_err)
            step_records.append(
                {
                    "phase": "oracle",
                    "step": int(len(executed_actions_raw)),
                    "cube_goal_distance": dist,
                    "cube_yaw_error": yaw_err,
                    "oracle_grasped": bool(oracle_grasped),
                }
            )
            if oracle_grasped:
                handoff_step = int(len(executed_actions_raw))
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break

    if oracle_grasped:
        handoff_info = current_info

    success = float(np.min(cube_goal_distances)) <= float(args.cube_success_threshold)
    if success:
        stop_reason = "goal_reached"

    if oracle_grasped and not (terminated or truncated) and not success:
        for policy_step in range(int(args.eval_budget)):
            if args.method == "ilqr":
                assert ilqr_assets is not None
                ilqr_policy, _, _, _, _, _ = ilqr_assets
                action_raw, record = ilqr_policy.get_action()
            else:
                assert swm_policy is not None and episode_history is not None
                action_batch = swm_policy.get_action(episode_history.info())
                action_raw = np.asarray(action_batch, dtype=np.float32).reshape(-1, env.action_space.shape[0])[0]
                record = {}

            executed_actions_raw.append(action_raw.copy())
            _, _, terminated, truncated, step_info = env.step(action_raw)
            current_info = step_info
            current_obs = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
            current_frame = np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)
            policy_steps_executed += 1
            rollout_frames.append(current_frame.copy())

            if args.method == "ilqr":
                assert ilqr_assets is not None
                ilqr_policy, ilqr_model, ilqr_config, pixel_mean, pixel_std, _ = ilqr_assets
                next_emb = encode_single_frame(
                    ilqr_model,
                    current_frame,
                    device=device,
                    img_size=int(ilqr_config["img_size"]),
                    pixel_mean=pixel_mean,
                    pixel_std=pixel_std,
                )
                ilqr_policy.append_embedding(next_emb)
            else:
                assert episode_history is not None
                episode_history.append(
                    frame=current_frame,
                    action=action_raw,
                    info=current_info,
                    observation=current_obs,
                    step_idx=case.start_step + oracle_steps_executed + policy_step + 1,
                )

            dist = cube_distance(current_info, goal_block_pos)
            yaw_err = cube_yaw_error(current_info, goal_block_yaw)
            cube_goal_distances.append(dist)
            cube_yaw_errors.append(yaw_err)
            success = dist <= float(args.cube_success_threshold)
            step_records.append(
                {
                    "phase": "policy",
                    "step": int(len(executed_actions_raw)),
                    "policy_step": int(policy_step + 1),
                    "cube_goal_distance": dist,
                    "cube_yaw_error": yaw_err,
                    **record,
                }
            )
            if success:
                stop_reason = "goal_reached"
                break
            if terminated or truncated:
                stop_reason = "terminated" if terminated else "truncated"
                break
        if policy_steps_executed >= int(args.eval_budget) and not success and not (terminated or truncated):
            stop_reason = "eval_budget"
    elif not oracle_grasped and not (terminated or truncated) and not success:
        stop_reason = "oracle_failed_to_grasp"

    video_path = None
    if not args.no_videos:
        video_path = str(save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))
    env.close()

    summary = {
        **asdict(case),
        "success": bool(success),
        "success_metric": "cube_position_l2",
        "cube_success_threshold": float(args.cube_success_threshold),
        "initial_cube_goal_distance": float(cube_goal_distances[0]),
        "handoff_cube_goal_distance": float(cube_distance(handoff_info, goal_block_pos)),
        "final_cube_goal_distance": float(cube_goal_distances[-1]),
        "min_cube_goal_distance": float(np.min(cube_goal_distances)),
        "initial_cube_yaw_error": float(cube_yaw_errors[0]),
        "handoff_cube_yaw_error": float(cube_yaw_error(handoff_info, goal_block_yaw)),
        "final_cube_yaw_error": float(cube_yaw_errors[-1]),
        "min_cube_yaw_error": float(np.min(cube_yaw_errors)),
        "used_oracle_before_policy": bool(used_oracle_before_policy),
        "oracle_grasped": bool(oracle_grasped),
        "oracle_steps_executed": int(oracle_steps_executed),
        "handoff_step": None if not oracle_grasped else int(handoff_step),
        "policy_steps_executed": int(policy_steps_executed),
        "steps_executed": int(len(executed_actions_raw)),
        "stop_reason": stop_reason,
        "episode_seed": int(episode_seed),
        "env_name": env_name,
        "camera": camera,
        "dataset_max_episode_steps": int(dataset_max_episode_steps),
        "env_max_episode_steps": int(max_episode_steps),
        "goal_block_pos": goal_block_pos.tolist(),
        "goal_block_yaw": float(goal_block_yaw),
        "handoff_block_pos": np.asarray(handoff_info["privileged/block_0_pos"], dtype=np.float32).tolist(),
        "handoff_block_yaw": float(handoff_info["privileged/block_0_yaw"][0]),
        "handoff_effector_pos": np.asarray(handoff_info["proprio/effector_pos"], dtype=np.float32).tolist(),
        "handoff_qpos": np.asarray(handoff_info["qpos"], dtype=np.float32).tolist(),
        "handoff_qvel": np.asarray(handoff_info["qvel"], dtype=np.float32).tolist(),
        "final_block_pos": np.asarray(current_info["privileged/block_0_pos"], dtype=np.float32).tolist(),
        "final_block_yaw": float(current_info["privileged/block_0_yaw"][0]),
        "final_effector_pos": np.asarray(current_info["proprio/effector_pos"], dtype=np.float32).tolist(),
        "final_qpos": np.asarray(current_info["qpos"], dtype=np.float32).tolist(),
        "final_qvel": np.asarray(current_info["qvel"], dtype=np.float32).tolist(),
        "video_path": video_path,
        "step_records": step_records,
    }
    save_case_summary(case_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.max_oracle_steps < 0:
        raise ValueError("--max-oracle-steps must be non-negative.")
    if args.cube_success_threshold <= 0:
        raise ValueError("--cube-success-threshold must be positive.")

    device = require_device(args.device)
    ep_len = load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)

    run_name = f"{int(time.time())}_{args.method}_seed_{args.seed}"
    out_root = args.out_dir.expanduser().resolve() / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    ilqr_assets = None
    swm_policy = None
    swm_history_size = None
    method_config: dict[str, Any]
    if args.method == "ilqr":
        ilqr_assets = load_ilqr_assets(args, device)
        method_config = ilqr_assets[2]
    else:
        probe_episode = load_dataset_episode(args.dataset_path, cases[0].episode_idx)
        env_for_space = make_env(
            env_name=str(probe_episode["env_name"]),
            physics_timestep=float(probe_episode["physics_timestep"]),
            control_timestep=float(probe_episode["control_timestep"]),
            max_episode_steps=int(probe_episode["max_episode_steps"]),
            width=int(probe_episode["width"]),
            height=int(probe_episode["height"]),
        )
        swm_policy, swm_history_size, method_config = make_swm_policy(args, device, env_for_space.action_space)
        env_for_space.close()

    case_results = []
    for case_idx, case in enumerate(tqdm(cases, desc="Hard OGBench cube eval")):
        case_results.append(
            run_case(
                args=args,
                case=case,
                case_idx=case_idx,
                out_root=out_root,
                device=device,
                ilqr_assets=ilqr_assets,
                swm_policy=swm_policy,
                swm_history_size=swm_history_size,
            )
        )

    successes = np.asarray([case["success"] for case in case_results], dtype=bool)
    metrics = {
        "success_rate": float(np.mean(successes) * 100.0),
        "episode_successes": successes.astype(int).tolist(),
        "success_metric": "cube_position_l2",
        "cube_success_threshold": float(args.cube_success_threshold),
        "method": args.method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "num_eval": len(case_results),
        "eval_budget": int(args.eval_budget),
        "max_oracle_steps": int(args.max_oracle_steps),
        "goal_protocol": "start_step_0_to_final_episode_step_with_oracle_grasp",
        "cases": case_results,
    }
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
