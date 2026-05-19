#!/usr/bin/env python3
"""Hard rope benchmark for iLQR and stable-worldmodel policies.

Every method starts from episode step 0, uses the final episode state/frame as
the goal, and is evaluated with the existing rope planner's task-target metric.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import faulthandler
import io
import json
import multiprocessing
import os
import queue
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
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
    message=r"Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)
faulthandler.enable(all_threads=True)

ROOT_DIR = Path(__file__).resolve().parents[2]
STABLE_WORLDMODEL_DIR = ROOT_DIR / "third_party" / "stable-worldmodel"
LE_WM_DIR = ROOT_DIR / "third_party" / "le-wm"
for path in (ROOT_DIR, STABLE_WORLDMODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(LE_WM_DIR) not in sys.path:
    sys.path.append(str(LE_WM_DIR))

import h5py
import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

from rope.shared.lab_env import LabEnv, TaskState
import rope.shared.models as rope_shared_models
from rope.train.mlpdyn_train import LeWMRopeDataset, build_markov_state, required_markov_history

IMAGENET_NORMALIZE = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def register_legacy_checkpoint_aliases() -> None:
    """Resolve older object checkpoints pickled under pre-merge module names."""

    ogbench_cube_pkg = sys.modules.setdefault("ogbench_cube", types.ModuleType("ogbench_cube"))
    ogbench_cube_pkg.__path__ = []  # type: ignore[attr-defined]
    ogbench_cube_shared_pkg = sys.modules.setdefault("ogbench_cube.shared", types.ModuleType("ogbench_cube.shared"))
    ogbench_cube_shared_pkg.__path__ = []  # type: ignore[attr-defined]
    setattr(ogbench_cube_pkg, "shared", ogbench_cube_shared_pkg)
    setattr(ogbench_cube_shared_pkg, "models", rope_shared_models)
    sys.modules.setdefault("ogbench_cube.shared.models", rope_shared_models)


register_legacy_checkpoint_aliases()


DEFAULT_DATASET_PATH = "rope/data/rope_random_cubic_spline.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn"
DEFAULT_OUT_DIR = "rope/plan/rope_hard_eval"
DEFAULT_SOLVER_CONFIG = (
    STABLE_WORLDMODEL_DIR / "scripts" / "plan" / "config" / "solver" / "cem.yaml"
)
DEFAULT_NUM_EVAL = 50
DEFAULT_EVAL_BUDGET = 120
DEFAULT_SEED = 42
DEFAULT_SWM_HISTORY_SIZE = 3

DEVICE = "auto"
HORIZON = 25
Q_TERMINAL = 5.0
Q_STAGE = 0.005
R_CONTROL = 0.001
VIDEO_FPS = 20


@dataclass(frozen=True)
class EvalCase:
    episode_idx: int
    start_step: int
    goal_step: int
    ep_len: int


class SingleVectorEnvAdapter:
    def __init__(self, action_dim: int) -> None:
        action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        )
        self.num_envs = 1
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=action_space.low[None, ...],
            high=action_space.high[None, ...],
            dtype=action_space.dtype,
        )


class RopeEpisodeHistory:
    def __init__(
        self,
        *,
        history_size: int,
        action_dim: int,
        case_id: int,
        goal_frame: np.ndarray,
        goal_info: dict[str, np.ndarray],
        goal_step: int,
    ) -> None:
        self.history_size = int(history_size)
        self.action_dim = int(action_dim)
        self.case_id = int(case_id)
        self.goal_frame = np.asarray(goal_frame, dtype=np.uint8)
        self.goal_step = int(goal_step)
        self.goal_observation = np.asarray(goal_info["observation"], dtype=np.float32)
        self.goal_task_target = np.asarray(goal_info["task_target"], dtype=np.float32)
        self.goal_qpos = np.asarray(goal_info["qpos"], dtype=np.float32)
        self.goal_qvel = np.asarray(goal_info["qvel"], dtype=np.float32)
        self.goal_control = np.asarray(goal_info["control"], dtype=np.float32)
        self.goal_left_attachment_pos = np.asarray(goal_info["left_attachment_pos"], dtype=np.float32)
        self.goal_right_attachment_pos = np.asarray(goal_info["right_attachment_pos"], dtype=np.float32)
        self.goal_rope_length = np.asarray(goal_info["rope_length"], dtype=np.float32)

        self.frames: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.actions: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.observations: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.task_target: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qpos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.qvel: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.control: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.left_attachment_pos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.right_attachment_pos: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.rope_length: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.step_idx: deque[int] = deque(maxlen=self.history_size)

    def reset(self, *, frame: np.ndarray, info: dict[str, np.ndarray], step_idx: int) -> None:
        zero_action = np.zeros((self.action_dim,), dtype=np.float32)
        for _ in range(self.history_size):
            self.append(frame=frame, action=zero_action, info=info, step_idx=step_idx)

    def append(
        self,
        *,
        frame: np.ndarray,
        action: np.ndarray,
        info: dict[str, np.ndarray],
        step_idx: int,
    ) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.observations.append(np.asarray(info["observation"], dtype=np.float32).copy())
        self.task_target.append(np.asarray(info["task_target"], dtype=np.float32).copy())
        self.qpos.append(np.asarray(info["qpos"], dtype=np.float32).copy())
        self.qvel.append(np.asarray(info["qvel"], dtype=np.float32).copy())
        self.control.append(np.asarray(info["control"], dtype=np.float32).copy())
        self.left_attachment_pos.append(np.asarray(info["left_attachment_pos"], dtype=np.float32).copy())
        self.right_attachment_pos.append(np.asarray(info["right_attachment_pos"], dtype=np.float32).copy())
        self.rope_length.append(np.asarray(info["rope_length"], dtype=np.float32).copy())
        self.step_idx.append(int(step_idx))

    def info(self) -> dict[str, np.ndarray]:
        frames = np.stack(list(self.frames), axis=0)
        actions = np.stack(list(self.actions), axis=0)
        observations = np.stack(list(self.observations), axis=0)
        task_target = np.stack(list(self.task_target), axis=0)
        qpos = np.stack(list(self.qpos), axis=0)
        qvel = np.stack(list(self.qvel), axis=0)
        control = np.stack(list(self.control), axis=0)
        left_attachment_pos = np.stack(list(self.left_attachment_pos), axis=0)
        right_attachment_pos = np.stack(list(self.right_attachment_pos), axis=0)
        rope_length = np.stack(list(self.rope_length), axis=0)
        step_idx = np.asarray(list(self.step_idx), dtype=np.int64)
        ids = np.full((self.history_size,), self.case_id, dtype=np.int64)
        goal_ids = np.asarray([self.case_id], dtype=np.int64)
        goal_step_idx = np.asarray([self.goal_step], dtype=np.int64)

        return {
            "pixels": frames[None, ...],
            "goal": self.goal_frame[None, None, ...],
            "action": actions[None, ...],
            "observation": observations[None, ...],
            "goal_observation": self.goal_observation[None, None, ...],
            "task_target": task_target[None, ...],
            "goal_task_target": self.goal_task_target[None, None, ...],
            "qpos": qpos[None, ...],
            "goal_qpos": self.goal_qpos[None, None, ...],
            "qvel": qvel[None, ...],
            "goal_qvel": self.goal_qvel[None, None, ...],
            "control": control[None, ...],
            "goal_control": self.goal_control[None, None, ...],
            "left_attachment_pos": left_attachment_pos[None, ...],
            "goal_left_attachment_pos": self.goal_left_attachment_pos[None, None, ...],
            "right_attachment_pos": right_attachment_pos[None, ...],
            "goal_right_attachment_pos": self.goal_right_attachment_pos[None, None, ...],
            "rope_length": rope_length[None, ...],
            "goal_rope_length": self.goal_rope_length[None, None, ...],
            "step_idx": step_idx[None, ...],
            "goal_step_idx": goal_step_idx[None, ...],
            "id": ids[None, ...],
            "goal_id": goal_ids[None, ...],
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
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Start index into the deterministic --num-eval sampled case list.",
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=None,
        help="Evaluate only this many cases from the deterministic sampled case list.",
    )
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument("--goal-tolerance", type=float, default=None)
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

    parser.add_argument("--policy", default=None, help="Stable-worldmodel run name, directory, or object checkpoint.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--solver-config", type=Path, default=DEFAULT_SOLVER_CONFIG)
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=1)
    parser.add_argument("--swm-history-size", type=int, default=None)
    parser.add_argument("--swm-img-size", type=int, default=224)
    parser.add_argument("--process-key", action="append", default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--show-solver-output", action="store_true", help="Do not suppress solver stdout such as CEM timing prints.")
    parser.add_argument("--debug", action="store_true", help="Print and save detailed phase/resource diagnostics.")
    parser.add_argument("--debug-log", type=Path, default=None, help="Optional debug log path. Defaults to <run_dir>/debug.log.")
    parser.add_argument("--debug-every-steps", type=int, default=10, help="Rollout step interval for debug logging.")
    parser.add_argument(
        "--isolate-cases",
        action="store_true",
        help="Run each case in a spawned subprocess. More robust to native MuJoCo/CUDA aborts, but slower.",
    )
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _cuda_status() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024.0**2)
        reserved = torch.cuda.memory_reserved(device) / (1024.0**2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        return (
            f"cuda_device={device} "
            f"cuda_alloc_mb={allocated:.1f} "
            f"cuda_reserved_mb={reserved:.1f} "
            f"cuda_max_alloc_mb={max_allocated:.1f}"
        )
    except Exception as exc:
        return f"cuda_status_error={type(exc).__name__}:{exc}"


def _nvidia_smi_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        return f"nvidia_smi_error={type(exc).__name__}:{exc}"
    if result.returncode != 0:
        return f"nvidia_smi_returncode={result.returncode}"
    compact = " | ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    return f"nvidia_smi=[{compact}]"


def debug_log(args: argparse.Namespace, message: str, *, log_path: Path | None = None, nvidia_smi: bool = False) -> None:
    if not getattr(args, "debug", False):
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    parts = [
        timestamp,
        f"pid={os.getpid()}",
        f"rss_mb={_rss_mb():.1f}",
        _cuda_status(),
        message,
    ]
    if nvidia_smi:
        parts.insert(-1, _nvidia_smi_status())
    line = " | ".join(parts)
    print(f"[rope-debug] {line}", flush=True)
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


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
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Checkpoint {checkpoint_path} did not contain a torch module.")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.ascontiguousarray(image))


class FFMpegRolloutWriter:
    """Stream rollout frames directly to an mp4 without writing frame PNGs."""

    def __init__(
        self,
        path: Path,
        *,
        fps: int,
        first_frame: np.ndarray,
    ) -> None:
        self.path = path
        self.frames_written = 0
        self.error: str | None = None
        self.proc: subprocess.Popen[bytes] | None = None

        frame = np.asarray(first_frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.error = f"expected RGB frame with shape HxWx3, got {frame.shape}"
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.error = "ffmpeg executable not found"
            return

        height, width = int(frame.shape[0]), int(frame.shape[1])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.proc = None
            return
        self.write(frame)

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None or self.error is not None:
            return
        try:
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                self.error = f"expected RGB frame with shape HxWx3, got {arr.shape}"
                return
            self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())
            self.frames_written += 1
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def close(self, *, timeout: float = 120.0) -> tuple[Path | None, str | None]:
        if self.proc is None:
            return None, self.error
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
                self.proc.stdin = None
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
        try:
            stdout, stderr = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            stdout, stderr = self.proc.communicate()
            self.error = "ffmpeg timed out while finalizing rollout.mp4"
        if self.proc.returncode != 0 and self.error is None:
            detail = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            self.error = f"ffmpeg_returncode={self.proc.returncode}: {detail}"
        if self.error is not None:
            return None, self.error
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return None, f"ffmpeg finished but did not create {self.path}"
        return self.path, None


def wait_for_video_file(path: Path, *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_count = 0
    while time.monotonic() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 2:
                    return True
            else:
                stable_count = 0
            last_size = size
        time.sleep(0.25)
    return path.is_file() and path.stat().st_size > 0


def update_case_summary(case_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary.update(updates)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def finalize_case_video(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    summary: dict[str, Any],
    debug_log_path: Path | None,
) -> dict[str, Any]:
    if args.no_videos:
        return summary
    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    existing = summary.get("video_path")
    if existing and Path(existing).is_file():
        return summary
    mp4_path = case_dir / "rollout.mp4"
    debug_log(args, f"parent_video_check_start case_idx={case_idx} path={mp4_path}", log_path=debug_log_path)
    video_path = mp4_path if wait_for_video_file(mp4_path) else None
    error = summary.get("video_error")
    if video_path is None and error is None:
        error = "rollout.mp4 was not produced"
    updates = {
        "video_path": None if video_path is None else str(video_path),
        "video_error": error,
    }
    debug_log(
        args,
        f"parent_video_check_done case_idx={case_idx} video={updates['video_path']} error={error}",
        log_path=debug_log_path,
    )
    summary.update(updates)
    return update_case_summary(case_dir, updates)


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


def load_dataset_episode(dataset_path: Path, episode_idx: int) -> dict[str, Any]:
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        return {
            "pixels": np.asarray(h5["pixels"][rows], dtype=np.uint8),
            "action": np.asarray(h5["action"][rows], dtype=np.float32),
            "observation": np.asarray(h5["observation"][rows], dtype=np.float32),
            "task_target": np.asarray(h5["task_target"][rows], dtype=np.float32),
            "qpos": np.asarray(h5["qpos"][rows], dtype=np.float32),
            "qvel": np.asarray(h5["qvel"][rows], dtype=np.float32),
            "control": np.asarray(h5["control"][rows], dtype=np.float32),
            "left_attachment_pos": np.asarray(h5["left_attachment_pos"][rows], dtype=np.float32),
            "right_attachment_pos": np.asarray(h5["right_attachment_pos"][rows], dtype=np.float32),
            "rope_length": np.asarray(h5["rope_length"][rows], dtype=np.float32),
            "time": np.asarray(h5["time"][rows], dtype=np.float32),
            "episode_seed": int(h5["episode_seed"][episode_idx]),
            "camera": str(h5.attrs.get("camera", "video_cam")),
            "mode": str(h5.attrs.get("mode", "")),
            "width": int(h5["pixels"].shape[2]),
            "height": int(h5["pixels"].shape[1]),
            "control_timestep": float(h5.attrs.get("control_timestep", 25.0 / 500.0)),
            "control_decimation": int(h5.attrs.get("control_decimation", 25)),
            "max_episode_steps": int(h5.attrs.get("max_episode_steps", max(ep_len - 1, 1))),
            "goal_tolerance": float(h5.attrs.get("goal_tolerance", 1e-4)),
            "terminated": bool(h5["terminated"][episode_idx]) if "terminated" in h5 else False,
            "truncated": bool(h5["truncated"][episode_idx]) if "truncated" in h5 else False,
        }


def render_rgb_frame(renderer: mujoco.Renderer, env: LabEnv, camera_id: int) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera_id)
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def extract_step_info(env: LabEnv, *, elapsed_time: float) -> dict[str, np.ndarray]:
    left_pos = env.data.site_xpos[env.arm1_site_id].copy().astype(np.float32)
    right_pos = env.data.site_xpos[env.arm2_site_id].copy().astype(np.float32)
    rope_length = env.data.ten_length.copy().astype(np.float32)
    task_target = env.task_controller.desired_state.as_array().astype(np.float32)
    qpos = env.data.qpos.copy().astype(np.float32)
    qvel = env.data.qvel.copy().astype(np.float32)
    control = env.data.ctrl.copy().astype(np.float32)
    observation = np.concatenate(
        [
            task_target,
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
        "task_target": task_target,
        "qpos": qpos,
        "qvel": qvel,
        "control": control,
        "left_attachment_pos": left_pos,
        "right_attachment_pos": right_pos,
        "rope_length": rope_length,
        "time": np.asarray([elapsed_time], dtype=np.float32),
    }


def reset_env_to_state(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control: np.ndarray,
    task_target: np.ndarray,
    camera_id: int,
    elapsed_time: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.reset(TaskState.from_array(task_target))
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.joint_controller.set_target(np.asarray(control, dtype=np.float64))
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.data.ctrl[:] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    frame = render_rgb_frame(renderer, env, camera_id)
    info = extract_step_info(env, elapsed_time=elapsed_time)
    return frame, info


def step_env_with_action(
    env: LabEnv,
    renderer: mujoco.Renderer,
    *,
    action: np.ndarray,
    control_decimation: int,
    camera_id: int,
    elapsed_time: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    env.apply_task_delta(np.asarray(action, dtype=np.float64))
    env.step(int(control_decimation))
    frame = render_rgb_frame(renderer, env, camera_id)
    info = extract_step_info(env, elapsed_time=elapsed_time)
    return frame, info


def task_target_distance(info: dict[str, np.ndarray], goal_task_target: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(info["task_target"], dtype=np.float32) - goal_task_target))


def left_attachment_distance(info: dict[str, np.ndarray], goal_left_pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(info["left_attachment_pos"], dtype=np.float32) - goal_left_pos))


def right_attachment_distance(info: dict[str, np.ndarray], goal_right_pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(info["right_attachment_pos"], dtype=np.float32) - goal_right_pos))


def rope_length_error(info: dict[str, np.ndarray], goal_rope_length: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(info["rope_length"], dtype=np.float32) - goal_rope_length))


class MarkovDynamicsTorch:
    def __init__(self, model: torch.nn.Module, state_dim: int, action_dim: int, device: torch.device) -> None:
        predictor = model.predictor
        if predictor.history_size != 1 or predictor.action_history_size != 1 or predictor.num_preds != 1:
            raise ValueError(
                "Expected a one-step Markov MLP dynamics model with "
                "history_size=1, action_history_size=1, and num_preds=1."
            )
        if type(model.action_encoder).__name__ != "Identity":
            raise ValueError("Expected identity action encoder for iLQR.")
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
        solve_time = time.perf_counter() - t0
        return (
            x_traj.detach().cpu().numpy().astype(np.float64),
            u_seq.detach().cpu().numpy().astype(np.float64),
            solve_time,
            iterations,
            current_cost,
        )


class ILQRPolicyAdapter:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: dict[str, Any],
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
        self.action_mean = action_mean
        self.action_std = action_std
        self.device = device
        self.markov_deriv = int(config.get("markov_deriv", 1))
        embed_dim = int(config.get("embed_dim", 32))
        self.markov_state_dim = int(config.get("markov_state_dim", (self.markov_deriv + 1) * embed_dim))
        self.action_dim = int(config.get("action_dim", 3))
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


def slice_eval_cases(args: argparse.Namespace, cases: list[EvalCase]) -> list[EvalCase]:
    if args.episode_idx is not None:
        if int(args.case_offset) != 0 or args.case_count is not None:
            raise ValueError("--case-offset/--case-count are for sampled benchmark sets, not --episode-idx debug runs.")
        return cases
    if int(args.case_offset) < 0:
        raise ValueError("--case-offset must be non-negative.")
    if args.case_count is not None and int(args.case_count) < 1:
        raise ValueError("--case-count must be positive when provided.")
    start = int(args.case_offset)
    if start >= len(cases):
        raise ValueError(f"--case-offset {start} is outside sampled case list of length {len(cases)}.")
    stop = len(cases) if args.case_count is None else min(len(cases), start + int(args.case_count))
    return cases[start:stop]


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
            transforms.Normalize(**IMAGENET_NORMALIZE),
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


def infer_action_encoder_input_dim(model: torch.nn.Module) -> int | None:
    action_encoder = getattr(model, "action_encoder", None)
    if action_encoder is None:
        return None
    if hasattr(action_encoder, "input_dim"):
        try:
            return int(getattr(action_encoder, "input_dim"))
        except (TypeError, ValueError):
            return None
    patch_embed = getattr(action_encoder, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        try:
            return int(getattr(patch_embed, "in_channels"))
        except (TypeError, ValueError):
            return None
    return None


def validate_swm_cost_action_block(model: torch.nn.Module, *, action_dim: int, action_block: int) -> None:
    encoder_input_dim = infer_action_encoder_input_dim(model)
    if int(action_block) <= 1 or encoder_input_dim is None:
        return
    expanded_action_dim = int(action_dim) * int(action_block)
    if encoder_input_dim == int(action_dim):
        raise ValueError(
            f"--action-block {action_block} is incompatible with this swm_cost checkpoint. "
            f"The model action encoder expects raw action_dim={action_dim}, but stable-worldmodel "
            f"expands CEM candidate actions to action_dim * action_block = {expanded_action_dim}. "
            "Use --action-block 1 for this checkpoint. For faster runs, reduce the CEM solver "
            "work with a lighter --solver-config instead of increasing --action-block."
        )


def validate_parent_swm_args(args: argparse.Namespace, action_dim: int) -> None:
    if args.method != "swm_cost" or int(args.action_block) <= 1:
        return
    policy_name = require_swm_policy(args)
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None
    model = load_swm_model(policy_name, "get_cost", cache_dir)
    validate_swm_cost_action_block(model, action_dim=action_dim, action_block=int(args.action_block))


def reset_swm_policy_state(policy: Any) -> None:
    """Clear MPC warm-start/action buffers between independent episodes."""

    action_buffer = getattr(policy, "_action_buffer", None)
    if action_buffer is not None and hasattr(action_buffer, "clear"):
        action_buffer.clear()
    public_buffer = getattr(policy, "action_buffer", None)
    if public_buffer is not None and hasattr(public_buffer, "clear"):
        public_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None


def make_swm_policy(args: argparse.Namespace, device: torch.device, action_dim: int) -> tuple[Any, int, dict[str, Any]]:
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
        validate_swm_cost_action_block(model, action_dim=action_dim, action_block=int(args.action_block))
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
    if args.method == "swm_cost" and int(args.plan_horizon) < history_size:
        raise ValueError(
            f"--plan-horizon ({args.plan_horizon}) must be >= SWM history size ({history_size}) "
            "for PLDM/LeWM rollout."
        )
    if args.method == "swm_cost" and int(args.receding_horizon) > int(args.plan_horizon):
        raise ValueError("--receding-horizon must be <= --plan-horizon.")
    policy.set_env(SingleVectorEnvAdapter(action_dim))
    method_config["action_dim"] = int(action_dim)
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
    action_dim = int(config.get("action_dim", 3))
    frameskip = int(config.get("frameskip", 1))
    if frameskip != 1:
        raise ValueError(f"This rope benchmark currently expects frameskip=1 for iLQR, got {frameskip}.")
    stats_dataset_path = (
        args.stats_dataset_path.expanduser().resolve()
        if args.stats_dataset_path is not None
        else args.dataset_path
    )
    train_stats_dataset = LeWMRopeDataset(
        stats_dataset_path,
        markov_deriv=markov_deriv,
        num_preds=1,
        frameskip=frameskip,
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
        "action_dim": action_dim,
        "horizon": int(args.horizon),
        "q_terminal": float(args.q_terminal),
        "q_stage": float(args.q_stage),
        "r_control": float(args.r_control),
        "ilqr_max_iters": int(args.ilqr_max_iters),
    }
    return policy, model, method_config, pixel_mean, pixel_std, checkpoint_path


def run_case(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    env: LabEnv,
    renderer: mujoco.Renderer,
    device: torch.device,
    ilqr_assets: tuple[ILQRPolicyAdapter, torch.nn.Module, dict[str, Any], torch.Tensor, torch.Tensor, Path] | None,
    swm_policy: Any | None,
    swm_history_size: int | None,
    action_dim: int,
    debug_log_path: Path | None = None,
) -> dict[str, Any]:
    debug_log(args, f"case_start case_idx={case_idx} episode_idx={case.episode_idx}", log_path=debug_log_path, nvidia_smi=True)
    episode = load_dataset_episode(args.dataset_path, case.episode_idx)
    debug_log(
        args,
        f"case_loaded case_idx={case_idx} ep_len={case.ep_len} budget={args.eval_budget} videos={not args.no_videos}",
        log_path=debug_log_path,
    )
    pixels_np = np.asarray(episode["pixels"], dtype=np.uint8)
    task_target_np = np.asarray(episode["task_target"], dtype=np.float32)
    qpos_np = np.asarray(episode["qpos"], dtype=np.float32)
    qvel_np = np.asarray(episode["qvel"], dtype=np.float32)
    control_np = np.asarray(episode["control"], dtype=np.float32)
    time_np = np.asarray(episode["time"], dtype=np.float32)
    camera = str(episode["camera"])
    width = int(episode["width"])
    height = int(episode["height"])
    control_decimation = int(episode["control_decimation"])
    goal_tolerance = float(args.goal_tolerance) if args.goal_tolerance is not None else float(episode["goal_tolerance"])

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    camera_id = env.model.camera(camera).id

    if True:
        debug_log(args, f"case_reset_start case_idx={case_idx}", log_path=debug_log_path)
        start_frame, start_info = reset_env_to_state(
            env,
            renderer,
            qpos=qpos_np[case.start_step],
            qvel=qvel_np[case.start_step],
            control=control_np[case.start_step],
            task_target=task_target_np[case.start_step],
            camera_id=camera_id,
            elapsed_time=float(time_np[case.start_step, 0]),
        )
        goal_frame, goal_info = reset_env_to_state(
            env,
            renderer,
            qpos=qpos_np[case.goal_step],
            qvel=qvel_np[case.goal_step],
            control=control_np[case.goal_step],
            task_target=task_target_np[case.goal_step],
            camera_id=camera_id,
            elapsed_time=float(time_np[case.goal_step, 0]),
        )
        current_frame, current_info = reset_env_to_state(
            env,
            renderer,
            qpos=qpos_np[case.start_step],
            qvel=qvel_np[case.start_step],
            control=control_np[case.start_step],
            task_target=task_target_np[case.start_step],
            camera_id=camera_id,
            elapsed_time=float(time_np[case.start_step, 0]),
        )
        debug_log(args, f"case_reset_done case_idx={case_idx}", log_path=debug_log_path)

        debug_log(args, f"case_save_images_start case_idx={case_idx}", log_path=debug_log_path)
        save_rgb_image(case_dir / "start_image.png", start_frame)
        save_rgb_image(case_dir / "goal_image.png", goal_frame)
        debug_log(args, f"case_save_images_done case_idx={case_idx}", log_path=debug_log_path)

        video_writer: FFMpegRolloutWriter | None = None
        if not args.no_videos:
            debug_log(args, f"case_video_stream_start case_idx={case_idx}", log_path=debug_log_path)
            video_writer = FFMpegRolloutWriter(
                case_dir / "rollout.mp4",
                fps=args.video_fps,
                first_frame=current_frame,
            )
            if video_writer.error is not None:
                debug_log(
                    args,
                    f"case_video_stream_error case_idx={case_idx} error={video_writer.error}",
                    log_path=debug_log_path,
                )
            else:
                debug_log(args, f"case_video_stream_ready case_idx={case_idx}", log_path=debug_log_path)

        goal_task_target = np.asarray(goal_info["task_target"], dtype=np.float32)
        goal_left_pos = np.asarray(goal_info["left_attachment_pos"], dtype=np.float32)
        goal_right_pos = np.asarray(goal_info["right_attachment_pos"], dtype=np.float32)
        goal_rope_length = np.asarray(goal_info["rope_length"], dtype=np.float32)

        task_target_distances = [task_target_distance(current_info, goal_task_target)]
        left_attachment_distances = [left_attachment_distance(current_info, goal_left_pos)]
        right_attachment_distances = [right_attachment_distance(current_info, goal_right_pos)]
        rope_length_errors = [rope_length_error(current_info, goal_rope_length)]
        latent_goal_distances: list[float] = []
        embedding_goal_distances: list[float] = []
        step_records: list[dict[str, Any]] = []
        executed_actions_raw: list[np.ndarray] = []
        executed_actions_norm: list[np.ndarray] = []

        if args.method == "ilqr":
            debug_log(args, f"case_ilqr_encode_start case_idx={case_idx}", log_path=debug_log_path)
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
            current_state = make_markov_state(ilqr_policy.current_history, ilqr_policy.markov_deriv)
            goal_state = torch.tensor(ilqr_policy.goal_state, dtype=torch.float32, device=device)
            latent_goal_distances.append(float(torch.linalg.vector_norm(current_state - goal_state).item()))
            embedding_goal_distances.append(float(torch.linalg.vector_norm(start_emb - goal_emb).item()))
            episode_history = None
            debug_log(args, f"case_ilqr_encode_done case_idx={case_idx}", log_path=debug_log_path)
        else:
            debug_log(args, f"case_swm_history_start case_idx={case_idx}", log_path=debug_log_path)
            assert swm_policy is not None
            reset_swm_policy_state(swm_policy)
            assert swm_history_size is not None
            episode_history = RopeEpisodeHistory(
                history_size=swm_history_size,
                action_dim=action_dim,
                case_id=case_idx,
                goal_frame=goal_frame,
                goal_info=goal_info,
                goal_step=case.goal_step,
            )
            episode_history.reset(frame=current_frame, info=current_info, step_idx=case.start_step)
            debug_log(args, f"case_swm_history_done case_idx={case_idx}", log_path=debug_log_path)

        success = task_target_distances[-1] <= goal_tolerance
        stop_reason = "goal_tolerance_reached" if success else "eval_budget"
        policy_steps_executed = 0

        if not success:
            for policy_step in range(int(args.eval_budget)):
                if policy_step == 0 or (
                    int(args.debug_every_steps) > 0 and policy_step % int(args.debug_every_steps) == 0
                ):
                    debug_log(
                        args,
                        f"case_step_begin case_idx={case_idx} step={policy_step + 1}/{args.eval_budget}",
                        log_path=debug_log_path,
                    )
                if args.method == "ilqr":
                    assert ilqr_assets is not None
                    ilqr_policy, _, _, _, _, _ = ilqr_assets
                    action_raw, record = ilqr_policy.get_action()
                    action_norm = action_to_standardized(action_raw, ilqr_policy.action_mean, ilqr_policy.action_std)
                    executed_actions_norm.append(action_norm.copy())
                else:
                    assert swm_policy is not None and episode_history is not None
                    debug_log(
                        args,
                        f"case_swm_get_action_start case_idx={case_idx} step={policy_step + 1}",
                        log_path=debug_log_path,
                    )
                    if args.show_solver_output:
                        action_batch = swm_policy.get_action(episode_history.info())
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            action_batch = swm_policy.get_action(episode_history.info())
                    debug_log(
                        args,
                        f"case_swm_get_action_done case_idx={case_idx} step={policy_step + 1}",
                        log_path=debug_log_path,
                    )
                    action_raw = np.asarray(action_batch, dtype=np.float32).reshape(-1, action_dim)[0]
                    record = {}

                executed_actions_raw.append(action_raw.copy())
                current_time = float(current_info["time"][0]) + float(episode["control_timestep"])
                debug_log(
                    args,
                    f"case_env_step_start case_idx={case_idx} step={policy_step + 1}",
                    log_path=debug_log_path,
                )
                current_frame, current_info = step_env_with_action(
                    env,
                    renderer,
                    action=action_raw,
                    control_decimation=control_decimation,
                    camera_id=camera_id,
                    elapsed_time=current_time,
                )
                debug_log(
                    args,
                    f"case_env_step_done case_idx={case_idx} step={policy_step + 1}",
                    log_path=debug_log_path,
                )
                policy_steps_executed += 1
                if video_writer is not None:
                    video_writer.write(current_frame)

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
                    current_state = make_markov_state(ilqr_policy.current_history, ilqr_policy.markov_deriv)
                    goal_state = torch.tensor(ilqr_policy.goal_state, dtype=torch.float32, device=device)
                    latent_goal_distances.append(float(torch.linalg.vector_norm(current_state - goal_state).item()))
                    embedding_goal_distances.append(float(torch.linalg.vector_norm(next_emb - goal_emb).item()))
                else:
                    assert episode_history is not None
                    episode_history.append(
                        frame=current_frame,
                        action=action_raw,
                        info=current_info,
                        step_idx=case.start_step + policy_step + 1,
                    )

                task_dist = task_target_distance(current_info, goal_task_target)
                left_dist = left_attachment_distance(current_info, goal_left_pos)
                right_dist = right_attachment_distance(current_info, goal_right_pos)
                length_err = rope_length_error(current_info, goal_rope_length)
                task_target_distances.append(task_dist)
                left_attachment_distances.append(left_dist)
                right_attachment_distances.append(right_dist)
                rope_length_errors.append(length_err)
                success = task_dist <= goal_tolerance
                step_records.append(
                    {
                        "phase": "policy",
                        "step": int(policy_step + 1),
                        "task_target_distance": task_dist,
                        "left_attachment_distance": left_dist,
                        "right_attachment_distance": right_dist,
                        "rope_length_error": length_err,
                        **record,
                    }
                )
                if success:
                    stop_reason = "goal_tolerance_reached"
                    break
                if policy_step == 0 or (
                    int(args.debug_every_steps) > 0 and (policy_step + 1) % int(args.debug_every_steps) == 0
                ):
                    debug_log(
                        args,
                        (
                            f"case_step_done case_idx={case_idx} step={policy_step + 1} "
                            f"task_dist={task_dist:.6f} left_dist={left_dist:.6f} "
                            f"right_dist={right_dist:.6f} rope_len_err={length_err:.6f}"
                        ),
                        log_path=debug_log_path,
                    )
            if policy_steps_executed >= int(args.eval_budget) and not success:
                stop_reason = "eval_budget"

    video_path = None
    video_error = None
    video_frames_written = 0
    if not args.no_videos and video_writer is not None:
        debug_log(
            args,
            f"case_video_stream_finalize_start case_idx={case_idx} frames={video_writer.frames_written}",
            log_path=debug_log_path,
        )
        video, video_error = video_writer.close()
        video_path = None if video is None else str(video)
        video_frames_written = int(video_writer.frames_written)
        debug_log(
            args,
            f"case_video_stream_finalize_done case_idx={case_idx} path={video_path} error={video_error}",
            log_path=debug_log_path,
        )
    final_info = current_info
    summary = {
        **asdict(case),
        "success": bool(success),
        "success_metric": "task_target_l2",
        "goal_tolerance": float(goal_tolerance),
        "policy_steps_executed": int(policy_steps_executed),
        "steps_executed": int(policy_steps_executed),
        "stop_reason": stop_reason,
        "episode_seed": int(episode["episode_seed"]),
        "dataset_terminated": bool(episode["terminated"]),
        "dataset_truncated": bool(episode["truncated"]),
        "mode": str(episode["mode"]),
        "camera": camera,
        "dataset_max_episode_steps": int(episode["max_episode_steps"]),
        "control_decimation": int(control_decimation),
        "control_timestep": float(episode["control_timestep"]),
        "initial_task_target_distance": float(task_target_distances[0]),
        "final_task_target_distance": float(task_target_distances[-1]),
        "min_task_target_distance": float(np.min(task_target_distances)),
        "initial_left_attachment_distance": float(left_attachment_distances[0]),
        "final_left_attachment_distance": float(left_attachment_distances[-1]),
        "min_left_attachment_distance": float(np.min(left_attachment_distances)),
        "initial_right_attachment_distance": float(right_attachment_distances[0]),
        "final_right_attachment_distance": float(right_attachment_distances[-1]),
        "min_right_attachment_distance": float(np.min(right_attachment_distances)),
        "initial_rope_length_error": float(rope_length_errors[0]),
        "final_rope_length_error": float(rope_length_errors[-1]),
        "min_rope_length_error": float(np.min(rope_length_errors)),
        "latent_goal_distance_initial": None if not latent_goal_distances else float(latent_goal_distances[0]),
        "latent_goal_distance_final": None if not latent_goal_distances else float(latent_goal_distances[-1]),
        "embedding_goal_distance_initial": None if not embedding_goal_distances else float(embedding_goal_distances[0]),
        "embedding_goal_distance_final": None if not embedding_goal_distances else float(embedding_goal_distances[-1]),
        "goal_task_target": goal_task_target.tolist(),
        "final_task_target": np.asarray(final_info["task_target"], dtype=np.float32).tolist(),
        "goal_left_attachment_pos": goal_left_pos.tolist(),
        "goal_right_attachment_pos": goal_right_pos.tolist(),
        "final_left_attachment_pos": np.asarray(final_info["left_attachment_pos"], dtype=np.float32).tolist(),
        "final_right_attachment_pos": np.asarray(final_info["right_attachment_pos"], dtype=np.float32).tolist(),
        "goal_rope_length": goal_rope_length.tolist(),
        "final_rope_length": np.asarray(final_info["rope_length"], dtype=np.float32).tolist(),
        "final_qpos": np.asarray(final_info["qpos"], dtype=np.float32).tolist(),
        "final_qvel": np.asarray(final_info["qvel"], dtype=np.float32).tolist(),
        "final_control": np.asarray(final_info["control"], dtype=np.float32).tolist(),
        "video_path": video_path,
        "video_error": video_error,
        "video_frames_written": int(video_frames_written),
        "task_target_distances": task_target_distances,
        "left_attachment_distances": left_attachment_distances,
        "right_attachment_distances": right_attachment_distances,
        "rope_length_errors": rope_length_errors,
        "latent_goal_distances": latent_goal_distances,
        "embedding_goal_distances": embedding_goal_distances,
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "dataset_start_pixel_l2": float(np.linalg.norm(start_frame.astype(np.float32) - pixels_np[case.start_step].astype(np.float32))),
        "dataset_goal_pixel_l2": float(np.linalg.norm(goal_frame.astype(np.float32) - pixels_np[case.goal_step].astype(np.float32))),
        "step_records": step_records,
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    debug_log(
        args,
        f"case_done case_idx={case_idx} success={success} steps={policy_steps_executed} stop={stop_reason}",
        log_path=debug_log_path,
        nvidia_smi=True,
    )
    return summary


def read_action_dim(dataset_path: Path) -> int:
    with h5py.File(dataset_path, "r") as h5:
        return int(h5["action"].shape[-1])


def build_run_metrics(
    *,
    args: argparse.Namespace,
    method_config: dict[str, Any],
    sampled_cases: list[EvalCase],
    evaluated_cases: list[EvalCase],
    case_results: list[dict[str, Any]],
    partial: bool,
) -> dict[str, Any]:
    successes = np.asarray([case["success"] for case in case_results], dtype=bool)
    success_rate = 0.0 if successes.size == 0 else float(np.mean(successes) * 100.0)
    return {
        "partial": bool(partial),
        "success_rate": success_rate,
        "episode_successes": successes.astype(int).tolist(),
        "success_metric": "task_target_l2",
        "method": args.method,
        "method_config": method_config,
        "dataset_path": str(args.dataset_path),
        "seed": int(args.seed),
        "num_eval": len(case_results),
        "requested_num_eval": int(args.num_eval),
        "sampled_case_count": len(sampled_cases),
        "case_offset": int(args.case_offset),
        "case_count": None if args.case_count is None else int(args.case_count),
        "evaluated_case_count": len(evaluated_cases),
        "sampled_episode_indices": [int(case.episode_idx) for case in sampled_cases],
        "evaluated_episode_indices": [int(case.episode_idx) for case in evaluated_cases],
        "eval_budget": int(args.eval_budget),
        "goal_tolerance_override": None if args.goal_tolerance is None else float(args.goal_tolerance),
        "goal_protocol": "start_step_0_to_final_episode_step",
        "cases": case_results,
    }


def make_failed_case_summary(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    stop_reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    goal_tolerance = args.goal_tolerance
    try:
        episode = load_dataset_episode(args.dataset_path, case.episode_idx)
        if goal_tolerance is None:
            goal_tolerance = float(episode["goal_tolerance"])
    except Exception:
        goal_tolerance = None
    mp4_path = case_dir / "rollout.mp4"
    has_video = False if args.no_videos else wait_for_video_file(mp4_path, timeout=10.0)
    video_error = None
    if not args.no_videos:
        if has_video:
            video_error = f"{stop_reason}: rollout.mp4 may contain only frames written before the worker failed"
        else:
            video_error = "worker exited before rollout.mp4 was finalized"
    summary = {
        **asdict(case),
        "success": False,
        "success_metric": "task_target_l2",
        "goal_tolerance": None if goal_tolerance is None else float(goal_tolerance),
        "policy_steps_executed": 0,
        "steps_executed": 0,
        "stop_reason": stop_reason,
        "worker_error": error,
        "initial_task_target_distance": None,
        "final_task_target_distance": None,
        "min_task_target_distance": None,
        "initial_left_attachment_distance": None,
        "final_left_attachment_distance": None,
        "min_left_attachment_distance": None,
        "initial_right_attachment_distance": None,
        "final_right_attachment_distance": None,
        "min_right_attachment_distance": None,
        "initial_rope_length_error": None,
        "final_rope_length_error": None,
        "min_rope_length_error": None,
        "video_path": str(mp4_path) if has_video else None,
        "video_error": video_error,
        "video_frames_written": None,
        "step_records": [],
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def isolated_case_worker(
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    action_dim: int,
    result_queue: multiprocessing.Queue,
) -> None:
    debug_log_path = args.debug_log.expanduser().resolve() if args.debug_log is not None else out_root / "debug.log"
    try:
        debug_log(args, f"worker_start case_idx={case_idx} episode_idx={case.episode_idx}", log_path=debug_log_path, nvidia_smi=True)
        device = require_device(args.device)
        debug_log(args, f"worker_device_ready case_idx={case_idx} device={device}", log_path=debug_log_path)
        ilqr_assets = None
        swm_policy = None
        swm_history_size = None
        if args.method == "ilqr":
            debug_log(args, f"worker_load_ilqr_start case_idx={case_idx}", log_path=debug_log_path)
            ilqr_assets = load_ilqr_assets(args, device)
            debug_log(args, f"worker_load_ilqr_done case_idx={case_idx}", log_path=debug_log_path, nvidia_smi=True)
        else:
            debug_log(args, f"worker_make_swm_start case_idx={case_idx}", log_path=debug_log_path)
            swm_policy, swm_history_size, _ = make_swm_policy(args, device, action_dim)
            debug_log(args, f"worker_make_swm_done case_idx={case_idx}", log_path=debug_log_path, nvidia_smi=True)

        probe_episode = load_dataset_episode(args.dataset_path, case.episode_idx)
        debug_log(args, f"worker_labenv_start case_idx={case_idx}", log_path=debug_log_path)
        env = LabEnv()
        debug_log(args, f"worker_labenv_done case_idx={case_idx}", log_path=debug_log_path)
        debug_log(args, f"worker_renderer_start case_idx={case_idx}", log_path=debug_log_path)
        with mujoco.Renderer(
            env.model,
            height=int(probe_episode["height"]),
            width=int(probe_episode["width"]),
        ) as renderer:
            debug_log(args, f"worker_renderer_done case_idx={case_idx}", log_path=debug_log_path)
            summary = run_case(
                args=args,
                case=case,
                case_idx=case_idx,
                out_root=out_root,
                env=env,
                renderer=renderer,
                device=device,
                ilqr_assets=ilqr_assets,
                swm_policy=swm_policy,
                swm_history_size=swm_history_size,
                action_dim=action_dim,
                debug_log_path=debug_log_path,
            )
        debug_log(args, f"worker_case_done case_idx={case_idx}", log_path=debug_log_path, nvidia_smi=True)
        result_queue.put({"ok": True, "summary": summary})
        debug_log(args, f"worker_result_put case_idx={case_idx}", log_path=debug_log_path)
    except BaseException:
        tb = traceback.format_exc()
        debug_log(args, f"worker_exception case_idx={case_idx} traceback={tb}", log_path=debug_log_path, nvidia_smi=True)
        result_queue.put({"ok": False, "traceback": tb})


def run_case_isolated(
    *,
    args: argparse.Namespace,
    case: EvalCase,
    case_idx: int,
    out_root: Path,
    action_dim: int,
) -> dict[str, Any]:
    debug_log_path = args.debug_log.expanduser().resolve() if args.debug_log is not None else out_root / "debug.log"
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    debug_log(args, f"parent_spawn_start case_idx={case_idx} episode_idx={case.episode_idx}", log_path=debug_log_path, nvidia_smi=True)
    proc = ctx.Process(
        target=isolated_case_worker,
        args=(args, case, case_idx, out_root, action_dim, result_queue),
    )
    proc.start()
    debug_log(args, f"parent_spawn_done case_idx={case_idx} worker_pid={proc.pid}", log_path=debug_log_path)
    proc.join()
    debug_log(args, f"parent_join_done case_idx={case_idx} exitcode={proc.exitcode}", log_path=debug_log_path, nvidia_smi=True)
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = None
    finally:
        result_queue.close()
        result_queue.join_thread()

    if proc.exitcode == 0 and result and result.get("ok"):
        debug_log(args, f"parent_case_success case_idx={case_idx}", log_path=debug_log_path)
        return finalize_case_video(
            args=args,
            case=case,
            case_idx=case_idx,
            out_root=out_root,
            summary=result["summary"],
            debug_log_path=debug_log_path,
        )
    if result and not result.get("ok"):
        debug_log(args, f"parent_case_python_exception case_idx={case_idx}", log_path=debug_log_path)
        summary = make_failed_case_summary(
            args=args,
            case=case,
            case_idx=case_idx,
            out_root=out_root,
            stop_reason="worker_python_exception",
            error=result.get("traceback"),
        )
        return finalize_case_video(
            args=args,
            case=case,
            case_idx=case_idx,
            out_root=out_root,
            summary=summary,
            debug_log_path=debug_log_path,
        )
    debug_log(args, f"parent_case_native_exit case_idx={case_idx} exitcode={proc.exitcode}", log_path=debug_log_path)
    summary = make_failed_case_summary(
        args=args,
        case=case,
        case_idx=case_idx,
        out_root=out_root,
        stop_reason=f"worker_native_exit_{proc.exitcode}",
        error=None,
    )
    return finalize_case_video(
        args=args,
        case=case,
        case_idx=case_idx,
        out_root=out_root,
        summary=summary,
        debug_log_path=debug_log_path,
    )


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.eval_budget < 1:
        raise ValueError("--eval-budget must be positive.")
    if args.goal_tolerance is not None and args.goal_tolerance <= 0:
        raise ValueError("--goal-tolerance must be positive.")
    if args.case_offset < 0:
        raise ValueError("--case-offset must be non-negative.")
    if args.case_count is not None and args.case_count < 1:
        raise ValueError("--case-count must be positive when provided.")

    device = require_device(args.device)
    ep_len = load_episode_lengths(args.dataset_path)
    sampled_cases = sample_eval_cases(args, ep_len)
    cases = slice_eval_cases(args, sampled_cases)
    action_dim = read_action_dim(args.dataset_path)
    validate_parent_swm_args(args, action_dim)

    run_name = f"{int(time.time())}_{args.method}_seed_{args.seed}"
    out_root = args.out_dir.expanduser().resolve() / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    debug_log_path = args.debug_log.expanduser().resolve() if args.debug_log is not None else out_root / "debug.log"
    if args.debug_log is None:
        args.debug_log = debug_log_path
    debug_log(
        args,
        (
            f"run_start method={args.method} num_eval={args.num_eval} budget={args.eval_budget} "
            f"case_offset={args.case_offset} case_count={args.case_count} actual_cases={len(cases)} "
            f"isolate_cases={args.isolate_cases} videos={not args.no_videos} out_root={out_root}"
        ),
        log_path=debug_log_path,
        nvidia_smi=True,
    )

    ilqr_assets = None
    swm_policy = None
    swm_history_size = None
    method_config: dict[str, Any]
    if args.method == "ilqr":
        if args.isolate_cases:
            model_dir = args.model_dir.expanduser().resolve()
            config = load_config(model_dir)
            checkpoint_path = (
                args.checkpoint.expanduser().resolve()
                if args.checkpoint is not None
                else latest_object_checkpoint(model_dir).resolve()
            )
            method_config = {
                "model_dir": str(model_dir),
                "checkpoint": str(checkpoint_path),
                "img_size": int(config.get("img_size", 224)),
                "action_dim": int(config.get("action_dim", action_dim)),
                "horizon": int(args.horizon),
                "q_terminal": float(args.q_terminal),
                "q_stage": float(args.q_stage),
                "r_control": float(args.r_control),
                "ilqr_max_iters": int(args.ilqr_max_iters),
            }
        else:
            ilqr_assets = load_ilqr_assets(args, device)
            method_config = ilqr_assets[2]
    else:
        if args.isolate_cases:
            policy_name = require_swm_policy(args)
            process_keys = args.process_key if args.process_key is not None else ["action"]
            method_config = {
                "policy": policy_name,
                "solver_config": str(args.solver_config) if args.method == "swm_cost" else None,
                "plan_config": None
                if args.method != "swm_cost"
                else {
                    "horizon": int(args.plan_horizon),
                    "receding_horizon": int(args.receding_horizon),
                    "action_block": int(args.action_block),
                    "warm_start": not bool(args.no_warm_start),
                },
                "process_keys": process_keys,
                "action_dim": int(action_dim),
                "swm_img_size": int(args.swm_img_size),
                "isolated_case_workers": True,
            }
        else:
            swm_policy, swm_history_size, method_config = make_swm_policy(args, device, action_dim)

    case_results = []
    if args.isolate_cases:
        for case_idx, case in enumerate(tqdm(cases, desc="Hard rope eval")):
            debug_log(args, f"run_case_loop_start case_idx={case_idx}", log_path=debug_log_path)
            case_results.append(
                run_case_isolated(
                    args=args,
                    case=case,
                    case_idx=case_idx,
                    out_root=out_root,
                    action_dim=action_dim,
                )
            )
            debug_log(args, f"run_case_loop_done case_idx={case_idx}", log_path=debug_log_path)
            partial_metrics = build_run_metrics(
                args=args,
                method_config=method_config,
                sampled_cases=sampled_cases,
                evaluated_cases=cases,
                case_results=case_results,
                partial=True,
            )
            with (out_root / "metrics_partial.json").open("w", encoding="utf-8") as handle:
                json.dump(partial_metrics, handle, indent=2)
    else:
        probe_episode = load_dataset_episode(args.dataset_path, cases[0].episode_idx)
        debug_log(args, "run_labenv_start", log_path=debug_log_path)
        env = LabEnv()
        debug_log(args, "run_labenv_done", log_path=debug_log_path)
        debug_log(args, "run_renderer_start", log_path=debug_log_path)
        with mujoco.Renderer(
            env.model,
            height=int(probe_episode["height"]),
            width=int(probe_episode["width"]),
        ) as renderer:
            debug_log(args, "run_renderer_done", log_path=debug_log_path)
            for case_idx, case in enumerate(tqdm(cases, desc="Hard rope eval")):
                debug_log(args, f"run_case_loop_start case_idx={case_idx}", log_path=debug_log_path)
                case_results.append(
                    run_case(
                        args=args,
                        case=case,
                        case_idx=case_idx,
                        out_root=out_root,
                        env=env,
                        renderer=renderer,
                        device=device,
                        ilqr_assets=ilqr_assets,
                        swm_policy=swm_policy,
                        swm_history_size=swm_history_size,
                        action_dim=action_dim,
                        debug_log_path=debug_log_path,
                    )
                )
                debug_log(args, f"run_case_loop_done case_idx={case_idx}", log_path=debug_log_path)
                partial_metrics = build_run_metrics(
                    args=args,
                    method_config=method_config,
                    sampled_cases=sampled_cases,
                    evaluated_cases=cases,
                    case_results=case_results,
                    partial=True,
                )
                with (out_root / "metrics_partial.json").open("w", encoding="utf-8") as handle:
                    json.dump(partial_metrics, handle, indent=2)

    metrics = build_run_metrics(
        args=args,
        method_config=method_config,
        sampled_cases=sampled_cases,
        evaluated_cases=cases,
        case_results=case_results,
        partial=False,
    )
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    debug_log(args, f"run_done success_rate={metrics['success_rate']:.2f} cases={len(case_results)}", log_path=debug_log_path, nvidia_smi=True)

    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
