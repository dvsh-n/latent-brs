from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_OGBENCH_ENV_TYPE = "cube"
DEFAULT_OGBENCH_CUBE_TYPE = "single"
DEFAULT_OGBENCH_OB_TYPE = "pixels"
DEFAULT_IMG_WIDTH = 224
DEFAULT_IMG_HEIGHT = 224
DEFAULT_ACTION_DIM = 5


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_ogbench_env(
    env_type: str = DEFAULT_OGBENCH_ENV_TYPE,
    *,
    cube_type: str = DEFAULT_OGBENCH_CUBE_TYPE,
    ob_type: str = DEFAULT_OGBENCH_OB_TYPE,
    width: int = DEFAULT_IMG_WIDTH,
    height: int = DEFAULT_IMG_HEIGHT,
    **kwargs,
):
    """Create an OGBench environment (Cube, Scene, or PointMaze).

    Args:
        env_type: One of 'cube', 'scene', or 'pointmaze'.
        cube_type: For cube envs, one of 'single', 'double', 'triple',
            'quadruple', 'octuple'.
        ob_type: Observation type, either 'pixels' or 'states'.
        width: Render width in pixels.
        height: Render height in pixels.
        **kwargs: Additional keyword arguments forwarded to the env constructor.
    """
    from stable_worldmodel.envs.ogbench import CubeEnv, PointMazeEnv, SceneEnv

    if env_type == "cube":
        return CubeEnv(
            env_type=cube_type,
            ob_type=ob_type,
            width=width,
            height=height,
            **kwargs,
        )
    elif env_type == "scene":
        return SceneEnv(
            ob_type=ob_type,
            **kwargs,
        )
    elif env_type == "pointmaze":
        return PointMazeEnv(
            ob_type=ob_type,
            width=width,
            height=height,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown env_type '{env_type}'. Use 'cube', 'scene', or 'pointmaze'."
        )


def render_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, tuple):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame
