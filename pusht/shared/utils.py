from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_PUSHT_ENV_ID = "gym_pusht/PushT-v0"
DEFAULT_EXPERT_MODEL_DIR = Path("pusht/models/diffusion_pusht")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_LEROBOT_SRC = _REPO_ROOT / "third_party" / "lerobot" / "src"


@dataclass
class ExpertPolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device


def ensure_lerobot_available() -> None:
    """Use the repo's LeRobot checkout when the active environment needs it."""
    try:
        from lerobot.configs import PreTrainedConfig as _PreTrainedConfig  # noqa: F401

        return
    except (ImportError, SyntaxError) as exc:
        import_error = exc

    if _VENDORED_LEROBOT_SRC.exists():
        if sys.version_info < (3, 12):
            raise ImportError(
                "The LeRobot checkout in third_party/lerobot requires Python 3.12+. "
                f"You are running Python {sys.version_info.major}.{sys.version_info.minor} "
                f"from {sys.executable}. Run this script with the Python 3.12 environment, e.g. "
                "`/home/devesh/latent-brs/temp_venv/bin/python pusht/eval/expert_viz.py`, "
                "or recreate the active venv with Python 3.12 and install LeRobot's pusht extra."
            ) from import_error

        vendor_path = str(_VENDORED_LEROBOT_SRC)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        for module_name in list(sys.modules):
            if module_name == "lerobot" or module_name.startswith("lerobot."):
                del sys.modules[module_name]
        try:
            from lerobot.configs import PreTrainedConfig as _PreTrainedConfig  # noqa: F401

            return
        except (ImportError, SyntaxError) as exc:
            import_error = exc

    raise ModuleNotFoundError(
        "A compatible LeRobot install is required. Install LeRobot in the active environment, "
        "or use this repo's Python 3.12 environment at /home/devesh/latent-brs/temp_venv."
    ) from import_error


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_pusht_env(
    env_id: str = DEFAULT_PUSHT_ENV_ID,
    *,
    obs_type: str = "pixels_agent_pos",
    render_mode: str = "rgb_array",
    max_episode_steps: int = 300,
    visualization_width: int = 384,
    visualization_height: int = 384,
):
    """Create a PushT Gymnasium environment.

    This prefers Hugging Face's `gym-pusht` package because LeRobot's PushT
    policies are trained against that action/observation convention.
    """
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install gymnasium in the active environment.") from exc

    package_name = env_id.split("/", maxsplit=1)[0] if "/" in env_id else None
    if package_name:
        try:
            __import__(package_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Environment package '{package_name}' is not installed. "
                "For the default PushT env, install LeRobot's pusht extra or gym-pusht."
            ) from exc

    kwargs = {
        "obs_type": obs_type,
        "render_mode": render_mode,
        "max_episode_steps": max_episode_steps,
        "visualization_width": visualization_width,
        "visualization_height": visualization_height,
    }
    try:
        return gym.make(env_id, disable_env_checker=True, **kwargs)
    except TypeError:
        # Some local PushT variants do not accept the visualization kwargs.
        kwargs.pop("visualization_width", None)
        kwargs.pop("visualization_height", None)
        return gym.make(env_id, disable_env_checker=True, **kwargs)


def load_expert_policy_bundle(model_dir: str | Path = DEFAULT_EXPERT_MODEL_DIR, device: str = "auto") -> ExpertPolicyBundle:
    """Load the saved PushT diffusion expert without importing LeRobot."""
    from pusht.shared.native_diffusion_policy import load_native_diffusion_policy_bundle

    model_dir = Path(model_dir)
    resolved_device = resolve_device(device)
    policy, preprocessor, postprocessor = load_native_diffusion_policy_bundle(model_dir, resolved_device)
    return ExpertPolicyBundle(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, device=resolved_device)


def _extract_agent_pos(observation: dict[str, Any]) -> np.ndarray:
    if "agent_pos" in observation:
        return np.asarray(observation["agent_pos"], dtype=np.float32)
    if "proprio" in observation:
        return np.asarray(observation["proprio"][:2], dtype=np.float32)
    if "state" in observation:
        return np.asarray(observation["state"][:2], dtype=np.float32)
    raise KeyError("Could not find agent position in observation. Expected 'agent_pos', 'proprio', or 'state'.")


def _extract_pixels(observation: dict[str, Any], env: Any | None = None) -> np.ndarray:
    if "pixels" in observation:
        pixels = observation["pixels"]
    elif "image" in observation:
        pixels = observation["image"]
    elif env is not None:
        pixels = env.render()
    else:
        raise KeyError("Could not find image pixels in observation and no env was provided for rendering.")

    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        if pixels.max() <= 1.0:
            pixels = pixels * 255.0
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return pixels


def _resize_hwc_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def pusht_observation_to_lerobot_batch(
    observation: dict[str, Any],
    *,
    env: Any | None,
    image_shape: tuple[int, int, int] = (3, 96, 96),
) -> dict[str, torch.Tensor]:
    """Convert a PushT env observation into the keys expected by the expert policy."""
    channels, height, width = image_shape
    if channels != 3:
        raise ValueError(f"Expected a 3-channel image feature, got {image_shape=}.")

    pixels = _extract_pixels(observation, env)
    pixels = _resize_hwc_uint8(pixels, height, width)
    image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float() / 255.0
    state = torch.from_numpy(_extract_agent_pos(observation)).float()

    return {
        "observation.image": image.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
    }


def env_action_from_policy_action(
    policy_action: np.ndarray,
    *,
    env: Any,
    observation: dict[str, Any],
    action_mode: str = "auto",
) -> np.ndarray:
    """Convert absolute LeRobot PushT actions to the active env's action convention."""
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)[:2]
    if action_mode == "absolute":
        return action
    if action_mode == "relative":
        agent_pos = _extract_agent_pos(observation)
        return np.clip((action - agent_pos) / 100.0, -1.0, 1.0).astype(np.float32)
    if action_mode != "auto":
        raise ValueError(f"Unknown action_mode '{action_mode}'. Use auto, absolute, or relative.")

    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        high = np.asarray(action_space.high)
        low = np.asarray(action_space.low)
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            agent_pos = _extract_agent_pos(observation)
            return np.clip((action - agent_pos) / 100.0, low, high).astype(np.float32)
    return action


@torch.no_grad()
def select_expert_action(
    bundle: ExpertPolicyBundle,
    observation: dict[str, Any],
    *,
    env: Any,
    action_mode: str = "auto",
) -> np.ndarray:
    image_shape = tuple(bundle.policy.config.input_features["observation.image"].shape)
    batch = pusht_observation_to_lerobot_batch(observation, env=env, image_shape=image_shape)
    batch = bundle.preprocessor(batch)
    action = bundle.policy.select_action(batch)
    action = bundle.postprocessor(action)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return env_action_from_policy_action(action, env=env, observation=observation, action_mode=action_mode)


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
