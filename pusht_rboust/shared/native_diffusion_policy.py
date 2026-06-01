"""Shared loaders for the native PushT diffusion policy runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from pusht_rboust.test.native_diffusion_policy import (
    ACTION,
    OBS_STATE,
    NativePolicyPostprocessor,
    NativePolicyPreprocessor,
    load_native_diffusion_policy_bundle as _load_safetensors_bundle,
)


def _nested_stats_from_state_dict(flat: dict[str, Tensor], *, device: torch.device) -> dict[str, dict[str, Tensor]]:
    key_prefix_map = {
        "normalize_inputs.buffer_observation_image.": "observation.image.",
        "normalize_inputs.buffer_observation_state.": f"{OBS_STATE}.",
        "normalize_targets.buffer_action.": f"{ACTION}.",
        "unnormalize_outputs.buffer_action.": f"{ACTION}.",
    }
    nested: dict[str, dict[str, Tensor]] = {}
    for flat_key, value in flat.items():
        remapped = None
        for prefix, target_prefix in key_prefix_map.items():
            if flat_key.startswith(prefix):
                remapped = f"{target_prefix}{flat_key[len(prefix):]}"
                break
        if remapped is None:
            continue
        key, stat_name = remapped.rsplit(".", 1)
        nested.setdefault(key, {})[stat_name] = value.to(device=device, dtype=torch.float32)
    return nested


class NativePolicyPreprocessorFromStateDict(NativePolicyPreprocessor):
    def __init__(self, stats_state_dict: dict[str, Tensor], device: torch.device, eps: float = 1e-8):
        self.stats = _nested_stats_from_state_dict(stats_state_dict, device=device)
        self.device = device
        self.eps = eps


class NativePolicyPostprocessorFromStateDict(NativePolicyPostprocessor):
    def __init__(self, stats_state_dict: dict[str, Tensor], eps: float = 1e-8):
        self.stats = _nested_stats_from_state_dict(stats_state_dict, device=torch.device("cpu"))
        self.eps = eps


def load_native_diffusion_policy_ckpt_bundle(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint at {checkpoint_path}, got {type(checkpoint).__name__}.")
    if "policy" not in checkpoint or "normalization_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} must contain 'policy' and 'normalization_state_dict' keys."
        )

    policy = checkpoint["policy"].to(device)
    policy.eval()
    policy.requires_grad_(False)
    normalization_state_dict = checkpoint["normalization_state_dict"]
    if not isinstance(normalization_state_dict, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has invalid normalization_state_dict type "
            f"{type(normalization_state_dict).__name__}."
        )

    preprocessor = NativePolicyPreprocessorFromStateDict(normalization_state_dict, device=device)
    postprocessor = NativePolicyPostprocessorFromStateDict(normalization_state_dict)
    return policy, preprocessor, postprocessor


def load_native_diffusion_policy_bundle(model_dir: str | Path, device: torch.device):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "model.ckpt"
    if checkpoint_path.is_file():
        return load_native_diffusion_policy_ckpt_bundle(checkpoint_path, device)
    return _load_safetensors_bundle(model_dir, device)
