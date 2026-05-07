from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Normalize a (N, C, H, W) float [0, 1] tensor with ImageNet stats."""
    return (pixels - IMAGENET_MEAN) / IMAGENET_STD


def resize_hwc_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def compute_action_stats(
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean and std from an array of actions, ignoring NaNs."""
    finite = actions[~np.isnan(actions).any(axis=1)]
    if finite.size == 0:
        raise ValueError("No finite actions found in dataset.")
    mean = finite.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(finite.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
    return mean, std
