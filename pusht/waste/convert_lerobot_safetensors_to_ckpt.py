from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from pusht.test.native_diffusion_policy import NativeDiffusionConfig, NativeDiffusionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot PushT diffusion model.safetensors export into a torch .ckpt file."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("pusht/models"),
        help="Directory containing model.safetensors and config.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to <model-dir>/model.ckpt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to instantiate and load weights on before saving.",
    )
    return parser.parse_args()


def split_normalization_tensors(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    normalization_prefixes = (
        "normalize_inputs.",
        "normalize_targets.",
        "unnormalize_outputs.",
        "normalize.",
        "unnormalize.",
        "input_normalizer.",
        "output_normalizer.",
        "normalalize_inputs.",
        "unnormalize_targets.",
    )
    model_tensors: dict[str, torch.Tensor] = {}
    normalization_tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        target = normalization_tensors if key.startswith(normalization_prefixes) else model_tensors
        target[key] = value
    return model_tensors, normalization_tensors


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    safetensors_path = model_dir / "model.safetensors"
    output_path = (args.output or (model_dir / "model.ckpt")).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors: {safetensors_path}")

    device = torch.device(args.device)
    config = NativeDiffusionConfig.from_json(config_path)
    policy = NativeDiffusionPolicy(config).to(device)
    raw_state_dict = load_file(safetensors_path, device=str(device))
    model_state_dict, normalization_state_dict = split_normalization_tensors(raw_state_dict)
    policy.load_state_dict(model_state_dict, strict=True)
    policy.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open() as f:
        raw_config = json.load(f)

    checkpoint = {
        "format": "lerobot_native_diffusion_policy",
        "config": raw_config,
        "normalization_state_dict": normalization_state_dict,
        "policy": policy,
    }
    torch.save(checkpoint, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
