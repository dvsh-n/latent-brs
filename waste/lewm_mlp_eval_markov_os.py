#!/usr/bin/env python3
"""Report one-step Markov MLP LE-WM error norm quantiles from saved error data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_DATA_PATH = "data/lewm_one_step_error_data.pt"
DEFAULT_OUT_DIR = "eval/lewm_mlpdyn_markov_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-name", default="one_step_error_quantiles.json")
    parser.add_argument(
        "--error-space",
        choices=("full-state", "latent", "delta"),
        default="full-state",
        help="Which part of the saved Markov-state error vector to measure.",
    )
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    return parser.parse_args()


def select_error_space(error: torch.Tensor, metadata: dict[str, object], error_space: str) -> torch.Tensor:
    if error.ndim < 2:
        raise ValueError(f"Expected error tensor with shape (num_samples, error_dim), got {tuple(error.shape)}.")
    error = error.reshape(error.shape[0], -1).float()
    if error_space == "full-state":
        return error

    embed_dim = metadata.get("embed_dim")
    if embed_dim is None:
        if error.shape[-1] % 2 != 0:
            raise ValueError("Cannot infer embed_dim from an odd-sized error vector.")
        embed_dim = error.shape[-1] // 2
    embed_dim = int(embed_dim)
    if error.shape[-1] < 2 * embed_dim:
        raise ValueError(f"error_dim={error.shape[-1]} is too small for embed_dim={embed_dim}.")

    if error_space == "latent":
        return error[:, :embed_dim]
    if error_space == "delta":
        return error[:, embed_dim : 2 * embed_dim]
    raise ValueError(f"Unknown error space: {error_space}")


def summarize_norms(norms: torch.Tensor, quantiles: list[float]) -> dict[str, object]:
    if norms.numel() == 0:
        raise ValueError("No finite error norms found.")
    q_tensor = torch.tensor(quantiles, dtype=norms.dtype)
    q_values = torch.quantile(norms, q_tensor)
    return {
        "num_samples": int(norms.numel()),
        "mean_l2": float(norms.mean()),
        "std_l2": float(norms.std(unbiased=False)),
        "min_l2": float(norms.min()),
        "max_l2": float(norms.max()),
        "quantiles": {f"{int(q * 100)}%": float(value) for q, value in zip(quantiles, q_values)},
    }


def main() -> None:
    args = parse_args()
    if any(q < 0.0 or q > 1.0 for q in args.quantiles):
        raise ValueError("--quantiles values must be in [0, 1].")

    data_path = args.data_path.expanduser().resolve()
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict payload in {data_path}, got {type(payload).__name__}.")
    if "error" not in payload or not torch.is_tensor(payload["error"]):
        raise KeyError(f"Expected tensor key 'error' in {data_path}.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    error = select_error_space(payload["error"], metadata, args.error_space)
    norms = torch.linalg.vector_norm(error, ord=2, dim=-1)
    norms = norms[torch.isfinite(norms)].contiguous()

    summary = summarize_norms(norms, args.quantiles)
    summary.update(
        {
            "data_path": str(data_path),
            "error_space": args.error_space,
            "error_dim": int(error.shape[-1]),
            "source_metadata": metadata,
        }
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output_name
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"output_path": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
