#!/usr/bin/env python3
"""Run random latent rollouts in the reacher latent-safety env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reacher.safety.latent_env import ReacherLatentSafetyEnv, load_latent_safety_components

DEFAULT_CACHE_PATH = "reacher/safety/cache/reacher_latent_safety_smoke.pt"
DEFAULT_CLASSIFIER_CHECKPOINT = "reacher/Haoran_obs_data/obs_net_sm_model/8acfaa546b7cc1b6/model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=Path(DEFAULT_CACHE_PATH))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle", choices=("knn", "classifier"), default="knn")
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER_CHECKPOINT))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--allow-classifier-latent-slice", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--optimistic-knn", action="store_true")
    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive.")

    components = load_latent_safety_components(
        cache_path=args.cache_path,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device_arg=args.device,
        oracle_kind=args.oracle,
        classifier_checkpoint=args.classifier_checkpoint,
        classifier_threshold=str(args.classifier_threshold),
        allow_classifier_latent_slice=bool(args.allow_classifier_latent_slice),
        knn_k=args.knn_k,
        pessimistic=not args.optimistic_knn,
    )
    env = ReacherLatentSafetyEnv(
        dynamics=components.dynamics,
        cache=components.cache,
        oracle=components.oracle,
        device=components.device,
        max_episode_steps=args.max_episode_steps,
        action_low=args.action_low,
        action_high=args.action_high,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    episodes = []
    for episode_idx in range(args.episodes):
        _, info = env.reset(seed=args.seed + episode_idx)
        margins = [float(info["safety_margin"])]
        failed = bool(info["is_failure"])
        steps = 0
        while not failed and steps < args.max_episode_steps:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            _, reward, terminated, truncated, _info = env.step(action)
            margins.append(float(reward))
            steps += 1
            failed = bool(terminated)
            if truncated:
                break
        episodes.append(
            {
                "episode": episode_idx,
                "steps": steps,
                "failed": failed,
                "min_margin": float(np.min(margins)),
                "mean_margin": float(np.mean(margins)),
                "start_margin": float(margins[0]),
                "end_margin": float(margins[-1]),
            }
        )

    min_margins = np.asarray([item["min_margin"] for item in episodes], dtype=np.float32)
    summary = {
        "cache_path": str(args.cache_path.expanduser().resolve()),
        "device": str(components.device),
        "oracle": args.oracle,
        "episodes": episodes,
        "failure_rate": float(np.mean([item["failed"] for item in episodes])),
        "mean_min_margin": float(np.mean(min_margins)),
        "worst_min_margin": float(np.min(min_margins)),
        "metadata": jsonable(components.cache["metadata"]),
        "note": "Random smoke run only: this is not planning and not PyHJ training.",
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
