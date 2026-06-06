#!/usr/bin/env python3
"""Evaluate a trained ogbench PyHJ policy/value function against cached margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ogbench.safety.train_pyhj_cube import build_policy
from ogbench.safety.latent_env import OGBenchCubeLatentSafetyEnv, load_latent_safety_components

from PyHJ.data import Batch  # noqa: E402

DEFAULT_CACHE_PATH = "ogbench/safety/cache/cube_latent_safety_classifier_train_tanh2.pt"
DEFAULT_POLICY_PATH = "ogbench/safety/runs/pyhj_cube_train_tanh2/policy_latest.pth"
DEFAULT_CLASSIFIER_CHECKPOINT = "ogbench/safety/obs_net/model.pt"


def parse_hidden_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.replace(",", " ").split() if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer hidden sizes, got {value!r}") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("Hidden sizes must be one or more positive integers.")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=Path(DEFAULT_CACHE_PATH))
    parser.add_argument("--policy-path", type=Path, default=Path(DEFAULT_POLICY_PATH))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path(DEFAULT_CLASSIFIER_CHECKPOINT))
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=25)
    parser.add_argument("--value-samples", type=int, default=20000)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=parse_hidden_sizes("512 512 512 512"))
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--actor-gradient-steps", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--policy-variant", choices=("dinowm", "generic"), default="dinowm")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUC where larger score should indicate label==1."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        stop = start + 1
        while stop < scores.shape[0] and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        avg_rank = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = avg_rank
        start = stop
    pos_rank_sum = float(np.sum(ranks[pos]))
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def policy_action(policy: Any, obs: np.ndarray, *, device: torch.device) -> np.ndarray:
    batch = Batch(obs=np.asarray(obs, dtype=np.float32)[None, :], info=Batch())
    raw = policy(batch, model="actor").act
    raw_np = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
    mapped = policy.map_action(raw_np)
    return np.asarray(mapped[0], dtype=np.float32)


def rollout_policy(
    components: Any,
    policy: Any | None,
    *,
    episodes: int,
    seed: int,
    max_episode_steps: int,
    action_low: float,
    action_high: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = OGBenchCubeLatentSafetyEnv(
        dynamics=components.dynamics,
        cache=components.cache,
        oracle=components.oracle,
        device=components.device,
        max_episode_steps=max_episode_steps,
        action_low=action_low,
        action_high=action_high,
        seed=seed,
    )
    summaries: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        obs, info = env.reset(seed=seed + episode)
        margins = [float(info["safety_margin"])]
        initial_failure = bool(info["is_failure"])
        failed = initial_failure
        rollout_failure = False
        steps = 0
        while not failed and steps < max_episode_steps:
            if policy is None:
                action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            else:
                action = policy_action(policy, obs, device=components.device)
            obs, reward, terminated, truncated, _info = env.step(action)
            margins.append(float(reward))
            steps += 1
            failed = bool(terminated)
            rollout_failure = bool(terminated)
            if truncated:
                break
        summaries.append(
            {
                "episode": episode,
                "steps": steps,
                "failed": bool(failed),
                "initial_failure": bool(initial_failure),
                "rollout_failure": bool(rollout_failure),
                "min_margin": float(np.min(margins)),
                "mean_margin": float(np.mean(margins)),
                "start_margin": float(margins[0]),
                "end_margin": float(margins[-1]),
            }
        )
    min_margins = np.asarray([item["min_margin"] for item in summaries], dtype=np.float32)
    failures = np.asarray([item["failed"] for item in summaries], dtype=bool)
    initial_failures = np.asarray([item["initial_failure"] for item in summaries], dtype=bool)
    rollout_failures = np.asarray([item["rollout_failure"] for item in summaries], dtype=bool)
    safe_start_mask = ~initial_failures
    safe_start_rollout_failures = rollout_failures[safe_start_mask]
    safe_start_min_margins = min_margins[safe_start_mask]
    return {
        "episodes": int(episodes),
        "failure_rate": float(np.mean(failures)) if failures.size else 0.0,
        "initial_failure_rate": float(np.mean(initial_failures)) if initial_failures.size else 0.0,
        "rollout_failure_rate": float(np.mean(rollout_failures)) if rollout_failures.size else 0.0,
        "safe_start_count": int(np.sum(safe_start_mask)),
        "safe_start_rollout_failure_rate": (
            float(np.mean(safe_start_rollout_failures)) if safe_start_rollout_failures.size else 0.0
        ),
        "mean_min_margin": float(np.mean(min_margins)) if min_margins.size else 0.0,
        "safe_start_mean_min_margin": (
            float(np.mean(safe_start_min_margins)) if safe_start_min_margins.size else 0.0
        ),
        "worst_min_margin": float(np.min(min_margins)) if min_margins.size else 0.0,
        "mean_steps": float(np.mean([item["steps"] for item in summaries])) if summaries else 0.0,
        "episode_summaries": summaries,
    }


@torch.no_grad()
def evaluate_values(
    cache: dict[str, Any],
    policy: Any,
    *,
    device: torch.device,
    sample_count: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    states = cache["markov_state"].float()
    margins = cache["safety_margin"].float()
    total = int(states.shape[0])
    count = min(int(sample_count), total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False) if count < total else np.arange(total)
    values: list[np.ndarray] = []
    for start in range(0, count, int(batch_size)):
        idx = indices[start : start + int(batch_size)]
        obs = states[idx].to(device=device, dtype=torch.float32)
        batch = Batch(obs=obs, info=Batch())
        act = policy(batch, model="actor").act
        q = policy.critic(obs, act).flatten()
        values.append(q.detach().cpu().numpy().astype(np.float32))
    value_np = np.concatenate(values, axis=0)
    margin_np = margins[indices].detach().cpu().numpy().astype(np.float32)
    safe_label = (margin_np > 0.0).astype(np.int64)
    clipped_value = np.minimum(value_np, margin_np)

    unsafe = safe_label == 0
    safe = safe_label == 1
    return {
        "sample_count": int(count),
        "unsafe_fraction": float(np.mean(unsafe)) if count else 0.0,
        "critic_value_mean": float(np.mean(value_np)),
        "critic_value_safe_mean": float(np.mean(value_np[safe])) if np.any(safe) else None,
        "critic_value_unsafe_mean": float(np.mean(value_np[unsafe])) if np.any(unsafe) else None,
        "critic_value_margin_corr": safe_corr(value_np, margin_np),
        "critic_value_safe_auc": rank_auc(value_np, safe_label),
        "clipped_value_mean": float(np.mean(clipped_value)),
        "clipped_value_safe_mean": float(np.mean(clipped_value[safe])) if np.any(safe) else None,
        "clipped_value_unsafe_mean": float(np.mean(clipped_value[unsafe])) if np.any(unsafe) else None,
        "clipped_value_margin_corr": safe_corr(clipped_value, margin_np),
        "clipped_value_safe_auc": rank_auc(clipped_value, safe_label),
        "margin_mean": float(np.mean(margin_np)),
        "margin_safe_mean": float(np.mean(margin_np[safe])) if np.any(safe) else None,
        "margin_unsafe_mean": float(np.mean(margin_np[unsafe])) if np.any(unsafe) else None,
    }


def main() -> None:
    args = parse_args()
    components = load_latent_safety_components(
        cache_path=args.cache_path,
        device_arg=args.device,
        oracle_kind="classifier",
        classifier_checkpoint=args.classifier_checkpoint,
        classifier_threshold=str(args.classifier_threshold),
        margin_transform=str(args.margin_transform),
    )
    metadata = components.cache["metadata"]
    state_dim = int(metadata["markov_state_dim"])
    action_dim = int(metadata["action_dim"])

    probe_env = OGBenchCubeLatentSafetyEnv(
        dynamics=components.dynamics,
        cache=components.cache,
        oracle=components.oracle,
        device=components.device,
        max_episode_steps=args.max_episode_steps,
        action_low=args.action_low,
        action_high=args.action_high,
        seed=args.seed,
    )
    policy = build_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=probe_env.action_space,
        device=components.device,
        args=args,
    )
    policy.load_state_dict(torch.load(args.policy_path.expanduser().resolve(), map_location=components.device))
    policy.eval()

    random_rollout = rollout_policy(
        components,
        None,
        episodes=args.episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    policy_rollout = rollout_policy(
        components,
        policy,
        episodes=args.episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    value_metrics = evaluate_values(
        components.cache,
        policy,
        device=components.device,
        sample_count=args.value_samples,
        batch_size=args.value_batch_size,
        seed=args.seed,
    )
    result = {
        "cache_path": str(args.cache_path.expanduser().resolve()),
        "policy_path": str(args.policy_path.expanduser().resolve()),
        "device": str(components.device),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "cache_unsafe_fraction": float(metadata["unsafe_fraction"]),
        "random_rollout": random_rollout,
        "policy_rollout": policy_rollout,
        "value_metrics": value_metrics,
        "interpretation": {
            "good_rollout_signal": "policy failure_rate lower than random failure_rate and policy mean_min_margin higher",
            "good_safe_start_signal": "policy safe_start_rollout_failure_rate lower than random safe_start_rollout_failure_rate",
            "good_value_signal": "critic_value_safe_auc above 0.5, positive value-margin correlation, safe mean value greater than unsafe mean value",
            "clipped_value": "min(critic_value, classifier_margin), matching the latent-safety plotting convention",
        },
    }
    print(json.dumps(jsonable(result), indent=2))
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
