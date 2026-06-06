#!/usr/bin/env python3
"""Train latent-safety/PyHJ avoid-DDPG on the ogbench latent safety env."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
LATENT_SAFETY_DIR = ROOT_DIR / "third_party" / "latent-safety"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LATENT_SAFETY_DIR) not in sys.path:
    sys.path.insert(0, str(LATENT_SAFETY_DIR))


def _install_gym_fallback() -> None:
    try:
        import gym  # noqa: F401
    except ModuleNotFoundError:
        import gymnasium

        sys.modules["gym"] = gymnasium
    import gymnasium.envs.registration as gym_registration

    if not hasattr(gym_registration, "load_plugin_envs"):
        gym_registration.load_plugin_envs = lambda: None


def _install_numba_fallback() -> None:
    if os.environ.get("LATENT_SAFETY_DISABLE_NUMBA", "").lower() in {"1", "true", "yes"}:
        module = types.ModuleType("numba")

        def njit(*decorator_args: Any, **_decorator_kwargs: Any):
            if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
                return decorator_args[0]

            def _decorate(fn: Any) -> Any:
                return fn

            return _decorate

        module.njit = njit
        sys.modules["numba"] = module
        warnings.warn(
            "LATENT_SAFETY_DISABLE_NUMBA is set; PyHJ njit helpers will run as plain Python.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("numba")

        def njit(*decorator_args: Any, **_decorator_kwargs: Any):
            if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
                return decorator_args[0]

            def _decorate(fn: Any) -> Any:
                return fn

            return _decorate

        module.njit = njit
        sys.modules["numba"] = module
        warnings.warn(
            "numba is not installed; PyHJ njit helpers will run as plain Python. "
            "This is okay for --dry-run but install numba before real training.",
            RuntimeWarning,
            stacklevel=2,
        )


_install_gym_fallback()
_install_numba_fallback()

from PyHJ.data import Collector, VectorReplayBuffer  # noqa: E402
from PyHJ.env import DummyVectorEnv  # noqa: E402
from PyHJ.exploration import GaussianNoise  # noqa: E402
from PyHJ.policy import avoid_DDPGPolicy_annealing, avoid_DDPGPolicy_annealing_dinowm  # noqa: E402
from PyHJ.trainer import offpolicy_trainer  # noqa: E402
from PyHJ.utils.net.common import Net  # noqa: E402
from PyHJ.utils.net.continuous import Actor, Critic  # noqa: E402

from ogbench.safety.latent_env import OGBenchCubeLatentSafetyEnv, load_latent_safety_components  # noqa: E402

DEFAULT_CACHE_PATH = "ogbench/safety/cache/cube_latent_safety_classifier_train_tanh2.pt"
DEFAULT_RUN_ROOT = "ogbench/safety/runs"


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
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--oracle", choices=("knn", "classifier"), default="classifier")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--classifier-threshold", default="conformal")
    parser.add_argument("--margin-transform", choices=("auto", "identity", "tanh", "tanh2"), default="auto")
    parser.add_argument("--allow-classifier-latent-slice", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--optimistic-knn", action="store_true")

    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=25)
    parser.add_argument("--action-low", type=float, default=-2.0)
    parser.add_argument("--action-high", type=float, default=2.0)

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=40000)
    parser.add_argument("--step-per-collect", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=40000)
    parser.add_argument("--update-per-step", type=float, default=0.125)
    parser.add_argument("--warmup-epochs", type=int, default=1)

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
    parser.add_argument("--resume-policy", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "training_num",
        "test_num",
        "max_episode_steps",
        "epoch",
        "step_per_epoch",
        "step_per_collect",
        "batch_size",
        "buffer_size",
        "actor_gradient_steps",
        "n_step",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs cannot be negative.")
    if args.update_per_step < 0:
        raise ValueError("--update-per-step cannot be negative.")
    if not (0.0 <= args.tau <= 1.0):
        raise ValueError("--tau must be in [0, 1].")
    if not (0.0 <= args.gamma <= 1.0):
        raise ValueError("--gamma must be in [0, 1].")
    if args.action_low >= args.action_high:
        raise ValueError("--action-low must be smaller than --action-high.")
    if args.oracle == "classifier" and args.classifier_checkpoint is None:
        raise ValueError("--classifier-checkpoint is required for --oracle classifier.")


def default_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.expanduser().resolve()
    name = args.run_name or time.strftime("pyhj_cube_%Y%m%d_%H%M%S")
    return (ROOT_DIR / DEFAULT_RUN_ROOT / name).resolve()


def make_env_factory(components: Any, args: argparse.Namespace, seed_offset: int):
    def _factory() -> OGBenchCubeLatentSafetyEnv:
        return OGBenchCubeLatentSafetyEnv(
            dynamics=components.dynamics,
            cache=components.cache,
            oracle=components.oracle,
            device=components.device,
            max_episode_steps=args.max_episode_steps,
            action_low=args.action_low,
            action_high=args.action_high,
            seed=args.seed + seed_offset,
        )

    return _factory


def build_policy(
    *,
    state_dim: int,
    action_dim: int,
    action_space: Any,
    device: torch.device,
    args: argparse.Namespace,
):
    actor_net = Net(state_dim, hidden_sizes=args.actor_hidden, activation=torch.nn.ReLU, device=device)
    actor = Actor(actor_net, (action_dim,), max_action=1.0, device=device).to(device)
    optimizer_name = getattr(args, "optimizer", "adam")
    if optimizer_name == "adam":
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    elif optimizer_name == "adamw":
        actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}.")

    critic_net = Net(
        state_dim,
        (action_dim,),
        hidden_sizes=args.critic_hidden,
        activation=torch.nn.ReLU,
        concat=True,
        device=device,
    )
    critic = Critic(critic_net, device=device).to(device)
    if optimizer_name == "adam":
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    else:
        critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)

    policy_variant = getattr(args, "policy_variant", "dinowm")
    policy_cls = {
        "generic": avoid_DDPGPolicy_annealing,
        "dinowm": avoid_DDPGPolicy_annealing_dinowm,
    }[policy_variant]

    return policy_cls(
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=False,
        estimation_step=args.n_step,
        action_space=action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
        action_scaling=True,
        action_bound_method="clip",
    )


def save_policy(policy: Any, run_dir: Path, name: str = "policy_latest.pth") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    torch.save(policy.state_dict(), path)
    return path


def write_config(args: argparse.Namespace, run_dir: Path, metadata: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    serializable = vars(args).copy()
    for key, value in list(serializable.items()):
        if isinstance(value, Path):
            serializable[key] = str(value)
    payload = {"args": serializable, "cache_metadata": metadata, "latent_safety_dir": str(LATENT_SAFETY_DIR)}
    (run_dir / "train_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = default_run_dir(args)
    components = load_latent_safety_components(
        cache_path=args.cache_path,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device_arg=args.device,
        oracle_kind=args.oracle,
        classifier_checkpoint=args.classifier_checkpoint,
        classifier_threshold=str(args.classifier_threshold),
        margin_transform=str(args.margin_transform),
        allow_classifier_latent_slice=bool(args.allow_classifier_latent_slice),
        knn_k=args.knn_k,
        pessimistic=not args.optimistic_knn,
    )
    metadata = components.cache["metadata"]
    state_dim = int(metadata["markov_state_dim"])
    action_dim = int(metadata["action_dim"])

    train_envs = DummyVectorEnv([make_env_factory(components, args, i) for i in range(args.training_num)])
    test_envs = DummyVectorEnv([make_env_factory(components, args, 100_000 + i) for i in range(args.test_num)])
    train_envs.seed(args.seed)
    test_envs.seed(args.seed + 100_000)

    action_space = train_envs.action_space[0] if isinstance(train_envs.action_space, list) else train_envs.action_space
    policy = build_policy(state_dim=state_dim, action_dim=action_dim, action_space=action_space, device=components.device, args=args)
    if args.resume_policy is not None:
        policy.load_state_dict(torch.load(args.resume_policy.expanduser().resolve(), map_location=components.device))

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)), exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    write_config(args, run_dir, metadata)

    summary = {
        "run_dir": str(run_dir),
        "dry_run": bool(args.dry_run),
        "device": str(components.device),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "training_num": args.training_num,
        "test_num": args.test_num,
        "buffer_size": args.buffer_size,
        "oracle": args.oracle,
        "latent_safety_dir": str(LATENT_SAFETY_DIR),
        "policy_checkpoint": str(run_dir / "policy_latest.pth"),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    def stop_fn(_mean_rewards: float) -> bool:
        return False

    def save_best_fn(current_policy: Any) -> None:
        save_policy(current_policy, run_dir)

    def train_fn(epoch: int, _env_step: int) -> None:
        if epoch <= args.warmup_epochs:
            policy._gamma = 0.0
            policy.warmup = True
        else:
            policy._gamma = args.gamma
            policy.warmup = False

    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
    )
    latest = save_policy(policy, run_dir)
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": result, "saved_policy": str(latest)}, indent=2))


if __name__ == "__main__":
    main()
