#!/usr/bin/env python
"""Train a LeRobot Diffusion Policy expert on PushT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from pusht.shared.utils import ensure_lerobot_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default="lerobot/pusht")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=Path("pusht/runs/pusht_diffusion_expert"))
    parser.add_argument("--model-dir", type=Path, default=Path("pusht/models/pusht_diffusion_expert"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --run-dir.",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--log-every",
        "--log-freq",
        dest="log_every",
        type=int,
        default=100,
        help="Print loss every N steps. 0 disables.",
    )
    parser.add_argument(
        "--save-every",
        "--save-freq",
        dest="save_every",
        type=int,
        default=10000,
        help="Save step checkpoints every N steps. 0 disables.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--no-episode-aware-sampler", action="store_true")
    args = parser.parse_args()
    if args.output_dir is not None:
        args.run_dir = args.output_dir
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_policy_config(dataset_metadata: LeRobotDatasetMetadata) -> DiffusionConfig:
    from lerobot.configs import FeatureType
    from lerobot.policies.diffusion import DiffusionConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    return DiffusionConfig(input_features=input_features, output_features=output_features)


def build_delta_timestamps(cfg: DiffusionConfig, dataset_metadata: LeRobotDatasetMetadata) -> dict[str, list[float]]:
    return {
        key: [idx / dataset_metadata.fps for idx in cfg.observation_delta_indices]
        for key in cfg.input_features
        if key.startswith("observation.")
    } | {
        key: [idx / dataset_metadata.fps for idx in cfg.action_delta_indices]
        for key in cfg.output_features
    }


def save_bundle(
    save_dir: Path,
    policy: DiffusionPolicy,
    preprocessor,
    postprocessor,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


def save_run_config(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with open(run_dir / "train_args.json", "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = parse_args()

    ensure_lerobot_available()
    from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.diffusion import DiffusionPolicy
    from lerobot.utils.random_utils import set_seed

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args.run_dir, args)

    print(f"Using device: {device}")
    print(f"Saving run outputs to: {args.run_dir}")
    print(f"Saving final model to: {args.model_dir}")
    print(f"Loading dataset metadata: {args.dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    cfg = build_policy_config(dataset_metadata)
    cfg.device = str(device)

    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    delta_timestamps = build_delta_timestamps(cfg, dataset_metadata)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )

    sampler = None
    shuffle = True
    if not args.no_episode_aware_sampler:
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=args.weight_decay,
    )

    step = 0
    progress = tqdm(total=args.steps, desc="Training", unit="step")
    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)

            if args.log_every > 0 and step % args.log_every == 0:
                loss_value = loss.item()
                progress.set_postfix(loss=f"{loss_value:.4f}")
                progress.write(f"step={step} loss={loss_value:.6f}")

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_dir = args.run_dir / "checkpoints" / f"step_{step:08d}"
                save_bundle(checkpoint_dir, policy, preprocessor, postprocessor)
                progress.write(f"Saved checkpoint to {checkpoint_dir}")

            if step >= args.steps:
                break

    progress.close()
    save_bundle(args.model_dir, policy, preprocessor, postprocessor)
    print(f"Saved final policy bundle to {args.model_dir}")


if __name__ == "__main__":
    main()
