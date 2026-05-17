#!/usr/bin/env python3
"""Train a history-conditioned Koopman model on Reacher DINO patch latents."""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from test.models_koopman import VisualHistoryKoopman


DEFAULT_DATASET_PATH = Path("data/expert_data/expert_data.pt")
DEFAULT_LATENT_METADATA_PATH = Path("data/expert_data/latents/dinov2_vits14_all_tokens/metadata.pt")
DEFAULT_MODEL_SAVE_PATH = Path("models/reacher_visual_koopman.pt")
DEFAULT_LOG_DIR = Path("runs/reacher_visual_koopman")

CHUNK_FILENAME_RE = re.compile(r"latents_(\d+)_(\d+)\.pt$")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if value in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--latent-metadata-path", type=Path, default=DEFAULT_LATENT_METADATA_PATH)
    parser.add_argument("--model-save-path", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--multi-step-horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--observable-dim", type=int, default=256)
    parser.add_argument("--adapter-hidden-channels", type=int, default=128)
    parser.add_argument("--observable-hidden-width", type=int, default=256)
    parser.add_argument("--observable-hidden-depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-latent", type=float, default=0.1)
    parser.add_argument(
        "--stop-grad-targets",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Detach the direct-encoding target branch before computing state and latent losses.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="TensorBoard logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save an intermediate checkpoint every N epochs.",
    )
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_chunk_range(chunk_path: str | Path) -> tuple[int, int]:
    match = CHUNK_FILENAME_RE.search(Path(chunk_path).name)
    if match is None:
        raise ValueError(f"Could not parse trajectory range from chunk path: {chunk_path}")
    start_idx, end_idx = match.groups()
    return int(start_idx), int(end_idx) + 1


class ChunkedDinoLatentStore:
    def __init__(self, dataset_path: Path, latent_metadata_path: Path) -> None:
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        metadata = torch.load(latent_metadata_path, map_location="cpu", weights_only=False)

        self.actions = dataset["actions"].to(torch.float32).contiguous()
        self.states = dataset["states"].to(torch.float32).contiguous()
        self.dataset_path = dataset_path
        self.latent_metadata_path = latent_metadata_path
        self.metadata = metadata

        self.num_trajectories = int(metadata["num_trajectories"])
        self.frames_per_trajectory = int(metadata["frames_per_trajectory"])
        self.action_steps = int(self.actions.shape[1])
        self.control_dim = int(self.actions.shape[-1])
        self.token_dim = int(self.metadata["latent_dim"])
        self.num_tokens = int(self.metadata["num_tokens"])
        self.has_cls_token = self.metadata["latent_type"] == "all_tokens"
        self.num_patch_tokens = self.num_tokens - 1 if self.has_cls_token else self.num_tokens

        if tuple(dataset["actions"].shape[:2]) != (self.num_trajectories, self.action_steps):
            raise ValueError("Action tensor shape does not match latent metadata trajectory count")
        if int(self.states.shape[1]) != self.frames_per_trajectory:
            raise ValueError("State tensor frame count does not match latent metadata")

        self.chunk_infos: list[dict[str, object]] = []
        for chunk_path_str in metadata["chunk_paths"]:
            chunk_path = Path(chunk_path_str)
            start_idx, end_idx = parse_chunk_range(chunk_path)
            self.chunk_infos.append(
                {
                    "path": chunk_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    def total_samples_per_epoch(self, history_len: int, horizon: int) -> int:
        valid_steps = self.valid_steps_per_trajectory(history_len, horizon)
        return self.num_trajectories * valid_steps

    def valid_steps_per_trajectory(self, history_len: int, horizon: int) -> int:
        valid_steps = self.frames_per_trajectory - history_len - horizon
        if valid_steps <= 0:
            raise ValueError(
                f"history_len={history_len} and horizon={horizon} leave no valid samples for "
                f"{self.frames_per_trajectory} frames"
            )
        return valid_steps

    def load_chunk(self, chunk_info: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_data = torch.load(chunk_info["path"], map_location="cpu", weights_only=False)
        latents = chunk_data["latents"].to(torch.float32)
        start_idx = int(chunk_info["start_idx"])
        end_idx = int(chunk_info["end_idx"])
        actions = self.actions[start_idx:end_idx]
        return latents, actions


def iter_chunk_batches(
    latents: torch.Tensor,
    actions: torch.Tensor,
    *,
    history_len: int,
    horizon: int,
    batch_size: int,
    generator: torch.Generator,
):
    n_traj = latents.shape[0]
    valid_steps = latents.shape[1] - history_len - horizon
    samples_per_chunk = n_traj * valid_steps
    frame_offsets = torch.arange(-history_len, horizon + 1, dtype=torch.long)
    control_offsets = torch.arange(horizon, dtype=torch.long)
    order = torch.randperm(samples_per_chunk, generator=generator)

    for batch_start in range(0, samples_per_chunk, batch_size):
        batch_ids = order[batch_start : batch_start + batch_size]
        traj_ids = batch_ids // valid_steps
        time_ids = history_len + (batch_ids % valid_steps)

        frame_ids = time_ids.unsqueeze(1) + frame_offsets.unsqueeze(0)
        control_ids = time_ids.unsqueeze(1) + control_offsets.unsqueeze(0)

        latent_batch = latents[traj_ids.unsqueeze(1), frame_ids]
        action_batch = actions[traj_ids.unsqueeze(1), control_ids]
        yield latent_batch, action_batch


def train(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path.expanduser().resolve()
    latent_metadata_path = args.latent_metadata_path.expanduser().resolve()
    model_save_path = args.model_save_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not latent_metadata_path.is_file():
        raise FileNotFoundError(f"Latent metadata file not found: {latent_metadata_path}")

    device = require_device(args.device)
    set_seed(args.seed)

    store = ChunkedDinoLatentStore(dataset_path, latent_metadata_path)
    total_samples = store.total_samples_per_epoch(args.history_len, args.multi_step_horizon)
    steps_per_epoch = math.ceil(total_samples / args.batch_size)

    model = VisualHistoryKoopman(
        control_dim=store.control_dim,
        token_dim=store.token_dim,
        num_patch_tokens=store.num_patch_tokens,
        state_dim=args.state_dim,
        observable_dim=args.observable_dim,
        history_len=args.history_len,
        adapter_hidden_channels=args.adapter_hidden_channels,
        observable_hidden_width=args.observable_hidden_width,
        observable_hidden_depth=args.observable_hidden_depth,
        has_cls_token=store.has_cls_token,
    ).to(device)

    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * steps_per_epoch,
        eta_min=args.min_lr,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    def build_save_data() -> dict[str, object]:
        return {
            "model_config": {
                "control_dim": store.control_dim,
                "token_dim": store.token_dim,
                "num_patch_tokens": store.num_patch_tokens,
                "state_dim": args.state_dim,
                "observable_dim": args.observable_dim,
                "history_len": args.history_len,
                "adapter_hidden_channels": args.adapter_hidden_channels,
                "observable_hidden_width": args.observable_hidden_width,
                "observable_hidden_depth": args.observable_hidden_depth,
                "has_cls_token": store.has_cls_token,
            },
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "grad_clip_norm": args.grad_clip_norm,
                "history_len": args.history_len,
                "multi_step_horizon": args.multi_step_horizon,
                "lambda_state": args.lambda_state,
                "lambda_latent": args.lambda_latent,
                "stop_grad_targets": args.stop_grad_targets,
                "checkpoint_every": args.checkpoint_every,
                "seed": args.seed,
            },
            "data_config": {
                "dataset_path": str(dataset_path),
                "latent_metadata_path": str(latent_metadata_path),
                "num_trajectories": store.num_trajectories,
                "frames_per_trajectory": store.frames_per_trajectory,
                "num_tokens": store.num_tokens,
                "num_patch_tokens": store.num_patch_tokens,
                "token_dim": store.token_dim,
                "control_dim": store.control_dim,
            },
            "state_dict": model.state_dict(),
        }

    print(f"Using device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Latent metadata: {latent_metadata_path}")
    print(f"Trajectories: {store.num_trajectories}")
    print(f"Frames per trajectory: {store.frames_per_trajectory}")
    print(f"Token layout: {store.num_patch_tokens} patch tokens x {store.token_dim} dims")
    print(f"History length: {args.history_len}")
    print(f"Rollout horizon: {args.multi_step_horizon}")
    print(f"State dim: {args.state_dim}")
    print(f"Observable dim: {args.observable_dim}")
    print(f"Stop-grad targets: {args.stop_grad_targets}")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
        chunk_order = torch.randperm(len(store.chunk_infos), generator=epoch_generator).tolist()
        epoch_loss = 0.0
        epoch_state_loss = 0.0
        epoch_latent_loss = 0.0
        epoch_batches = 0

        pbar = tqdm(chunk_order, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="chunk")
        for chunk_rank, chunk_idx in enumerate(pbar):
            chunk_info = store.chunk_infos[chunk_idx]
            latents_cpu, actions_cpu = store.load_chunk(chunk_info)
            batch_generator = torch.Generator(device="cpu").manual_seed(
                args.seed + epoch * 10_000 + chunk_rank
            )

            for latent_batch_cpu, action_batch_cpu in iter_chunk_batches(
                latents_cpu,
                actions_cpu,
                history_len=args.history_len,
                horizon=args.multi_step_horizon,
                batch_size=args.batch_size,
                generator=batch_generator,
            ):
                latent_batch = latent_batch_cpu.to(device, non_blocking=True)
                action_batch = action_batch_cpu.to(device, non_blocking=True)

                z_pred_seq, x_pred_seq, z_target_seq, x_target_seq = model(latent_batch, action_batch)
                if args.stop_grad_targets:
                    z_target_seq = z_target_seq.detach()
                    x_target_seq = x_target_seq.detach()

                loss_state = nn.functional.mse_loss(x_pred_seq, x_target_seq)
                loss_latent = nn.functional.mse_loss(z_pred_seq, z_target_seq)
                loss = args.lambda_state * loss_state + args.lambda_latent * loss_latent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                epoch_state_loss += float(loss_state.item())
                epoch_latent_loss += float(loss_latent.item())
                epoch_batches += 1

                if global_step % args.log_every == 0:
                    with torch.no_grad():
                        writer.add_scalar("loss/total", float(loss.item()), global_step)
                        writer.add_scalar("loss/state", float(loss_state.item()), global_step)
                        writer.add_scalar("loss/latent", float(loss_latent.item()), global_step)
                        writer.add_scalar("stats/x_target_std", float(x_target_seq.std().item()), global_step)
                        writer.add_scalar("stats/z_target_std", float(z_target_seq.std().item()), global_step)
                        writer.add_scalar("stats/a_weight_norm", float(model.A.weight.norm().item()), global_step)
                        writer.add_scalar("stats/b_weight_norm", float(model.B.weight.norm().item()), global_step)
                        writer.add_scalar("lr", float(scheduler.get_last_lr()[0]), global_step)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    state=f"{loss_state.item():.4f}",
                    latent=f"{loss_latent.item():.4f}",
                )
                global_step += 1

            del latents_cpu
            del actions_cpu

        mean_loss = epoch_loss / max(epoch_batches, 1)
        mean_state_loss = epoch_state_loss / max(epoch_batches, 1)
        mean_latent_loss = epoch_latent_loss / max(epoch_batches, 1)
        writer.add_scalar("epoch/loss_total", mean_loss, epoch)
        writer.add_scalar("epoch/loss_state", mean_state_loss, epoch)
        writer.add_scalar("epoch/loss_latent", mean_latent_loss, epoch)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={mean_loss:.6f} state={mean_state_loss:.6f} latent={mean_latent_loss:.6f}"
        )

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            checkpoint_path = model_save_path.with_name(
                f"{model_save_path.stem}_epoch_{epoch + 1}{model_save_path.suffix}"
            )
            torch.save(build_save_data(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()

    torch.save(build_save_data(), model_save_path)
    print(f"Saved model checkpoint to {model_save_path}")
    print(f"TensorBoard logdir: {log_dir}")


if __name__ == "__main__":
    train(parse_args())
