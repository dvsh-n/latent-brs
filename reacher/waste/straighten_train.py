#!/usr/bin/env python3
"""Train a Temporal Straightening style world model on Reacher expert data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from shared.models import (
    LocalDinoV2Encoder,
    ProprioActionEncoder,
    StraighteningWorldModel,
    ViTPredictor,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "expert_data" / "prepocessed"
DEFAULT_SAVE_PATH = REPO_ROOT / "models" / "straighten_reacher.pt"
DEFAULT_LOG_DIR = REPO_ROOT / "runs" / "straighten_reacher"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-hist", type=int, default=3)
    parser.add_argument("--num-pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--stop-grad", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--straighten", default="False")
    parser.add_argument("--encoder-name", default="dinov2_vits14")
    parser.add_argument("--encoder-type", choices=("dino", "dino_channel", "dino_global"), default="dino_global")
    parser.add_argument("--projector-out-dim", type=int, default=128)
    parser.add_argument("--projector-hidden-dim", type=int, default=384)
    parser.add_argument("--projector-target-hw", type=int, default=1)
    parser.add_argument("--agg-type", choices=("flatten", "mean", "mlp"), default="mlp")
    parser.add_argument("--agg-out-dim", type=int, default=128)
    parser.add_argument("--agg-mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--action-emb-dim", type=int, default=10)
    parser.add_argument("--proprio-emb-dim", type=int, default=10)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=16)
    parser.add_argument("--predictor-mlp-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ProjectorSpec:
    name: str
    kwargs: dict


def build_projector_spec(args: argparse.Namespace) -> ProjectorSpec:
    if args.encoder_type == "dino":
        return ProjectorSpec(name="none", kwargs={})
    if args.encoder_type == "dino_channel":
        conv_layers = [
            {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "in_dim": 384,
                "out_dim": args.projector_hidden_dim,
            },
            {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "in_dim": args.projector_hidden_dim,
                "out_dim": args.projector_out_dim,
            },
        ]
        return ProjectorSpec(
            name="channel",
            kwargs={"norm_type": "layer", "conv_layers": conv_layers},
        )
    conv_layers = [
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "in_dim": args.projector_hidden_dim,
            "out_dim": args.projector_hidden_dim,
        },
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "in_dim": args.projector_hidden_dim,
            "out_dim": args.projector_hidden_dim,
        },
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "in_dim": args.projector_hidden_dim,
            "out_dim": args.projector_hidden_dim,
        },
    ]
    return ProjectorSpec(
        name="global",
        kwargs={
            "in_dim": 384,
            "out_dim": args.projector_out_dim,
            "hidden": args.projector_hidden_dim,
            "pool_hw": args.projector_target_hw,
            "conv_layers": conv_layers,
        },
    )


class ReacherStraighteningDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        num_hist: int,
        num_pred: int,
        frameskip: int = 1,
    ) -> None:
        self.data_dir = data_dir
        self.obses_dir = data_dir / "obses"
        self.states = torch.load(data_dir / "states.pth", map_location="cpu").float()
        self.actions = torch.load(data_dir / "actions_padded.pth", map_location="cpu").float()
        self.seq_lengths = torch.load(data_dir / "seq_lengths.pth", map_location="cpu").long()
        self.num_frames = num_hist + num_pred
        self.frameskip = frameskip
        self.state_dim = int(self.states.shape[-1])
        self.action_dim = int(self.actions.shape[-1])
        self.proprio_dim = self.state_dim
        self.slices: list[tuple[int, int, int]] = []
        for traj_idx in range(len(self.seq_lengths)):
            traj_len = int(self.seq_lengths[traj_idx])
            span = self.num_frames * self.frameskip
            if traj_len < span:
                continue
            for start in range(traj_len - span + 1):
                self.slices.append((traj_idx, start, start + span))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        traj_idx, start, end = self.slices[idx]
        frame_idx = list(range(start, end, self.frameskip))
        visual = torch.load(self.obses_dir / f"episode_{traj_idx:05d}.pth", map_location="cpu")[frame_idx]
        proprio = self.states[traj_idx, frame_idx]
        act = self.actions[traj_idx, frame_idx]
        return {"visual": visual, "proprio": proprio}, act


def make_encoder(args: argparse.Namespace) -> tuple[LocalDinoV2Encoder, ProjectorSpec]:
    projector = build_projector_spec(args)
    encoder = LocalDinoV2Encoder(
        name=args.encoder_name,
        feature_key="x_norm_patchtokens",
        projector=projector.name,
        projector_kwargs=projector.kwargs,
        agg_type=args.agg_type,
        agg_out_dim=args.agg_out_dim,
        agg_mlp_hidden_dim=args.agg_mlp_hidden_dim,
    )
    encoder.freeze_backbone()
    return encoder, projector


def split_dataset(dataset: Dataset, train_split: float, seed: int) -> tuple[Dataset, Dataset]:
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_model(args: argparse.Namespace) -> StraighteningWorldModel:
    encoder, _projector = make_encoder(args)
    action_encoder = ProprioActionEncoder(in_dim=2, emb_dim=args.action_emb_dim)
    proprio_encoder = ProprioActionEncoder(in_dim=6, emb_dim=args.proprio_emb_dim)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 196, 196)
        token_count = int(encoder(dummy).shape[1])
    predictor = ViTPredictor(
        num_patches=token_count,
        num_frames=args.num_hist,
        dim=encoder.emb_dim + args.action_emb_dim + args.proprio_emb_dim,
        depth=args.predictor_depth,
        heads=args.predictor_heads,
        mlp_dim=args.predictor_mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    )
    return StraighteningWorldModel(
        encoder=encoder,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        predictor=predictor,
        image_size=args.img_size,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        stop_grad=args.stop_grad,
        straighten=args.straighten,
    )


def train_epoch(
    model: StraighteningWorldModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
    log_every: int,
) -> tuple[dict[str, float], int]:
    model.train()
    running: dict[str, float] = {}
    count = 0

    for batch_idx, (obs, act) in enumerate(tqdm(loader, desc="Train", leave=False)):
        obs = {key: value.to(device, non_blocking=True) for key, value in obs.items()}
        act = act.to(device, non_blocking=True)

        outputs = model(obs, act)
        loss = outputs["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                running[key] = running.get(key, 0.0) + float(value.item())
        count += 1
        global_step += 1

        if global_step % log_every == 0:
            for key, total in running.items():
                writer.add_scalar(f"train/{key}", total / count, global_step)

    return {key: total / max(count, 1) for key, total in running.items()}, global_step


@torch.no_grad()
def eval_epoch(
    model: StraighteningWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running: dict[str, float] = {}
    count = 0
    for obs, act in tqdm(loader, desc="Valid", leave=False):
        obs = {key: value.to(device, non_blocking=True) for key, value in obs.items()}
        act = act.to(device, non_blocking=True)
        outputs = model(obs, act)
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                running[key] = running.get(key, 0.0) + float(value.item())
        count += 1
    return {key: total / max(count, 1) for key, total in running.items()}


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    save_path = args.save_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")

    dataset = ReacherStraighteningDataset(
        data_dir=data_dir,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
    )
    train_set, val_set = split_dataset(dataset, train_split=args.train_split, seed=args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.") and not name.startswith("encoder.base_model."):
            encoder_params.append(param)
        else:
            other_params.append(param)
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_val_loss = math.inf
    global_step = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            writer=writer,
            global_step=global_step,
            log_every=args.log_every,
        )
        val_metrics = eval_epoch(model=model, loader=val_loader, device=device)

        for key, value in train_metrics.items():
            writer.add_scalar(f"epoch_train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"epoch_valid/{key}", value, epoch)

        epoch_log = {"epoch": epoch, "train": train_metrics, "valid": val_metrics}
        history.append(epoch_log)
        print(json.dumps(epoch_log, indent=2))

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "valid_metrics": val_metrics,
            "history": history,
        }
        if epoch % args.save_every == 0:
            torch.save(ckpt, save_path.with_name(f"{save_path.stem}_epoch_{epoch}{save_path.suffix}"))
        val_loss = float(val_metrics.get("loss", float("inf")))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_path)

    writer.close()


if __name__ == "__main__":
    main()
