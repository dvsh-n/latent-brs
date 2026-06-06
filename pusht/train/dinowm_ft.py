#!/usr/bin/env python3
"""Finetune a PushT DINO-WM/PreJEPA checkpoint on local HDF5 data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "third_party" / "stable-worldmodel",
    REPO_ROOT / "third_party" / "stable-pretraining",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import h5py
import hdf5plugin  # noqa: F401
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, Subset, random_split


DEFAULT_DATASET_PATH = Path("pusht/data/train_data/pusht_diffusion_train_combined.h5")
DEFAULT_INIT_CHECKPOINT = Path(
    "third_party/stable-worldmodel/experiments/pusht_dinowm/checkpoints/models/dinowm/dinowm_object.ckpt"
)
DEFAULT_RUN_DIR = Path("pusht/models/dinowm_ft_combined_15000ep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume training from a Lightning checkpoint and restore optimizer/scheduler state.",
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-model-name", default="dinowm")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument("--max-episodes", type=int, default=15000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use at most this many randomly selected training windows after episode filtering.",
    )

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--proprio-dim", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the DINO image backbone. The current PreJEPA encoder path detaches image embeddings, so this should stay true unless that model code is changed.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep training dataloader workers alive across epochs. Validation workers stay non-persistent to avoid doubled RAM usage.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=1)
    args = parser.parse_args()

    if not 0 < args.train_split <= 1:
        parser.error("--train-split must be in (0, 1].")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive when provided.")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive when provided.")
    if args.history_size < 1:
        parser.error("--history-size must be positive.")
    if args.num_preds < 1:
        parser.error("--num-preds must be positive.")
    if args.frameskip < 1:
        parser.error("--frameskip must be positive.")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be positive.")
    if args.action_dim < 1 or args.proprio_dim < 1:
        parser.error("--action-dim and --proprio-dim must be positive.")

    args.num_steps = args.history_size + args.num_preds
    args.effective_action_dim = args.frameskip * args.action_dim
    return args


class DINOWMPushTDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        *,
        num_steps: int,
        frameskip: int,
        sample_stride: int,
        img_size: int,
        action_dim: int,
        proprio_dim: int,
        max_episodes: int | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.num_steps = int(num_steps)
        self.frameskip = int(frameskip)
        self.sample_stride = int(sample_stride)
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.max_episodes = max_episodes
        self.span = self.num_steps * self.frameskip
        self._h5: h5py.File | None = None

        with h5py.File(self.dataset_path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if self.max_episodes is not None:
                self.ep_len = self.ep_len[: self.max_episodes]
                self.ep_offset = self.ep_offset[: self.max_episodes]
            if int(h5["action"].shape[-1]) != self.action_dim:
                raise ValueError(f"Expected action_dim={self.action_dim}, got {h5['action'].shape[-1]}.")
            if int(h5["proprio"].shape[-1]) != self.proprio_dim:
                raise ValueError(f"Expected proprio_dim={self.proprio_dim}, got {h5['proprio'].shape[-1]}.")

            self.action_mean, self.action_std = self._column_stats(h5["action"])
            self.proprio_mean, self.proprio_std = self._column_stats(h5["proprio"])

        self.samples: list[tuple[int, int]] = []
        for ep_idx, ep_len in enumerate(self.ep_len.tolist()):
            if ep_len < self.span:
                continue
            for start in range(0, ep_len - self.span + 1, self.sample_stride):
                self.samples.append((ep_idx, start))
        if not self.samples:
            raise ValueError("No valid training windows found. Check max_episodes/frameskip/history_size/num_preds.")

    @staticmethod
    def _column_stats(dataset: h5py.Dataset) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(dataset[:], dtype=np.float32)
        values = values[~np.isnan(values).any(axis=1)]
        mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(values.std(axis=0, keepdims=True).astype(np.float32), 1e-6)
        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        ep_idx, start = self.samples[index]
        base = int(self.ep_offset[ep_idx]) + start

        frame_rows = base + np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixels_np = np.asarray(h5["pixels"][frame_rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        pixels = preprocess_pixels(pixels, self.img_size)

        proprio_np = np.asarray(h5["proprio"][frame_rows], dtype=np.float32)
        proprio_np = (np.nan_to_num(proprio_np, nan=0.0) - self.proprio_mean) / self.proprio_std

        action_np = np.asarray(h5["action"][base : base + self.span], dtype=np.float32)
        action_np = (np.nan_to_num(action_np, nan=0.0) - self.action_mean) / self.action_std
        action = action_np.reshape(self.num_steps, self.frameskip * self.action_dim)

        return {
            "pixels": pixels,
            "proprio": torch.from_numpy(proprio_np).float(),
            "action": torch.from_numpy(action).float(),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
            self._h5 = None


class ModelObjectCallback(Callback):
    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt")
        torch.save(pl_module.model, self.dirpath / f"{self.filename}_object.ckpt")


class StateDictCallback(Callback):
    def __init__(self, dirpath: Path, epoch_interval: int = 1) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model.state_dict(), self.dirpath / f"weights_epoch_{epoch}.pt")
        torch.save(pl_module.model.state_dict(), self.dirpath / "weights.pt")


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        pixels = F.interpolate(
            pixels,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    pixel_mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    pixel_std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (pixels - pixel_mean) / pixel_std


def strip_action_dims(tensor: torch.Tensor, action_range: tuple[int, int]) -> torch.Tensor:
    lo, hi = action_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def dinowm_forward(self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace):
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)

    batch = self.model.encode(batch, target="embed", is_video=False)
    embedding = batch["embed"][:, : args.history_size]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch["embed"][:, args.num_preds :].detach()

    pixels_dim = batch["pixels_embed"].size(-1)
    batch["pixels_loss"] = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim])

    start = pixels_dim
    action_range = (0, 0)
    for key in self.model.extra_encoders:
        dim = batch[f"{key}_embed"].size(-1)
        lo, hi = start, start + dim
        if key == "action":
            action_range = (lo, hi)
        else:
            batch[f"{key}_loss"] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    batch["actionless_pred_embed"] = strip_action_dims(pred_embedding, action_range)
    batch["actionless_target_embed"] = strip_action_dims(target_embedding, action_range)
    batch["loss"] = F.mse_loss(
        batch["actionless_pred_embed"],
        batch["actionless_target_embed"].detach(),
    )

    if batch["loss"].isnan():
        raise ValueError("NaN loss encountered.")

    log_prefix = "train" if stage == "fit" else stage
    losses = {f"{log_prefix}/{key}": value.detach() for key, value in batch.items() if key.endswith("_loss") or key == "loss"}
    self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return batch


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator == "gpu",
        "persistent_workers": args.num_workers > 0 and persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def sanitize_hparams(args: argparse.Namespace) -> dict[str, object]:
    hparams = vars(args).copy()
    for key, value in hparams.items():
        if isinstance(value, Path):
            hparams[key] = str(value)
    return hparams


def resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_checkpoint is None:
        return None
    return resolve_file(args.resume_checkpoint, "Resume checkpoint")


def load_pretrained_model(checkpoint_path: Path) -> torch.nn.Module:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ("backbone", "predictor", "extra_encoders", "history_size", "num_pred")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        source_type = f"{type(model).__module__}.{type(model).__name__}"
        raise TypeError(f"Expected a PreJEPA-compatible object checkpoint, got {source_type}; missing {missing}.")
    model.train()
    model.requires_grad_(True)
    return model


def validate_model(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if int(model.history_size) != args.history_size:
        raise ValueError(f"Checkpoint history_size={model.history_size}, but --history-size={args.history_size}.")
    if int(model.num_pred) != args.num_preds:
        raise ValueError(f"Checkpoint num_pred={model.num_pred}, but --num-preds={args.num_preds}.")

    expected_extra_dims = {
        "proprio": args.proprio_dim,
        "action": args.effective_action_dim,
    }
    for key, expected_dim in expected_extra_dims.items():
        if key not in model.extra_encoders:
            raise ValueError(f"Checkpoint is missing extra encoder '{key}'.")
        in_chans = getattr(model.extra_encoders[key], "in_chans", None)
        if in_chans is not None and int(in_chans) != int(expected_dim):
            raise ValueError(
                f"Checkpoint extra encoder '{key}' expects input dim {in_chans}, but this script will provide {expected_dim}."
            )


def freeze_backbone(model: torch.nn.Module) -> None:
    if hasattr(model, "backbone"):
        model.backbone.requires_grad_(False)
        model.backbone.eval()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dataset_path = resolve_file(args.dataset_path, "Dataset")
    run_dir = args.run_dir.expanduser().resolve()
    resume_checkpoint_path = resolve_resume_checkpoint(args)
    init_checkpoint_path = None if resume_checkpoint_path is not None else resolve_file(args.init_checkpoint, "Init checkpoint")

    if run_dir.exists() and resume_checkpoint_path is None:
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    if not run_dir.exists() and resume_checkpoint_path is not None:
        raise FileNotFoundError(f"Run dir does not exist for resume: {run_dir}")

    spt.set(cache_dir=str(run_dir))

    dataset: Dataset = DINOWMPushTDataset(
        dataset_path,
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        sample_stride=args.sample_stride,
        img_size=args.img_size,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        max_episodes=args.max_episodes,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need at least 2 valid training windows for train/val splitting, got {len(dataset)}.")

    generator = torch.Generator().manual_seed(args.seed)
    if args.max_samples is not None and args.max_samples < len(dataset):
        indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples].tolist()
        dataset = Subset(dataset, indices)

    train_len = int(len(dataset) * args.train_split)
    val_len = len(dataset) - train_len
    if train_len < 1:
        train_len = 1
        val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = make_loader(
        train_set,
        args,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.persistent_workers,
    )
    val_loader = (
        make_loader(
            val_set,
            args,
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
        )
        if val_len
        else None
    )

    world_model = load_pretrained_model(init_checkpoint_path or args.init_checkpoint.expanduser().resolve())
    validate_model(world_model, args)
    if args.freeze_backbone:
        freeze_backbone(world_model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    module = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, args=args),
        optim=optimizers,
        hparams=sanitize_hparams(args),
    )

    if resume_checkpoint_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = vars(args).copy()
        config["init_checkpoint"] = str(init_checkpoint_path)
        config["dataset_path"] = str(dataset_path)
        config["run_dir"] = str(run_dir)
        config["num_windows"] = len(dataset)
        config["train_windows"] = train_len
        config["val_windows"] = val_len
        with (run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, default=str)

        init_report = {
            "init_checkpoint": str(init_checkpoint_path),
            "loaded_modules": ["backbone", "predictor", "extra_encoders", "decoder"],
            "reinitialized_modules": [],
            "freeze_backbone": bool(args.freeze_backbone),
            "note": "PreJEPA._encode_image detaches pixel embeddings, so the image backbone is effectively frozen in this model implementation.",
        }
        with (run_dir / "init_report.json").open("w") as f:
            json.dump(init_report, f, indent=2)
    else:
        resume_report = {
            "resume_checkpoint": str(resume_checkpoint_path),
            "run_dir": str(run_dir),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "freeze_backbone": bool(args.freeze_backbone),
        }
        with (run_dir / "resume_report.json").open("w") as f:
            json.dump(resume_report, f, indent=2)

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=run_dir,
            filename=f"{args.output_model_name}" + "_{epoch:03d}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        ),
        ModelObjectCallback(run_dir, args.output_model_name, epoch_interval=args.save_object_every),
        StateDictCallback(run_dir, epoch_interval=args.save_object_every),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        default_root_dir=run_dir,
        logger=False,
        num_sanity_val_steps=1 if val_loader is not None else 0,
        enable_checkpointing=True,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_checkpoint_path) if resume_checkpoint_path else None)


if __name__ == "__main__":
    main()
