#!/usr/bin/env python3
"""Finetune a PushT LeWM checkpoint on local HDF5 data."""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import hdf5plugin  # noqa: F401
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from pusht.shared.models import SIGReg


DEFAULT_DATASET_PATHS = [
    Path("pusht/data/pusht_diffusion_train.h5"),
    Path("pusht/data/pusht_diffusion_edge.h5"),
    # Path("pusht/data/pusht_diffusion_random.h5"),
]
DEFAULT_INIT_CHECKPOINT = "lewm_home/checkpoints/pusht/lewm_object.ckpt"
DEFAULT_RUN_DIR = "pusht/models/lewm_ft_combined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-run-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume training from a Lightning checkpoint (for example, run_dir/last.ckpt) and restore optimizer/scheduler state.",
    )
    parser.add_argument("--dataset-path", type=Path, nargs="+", default=DEFAULT_DATASET_PATHS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-model-name", default="lewm")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use at most this many randomly selected training windows before train/val splitting.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Use at most this many episodes from each dataset before generating training windows.",
    )

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-scale", default="tiny")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1, help="Number of future frame steps to predict.")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=2)

    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--straighten", action="store_true", default=True, help="Apply temporal straightening to encoder latents.")
    parser.add_argument("--straighten-weight", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    parser.add_argument("--freeze-projector-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=110)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep training dataloader workers alive across epochs. Validation workers stay non-persistent to avoid doubled RAM usage.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=1)
    args = parser.parse_args()
    if args.freeze_encoder_epochs < 0:
        parser.error("--freeze-encoder-epochs must be non-negative.")
    if args.freeze_projector_epochs < 0:
        parser.error("--freeze-projector-epochs must be non-negative.")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive when provided.")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive when provided.")
    if args.history_size < 1:
        parser.error("--history-size must be positive.")
    if args.num_preds < 1:
        parser.error("--num-preds must be positive.")
    if args.frameskip < 1:
        parser.error("--frameskip must be positive.")
    args.effective_action_dim = args.frameskip * args.action_dim
    return args


class LeWMPushTDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        *,
        history_size: int,
        num_preds: int,
        frameskip: int,
        img_size: int,
        action_dim: int,
        max_episodes: int | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.history_size = int(history_size)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        self.num_steps = self.history_size + self.num_preds
        self.action_steps = self.history_size
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.max_episodes = max_episodes
        self.effective_action_dim = self.frameskip * self.action_dim
        self._h5: h5py.File | None = None

        if self.history_size < 1:
            raise ValueError("history_size must be positive.")
        if self.num_preds < 1:
            raise ValueError("num_preds must be positive.")
        if self.frameskip < 1:
            raise ValueError("frameskip must be positive.")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("max_episodes must be positive when provided.")

        with h5py.File(self.dataset_path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if self.max_episodes is not None:
                self.ep_len = self.ep_len[: self.max_episodes]
                self.ep_offset = self.ep_offset[: self.max_episodes]
            if int(h5["action"].shape[-1]) != self.action_dim:
                raise ValueError(f"Expected action_dim={self.action_dim}, got {h5['action'].shape[-1]}.")
            finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
            finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
            self.action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
            self.action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
            self.action_std = np.maximum(self.action_std, 1e-6)

        self.samples: list[tuple[int, int]] = []
        required_last_frame_offset = (self.num_steps - 1) * self.frameskip
        required_action_end_offset = self.action_steps * self.frameskip
        required_offset = max(required_last_frame_offset, required_action_end_offset)
        for ep_idx, ep_len in enumerate(self.ep_len.tolist()):
            max_start = ep_len - 1 - required_offset
            for start in range(max_start + 1):
                self.samples.append((ep_idx, start))
        if not self.samples:
            raise ValueError("No valid training windows found. Check history_size/frameskip/num_preds.")

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        ep_idx, start = self.samples[index]
        base = int(self.ep_offset[ep_idx]) + start

        frame_offsets = np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixel_rows = base + frame_offsets
        pixels_np = np.asarray(h5["pixels"][pixel_rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()

        action_blocks = []
        for step in range(self.action_steps):
            action_start = base + step * self.frameskip
            action_stop = action_start + self.frameskip
            block = np.asarray(h5["action"][action_start:action_stop], dtype=np.float32)
            block = (np.nan_to_num(block, nan=0.0) - self.action_mean) / self.action_std
            action_blocks.append(torch.from_numpy(block.reshape(-1)))

        return {
            "pixels": pixels,
            "action": torch.stack(action_blocks, dim=0).float(),
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


class FreezeModuleCallback(Callback):
    def __init__(self, module_name: str, freeze_epochs: int) -> None:
        super().__init__()
        self.module_name = module_name
        self.freeze_epochs = int(freeze_epochs)
        self._module_frozen = False

    def _set_module_requires_grad(self, pl_module: pl.LightningModule, enabled: bool) -> None:
        module = getattr(pl_module.model, self.module_name, None)
        if module is None:
            raise AttributeError(f"Expected model.{self.module_name} to exist for finetuning.")
        module.requires_grad_(enabled)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.freeze_epochs <= 0:
            return
        self._set_module_requires_grad(pl_module, enabled=False)
        self._module_frozen = True

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._module_frozen and trainer.current_epoch >= self.freeze_epochs:
            self._set_module_requires_grad(pl_module, enabled=True)
            self._module_frozen = False


def temporal_straightening_loss(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if emb.shape[1] < 3:
        return emb.new_zeros(())
    vel_prev = emb[:, 1:-1] - emb[:, :-2]
    vel_next = emb[:, 2:] - emb[:, 1:-1]
    cosine = F.cosine_similarity(vel_prev, vel_next, dim=-1, eps=eps)
    return (1.0 - cosine).mean()


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        batch_size, time_steps = pixels.shape[:2]
        pixels = F.interpolate(
            pixels.view(batch_size * time_steps, *pixels.shape[2:]),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, time_steps, *pixels.shape[2:3], img_size, img_size)
    pixel_mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    pixel_std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (pixels - pixel_mean) / pixel_std


def lewm_forward(self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace):
    lambd = args.sigreg_weight
    history_size = args.history_size

    pixels = preprocess_pixels(batch["pixels"], args.img_size)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode({"pixels": pixels, "action": batch["action"]})
    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, args.num_preds :]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_emb"] = pred_emb
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["straighten_loss"] = temporal_straightening_loss(emb) if args.straighten else emb.new_zeros(())
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + args.straighten_weight * output["straighten_loss"]
    )

    log_prefix = "train" if stage == "fit" else stage
    losses = {f"{log_prefix}/{key}": value.detach() for key, value in output.items() if "loss" in key}
    self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return output


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
        elif isinstance(value, list):
            hparams[key] = [str(item) if isinstance(item, Path) else item for item in value]
    return hparams


def resolve_dataset_paths(dataset_paths: list[Path | str]) -> list[Path]:
    resolved_paths = [Path(path).expanduser().resolve() for path in dataset_paths]
    missing_paths = [path for path in resolved_paths if not path.is_file()]
    if missing_paths:
        missing_str = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Dataset file(s) not found: {missing_str}")
    return resolved_paths


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_init_checkpoint(args: argparse.Namespace) -> Path:
    if args.init_checkpoint is not None:
        checkpoint_path = args.init_checkpoint.expanduser().resolve()
    else:
        checkpoint_path = latest_object_checkpoint(args.init_run_dir.expanduser().resolve()).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_resume_checkpoint(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.resume_checkpoint is not None:
        checkpoint_path = args.resume_checkpoint.expanduser().resolve()
    else:
        checkpoint_path = None
    if checkpoint_path is None:
        return None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_pretrained_model(checkpoint_path: Path) -> torch.nn.Module:
    source_model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = getattr(source_model, "encoder", None)
    predictor = getattr(source_model, "predictor", None)
    action_encoder = getattr(source_model, "action_encoder", None)
    if encoder is None or predictor is None or action_encoder is None:
        source_type = f"{type(source_model).__module__}.{type(source_model).__name__}"
        raise TypeError(
            "Expected a JEPA-compatible object checkpoint with "
            f"`encoder`, `predictor`, and `action_encoder`, got {source_type}."
        )
    source_model.train()
    source_model.requires_grad_(True)
    return source_model


def validate_lewm_checkpoint(model: torch.nn.Module, args: argparse.Namespace) -> None:
    predictor = getattr(model, "predictor", None)
    action_encoder = getattr(model, "action_encoder", None)
    expected_action_dim = args.effective_action_dim

    predictor_frames = getattr(predictor, "num_frames", None)
    if predictor_frames is not None and int(predictor_frames) != args.history_size:
        raise ValueError(
            f"Checkpoint predictor expects history_size={predictor_frames}, "
            f"but args.history_size={args.history_size}."
        )

    encoder_action_dim = getattr(action_encoder, "input_dim", None)
    if encoder_action_dim is not None and int(encoder_action_dim) != expected_action_dim:
        raise ValueError(
            f"Checkpoint action encoder expects action dim {encoder_action_dim}, "
            f"but frameskip * action_dim = {args.frameskip} * {args.action_dim} = {expected_action_dim}."
        )


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    dataset_paths = resolve_dataset_paths(args.dataset_path)
    run_dir = args.run_dir.expanduser().resolve()
    resume_checkpoint_path = resolve_resume_checkpoint(args, run_dir)
    init_checkpoint_path = None if resume_checkpoint_path is not None else resolve_init_checkpoint(args)

    if run_dir.exists() and resume_checkpoint_path is None:
        raise FileExistsError(f"Run dir already exists: {run_dir}")
    if not run_dir.exists() and resume_checkpoint_path is not None:
        raise FileNotFoundError(f"Run dir does not exist for resume: {run_dir}")

    spt.set(cache_dir=str(run_dir))

    datasets = [
        LeWMPushTDataset(
            dataset_path,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            img_size=args.img_size,
            action_dim=args.action_dim,
            max_episodes=args.max_episodes,
        )
        for dataset_path in dataset_paths
    ]
    dataset: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
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

    world_model = load_pretrained_model(init_checkpoint_path or resolve_init_checkpoint(args))
    validate_lewm_checkpoint(world_model, args)

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
        sigreg=SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj),
        forward=partial(lewm_forward, args=args),
        optim=optimizers,
        hparams=sanitize_hparams(args),
    )

    if resume_checkpoint_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = vars(args).copy()
        config["init_checkpoint"] = str(init_checkpoint_path)
        config["run_dir"] = str(run_dir)
        with (run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, default=str)

        init_report = {
            "init_checkpoint": str(init_checkpoint_path),
            "loaded_modules": ["encoder", "projector", "predictor", "action_encoder", "pred_proj"],
            "reinitialized_modules": [],
            "freeze_encoder_epochs": int(args.freeze_encoder_epochs),
            "freeze_projector_epochs": int(args.freeze_projector_epochs),
        }
        with (run_dir / "init_report.json").open("w") as f:
            json.dump(init_report, f, indent=2)
    else:
        resume_report = {
            "resume_checkpoint": str(resume_checkpoint_path),
            "run_dir": str(run_dir),
            "sigreg_weight": float(args.sigreg_weight),
            "freeze_encoder_epochs": int(args.freeze_encoder_epochs),
            "freeze_projector_epochs": int(args.freeze_projector_epochs),
        }
        with (run_dir / "resume_report.json").open("w") as f:
            json.dump(resume_report, f, indent=2)

    callbacks: list[Callback] = [
        FreezeModuleCallback("encoder", args.freeze_encoder_epochs),
        FreezeModuleCallback("projector", args.freeze_projector_epochs),
        ModelCheckpoint(
            dirpath=run_dir,
            filename=f"{args.output_model_name}" + "_{epoch:03d}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        ),
        ModelObjectCallback(run_dir, args.output_model_name, epoch_interval=args.save_object_every),
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
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])
    trainer.fit(module, datamodule=data_module, ckpt_path=str(resume_checkpoint_path) if resume_checkpoint_path else None)


if __name__ == "__main__":
    main()
