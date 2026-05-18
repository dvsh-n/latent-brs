#!/usr/bin/env python3
"""Train a goal-conditioned Mahalanobis distance metric based on the SAC Value Function."""

import argparse
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, Dataset, random_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=Path("reacher/data/expert_data_50hz/reacher_expert.h5"))
    parser.add_argument("--sac-model-path", type=Path, default=Path("reacher/models/reacher-dm-control-sac/final_model.zip"))
    parser.add_argument("--jepa-checkpoint", type=Path, default=Path("reacher/models/mlpdyn_ft_1/lewm_epoch_30_object.ckpt"))
    parser.add_argument("--run-dir", type=Path, default=Path("reacher/models/goal_conditioned_metric"))
    
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=18)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class ValueMetricDataset(Dataset):
    """Loads current frames, goal frames, and flat observations for SAC."""
    def __init__(self, dataset_path: Path, img_size: int):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self._h5 = None
        
        with h5py.File(self.dataset_path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            
        self.samples = []
        for ep_idx, ep_len in enumerate(self.ep_len):
            if ep_len < 2:
                continue
            goal_step = ep_len - 1
            for step in range(ep_len - 1):
                self.samples.append((ep_idx, step, goal_step))

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        h5 = self._file()
        ep_idx, step, goal_step = self.samples[idx]
        
        base_offset = int(self.ep_offset[ep_idx])
        curr_row = base_offset + step
        goal_row = base_offset + goal_step

        # Get flat observation for SAC Critic
        obs = torch.from_numpy(np.asarray(h5["observation"][curr_row], dtype=np.float32))

        # Get images
        curr_img = np.asarray(h5["pixels"][curr_row], dtype=np.uint8)
        goal_img = np.asarray(h5["pixels"][goal_row], dtype=np.uint8)
        
        curr_pixel = torch.from_numpy(curr_img).permute(2, 0, 1).float().div_(255.0)
        goal_pixel = torch.from_numpy(goal_img).permute(2, 0, 1).float().div_(255.0)
        
        if curr_pixel.shape[-2:] != (self.img_size, self.img_size):
            curr_pixel = F.interpolate(curr_pixel.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear")[0]
            goal_pixel = F.interpolate(goal_pixel.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear")[0]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        curr_pixel = (curr_pixel - mean) / std
        goal_pixel = (goal_pixel - mean) / std

        return {"pixels": curr_pixel, "goal_pixels": goal_pixel, "observation": obs}


class GoalConditionedValueMetric(nn.Module):
    """Neural Network that outputs a goal-conditioned PSD metric matrix M(z, z_goal)."""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.tril_size = (embed_dim * (embed_dim + 1)) // 2
        
        # Input dimension is 2 * embed_dim to accept both state and goal
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.tril_size)
        )
        
        tril_indices = torch.tril_indices(row=embed_dim, col=embed_dim, offset=0)
        self.register_buffer("row_indices", tril_indices[0])
        self.register_buffer("col_indices", tril_indices[1])

    def get_M_matrix(self, z: torch.Tensor, z_goal: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Concatenate current state and goal state
        net_input = torch.cat([z, z_goal], dim=-1)
        tril_elements = self.net(net_input)
        
        L = torch.zeros(batch_size, self.embed_dim, self.embed_dim, device=z.device)
        L[:, self.row_indices, self.col_indices] = tril_elements
        
        # Softplus ensures strict positivity on the diagonal (Valid Cholesky factor)
        diag_indices = torch.arange(self.embed_dim, device=z.device)
        L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices]) + eps
        
        # M(z, z_goal) = L * L^T
        M = torch.bmm(L, L.transpose(1, 2))
        return M

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        M = self.get_M_matrix(z, z_goal)
        delta = (z - z_goal).unsqueeze(2)           
        delta_T = (z - z_goal).unsqueeze(1)         
        dist = torch.bmm(torch.bmm(delta_T, M), delta) 
        return dist.squeeze(-1).squeeze(-1) 


class GoalConditionedMetricTrainer(L.LightningModule):
    def __init__(self, jepa_checkpoint: Path, sac_model_path: Path, embed_dim: int, hidden_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 1. Load and Freeze JEPA
        source_model = torch.load(jepa_checkpoint, map_location="cpu", weights_only=False)
        self.encoder = source_model.encoder
        self.projector = source_model.projector
        self.encoder.eval()
        self.projector.eval()
        for param in self.encoder.parameters(): param.requires_grad = False
        for param in self.projector.parameters(): param.requires_grad = False

        # 2. Load and Freeze SAC
        sac_model = SAC.load(str(sac_model_path), device="cpu")
        self.sac_actor = sac_model.policy.actor
        self.sac_critic = sac_model.policy.critic
        self.sac_actor.eval()
        self.sac_critic.eval()
        for param in self.sac_actor.parameters(): param.requires_grad = False
        for param in self.sac_critic.parameters(): param.requires_grad = False

        # 3. The Learnable Goal-Conditioned Metric Network
        self.metric_net = GoalConditionedValueMetric(embed_dim, hidden_dim)

    @torch.no_grad()
    def get_z(self, pixels: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixels, interpolate_pos_encoding=True)
        return self.projector(out.last_hidden_state[:, 0])

    def training_step(self, batch, batch_idx):
        pixels = batch["pixels"]
        goal_pixels = batch["goal_pixels"]
        obs = batch["observation"]

        with torch.no_grad():
            actions = self.sac_actor(obs, deterministic=True)
            q1, q2 = self.sac_critic(obs, actions)
            cost_target = -torch.min(q1, q2).squeeze()
            
            z = self.get_z(pixels)
            z_goal = self.get_z(goal_pixels)

        predicted_dist = self.metric_net(z, z_goal)

        # Huber loss handles potential large Q-value outliers
        loss = F.smooth_l1_loss(predicted_dist, cost_target)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pixels = batch["pixels"]
            goal_pixels = batch["goal_pixels"]
            obs = batch["observation"]
            
            actions = self.sac_actor(obs, deterministic=True)
            q1, q2 = self.sac_critic(obs, actions)
            cost_target = -torch.min(q1, q2).squeeze()
            
            z = self.get_z(pixels)
            z_goal = self.get_z(goal_pixels)
            
            predicted_dist = self.metric_net(z, z_goal)
            val_loss = F.mse_loss(predicted_dist, cost_target)
            
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.metric_net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    
    args.run_dir.mkdir(parents=True, exist_ok=True)

    dataset = ValueMetricDataset(args.dataset_path, args.img_size)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = GoalConditionedMetricTrainer(
        jepa_checkpoint=args.jepa_checkpoint,
        sac_model_path=args.sac_model_path,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-gc-metric-{epoch:02d}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.run_dir,
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the standalone metric network for easy loading in the planner
    best_model = GoalConditionedMetricTrainer.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.metric_net, args.run_dir / "gc_metric_net.pt")
    print(f"\nTraining Complete. Goal-Conditioned Metric Network saved to {args.run_dir / 'gc_metric_net.pt'}")


if __name__ == "__main__":
    main()