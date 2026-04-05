"""
DAgger training for linear-decoder Deep Koopman quadrotor with GPU batch ADMM planning.
"""
import json
import os
import shutil
import sys
from pathlib import Path


def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT_DIR))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import DeepKoopmanLinDec
from train.koop_train_lindec import QuadrotorDataset
from utils import get_activation_fn, get_unique_path, resolve_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_DAGGER_DIR = ROOT_DIR / "train" / "dagger_lindec"
DAGGER_DATA_PATH = "train/dagger_lindec/data/dagger_data.pt"
DAGGER_CKPT_DIR = "train/dagger_lindec/ckpts"
DAGGER_LOG_DIR = "train/dagger_lindec/runs"
DAGGER_EVAL_DIR = "train/dagger_lindec/eval"
DAGGER_EVAL_LOG_DIR = "train/dagger_lindec/eval_runs"
FINAL_MODEL_PATH = "models/koop_quad3d_lindec_dagger.pt"

TRAIN_DATA_PATHS = ["data/random_data.pt"]
INIT_NUM_TRAJ = 15_000

HORIZON = 200
DT = 0.025
Q_TERMINAL = 100.0
R_EFFORT = 0.25
RHO = 10.0
ADMM_STEPS = 100
ENCODE_BATCH = 1024
PLAN_BATCH = 1024

MASS = 1.0
GRAV = -9.81
U1_MIN, U1_MAX = 0.0, -2.0 * GRAV * MASS
U_TORQUE_MIN, U_TORQUE_MAX = -0.1, 0.1

MIN_DIST = 2.5
LOW_BOUNDS = np.array([0.1, 0.1, 0.1], dtype=np.float32)
HIGH_BOUNDS = np.array([4.9, 4.9, 4.9], dtype=np.float32)
ENFORCE_POS_BOUNDS = True

STATE_DIM = 12
CONTROL_DIM = 4
LATENT_DIM = 24
HIDDEN_WIDTH = 64
HIDDEN_DEPTH = 3
ACTIVATION = "gelu"

LR = 5e-4
MIN_LR = 5e-8
WEIGHT_DECAY = 1e-6
EPOCHS_PER_ITER = 25
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 75
NUM_WORKERS = 8
ENABLE_NORMALIZATION = False
LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.1

NUM_ITERATIONS = 10
NEW_TRAJ_PER_ITER = 5_000
FINAL_CKPT_ONLY = True

EVAL_EPISODES = 25
EVAL_PLOT_EPISODES = 5
EVAL_SEED = 123
EVAL_GOAL_TOL = 0.2
EVAL_SAVE_PLOTS = True


def quadrotor_dynamics_np(x, u, dt=DT):
    psi, theta, phi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

    cos_theta = np.cos(theta)
    if np.abs(cos_theta) < 1e-6:
        cos_theta = 1e-6 * np.sign(cos_theta) if cos_theta != 0 else 1e-6

    xdot = np.array(
        [
            x_dot,
            y_dot,
            z_dot,
            q * np.sin(phi) / cos_theta + r * np.cos(phi) / cos_theta,
            q * np.cos(phi) - r * np.sin(phi),
            p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta),
            u1 / MASS * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)),
            u1 / MASS * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)),
            GRAV + u1 / MASS * (np.cos(phi) * np.cos(theta)),
            ((0.1 - 0.3) / 0.5) * q * r + u2 / 0.5,
            ((0.3 - 0.5) / 0.1) * p * r + u3 / 0.1,
            ((0.5 - 0.1) / 0.3) * p * q + u4 / 0.3,
        ],
        dtype=np.float32,
    )
    return x + dt * xdot


def simulate_true_dynamics(x0, u_seq):
    traj = np.zeros((len(u_seq) + 1, 12), dtype=np.float32)
    traj[0] = x0
    for k in range(len(u_seq)):
        traj[k + 1] = quadrotor_dynamics_np(traj[k], u_seq[k])
    return traj


class Normalizer:
    def __init__(self, stats):
        self.min = stats["min"].cpu().numpy()
        self.max = stats.get("max")
        self.range = stats.get("range")
        if self.range is None:
            self.range = (self.max - self.min).cpu().numpy()
        else:
            self.range = self.range.cpu().numpy()
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def denormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min


def preprocess_state(x, normalizer, enable_normalization):
    return normalizer.normalize(x) if enable_normalization else x


def postprocess_state(x, normalizer, enable_normalization):
    return normalizer.denormalize(x) if enable_normalization else x


class GPUADMMBatchTrajOptSolver:
    """
    Open-loop latent trajectory optimization with ADMM.
    Cost terms: terminal error + control effort only.
    """

    def __init__(
        self,
        A_np,
        B_np,
        horizon,
        u_min,
        u_max,
        q_terminal,
        r_effort,
        rho,
        n_admm_steps,
        device,
    ):
        self.device = str(device)
        self.H = int(horizon)
        self.rho = float(rho)
        self.steps = int(n_admm_steps)
        self.q_terminal = float(q_terminal)
        self.r_effort = float(r_effort)

        self.A = torch.as_tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.as_tensor(B_np, dtype=torch.float32, device=device)
        self.nx = int(self.A.shape[0])
        self.nu = int(self.B.shape[1])

        self.u_hover = torch.tensor([-GRAV * MASS, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        self.u_min = torch.as_tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.as_tensor(u_max, dtype=torch.float32, device=device)
        self.u_min_tiled = self.u_min.repeat(self.H).view(1, -1)
        self.u_max_tiled = self.u_max.repeat(self.H).view(1, -1)

        self.nz_total = (self.H + 1) * self.nx
        self.nu_total = self.H * self.nu
        self.n_vars = self.nu_total + self.nz_total
        self.n_eq = (self.H + 1) * self.nx
        self.u_slice = slice(0, self.nu_total)
        self.xN_idx = self.H * self.nu + self.H * self.nx

        self.P = self._build_hessian()
        self.A_eq = self._build_dynamics_matrix()
        self.KKT_inv = self._build_kkt_inverse()

    def _build_hessian(self):
        H, nu, nx = self.H, self.nu, self.nx
        r_eff = torch.eye(nu, device=self.A.device) * max(self.r_effort, 1e-6)
        p_u = torch.zeros((H * nu, H * nu), dtype=torch.float32, device=self.A.device)
        for k in range(H):
            sl = slice(k * nu, (k + 1) * nu)
            p_u[sl, sl] = r_eff + self.rho * torch.eye(nu, device=self.A.device)

        p_x = torch.eye((H + 1) * nx, dtype=torch.float32, device=self.A.device) * 1e-4
        p_x[-nx:, -nx:] += torch.eye(nx, dtype=torch.float32, device=self.A.device) * self.q_terminal

        top = torch.cat([p_u, torch.zeros((H * nu, (H + 1) * nx), device=self.A.device)], dim=1)
        bot = torch.cat([torch.zeros(((H + 1) * nx, H * nu), device=self.A.device), p_x], dim=1)
        return torch.cat([top, bot], dim=0)

    def _build_dynamics_matrix(self):
        H, nx, nu = self.H, self.nx, self.nu
        a_eq = torch.zeros((self.n_eq, self.n_vars), dtype=torch.float32, device=self.A.device)

        x_offset = H * nu
        a_eq[0:nx, x_offset: x_offset + nx] = torch.eye(nx, device=self.A.device)

        for k in range(H):
            row = (k + 1) * nx
            col_u = k * nu
            col_x = x_offset + k * nx
            col_x_next = x_offset + (k + 1) * nx

            a_eq[row: row + nx, col_u: col_u + nu] = -self.B
            a_eq[row: row + nx, col_x: col_x + nx] = -self.A
            a_eq[row: row + nx, col_x_next: col_x_next + nx] = torch.eye(nx, device=self.A.device)

        return a_eq

    def _build_kkt_inverse(self):
        top = torch.cat([self.P, self.A_eq.T], dim=1)
        bot = torch.cat([self.A_eq, torch.zeros((self.n_eq, self.n_eq), device=self.A.device)], dim=1)
        kkt = torch.cat([top, bot], dim=0)
        return torch.linalg.inv(kkt)

    def solve_batch(self, z0_batch_np, zg_batch_np):
        z0_batch = torch.as_tensor(z0_batch_np, dtype=torch.float32, device=self.A.device)
        zg_batch = torch.as_tensor(zg_batch_np, dtype=torch.float32, device=self.A.device)
        bs = z0_batch.shape[0]

        sol = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.A.device)
        z_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.A.device)
        y_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.A.device)

        q = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.A.device)
        lin_u_cost = -(self.r_effort * self.u_hover)
        for k in range(self.H):
            q[:, k * self.nu: (k + 1) * self.nu] = lin_u_cost
        q[:, self.xN_idx: self.xN_idx + self.nx] = -self.q_terminal * zg_batch

        b_eq = torch.zeros((bs, self.n_eq), dtype=torch.float32, device=self.A.device)
        b_eq[:, : self.nx] = z0_batch

        for _ in range(self.steps):
            current_q = q.clone()
            current_q[:, self.u_slice] -= self.rho * (z_u - y_u)
            rhs = torch.cat([-current_q, b_eq], dim=1)
            sol_aug = rhs @ self.KKT_inv.T
            sol = sol_aug[:, : self.n_vars]
            u_sol = sol[:, self.u_slice]
            z_u = torch.maximum(torch.minimum(u_sol + y_u, self.u_max_tiled), self.u_min_tiled)
            y_u = y_u + (u_sol - z_u)

        u_flat = sol[:, : self.nu_total]
        z_flat = sol[:, self.nu_total:]
        u_traj = u_flat.view(bs, self.H, self.nu).detach().cpu().numpy()
        z_traj = z_flat.view(bs, self.H + 1, self.nx).detach().cpu().numpy()
        return z_traj, u_traj


def sample_start_goal(rng):
    start = np.zeros(12, dtype=np.float32)
    goal = np.zeros(12, dtype=np.float32)
    start_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
    while True:
        goal_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
        if np.linalg.norm(goal_pos - start_pos) > MIN_DIST:
            break
    start[:3] = start_pos
    goal[:3] = goal_pos
    return start, goal


def _truncate_oob_rollout(x_traj, u_traj):
    in_bounds = np.all((x_traj[:, :3] >= LOW_BOUNDS) & (x_traj[:, :3] <= HIGH_BOUNDS), axis=1)
    if np.all(in_bounds):
        return x_traj, u_traj

    first_bad = int(np.argmax(~in_bounds))
    if first_bad <= 1:
        return None, None

    x_trunc = x_traj[:first_bad]
    u_trunc = u_traj[: first_bad - 1]
    if len(u_trunc) < MULTI_STEP_HORIZON:
        return None, None
    return x_trunc, u_trunc


def _all_in_bounds(x_traj):
    in_bounds = np.all((x_traj[:, :3] >= LOW_BOUNDS) & (x_traj[:, :3] <= HIGH_BOUNDS), axis=1)
    return bool(np.all(in_bounds))


def rollout_model(model, normalizer, x0, u_seq, enable_normalization):
    device = next(model.parameters()).device
    with torch.no_grad():
        x0_proc = preprocess_state(x0, normalizer, enable_normalization)
        x0_t = torch.tensor(x0_proc, dtype=torch.float32, device=device).unsqueeze(0)
        z_k = model.encoder(x0_t)

        x_hat_proc = [model.C(z_k).squeeze(0)]
        for u_k in u_seq:
            u_t = torch.tensor(u_k, dtype=torch.float32, device=device).unsqueeze(0)
            z_k = model.A(z_k) + model.B(u_t)
            x_hat_proc.append(model.C(z_k).squeeze(0))

        x_hat_proc = torch.stack(x_hat_proc, dim=0).cpu().numpy()
        x_hat = postprocess_state(x_hat_proc, normalizer, enable_normalization)
    return x_hat


def plot_xyz_rollout(x_true, x_pred, start_state, goal_state, save_path):
    t = np.arange(min(len(x_true), len(x_pred)))
    labels = ["X", "Y", "Z"]

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, x_true[: len(t), i], label="true", color="blue", linewidth=2)
        ax.plot(t, x_pred[: len(t), i], label="model", color="red", linestyle="--", linewidth=2)
        ax.axhline(goal_state[i], color="green", linestyle=":", linewidth=1, label="goal" if i == 0 else None)
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True, linestyle=":")

    axs[-1].set_xlabel("Step")
    fig.suptitle(f"XYZ Rollout (start={start_state[:3]}, goal={goal_state[:3]})")
    handles, leg_labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _append_metrics_csv(csv_path, metrics):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "iteration,success_rate,final_pos_err_mean,final_pos_err_median,"
        "pos_rmse_mean,state_rmse_mean,plan_fail_rate,oob_rate,valid_rollouts\n"
    )
    line = (
        f"{metrics['iteration']},{metrics['success_rate']:.6f},"
        f"{metrics['final_pos_err_mean']:.6f},{metrics['final_pos_err_median']:.6f},"
        f"{metrics['pos_rmse_mean']:.6f},{metrics['state_rmse_mean']:.6f},"
        f"{metrics['plan_fail_rate']:.6f},{metrics['oob_rate']:.6f},"
        f"{metrics['valid_rollouts']}\n"
    )
    if not csv_path.exists():
        csv_path.write_text(header, encoding="utf-8")
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line)


def evaluate_iteration(model, normalizer, iteration, enable_normalization, writer=None):
    eval_rng = np.random.default_rng(EVAL_SEED)
    eval_dir = Path(resolve_path(DAGGER_EVAL_DIR)) / f"iter_{iteration}"
    plot_dir = eval_dir / "plots"
    if EVAL_SAVE_PLOTS:
        plot_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    A_np = model.A.weight.detach().cpu().numpy()
    B_np = model.B.weight.detach().cpu().numpy()
    solver = GPUADMMBatchTrajOptSolver(
        A_np=A_np,
        B_np=B_np,
        horizon=HORIZON,
        u_min=np.array([U1_MIN, U_TORQUE_MIN, U_TORQUE_MIN, U_TORQUE_MIN], dtype=np.float32),
        u_max=np.array([U1_MAX, U_TORQUE_MAX, U_TORQUE_MAX, U_TORQUE_MAX], dtype=np.float32),
        q_terminal=Q_TERMINAL,
        r_effort=R_EFFORT,
        rho=RHO,
        n_admm_steps=ADMM_STEPS,
        device=device,
    )

    starts, goals = [], []
    for _ in range(EVAL_EPISODES):
        s, g = sample_start_goal(eval_rng)
        starts.append(s)
        goals.append(g)
    starts = np.stack(starts)
    goals = np.stack(goals)

    z0_parts, zg_parts = [], []
    with torch.no_grad():
        x0_proc = preprocess_state(starts, normalizer, enable_normalization)
        xg_proc = preprocess_state(goals, normalizer, enable_normalization)
        for i in range(0, EVAL_EPISODES, ENCODE_BATCH):
            j = min(i + ENCODE_BATCH, EVAL_EPISODES)
            x0_t = torch.tensor(x0_proc[i:j], dtype=torch.float32, device=device)
            xg_t = torch.tensor(xg_proc[i:j], dtype=torch.float32, device=device)
            z0_parts.append(model.encoder(x0_t).cpu().numpy())
            zg_parts.append(model.encoder(xg_t).cpu().numpy())
    z0 = np.concatenate(z0_parts, axis=0)
    zg = np.concatenate(zg_parts, axis=0)

    _, u_all = solver.solve_batch(z0, zg)

    pos_rmses, state_rmses, final_errs = [], [], []
    success, oob_fail = 0, 0
    for ep in range(EVAL_EPISODES):
        u_traj = u_all[ep]
        x_true = simulate_true_dynamics(starts[ep], u_traj)
        x_pred = rollout_model(model, normalizer, starts[ep], u_traj, enable_normalization)
        in_bounds = _all_in_bounds(x_true) if ENFORCE_POS_BOUNDS else True
        if not in_bounds:
            oob_fail += 1

        final_err = float(np.linalg.norm(x_true[-1, :3] - goals[ep, :3]))
        final_errs.append(final_err)
        if in_bounds and final_err <= EVAL_GOAL_TOL:
            success += 1

        pos_diff = x_true[:, :3] - x_pred[:, :3]
        state_diff = x_true - x_pred
        pos_mse = float(np.mean(pos_diff ** 2))
        state_mse = float(np.mean(state_diff ** 2))
        pos_rmses.append(float(np.sqrt(pos_mse)))
        state_rmses.append(float(np.sqrt(state_mse)))

        if EVAL_SAVE_PLOTS and ep < EVAL_PLOT_EPISODES:
            save_path = plot_dir / f"episode_{ep:03d}.png"
            plot_xyz_rollout(x_true, x_pred, starts[ep], goals[ep], str(save_path))

    total = EVAL_EPISODES
    metrics = {
        "iteration": iteration,
        "eval_episodes": EVAL_EPISODES,
        "goal_tolerance": EVAL_GOAL_TOL,
        "success_rate": success / total,
        "final_pos_err_mean": float(np.mean(final_errs)) if final_errs else float("nan"),
        "final_pos_err_median": float(np.median(final_errs)) if final_errs else float("nan"),
        "pos_rmse_mean": float(np.mean(pos_rmses)) if pos_rmses else float("nan"),
        "state_rmse_mean": float(np.mean(state_rmses)) if state_rmses else float("nan"),
        "plan_fail_rate": 0.0,
        "oob_rate": oob_fail / total,
        "valid_rollouts": len(pos_rmses),
    }

    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _append_metrics_csv(resolve_path(f"{DAGGER_EVAL_DIR}/metrics.csv"), metrics)

    if writer is not None:
        writer.add_scalar("eval/success_rate", metrics["success_rate"], iteration)
        writer.add_scalar("eval/final_pos_err_mean", metrics["final_pos_err_mean"], iteration)
        writer.add_scalar("eval/final_pos_err_median", metrics["final_pos_err_median"], iteration)
        writer.add_scalar("eval/pos_rmse_mean", metrics["pos_rmse_mean"], iteration)
        writer.add_scalar("eval/state_rmse_mean", metrics["state_rmse_mean"], iteration)
        writer.add_scalar("eval/plan_fail_rate", metrics["plan_fail_rate"], iteration)
        writer.add_scalar("eval/oob_rate", metrics["oob_rate"], iteration)
        writer.add_scalar("eval/valid_rollouts", metrics["valid_rollouts"], iteration)
        writer.flush()
    return metrics


def _find_source_data():
    for p in TRAIN_DATA_PATHS:
        resolved = resolve_path(p)
        if Path(resolved).exists():
            return resolved
    raise FileNotFoundError(
        f"Source data not found. Tried: {TRAIN_DATA_PATHS}. Generate with data generation script first."
    )


def init_dagger_data():
    data_path = resolve_path(DAGGER_DATA_PATH)
    train_path = _find_source_data()
    os.makedirs(Path(data_path).parent, exist_ok=True)

    demos = torch.load(train_path, weights_only=False)
    n_total = len(demos)
    n_use = min(INIT_NUM_TRAJ, n_total)
    selected = demos[:n_use]
    torch.save(selected, data_path)
    print(f"Initialized dagger data: {n_use} trajectories from {n_total} in {Path(train_path).name}")
    print(f"Saved to {data_path}")
    return data_path


def train_iteration(data_path, ckpt_path, log_dir, iteration, load_from_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Path(ckpt_path).parent, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = QuadrotorDataset(data_path, MULTI_STEP_HORIZON, enable_normalization=ENABLE_NORMALIZATION)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    activation_fn = get_activation_fn(ACTIVATION)
    model_config = {
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "depth": HIDDEN_DEPTH,
        "activation_fn": activation_fn,
    }
    model = DeepKoopmanLinDec(**model_config).to(device)

    if load_from_ckpt and Path(load_from_ckpt).exists():
        ckpt = torch.load(load_from_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded previous checkpoint from {load_from_ckpt}")
    else:
        nn.init.eye_(model.A.weight)
        nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PER_ITER * len(dataloader), eta_min=MIN_LR
    )
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(EPOCHS_PER_ITER), desc=f"Train iter {iteration}"):
        for x_k, u_seq, x_kp1_seq in dataloader:
            x_k = x_k.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            x_kp1_seq = x_kp1_seq.to(device, non_blocking=True)

            z_pred_seq, x_pred_seq, z_target_seq = model(x_k, u_seq, x_kp1_seq)
            loss_state = nn.functional.mse_loss(x_pred_seq, x_kp1_seq)
            loss_latent = nn.functional.mse_loss(z_pred_seq, z_target_seq)
            loss = LAMBDA_STATE * loss_state + LAMBDA_LATENT * loss_latent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        writer.add_scalar("train/loss_total", float(loss.item()), epoch)
        writer.add_scalar("train/loss_state", float(loss_state.item()), epoch)
        writer.add_scalar("train/loss_latent", float(loss_latent.item()), epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        writer.flush()

    writer.close()

    model_config_save = {
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "depth": HIDDEN_DEPTH,
        "activation_fn": ACTIVATION,
    }
    training_config = {
        "lr": LR,
        "min_lr": MIN_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs_per_iter": EPOCHS_PER_ITER,
        "batch_size": BATCH_SIZE,
        "multi_step_horizon": MULTI_STEP_HORIZON,
        "num_workers": NUM_WORKERS,
        "enable_normalization": ENABLE_NORMALIZATION,
        "lambda_state": LAMBDA_STATE,
        "lambda_latent": LAMBDA_LATENT,
        "dagger_num_iterations": NUM_ITERATIONS,
        "dagger_new_traj_per_iter": NEW_TRAJ_PER_ITER,
        "dagger_final_ckpt_only": FINAL_CKPT_ONLY,
        "planner_horizon": HORIZON,
        "planner_dt": DT,
        "planner_q_terminal": Q_TERMINAL,
        "planner_r_effort": R_EFFORT,
        "planner_rho": RHO,
        "planner_admm_steps": ADMM_STEPS,
        "planner_encode_batch": ENCODE_BATCH,
        "planner_plan_batch": PLAN_BATCH,
        "enforce_pos_bounds": ENFORCE_POS_BOUNDS,
    }
    save_data = {
        "model_config": model_config_save,
        "training_config": training_config,
        "iteration": int(iteration),
        "state_dict": model.state_dict(),
        "normalization_stats": {"min": dataset.min, "max": dataset.max, "range": dataset.range},
    }
    torch.save(save_data, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def collect_and_aggregate(model, normalizer, rng, data_path, n_traj, enable_normalization):
    device = next(model.parameters()).device
    A_np = model.A.weight.detach().cpu().numpy()
    B_np = model.B.weight.detach().cpu().numpy()
    solver = GPUADMMBatchTrajOptSolver(
        A_np=A_np,
        B_np=B_np,
        horizon=HORIZON,
        u_min=np.array([U1_MIN, U_TORQUE_MIN, U_TORQUE_MIN, U_TORQUE_MIN], dtype=np.float32),
        u_max=np.array([U1_MAX, U_TORQUE_MAX, U_TORQUE_MAX, U_TORQUE_MAX], dtype=np.float32),
        q_terminal=Q_TERMINAL,
        r_effort=R_EFFORT,
        rho=RHO,
        n_admm_steps=ADMM_STEPS,
        device=device,
    )

    starts, goals = [], []
    for _ in range(n_traj):
        s, g = sample_start_goal(rng)
        starts.append(s)
        goals.append(g)
    starts = np.stack(starts)
    goals = np.stack(goals)

    z0_parts, zg_parts = [], []
    with torch.no_grad():
        x0_proc = preprocess_state(starts, normalizer, enable_normalization)
        xg_proc = preprocess_state(goals, normalizer, enable_normalization)
        for i in range(0, n_traj, ENCODE_BATCH):
            j = min(i + ENCODE_BATCH, n_traj)
            x0_t = torch.tensor(x0_proc[i:j], dtype=torch.float32, device=device)
            xg_t = torch.tensor(xg_proc[i:j], dtype=torch.float32, device=device)
            z0_parts.append(model.encoder(x0_t).cpu().numpy())
            zg_parts.append(model.encoder(xg_t).cpu().numpy())
    z0_all = np.concatenate(z0_parts, axis=0)
    zg_all = np.concatenate(zg_parts, axis=0)

    new_demos = []
    for i in tqdm(range(0, n_traj, PLAN_BATCH), desc="Collecting trajectories (ADMM batch)"):
        j = min(i + PLAN_BATCH, n_traj)
        _, u_batch = solver.solve_batch(z0_all[i:j], zg_all[i:j])
        for b in range(j - i):
            idx = i + b
            u_traj = u_batch[b]
            x_true = simulate_true_dynamics(starts[idx], u_traj)
            if ENFORCE_POS_BOUNDS:
                x_true, u_traj = _truncate_oob_rollout(x_true, u_traj)
                if x_true is None:
                    continue
            new_demos.append(
                {
                    "start": starts[idx],
                    "goal": goals[idx],
                    "states": x_true.T.astype(np.float32),
                    "controls": u_traj.T.astype(np.float32),
                }
            )

    if not new_demos:
        print("Warning: no new demos collected")
        return

    current = torch.load(data_path, weights_only=False)
    current.extend(new_demos)
    torch.save(current, data_path)
    print(f"Aggregated {len(new_demos)} trajectories. Total: {len(current)}")


def export_final_model(src_ckpt_path):
    src = Path(src_ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"Cannot export final model; checkpoint missing: {src}")
    dst = Path(get_unique_path(resolve_path(FINAL_MODEL_PATH)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(src, map_location="cpu", weights_only=False)
    torch.save(payload, dst)
    print(f"Saved final model to {dst}")


def cleanup_dagger_workspace():
    if TRAIN_DAGGER_DIR.exists():
        shutil.rmtree(TRAIN_DAGGER_DIR)
        print(f"Deleted DAgger workspace: {TRAIN_DAGGER_DIR}")


def main():
    data_path = resolve_path(DAGGER_DATA_PATH)
    Path(resolve_path(DAGGER_CKPT_DIR)).mkdir(parents=True, exist_ok=True)
    Path(resolve_path(DAGGER_LOG_DIR)).mkdir(parents=True, exist_ok=True)
    Path(resolve_path(DAGGER_EVAL_DIR)).mkdir(parents=True, exist_ok=True)
    Path(resolve_path(DAGGER_EVAL_LOG_DIR)).mkdir(parents=True, exist_ok=True)

    init_dagger_data()

    rng = np.random.default_rng()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_writer = SummaryWriter(resolve_path(DAGGER_EVAL_LOG_DIR))

    for it in range(NUM_ITERATIONS):
        print(f"\n{'=' * 60}")
        print(f"DAgger Iteration {it + 1}/{NUM_ITERATIONS}")
        print(f"{'=' * 60}")

        if FINAL_CKPT_ONLY:
            ckpt_path = resolve_path(f"{DAGGER_CKPT_DIR}/dagger_model.pt")
            log_dir = resolve_path(f"{DAGGER_LOG_DIR}/iter_{it}")
            prev_ckpt = ckpt_path if it > 0 and Path(ckpt_path).exists() else None
        else:
            ckpt_path = resolve_path(f"{DAGGER_CKPT_DIR}/iter_{it}.pt")
            log_dir = resolve_path(f"{DAGGER_LOG_DIR}/iter_{it}")
            prev_ckpt = resolve_path(f"{DAGGER_CKPT_DIR}/iter_{it - 1}.pt") if it > 0 else None

        train_iteration(data_path, ckpt_path, log_dir, it, load_from_ckpt=prev_ckpt)

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = checkpoint["model_config"]
        if isinstance(cfg.get("activation_fn"), str):
            cfg["activation_fn"] = get_activation_fn(cfg["activation_fn"])
        model = DeepKoopmanLinDec(
            **{
                k: v
                for k, v in cfg.items()
                if k in ["state_dim", "control_dim", "latent_dim", "hidden_width", "depth", "activation_fn"]
            }
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        training_config = checkpoint.get("training_config", {})
        enable_normalization = training_config.get("enable_normalization", ENABLE_NORMALIZATION)
        normalizer = Normalizer(checkpoint["normalization_stats"])
        evaluate_iteration(model, normalizer, it, enable_normalization, eval_writer)
        collect_and_aggregate(model, normalizer, rng, data_path, NEW_TRAJ_PER_ITER, enable_normalization)

    eval_writer.close()

    final_ckpt = resolve_path(f"{DAGGER_CKPT_DIR}/dagger_model.pt") if FINAL_CKPT_ONLY else resolve_path(
        f"{DAGGER_CKPT_DIR}/iter_{NUM_ITERATIONS - 1}.pt"
    )
    export_final_model(final_ckpt)
    if FINAL_CKPT_ONLY:
        cleanup_dagger_workspace()

    if FINAL_CKPT_ONLY:
        print("\nDAgger complete. Intermediate DAgger workspace was deleted (FINAL_CKPT_ONLY=True).")
    else:
        print(f"\nDAgger complete. Data: {resolve_path(DAGGER_DATA_PATH)}")
        print(f"Checkpoints: {resolve_path(DAGGER_CKPT_DIR)}")
    print(f"Final model: {resolve_path(FINAL_MODEL_PATH)}")


if __name__ == "__main__":
    main()
