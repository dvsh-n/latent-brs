import sys
from pathlib import Path

# --- Root Path Setup ---
def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path

ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Import models/utils exactly as before
from models import DeepKoopman
from utils import resolve_path, get_activation_fn

# ====== CONFIGURATION ======
KOOPMAN_PATH = "models/koop_quad3d_dagger_1.pt"
DEVICE = "cuda"

HORIZON = 50
DT = 0.025
MAX_MPC_STEPS = 500
GOAL_TOL = 0.1  # meters
ENABLE_ANIM = True
ENABLE_VIDEO_GEN = True
VIDEO_SAVE_PATH = "plan/videos/plan_admm_mpc.gif"
ENABLE_TIMING_BREAKDOWN = True

# Cost weights
Q = 100.0
R = 0.25

# Solver config
RHO = 10.0
MAX_ADMM_STEPS = 5
ENABLE_WARM_START = True          
ENABLE_EARLY_STOP = False          
ADMM_PRIMAL_TOL = 1e-4
ADMM_DUAL_TOL = 1e-2

# Physics and control bounds
MASS = 1.0
GRAV = -9.81
U1_MIN, U1_MAX = 0.0, -2.0 * GRAV * MASS
U_TORQUE_MIN, U_TORQUE_MAX = -0.1, 0.1

# Start/goal setup
MIN_DIST = 2.5
LOW_BOUNDS = np.array([0.1, 0.1, 0.1], dtype=np.float32)
HIGH_BOUNDS = np.array([4.9, 4.9, 4.9], dtype=np.float32)


# ====== HELPERS ======
def maybe_cuda_synchronize(device):
    if isinstance(device, str):
        is_cuda = device.startswith("cuda")
    else:
        is_cuda = getattr(device, "type", None) == "cuda"

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


class Normalizer:
    def __init__(self, stats):
        self.min = stats["min"].detach().cpu().numpy()
        self.range = stats["range"].detach().cpu().numpy()
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        return 2 * (x - self.min) / self.range - 1

    def denormalize(self, x_norm):
        return (x_norm + 1) / 2 * self.range + self.min

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
    # start[:3] = [3.9499183, 1.9324102, 2.9330184]
    # goal[:3] = [0.5086885, 3.824969, 0.8882106]
    return start, goal

def quadrotor_dynamics_np(x, u, dt=DT):
    psi, theta, phi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

    cos_theta = np.cos(theta)
    if np.abs(cos_theta) < 1e-6:
        cos_theta = 1e-6 * np.sign(cos_theta) if cos_theta != 0 else 1e-6

    xdot = np.array([
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
    ], dtype=np.float32)

    return x + dt * xdot

def rollout_true_dynamics_np(x0, u_seq, dt=DT):
    """Roll out true dynamics for a control sequence, including initial state."""
    traj = [x0.copy()]
    x = x0.copy()
    for u in u_seq:
        x = quadrotor_dynamics_np(x, u, dt=dt)
        traj.append(x.copy())
    return np.asarray(traj, dtype=np.float32)

def print_planning_error_metrics(step_traj_mse, step_pos_traj_mse, k_state_mse_curve, k_pos_mse_curve):
    """Print summary metrics for model-vs-true rollout consistency under planned controls."""
    if len(step_traj_mse) == 0:
        print("No planning-error metrics available (no MPC steps executed).")
        return

    step_traj_mse = np.asarray(step_traj_mse, dtype=np.float64)
    step_pos_traj_mse = np.asarray(step_pos_traj_mse, dtype=np.float64)
    k_state_mse_curve = np.asarray(k_state_mse_curve, dtype=np.float64)
    k_pos_mse_curve = np.asarray(k_pos_mse_curve, dtype=np.float64)

    print("\n" + "=" * 62)
    print("Planning Model-Error Metrics (Decoded Plan vs True Rollout)")
    print("=" * 62)
    print("Per-MPC-step trajectory MSE (full horizon under planned controls):")
    print(f"  state_12d_mse: mean={step_traj_mse.mean():.8f}, std={step_traj_mse.std():.8f}, min={step_traj_mse.min():.8f}, max={step_traj_mse.max():.8f}")
    print(f"  pos_xyz_mse:   mean={step_pos_traj_mse.mean():.8f}, std={step_pos_traj_mse.std():.8f}, min={step_pos_traj_mse.min():.8f}, max={step_pos_traj_mse.max():.8f}")

    print("\nK-step MSE curves (averaged over all MPC replans):")
    print(f"  k=1   : state_12d_mse={k_state_mse_curve[0]:.8f}, pos_xyz_mse={k_pos_mse_curve[0]:.8f}")
    print(f"  k={len(k_state_mse_curve):<3d}: state_12d_mse={k_state_mse_curve[-1]:.8f}, pos_xyz_mse={k_pos_mse_curve[-1]:.8f}")
    print("=" * 62)


def print_timing_breakdown_metrics(timing_stats, n_steps):
    if n_steps == 0:
        print("No timing metrics available (no MPC steps executed).")
        return

    print("\n" + "=" * 62)
    print("Per-MPC-Step Timing Breakdown")
    print("=" * 62)
    for key, value in timing_stats.items():
        print(f"  {key:<22}: {(value / n_steps) * 1000.0:.4f} ms")
    print("=" * 62)

# ====== NEW SOLVER CLASS (Paper-Based ADMM) ======
class GPUADMM_MPCSolver:
    def __init__(self, A_np, B_np, Q_val, R_val, 
                 u_min, u_max,
                 horizon, rho=1.0, n_admm_steps=15, device=DEVICE):
        """
        Replaces LatentMPCSolver. 
        Implements GPU-accelerated ADMM for LTI systems (Eqs. 10-12, 33-44).
        Pre-computes KKT inverse for O(1) solving online.
        """
        self.H = horizon
        self.nx = A_np.shape[0]
        self.nu = B_np.shape[1]
        self.rho = rho
        self.steps = n_admm_steps
        self.device = device
        
        # Physics constants for cost
        self.u_hover = torch.tensor([-GRAV * MASS, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        # 1. Transfer Dynamics & Weights to GPU
        self.A = torch.tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)
        
        # Cost Matrices
        # Using R_SMOOTH as main R penalty to approximate smoothness in standard LQR form
        self.Q = torch.eye(self.nx, device=device) * Q_val
        self.R = torch.eye(self.nu, device=device) * (R_val if R_val > 0 else 1.0) 
        
        # Bounds
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=device)

        # [cite_start]2. Build KKT Matrix (Offline Pre-computation) [cite: 288]
        self.nz_total = (self.H + 1) * self.nx
        self.nu_total = self.H * self.nu
        self.n_vars = self.nz_total + self.nu_total
        self.n_eq = (self.H + 1) * self.nx
        self.u_slice = slice(0, self.nu_total)
        self.xN_idx = self.H * self.nu + self.H * self.nx
        self.u_min_tiled = self.u_min.repeat(self.H)
        self.u_max_tiled = self.u_max.repeat(self.H)

        # Warm-start buffers
        self._warm_sol = torch.zeros(self.n_vars, device=self.device)
        self._warm_Z_u = torch.zeros(self.nu_total, device=self.device)
        self._warm_Y_u = torch.zeros(self.nu_total, device=self.device)

        # Reusable work buffers to avoid per-solve allocations.
        self._sol = torch.zeros(self.n_vars, device=self.device)
        self._Z_u = torch.zeros(self.nu_total, device=self.device)
        self._Y_u = torch.zeros(self.nu_total, device=self.device)
        self._q_base = torch.zeros(self.n_vars, device=self.device)
        self._q_nominal = torch.empty(self.n_vars, device=self.device)
        self._current_q = torch.empty(self.n_vars, device=self.device)
        self._rhs = torch.empty(self.n_vars + self.n_eq, device=self.device)
        self._z_prev = torch.empty(self.nu_total, device=self.device)
        self._b_eq = torch.zeros(self.n_eq, device=self.device)

        # Static linear term on controls.
        lin_u_cost = -(self.R @ self.u_hover)
        for k in range(self.H):
            self._q_base[k * self.nu:(k + 1) * self.nu] = lin_u_cost

        self._build_kkt_inverse()

    def _build_kkt_inverse(self):
        """Constructs and inverts the KKT matrix offline."""
        nx, nu, H = self.nx, self.nu, self.H
        
        # --- Construct P (Hessian) ---
        # Block diagonal P: [u0, ..., uH-1, x0, ..., xH]
        # P_u = R + rho * I (Augmented Lagrangian term)
        P_blocks = []
        R_prox = self.R + self.rho * torch.eye(nu, device=self.device)
        
        for _ in range(H):
            P_blocks.append(R_prox)
            
        # P_x (Small regularization for stability + Terminal Cost)
        eye_nx = torch.eye(nx, device=self.device)
        for _ in range(H):
            P_blocks.append(1e-3 * eye_nx)
        P_blocks.append(1e-3 * eye_nx + self.Q)  # Terminal cost on xN
        
        self.P_mat = torch.block_diag(*P_blocks)

        # --- Construct A_eq (Dynamics) ---
        # Constraints: x0 = init, x_{k+1} = A x_k + B u_k
        A_eq = torch.zeros((self.n_eq, self.n_vars), device=self.device)
        u_start = 0
        x_start = H * nu
        
        # x0 constraint
        A_eq[0:nx, x_start:x_start+nx] = torch.eye(nx, device=self.device)
        
        # Dynamics
        for k in range(H):
            row = (k + 1) * nx
            col_u = u_start + k * nu
            col_x = x_start + k * nx
            col_x_next = x_start + (k + 1) * nx
            
            A_eq[row:row+nx, col_u:col_u+nu] = -self.B
            A_eq[row:row+nx, col_x:col_x+nx] = -self.A
            A_eq[row:row+nx, col_x_next:col_x_next+nx] = torch.eye(nx, device=self.device)
            
        # --- Invert KKT ---
        # KKT = [[P, A.T], [A, 0]]
        top = torch.cat([self.P_mat, A_eq.T], dim=1)
        bot = torch.cat([A_eq, torch.zeros((self.n_eq, self.n_eq), device=self.device)], dim=1)
        KKT = torch.cat([top, bot], dim=0)
        
        # Pre-compute inverse for O(1) online solve
        self.KKT_inv = torch.linalg.inv(KKT)

    def _shift_warm_start(self, z0):
        if self.H <= 1:
            return

        u_prev = self._warm_sol[self.u_slice].view(self.H, self.nu)
        z_prev = self._warm_sol[self.nu_total:].view(self.H + 1, self.nx)
        zu_prev = self._warm_Z_u.view(self.H, self.nu)
        yu_prev = self._warm_Y_u.view(self.H, self.nu)

        u_shift = torch.empty_like(u_prev)
        z_shift = torch.empty_like(z_prev)
        zu_shift = torch.empty_like(zu_prev)
        yu_shift = torch.empty_like(yu_prev)

        u_shift[:-1] = u_prev[1:]
        u_shift[-1] = u_prev[-1]

        zu_shift[:-1] = zu_prev[1:]
        zu_shift[-1] = zu_prev[-1]

        yu_shift[:-1] = yu_prev[1:]
        yu_shift[-1] = yu_prev[-1]

        z_shift[0] = z0
        z_shift[1:-1] = z_prev[2:]
        z_shift[-1] = z_prev[-1]

        self._sol[self.u_slice] = u_shift.reshape(-1)
        self._sol[self.nu_total:] = z_shift.reshape(-1)
        self._Z_u.copy_(zu_shift.reshape(-1))
        self._Y_u.copy_(yu_shift.reshape(-1))

    def _solve_torch_single(self, z0, zg):
        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()
        
        # Initialize variables
        if ENABLE_WARM_START:
            self._sol.copy_(self._warm_sol)
            self._Z_u.copy_(self._warm_Z_u)
            self._Y_u.copy_(self._warm_Y_u)
            self._shift_warm_start(z0)
        else:
            self._sol.zero_()
            self._Z_u.zero_()
            self._Y_u.zero_()
        
        sol = self._sol
        Z_u = self._Z_u
        Y_u = self._Y_u
        q_base = self._q_base
        q_nominal = self._q_nominal
        current_q = self._current_q
        rhs = self._rhs
        z_prev = self._z_prev
        b_eq = self._b_eq

        # Linear cost on xN: -Q * z_goal
        q_nominal.copy_(q_base)
        q_nominal[self.xN_idx:] += -(self.Q @ zg)

        # Equality RHS (Initial state)
        b_eq.zero_()
        b_eq[:self.nx] = z0
        
        # ADMM Loop
        for _ in range(self.steps):
            # 1. Primal Update (Linear Solve)
            # Add ADMM penalties to q: -rho * (Z - Y)
            current_q.copy_(q_nominal)
            current_q[self.u_slice] -= self.rho * (Z_u - Y_u)
            
            rhs[:self.n_vars] = -current_q
            rhs[self.n_vars:] = b_eq
            sol_augmented = self.KKT_inv @ rhs
            
            sol = sol_augmented[:self.n_vars]
            u_sol = sol[self.u_slice]
            
            # [cite_start]2. Projection (Z Update) [cite: 234]
            # Project u onto box constraints
            z_prev.copy_(Z_u)
            torch.clamp(u_sol + Y_u, min=self.u_min_tiled, max=self.u_max_tiled, out=Z_u)
            
            # [cite_start]3. Dual Update (Y Update) [cite: 235]
            Y_u.add_(u_sol - Z_u)

            if ENABLE_EARLY_STOP:
                primal_res = torch.linalg.vector_norm(u_sol - Z_u)
                dual_res = torch.linalg.vector_norm(self.rho * (Z_u - z_prev))
                if primal_res <= ADMM_PRIMAL_TOL and dual_res <= ADMM_DUAL_TOL:
                    break

        # Unpack
        if ENABLE_WARM_START:
            self._warm_sol.copy_(sol)
            self._warm_Z_u.copy_(Z_u)
            self._warm_Y_u.copy_(Y_u)

        u_flat = sol[:self.nu_total]
        z_flat = sol[self.nu_total:]
        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0

        z_traj = z_flat.reshape(self.H + 1, self.nx)
        u_traj = u_flat.reshape(self.H, self.nu)
        return z_traj, u_traj, solve_time

    def solve_torch(self, z0, zg):
        """
        Solve MPC in torch.
        Accepts either single inputs [nx] / [nx] or batched inputs [B, nx] / [B, nx].
        Returns:
          - single input: z_traj [H+1, nx], u_traj [H, nu], solve_time (float)
          - batched input: z_traj [B, H+1, nx], u_traj [B, H, nu], solve_time (float)
        """
        if z0.ndim == 1 and zg.ndim == 1:
            return self._solve_torch_single(z0, zg)

        if z0.ndim != 2 or zg.ndim != 2:
            raise ValueError(
                f"Expected z0/zg shapes [nx] or [B,nx], got z0={tuple(z0.shape)}, zg={tuple(zg.shape)}"
            )
        if z0.shape != zg.shape:
            raise ValueError(f"z0 and zg must have the same shape, got {tuple(z0.shape)} vs {tuple(zg.shape)}")
        if z0.shape[1] != self.nx:
            raise ValueError(f"Latent dim mismatch: expected {self.nx}, got {z0.shape[1]}")

        maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()
        bs = z0.shape[0]

        # Batched ADMM work buffers.
        sol = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.device)
        Z_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)
        Y_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)
        b_eq = torch.zeros((bs, self.n_eq), dtype=torch.float32, device=self.device)

        # q_nominal = q_base with terminal linear term -Q * z_goal for each batch element.
        q_nominal = self._q_base.unsqueeze(0).repeat(bs, 1)
        q_nominal[:, self.xN_idx:] += -(zg @ self.Q.T)
        b_eq[:, : self.nx] = z0

        for _ in range(self.steps):
            current_q = q_nominal.clone()
            current_q[:, self.u_slice] -= self.rho * (Z_u - Y_u)

            rhs = torch.cat([-current_q, b_eq], dim=1)
            sol_augmented = rhs @ self.KKT_inv.T
            sol = sol_augmented[:, : self.n_vars]
            u_sol = sol[:, self.u_slice]

            z_prev = Z_u.clone()
            torch.clamp(u_sol + Y_u, min=self.u_min_tiled, max=self.u_max_tiled, out=Z_u)
            Y_u.add_(u_sol - Z_u)

            if ENABLE_EARLY_STOP:
                primal_res = torch.linalg.vector_norm(u_sol - Z_u, dim=1)
                dual_res = torch.linalg.vector_norm(self.rho * (Z_u - z_prev), dim=1)
                if bool(torch.all(primal_res <= ADMM_PRIMAL_TOL) and torch.all(dual_res <= ADMM_DUAL_TOL)):
                    break

        u_flat = sol[:, : self.nu_total]
        z_flat = sol[:, self.nu_total :]
        maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0

        z_traj = z_flat.reshape(bs, self.H + 1, self.nx)
        u_traj = u_flat.reshape(bs, self.H, self.nu)
        return z_traj, u_traj, solve_time

    def solve(self, z0_np, zg_np):
        z0 = torch.as_tensor(z0_np, dtype=torch.float32, device=self.device)
        zg = torch.as_tensor(zg_np, dtype=torch.float32, device=self.device)
        z_traj_t, u_traj_t, solve_time = self.solve_torch(z0, zg)
        z_traj = z_traj_t.cpu().numpy()
        u_traj = u_traj_t.cpu().numpy()
        return z_traj, u_traj, solve_time


# ====== VISUALIZATION HELPERS (Unchanged) ======
def plot_3d(x_decoded, x_true, start_state, goal_state, planned_trajs=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if planned_trajs:
        for traj in planned_trajs:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="gray", alpha=0.15, linewidth=1)

    ax.plot(x_decoded[:, 0], x_decoded[:, 1], x_decoded[:, 2], "r", linewidth=2, label="Latent Plan (Decoded)")
    ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "b", linewidth=2, label="True Dynamics (Applied U)")

    ax.scatter(*start_state[:3], c="g", s=60, label="Start")
    ax.scatter(*goal_state[:3], c="r", s=60, label="Goal")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(0, 5)
    ax.set_title("MPC: Decoded Plan vs True Dynamics")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    return fig

def plot_controls(u_traj):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(u_traj[:, 0], label="u1 (thrust)")
    ax1.set_title("Applied Controls")
    ax1.set_ylabel("Thrust")
    ax1.grid(True); ax1.legend()

    ax2.plot(u_traj[:, 1], label="u2")
    ax2.plot(u_traj[:, 2], label="u3")
    ax2.plot(u_traj[:, 3], label="u4")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Torques")
    ax2.grid(True); ax2.legend()
    plt.tight_layout()
    return fig

def init_live_plot(start_state, goal_state, avg_solve_s):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*start_state[:3], c="g", s=60, label="Start")
    ax.scatter(*goal_state[:3], c="r", s=60, label="Goal")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(0, 5)
    ax.set_title(f"MPC, N = {HORIZON}, Avg Solve Time = {(avg_solve_s*1000):.3f}ms")
    ax.set_box_aspect([1, 1, 1])

    planned_line, = ax.plot([], [], [], "r", linewidth=2, label="Latent Plan (Decoded)")
    true_line, = ax.plot([], [], [], "b", linewidth=2, label="True Dynamics")
    ax.legend()

    plt.tight_layout()
    return fig, ax, planned_line, true_line

def update_live_plot(planned_line, true_line, x_decoded, true_traj):
    planned_line.set_data(x_decoded[:, 0], x_decoded[:, 1])
    planned_line.set_3d_properties(x_decoded[:, 2])
    true_line.set_data(true_traj[:, 0], true_traj[:, 1])
    true_line.set_3d_properties(true_traj[:, 2])

def save_animation(anim, video_save_path):
    video_save_path = Path(resolve_path(video_save_path))
    video_save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=max(1, int(round(1.0 / DT))))
        anim.save(video_save_path, writer=writer)
        print(f"Saved video to {video_save_path}")
    except Exception as exc:
        fallback_path = video_save_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=max(1, int(round(1.0 / DT))))
        anim.save(fallback_path, writer=writer)
        print(f"Saved video to {fallback_path} (mp4 unavailable: {exc})")


def show_3d_animation(start_state, goal_state, avg_solve_s, planned_trajs, true_traj, enable_video_gen=False, video_save_path=VIDEO_SAVE_PATH):
    if not planned_trajs:
        return None, None

    fig, _, planned_line, true_line = init_live_plot(start_state, goal_state, avg_solve_s)

    def _animate(frame_idx):
        plan = planned_trajs[frame_idx]
        update_live_plot(planned_line, true_line, plan, true_traj[:frame_idx + 2])
        return planned_line, true_line

    anim = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(planned_trajs),
        interval=DT * 1000.0,
        blit=False,
        repeat=False,
    )
    fig._anim = anim
    return fig, anim


# ====== MAIN LOOP ======
def main():
    print(f"Using device: {DEVICE}")

    model_path = resolve_path(KOOPMAN_PATH)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if model_config is None:
        print("ERROR: Missing model config in checkpoint.")
        return

    model_params = {k: v for k, v in model_config.items()
                    if k in ["state_dim", "control_dim", "latent_dim", "hidden_width", "depth", "activation_fn"]}
    if isinstance(model_params.get("activation_fn"), str):
        model_params["activation_fn"] = get_activation_fn(model_params["activation_fn"])

    model = DeepKoopman(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalizer = Normalizer(checkpoint["normalization_stats"])
    rng = np.random.default_rng()
    start_state, goal_state = sample_start_goal(rng)

    # Encode start/goal
    with torch.inference_mode():
        x0_norm = normalizer.normalize(start_state)
        xg_norm = normalizer.normalize(goal_state)
        x0_t = torch.tensor(x0_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        xg_t = torch.tensor(xg_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        z0 = model.encoder(x0_t).squeeze(0)
        zg = model.encoder(xg_t).squeeze(0)

    A_np = model.A.weight.detach().cpu().numpy()
    B_np = model.B.weight.detach().cpu().numpy()

    print(f"Start position: {start_state[:3]}")
    print(f"Goal position:  {goal_state[:3]}")
    print("Initializing GPU ADMM Solver...")

    # --- SWAP: Use new GPUADMM_MPCSolver instead of LatentMPCSolver ---
    mpc_solver = GPUADMM_MPCSolver(
        A_np, B_np,
        Q_val=Q,
        R_val=R, # Using Smoothness weight as main R weight for approximation
        u_min=np.array([U1_MIN, U_TORQUE_MIN, U_TORQUE_MIN, U_TORQUE_MIN]),
        u_max=np.array([U1_MAX, U_TORQUE_MAX, U_TORQUE_MAX, U_TORQUE_MAX]),
        horizon=HORIZON,
        rho=RHO,      # Tunable ADMM penalty parameter
        n_admm_steps=MAX_ADMM_STEPS, # Fixed iterations for constant time
        device=DEVICE
    )
    print("Solver ready. Running MPC loop...")

    current_state = start_state.copy()
    true_traj = [current_state.copy()]
    applied_controls = []
    planned_trajs = []
    
    solve_times = []
    timing_sums = {
        "encode": 0.0,
        "solve": 0.0,
        "plan_extract_decode": 0.0,
        "control_transfer": 0.0,
        "rollout": 0.0,
        "metrics": 0.0,
        "control_extract_apply": 0.0,
        "total_step": 0.0,
    }
    # Metrics for:
    # 1) Per-MPC-step rollout consistency under planned controls
    step_traj_mse = []
    step_pos_traj_mse = []
    # 2) 1-step and k-step error curves, k=1..H
    k_state_mse_sum = np.zeros(HORIZON, dtype=np.float64)
    k_pos_mse_sum = np.zeros(HORIZON, dtype=np.float64)
    k_count = np.zeros(HORIZON, dtype=np.int64)

    pbar = tqdm(range(MAX_MPC_STEPS), desc="MPC Steps")
    for step in pbar:
        step_t0 = time.perf_counter()

        # Encode current state
        encode_t0 = time.perf_counter()
        with torch.inference_mode():
            x_norm = normalizer.normalize(current_state)
            x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            z_curr = model.encoder(x_t).squeeze(0)
        maybe_cuda_synchronize(DEVICE)
        timing_sums["encode"] += time.perf_counter() - encode_t0

        # Solve (Using the new GPU solver)
        solve_t0 = time.perf_counter()
        z_traj_t, u_traj_t, solve_time = mpc_solver.solve_torch(z_curr, zg)
        timing_sums["solve"] += time.perf_counter() - solve_t0
        solve_times.append(solve_time)
        
        # Display solve time (should be < 1ms on GPU usually)
        pbar.set_postfix({"solve_ms": f"{solve_time*1000:.4f}"})

        # Decode plan for visualization
        plan_extract_decode_t0 = time.perf_counter()
        with torch.inference_mode():
            x_dec_norm = model.decoder(z_traj_t).cpu().numpy()
            x_decoded = normalizer.denormalize(x_dec_norm)
            planned_trajs.append(x_decoded)
        timing_sums["plan_extract_decode"] += time.perf_counter() - plan_extract_decode_t0

        # Metric 1 + 2: compare decoded prediction to true rollout under the same u plan.
        control_transfer_t0 = time.perf_counter()
        u_plan_np = u_traj_t.cpu().numpy()
        timing_sums["control_transfer"] += time.perf_counter() - control_transfer_t0

        rollout_t0 = time.perf_counter()
        x_true_rollout = rollout_true_dynamics_np(current_state, u_plan_np, dt=DT)
        timing_sums["rollout"] += time.perf_counter() - rollout_t0

        # Skip k=0 (same initial state); evaluate prediction error at k=1..H.
        metrics_t0 = time.perf_counter()
        sq_err = (x_decoded[1:] - x_true_rollout[1:]) ** 2  # [H, 12]

        # 1) Per-step full-horizon trajectory error.
        step_traj_mse.append(float(np.mean(sq_err)))
        step_pos_traj_mse.append(float(np.mean(sq_err[:, :3])))

        # 2) k-step error curves accumulated over replans.
        k_state_mse_sum += np.mean(sq_err, axis=1)
        k_pos_mse_sum += np.mean(sq_err[:, :3], axis=1)
        k_count += 1
        timing_sums["metrics"] += time.perf_counter() - metrics_t0

        # Apply control
        control_extract_apply_t0 = time.perf_counter()
        u0 = u_traj_t[0].cpu().numpy()
        applied_controls.append(u0)
        current_state = quadrotor_dynamics_np(current_state, u0)
        true_traj.append(current_state.copy())
        timing_sums["control_extract_apply"] += time.perf_counter() - control_extract_apply_t0
        timing_sums["total_step"] += time.perf_counter() - step_t0

        dist_to_goal = np.linalg.norm(current_state[:3] - goal_state[:3])
        if dist_to_goal <= GOAL_TOL:
            break

    true_traj = np.array(true_traj)
    applied_controls = np.array(applied_controls)

    print(f"Final position: {true_traj[-1, :3]}")
    print(f"Goal position:  {goal_state[:3]}")
    if solve_times:
        avg_solve = float(np.mean(solve_times))
        print(f"Average solve time: {avg_solve*1000:.4f} ms")
    else:
        avg_solve = 0.0

    if np.all(k_count > 0):
        k_state_mse_curve = k_state_mse_sum / k_count
        k_pos_mse_curve = k_pos_mse_sum / k_count
        print_planning_error_metrics(step_traj_mse, step_pos_traj_mse, k_state_mse_curve, k_pos_mse_curve)

    if ENABLE_TIMING_BREAKDOWN:
        print_timing_breakdown_metrics(timing_sums, len(solve_times))

    anim = None
    anim_save_path = VIDEO_SAVE_PATH
    if ENABLE_ANIM:
        _, anim = show_3d_animation(
            start_state,
            goal_state,
            avg_solve,
            planned_trajs,
            true_traj,
            enable_video_gen=ENABLE_VIDEO_GEN,
            video_save_path=VIDEO_SAVE_PATH,
        )

    last_plan = planned_trajs[-1] if planned_trajs else true_traj
    plot_3d(last_plan, true_traj, start_state, goal_state, planned_trajs=planned_trajs)
    if len(applied_controls) > 0:
        plot_controls(applied_controls)

    plt.show()

    if ENABLE_ANIM and ENABLE_VIDEO_GEN and anim is not None:
        save_animation(anim, anim_save_path)

if __name__ == "__main__":
    main()
