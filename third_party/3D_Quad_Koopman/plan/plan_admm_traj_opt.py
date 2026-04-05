import sys
from pathlib import Path
import time


def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from models import DeepKoopman
from utils import resolve_path, get_activation_fn


# ================== CONFIG ==================
KOOPMAN_PATH = "models/koop_self_train_3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HORIZON = 100
DT = 0.025
N_ADMM_STEPS = 25
RHO = 10.0
BATCH = 1024
TO_SHOW = 5

# Cost weights
Q_TERMINAL = 100.0
R_EFFORT = 0.25

# Physical bounds
MASS = 1.0
GRAV = -9.81
U1_MIN, U1_MAX = 0.0, -2.0 * GRAV * MASS
U_TORQUE_MIN, U_TORQUE_MAX = -0.1, 0.1

# Sampling bounds
MIN_DIST = 2.5
LOW_BOUNDS = np.array([0.1, 0.1, 0.1], dtype=np.float32)
HIGH_BOUNDS = np.array([4.9, 4.9, 4.9], dtype=np.float32)

# Optional post-rollout truncation to in-domain trajectories
ENFORCE_POS_BOUNDS = True

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINED_PLOT_PATH = SCRIPT_DIR / "traj_opt_batch_overlay.png"


# ================== HELPERS ==================
class Normalizer:
    def __init__(self, stats):
        self.min = stats["min"].detach().cpu().numpy()
        self.range = stats["range"].detach().cpu().numpy()
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        return 2.0 * (x - self.min) / self.range - 1.0

    def denormalize(self, x_norm):
        return (x_norm + 1.0) / 2.0 * self.range + self.min


def sample_start_goal(rng, min_dist=MIN_DIST):
    start = np.zeros(12, dtype=np.float32)
    goal = np.zeros(12, dtype=np.float32)

    start_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
    while True:
        goal_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
        if np.linalg.norm(goal_pos - start_pos) > float(min_dist):
            break

    start[:3] = start_pos
    goal[:3] = goal_pos
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


def rollout_true_dynamics(x0, u_seq):
    traj = np.zeros((u_seq.shape[0] + 1, 12), dtype=np.float32)
    traj[0] = x0
    for k in range(u_seq.shape[0]):
        traj[k + 1] = quadrotor_dynamics_np(traj[k], u_seq[k])
    return traj


def truncate_oob_rollout(x_traj, u_traj):
    in_bounds = np.all(
        (x_traj[:, :3] >= LOW_BOUNDS) & (x_traj[:, :3] <= HIGH_BOUNDS),
        axis=1,
    )
    if np.all(in_bounds):
        return x_traj, u_traj

    first_bad = int(np.argmax(~in_bounds))
    if first_bad <= 1:
        return None, None

    return x_traj[:first_bad], u_traj[: first_bad - 1]


# ================== ADMM TRAJ OPT ==================
class GPUADMMTrajOptSolver:
    """
    Open-loop latent trajectory optimization with ADMM.

    Decision vars: [u_0...u_{H-1}, z_0...z_H]
    Constraints: z_{k+1} = A z_k + B u_k, z_0 fixed.
    Cost: terminal(z_H-z_g) + effort(u-u_hover).
    """

    def __init__(
        self,
        A_np,
        B_np,
        horizon,
        u_min,
        u_max,
        q_terminal=Q_TERMINAL,
        r_effort=R_EFFORT,
        rho=RHO,
        n_admm_steps=N_ADMM_STEPS,
        device=DEVICE,
    ):
        self.device = device
        self.H = int(horizon)
        self.rho = float(rho)
        self.steps = int(n_admm_steps)

        self.A = torch.tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        self.u_hover = torch.tensor([-GRAV * MASS, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=device)

        self.nz_total = (self.H + 1) * self.nx
        self.nu_total = self.H * self.nu
        self.n_vars = self.nu_total + self.nz_total
        self.n_eq = (self.H + 1) * self.nx
        self.u_slice = slice(0, self.nu_total)
        self.xN_idx = self.H * self.nu + self.H * self.nx

        self.P = self._build_hessian(q_terminal, r_effort)
        self.A_eq = self._build_dynamics_matrix()
        self.KKT_inv = self._build_kkt_inverse()

    def _build_hessian(self, q_terminal, r_effort):
        H, nu, nx = self.H, self.nu, self.nx

        # Control Hessian: effort + ADMM proximal term.
        R_eff = torch.eye(nu, device=self.device) * max(r_effort, 1e-6)
        P_u = torch.zeros((H * nu, H * nu), dtype=torch.float32, device=self.device)
        for k in range(H):
            sl = slice(k * nu, (k + 1) * nu)
            P_u[sl, sl] = R_eff + self.rho * torch.eye(nu, device=self.device)

        P_x = torch.eye((H + 1) * nx, dtype=torch.float32, device=self.device) * 1e-4
        P_x[-nx:, -nx:] += torch.eye(nx, dtype=torch.float32, device=self.device) * q_terminal

        top = torch.cat([P_u, torch.zeros((H * nu, (H + 1) * nx), device=self.device)], dim=1)
        bot = torch.cat([torch.zeros(((H + 1) * nx, H * nu), device=self.device), P_x], dim=1)
        return torch.cat([top, bot], dim=0)

    def _build_dynamics_matrix(self):
        H, nx, nu = self.H, self.nx, self.nu
        A_eq = torch.zeros((self.n_eq, self.n_vars), dtype=torch.float32, device=self.device)

        x_offset = H * nu
        A_eq[0:nx, x_offset : x_offset + nx] = torch.eye(nx, device=self.device)

        for k in range(H):
            row = (k + 1) * nx
            col_u = k * nu
            col_x = x_offset + k * nx
            col_x_next = x_offset + (k + 1) * nx

            A_eq[row : row + nx, col_u : col_u + nu] = -self.B
            A_eq[row : row + nx, col_x : col_x + nx] = -self.A
            A_eq[row : row + nx, col_x_next : col_x_next + nx] = torch.eye(nx, device=self.device)

        return A_eq

    def _build_kkt_inverse(self):
        top = torch.cat([self.P, self.A_eq.T], dim=1)
        bot = torch.cat([self.A_eq, torch.zeros((self.n_eq, self.n_eq), device=self.device)], dim=1)
        kkt = torch.cat([top, bot], dim=0)
        return torch.linalg.inv(kkt)

    def solve(self, z0_np, zg_np):
        t0 = time.perf_counter()
        z0 = torch.as_tensor(z0_np, dtype=torch.float32, device=self.device)
        zg = torch.as_tensor(zg_np, dtype=torch.float32, device=self.device)

        sol = torch.zeros((self.n_vars,), dtype=torch.float32, device=self.device)
        Z_u = torch.zeros((self.nu_total,), dtype=torch.float32, device=self.device)
        Y_u = torch.zeros((self.nu_total,), dtype=torch.float32, device=self.device)

        q = torch.zeros((self.n_vars,), dtype=torch.float32, device=self.device)
        lin_u_cost = -(R_EFFORT * self.u_hover)
        for k in range(self.H):
            q[k * self.nu : (k + 1) * self.nu] = lin_u_cost
        q[self.xN_idx :] = -Q_TERMINAL * zg

        b_eq = torch.zeros((self.n_eq,), dtype=torch.float32, device=self.device)
        b_eq[: self.nx] = z0

        for _ in range(self.steps):
            current_q = q.clone()
            current_q[self.u_slice] -= self.rho * (Z_u - Y_u)

            rhs = torch.cat([-current_q, b_eq], dim=0)
            sol_aug = self.KKT_inv @ rhs
            sol = sol_aug[: self.n_vars]
            u_sol = sol[self.u_slice]

            Z_u = torch.clamp(u_sol + Y_u, min=self.u_min.repeat(self.H), max=self.u_max.repeat(self.H))
            Y_u = Y_u + (u_sol - Z_u)

        u_flat = sol[: self.nu_total]
        z_flat = sol[self.nu_total :]

        u_traj = u_flat.reshape(self.H, self.nu).detach().cpu().numpy()
        z_traj = z_flat.reshape(self.H + 1, self.nx).detach().cpu().numpy()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        solve_time = time.perf_counter() - t0
        return z_traj, u_traj, solve_time

    def solve_batch(self, z0_batch_np, zg_batch_np):
        t0 = time.perf_counter()
        z0_batch = torch.as_tensor(z0_batch_np, dtype=torch.float32, device=self.device)
        zg_batch = torch.as_tensor(zg_batch_np, dtype=torch.float32, device=self.device)
        bs = z0_batch.shape[0]

        sol = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.device)
        Z_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)
        Y_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)

        q = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.device)
        lin_u_cost = -(R_EFFORT * self.u_hover)
        for k in range(self.H):
            q[:, k * self.nu : (k + 1) * self.nu] = lin_u_cost
        q[:, self.xN_idx :] = -Q_TERMINAL * zg_batch

        b_eq = torch.zeros((bs, self.n_eq), dtype=torch.float32, device=self.device)
        b_eq[:, : self.nx] = z0_batch

        u_min = self.u_min.repeat(self.H).view(1, -1)
        u_max = self.u_max.repeat(self.H).view(1, -1)

        for _ in range(self.steps):
            current_q = q.clone()
            current_q[:, self.u_slice] -= self.rho * (Z_u - Y_u)

            rhs = torch.cat([-current_q, b_eq], dim=1)
            sol_aug = rhs @ self.KKT_inv.T
            sol = sol_aug[:, : self.n_vars]
            u_sol = sol[:, self.u_slice]

            Z_u = torch.maximum(torch.minimum(u_sol + Y_u, u_max), u_min)
            Y_u = Y_u + (u_sol - Z_u)

        u_flat = sol[:, : self.nu_total]
        z_flat = sol[:, self.nu_total :]
        u_traj = u_flat.reshape(bs, self.H, self.nu).detach().cpu().numpy()
        z_traj = z_flat.reshape(bs, self.H + 1, self.nx).detach().cpu().numpy()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        solve_time = time.perf_counter() - t0
        return z_traj, u_traj, solve_time


# ================== VIZ ==================
def plot_3d_batch_overlay(samples, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for j, sample in enumerate(samples):
        _, start_state, goal_state, x_plan, x_true, _, _ = sample
        plan_label = "Decoded latent plan" if j == 0 else None
        true_label = "True rollout" if j == 0 else None
        start_label = "Start" if j == 0 else None
        goal_label = "Goal" if j == 0 else None

        ax.plot(x_plan[:, 0], x_plan[:, 1], x_plan[:, 2], "r", linewidth=1.3, alpha=0.65, label=plan_label)
        ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "b", linewidth=1.3, alpha=0.75, label=true_label)
        ax.scatter(*start_state[:3], c="g", s=25, alpha=0.8, label=start_label)
        ax.scatter(*goal_state[:3], c="m", s=25, alpha=0.8, label=goal_label)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Open-loop ADMM Traj Opt Overlay ({len(samples)} rollouts)")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ================== MAIN ==================
def main():
    print(f"Using device: {DEVICE}")
    model_path = resolve_path(KOOPMAN_PATH)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if model_config is None:
        raise RuntimeError("Checkpoint missing model_config/config")

    model_params = {
        k: v
        for k, v in model_config.items()
        if k in ["state_dim", "control_dim", "latent_dim", "hidden_width", "depth", "activation_fn"]
    }
    if isinstance(model_params.get("activation_fn"), str):
        model_params["activation_fn"] = get_activation_fn(model_params["activation_fn"])

    model = DeepKoopman(**model_params).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalizer = Normalizer(checkpoint["normalization_stats"])
    rng = np.random.default_rng()

    starts, goals = [], []
    for _ in range(BATCH):
        s, g = sample_start_goal(rng)
        starts.append(s)
        goals.append(g)
    starts = np.stack(starts)
    goals = np.stack(goals)

    with torch.no_grad():
        x0_norm = normalizer.normalize(starts)
        xg_norm = normalizer.normalize(goals)
        x0_t = torch.tensor(x0_norm, dtype=torch.float32, device=DEVICE)
        xg_t = torch.tensor(xg_norm, dtype=torch.float32, device=DEVICE)
        z0_batch = model.encoder(x0_t).cpu().numpy()
        zg_batch = model.encoder(xg_t).cpu().numpy()

    A_np = model.A.weight.detach().cpu().numpy()
    B_np = model.B.weight.detach().cpu().numpy()

    solver = GPUADMMTrajOptSolver(
        A_np=A_np,
        B_np=B_np,
        horizon=HORIZON,
        u_min=np.array([U1_MIN, U_TORQUE_MIN, U_TORQUE_MIN, U_TORQUE_MIN], dtype=np.float32),
        u_max=np.array([U1_MAX, U_TORQUE_MAX, U_TORQUE_MAX, U_TORQUE_MAX], dtype=np.float32),
        q_terminal=Q_TERMINAL,
        r_effort=R_EFFORT,
        rho=RHO,
        n_admm_steps=N_ADMM_STEPS,
        device=DEVICE,
    )

    z_plans, u_plans, solve_time = solver.solve_batch(z0_batch, zg_batch)
    print(f"Solved {BATCH} trajectories (N={HORIZON}) in {solve_time:.4f} s ({solve_time / BATCH:.6f} s/problem)")

    with torch.no_grad():
        z_t = torch.tensor(z_plans, dtype=torch.float32, device=DEVICE)
        x_dec_norm = model.decoder(z_t.view(-1, z_t.shape[-1])).view(BATCH, HORIZON + 1, -1).cpu().numpy()
        x_decoded_batch = normalizer.denormalize(x_dec_norm)

    collected = []
    final_errs = []
    for i in range(BATCH):
        x_true = rollout_true_dynamics(starts[i], u_plans[i])
        u_use = u_plans[i]
        if ENFORCE_POS_BOUNDS:
            x_true_trunc, u_trunc = truncate_oob_rollout(x_true, u_use)
            if x_true_trunc is None:
                continue
            x_true = x_true_trunc
            u_use = u_trunc

        final_err = float(np.linalg.norm(x_true[-1, :3] - goals[i, :3]))
        final_errs.append(final_err)
        x_plan = x_decoded_batch[i][: x_true.shape[0]]
        collected.append((i, starts[i], goals[i], x_plan, x_true, u_use, final_err))

    print(f"Valid collected rollouts: {len(collected)}/{BATCH}")
    if final_errs:
        print(f"Mean final position error: {float(np.mean(final_errs)):.4f} m")

    if not collected:
        print("No valid rollout to plot.")
        return

    n_show = min(TO_SHOW, len(collected))
    show_ids = rng.choice(len(collected), size=n_show, replace=False)
    selected_samples = [collected[int(i)] for i in show_ids]
    plot_3d_batch_overlay(selected_samples, COMBINED_PLOT_PATH)
    print(f"Saved combined overlay plot: {COMBINED_PLOT_PATH}")
    for j, sample in enumerate(selected_samples):
        idx, _, _, _, _, _, final_err = sample
        print(f"[overlay {j + 1}/{n_show}] idx={idx} err={final_err:.4f} m")


if __name__ == "__main__":
    main()
