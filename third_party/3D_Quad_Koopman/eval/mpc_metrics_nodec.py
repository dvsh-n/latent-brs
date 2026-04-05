import sys
import argparse
import time
from pathlib import Path


def find_root(start_path, marker=".root"):
    for parent in [start_path] + list(start_path.parents):
        if (parent / marker).exists():
            return parent
    return start_path


ROOT_DIR = find_root(Path(__file__).resolve().parent)
sys.path.append(str(ROOT_DIR))

import numpy as np
import torch
from tqdm import tqdm

from models import DeepKoopmanNoDec
from utils import resolve_path, get_activation_fn, print_table, print_config_table


MODEL_PATH = "models/koop_quad3d_nodec_dagger_1.pt"
DEVICE = "cuda"

HORIZON_START = 25
HORIZON_END = 125
HORIZON_STEP = 20

NUM_PROBLEMS = 50
MAX_MPC_STEPS = 500
GOAL_TOL = 0.10
RNG_SEED = 0

Q = 100.0
R = 0.25

RHO = 10.0
MAX_ADMM_STEPS = 100
ENABLE_EARLY_STOP = False
ADMM_PRIMAL_TOL = 1e-4
ADMM_DUAL_TOL = 1e-2

DT = 0.025
MASS = 1.0
GRAV = -9.81
IX, IY, IZ = 0.5, 0.1, 0.3
U1_MIN, U1_MAX = 0.0, -2.0 * GRAV * MASS
U_TORQUE_MIN, U_TORQUE_MAX = -0.1, 0.1

MIN_DIST = 2.5
LOW_BOUNDS = np.array([0.1, 0.1, 0.1], dtype=np.float32)
HIGH_BOUNDS = np.array([4.9, 4.9, 4.9], dtype=np.float32)


class Normalizer:
    def __init__(self, stats):
        self.min = stats["min"].detach().cpu().numpy()
        self.range = stats["range"].detach().cpu().numpy()
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        return 2 * (x - self.min) / self.range - 1

    def denormalize(self, x_norm):
        return (x_norm + 1) / 2 * self.range + self.min


def preprocess_state(x, normalizer, enable_normalization):
    return normalizer.normalize(x) if enable_normalization else x


def postprocess_state(x, normalizer, enable_normalization):
    return normalizer.denormalize(x) if enable_normalization else x


def quadrotor_dynamics_np(x, u, dt=DT):
    psi, theta, phi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    u1, u2, u3, u4 = u[0], u[1], u[2], u[3]

    cos_theta = np.cos(theta)
    if np.abs(cos_theta) < 1e-6:
        cos_theta = 1e-6 if cos_theta == 0 else 1e-6 * np.sign(cos_theta)

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
            ((IY - IZ) / IX) * q * r + u2 / IX,
            ((IZ - IX) / IY) * p * r + u3 / IY,
            ((IX - IY) / IZ) * p * q + u4 / IZ,
        ],
        dtype=np.float32,
    )
    return x + dt * xdot


def rollout_true_dynamics_np(x0, u_seq, dt=DT):
    traj = [x0.copy()]
    x = x0.copy()
    for u in u_seq:
        x = quadrotor_dynamics_np(x, u, dt=dt)
        traj.append(x.copy())
    return np.asarray(traj, dtype=np.float32)


def sample_start_goal_batch(rng, n):
    starts = np.zeros((n, 12), dtype=np.float32)
    goals = np.zeros((n, 12), dtype=np.float32)

    for i in range(n):
        start_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
        while True:
            goal_pos = rng.uniform(low=LOW_BOUNDS, high=HIGH_BOUNDS)
            if np.linalg.norm(goal_pos - start_pos) > MIN_DIST:
                break
        starts[i, :3] = start_pos
        goals[i, :3] = goal_pos

    return starts, goals


class GPUADMM_MPCSolver:
    def __init__(
        self,
        A_np,
        B_np,
        Q_val,
        R_val,
        u_min,
        u_max,
        horizon,
        rho=1.0,
        n_admm_steps=15,
        device="cuda",
    ):
        self.H = horizon
        self.nx = A_np.shape[0]
        self.nu = B_np.shape[1]
        self.rho = rho
        self.steps = n_admm_steps
        self.device = device

        self.u_hover = torch.tensor([-GRAV * MASS, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)

        self.A = torch.tensor(A_np, dtype=torch.float32, device=device)
        self.B = torch.tensor(B_np, dtype=torch.float32, device=device)
        self.Q = torch.eye(self.nx, device=device) * Q_val
        self.R = torch.eye(self.nu, device=device) * (R_val if R_val > 0 else 1.0)

        self.u_min = torch.tensor(u_min, dtype=torch.float32, device=device)
        self.u_max = torch.tensor(u_max, dtype=torch.float32, device=device)

        self.nz_total = (self.H + 1) * self.nx
        self.nu_total = self.H * self.nu
        self.n_vars = self.nz_total + self.nu_total
        self.n_eq = (self.H + 1) * self.nx
        self.u_slice = slice(0, self.nu_total)
        self.xN_idx = self.H * self.nu + self.H * self.nx
        self.u_min_tiled = self.u_min.repeat(self.H)
        self.u_max_tiled = self.u_max.repeat(self.H)

        self._q_base = torch.zeros(self.n_vars, device=self.device)
        lin_u_cost = -(self.R @ self.u_hover)
        for k in range(self.H):
            self._q_base[k * self.nu : (k + 1) * self.nu] = lin_u_cost

        self._build_kkt_inverse()

    def _build_kkt_inverse(self):
        nx, nu, H = self.nx, self.nu, self.H

        P_blocks = []
        R_prox = self.R + self.rho * torch.eye(nu, device=self.device)
        for _ in range(H):
            P_blocks.append(R_prox)

        eye_nx = torch.eye(nx, device=self.device)
        for _ in range(H):
            P_blocks.append(1e-3 * eye_nx)
        P_blocks.append(1e-3 * eye_nx + self.Q)

        P_mat = torch.block_diag(*P_blocks)

        A_eq = torch.zeros((self.n_eq, self.n_vars), device=self.device)
        u_start = 0
        x_start = H * nu

        A_eq[0:nx, x_start : x_start + nx] = torch.eye(nx, device=self.device)

        for k in range(H):
            row = (k + 1) * nx
            col_u = u_start + k * nu
            col_x = x_start + k * nx
            col_x_next = x_start + (k + 1) * nx

            A_eq[row : row + nx, col_u : col_u + nu] = -self.B
            A_eq[row : row + nx, col_x : col_x + nx] = -self.A
            A_eq[row : row + nx, col_x_next : col_x_next + nx] = torch.eye(nx, device=self.device)

        top = torch.cat([P_mat, A_eq.T], dim=1)
        bot = torch.cat([A_eq, torch.zeros((self.n_eq, self.n_eq), device=self.device)], dim=1)
        KKT = torch.cat([top, bot], dim=0)
        self.KKT_inv = torch.linalg.inv(KKT)

    def solve_torch(self, z0, zg):
        if z0.ndim != 2 or zg.ndim != 2:
            raise ValueError(f"Expected [B,nx] tensors, got z0={tuple(z0.shape)} zg={tuple(zg.shape)}")
        if z0.shape != zg.shape:
            raise ValueError(f"z0 and zg shape mismatch: {tuple(z0.shape)} vs {tuple(zg.shape)}")

        t0 = time.perf_counter()
        bs = z0.shape[0]

        sol = torch.zeros((bs, self.n_vars), dtype=torch.float32, device=self.device)
        Z_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)
        Y_u = torch.zeros((bs, self.nu_total), dtype=torch.float32, device=self.device)
        b_eq = torch.zeros((bs, self.n_eq), dtype=torch.float32, device=self.device)

        q_nominal = self._q_base.unsqueeze(0).repeat(bs, 1)
        q_nominal[:, self.xN_idx :] += -(zg @ self.Q.T)
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
        solve_time = time.perf_counter() - t0

        z_traj = z_flat.reshape(bs, self.H + 1, self.nx)
        u_traj = u_flat.reshape(bs, self.H, self.nu)
        return z_traj, u_traj, solve_time


def load_model(model_path, device):
    checkpoint = torch.load(resolve_path(model_path), map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if model_config is None:
        raise RuntimeError("Missing model config in checkpoint")

    model_params = {
        k: v
        for k, v in model_config.items()
        if k in ["state_dim", "control_dim", "embedding_dim", "hidden_width", "depth", "activation_fn"]
    }
    if isinstance(model_params.get("activation_fn"), str):
        model_params["activation_fn"] = get_activation_fn(model_params["activation_fn"])

    model = DeepKoopmanNoDec(**model_params).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def print_checkpoint_configs(checkpoint):
    model_cfg = checkpoint.get("model_config", checkpoint.get("config"))
    train_cfg = checkpoint.get("training_config")

    if isinstance(model_cfg, dict):
        print_config_table(model_cfg, title="Checkpoint Model Configuration")
    else:
        print("Checkpoint Model Configuration: not found")

    if isinstance(train_cfg, dict):
        print_config_table(train_cfg, title="Checkpoint Training Configuration")
    else:
        print("Checkpoint Training Configuration: not found")


def evaluate_horizon(model, normalizer, solver, starts, goals, device, max_mpc_steps, goal_tol, enable_normalization):
    n = starts.shape[0]
    curr = starts.copy()

    reached = np.zeros(n, dtype=bool)
    steps_to_goal = np.full(n, max_mpc_steps, dtype=np.int32)

    plan_state_mse = []
    plan_pos_mse = []

    batch_solve_times = []
    problem_steps_total = 0

    with torch.inference_mode():
        goals_proc = preprocess_state(goals, normalizer, enable_normalization)
        goals_t = torch.tensor(goals_proc, dtype=torch.float32, device=device)
        z_goals = model.lift_state(goals_t)

    for step in range(max_mpc_steps):
        active_idx = np.where(~reached)[0]
        if active_idx.size == 0:
            break

        problem_steps_total += int(active_idx.size)

        with torch.inference_mode():
            curr_proc = preprocess_state(curr[active_idx], normalizer, enable_normalization)
            curr_t = torch.tensor(curr_proc, dtype=torch.float32, device=device)
            z_curr = model.lift_state(curr_t)
            z_goal_active = z_goals[active_idx]

            z_plan, u_plan, solve_time = solver.solve_torch(z_curr, z_goal_active)
            batch_solve_times.append(float(solve_time))

            bsz = z_plan.shape[0]
            x_plan_proc = z_plan[..., : model.state_dim].cpu().numpy()
            x_plan = postprocess_state(x_plan_proc.reshape(-1, model.state_dim), normalizer, enable_normalization)
            x_plan = x_plan.reshape(bsz, solver.H + 1, model.state_dim)

        u_plan_np = u_plan.cpu().numpy()

        for j, idx in enumerate(active_idx):
            x_true_rollout = rollout_true_dynamics_np(curr[idx], u_plan_np[j], dt=DT)
            sq_err = (x_plan[j, 1:] - x_true_rollout[1:]) ** 2
            plan_state_mse.append(float(np.mean(sq_err)))
            plan_pos_mse.append(float(np.mean(sq_err[:, :3])))

            u0 = u_plan_np[j, 0]
            curr[idx] = quadrotor_dynamics_np(curr[idx], u0, dt=DT)

        dist = np.linalg.norm(curr[active_idx, :3] - goals[active_idx, :3], axis=1)
        hit = dist <= goal_tol
        if np.any(hit):
            hit_idx = active_idx[hit]
            reached[hit_idx] = True
            steps_to_goal[hit_idx] = step + 1

    success_count = int(np.sum(reached))
    goal_reach_pct = 100.0 * success_count / max(n, 1)

    success_steps = steps_to_goal[reached]
    avg_solve_steps_success = float(np.mean(success_steps)) if success_steps.size > 0 else float("nan")
    avg_solve_steps_all = float(np.mean(steps_to_goal))

    avg_batch_solve_ms = 1000.0 * float(np.mean(batch_solve_times)) if batch_solve_times else float("nan")
    avg_per_problem_step_ms = (
        1000.0 * float(np.sum(batch_solve_times)) / problem_steps_total if problem_steps_total > 0 else float("nan")
    )

    mean_plan_state_mse = float(np.mean(plan_state_mse)) if plan_state_mse else float("nan")
    mean_plan_pos_mse = float(np.mean(plan_pos_mse)) if plan_pos_mse else float("nan")

    final_pos_mse = float(np.mean((curr[:, :3] - goals[:, :3]) ** 2))

    return {
        "success": success_count,
        "total": n,
        "goal_reach_pct": goal_reach_pct,
        "mean_plan_state_mse": mean_plan_state_mse,
        "mean_plan_pos_mse": mean_plan_pos_mse,
        "final_pos_mse": final_pos_mse,
        "avg_solve_steps_success": avg_solve_steps_success,
        "avg_solve_steps_all": avg_solve_steps_all,
        "avg_batch_solve_ms": avg_batch_solve_ms,
        "avg_per_problem_step_ms": avg_per_problem_step_ms,
    }


def build_horizon_list(start, end, step):
    if step <= 0:
        raise ValueError("horizon_step must be > 0")
    if end < start:
        raise ValueError("horizon_end must be >= horizon_start")
    horizons = list(range(start, end + 1, step))
    if horizons[-1] != end:
        horizons.append(end)
    return horizons


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate decoder-free ADMM-MPC metrics across horizon sweep")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to trained Koopman model")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device: cuda or cpu")

    parser.add_argument("--horizon_start", type=int, default=HORIZON_START, help="First horizon in sweep")
    parser.add_argument("--horizon_end", type=int, default=HORIZON_END, help="Last horizon in sweep")
    parser.add_argument("--horizon_step", type=int, default=HORIZON_STEP, help="Horizon stride in sweep")

    parser.add_argument("--num_problems", type=int, default=NUM_PROBLEMS, help="Number of start-goal problems")
    parser.add_argument("--max_mpc_steps", type=int, default=MAX_MPC_STEPS, help="Max closed-loop MPC steps")
    parser.add_argument("--goal_tol", type=float, default=GOAL_TOL, help="Goal tolerance in meters")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")

    parser.add_argument("--q", type=float, default=Q, help="Terminal state weight")
    parser.add_argument("--r", type=float, default=R, help="Control effort weight")
    parser.add_argument("--rho", type=float, default=RHO, help="ADMM rho")
    parser.add_argument("--max_admm_steps", type=int, default=MAX_ADMM_STEPS, help="ADMM iterations")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("Requested CUDA but unavailable. Falling back to CPU.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, checkpoint = load_model(args.model_path, device)
    print_checkpoint_configs(checkpoint)
    training_config = checkpoint.get("training_config", {})
    enable_normalization = training_config.get("enable_normalization", True)
    normalizer = Normalizer(checkpoint["normalization_stats"])
    A_np = model.A.weight.detach().cpu().numpy()
    B_np = model.B.weight.detach().cpu().numpy()

    starts, goals = sample_start_goal_batch(rng, args.num_problems)
    horizons = build_horizon_list(args.horizon_start, args.horizon_end, args.horizon_step)

    config = {
        "model_path": args.model_path,
        "device": device.type,
        "num_problems": args.num_problems,
        "max_mpc_steps": args.max_mpc_steps,
        "goal_tol": args.goal_tol,
        "horizons": horizons,
        "q": args.q,
        "r": args.r,
        "rho": args.rho,
        "max_admm_steps": args.max_admm_steps,
        "seed": args.seed,
        "enable_normalization": enable_normalization,
    }
    print_config_table(config, title="MPC Metrics Configuration")

    rows_summary = []
    rows_perf = []
    pbar = tqdm(horizons, desc="Horizon Sweep")
    for horizon in pbar:
        solver = GPUADMM_MPCSolver(
            A_np,
            B_np,
            Q_val=args.q,
            R_val=args.r,
            u_min=np.array([U1_MIN, U_TORQUE_MIN, U_TORQUE_MIN, U_TORQUE_MIN], dtype=np.float32),
            u_max=np.array([U1_MAX, U_TORQUE_MAX, U_TORQUE_MAX, U_TORQUE_MAX], dtype=np.float32),
            horizon=horizon,
            rho=args.rho,
            n_admm_steps=args.max_admm_steps,
            device=device,
        )

        metrics = evaluate_horizon(
            model=model,
            normalizer=normalizer,
            solver=solver,
            starts=starts,
            goals=goals,
            device=device,
            max_mpc_steps=args.max_mpc_steps,
            goal_tol=args.goal_tol,
            enable_normalization=enable_normalization,
        )

        pbar.set_postfix(goal=f"{metrics['goal_reach_pct']:.1f}%", mse=f"{metrics['mean_plan_pos_mse']:.3e}")

        rows_summary.append(
            (
                horizon,
                f"{metrics['success']}/{metrics['total']}",
                f"{metrics['goal_reach_pct']:.2f}",
                f"{metrics['mean_plan_state_mse']:.6e}",
                f"{metrics['mean_plan_pos_mse']:.6e}",
                f"{metrics['final_pos_mse']:.6e}",
            )
        )
        rows_perf.append(
            (
                horizon,
                f"{metrics['avg_solve_steps_success']:.2f}" if not np.isnan(metrics['avg_solve_steps_success']) else "N/A",
                f"{metrics['avg_solve_steps_all']:.2f}",
                f"{metrics['avg_batch_solve_ms']:.3f}",
                f"{metrics['avg_per_problem_step_ms']:.4f}",
            )
        )

    print_table(
        "MPC Horizon Sweep Metrics (Quality)",
        [
            "H",
            "Reached",
            "Goal Reach %",
            "Plan MSE (12D)",
            "Plan MSE (XYZ)",
            "Final Pos MSE",
        ],
        rows_summary,
    )
    print_table(
        "MPC Horizon Sweep Metrics (Solve Performance)",
        [
            "H",
            "Solve Steps (Succ)",
            "Solve Steps (All)",
            "Solve ms/Batch",
            "Solve ms/ProblemStep",
        ],
        rows_perf,
    )


if __name__ == "__main__":
    main()
