#!/usr/bin/env python3
"""Plan in Reacher latent space with a locally constrained obstacle-aware MPC variant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from reacher.plan import analyze_conformal_obstacle as obstacle_analysis
from reacher.plan import plan_ilqr_mpc as planner
from reacher.train.mlpdyn_train import LeWMReacherDataset

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_3"
DEFAULT_OUT_DIR = "reacher/plan/ilqr_mpc_obs"
DEFAULT_EPISODE_IDX = 829
DEFAULT_HORIZON = 20
DEFAULT_UNCONSTRAINED_MAX_MPC_STEPS = 100
DEFAULT_CONSTRAINED_MAX_MPC_STEPS = 150
DEFAULT_OBSTACLE_STEP = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--background-dataset-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-idx", type=int, default=DEFAULT_EPISODE_IDX)
    parser.add_argument("--obstacle-step", type=int, default=DEFAULT_OBSTACLE_STEP)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=None)
    parser.add_argument("--unconstrained-max-mpc-steps", type=int, default=DEFAULT_UNCONSTRAINED_MAX_MPC_STEPS)
    parser.add_argument("--constrained-max-mpc-steps", type=int, default=DEFAULT_CONSTRAINED_MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--q-terminal", type=float, default=10.0)
    parser.add_argument("--q-stage", type=float, default=0.005)
    parser.add_argument("--r-control", type=float, default=0.1)
    parser.add_argument("--ilqr-max-iters", type=int, default=35)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--joint1-range", type=float, default=0.35)
    parser.add_argument("--joint2-range", type=float, default=0.35)
    parser.add_argument("--set-pca-count", type=int, default=192)
    parser.add_argument("--set-norm-count", type=int, default=192)
    parser.add_argument("--set-cal-count", type=int, default=192)
    parser.add_argument("--set-test-count", type=int, default=512)
    parser.add_argument("--background-samples", type=int, default=4000)
    parser.add_argument("--context-background-count", type=int, default=512)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--conformal-eps", type=float, default=1e-12)
    parser.add_argument("--eigval-floor", type=float, default=1e-10)
    parser.add_argument("--membership-tol", type=float, default=1e-8)
    parser.add_argument("--overlay-sample-count", type=int, default=96)
    parser.add_argument("--overlay-perturb-alpha", type=float, default=0.05)
    parser.add_argument("--constraint-margin", type=float, default=0.08)
    parser.add_argument("--constraint-activation", type=float, default=1.35)
    parser.add_argument("--constraint-rho", type=float, default=1.0)
    parser.add_argument("--constraint-admm-iters", type=int, default=60)
    parser.add_argument("--constraint-admm-tol", type=float, default=1e-4)
    parser.add_argument("--qp-regularization", type=float, default=1e-5)
    parser.add_argument("--force-rerun-rollout", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_mpc_steps is not None:
        args.unconstrained_max_mpc_steps = int(args.max_mpc_steps)
        args.constrained_max_mpc_steps = int(args.max_mpc_steps)
    return args


def log_progress(message: str) -> None:
    print(f"[plan_ilqr_mpc_obs] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


class StagewiseConstraintADMM:
    def __init__(self, *, rho: float, max_iters: int, tol: float, regularization: float) -> None:
        self.rho = float(rho)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.regularization = float(regularization)

    def _solve_unconstrained_lqr(
        self,
        q_xx: np.ndarray,
        q_x: np.ndarray,
        r_uu: np.ndarray,
        r_u: np.ndarray,
        m_xu: np.ndarray,
        a_seq: np.ndarray,
        b_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        horizon = int(a_seq.shape[0])
        state_dim = int(a_seq.shape[1])
        action_dim = int(b_seq.shape[2])
        dx = np.zeros((horizon + 1, state_dim), dtype=np.float64)
        du = np.zeros((horizon, action_dim), dtype=np.float64)
        k_seq = np.zeros((horizon, action_dim), dtype=np.float64)
        kk_seq = np.zeros((horizon, action_dim, state_dim), dtype=np.float64)

        v_x = q_x[horizon].copy()
        v_xx = 0.5 * (q_xx[horizon] + q_xx[horizon].T)
        reg_eye = self.regularization * np.eye(action_dim, dtype=np.float64)

        for step in range(horizon - 1, -1, -1):
            a = a_seq[step]
            b = b_seq[step]
            qx = q_x[step]
            qu = r_u[step]
            qxx = 0.5 * (q_xx[step] + q_xx[step].T)
            quu = 0.5 * (r_uu[step] + r_uu[step].T)
            qxu = m_xu[step]
            qux = qxu.T

            local_qx = qx + a.T @ v_x
            local_qu = qu + b.T @ v_x
            local_qxx = qxx + a.T @ v_xx @ a
            local_quu = quu + b.T @ v_xx @ b + reg_eye
            local_qxu = qxu + a.T @ v_xx @ b
            local_qux = qux + b.T @ v_xx @ a
            local_quu = 0.5 * (local_quu + local_quu.T)

            try:
                k = -np.linalg.solve(local_quu, local_qu)
                kk = -np.linalg.solve(local_quu, local_qux)
            except np.linalg.LinAlgError:
                return dx, du, False

            k_seq[step] = k
            kk_seq[step] = kk
            v_x = local_qx + local_qxu @ k + kk.T @ local_qu + kk.T @ local_quu @ k
            v_xx = local_qxx + local_qxu @ kk + kk.T @ local_qux + kk.T @ local_quu @ kk
            v_xx = 0.5 * (v_xx + v_xx.T)

        for step in range(horizon):
            du[step] = k_seq[step] + kk_seq[step] @ dx[step]
            dx[step + 1] = a_seq[step] @ dx[step] + b_seq[step] @ du[step]

        return dx, du, True

    def solve(
        self,
        q_xx: np.ndarray,
        q_x: np.ndarray,
        r_uu: np.ndarray,
        r_u: np.ndarray,
        m_xu: np.ndarray,
        a_seq: np.ndarray,
        b_seq: np.ndarray,
        c_mat: np.ndarray,
        d_mat: np.ndarray,
        f_vec: np.ndarray,
        z_init: np.ndarray | None = None,
        y_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        horizon = int(a_seq.shape[0])
        n_stages = horizon + 1
        n_constraints = int(c_mat.shape[1])
        z = np.zeros((n_stages, n_constraints), dtype=np.float64) if z_init is None else z_init.copy()
        y = np.zeros_like(z) if y_init is None else y_init.copy()
        z = np.minimum(z, f_vec)

        dx = np.zeros((n_stages, q_x.shape[1]), dtype=np.float64)
        du = np.zeros((horizon, r_u.shape[1]), dtype=np.float64)
        primal = 0.0
        dual = 0.0
        converged = False
        backward_ok = True

        for admm_iter in range(self.max_iters):
            aug_q_xx = q_xx.copy()
            aug_q_x = q_x.copy()
            aug_r_uu = r_uu.copy()
            aug_r_u = r_u.copy()
            aug_m_xu = m_xu.copy()

            for step in range(n_stages):
                c_step = c_mat[step]
                d_step = d_mat[step]
                shift = self.rho * (y[step] - z[step])
                aug_q_xx[step] += self.rho * (c_step.T @ c_step)
                aug_q_x[step] += c_step.T @ shift
                if step < horizon:
                    aug_r_uu[step] += self.rho * (d_step.T @ d_step)
                    aug_r_u[step] += d_step.T @ shift
                    aug_m_xu[step] += self.rho * (c_step.T @ d_step)

            dx, du, backward_ok = self._solve_unconstrained_lqr(
                aug_q_xx,
                aug_q_x,
                aug_r_uu,
                aug_r_u,
                aug_m_xu,
                a_seq,
                b_seq,
            )
            if not backward_ok:
                break

            z_bar = np.einsum("kcx,kx->kc", c_mat, dx)
            z_bar[:-1] += np.einsum("kcu,ku->kc", d_mat[:-1], du)
            z_prev = z.copy()
            z = np.minimum(z_bar + y, f_vec)
            y = y + z_bar - z

            primal = float(np.linalg.norm(z_bar - z, ord=np.inf))
            dual = float(self.rho * np.linalg.norm(z - z_prev, ord=np.inf))
            if primal <= self.tol and dual <= self.tol:
                converged = True
                return dx, du, z, y, {
                    "iters": float(admm_iter + 1),
                    "primal_residual": primal,
                    "dual_residual": dual,
                    "converged": 1.0,
                    "backward_ok": 1.0,
                }

        return dx, du, z, y, {
            "iters": float(self.max_iters if backward_ok else 0.0),
            "primal_residual": primal,
            "dual_residual": dual,
            "converged": float(1.0 if converged else 0.0),
            "backward_ok": float(1.0 if backward_ok else 0.0),
        }


class ObstacleConstrainedILQRMPCSolver(planner.ILQRMPCSolver):
    def __init__(
        self,
        dynamics: planner.MarkovDynamicsTorch,
        *,
        embed_dim: int,
        obstacle_center: np.ndarray,
        obstacle_right_pinv: np.ndarray,
        obstacle_margin: float,
        obstacle_activation: float,
        constraint_rho: float,
        constraint_admm_iters: int,
        constraint_admm_tol: float,
        qp_regularization: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(dynamics, **kwargs)
        self.embed_dim = int(embed_dim)
        self.obstacle_center = np.asarray(obstacle_center, dtype=np.float64).copy()
        self.obstacle_right_pinv = np.asarray(obstacle_right_pinv, dtype=np.float64).copy()
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_activation = float(obstacle_activation)
        self.subproblem_solver = StagewiseConstraintADMM(
            rho=constraint_rho,
            max_iters=constraint_admm_iters,
            tol=constraint_admm_tol,
            regularization=qp_regularization,
        )
        self.last_solve_stats: dict[str, float] = {}
        self.prev_constraint_z = np.zeros((self.horizon + 1, 1), dtype=np.float64)
        self.prev_constraint_y = np.zeros((self.horizon + 1, 1), dtype=np.float64)

    def _embedding_coords(self, embedding: np.ndarray) -> np.ndarray:
        return (embedding - self.obstacle_center) @ self.obstacle_right_pinv

    def _constraint_value(self, embedding: np.ndarray, normal: np.ndarray, rhs: float) -> float:
        return float(normal @ embedding - rhs)

    def _state_inside_obstacle(self, embedding: np.ndarray) -> bool:
        coords = self._embedding_coords(embedding)
        return bool(np.max(np.abs(coords)) <= (1.0 + self.obstacle_margin))

    def _select_constraint_rows(self, x_traj: np.ndarray) -> list[tuple[np.ndarray, float] | None]:
        rows: list[tuple[np.ndarray, float] | None] = [None] * x_traj.shape[0]
        for step in range(1, x_traj.shape[0]):
            embedding = x_traj[step, : self.embed_dim]
            coords = self._embedding_coords(embedding)
            max_abs = float(np.max(np.abs(coords)))
            if max_abs >= self.obstacle_activation:
                continue
            face_idx = int(np.argmax(np.abs(coords)))
            face_sign = 1.0 if coords[face_idx] >= 0.0 else -1.0
            if abs(coords[face_idx]) <= 1e-8:
                face_sign = 1.0
            normal = face_sign * self.obstacle_right_pinv[:, face_idx]
            rhs = 1.0 + self.obstacle_margin + float(normal @ self.obstacle_center)
            rows[step] = (normal.astype(np.float64), rhs)
        return rows

    def _constraints_satisfied(
        self,
        x_traj: np.ndarray,
        rows: list[tuple[np.ndarray, float] | None],
        tol: float = 1e-7,
    ) -> bool:
        for step, row in enumerate(rows):
            if row is None:
                continue
            normal, rhs = row
            embedding = x_traj[step, : self.embed_dim]
            if self._constraint_value(embedding, normal, rhs) < -tol:
                return False
        return True

    def _build_local_model(
        self,
        x_traj: np.ndarray,
        u_seq: np.ndarray,
        x_goal: np.ndarray,
        rows: list[tuple[np.ndarray, float] | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_xx = np.zeros((self.horizon + 1, self.state_dim, self.state_dim), dtype=np.float64)
        q_x = np.zeros((self.horizon + 1, self.state_dim), dtype=np.float64)
        r_uu = np.zeros((self.horizon, self.action_dim, self.action_dim), dtype=np.float64)
        r_u = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        m_xu = np.zeros((self.horizon, self.state_dim, self.action_dim), dtype=np.float64)
        c_mat = np.zeros((self.horizon + 1, 1, self.state_dim), dtype=np.float64)
        d_mat = np.zeros((self.horizon + 1, 1, self.action_dim), dtype=np.float64)
        f_vec = np.full((self.horizon + 1, 1), 1e12, dtype=np.float64)

        q_stage = 2.0 * self.q_stage
        q_terminal = 2.0 * self.q_terminal
        r_control = 2.0 * self.r_control
        eye_x = np.eye(self.state_dim, dtype=np.float64)
        eye_u = np.eye(self.action_dim, dtype=np.float64)

        for step in range(self.horizon):
            q_xx[step] = q_stage * eye_x
            q_x[step] = q_stage * (x_traj[step] - x_goal)
            r_uu[step] = r_control * eye_u
            r_u[step] = r_control * u_seq[step]

        q_xx[self.horizon] = q_terminal * eye_x
        q_x[self.horizon] = q_terminal * (x_traj[self.horizon] - x_goal)

        for step, row in enumerate(rows):
            if row is None:
                continue
            normal, rhs = row
            c_mat[step, 0, : self.embed_dim] = -normal
            f_vec[step, 0] = float(self._constraint_value(x_traj[step, : self.embed_dim], normal, rhs))

        return q_xx, q_x, r_uu, r_u, m_xu, c_mat, d_mat, f_vec

    def solve(self, x0_np: np.ndarray, x_goal_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, float]:
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.tensor(x_goal_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()

        planner.maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        x_traj = self._rollout(x0, u_seq)
        current_cost = float(self._trajectory_cost(x_traj, u_seq, x_goal).item())
        iterations = 0
        current_rows = self._select_constraint_rows(x_traj.detach().cpu().numpy().astype(np.float64))
        feasible_initial = self._constraints_satisfied(
            x_traj.detach().cpu().numpy().astype(np.float64),
            current_rows,
        )
        accepted_constraints = float(sum(row is not None for row in current_rows))
        admm_iters = 0.0
        admm_primal = 0.0
        admm_dual = 0.0
        admm_converged = 0.0
        accepted = False
        stop_reason = "max_iters"

        for iteration in range(self.max_iters):
            iterations = iteration + 1
            a_t, b_t = self._linearize_dynamics(x_traj, u_seq)
            x_np = x_traj.detach().cpu().numpy().astype(np.float64)
            u_np = u_seq.detach().cpu().numpy().astype(np.float64)
            a_np = a_t.detach().cpu().numpy().astype(np.float64)
            b_np = b_t.detach().cpu().numpy().astype(np.float64)
            current_rows = self._select_constraint_rows(x_np)
            q_xx, q_x, r_uu, r_u, m_xu, c_mat, d_mat, f_vec = self._build_local_model(
                x_np,
                u_np,
                x_goal_np,
                current_rows,
            )
            dx_seq, du_seq_np, z_next, y_next, qp_stats = self.subproblem_solver.solve(
                q_xx,
                q_x,
                r_uu,
                r_u,
                m_xu,
                a_np,
                b_np,
                c_mat,
                d_mat,
                f_vec,
                z_init=self.prev_constraint_z,
                y_init=self.prev_constraint_y,
            )
            if qp_stats["backward_ok"] < 0.5:
                stop_reason = "backward_pass_failed"
                break

            du_seq = torch.tensor(du_seq_np, dtype=torch.float32, device=self.device)

            best_candidate: tuple[torch.Tensor, torch.Tensor, float, float] | None = None
            for alpha in self.line_search_alphas:
                u_new = u_seq + alpha * du_seq
                x_new = self._rollout(x0, u_new)
                x_new_np = x_new.detach().cpu().numpy().astype(np.float64)
                if not self._constraints_satisfied(x_new_np, current_rows):
                    continue
                new_cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
                if np.isfinite(new_cost) and new_cost < current_cost:
                    best_candidate = (x_new, u_new, new_cost, alpha)
                    break

            if best_candidate is None:
                stop_reason = "no_feasible_step"
                break

            accepted = True
            x_traj, u_seq, current_cost, alpha = best_candidate
            self.prev_constraint_z = z_next
            self.prev_constraint_y = y_next
            accepted_constraints = float(sum(row is not None for row in current_rows))
            admm_iters = qp_stats["iters"]
            admm_primal = qp_stats["primal_residual"]
            admm_dual = qp_stats["dual_residual"]
            admm_converged = qp_stats["converged"]
            max_du = float(np.max(np.abs(alpha * du_seq_np)))
            step_norm = float(np.max(np.abs(alpha * dx_seq)))
            if max(max_du, step_norm) <= self.tol:
                stop_reason = "converged"
                break

        self.prev_u_guess = u_seq.detach().clone()
        planner.maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0
        final_cost = float(self._trajectory_cost(x_traj, u_seq, x_goal).item())
        final_x_np = x_traj.detach().cpu().numpy().astype(np.float64)
        final_rows = self._select_constraint_rows(final_x_np)
        feasible_final = self._constraints_satisfied(final_x_np, final_rows)
        self.last_solve_stats = {
            "cost": float(current_cost),
            "active_constraints": float(accepted_constraints),
            "admm_iters": float(admm_iters),
            "admm_primal_residual": float(admm_primal),
            "admm_dual_residual": float(admm_dual),
            "admm_converged": float(admm_converged),
            "accepted": float(1.0 if accepted else 0.0),
            "feasible_initial": float(1.0 if feasible_initial else 0.0),
            "feasible_final": float(1.0 if feasible_final else 0.0),
            "stop_reason": stop_reason,
        }
        return (
            x_traj.detach().cpu().numpy().astype(np.float64),
            u_seq.detach().cpu().numpy().astype(np.float64),
            solve_time,
            iterations,
            final_cost,
        )


def run_constrained_rollout(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    rollout: dict[str, Any],
    zonotope: dict[str, np.ndarray | float],
    device: torch.device,
    frame_batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 24))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for this planner, got {history_size}.")

    train_dataset_path = Path(str(config.get("dataset_path", args.dataset_path))).expanduser().resolve()
    train_stats_dataset = LeWMReacherDataset(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

    goal_obs = np.asarray(rollout["goal_obs"], dtype=np.float32)
    obs_dim = int(goal_obs.shape[0])
    width = int(rollout["width"])
    height = int(rollout["height"])
    episode_seed = int(rollout["episode_seed"])
    time_limit = float(rollout["time_limit"])
    physics_freq_hz = float(rollout["physics_freq_hz"])
    qpos0 = np.asarray(rollout["dataset_qpos"][0], dtype=np.float32)
    qvel0 = np.asarray(rollout["dataset_qvel"][0], dtype=np.float32)
    goal_embedding = np.asarray(rollout["dataset_emb"][-1], dtype=np.float32)
    goal_state_np = np.concatenate(
        (goal_embedding, np.zeros_like(goal_embedding)),
        axis=0,
    ).astype(np.float64)
    goal_state_t = torch.tensor(goal_state_np, dtype=torch.float32, device=device)

    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)
    env = planner.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    current_frame = planner.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos0,
        qvel=qvel0,
        height=height,
        width=width,
    )
    dynamics = planner.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)
    mpc_solver = ObstacleConstrainedILQRMPCSolver(
        dynamics,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        device=device,
        embed_dim=embed_dim,
        obstacle_center=np.asarray(zonotope["center"], dtype=np.float64),
        obstacle_right_pinv=np.asarray(zonotope["right_pinv"], dtype=np.float64),
        obstacle_margin=args.constraint_margin,
        obstacle_activation=args.constraint_activation,
        constraint_rho=args.constraint_rho,
        constraint_admm_iters=args.constraint_admm_iters,
        constraint_admm_tol=args.constraint_admm_tol,
        qp_regularization=args.qp_regularization,
    )

    current_emb = planner.encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = planner.make_markov_state(current_emb)
    current_obs = planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)

    rollout_frames = [current_frame.copy()]
    rollout_qpos = [np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy()]
    rollout_qvel = [np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy()]
    rollout_emb = [current_emb.detach().cpu().numpy().astype(np.float32)]
    rollout_markov = [current_state.detach().cpu().numpy().astype(np.float32)]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    observation_goal_distances = [planner.compute_observation_goal_distance(current_obs, goal_obs)]
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state_t).item())]
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    plan_costs: list[float] = []
    active_constraints: list[float] = []
    admm_iterations: list[float] = []
    solver_feasible: list[float] = []
    stop_reason = "max_mpc_steps"

    for _ in tqdm(range(args.constrained_max_mpc_steps), desc="Obstacle-aware MPC rollout"):
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        x_plan, u_plan, solve_time, n_iters, plan_cost = mpc_solver.solve(current_state_np, goal_state_np)
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))
        plan_costs.append(float(plan_cost))
        active_constraints.append(float(mpc_solver.last_solve_stats.get("active_constraints", 0.0)))
        admm_iterations.append(float(mpc_solver.last_solve_stats.get("admm_iters", 0.0)))
        solver_feasible.append(float(mpc_solver.last_solve_stats.get("feasible_final", 0.0)))

        if float(mpc_solver.last_solve_stats.get("accepted", 0.0)) < 0.5:
            stop_reason = "constraint_infeasible"
            break

        next_embedding_plan = np.asarray(x_plan[1, :embed_dim], dtype=np.float64)
        if mpc_solver._state_inside_obstacle(next_embedding_plan):
            stop_reason = "predicted_obstacle_violation"
            break

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = planner.normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = planner.encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = planner.make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        rollout_qpos.append(np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy())
        rollout_qvel.append(np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy())
        rollout_emb.append(current_emb.detach().cpu().numpy().astype(np.float32))
        rollout_markov.append(current_state.detach().cpu().numpy().astype(np.float32))
        observation_goal_distances.append(planner.compute_observation_goal_distance(current_obs, goal_obs))
        latent_goal_distances.append(float(torch.linalg.vector_norm(current_state - goal_state_t).item()))

        if mpc_solver._state_inside_obstacle(np.asarray(rollout_emb[-1][:embed_dim], dtype=np.float64)):
            stop_reason = "observed_obstacle_violation"
            break

        reached_goal, _ = planner.goal_reached(current_obs, goal_obs)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    env.close()
    return {
        "rollout_frames": np.asarray(rollout_frames, dtype=np.uint8),
        "rollout_qpos": np.asarray(rollout_qpos, dtype=np.float32),
        "rollout_qvel": np.asarray(rollout_qvel, dtype=np.float32),
        "rollout_emb": np.asarray(rollout_emb, dtype=np.float32),
        "rollout_markov": np.asarray(rollout_markov, dtype=np.float32),
        "executed_actions_raw": np.asarray(executed_actions_raw, dtype=np.float32),
        "executed_actions_norm": np.asarray(executed_actions_norm, dtype=np.float32),
        "observation_goal_distances": np.asarray(observation_goal_distances, dtype=np.float32),
        "latent_goal_distances": np.asarray(latent_goal_distances, dtype=np.float32),
        "solve_times_ms": np.asarray(solve_times_ms, dtype=np.float32),
        "ilqr_iterations": np.asarray(ilqr_iterations, dtype=np.int32),
        "plan_costs": np.asarray(plan_costs, dtype=np.float32),
        "active_constraints": np.asarray(active_constraints, dtype=np.float32),
        "admm_iterations": np.asarray(admm_iterations, dtype=np.float32),
        "solver_feasible": np.asarray(solver_feasible, dtype=np.float32),
        "stop_reason": stop_reason,
    }


def make_comparison_plot(
    out_path: Path,
    *,
    title: str,
    background: np.ndarray,
    construction_samples: np.ndarray,
    test_inside_samples: np.ndarray,
    test_outside_samples: np.ndarray,
    dataset_traj: np.ndarray,
    nominal_traj: np.ndarray,
    constrained_traj: np.ndarray,
    start_point: np.ndarray,
    dataset_goal_point: np.ndarray,
    nominal_goal_point: np.ndarray,
    constrained_goal_point: np.ndarray,
    nominal_point: np.ndarray,
    zonotope_boundary: np.ndarray,
    explained_ratio: np.ndarray,
    focus_main_view: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.scatter(background[:, 0], background[:, 1], s=3, alpha=0.18, color="0.5", label="background", zorder=1)
    if construction_samples.shape[0] > 0:
        ax.scatter(construction_samples[:, 0], construction_samples[:, 1], s=10, alpha=0.3, color="tab:red", label="obstacle samples", zorder=2)
    if test_inside_samples.shape[0] > 0:
        ax.scatter(test_inside_samples[:, 0], test_inside_samples[:, 1], s=16, alpha=0.65, color="tab:orange", label="held-out inside", zorder=3)
    if test_outside_samples.shape[0] > 0:
        ax.scatter(test_outside_samples[:, 0], test_outside_samples[:, 1], s=24, alpha=0.95, marker="x", linewidths=1.2, color="black", label="held-out outside", zorder=4)
    if zonotope_boundary.shape[0] >= 2:
        ax.plot(zonotope_boundary[:, 0], zonotope_boundary[:, 1], color="tab:orange", linewidth=2.0, label="zonotope boundary", zorder=8)
    ax.plot(dataset_traj[:, 0], dataset_traj[:, 1], color="tab:blue", linewidth=1.5, label="dataset trajectory", zorder=5)
    ax.plot(nominal_traj[:, 0], nominal_traj[:, 1], color="tab:green", linewidth=1.7, label="executed MPC rollout", zorder=6)
    ax.plot(constrained_traj[:, 0], constrained_traj[:, 1], color="tab:purple", linewidth=1.8, linestyle="--", label="constrained MPC rollout", zorder=7)
    ax.scatter(start_point[0], start_point[1], s=90, marker="o", color="black", label="start", zorder=9)
    ax.scatter(dataset_goal_point[0], dataset_goal_point[1], s=115, marker="*", color="gold", edgecolor="black", linewidth=0.8, label="dataset goal", zorder=9)
    ax.scatter(nominal_goal_point[0], nominal_goal_point[1], s=90, marker="P", color="limegreen", edgecolor="black", linewidth=0.8, label="nominal goal", zorder=9)
    ax.scatter(constrained_goal_point[0], constrained_goal_point[1], s=90, marker="D", color="tab:purple", edgecolor="black", linewidth=0.8, label="constrained goal", zorder=9)
    ax.scatter(nominal_point[0], nominal_point[1], s=80, marker="X", color="tab:red", edgecolor="black", linewidth=0.8, label="obstacle center", zorder=9)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({100.0 * explained_ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100.0 * explained_ratio[1]:.1f}%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if focus_main_view:
        focus_stack = np.concatenate(
            [
                construction_samples,
                test_inside_samples,
                test_outside_samples,
                dataset_traj,
                nominal_traj,
                constrained_traj,
                start_point[None, :],
                dataset_goal_point[None, :],
                nominal_goal_point[None, :],
                constrained_goal_point[None, :],
                nominal_point[None, :],
            ]
            + ([zonotope_boundary] if zonotope_boundary.shape[0] >= 2 else []),
            axis=0,
        )
        mins = focus_stack.min(axis=0)
        maxs = focus_stack.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.4 * spans
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_progress(f"Starting obstacle-aware planning for episode {args.episode_idx}, requested obstacle step {args.obstacle_step}.")
    rng = np.random.default_rng(args.seed)
    device = planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else planner.latest_object_checkpoint(model_dir).resolve()
    )
    model = planner.load_model(checkpoint_path, device)
    background_dataset_path = (
        args.background_dataset_path.expanduser().resolve()
        if args.background_dataset_path is not None
        else Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    )

    nominal_cache_path = obstacle_analysis.infer_rollout_cache_path(out_root, checkpoint_path, args.episode_idx, args)
    nominal_args = argparse.Namespace(**vars(args))
    nominal_args.max_mpc_steps = int(args.unconstrained_max_mpc_steps)
    nominal = obstacle_analysis.run_or_load_rollout(
        planner=planner,
        dataset_cls=LeWMReacherDataset,
        cache_path=nominal_cache_path,
        force_rerun=args.force_rerun_rollout,
        model=model,
        config=config,
        dataset_path=dataset_path,
        episode_idx=args.episode_idx,
        device=device,
        frame_batch_size=args.frame_batch_size,
        args=nominal_args,
    )
    nominal_qpos = np.asarray(nominal["rollout_qpos"], dtype=np.float64)
    nominal_emb = np.asarray(nominal["rollout_emb"], dtype=np.float64)
    dataset_emb = np.asarray(nominal["dataset_emb"], dtype=np.float64)
    obstacle_step = int(args.obstacle_step)
    if obstacle_step == -1:
        obstacle_step = int(nominal_qpos.shape[0] - 1)
    if obstacle_step < 0 or obstacle_step >= nominal_qpos.shape[0]:
        raise ValueError(f"--obstacle-step must be in [0, {nominal_qpos.shape[0] - 1}], got {args.obstacle_step}.")

    episode_dir = out_root / f"episode_{args.episode_idx:05d}" / f"step_{obstacle_step:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    log_progress("Loaded nominal rollout cache.")

    img_size = int(config.get("img_size", 224))
    embed_dim = int(config.get("embed_dim", 24))
    pixel_mean, pixel_std = planner.imagenet_pixel_stats(device)

    env = planner.make_render_env(
        seed=int(nominal["episode_seed"]),
        time_limit=float(nominal["time_limit"]),
        width=int(nominal["width"]),
        height=int(nominal["height"]),
        physics_freq_hz=float(nominal["physics_freq_hz"]),
    )
    lower, upper = obstacle_analysis.joint_limits_from_env(env)
    center_qpos = nominal_qpos[obstacle_step]
    joint_ranges = np.array([args.joint1_range, args.joint2_range], dtype=np.float64)
    total_obstacle = args.set_pca_count + args.set_norm_count + args.set_cal_count + args.set_test_count
    sampled_qpos = obstacle_analysis.sample_local_perturbations(
        center_qpos,
        lower,
        upper,
        rng,
        joint_ranges=joint_ranges,
        count=total_obstacle,
    )
    perturb_frames = obstacle_analysis.render_qpos_batch(
        planner,
        env,
        int(nominal["episode_seed"]),
        sampled_qpos,
        height=int(nominal["height"]),
        width=int(nominal["width"]),
    )
    env.close()

    perturb_pixels = planner.preprocess_pixels(
        perturb_frames,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    perturb_emb = planner.encode_frames(
        model,
        perturb_pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    ).detach().cpu().numpy().astype(np.float64)
    position_samples = perturb_emb[:, :embed_dim]
    permutation = rng.permutation(total_obstacle)
    position_samples = position_samples[permutation]
    sampled_qpos = sampled_qpos[permutation]

    pca_end = args.set_pca_count
    norm_end = pca_end + args.set_norm_count
    cal_end = norm_end + args.set_cal_count
    pca_samples = position_samples[:pca_end]
    norm_samples = position_samples[pca_end:norm_end]
    cal_samples = position_samples[norm_end:cal_end]
    test_samples = position_samples[cal_end:]
    zonotope = obstacle_analysis.build_conformal_zonotope(
        pca_samples,
        norm_samples,
        cal_samples,
        delta=args.delta,
        eps=args.conformal_eps,
        eigval_floor=args.eigval_floor,
    )
    nominal_position = nominal_emb[obstacle_step, :embed_dim]
    heldout_inside = obstacle_analysis.zonotope_contains(test_samples, zonotope, tol=args.membership_tol)
    empirical_coverage = float(np.mean(heldout_inside))
    log_progress(f"Built obstacle zonotope with held-out coverage {empirical_coverage:.4f}.")

    log_progress("Running constrained rollout.")
    constrained = run_constrained_rollout(
        model=model,
        config=config,
        rollout=nominal,
        zonotope=zonotope,
        device=device,
        frame_batch_size=args.frame_batch_size,
        args=args,
    )

    log_progress("Preparing overlays, videos, and comparison plots.")
    nominal_frames = np.asarray(nominal["rollout_frames"], dtype=np.uint8)
    constrained_frames = np.asarray(constrained["rollout_frames"], dtype=np.uint8)
    nominal_video_dir = episode_dir / "nominal_rollout"
    constrained_video_dir = episode_dir / "constrained_rollout"
    nominal_video_dir.mkdir(parents=True, exist_ok=True)
    constrained_video_dir.mkdir(parents=True, exist_ok=True)
    planner.save_rollout_video([frame.copy() for frame in nominal_frames], nominal_video_dir, fps=args.video_fps)
    planner.save_rollout_video([frame.copy() for frame in constrained_frames], constrained_video_dir, fps=args.video_fps)
    planner.save_rgb_image(episode_dir / "nominal_goal_image.png", nominal_frames[-1])
    planner.save_rgb_image(episode_dir / "constrained_goal_image.png", constrained_frames[-1])

    overlay_count = min(int(args.overlay_sample_count), int(sampled_qpos.shape[0]))
    overlay_indices = np.linspace(0, sampled_qpos.shape[0] - 1, num=overlay_count, dtype=np.int64)
    overlay_qpos_batch = np.concatenate((center_qpos[None, :], sampled_qpos[overlay_indices]), axis=0)
    overlay_env = planner.make_render_env(
        seed=int(nominal["episode_seed"]),
        time_limit=float(nominal["time_limit"]),
        width=int(nominal["width"]),
        height=int(nominal["height"]),
        physics_freq_hz=float(nominal["physics_freq_hz"]),
    )
    overlay_frames, overlay_masks = obstacle_analysis.render_masked_qpos_batch(
        overlay_env,
        int(nominal["episode_seed"]),
        overlay_qpos_batch,
        height=int(nominal["height"]),
        width=int(nominal["width"]),
    )
    overlay_env.close()
    obstacle_overlay = obstacle_analysis.make_obstacle_overlay_image(
        nominal_frame=overlay_frames[0],
        nominal_mask=overlay_masks[0],
        perturb_frames=overlay_frames[1:],
        perturb_masks=overlay_masks[1:],
        perturb_alpha=float(args.overlay_perturb_alpha),
    )
    planner.save_rgb_image(episode_dir / "obstacle_overlay_all.png", obstacle_overlay)

    background_rows = obstacle_analysis.sample_background_rows(background_dataset_path, rng, args.background_samples)
    background_emb = obstacle_analysis.encode_dataset_rows(
        planner,
        model,
        background_dataset_path,
        background_rows,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        frame_batch_size=args.frame_batch_size,
    )
    background_pos = background_emb[:, :embed_dim]
    dataset_pos = dataset_emb[:, :embed_dim]
    nominal_pos = nominal_emb[:, :embed_dim]
    constrained_pos = np.asarray(constrained["rollout_emb"], dtype=np.float64)[:, :embed_dim]
    global_pca = obstacle_analysis.fit_pca_2d(background_pos)
    local_pca = obstacle_analysis.fit_pca_2d(position_samples)
    local_boundary = obstacle_analysis.zonotope_boundary_2d(*obstacle_analysis.project_zonotope(zonotope, local_pca))
    heldout_inside_samples = test_samples[heldout_inside]
    heldout_outside_samples = test_samples[~heldout_inside]

    make_comparison_plot(
        episode_dir / "pca_global.png",
        title=f"Global PCA comparison | episode {args.episode_idx} step {obstacle_step}",
        background=obstacle_analysis.project_points(background_pos, global_pca),
        construction_samples=obstacle_analysis.project_points(position_samples, global_pca),
        test_inside_samples=np.zeros((0, 2), dtype=np.float64),
        test_outside_samples=np.zeros((0, 2), dtype=np.float64),
        dataset_traj=obstacle_analysis.project_points(dataset_pos, global_pca),
        nominal_traj=obstacle_analysis.project_points(nominal_pos, global_pca),
        constrained_traj=obstacle_analysis.project_points(constrained_pos, global_pca),
        start_point=obstacle_analysis.project_points(dataset_pos[[0]], global_pca)[0],
        dataset_goal_point=obstacle_analysis.project_points(dataset_pos[[-1]], global_pca)[0],
        nominal_goal_point=obstacle_analysis.project_points(nominal_pos[[-1]], global_pca)[0],
        constrained_goal_point=obstacle_analysis.project_points(constrained_pos[[-1]], global_pca)[0],
        nominal_point=obstacle_analysis.project_points(nominal_position[None, :], global_pca)[0],
        zonotope_boundary=np.zeros((0, 2), dtype=np.float64),
        explained_ratio=np.asarray(global_pca["explained_variance_ratio"], dtype=np.float64),
        focus_main_view=True,
    )
    make_comparison_plot(
        episode_dir / "pca_local.png",
        title=f"Local PCA comparison | episode {args.episode_idx} step {obstacle_step}",
        background=obstacle_analysis.project_points(background_pos, local_pca),
        construction_samples=obstacle_analysis.project_points(position_samples[:cal_end], local_pca),
        test_inside_samples=obstacle_analysis.project_points(heldout_inside_samples, local_pca),
        test_outside_samples=obstacle_analysis.project_points(heldout_outside_samples, local_pca),
        dataset_traj=obstacle_analysis.project_points(dataset_pos, local_pca),
        nominal_traj=obstacle_analysis.project_points(nominal_pos, local_pca),
        constrained_traj=obstacle_analysis.project_points(constrained_pos, local_pca),
        start_point=obstacle_analysis.project_points(dataset_pos[[0]], local_pca)[0],
        dataset_goal_point=obstacle_analysis.project_points(dataset_pos[[-1]], local_pca)[0],
        nominal_goal_point=obstacle_analysis.project_points(nominal_pos[[-1]], local_pca)[0],
        constrained_goal_point=obstacle_analysis.project_points(constrained_pos[[-1]], local_pca)[0],
        nominal_point=obstacle_analysis.project_points(nominal_position[None, :], local_pca)[0],
        zonotope_boundary=local_boundary,
        explained_ratio=np.asarray(local_pca["explained_variance_ratio"], dtype=np.float64),
        focus_main_view=False,
    )

    obstacle_coords = obstacle_analysis.zonotope_coords(position_samples, zonotope)
    obstacle_analysis.make_zonotope_coords_plot(
        episode_dir / "zonotope_coords.png",
        construction_coords=obstacle_coords[:cal_end, :2],
        test_coords=obstacle_coords[cal_end:, :2],
        test_inside_mask=heldout_inside,
        nominal_coords=obstacle_analysis.zonotope_coords(nominal_position[None, :], zonotope)[0, :2],
    )

    torch.save(
        {
            "zonotope": zonotope,
            "global_pca": global_pca,
            "local_pca": local_pca,
            "nominal_rollout_cache_path": str(nominal_cache_path),
            "background_rows": background_rows.astype(np.int64),
            "sampled_qpos": sampled_qpos.astype(np.float64),
            "obstacle_position_samples": position_samples.astype(np.float64),
            "nominal_rollout_emb": nominal_pos.astype(np.float64),
            "constrained_rollout_emb": constrained_pos.astype(np.float64),
            "empirical_coverage_mask": heldout_inside.astype(bool),
        },
        episode_dir / "analysis.pt",
    )
    save_json(
        episode_dir / "summary.json",
        {
            "episode_idx": int(args.episode_idx),
            "obstacle_step": int(obstacle_step),
            "nominal_rollout_cache_path": str(nominal_cache_path),
            "checkpoint_path": str(checkpoint_path),
            "background_dataset_path": str(background_dataset_path),
            "constraint_margin": float(args.constraint_margin),
            "constraint_activation": float(args.constraint_activation),
            "unconstrained_max_mpc_steps": int(args.unconstrained_max_mpc_steps),
            "constrained_max_mpc_steps": int(args.constrained_max_mpc_steps),
            "empirical_coverage": empirical_coverage,
            "nominal_stop_reason": str(nominal["stop_reason"]),
            "constrained_stop_reason": str(constrained["stop_reason"]),
            "nominal_final_obs_goal_distance": float(np.asarray(nominal["observation_goal_distances"], dtype=np.float64)[-1]),
            "constrained_final_obs_goal_distance": float(np.asarray(constrained["observation_goal_distances"], dtype=np.float64)[-1]),
            "constrained_mean_active_constraints": float(np.mean(np.asarray(constrained["active_constraints"], dtype=np.float64))) if np.asarray(constrained["active_constraints"]).size > 0 else 0.0,
            "constrained_mean_solver_feasible": float(np.mean(np.asarray(constrained["solver_feasible"], dtype=np.float64))) if np.asarray(constrained["solver_feasible"]).size > 0 else 0.0,
        },
    )

    log_progress("Obstacle-aware planning complete.")
    print(f"Output dir: {episode_dir}")


if __name__ == "__main__":
    main()
