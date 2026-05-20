#!/usr/bin/env python3
"""Plan in Reacher pixel space with a conformalized local obstacle net and Torch SQP."""

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

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from reacher.plan import plan_ilqr_mpc as nominal_planner
from reacher.train.mlpdyn_train import LeWMReacherDataset

DEFAULT_TEST_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_5"
DEFAULT_OUT_DIR = "reacher/plan/ilqr_mpc_obs"
DEFAULT_OBSTACLE_DIR = "reacher/plan/obstacle_nets/episode_00520/step_0100/7128698f4baacd83"
DEVICE = "cuda"
HORIZON = 20
MAX_MPC_STEPS = 100
Q_TERMINAL = 10.0
Q_STAGE = 0.005
R_CONTROL = 0.1
VIDEO_FPS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--obstacle-cache-dir", type=Path, default=Path(DEFAULT_OBSTACLE_DIR))
    parser.add_argument("--obstacle-model-path", type=Path, default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--episode-idx", type=int, default=520)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--sqp-max-iters", type=int, default=8)
    parser.add_argument("--sqp-tol", type=float, default=1e-4)
    parser.add_argument("--qp-regularization", type=float, default=1e-5)
    parser.add_argument("--qp-active-set-iters", type=int, default=96)
    parser.add_argument("--trust-region", type=float, default=0.6)
    parser.add_argument("--line-search-min-alpha", type=float, default=0.05)
    parser.add_argument("--merit-obstacle-weight", type=float, default=1e3)
    parser.add_argument("--nonlinear-obstacle-tol", type=float, default=1e-4)
    parser.add_argument("--obstacle-margin-logit", type=float, default=0.05)
    parser.add_argument("--force-unconstrained-fallback", action="store_true", default=False)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def log_progress(message: str) -> None:
    print(f"[plan_ilqr_mpc_obs] {message}", flush=True)


def clamp_probability(prob: torch.Tensor | float, eps: float = 1e-6) -> torch.Tensor:
    tensor = prob if isinstance(prob, torch.Tensor) else torch.tensor(prob, dtype=torch.float64)
    return torch.clamp(tensor, eps, 1.0 - eps)


class ObstacleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Expected depth >= 1, got {depth}.")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ObstacleClassifierTorch:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        payload = torch.load(model_path, map_location=device, weights_only=False)
        input_dim = int(payload["input_dim"])
        hidden_dim = int(payload["hidden_dim"])
        depth = int(payload["depth"])
        dropout = float(payload["dropout"])
        self.net = ObstacleMLP(input_dim, hidden_dim, depth, dropout).to(device)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.net.requires_grad_(False)
        self.device = device
        self.embed_dim = input_dim
        self.feature_mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
        self.feature_std = torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device)
        tau = float(payload["conformal_threshold"])
        tau_tensor = clamp_probability(tau)
        self.threshold_prob = float(tau_tensor.item())
        self.threshold_logit = float(torch.log(tau_tensor / (1.0 - tau_tensor)).item())
        self.nominal_position = torch.as_tensor(payload["nominal_position"], dtype=torch.float32, device=device)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.feature_mean) / self.feature_std

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.ndim == 1:
            z = z.unsqueeze(0)
            squeeze = True
        logits = self.net(self.normalize(z))
        return logits[0] if squeeze else logits

    def probs(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(z))

    def linearize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_var = z.detach().clone().requires_grad_(True)
        logit = self.net(self.normalize(z_var.unsqueeze(0)))[0]
        grad = torch.autograd.grad(logit, z_var, retain_graph=False, create_graph=False)[0]
        return logit.detach(), grad.detach()


class ConstrainedSQPMPCSolver:
    def __init__(
        self,
        dynamics: nominal_planner.MarkovDynamicsTorch,
        obstacle_model: ObstacleClassifierTorch,
        *,
        horizon: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        sqp_max_iters: int,
        sqp_tol: float,
        qp_regularization: float,
        qp_active_set_iters: int,
        trust_region: float,
        u_min: np.ndarray,
        u_max: np.ndarray,
        line_search_min_alpha: float,
        merit_obstacle_weight: float,
        nonlinear_obstacle_tol: float,
        obstacle_margin_logit: float,
        device: torch.device,
    ) -> None:
        self.dynamics = dynamics
        self.obstacle_model = obstacle_model
        self.state_dim = dynamics.state_dim
        self.action_dim = dynamics.action_dim
        self.embed_dim = obstacle_model.embed_dim
        self.horizon = int(horizon)
        self.q_terminal = float(q_terminal)
        self.q_stage = float(q_stage)
        self.r_control = float(r_control)
        self.sqp_max_iters = int(sqp_max_iters)
        self.sqp_tol = float(sqp_tol)
        self.qp_regularization = float(qp_regularization)
        self.qp_active_set_iters = int(qp_active_set_iters)
        self.trust_region = float(trust_region)
        self.line_search_min_alpha = float(line_search_min_alpha)
        self.merit_obstacle_weight = float(merit_obstacle_weight)
        self.nonlinear_obstacle_tol = float(nonlinear_obstacle_tol)
        self.obstacle_logit_limit = float(obstacle_model.threshold_logit - obstacle_margin_logit)
        self.device = device
        self.line_search_alphas = (1.0, 0.5, 0.25, 0.1, 0.05)
        self.u_min = torch.as_tensor(u_min, dtype=torch.float32, device=device).reshape(-1)
        self.u_max = torch.as_tensor(u_max, dtype=torch.float32, device=device).reshape(-1)
        self.prev_u_guess = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32, device=device)

    def _make_initial_action_guess(self) -> torch.Tensor:
        if self.horizon <= 1:
            return self.prev_u_guess.clone()
        guess = torch.empty_like(self.prev_u_guess)
        guess[:-1] = self.prev_u_guess[1:]
        guess[-1] = self.prev_u_guess[-1]
        return torch.clamp(guess, min=self.u_min, max=self.u_max)

    def _rollout(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        x_traj = torch.empty((self.horizon + 1, self.state_dim), dtype=x0.dtype, device=self.device)
        x_traj[0] = x0
        x_curr = x0
        for step in range(self.horizon):
            x_curr = self.dynamics.step(x_curr, u_seq[step])
            x_traj[step + 1] = x_curr
        return x_traj

    def _trajectory_cost(self, x_traj: torch.Tensor, u_seq: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros((), dtype=x_traj.dtype, device=x_traj.device)
        for step in range(self.horizon):
            x_err = x_traj[step] - x_goal
            cost = cost + self.q_stage * torch.dot(x_err, x_err)
            cost = cost + self.r_control * torch.dot(u_seq[step], u_seq[step])
        terminal_err = x_traj[self.horizon] - x_goal
        cost = cost + self.q_terminal * torch.dot(terminal_err, terminal_err)
        return cost

    def _linearize_dynamics(self, x_traj: torch.Tensor, u_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_list = []
        b_list = []

        def dyn_cat(inp: torch.Tensor) -> torch.Tensor:
            x = inp[: self.state_dim]
            u = inp[self.state_dim :]
            return self.dynamics.step(x, u)

        for step in range(self.horizon):
            xu = torch.cat((x_traj[step], u_seq[step]), dim=0).detach().requires_grad_(True)
            jac = torch.autograd.functional.jacobian(dyn_cat, xu, vectorize=True)
            a_list.append(jac[:, : self.state_dim].detach())
            b_list.append(jac[:, self.state_dim :].detach())
        return torch.stack(a_list, dim=0), torch.stack(b_list, dim=0)

    def _condensed_state_maps(self, a_seq: torch.Tensor, b_seq: torch.Tensor) -> list[torch.Tensor]:
        dtype = torch.float64
        du_dim = self.horizon * self.action_dim
        maps = [torch.zeros((self.state_dim, du_dim), dtype=dtype, device=self.device)]
        for step in range(self.horizon):
            prev = maps[-1]
            next_map = a_seq[step].to(dtype) @ prev
            block = slice(step * self.action_dim, (step + 1) * self.action_dim)
            next_map[:, block] = next_map[:, block] + b_seq[step].to(dtype)
            maps.append(next_map)
        return maps

    def _build_qp(
        self,
        x_traj: torch.Tensor,
        u_seq: torch.Tensor,
        x_goal: torch.Tensor,
        a_seq: torch.Tensor,
        b_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        dtype = torch.float64
        du_dim = self.horizon * self.action_dim
        h_mat = torch.eye(du_dim, dtype=dtype, device=self.device) * self.qp_regularization
        g_vec = torch.zeros((du_dim,), dtype=dtype, device=self.device)
        state_maps = self._condensed_state_maps(a_seq, b_seq)
        qp_rows: list[torch.Tensor] = []
        qp_rhs: list[torch.Tensor] = []
        obstacle_stage_logits: list[float] = []

        for step in range(self.horizon):
            s_map = state_maps[step]
            x_err = (x_traj[step] - x_goal).to(dtype)
            h_mat = h_mat + 2.0 * self.q_stage * (s_map.T @ s_map)
            g_vec = g_vec + 2.0 * self.q_stage * (s_map.T @ x_err)

            block = slice(step * self.action_dim, (step + 1) * self.action_dim)
            e_mat = torch.zeros((self.action_dim, du_dim), dtype=dtype, device=self.device)
            e_mat[:, block] = torch.eye(self.action_dim, dtype=dtype, device=self.device)
            u_nom = u_seq[step].to(dtype)
            h_mat = h_mat + 2.0 * self.r_control * (e_mat.T @ e_mat)
            g_vec = g_vec + 2.0 * self.r_control * (e_mat.T @ u_nom)

            upper_rhs = (self.u_max - u_seq[step]).to(dtype).reshape(-1)
            lower_rhs = (u_seq[step] - self.u_min).to(dtype).reshape(-1)
            qp_rows.append(e_mat)
            qp_rhs.append(upper_rhs)
            qp_rows.append(-e_mat)
            qp_rhs.append(lower_rhs)
            qp_rows.append(e_mat)
            qp_rhs.append(torch.full((self.action_dim,), self.trust_region, dtype=dtype, device=self.device))
            qp_rows.append(-e_mat)
            qp_rhs.append(torch.full((self.action_dim,), self.trust_region, dtype=dtype, device=self.device))

        terminal_map = state_maps[self.horizon]
        terminal_err = (x_traj[self.horizon] - x_goal).to(dtype)
        h_mat = h_mat + 2.0 * self.q_terminal * (terminal_map.T @ terminal_map)
        g_vec = g_vec + 2.0 * self.q_terminal * (terminal_map.T @ terminal_err)

        for step in range(1, self.horizon + 1):
            z_nom = x_traj[step, : self.embed_dim]
            logit_nom, grad_z = self.obstacle_model.linearize(z_nom)
            full_grad = torch.zeros((self.state_dim,), dtype=torch.float32, device=self.device)
            full_grad[: self.embed_dim] = grad_z
            row = (full_grad.to(dtype) @ state_maps[step]).reshape(1, -1)
            rhs = torch.tensor(
                [self.obstacle_logit_limit - float(logit_nom.item())],
                dtype=dtype,
                device=self.device,
            )
            qp_rows.append(row)
            qp_rhs.append(rhs)
            obstacle_stage_logits.append(float(logit_nom.item()))

        if qp_rows:
            a_ineq = torch.cat(qp_rows, dim=0)
            b_ineq = torch.cat(qp_rhs, dim=0)
        else:
            a_ineq = torch.zeros((0, du_dim), dtype=dtype, device=self.device)
            b_ineq = torch.zeros((0,), dtype=dtype, device=self.device)
        h_mat = 0.5 * (h_mat + h_mat.T)
        info = {
            "num_constraints": int(a_ineq.shape[0]),
            "max_nominal_obstacle_logit": float(max(obstacle_stage_logits)) if obstacle_stage_logits else float("-inf"),
        }
        return h_mat, g_vec, a_ineq, b_ineq, info

    def _solve_qp_active_set(
        self,
        h_mat: torch.Tensor,
        g_vec: torch.Tensor,
        a_ineq: torch.Tensor,
        b_ineq: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, dict[str, Any]]:
        du_dim = int(g_vec.shape[0])
        h_eye = torch.eye(du_dim, dtype=h_mat.dtype, device=h_mat.device) * self.qp_regularization

        def solve_unconstrained() -> torch.Tensor:
            return torch.linalg.solve(h_mat + h_eye, -g_vec)

        if a_ineq.numel() == 0:
            sol = solve_unconstrained()
            return sol, True, {"active_count": 0, "max_violation": 0.0}

        x = solve_unconstrained()
        active: list[int] = []
        lam = torch.zeros((0,), dtype=h_mat.dtype, device=h_mat.device)
        for _ in range(self.qp_active_set_iters):
            if active:
                a_act = a_ineq[active]
                zeros = torch.zeros((len(active), len(active)), dtype=h_mat.dtype, device=h_mat.device)
                kkt = torch.cat(
                    (
                        torch.cat((h_mat + h_eye, a_act.T), dim=1),
                        torch.cat((a_act, zeros), dim=1),
                    ),
                    dim=0,
                )
                rhs = torch.cat((-g_vec, b_ineq[active]), dim=0)
                try:
                    sol = torch.linalg.solve(kkt, rhs)
                except RuntimeError:
                    return x, False, {"active_count": len(active), "max_violation": float("inf")}
                x = sol[:du_dim]
                lam = sol[du_dim:]
            else:
                x = solve_unconstrained()
                lam = torch.zeros((0,), dtype=h_mat.dtype, device=h_mat.device)

            violations = a_ineq @ x - b_ineq
            max_violation, max_idx = torch.max(violations, dim=0)
            if float(max_violation.item()) <= 1e-7:
                if active and float(torch.min(lam).item()) < -1e-7:
                    remove_idx = int(torch.argmin(lam).item())
                    del active[remove_idx]
                    continue
                return x, True, {"active_count": len(active), "max_violation": float(max_violation.item())}

            next_idx = int(max_idx.item())
            if next_idx not in active:
                active.append(next_idx)
                continue
            if active and float(torch.min(lam).item()) < -1e-7:
                remove_idx = int(torch.argmin(lam).item())
                del active[remove_idx]
                continue
            return x, False, {"active_count": len(active), "max_violation": float(max_violation.item())}

        violations = a_ineq @ x - b_ineq
        max_violation = float(torch.max(violations).item()) if violations.numel() > 0 else 0.0
        return x, False, {"active_count": len(active), "max_violation": max_violation}

    def _obstacle_violation(self, x_traj: torch.Tensor) -> tuple[float, float]:
        if x_traj.shape[0] <= 1:
            return float("-inf"), 0.0
        z = x_traj[1:, : self.embed_dim]
        logits = self.obstacle_model.logits(z)
        max_logit = float(torch.max(logits).item())
        violation = float(torch.max(logits - self.obstacle_logit_limit).item())
        return max_logit, violation

    def _evaluate_candidate(
        self,
        x0: torch.Tensor,
        x_goal: torch.Tensor,
        u_nominal: torch.Tensor,
        du: torch.Tensor,
        alpha: float,
    ) -> dict[str, Any]:
        u_new = u_nominal + alpha * du.reshape(self.horizon, self.action_dim).to(u_nominal.dtype)
        u_new = torch.clamp(u_new, min=self.u_min, max=self.u_max)
        x_new = self._rollout(x0, u_new)
        cost = float(self._trajectory_cost(x_new, u_new, x_goal).item())
        max_logit, obstacle_violation = self._obstacle_violation(x_new)
        merit = cost + self.merit_obstacle_weight * max(obstacle_violation, 0.0) ** 2
        return {
            "x_traj": x_new,
            "u_seq": u_new,
            "cost": cost,
            "merit": merit,
            "max_logit": max_logit,
            "obstacle_violation": obstacle_violation,
            "alpha": float(alpha),
        }

    def solve(self, x0_np: np.ndarray, x_goal_np: np.ndarray, *, allow_unconstrained_fallback: bool = False) -> dict[str, Any]:
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        x_goal = torch.as_tensor(x_goal_np, dtype=torch.float32, device=self.device)
        u_seq = self._make_initial_action_guess()

        nominal_planner.maybe_cuda_synchronize(self.device)
        t0 = time.perf_counter()

        current = self._evaluate_candidate(x0, x_goal, u_seq, torch.zeros_like(u_seq).reshape(-1), 0.0)
        qp_success = True
        sqp_iterations = 0
        last_qp_info: dict[str, Any] = {"active_count": 0, "max_violation": 0.0}

        for iteration in range(self.sqp_max_iters):
            sqp_iterations = iteration + 1
            x_traj = current["x_traj"]
            u_nominal = current["u_seq"]
            a_seq, b_seq = self._linearize_dynamics(x_traj, u_nominal)
            h_mat, g_vec, a_ineq, b_ineq, qp_meta = self._build_qp(x_traj, u_nominal, x_goal, a_seq, b_seq)
            du, solved, qp_info = self._solve_qp_active_set(h_mat, g_vec, a_ineq, b_ineq)
            last_qp_info = {**qp_info, **qp_meta}
            if not solved:
                qp_success = False
                if not allow_unconstrained_fallback:
                    break
                du = torch.linalg.solve(h_mat + torch.eye(h_mat.shape[0], dtype=h_mat.dtype, device=h_mat.device) * self.qp_regularization, -g_vec)

            step_norm = float(torch.max(torch.abs(du)).item()) if du.numel() > 0 else 0.0
            if step_norm <= self.sqp_tol:
                break

            accepted = None
            for alpha in self.line_search_alphas:
                if alpha < self.line_search_min_alpha:
                    continue
                candidate = self._evaluate_candidate(x0, x_goal, u_nominal, du, alpha)
                better_merit = candidate["merit"] < current["merit"] - 1e-9
                feasible_enough = candidate["obstacle_violation"] <= self.nonlinear_obstacle_tol
                if feasible_enough and better_merit:
                    accepted = candidate
                    break
                if accepted is None and better_merit:
                    accepted = candidate

            if accepted is None:
                break

            current = accepted
            if abs(current["merit"] - current["cost"]) <= self.sqp_tol and step_norm <= self.trust_region * 0.25:
                break

        self.prev_u_guess = current["u_seq"].detach().clone()
        nominal_planner.maybe_cuda_synchronize(self.device)
        solve_time = time.perf_counter() - t0
        return {
            "x_traj": current["x_traj"].detach().cpu().numpy().astype(np.float64),
            "u_seq": current["u_seq"].detach().cpu().numpy().astype(np.float64),
            "solve_time": float(solve_time),
            "iterations": int(sqp_iterations),
            "cost": float(current["cost"]),
            "merit": float(current["merit"]),
            "max_logit": float(current["max_logit"]),
            "obstacle_violation": float(current["obstacle_violation"]),
            "qp_success": bool(qp_success),
            "qp_info": last_qp_info,
        }


def resolve_obstacle_model_path(args: argparse.Namespace) -> Path:
    if args.obstacle_model_path is not None:
        return args.obstacle_model_path.expanduser().resolve()
    cache_dir = args.obstacle_cache_dir.expanduser().resolve()
    path = cache_dir / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Obstacle model not found: {path}")
    return path


def main() -> None:
    args = parse_args()
    log_progress("Loading world model, obstacle net, and planning episode.")
    device = nominal_planner.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    config = nominal_planner.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else nominal_planner.latest_object_checkpoint(model_dir).resolve()
    )
    world_model = nominal_planner.load_model(checkpoint_path, device)
    obstacle_model_path = resolve_obstacle_model_path(args)
    obstacle_model = ObstacleClassifierTorch(obstacle_model_path, device)

    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for the finetuned MLP model, got {history_size}.")
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    if obstacle_model.embed_dim != embed_dim:
        raise ValueError(
            f"Obstacle net embed_dim mismatch: planner model uses {embed_dim}, obstacle net uses {obstacle_model.embed_dim}."
        )

    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = LeWMReacherDataset(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    pixel_mean, pixel_std = nominal_planner.imagenet_pixel_stats(device)
    action_mean = train_stats_dataset.action_mean.astype(np.float32)
    action_std = train_stats_dataset.action_std.astype(np.float32)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
    valid_episodes = np.flatnonzero(ep_len >= 2)
    if valid_episodes.size == 0:
        raise ValueError("Need at least one test trajectory with 2 or more frames.")

    rng = np.random.default_rng(args.seed)
    if args.episode_idx is None:
        episode_idx = int(rng.choice(valid_episodes))
    else:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        if ep_len[episode_idx] < 2:
            raise ValueError(f"--episode-idx {episode_idx} must have at least 2 frames, got {ep_len[episode_idx]}.")

    episode = nominal_planner.load_dataset_episode(dataset_path, episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    run_name = f"{int(time.time())}_episode_{episode_idx:05d}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pixels = nominal_planner.preprocess_pixels(
        pixels_np,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    true_latents = nominal_planner.encode_frames(
        world_model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_emb = true_latents[0]
    goal_emb = true_latents[-1]
    start_state = nominal_planner.make_markov_state(start_emb)
    goal_state = nominal_planner.make_markov_state(goal_emb)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    nominal_planner.save_rgb_image(out_dir / "start_image.png", pixels_np[0])
    nominal_planner.save_rgb_image(out_dir / "goal_image.png", pixels_np[-1])

    env = nominal_planner.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    render_start = nominal_planner.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[0],
        qvel=qvel_np[0],
        height=height,
        width=width,
    )
    action_low_raw = np.asarray(env.action_space.low, dtype=np.float32)
    action_high_raw = np.asarray(env.action_space.high, dtype=np.float32)
    action_low_norm = (action_low_raw - action_mean) / action_std
    action_high_norm = (action_high_raw - action_mean) / action_std

    dynamics = nominal_planner.MarkovDynamicsTorch(world_model, markov_state_dim, action_dim, device)
    mpc_solver = ConstrainedSQPMPCSolver(
        dynamics,
        obstacle_model,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        sqp_max_iters=args.sqp_max_iters,
        sqp_tol=args.sqp_tol,
        qp_regularization=args.qp_regularization,
        qp_active_set_iters=args.qp_active_set_iters,
        trust_region=args.trust_region,
        u_min=action_low_norm,
        u_max=action_high_norm,
        line_search_min_alpha=args.line_search_min_alpha,
        merit_obstacle_weight=args.merit_obstacle_weight,
        nonlinear_obstacle_tol=args.nonlinear_obstacle_tol,
        obstacle_margin_logit=args.obstacle_margin_logit,
        device=device,
    )

    current_frame = render_start
    current_emb = nominal_planner.encode_single_frame(
        world_model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = nominal_planner.make_markov_state(current_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_obs = obs_np[-1].astype(np.float32)
    obs_dim = int(goal_obs.shape[0])
    current_obs = nominal_planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)

    rollout_frames = [current_frame.copy()]
    rollout_qpos = [np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy()]
    rollout_qvel = [np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy()]
    rollout_emb = [current_emb.detach().cpu().numpy().astype(np.float32)]
    executed_actions_raw: list[np.ndarray] = []
    executed_actions_norm: list[np.ndarray] = []
    latent_goal_distances = [float(torch.linalg.vector_norm(current_state - goal_state).item())]
    embedding_goal_distances = [float(torch.linalg.vector_norm(current_emb - goal_emb).item())]
    observation_goal_distances = [nominal_planner.compute_observation_goal_distance(current_obs, goal_obs)]
    solve_times_ms: list[float] = []
    sqp_iterations: list[int] = []
    sqp_costs: list[float] = []
    sqp_merits: list[float] = []
    max_predicted_logits: list[float] = []
    max_executed_logits = [float(obstacle_model.logits(current_emb).item())]
    obstacle_violations: list[float] = []
    stop_reason = "max_mpc_steps"

    log_progress("Starting constrained MPC rollout.")
    pbar = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in pbar:
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        solve_payload = mpc_solver.solve(
            current_state_np,
            goal_state_np,
            allow_unconstrained_fallback=args.force_unconstrained_fallback,
        )
        x_plan = solve_payload["x_traj"]
        u_plan = solve_payload["u_seq"]
        solve_times_ms.append(float(solve_payload["solve_time"]) * 1000.0)
        sqp_iterations.append(int(solve_payload["iterations"]))
        sqp_costs.append(float(solve_payload["cost"]))
        sqp_merits.append(float(solve_payload["merit"]))
        max_predicted_logits.append(float(solve_payload["max_logit"]))
        obstacle_violations.append(float(solve_payload["obstacle_violation"]))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = nominal_planner.normalized_to_raw_action(u0_norm, action_mean, action_std)
        executed_actions_norm.append(u0_norm.copy())
        executed_actions_raw.append(u0_raw.copy())

        _, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = nominal_planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = nominal_planner.encode_single_frame(
            world_model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = nominal_planner.make_markov_state(next_emb, current_emb)
        current_emb = next_emb

        rollout_frames.append(current_frame.copy())
        rollout_qpos.append(np.asarray(env._env.physics.data.qpos[:2], dtype=np.float32).copy())
        rollout_qvel.append(np.asarray(env._env.physics.data.qvel[:2], dtype=np.float32).copy())
        rollout_emb.append(current_emb.detach().cpu().numpy().astype(np.float32))
        latent_goal_distance = float(torch.linalg.vector_norm(current_state - goal_state).item())
        embedding_goal_distance = float(torch.linalg.vector_norm(current_emb - goal_emb).item())
        obs_goal_distance = nominal_planner.compute_observation_goal_distance(current_obs, goal_obs)
        executed_logit = float(obstacle_model.logits(current_emb).item())
        latent_goal_distances.append(latent_goal_distance)
        embedding_goal_distances.append(embedding_goal_distance)
        observation_goal_distances.append(obs_goal_distance)
        max_executed_logits.append(executed_logit)

        pbar.set_postfix(
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            sqp_it=f"{sqp_iterations[-1]}",
            obs_logit=f"{executed_logit:.2f}",
            obs_goal=f"{obs_goal_distance:.3f}",
        )

        reached_goal, _ = nominal_planner.goal_reached(current_obs, goal_obs)
        if reached_goal:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

    final_obs = nominal_planner.build_observation_from_env(env, obs_dim=obs_dim, goal_obs=goal_obs)
    video_path = str(nominal_planner.save_rollout_video(rollout_frames, out_dir, fps=args.video_fps)) if rollout_frames else None
    env.close()

    summary = {
        "episode_idx": int(episode_idx),
        "checkpoint_path": str(checkpoint_path),
        "obstacle_model_path": str(obstacle_model_path),
        "stop_reason": stop_reason,
        "num_steps_executed": int(len(executed_actions_raw)),
        "solve_times_ms": solve_times_ms,
        "sqp_iterations": sqp_iterations,
        "sqp_costs": sqp_costs,
        "sqp_merits": sqp_merits,
        "predicted_obstacle_max_logits": max_predicted_logits,
        "predicted_obstacle_violations": obstacle_violations,
        "executed_obstacle_logits": max_executed_logits,
        "latent_goal_distances": latent_goal_distances,
        "embedding_goal_distances": embedding_goal_distances,
        "observation_goal_distances": observation_goal_distances,
        "executed_actions_raw": [action.tolist() for action in executed_actions_raw],
        "executed_actions_norm": [action.tolist() for action in executed_actions_norm],
        "rollout_qpos": [q.tolist() for q in rollout_qpos],
        "rollout_qvel": [q.tolist() for q in rollout_qvel],
        "rollout_emb": [emb.tolist() for emb in rollout_emb],
        "goal_obs": goal_obs.tolist(),
        "final_obs": final_obs.tolist(),
        "video_path": video_path,
        "args": vars(args),
        "obstacle_logit_limit": float(mpc_solver.obstacle_logit_limit),
        "obstacle_prob_threshold": float(obstacle_model.threshold_prob),
        "obstacle_logit_threshold": float(obstacle_model.threshold_logit),
    }
    save_json(out_dir / "summary.json", summary)

    print(f"Saved to: {out_dir}")
    print(f"Stop reason: {stop_reason}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
