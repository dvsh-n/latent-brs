#!/usr/bin/env python3
"""Benchmark Reacher iLQR MPC with the same replay protocol as LeWM eval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from reacher.plan import plan_ilqr_mpc as base

DEFAULT_OUT_DIR = "reacher/plan/ilqr_mpc_lewm_protocol"


@dataclass(frozen=True)
class EvalCase:
    episode_idx: int
    start_step: int
    goal_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(base.DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(base.DEFAULT_TEST_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default=base.DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=base.HORIZON)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=base.VIDEO_FPS)
    parser.add_argument("--q-terminal", type=float, default=base.Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=base.Q_STAGE)
    parser.add_argument("--r-control", type=float, default=base.R_CONTROL)
    parser.add_argument("--ilqr-max-iters", type=int, default=15)
    parser.add_argument("--ilqr-tol", type=float, default=1e-4)
    parser.add_argument("--ilqr-regularization", type=float, default=1e-3)
    parser.add_argument("--qpos-threshold", type=float, default=0.05)
    parser.add_argument(
        "--success-mode",
        choices=("qpos", "observation"),
        default="qpos",
        help="qpos matches LeWM's qpos_match termination; observation matches plan_ilqr_mpc.py.",
    )
    parser.add_argument("--observation-threshold", type=float, default=0.05)
    parser.add_argument("--no-videos", action="store_true")
    return parser.parse_args()


def load_episode_lengths(dataset_path: Path) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        return np.asarray(h5["ep_len"][:], dtype=np.int64)


def sample_eval_cases(args: argparse.Namespace, ep_len: np.ndarray) -> list[EvalCase]:
    goal_offset = int(args.goal_offset_steps)
    if goal_offset < 1:
        raise ValueError("--goal-offset-steps must be positive.")
    if int(args.eval_budget) < 1:
        raise ValueError("--eval-budget must be positive.")

    if args.episode_idx is not None:
        episode_idx = int(args.episode_idx)
        if episode_idx < 0 or episode_idx >= ep_len.shape[0]:
            raise ValueError(f"--episode-idx must be in [0, {ep_len.shape[0] - 1}], got {episode_idx}.")
        max_start = int(ep_len[episode_idx]) - goal_offset - 1
        if max_start < 0:
            raise ValueError(
                f"Episode {episode_idx} length {ep_len[episode_idx]} is shorter than "
                f"goal_offset_steps + 1 ({goal_offset + 1})."
            )
        start_step = 0 if args.start_step is None else int(args.start_step)
        if start_step < 0 or start_step > max_start:
            raise ValueError(f"--start-step must be in [0, {max_start}] for episode {episode_idx}.")
        return [EvalCase(episode_idx, start_step, start_step + goal_offset)]

    valid_cases = [
        EvalCase(int(ep), int(start), int(start + goal_offset))
        for ep, length in enumerate(ep_len)
        for start in range(max(0, int(length) - goal_offset))
    ]
    if not valid_cases:
        raise ValueError("No valid dataset rows for the requested --goal-offset-steps.")
    # Mirror third_party/le-wm/eval.py exactly: it samples from
    # np.arange(len(valid_indices) - 1), excluding the final valid row.
    sample_population = len(valid_cases) - 1
    if sample_population < 1:
        raise ValueError("Need at least two valid dataset rows to mirror LeWM sampling.")
    if args.num_eval > sample_population:
        raise ValueError(f"Requested {args.num_eval} eval cases but only {sample_population} are valid.")

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(sample_population, size=int(args.num_eval), replace=False)
    return [valid_cases[int(i)] for i in np.sort(selected)]


def qpos_success(current_qpos: np.ndarray, goal_qpos: np.ndarray, threshold: float) -> tuple[bool, float]:
    err = np.abs(current_qpos - goal_qpos)
    return bool(np.all(err < threshold)), float(np.max(err))


def load_preprocessing(
    *,
    dataset_path: Path,
    config: dict[str, object],
    history_size: int,
    img_size: int,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    train_dataset_path = Path(str(config.get("dataset_path", dataset_path))).expanduser().resolve()
    train_stats_dataset = base.LeWMReacherDataset(
        train_dataset_path,
        history_size=history_size,
        num_preds=1,
        frameskip=int(config.get("frameskip", 1)),
        img_size=img_size,
        action_dim=action_dim,
    )
    return (
        train_stats_dataset.pixel_mean,
        train_stats_dataset.pixel_std,
        train_stats_dataset.action_mean.astype(np.float32),
        train_stats_dataset.action_std.astype(np.float32),
    )


def run_case(
    *,
    case: EvalCase,
    case_idx: int,
    args: argparse.Namespace,
    model: torch.nn.Module,
    dynamics: base.MarkovDynamicsTorch,
    device: torch.device,
    img_size: int,
    markov_state_dim: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    out_root: Path,
) -> dict[str, object]:
    episode = base.load_dataset_episode(args.dataset_path, case.episode_idx)
    pixels_np = np.asarray(episode["pixels"])
    qpos_np = np.asarray(episode["qpos"])
    qvel_np = np.asarray(episode["qvel"])
    obs_np = np.asarray(episode["observation"])
    episode_seed = int(episode["episode_seed"])
    physics_freq_hz = float(episode["physics_freq_hz"])
    time_limit = float(episode["time_limit"])
    height = int(episode["height"])
    width = int(episode["width"])

    pixels = base.preprocess_pixels(
        pixels_np[[case.start_step, case.goal_step]],
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    start_emb, goal_emb = base.encode_frames(
        model,
        pixels,
        device=device,
        frame_batch_size=args.frame_batch_size,
    )
    start_state = base.make_markov_state(start_emb)
    goal_state = base.make_markov_state(goal_emb)
    if int(start_state.numel()) != markov_state_dim:
        raise ValueError(f"State dimension mismatch: config says {markov_state_dim}, built {start_state.numel()}.")

    case_dir = out_root / f"case_{case_idx:04d}_episode_{case.episode_idx:05d}_start_{case.start_step:05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    base.save_rgb_image(case_dir / "start_image.png", pixels_np[case.start_step])
    base.save_rgb_image(case_dir / "goal_image.png", pixels_np[case.goal_step])

    env = base.make_render_env(
        seed=episode_seed,
        time_limit=time_limit,
        width=width,
        height=height,
        physics_freq_hz=physics_freq_hz,
    )
    current_frame = base.reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_np[case.start_step],
        qvel=qvel_np[case.start_step],
        height=height,
        width=width,
    )

    solver = base.ILQRMPCSolver(
        dynamics,
        horizon=args.horizon,
        q_terminal=args.q_terminal,
        q_stage=args.q_stage,
        r_control=args.r_control,
        max_iters=args.ilqr_max_iters,
        tol=args.ilqr_tol,
        regularization=args.ilqr_regularization,
        device=device,
    )

    current_emb = base.encode_single_frame(
        model,
        current_frame,
        device=device,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    current_state = base.make_markov_state(current_emb)
    goal_state_np = goal_state.detach().cpu().numpy().astype(np.float64)
    goal_qpos = qpos_np[case.goal_step].astype(np.float32)
    goal_obs = obs_np[case.goal_step].astype(np.float32)

    rollout_frames = [current_frame.copy()]
    solve_times_ms: list[float] = []
    ilqr_iterations: list[int] = []
    success = False
    success_distance = float("inf")
    stop_reason = "eval_budget"

    for step_idx in range(int(args.eval_budget)):
        current_state_np = current_state.detach().cpu().numpy().astype(np.float64)
        _, u_plan, solve_time, n_iters, _ = solver.solve(current_state_np, goal_state_np)
        solve_times_ms.append(solve_time * 1000.0)
        ilqr_iterations.append(int(n_iters))

        u0_norm = u_plan[0].astype(np.float32)
        u0_raw = base.normalized_to_raw_action(u0_norm, action_mean, action_std)
        obs, _, terminated, truncated, _ = env.step(u0_raw)
        current_obs = np.asarray(obs, dtype=np.float32)
        current_qpos = np.asarray(env._env.physics.data.qpos[: goal_qpos.shape[0]], dtype=np.float32)

        if args.success_mode == "qpos":
            success, success_distance = qpos_success(current_qpos, goal_qpos, float(args.qpos_threshold))
        else:
            success, success_distance = base.goal_reached(current_obs, goal_obs, float(args.observation_threshold))
        if success:
            stop_reason = "goal_reached"
            break
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break

        current_frame = env._env.physics.render(height=height, width=width, camera_id=0)
        next_emb = base.encode_single_frame(
            model,
            current_frame,
            device=device,
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        current_state = base.make_markov_state(next_emb, current_emb)
        current_emb = next_emb
        rollout_frames.append(current_frame.copy())

    final_qpos = np.asarray(env._env.physics.data.qpos[: goal_qpos.shape[0]], dtype=np.float32)
    env.close()

    video_path = None
    if not args.no_videos:
        video_path = str(base.save_rollout_video(rollout_frames, case_dir, fps=args.video_fps))

    return {
        **asdict(case),
        "success": bool(success),
        "success_distance": float(success_distance),
        "stop_reason": stop_reason,
        "steps_executed": len(solve_times_ms),
        "final_qpos": final_qpos.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "mean_solve_time_ms": float(np.mean(solve_times_ms)) if solve_times_ms else float("nan"),
        "mean_ilqr_iterations": float(np.mean(ilqr_iterations)) if ilqr_iterations else float("nan"),
        "video_path": video_path,
    }


def main() -> None:
    args = parse_args()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    device = base.require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve() / f"{int(time.time())}_seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    config = base.load_config(model_dir)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else base.latest_object_checkpoint(model_dir).resolve()
    )
    model = base.load_model(checkpoint_path, device)

    history_size = int(config.get("history_size", 1))
    if history_size != 1:
        raise ValueError(f"Expected history_size=1 for the finetuned MLP model, got {history_size}.")

    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 2))
    embed_dim = int(config.get("embed_dim", 18))
    markov_state_dim = int(config.get("markov_state_dim", 2 * embed_dim))
    pixel_mean, pixel_std, action_mean, action_std = load_preprocessing(
        dataset_path=args.dataset_path,
        config=config,
        history_size=history_size,
        img_size=img_size,
        action_dim=action_dim,
    )
    dynamics = base.MarkovDynamicsTorch(model, markov_state_dim, action_dim, device)

    ep_len = load_episode_lengths(args.dataset_path)
    cases = sample_eval_cases(args, ep_len)
    case_results = []
    for case_idx, case in enumerate(tqdm(cases, desc="Eval cases")):
        case_results.append(
            run_case(
                case=case,
                case_idx=case_idx,
                args=args,
                model=model,
                dynamics=dynamics,
                device=device,
                img_size=img_size,
                markov_state_dim=markov_state_dim,
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                action_mean=action_mean,
                action_std=action_std,
                out_root=out_root,
            )
        )

    episode_successes = np.asarray([r["success"] for r in case_results], dtype=bool)
    metrics = {
        "success_rate": float(np.mean(episode_successes) * 100.0),
        "episode_successes": episode_successes.astype(int).tolist(),
        "num_eval": len(case_results),
        "seed": int(args.seed),
        "goal_offset_steps": int(args.goal_offset_steps),
        "eval_budget": int(args.eval_budget),
        "success_mode": args.success_mode,
        "qpos_threshold": float(args.qpos_threshold),
        "observation_threshold": float(args.observation_threshold),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "dataset_path": str(args.dataset_path),
        "cases": case_results,
    }
    with (out_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
