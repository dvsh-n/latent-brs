#!/usr/bin/env python3
"""Plan in OGBench space using one MPPI bootstrap pass followed by pure SLS MPC."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyrallis
import torch
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import gymnasium
import imageio.v2 as imageio

from error_model import MGNLLPredictor
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.mppi_planner import MPPIPlanner

from ogbench_cube.data.ogbench_cube_data_gen import LocalCubePlanOracle
from ogbench_cube.train.mlpdyn_train import LeWMOGBenchCubeDataset
from ogbench_cube.plan.plan_sls_mpc_mppi import (
    PlanSLSMoppiCubeConfig,
    build_equinox_mlp_from_pytorch,
    build_jax_obstacle_from_artifact,
    build_jax_state_region_from_artifact,
    cube_is_grasped,
    encode_single_frame,
    latest_object_checkpoint,
    load_calibrated_cholesky,
    load_planning_episode,
    make_action_reference,
    make_action_weights,
    make_constant_jax_disturbance,
    make_control_box_constraints,
    make_jax_disturbance,
    make_mppi_rollout_and_eval,
    make_obstacle_constraint,
    make_state_region_constraint,
    make_tracking_cost,
    normalized_to_raw_action,
    ogbench_success,
    render_without_target_cube,
    reset_env_to_state,
    resolve_action_stats_dataset_path,
    save_rgb_image,
    synthesize_qpos_qvel_from_block_pose,
)


def _shift_state_sequence(X: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([X[1:], X[-1:]], axis=0)


def _shift_action_sequence(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([U[1:], U[-1:]], axis=0)


@dataclass
class PlanBootstrapMPPICubeConfig(PlanSLSMoppiCubeConfig):
    """Configuration for MPPI-bootstrapped SLS MPC."""

    bootstrap_mppi_steps: int = field(default=5)


def main() -> None:
    cfg = pyrallis.parse(config_class=PlanBootstrapMPPICubeConfig)
    if cfg.bootstrap_mppi_steps < 0:
        raise ValueError("bootstrap_mppi_steps must be non-negative.")
    if cfg.mppi_state_region_penalty is None:
        cfg.mppi_state_region_penalty = float(cfg.mppi_state_box_penalty)
    default_out_dir = PlanBootstrapMPPICubeConfig.__dataclass_fields__["out_dir"].default
    if cfg.out_dir == default_out_dir:
        cfg.out_dir = Path("ogbench_cube/plan/sls_mpc_bootstrap_mppi")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    out_dir = cfg.out_dir.expanduser().resolve() / f"{int(time.time())}_bootstrap_sls_cube"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = cfg.model_dir.expanduser().resolve()
    with open(model_dir / "config.json", "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)

    model = torch.load(latest_object_checkpoint(model_dir), map_location=device, weights_only=False).eval()
    state_dim = int(config_dict.get("markov_state_dim", 48))
    action_dim = int(config_dict.get("action_dim", 5))
    img_size = int(config_dict.get("img_size", 224))

    init_key = jax.random.PRNGKey(cfg.seed)
    k1, k2, k3 = jax.random.split(init_key, 3)
    eqx_dyn = build_equinox_mlp_from_pytorch(model.predictor.net, k1)
    dynamics = lambda x, u, t=0.0, parameter=1.0: eqx_dyn(jnp.concatenate([x, u], axis=-1))

    obstacle_model = None
    obstacle_constraint = None
    state_region = None
    state_region_constraint = None
    if cfg.enable_obstacle:
        obstacle_model = build_jax_obstacle_from_artifact(cfg.obstacle_model_path, k3)
        if obstacle_model.input_dim > state_dim:
            raise ValueError(
                f"Obstacle classifier input_dim={obstacle_model.input_dim} exceeds planner state_dim={state_dim}."
            )
        obstacle_constraint = make_obstacle_constraint(obstacle_model, cfg.obstacle_margin)
        print(
            f"Using conformal obstacle classifier from {cfg.obstacle_model_path} "
            f"with threshold {float(obstacle_model.threshold):.6g} and margin {cfg.obstacle_margin:.6g}"
        )
    else:
        print("Obstacle avoidance disabled.")
    if cfg.state_region_path is not None:
        state_region = build_jax_state_region_from_artifact(cfg.state_region_path, state_dim)
        state_region_constraint = make_state_region_constraint(state_region)
        print(f"Using conformal Markov state region from {cfg.state_region_path}")
    else:
        print("Conformal latent state-region constraint disabled.")

    if cfg.use_constant_covariance:
        calibrated_cholesky = load_calibrated_cholesky(cfg.constant_covariance_path)
        disturbance = make_constant_jax_disturbance(calibrated_cholesky, state_dim)
        print(f"Using fixed calibrated covariance disturbance from {cfg.constant_covariance_path}")
    else:
        error_model = MGNLLPredictor.load_from_checkpoint(cfg.error_model_ckpt).to(device).eval()
        disturbance = make_jax_disturbance(
            build_equinox_mlp_from_pytorch(error_model.net, k2),
            cfg.q_learned,
            state_dim,
            error_model.diagonal,
        )

    episode, episode_idx = load_planning_episode(cfg.dataset_path, cfg.episode_idx, cfg.seed)
    qpos_init = episode["qpos_init"]
    qvel_init = episode["qvel_init"]
    qpos_goal = episode["qpos_goal"]
    qvel_goal = episode["qvel_goal"]
    target_block_pos_init = episode["target_block_pos_init"]
    target_block_yaw_init = float(episode["target_block_yaw_init"])
    target_block_pos_goal = episode["target_block_pos_goal"]
    target_block_yaw_goal = float(episode["target_block_yaw_goal"])
    start_pixels = episode.get("start_pixels")
    goal_pixels = episode.get("goal_pixels")
    episode_seed = int(episode["episode_seed"])
    env_name = str(episode["env_name"])
    camera = str(episode["camera"])

    env = gymnasium.make(
        env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=cfg.visualize_success_colors,
        width=256,
        height=256,
    )
    oracle = LocalCubePlanOracle(env=env, segment_dt=0.4, noise=0.0)

    if bool(episode.get("needs_qpos_synthesis", False)):
        qpos_init, qvel_init = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_init"], dtype=np.float32),
            float(episode["block_yaw_init"]),
            episode_seed,
        )
        qpos_goal, qvel_goal = synthesize_qpos_qvel_from_block_pose(
            env,
            np.asarray(episode["block_pos_goal"], dtype=np.float32),
            float(episode["block_yaw_goal"]),
            episode_seed,
        )

    goal_frame, _ = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_goal,
        qvel=qvel_goal,
        target_block_pos=target_block_pos_goal,
        target_block_yaw=target_block_yaw_goal,
        camera=camera,
    )
    current_frame, current_info = reset_env_to_state(
        env,
        seed=episode_seed,
        qpos=qpos_init,
        qvel=qvel_init,
        target_block_pos=target_block_pos_init,
        target_block_yaw=target_block_yaw_init,
        camera=camera,
    )

    start_image = np.asarray(start_pixels, dtype=np.uint8).copy() if start_pixels is not None else current_frame.copy()
    goal_image = np.asarray(goal_pixels, dtype=np.uint8).copy() if goal_pixels is not None else goal_frame.copy()
    save_rgb_image(out_dir / "start_image.png", start_image)
    save_rgb_image(out_dir / "goal_image.png", goal_image)

    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    goal_emb = encode_single_frame(model, goal_image, device, img_size, pixel_mean, pixel_std)
    goal_state = torch.cat([goal_emb, torch.zeros_like(goal_emb)], dim=-1).cpu().numpy().astype(np.float64)

    rollout_frames = [current_frame.copy()]
    grasped = False
    oracle.reset(None, current_info)
    for _ in range(cfg.max_oracle_steps):
        if cube_is_grasped(current_info, cfg.grasp_contact_threshold, cfg.grasp_alignment_threshold):
            grasped = True
            break
        current_info = env.step(np.asarray(oracle.select_action(None, current_info), dtype=np.float32))[4]
        rollout_frames.append(render_without_target_cube(env, camera))
    if not grasped:
        env.close()
        return

    action_stats_dataset_path = resolve_action_stats_dataset_path(cfg)
    train_stats = LeWMOGBenchCubeDataset(
        str(action_stats_dataset_path),
        markov_deriv=1,
        num_preds=1,
        frameskip=1,
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean = train_stats.action_mean.astype(np.float32)
    action_std = train_stats.action_std.astype(np.float32)
    print(f"Using action statistics from {action_stats_dataset_path}")

    W_mppi_stage = jnp.ones((state_dim,)) * cfg.mppi_q_stage
    W_mppi_stage = W_mppi_stage.at[state_dim // 2 :].set(1.0)
    W_mppi_terminal = jnp.ones((state_dim,)) * cfg.mppi_q_terminal
    W_mppi_terminal = W_mppi_terminal.at[state_dim // 2 :].set(1.0)
    W_stage_scaled = jnp.ones((state_dim,)) * cfg.q_stage
    W_stage_scaled = W_stage_scaled.at[state_dim // 2 :].set(1.0)
    W_terminal_scaled = jnp.ones((state_dim,)) * cfg.q_terminal
    W_terminal_scaled = W_terminal_scaled.at[state_dim // 2 :].set(1.0)

    u_min, u_max = -2.0 * jnp.ones(action_dim), 2.0 * jnp.ones(action_dim)
    u_max = u_max.at[4].set(2.0)
    u_min = u_min.at[4].set(-2.0)
    action_ref = make_action_reference(action_dim, u_min, u_max)

    mppi_roll, mppi_ev = make_mppi_rollout_and_eval(
        dynamics,
        W_mppi_stage,
        W_mppi_terminal,
        jnp.asarray(goal_state),
        obstacle_model=obstacle_model,
        state_region=state_region,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
        state_region_penalty_weight=(
            float(cfg.mppi_state_region_penalty) if state_region is not None else 0.0
        ),
        r_control=cfg.mppi_r_control,
        r_control_u4=cfg.mppi_r_control_u4,
        action_ref=action_ref,
    )
    mppi_planner = MPPIPlanner(
        config={
            "planning": {
                "action_dim": action_dim,
                "n_sample": cfg.mppi_samples,
                "horizon": cfg.horizon,
                "n_update_iter": cfg.mppi_update_iter,
                "use_last": True,
                "reject_bad": False,
                "mppi": {
                    "reward_weight": cfg.mppi_reward_weight,
                    "noise_level": cfg.mppi_noise_level,
                    "noise_decay": 1.0,
                    "beta_filter": cfg.mppi_beta_filter,
                },
            }
        },
        model_rollout_fn=mppi_roll,
        evaluate_traj_fn=mppi_ev,
        action_lower_lim=u_min,
        action_upper_lim=u_max,
    )

    def run_mppi_opt(key_arg, state_arg, actions_arg):
        return mppi_planner.trajectory_optimization(key_arg, state_arg, actions_arg, skip=False)

    action_weights = make_action_weights(action_dim, cfg.r_control, cfg.r_control_u4)
    cost = make_tracking_cost(
        action_weights=action_weights,
        horizon=cfg.horizon,
        W_stage=W_stage_scaled,
        W_terminal=W_terminal_scaled,
        goal_state=jnp.asarray(goal_state),
        action_ref=action_ref,
        obstacle_model=obstacle_model,
        obstacle_margin=cfg.obstacle_margin,
        obstacle_penalty_weight=(cfg.obstacle_penalty_weight if obstacle_model is not None else 0.0),
    )
    constraints_all = lambda x, u, t: jnp.concatenate(
        [
            make_control_box_constraints(u_min, u_max)(x, u, t),
            *(() if state_region_constraint is None else (state_region_constraint(x, u, t),)),
            *(() if obstacle_constraint is None else (obstacle_constraint(x, u, t),)),
        ],
        axis=0,
    )

    u_init = jnp.zeros((cfg.horizon, action_dim))
    u_init = u_init.at[:, 4].set(action_ref[4])
    controller = GenericMPC(
        SLSConfig(max_sls_iterations=1, enable_fastsls=False, initialize_nominal=True, warm_start=True, rti=True),
        SQPConfig(max_sqp_iterations=1, warm_start=False, feas_tol=1e-2, step_tol=1e-4, line_search=False),
        ADMMConfig(eps_abs=1e-2, eps_rel=1e-4, rho_max=1e2, max_iterations=300, initial_rho=1.0),
        config=MPCConfig(n=state_dim, nu=action_dim, N=cfg.horizon, W=W_stage_scaled, u_ref=action_ref, dt=1.0 / 20.0),
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)),
        cost=cost,
        num_constraints=2 * action_dim + (1 if state_region_constraint is not None else 0) + (1 if obstacle_constraint is not None else 0),
        disturbance=disturbance,
        shift=1,
        X_in=jnp.zeros((cfg.horizon + 1, state_dim)),
        U_in=u_init,
    )

    current_emb = encode_single_frame(model, rollout_frames[-1], device, img_size, pixel_mean, pixel_std)
    current_state = torch.cat([current_emb, torch.zeros_like(current_emb)], dim=-1).cpu().numpy().astype(np.float64)

    if obstacle_model is not None:
        start_score = float(obstacle_model(jnp.asarray(current_state)))
        goal_score = float(obstacle_model(jnp.asarray(goal_state)))
        required_score = float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        if start_score <= required_score or goal_score <= required_score:
            print(
                "Terminating: start and goal must both be outside the conformal obstacle set. "
                f"Required score > {required_score:.6g}; start_score={start_score:.6g}, goal_score={goal_score:.6g}."
            )
            env.close()
            return
        print(
            "Obstacle sanity check passed: "
            f"start_score={start_score:.6g}, goal_score={goal_score:.6g}, required_score>{required_score:.6g}"
        )
    if state_region is not None:
        start_region_score = float(state_region.score(jnp.asarray(current_state)))
        goal_region_score = float(state_region.score(jnp.asarray(goal_state)))
        if start_region_score > 1.0 or goal_region_score > 1.0:
            print(
                "Terminating: start and goal must both lie inside the conformal Markov state region. "
                f"Required score <= 1; start_score={start_region_score:.6g}, goal_score={goal_region_score:.6g}."
            )
            env.close()
            return
        print(
            "State-region sanity check passed: "
            f"start_score={start_region_score:.6g}, goal_score={goal_region_score:.6g}, required_score<=1"
        )

    goal_ref = jnp.tile(jnp.asarray(goal_state)[None, :], (cfg.horizon + 1, 1))
    prev_X = jnp.zeros((cfg.horizon + 1, state_dim))
    prev_U = u_init
    prev_u0 = np.zeros(action_dim, dtype=np.float32)
    executed_actions_norm: list[np.ndarray] = []
    executed_actions_raw: list[np.ndarray] = []

    jax_seed = jax.random.PRNGKey(cfg.seed)
    controller.X_in = prev_X
    controller.U_in = prev_U
    print(f"Running MPPI warmstart before SLS for the first {cfg.bootstrap_mppi_steps} MPC step(s)")

    mpc_pbar = tqdm(range(cfg.max_mpc_steps), desc="Bootstrap MPPI -> SLS MPC")
    for step_idx in mpc_pbar:
        if step_idx > 0:
            controller.X_in = _shift_state_sequence(prev_X)
            controller.U_in = _shift_action_sequence(prev_U)

        reference = goal_ref
        use_mppi_warmstart = step_idx < cfg.bootstrap_mppi_steps
        mppi_status = "not_used"
        if use_mppi_warmstart:
            try:
                jax_seed, mppi_key = jax.random.split(jax_seed)
                mppi_res = run_mppi_opt(mppi_key, jnp.asarray(current_state), controller.U_in)
                X_mppi = jnp.concatenate([jnp.asarray(current_state)[None, :], jnp.asarray(mppi_res["state_seq"])], axis=0)
                U_mppi = jnp.asarray(mppi_res["act_seq"])
                if np.all(np.isfinite(np.asarray(X_mppi))) and np.all(np.isfinite(np.asarray(U_mppi))):
                    controller.X_in = X_mppi
                    controller.U_in = U_mppi
                    reference = X_mppi
                    mppi_status = "mppi_warmstart"
                else:
                    mppi_status = "mppi_nonfinite"
            except Exception:
                mppi_status = "mppi_exception"

        try:
            u0, X_pred, U_pred, *_ = controller.run(
                x0=current_state,
                reference=reference,
                parameter=1.0 / 20.0,
            )
            status = f"{mppi_status}_sls" if use_mppi_warmstart else "sls_mpc"
        except Exception:
            u0, X_pred, U_pred = None, None, None
            status = "exception_fallback"

        if u0 is None or X_pred is None or U_pred is None:
            u0, X_pred, U_pred = None, None, None
        elif not (
            np.all(np.isfinite(np.asarray(u0)))
            and np.all(np.isfinite(np.asarray(X_pred)))
            and np.all(np.isfinite(np.asarray(U_pred)))
        ):
            u0, X_pred, U_pred = None, None, None
            status = "nonfinite_fallback"

        if u0 is None:
            u0 = prev_u0
            status = "frozen_fallback"
        else:
            prev_u0 = np.asarray(u0, dtype=np.float32)
            prev_X = jnp.asarray(X_pred)
            prev_U = jnp.asarray(U_pred)

        u0_norm = np.asarray(u0, dtype=np.float64).reshape(-1)
        u_raw = normalized_to_raw_action(u0_norm, action_mean, action_std)
        current_info = env.step(u_raw)[4]
        executed_actions_norm.append(u0_norm.astype(np.float32))
        executed_actions_raw.append(u_raw.astype(np.float32))
        reached_ogbench_success = ogbench_success(current_info)

        frame = render_without_target_cube(env, camera)
        rollout_frames.append(frame)

        next_emb = encode_single_frame(model, frame, device, img_size, pixel_mean, pixel_std)
        current_state = torch.cat([next_emb, next_emb - current_emb], dim=-1).cpu().numpy().astype(np.float64)
        current_emb = next_emb

        lat_err = float(np.linalg.norm(current_state - goal_state))
        if reached_ogbench_success:
            status = "ogbench_success"
        obs_free = "n/a"
        if obstacle_model is not None:
            obstacle_score = float(obstacle_model(jnp.asarray(current_state)))
            obs_free = obstacle_score > float(obstacle_model.threshold) + float(cfg.obstacle_margin)
        mpc_pbar.set_postfix(latent_err=f"{lat_err:.4f}", obs_free=obs_free, status=status)
        if cfg.terminate_on_ogbench_success and reached_ogbench_success:
            break
        if lat_err <= 0.05:
            break

    imageio.mimwrite(out_dir / "cube_bootstrap_sls.mp4", rollout_frames, fps=cfg.video_fps)
    np.savez(
        out_dir / "executed_actions.npz",
        executed_actions_norm=np.asarray(executed_actions_norm, dtype=np.float32),
        executed_actions_raw=np.asarray(executed_actions_raw, dtype=np.float32),
    )
    env.close()


if __name__ == "__main__":
    main()
