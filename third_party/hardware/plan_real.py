import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import jax

import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from hardware.iiwa import IiwaHardwareEnv
from hardware.realsense import PlanarPoseDetectorAPI
# jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import equinox as eqx
import pickle
import time
import datetime

from envs.T_pushing.t_sim import generate_init_target_states, T_Sim
from models.load import load_model
from models.T_pushing.dt_dyn import T_Dynamics
from models.T_pushing.ct_dyn import Continuous_T_Dynamics
from models.T_pushing.ct_ctl import T_controller
from planning.planner import MPPIPlanner, CEMPlanner
from planning.T_pushing.plan_utils import XX, generate_test_cases, get_abs_states, make_rollout_and_reward_fns, plot_cost_stat, plot_plan_from_poses
from utils.T_pushing import hole_to_walls_aabbs
from utils.misc import box_corners_nd

from envs.T_pushing.t_sim import gen_vertices_from_poses
from planning.T_pushing.plan_utils import merge_t_shape

def visualize_initial_setup(detector: PlanarPoseDetectorAPI,
                            i: int,
                            init_pose: np.ndarray,
                            target_pose: np.ndarray,
                            stem_size: np.ndarray,
                            bar_size: np.ndarray,
                            window_size: int,
                            real_vis_offset: np.ndarray,
                            out_dir: str,
                            close_thresh: np.ndarray = np.array([5.0, 5.0, 0.1])):
    plt.ion()  # interactive mode ON

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax0, ax1 = ax  # two axes

    for a in (ax0, ax1):
        a.set_xlim([0, window_size])
        a.set_ylim([0, window_size])
        a.set_aspect("equal")

    ax0.set_title("Current (blue) and Init (green)")
    ax1.set_title("Current (blue) and Target (red)")

    # create empty patches once
    cur0 = Polygon(np.zeros((3, 2)), closed=True, color="blue", alpha=0.5)
    ini0 = Polygon(np.zeros((3, 2)), closed=True, color="green", alpha=0.5)
    cur1 = Polygon(np.zeros((3, 2)), closed=True, color="blue", alpha=0.5)
    tgt1 = Polygon(np.zeros((3, 2)), closed=True, color="red", alpha=0.5)

    ax0.add_patch(cur0); ax0.add_patch(ini0)
    ax1.add_patch(cur1); ax1.add_patch(tgt1)

    plt.tight_layout()
    fig.show()

    while True:
        state_cur = detector.get_planar_pose_in_world_mm(blocking=True, store_image=False)

        curr_polys = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, state_cur[:3] + real_vis_offset)))
        init_polys = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, init_pose + real_vis_offset)))
        tgt_polys  = merge_t_shape(*np.array(gen_vertices_from_poses(stem_size, bar_size, target_pose + real_vis_offset)))

        # update patch vertices (Nx2)
        cur0.set_xy(curr_polys)
        ini0.set_xy(init_polys)
        cur1.set_xy(curr_polys)
        tgt1.set_xy(tgt_polys)

        fig.canvas.draw_idle()
        plt.pause(0.03)  # ~30 FPS-ish UI update

        if (np.all(np.abs(state_cur[:3] - init_pose) < close_thresh)):
            # save figure
            fig_path = os.path.join(out_dir, f"initial_setup_{i:04d}.png")
            fig.savefig(fig_path, dpi=200)
            print(f"Saved initial setup figure to {fig_path}")
            plt.ioff()
            plt.close(fig)
            break

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="T_pushing_real.yaml")
def main(config: DictConfig):
    if "testing" in config:
        testing_config = config["testing"]
        mode = testing_config.get("mode", "certified")
        assert mode in {"certified", "regular"}, f"Unknown testing mode: {mode}"
        model_config = testing_config[mode]
        with open_dict(config):
            config["test_models"] = model_config
    data_config = config["data"]
    train_config = config["train_dt_dyn"]
    planning_config = config["planning"]
    verbose = planning_config.get("verbose", False)
    seed = config["settings"]["seed"]
    key = jax.random.PRNGKey(seed)

    dt_dyn_dir = config["test_models"]["dt_dyn_dir"]
    dt_dyn: T_Dynamics = load_model(model_dir=dt_dyn_dir, model_type="dt_dyn", mode="best")
    abs_pose = dt_dyn.abs_pose
    pred_mode = dt_dyn.pred_mode

    # -------------- real --------------
    # real_to_sim_scale = planning_config.get("real_to_sim_scale", 0.6)
    real_to_sim_scale = 1.0 
    assert real_to_sim_scale == 1.0
    sim_to_real_scale = 1.0 / real_to_sim_scale

    stem_size = np.array(data_config["stem_size"]) * sim_to_real_scale
    bar_size = np.array(data_config["bar_size"]) * sim_to_real_scale
    pusher_size = np.array(data_config["pusher_size"]) * sim_to_real_scale
    window_size = int(data_config["window_size"] * sim_to_real_scale)
    scale = float(data_config["scale"]) * sim_to_real_scale
    action_bound = float(planning_config["action_bound"]) * sim_to_real_scale
    # -------------- real --------------

    param_dict = {"stem_size": stem_size,
                "bar_size": bar_size, 
                "pusher_size": pusher_size,
                "save_img": True,
                "enable_vis": False,
                "window_size": window_size,}

    state_dim, pose_dim, action_dim = data_config["state_dim"], data_config["pose_dim"], data_config["action_dim"]
    T_dim = state_dim if pred_mode == "state" else pose_dim
    action_lower_lim = -action_bound * jnp.ones((action_dim,)) / scale
    action_upper_lim = action_bound * jnp.ones((action_dim,)) / scale
    
    cost_norm, only_final_cost = planning_config["cost_norm"], planning_config["only_final_cost"]
    max_steps = planning_config["max_steps"] + 1  # +1 to account for initial step
    horizon = planning_config["horizon"]
    n_act_step = planning_config["n_act_step"]

    hole_config = planning_config.get("hole", {})
    hole_enable = hole_config.get("enable", False)
    obs_dict = {}
    if hole_enable:
        hole_center = np.array(hole_config["center"]) * sim_to_real_scale
        hole_size = np.array(hole_config["size"]) * sim_to_real_scale
        c_wall, h_wall = hole_to_walls_aabbs(hole_center, hole_size, window_size=window_size)
        c_wall = np.array([[XX - 195, 755], [XX + 195, 755]])
        h_wall = np.array([[140, 45], [140, 45]])
        obs_dict["obs_pos_list"] = c_wall
        obs_dict["obs_size_list"] = h_wall
        obs_dict["obs_norm"] = 1
        param_dict.update(obs_dict)

    enable_ctl = planning_config.get("enable_ctl", False)
    if enable_ctl:
        assert abs_pose, "Controller can only be enabled when using absolute pose prediction."
    ct_ctl_dir = config["test_models"]["ct_ctl_dir"]
    ct_ctl: T_controller = load_model(model_dir=ct_ctl_dir, model_type="ct_ctl", mode="best")
    # ctl_frequency = ct_ctl.ctl_frequency
    ctl_frequency = 5

    out_dir = os.path.join(planning_config["out_path"], f"{dt_dyn_dir[-6:]}_{ct_ctl_dir[-6:]}")
    if planning_config.get("add_timestamp", False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    noise_type = planning_config.get("disturbance", {}).get("type", "none")
    assert noise_type in {"none", "normal", "uniform"}, f"Unknown disturbance type: {noise_type}"
    if noise_type != "none":
        assert pred_mode == "pose"
    noise_init = planning_config.get("disturbance", {}).get("init", 0.0)
    noise_inter = planning_config.get("disturbance", {}).get("inter", 0.0)

    # reward and planning part
    rollout_fn, reward_fn, step_cost_fn, step_cost_fn_np = make_rollout_and_reward_fns(
        dt_dyn,
        config,
        abs_pose,
        pred_mode,
    )
    planner_type = planning_config.get("planner", "mppi").lower()
    if planner_type == "mppi":
        planner = MPPIPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    elif planner_type == "cem":
        planner = CEMPlanner(config, rollout_fn, reward_fn, action_lower_lim, action_upper_lim)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    # JIT compile
    compile_start = time.time()
    jit_trajopt = eqx.filter_jit(planner.trajectory_optimization)
    jit_trajopt(key, jnp.zeros((T_dim + action_dim,)), jnp.zeros((horizon, action_dim)), skip=True, target_state=jnp.zeros((T_dim,)), pusher_pos=jnp.zeros((action_dim,)))
    jit_trajopt(key, jnp.zeros((T_dim + action_dim,)), jnp.zeros((horizon, action_dim)), skip=False, target_state=jnp.zeros((T_dim,)), pusher_pos=jnp.zeros((action_dim,)))

    jit_ctl = eqx.filter_jit(ct_ctl.forward_batchless)
    jit_ctl(jnp.zeros((T_dim + action_dim,)), jnp.zeros((T_dim,)), jnp.zeros((action_dim,)))
    compile_time = time.time() - compile_start
    print(f"JIT compilation time: {compile_time} seconds")

    num_test = planning_config["num_test"]
    test_id = planning_config.get("test_id", 0)
    init_pusher_pos_list, init_pose_list, target_pose_list = generate_test_cases(seed, num_test, test_id=test_id)

    cost_stat = []
    plan_time_stat = []
    for i in range(num_test):
        init_pusher_pos = init_pusher_pos_list[i]
        init_pose = init_pose_list[i]
        target_pose = target_pose_list[i]
        print(f"Test case {i}:")
        # print(f"  Init pusher pos: {init_pusher_pos}")
        # print(f"  Init pose: {init_pose}")
        # print(f"  Target pusher pos: {target_pusher_pos}")
        # print(f"  Target pose: {target_pose}")

        if pred_mode == "pose":
            target_state = target_pose
            scaled_target_state = target_state / scale
            scaled_target_state[2] = target_state[2]  # do not scale angle
        else:
            init_state, target_state = generate_init_target_states(
                init_pose, target_pose, param_dict={"stem_size": stem_size, "bar_size": bar_size}
            )
            scaled_target_state = target_state / scale
        scaled_target_state = jnp.array(scaled_target_state)

        # -------------- real --------------

        detector = PlanarPoseDetectorAPI(w = 1280, h = 720, fps = 15, max_stored_images = 1000)
        real_vis_offset = np.array(planning_config.get("real_vis_offset", [0.0, 0.0, 0.0]))
        # visualize initial setup
        visualize_initial_setup(
            detector,
            i,
            init_pose,
            target_pose,
            stem_size,
            bar_size,
            window_size,
            real_vis_offset,
            out_dir,
            close_thresh=np.array([10.0, 10.0, 0.2])
        )

        env = IiwaHardwareEnv(
            use_ik=True,
            period_sec=0.01,
            max_delta_per_step=0.001,
            realtime_rate=1.0,
            ik_update_period_sec=0.01
        )
        # One-call safe startup (holds current pose, then enables).
        env.start(timeout_sec=1.0)

        z_hi = 330
        z_lo = 280
        timing_bucket_names = ("total", "get_state", "planning", "waypoint_generation", "command_enqueue")
        timing_sums = {name: 0.0 for name in timing_bucket_names}
        timing_count = 0
        timing_print_first_n = 5
        timing_print_period = 10
        step_plan_vis_dir = None
        if pred_mode == "pose":
            step_plan_vis_dir = os.path.join(out_dir, f"plan_vis_steps_{i:04d}")
            os.makedirs(step_plan_vis_dir, exist_ok=True)

        def print_timing_summary(step_idx, latest_timing):
            nonlocal timing_count
            timing_count += 1
            for name in timing_bucket_names:
                timing_sums[name] += float(latest_timing.get(name, 0.0))

            # should_print = (timing_count <= timing_print_first_n) or (step_idx % timing_print_period == 0)
            should_print = True
            if not should_print:
                return

            latest_ms = {name: latest_timing[name] * 1e3 for name in timing_bucket_names}
            avg_ms = {name: (timing_sums[name] / timing_count) * 1e3 for name in timing_bucket_names}
            print(
                f"[timing] step={step_idx:03d} "
                f"latest(ms) total={latest_ms['total']:.1f} state={latest_ms['get_state']:.1f} "
                f"plan={latest_ms['planning']:.1f} waypoint={latest_ms['waypoint_generation']:.1f} "
                f"cmd={latest_ms['command_enqueue']:.3f} | "
                f"avg(ms) total={avg_ms['total']:.1f} state={avg_ms['get_state']:.1f} "
                f"plan={avg_ms['planning']:.1f} waypoint={avg_ms['waypoint_generation']:.1f} "
                f"cmd={avg_ms['command_enqueue']:.3f}"
            )

        background_control_started = False
        enable_background_detection = True
        background_detection_started = False
        low_height_reached = False
        try:
            # move to init position (transformation from workspace to robot frame is inside the function)
            t_block = time.perf_counter()
            env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_hi], dtype=float), timeout_sec=5.0, verbose=False, hold_count_required=5)
            print(f"[plan_real] init move to z_hi took {time.perf_counter() - t_block:.3f}s")
            t_block = time.perf_counter()
            env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_lo], dtype=float), timeout_sec=5.0, verbose=False, hold_count_required=5)
            print(f"[plan_real] init move to z_lo took {time.perf_counter() - t_block:.3f}s")
            low_height_reached = True

            env.start_background_control(control_period_sec=0.01)
            background_control_started = True
            env.set_latest_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_lo], dtype=float))
            print("[plan_real] Started IIWA background control for execution loop.")

            if enable_background_detection:
                detector.start_background_detection(store_image=True)
                background_detection_started = True

                t_pose_wait = time.perf_counter()
                pose_wait_timeout_sec = 5.0
                while detector.get_latest_planar_pose_in_world_mm(require_valid=False) is None:
                    if time.perf_counter() - t_pose_wait > pose_wait_timeout_sec:
                        raise RuntimeError("Timed out waiting for first cached AprilTag pose.")
                    time.sleep(0.01)
                print(f"[plan_real] First cached detector pose ready after {time.perf_counter() - t_pose_wait:.3f}s")

            # mimic sim api
            def get_state():
                # Background detection owns camera/tag updates; the foreground loop
                # reads the latest cached planar pose and current pusher position.
                if enable_background_detection:
                    state_cur = detector.get_latest_planar_pose_in_world_mm(require_valid=False)
                else:
                    state_cur = detector.get_planar_pose_in_world_mm(blocking=True, store_image=True)

                # current pusher pos in base frame in mm
                start_time = time.time()
                pusher_pos = env.get_tool_metrics_in_world_mm()[:2]
                print(f"[plan_real] get_tool_metrics_in_world_mm took {time.time() - start_time:.3f}s")
                
                env_state = np.concatenate([state_cur, pusher_pos], axis=0)

                # in NN scale
                state_cur[:2] = state_cur[:2] / scale
                pusher_pos = pusher_pos / scale
                state_cur = jnp.concatenate([jnp.array(state_cur), jnp.array(pusher_pos)], axis=0)
                pusher_pos = jnp.array(pusher_pos)
                return state_cur, env_state, pusher_pos

            t_state0 = time.perf_counter()
            state_cur, env_state, pusher_pos = get_state()
            print(f"[plan_real] initial get_state took {time.perf_counter() - t_state0:.3f}s")

            # -------------- real --------------

            # executtion loop
            planning_res_list = []
            step_cost_list = []
            gt_states = [env_state]
            t = 0
            succeed = False
            init_follow = True
            init_follow_steps = planning_config.get("init_follow_steps", 1)
            while t < max_steps and not succeed:
                # -------------- real --------------
                cycle_get_state_t0 = time.perf_counter()
                state_cur, env_state, pusher_pos = get_state()
                cycle_get_state_time = time.perf_counter() - cycle_get_state_t0
                # -------------- real --------------

                key = jax.random.PRNGKey(seed + t)

                # noise_param = noise_init
                # noise = jnp.zeros((T_dim,))
                # if not succeed:
                #     key, subkey = jax.random.split(key)
                #     if noise_type == "normal":
                #         noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
                #     elif noise_type == "uniform":
                #         noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param
                # state_cur = state_cur.at[0:T_dim].add(noise)

                # key, subkey = jax.random.split(key)
                # init_act_seq = jax.random.uniform(subkey,(horizon, action_dim),minval=action_lower_lim,maxval=action_upper_lim,)

                init_act_seq = jnp.zeros((horizon, action_dim))
                key, subkey = jax.random.split(key)
                # with jax.disable_jit():
                #     planning_res = eqx.filter_jit(planner.trajectory_optimization)(key, state_cur, init_act_seq, skip=False, target_state=scaled_target_state, pusher_pos=pusher_pos)
                start_plan_time = time.perf_counter()
                planning_res = jit_trajopt(subkey, state_cur, init_act_seq, skip=succeed, target_state=scaled_target_state, pusher_pos=pusher_pos)
                planning_time = time.perf_counter() - start_plan_time
                plan_time_stat.append(planning_time)
                if verbose and 'collision_loss' in planning_res['aux']['eval_out'] and planning_res['aux']['eval_out']['collision_loss'] > 0:
                    print(f"Step {t} planning result:")
                    print(f"   collision loss: {planning_res['aux']['eval_out']['collision_loss']}")

                waypoint_prep_t0 = time.perf_counter()
                scaled_act_seq = planning_res["act_seq"]
                scaled_state_seq = planning_res["state_seq"]
                act_seq = scaled_act_seq * scale  # (horizon, action_dim)
                print(f"Step {t} planned act_seq (scaled): {act_seq[:n_act_step]}")  # only print the first n_act_step actions for brevity
                if abs_pose:
                    abs_scaled_state_seq = jnp.concatenate([state_cur[None, :], scaled_state_seq], axis=0)
                    abs_state_seq = abs_scaled_state_seq * scale
                    if pred_mode == "pose":
                        abs_state_seq = abs_state_seq.at[:, pose_dim - 1].set(abs_state_seq[:, pose_dim - 1]/scale)  # do not scale angle
                    pusher_pos_seq = abs_scaled_state_seq[:, -action_dim:] * scale
                else:
                    abs_state_seq, pusher_pos_seq = get_abs_states(scaled_state_seq * scale[None, :, :], pusher_pos * scale, act_seq[None, :, :], pred_mode=pred_mode)
                    abs_state_seq = abs_state_seq[0]
                    pusher_pos_seq = pusher_pos_seq[0]
                res = {
                    "time_step": t,
                    "act_seq": act_seq,
                    "state_seq": abs_state_seq,
                    "pusher_pos_seq": pusher_pos_seq,
                    "planning_res": planning_res,
                }
                if verbose:
                    print(f"reach vol: {planning_res['aux']['eval_out']['reach_aux'].get('reach_vol', None)}")
                planning_res_list.append(res)
                if pred_mode == "pose":
                    plot_plan_from_poses(
                        state_seqs=np.asarray(res["state_seq"])[None, None, :, :pose_dim],
                        pusher_pos_seqs=np.asarray(res["pusher_pos_seq"])[None],
                        target_pose=target_pose,
                        stem_size=stem_size,
                        bar_size=bar_size,
                        window_size=(window_size, window_size),
                        obs_dict=obs_dict,
                        fps=1,
                        save_path=os.path.join(step_plan_vis_dir, f"plan_step.gif"),
                        real_vis_offset=real_vis_offset,
                    )
                waypoint_prep_time = time.perf_counter() - waypoint_prep_t0
                cycle_step_timings = []
                for step in range(n_act_step):
                    sub_target = abs_scaled_state_seq[step + 1, :-action_dim]
                    ref_action = scaled_act_seq[step, :]
                    sub_env_states = []
                    step_t0 = time.perf_counter()
                    step_waypoint_time = 0.0
                    step_command_time = 0.0
                    step_get_state_time = 0.0

                    if (step_cost_fn(sub_target, state_cur[:-action_dim]) < 0) or (init_follow) or succeed:
                        use_ctl = False
                    else:
                        use_ctl = enable_ctl

                    if enable_ctl:
                        for ctl_step in range(ctl_frequency):
                            waypoint_inner_t0 = time.perf_counter()
                            print(f"   step_cost: {step_cost_fn(sub_target, state_cur[:-action_dim])}, init_follow: {init_follow}")
                            if use_ctl:
                                # if verbose:
                                #     print(f"   step_cost: {step_cost_fn(sub_target, state_cur[:-action_dim])}, init_follow: {init_follow}")
                                next_action = jit_ctl(state_cur, sub_target, ref_action)
                                if verbose:
                                    print(f"   controller action: {next_action}, curr pusher pos: {pusher_pos}, curr state: {state_cur[:-action_dim]}")
                            else:
                                next_action = ref_action
                                if verbose:
                                    print(f"   skip controller, action: {next_action}, curr pusher pos: {pusher_pos}, curr state: {state_cur[:-action_dim]}")
                                
                            next_pusher_pos = (pusher_pos + next_action / 2) * scale
                            target_world_mm = np.array([next_pusher_pos[0], next_pusher_pos[1], z_lo], dtype=float)
                            step_waypoint_time += time.perf_counter() - waypoint_inner_t0
                            command_t0 = time.perf_counter()
                            env.set_latest_target_xyz_in_world_mm(target_world_mm)
                            step_command_time += time.perf_counter() - command_t0
                            if verbose:
                                print(f"   command target mm: {target_world_mm}")

                            time.sleep(0.1)  # allow some time for the command to take effect and the state to update

                            state_read_t0 = time.perf_counter()
                            state_cur, env_state, pusher_pos = get_state()
                            step_get_state_time += time.perf_counter() - state_read_t0

                            # noise_param = noise_inter
                            # noise = jnp.zeros((T_dim,))
                            # if not succeed:
                            #     key, subkey = jax.random.split(key)
                            #     if noise_type == "normal":
                            #         noise = jax.random.normal(subkey, shape=(T_dim,)) * noise_param
                            #     elif noise_type == "uniform":
                            #         noise = jax.random.uniform(subkey, shape=(T_dim,), minval=-1.0, maxval=1.0) * noise_param

                            # # state_cur = state_cur.at[0:T_dim].add(noise)
                            # env.force_update([[noise[0] * scale, noise[1] * scale, noise[2]]])  # apply disturbance

                            sub_env_states.append(env_state)
                        env_state = np.array(sub_env_states)
                        step_cost = step_cost_fn_np(env_state[-1][:-action_dim], target_state)
                    else:
                        waypoint_inner_t0 = time.perf_counter()
                        next_pusher_pos = pusher_pos_seq[step + 1, :]
                        target_world_mm = np.array([next_pusher_pos[0], next_pusher_pos[1], z_lo], dtype=float)
                        step_waypoint_time += time.perf_counter() - waypoint_inner_t0
                        command_t0 = time.perf_counter()
                        env.set_latest_target_xyz_in_world_mm(target_world_mm)
                        step_command_time += time.perf_counter() - command_t0
                        if verbose:
                            print(f"   command target mm: {target_world_mm}")
                        state_read_t0 = time.perf_counter()
                        state_cur, env_state, pusher_pos = get_state()
                        step_get_state_time += time.perf_counter() - state_read_t0
                        step_cost = step_cost_fn_np(env_state[:-action_dim], target_state)

                    cycle_step_timings.append({
                        "total": time.perf_counter() - step_t0,
                        "get_state": step_get_state_time,
                        "waypoint_generation": step_waypoint_time,
                        "command_enqueue": step_command_time,
                    })
                    t += 1
                    if t > init_follow_steps:
                        init_follow = False
                    gt_states.append(env_state)
                    if verbose:
                        print(f"   step {t} cost: {step_cost}, state: {state_cur[:-action_dim]}, pusher_pos: {pusher_pos}")
                    step_cost_list.append(step_cost)
                    if (not succeed) and step_cost < 0.25:
                        print(f"Task succeeded at step {t} with step cost {step_cost}")
                        succeed = True
                        break
                    if t >= max_steps:
                        break

                executed_steps = len(cycle_step_timings)
                if executed_steps > 0:
                    shared_get_state = cycle_get_state_time / executed_steps
                    shared_planning = planning_time / executed_steps
                    shared_waypoint = waypoint_prep_time / executed_steps
                    start_step_idx = t - executed_steps + 1
                    for idx, step_timing in enumerate(cycle_step_timings):
                        step_idx = start_step_idx + idx
                        latest_timing = {
                            "total": step_timing["total"] + shared_get_state + shared_planning + shared_waypoint,
                            "get_state": step_timing["get_state"] + shared_get_state,
                            "planning": shared_planning,
                            "waypoint_generation": step_timing["waypoint_generation"] + shared_waypoint,
                            "command_enqueue": step_timing["command_enqueue"],
                        }
                        print_timing_summary(step_idx, latest_timing)

            cost_stat.append(step_cost_list)
            print(f"final cost: {step_cost_list[-1]}, total cost: {sum(step_cost_list)}")
        finally:
            if background_detection_started and detector.is_background_detection_running():
                detector.stop_background_detection()
            if background_control_started and env.is_background_control_running():
                print("[plan_real] Stopping IIWA background control before retreat.")
                env.stop_background_control()
            if low_height_reached:
                t_block = time.perf_counter()
                env.move_to_target_xyz_in_world_mm(np.array([init_pusher_pos[0], init_pusher_pos[1], z_hi], dtype=float), timeout_sec=10.0, verbose=False, hold_count_required=5)
                print(f"[plan_real] retreat move to z_hi took {time.perf_counter() - t_block:.3f}s")
            # capture final images
            detector.get_planar_pose_in_world_mm(blocking=True, store_image=True)  # ensure latest pose is captured
            imgs_dir = os.path.join(out_dir, f"vis_{i:04d}")
            os.makedirs(imgs_dir, exist_ok=True)
            detector.save_images(imgs_dir)
            detector.close()

        # -------------- real --------------


        # save results
        # planning_res_path = os.path.join(out_dir, "planning_res.npy")
        # np.save(planning_res_path, planning_res_list)
        # gt_states_path = os.path.join(out_dir, "gt_states.npy")
        # np.save(gt_states_path, np.array(gt_states))
        planning_res_path = os.path.join(out_dir, f"planning_res_{i:04d}.pkl")
        with open(planning_res_path, "wb") as f:
            pickle.dump(planning_res_list, f)
        gt_states_path = os.path.join(out_dir, f"gt_states_{i:04d}.pkl")
        with open(gt_states_path, "wb") as f:
            pickle.dump(gt_states, f)

        if pred_mode == "pose":
            act_seqs = np.array([d["act_seq"] for d in planning_res_list])
            state_seqs = np.array([d["state_seq"] for d in planning_res_list])[..., :pose_dim]
            pusher_pos_seqs = np.array([d["pusher_pos_seq"] for d in planning_res_list])

            plot_plan_from_poses(
                state_seqs=state_seqs[None],
                pusher_pos_seqs=pusher_pos_seqs,
                target_pose=target_pose,
                stem_size=stem_size,
                bar_size=bar_size,
                window_size=(window_size, window_size),
                obs_dict=obs_dict,
                fps=5,
                save_path=os.path.join(out_dir, f"plan_vis_{i:04d}.gif"),
                real_vis_offset=real_vis_offset,
            )
            reach_config = planning_config.get("reach_in_obj", {})
            refine_config = planning_config.get("refinement", {})
            reach_refine_config = refine_config.get("reach_in_obj", {})
            enable_reach = reach_config.get("enable", False) or (refine_config.get('enable', False) and reach_refine_config.get("enable", False))
            if enable_reach:
                # r_lo_seqs, r_up_seqs: (n_sim_steps+1, horizon+1, 3)
                r_lo_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_lo'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]
                r_up_seqs = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['r_up'] for d in planning_res_list]).reshape((*state_seqs.shape[:2], -1))[..., :pose_dim]
                reach_vols = np.array([d['planning_res']['aux']['eval_out']['reach_aux']['reach_vol'] for d in planning_res_list]).reshape((state_seqs.shape[0], -1))
                print(f"Average reach volume over time: {np.mean(reach_vols, axis=0)}")

                vis_reach_steps = planning_config.get("vis_reach_steps", horizon)
                r_lo_seqs = r_lo_seqs[:, :vis_reach_steps + 1, :]
                r_up_seqs = r_up_seqs[:, :vis_reach_steps + 1, :]
                state_seqs = state_seqs[:, :vis_reach_steps + 1, :]
                pusher_pos_seqs = pusher_pos_seqs[:, :vis_reach_steps + 1, :]

                r_lo_seqs[..., :2] = r_lo_seqs[..., :2] * scale
                r_up_seqs[..., :2] = r_up_seqs[..., :2] * scale

                # sample from r_lo_seqs and r_up_seqs: choose corners for each dimension, 2^3 = 8 samples
                sample_states = box_corners_nd(r_lo_seqs, r_up_seqs)  # (8, n_sim_steps+1, horizon+1, 3)
                n_samples = sample_states.shape[0]
                sample_states = np.random.uniform(size=(n_samples, *state_seqs.shape))
                sample_state_seqs = r_lo_seqs[None] + sample_states * (r_up_seqs - r_lo_seqs)[None]
                plot_plan_from_poses(
                    state_seqs=sample_state_seqs,
                    pusher_pos_seqs=pusher_pos_seqs,
                    target_pose=target_pose,
                    stem_size=stem_size,
                    bar_size=bar_size,
                    window_size=(window_size, window_size),
                    obs_dict=obs_dict,
                    fps=5,
                    save_path=os.path.join(out_dir, f"plan_reach_vis_{i:04d}.gif"),
                    real_vis_offset=real_vis_offset,    
                )
        


    cost_stat = np.array(cost_stat)  # (num_test, max_steps)
    avg_step_cost = np.mean(cost_stat, axis=0)
    print(f"Average step cost over time over {num_test} test cases: {avg_step_cost}")
    plot_cost_stat(cost_stat, os.path.join(out_dir, "step_costs.png"))

    plan_time_stat = np.array(plan_time_stat)
    avg_plan_time = np.mean(plan_time_stat)
    print(f"Average planning time per step over {num_test} test cases: {avg_plan_time} seconds")

    # save overall stats
    stats = {
        "cost_stat": cost_stat,
        "plan_time_stat": plan_time_stat,
        "jit_compile_time": compile_time,
    }

    stats_path = os.path.join(out_dir, "planning_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # copy config file to out_dir
    with open(os.path.join(out_dir, "planning_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config, resolve=True))


    return

if __name__ == "__main__":
    main()
