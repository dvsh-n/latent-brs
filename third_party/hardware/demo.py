import time
import numpy as np

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.solvers import Solve
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.autodiffutils import InitializeAutoDiff
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import Demultiplexer, Multiplexer, ConstantVectorSource, VectorLogSink
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.all import BasicVector, LeafSystem
from pydrake.geometry import Meshcat
from pydrake.visualization import AddDefaultVisualization

class JointLinearTrajectory(LeafSystem):
    """Outputs q_des(t) that linearly interpolates from q0 to q1 over [0, T]."""
    def __init__(self, q0: np.ndarray, q1: np.ndarray, T: float):
        super().__init__()
        self._q0 = np.asarray(q0).copy()
        self._q1 = np.asarray(q1).copy()
        self._T = float(T)
        assert self._q0.shape == self._q1.shape
        self._n = self._q0.size
        self.DeclareVectorOutputPort("q_des", BasicVector(self._n), self._calc)

    def _calc(self, context, output):
        t = context.get_time()
        if self._T <= 1e-9:
            alpha = 1.0
        else:
            alpha = np.clip(t / self._T, 0.0, 1.0)
        q = (1.0 - alpha) * self._q0 + alpha * self._q1
        output.SetFromVector(q)

def eval_tool_pose_and_metrics(
    plant,
    plant_context,
    model_instance,
    tool_frame,
    q,
    target_xyz=None,
    z_fixed=None,
):
    """
    Shared evaluation routine for tool pose + optional metrics.

    Returns a dict with:
      - p_WT: np.ndarray (3,)
      - align_score: float (dot(tool_z_world, -world_z))
    and if target_xyz/z_fixed provided:
      - xy_err_vec: np.ndarray (2,)
      - xy_err_norm: float
      - z_dev: float
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    plant.SetPositions(plant_context, model_instance, q)

    X_WT = plant.CalcRelativeTransform(
        plant_context, plant.world_frame(), tool_frame
    )
    p = np.array(X_WT.translation(), dtype=float)
    R = X_WT.rotation().matrix()
    tool_z_world = R[:, 2]
    align_score = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))

    out = {"p_WT": p, "align_score": align_score}

    if target_xyz is not None:
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        xy_err_vec = p[:2] - target_xyz[:2]
        out["xy_err_vec"] = xy_err_vec
        out["xy_err_norm"] = float(np.linalg.norm(xy_err_vec))

    if z_fixed is not None:
        out["z_dev"] = float(abs(p[2] - float(z_fixed)))

    return out

class ToolPoseAndError(LeafSystem):
    """
    Logs [x, y, z, xy_err_norm, z_dev, align_score]
    where align_score = dot(tool_z_world, -world_z) (1 is perfect "down").
    """
    def __init__(self, plant, model_instance, tool_frame, target_xyz, z_fixed):
        super().__init__()
        self._plant = plant
        self._model = model_instance
        self._tool = tool_frame
        self._target = np.asarray(target_xyz).copy()
        self._z_fixed = float(z_fixed)

        # Cache a context once (important for future hardware reuse)
        self._plant_context = self._plant.CreateDefaultContext()

        nq = plant.num_positions(model_instance)
        self.DeclareVectorInputPort("q", BasicVector(nq))
        self.DeclareVectorOutputPort("metrics", BasicVector(6), self._calc)

    def _calc(self, context, output):
        q = self.get_input_port(0).Eval(context)

        r = eval_tool_pose_and_metrics(
            plant=self._plant,
            plant_context=self._plant_context,
            model_instance=self._model,
            tool_frame=self._tool,
            q=q,
            target_xyz=self._target,
            z_fixed=self._z_fixed,
        )

        p = r["p_WT"]
        output.SetFromVector(
            np.array([p[0], p[1], p[2], r["xy_err_norm"], r["z_dev"], r["align_score"]])
        )


class IiwaPushingEnv:
    """
    A reusable environment:
      - holds an IK plant (finalized) with a tool frame
      - given a 2D/3D target, runs: IK -> FK feasibility check -> sim tracking -> report
    """

    def __init__(
        self,
        urdf_url="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
        time_step=0.005,
        tool_offset_z=0.10,
        kp=300.0,
        ki=0.0,
        kd=30.0,
        tol_xy=4e-3,
        tol_z=1.5e-3,
        ang_tol_deg=10.0,
        w_pos=1000.0,
        w_q=1.0,
    ):
        self.urdf_url = urdf_url
        self.time_step = float(time_step)
        self.tool_offset_z = float(tool_offset_z)

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.tol_xy = float(tol_xy)
        self.tol_z = float(tol_z)
        self.ang_tol_deg = float(ang_tol_deg)

        self.w_pos = float(w_pos)  # if you already added center-preference cost in IK
        self.w_q = float(w_q)

        # IK / FK plant
        self.plant = None
        self.scene_graph = None
        self.model_instance = None
        self.tool_frame = None

        self.nq = None
        self.nv = None
        self.na = None

        # default joint state
        self.q0 = None

        # cache current z_fixed based on q0 FK
        self.z_fixed = None

        self._build_ik_plant()
        # Cache FK context for repeated evaluations
        self._ik_plant_context = self.plant.CreateDefaultContext()
        self.reset()

    # -------------------------
    # Public API
    # -------------------------
    def reset(self, q0=None):
        """Set nominal start configuration and cache z_fixed from FK."""
        if q0 is None:
            q0 = np.zeros(self.nq)
        self.q0 = np.asarray(q0, dtype=float).reshape(-1)

        p0, _align0 = self._fk_tool(self.q0)
        self.z_fixed = float(p0[2])
        return p0

    def run(self, target, sim_T=10.0, traj_T=3.0, ang_tol_deg=None, verbose=True):
        """
        Backward-compatible wrapper:
          Phase 1 (plan_ik) -> Phase 2 (execute_sim)
        """
        plan = self.plan_ik(target, ang_tol_deg=ang_tol_deg, warm_start=True)

        exec_stats = self.execute_sim(
            q_goal=plan["q_goal"],
            target_xyz=plan["target_xyz"],
            traj_T=traj_T,
            sim_T=sim_T,
        )

        stats = {
            "nq": self.nq, "nv": self.nv, "na": self.na,
            "target_xyz": plan["target_xyz"],
            "z_fixed": float(plan["target_xyz"][2]),

            "ik_success": plan["ik_success"],
            "ik_time_sec": plan["ik_time_sec"],

            # IK feasibility (FK at q_goal)
            "fk_at_q_goal": plan["fk_at_q_goal"],
            "fk_axis_err": plan["fk_axis_err"],
            "fk_pos_err": plan["fk_pos_err"],
            "fk_xy_err": plan["fk_xy_err"],
            "fk_z_err": plan["fk_z_err"],
            "fk_align_score": plan["fk_align_score"],
            "fk_down_angle_deg": plan["fk_down_angle_deg"],
        }
        stats.update(exec_stats)

        if verbose:
            self._pretty_print(stats)
        return stats

    def plan_ik(self, target, ang_tol_deg=None, warm_start=True):
        """
        Phase 1: IK planning + FK feasibility check (simulation-agnostic).

        Returns dict with keys:
          - target_xyz
          - ik_success, ik_time_sec
          - q_goal
          - fk_at_q_goal, fk_axis_err, fk_pos_err, fk_xy_err, fk_z_err
          - fk_align_score, fk_down_angle_deg
        """
        if ang_tol_deg is None:
            ang_tol_deg = self.ang_tol_deg

        target_xyz = self._normalize_target(target)

        t0 = time.time()
        q_goal, ok = self._solve_pushing_ik_once(
            q_seed=self.q0,
            target_xyz=target_xyz,
            ang_tol_deg=float(ang_tol_deg),
        )
        t1 = time.time()

        if ok and warm_start:
            # keep your existing warm-start behavior
            self.q0 = np.asarray(q_goal).copy()

        # FK feasibility check at q_goal
        p_goal, align_goal = self._fk_tool(q_goal)
        axis_err = (p_goal - target_xyz)
        pos_err = float(np.linalg.norm(axis_err))
        xy_err = float(np.linalg.norm(axis_err[:2]))
        z_err = float(abs(axis_err[2]))
        down_angle_deg = float(np.degrees(np.arccos(np.clip(align_goal, -1.0, 1.0))))

        return {
            "target_xyz": target_xyz,
            "ik_success": bool(ok),
            "ik_time_sec": float(t1 - t0),
            "q_goal": np.asarray(q_goal, dtype=float),

            "fk_at_q_goal": p_goal,
            "fk_axis_err": axis_err,
            "fk_pos_err": pos_err,
            "fk_xy_err": xy_err,
            "fk_z_err": z_err,
            "fk_align_score": float(align_goal),
            "fk_down_angle_deg": down_angle_deg,
        }

    def execute_sim(self, q_goal, target_xyz, traj_T=3.0, sim_T=10.0):
        """
        Phase 2: execute tracking in simulation.

        Returns dict with keys:
          - sim_init_sec, sim_run_sec
          - plus tracking stats from _simulate_tracking(...)
        """
        sim_init_t, sim_run_t, sim_stats = self._simulate_tracking(
            q_goal=q_goal,
            target_xyz=target_xyz,
            z_fixed=float(target_xyz[2]),
            traj_T=float(traj_T),
            sim_T=float(sim_T),
        )
        out = {
            "sim_init_sec": float(sim_init_t),
            "sim_run_sec": float(sim_run_t),
        }
        out.update(sim_stats)
        return out


    def make_target_xy_relative(self, dx, dy):
        """Convenience: target = current tool position + (dx,dy) in world XY, z fixed."""
        p0, _ = self._fk_tool(self.q0)
        return np.array([p0[0] + float(dx), p0[1] + float(dy)], dtype=float)

    # -------------------------
    # Internals: plant + IK + FK
    # -------------------------
    def _build_ik_plant(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.time_step)
        parser = Parser(plant, scene_graph)
        model_instance = parser.AddModelsFromUrl(self.urdf_url)[0]

        # Weld base for 7-DOF
        base_body = plant.GetBodyByName("base", model_instance)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

        # Tool frame
        link7 = plant.GetFrameByName("iiwa_link_7", model_instance)
        X_7T = RigidTransform(
            RotationMatrix.MakeXRotation(np.pi),
            [0.0, 0.0, self.tool_offset_z],
        )
        tool_frame = plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))

        plant.Finalize()

        nq = plant.num_positions(model_instance)
        nv = plant.num_velocities(model_instance)
        na = plant.num_actuated_dofs(model_instance)
        if not (nq == nv == na == 7):
            raise RuntimeError(f"Expected fixed-base iiwa: nq=nv=na=7, got nq={nq}, nv={nv}, na={na}.")

        self.plant = plant
        self.scene_graph = scene_graph
        self.model_instance = model_instance
        self.tool_frame = tool_frame
        self.nq, self.nv, self.na = nq, nv, na

    def _normalize_target(self, target):
        t = np.asarray(target, dtype=float).reshape(-1)
        if t.shape[0] == 2:
            return np.array([t[0], t[1], self.z_fixed], dtype=float)
        if t.shape[0] == 3:
            return t
        raise ValueError(f"target must be shape (2,) or (3,), got {t.shape}")

    def _fk_tool(self, q):
        r = eval_tool_pose_and_metrics(
            plant=self.plant,
            plant_context=self._ik_plant_context,
            model_instance=self.model_instance,
            tool_frame=self.tool_frame,
            q=q,
        )
        return r["p_WT"], r["align_score"]

    def _solve_pushing_ik_once(self, q_seed, target_xyz, ang_tol_deg):
        """
        Original working IK:
        - position box (tol_xy/tol_z)
        - tool +z aligned with world -z within ang_tol_deg
        - soft position-centering cost via AutoDiff (w_pos)
        - no joint-space quadratic regularization
        """
        plant = self.plant
        model_instance = self.model_instance
        tool_frame = self.tool_frame

        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

        plant_context = plant.CreateDefaultContext()
        plant.SetPositions(plant_context, model_instance, q_seed)

        ik = InverseKinematics(plant, plant_context)
        q = ik.q()
        prog = ik.prog()

        # --- tolerances: keep identical to your working version ---
        tol_xy = 4e-3
        tol_z  = 1.5e-3

        lower = target_xyz + np.array([-tol_xy, -tol_xy, -tol_z])
        upper = target_xyz + np.array([+tol_xy, +tol_xy, +tol_z])

        ik.AddPositionConstraint(
            frameB=tool_frame,
            p_BQ=np.zeros(3),
            frameA=plant.world_frame(),
            p_AQ_lower=lower,
            p_AQ_upper=upper,
        )

        # Axis alignment: tool +z aligned with world -z (yaw free)
        theta = float(ang_tol_deg) * np.pi / 180.0
        na_A = np.array([[0.0], [0.0], [-1.0]])  # world -z
        nb_B = np.array([[0.0], [0.0], [ 1.0]])  # tool +z
        ik.AddAngleBetweenVectorsConstraint(
            plant.world_frame(), na_A,
            tool_frame, nb_B,
            0.0, theta
        )

        # Soft cost: prefer center of the box (minimize ||p(q)-target||^2)
        w_pos = 1000.0

        plant_ad = plant.ToAutoDiffXd()
        ctx_ad = plant_ad.CreateDefaultContext()
        # Use same frame name in AutoDiff plant
        tool_ad = plant_ad.GetFrameByName(tool_frame.name(), model_instance)

        def pos_cost(x):
            # x can be float ndarray or AutoDiffXd object ndarray (SNOPT).
            if getattr(x, "dtype", None) == object:
                q_ad = x
            else:
                x = np.asarray(x, dtype=float).reshape((-1, 1))
                q_ad = InitializeAutoDiff(x, num_derivatives=x.shape[0]).reshape((-1,))

            plant_ad.SetPositions(ctx_ad, model_instance, q_ad)
            X_WT = plant_ad.CalcRelativeTransform(ctx_ad, plant_ad.world_frame(), tool_ad)
            p = X_WT.translation()
            e = p - target_xyz
            return w_pos * e.dot(e)

        prog.AddCost(pos_cost, q)

        # Seed
        prog.SetInitialGuess(q, q_seed)

        result = Solve(prog)
        if not result.is_success():
            return q_seed, False

        return result.GetSolution(q), True

    # -------------------------
    # Internals: simulation + logging
    # -------------------------
    def _simulate_tracking(self, q_goal, target_xyz, z_fixed, traj_T, sim_T):
        """
        Builds a fresh diagram per run (safe + easy).
        Returns (init_time, run_time, stats_dict).
        """
        builder = DiagramBuilder()

        # Build a fresh plant for simulation (recommended; avoids re-adding the same System).
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.time_step)
        parser = Parser(plant, scene_graph)
        model_instance = parser.AddModelsFromUrl(self.urdf_url)[0]

        base_body = plant.GetBodyByName("base", model_instance)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

        link7 = plant.GetFrameByName("iiwa_link_7", model_instance)
        X_7T = RigidTransform(
            RotationMatrix.MakeXRotation(np.pi),
            [0.0, 0.0, self.tool_offset_z],
        )
        tool_frame = plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))
        plant.Finalize()

        nq = plant.num_positions(model_instance)
        nv = plant.num_velocities(model_instance)

        demux = builder.AddSystem(Demultiplexer([nq, nv]))

        # Desired joint trajectory (Sapien-like interpolation in joint space)
        traj = builder.AddSystem(JointLinearTrajectory(self.q0, q_goal, T=traj_T))

        # IDC
        Kp = self.kp * np.ones(nq)
        Ki = self.ki * np.ones(nq)
        Kd = self.kd * np.ones(nq)
        idc = builder.AddSystem(InverseDynamicsController(plant, Kp, Ki, Kd, has_reference_acceleration=False))

        # plant state -> demux
        builder.Connect(plant.get_state_output_port(model_instance), demux.get_input_port(0))

        # estimated_state = [q; v]
        mux_est = builder.AddSystem(Multiplexer([nq, nv]))
        builder.Connect(demux.get_output_port(0), mux_est.get_input_port(0))
        builder.Connect(demux.get_output_port(1), mux_est.get_input_port(1))
        builder.Connect(mux_est.get_output_port(0), idc.get_input_port_estimated_state())

        # desired_state = [q_des; v_des=0]
        v_des_src = builder.AddSystem(ConstantVectorSource(np.zeros(nv)))
        mux_des = builder.AddSystem(Multiplexer([nq, nv]))
        builder.Connect(traj.get_output_port(0), mux_des.get_input_port(0))
        builder.Connect(v_des_src.get_output_port(0), mux_des.get_input_port(1))
        builder.Connect(mux_des.get_output_port(0), idc.get_input_port_desired_state())

        # torque -> plant
        builder.Connect(idc.get_output_port_control(), plant.get_actuation_input_port(model_instance))

        # Metrics system (reuse your existing ToolPoseAndError exactly)
        metrics_sys = builder.AddSystem(ToolPoseAndError(
            plant=plant,
            model_instance=model_instance,
            tool_frame=tool_frame,
            target_xyz=target_xyz,
            z_fixed=z_fixed,
        ))
        builder.Connect(demux.get_output_port(0), metrics_sys.get_input_port(0))

        log_metrics = builder.AddSystem(VectorLogSink(6))
        builder.Connect(metrics_sys.get_output_port(0), log_metrics.get_input_port(0))

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        plant.SetPositions(plant_context, model_instance, self.q0)
        plant.SetVelocities(plant_context, model_instance, np.zeros(nv))

        t0 = time.time()
        sim = Simulator(diagram, context)
        sim.Initialize()
        t1 = time.time()
        sim.AdvanceTo(sim_T)
        t2 = time.time()

        data = log_metrics.FindLog(context).data().T
        # columns: [x, y, z, xy_err_norm, z_dev, align_score]
        xy_err = data[:, 3]
        z_dev = data[:, 4]
        align = data[:, 5]
        p_final = data[-1, 0:3]
        xy_err_vec = p_final[0:2] - target_xyz[0:2]
        down_angle_deg = float(np.degrees(np.arccos(np.clip(float(align[-1]), -1.0, 1.0))))

        stats = {
            "final_xy_err": float(xy_err[-1]),
            "max_xy_err": float(np.max(xy_err)),
            "final_z_dev": float(z_dev[-1]),
            "max_z_dev": float(np.max(z_dev)),
            "final_align_score": float(align[-1]),
            "min_align_score": float(np.min(align)),
            "final_down_angle_deg": down_angle_deg,
            "final_tool_pos": p_final,
            "final_xy_err_vec": xy_err_vec,
        }
        return (t1 - t0), (t2 - t1), stats

    def _pretty_print(self, s):
        print(f"nq = {s['nq']} nv = {s['nv']} na = {s['na']}")
        print(f"IK success: {s['ik_success']}  (time {s['ik_time_sec']:.6f}s)")
        print("=== IK FEASIBILITY CHECK (FK at q_goal) ===")
        print("FK tool position at q_goal:", s["fk_at_q_goal"])
        print(f"Axis err: {s['fk_axis_err']}")
        print(f"Position err: {s['fk_pos_err']:.6e}  (xy {s['fk_xy_err']:.6e}, z {s['fk_z_err']:.6e})")
        print(f"Align score at q_goal: {s['fk_align_score']:.9f}")
        print(f"Align angle deg at q_goal: {s['fk_down_angle_deg']:.3f}")
        print("=== Demo Results ===")
        print(f"Target XYZ: {s['target_xyz']} z_fixed: {s['z_fixed']}")
        print(f"Final XY error: {s['final_xy_err']:.6f} m (max {s['max_xy_err']:.6f})")
        print(f"Final z dev:    {s['final_z_dev']:.6f} m (max {s['max_z_dev']:.6f})")
        print(f"Align score: final {s['final_align_score']:.6f}, min {s['min_align_score']:.6f}")
        print(f"Final down-axis angle error: {s['final_down_angle_deg']:.2f} deg")
        print("Final tool position:", s["final_tool_pos"])
        print("Final XY err vec:   ", s["final_xy_err_vec"])
        print(f"Sim init: {s['sim_init_sec']:.6f}s, sim run: {s['sim_run_sec']:.6f}s")
# def main():
#     env = IiwaPushingEnv(
#         time_step=0.005,
#         tool_offset_z=0.10,
#         kp=300.0, ki=0.0, kd=30.0,
#         tol_xy=4e-3, tol_z=1.5e-3,
#         ang_tol_deg=10.0,
#     )

#     # Example: generate a 2D target relative to current tool pose
#     target_xy = env.make_target_xy_relative(dx=0.05, dy=0.02)

#     # One call runs the whole pipeline (IK -> check -> simulate -> report)
#     env.run(target_xy, sim_T=10.0, traj_T=3.0, verbose=True)
# if __name__ == "__main__":
#     main()

def main():
    """
    Interactive keyboard demo (terminal):
      - W/S: +/-Y
      - A/D: -/+X
      - Up/Down arrows: +/-Z
      - R: reset target to current tool pose (x,y) and z_fixed
      - Q or Esc: quit

    Notes:
      - This drives the *target* interactively and runs a short sim segment each keypress.
      - It assumes your env.run(...) accepts a 3D target and uses its z (i.e., NOT forcing z_fixed).
        If your env normalizes to z_fixed always, then Up/Down won't change z; remove that normalization.
    """
    import curses
    import numpy as np
    import time

    # --- Create env ---
    env = IiwaPushingEnv(
        time_step=0.005,
        tool_offset_z=0.10,
        kp=300.0, ki=0.0, kd=30.0,
        # IK tolerances (same as your working setup)
        tol_xy=4e-3, tol_z=1.5e-3,
        ang_tol_deg=10.0,
    )

    # Initialize target at current tool pose (x,y) and z_fixed
    q0 = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 0.8, 0.0])
    p0 = env.reset(q0=q0)
    target_xyz = np.array([p0[0], p0[1], env.z_fixed], dtype=float)

    # Step sizes
    step_xy = 0.01   # meters per keypress
    step_z  = 0.01   # meters per keypress

    # Per-keypress simulation settings
    sim_T = 0.6       # seconds (short burst)
    traj_T = 0.3      # seconds (joint interpolation horizon)

    def _draw(stdscr, msg_lines):
        stdscr.erase()
        for i, line in enumerate(msg_lines):
            stdscr.addstr(i, 0, line)
        stdscr.refresh()

    def _run_once(stdscr):
        nonlocal target_xyz
        # Run pipeline (IK -> FK check -> sim) for the current target
        stats = env.run(target_xyz, sim_T=sim_T, traj_T=traj_T, verbose=False)

        lines = []
        lines.append("=== IIWA Pushing Interactive Demo ===")
        lines.append("Controls: W/S=+/-Y, A/D=-/+X, Up/Down=+/-Z, R=reset target, Q/Esc=quit")
        lines.append(f"Step: xy={step_xy:.3f} m, z={step_z:.3f} m | sim_T={sim_T:.2f}s traj_T={traj_T:.2f}s")
        lines.append("")
        lines.append(f"Current target_xyz: [{target_xyz[0]: .4f}, {target_xyz[1]: .4f}, {target_xyz[2]: .4f}]")
        lines.append(f"Env z_fixed (from reset): {env.z_fixed: .6f}")
        lines.append("")
        lines.append(f"IK success: {stats['ik_success']}  | IK time: {stats['ik_time_sec']*1e3: .3f} ms")
        lines.append("IK(FK@q_goal): "
                     f"xy_err={stats['fk_xy_err']*1e3: .3f} mm, "
                     f"z_err={stats['fk_z_err']*1e3: .3f} mm, "
                     f"down_angle={stats['fk_down_angle_deg']: .2f} deg")
        lines.append("SIM(final):  "
                     f"xy_err={stats['final_xy_err']*1e3: .3f} mm, "
                     f"z_dev={stats['final_z_dev']*1e3: .3f} mm, "
                     f"down_angle={stats['final_down_angle_deg']: .2f} deg")
        lines.append(f"Final tool pos: [{stats['final_tool_pos'][0]: .4f}, {stats['final_tool_pos'][1]: .4f}, {stats['final_tool_pos'][2]: .4f}]")
        lines.append(f"Sim runtime: {stats['sim_run_sec']: .3f} s (init {stats['sim_init_sec']: .3f} s)")
        _draw(stdscr, lines)

    def _curses_main(stdscr):
        nonlocal target_xyz, step_xy, step_z, sim_T, traj_T

        curses.curs_set(0)
        stdscr.nodelay(True)   # non-blocking getch()
        stdscr.keypad(True)    # enable arrow keys
        _run_once(stdscr)

        while True:
            ch = stdscr.getch()
            if ch == -1:
                time.sleep(0.02)
                continue

            moved = False

            # Quit
            if ch in (ord('q'), ord('Q'), 27):  # 27 = ESC
                break

            # Reset target to current tool pose + z_fixed
            if ch in (ord('r'), ord('R')):
                p0 = env.reset()
                target_xyz = np.array([p0[0], p0[1], env.z_fixed], dtype=float)
                moved = True

            # WASD for XY
            elif ch in (ord('w'), ord('W')):
                target_xyz[1] += step_xy
                moved = True
            elif ch in (ord('s'), ord('S')):
                target_xyz[1] -= step_xy
                moved = True
            elif ch in (ord('a'), ord('A')):
                target_xyz[0] -= step_xy
                moved = True
            elif ch in (ord('d'), ord('D')):
                target_xyz[0] += step_xy
                moved = True

            # Arrow up/down for Z
            elif ch == curses.KEY_UP:
                target_xyz[2] += step_z
                moved = True
            elif ch == curses.KEY_DOWN:
                target_xyz[2] -= step_z
                moved = True

            # Optional: adjust step sizes quickly (1/2 for xy, 9/0 for z)
            elif ch == ord('1'):
                step_xy = max(0.001, step_xy * 0.5)
            elif ch == ord('2'):
                step_xy = min(0.05, step_xy * 2.0)
            elif ch == ord('9'):
                step_z = max(0.001, step_z * 0.5)
            elif ch == ord('0'):
                step_z = min(0.05, step_z * 2.0)

            # Optional: adjust sim burst duration ([-] and [+])
            elif ch == ord('-'):
                sim_T = max(0.1, sim_T - 0.1)
                traj_T = max(0.05, min(traj_T, sim_T))
            elif ch in (ord('+'), ord('=')):
                sim_T = min(3.0, sim_T + 0.1)
                traj_T = min(2.0, max(traj_T, 0.5 * sim_T))

            if moved:
                _run_once(stdscr)

    curses.wrapper(_curses_main)


if __name__ == "__main__":
    main()
