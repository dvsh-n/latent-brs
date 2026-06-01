import numpy as np
import time
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem, BasicVector
from pydrake.systems.primitives import Demultiplexer, VectorLogSink
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.autodiffutils import InitializeAutoDiff
from pydrake.systems.primitives import (
    Demultiplexer, VectorLogSink, Multiplexer, ConstantVectorSource
)
from pydrake.systems.controllers import InverseDynamicsController
# --------------------------
# Helper systems
# --------------------------

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

        nq = plant.num_positions(model_instance)
        self.DeclareVectorInputPort("q", BasicVector(nq))
        self.DeclareVectorOutputPort("metrics", BasicVector(6), self._calc)

    def _calc(self, context, output):
        q = self.get_input_port(0).Eval(context)
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(plant_context, self._model, q)

        X_WT = self._plant.CalcRelativeTransform(
            plant_context, self._plant.world_frame(), self._tool
        )
        p = X_WT.translation()
        R = X_WT.rotation().matrix()
        tool_z_world = R[:, 2]
        align_score = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))

        xy_err = p[:2] - self._target[:2]
        xy_err_norm = float(np.linalg.norm(xy_err))
        z_dev = float(abs(p[2] - self._z_fixed))

        output.SetFromVector(np.array([p[0], p[1], p[2], xy_err_norm, z_dev, align_score]))


# --------------------------
# IK solve (Sapien-like: solve once)
# --------------------------

def solve_pushing_ik_once(
    plant,
    model_instance,
    tool_frame,
    q_seed: np.ndarray,
    target_xyz: np.ndarray,
    pos_tol: float = 3e-3,
    ang_tol_deg: float = 10.0,
):
    """
    Solve IK for:
      - tool position near target_xyz (box tolerance)
      - tool z-axis aligned with world -z within angle tolerance (yaw free)
    """
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, model_instance, q_seed)

    ik = InverseKinematics(plant, plant_context)
    q = ik.q()
    prog = ik.prog()

    tol_xy = 4e-3   # 2 mm in x/y
    tol_z  = 1.5e-3   # 0.5 mm in z

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

    # Soft cost: prefer the center of the position box (minimize ||p(q)-target||^2).
    w_pos = 1000.0  # try 1e3 first so it dominates the q_seed cost

    plant_ad = plant.ToAutoDiffXd()
    ctx_ad = plant_ad.CreateDefaultContext()
    tool_ad = plant_ad.GetFrameByName(tool_frame.name(), model_instance)

    def pos_cost(x):
        # x can be either float ndarray (during some evaluations) or AutoDiff object ndarray (SNOPT).
        if getattr(x, "dtype", None) == object:
            q_ad = x  # already AutoDiffXd vector
        else:
            # InitializeAutoDiff expects a float matrix; give it shape (n,1) then flatten back.
            x = np.asarray(x, dtype=float).reshape((-1, 1))
            q_ad = InitializeAutoDiff(x, num_derivatives=x.shape[0]).reshape((-1,))

        plant_ad.SetPositions(ctx_ad, model_instance, q_ad)
        X_WT = plant_ad.CalcRelativeTransform(ctx_ad, plant_ad.world_frame(), tool_ad)
        p = X_WT.translation()
        e = p - target_xyz
        return w_pos * e.dot(e)

    prog.AddCost(pos_cost, q)

    # Regularize towards seed
    # prog.AddQuadraticErrorCost(np.eye(q.shape[0]), q_seed, q)
    prog.SetInitialGuess(q, q_seed)

    result = Solve(prog)
    if not result.is_success():
        return q_seed, False

    return result.GetSolution(q), True


# --------------------------
# Main demo
# --------------------------
class IiwaPushingDemo:
    """Wraps plant + diagram build + simulation for the iiwa pushing IK demo."""

    def __init__(
        self,
        urdf_url="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
        time_step=0.005,
        tool_offset_z=0.10,
        q0=None,
        traj_T=3.0,
        sim_T=10.0,
        Kp=300.0,
        Ki=0.0,
        Kd=30.0,
    ):
        self.urdf_url = urdf_url
        self.time_step = float(time_step)
        self.tool_offset_z = float(tool_offset_z)
        self.traj_T = float(traj_T)
        self.sim_T = float(sim_T)

        self._Kp_scalar = float(Kp)
        self._Ki_scalar = float(Ki)
        self._Kd_scalar = float(Kd)

        self.q0 = None if q0 is None else np.asarray(q0).reshape(-1)

        # Filled after build_plant()
        self.plant = None
        self.scene_graph = None
        self.model_instance = None
        self.tool_frame = None
        self.nq = None
        self.nv = None
        self.na = None

        # Filled after build_diagram()
        self.diagram = None
        self.context = None
        self.log_metrics = None
        self.target_xyz = None
        self.z_fixed = None

        self.build_plant()

    def build_plant(self):
        """Create a standalone plant (no controllers) for IK + FK checks."""
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.time_step)
        parser = Parser(plant, scene_graph)

        model_instance = parser.AddModelsFromUrl(self.urdf_url)[0]

        # Weld base to world so nq=nv=na=7.
        base_body = plant.GetBodyByName("base", model_instance)
        plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

        # Define tool frame.
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
            raise RuntimeError(
                f"This demo assumes fixed-base iiwa with 7 DOF, got nq={nq}, nv={nv}, na={na}."
            )

        self.plant = plant
        self.scene_graph = scene_graph
        self.model_instance = model_instance
        self.tool_frame = tool_frame
        self.nq, self.nv, self.na = nq, nv, na

        if self.q0 is None:
            self.q0 = np.zeros(self.nq)

    def initial_tool_pose(self):
        """Return (p0, z_fixed) at q0."""
        ctx = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ctx, self.model_instance, self.q0)
        X_WT0 = self.plant.CalcRelativeTransform(ctx, self.plant.world_frame(), self.tool_frame)
        p0 = X_WT0.translation()
        return p0, float(p0[2])

    def solve_ik(self, target_xyz, ang_tol_deg=10.0):
        """Solve IK once; returns (q_goal, ok, solve_time_sec)."""
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        t0 = time.time()
        q_goal, ok = solve_pushing_ik_once(
            plant=self.plant,
            model_instance=self.model_instance,
            tool_frame=self.tool_frame,
            q_seed=self.q0,
            target_xyz=target_xyz,
            pos_tol=1e-2,         # unused in your current solve_pushing_ik_once (tol_xy/tol_z inside)
            ang_tol_deg=ang_tol_deg,
        )
        t1 = time.time()
        return q_goal, ok, (t1 - t0)

    def fk_tool(self, q):
        """FK tool pose at q; returns (p, align_score)."""
        ctx = self.plant.CreateDefaultContext()
        self.plant.SetPositions(ctx, self.model_instance, np.asarray(q).reshape(-1))
        X_WT = self.plant.CalcRelativeTransform(ctx, self.plant.world_frame(), self.tool_frame)
        p = X_WT.translation()
        R = X_WT.rotation().matrix()
        tool_z_world = R[:, 2]
        align_score = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))
        return p, align_score

    def build_diagram(self, q_goal, target_xyz, z_fixed):
        """Build the full diagram: trajectory -> IDC -> plant + metrics logging."""
        # Rebuild everything in one DiagramBuilder using the *same* finalized plant.
        builder = DiagramBuilder()

        # Add the already-finalized plant/scene_graph systems into this builder.
        # (Drake allows adding an existing System; here we reuse the plant object itself.)
        plant = self.plant
        model_instance = self.model_instance

        builder.AddSystem(plant)

        nq, nv = self.nq, self.nv

        demux = builder.AddSystem(Demultiplexer([nq, nv]))
        traj = builder.AddSystem(JointLinearTrajectory(self.q0, q_goal, T=self.traj_T))

        Kp = self._Kp_scalar * np.ones(nq)
        Ki = self._Ki_scalar * np.ones(nq)
        Kd = self._Kd_scalar * np.ones(nq)

        idc = builder.AddSystem(
            InverseDynamicsController(plant, Kp, Ki, Kd, has_reference_acceleration=False)
        )

        # plant state -> demux
        builder.Connect(plant.get_state_output_port(model_instance), demux.get_input_port(0))

        # estimated_state = [q; v]
        mux_est = builder.AddSystem(Multiplexer([nq, nv]))
        builder.Connect(demux.get_output_port(0), mux_est.get_input_port(0))
        builder.Connect(demux.get_output_port(1), mux_est.get_input_port(1))
        builder.Connect(mux_est.get_output_port(0), idc.get_input_port_estimated_state())

        # desired_state = [q_des; v_des(=0)]
        v_des_src = builder.AddSystem(ConstantVectorSource(np.zeros(nv)))
        mux_des = builder.AddSystem(Multiplexer([nq, nv]))
        builder.Connect(traj.get_output_port(0), mux_des.get_input_port(0))
        builder.Connect(v_des_src.get_output_port(0), mux_des.get_input_port(1))
        builder.Connect(mux_des.get_output_port(0), idc.get_input_port_desired_state())

        # torque -> plant
        builder.Connect(idc.get_output_port_control(), plant.get_actuation_input_port(model_instance))

        # Metrics + logging
        metrics_sys = builder.AddSystem(ToolPoseAndError(
            plant=plant,
            model_instance=model_instance,
            tool_frame=self.tool_frame,
            target_xyz=target_xyz,
            z_fixed=z_fixed,
        ))
        builder.Connect(demux.get_output_port(0), metrics_sys.get_input_port(0))

        log_metrics = builder.AddSystem(VectorLogSink(6))
        builder.Connect(metrics_sys.get_output_port(0), log_metrics.get_input_port(0))

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        # Initialize plant at q0
        plant.SetPositions(plant_context, model_instance, self.q0)
        plant.SetVelocities(plant_context, model_instance, np.zeros(nv))

        self.diagram = diagram
        self.context = context
        self.log_metrics = log_metrics
        self.target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        self.z_fixed = float(z_fixed)

    def run(self):
        """Run simulation for self.sim_T, returns (sim_init_time, sim_time)."""
        if self.diagram is None or self.context is None:
            raise RuntimeError("Call build_diagram(...) before run().")

        t0 = time.time()
        sim = Simulator(self.diagram, self.context)
        sim.Initialize()
        t1 = time.time()
        sim.AdvanceTo(self.sim_T)
        t2 = time.time()
        return (t1 - t0), (t2 - t1)

    def summarize(self):
        """Return a dict of metrics + print-friendly values."""
        data = self.log_metrics.FindLog(self.context).data().T
        # columns: [x, y, z, xy_err_norm, z_dev, align_score]
        xy_err = data[:, 3]
        z_dev = data[:, 4]
        align = data[:, 5]
        p_final = data[-1, 0:3]
        xy_err_vec = p_final[0:2] - self.target_xyz[0:2]
        angle_deg = np.degrees(np.arccos(np.clip(float(align[-1]), -1.0, 1.0)))

        return {
            "target_xyz": self.target_xyz,
            "z_fixed": self.z_fixed,
            "final_xy_err": float(xy_err[-1]),
            "max_xy_err": float(np.max(xy_err)),
            "final_z_dev": float(z_dev[-1]),
            "max_z_dev": float(np.max(z_dev)),
            "final_align": float(align[-1]),
            "min_align": float(np.min(align)),
            "final_down_angle_deg": float(angle_deg),
            "final_tool_pos": p_final,
            "final_xy_err_vec": xy_err_vec,
        }

def main():
    demo = IiwaPushingDemo(
        time_step=0.005,
        tool_offset_z=0.10,
        traj_T=3.0,
        sim_T=10.0,
        Kp=300.0,
        Ki=0.0,
        Kd=30.0,
    )

    print("nq =", demo.nq, "nv =", demo.nv, "na =", demo.na)

    # --- Generate target (same logic as your current file) ---
    p0, z_fixed = demo.initial_tool_pose()
    target_xyz = np.array([float(p0[0] + 0.05), float(p0[1] + 0.02), float(z_fixed)])
    print("Initial tool p0 =", p0, "=> z_fixed =", z_fixed, "target_xyz =", target_xyz)

    # --- IK ---
    q_goal, ok, ik_time = demo.solve_ik(target_xyz, ang_tol_deg=10.0)
    print(f"IK success: {ok}, time taken: {ik_time:.6f} seconds")
    if not ok:
        print("IK failed. Try increasing ang_tol_deg or adjusting tolerances/tool frame.")
        return

    # --- IK feasibility check (FK at q_goal) ---
    p_goal, align_goal = demo.fk_tool(q_goal)
    pos_err = np.linalg.norm(p_goal - target_xyz)
    xy_err = np.linalg.norm(p_goal[:2] - target_xyz[:2])
    z_err = abs(p_goal[2] - target_xyz[2])
    print("=== IK FEASIBILITY CHECK (FK at q_goal) ===")
    print("FK tool position at q_goal:", p_goal)
    print(f"Position err: {pos_err:.6e}  (xy {xy_err:.6e}, z {z_err:.6e})")
    print(f"Align score at q_goal: {align_goal:.9f}")
    print(f"Align angle deg at q_goal: {np.degrees(np.arccos(np.clip(align_goal, -1, 1))):.3f}")

    # --- Build diagram + simulate ---
    demo.build_diagram(q_goal=q_goal, target_xyz=target_xyz, z_fixed=z_fixed)
    sim_init_t, sim_t = demo.run()
    print(f"Simulator initialization time: {sim_init_t:.6f} seconds")
    print(f"Simulation time ({demo.sim_T:.1f}s): {sim_t:.6f} seconds")

    # --- Report ---
    stats = demo.summarize()
    print("=== Demo Results ===")
    print("Target XYZ:", stats["target_xyz"], "z_fixed:", stats["z_fixed"])
    print(f"Final XY error: {stats['final_xy_err']:.6f} m (max {stats['max_xy_err']:.6f})")
    print(f"Final z dev:    {stats['final_z_dev']:.6f} m (max {stats['max_z_dev']:.6f})")
    print(f"Align score (1 is perfect down): final {stats['final_align']:.6f}, min {stats['min_align']:.6f}")
    print(f"Final down-axis angle error: {stats['final_down_angle_deg']:.2f} deg")
    print("Final tool position:", stats["final_tool_pos"])
    print("Target position:    ", stats["target_xyz"])
    print("Final XY err vec:   ", stats["final_xy_err_vec"])


if __name__ == "__main__":
    main()
