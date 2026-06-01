import numpy as np
import threading

from drake import lcmt_iiwa_command, lcmt_iiwa_status
from pydrake.all import (
    DiagramBuilder,
    DrakeLcm,
    LcmInterfaceSystem,
    LcmSubscriberSystem,
    LcmPublisherSystem,
    IiwaStatusReceiver,
    IiwaCommandSender,
    IiwaControlMode,
    position_enabled,
    LeafSystem,
    BasicVector,
    Simulator,
    InitializeAutoDiff,
    ConstantVectorSource,
    RollPitchYaw
)

def MakeLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm

def MakeSingleIiwaLcmIO(builder, lcm, lcm_channel_suffix="", export_position_input=False):
    control_mode = IiwaControlMode.kPositionAndTorque
    assert position_enabled(control_mode)

    cmd_sender = builder.AddSystem(IiwaCommandSender(control_mode=control_mode))
    cmd_pub = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel="IIWA_COMMAND" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_command,
            lcm=lcm,
            publish_period=0.005,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(cmd_sender.get_output_port(), cmd_pub.get_input_port())

    # IMPORTANT: only export if you explicitly want an external input
    if export_position_input:
        builder.ExportInput(cmd_sender.get_position_input_port(), "position")

    status_recv = builder.AddSystem(IiwaStatusReceiver())
    status_sub = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_status,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )
    builder.Connect(status_sub.get_output_port(), status_recv.get_input_port())

    # If you still want a diagram output named position_measured, keep this:
    builder.ExportOutput(status_recv.get_position_measured_output_port(), "position_measured")

    return cmd_sender, status_recv

class JointPositionStreamer(LeafSystem):
    """
    Inputs:
      - q_measured (7)
      - q_goal     (7)
      - enable     (1)  (0: hold last q_cmd, 1: update toward q_goal)
    Output:
      - q_cmd      (7)  (stateful, rate-limited)
    """
    def __init__(self, period_sec=0.005, max_delta_per_step=0.01):
        super().__init__()
        self._max_delta = float(max_delta_per_step)

        self.DeclareVectorInputPort("q_measured", 7)
        self.DeclareVectorInputPort("q_goal", 7)
        self.DeclareVectorInputPort("enable", 1)

        state_idx = self.DeclareDiscreteState(7)  # q_cmd
        self.DeclareStateOutputPort("q_cmd", state_idx)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=float(period_sec),
            offset_sec=0.0,
            update=self._update,
        )

    def _update(self, context, discrete_state):
        enable = float(self.get_input_port(2).Eval(context)[0])
        q_meas = np.asarray(self.get_input_port(0).Eval(context), dtype=float)
        q_goal = np.asarray(self.get_input_port(1).Eval(context), dtype=float)

        q_cmd = discrete_state.get_vector().CopyToVector()

        # Initialize q_cmd from q_measured on first tick
        if (not np.isfinite(q_cmd).all()) or (np.linalg.norm(q_cmd) < 1e-12):
            q_cmd = q_meas.copy()

        if enable < 0.5:
            # Hold current command
            discrete_state.set_value(q_cmd)
            return

        # --- NEW: vector-norm–clipped joint-space step ---
        dq = q_goal - q_cmd
        dq_norm = np.linalg.norm(dq)

        if dq_norm > self._max_delta:
            dq = dq * (self._max_delta / dq_norm)

        q_next = q_cmd + dq
        discrete_state.set_value(q_next)


from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame

class ToolPoseMetrics(LeafSystem):
    """
    Input:  q (7)
    Output: metrics (4) = [x, y, z, align_score]
      align_score = dot(tool_z_world, -world_z) in [ -1, 1 ]
    """
    def __init__(self, plant: MultibodyPlant, model_instance, tool_frame):
        super().__init__()
        self._plant = plant
        self._model = model_instance
        self._tool = tool_frame
        self._ctx = self._plant.CreateDefaultContext()

        self.DeclareVectorInputPort("q", 7)
        self.DeclareVectorOutputPort("metrics", BasicVector(4), self._calc)

    def _calc(self, context, output):
        q = self.get_input_port(0).Eval(context)
        self._plant.SetPositions(self._ctx, self._model, q)

        X_WT = self._plant.CalcRelativeTransform(self._ctx, self._plant.world_frame(), self._tool)
        p = np.array(X_WT.translation(), dtype=float)
        R = X_WT.rotation().matrix()
        tool_z_world = R[:, 2]
        align_score = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))

        output.SetFromVector(np.array([p[0], p[1], p[2], align_score], dtype=float))


def BuildFkPlantWithToolFrame(
    urdf_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
    tool_offset_z=0.10,
):
    """
    Minimal plant (no SceneGraph) for FK computations only.
    Tool frame definition matches your sim: link7 + XRotation(pi) + z offset. :contentReference[oaicite:6]{index=6}
    """
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    model_instance = parser.AddModelsFromUrl(urdf_url)[0]

    # Weld base (fixed-base IIWA) :contentReference[oaicite:7]{index=7}
    # base_body = plant.GetBodyByName("base", model_instance)
    # plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

    # Weld model base to world. Different models name the base body differently.
    try:
        base_body = plant.GetBodyByName("base", model_instance)          # common in some URDFs
    except RuntimeError:
        base_body = plant.GetBodyByName("iiwa_link_0", model_instance)   # iiwa SDFs

    plant.WeldFrames(plant.world_frame(), base_body.body_frame(), RigidTransform())

    # Tool frame :contentReference[oaicite:8]{index=8}
    link7 = plant.GetFrameByName("iiwa_link_7", model_instance)
    X_7T = RigidTransform(
        RotationMatrix.Identity(),
        [0.0, 0.0, float(tool_offset_z)],
    )
    tool_frame = plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))

    plant.Finalize()
    return plant, model_instance, tool_frame

def MakeIiwaRealStation(
    lcm_channel_suffix="",
    urdf_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
    tool_offset_z=0.10,
    period_sec=0.005,
    max_delta_per_step=0.01,
    use_ik=False,
    ik_update_period_sec=0.05,
    ang_tol_deg=5.0,
):
    builder = DiagramBuilder()
    lcm = MakeLcm(builder)

    cmd_sender, status_recv = MakeSingleIiwaLcmIO(
        builder, lcm, lcm_channel_suffix=lcm_channel_suffix, export_position_input=False
    )

    streamer = builder.AddSystem(
        JointPositionStreamer(period_sec=period_sec, max_delta_per_step=max_delta_per_step)
    )
    builder.Connect(
        status_recv.get_position_measured_output_port(),
        streamer.GetInputPort("q_measured"),
    )

    # Streamer internal enable: keep updates running; safety is enforced by CommandGate.
    _streamer_enable = builder.AddSystem(ConstantVectorSource([1.0]))
    builder.Connect(_streamer_enable.get_output_port(), streamer.GetInputPort("enable"))


    # Always allow external joint-goal input (for init/homing)
    # (even if use_ik=True)
    q_goal_in = streamer.GetInputPort("q_goal")  # default wiring when use_ik=False

    if use_ik:
        # IK manager
        ik_mgr = builder.AddSystem(
            IiwaIkGoalManager(
                urdf_url=urdf_url,
                tool_offset_z=tool_offset_z,
                ang_tol_deg=ang_tol_deg,
                update_period_sec=ik_update_period_sec,
            )
        )
        builder.Connect(
            status_recv.get_position_measured_output_port(),
            ik_mgr.GetInputPort("q_measured"),
        )

        # Selector between direct joint goal and IK-computed goal
        selector = builder.AddSystem(QGoalSelector())

        # Export inputs:
        builder.ExportInput(ik_mgr.GetInputPort("target_xyz"), "target_xyz")
        builder.ExportInput(selector.GetInputPort("mode"), "mode")              # 0=joint, 1=IK
        builder.ExportInput(selector.GetInputPort("q_goal_direct"), "q_goal")   # joint init target

        # Wire IK goal into selector
        builder.Connect(ik_mgr.GetOutputPort("q_goal"), selector.GetInputPort("q_goal_ik"))

        # Wire selector output into streamer q_goal
        builder.Connect(selector.GetOutputPort("q_goal_out"), streamer.GetInputPort("q_goal"))

        # Export IK status
        builder.ExportOutput(ik_mgr.GetOutputPort("ik_status"), "ik_status")
        builder.ExportOutput(ik_mgr.GetOutputPort("q_goal"), "q_goal")

    else:
        # No IK: external q_goal goes straight to streamer
        builder.ExportInput(streamer.GetInputPort("q_goal"), "q_goal")

    # Command out (gated for safety): when enable=0, hold q_measured; when enable=1, follow streamer q_cmd.
    gate = builder.AddSystem(CommandGate())
    builder.Connect(streamer.GetOutputPort("q_cmd"), gate.GetInputPort("q_cmd_in"))
    builder.Connect(status_recv.get_position_measured_output_port(), gate.GetInputPort("q_hold"))
    builder.ExportInput(gate.GetInputPort("enable"), "enable")
    builder.Connect(gate.GetOutputPort("q_cmd_out"), cmd_sender.get_position_input_port())

    # Export measured q
    builder.ExportOutput(status_recv.get_position_measured_output_port(), "q_measured")

    # Tool metrics
    fk_plant, fk_model, fk_tool = BuildFkPlantWithToolFrame(
        urdf_url=urdf_url, tool_offset_z=tool_offset_z
    )
    metrics_sys = builder.AddSystem(ToolPoseMetrics(fk_plant, fk_model, fk_tool))
    builder.Connect(
        status_recv.get_position_measured_output_port(),
        metrics_sys.GetInputPort("q"),
    )
    builder.ExportOutput(metrics_sys.GetOutputPort("metrics"), "tool_metrics")

    return builder.Build()

import numpy as np
import time

from pydrake.all import (
    LeafSystem, BasicVector,
    MultibodyPlant, Parser,
    RigidTransform, RotationMatrix,
    InverseKinematics, Solve,
)
from pydrake.multibody.tree import FixedOffsetFrame


class QGoalSelector(LeafSystem):
    """
    Inputs:
      - mode (1): 0 -> use q_goal_direct, 1 -> use q_goal_ik
      - q_goal_direct (7)
      - q_goal_ik (7)
    Output:
      - q_goal_out (7)
    """
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("mode", 1)
        self.DeclareVectorInputPort("q_goal_direct", 7)
        self.DeclareVectorInputPort("q_goal_ik", 7)
        self.DeclareVectorOutputPort("q_goal_out", BasicVector(7), self._calc)

    def _calc(self, context, output):
        mode = float(self.get_input_port(0).Eval(context)[0])
        if mode >= 0.5:
            q_ik = self.get_input_port(2).Eval(context)
            output.SetFromVector(q_ik)
        else:
            q_direct = self.get_input_port(1).Eval(context)
            output.SetFromVector(q_direct)

class CommandGate(LeafSystem):
    """
    Gate joint commands for safety.

    Inputs:
      - q_cmd_in   (7): upstream command (e.g., from JointPositionStreamer)
      - q_hold     (7): safe hold command (typically q_measured)
      - enable     (1): 0 -> output q_hold, 1 -> output q_cmd_in
    Output:
      - q_cmd_out  (7)
    """
    def __init__(self):
        super().__init__()
        self.DeclareVectorInputPort("q_cmd_in", 7)
        self.DeclareVectorInputPort("q_hold", 7)
        self.DeclareVectorInputPort("enable", 1)
        self.DeclareVectorOutputPort("q_cmd_out", BasicVector(7), self._calc)

    def _calc(self, context, output):
        enable = float(self.get_input_port(2).Eval(context)[0])
        if enable >= 0.5:
            q = self.get_input_port(0).Eval(context)
        else:
            q = self.get_input_port(1).Eval(context)
        output.SetFromVector(q)

class IiwaIkGoalManager(LeafSystem):
    """
    Event-driven IK manager:
      - runs IK at a low rate (e.g., 10-20 Hz)
      - holds last good q_goal
      - outputs q_goal for a fast streaming controller (200 Hz)
    """
    def __init__(
        self,
        urdf_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
        tool_offset_z=0.10,
        ang_tol_deg=5.0,
        update_period_sec=0.05,      # 20 Hz IK loop
        target_change_tol=1e-4,      # meters
        seed_blend=0.0,              # optional: blend last q_goal with q_measured
    ):
        super().__init__()
        self._ang_tol_deg = float(ang_tol_deg)
        self._update_period = float(update_period_sec)
        self._target_tol = float(target_change_tol)
        self._seed_blend = float(seed_blend)

        # Build a tiny IK plant (no SceneGraph) for solving IK
        self._plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self._plant)
        self._model = parser.AddModelsFromUrl(urdf_url)[0]
        self._model_name = self._plant.GetModelInstanceName(self._model)

        # Weld model base to world. Different models name the base body differently.
        try:
            base_body = self._plant.GetBodyByName("base", self._model)          # common in some URDFs
        except RuntimeError:
            base_body = self._plant.GetBodyByName("iiwa_link_0", self._model)   # iiwa SDFs

        self._plant.WeldFrames(self._plant.world_frame(), base_body.body_frame(), RigidTransform())

        link7 = self._plant.GetFrameByName("iiwa_link_7", self._model)
        X_7T = RigidTransform(
            RotationMatrix.Identity(),
            [0.0, 0.0, float(tool_offset_z)],
        )
        self._tool = self._plant.AddFrame(FixedOffsetFrame("tool_frame", link7, X_7T))
        self._plant.Finalize()
        self._plant_ad = self._plant.ToAutoDiffXd()
        self._ctx_ad = self._plant_ad.CreateDefaultContext()
        self._model_ad = self._plant_ad.GetModelInstanceByName(self._model_name)
        self._tool_ad = self._plant_ad.GetFrameByName(self._tool.name(), self._model_ad)



        self._nq = self._plant.num_positions(self._model)

        # Ports
        self.DeclareVectorInputPort("target_xyz", 3)
        self.DeclareVectorInputPort("q_measured", self._nq)

        # Discrete state:
        #   [0:3]     last_target_xyz
        #   [3:3+nq]  q_goal
        #   [..]      success_flag (0/1)
        #   [..]      last_solve_time_sec
        self._idx_last_target = self.DeclareDiscreteState(3)
        self._idx_q_goal = self.DeclareDiscreteState(self._nq)
        self._idx_meta = self.DeclareDiscreteState(2)  # [success_flag, last_solve_time]

        self.DeclareStateOutputPort("q_goal", self._idx_q_goal)
        self.DeclareVectorOutputPort("ik_status", BasicVector(2), self._calc_status)

        # Periodic update (slow loop)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self._update_period,
            offset_sec=0.0,
            update=self._update,
        )

    def _calc_status(self, context, output):
        meta = context.get_discrete_state(self._idx_meta).get_value()
        output.SetFromVector(meta)

    def _update(self, context, discrete_state):
        target = np.array(self.get_input_port(0).Eval(context), dtype=float).reshape(3)
        q_meas = np.array(self.get_input_port(1).Eval(context), dtype=float).reshape(self._nq)

        last_target = discrete_state.get_vector(self._idx_last_target).CopyToVector()
        q_goal_prev = discrete_state.get_vector(self._idx_q_goal).CopyToVector()

        # Determine if we should solve IK again
        need_solve = True
        if np.isfinite(last_target).all():
            if np.linalg.norm(target - last_target) <= self._target_tol:
                need_solve = False

        if not need_solve:
            # Keep q_goal as-is, keep meta as-is.
            return

        # Seed choice: measured, or blended measured+previous
        if np.isfinite(q_goal_prev).all():
            q_seed = (1.0 - self._seed_blend) * q_meas + self._seed_blend * q_goal_prev
        else:
            q_seed = q_meas

        t0 = time.time()
        q_sol, ok = self._solve_ik(target_xyz=target, q_seed=q_seed)
        t1 = time.time()

        # if ok:
        #     ok_feas, _ = self.check_ik_feasibility(q_sol, target, ang_tol_deg=self._ang_tol_deg)
        #     ok = bool(ok_feas)

        # Update state
        discrete_state.get_mutable_vector(self._idx_last_target).set_value(target)

        if ok:
            discrete_state.get_mutable_vector(self._idx_q_goal).set_value(q_sol)
            discrete_state.get_mutable_vector(self._idx_meta).set_value(
                np.array([1.0, float(t1 - t0)], dtype=float)
            )
        else:
            # If IK fails, keep the previous q_goal (don’t jerk the robot)
            discrete_state.get_mutable_vector(self._idx_meta).set_value(
                np.array([0.0, float(t1 - t0)], dtype=float)
            )

    def _solve_ik(self, target_xyz, q_seed):
        """
        Cached AutoDiff version of your working IK:
        - position box (tol_xy/tol_z)
        - tool +z aligned with world -z within ang_tol
        - soft position-centering cost via cached AutoDiff plant/context
        """
        plant = self._plant
        model_instance = self._model
        tool_frame = self._tool

        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

        # Seed context (Float plant) at q_seed
        plant_context = plant.CreateDefaultContext()
        plant.SetPositions(plant_context, model_instance, q_seed)

        ik = InverseKinematics(plant, plant_context)
        q = ik.q()
        prog = ik.prog()

        # --- tolerances: identical to your working version ---
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

        Q = 1.0 * np.eye(len(q_seed))
        prog.AddQuadraticErrorCost(Q, q_seed, q)

        # # Axis alignment: tool +z aligned with world -z (yaw free)
        # theta = float(self._ang_tol_deg) * np.pi / 180.0
        # na_A = np.array([[0.0], [0.0], [-1.0]])  # world -z
        # nb_B = np.array([[0.0], [0.0], [ 1.0]])  # tool +z
        # ik.AddAngleBetweenVectorsConstraint(
        #     plant.world_frame(), na_A,
        #     tool_frame, nb_B,
        #     0.0, theta
        # )


        roll = 0.0
        pitch = np.pi     # example; depends on your tool frame definition
        yaw = 0.0         # choose a fixed yaw (Sapien uses fixed self.rpy)

        R_WT_des = RotationMatrix(RollPitchYaw(roll, pitch, yaw))

        # small tolerance (degrees -> radians). Start a bit loose then tighten.
        theta_bound = np.deg2rad(2.0)

        ik.AddOrientationConstraint(
            frameAbar=plant.world_frame(),
            R_AbarA=R_WT_des,
            frameBbar=tool_frame,
            R_BbarB=RotationMatrix(),   # identity: tool_frame itself
            theta_bound=theta_bound,
        )

        # Soft cost: prefer center of the box (minimize ||p(q)-target||^2)
        w_pos = 1000.0

        plant_ad = self._plant_ad
        ctx_ad = self._ctx_ad
        model_ad = self._model_ad
        tool_ad = self._tool_ad

        def pos_cost(x):
            # x can be float ndarray or AutoDiffXd ndarray (SNOPT).
            if getattr(x, "dtype", None) == object:
                q_ad = x
            else:
                x = np.asarray(x, dtype=float).reshape((-1, 1))
                q_ad = InitializeAutoDiff(x, num_derivatives=x.shape[0]).reshape((-1,))

            plant_ad.SetPositions(ctx_ad, model_ad, q_ad)
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

        return np.asarray(result.GetSolution(q), dtype=float), True


    def fk_tool(self, q):
        """
        Forward-kinematics for the tool frame at joint configuration q.
        Returns:
        p_goal: (3,)
        align_score: float = dot(tool +z in world, world -z)
        """
        q = np.asarray(q, dtype=float).reshape(self._nq)
        ctx = self._plant.CreateDefaultContext()
        self._plant.SetPositions(ctx, self._model, q)

        X_WT = self._plant.CalcRelativeTransform(ctx, self._plant.world_frame(), self._tool)
        p = np.asarray(X_WT.translation(), dtype=float)

        # tool +z expressed in world
        R = X_WT.rotation().matrix()  # (3,3)
        tool_z_W = R @ np.array([0.0, 0.0, 1.0], dtype=float)
        align_score = float(tool_z_W.dot(np.array([0.0, 0.0, -1.0], dtype=float)))

        return p, align_score


    def check_ik_feasibility(self, q_goal, target_xyz, *, ang_tol_deg=None,
                            tol_xy=4e-3, tol_z=1.5e-3):
        """
        Check feasibility by evaluating FK at q_goal and comparing against:
        - position box tolerance (tol_xy/tol_z)
        - alignment tolerance ang_tol_deg

        Returns:
        ok_feas: bool
        info: dict with pos_err/xy_err/z_err/align_score/align_angle_deg
        """
        if ang_tol_deg is None:
            ang_tol_deg = self._ang_tol_deg

        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        p_goal, align_score = self.fk_tool(q_goal)
        pos_err = float(np.linalg.norm(p_goal - target_xyz))
        xy_err = float(np.linalg.norm(p_goal[:2] - target_xyz[:2]))
        z_err = float(abs(p_goal[2] - target_xyz[2]))

        # alignment angle in degrees
        c = float(np.clip(align_score, -1.0, 1.0))
        align_angle_deg = float(np.degrees(np.arccos(c)))

        ok_pos = (xy_err <= float(tol_xy)) and (z_err <= float(tol_z))
        ok_align = (align_angle_deg <= float(ang_tol_deg))
        ok_feas = bool(ok_pos and ok_align)

        info = {
            "p_goal": p_goal,
            "pos_err": pos_err,
            "xy_err": xy_err,
            "z_err": z_err,
            "align_score": float(align_score),
            "align_angle_deg": align_angle_deg,
            "ok_pos": bool(ok_pos),
            "ok_align": bool(ok_align),
        }
        return ok_feas, info


class IiwaHardwareEnv:
    """Safe-by-default wrapper around the real IIWA station.

    Design goals:
      - No unexpected motion on startup (commands are gated until env.start()).
      - Simple public API for demos and task pipelines.
      - Keeps planning/IK separate from execution loops.
    """
    def __init__(
        self,
        *,
        use_ik=False,
        lcm_channel_suffix="",
        urdf_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
        tool_offset_z=0.10,
        period_sec=0.005,
        max_delta_per_step=0.01,
        realtime_rate=1.0,
        ik_update_period_sec=0.05,
        ang_tol_deg=2.0,
    ):
        self.use_ik = bool(use_ik)
        self.station = MakeIiwaRealStation(
            lcm_channel_suffix=lcm_channel_suffix,
            urdf_url=urdf_url,
            tool_offset_z=tool_offset_z,
            period_sec=period_sec,
            max_delta_per_step=max_delta_per_step,
            use_ik=self.use_ik,
            ik_update_period_sec=ik_update_period_sec,
            ang_tol_deg=ang_tol_deg,
        )
        self.ctx = self.station.CreateDefaultContext()


        # Cache ports
        self._port_q_meas = self.station.GetOutputPort("q_measured")
        self._port_tool = self.station.GetOutputPort("tool_metrics")
        self._port_enable = self.station.GetInputPort("enable")

        self._port_ik_status = None
        if self.use_ik:
            self._port_mode = self.station.GetInputPort("mode")
            self._port_q_goal = self.station.GetInputPort("q_goal")
            self._port_target = self.station.GetInputPort("target_xyz")
            self._port_ik_status = self.station.GetOutputPort("ik_status")
            # Provide dummy values so every required input is connected.
            self._port_mode.FixValue(self.ctx, np.array([0.0], dtype=float))          # start in joint mode
            self._port_q_goal.FixValue(self.ctx, np.zeros(7, dtype=float))           # dummy
            self._port_target.FixValue(self.ctx, np.zeros(3, dtype=float))           # dummy
        else:
            self._port_q_goal = self.station.GetInputPort("q_goal")

        # Safety gate: disabled by default (no motion).
        self._port_enable.FixValue(self.ctx, np.array([0.0], dtype=float))
        self._started = False

        self.sim = Simulator(self.station, self.ctx)
        self.sim.set_target_realtime_rate(float(realtime_rate))
        self.sim.Initialize()

        # after self.station / self.ctx are created:
        self._fk_plant, self._fk_model, self._fk_tool = BuildFkPlantWithToolFrame(
            urdf_url=urdf_url, tool_offset_z=tool_offset_z
        )
        self._fk_ctx = self._fk_plant.CreateDefaultContext()

        # Threading model:
        # - all simulator/context access is guarded by _sim_lock
        # - when background control is running, only that thread may advance time
        # - the foreground thread only updates the latest-target buffer
        self._sim_lock = threading.RLock()
        self._target_lock = threading.Lock()
        self._bg_thread = None
        self._bg_stop_event = None
        self._bg_control_period = None
        self._bg_thread_ident = None
        self._latest_target_xyz = None
        self._latest_target_set = False
        self._bg_first_target_logged = False
        self._obs_lock = threading.Lock()
        self._obs_cache_valid = False
        self._cached_q_measured = None
        self._cached_tool_metrics = None
        self._cached_ik_status = None

    def _ik_mgr_fk_tool(self, q):
        q = np.asarray(q, dtype=float).reshape(7)
        self._fk_plant.SetPositions(self._fk_ctx, self._fk_model, q)
        X_WT = self._fk_plant.CalcRelativeTransform(self._fk_ctx, self._fk_plant.world_frame(), self._fk_tool)
        p = np.asarray(X_WT.translation(), dtype=float)
        R = X_WT.rotation().matrix()
        tool_z_world = R[:, 2]
        align = float(np.dot(tool_z_world, np.array([0.0, 0.0, -1.0])))
        return p, align



    # ---------- time / stepping ----------
    def time(self):
        with self._sim_lock:
            return float(self.ctx.get_time())

    def _manual_stepping_allowed(self):
        if not self.is_background_control_running():
            return
        if threading.get_ident() != self._bg_thread_ident:
            raise RuntimeError(
                "Background control owns simulator stepping; stop it before calling step()/advance_to() from another thread."
            )

    def step(self, dt):
        with self._sim_lock:
            self._manual_stepping_allowed()
            self.sim.AdvanceTo(float(self.ctx.get_time()) + float(dt))

    def advance_to(self, t_abs):
        with self._sim_lock:
            self._manual_stepping_allowed()
            self.sim.AdvanceTo(float(t_abs))

    # ---------- observations ----------
    def _refresh_obs_cache_locked(self):
        q_measured = np.asarray(self._port_q_meas.Eval(self.ctx), dtype=float).copy()
        tool_metrics = np.asarray(self._port_tool.Eval(self.ctx), dtype=float).copy()
        ik_status = None
        if self.use_ik:
            ik_status = np.asarray(self._port_ik_status.Eval(self.ctx), dtype=float).copy()

        with self._obs_lock:
            self._cached_q_measured = q_measured
            self._cached_tool_metrics = tool_metrics
            self._cached_ik_status = ik_status
            self._obs_cache_valid = True

    def get_q_measured(self):
        if self.is_background_control_running():
            with self._obs_lock:
                if self._obs_cache_valid and self._cached_q_measured is not None:
                    return self._cached_q_measured.copy()
        with self._sim_lock:
            return np.asarray(self._port_q_meas.Eval(self.ctx), dtype=float)

    def get_q_goal_ik(self):
        """Return current IK manager output q_goal (7,) if use_ik=True."""
        if not self.use_ik:
            raise RuntimeError("get_q_goal_ik requires use_ik=True.")
        with self._sim_lock:
            return np.asarray(self.station.GetOutputPort("q_goal").Eval(self.ctx), dtype=float)

    def get_tool_metrics_at_q(self, q):
        """Compute tool metrics [x,y,z,align] at an arbitrary joint config q (FK only)."""
        # Reuse the same FK plant/tool frame used by ToolPoseMetrics would be ideal,
        # but simplest is to instantiate a ToolPoseMetrics-like helper once.
        # If you already have a cached FK plant in env, use it; otherwise keep it minimal:
        p, align = self._ik_mgr_fk_tool(q)  # <-- see note below
        return np.array([p[0], p[1], p[2], float(align)], dtype=float)

    def get_tool_metrics_ik_goal(self):
        """Tool metrics evaluated at IK final pose q_goal."""
        qg = self.get_q_goal_ik()
        return self.get_tool_metrics_at_q(qg)

    def get_tool_metrics_in_world_mm(self):
        """Tool metrics [x,y,z,align] in world frame (rotated from robot frame)."""
        tm = self.get_tool_metrics()
        # robot frame rotates CCW 90 deg around z axis w.r.t. workspace frame
        x_world = -tm[1] * 1000.0
        y_world = tm[0] * 1000.0
        z_world = tm[2] * 1000.0
        align = tm[3]
        return np.array([x_world, y_world, z_world, align], dtype=float)

    def get_tool_metrics(self):
        if self.is_background_control_running():
            with self._obs_lock:
                if self._obs_cache_valid and self._cached_tool_metrics is not None:
                    return self._cached_tool_metrics.copy()
        with self._sim_lock:
            return np.asarray(self._port_tool.Eval(self.ctx), dtype=float)

    def get_ik_status(self):
        if not self.use_ik:
            return None
        if self.is_background_control_running():
            with self._obs_lock:
                if self._obs_cache_valid and self._cached_ik_status is not None:
                    return self._cached_ik_status.copy()
        with self._sim_lock:
            return np.asarray(self._port_ik_status.Eval(self.ctx), dtype=float)

    # ---------- internal safety helpers ----------
    def _set_enable(self, on: bool):
        self._port_enable.FixValue(self.ctx, np.array([1.0 if on else 0.0], dtype=float))

    def _set_mode_joint(self):
        if self.use_ik:
            self._port_mode.FixValue(self.ctx, np.array([0.0], dtype=float))

    def _set_mode_ik(self):
        if self.use_ik:
            self._port_mode.FixValue(self.ctx, np.array([1.0], dtype=float))

    def _set_q_goal(self, q_goal):
        self._port_q_goal.FixValue(self.ctx, np.asarray(q_goal, dtype=float).reshape(7))

    def _set_target_xyz(self, target_xyz):
        if not self.use_ik:
            raise RuntimeError("set_target_xyz requires use_ik=True.")
        self._port_target.FixValue(self.ctx, np.asarray(target_xyz, dtype=float).reshape(3))

    def _wait_for_status(self, timeout_sec=2.0, dt=0.02, eps=1e-6):
        """Advance time until we see a nontrivial q_measured."""
        t_end = self.time() + float(timeout_sec)
        while self.time() < t_end:
            self.step(dt)
            q = self.get_q_measured()
            if np.isfinite(q).all() and np.linalg.norm(q) > eps:
                return q
        raise RuntimeError("No nontrivial IIWA_STATUS received within timeout.")

    def _ik_ok(self):
        st = self.get_ik_status()
        return (st is not None) and (int(st[0]) == 1)

    def _wait_for_ik_success(self, timeout_sec=2.0, dt=0.001):
        t_end = self.time() + float(timeout_sec)
        while self.time() < t_end:
            self.step(dt)
            if self._ik_ok():
                return True
        return False

    # ---------- public high-level API ----------
    def start(self, timeout_sec=2.0):
        """Safe startup: latch current measured joints, hold them, then enable motion."""
        with self._sim_lock:
            if self._started:
                return

            # Keep in joint mode and hold pose before enabling.
            if self.use_ik:
                self._set_mode_joint()

            q0 = self._wait_for_status(timeout_sec=timeout_sec)

            # Hold current pose as the direct goal.
            self._set_q_goal(q0)

            # Now it's safe to allow commands through.
            self._set_enable(True)
            self._started = True

    def home(self, q_home, timeout_sec=10.0, **kwargs):
        """Move to a specified joint pose (safe init/homing)."""
        self.stop_background_control()
        self.start()
        if self.use_ik:
            self._set_mode_joint()
        self._set_q_goal(q_home)
        return self.wait_until_joint_reached(q_home, timeout_sec=timeout_sec, **kwargs)

    def move_to_target_xyz_in_world_mm(self, target_xyz_world_mm, timeout_sec=10.0, **kwargs):
        # robot frame rotates CCW 90 deg around z axis w.r.t. workspace frame
        target_xyz_robot = np.array([
            target_xyz_world_mm[1] / 1000.0,
            -target_xyz_world_mm[0] / 1000.0,
            target_xyz_world_mm[2] / 1000.0,
        ], dtype=float)
        return self.move_to_target_xyz(target_xyz_robot, timeout_sec=timeout_sec, **kwargs)

    def set_latest_target_xyz_in_world_mm(self, target_xyz_world_mm):
        target_xyz_robot = np.array([
            target_xyz_world_mm[1] / 1000.0,
            -target_xyz_world_mm[0] / 1000.0,
            target_xyz_world_mm[2] / 1000.0,
        ], dtype=float)
        self.set_latest_target_xyz(target_xyz_robot)

    def move_to_target_xyz(self, target_xyz, timeout_sec=10.0, **kwargs):
        """IK move: set target, wait for IK to succeed, switch to IK mode, then wait until reached."""
        if not self.use_ik:
            raise RuntimeError("move_to_target_xyz requires use_ik=True.")
        self.stop_background_control()
        self.start()

        self._set_target_xyz(target_xyz)

        # Allow IK manager to compute at least one feasible goal before switching to IK mode.
        ok = self._wait_for_ik_success(timeout_sec=min(2.0, float(timeout_sec)))
        if not ok:
            raise RuntimeError("IK did not succeed (target likely infeasible).")
        self._set_mode_ik()

        return self.wait_until_reached(target_xyz, timeout_sec=timeout_sec, **kwargs)

    def set_target_xyz(self, target_xyz):
        """For interactive demos: update target; env handles safe start."""
        if not self.use_ik:
            raise RuntimeError("set_target_xyz requires use_ik=True.")
        # While background control is running, the foreground thread only updates
        # the shared latest-target buffer; the background thread owns simulator stepping.
        if self.is_background_control_running():
            self.set_latest_target_xyz(target_xyz)
            return

        self.start()
        with self._sim_lock:
            self._set_target_xyz(target_xyz)
            # Optionally auto-switch once IK is feasible.
            if self._ik_ok():
                self._set_mode_ik()

    def set_latest_target_xyz(self, target_xyz):
        """Update the shared latest Cartesian target; the background loop consumes it."""
        if not self.use_ik:
            raise RuntimeError("set_latest_target_xyz requires use_ik=True.")
        if not self._started:
            raise RuntimeError(
                "set_latest_target_xyz requires the env to be started already. "
                "Call start() or start_background_control() first."
            )
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        with self._target_lock:
            self._latest_target_xyz = target_xyz.copy()
            self._latest_target_set = True

    def is_background_control_running(self):
        return self._bg_thread is not None and self._bg_thread.is_alive()

    def _background_control_loop(self):
        self._bg_thread_ident = threading.get_ident()
        print(f"[iiwa] Background control loop started (dt={self._bg_control_period:.3f}s)")
        try:
            while not self._bg_stop_event.is_set():
                with self._target_lock:
                    has_target = self._latest_target_set
                    target_xyz = None if self._latest_target_xyz is None else self._latest_target_xyz.copy()

                with self._sim_lock:
                    if has_target and target_xyz is not None:
                        self._set_target_xyz(target_xyz)
                    self.sim.AdvanceTo(float(self.ctx.get_time()) + float(self._bg_control_period))
                    if has_target and self._ik_ok():
                        self._set_mode_ik()
                    self._refresh_obs_cache_locked()

                if has_target and (not self._bg_first_target_logged):
                    print(f"[iiwa] Background control received first target: {target_xyz}")
                    self._bg_first_target_logged = True
        finally:
            self._bg_thread_ident = None
            print("[iiwa] Background control loop stopped")

    def start_background_control(self, control_period_sec=0.01):
        """Start a daemon thread that owns simulator stepping for latest-target IK tracking."""
        if not self.use_ik:
            raise RuntimeError("start_background_control requires use_ik=True.")
        if self.is_background_control_running():
            return

        self.start()
        self._bg_control_period = float(control_period_sec)
        self._bg_stop_event = threading.Event()
        self._bg_first_target_logged = False
        self._bg_thread = threading.Thread(
            target=self._background_control_loop,
            name="iiwa_background_control",
            daemon=True,
        )
        self._bg_thread.start()

    def stop_background_control(self, join_timeout_sec=2.0):
        """Stop the background stepping thread cleanly."""
        if not self.is_background_control_running():
            return

        print("[iiwa] Stopping background control loop...")
        self._bg_stop_event.set()
        self._bg_thread.join(timeout=float(join_timeout_sec))
        if self._bg_thread.is_alive():
            raise RuntimeError("Background control loop did not stop cleanly.")
        self._bg_thread = None
        self._bg_stop_event = None

    # ---------- termination helpers ----------
    def reached(self, target_xyz, xy_tol=5e-3, z_tol=3e-3, align_tol=0.985):
        tool = self.get_tool_metrics()
        return is_reached(tool, target_xyz, xy_tol=xy_tol, z_tol=z_tol, align_tol=align_tol)

    def wait_until_reached(
        self,
        target_xyz,
        *,
        timeout_sec=10.0,
        check_dt=0.005,
        hold_count_required=5,
        xy_tol=5e-3,
        z_tol=3e-3,
        align_tol=0.985,
        print_dt=0.2,
        verbose=True,
        tol_scale=1,
    ):
        xy_tol = float(xy_tol) * float(tol_scale)
        z_tol = float(z_tol) * float(tol_scale)
        align_tol = float(align_tol) * float(tol_scale)

        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        t_end = self.time() + float(timeout_sec)

        hold = 0
        t_next_print = self.time()

        while self.time() < t_end:
            self.step(check_dt)

            tool = self.get_tool_metrics()
            if is_reached(tool, target_xyz, xy_tol=xy_tol, z_tol=z_tol, align_tol=align_tol):
                hold += 1
            else:
                hold = 0

            if verbose and self.time() >= t_next_print:
                p = tool[:3]
                align = float(tool[3])
                xy_err = float(np.linalg.norm(p[:2] - target_xyz[:2]))
                z_err = float(abs(p[2] - target_xyz[2]))
                if self.use_ik:
                    ik_status = self.get_ik_status()
                    ik_ok = int(ik_status[0])
                    ik_ms = ik_status[1] * 1e3
                    
                    print(
                        f"t={self.time():.2f} tool=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                        f"xy_err={xy_err*1e3:.1f}mm z_err={z_err*1e3:.1f}mm align={align:+.3f} "
                        f"IK_ok={ik_ok} IK_dt={ik_ms:.1f}ms hold={hold}/{hold_count_required}"
                    )

                    qg = self.get_q_goal_ik()
                    tool_g = self.get_tool_metrics_ik_goal()  # FK at IK final pose
                    p_g = tool_g[:3]
                    align_g = tool_g[3]
                    print(f"IK_goal tool=({p_g[0]:+.3f},{p_g[1]:+.3f},{p_g[2]:+.3f}) align={align_g:+.3f} "
                        # f"| ||Δq_goal||={np.linalg.norm(qg - qg_prev):.4f}"
                        )
                    qg_prev = qg
                else:
                    print(
                        f"t={self.time():.2f} tool=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                        f"xy_err={xy_err*1e3:.1f}mm z_err={z_err*1e3:.1f}mm align={align:+.3f} "
                        f"hold={hold}/{hold_count_required}"
                    )
                t_next_print += print_dt

            if hold >= hold_count_required:
                return {"reached": True, "t_end": self.time(), "hold": hold}

        return {"reached": False, "t_end": self.time(), "hold": hold}

    def wait_until_joint_reached(self, q_goal, *, timeout_sec=10.0, check_dt=0.005, tol=1e-2, verbose=False):
        q_goal = np.asarray(q_goal, dtype=float).reshape(7)
        t_end = self.time() + float(timeout_sec)
        while self.time() < t_end:
            self.step(check_dt)
            q = self.get_q_measured()
            tool = self.get_tool_metrics()
            err = float(np.linalg.norm(q - q_goal))
            if verbose:
                print(f"t={self.time():.2f} ||q-q_goal||={err:.4f}, tool=({tool[0]:+.3f},{tool[1]:+.3f},{tool[2]:+.3f}) align={tool[3]:+.3f}")
            if err <= float(tol):
                return {"reached": True, "t_end": self.time(), "err": err}
        return {"reached": False, "t_end": self.time(), "err": err}


    def move_in_xy(self, target_xy, z_fixed, *, step_xy=0.005, dt=0.02, timeout_sec=10.0):
        # assumes env is already in IK mode and running
        self.stop_background_control()
        xy = np.asarray(target_xy, float).reshape(2)
        t_end = self.time() + timeout_sec

        while self.time() < t_end:
            tool = self.get_tool_metrics()
            cur_xy = tool[:2]
            d = xy - cur_xy
            dist = float(np.linalg.norm(d))
            if dist < 1e-3:
                return True

            # step towards target in XY only
            step = min(step_xy, dist)
            nxt_xy = cur_xy + (d / dist) * step
            nxt = np.array([nxt_xy[0], nxt_xy[1], z_fixed], float)

            # clamp for safety + send
            self.set_target_xyz(nxt)
            self.step(dt)

        return False


def is_reached(tool_metrics, target_xyz, xy_tol=5e-3, z_tol=3e-3, align_tol=0.985):
    p = tool_metrics[:3]
    align = float(tool_metrics[3])
    target_xyz = np.asarray(target_xyz, float)

    xy_err = np.linalg.norm(p[:2] - target_xyz[:2])
    z_err = abs(p[2] - target_xyz[2])
    return (xy_err <= xy_tol) and (z_err <= z_tol) and (align >= align_tol)

def main_smoke():
    env = IiwaHardwareEnv(
        use_ik=True,
        period_sec=0.01,
        max_delta_per_step=0.001,
        realtime_rate=1.0,
        ik_update_period_sec=0.01
    )

    # ===== Tunable test parameters =====
    start_target = np.array([0.5, 0.05, 0.38], dtype=float)
    start_target = np.array([0.4, 0.0, 0.38], dtype=float)
    square_side = 0.1          # meters
    sleep_between_waypoints = 1.0  # seconds
    num_laps = 1
    control_period_sec = 0.01

    # Final wait settings
    final_timeout_s = 10.0
    pos_tol_xy = 0.005   # 5 mm
    pos_tol_z = 0.003    # 3 mm
    align_tol = 0.995

    env.start(timeout_sec=1.0)

    # Move to the square start point first.
    env.move_to_target_xyz(
        start_target,
        timeout_sec=10.0,
        verbose=True,
        hold_count_required=5
    )
    exit()
    # Define 4 corners of a square in the xy plane.
    # Start from lower-left and go counter-clockwise.
    half = square_side / 2.0
    cx, cy, cz = start_target
    waypoints = [
        np.array([cx - half, cy - half, cz], dtype=float),
        np.array([cx + half, cy - half, cz], dtype=float),
        np.array([cx + half, cy + half, cz], dtype=float),
        np.array([cx - half, cy + half, cz], dtype=float),
    ]

    print("[smoke] square waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    env.start_background_control(control_period_sec=control_period_sec)
    try:
        final_target = None

        for lap in range(num_laps):
            for i, subgoal in enumerate(waypoints):
                final_target = subgoal.copy()
                start_time = time.time()
                env.set_latest_target_xyz(subgoal)
                dt = time.time() - start_time
                print(
                    f"[smoke] lap={lap} wp={i} queued latest target {subgoal} "
                    f"in {dt:.3f} sec"
                )
                time.sleep(sleep_between_waypoints)

        # Wait until the final waypoint is reached before stopping.
        t0 = time.time()
        reached = False
        while time.time() - t0 < final_timeout_s:
            tool = env.get_tool_metrics()
            tool_xyz = np.asarray(tool[:3], dtype=float)
            align = float(tool[3])

            xy_err = np.linalg.norm(tool_xyz[:2] - final_target[:2])
            z_err = abs(tool_xyz[2] - final_target[2])

            print(
                f"[smoke] waiting final target: "
                f"tool=({tool_xyz[0]:+.3f},{tool_xyz[1]:+.3f},{tool_xyz[2]:+.3f}) "
                f"target=({final_target[0]:+.3f},{final_target[1]:+.3f},{final_target[2]:+.3f}) "
                f"xy_err={xy_err*1000:.1f}mm z_err={z_err*1000:.1f}mm align={align:+.3f}"
            )

            if xy_err <= pos_tol_xy and z_err <= pos_tol_z and align >= align_tol:
                print(
                    f"[smoke] final target reached: "
                    f"tool={tool_xyz}, target={final_target}, align={align:.3f}"
                )
                reached = True
                break

            time.sleep(0.05)

        if not reached:
            print("[smoke] final target not reached before timeout.")

        tool = env.get_tool_metrics()
        tool_xyz = np.asarray(tool[:3], dtype=float)
        align = float(tool[3])
        print(
            f"[smoke] final tool=({tool_xyz[0]:+.3f},{tool_xyz[1]:+.3f},{tool_xyz[2]:+.3f}) "
            f"target=({final_target[0]:+.3f},{final_target[1]:+.3f},{final_target[2]:+.3f}) "
            f"align={align:+.3f}"
        )

    finally:
        env.stop_background_control()

    return

def main():
    """
    Real-arm interactive keyboard demo (terminal, curses).
      - W/S: +/-Y
      - A/D: -/+X
      - Up/Down arrows: +/-Z
      - R: reset target to INIT_TARGET (clamped)
      - Q or Esc: quit

    SAFETY:
      - target is ALWAYS clamped before sending:
        x in [0.61, 0.81], y in [-0.05, 0.05], z in [0.28, 0.38]
    """
    import curses
    import time
    import numpy as np

    # --- Safety bounds (STRICT) ---
    X_MIN, X_MAX = 0.61, 0.81
    Y_MIN, Y_MAX = -0.05, 0.05
    Z_MIN, Z_MAX = 0.28, 0.38

    def clamp_target(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float).reshape(3).copy()
        xyz[0] = min(max(xyz[0], X_MIN), X_MAX)
        xyz[1] = min(max(xyz[1], Y_MIN), Y_MAX)
        xyz[2] = min(max(xyz[2], Z_MIN), Z_MAX)
        return xyz

    # --- Create env (real arm) ---
    # Assumes you already have IiwaHardwareEnv with move_to_target_xyz(), set_target_xyz(), step(), etc.
    env = IiwaHardwareEnv(
        use_ik=True,
        period_sec=0.002,
        max_delta_per_step=0.0002,
        realtime_rate=1.0,
        ik_update_period_sec=0.01
    )

    # --- Init move (STRICT) ---
    INIT_TARGET = np.array([0.4, 0.00, 0.38], dtype=float)
    target_xyz = clamp_target(INIT_TARGET)
    env.move_to_target_xyz(target_xyz, timeout_sec=10.0, verbose=True)
    env.start_background_control(control_period_sec=0.01)

    # Step sizes
    step_xy = 0.002  # meters per keypress
    step_z  = 0.001  # meters per keypress
    refresh_dt = 0.02  # periodic UI refresh while background control keeps tracking

    def _draw(stdscr, lines):
        stdscr.erase()
        for i, line in enumerate(lines):
            stdscr.addstr(i, 0, line)
        stdscr.refresh()

    def _run_once(stdscr):
        nonlocal target_xyz

        # STRICT clamp before sending
        target_xyz = clamp_target(target_xyz)

        # Background control owns simulator stepping; the foreground UI only
        # updates the latest target and reads back state for display.
        env.set_target_xyz(target_xyz)

        # Read back metrics
        tool = env.get_tool_metrics()  # [x,y,z,align_score]
        q_meas = env.get_q_measured()
        ok, ik_dt = env.get_ik_status()
        q_ik = env.get_q_goal_ik()

        # radians -> degrees for readability
        q_meas_deg = np.degrees(q_meas)
        q_ik_deg = np.degrees(q_ik)
        q_err_deg = q_ik_deg - q_meas_deg


        # Errors in the SAME frame as tool_metrics
        xy_err = float(np.linalg.norm(tool[:2] - target_xyz[:2]))
        z_err = float(abs(tool[2] - target_xyz[2]))

        lines = []
        lines.append("=== IIWA Real-Arm Interactive Demo (IK) ===")
        lines.append("Controls: W/S=+/-Y, A/D=-/+X, Up/Down=+/-Z, R=reset init, Q/Esc=quit")
        lines.append(f"Bounds: x[{X_MIN:.2f},{X_MAX:.2f}] y[{Y_MIN:.2f},{Y_MAX:.2f}] z[{Z_MIN:.2f},{Z_MAX:.2f}]")
        lines.append(f"Step: xy={step_xy*1e3:.1f}mm  z={step_z*1e3:.1f}mm")
        lines.append("")
        lines.append(f"Target(clamped): [{target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}]")
        lines.append(f"Tool:           [{tool[0]:+.3f}, {tool[1]:+.3f}, {tool[2]:+.3f}]  align={tool[3]:+.3f}")
        lines.append(f"Err: xy={xy_err*1e3:.1f}mm z={z_err*1e3:.1f}mm | IK_ok={int(ok)} IK_dt={ik_dt*1e3:.1f}ms")
        lines.append(f"q_measured norm: {float(np.linalg.norm(q_meas)):.3f}")
        lines.append("Joint angles (degrees):")
        lines.append(f"  Measured: " + " ".join([f"{angle:+.1f}" for angle in q_meas_deg]))
        lines.append(f"  IK Goal:  " + " ".join([f"{angle:+.1f}" for angle in q_ik_deg]))
        lines.append(f"  Errors:   " + " ".join([f"{angle:+.1f}" for angle in q_err_deg]))
        _draw(stdscr, lines)

    def _curses_main(stdscr):
        nonlocal target_xyz, step_xy, step_z

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        _run_once(stdscr)

        while True:
            ch = stdscr.getch()
            if ch == -1:
                time.sleep(refresh_dt)
                _run_once(stdscr)
                continue

            moved = False

            # Quit
            if ch in (ord('q'), ord('Q'), 27):
                break

            # Reset to init target (clamped)
            if ch in (ord('r'), ord('R')):
                target_xyz = clamp_target(INIT_TARGET)
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

            # Optional: adjust step sizes
            elif ch == ord('1'):
                step_xy = max(0.001, step_xy * 0.5)
            elif ch == ord('2'):
                step_xy = min(0.02, step_xy * 2.0)
            elif ch == ord('9'):
                step_z = max(0.001, step_z * 0.5)
            elif ch == ord('0'):
                step_z = min(0.02, step_z * 2.0)

            if moved:
                # STRICT clamp before sending
                target_xyz = clamp_target(target_xyz)
            _run_once(stdscr)

    try:
        curses.wrapper(_curses_main)
    finally:
        env.stop_background_control()


if __name__ == "__main__":
    main_smoke()
    # main()
