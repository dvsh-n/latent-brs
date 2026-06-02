import numpy as np

from pydrake.all import (
    JacobianWrtVariable,
    InverseKinematics,
    MathematicalProgram,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RotationMatrix,
    Solve,
)


def desired_link7_rotation(ee_z_rotation_offset_rad: float = 0.0) -> RotationMatrix:
    return RotationMatrix.MakeZRotation(
        np.pi / 2 + float(ee_z_rotation_offset_rad)
    ).multiply(RotationMatrix.MakeXRotation(np.pi))


class SingleIiwaPositionIK:
    """
    Single-arm iiwa position-only IK helper.

    Coordinate convention:
        The target position p_WQ is expressed in this single iiwa model's world frame.
        Since we use each arm independently, this is effectively each arm's own base frame.

    Controlled point:
        p_BQ is a configurable point on iiwa_link_7.
    """

    def __init__(
        self,
        model_url="package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf",
        constrain_orientation=True,
        orientation_theta_bound=np.deg2rad(3.0),
        orientation_constraint_mode="full",
        ee_z_rotation_offset_rad=0.0,
        ee_position_offset_B=(0.0, 0.0, 0.0),
    ):
        self.constrain_orientation = bool(constrain_orientation)
        self.orientation_theta_bound = float(orientation_theta_bound)
        self.orientation_constraint_mode = str(orientation_constraint_mode)
        valid_orientation_modes = {"full", "z_axis_down"}
        if self.orientation_constraint_mode not in valid_orientation_modes:
            raise ValueError(
                f"orientation_constraint_mode must be one of {sorted(valid_orientation_modes)}, "
                f"got {self.orientation_constraint_mode!r}"
            )
        self.ee_z_rotation_offset_rad = float(ee_z_rotation_offset_rad)
        self.ee_position_offset_B = np.asarray(ee_position_offset_B, dtype=float).reshape(3)
        self.R_W7_desired = desired_link7_rotation(self.ee_z_rotation_offset_rad)
        self.plant = MultibodyPlant(time_step=0.0)
        parser = Parser(self.plant)

        models = parser.AddModelsFromUrl(model_url)
        assert len(models) == 1
        self.iiwa = models[0]

        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("iiwa_link_0", self.iiwa),
            RigidTransform(),
        )
        self.plant.Finalize()

        self.context = self.plant.CreateDefaultContext()
        self.link7 = self.plant.GetBodyByName("iiwa_link_7", self.iiwa)
        self.link7_frame = self.link7.body_frame()

    def set_q(self, q):
        q = np.asarray(q, dtype=float).reshape(7)
        self.plant.SetPositions(self.context, self.iiwa, q)

    def fk_position(self, q):
        self.set_q(q)
        X_W7 = self.link7_frame.CalcPoseInWorld(self.context)
        return np.asarray(X_W7 @ self.ee_position_offset_B).reshape(3)

    def fk_rotation(self, q):
        self.set_q(q)
        X_W7 = self.link7_frame.CalcPoseInWorld(self.context)
        return X_W7.rotation()

    def orientation_error(self, q):
        R_W7 = self.fk_rotation(q)
        return float(self.R_W7_desired.inverse().multiply(R_W7).ToAngleAxis().angle())

    def z_axis_down_error(self, q):
        R_W7 = self.fk_rotation(q).matrix()
        link7_z_W = R_W7 @ np.array([0.0, 0.0, 1.0])
        cos_angle = float(np.clip(link7_z_W.dot(np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        return float(np.arccos(cos_angle))

    def orientation_constraint_error(self, q):
        if self.orientation_constraint_mode == "z_axis_down":
            return self.z_axis_down_error(q)
        return self.orientation_error(q)

    def translational_jacobian(self, q):
        self.set_q(q)
        Jv_WQ = self.plant.CalcJacobianTranslationalVelocity(
            self.context,
            JacobianWrtVariable.kQDot,
            self.link7_frame,
            self.ee_position_offset_B,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        return np.asarray(Jv_WQ)

    def singularity_metric(self, q):
        """
        Returns:
            sigma_min: minimum singular value of the 3x7 translational Jacobian.
            cond: sigma_max / sigma_min.
        """
        J = self.translational_jacobian(q)
        s = np.linalg.svd(J, compute_uv=False)

        sigma_min = float(s[-1])
        sigma_max = float(s[0])
        cond = float(sigma_max / max(sigma_min, 1e-12))
        return sigma_min, cond

    def solve_position_ik(
        self,
        p_WQ,
        q_seed,
        position_tol=0.005,
        max_joint_move_from_seed=np.deg2rad(12.0),
        joint_lower_bounds=None,
        joint_upper_bounds=None,
        min_sigma=1e-4,
        max_cond=1e4,
    ):
        """
        Solve position-only IK.

        Args:
            p_WQ: desired 3D position in this arm's local/world frame.
            q_seed: previous waypoint solution or current measured q.
            position_tol: box tolerance around target position, meters.
            max_joint_move_from_seed: optional soft safety bound per waypoint.
            min_sigma: reject near translational singularity.
            max_cond: reject ill-conditioned translational Jacobian.

        Returns:
            q_sol, info dict.
        """
        p_WQ = np.asarray(p_WQ, dtype=float).reshape(3)
        q_seed = np.asarray(q_seed, dtype=float).reshape(7)

        ik = InverseKinematics(self.plant, with_joint_limits=True)
        q = ik.q()
        prog = ik.prog()

        frame_W = self.plant.world_frame()

        ik.AddPositionConstraint(
            frameB=self.link7_frame,
            p_BQ=self.ee_position_offset_B,
            frameA=frame_W,
            p_AQ_lower=p_WQ - position_tol,
            p_AQ_upper=p_WQ + position_tol,
        )
        if self.constrain_orientation and self.orientation_constraint_mode == "full":
            ik.AddOrientationConstraint(
                frameAbar=frame_W,
                R_AbarA=self.R_W7_desired,
                frameBbar=self.link7_frame,
                R_BbarB=RotationMatrix(),
                theta_bound=self.orientation_theta_bound,
            )
        elif self.constrain_orientation and self.orientation_constraint_mode == "z_axis_down":
            ik.AddAngleBetweenVectorsConstraint(
                frameA=frame_W,
                na_A=np.array([0.0, 0.0, -1.0]),
                frameB=self.link7_frame,
                nb_B=np.array([0.0, 0.0, 1.0]),
                angle_lower=0.0,
                angle_upper=self.orientation_theta_bound,
            )

        # Prefer solutions close to the previous waypoint.
        Q = np.eye(7)
        prog.AddQuadraticErrorCost(Q, q_seed, q)
        prog.SetInitialGuess(q, q_seed)

        # Per-waypoint trust region. This prevents IK from jumping to a different
        # elbow configuration between nearby Cartesian waypoints.
        if max_joint_move_from_seed is not None:
            prog.AddBoundingBoxConstraint(
                q_seed - max_joint_move_from_seed,
                q_seed + max_joint_move_from_seed,
                q,
            )
        if joint_lower_bounds is not None or joint_upper_bounds is not None:
            if joint_lower_bounds is None or joint_upper_bounds is None:
                raise ValueError("joint_lower_bounds and joint_upper_bounds must be provided together.")
            prog.AddBoundingBoxConstraint(
                np.asarray(joint_lower_bounds, dtype=float).reshape(7),
                np.asarray(joint_upper_bounds, dtype=float).reshape(7),
                q,
            )

        result = Solve(prog)

        if not result.is_success():
            return None, {
                "success": False,
                "reason": "IK solve failed",
                "p_target": p_WQ,
            }

        q_sol = np.asarray(result.GetSolution(q)).reshape(7)

        p_actual = self.fk_position(q_sol)
        pos_err = float(np.linalg.norm(p_actual - p_WQ))
        orientation_err = self.orientation_constraint_error(q_sol)
        full_orientation_err = self.orientation_error(q_sol)
        z_axis_down_err = self.z_axis_down_error(q_sol)

        sigma_min, cond = self.singularity_metric(q_sol)
        if sigma_min < min_sigma or cond > max_cond:
            return None, {
                "success": False,
                "reason": "near singularity",
                "p_target": p_WQ,
                "p_actual": p_actual,
                "pos_err": pos_err,
                "orientation_err": orientation_err,
                "full_orientation_err": full_orientation_err,
                "z_axis_down_err": z_axis_down_err,
                "sigma_min": sigma_min,
                "cond": cond,
            }

        return q_sol, {
            "success": True,
            "p_target": p_WQ,
            "p_actual": p_actual,
            "pos_err": pos_err,
            "orientation_err": orientation_err,
            "full_orientation_err": full_orientation_err,
            "z_axis_down_err": z_axis_down_err,
            "sigma_min": sigma_min,
            "cond": cond,
        }


def quintic_smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def make_cartesian_line_waypoints(p_start, p_goal, num_waypoints):
    p_start = np.asarray(p_start, dtype=float).reshape(3)
    p_goal = np.asarray(p_goal, dtype=float).reshape(3)

    waypoints = []
    for i in range(num_waypoints):
        s = i / max(num_waypoints - 1, 1)
        a = quintic_smoothstep(s)
        p = p_start + a * (p_goal - p_start)
        waypoints.append(p)

    return np.asarray(waypoints)


def solve_cartesian_path(
    ik_solver,
    p_waypoints,
    q_seed,
    position_tol=0.005,
    max_joint_move_from_seed=np.deg2rad(8.0),
    min_sigma=1e-4,
    max_cond=1e4,
):
    """
    Sequential waypoint IK. Each waypoint uses previous q solution as seed.
    """
    q_seed = np.asarray(q_seed, dtype=float).reshape(7)

    q_list = []
    info_list = []

    q_prev = q_seed.copy()

    for k, p in enumerate(p_waypoints):
        q_sol, info = ik_solver.solve_position_ik(
            p_WQ=p,
            q_seed=q_prev,
            position_tol=position_tol,
            max_joint_move_from_seed=max_joint_move_from_seed,
            min_sigma=min_sigma,
            max_cond=max_cond,
        )

        info["waypoint"] = k
        info_list.append(info)

        if q_sol is None:
            raise RuntimeError(
                f"IK failed at waypoint {k}: {info}"
            )

        q_list.append(q_sol)
        q_prev = q_sol

    return np.asarray(q_list), info_list
