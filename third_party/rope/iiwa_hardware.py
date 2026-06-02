import numpy as np
from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
)
from pydrake.all import (
    Gain,
    OutputPort,
    Adder,
    Diagram,
    DiagramBuilder,
    IiwaCommandSender,
    IiwaControlMode,
    IiwaStatusReceiver,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    position_enabled,
    torque_enabled,
    Multiplexer,
    ConstantVectorSource,
    MatrixGain,
    DrakeLcmInterface,
    DrakeLcm,
    LcmInterfaceSystem,
    LeafSystem,
    Simulator,
)


LCM_CHANNEL_SUFFIXS = ["", "_2"]


def _MakeIiwaRobot(
    lcm: DrakeLcmInterface,
    control_mode=IiwaControlMode.kPositionAndTorque,
    lcm_channel_suffix=""
) -> Diagram:
    assert isinstance(control_mode, IiwaControlMode)

    builder = DiagramBuilder()

    # Publish IIWA command.
    # IIWA driver won't respond faster than 1000Hz in torque_only mode and
    # 200Hz in other modes
    publish_period = 0.005
    if control_mode == IiwaControlMode.kTorqueOnly:
        publish_period = 0.001

    iiwa_command_sender = builder.AddSystem(
        IiwaCommandSender(control_mode=control_mode)
    )
    iiwa_command_publisher = builder.AddSystem(
        LcmPublisherSystem.Make(
            channel="IIWA_COMMAND" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_command,
            lcm=lcm,
            publish_period=publish_period,
            use_cpp_serializer=True,
        )
    )
    builder.Connect(
        iiwa_command_sender.get_output_port(),
        iiwa_command_publisher.get_input_port(),
    )
    if position_enabled(control_mode):
        builder.ExportInput(
            iiwa_command_sender.get_position_input_port(),
            "position",
        )
    if torque_enabled(control_mode):
        builder.ExportInput(
            iiwa_command_sender.get_torque_input_port(),
            "torque",
        )
    # Receive IIWA status and populate the output ports.
    iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
    iiwa_status_subscriber = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel="IIWA_STATUS" + lcm_channel_suffix,
            lcm_type=lcmt_iiwa_status,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )
    builder.Connect(
        iiwa_status_subscriber.get_output_port(),
        iiwa_status_receiver.get_input_port(),
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_position_commanded_output_port(),
        "position_commanded",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_position_measured_output_port(),
        "position_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_velocity_estimated_output_port(),
        "velocity_estimated",
    )

    # These are negated as outlined in drake/manipulation/README.
    def NegatedPort(
        builder: DiagramBuilder, output_port: OutputPort
    ) -> OutputPort:
        negater = builder.AddNamedSystem(
            f"signflip_{output_port.get_name()}", Gain(-1,
                                                       size=output_port.size())
        )
        builder.Connect(output_port, negater.get_input_port())
        return negater.get_output_port()

    builder.ExportOutput(
        NegatedPort(
            builder=builder,
            output_port=iiwa_status_receiver.get_torque_commanded_output_port(),
        ),
        "torque_commanded",
    )
    builder.ExportOutput(
        NegatedPort(
            builder=builder,
            output_port=iiwa_status_receiver.get_torque_measured_output_port(),
        ),
        "torque_measured",
    )
    builder.ExportOutput(
        iiwa_status_receiver.get_torque_external_output_port(),
        "torque_external",
    )

    mux = builder.AddSystem(Multiplexer(input_sizes=[7, 7]))
    builder.Connect(
        iiwa_status_receiver.get_position_measured_output_port(),
        mux.get_input_port(0),
    )
    builder.Connect(
        iiwa_status_receiver.get_velocity_estimated_output_port(),
        mux.get_input_port(1),
    )
    builder.ExportOutput(
        mux.get_output_port(),
        "state_estimated",
    )

    return builder.Build()


def _MakeLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm


def MakePlanarBimanualStation() -> Diagram:
    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)

    for i, lcm_channel_suffix in enumerate(LCM_CHANNEL_SUFFIXS):
        iiwa = builder.AddSystem(
            _MakeIiwaRobot(
                lcm=lcm,
                control_mode=IiwaControlMode.kPositionAndTorque,
                lcm_channel_suffix=lcm_channel_suffix,
            )
        )

        nominal_position_source = builder.AddSystem(
            ConstantVectorSource([0, np.pi/2, -np.pi/2, 0, 0, 0, 0])
        )

        B = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0],
        ]).T
        gain = builder.AddSystem(
            MatrixGain(B)
        )

        adder = builder.AddSystem(Adder(num_inputs=2, size=7))

        builder.Connect(
            gain.get_output_port(),
            adder.get_input_port(0),
        )
        builder.Connect(
            nominal_position_source.get_output_port(),
            adder.get_input_port(1),
        )
        builder.Connect(
            adder.get_output_port(),
            iiwa.GetInputPort("position"),
        )
        builder.ExportInput(
            gain.get_input_port(),
            f"input_{i}"
        )

        gain = builder.AddSystem(
            MatrixGain(B.T)
        )
        builder.Connect(
            iiwa.GetOutputPort("position_measured"),
            gain.get_input_port(),
        )
        builder.ExportOutput(
            gain.get_output_port(),
            f"position_{i}"
        )

    return builder.Build()


def HomePlanarBimanualStation(robot0_pos, robot1_pos):
    assert len(robot0_pos) == 3
    assert len(robot1_pos) == 3

    def get_joint_angles(robot_pos):
        return [
            robot_pos[0],
            np.pi/2,
            -np.pi/2,
            robot_pos[1],
            0,
            -robot_pos[2],
            0,
        ]

    HomeIiwaRobot(
        robot_index=0,
        home_joint_angles=get_joint_angles(robot0_pos),
    )
    HomeIiwaRobot(
        robot_index=1,
        home_joint_angles=get_joint_angles(robot1_pos),
    )

def MakeFullBimanualStation(
    control_mode=IiwaControlMode.kPositionAndTorque,
) -> Diagram:
    """
    Full 7-DoF bimanual hardware station.

    Inputs:
        position_0: 7D joint command for arm 0
        position_1: 7D joint command for arm 1

    Outputs:
        position_measured_0: 7D measured joint position for arm 0
        position_measured_1: 7D measured joint position for arm 1
        state_estimated_0: 14D [q; v] for arm 0
        state_estimated_1: 14D [q; v] for arm 1
    """
    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)

    for i, lcm_channel_suffix in enumerate(LCM_CHANNEL_SUFFIXS):
        iiwa = builder.AddSystem(
            _MakeIiwaRobot(
                lcm=lcm,
                control_mode=control_mode,
                lcm_channel_suffix=lcm_channel_suffix,
            )
        )

        if position_enabled(control_mode):
            builder.ExportInput(
                iiwa.GetInputPort("position"),
                f"position_{i}",
            )

        if torque_enabled(control_mode):
            builder.ExportInput(
                iiwa.GetInputPort("torque"),
                f"torque_{i}",
            )

        builder.ExportOutput(
            iiwa.GetOutputPort("position_measured"),
            f"position_measured_{i}",
        )
        builder.ExportOutput(
            iiwa.GetOutputPort("position_commanded"),
            f"position_commanded_{i}",
        )
        builder.ExportOutput(
            iiwa.GetOutputPort("velocity_estimated"),
            f"velocity_estimated_{i}",
        )
        builder.ExportOutput(
            iiwa.GetOutputPort("state_estimated"),
            f"state_estimated_{i}",
        )
        builder.ExportOutput(
            iiwa.GetOutputPort("torque_external"),
            f"torque_external_{i}",
        )

    return builder.Build()


def HomeIiwaRobot(
    robot_index: int,
    home_joint_angles: list[float],
    control_mode=IiwaControlMode.kPositionOnly,
    max_joint_speed=0.15,   # rad/s, about 8.6 deg/s
    timeout=60.0,
):
    assert 0 <= robot_index and robot_index < len(LCM_CHANNEL_SUFFIXS)
    assert len(home_joint_angles) == 7

    class GentleHomingController(LeafSystem):
        def __init__(self):
            super().__init__()
            self.x_target = np.array(home_joint_angles, dtype=float)
            self.max_step = max_joint_speed * 0.005

            self.DeclareVectorInputPort("position_estimated", 7)
            self.DeclareVectorOutputPort(
                "position_command",
                7,
                self.CalcOutput,
            )

        def CalcOutput(self, context, output):
            x = self.get_input_port().Eval(context)
            delta_x = np.clip(
                self.x_target - x,
                -self.max_step,
                self.max_step,
            )
            x_cmd = x + delta_x
            output.SetFromVector(x_cmd)

    builder = DiagramBuilder()
    lcm = _MakeLcm(builder)

    iiwa = builder.AddSystem(
        _MakeIiwaRobot(
            lcm_channel_suffix=LCM_CHANNEL_SUFFIXS[robot_index],
            lcm=lcm,
            control_mode=control_mode,
        )
    )

    controller = builder.AddSystem(GentleHomingController())

    builder.Connect(
        iiwa.GetOutputPort("position_measured"),
        controller.get_input_port(),
    )
    builder.Connect(
        controller.get_output_port(),
        iiwa.GetInputPort("position"),
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)

    iiwa_context = diagram.GetSubsystemContext(iiwa, context)

    print(f"[home arm {robot_index}] target deg = {np.round(np.rad2deg(home_joint_angles), 2)}")

    while context.get_time() < timeout:
        simulator.AdvanceTo(context.get_time() + 0.1)

        joint_angles = iiwa.GetOutputPort("position_measured").Eval(iiwa_context)
        err = np.max(np.abs(joint_angles - home_joint_angles))

        print(
            f"[home arm {robot_index}] "
            f"t={context.get_time():.1f}, "
            f"max_err_deg={np.rad2deg(err):.2f}"
        )

        if np.all(np.abs(joint_angles - home_joint_angles) < np.deg2rad(0.5)):
            print(f"[home arm {robot_index}] reached target.")
            return True

    print(f"[home arm {robot_index}] timeout, target not fully reached.")
    return False

def HomeFullBimanualStation(
    robot0_q,
    robot1_q,
    max_joint_speed=0.15,
    timeout=60.0,
):
    assert len(robot0_q) == 7
    assert len(robot1_q) == 7

    print("Homing arm 0 gently...")
    ok0 = HomeIiwaRobot(
        robot_index=0,
        home_joint_angles=robot0_q,
        control_mode=IiwaControlMode.kPositionOnly,
        max_joint_speed=max_joint_speed,
        timeout=timeout,
    )

    print("Homing arm 1 gently...")
    ok1 = HomeIiwaRobot(
        robot_index=1,
        home_joint_angles=robot1_q,
        control_mode=IiwaControlMode.kPositionOnly,
        max_joint_speed=max_joint_speed,
        timeout=timeout,
    )

    print(f"Homing result: arm0={ok0}, arm1={ok1}")
    return ok0 and ok1

if __name__ == "__main__":
    print("Starting homing test...")

    HomePlanarBimanualStation(
        robot0_pos=[0.0, 0.0, 0.0],
        robot1_pos=[0.0, 0.0, 0.0],
    )

    print("Done.")