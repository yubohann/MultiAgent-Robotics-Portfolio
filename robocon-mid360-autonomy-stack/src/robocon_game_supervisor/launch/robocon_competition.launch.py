"""Launch the MID-360 localization stack and the competition supervisor together."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    localization_share = get_package_share_directory("mid360_localization_contract")
    supervisor_share = get_package_share_directory("robocon_game_supervisor")
    camera_share = get_package_share_directory("robocon_camera_yolo_adapter")
    map_localizer_share = get_package_share_directory("mid360_map_localizer")
    perception_share = get_package_share_directory("robocon_perception_adapter")
    localization_launch = localization_share + "/launch/robocon_mid360_bringup.launch.py"
    map_localizer_launch = map_localizer_share + "/launch/fixed_map_localization.launch.py"
    target_gate_launch = perception_share + "/launch/target_gate.launch.py"
    supervisor_params = LaunchConfiguration("supervisor_params_file")
    pose_bridge_params = LaunchConfiguration("pose_bridge_params_file")
    supervisor = Node(
        package="robocon_game_supervisor",
        executable="robocon_game_supervisor",
        name="robocon_game_supervisor",
        output="screen",
        parameters=[supervisor_params],
    )
    pose_bridge = Node(
        package="robocon_pose_command_bridge",
        executable="pose_command_bridge",
        name="robocon_pose_command_bridge",
        output="screen",
        parameters=[pose_bridge_params],
        condition=IfCondition(LaunchConfiguration("start_pose_bridge")),
    )
    action_simulator = Node(
        package="robocon_game_supervisor",
        executable="robocon_action_simulator",
        name="robocon_action_simulator",
        output="screen",
        parameters=[supervisor_params],
        condition=IfCondition(LaunchConfiguration("start_action_simulator")),
    )
    camera_yolo = Node(
        package="robocon_camera_yolo_adapter",
        executable="camera_yolo_adapter",
        name="robocon_camera_yolo_adapter",
        output="screen",
        parameters=[LaunchConfiguration("camera_yolo_params_file")],
        condition=IfCondition(LaunchConfiguration("start_camera_yolo")),
    )
    target_gate = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(target_gate_launch),
        condition=IfCondition(LaunchConfiguration("start_target_gate")),
    )
    fixed_map_localizer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(map_localizer_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("fixed_map_params_file"),
            "map_file": LaunchConfiguration("map_file"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_fixed_map_localizer")),
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch),
        launch_arguments={
            "start_driver": LaunchConfiguration("start_driver"),
            "start_fast_lio": LaunchConfiguration("start_fast_lio"),
            "driver_config_path": LaunchConfiguration("driver_config_path"),
            "fast_lio_params_file": LaunchConfiguration("fast_lio_params_file"),
            "contract_params_file": LaunchConfiguration("contract_params_file"),
        }.items(),
    )
    supervisor_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=supervisor,
            on_exit=[EmitEvent(event=Shutdown(reason="competition supervisor exited"))],
        )
    )
    return LaunchDescription([
        DeclareLaunchArgument("start_driver", default_value="true"),
        DeclareLaunchArgument("start_fast_lio", default_value="true"),
        DeclareLaunchArgument("start_pose_bridge", default_value="true"),
        DeclareLaunchArgument("start_action_simulator", default_value="false"),
        DeclareLaunchArgument("start_camera_yolo", default_value="false"),
        DeclareLaunchArgument("start_target_gate", default_value="true"),
        DeclareLaunchArgument("start_fixed_map_localizer", default_value="true"),
        DeclareLaunchArgument("driver_config_path", default_value=""),
        DeclareLaunchArgument("fast_lio_params_file", default_value=""),
        DeclareLaunchArgument("map_file", default_value=""),
        DeclareLaunchArgument(
            "contract_params_file",
            default_value=localization_share + "/config/competition.yaml",
        ),
        DeclareLaunchArgument(
            "supervisor_params_file",
            default_value=supervisor_share + "/config/supervisor.yaml",
        ),
        DeclareLaunchArgument(
            "pose_bridge_params_file",
            default_value=get_package_share_directory("robocon_pose_command_bridge") + "/config/pose_command_bridge.yaml",
        ),
        DeclareLaunchArgument(
            "camera_yolo_params_file",
            default_value=camera_share + "/config/camera_yolo.yaml",
        ),
        DeclareLaunchArgument(
            "fixed_map_params_file",
            default_value=map_localizer_share + "/config/fixed_map_localization.yaml",
        ),
        localization,
        fixed_map_localizer,
        target_gate,
        supervisor,
        pose_bridge,
        action_simulator,
        camera_yolo,
        supervisor_exit,
    ])
