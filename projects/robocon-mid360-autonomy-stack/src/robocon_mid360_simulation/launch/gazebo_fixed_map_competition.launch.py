"""Run the Gazebo localization and competition safety chain together.

This launch file is an integration profile for ``gazebo_simulation`` evidence.
It deliberately keeps the synthetic action executor and target gate explicit;
they are adapters for exercising ROS contracts, not physical hardware.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    simulation_share = get_package_share_directory("robocon_mid360_simulation")
    map_localizer_share = get_package_share_directory("mid360_map_localizer")
    supervisor_share = get_package_share_directory("robocon_game_supervisor")
    mapping_launch = simulation_share + "/launch/gazebo_mid360_mapping.launch.py"
    localizer_launch = map_localizer_share + "/launch/fixed_map_localization.launch.py"

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mapping_launch),
        launch_arguments={
            "use_gui": LaunchConfiguration("use_gui"),
            "lidar_samples": LaunchConfiguration("lidar_samples"),
            "lidar_downsample": LaunchConfiguration("lidar_downsample"),
            "world": LaunchConfiguration("world"),
            "map_output_file": LaunchConfiguration("map_output_file"),
            "metadata_output_file": LaunchConfiguration("metadata_output_file"),
            "enable_ground_truth": "false",
        }.items(),
    )
    localizer = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localizer_launch),
        launch_arguments={
            "map_file": LaunchConfiguration("map_file"),
            "params_file": LaunchConfiguration("fixed_map_params_file"),
        }.items(),
    )
    supervisor = Node(
        package="robocon_game_supervisor",
        executable="robocon_game_supervisor",
        name="robocon_game_supervisor",
        output="screen",
        parameters=[LaunchConfiguration("supervisor_params_file"), {
            "task_id": LaunchConfiguration("task_id"),
            "require_teammate_heartbeat": True,
            "auto_recovery_on_signal_loss": True,
            "action_ttl_sec": 2.0,
        }],
    )
    action_simulator = Node(
        package="robocon_game_supervisor",
        executable="robocon_action_simulator",
        name="robocon_action_simulator",
        output="screen",
        parameters=[{"failure_action": ""}],
    )
    target_gate = Node(
        package="robocon_perception_adapter",
        executable="target_gate",
        name="robocon_target_gate",
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="false"),
        DeclareLaunchArgument("lidar_samples", default_value="30000"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument(
            "world",
            default_value=simulation_share + "/worlds/indoor_competition_candidate.world",
        ),
        DeclareLaunchArgument("map_file", default_value=""),
        DeclareLaunchArgument("map_output_file", default_value=""),
        DeclareLaunchArgument("metadata_output_file", default_value=""),
        DeclareLaunchArgument("task_id", default_value="gazebo-fixed-map-interlock"),
        DeclareLaunchArgument(
            "fixed_map_params_file",
            default_value=map_localizer_share + "/config/fixed_map_localization.yaml",
        ),
        DeclareLaunchArgument(
            "supervisor_params_file",
            default_value=supervisor_share + "/config/supervisor.yaml",
        ),
        mapping,
        localizer,
        target_gate,
        action_simulator,
        supervisor,
    ])
