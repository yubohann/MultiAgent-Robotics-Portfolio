"""Launch the MID-360 interface contract around an already-running FAST-LIO2 node."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_params = (
        get_package_share_directory("mid360_localization_contract")
        + "/config/competition.yaml"
    )
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to the private deployment parameter file.",
        ),
        Node(
            package="mid360_localization_contract",
            executable="mid360_input_guard",
            name="mid360_input_guard",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="mid360_localization_contract",
            executable="mid360_pose_bridge",
            name="mid360_pose_bridge",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="mid360_localization_contract",
            executable="mid360_map_odom_anchor",
            name="mid360_map_odom_anchor",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="mid360_localization_contract",
            executable="mid360_static_sensor_frames",
            name="mid360_static_sensor_frames",
            output="screen",
            parameters=[params_file],
        ),
        Node(
            package="mid360_localization_contract",
            executable="mid360_preflight",
            name="mid360_preflight",
            output="screen",
            parameters=[params_file],
        ),
    ])
