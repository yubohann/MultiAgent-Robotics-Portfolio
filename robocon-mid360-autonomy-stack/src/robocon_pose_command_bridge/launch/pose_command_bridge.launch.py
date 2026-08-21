from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("robocon_pose_command_bridge")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([share, "config", "pose_command_bridge.yaml"]),
            description="Pose command bridge parameter file.",
        ),
        Node(
            package="robocon_pose_command_bridge",
            executable="pose_command_bridge",
            name="robocon_pose_command_bridge",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
