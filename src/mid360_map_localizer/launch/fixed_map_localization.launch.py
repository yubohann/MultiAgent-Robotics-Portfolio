from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("mid360_map_localizer")
    params_file = LaunchConfiguration("params_file")
    overrides = {
        "map_file": LaunchConfiguration("map_file"),
        "scan_topic": LaunchConfiguration("scan_topic"),
        "odom_topic": LaunchConfiguration("odom_topic"),
    }
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([share, "config", "fixed_map_localization.yaml"]),
            description="Fixed-map scan matcher parameters.",
        ),
        DeclareLaunchArgument("map_file", default_value="", description="Private PCD map path."),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/mid360/cloud_registered_reliable",
            description="Registered cloud in the local odometry frame.",
        ),
        DeclareLaunchArgument(
            "odom_topic",
            default_value="/mid360/local_odometry",
            description="Canonical local odometry topic.",
        ),
        Node(
            package="mid360_map_localizer",
            executable="mid360_scan_matcher_node",
            name="mid360_scan_matcher",
            output="screen",
            parameters=[params_file, overrides],
        ),
    ])
