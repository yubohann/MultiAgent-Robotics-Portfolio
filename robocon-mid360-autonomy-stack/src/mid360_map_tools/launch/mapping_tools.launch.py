from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("mid360_map_tools")
    params = LaunchConfiguration("params_file")
    input_topic = LaunchConfiguration("input_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    output_frame = LaunchConfiguration("map_frame")

    common = {
        "input_topic": input_topic,
        "odom_topic": odom_topic,
        "output_frame": output_frame,
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([package_share, "config", "mapping_tools.yaml"]),
            description="Mapping and occupancy-grid parameter file.",
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/cloud_registered",
            description="FAST-LIO registered PointCloud2 topic.",
        ),
        DeclareLaunchArgument(
            "odom_topic",
            default_value="/mid360/local_odometry",
            description="Canonical local odometry topic.",
        ),
        DeclareLaunchArgument(
            "map_frame",
            default_value="",
            description="Optional frame override. Empty preserves the cloud frame.",
        ),
        Node(
            package="mid360_map_tools",
            executable="fastlio_cloud_mapper_node",
            name="mid360_quality_mapper",
            output="screen",
            parameters=[params, common],
        ),
        Node(
            package="mid360_map_tools",
            executable="pointcloud_occupancy_grid_node",
            name="mid360_occupancy_grid",
            output="screen",
            parameters=[params, common],
        ),
    ])
