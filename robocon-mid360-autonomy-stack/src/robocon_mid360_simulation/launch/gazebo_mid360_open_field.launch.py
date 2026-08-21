"""Open-field degradation profile without an overhead return plane."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("robocon_mid360_simulation")
    source = PathJoinSubstitution([share, "launch", "gazebo_mid360_candidate.launch.py"])
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="false"),
        DeclareLaunchArgument("lidar_samples", default_value="2000"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument("enable_rgbd", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(source),
            launch_arguments={
                "world": PathJoinSubstitution([share, "worlds", "open_field_degraded.world"]),
                "use_gui": LaunchConfiguration("use_gui"),
                "lidar_samples": LaunchConfiguration("lidar_samples"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "enable_rgbd": LaunchConfiguration("enable_rgbd"),
            }.items(),
        ),
    ])
