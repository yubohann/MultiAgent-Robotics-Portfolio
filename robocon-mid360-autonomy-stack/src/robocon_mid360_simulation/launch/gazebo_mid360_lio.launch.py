"""Compose Gazebo sensors, FAST-LIO2, and the MID-360 contract for simulation."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    simulation_share = get_package_share_directory("robocon_mid360_simulation")
    contract_share = get_package_share_directory("mid360_localization_contract")
    fast_lio_parameters = LaunchConfiguration("fast_lio_parameters")
    contract_parameters = LaunchConfiguration("contract_parameters")
    lidar_samples = LaunchConfiguration("lidar_samples")
    lidar_downsample = LaunchConfiguration("lidar_downsample")
    enable_rgbd = LaunchConfiguration("enable_rgbd")
    enable_ground_truth = LaunchConfiguration("enable_ground_truth")
    world = LaunchConfiguration("world")

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([simulation_share, "launch", "gazebo_mid360_candidate.launch.py"])
        ),
        launch_arguments={
            "lidar_samples": lidar_samples,
            "lidar_downsample": lidar_downsample,
            "enable_rgbd": enable_rgbd,
            "enable_ground_truth": enable_ground_truth,
            "world": world,
        }.items(),
    )
    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        parameters=[fast_lio_parameters, {"use_sim_time": True}],
        output="screen",
    )
    contract = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([contract_share, "launch", "competition_localization.launch.py"])
        ),
        launch_arguments={"params_file": contract_parameters}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "fast_lio_parameters",
            default_value=PathJoinSubstitution([simulation_share, "config", "fast_lio_simulation.yaml"]),
        ),
        DeclareLaunchArgument(
            "contract_parameters",
            default_value=PathJoinSubstitution([simulation_share, "config", "mid360_simulation_contract.yaml"]),
        ),
        DeclareLaunchArgument(
            "lidar_samples",
            default_value="30000",
            description="Rays per simulated packet. 30000 is the higher-density LIO profile; use 2000 only for sensor smoke.",
        ),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument("enable_rgbd", default_value="false"),
        DeclareLaunchArgument(
            "enable_ground_truth",
            default_value="false",
            description="Enable the noiseless Gazebo pose topic for simulation-only error measurement.",
        ),
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([simulation_share, "worlds", "robocon25_candidate.world"]),
        ),
        simulation,
        fast_lio,
        contract,
    ])
