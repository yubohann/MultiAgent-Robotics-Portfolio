"""Run the simulation, FAST-LIO2, localization contract, and map recorder."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    simulation_share = get_package_share_directory("robocon_mid360_simulation")
    map_tools_share = get_package_share_directory("mid360_map_tools")
    lio_launch = PathJoinSubstitution([simulation_share, "launch", "gazebo_mid360_lio.launch.py"])
    mapping_fast_lio_parameters = PathJoinSubstitution([
        simulation_share, "config", "fast_lio_mapping_simulation.yaml"
    ])
    map_params = PathJoinSubstitution([map_tools_share, "config", "mapping_tools.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="false"),
        DeclareLaunchArgument("lidar_samples", default_value="30000"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument(
            "sparse_coverage",
            default_value="false",
            description=(
                "Diagnostic profile for low-density scans: preserves sparse returns "
                "for accumulated coverage but is not a fixed-map quality profile."
            ),
        ),
        DeclareLaunchArgument("enable_rgbd", default_value="false"),
        DeclareLaunchArgument("enable_ground_truth", default_value="false"),
        DeclareLaunchArgument(
            "fast_lio_parameters",
            default_value=mapping_fast_lio_parameters,
            description="Mapping-only FAST-LIO profile with registered-scan publication enabled.",
        ),
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution([
            simulation_share, "worlds", "indoor_competition_candidate.world"
        ])),
        DeclareLaunchArgument("map_output_file", description="Output ASCII PCD path."),
        DeclareLaunchArgument("metadata_output_file", description="Output YAML metadata path."),
        DeclareLaunchArgument("map_id", default_value="indoor_candidate_v1"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lio_launch),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "lidar_samples": LaunchConfiguration("lidar_samples"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "enable_rgbd": LaunchConfiguration("enable_rgbd"),
                "enable_ground_truth": LaunchConfiguration("enable_ground_truth"),
                "fast_lio_parameters": LaunchConfiguration("fast_lio_parameters"),
                "world": LaunchConfiguration("world"),
            }.items(),
        ),
        Node(
            package="mid360_map_tools",
            executable="fastlio_cloud_mapper_node",
            name="mid360_quality_mapper",
            output="screen",
            condition=UnlessCondition(LaunchConfiguration("sparse_coverage")),
            parameters=[map_params, {
                "input_topic": "/cloud_registered",
                "odom_topic": "/mid360/local_odometry",
                "output_frame": "camera_init",
                "require_odom_for_map": False,
                "min_map_hits": 1,
                "map_window_frames": 0,
                "save_map_on_shutdown": True,
                "map_output_file": LaunchConfiguration("map_output_file"),
                "metadata_output_file": LaunchConfiguration("metadata_output_file"),
                "map_id": LaunchConfiguration("map_id"),
                "map_save_interval_sec": 5.0,
            }],
        ),
        Node(
            package="mid360_map_tools",
            executable="fastlio_cloud_mapper_node",
            name="mid360_sparse_coverage_mapper",
            output="screen",
            condition=IfCondition(LaunchConfiguration("sparse_coverage")),
            parameters=[map_params, {
                "input_topic": "/cloud_registered",
                "odom_topic": "/mid360/local_odometry",
                "output_frame": "camera_init",
                "require_odom_for_map": False,
                "enable_quality_gate": True,
                "radius_filter": 0.0,
                "radius_min_neighbors": 0,
                "z_min": -0.6,
                "z_max": 4.5,
                "scan_voxel": 0.05,
                "map_voxel": 0.08,
                "min_map_hits": 1,
                "map_window_frames": 0,
                "min_scan_map_overlap": 0.01,
                "overlap_neighbor_voxels": 3,
                "save_map_on_shutdown": True,
                "map_output_file": LaunchConfiguration("map_output_file"),
                "metadata_output_file": LaunchConfiguration("metadata_output_file"),
                "map_id": "sparse_coverage_diagnostic",
                "map_save_interval_sec": 5.0,
            }],
        ),
    ])
