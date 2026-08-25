"""Dual ROBOCON basketball court with robot-1 FAST-LIO and map views."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    simulation_share = get_package_share_directory("robocon_mid360_simulation")
    map_tools_share = get_package_share_directory("mid360_map_tools")
    dual_launch = PathJoinSubstitution([simulation_share, "launch", "gazebo_mid360_dual.launch.py"])
    lio_params = PathJoinSubstitution([simulation_share, "config", "fast_lio_mapping_simulation.yaml"])
    map_params = PathJoinSubstitution([map_tools_share, "config", "mapping_tools.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("lidar_samples", default_value="30000"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument("allow_unsafe_density", default_value="false"),
        DeclareLaunchArgument("enable_rgbd", default_value="true"),
        DeclareLaunchArgument("lio_params", default_value=lio_params),
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution([
            simulation_share, "worlds", "robocon25_candidate.world"
        ])),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(dual_launch),
            launch_arguments={
                "use_gui": LaunchConfiguration("use_gui"),
                "lidar_samples": LaunchConfiguration("lidar_samples"),
                "lidar_downsample": LaunchConfiguration("lidar_downsample"),
                "allow_unsafe_density": LaunchConfiguration("allow_unsafe_density"),
                "enable_rgbd": LaunchConfiguration("enable_rgbd"),
                "world": LaunchConfiguration("world"),
            }.items(),
        ),
        Node(
            package="fast_lio",
            executable="fastlio_mapping",
            name="fastlio_mapping_robot1",
            parameters=[LaunchConfiguration("lio_params"), {"use_sim_time": True, "publish.tf": True}],
            remappings=[
                ("/livox/lidar", "/robot1/livox/lidar"),
                ("/livox/imu", "/robot1/livox/imu"),
                ("/Odometry", "/mid360/local_odometry"),
            ],
            output="screen",
        ),
        Node(
            package="mid360_map_tools",
            executable="fastlio_cloud_mapper_node",
            name="mid360_quality_mapper_dual",
            parameters=[map_params, {
                "input_topic": "/cloud_registered",
                "odom_topic": "/mid360/local_odometry",
                "output_frame": "camera_init",
                "require_odom_for_map": False,
                "enable_quality_gate": False,
                "scan_voxel": 0.08,
                "map_voxel": 0.12,
                "map_window_frames": 0,
                "min_map_hits": 1,
                "map_publish_every": 1,
                "save_map_on_shutdown": True,
                "map_output_file": "/tmp/robocon_dual_demo_map.pcd",
                "metadata_output_file": "/tmp/robocon_dual_demo_map.yaml",
                "map_id": "robocon25_dual_demo",
            }],
            output="screen",
        ),
        Node(
            package="mid360_map_tools",
            executable="pointcloud_occupancy_grid_node",
            name="mid360_occupancy_grid_dual",
            parameters=[map_params, {
                "input_topic": "/cloud_registered",
                "grid_topic": "/mid360/occupancy_grid",
                "output_frame": "camera_init",
                "odom_topic": "/mid360/local_odometry",
                "use_odometry": True,
                "raycast_free_space": True,
                "occupied_increment": 12,
                "free_decrement": 2,
                "log_odds_decay_per_cloud": 0,
                "stale_after_clouds": 0,
                "resolution": 0.08,
                "width_m": 24.0,
                "height_m": 12.0,
                "origin_x": -12.0,
                "origin_y": -6.0,
                "publish_every_n_clouds": 1,
            }],
            output="screen",
        ),
        Node(
            package="robocon_mid360_simulation",
            executable="depth_visualizer.py",
            name="robot1_depth_visualizer",
            condition=IfCondition(LaunchConfiguration("enable_rgbd")),
            parameters=[{
                "input_topic": "/robot1/simulated_rgbd_camera/depth/image_raw",
                "output_topic": "/robot1/simulated_rgbd_camera/depth/image_visualized",
                "min_depth": 0.2,
                "max_depth": 12.0,
            }],
            output="screen",
        ),
        Node(
            package="robocon_mid360_simulation",
            executable="depth_visualizer.py",
            name="robot2_depth_visualizer",
            condition=IfCondition(LaunchConfiguration("enable_rgbd")),
            parameters=[{
                "input_topic": "/robot2/simulated_rgbd_camera/depth/image_raw",
                "output_topic": "/robot2/simulated_rgbd_camera/depth/image_visualized",
                "min_depth": 0.2,
                "max_depth": 12.0,
            }],
            output="screen",
        ),
    ])
