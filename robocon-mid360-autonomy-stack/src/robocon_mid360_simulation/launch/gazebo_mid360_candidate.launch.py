"""Start the headless Gazebo MID-360 simulation with the public candidate asset."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("robocon_mid360_simulation")
    gazebo_share = get_package_share_directory("gazebo_ros")
    world = LaunchConfiguration("world")
    use_gui = LaunchConfiguration("use_gui")
    lidar_samples = LaunchConfiguration("lidar_samples")
    lidar_downsample = LaunchConfiguration("lidar_downsample")
    enable_rgbd = LaunchConfiguration("enable_rgbd")
    enable_ground_truth = LaunchConfiguration("enable_ground_truth")
    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([package_share, "urdf", "robocon25_mid360_robot.xacro"]), " ",
        "lidar_samples:=", lidar_samples, " ",
        "lidar_downsample:=", lidar_downsample, " ",
        "enable_rgbd:=", enable_rgbd, " ",
        "enable_ground_truth:=", enable_ground_truth,
    ])

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([gazebo_share, "launch", "gzserver.launch.py"])),
        launch_arguments={"world": world}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([gazebo_share, "launch", "gzclient.launch.py"])),
        condition=IfCondition(use_gui),
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        output="screen",
    )
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "robocon25_mid360_robot", "-topic", "robot_description", "-x", "0", "-y", "0", "-z", "0.05"],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([package_share, "worlds", "robocon25_candidate.world"]),
            description="Public candidate Gazebo world; it is not official-rule verified.",
        ),
        DeclareLaunchArgument("use_gui", default_value="false"),
        DeclareLaunchArgument(
            "lidar_samples",
            default_value="2000",
            description="Rays per 10 Hz simulated packet; 2000 is the WSL profile, 30000 is upstream high-density.",
        ),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument(
            "enable_rgbd",
            default_value="false",
            description="Enable the optional Gazebo RGB-D interface for synthetic perception tests.",
        ),
        DeclareLaunchArgument(
            "enable_ground_truth",
            default_value="false",
            description="Enable the noiseless Gazebo pose topic for simulation-only error measurement.",
        ),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", PathJoinSubstitution([package_share, "models"])),
        gzserver,
        state_publisher,
        TimerAction(period=2.0, actions=[spawn_robot]),
        gzclient,
    ])
