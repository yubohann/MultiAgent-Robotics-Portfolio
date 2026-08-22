"""Start two basketball robot proxies on the local ROBOCON 2025 candidate court."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _robot_description(package_share, robot_ns, lidar_topic, lidar_node_name, imu_topic, cmd_topic,
                       color_topic, depth_topic, info_topic, lidar_samples,
                       lidar_downsample, enable_rgbd, enable_ground_truth):
    return Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([package_share, "urdf", "robocon25_mid360_robot.xacro"]), " ",
        "robot_ns:=", robot_ns, " ",
        "lidar_topic:=", lidar_topic, " ",
        "lidar_node_name:=", lidar_node_name, " ",
        "imu_topic:=", imu_topic, " ",
        "cmd_vel_topic:=", cmd_topic, " ",
        "camera_color_topic:=", color_topic, " ",
        "camera_depth_topic:=", depth_topic, " ",
        "camera_info_topic:=", info_topic, " ",
        "show_loaded_ball:=false ",
        "lidar_samples:=", lidar_samples, " ",
        "lidar_downsample:=", lidar_downsample, " ",
        "enable_rgbd:=", enable_rgbd, " ",
        "enable_ground_truth:=", enable_ground_truth,
    ])


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("robocon_mid360_simulation")
    gazebo_share = get_package_share_directory("gazebo_ros")
    world = LaunchConfiguration("world")
    use_gui = LaunchConfiguration("use_gui")
    lidar_samples = LaunchConfiguration("lidar_samples")
    lidar_downsample = LaunchConfiguration("lidar_downsample")
    enable_rgbd = LaunchConfiguration("enable_rgbd")
    enable_ground_truth = LaunchConfiguration("enable_ground_truth")
    allow_unsafe_density = LaunchConfiguration("allow_unsafe_density")

    def density_guard(context):
        samples = int(lidar_samples.perform(context))
        downsample = int(lidar_downsample.perform(context))
        unsafe_override = allow_unsafe_density.perform(context).lower() == "true"
        if samples >= 200000 and downsample < 4 and not unsafe_override:
            return [
                LogInfo(
                    msg=(
                        "ERROR: Refusing unsafe dual density: lidar_samples=%d with "
                        "lidar_downsample=%d exceeds the WSL memory profile. "
                        "Use lidar_downsample:=4 (recommended) or pass "
                        "allow_unsafe_density:=true only after increasing WSL memory."
                    ) % (samples, downsample)
                ),
                Shutdown(reason="unsafe dual MID-360 density blocked before ray allocation"),
            ]
        return []

    robot1_description = _robot_description(
        package_share, "/robot1", "/robot1/livox/lidar", "robot1_lidar_mid360_plugin", "/robot1/livox/imu",
        "/robot1/cmd_vel_chassis", "/robot1/camera/color/image_raw",
        "/robot1/camera/depth/image_raw", "/robot1/camera/color/camera_info",
        lidar_samples, lidar_downsample, enable_rgbd, enable_ground_truth,
    )
    robot2_description = _robot_description(
        package_share, "/robot2", "/robot2/livox/lidar", "robot2_lidar_mid360_plugin", "/robot2/livox/imu",
        "/robot2/cmd_vel_chassis", "/robot2/camera/color/image_raw",
        "/robot2/camera/depth/image_raw", "/robot2/camera/color/camera_info",
        lidar_samples, lidar_downsample, enable_rgbd, enable_ground_truth,
    )

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([gazebo_share, "launch", "gzserver.launch.py"])),
        launch_arguments={"world": world}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([gazebo_share, "launch", "gzclient.launch.py"])),
        condition=IfCondition(use_gui),
    )

    state1 = Node(
        package="robot_state_publisher", executable="robot_state_publisher", namespace="robot1",
        parameters=[{"use_sim_time": True, "robot_description": robot1_description}], output="screen",
    )
    state2 = Node(
        package="robot_state_publisher", executable="robot_state_publisher", namespace="robot2",
        parameters=[{"use_sim_time": True, "robot_description": robot2_description}], output="screen",
    )
    spawn1 = Node(
        package="gazebo_ros", executable="spawn_entity.py",
        arguments=["-entity", "robocon25_robot1", "-topic", "/robot1/robot_description",
                   "-x", "-3.0", "-y", "-1.35", "-z", "0.18"], output="screen",
    )
    spawn2 = Node(
        package="gazebo_ros", executable="spawn_entity.py",
        arguments=["-entity", "robocon25_robot2", "-topic", "/robot2/robot_description",
                   "-x", "2.2", "-y", "1.10", "-z", "0.18"], output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value=PathJoinSubstitution([
            package_share, "worlds", "robocon25_candidate.world"
        ]), description="Local ROBOCON 2025 candidate basketball court."),
        DeclareLaunchArgument("use_gui", default_value="false"),
        DeclareLaunchArgument("lidar_samples", default_value="30000"),
        DeclareLaunchArgument("lidar_downsample", default_value="1"),
        DeclareLaunchArgument(
            "allow_unsafe_density",
            default_value="false",
            description="Allow >=200000 samples with downsample <4; may OOM WSL.",
        ),
        DeclareLaunchArgument("enable_rgbd", default_value="false"),
        DeclareLaunchArgument("enable_ground_truth", default_value="false"),
        OpaqueFunction(function=density_guard),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH", PathJoinSubstitution([package_share, "models"])),
        gzserver,
        state1,
        state2,
        # Gazebo Classic serializes /spawn_entity handling. Starting both heavy
        # RGB-D + Livox models together can leave the first request blocked on
        # WSL, so make the two physical spawns deterministic and sequential.
        TimerAction(period=6.0, actions=[spawn1]),
        TimerAction(period=32.0, actions=[spawn2]),
        gzclient,
    ])
