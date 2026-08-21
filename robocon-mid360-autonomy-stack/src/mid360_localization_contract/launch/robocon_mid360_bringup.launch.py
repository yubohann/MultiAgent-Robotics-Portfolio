"""Start the official Livox driver, FAST-LIO2, and the localization contract."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("mid360_localization_contract")
    contract_launch = package_share + "/launch/competition_localization.launch.py"

    start_driver = LaunchConfiguration("start_driver")
    start_fast_lio = LaunchConfiguration("start_fast_lio")
    driver_config_path = LaunchConfiguration("driver_config_path")
    fast_lio_params_file = LaunchConfiguration("fast_lio_params_file")
    contract_params_file = LaunchConfiguration("contract_params_file")

    livox_driver = Node(
        package="livox_ros_driver2",
        executable="livox_ros_driver2_node",
        name="livox_lidar_publisher",
        output="screen",
        parameters=[{
            "xfer_format": 1,
            "multi_topic": 0,
            "data_src": 0,
            "publish_freq": 10.0,
            "output_data_type": 0,
            "frame_id": "lidar_mid360",
            "user_config_path": driver_config_path,
        }],
        condition=IfCondition(start_driver),
    )

    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        output="screen",
        parameters=[fast_lio_params_file],
        condition=IfCondition(start_fast_lio),
    )

    contract = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(contract_launch),
        launch_arguments={"params_file": contract_params_file}.items(),
    )

    driver_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=livox_driver,
            on_exit=[
                LogInfo(msg="Livox driver exited; shutting down the MID-360 bringup."),
                EmitEvent(event=Shutdown(reason="livox_ros_driver2 exited")),
            ],
        )
    )
    fast_lio_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=fast_lio,
            on_exit=[
                LogInfo(msg="FAST-LIO2 exited; shutting down the MID-360 bringup."),
                EmitEvent(event=Shutdown(reason="fast_lio exited")),
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_driver",
            default_value="true",
            description="Start livox_ros_driver2 directly in this bringup.",
        ),
        DeclareLaunchArgument(
            "start_fast_lio",
            default_value="true",
            description="Start the selected FAST-LIO2 ROS2 executable.",
        ),
        DeclareLaunchArgument(
            "driver_config_path",
            default_value="",
            description="Absolute path to the private MID360_config.json.",
        ),
        DeclareLaunchArgument(
            "fast_lio_params_file",
            default_value="",
            description="Absolute path to the private FAST-LIO2 YAML.",
        ),
        DeclareLaunchArgument(
            "contract_params_file",
            default_value=package_share + "/config/competition.yaml",
            description="Path to the localization contract YAML.",
        ),
        livox_driver,
        fast_lio,
        contract,
        driver_exit_handler,
        fast_lio_exit_handler,
    ])
