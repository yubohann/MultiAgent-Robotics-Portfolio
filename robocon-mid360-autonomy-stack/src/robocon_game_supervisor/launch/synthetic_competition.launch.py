"""Run the supervisor, perception gate, and synthetic adapters locally."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("robocon_game_supervisor")
    params = share + "/config/supervisor.yaml"
    return LaunchDescription([
        Node(
            package="robocon_game_supervisor",
            executable="robocon_synthetic_inputs",
            name="robocon_synthetic_inputs",
            output="screen",
        ),
        Node(
            package="robocon_perception_adapter",
            executable="synthetic_target_source",
            name="robocon_synthetic_target_source",
            output="screen",
        ),
        Node(
            package="robocon_perception_adapter",
            executable="target_gate",
            name="robocon_target_gate",
            output="screen",
        ),
        Node(
            package="robocon_game_supervisor",
            executable="robocon_game_supervisor",
            name="robocon_game_supervisor",
            output="screen",
            parameters=[params, {"task_id": "synthetic"}],
        ),
        Node(
            package="robocon_game_supervisor",
            executable="robocon_action_simulator",
            name="robocon_action_simulator",
            output="screen",
            parameters=[params, {"task_id": "synthetic"}],
        ),
    ])
