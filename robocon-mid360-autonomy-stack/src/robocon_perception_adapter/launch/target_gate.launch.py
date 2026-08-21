from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="robocon_perception_adapter",
            executable="target_gate",
            name="robocon_target_gate",
            output="screen",
        ),
    ])
