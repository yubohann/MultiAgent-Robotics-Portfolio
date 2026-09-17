from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="camera_yolo_adapter",
            executable="camera_yolo_adapter",
            name="camera_yolo_adapter",
            output="screen",
        ),
    ])
