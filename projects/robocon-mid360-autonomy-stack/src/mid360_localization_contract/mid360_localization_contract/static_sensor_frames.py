"""Publish static sensor frames only after the mechanical measurements are approved."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .geometry import quaternion_from_rpy


class Mid360StaticSensorFrames(Node):
    def __init__(self) -> None:
        super().__init__("mid360_static_sensor_frames")
        self.declare_parameter("enabled", False)
        self.declare_parameter("calibration_ready", False)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("imu_frame", "imu_link")
        self.declare_parameter("lidar_frame", "lidar_mid360")
        self.declare_parameter("base_to_imu_translation_m", [0.0, 0.0, 0.0])
        self.declare_parameter("base_to_imu_rpy_rad", [0.0, 0.0, 0.0])
        self.declare_parameter("imu_to_lidar_translation_m", [0.0, 0.0, 0.0])
        self.declare_parameter("imu_to_lidar_rpy_rad", [0.0, 0.0, 0.0])

        enabled = bool(self.get_parameter("enabled").value)
        calibration_ready = bool(self.get_parameter("calibration_ready").value)
        if not enabled:
            self.get_logger().warn("Static sensor TF publication is disabled.")
            return
        if not calibration_ready:
            raise ValueError("Static sensor TF publication refused because calibration_ready is false.")

        self.broadcaster = StaticTransformBroadcaster(self)
        transforms = [
            self._make_transform(
                str(self.get_parameter("base_frame").value),
                str(self.get_parameter("imu_frame").value),
                self.get_parameter("base_to_imu_translation_m").value,
                self.get_parameter("base_to_imu_rpy_rad").value,
            ),
            self._make_transform(
                str(self.get_parameter("imu_frame").value),
                str(self.get_parameter("lidar_frame").value),
                self.get_parameter("imu_to_lidar_translation_m").value,
                self.get_parameter("imu_to_lidar_rpy_rad").value,
            ),
        ]
        self.broadcaster.sendTransform(transforms)
        self.get_logger().info("Published the approved base_link, IMU, and MID-360 static transforms.")

    def _make_transform(self, parent: str, child: str, translation_values, rpy_values) -> TransformStamped:
        if parent == child:
            raise ValueError("A static transform cannot have identical parent and child frames.")
        if len(translation_values) != 3 or len(rpy_values) != 3:
            raise ValueError("Static transform translation and RPY parameters require three values.")
        rotation = quaternion_from_rpy(*(float(value) for value in rpy_values))
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = float(translation_values[0])
        transform.transform.translation.y = float(translation_values[1])
        transform.transform.translation.z = float(translation_values[2])
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
        return transform


def main() -> None:
    rclpy.init()
    node = Mid360StaticSensorFrames()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
