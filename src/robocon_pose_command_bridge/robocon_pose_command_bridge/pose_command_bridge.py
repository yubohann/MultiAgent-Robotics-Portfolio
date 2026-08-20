"""Publish safety-gated pose commands for the legacy serial control boundary."""

from __future__ import annotations

import json
import time
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .core import format_command, make_pose_command, pose_command_gate, quaternion_to_yaw


class PoseCommandBridge(Node):
    def __init__(self) -> None:
        super().__init__("robocon_pose_command_bridge")
        self.target_x = float(self.declare_parameter("target_x", 4.0).value)
        self.target_y = float(self.declare_parameter("target_y", 15.0).value)
        self.pose_topic = str(self.declare_parameter("pose_topic", "/mid360/localization_odometry").value)
        self.pose_valid_topic = str(self.declare_parameter("pose_valid_topic", "/mid360/pose_valid").value)
        self.map_locked_topic = str(self.declare_parameter("map_locked_topic", "/mid360/map_locked").value)
        self.command_topic = str(self.declare_parameter("command_topic", "/robocon/control/pose_target").value)
        self.max_pose_age_sec = float(self.declare_parameter("max_pose_age_sec", 0.30).value)
        self.publish_hz = float(self.declare_parameter("publish_hz", 10.0).value)
        self.serial_enabled = bool(self.declare_parameter("serial_enabled", False).value)
        self.serial_port = str(self.declare_parameter("serial_port", "/dev/ttyESP").value)
        self.serial_baudrate = int(self.declare_parameter("serial_baudrate", 115200).value)

        if self.max_pose_age_sec <= 0.0 or self.publish_hz <= 0.0:
            raise ValueError("max_pose_age_sec and publish_hz must be positive")

        self.pose: Optional[Odometry] = None
        self.pose_received_at: Optional[float] = None
        self.pose_valid = False
        self.map_locked = False
        self.serial = None
        if self.serial_enabled:
            try:
                import serial

                self.serial = serial.Serial(self.serial_port, self.serial_baudrate, timeout=1.0)
            except Exception as exc:  # pragma: no cover - hardware-only path
                self.get_logger().error("serial_enabled but port could not be opened: %s", exc)
                raise

        self.command_pub = self.create_publisher(String, self.command_topic, 10)
        self.create_subscription(Odometry, self.pose_topic, self._pose_callback, 10)
        self.create_subscription(Bool, self.pose_valid_topic, self._pose_valid_callback, 10)
        self.create_subscription(Bool, self.map_locked_topic, self._map_locked_callback, 10)
        self.create_timer(1.0 / self.publish_hz, self._publish_command)

    def _pose_callback(self, message: Odometry) -> None:
        if message.header.frame_id != "map" or message.child_frame_id != "base_link":
            self.get_logger().warning("Ignoring pose with frame %s -> %s", message.header.frame_id, message.child_frame_id)
            return
        self.pose = message
        self.pose_received_at = time.monotonic()

    def _pose_valid_callback(self, message: Bool) -> None:
        self.pose_valid = bool(message.data)

    def _map_locked_callback(self, message: Bool) -> None:
        self.map_locked = bool(message.data)

    def _publish_command(self) -> None:
        pose_age = (
            time.monotonic() - self.pose_received_at
            if self.pose_received_at is not None
            else float("inf")
        )
        allowed, _reason = pose_command_gate(
            pose_available=self.pose is not None,
            pose_valid=self.pose_valid,
            map_locked=self.map_locked,
            pose_age_sec=pose_age,
            max_pose_age_sec=self.max_pose_age_sec,
        )
        if not allowed:
            return
        try:
            orientation = self.pose.pose.pose.orientation
            yaw = quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
            position = self.pose.pose.pose.position
            command = make_pose_command(position.x, position.y, yaw, self.target_x, self.target_y)
        except ValueError as exc:
            self.get_logger().warning("Pose rejected: %s", exc)
            return
        stamp = self.get_clock().now().nanoseconds
        payload = format_command(command, stamp)
        self.command_pub.publish(String(data=payload))
        if self.serial is not None:  # pragma: no cover - hardware-only path
            self.serial.write((payload + "\n").encode("utf-8"))

    def destroy_node(self):
        if self.serial is not None:  # pragma: no cover - hardware-only path
            self.serial.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseCommandBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
