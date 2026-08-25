#!/usr/bin/env python3
"""Publish a fixed-range mono8 view of a Gazebo 32FC1 depth image."""

from __future__ import annotations

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class DepthVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("depth_visualizer")
        self.input_topic = self.declare_parameter(
            "input_topic", "/robot1/simulated_rgbd_camera/depth/image_raw"
        ).value
        self.output_topic = self.declare_parameter(
            "output_topic", "/robot1/simulated_rgbd_camera/depth/image_visualized"
        ).value
        self.min_depth = float(self.declare_parameter("min_depth", 0.2).value)
        self.max_depth = float(self.declare_parameter("max_depth", 12.0).value)
        if not 0.0 < self.min_depth < self.max_depth:
            raise ValueError("depth range must satisfy 0 < min_depth < max_depth")
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.subscription = self.create_subscription(
            Image, self.input_topic, self._on_depth, qos_profile_sensor_data
        )

    def _on_depth(self, message: Image) -> None:
        encoding = str(message.encoding).lower()
        if encoding in {"32fc1", "32fc"}:
            value_format, value_size, scale = "f", 4, 1.0
        elif encoding in {"16uc1", "16uc", "mono16"}:
            value_format, value_size, scale = "H", 2, 0.001
        else:
            self.get_logger().warning("unsupported depth encoding: %s", message.encoding)
            return

        width, height = int(message.width), int(message.height)
        output = bytearray(width * height)
        for row in range(height):
            source_row = row * int(message.step)
            for column in range(width):
                offset = source_row + column * value_size
                if offset + value_size > len(message.data):
                    continue
                value = float(struct.unpack_from("<" + value_format, message.data, offset)[0]) * scale
                if not math.isfinite(value) or value <= self.min_depth:
                    continue
                normalized = (self.max_depth - min(self.max_depth, value)) / (
                    self.max_depth - self.min_depth
                )
                output[row * width + column] = int(round(255.0 * normalized))

        visualized = Image()
        visualized.header = message.header
        visualized.height = height
        visualized.width = width
        visualized.encoding = "mono8"
        visualized.is_bigendian = False
        visualized.step = width
        visualized.data = bytes(output)
        self.publisher.publish(visualized)


def main() -> int:
    rclpy.init()
    node = DepthVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
