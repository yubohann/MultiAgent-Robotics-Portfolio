#!/usr/bin/env python3
"""Confirm that the simulator publishes both required sensor topics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class StreamProbe(Node):
    def __init__(self, minimum_lidar_messages: int, minimum_imu_messages: int) -> None:
        super().__init__("mid360_simulation_stream_probe")
        self.minimum_lidar_messages = minimum_lidar_messages
        self.minimum_imu_messages = minimum_imu_messages
        self.lidar: dict[str, int] | None = None
        self.imu: dict[str, str] | None = None
        self.lidar_received = 0
        self.lidar_accepted = 0
        self.imu_received = 0
        self.imu_accepted = 0
        self.create_subscription(CustomMsg, "/livox/lidar", self._lidar, qos_profile_sensor_data)
        self.create_subscription(Imu, "/livox/imu", self._imu, qos_profile_sensor_data)

    def _lidar(self, message: CustomMsg) -> None:
        self.lidar_received += 1
        if int(message.point_num) == len(message.points) and len(message.points) > 0:
            self.lidar_accepted += 1
        if self.lidar is None or self.lidar_accepted >= self.minimum_lidar_messages:
            self.lidar = {
                "point_num": int(message.point_num),
                "array_length": len(message.points),
                "timebase": int(message.timebase),
            }

    def _imu(self, message: Imu) -> None:
        self.imu_received += 1
        self.imu_accepted += 1
        if self.imu is None or self.imu_accepted >= self.minimum_imu_messages:
            self.imu = {"frame_id": message.header.frame_id}

    @property
    def ready(self) -> bool:
        return (
            self.lidar_accepted >= self.minimum_lidar_messages
            and self.imu_accepted >= self.minimum_imu_messages
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--minimum-lidar-messages", type=int, default=1)
    parser.add_argument("--minimum-imu-messages", type=int, default=1)
    args = parser.parse_args()
    if args.minimum_lidar_messages <= 0 or args.minimum_imu_messages <= 0:
        raise SystemExit("minimum message counts must be positive")

    rclpy.init()
    node = StreamProbe(args.minimum_lidar_messages, args.minimum_imu_messages)
    deadline = time.monotonic() + args.timeout_sec
    try:
        while time.monotonic() < deadline and not node.ready:
            rclpy.spin_once(node, timeout_sec=0.2)
        payload = {
            "lidar": node.lidar,
            "imu": node.imu,
            "timeout_sec": args.timeout_sec,
            "minimum_lidar_messages": args.minimum_lidar_messages,
            "minimum_imu_messages": args.minimum_imu_messages,
            "lidar_received": node.lidar_received,
            "lidar_accepted": node.lidar_accepted,
            "imu_received": node.imu_received,
            "imu_accepted": node.imu_accepted,
            "ready": node.ready,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return 0 if payload["ready"] else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
