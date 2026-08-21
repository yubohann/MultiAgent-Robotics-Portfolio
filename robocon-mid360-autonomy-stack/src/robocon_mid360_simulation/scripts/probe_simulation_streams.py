#!/usr/bin/env python3
"""Confirm that the simulator publishes both required sensor topics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class StreamProbe(Node):
    def __init__(
        self,
        minimum_lidar_messages: int,
        minimum_imu_messages: int,
        lidar_topic: str,
        imu_topic: str,
    ) -> None:
        super().__init__("mid360_simulation_stream_probe")
        self.minimum_lidar_messages = minimum_lidar_messages
        self.minimum_imu_messages = minimum_imu_messages
        self.lidar: dict[str, int] | None = None
        self.imu: dict[str, str] | None = None
        self.lidar_received = 0
        self.lidar_accepted = 0
        self.imu_received = 0
        self.imu_accepted = 0
        self.previous_lidar_timebase: int | None = None
        self.previous_lidar_end_time: int | None = None
        self.lidar_timebase_monotonic = True
        self.lidar_packets_non_overlapping = True
        self.create_subscription(CustomMsg, lidar_topic, self._lidar, qos_profile_sensor_data)
        self.create_subscription(Imu, imu_topic, self._imu, qos_profile_sensor_data)

    def _lidar(self, message: CustomMsg) -> None:
        self.lidar_received += 1
        if int(message.point_num) == len(message.points) and len(message.points) > 0:
            self.lidar_accepted += 1
        azimuth_bins = [0] * 12
        elevations: list[float] = []
        previous_offset = -1
        offsets_monotonic = True
        finite_points = 0
        for point in message.points:
            offset = int(point.offset_time)
            offsets_monotonic = offsets_monotonic and offset >= previous_offset
            previous_offset = offset
            x, y, z = float(point.x), float(point.y), float(point.z)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            planar_range = math.hypot(x, y)
            if planar_range == 0.0 and z == 0.0:
                continue
            finite_points += 1
            azimuth = math.atan2(y, x)
            azimuth_index = min(11, int((azimuth + math.pi) / (2.0 * math.pi) * 12.0))
            azimuth_bins[azimuth_index] += 1
            elevations.append(math.degrees(math.atan2(z, planar_range)))
        if int(message.point_num) == len(message.points) and len(message.points) > 0:
            timebase = int(message.timebase)
            end_time = timebase + max(0, previous_offset)
            if self.previous_lidar_timebase is not None:
                self.lidar_timebase_monotonic = (
                    self.lidar_timebase_monotonic
                    and timebase >= self.previous_lidar_timebase
                )
            if self.previous_lidar_end_time is not None:
                self.lidar_packets_non_overlapping = (
                    self.lidar_packets_non_overlapping
                    and timebase >= self.previous_lidar_end_time
                )
            self.previous_lidar_timebase = timebase
            self.previous_lidar_end_time = end_time
            if self.lidar is None or self.lidar_accepted >= self.minimum_lidar_messages:
                self.lidar = {
                    "point_num": int(message.point_num),
                    "array_length": len(message.points),
                    "timebase": timebase,
                    "finite_points": finite_points,
                    "azimuth_bins": azimuth_bins,
                    "azimuth_nonempty_bins": sum(count > 0 for count in azimuth_bins),
                    "elevation_min_deg": min(elevations) if elevations else None,
                    "elevation_max_deg": max(elevations) if elevations else None,
                    "offset_time_monotonic": offsets_monotonic,
                    "offset_span_ns": max(0, previous_offset),
                    "timebase_monotonic": self.lidar_timebase_monotonic,
                    "packets_non_overlapping": self.lidar_packets_non_overlapping,
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
            and self.lidar is not None
            and bool(self.lidar["offset_time_monotonic"])
            and bool(self.lidar["timebase_monotonic"])
            and bool(self.lidar["packets_non_overlapping"])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--minimum-lidar-messages", type=int, default=1)
    parser.add_argument("--minimum-imu-messages", type=int, default=1)
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--imu-topic", default="/livox/imu")
    args = parser.parse_args()
    if args.minimum_lidar_messages <= 0 or args.minimum_imu_messages <= 0:
        raise SystemExit("minimum message counts must be positive")

    rclpy.init()
    node = StreamProbe(
        args.minimum_lidar_messages,
        args.minimum_imu_messages,
        args.lidar_topic,
        args.imu_topic,
    )
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
            "lidar_topic": args.lidar_topic,
            "imu_topic": args.imu_topic,
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
