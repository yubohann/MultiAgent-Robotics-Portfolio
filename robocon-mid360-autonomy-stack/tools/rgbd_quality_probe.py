#!/usr/bin/env python3
"""Measure the content and timing contract of the Gazebo RGB-D topics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def _stamp_seconds(message: Any) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9


def _finite_depth_samples(message: Image, limit: int = 12000) -> tuple[int, int, list[float]]:
    """Return (valid, total, values_m) using a bounded spatial sample."""
    encoding = str(message.encoding).lower()
    if message.width <= 0 or message.height <= 0 or message.step <= 0:
        return 0, 0, []
    if encoding in {"32fc1", "32fc"}:
        fmt, size, scale = "<f", 4, 1.0
    elif encoding in {"16uc1", "16uc", "mono16"}:
        fmt, size, scale = "<H", 2, 0.001
    else:
        return 0, 0, []
    stride = max(1, int(math.sqrt((message.width * message.height) / limit)))
    valid = total = 0
    values: list[float] = []
    for row in range(0, message.height, stride):
        base = row * message.step
        for column in range(0, message.width, stride):
            offset = base + column * size
            if offset + size > len(message.data):
                continue
            total += 1
            value = float(struct.unpack_from(fmt, message.data, offset)[0]) * scale
            if math.isfinite(value) and value > 0.0:
                valid += 1
                if len(values) < limit:
                    values.append(value)
    return valid, total, values


class RgbdQualityProbe(Node):
    def __init__(self, run_dir: Path, duration_sec: float) -> None:
        super().__init__("rgbd_quality_probe")
        self.run_dir = run_dir
        self.duration_sec = duration_sec
        self.started = time.monotonic()
        self.finished = False
        self.color_count = 0
        self.depth_count = 0
        self.info_count = 0
        self.color_meta: dict[str, Any] = {}
        self.depth_meta: dict[str, Any] = {}
        self.info_meta: dict[str, Any] = {}
        self.color_arrival: list[float] = []
        self.depth_arrival: list[float] = []
        self.color_stamps: list[float] = []
        self.depth_stamps: list[float] = []
        self.skews: list[float] = []
        self.depth_valid = 0
        self.depth_total = 0
        self.depth_values: list[float] = []
        self._last_color_stamp: float | None = None
        self._last_depth_stamp: float | None = None
        self.create_subscription(Image, "/simulated_rgbd_camera/image_raw", self._color, qos_profile_sensor_data)
        self.create_subscription(Image, "/simulated_rgbd_camera/depth/image_raw", self._depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/simulated_rgbd_camera/camera_info", self._info, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)

    def _color(self, message: Image) -> None:
        self.color_count += 1
        stamp = _stamp_seconds(message)
        self.color_stamps.append(stamp)
        self.color_arrival.append(time.monotonic())
        self._last_color_stamp = stamp
        if not self.color_meta:
            self.color_meta = {
                "width": int(message.width), "height": int(message.height),
                "encoding": str(message.encoding), "step": int(message.step),
                "frame_id": str(message.header.frame_id),
            }
        self._record_skew()

    def _depth(self, message: Image) -> None:
        self.depth_count += 1
        stamp = _stamp_seconds(message)
        self.depth_stamps.append(stamp)
        self.depth_arrival.append(time.monotonic())
        self._last_depth_stamp = stamp
        if not self.depth_meta:
            self.depth_meta = {
                "width": int(message.width), "height": int(message.height),
                "encoding": str(message.encoding), "step": int(message.step),
                "frame_id": str(message.header.frame_id),
            }
        valid, total, values = _finite_depth_samples(message)
        self.depth_valid += valid
        self.depth_total += total
        self.depth_values.extend(values)
        if len(self.depth_values) > 12000:
            self.depth_values = self.depth_values[-12000:]
        self._record_skew()

    def _record_skew(self) -> None:
        if self._last_color_stamp is not None and self._last_depth_stamp is not None:
            self.skews.append(abs(self._last_color_stamp - self._last_depth_stamp))

    def _info(self, message: CameraInfo) -> None:
        self.info_count += 1
        if not self.info_meta:
            self.info_meta = {
                "width": int(message.width), "height": int(message.height),
                "frame_id": str(message.header.frame_id),
                "distortion_model": str(message.distortion_model),
                "fx": float(message.k[0]), "fy": float(message.k[4]),
                "cx": float(message.k[2]), "cy": float(message.k[5]),
            }

    def _tick(self) -> None:
        if time.monotonic() - self.started >= self.duration_sec:
            self.finished = True

    def summary(self) -> dict[str, Any]:
        color_rate = self.color_count / max(1e-9, time.monotonic() - self.started)
        depth_rate = self.depth_count / max(1e-9, time.monotonic() - self.started)
        valid_ratio = self.depth_valid / self.depth_total if self.depth_total else None
        return {
            "evidence_level": "gazebo_simulation",
            "outcome": "passed" if self.color_count > 0 and self.depth_count > 0 and self.info_count > 0 else "failed",
            "duration_sec": self.duration_sec,
            "color_samples": self.color_count,
            "depth_samples": self.depth_count,
            "camera_info_samples": self.info_count,
            "color_rate_hz_wall": color_rate,
            "depth_rate_hz_wall": depth_rate,
            "color": self.color_meta,
            "depth": self.depth_meta,
            "camera_info": self.info_meta,
            "depth_sample_count": self.depth_total,
            "depth_valid_count": self.depth_valid,
            "depth_valid_ratio": valid_ratio,
            "depth_min_m": min(self.depth_values) if self.depth_values else None,
            "depth_median_m": statistics.median(self.depth_values) if self.depth_values else None,
            "depth_max_m": max(self.depth_values) if self.depth_values else None,
            "rgb_depth_stamp_skew_p95_sec": _percentile(self.skews, 0.95),
            "rgb_depth_stamp_skew_max_sec": max(self.skews) if self.skews else None,
            "notes": [
                "Content and timing measurements are from the local Gazebo simulation.",
                "No detector precision, recall, calibration, or physical-camera claim is made.",
            ],
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--max-wall-sec", type=float, default=60.0)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.max_wall_sec <= 0:
        parser.error("durations must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RgbdQualityProbe(args.run_dir, args.duration_sec)
    deadline = time.monotonic() + args.max_wall_sec
    try:
        while rclpy.ok() and not node.finished and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        summary = node.summary()
        if not node.finished:
            summary["outcome"] = "failed"
            summary["failure_reason"] = "wall-clock timeout"
        (args.run_dir / "rgbd_quality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if summary["outcome"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
