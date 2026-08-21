#!/usr/bin/env python3
"""Run a bounded Gazebo motion sequence and record LIO evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from gazebo_msgs.msg import PerformanceMetrics
from geometry_msgs.msg import Twist
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from run_contract import StreamReadiness


SEGMENTS = (
    ("stationary", 5.0, (0.0, 0.0, 0.0)),
    ("forward", 15.0, (0.25, 0.0, 0.0)),
    ("lateral", 15.0, (0.0, 0.25, 0.0)),
    ("yaw_rotation", 15.0, (0.0, 0.0, 0.35)),
    ("stop", 10.0, (0.0, 0.0, 0.0)),
)
SEQUENCE_DURATION_SEC = sum(item[1] for item in SEGMENTS)


def _stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _diagnostic_values(message: DiagnosticArray) -> dict[str, str]:
    values: dict[str, str] = {}
    for status in message.status:
        values[f"{status.name}.level"] = str(status.level)
        values[f"{status.name}.message"] = status.message
        for item in status.values:
            values[item.key] = item.value
    return values


class ControlledLioRecorder(Node):
    def __init__(
        self,
        run_dir: Path,
        duration_sec: float,
        motion_scale: float,
        ready_timeout_sec: float,
        required_lidar_packets: int,
        required_imu_packets: int,
    ) -> None:
        super().__init__(
            "controlled_lio_recorder",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self.run_dir = run_dir
        self.events = (run_dir / "telemetry.jsonl").open("w", encoding="utf-8")
        self.commands = (run_dir / "commands.jsonl").open("w", encoding="utf-8")
        self.created_wall = time.monotonic()
        self.ready_wall: float | None = None
        self.ready_sim_ns: int | None = None
        self.finished_wall: float | None = None
        self.duration_sec = duration_sec
        self.motion_scale = motion_scale
        self.ready_timeout_sec = ready_timeout_sec
        self.readiness = StreamReadiness(required_lidar_packets, required_imu_packets)
        self.segment_index = 0
        self.sequence_index = 0
        self.samples = {"lidar": 0, "imu": 0, "odom": 0, "diagnostic": 0, "performance": 0}
        self.last_lidar_stamp_ns: int | None = None
        self.last_imu_stamp_ns: int | None = None
        self.last_odom_arrival: float | None = None
        self.odom_intervals: list[float] = []
        self.pose_ages: list[float] = []
        self.lidar_raw_points: list[int] = []
        self.lidar_finite_points: list[int] = []
        self.lidar_offset_spans_ns: list[int] = []
        self.lidar_stamps_ns: list[int] = []
        self.odom_stamps_ns: list[int] = []
        self.imu_intervals: list[float] = []
        self.last_imu_arrival: float | None = None
        self.error_counts: dict[str, int] = {}
        self.startup_error_counts: dict[str, int] = {}
        self.startup_diagnostic_count = 0
        self.real_time_factors: list[float] = []
        self.finished = False
        self.outcome = "running"
        self.failure_reason = ""
        self.exit_code = 0

        self.cmd_publisher = self.create_publisher(Twist, "/cmd_vel_chassis", 10)
        self.create_subscription(CustomMsg, "/livox/lidar", self._lidar_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, "/livox/imu", self._imu_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/mid360/local_odometry", self._odom_callback, qos_profile_sensor_data)
        self.create_subscription(PerformanceMetrics, "/gazebo/performance_metrics", self._performance_callback, 10)
        for topic in (
            "/mid360/input_diagnostics",
            "/mid360/pose_diagnostics",
            "/mid360/localization_diagnostics",
            "/mid360/preflight_diagnostics",
        ):
            self.create_subscription(DiagnosticArray, topic, self._diagnostic_callback, 10)

    @property
    def elapsed(self) -> float:
        if self.ready_sim_ns is None:
            return 0.0
        return max(0.0, (int(self.get_clock().now().nanoseconds) - self.ready_sim_ns) / 1e9)

    @property
    def readiness_wait_sec(self) -> float:
        end = self.ready_wall if self.ready_wall is not None else time.monotonic()
        return max(0.0, end - self.created_wall)

    def _write(self, stream: Any, payload: dict[str, Any]) -> None:
        payload.setdefault("wall_time_unix", time.time())
        payload.setdefault("elapsed_sec", self.elapsed)
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()

    def _lidar_callback(self, message: CustomMsg) -> None:
        now = time.monotonic()
        stamp_ns = _stamp_ns(message)
        points = list(message.points)
        finite = sum(
            1
            for point in points
            if all(math.isfinite(float(value)) for value in (point.x, point.y, point.z))
        )
        offsets = [int(point.offset_time) for point in points]
        self.samples["lidar"] += 1
        self.lidar_raw_points.append(len(points))
        self.lidar_finite_points.append(finite)
        self.lidar_offset_spans_ns.append(max(offsets) - min(offsets) if offsets else 0)
        self.readiness.observe_lidar(
            len(points) > 0 and finite > 0 and int(message.point_num) == len(points)
        )
        if self.last_lidar_stamp_ns is not None and stamp_ns > self.last_lidar_stamp_ns:
            interval = (stamp_ns - self.last_lidar_stamp_ns) / 1e9
            if interval < 2.0:
                self._record_interval("lidar_period_sec", interval)
        self.last_lidar_stamp_ns = stamp_ns
        self.lidar_stamps_ns.append(stamp_ns)
        self._write(self.events, {
            "kind": "lidar",
            "sequence": self.samples["lidar"],
            "stamp_ns": stamp_ns,
            "point_count": len(points),
            "finite_point_count": finite,
            "offset_span_ns": self.lidar_offset_spans_ns[-1],
            "timebase": int(message.timebase),
        })
        self._start_motion_if_ready()

    def _imu_callback(self, message: Imu) -> None:
        now = time.monotonic()
        stamp_ns = _stamp_ns(message)
        self.samples["imu"] += 1
        self.readiness.observe_imu(all(math.isfinite(float(value)) for value in (
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        )))
        if self.last_imu_arrival is not None:
            self.imu_intervals.append(now - self.last_imu_arrival)
        self.last_imu_arrival = now
        self.last_imu_stamp_ns = stamp_ns
        self._write(self.events, {
            "kind": "imu",
            "sequence": self.samples["imu"],
            "stamp_ns": stamp_ns,
            "frame_id": message.header.frame_id,
        })
        self._start_motion_if_ready()

    def _odom_callback(self, message: Odometry) -> None:
        now = time.monotonic()
        self.samples["odom"] += 1
        self.odom_stamps_ns.append(_stamp_ns(message))
        if self.last_odom_arrival is not None:
            self.odom_intervals.append(now - self.last_odom_arrival)
        self.last_odom_arrival = now
        ros_now_ns = int(self.get_clock().now().nanoseconds)
        stamp_ns = _stamp_ns(message)
        age = (ros_now_ns - stamp_ns) / 1e9 if ros_now_ns > 0 and stamp_ns > 0 else None
        # Do not mix startup samples with the motion-window latency metric.
        # Before the readiness gate, Gazebo may still publish a zero/old clock
        # stamp and produce an artificial multi-year pose age.
        if self.ready_wall is not None and age is not None and age >= 0.0:
            self.pose_ages.append(age)
        self._write(self.events, {
            "kind": "odometry",
            "sequence": self.samples["odom"],
            "stamp_ns": _stamp_ns(message),
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "pose_age_sec": age,
            "x": message.pose.pose.position.x,
            "y": message.pose.pose.position.y,
            "z": message.pose.pose.position.z,
        })

    def _diagnostic_callback(self, message: DiagnosticArray) -> None:
        values = _diagnostic_values(message)
        self.samples["diagnostic"] += 1
        ready = self.ready_wall is not None
        if not ready:
            self.startup_diagnostic_count += 1
        for key, value in values.items():
            if key.endswith("pose_age_sec"):
                try:
                    age = float(value)
                except ValueError:
                    continue
                if self.ready_wall is not None and math.isfinite(age):
                    self.pose_ages.append(age)
        for key, value in values.items():
            lowered = value.lower()
            if any(token in lowered for token in ("no point", "too few", "effective points", "error", "invalid")):
                counts = self.error_counts if ready else self.startup_error_counts
                counts[value] = counts.get(value, 0) + 1
        self._write(self.events, {"kind": "diagnostic", "values": values})

    def _performance_callback(self, message: PerformanceMetrics) -> None:
        self.samples["performance"] += 1
        factor = float(message.real_time_factor)
        if math.isfinite(factor) and factor >= 0.0:
            self.real_time_factors.append(factor)
        self._write(self.events, {
            "kind": "performance",
            "real_time_factor": factor,
            "sensor_count": len(message.sensors),
        })

    def _record_interval(self, key: str, value: float) -> None:
        self._write(self.events, {"kind": "interval", "metric": key, "value": value})

    def _start_motion_if_ready(self) -> None:
        if self.ready_wall is not None or not self.readiness.ready:
            return
        self.ready_wall = time.monotonic()
        self.ready_sim_ns = int(self.get_clock().now().nanoseconds)
        self._write(self.events, {
            "kind": "input_ready",
            "wall_wait_sec": self.readiness_wait_sec,
            "sim_start_ns": self.ready_sim_ns,
            "readiness": self.readiness.snapshot(),
        })

    def control_tick(self) -> None:
        if self.ready_wall is None:
            if self.readiness_wait_sec >= self.ready_timeout_sec:
                self._finish("input_not_ready", "LiDAR and IMU readiness thresholds were not met")
            return
        if self.elapsed >= self.duration_sec:
            self._finish()
            return
        # Repeat the full motion pattern when a longer mapping run is
        # requested.  Previously the recorder stopped at the end of the
        # first 60-second sequence even if duration_sec was larger, which
        # made a "longer" run collect no additional coverage.
        cycle_elapsed = self.elapsed % SEQUENCE_DURATION_SEC
        cycle_index = int(self.elapsed // SEQUENCE_DURATION_SEC)
        elapsed_until_segment = 0.0
        self.segment_index = 0
        for index, (_, segment_duration, _) in enumerate(SEGMENTS):
            if cycle_elapsed < elapsed_until_segment + segment_duration:
                self.segment_index = index
                break
            elapsed_until_segment += segment_duration
        self.sequence_index = cycle_index
        name, _, (vx, vy, wz) = SEGMENTS[self.segment_index]
        command = Twist()
        command.linear.x = vx * self.motion_scale
        command.linear.y = vy * self.motion_scale
        command.angular.z = wz * self.motion_scale
        self.cmd_publisher.publish(command)
        self._write(self.commands, {
            "segment": name,
            "sequence_index": self.sequence_index,
            "motion_scale": self.motion_scale,
            "linear_x_mps": command.linear.x,
            "linear_y_mps": command.linear.y,
            "angular_z_radps": command.angular.z,
        })

    def _finish(self, outcome: str = "complete", failure_reason: str = "") -> None:
        if self.finished:
            return
        self.cmd_publisher.publish(Twist())
        self.finished_wall = time.monotonic()
        self.outcome = outcome
        self.failure_reason = failure_reason
        self.exit_code = 0 if outcome == "complete" else 2
        motion_wall_sec = (
            self.finished_wall - self.ready_wall if self.ready_wall is not None else 0.0
        )
        sim_to_wall_ratio = (
            self.elapsed / motion_wall_sec
            if motion_wall_sec > 0.0 and self.elapsed >= 0.0
            else None
        )
        summary = {
            "kind": "summary",
            "outcome": self.outcome,
            "failure_reason": self.failure_reason or None,
            "duration_sec": self.elapsed,
            "motion_duration_target_sec": self.duration_sec,
            "readiness_wait_wall_sec": self.readiness_wait_sec,
            "motion_wall_sec": motion_wall_sec,
            # This is a derived simulation/wall-clock ratio, not Gazebo's
            # official real-time-factor metric.
            "sim_to_wall_ratio": sim_to_wall_ratio,
            "readiness": self.readiness.snapshot(),
            "startup_diagnostic_count": self.startup_diagnostic_count,
            "startup_error_counts": self.startup_error_counts,
            "samples": self.samples,
            "lidar_raw_points_min": min(self.lidar_raw_points) if self.lidar_raw_points else None,
            "lidar_raw_points_median": sorted(self.lidar_raw_points)[len(self.lidar_raw_points) // 2] if self.lidar_raw_points else None,
            "lidar_finite_points_min": min(self.lidar_finite_points) if self.lidar_finite_points else None,
            "offset_span_ns_median": sorted(self.lidar_offset_spans_ns)[len(self.lidar_offset_spans_ns) // 2] if self.lidar_offset_spans_ns else None,
            "odom_rate_hz": 1.0 / (sum(self.odom_intervals) / len(self.odom_intervals)) if self.odom_intervals else 0.0,
            "imu_rate_hz": 1.0 / (sum(self.imu_intervals) / len(self.imu_intervals)) if self.imu_intervals else 0.0,
            "lidar_wall_rate_hz": _wall_rate(self.lidar_stamps_ns, motion_wall_sec),
            "lidar_sim_rate_hz": _sim_rate(self.lidar_stamps_ns),
            "lidar_sim_span_sec": _sim_span(self.lidar_stamps_ns),
            "odom_wall_rate_hz": _wall_rate(self.odom_stamps_ns, motion_wall_sec),
            "odom_sim_rate_hz": _sim_rate(self.odom_stamps_ns),
            "odom_sim_span_sec": _sim_span(self.odom_stamps_ns),
            "pose_age_p95_sec": _percentile(self.pose_ages, 0.95),
            "gazebo_real_time_factor_median": _percentile(self.real_time_factors, 0.5),
            "gazebo_real_time_factor_p05": _percentile(self.real_time_factors, 0.05),
            "error_counts": self.error_counts,
        }
        self._write(self.events, summary)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        self.close()
        self.finished = True

    def close(self) -> None:
        self.events.close()
        self.commands.close()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _sim_span(stamps_ns: list[int]) -> float | None:
    if len(stamps_ns) < 2:
        return None
    return max(0.0, (stamps_ns[-1] - stamps_ns[0]) / 1e9)


def _sim_rate(stamps_ns: list[int]) -> float:
    span = _sim_span(stamps_ns)
    if span is None or span <= 0.0:
        return 0.0
    return (len(stamps_ns) - 1) / span


def _wall_rate(stamps_ns: list[int], duration_sec: float) -> float:
    if duration_sec <= 0.0:
        return 0.0
    return len(stamps_ns) / duration_sec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--duration-sec", type=float, default=sum(item[1] for item in SEGMENTS))
    parser.add_argument("--lidar-samples", type=int, default=30000)
    parser.add_argument("--lidar-downsample", type=int, default=1)
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=1.0,
        help="Scale commanded linear/angular motion; 0.5 gives a slower, gentler trajectory.",
    )
    parser.add_argument("--scene", default="public-candidate-not-official")
    parser.add_argument("--profile", default="controlled-motion")
    parser.add_argument("--ready-timeout-sec", type=float, default=60.0)
    parser.add_argument("--required-lidar-packets", type=int, default=3)
    parser.add_argument("--required-imu-packets", type=int, default=10)
    parser.add_argument("--max-wall-sec", type=float, default=300.0)
    args = parser.parse_args()
    if (
        args.duration_sec <= 0.0
        or args.motion_scale <= 0.0
        or args.ready_timeout_sec <= 0.0
        or args.max_wall_sec <= 0.0
    ):
        raise SystemExit("duration, motion scale, readiness timeout, and wall timeout must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "evidence_level": "gazebo_simulation",
        "profile": args.profile,
        "scene": args.scene,
        "duration_sec": args.duration_sec,
        "lidar_samples_requested": args.lidar_samples,
        "lidar_downsample": args.lidar_downsample,
        "motion_scale": args.motion_scale,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "ready_timeout_sec": args.ready_timeout_sec,
        "required_lidar_packets": args.required_lidar_packets,
        "required_imu_packets": args.required_imu_packets,
        "max_wall_sec": args.max_wall_sec,
        "sequence_duration_sec": SEQUENCE_DURATION_SEC,
        "segments": [{"name": name, "duration_sec": duration, "command": velocity} for name, duration, velocity in SEGMENTS],
        "pid": os.getpid(),
    }
    (args.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    rclpy.init()
    node = ControlledLioRecorder(
        args.run_dir,
        args.duration_sec,
        args.motion_scale,
        args.ready_timeout_sec,
        args.required_lidar_packets,
        args.required_imu_packets,
    )
    wall_deadline = time.monotonic() + args.max_wall_sec
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
            node.control_tick()
            if time.monotonic() >= wall_deadline and not node.finished:
                node._finish("wall_timeout", "simulation did not complete within the wall-clock budget")
    except KeyboardInterrupt:
        if not node.finished:
            node._finish("interrupted", "recorder interrupted")
    finally:
        if not node.events.closed:
            node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return node.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
