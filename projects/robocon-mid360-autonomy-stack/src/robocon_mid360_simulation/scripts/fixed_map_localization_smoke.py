#!/usr/bin/env python3
"""Run a bounded Gazebo fixed-map localization check and preserve its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String


def _pose_matrix(pose: Any) -> list[list[float]]:
    """Return a homogeneous transform for a ROS pose without numpy."""
    x, y, z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
    qx, qy, qz, qw = (float(pose.orientation.x), float(pose.orientation.y),
                      float(pose.orientation.z), float(pose.orientation.w))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        qx = qy = qz = 0.0
        qw = 1.0
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw), x],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw), y],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy), z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)] for row in range(4)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [[matrix[row][column] for column in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    result = [[rotation[column][row] for column in range(3)] + [0.0] for row in range(3)]
    for row in range(3):
        result[row][3] = -sum(result[row][column] * translation[column] for column in range(3))
    result.append([0.0, 0.0, 0.0, 1.0])
    return result


def _yaw(matrix: list[list[float]]) -> float:
    return math.atan2(matrix[1][0], matrix[0][0])


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def pose_error(
    truth: Odometry,
    odom: Odometry,
    map_to_odom: Odometry,
) -> tuple[float, float]:
    """Compare corrected map pose with Gazebo truth in the aligned origin frame."""
    estimated = _matmul(_pose_matrix(map_to_odom.pose.pose), _pose_matrix(odom.pose.pose))
    actual = _pose_matrix(truth.pose.pose)
    translation = math.sqrt(sum((estimated[index][3] - actual[index][3]) ** 2 for index in range(3)))
    yaw_error = abs(_wrap_angle(_yaw(estimated) - _yaw(actual)))
    return translation, yaw_error


def relative_pose_error(
    baseline_estimated: list[list[float]],
    baseline_actual: list[list[float]],
    estimated: list[list[float]],
    actual: list[list[float]],
) -> tuple[float, float]:
    """Measure drift after removing the unknown initial map/world offset."""
    estimated_delta = _matmul(_inverse(baseline_estimated), estimated)
    actual_delta = _matmul(_inverse(baseline_actual), actual)
    discrepancy = _matmul(_inverse(actual_delta), estimated_delta)
    translation = math.sqrt(sum(discrepancy[index][3] ** 2 for index in range(3)))
    return translation, abs(_wrap_angle(_yaw(discrepancy)))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class FixedMapSmoke(Node):
    def __init__(
        self,
        run_dir: Path,
        map_file: Path,
        duration_sec: float,
        readiness_timeout_sec: float,
        truth_topic: str,
    ) -> None:
        super().__init__("fixed_map_localization_smoke")
        self.run_dir = run_dir
        self.map_file = str(map_file)
        self.duration_sec = duration_sec
        self.readiness_timeout_sec = readiness_timeout_sec
        self.truth_topic = truth_topic
        self.created_at = time.monotonic()
        self.started_at: float | None = None
        self.initialpose_sent_at: float | None = None
        self.finished = False
        self.accepted = False
        self.latest_odom: Odometry | None = None
        self.latest_truth: Odometry | None = None
        self.latest_correction: Odometry | None = None
        self.truth_translation_errors: list[float] = []
        self.truth_yaw_errors: list[float] = []
        self.truth_drift_translation_errors: list[float] = []
        self.truth_drift_yaw_errors: list[float] = []
        self.baseline_estimated: list[list[float]] | None = None
        self.baseline_actual: list[list[float]] | None = None
        self.last_truth_sample_wall = 0.0
        self.latest_status: dict[str, Any] = {}
        self.status_history: list[dict[str, Any]] = []
        self.correction_count = 0
        self.map_locked_values: list[bool] = []
        self.last_fitness: float | None = None
        self.last_scan_points = 0
        self.events = (run_dir / "localization_telemetry.jsonl").open("w", encoding="utf-8")
        self.map_sha256 = hashlib.sha256(map_file.read_bytes()).hexdigest()

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_chassis", 10)
        self.create_subscription(Odometry, "/mid360/local_odometry", self._odom_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/mid360/map_to_odom_correction", self._correction_callback, 10)
        self.create_subscription(Odometry, truth_topic, self._truth_callback, qos_profile_sensor_data)
        self.create_subscription(String, "/mid360/map_localization_diagnostics", self._diagnostic_callback, 10)
        self.create_subscription(String, "/mid360/pose_status", self._pose_status_callback, 10)
        self.create_subscription(Bool, "/mid360/map_locked", self._locked_callback, 10)
        self.create_timer(0.05, self._tick)

    def _write(self, payload: dict[str, Any]) -> None:
        payload["wall_time_unix"] = time.time()
        self.events.write(json.dumps(payload, sort_keys=True) + "\n")
        self.events.flush()

    def _odom_callback(self, message: Odometry) -> None:
        self.latest_odom = message
        self._record_truth_error()

    def _truth_callback(self, message: Odometry) -> None:
        self.latest_truth = message
        self._record_truth_error()

    def _correction_callback(self, message: Odometry) -> None:
        self.latest_correction = message
        self.correction_count += 1
        self._record_truth_error()
        self._write({
            "kind": "map_to_odom_correction",
            "sequence": self.correction_count,
            "frame_id": message.header.frame_id,
            "child_frame_id": message.child_frame_id,
            "x": message.pose.pose.position.x,
            "y": message.pose.pose.position.y,
            "z": message.pose.pose.position.z,
        })

    def _record_truth_error(self) -> None:
        if self.latest_truth is None or self.latest_odom is None or self.latest_correction is None:
            return
        now = time.monotonic()
        if now - self.last_truth_sample_wall < 0.10:
            return
        estimated = _matmul(
            _pose_matrix(self.latest_correction.pose.pose),
            _pose_matrix(self.latest_odom.pose.pose),
        )
        actual = _pose_matrix(self.latest_truth.pose.pose)
        translation, yaw = pose_error(self.latest_truth, self.latest_odom, self.latest_correction)
        if self.baseline_estimated is None or self.baseline_actual is None:
            self.baseline_estimated = estimated
            self.baseline_actual = actual
        drift_translation, drift_yaw = relative_pose_error(
            self.baseline_estimated, self.baseline_actual, estimated, actual
        )
        self.truth_translation_errors.append(translation)
        self.truth_yaw_errors.append(yaw)
        self.truth_drift_translation_errors.append(drift_translation)
        self.truth_drift_yaw_errors.append(drift_yaw)
        self.last_truth_sample_wall = now
        self._write({
            "kind": "ground_truth_error",
            "translation_error_m": translation,
            "yaw_error_deg": math.degrees(yaw),
        })

    def _diagnostic_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self._write({"kind": "map_diagnostic", "malformed": True, "raw": message.data})
            return
        self.latest_status = payload
        self.status_history.append(payload)
        fitness = payload.get("last_fitness")
        if isinstance(fitness, (int, float)) and math.isfinite(float(fitness)):
            self.last_fitness = float(fitness)
        self.last_scan_points = max(self.last_scan_points, int(payload.get("last_scan_points", 0)))
        self._write({"kind": "map_diagnostic", "payload": payload})

    def _pose_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self._write({"kind": "pose_status", "payload": payload})

    def _locked_callback(self, message: Bool) -> None:
        self.map_locked_values.append(bool(message.data))
        self._write({"kind": "map_locked", "value": bool(message.data)})

    def _send_initialpose(self) -> None:
        assert self.latest_odom is not None
        message = PoseWithCovarianceStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        # The frozen map was built from this same Gazebo origin. This is a
        # controlled initial-pose test, not a claim about a physical pose.
        message.pose.pose.position.x = 0.0
        message.pose.pose.position.y = 0.0
        message.pose.pose.position.z = 0.0
        message.pose.pose.orientation.w = 1.0
        message.pose.covariance[0] = 0.04
        message.pose.covariance[7] = 0.04
        message.pose.covariance[35] = 0.08
        self.initialpose_pub.publish(message)
        self.initialpose_sent_at = time.monotonic()
        self.started_at = self.initialpose_sent_at
        self._write({"kind": "initialpose", "frame_id": "map", "x": 0.0, "y": 0.0, "z": 0.0})

    def _finish(self, outcome: str, failure_reason: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        self.cmd_pub.publish(Twist())
        summary = {
            "evidence_level": "gazebo_simulation",
            "outcome": outcome,
            "failure_reason": failure_reason or None,
            "map_file": self.map_file,
            "map_sha256": self.map_sha256,
            "initialpose_sent": self.initialpose_sent_at is not None,
            "map_status": self.latest_status.get("status", "NO_MAP_DIAGNOSTIC"),
            "correction_count": self.correction_count,
            "map_locked_seen": any(self.map_locked_values),
            "last_fitness": self.last_fitness,
            "last_scan_points": self.last_scan_points,
            "status_sequence": [item.get("status", "unknown") for item in self.status_history],
            "ground_truth_topic": self.truth_topic,
            "ground_truth_samples": len(self.truth_translation_errors),
            "ground_truth_available": bool(self.truth_translation_errors),
            "translation_error_p50_m": _percentile(self.truth_translation_errors, 0.50),
            "translation_error_p95_m": _percentile(self.truth_translation_errors, 0.95),
            "translation_error_max_m": max(self.truth_translation_errors) if self.truth_translation_errors else None,
            "yaw_error_p50_deg": math.degrees(_percentile(self.truth_yaw_errors, 0.50)) if self.truth_yaw_errors else None,
            "yaw_error_p95_deg": math.degrees(_percentile(self.truth_yaw_errors, 0.95)) if self.truth_yaw_errors else None,
            "yaw_error_max_deg": math.degrees(max(self.truth_yaw_errors)) if self.truth_yaw_errors else None,
            "drift_translation_error_p50_m": _percentile(self.truth_drift_translation_errors, 0.50),
            "drift_translation_error_p95_m": _percentile(self.truth_drift_translation_errors, 0.95),
            "drift_translation_error_max_m": max(self.truth_drift_translation_errors) if self.truth_drift_translation_errors else None,
            "drift_yaw_error_p50_deg": math.degrees(_percentile(self.truth_drift_yaw_errors, 0.50)) if self.truth_drift_yaw_errors else None,
            "drift_yaw_error_p95_deg": math.degrees(_percentile(self.truth_drift_yaw_errors, 0.95)) if self.truth_drift_yaw_errors else None,
            "drift_yaw_error_max_deg": math.degrees(max(self.truth_drift_yaw_errors)) if self.truth_drift_yaw_errors else None,
        }
        (self.run_dir / "localization_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write({"kind": "summary", **summary})

    def _tick(self) -> None:
        if self.finished:
            return
        if self.initialpose_sent_at is None:
            if time.monotonic() - self.created_at >= self.readiness_timeout_sec:
                self._finish("failed", "Timed out waiting for fresh odometry and localizer readiness")
                return
            if self.latest_odom is None:
                return
            if self.latest_status.get("status") == "WAITING_FOR_INITIALPOSE":
                self._send_initialpose()
            return
        assert self.started_at is not None
        elapsed = time.monotonic() - self.started_at
        command = Twist()
        if elapsed < self.duration_sec:
            command.linear.x = 0.12
            self.cmd_pub.publish(command)
        if any(self.map_locked_values) and self.correction_count > 0:
            self.accepted = True
        if elapsed >= self.duration_sec:
            if self.accepted:
                self._finish("passed")
            else:
                self._finish("failed", "No accepted map-to-odom correction and map lock before deadline")

    def start(self) -> None:
        self._write({"kind": "started", "map_sha256": self.map_sha256, "duration_sec": self.duration_sec})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--map-file", required=True, type=Path)
    parser.add_argument("--duration-sec", type=float, default=25.0)
    parser.add_argument("--readiness-timeout-sec", type=float, default=180.0)
    parser.add_argument("--truth-topic", default="/simulation/ground_truth/odom")
    args = parser.parse_args()
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.readiness_timeout_sec <= 0:
        parser.error("--readiness-timeout-sec must be positive")
    if not args.map_file.is_file():
        parser.error(f"map file does not exist: {args.map_file}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FixedMapSmoke(
        args.run_dir,
        args.map_file,
        args.duration_sec,
        args.readiness_timeout_sec,
        args.truth_topic,
    )
    node.start()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        node._finish("interrupted", "Interrupted by operator")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    summary = json.loads((args.run_dir / "localization_summary.json").read_text(encoding="utf-8"))
    return 0 if summary["outcome"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
