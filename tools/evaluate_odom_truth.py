#!/usr/bin/env python3
"""Compare an estimator odometry stream with Gazebo truth from one replay.

The comparison is trajectory-aligned at the first timestamped pair. This
removes the arbitrary estimator origin and fixed sensor mounting offset while
retaining drift and motion error. It is a simulation/bag diagnostic, not a
physical calibration result.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import signal
from dataclasses import dataclass
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data


@dataclass(frozen=True)
class Pose:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float


def _stamp_ns(message: Odometry) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _yaw(message: Odometry) -> float:
    q = message.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _pose(message: Odometry) -> Pose:
    p = message.pose.pose.position
    return Pose(_stamp_ns(message), float(p.x), float(p.y), float(p.z), _yaw(message))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _aligned_error(truth: Pose, estimate: Pose, first_truth: Pose, first_estimate: Pose) -> tuple[float, float]:
    yaw_offset = first_estimate.yaw - first_truth.yaw
    c, s = math.cos(yaw_offset), math.sin(yaw_offset)
    # Align truth's first point to the estimator's first point in SE(2).
    tx = first_estimate.x - (c * first_truth.x - s * first_truth.y)
    ty = first_estimate.y - (s * first_truth.x + c * first_truth.y)
    aligned_x = c * truth.x - s * truth.y + tx
    aligned_y = s * truth.x + c * truth.y + ty
    translation = math.sqrt(
        (estimate.x - aligned_x) ** 2
        + (estimate.y - aligned_y) ** 2
        + (estimate.z - truth.z) ** 2
    )
    yaw_error = abs(_wrap(estimate.yaw - (truth.yaw + yaw_offset)))
    return translation, yaw_error


class TruthEvaluator:
    def __init__(self, output: Path, truth_topic: str, estimate_topic: str, max_delta_sec: float) -> None:
        self.output = output
        self.truth_topic = truth_topic
        self.estimate_topic = estimate_topic
        self.max_delta_ns = int(max_delta_sec * 1_000_000_000)
        self.truth: list[Pose] = []
        self.estimates: list[Pose] = []
        self.node = rclpy.create_node("odom_truth_evaluator")
        self.node.create_subscription(Odometry, truth_topic, self._truth_callback, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, estimate_topic, self._estimate_callback, qos_profile_sensor_data)
        # Persist progress during replay; ROS shutdown can be delayed by a
        # bag player or an estimator that does not handle SIGINT promptly.
        self.node.create_timer(1.0, self.write_summary)
        self.write_summary()

    def _truth_callback(self, message: Odometry) -> None:
        self.truth.append(_pose(message))

    def _estimate_callback(self, message: Odometry) -> None:
        self.estimates.append(_pose(message))

    def _match(self) -> list[tuple[Pose, Pose]]:
        if not self.truth or not self.estimates:
            return []
        truth_stamps = [item.stamp_ns for item in self.truth]
        pairs: list[tuple[Pose, Pose]] = []
        for estimate in self.estimates:
            index = bisect.bisect_left(truth_stamps, estimate.stamp_ns)
            candidates = []
            if index < len(self.truth):
                candidates.append(self.truth[index])
            if index > 0:
                candidates.append(self.truth[index - 1])
            if not candidates:
                continue
            truth = min(candidates, key=lambda item: abs(item.stamp_ns - estimate.stamp_ns))
            if abs(truth.stamp_ns - estimate.stamp_ns) <= self.max_delta_ns:
                pairs.append((truth, estimate))
        return pairs

    def summary(self) -> dict[str, object]:
        pairs = self._match()
        errors: list[float] = []
        yaw_errors: list[float] = []
        if pairs:
            first_truth, first_estimate = pairs[0]
            for truth, estimate in pairs:
                translation, yaw = _aligned_error(truth, estimate, first_truth, first_estimate)
                errors.append(translation)
                yaw_errors.append(yaw)
        return {
            "evidence_level": "gazebo_simulation/bag_replay",
            "diagnostic_only": True,
            "alignment_mode": "first_pose_se2",
            "truth_topic": self.truth_topic,
            "estimate_topic": self.estimate_topic,
            "truth_samples": len(self.truth),
            "estimate_samples": len(self.estimates),
            "matched_samples": len(pairs),
            "max_timestamp_delta_sec": self.max_delta_ns / 1_000_000_000.0,
            "translation_error_p50_m": _percentile(errors, 0.50),
            "translation_error_p95_m": _percentile(errors, 0.95),
            "translation_error_max_m": max(errors) if errors else None,
            "yaw_error_p50_deg": math.degrees(_percentile(yaw_errors, 0.50)) if yaw_errors else None,
            "yaw_error_p95_deg": math.degrees(_percentile(yaw_errors, 0.95)) if yaw_errors else None,
            "yaw_error_max_deg": math.degrees(max(yaw_errors)) if yaw_errors else None,
            "status": "passed" if len(pairs) >= 3 else "insufficient_matches",
        }

    def write_summary(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-topic", default="/simulation/ground_truth/odom")
    parser.add_argument("--estimate-topic", default="/Odometry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-delta-sec", type=float, default=0.10)
    args = parser.parse_args()
    if args.max_delta_sec <= 0:
        parser.error("--max-delta-sec must be positive")

    rclpy.init()
    evaluator = TruthEvaluator(args.output, args.truth_topic, args.estimate_topic, args.max_delta_sec)

    def stop_handler(_signum: int, _frame: object) -> None:
        evaluator.write_summary()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        rclpy.spin(evaluator.node)
    finally:
        evaluator.write_summary()
        evaluator.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
