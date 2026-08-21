"""Validate camera/depth target observations before they reach shot control."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Bool, String


@dataclass(frozen=True)
class TargetObservation:
    confidence: float
    distance_m: float
    stable: bool
    observed_at_ns: int
    target_type: str = "unknown"
    evidence_level: str = "runtime_observation"
    evidence_source: str = "unknown"


def parse_observation(value: str | dict[str, object]) -> TargetObservation:
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("target observation must be a JSON object")
    distance = payload.get("distance_m")
    if distance is None and payload.get("distance_mm") is not None:
        distance = float(payload["distance_mm"]) / 1000.0
    observation = TargetObservation(
        confidence=float(payload["confidence"]),
        distance_m=float(distance),
        stable=bool(payload.get("stable", False)),
        observed_at_ns=int(payload["observed_at_ns"]),
        target_type=str(payload.get("target_type", "unknown")),
        evidence_level=str(payload.get("evidence_level", "runtime_observation")),
        evidence_source=str(payload.get("evidence_source", "unknown")),
    )
    if not math.isfinite(observation.confidence) or not math.isfinite(observation.distance_m):
        raise ValueError("confidence and distance must be finite")
    if observation.confidence < 0.0 or observation.confidence > 1.0:
        raise ValueError("confidence must be within [0, 1]")
    if observation.distance_m < 0.0:
        raise ValueError("distance_m must be non-negative")
    if observation.observed_at_ns <= 0:
        raise ValueError("observed_at_ns must be positive")
    return observation


def validate_observation(
    observation: TargetObservation,
    *,
    now_ns: int,
    min_confidence: float,
    min_distance_m: float,
    max_distance_m: float,
    max_age_sec: float,
    require_stable: bool,
) -> tuple[bool, str]:
    age_sec = (now_ns - observation.observed_at_ns) / 1_000_000_000.0
    if age_sec < 0.0 or age_sec > max_age_sec:
        return False, "target_observation_stale"
    if observation.confidence < min_confidence:
        return False, "target_confidence_below_threshold"
    if observation.distance_m < min_distance_m or observation.distance_m > max_distance_m:
        return False, "target_distance_out_of_range"
    if require_stable and not observation.stable:
        return False, "target_observation_unstable"
    return True, "target_valid"


class TargetGate(Node):
    def __init__(self) -> None:
        super().__init__("robocon_target_gate")
        self.declare_parameter("observation_topic", "/camera/target_observation")
        self.declare_parameter("valid_topic", "/robocon/perception/target_valid")
        self.declare_parameter("status_topic", "/robocon/perception/target_status")
        self.declare_parameter("diagnostic_topic", "/robocon/perception/diagnostics")
        self.declare_parameter("min_confidence", 0.70)
        self.declare_parameter("min_distance_m", 1.0)
        self.declare_parameter("max_distance_m", 10.0)
        self.declare_parameter("max_age_sec", 0.50)
        self.declare_parameter("require_stable", True)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_distance_m = float(self.get_parameter("min_distance_m").value)
        self.max_distance_m = float(self.get_parameter("max_distance_m").value)
        self.max_age_sec = float(self.get_parameter("max_age_sec").value)
        self.require_stable = bool(self.get_parameter("require_stable").value)
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.min_distance_m < 0.0 or self.max_distance_m <= self.min_distance_m:
            raise ValueError("distance range is invalid")
        if self.max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be positive")
        self._latest: TargetObservation | None = None
        self._last_reason = "no target observation received"
        self._valid_pub = self.create_publisher(Bool, str(self.get_parameter("valid_topic").value), 10)
        self._status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("observation_topic").value),
            self._observation_callback,
            10,
        )
        self.create_timer(0.05, self._publish_status)

    def _observation_callback(self, message: String) -> None:
        try:
            self._latest = parse_observation(message.data)
            _, self._last_reason = validate_observation(
                self._latest,
                now_ns=time.time_ns(),
                min_confidence=self.min_confidence,
                min_distance_m=self.min_distance_m,
                max_distance_m=self.max_distance_m,
                max_age_sec=self.max_age_sec,
                require_stable=self.require_stable,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._latest = None
            self._last_reason = f"malformed_target_observation: {error}"

    def _publish_status(self) -> None:
        now_ns = time.time_ns()
        valid = False
        age_sec = math.inf
        if self._latest is not None:
            valid, self._last_reason = validate_observation(
                self._latest,
                now_ns=now_ns,
                min_confidence=self.min_confidence,
                min_distance_m=self.min_distance_m,
                max_distance_m=self.max_distance_m,
                max_age_sec=self.max_age_sec,
                require_stable=self.require_stable,
            )
            age_sec = (now_ns - self._latest.observed_at_ns) / 1_000_000_000.0
        self._valid_pub.publish(Bool(data=valid))
        payload = {
            "version": 1,
            "valid": valid,
            "reason": self._last_reason,
            "age_sec": age_sec if math.isfinite(age_sec) else None,
            "confidence": self._latest.confidence if self._latest else None,
            "distance_m": self._latest.distance_m if self._latest else None,
            "stable": self._latest.stable if self._latest else False,
            "target_type": self._latest.target_type if self._latest else None,
            "evidence_level": self._latest.evidence_level if self._latest else "unknown",
            "evidence_source": self._latest.evidence_source if self._latest else "unknown",
        }
        self._status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        status = DiagnosticStatus()
        status.name = f"{self.get_name()}/target"
        status.level = DiagnosticStatus.OK if valid else DiagnosticStatus.WARN
        status.message = self._last_reason
        status.values = [
            KeyValue(key="target_valid", value=str(valid).lower()),
            KeyValue(key="reason", value=self._last_reason),
            KeyValue(key="target_age_sec", value=str(age_sec)),
            KeyValue(key="evidence_level", value=payload["evidence_level"]),
        ]
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        diagnostic.status = [status]
        self._diagnostic_pub.publish(diagnostic)


def main() -> None:
    rclpy.init()
    node = TargetGate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
