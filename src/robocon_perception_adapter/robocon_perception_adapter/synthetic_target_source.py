"""Explicit Gazebo/synthetic target-truth adapter for local integration only."""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .target_gate import TargetObservation, parse_observation


def normalize_synthetic_truth(value: str | dict[str, object]) -> tuple[TargetObservation, str]:
    """Validate an explicitly synthetic truth message before forwarding it.

    This adapter must never be confused with a camera model or a detector.  It
    only permits a declared synthetic source and preserves that identity in the
    resulting observation.
    """

    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("synthetic target truth must be a JSON object")
    source = str(payload.get("source", "")).strip()
    if not source:
        raise ValueError("synthetic target truth requires a source")
    if payload.get("evidence_level") != "synthetic":
        raise ValueError("synthetic target truth must declare evidence_level=synthetic")
    normalized = dict(payload)
    normalized["evidence_source"] = source
    return parse_observation(normalized), source


class SyntheticTargetSource(Node):
    """Convert a declared synthetic target event into the shared observation contract."""

    def __init__(self) -> None:
        super().__init__("robocon_synthetic_target_source")
        self.declare_parameter("truth_topic", "/gazebo/synthetic_target_truth")
        self.declare_parameter("observation_topic", "/camera/target_observation")
        self._publisher = self.create_publisher(
            String, str(self.get_parameter("observation_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("truth_topic").value),
            self._truth_callback,
            10,
        )

    def _truth_callback(self, message: String) -> None:
        try:
            observation, source = normalize_synthetic_truth(message.data)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Rejected synthetic target truth: %s", error)
            return
        payload = {
            "confidence": observation.confidence,
            "distance_m": observation.distance_m,
            "stable": observation.stable,
            "observed_at_ns": observation.observed_at_ns,
            "target_type": observation.target_type,
            "evidence_level": "synthetic",
            "evidence_source": source,
            "forwarded_at_ns": time.time_ns(),
        }
        self._publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = SyntheticTargetSource()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
