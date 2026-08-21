"""Publish explicit synthetic readiness and safety inputs for local integration tests."""

from __future__ import annotations

import json
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Bool, String


class SyntheticInputs(Node):
    """Provide all non-localization booleans as synthetic evidence only."""

    def __init__(self) -> None:
        super().__init__("robocon_synthetic_inputs")
        topic_names = {
            "preflight_ready": "/mid360/preflight_ready",
            "input_valid": "/mid360/input_valid",
            "pose_valid": "/mid360/pose_valid",
            "map_locked": "/mid360/map_locked",
            "teammate_safe": "/robocon/team/teammate_safe",
        }
        self._signal_publishers = {
            name: self.create_publisher(Bool, topic, 10)
            for name, topic in topic_names.items()
        }
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, "/mid360/pose_diagnostics", 10
        )
        self._team_pub = self.create_publisher(String, "/robocon/team/message", 10)
        self._target_truth_pub = self.create_publisher(
            String, "/gazebo/synthetic_target_truth", 10
        )
        self.create_timer(0.05, self._publish)

    def _publish(self) -> None:
        for publisher in self._signal_publishers.values():
            publisher.publish(Bool(data=True))
        self._target_truth_pub.publish(String(data=json.dumps({
            "confidence": 0.90,
            "distance_m": 3.0,
            "stable": True,
            "observed_at_ns": time.time_ns(),
            "target_type": "hoop",
            "evidence_level": "synthetic",
            "source": "synthetic_competition_plant",
        }, separators=(",", ":"))))
        status = DiagnosticStatus()
        status.name = "robocon_synthetic_inputs/pose"
        status.level = DiagnosticStatus.OK
        status.message = "synthetic evidence; not hardware data"
        status.values = [KeyValue(key="pose_age_sec", value="0.02")]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self._diagnostic_pub.publish(message)
        self._team_pub.publish(String(data=json.dumps({
            "protocol_version": 1,
            "message_type": "heartbeat",
            "task_id": "synthetic",
            "sender_id": "ball_handler",
            "sequence": int(time.time() * 20),
            "created_at_ns": time.time_ns(),
            "expires_at_ns": time.time_ns() + 300_000_000,
            "payload": {"safe": True, "evidence_level": "synthetic"},
        }, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = SyntheticInputs()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
