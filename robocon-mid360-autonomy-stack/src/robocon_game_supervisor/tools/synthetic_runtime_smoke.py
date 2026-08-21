#!/usr/bin/env python3
"""Finite ROS 2 smoke test for the synthetic competition action path."""

from __future__ import annotations

import json
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String


STEPS = (
    ("preflight", "WAIT_START"),
    ("start", "ACTIVE"),
    ("prepare_receive", "receiver_ready"),
    ("prepare_pass", "pass_armed"),
    ("execute_pass", "pass_executed"),
    ("collect_ball", "receipt_confirmed"),
    ("prepare_shot", "prepare_shot_complete"),
    ("fire_shot", "shot_executed"),
)


class SmokeClient(Node):
    def __init__(self) -> None:
        super().__init__("robocon_synthetic_smoke_client")
        self.publisher = self.create_publisher(String, "/robocon/operator_command", 10)
        self.create_subscription(String, "/robocon/game/state", self._state_callback, 10)
        self.create_subscription(String, "/robocon/game/action_decision", self._decision_callback, 10)
        self.create_subscription(
            Bool, "/mid360/preflight_ready", lambda message: self._bool_signal("preflight", message), 10
        )
        self.create_subscription(
            Bool, "/robocon/perception/target_valid", lambda message: self._bool_signal("target", message), 10
        )
        self.step_index = 0
        self.awaiting_step = False
        self.latest_state: dict[str, object] | None = None
        self.latest_decision: dict[str, object] | None = None
        self._signals = {"preflight": False, "target": False}
        self._signal_seen_at: dict[str, float | None] = {"preflight": None, "target": None}
        self.exit_code: int | None = None
        self.create_timer(0.05, self._advance)

    def _advance(self) -> None:
        if self.exit_code is not None or self.step_index >= len(STEPS):
            return
        if self.publisher.get_subscription_count() == 0:
            return
        command, expected = STEPS[self.step_index]
        if command == "preflight" and not self._signal_stable("preflight"):
            return
        if command == "fire_shot" and not self._signal_stable("target"):
            return
        if self.awaiting_step:
            if not self._step_completed(expected):
                return
            self.step_index += 1
            self.awaiting_step = False
            return
        self.publisher.publish(String(data=command))
        self.get_logger().info(f"synthetic smoke command: {command}")
        self.awaiting_step = True

    def _bool_signal(self, name: str, message: Bool) -> None:
        self._signals[name] = bool(message.data)
        if message.data and self._signal_seen_at[name] is None:
            self._signal_seen_at[name] = time.monotonic()
        if not message.data:
            self._signal_seen_at[name] = None

    def _signal_stable(self, name: str) -> bool:
        seen_at = self._signal_seen_at[name]
        return self._signals[name] and seen_at is not None and time.monotonic() - seen_at >= 0.20

    def _state_callback(self, message: String) -> None:
        try:
            self.latest_state = json.loads(message.data)
        except json.JSONDecodeError:
            return
        task_state = self.latest_state.get("task_state", {})
        if task_state.get("shooter") == "shot_executed":
            self.get_logger().info("synthetic competition path reached shot_executed")
            self.exit_code = 0

    def _decision_callback(self, message: String) -> None:
        try:
            self.latest_decision = json.loads(message.data)
        except json.JSONDecodeError:
            return

    def _step_completed(self, expected: str) -> bool:
        if self.latest_state is None:
            return False
        if expected in {"WAIT_START", "ACTIVE"}:
            return self.latest_state.get("state") == expected
        if expected == "prepare_shot_complete":
            completed = self.latest_state.get("completed_actions", {})
            result = completed.get("PrepareShot", {}) if isinstance(completed, dict) else {}
            return result.get("state") == "succeeded"
        task_state = self.latest_state.get("task_state", {})
        return task_state.get("shooter") == expected


def main() -> int:
    rclpy.init()
    node = SmokeClient()
    result = 2
    try:
        deadline = time.monotonic() + 8.0
        try:
            while rclpy.ok() and node.exit_code is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        except ExternalShutdownException:
            # ros2 launch may shut down the shared context while this finite
            # client is leaving; do not call rclpy.shutdown() a second time.
            pass
        if node.exit_code is None:
            node.get_logger().error(
                f"synthetic competition path timed out without a terminal state: {node.latest_state}"
            )
            result = 2
        else:
            result = node.exit_code
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())
