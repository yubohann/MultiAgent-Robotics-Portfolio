#!/usr/bin/env python3
"""Exercise fixed-map tracking loss and competition interlocks over ROS 2.

This is a short contract test. Its odometry, pose validity, and map correction
messages are explicitly synthetic; it never represents a physical localization
or mechanism result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


class RecoveryInterlockSmoke(Node):
    def __init__(self, run_dir: Path, auto_recovery: bool) -> None:
        super().__init__("localization_recovery_interlock_smoke")
        self.run_dir = run_dir
        self.auto_recovery = auto_recovery
        self.started_at = time.monotonic()
        self.phase_started_at = self.started_at
        self.phase = "WAITING_FOR_TRACKING"
        self.finished = False
        self.failure_reason = ""
        self.map_states: list[str] = []
        self.game_states: list[str] = []
        self.decisions: list[dict[str, Any]] = []
        self.action_requests: list[dict[str, Any]] = []
        self.receiver_ready_seen = False
        self.initial_correction_sent = False
        self.relocalization_requested = False
        self.recovery_correction_sent = False
        self.command_sent: set[str] = set()
        self.blocked_request_start = 0
        self.blocked_request_end = 0
        self.signal_ready = False
        self.events = (run_dir / "interlock_telemetry.jsonl").open("w", encoding="utf-8")

        self.odom_pub = self.create_publisher(Odometry, "/mid360/local_odometry", 10)
        self.pose_valid_pub = self.create_publisher(Bool, "/mid360/pose_valid", 10)
        self.relocalization_pub = self.create_publisher(Bool, "/mid360/relocalization_request", 10)
        self.correction_pub = self.create_publisher(Odometry, "/mid360/map_to_odom_correction", 10)
        self.command_pub = self.create_publisher(String, "/robocon/operator_command", 10)
        self.feedback_pub = self.create_publisher(String, "/robocon/action/feedback", 10)
        self.signal_publishers = {
            topic: self.create_publisher(Bool, topic, 10)
            for topic in (
                "/mid360/preflight_ready",
                "/robocon/perception/target_valid",
                "/robocon/mechanism/ball_present",
                "/robocon/mechanism/healthy",
                "/robocon/team/teammate_safe",
            )
        }
        self.create_subscription(String, "/mid360/pose_status", self._pose_status_callback, 10)
        self.create_subscription(String, "/robocon/game/state", self._game_state_callback, 10)
        self.create_subscription(String, "/robocon/game/action_decision", self._decision_callback, 10)
        self.create_subscription(String, "/robocon/action/request", self._action_request_callback, 10)
        self.create_timer(0.05, self._tick)

    def _write(self, payload: dict[str, Any]) -> None:
        payload["wall_time_unix"] = time.time()
        self.events.write(json.dumps(payload, sort_keys=True) + "\n")
        self.events.flush()

    def _pose_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        state = str(payload.get("tracking_state", "unknown"))
        if not self.map_states or self.map_states[-1] != state:
            self.map_states.append(state)
            self._write({"kind": "map_state", "payload": payload})

    def _game_state_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        state = str(payload.get("state", "unknown"))
        task_state = payload.get("task_state", {})
        if isinstance(task_state, dict) and task_state.get("shooter") == "receiver_ready":
            self.receiver_ready_seen = True
        if not self.game_states or self.game_states[-1] != state:
            self.game_states.append(state)
            self._write({"kind": "game_state", "payload": payload})

    def _decision_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.decisions.append(payload)
        self._write({"kind": "decision", "payload": payload})

    def _action_request_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.action_requests.append(payload)
        self._write({"kind": "action_request", "payload": payload})
        if payload.get("action") == "PrepareReceive":
            self.feedback_pub.publish(String(data=json.dumps({
                "protocol_version": 1,
                "action_id": payload["action_id"],
                "task_id": payload["task_id"],
                "action": "PrepareReceive",
                "sender_id": "synthetic_mechanism",
                "state": "succeeded",
                "created_at_ns": time.time_ns(),
                "reason": "synthetic receiver-ready feedback",
                "evidence": {"receiver_ready": True, "evidence_level": "synthetic"},
            }, separators=(",", ":"))))

    def _publish_odom(self) -> None:
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        message.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(message)

    def _publish_correction(self) -> None:
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.child_frame_id = "odom"
        message.pose.pose.orientation.w = 1.0
        self.correction_pub.publish(message)
        self._write({"kind": "synthetic_map_to_odom_correction"})

    def _publish_command(self, command: str) -> None:
        self.command_pub.publish(String(data=command))
        self.command_sent.add(command)
        self._write({"kind": "operator_command", "command": command})

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_at = time.monotonic()
        self._write({"kind": "phase", "phase": phase})

    def _elapsed(self) -> float:
        return time.monotonic() - self.phase_started_at

    def _has_map_state(self, state: str) -> bool:
        return state in self.map_states

    def _latest_map_state(self) -> str:
        return self.map_states[-1] if self.map_states else "unknown"

    def _latest_game_state(self) -> str:
        return self.game_states[-1] if self.game_states else "unknown"

    def _decision_for(self, action: str) -> dict[str, Any] | None:
        for decision in reversed(self.decisions):
            if decision.get("action") == action:
                return decision
        return None

    def _finish(self, outcome: str, reason: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        blocked_actions = {"navigate_to_pose", "prepare_pass", "fire_shot"}
        blocked_decisions = {
            action: self._decision_for(action) for action in sorted(blocked_actions)
        }
        blocked_requests = [
            request for request in self.action_requests[
                self.blocked_request_start:self.blocked_request_end
            ]
            if request.get("action") in {"NavigateToPose", "PreparePass", "FireShot"}
        ]
        resumed = self._decision_for("navigate_to_pose_after_relocalization")
        summary = {
            "evidence_level": "synthetic",
            "outcome": outcome,
            "failure_reason": reason or None,
            "auto_recovery_on_signal_loss": self.auto_recovery,
            "map_state_sequence": self.map_states,
            "game_state_sequence": self.game_states,
            "blocked_decisions": blocked_decisions,
            "blocked_action_request_count": len(blocked_requests),
            "resumed_navigation_decision": resumed,
            "action_request_count": len(self.action_requests),
        }
        (self.run_dir / "interlock_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write({"kind": "summary", **summary})

    def _all_blocked(self) -> bool:
        for action in ("navigate_to_pose", "prepare_pass", "fire_shot"):
            decision = self._decision_for(action)
            if decision is None or bool(decision.get("accepted", True)):
                return False
            reason = str(decision.get("reason", ""))
            if not any(
                marker in reason
                for marker in (
                    "map_locked is false",
                    "pose_valid is false",
                    "pose is stale",
                    "supervisor is not ACTIVE",
                )
            ):
                return False
        return True

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() - self.started_at > 70.0:
            self._finish("failed", f"timed out in phase {self.phase}")
            return

        self._publish_odom()
        pose_is_valid = self.phase not in {
            "INJECT_LOST",
            "BLOCK_ACTIONS",
            "REQUEST_RELOCALIZATION",
            "REACQUIRE_TRACKING",
        }
        self.pose_valid_pub.publish(Bool(data=pose_is_valid))
        for publisher in self.signal_publishers.values():
            publisher.publish(Bool(data=True))

        if self.phase == "WAITING_FOR_TRACKING":
            self.signal_ready = (
                self.correction_pub.get_subscription_count() > 0
                and self.odom_pub.get_subscription_count() > 0
                and self.pose_valid_pub.get_subscription_count() > 0
            )
            if not self.initial_correction_sent and self.signal_ready and self._elapsed() >= 1.0:
                self._publish_correction()
                self.initial_correction_sent = True
            if self._has_map_state("TRACKING") and self._elapsed() >= 1.5:
                self._set_phase("ACTIVATE_PREFLIGHT")
            return

        if self.phase == "ACTIVATE_PREFLIGHT":
            if "preflight" not in self.command_sent and self.command_pub.get_subscription_count() > 0:
                self._publish_command("preflight")
            if self._latest_game_state() == "WAIT_START":
                self._set_phase("ACTIVATE_START")
            return

        if self.phase == "ACTIVATE_START":
            if "start" not in self.command_sent and self.command_pub.get_subscription_count() > 0:
                self._publish_command("start")
            if self._latest_game_state() == "ACTIVE":
                self._set_phase("PREPARE_RECEIVER")
            return

        if self.phase == "PREPARE_RECEIVER":
            if "prepare_receive" not in self.command_sent and self.command_pub.get_subscription_count() > 0:
                self._publish_command("prepare_receive")
            if self.receiver_ready_seen:
                self._set_phase("INJECT_LOST")
            return

        if self.phase == "INJECT_LOST":
            if self._has_map_state("LOST"):
                self.blocked_request_start = len(self.action_requests)
                self._set_phase("BLOCK_ACTIONS")
            return

        if self.phase == "BLOCK_ACTIONS":
            for command in ("navigate_to_pose", "prepare_pass", "fire_shot"):
                if command not in self.command_sent and self.command_pub.get_subscription_count() > 0:
                    self._publish_command(command)
                    return
            if self._all_blocked():
                self.blocked_request_end = len(self.action_requests)
                self._set_phase("REQUEST_RELOCALIZATION")
            return

        if self.phase == "REQUEST_RELOCALIZATION":
            if not self.relocalization_requested and self.relocalization_pub.get_subscription_count() > 0:
                self.relocalization_pub.publish(Bool(data=True))
                self.relocalization_requested = True
                self._write({"kind": "relocalization_request", "value": True})
            if self._latest_map_state() == "RELOCALIZING":
                self._set_phase("REACQUIRE_TRACKING")
            return

        if self.phase == "REACQUIRE_TRACKING":
            if (not self.recovery_correction_sent and self.signal_ready
                    and self._elapsed() >= 0.3):
                self._publish_correction()
                self.recovery_correction_sent = True
            if self.recovery_correction_sent and self._elapsed() >= 0.8:
                self.pose_valid_pub.publish(Bool(data=True))
            if self.recovery_correction_sent and self._elapsed() >= 1.0 and self._latest_map_state() == "TRACKING":
                if self.auto_recovery:
                    self._set_phase("RESTART_AFTER_RECOVERY")
                else:
                    self._set_phase("VERIFY_RESUMED_ACTION")
            return

        if self.phase == "RESTART_AFTER_RECOVERY":
            if self._latest_game_state() == "RECOVERY":
                if "preflight_after_relocalization" not in self.command_sent:
                    self.command_sent.add("preflight_after_relocalization")
                    self._write({"kind": "recovery_preflight_marker"})
                    self.command_pub.publish(String(data="preflight"))
                    return
                if self._latest_game_state() == "WAIT_START":
                    self.command_pub.publish(String(data="start"))
                    self._set_phase("WAIT_RESTART_ACTIVE")
                    return
            elif self._latest_game_state() == "WAIT_START":
                self.command_pub.publish(String(data="start"))
                self._set_phase("WAIT_RESTART_ACTIVE")
            return

        if self.phase == "WAIT_RESTART_ACTIVE":
            if self._latest_game_state() == "ACTIVE":
                self._set_phase("VERIFY_RESUMED_ACTION")
            return

        if self.phase == "VERIFY_RESUMED_ACTION":
            command = "navigate_to_pose_after_relocalization"
            if command not in self.command_sent and self.command_pub.get_subscription_count() > 0:
                # This alias is recorded distinctly but dispatched as the
                # normal navigation action through the ROS adapter contract.
                self.command_pub.publish(String(data="navigate_to_pose"))
                self.command_sent.add(command)
                self._write({"kind": "operator_command", "command": "navigate_to_pose"})
                return
            resumed = self._decision_for("NavigateToPose")
            if resumed is not None and bool(resumed.get("accepted", False)):
                # Preserve the resume decision under a unique key in the
                # evidence file, because its ROS action is NavigateToPose.
                self.decisions.append({**resumed, "action": command})
                self._finish("passed")
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--auto-recovery", choices=("true", "false"), default="false")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = RecoveryInterlockSmoke(args.run_dir, auto_recovery=args.auto_recovery == "true")
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._finish("interrupted", "interrupted by operator")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    summary = json.loads((args.run_dir / "interlock_summary.json").read_text(encoding="utf-8"))
    return 0 if summary["outcome"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
