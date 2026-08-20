#!/usr/bin/env python3
"""ROS-level fault checks for teammate transport, target gating, and actions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


class CompetitionFaultSmoke(Node):
    def __init__(self, run_dir: Path, mode: str) -> None:
        super().__init__(f"competition_fault_runtime_{mode}")
        self.run_dir = run_dir
        self.mode = mode
        self.started_at = time.monotonic()
        self.phase = "WAITING_FOR_NODES"
        self.phase_started_at = self.started_at
        self.finished = False
        self.map_locked = True
        self.decisions: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []
        self.action_requests: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []
        self.target_statuses: list[dict[str, Any]] = []
        self.command_sent: set[str] = set()
        self.heartbeat_sequence = 0
        self.heartbeat_enabled = mode == "protocol"
        self.target_valid_commanded = mode == "protocol"
        self.failure_reason = ""
        self.events = (run_dir / "fault_telemetry.jsonl").open("w", encoding="utf-8")

        self.command_pub = self.create_publisher(String, "/robocon/operator_command", 10)
        self.team_pub = self.create_publisher(String, "/robocon/team/message", 10)
        self.target_observation_pub = self.create_publisher(String, "/camera/target_observation", 10)
        self.signal_publishers = {
            topic: self.create_publisher(Bool, topic, 10)
            for topic in (
                "/mid360/preflight_ready",
                "/mid360/pose_valid",
                "/mid360/map_locked",
                "/robocon/mechanism/ball_present",
                "/robocon/mechanism/healthy",
                "/robocon/team/teammate_safe",
            )
        }
        if mode == "protocol":
            self.signal_publishers["/robocon/perception/target_valid"] = self.create_publisher(
                Bool, "/robocon/perception/target_valid", 10
            )
        self.create_subscription(String, "/robocon/game/state", self._state_callback, 10)
        self.create_subscription(String, "/robocon/game/action_decision", self._decision_callback, 10)
        self.create_subscription(String, "/robocon/action/request", self._action_callback, 10)
        self.create_subscription(String, "/robocon/team/ack", self._ack_callback, 10)
        self.create_subscription(String, "/robocon/perception/target_status", self._target_status_callback, 10)
        self.create_subscription(Bool, "/robocon/perception/target_valid", self._target_valid_callback, 10)
        self.latest_game_state = "BOOT"
        self.latest_task_state: dict[str, Any] = {}
        self.first_game_state_at: float | None = None
        self.latest_target_valid = False
        self.create_timer(0.05, self._tick)

    def _write(self, payload: dict[str, Any]) -> None:
        payload["wall_time_unix"] = time.time()
        self.events.write(json.dumps(payload, sort_keys=True) + "\n")
        self.events.flush()

    def _state_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.latest_game_state = str(payload.get("state", "unknown"))
        self.latest_task_state = dict(payload.get("task_state", {}))
        if self.first_game_state_at is None:
            self.first_game_state_at = time.monotonic()
        self.states.append(payload)
        self._write({"kind": "game_state", "payload": payload})

    def _decision_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.decisions.append(payload)
        self._write({"kind": "decision", "payload": payload})

    def _action_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.action_requests.append(payload)
        self._write({"kind": "action_request", "payload": payload})

    def _ack_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.acks.append(payload)
        self._write({"kind": "team_ack", "payload": payload})

    def _target_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        self.target_statuses.append(payload)
        self._write({"kind": "target_status", "payload": payload})

    def _target_valid_callback(self, message: Bool) -> None:
        self.latest_target_valid = bool(message.data)

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_at = time.monotonic()
        self._write({"kind": "phase", "phase": phase})

    def _publish_command(self, command: str) -> None:
        self.command_pub.publish(String(data=command))
        self.command_sent.add(command)
        self._write({"kind": "operator_command", "command": command})

    def _publish_signals(self) -> None:
        values = {
            "/mid360/preflight_ready": True,
            "/mid360/pose_valid": True,
            "/mid360/map_locked": self.map_locked,
            "/robocon/mechanism/ball_present": True,
            "/robocon/mechanism/healthy": True,
            "/robocon/team/teammate_safe": True,
        }
        for topic, value in values.items():
            publisher = self.signal_publishers.get(topic)
            if publisher is not None:
                publisher.publish(Bool(data=value))
        if "/robocon/perception/target_valid" in self.signal_publishers:
            self.signal_publishers["/robocon/perception/target_valid"].publish(Bool(data=True))

    def _publish_target(self, confidence: float, age_ns: int = 0) -> None:
        observed_at_ns = time.time_ns() - age_ns
        payload = {
            "confidence": confidence,
            "distance_m": 3.0,
            "stable": True,
            "observed_at_ns": observed_at_ns,
            "target_type": "hoop",
            "evidence_level": "synthetic",
            "evidence_source": "competition_fault_runtime_smoke",
        }
        self.target_observation_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _publish_heartbeat(self, *, duplicate: bool = False, expired: bool = False) -> str:
        now_ns = time.time_ns()
        if not duplicate:
            self.heartbeat_sequence += 1
        sequence = self.heartbeat_sequence
        if expired:
            created_at_ns = now_ns - 2_000_000_000
            expires_at_ns = now_ns - 1_000_000_000
        else:
            created_at_ns = now_ns
            expires_at_ns = now_ns + 1_000_000_000
        payload = {
            "protocol_version": 1,
            "message_type": "heartbeat",
            "task_id": "competition-fault-smoke",
            "sender_id": "ball_handler",
            "sequence": sequence,
            "created_at_ns": created_at_ns,
            "expires_at_ns": expires_at_ns,
            "payload": {"safe": True, "evidence_level": "synthetic"},
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.team_pub.publish(String(data=encoded))
        self._write({"kind": "team_message", "duplicate": duplicate, "expired": expired, "payload": payload})
        return encoded

    def _latest_decision(self, action: str) -> dict[str, Any] | None:
        for decision in reversed(self.decisions):
            if decision.get("action") == action:
                return decision
        return None

    def _completed(self, action: str, state: str) -> bool:
        for payload in reversed(self.states):
            completed = payload.get("completed_actions", {})
            result = completed.get(action, {}) if isinstance(completed, dict) else {}
            if result.get("state") == state:
                return True
        return False

    def _finish(self, outcome: str, reason: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        summary = {
            "evidence_level": "synthetic",
            "mode": self.mode,
            "outcome": outcome,
            "failure_reason": reason or None,
            "game_state_sequence": list(dict.fromkeys(str(item.get("state", "unknown")) for item in self.states)),
            "ack_payloads": self.acks,
            "target_status_tail": self.target_statuses[-8:],
            "action_request_actions": [item.get("action") for item in self.action_requests],
            "decisions_tail": self.decisions[-12:],
        }
        (self.run_dir / "fault_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write({"kind": "summary", **summary})

    def _protocol_passed(self) -> bool:
        reasons = [str(item.get("payload", {}).get("reason", "")) for item in self.acks]
        accepted = [bool(item.get("payload", {}).get("accepted", False)) for item in self.acks]
        return (
            len(self.acks) >= 3
            and accepted[0] is True
            and accepted[1] is False
            and accepted[2] is False
            and "stale_or_duplicate" in reasons
            and self.latest_game_state == "RECOVERY"
        )

    def _mechanism_passed(self) -> bool:
        invalid = self._latest_decision("fire_shot")
        return (
            invalid is not None
            and not bool(invalid.get("accepted", True))
            and "target" in str(invalid.get("reason", ""))
            and self._completed("ExecutePass", "failed")
            and self.latest_game_state == "RECOVERY"
            and not any(item.get("action") == "FireShot" for item in self.action_requests)
        )

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() - self.started_at > 30.0:
            self._finish("failed", f"timed out in phase {self.phase}")
            return
        self._publish_signals()
        if self.mode == "mechanism":
            if self.phase in {"WAITING_FOR_NODES", "ACTIVE", "TARGET_INVALID", "RESTORE_TARGET"}:
                if self.phase != "TARGET_INVALID":
                    self._publish_target(0.90)

        if self.phase == "WAITING_FOR_NODES":
            signal_subscriptions_ready = all(
                publisher.get_subscription_count() > 0
                for publisher in self.signal_publishers.values()
            )
            if (
                self.command_pub.get_subscription_count() > 0
                and signal_subscriptions_ready
                and self.first_game_state_at is not None
                and self.first_game_state_at + 0.8 < time.monotonic()
            ):
                self._publish_command("preflight")
                self._set_phase("WAIT_START")
            return
        if self.phase == "WAIT_START":
            if self.latest_game_state == "WAIT_START":
                if self.mode == "protocol":
                    self._publish_heartbeat()
                    self._set_phase("WAIT_HEARTBEAT_BASELINE")
                else:
                    self._publish_command("start")
                    self._set_phase("ACTIVE")
            return
        if self.phase == "WAIT_HEARTBEAT_BASELINE":
            if self.acks and bool(self.acks[-1].get("payload", {}).get("accepted", False)):
                self._publish_command("start")
                self._set_phase("ACTIVE")
            return
        if self.phase == "ACTIVE":
            if self.latest_game_state != "ACTIVE":
                return
            if self.mode == "protocol":
                if time.monotonic() - self.phase_started_at < 0.2:
                    return
                self._publish_heartbeat(duplicate=True)
                self._publish_heartbeat(expired=True)
                self.heartbeat_enabled = False
                self._set_phase("WAIT_HEARTBEAT_RECOVERY")
            else:
                self._set_phase("TARGET_INVALID")
            return
        if self.phase == "TARGET_INVALID":
            self._publish_target(0.20, age_ns=0)
            if self.target_statuses and self.target_statuses[-1].get("valid") is False:
                self._publish_command("fire_shot")
                self._set_phase("WAIT_TARGET_REJECTION")
            return
        if self.phase == "WAIT_TARGET_REJECTION":
            decision = self._latest_decision("fire_shot")
            if decision is not None and not bool(decision.get("accepted", True)):
                self._set_phase("RESTORE_TARGET")
            return
        if self.phase == "RESTORE_TARGET":
            self._publish_target(0.90)
            if self.target_statuses and self.target_statuses[-1].get("valid") is True:
                self._publish_command("prepare_receive")
                self._set_phase("WAIT_RECEIVE")
            return
        if self.phase == "WAIT_RECEIVE":
            if self._completed("PrepareReceive", "succeeded"):
                self._publish_command("prepare_pass")
                self._set_phase("WAIT_PASS_ARM")
            return
        if self.phase == "WAIT_PASS_ARM":
            if self._completed("PreparePass", "succeeded"):
                self._publish_command("execute_pass")
                self._set_phase("WAIT_PASS_FAILURE")
            return
        if self.phase == "WAIT_PASS_FAILURE":
            if self._mechanism_passed():
                self._finish("passed")
            return
        if self.phase == "WAIT_HEARTBEAT_RECOVERY":
            if self._protocol_passed():
                self._finish("passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("protocol", "mechanism"), required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = CompetitionFaultSmoke(args.run_dir, args.mode)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._finish("interrupted", "interrupted by operator")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    summary = json.loads((args.run_dir / "fault_summary.json").read_text(encoding="utf-8"))
    return 0 if summary["outcome"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
