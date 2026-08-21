#!/usr/bin/env python3
"""Exercise the real Gazebo localization chain and supervisor interlock.

The driver publishes only operator/perception/peer inputs and a controlled
Gazebo pause fault. FAST-LIO2, the mapper, fixed-map ICP, and localization
contract publish the pose and lock signals observed here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty


class FixedMapInterlockIntegration(Node):
    def __init__(self, run_dir: Path, map_file: Path, timeout_sec: float) -> None:
        super().__init__(
            "fixed_map_interlock_integration",
            parameter_overrides=[
                # The driver must keep ticking while Gazebo is intentionally
                # paused to exercise LOST and recovery. Sensor/estimator nodes
                # remain on simulation time; this orchestration node uses wall time.
                Parameter("use_sim_time", Parameter.Type.BOOL, False)
            ],
        )
        self.run_dir = run_dir
        self.map_file = map_file
        self.timeout_sec = timeout_sec
        self.started_wall = time.monotonic()
        self.phase = "WAITING_FOR_GRAPH"
        self.phase_started_wall = self.started_wall
        self.finished = False
        self.outcome = "running"
        self.failure_reason = ""
        self.latest_odom: Odometry | None = None
        self.latest_localizer: dict[str, Any] = {}
        self.latest_pose_status: dict[str, Any] = {}
        self.map_locked = False
        self.pose_valid = False
        self.preflight_ready = False
        self.supervisor_state = "UNKNOWN"
        self.decisions: list[dict[str, Any]] = []
        self.action_requests: list[dict[str, Any]] = []
        self.action_feedback: list[dict[str, Any]] = []
        self.state_history: list[dict[str, Any]] = []
        self.phase_history: list[dict[str, Any]] = []
        self.action_queue = [
            "prepare_receive",
            "prepare_pass",
            "execute_pass",
            "collect_ball",
            "prepare_shot",
        ]
        self.action_waiting_for: str | None = None
        self.fire_rejection_seen = False
        self.fire_success_seen = False
        self.pause_requested = False
        self.unpause_requested = False
        self.initialpose_sent = False
        self.started_match = False
        self.heartbeat_sequence = 0

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.command_pub = self.create_publisher(String, "/robocon/operator_command", 10)
        self.target_pub = self.create_publisher(String, "/camera/target_observation", 10)
        self.teammate_safe_pub = self.create_publisher(Bool, "/robocon/team/teammate_safe", 10)
        self.team_message_pub = self.create_publisher(String, "/robocon/team/message", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel_chassis", 10)
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")

        self.create_subscription(Odometry, "/mid360/local_odometry", self._odom_callback, 10)
        self.create_subscription(Bool, "/mid360/map_locked", self._map_locked_callback, 10)
        self.create_subscription(Bool, "/mid360/pose_valid", self._pose_valid_callback, 10)
        self.create_subscription(Bool, "/mid360/preflight_ready", self._preflight_callback, 10)
        self.create_subscription(String, "/mid360/map_localization_diagnostics", self._localizer_callback, 10)
        self.create_subscription(String, "/mid360/pose_status", self._pose_status_callback, 10)
        self.create_subscription(String, "/robocon/game/state", self._game_state_callback, 10)
        self.create_subscription(String, "/robocon/game/action_decision", self._decision_callback, 10)
        self.create_subscription(String, "/robocon/action/request", self._action_request_callback, 10)
        self.create_subscription(String, "/robocon/action/feedback", self._action_feedback_callback, 10)

        self.events = (run_dir / "integration_telemetry.jsonl").open("w", encoding="utf-8")
        self.create_timer(0.05, self._tick)

    def _write(self, kind: str, **payload: Any) -> None:
        record = {"kind": kind, "wall_time_unix": time.time(), "phase": self.phase, **payload}
        self.events.write(json.dumps(record, sort_keys=True) + "\n")
        self.events.flush()

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_wall = time.monotonic()
        entry = {"phase": phase, "wall_time_unix": time.time()}
        self.phase_history.append(entry)
        self._write("phase", **entry)

    def _odom_callback(self, message: Odometry) -> None:
        self.latest_odom = message

    def _map_locked_callback(self, message: Bool) -> None:
        self.map_locked = bool(message.data)

    def _pose_valid_callback(self, message: Bool) -> None:
        self.pose_valid = bool(message.data)

    def _preflight_callback(self, message: Bool) -> None:
        self.preflight_ready = bool(message.data)

    def _localizer_callback(self, message: String) -> None:
        try:
            self.latest_localizer = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._write("malformed_localizer_diagnostic", value=message.data)

    def _pose_status_callback(self, message: String) -> None:
        try:
            payload = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.latest_pose_status = payload
        self._write("pose_status", payload=payload)

    def _game_state_callback(self, message: String) -> None:
        try:
            payload = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.supervisor_state = str(payload.get("state", "UNKNOWN"))
        self.state_history.append(payload)
        self._write("game_state", payload=payload)

    def _decision_callback(self, message: String) -> None:
        try:
            payload = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.decisions.append(payload)
        self._write("decision", payload=payload)
        if payload.get("action") == "FireShot" and not bool(payload.get("accepted", True)):
            self.fire_rejection_seen = True
        if payload.get("action") == "fire_shot" and not bool(payload.get("accepted", True)):
            self.fire_rejection_seen = True

    def _action_request_callback(self, message: String) -> None:
        try:
            payload = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.action_requests.append(payload)
        self._write("action_request", payload=payload)

    def _action_feedback_callback(self, message: String) -> None:
        try:
            payload = dict(json.loads(message.data))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.action_feedback.append(payload)
        self._write("action_feedback", payload=payload)
        if payload.get("action") == "FireShot" and payload.get("state") == "succeeded":
            self.fire_success_seen = True

    def _publish_inputs(self) -> None:
        self.teammate_safe_pub.publish(Bool(data=True))
        # The boolean is a local safety signal; the expiring envelope is the
        # peer heartbeat consumed by the supervisor's transport monitor.
        self.heartbeat_sequence += 1
        now_ns = time.time_ns()
        heartbeat = {
            "protocol_version": 1,
            "message_type": "heartbeat",
            "task_id": "gazebo-fixed-map-interlock",
            "sender_id": "ball_handler",
            "sequence": self.heartbeat_sequence,
            "created_at_ns": now_ns,
            "expires_at_ns": now_ns + 1_000_000_000,
            "payload": {
                "safe": True,
                "evidence_level": "gazebo_simulation",
                "evidence_source": "fixed_map_interlock_integration",
            },
        }
        self.team_message_pub.publish(
            String(data=json.dumps(heartbeat, separators=(",", ":")))
        )
        # Carry the safety assertion through the same ordered transport as the
        # heartbeat.  Under a loaded Gazebo graph this avoids relying on the
        # separate Bool callback alone; the supervisor still requires both the
        # fresh safety assertion and the fresh heartbeat.
        teammate_safe = dict(heartbeat)
        teammate_safe["message_type"] = "teammate_safe"
        self.team_message_pub.publish(
            String(data=json.dumps(teammate_safe, separators=(",", ":")))
        )
        self._write("team_heartbeat", sequence=self.heartbeat_sequence)
        target = {
            "confidence": 0.95,
            "distance_m": 3.0,
            "stable": True,
            "observed_at_ns": time.time_ns(),
            "target_type": "hoop",
            "evidence_level": "gazebo_simulation",
            "evidence_source": "fixed_map_interlock_integration",
        }
        self.target_pub.publish(String(data=json.dumps(target, separators=(",", ":"))))

    def _send_command(self, command: str) -> None:
        self.command_pub.publish(String(data=command))
        self._write("operator_command", command=command)

    def _send_initialpose(self) -> None:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.pose.orientation.w = 1.0
        for index in (0, 7, 14):
            message.pose.covariance[index] = 0.05
        self.initialpose_pub.publish(message)
        self.initialpose_sent = True
        self._write("initialpose", frame_id=message.header.frame_id)

    def _send_next_action(self) -> None:
        if self.action_waiting_for is not None:
            return
        if not self.action_queue:
            self._set_phase("REQUEST_PAUSE")
            return
        action = self.action_queue.pop(0)
        self.action_waiting_for = action
        self._send_command(action)

    def _action_succeeded(self, action: str) -> bool:
        return any(
            str(item.get("action")) == action.replace("_", "").lower()
            and str(item.get("state")) == "succeeded"
            for item in self.action_feedback
        ) or any(
            str(item.get("action")) == {
                "prepare_receive": "PrepareReceive",
                "prepare_pass": "PreparePass",
                "execute_pass": "ExecutePass",
                "collect_ball": "CollectBall",
                "prepare_shot": "PrepareShot",
            }.get(action, action)
            and str(item.get("state")) == "succeeded"
            for item in self.action_feedback
        )

    def _request_physics(self, pause: bool) -> None:
        client = self.pause_client if pause else self.unpause_client
        if not client.service_is_ready():
            return
        if pause and self.pause_requested:
            return
        if not pause and self.unpause_requested:
            return
        future = client.call_async(Empty.Request())
        if pause:
            self.pause_requested = True
        else:
            self.unpause_requested = True
        self._write("physics_request", action="pause" if pause else "unpause")
        future.add_done_callback(
            lambda done: self._write(
                "physics_response",
                action="pause" if pause else "unpause",
                success=done.exception() is None,
            )
        )

    def _finish(self, outcome: str, reason: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        self.outcome = outcome
        self.failure_reason = reason
        self.cmd_vel_pub.publish(Twist())
        summary = {
            "evidence_level": "gazebo_simulation",
            "diagnostic_only": True,
            "outcome": outcome,
            "failure_reason": reason or None,
            "map_file": str(self.map_file),
            "map_locked_seen": self.map_locked,
            "pose_valid_seen": self.pose_valid,
            "initialpose_sent": self.initialpose_sent,
            "localizer_status_tail": self.latest_localizer,
            "pose_status_tail": self.latest_pose_status,
            "supervisor_state": self.supervisor_state,
            "phase_history": self.phase_history,
            "fire_rejection_seen": self.fire_rejection_seen,
            "fire_success_seen": self.fire_success_seen,
            "fire_action_requests": [
                item for item in self.action_requests
                if str(item.get("action")) == "FireShot"
            ],
            "action_feedback_tail": self.action_feedback[-20:],
            "decision_tail": self.decisions[-20:],
            "state_tail": self.state_history[-10:],
            "elapsed_wall_sec": time.monotonic() - self.started_wall,
        }
        (self.run_dir / "integration_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._write("summary", **summary)
        self.events.close()

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() - self.started_wall > self.timeout_sec:
            self._finish("failed", f"timed out in phase {self.phase}")
            return
        self._publish_inputs()

        if self.phase == "WAITING_FOR_GRAPH":
            if self.latest_odom is not None and self.latest_localizer:
                status = str(self.latest_localizer.get("status", ""))
                if status in {"WAITING_FOR_INITIALPOSE", "WAITING_FOR_FRESH_ODOM"}:
                    self._send_initialpose()
                    self._set_phase("WAITING_FOR_TRACKING")
            return
        if self.phase == "WAITING_FOR_TRACKING":
            if self.map_locked and self.pose_valid and self.preflight_ready:
                self._send_command("preflight")
                self._set_phase("WAITING_FOR_START")
            return
        if self.phase == "WAITING_FOR_START":
            if self.supervisor_state == "WAIT_START":
                self._send_command("start")
                self._set_phase("ACTIVE_MOTION")
            return
        if self.phase == "ACTIVE_MOTION":
            elapsed = time.monotonic() - self.phase_started_wall
            command = Twist()
            command.linear.x = 0.08 if elapsed < 5.0 else 0.0
            self.cmd_vel_pub.publish(command)
            if elapsed >= 6.0 and self.supervisor_state == "ACTIVE":
                self._set_phase("MECHANISM_SEQUENCE")
            return
        if self.phase == "MECHANISM_SEQUENCE":
            if self.supervisor_state != "ACTIVE":
                return
            if self.action_waiting_for is not None:
                action_name = {
                    "prepare_receive": "PrepareReceive",
                    "prepare_pass": "PreparePass",
                    "execute_pass": "ExecutePass",
                    "collect_ball": "CollectBall",
                    "prepare_shot": "PrepareShot",
                }[self.action_waiting_for]
                if self._action_succeeded(action_name):
                    self._write("action_gate_succeeded", action=action_name)
                    self.action_waiting_for = None
            self._send_next_action()
            return
        if self.phase == "REQUEST_PAUSE":
            self._request_physics(True)
            self._set_phase("WAITING_FOR_LOST")
            return
        if self.phase == "WAITING_FOR_LOST":
            lost = (
                not self.map_locked
                or not self.pose_valid
                or str(self.latest_pose_status.get("tracking_state", "")) == "LOST"
            )
            if lost:
                self._send_command("fire_shot")
                self._set_phase("WAITING_FOR_FIRE_REJECTION")
            return
        if self.phase == "WAITING_FOR_FIRE_REJECTION":
            if self.fire_rejection_seen:
                self._request_physics(False)
                self._set_phase("WAITING_FOR_RECOVERY")
            return
        if self.phase == "WAITING_FOR_RECOVERY":
            recovered = self.map_locked and self.pose_valid and str(
                self.latest_pose_status.get("tracking_state", "")
            ) == "TRACKING"
            if recovered:
                # The localization contract publishes readiness asynchronously
                # after odometry/map recovery.  Do not send preflight in the
                # same tick as TRACKING: the supervisor may still hold the
                # previous false readiness sample and would enter FAULT.
                self._set_phase("WAITING_FOR_RECOVERY_PREFLIGHT")
            return
        if self.phase == "WAITING_FOR_RECOVERY_PREFLIGHT":
            if self.preflight_ready:
                self._send_command("preflight")
                self._set_phase("WAITING_FOR_RECOVERY_START")
            return
        if self.phase == "WAITING_FOR_RECOVERY_START":
            if self.supervisor_state == "WAIT_START":
                self._send_command("start")
                self._set_phase("WAITING_FOR_ACTIVE_AFTER_RECOVERY")
            return
        if self.phase == "WAITING_FOR_ACTIVE_AFTER_RECOVERY":
            if self.supervisor_state == "ACTIVE":
                self._send_command("fire_shot")
                self._set_phase("WAITING_FOR_FIRE_SUCCESS")
            return
        if self.phase == "WAITING_FOR_FIRE_SUCCESS":
            if self.fire_success_seen:
                self._finish("passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--map-file", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    args = parser.parse_args()
    if not args.map_file.is_file():
        parser.error(f"map file does not exist: {args.map_file}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = FixedMapInterlockIntegration(args.run_dir, args.map_file, args.timeout_sec)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._finish("interrupted", "interrupted by operator")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    summary_path = args.run_dir / "integration_summary.json"
    if not summary_path.is_file():
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return 0 if summary.get("outcome") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
