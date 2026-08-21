"""ROS 2 adapter for the hardware-independent ROBOCON supervisor core."""

from __future__ import annotations

import json
import math
import time
import uuid

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .actions import ActionFeedback, ActionRequest
from .protocol import TeamLink, envelope_to_json
from .supervisor import GameSupervisor, SafetySnapshot


class RoboconGameSupervisorNode(Node):
    """Translate ROS status signals and operator commands into supervisor decisions."""

    def __init__(self) -> None:
        super().__init__("robocon_game_supervisor")
        self.declare_parameter("command_topic", "/robocon/operator_command")
        self.declare_parameter("state_topic", "/robocon/game/state")
        self.declare_parameter("decision_topic", "/robocon/game/action_decision")
        self.declare_parameter("failure_topic", "/robocon/game/failure")
        self.declare_parameter("action_request_topic", "/robocon/action/request")
        self.declare_parameter("action_feedback_topic", "/robocon/action/feedback")
        self.declare_parameter("action_sender_id", "target_computer_game_supervisor")
        self.declare_parameter("action_ttl_sec", 1.0)
        self.declare_parameter("task_id", "TBD")
        self.declare_parameter("team_message_topic", "/robocon/team/message")
        self.declare_parameter("team_ack_topic", "/robocon/team/ack")
        self.declare_parameter("team_robot_id", "shooter")
        self.declare_parameter("team_ack_ttl_sec", 1.0)
        self.declare_parameter("require_teammate_heartbeat", True)
        self.declare_parameter("teammate_heartbeat_timeout_sec", 0.50)
        self.declare_parameter("preflight_ready_topic", "/mid360/preflight_ready")
        self.declare_parameter("pose_valid_topic", "/mid360/pose_valid")
        self.declare_parameter("map_locked_topic", "/mid360/map_locked")
        self.declare_parameter("pose_diagnostic_topic", "/mid360/pose_diagnostics")
        self.declare_parameter("target_valid_topic", "/robocon/perception/target_valid")
        self.declare_parameter("ball_present_topic", "/robocon/mechanism/ball_present")
        self.declare_parameter("mechanism_healthy_topic", "/robocon/mechanism/healthy")
        self.declare_parameter("teammate_safe_topic", "/robocon/team/teammate_safe")
        self.declare_parameter("pose_age_limit_sec", 0.30)
        self.declare_parameter("target_age_limit_sec", 0.30)
        self.declare_parameter("signal_silence_limit_sec", 0.50)
        self.declare_parameter("auto_recovery_on_signal_loss", True)
        self.declare_parameter("safety_monitor_hz", 10.0)
        self.declare_parameter("publish_hz", 10.0)

        self.supervisor = GameSupervisor(
            float(self.get_parameter("pose_age_limit_sec").value),
            float(self.get_parameter("target_age_limit_sec").value),
        )
        self.signal_silence_limit_sec = float(self.get_parameter("signal_silence_limit_sec").value)
        if self.signal_silence_limit_sec <= 0.0:
            raise ValueError("signal_silence_limit_sec must be positive")

        self._signals: dict[str, tuple[bool, float | None]] = {
            "preflight_ready": (False, None),
            "pose_valid": (False, None),
            "map_locked": (False, None),
            "target_valid": (False, None),
            "ball_present": (False, None),
            "mechanism_healthy": (False, None),
            "teammate_safe": (False, None),
        }
        self._pose_age_sec = math.inf
        self._last_decision = {"accepted": True, "action": "BOOT", "reason": "initialized"}
        self._action_sequence = 0
        configured_task_id = str(self.get_parameter("task_id").value)
        self._task_id = configured_task_id if configured_task_id != "TBD" else f"task-{uuid.uuid4().hex[:12]}"
        self._action_sender_id = str(self.get_parameter("action_sender_id").value)
        self._action_ttl_sec = float(self.get_parameter("action_ttl_sec").value)
        if self._action_ttl_sec <= 0.0:
            raise ValueError("action_ttl_sec must be positive")
        self._pending_actions: dict[str, ActionRequest] = {}
        self._action_status: dict[str, str] = {}
        self._completed_actions: dict[str, dict[str, object]] = {}
        self._team_link = TeamLink(
            str(self.get_parameter("team_robot_id").value),
            float(self.get_parameter("team_ack_ttl_sec").value),
            expected_task_id=self._task_id,
        )
        self._require_teammate_heartbeat = bool(self.get_parameter("require_teammate_heartbeat").value)
        self._auto_recovery_on_signal_loss = bool(
            self.get_parameter("auto_recovery_on_signal_loss").value
        )
        self._teammate_heartbeat_timeout_sec = float(
            self.get_parameter("teammate_heartbeat_timeout_sec").value
        )
        if self._teammate_heartbeat_timeout_sec <= 0.0:
            raise ValueError("teammate_heartbeat_timeout_sec must be positive")

        self._state_pub = self.create_publisher(String, str(self.get_parameter("state_topic").value), 10)
        self._decision_pub = self.create_publisher(String, str(self.get_parameter("decision_topic").value), 10)
        self._failure_pub = self.create_publisher(String, str(self.get_parameter("failure_topic").value), 10)
        self._action_request_pub = self.create_publisher(
            String, str(self.get_parameter("action_request_topic").value), 10
        )
        self._team_ack_pub = self.create_publisher(
            String, str(self.get_parameter("team_ack_topic").value), 10
        )
        self.create_subscription(String, str(self.get_parameter("command_topic").value), self._command_callback, 10)
        self.create_subscription(
            String,
            str(self.get_parameter("action_feedback_topic").value),
            self._action_feedback_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("team_message_topic").value),
            self._team_message_callback,
            10,
        )
        for signal in ("preflight_ready", "pose_valid", "map_locked", "target_valid", "ball_present", "mechanism_healthy", "teammate_safe"):
            topic = str(self.get_parameter(f"{signal}_topic").value)
            self.create_subscription(Bool, topic, lambda msg, name=signal: self._bool_callback(name, msg), 10)
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter("pose_diagnostic_topic").value),
            self._pose_diagnostic_callback,
            10,
        )

        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        safety_monitor_hz = float(self.get_parameter("safety_monitor_hz").value)
        if safety_monitor_hz <= 0.0:
            raise ValueError("safety_monitor_hz must be positive")
        self.create_timer(1.0 / publish_hz, self._publish_state)
        self.create_timer(0.05, self._expire_actions)
        self.create_timer(1.0 / safety_monitor_hz, self._monitor_safety)

    def _bool_callback(self, name: str, message: Bool) -> None:
        self._signals[name] = (bool(message.data), time.monotonic())

    def _pose_diagnostic_callback(self, message: DiagnosticArray) -> None:
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            raw_age = values.get("pose_age_sec")
            if raw_age is not None:
                try:
                    self._pose_age_sec = float(raw_age)
                except ValueError:
                    self._pose_age_sec = math.inf

    def _command_callback(self, message: String) -> None:
        command = self._parse_command(message.data)
        if command is None:
            self._publish_decision(False, "invalid_command", "command must be plain text or JSON with a command field")
            return
        try:
            self._dispatch(command)
        except (RuntimeError, ValueError) as error:
            self._publish_decision(False, command, str(error))

    def _action_feedback_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            feedback = ActionFeedback(
                protocol_version=int(payload["protocol_version"]),
                action_id=str(payload["action_id"]),
                task_id=str(payload["task_id"]),
                action=str(payload["action"]),
                sender_id=str(payload["sender_id"]),
                state=str(payload["state"]),
                created_at_ns=int(payload["created_at_ns"]),
                reason=str(payload.get("reason", "")),
                evidence=dict(payload.get("evidence", {})),
            )
            feedback.validate()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._publish_decision(False, "action_feedback", f"malformed feedback: {error}")
            return
        request = self._pending_actions.get(feedback.action_id)
        if request is None or feedback.task_id != self._task_id or feedback.action != request.action:
            self._publish_decision(False, feedback.action, "feedback does not match a pending action")
            return
        if feedback.state in {"accepted", "running"}:
            self._action_status[feedback.action_id] = feedback.state
            self._publish_decision(True, feedback.action, feedback.reason or feedback.state)
            return
        self._pending_actions.pop(feedback.action_id, None)
        self._action_status[feedback.action_id] = feedback.state
        self._completed_actions[feedback.action] = {
            "action_id": feedback.action_id,
            "state": feedback.state,
            "reason": feedback.reason,
            "created_at_ns": feedback.created_at_ns,
            "evidence": feedback.evidence,
        }
        if feedback.state == "succeeded":
            decision = self.supervisor.apply_action_success(feedback.action, feedback.evidence)
            self._publish_decision(decision.accepted, feedback.action, decision.reason)
            return
        decision = self.supervisor.handle_action_failure(
            feedback.action, feedback.reason or f"action {feedback.state}"
        )
        self._publish_decision(False, feedback.action, decision.reason)

    def _team_message_callback(self, message: String) -> None:
        try:
            ack, accepted, reason = self._team_link.receive(message.data, time.time_ns())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._publish_decision(False, "team_message", f"malformed team message: {error}")
            return
        self._team_ack_pub.publish(String(data=envelope_to_json(ack)))
        if not accepted:
            return
        # Peer evidence is accepted only through the same ordered core methods.
        try:
            envelope = json.loads(message.data)
            message_type = str(envelope.get("message_type", ""))
            payload = dict(envelope.get("payload", {}))
            if message_type == "teammate_safe":
                self._signals["teammate_safe"] = (bool(payload.get("safe", False)), time.monotonic())
            elif message_type == "receiver_ready":
                self._publish_decision_from_core(message_type, self.supervisor.set_receiver_ready())
            elif message_type == "pass_armed":
                self._publish_decision_from_core(message_type, self.supervisor.arm_pass())
            elif message_type == "pass_executed":
                self._publish_decision_from_core(message_type, self.supervisor.confirm_pass_executed())
            elif message_type == "receipt_confirmed":
                self._publish_decision_from_core(message_type, self.supervisor.confirm_receipt())
            elif message_type == "heartbeat":
                self._publish_decision(True, message_type, reason)
        except (TypeError, ValueError) as error:
            self._publish_decision(False, "team_message", f"invalid team payload: {error}")

    @staticmethod
    def _parse_command(value: str) -> str | None:
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            command = payload.get("command") if isinstance(payload, dict) else None
            return str(command).strip() if command else None
        return raw

    def _dispatch(self, command: str) -> None:
        if command == "preflight":
            self.supervisor.enter_preflight()
            ready, _ = self._signals["preflight_ready"]
            self.supervisor.preflight_result(ready, "preflight_ready is false")
            self._publish_decision(ready, command, "preflight accepted" if ready else "preflight is not ready")
            return
        if command == "start":
            self.supervisor.start()
            self._publish_decision(True, command, "competition started")
            return
        if command == "pause":
            self.supervisor.pause()
            self._publish_decision(True, command, "competition paused")
            return
        if command == "resume":
            self.supervisor.resume()
            self._publish_decision(True, command, "competition resumed")
            return
        if command == "recover":
            self.supervisor.enter_recovery("operator recovery request")
            self._publish_decision(True, command, "recovery entered")
            return
        if command == "finish":
            self.supervisor.finish()
            self._publish_decision(True, command, "competition finished")
            return
        if command == "estop":
            self.supervisor.emergency_stop()
            self._publish_decision(True, command, "emergency stop latched")
            return
        legacy_evidence_commands = {
            "receiver_ready": "PrepareReceive",
            "pass_armed": "PreparePass",
            "pass_executed": "ExecutePass",
            "receipt_confirmed": "CollectBall",
            "fire_shot": "FireShot",
        }
        action_aliases = {
            "navigate_to_pose": "NavigateToPose",
            "prepare_receive": "PrepareReceive",
            "collect_ball": "CollectBall",
            "prepare_pass": "PreparePass",
            "execute_pass": "ExecutePass",
            "prepare_shot": "PrepareShot",
            "fire_shot": "FireShot",
            "abort_task": "AbortTask",
            "emergency_stop": "EmergencyStop",
        }
        action = action_aliases.get(command, legacy_evidence_commands.get(command))
        if action is not None:
            safety = (
                self._safety_snapshot()
                if action in self.supervisor._LOCALIZATION_GATED_ACTIONS
                else None
            )
            decision = self.supervisor.can_issue_action(action, safety)
            if not decision.accepted:
                self._publish_decision_from_core(command, decision)
                return
            if action == "EmergencyStop":
                self.supervisor.emergency_stop("operator requested emergency stop")
            elif action == "AbortTask" and self.supervisor.state.value in {"ACTIVE", "PAUSED"}:
                self.supervisor.enter_recovery("operator requested task abort")
            self._publish_action_request(action)
            return
        self._publish_decision(False, command, "unknown operator command")

    def _publish_action_request(self, action: str, parameters: dict[str, object] | None = None) -> None:
        now_ns = time.time_ns()
        self._action_sequence += 1
        request = ActionRequest(
            protocol_version=1,
            action_id=f"{self._task_id}-{self._action_sequence}",
            task_id=self._task_id,
            action=action,
            sender_id=self._action_sender_id,
            sequence=self._action_sequence,
            created_at_ns=now_ns,
            expires_at_ns=now_ns + int(self._action_ttl_sec * 1_000_000_000),
            parameters=parameters or {},
        )
        request.validate()
        self._pending_actions[request.action_id] = request
        self._action_status[request.action_id] = "published"
        payload = {
            "protocol_version": request.protocol_version,
            "action_id": request.action_id,
            "task_id": request.task_id,
            "action": request.action,
            "sender_id": request.sender_id,
            "sequence": request.sequence,
            "created_at_ns": request.created_at_ns,
            "expires_at_ns": request.expires_at_ns,
            "parameters": request.parameters,
        }
        self._action_request_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._publish_decision(True, action, "action request published")

    def _expire_actions(self) -> None:
        now_ns = time.time_ns()
        expired = [
            action_id
            for action_id, request in self._pending_actions.items()
            if now_ns > request.expires_at_ns
        ]
        for action_id in expired:
            request = self._pending_actions.pop(action_id)
            self._action_status[action_id] = "expired"
            decision = self.supervisor.handle_action_failure(
                request.action, "action request expired before feedback"
            )
            self._completed_actions[request.action] = {
                "action_id": request.action_id,
                "state": "expired",
                "reason": decision.reason,
                "created_at_ns": now_ns,
                "evidence": {},
            }
            self._publish_decision(False, request.action, decision.reason)

    def _monitor_safety(self) -> None:
        if not self._auto_recovery_on_signal_loss or self.supervisor.state.value != "ACTIVE":
            return
        decision = self.supervisor.monitor_safety(
            self._safety_snapshot(), require_teammate=self._require_teammate_heartbeat
        )
        if decision.accepted:
            return
        self._publish_decision(False, decision.action, decision.reason)

    def _safety_snapshot(self) -> SafetySnapshot:
        now = time.monotonic()

        def fresh(name: str) -> bool:
            value, updated = self._signals[name]
            return value and updated is not None and now - updated <= self.signal_silence_limit_sec

        target_updated = self._signals["target_valid"][1]
        target_age = math.inf if target_updated is None else max(0.0, now - target_updated)
        pose_age = self._pose_age_sec
        if not math.isfinite(pose_age):
            pose_updated = self._signals["pose_valid"][1]
            pose_age = math.inf if pose_updated is None else max(0.0, now - pose_updated)
        teammate_fresh = fresh("teammate_safe")
        if self._require_teammate_heartbeat:
            heartbeat_ns = self._team_link.last_heartbeat_ns
            teammate_fresh = teammate_fresh and heartbeat_ns is not None and (
                time.time() - (heartbeat_ns / 1_000_000_000.0)
                <= self._teammate_heartbeat_timeout_sec
            )
        return SafetySnapshot(
            pose_valid=fresh("pose_valid"),
            map_locked=fresh("map_locked"),
            pose_age_sec=pose_age,
            target_valid=fresh("target_valid"),
            target_age_sec=target_age,
            ball_present=fresh("ball_present"),
            mechanism_healthy=fresh("mechanism_healthy"),
            teammate_safe=teammate_fresh,
        )

    def _publish_decision_from_core(self, command: str, decision) -> None:
        self._publish_decision(decision.accepted, command, decision.reason)

    def _publish_decision(self, accepted: bool, action: str, reason: str) -> None:
        self._last_decision = {"accepted": accepted, "action": action, "reason": reason}
        payload = dict(self._last_decision)
        payload["state"] = self.supervisor.state.value
        payload["stamp_ns"] = time.time_ns()
        serialized = json.dumps(payload, separators=(",", ":"))
        self._decision_pub.publish(String(data=serialized))
        if not accepted:
            self._failure_pub.publish(String(data=serialized))

    def _publish_state(self) -> None:
        payload = {
            "state": self.supervisor.state.value,
            "task_state": {role.value: state.value for role, state in self.supervisor.task_state.items()},
            "last_failure_reason": self.supervisor.last_failure_reason,
            "last_decision": self._last_decision,
            "pending_actions": {
                action_id: self._action_status.get(action_id, "unknown")
                for action_id in self._pending_actions
            },
            "completed_actions": dict(self._completed_actions),
            "stamp_ns": time.time_ns(),
        }
        self._state_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = RoboconGameSupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
