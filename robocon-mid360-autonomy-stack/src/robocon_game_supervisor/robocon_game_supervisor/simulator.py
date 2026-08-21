"""State-evidenced synthetic action executor for local integration tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .actions import ActionDeduplicator, ACTION_NAMES, ActionFeedback, ActionRequest


@dataclass
class SyntheticMechanismPlant:
    """Minimal deterministic plant model used only by the synthetic profile.

    A request is never considered successful merely because time elapsed.  Each
    transition validates the previous simulated sensor state and returns the
    state evidence that made the transition legal.  It is deliberately not an
    emulation of a physical mechanism.
    """

    healthy: bool = True
    ball_present: bool = True
    receiver_ready: bool = False
    pass_armed: bool = False
    pass_executed: bool = False
    receipt_confirmed: bool = False
    shot_ready: bool = False
    shot_released: bool = False
    estop_latched: bool = False

    def apply(self, action: str) -> tuple[bool, str, dict[str, object]]:
        if action not in ACTION_NAMES:
            return False, "unsupported action", self.evidence()
        if self.estop_latched and action != "EmergencyStop":
            return False, "synthetic estop is latched", self.evidence()
        powered_actions = {
            "NavigateToPose", "PrepareReceive", "PreparePass", "ExecutePass",
            "CollectBall", "PrepareShot", "FireShot",
        }
        if action in powered_actions and not self.healthy:
            return False, "synthetic mechanism is unhealthy", self.evidence()

        if action == "PrepareReceive":
            self.receiver_ready = True
            return True, "synthetic receiver-ready sensor asserted", self.evidence()
        if action == "PreparePass":
            if not self.receiver_ready:
                return False, "receiver-ready sensor is false", self.evidence()
            if not self.ball_present:
                return False, "ball-present sensor is false", self.evidence()
            self.pass_armed = True
            return True, "synthetic pass-armed limit asserted", self.evidence()
        if action == "ExecutePass":
            if not self.pass_armed:
                return False, "pass-armed limit is false", self.evidence()
            if not self.ball_present:
                return False, "ball-present sensor is false", self.evidence()
            self.pass_executed = True
            self.ball_present = False
            return True, "synthetic transfer-release sensor asserted", self.evidence()
        if action == "CollectBall":
            if not self.pass_executed:
                return False, "pass-executed evidence is false", self.evidence()
            # This is a modeled arrival event, not a delay-based acknowledgement.
            self.receipt_confirmed = True
            self.ball_present = True
            return True, "synthetic receipt sensor asserted", self.evidence()
        if action == "PrepareShot":
            if not self.receipt_confirmed or not self.ball_present:
                return False, "receipt or ball-present sensor is false", self.evidence()
            self.shot_ready = True
            return True, "synthetic shot-ready limit asserted", self.evidence()
        if action == "FireShot":
            if not self.shot_ready or not self.ball_present:
                return False, "shot-ready or ball-present sensor is false", self.evidence()
            self.shot_released = True
            self.ball_present = False
            return True, "synthetic shot-release sensor asserted", self.evidence()
        if action == "EmergencyStop":
            self.estop_latched = True
            return True, "synthetic estop feedback asserted", self.evidence()
        if action == "AbortTask":
            self.pass_armed = False
            self.shot_ready = False
            return True, "synthetic task abort feedback asserted", self.evidence()
        return True, "synthetic navigation completion asserted", self.evidence()

    def evidence(self) -> dict[str, object]:
        return {
            "evidence_level": "synthetic",
            "evidence_source": "synthetic_mechanism_plant",
            "mechanism_healthy": self.healthy,
            "ball_present": self.ball_present,
            "receiver_ready": self.receiver_ready,
            "pass_armed": self.pass_armed,
            "pass_executed": self.pass_executed,
            "receipt_confirmed": self.receipt_confirmed,
            "shot_ready": self.shot_ready,
            "shot_released": self.shot_released,
            "estop_latched": self.estop_latched,
        }


class RoboconActionSimulator(Node):
    """Execute high-level requests against an explicit synthetic state model."""

    def __init__(self) -> None:
        super().__init__("robocon_action_simulator")
        self.declare_parameter("request_topic", "/robocon/action/request")
        self.declare_parameter("feedback_topic", "/robocon/action/feedback")
        self.declare_parameter("mechanism_state_topic", "/robocon/mechanism/state")
        self.declare_parameter("ball_present_topic", "/robocon/mechanism/ball_present")
        self.declare_parameter("mechanism_healthy_topic", "/robocon/mechanism/healthy")
        self.declare_parameter("failure_action", "")
        self.failure_action = str(self.get_parameter("failure_action").value)
        self.plant = SyntheticMechanismPlant()
        self._deduplicator = ActionDeduplicator()
        self.feedback_pub = self.create_publisher(
            String, str(self.get_parameter("feedback_topic").value), 10
        )
        self._state_pub = self.create_publisher(
            String, str(self.get_parameter("mechanism_state_topic").value), 10
        )
        self._ball_pub = self.create_publisher(
            Bool, str(self.get_parameter("ball_present_topic").value), 10
        )
        self._healthy_pub = self.create_publisher(
            Bool, str(self.get_parameter("mechanism_healthy_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("request_topic").value),
            self._request_callback,
            10,
        )
        self.create_timer(0.05, self._publish_plant_state)

    def _request_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            request = ActionRequest(
                protocol_version=int(payload["protocol_version"]),
                action_id=str(payload["action_id"]),
                task_id=str(payload["task_id"]),
                action=str(payload["action"]),
                sender_id=str(payload["sender_id"]),
                sequence=int(payload["sequence"]),
                created_at_ns=int(payload["created_at_ns"]),
                expires_at_ns=int(payload["expires_at_ns"]),
                parameters=dict(payload.get("parameters", {})),
            )
            request.validate()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning("Ignoring malformed action request: %s", error)
            return
        now_ns = time.time_ns()
        if not self._deduplicator.accept(request, now_ns):
            self._publish_feedback(request, "failed", "synthetic request is stale or duplicated")
            return
        self._publish_feedback(request, "accepted", "synthetic executor accepted request")
        if request.action == self.failure_action:
            self._publish_feedback(request, "failed", "synthetic failure injection")
            return
        succeeded, reason, evidence = self.plant.apply(request.action)
        self._publish_feedback(request, "succeeded" if succeeded else "failed", reason, evidence)
        self._publish_plant_state()

    def _publish_plant_state(self) -> None:
        evidence = self.plant.evidence()
        evidence["stamp_ns"] = time.time_ns()
        self._state_pub.publish(String(data=json.dumps(evidence, separators=(",", ":"))))
        self._ball_pub.publish(Bool(data=self.plant.ball_present))
        self._healthy_pub.publish(Bool(data=self.plant.healthy))

    def _publish_feedback(
        self,
        request: ActionRequest,
        state: str,
        reason: str,
        evidence: dict[str, object] | None = None,
    ) -> None:
        feedback = ActionFeedback(
            protocol_version=request.protocol_version,
            action_id=request.action_id,
            task_id=request.task_id,
            action=request.action,
            sender_id="synthetic_executor",
            state=state,
            created_at_ns=time.time_ns(),
            reason=reason,
            evidence=evidence or self.plant.evidence(),
        )
        payload = {
            "protocol_version": feedback.protocol_version,
            "action_id": feedback.action_id,
            "task_id": feedback.task_id,
            "action": feedback.action,
            "sender_id": feedback.sender_id,
            "state": feedback.state,
            "created_at_ns": feedback.created_at_ns,
            "reason": feedback.reason,
            "evidence": feedback.evidence,
        }
        self.feedback_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = RoboconActionSimulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
