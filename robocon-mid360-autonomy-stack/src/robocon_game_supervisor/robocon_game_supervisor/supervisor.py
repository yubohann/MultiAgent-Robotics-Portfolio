"""Deterministic game state and action safety gates without ROS dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .actions import ACTION_NAMES


class SupervisorState(str, Enum):
    BOOT = "BOOT"
    PREFLIGHT = "PREFLIGHT"
    WAIT_START = "WAIT_START"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    RECOVERY = "RECOVERY"
    FAULT = "FAULT"
    ESTOP = "ESTOP"
    FINISHED = "FINISHED"


class RobotRole(str, Enum):
    SHOOTER = "shooter"
    BALL_HANDLER = "ball_handler"


class RobotTaskState(str, Enum):
    IDLE = "idle"
    RECEIVER_READY = "receiver_ready"
    PASS_ARMED = "pass_armed"
    PASS_EXECUTED = "pass_executed"
    RECEIPT_CONFIRMED = "receipt_confirmed"
    PREPARING_SHOT = "preparing_shot"
    SHOT_EXECUTED = "shot_executed"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class SafetySnapshot:
    pose_valid: bool
    map_locked: bool
    pose_age_sec: float
    target_valid: bool
    target_age_sec: float
    ball_present: bool
    mechanism_healthy: bool
    teammate_safe: bool


@dataclass(frozen=True)
class ActionDecision:
    accepted: bool
    action: str
    reason: str


class GameSupervisor:
    # These requests move the robot or release a ball toward a field-relative
    # target. They must not cross the adapter boundary while localization is
    # unlocked, even before the periodic safety monitor enters RECOVERY.
    _LOCALIZATION_GATED_ACTIONS = {
        "NavigateToPose",
        "PreparePass",
        "ExecutePass",
        "PrepareShot",
        "FireShot",
    }

    def __init__(self, pose_age_limit_sec: float = 0.30, target_age_limit_sec: float = 0.30) -> None:
        if pose_age_limit_sec <= 0.0 or target_age_limit_sec <= 0.0:
            raise ValueError("age limits must be positive")
        self.state = SupervisorState.BOOT
        self.pose_age_limit_sec = pose_age_limit_sec
        self.target_age_limit_sec = target_age_limit_sec
        self.task_state = {
            RobotRole.SHOOTER: RobotTaskState.IDLE,
            RobotRole.BALL_HANDLER: RobotTaskState.IDLE,
        }
        self.last_failure_reason = ""

    def enter_preflight(self) -> None:
        self._require({SupervisorState.BOOT, SupervisorState.RECOVERY})
        self.state = SupervisorState.PREFLIGHT

    def preflight_result(self, ready: bool, reason: str = "") -> None:
        self._require({SupervisorState.PREFLIGHT})
        if ready:
            self.state = SupervisorState.WAIT_START
            self.last_failure_reason = ""
        else:
            self._fail(reason or "preflight is not ready")

    def start(self) -> None:
        self._require({SupervisorState.WAIT_START})
        self.state = SupervisorState.ACTIVE

    def pause(self, reason: str = "operator pause") -> None:
        self._require({SupervisorState.ACTIVE})
        self.state = SupervisorState.PAUSED
        self.last_failure_reason = reason

    def resume(self) -> None:
        self._require({SupervisorState.PAUSED})
        self.state = SupervisorState.ACTIVE

    def enter_recovery(self, reason: str) -> None:
        self._require({SupervisorState.ACTIVE, SupervisorState.PAUSED, SupervisorState.FAULT})
        self.state = SupervisorState.RECOVERY
        self.last_failure_reason = reason

    def fault(self, reason: str) -> None:
        self._require({SupervisorState.PREFLIGHT, SupervisorState.WAIT_START, SupervisorState.ACTIVE, SupervisorState.PAUSED, SupervisorState.RECOVERY})
        self._fail(reason)

    def emergency_stop(self, reason: str = "emergency stop") -> None:
        self.state = SupervisorState.ESTOP
        self.last_failure_reason = reason

    def finish(self) -> None:
        self._require({SupervisorState.ACTIVE, SupervisorState.PAUSED})
        self.state = SupervisorState.FINISHED

    def set_receiver_ready(self) -> ActionDecision:
        if self.state != SupervisorState.ACTIVE:
            return ActionDecision(False, "receiver_ready", "supervisor is not ACTIVE")
        self.task_state[RobotRole.SHOOTER] = RobotTaskState.RECEIVER_READY
        return ActionDecision(True, "receiver_ready", "receiver is ready")

    def arm_pass(self) -> ActionDecision:
        if self.task_state[RobotRole.SHOOTER] != RobotTaskState.RECEIVER_READY:
            return ActionDecision(False, "pass_armed", "receiver_ready evidence is missing")
        self.task_state[RobotRole.SHOOTER] = RobotTaskState.PASS_ARMED
        return ActionDecision(True, "pass_armed", "pass is armed")

    def confirm_pass_executed(self) -> ActionDecision:
        if self.task_state[RobotRole.SHOOTER] != RobotTaskState.PASS_ARMED:
            return ActionDecision(False, "pass_executed", "pass_armed evidence is missing")
        self.task_state[RobotRole.SHOOTER] = RobotTaskState.PASS_EXECUTED
        return ActionDecision(True, "pass_executed", "pass execution acknowledged")

    def confirm_receipt(self) -> ActionDecision:
        if self.task_state[RobotRole.SHOOTER] != RobotTaskState.PASS_EXECUTED:
            return ActionDecision(False, "receipt_confirmed", "pass_executed evidence is missing")
        self.task_state[RobotRole.SHOOTER] = RobotTaskState.RECEIPT_CONFIRMED
        return ActionDecision(True, "receipt_confirmed", "receipt confirmed")

    def request_fire_shot(self, safety: SafetySnapshot) -> ActionDecision:
        decision = self._fire_shot_decision(safety)
        if not decision.accepted:
            return decision
        self.task_state[RobotRole.SHOOTER] = RobotTaskState.SHOT_EXECUTED
        return decision

    def monitor_safety(self, safety: SafetySnapshot, require_teammate: bool = True) -> ActionDecision:
        """Move an active match into recovery when a critical runtime signal is lost."""
        if self.state != SupervisorState.ACTIVE:
            return ActionDecision(True, "safety_monitor", "runtime safety monitoring is inactive")

        failures = []
        if not safety.pose_valid:
            failures.append("pose_valid is false")
        if not safety.map_locked:
            failures.append("map_locked is false")
        if safety.pose_age_sec > self.pose_age_limit_sec:
            failures.append("pose is stale")
        if require_teammate and not safety.teammate_safe:
            failures.append("teammate safety is not confirmed")
        if not failures:
            return ActionDecision(True, "safety_monitor", "critical runtime signals are healthy")

        reason = "runtime safety degraded: " + "; ".join(failures)
        self.enter_recovery(reason)
        return ActionDecision(False, "safety_monitor", reason)

    def can_issue_action(self, action: str, safety: SafetySnapshot | None = None) -> ActionDecision:
        """Check a high-level action without mutating task state."""
        if action not in ACTION_NAMES:
            return ActionDecision(False, action, "unsupported action")
        if action == "EmergencyStop":
            return ActionDecision(True, action, "emergency stop is always allowed")
        if action == "AbortTask":
            if self.state in {SupervisorState.ACTIVE, SupervisorState.PAUSED, SupervisorState.RECOVERY, SupervisorState.FAULT}:
                return ActionDecision(True, action, "task abort is allowed")
            return ActionDecision(False, action, "task abort is not allowed in the current state")
        if self.state != SupervisorState.ACTIVE:
            return ActionDecision(False, action, "supervisor is not ACTIVE")
        prerequisites = {
            "NavigateToPose": True,
            "PrepareReceive": True,
            "CollectBall": self.task_state[RobotRole.SHOOTER] == RobotTaskState.PASS_EXECUTED,
            "PreparePass": self.task_state[RobotRole.SHOOTER] == RobotTaskState.RECEIVER_READY,
            "ExecutePass": self.task_state[RobotRole.SHOOTER] == RobotTaskState.PASS_ARMED,
            "PrepareShot": True,
        }
        if action == "FireShot":
            if safety is None:
                return ActionDecision(False, action, "safety snapshot is required")
            return self._fire_shot_decision(safety)
        if not prerequisites[action]:
            reasons = {
                "CollectBall": "pass_executed evidence is missing",
                "PreparePass": "receiver_ready evidence is missing",
                "ExecutePass": "pass_armed evidence is missing",
            }
            return ActionDecision(False, action, reasons[action])
        if action in self._LOCALIZATION_GATED_ACTIONS:
            if safety is None:
                return ActionDecision(False, action, "safety snapshot is required")
            localization_decision = self._localization_decision(action, safety)
            if not localization_decision.accepted:
                return localization_decision
        return ActionDecision(True, action, "action prerequisites are satisfied")

    def apply_action_success(self, action: str, evidence: dict[str, object] | None = None) -> ActionDecision:
        """Apply a trusted hardware feedback result to the competition state."""
        evidence = evidence or {}
        if action == "PrepareReceive":
            return self.set_receiver_ready()
        if action == "PreparePass":
            return self.arm_pass()
        if action == "ExecutePass":
            return self.confirm_pass_executed()
        if action == "CollectBall":
            if evidence.get("receipt_confirmed", True) is not True:
                return ActionDecision(False, action, "receipt confirmation evidence is missing")
            return self.confirm_receipt()
        if action == "FireShot":
            if self.state != SupervisorState.ACTIVE:
                return ActionDecision(False, action, "supervisor is not ACTIVE")
            self.task_state[RobotRole.SHOOTER] = RobotTaskState.SHOT_EXECUTED
            return ActionDecision(True, action, "shot execution feedback accepted")
        if action in {"NavigateToPose", "PrepareShot"}:
            return ActionDecision(True, action, "action execution feedback accepted")
        if action == "AbortTask":
            self.last_failure_reason = "task aborted by operator or hardware adapter"
            if self.state != SupervisorState.ESTOP:
                self.state = SupervisorState.RECOVERY
            return ActionDecision(True, action, "task aborted")
        if action == "EmergencyStop":
            self.emergency_stop("hardware adapter confirmed emergency stop")
            return ActionDecision(True, action, "emergency stop latched")
        return ActionDecision(False, action, "unsupported action feedback")

    def handle_action_failure(self, action: str, reason: str) -> ActionDecision:
        """Contain failed or expired actuator work before more tasks can be issued."""
        failure_reason = reason or f"action {action} did not complete"
        if action == "EmergencyStop":
            self.emergency_stop(failure_reason)
            return ActionDecision(False, action, failure_reason)
        if self.state in {SupervisorState.ACTIVE, SupervisorState.PAUSED, SupervisorState.FAULT}:
            self.enter_recovery(f"action failure: {action}: {failure_reason}")
        elif self.state == SupervisorState.RECOVERY:
            self.last_failure_reason = f"action failure: {action}: {failure_reason}"
        return ActionDecision(False, action, failure_reason)

    def _fire_shot_decision(self, safety: SafetySnapshot) -> ActionDecision:
        checks = (
            (safety.pose_valid, "pose_valid is false"),
            (safety.map_locked, "map_locked is false"),
            (safety.pose_age_sec <= self.pose_age_limit_sec, "pose is stale"),
            (safety.target_valid, "target is not valid"),
            (safety.target_age_sec <= self.target_age_limit_sec, "target observation is stale"),
            (safety.ball_present, "ball_present is false"),
            (safety.mechanism_healthy, "shooting mechanism is unhealthy"),
            (safety.teammate_safe, "teammate safety is not confirmed"),
        )
        for passed, reason in checks:
            if not passed:
                return ActionDecision(False, "FireShot", reason)
        return ActionDecision(True, "FireShot", "all shot interlocks are satisfied")

    def _localization_decision(self, action: str, safety: SafetySnapshot) -> ActionDecision:
        checks = (
            (safety.pose_valid, "pose_valid is false"),
            (safety.map_locked, "map_locked is false"),
            (safety.pose_age_sec <= self.pose_age_limit_sec, "pose is stale"),
        )
        for passed, reason in checks:
            if not passed:
                return ActionDecision(False, action, reason)
        return ActionDecision(True, action, "localization interlock is satisfied")

    def _require(self, allowed: set[SupervisorState]) -> None:
        if self.state not in allowed:
            raise RuntimeError(f"transition from {self.state.value} is not allowed")

    def _fail(self, reason: str) -> None:
        self.state = SupervisorState.FAULT
        self.last_failure_reason = reason
