"""Versioned high-level action and feedback contracts for hardware adapters."""

from __future__ import annotations

from dataclasses import dataclass, field


ACTION_NAMES = (
    "NavigateToPose",
    "PrepareReceive",
    "CollectBall",
    "PreparePass",
    "ExecutePass",
    "PrepareShot",
    "FireShot",
    "AbortTask",
    "EmergencyStop",
)

FEEDBACK_STATES = ("accepted", "running", "succeeded", "failed", "cancelled")


@dataclass(frozen=True)
class ActionRequest:
    protocol_version: int
    action_id: str
    task_id: str
    action: str
    sender_id: str
    sequence: int
    created_at_ns: int
    expires_at_ns: int
    parameters: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.action_id or not self.task_id or not self.sender_id:
            raise ValueError("action_id, task_id, and sender_id are required")
        if self.action not in ACTION_NAMES:
            raise ValueError(f"unsupported action: {self.action}")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.expires_at_ns <= self.created_at_ns:
            raise ValueError("expires_at_ns must be after created_at_ns")

    def is_fresh(self, now_ns: int) -> bool:
        self.validate()
        return self.created_at_ns <= now_ns <= self.expires_at_ns


@dataclass(frozen=True)
class ActionFeedback:
    protocol_version: int
    action_id: str
    task_id: str
    action: str
    sender_id: str
    state: str
    created_at_ns: int
    reason: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.action_id or not self.task_id or not self.sender_id:
            raise ValueError("action_id, task_id, and sender_id are required")
        if self.action not in ACTION_NAMES:
            raise ValueError(f"unsupported action: {self.action}")
        if self.state not in FEEDBACK_STATES:
            raise ValueError(f"unsupported feedback state: {self.state}")


class ActionDeduplicator:
    """Reject repeated action sequences from the same sender and task."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], int] = {}
        self.duplicate_count = 0

    def accept(self, request: ActionRequest, now_ns: int) -> bool:
        if not request.is_fresh(now_ns):
            return False
        key = (request.sender_id, request.task_id)
        latest = self._latest.get(key)
        if latest is not None and request.sequence <= latest:
            self.duplicate_count += 1
            return False
        self._latest[key] = request.sequence
        return True
