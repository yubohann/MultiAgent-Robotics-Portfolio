"""Pure Python competition contracts used by ROS and hardware adapters."""

from .protocol import Deduplicator, MessageEnvelope
from .supervisor import (
    ActionDecision,
    GameSupervisor,
    RobotRole,
    RobotTaskState,
    SafetySnapshot,
    SupervisorState,
)

__all__ = [
    "ActionDecision",
    "Deduplicator",
    "GameSupervisor",
    "MessageEnvelope",
    "RobotRole",
    "RobotTaskState",
    "SafetySnapshot",
    "SupervisorState",
]
