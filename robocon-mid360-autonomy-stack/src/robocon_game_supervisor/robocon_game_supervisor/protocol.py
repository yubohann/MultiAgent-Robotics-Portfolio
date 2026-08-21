"""Versioned, expiring, idempotent dual-robot message contracts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class MessageEnvelope:
    protocol_version: int
    message_type: str
    task_id: str
    sender_id: str
    sequence: int
    created_at_ns: int
    expires_at_ns: int
    payload: dict[str, object]

    def validate(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.message_type or not self.task_id or not self.sender_id:
            raise ValueError("message_type, task_id, and sender_id are required")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.expires_at_ns <= self.created_at_ns:
            raise ValueError("expires_at_ns must be after created_at_ns")

    def is_fresh(self, now_ns: int) -> bool:
        self.validate()
        return self.created_at_ns <= now_ns <= self.expires_at_ns


class Deduplicator:
    """Accept each sender/task/message sequence once and reject old sequences."""

    def __init__(self) -> None:
        self._latest: dict[tuple[str, str, str], int] = {}
        self.duplicate_count = 0
        self.stale_count = 0

    def accept(self, envelope: MessageEnvelope, now_ns: int) -> bool:
        if not envelope.is_fresh(now_ns):
            self.stale_count += 1
            return False
        key = (envelope.sender_id, envelope.task_id, envelope.message_type)
        latest = self._latest.get(key)
        if latest is not None and envelope.sequence <= latest:
            self.duplicate_count += 1
            return False
        self._latest[key] = envelope.sequence
        return True


def envelope_to_json(envelope: MessageEnvelope) -> str:
    """Serialize a transport-neutral message envelope."""
    envelope.validate()
    return json.dumps(
        {
            "protocol_version": envelope.protocol_version,
            "message_type": envelope.message_type,
            "task_id": envelope.task_id,
            "sender_id": envelope.sender_id,
            "sequence": envelope.sequence,
            "created_at_ns": envelope.created_at_ns,
            "expires_at_ns": envelope.expires_at_ns,
            "payload": envelope.payload,
        },
        separators=(",", ":"),
    )


def envelope_from_json(value: str | dict[str, object]) -> MessageEnvelope:
    """Parse and validate a JSON or already-decoded envelope."""
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict):
        raise ValueError("message envelope must be a JSON object")
    envelope = MessageEnvelope(
        protocol_version=int(payload["protocol_version"]),
        message_type=str(payload["message_type"]),
        task_id=str(payload["task_id"]),
        sender_id=str(payload["sender_id"]),
        sequence=int(payload["sequence"]),
        created_at_ns=int(payload["created_at_ns"]),
        expires_at_ns=int(payload["expires_at_ns"]),
        payload=dict(payload.get("payload", {})),
    )
    envelope.validate()
    return envelope


class TeamLink:
    """Transport-neutral freshness, deduplication, and ACK behavior."""

    def __init__(
        self,
        local_sender_id: str,
        ack_ttl_sec: float = 1.0,
        expected_task_id: str | None = None,
    ) -> None:
        if not local_sender_id:
            raise ValueError("local_sender_id is required")
        if ack_ttl_sec <= 0.0:
            raise ValueError("ack_ttl_sec must be positive")
        self.local_sender_id = local_sender_id
        self.ack_ttl_sec = ack_ttl_sec
        self.expected_task_id = expected_task_id
        self._ack_sequence = 0
        self.deduplicator = Deduplicator()
        self.last_heartbeat_ns: int | None = None
        self.task_mismatch_count = 0

    def receive(self, raw: str | dict[str, object], now_ns: int | None = None) -> tuple[MessageEnvelope, bool, str]:
        now_ns = time.time_ns() if now_ns is None else now_ns
        envelope = envelope_from_json(raw)
        if self.expected_task_id and envelope.task_id != self.expected_task_id:
            self.task_mismatch_count += 1
            accepted = False
            reason = "task_mismatch"
        else:
            accepted = self.deduplicator.accept(envelope, now_ns)
            reason = "accepted" if accepted else "stale_or_duplicate"
        if accepted and envelope.message_type == "heartbeat":
            self.last_heartbeat_ns = now_ns
        self._ack_sequence += 1
        ack = MessageEnvelope(
            protocol_version=envelope.protocol_version,
            message_type="ack",
            task_id=envelope.task_id,
            sender_id=self.local_sender_id,
            sequence=self._ack_sequence,
            created_at_ns=now_ns,
            expires_at_ns=now_ns + int(self.ack_ttl_sec * 1_000_000_000),
            payload={
                "ack_sequence": envelope.sequence,
                "ack_type": envelope.message_type,
                "accepted": accepted,
                "reason": reason,
            },
        )
        return ack, accepted, reason
