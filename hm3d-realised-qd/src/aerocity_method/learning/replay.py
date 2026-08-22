"""Immutable transition validation and deterministic bounded replay."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generic, TypeVar

from aerocity_method.contracts.io import canonical_sha256, finite_number, require_sha256
from aerocity_method.contracts.models import ABI_VERSION, FragmentReplayRecord


def _vector(values: object, name: str) -> tuple[float, ...]:
    resolved = tuple(finite_number(value, name) for value in tuple(values))  # type: ignore[arg-type]
    if not resolved:
        raise ValueError(f"{name} must not be empty")
    return resolved


def _candidate_matrix(values: object, name: str) -> tuple[tuple[float, ...], ...]:
    rows = tuple(_vector(row, name) for row in tuple(values))  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"{name} must contain candidates")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} candidate feature dimensions must match")
    return rows


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    context: tuple[float, ...]
    candidates: tuple[tuple[float, ...], ...]
    legal_mask: tuple[bool, ...]
    action: int
    reward: float
    cost: float
    preference: tuple[float, ...]
    behavior_features: tuple[float, ...]
    next_context: tuple[float, ...]
    next_candidates: tuple[tuple[float, ...], ...]
    next_legal_mask: tuple[bool, ...]
    next_preference: tuple[float, ...]
    done: bool
    duration: float
    outcome_hash: str
    terminated: bool | None = None
    truncated: bool = False
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ABI_VERSION:
            raise ValueError("unsupported transition schema version")
        context = _vector(self.context, "context")
        next_context = _vector(self.next_context, "next_context")
        if len(context) != len(next_context):
            raise ValueError("context and next_context dimensions must match")
        candidates = _candidate_matrix(self.candidates, "candidates")
        next_candidates = _candidate_matrix(self.next_candidates, "next_candidates")
        if len(candidates[0]) != len(next_candidates[0]):
            raise ValueError("candidate feature dimensions must remain stable")
        legal = tuple(self.legal_mask)
        next_legal = tuple(self.next_legal_mask)
        if len(legal) != len(candidates) or len(next_legal) != len(next_candidates):
            raise ValueError("legal masks must match candidate counts")
        if any(not isinstance(value, bool) for value in legal + next_legal):
            raise ValueError("legal masks must contain booleans")
        if not any(legal) or not any(next_legal):
            raise ValueError("each candidate set must have at least one legal action")
        if not isinstance(self.action, int) or isinstance(self.action, bool):
            raise ValueError("action must be an integer")
        if self.action < 0 or self.action >= len(candidates) or not legal[self.action]:
            raise ValueError("action must select a legal candidate")
        preference = tuple(finite_number(value, "preference") for value in self.preference)
        next_preference = tuple(
            finite_number(value, "next_preference") for value in self.next_preference
        )
        if len(preference) != len(next_preference):
            raise ValueError("preference dimensions must remain stable")
        behavior = tuple(
            finite_number(value, "behavior_features") for value in self.behavior_features
        )
        reward = finite_number(self.reward, "reward")
        cost = finite_number(self.cost, "cost")
        duration = finite_number(self.duration, "duration")
        if cost < 0.0 or duration < 0.0:
            raise ValueError("cost and duration must be non-negative")
        # ``done`` remains the Bellman-bootstrap mask for the fixed-horizon
        # candidate-selection task.  Preserve the cause separately so outcome
        # records never turn a normal time-budget truncation into a safety
        # terminal failure.
        terminated = self.done if self.terminated is None else self.terminated
        if not isinstance(self.done, bool) or not isinstance(terminated, bool):
            raise ValueError("done and terminated must be booleans")
        if not isinstance(self.truncated, bool):
            raise ValueError("truncated must be boolean")
        if terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if self.done != (terminated or self.truncated):
            raise ValueError("done must equal terminated or truncated")
        require_sha256(self.outcome_hash, "outcome_hash")
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "next_context", next_context)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "next_candidates", next_candidates)
        object.__setattr__(self, "legal_mask", legal)
        object.__setattr__(self, "next_legal_mask", next_legal)
        object.__setattr__(self, "preference", preference)
        object.__setattr__(self, "next_preference", next_preference)
        object.__setattr__(self, "behavior_features", behavior)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "terminated", terminated)


@dataclass(frozen=True, slots=True)
class PaddedCandidateBatch:
    contexts: tuple[tuple[float, ...], ...]
    candidates: tuple[tuple[tuple[float, ...], ...], ...]
    legal_masks: tuple[tuple[bool, ...], ...]
    counts: tuple[int, ...]


def pad_candidate_batch(
    contexts: tuple[tuple[float, ...], ...] | list[tuple[float, ...]],
    candidates: tuple[tuple[tuple[float, ...], ...], ...] | list[tuple[tuple[float, ...], ...]],
    legal_masks: tuple[tuple[bool, ...], ...] | list[tuple[bool, ...]],
) -> PaddedCandidateBatch:
    context_rows = tuple(_vector(row, "contexts") for row in contexts)
    candidate_rows = tuple(_candidate_matrix(row, "candidates") for row in candidates)
    mask_rows = tuple(tuple(row) for row in legal_masks)
    if (
        not context_rows
        or len(context_rows) != len(candidate_rows)
        or len(context_rows) != len(mask_rows)
    ):
        raise ValueError("batch inputs must have the same non-zero batch size")
    context_width = len(context_rows[0])
    candidate_width = len(candidate_rows[0][0])
    if any(len(row) != context_width for row in context_rows):
        raise ValueError("context dimensions must match within a batch")
    if any(len(candidate) != candidate_width for row in candidate_rows for candidate in row):
        raise ValueError("candidate dimensions must match within a batch")
    for row, mask in zip(candidate_rows, mask_rows, strict=True):
        if (
            len(row) != len(mask)
            or not any(mask)
            or any(not isinstance(value, bool) for value in mask)
        ):
            raise ValueError(
                "each legal mask must match its candidate set and contain a legal action"
            )
    maximum = max(len(row) for row in candidate_rows)
    zero = tuple(0.0 for _ in range(candidate_width))
    padded_candidates = tuple(
        row + tuple(zero for _ in range(maximum - len(row))) for row in candidate_rows
    )
    padded_masks = tuple(
        mask + tuple(False for _ in range(maximum - len(mask))) for mask in mask_rows
    )
    return PaddedCandidateBatch(
        contexts=context_rows,
        candidates=padded_candidates,
        legal_masks=padded_masks,
        counts=tuple(len(row) for row in candidate_rows),
    )


T = TypeVar("T")


class ReplayBuffer(Generic[T]):
    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._items: list[T] = []
        self._cursor = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: T) -> None:
        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._cursor] = item
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, count: int) -> tuple[T, ...]:
        if not isinstance(count, int) or count < 1 or count > len(self._items):
            raise ValueError("sample count must be within current replay size")
        return tuple(self._rng.sample(self._items, count))

    def snapshot(self) -> tuple[T, ...]:
        return tuple(self._items)

    def rng_state(self) -> object:
        return self._rng.getstate()

    def restore_rng_state(self, state: object) -> None:
        self._rng.setstate(state)


class FragmentReplayBuffer(ReplayBuffer[FragmentReplayRecord]):
    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        super().__init__(capacity, seed=seed)
        self._keys: set[tuple[str, str]] = set()

    def add(self, item: FragmentReplayRecord) -> None:
        key = (item.instance_fragment_id, item.outcome_hash)
        if key in self._keys:
            return
        if len(self._items) == self.capacity:
            evicted = self._items[self._cursor]
            self._keys.remove((evicted.instance_fragment_id, evicted.outcome_hash))
        super().add(item)
        self._keys.add(key)

    def state_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ABI_VERSION,
            "capacity": self.capacity,
            "cursor": self._cursor,
            "items": [item.to_dict() for item in self._items],
            "rng_state": self._rng.getstate(),
        }
        payload["checkpoint_hash"] = canonical_sha256(payload)
        return payload

    @staticmethod
    def _tuplify(value: object) -> object:
        if isinstance(value, list):
            return tuple(FragmentReplayBuffer._tuplify(child) for child in value)
        return value

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> FragmentReplayBuffer:
        if state.get("schema_version") != ABI_VERSION:
            raise ValueError("fragment replay checkpoint schema mismatch")
        supplied_hash = state.get("checkpoint_hash")
        unsigned = {key: value for key, value in state.items() if key != "checkpoint_hash"}
        if canonical_sha256(unsigned) != supplied_hash:
            raise ValueError("fragment replay checkpoint content hash mismatch")
        capacity = state.get("capacity")
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("fragment replay checkpoint capacity is invalid")
        restored = cls(capacity)
        items = state.get("items")
        if not isinstance(items, list) or len(items) > capacity:
            raise ValueError("fragment replay checkpoint items are invalid")
        for payload in items:
            if not isinstance(payload, dict):
                raise ValueError("fragment replay checkpoint item is invalid")
            restored.add(FragmentReplayRecord.from_dict(payload))
        cursor = state.get("cursor")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or not 0 <= cursor < capacity:
            raise ValueError("fragment replay checkpoint cursor is invalid")
        if cursor != len(items) % capacity:
            raise ValueError("fragment replay checkpoint cursor is inconsistent")
        rng_state = cls._tuplify(state.get("rng_state"))
        restored._rng.setstate(rng_state)  # type: ignore[arg-type]
        restored._cursor = cursor
        return restored
