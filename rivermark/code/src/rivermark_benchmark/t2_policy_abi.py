"""Public, bounded policy ABI for native City-Lite T2 control.

This module deliberately has no Isaac, Torch, evaluator-manifest, target, or
sensor-frame imports.  It defines the small public boundary that a native
runner must use before lowering a policy command through the calibrated CF2X
controller.  Native capture owns the actual sensor read, actuator write,
simulation step, and private evaluation; this module makes their causal
binding checkable without launching Kit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .citylite_scene import AGENT_COUNT
from .isaac_transfer import (
    ACTION_FIELDS,
    EXCLUDED_POLICY_INPUTS,
    STATE_FIELDS,
    FixedDecisionCadence,
    PhysicalState8D,
    StateOnlyTransferError,
    WorldCommandBounds,
    derive_physical_state_8d,
)
from .search_event_evaluator import EVENT_SUBMISSION_SCHEMA, parse_candidate_events

T2_POLICY_ABI_SCHEMA = "org.rivermark.native-t2-policy-abi.v1"
T2_DECISION_EVIDENCE_SCHEMA = "org.rivermark.native-t2-decision-evidence.v1"
T2_NATIVE_STEP_EVIDENCE_SCHEMA = "org.rivermark.native-t2-step-evidence.v2"
T2_EVENT_JOURNAL_SCHEMA = "org.rivermark.native-t2-event-journal.v1"
T2_INFORMATION_PROFILE = "state_only"
T2_CLAIM_BOUNDARY = "development_native_t2_canary_only"
_ACTION_SHAPE = (AGENT_COUNT, 4)
_STATE_SHAPE = (AGENT_COUNT, 8)
_HEX = frozenset("0123456789abcdef")


class T2PolicyAbiError(ValueError):
    """Raised when a T2 public-policy or causal-evidence boundary is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _finite_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise T2PolicyAbiError(f"{label} must be a finite numeric array") from exc
    if array.shape != shape:
        raise T2PolicyAbiError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise T2PolicyAbiError(f"{label} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _finite_vector(value: Any, *, length: int, label: str) -> tuple[float, ...]:
    array = _finite_array(value, shape=(length,), label=label)
    return tuple(float(item) for item in array.tolist())


def _positive_integer(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise T2PolicyAbiError(f"{label} must be an integer")
    result = int(value)
    if result < 0 or (not allow_zero and result == 0):
        comparator = "non-negative" if allow_zero else "positive"
        raise T2PolicyAbiError(f"{label} must be {comparator}")
    return result


def _finite_time_ns(value: Any, *, label: str) -> int:
    return _positive_integer(value, label=label, allow_zero=True)


def _sensor_observation_id(*, agent_id: int, capture_frame_index: int) -> str:
    return f"obs-a{agent_id:02d}-f{capture_frame_index:08d}"


@dataclass(frozen=True)
class T2PublicFleetObservation:
    """One public, pre-command state snapshot for all eight CF2X agents."""

    physics_step: int
    command_time_ns: int
    state: PhysicalState8D

    def __post_init__(self) -> None:
        step = _positive_integer(self.physics_step, label="physics_step", allow_zero=True)
        command_time_ns = _finite_time_ns(self.command_time_ns, label="command_time_ns")
        if not isinstance(self.state, PhysicalState8D):
            raise T2PolicyAbiError("state must be a PhysicalState8D")
        if self.state.agent_ids != tuple(range(AGENT_COUNT)):
            raise T2PolicyAbiError("T2 requires eight canonical agent rows")
        if self.state.values.shape != _STATE_SHAPE:
            raise T2PolicyAbiError("T2 state must have shape [8, 8]")
        object.__setattr__(self, "physics_step", step)
        object.__setattr__(self, "command_time_ns", command_time_ns)

    @classmethod
    def from_rigid_body_state(
        cls,
        *,
        physics_step: int,
        command_time_ns: int,
        position_w_m: Any,
        linear_velocity_w_mps: Any,
        quaternion_wxyz: Any,
        angular_velocity_b_radps: Any,
    ) -> T2PublicFleetObservation:
        """Construct the ABI snapshot from the pre-command native body arrays."""

        try:
            state = derive_physical_state_8d(
                position_w_m,
                linear_velocity_w_mps,
                quaternion_wxyz,
                angular_velocity_b_radps,
                agent_ids=tuple(range(AGENT_COUNT)),
            )
        except StateOnlyTransferError as exc:
            raise T2PolicyAbiError(str(exc)) from exc
        return cls(
            physics_step=physics_step,
            command_time_ns=command_time_ns,
            state=state,
        )

    def public_dict(self) -> dict[str, Any]:
        """Serialize only fields deliberately available to the policy."""

        return {
            "schema": T2_POLICY_ABI_SCHEMA,
            "claim_boundary": T2_CLAIM_BOUNDARY,
            "information_profile": T2_INFORMATION_PROFILE,
            "physics_step": self.physics_step,
            "command_time_ns": self.command_time_ns,
            "agent_ids": list(self.state.agent_ids),
            "state_fields": list(STATE_FIELDS),
            "state_8d": self.state.values.tolist(),
            "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.public_dict())


@dataclass(frozen=True)
class T2BoundedAction:
    """Raw and post-bound world-frame velocity/yaw actions for one decision."""

    raw_velocity_yaw_command: np.ndarray
    emitted_velocity_yaw_command: np.ndarray

    def __post_init__(self) -> None:
        raw = _finite_array(
            self.raw_velocity_yaw_command,
            shape=_ACTION_SHAPE,
            label="raw_velocity_yaw_command",
        )
        emitted = _finite_array(
            self.emitted_velocity_yaw_command,
            shape=_ACTION_SHAPE,
            label="emitted_velocity_yaw_command",
        )
        object.__setattr__(self, "raw_velocity_yaw_command", raw)
        object.__setattr__(self, "emitted_velocity_yaw_command", emitted)

    @classmethod
    def from_raw(
        cls, raw_velocity_yaw_command: Any, *, bounds: WorldCommandBounds
    ) -> T2BoundedAction:
        if not isinstance(bounds, WorldCommandBounds):
            raise T2PolicyAbiError("bounds must be a WorldCommandBounds")
        raw = _finite_array(
            raw_velocity_yaw_command,
            shape=_ACTION_SHAPE,
            label="raw_velocity_yaw_command",
        )
        try:
            velocity, yaw_rate = bounds.apply(raw[:, :3], raw[:, 3])
        except StateOnlyTransferError as exc:
            raise T2PolicyAbiError(str(exc)) from exc
        return cls(raw, np.concatenate((velocity, yaw_rate[:, None]), axis=1))

    def public_dict(self) -> dict[str, Any]:
        return {
            "action_fields": list(ACTION_FIELDS),
            "frame": "world",
            "raw_velocity_yaw_command": self.raw_velocity_yaw_command.tolist(),
            "emitted_velocity_yaw_command": self.emitted_velocity_yaw_command.tolist(),
        }


@dataclass(frozen=True)
class T2PolicyDecision:
    """A cadence-bound public observation and its bounded pre-step command."""

    decision_index: int
    observation: T2PublicFleetObservation
    action: T2BoundedAction
    bounds: WorldCommandBounds

    def __post_init__(self) -> None:
        index = _positive_integer(self.decision_index, label="decision_index", allow_zero=True)
        if not isinstance(self.observation, T2PublicFleetObservation):
            raise T2PolicyAbiError("observation must be a T2PublicFleetObservation")
        if not isinstance(self.action, T2BoundedAction):
            raise T2PolicyAbiError("action must be a T2BoundedAction")
        if not isinstance(self.bounds, WorldCommandBounds):
            raise T2PolicyAbiError("bounds must be a WorldCommandBounds")
        object.__setattr__(self, "decision_index", index)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": T2_DECISION_EVIDENCE_SCHEMA,
            "claim_boundary": T2_CLAIM_BOUNDARY,
            "command_before_step": True,
            "decision_index": self.decision_index,
            "observation_sha256": self.observation.sha256,
            "observation": self.observation.public_dict(),
            "action": self.action.public_dict(),
            "world_command_bounds": {
                "max_horizontal_speed_mps": self.bounds.max_horizontal_speed_mps,
                "max_vertical_speed_mps": self.bounds.max_vertical_speed_mps,
                "max_yaw_rate_rad_s": self.bounds.max_yaw_rate_rad_s,
            },
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.public_dict())


PolicyActionFn = Callable[[T2PublicFleetObservation], Any]


class T2PolicyRunner:
    """Run a caller-owned public policy at one integer physics cadence.

    The callable receives precisely one :class:`T2PublicFleetObservation` and
    must return a finite ``[8,4]`` world velocity/yaw array.  It does not get a
    task manifest, target truth, reward, evaluator result, seed, or future
    state.  Native code must lower ``emitted_velocity_yaw_command`` into thrust
    and append the resulting actuator/state evidence after the physical step.
    """

    def __init__(
        self,
        policy: PolicyActionFn,
        *,
        cadence: FixedDecisionCadence,
        bounds: WorldCommandBounds | None = None,
    ) -> None:
        if not callable(policy):
            raise T2PolicyAbiError("policy must be callable")
        if not isinstance(cadence, FixedDecisionCadence):
            raise T2PolicyAbiError("cadence must be a FixedDecisionCadence")
        self.policy = policy
        self.cadence = cadence
        self.bounds = bounds or WorldCommandBounds()
        if not isinstance(self.bounds, WorldCommandBounds):
            raise T2PolicyAbiError("bounds must be a WorldCommandBounds")

    def decide(self, observation: T2PublicFleetObservation) -> T2PolicyDecision:
        if not isinstance(observation, T2PublicFleetObservation):
            raise T2PolicyAbiError("observation must be a T2PublicFleetObservation")
        try:
            decision_index = self.cadence.decision_index(observation.physics_step)
        except StateOnlyTransferError as exc:
            raise T2PolicyAbiError(str(exc)) from exc
        raw_action = self.policy(observation)
        action = T2BoundedAction.from_raw(raw_action, bounds=self.bounds)
        return T2PolicyDecision(
            decision_index=decision_index,
            observation=observation,
            action=action,
            bounds=self.bounds,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": T2_POLICY_ABI_SCHEMA,
            "claim_boundary": T2_CLAIM_BOUNDARY,
            "information_profile": T2_INFORMATION_PROFILE,
            "policy_input_fields": list(STATE_FIELDS),
            "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
            "action_fields": list(ACTION_FIELDS),
            "action_frame": "world",
            "decision_cadence_physics_steps": self.cadence.every_physics_steps,
            "world_command_bounds": {
                "max_horizontal_speed_mps": self.bounds.max_horizontal_speed_mps,
                "max_vertical_speed_mps": self.bounds.max_vertical_speed_mps,
                "max_yaw_rate_rad_s": self.bounds.max_yaw_rate_rad_s,
            },
        }


@dataclass(frozen=True)
class T2PublicSensorObservation:
    """Public identity of one actual synchronized native sensor frame.

    This is intentionally distinct from a pre-command state snapshot. The
    native capture loop creates it only after reading the synchronized onboard
    sensor frame, then records the final stream hash separately in the capture
    receipt. A candidate event may cite this identity, never a policy-state
    placeholder.
    """

    agent_id: int
    capture_frame_index: int
    sensor_time_ns: int
    modality: str = "rgbd"

    def __post_init__(self) -> None:
        if isinstance(self.agent_id, bool) or not isinstance(self.agent_id, (int, np.integer)):
            raise T2PolicyAbiError("sensor observation agent_id must be an integer")
        agent = int(self.agent_id)
        if not 0 <= agent < AGENT_COUNT:
            raise T2PolicyAbiError("sensor observation agent_id is outside the T2 fleet")
        frame = _positive_integer(
            self.capture_frame_index,
            label="sensor observation capture_frame_index",
            allow_zero=True,
        )
        sensor_time_ns = _finite_time_ns(self.sensor_time_ns, label="sensor observation sensor_time_ns")
        if self.modality != "rgbd":
            raise T2PolicyAbiError("T2 candidate evidence currently requires rgbd modality")
        object.__setattr__(self, "agent_id", agent)
        object.__setattr__(self, "capture_frame_index", frame)
        object.__setattr__(self, "sensor_time_ns", sensor_time_ns)

    @property
    def observation_id(self) -> str:
        return _sensor_observation_id(
            agent_id=self.agent_id,
            capture_frame_index=self.capture_frame_index,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "agent_id": self.agent_id,
            "capture_frame_index": self.capture_frame_index,
            "sensor_time_ns": self.sensor_time_ns,
            "modality": self.modality,
        }


@dataclass(frozen=True)
class T2CandidateDetection:
    """A policy-side candidate localised from one public sensor observation.

    ``deduplication_key`` is an optional, capture-local key derived from the
    public semantic metadata.  It is intentionally not serialised into a
    candidate event: its only purpose is preventing repeated views of the
    same anonymous semantic instance from becoming repeated confirmations.
    """

    agent_id: int
    position_w_m: tuple[float, float, float]
    confidence: float
    deduplication_key: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.agent_id, bool) or not isinstance(self.agent_id, (int, np.integer)):
            raise T2PolicyAbiError("candidate agent_id must be an integer")
        agent = int(self.agent_id)
        if not 0 <= agent < AGENT_COUNT:
            raise T2PolicyAbiError("candidate agent_id is outside the T2 fleet")
        position = _finite_vector(self.position_w_m, length=3, label="candidate position_w_m")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise T2PolicyAbiError("candidate confidence must be finite in [0, 1]")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise T2PolicyAbiError("candidate confidence must be finite in [0, 1]")
        key = self.deduplication_key
        if key is not None and (
            not isinstance(key, str)
            or not key
            or len(key) > 128
            or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in key)
        ):
            raise T2PolicyAbiError("candidate deduplication_key must be a short ASCII token")
        object.__setattr__(self, "agent_id", agent)
        object.__setattr__(self, "position_w_m", position)
        object.__setattr__(self, "confidence", confidence)


class T2CandidateEventJournal:
    """Bind public candidate detections to public observation IDs and times."""

    def __init__(self, *, episode_id: str, event_time_origin_ns: int = 0) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise T2PolicyAbiError("episode_id must be a non-empty string")
        origin = _finite_time_ns(
            event_time_origin_ns, label="event_time_origin_ns"
        )
        self.episode_id = episode_id
        self.event_time_origin_ns = origin
        self._events: list[dict[str, Any]] = []
        self._next_event_index = 0

    def append(
        self,
        observation: T2PublicSensorObservation,
        detections: Sequence[T2CandidateDetection],
    ) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(observation, T2PublicSensorObservation):
            raise T2PolicyAbiError("observation must be a T2PublicSensorObservation")
        if observation.sensor_time_ns < self.event_time_origin_ns:
            raise T2PolicyAbiError("sensor observation precedes the T2 event time origin")
        if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
            raise T2PolicyAbiError("detections must be a sequence")
        rows: list[dict[str, Any]] = []
        for detection in detections:
            if not isinstance(detection, T2CandidateDetection):
                raise T2PolicyAbiError("detections must contain T2CandidateDetection values")
            if detection.agent_id != observation.agent_id:
                raise T2PolicyAbiError("candidate agent_id must match its source sensor observation")
            event = {
                "event_id": f"evt-{self._next_event_index:08d}",
                "timestamp_s": (
                    observation.sensor_time_ns - self.event_time_origin_ns
                )
                / 1_000_000_000.0,
                "agent_id": detection.agent_id,
                "source_observation_id": observation.observation_id,
                "position_w_m": list(detection.position_w_m),
                "confidence": detection.confidence,
            }
            self._next_event_index += 1
            rows.append(event)
        self._events.extend(rows)
        return tuple(dict(row) for row in rows)

    def submission(self) -> dict[str, Any]:
        submission = {
            "schema": EVENT_SUBMISSION_SCHEMA,
            "episode_id": self.episode_id,
            "events": [dict(event) for event in self._events],
        }
        try:
            parse_candidate_events(
                submission,
                expected_episode_id=self.episode_id,
                agent_count=AGENT_COUNT,
            )
        except ValueError as exc:
            raise T2PolicyAbiError(f"candidate event journal is invalid: {exc}") from exc
        return submission

    def public_dict(self) -> dict[str, Any]:
        submission = self.submission()
        return {
            "schema": T2_EVENT_JOURNAL_SCHEMA,
            "claim_boundary": T2_CLAIM_BOUNDARY,
            "event_time_origin_ns": self.event_time_origin_ns,
            "submission": submission,
            "submission_sha256": _sha256(submission),
        }


@dataclass(frozen=True)
class T2NativeStepEvidence:
    """Bind a pre-step decision to native applied thrust and post-step state."""

    decision: T2PolicyDecision
    applied_physics_step: int
    physical_command_time_ns: int
    effective_time_ns: int
    requested_thrust_n: np.ndarray
    applied_thrust_n: np.ndarray
    applied_wrench_body: np.ndarray
    post_step_state_8d: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.decision, T2PolicyDecision):
            raise T2PolicyAbiError("decision must be a T2PolicyDecision")
        applied_step = _positive_integer(
            self.applied_physics_step,
            label="applied_physics_step",
            allow_zero=True,
        )
        if applied_step < self.decision.observation.physics_step:
            raise T2PolicyAbiError(
                "applied_physics_step must not precede the decision physics_step"
            )
        physical_command_time = _finite_time_ns(
            self.physical_command_time_ns, label="physical_command_time_ns"
        )
        if physical_command_time < self.decision.observation.command_time_ns:
            raise T2PolicyAbiError(
                "physical_command_time_ns must not precede decision_command_time_ns"
            )
        effective_time = _finite_time_ns(self.effective_time_ns, label="effective_time_ns")
        if effective_time <= physical_command_time:
            raise T2PolicyAbiError(
                "effective_time_ns must be after physical_command_time_ns"
            )
        requested = _finite_array(
            self.requested_thrust_n,
            shape=(AGENT_COUNT, 4),
            label="requested_thrust_n",
        )
        applied = _finite_array(
            self.applied_thrust_n,
            shape=(AGENT_COUNT, 4),
            label="applied_thrust_n",
        )
        wrench = _finite_array(
            self.applied_wrench_body,
            shape=(AGENT_COUNT, 6),
            label="applied_wrench_body",
        )
        post_state = _finite_array(
            self.post_step_state_8d,
            shape=_STATE_SHAPE,
            label="post_step_state_8d",
        )
        if np.any(requested < 0.0) or np.any(applied < 0.0):
            raise T2PolicyAbiError("native thrust evidence must be non-negative")
        object.__setattr__(self, "applied_physics_step", applied_step)
        object.__setattr__(self, "physical_command_time_ns", physical_command_time)
        object.__setattr__(self, "effective_time_ns", effective_time)
        object.__setattr__(self, "requested_thrust_n", requested)
        object.__setattr__(self, "applied_thrust_n", applied)
        object.__setattr__(self, "applied_wrench_body", wrench)
        object.__setattr__(self, "post_step_state_8d", post_state)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": T2_NATIVE_STEP_EVIDENCE_SCHEMA,
            "command_before_step": True,
            "decision_sha256": self.decision.sha256,
            "decision_physics_step": self.decision.observation.physics_step,
            "applied_physics_step": self.applied_physics_step,
            "decision_command_time_ns": self.decision.observation.command_time_ns,
            "physical_command_time_ns": self.physical_command_time_ns,
            "effective_time_ns": self.effective_time_ns,
            "requested_thrust_n": self.requested_thrust_n.tolist(),
            "applied_thrust_n": self.applied_thrust_n.tolist(),
            "applied_wrench_body": self.applied_wrench_body.tolist(),
            "post_step_state_8d": self.post_step_state_8d.tolist(),
        }


__all__ = [
    "T2_CLAIM_BOUNDARY",
    "T2_DECISION_EVIDENCE_SCHEMA",
    "T2_NATIVE_STEP_EVIDENCE_SCHEMA",
    "T2_EVENT_JOURNAL_SCHEMA",
    "T2_INFORMATION_PROFILE",
    "T2_POLICY_ABI_SCHEMA",
    "T2BoundedAction",
    "T2CandidateDetection",
    "T2CandidateEventJournal",
    "T2NativeStepEvidence",
    "T2PolicyAbiError",
    "T2PolicyDecision",
    "T2PolicyRunner",
    "T2PublicFleetObservation",
    "T2PublicSensorObservation",
]
