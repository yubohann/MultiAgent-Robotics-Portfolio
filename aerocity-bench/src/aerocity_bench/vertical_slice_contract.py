"""Fail-closed integrity checks for the internal CF2X vertical-slice fixture.

The vertical slice is deliberately a non-formal, single-UAV preflight fixture.
It may consume evaluator-private witnesses, so this module validates private
evidence locally and returns only hashes and categorical status.  It must never
be used to promote a run to a formal L1 score.
"""

from __future__ import annotations

import math
from typing import Any

from .canonical import content_hash
from .contracts import ACTION_KINDS
from .errors import ValidationError
from .geometry import distance

PRIVATE_VERTICAL_SLICE_SCHEMA = "org.aerocity.bench.quadrotor-l1-vertical-slice-private.v1"
PRIVATE_VERTICAL_SLICE_SCOPE = "quadrotor_internal_vertical_slice_private_fixture"
_EXECUTION_RECEIPT_SCHEMA_V2 = "org.aerocity.bench.execution-receipt.v2"
_EXECUTION_RECEIPT_SCHEMA_V3 = "org.aerocity.bench.execution-receipt.v3"
_OBSERVATION_RECEIPT_FIELDS = frozenset(
    {"observation_id", "drone_id", "timestamp_s", "accepted", "reason", "receipt_hash"}
)
_EXECUTION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "episode_id",
        "drone_id",
        "action_sequence",
        "task_time_start_s",
        "task_time_end_s",
        "planning_latency_s",
        "action_requested",
        "action_executed",
        "status",
        "distance_m",
        "energy_used_j",
        "minimum_clearance_m",
        "collision",
        "out_of_bounds",
        "safety_intervention",
        "deadline_miss",
        "execution_level",
        "action_packet_hash",
        "source_observation_id",
        "source_observation_hash",
        "state_before_hash",
        "state_after_hash",
        "previous_receipt_hash",
        "confirmation_ids",
        "receipt_hash",
    }
)
_EXECUTION_RECEIPT_FIELDS_V3 = _EXECUTION_RECEIPT_FIELDS | {"planner_invoked"}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"vertical slice {field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValidationError(f"vertical slice {field} must be finite and non-negative")
    return numeric


def _position(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValidationError(f"vertical slice {field} must be a 3-D position")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValidationError(f"vertical slice {field} contains a non-finite coordinate")
    return result  # type: ignore[return-value]


def _require_hash(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValidationError(f"vertical slice {field} is not a SHA-256 digest")
    return str(value)


def _validate_observation_receipt(
    value: object,
    *,
    observation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _OBSERVATION_RECEIPT_FIELDS:
        raise ValidationError("vertical slice observation receipt fields differ")
    if (
        value.get("observation_id") != observation.get("observation_id")
        or value.get("drone_id") != observation.get("drone_id")
        or float(value.get("timestamp_s", math.nan))
        != float(observation.get("timestamp_s", math.nan))
        or not isinstance(value.get("accepted"), bool)
        or not isinstance(value.get("reason"), str)
        or not value["reason"]
    ):
        raise ValidationError("vertical slice observation receipt does not bind its observation")
    payload = {key: value[key] for key in value if key != "receipt_hash"}
    if content_hash(payload) != value.get("receipt_hash"):
        raise ValidationError("vertical slice observation receipt hash mismatch")
    return value


def _validate_receipt(
    receipt: object,
    *,
    action: dict[str, Any],
    observation: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    expected_episode_id: str,
    expected_drone_id: str,
    sequence: int,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValidationError("vertical slice execution receipt fields differ")
    schema = receipt.get("schema")
    expected_fields = (
        _EXECUTION_RECEIPT_FIELDS_V3
        if schema == _EXECUTION_RECEIPT_SCHEMA_V3
        else _EXECUTION_RECEIPT_FIELDS
    )
    if set(receipt) != expected_fields:
        raise ValidationError("vertical slice execution receipt fields differ")
    payload = dict(receipt)
    expected_hash = _require_hash(payload.pop("receipt_hash", None), "execution receipt hash")
    if content_hash(payload) != expected_hash:
        raise ValidationError("vertical slice execution receipt hash mismatch")
    if schema not in {_EXECUTION_RECEIPT_SCHEMA_V2, _EXECUTION_RECEIPT_SCHEMA_V3}:
        raise ValidationError("vertical slice execution receipt schema is invalid")
    if schema == _EXECUTION_RECEIPT_SCHEMA_V3 and not isinstance(
        receipt.get("planner_invoked"), bool
    ):
        raise ValidationError("vertical slice planner invocation flag is invalid")
    if receipt.get("execution_level") != "L1":
        raise ValidationError("vertical slice execution receipt is not L1")
    if (
        receipt.get("episode_id") != expected_episode_id
        or receipt.get("drone_id") != expected_drone_id
        or receipt.get("action_sequence") != sequence
    ):
        raise ValidationError("vertical slice execution receipt identity is inconsistent")
    if action.get("schema") != "org.aerocity.bench.action-packet.v1":
        raise ValidationError("vertical slice trace action schema is invalid")
    if observation.get("schema") != "org.aerocity.bench.observation-packet.v2":
        raise ValidationError("vertical slice trace observation schema is invalid")
    if (
        action.get("episode_id") != expected_episode_id
        or action.get("drone_id") != expected_drone_id
        or action.get("sequence") != sequence
        or observation.get("episode_id") != expected_episode_id
        or observation.get("drone_id") != expected_drone_id
        or observation.get("sequence") != sequence
    ):
        raise ValidationError("vertical slice trace packet identity is inconsistent")
    if action.get("kind") not in ACTION_KINDS or action.get("kind") != receipt.get(
        "action_requested"
    ):
        raise ValidationError("vertical slice requested action is invalid")
    if receipt.get("action_executed") != action.get("kind"):
        raise ValidationError("vertical slice action was not executed as recorded")
    if (
        receipt.get("source_observation_id") != observation.get("observation_id")
        or receipt.get("action_packet_hash") != content_hash(action)
        or receipt.get("source_observation_hash") != content_hash(observation)
        or receipt.get("state_before_hash") != content_hash(state_before)
        or receipt.get("state_after_hash") != content_hash(state_after)
    ):
        raise ValidationError("vertical slice execution receipt provenance is inconsistent")
    if (
        abs(
            float(action.get("issued_at_s", math.nan))
            - float(observation.get("timestamp_s", math.nan))
        )
        > 1.0e-9
    ):
        raise ValidationError("vertical slice action is not bound to its source observation time")
    if action.get("kind") == "OBSERVE":
        if action.get("source_observation_id") != observation.get("observation_id"):
            raise ValidationError("vertical slice OBSERVE lacks its source observation binding")
    elif action.get("source_observation_id") is not None:
        raise ValidationError("vertical slice non-OBSERVE action binds a private observation")
    for field in (
        "task_time_start_s",
        "task_time_end_s",
        "planning_latency_s",
        "distance_m",
        "energy_used_j",
    ):
        _finite_nonnegative(receipt.get(field), field)
    if float(receipt["task_time_end_s"]) < float(receipt["task_time_start_s"]):
        raise ValidationError("vertical slice execution receipt time runs backwards")
    clearance = receipt.get("minimum_clearance_m")
    if clearance is not None:
        _finite_nonnegative(clearance, "minimum_clearance_m")
    for field in (
        "collision",
        "out_of_bounds",
        "safety_intervention",
        "deadline_miss",
    ):
        if not isinstance(receipt.get(field), bool):
            raise ValidationError(f"vertical slice {field} flag is invalid")
    confirmation_ids = receipt.get("confirmation_ids")
    if (
        not isinstance(confirmation_ids, list)
        or any(not isinstance(item, str) or not item for item in confirmation_ids)
        or len(set(confirmation_ids)) != len(confirmation_ids)
    ):
        raise ValidationError("vertical slice execution receipt confirmation IDs are invalid")
    if previous is None:
        if (
            receipt.get("previous_receipt_hash") is not None
            or abs(float(receipt["task_time_start_s"])) > 1.0e-9
        ):
            raise ValidationError("vertical slice receipt chain does not begin at zero")
    else:
        if (
            receipt.get("previous_receipt_hash") != previous.get("receipt_hash")
            or receipt.get("state_before_hash") != previous.get("state_after_hash")
            or abs(float(receipt["task_time_start_s"]) - float(previous["task_time_end_s"]))
            > 1.0e-9
        ):
            raise ValidationError("vertical slice execution receipt chain is discontinuous")
    return receipt


def validate_private_vertical_slice_report(private: dict[str, Any]) -> dict[str, Any]:
    """Validate one private single-UAV vertical-slice report without promoting it.

    The function deliberately checks more than a report hash: each trace record
    is tied to a receipt, evaluator output, and the return closure.  A caller
    may persist the returned hashes in a public summary, but must not persist the
    private report itself with a benchmark release.
    """

    if private.get("schema") != PRIVATE_VERTICAL_SLICE_SCHEMA:
        raise ValidationError("unexpected private vertical slice schema")
    if private.get("formal_score_eligible") is not False:
        raise ValidationError("vertical slice must not claim formal-score eligibility")
    if private.get("evidence_scope") != PRIVATE_VERTICAL_SLICE_SCOPE:
        raise ValidationError("unexpected private vertical slice evidence scope")
    bindings = private.get("input_bindings")
    fixture = private.get("private_fixture")
    execution = private.get("execution")
    final = private.get("final")
    closure_contract = private.get("closure_contract")
    trace = private.get("trace_private")
    receipts = private.get("execution_receipts")
    observation_receipts = private.get("observation_receipts")
    confirmations = private.get("confirmation_receipts")
    failures = private.get("failure_records")
    audit = private.get("evaluator_private_audit")
    if not all(
        isinstance(value, dict)
        for value in (bindings, fixture, execution, final, closure_contract, audit)
    ):
        raise ValidationError("vertical slice private report has an invalid record section")
    if not all(
        isinstance(value, list)
        for value in (trace, receipts, observation_receipts, confirmations, failures)
    ):
        raise ValidationError("vertical slice private report has an invalid list section")
    episode_id = bindings.get("episode_id")
    drone_id = fixture.get("start_drone_id")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or not isinstance(drone_id, str)
        or not drone_id
    ):
        raise ValidationError("vertical slice private report lacks an episode/drone binding")
    if not _is_sha256(private.get("private_fixture_commitment")):
        raise ValidationError("vertical slice private fixture commitment is invalid")
    if len(trace) != len(receipts) or len(trace) != int(execution.get("control_action_count", -1)):
        raise ValidationError("vertical slice trace and execution receipt counts differ")
    if len(observation_receipts) != sum(
        isinstance(item, dict)
        and isinstance(item.get("action"), dict)
        and item["action"].get("kind") == "OBSERVE"
        for item in trace
    ):
        raise ValidationError(
            "vertical slice observation receipt count differs from OBSERVE actions"
        )

    accepted_observations: dict[str, dict[str, Any]] = {}
    receipt_confirmation_ids: list[str] = []
    previous: dict[str, Any] | None = None
    for sequence, (entry, receipt) in enumerate(zip(trace, receipts, strict=True)):
        if not isinstance(entry, dict):
            raise ValidationError("vertical slice trace entry is invalid")
        action = entry.get("action")
        observation = entry.get("source_observation")
        state_before = entry.get("state_before")
        state_after = entry.get("state_after")
        if not all(
            isinstance(value, dict) for value in (action, observation, state_before, state_after)
        ):
            raise ValidationError("vertical slice trace entry omits a bound packet or state")
        validated = _validate_receipt(
            receipt,
            action=action,
            observation=observation,
            state_before=state_before,
            state_after=state_after,
            expected_episode_id=episode_id,
            expected_drone_id=drone_id,
            sequence=sequence,
            previous=previous,
        )
        if entry.get("execution_receipt_hash") != validated.get("receipt_hash"):
            raise ValidationError("vertical slice trace receipt hash differs from receipt list")
        observation_receipt = entry.get("observation_receipt")
        if action.get("kind") == "OBSERVE":
            validated_observation = _validate_observation_receipt(
                observation_receipt, observation=observation
            )
            if validated_observation.get("accepted") is True:
                accepted_observations[str(observation["observation_id"])] = validated_observation
        elif observation_receipt is not None:
            raise ValidationError(
                "vertical slice non-OBSERVE action emitted an observation receipt"
            )
        receipt_confirmation_ids.extend(validated["confirmation_ids"])
        previous = validated

    if len(receipt_confirmation_ids) != len(set(receipt_confirmation_ids)):
        raise ValidationError("vertical slice repeats a confirmation across execution receipts")
    confirmation_by_id: dict[str, dict[str, Any]] = {}
    for confirmation in confirmations:
        if (
            not isinstance(confirmation, dict)
            or confirmation.get("schema") != "org.aerocity.bench.confirmation-receipt.v1"
        ):
            raise ValidationError("vertical slice confirmation receipt schema is invalid")
        confirmation_id = confirmation.get("confirmation_id")
        source_id = confirmation.get("source_observation_id")
        if (
            not isinstance(confirmation_id, str)
            or not confirmation_id
            or confirmation_id in confirmation_by_id
            or confirmation.get("drone_id") != drone_id
            or not isinstance(source_id, str)
            or source_id not in accepted_observations
            or not isinstance(confirmation.get("receipt_token"), str)
            or not confirmation["receipt_token"]
            or not isinstance(confirmation.get("anonymous_target_handle"), str)
            or not confirmation["anonymous_target_handle"]
        ):
            raise ValidationError(
                "vertical slice confirmation receipt is not bound to an accepted OBSERVE"
            )
        confirmation_by_id[confirmation_id] = confirmation
    if set(receipt_confirmation_ids) != set(confirmation_by_id):
        raise ValidationError(
            "vertical slice execution receipts do not bind evaluator confirmations"
        )

    for failure in failures:
        if (
            not isinstance(failure, dict)
            or failure.get("schema") != "org.aerocity.bench.failure-record.v1"
            or failure.get("episode_id") != episode_id
            or failure.get("drone_id") != drone_id
            or not isinstance(failure.get("category"), str)
            or not failure["category"]
            or not isinstance(failure.get("detail"), str)
            or not isinstance(failure.get("terminal"), bool)
        ):
            raise ValidationError("vertical slice failure record is invalid")
        _finite_nonnegative(failure.get("task_time_s"), "failure task_time_s")

    if previous is None:
        raise ValidationError("vertical slice contains no execution receipts")
    simulated_time_s = _finite_nonnegative(execution.get("simulated_time_s"), "simulated_time_s")
    if abs(simulated_time_s - float(previous["task_time_end_s"])) > 1.0e-9:
        raise ValidationError("vertical slice simulated time does not match the receipt chain")
    if audit.get("episode_id") != episode_id or int(audit.get("confirmed_count", -1)) != len(
        confirmations
    ):
        raise ValidationError("vertical slice evaluator audit disagrees with confirmations")
    if set(audit.get("confirmation_ids", [])) != set(confirmation_by_id):
        raise ValidationError("vertical slice evaluator audit confirmation IDs differ")

    receipt_collision = any(bool(receipt["collision"]) for receipt in receipts)
    receipt_out_of_bounds = any(bool(receipt["out_of_bounds"]) for receipt in receipts)
    if (
        final.get("collision_detected") is not receipt_collision
        or final.get("out_of_bounds_detected") is not receipt_out_of_bounds
        or final.get("confirmation_observed") is not bool(confirmations)
    ):
        raise ValidationError("vertical slice final state disagrees with receipt evidence")
    if not isinstance(final.get("returned_home"), bool) or not isinstance(
        final.get("closure_status"), str
    ):
        raise ValidationError("vertical slice final closure fields are invalid")
    home = _position(closure_contract.get("home_position"), "home_position")
    radius = _finite_nonnegative(closure_contract.get("home_radius_m"), "home_radius_m")
    final_state = final.get("final_state")
    if not isinstance(final_state, dict):
        raise ValidationError("vertical slice final state is invalid")
    at_home = (
        distance(_position(final_state.get("position"), "final position"), home) <= radius + 1.0e-9
    )
    if bool(final["returned_home"]) != at_home:
        raise ValidationError("vertical slice returned-home flag lacks geometric evidence")
    if final["closure_status"] == "PASS":
        if (
            not confirmations
            or not at_home
            or receipt_collision
            or receipt_out_of_bounds
            or failures
            or not any(receipt["action_requested"] == "RETURN" for receipt in receipts)
        ):
            raise ValidationError(
                "vertical slice PASS closure is not supported by the receipt evidence"
            )
    elif final["closure_status"] != "FAIL":
        raise ValidationError("vertical slice closure status must be PASS or FAIL")

    return {
        "status": "PASS",
        "formal_score_eligible": False,
        "execution_receipt_set_hash": content_hash(receipts),
        "confirmation_receipt_set_hash": content_hash(confirmations),
        "closure_status": final["closure_status"],
        "control_action_count": len(receipts),
    }
