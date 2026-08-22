"""Pure contracts for the non-formal shared-world CF2X fleet preflight.

The native runner is intentionally split from these helpers.  The helpers can
be checked on a CPU-only developer machine before an Isaac process is opened,
which avoids spending GPU time on malformed rosters, nondeterministic receipt
ordering, or public/private report leaks.  They do not grant formal-score
eligibility; the reviewed CF2X parameters and the complete L1 executor remain
unfrozen.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json
from .errors import ValidationError
from .measurement_evidence import validate_measurement_evidence_snapshot
from .planning_cadence import PLANNING_EVENT_TRIGGERS, validate_planning_cadence

FLEET_SIZE = 4
FLEET_PRECHECK_SCOPE = "cf2x_internal_shared_world_fleet_preflight"
FLEET_PRIVATE_SCOPE = "cf2x_internal_shared_world_fleet_preflight_private"
PRIVATE_WITNESS_FIXTURE_MODE = "private-witness-fixture"
EXTERNAL_PROCESS_POLICY_MODE = "external-process-policy"
SHORT_PREFLIGHT_PURPOSE = "short-engineering-preflight"
COMPLETE_CALIBRATION_PURPOSE = "complete-calibration-episode"
FROZEN_COMPLETE_CALIBRATION_DURATION_S = 300.0
EXECUTION_FAILURE_CATEGORIES = frozenset(
    {
        "collision",
        "out_of_bounds_failure",
        "external_adapter_failure",
        "method_failure",
        "deadline_failure",
        "planner_timeout",
        "planner_crash",
        "controller_failure",
        "reset_failure",
        "return_failure",
        "energy_exhausted",
    }
)
_DRONE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PRIVATE_KEY_TOKENS = (
    "target_id",
    "target_position",
    "target_coordinate",
    "target_normal",
    "target_validity",
    "witness",
    "legal_witness",
    "private_episode",
    "private_fixture",
    "private_audit",
    "trace_private",
    "receipt_secret",
)
_ALLOWED_PUBLIC_PRIVATE_KEYS = {
    "private_evaluator_commitment",
    "private_fixture_commitment",
    "private_report_file_sha256",
}


def validate_native_run_purpose(
    *,
    purpose: str,
    execution_mode: str,
    requested_sim_time_s: float,
    frozen_episode_duration_s: float,
) -> None:
    """Keep short engineering evidence separate from a complete calibration replay."""

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (requested_sim_time_s, frozen_episode_duration_s)
    ):
        raise ValidationError("native run durations must be finite and positive")
    if purpose == SHORT_PREFLIGHT_PURPOSE:
        if requested_sim_time_s >= frozen_episode_duration_s:
            raise ValidationError(
                "short native preflight must remain below the frozen episode duration"
            )
        return
    if purpose != COMPLETE_CALIBRATION_PURPOSE:
        raise ValidationError("native run purpose is unsupported")
    if execution_mode not in {"public-policy", EXTERNAL_PROCESS_POLICY_MODE}:
        raise ValidationError("complete calibration replay requires a public policy")
    if not math.isclose(
        frozen_episode_duration_s,
        FROZEN_COMPLETE_CALIBRATION_DURATION_S,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValidationError("complete calibration replay requires the frozen 300-second contract")
    if not math.isclose(
        requested_sim_time_s,
        frozen_episode_duration_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValidationError(
            "complete calibration replay must use the frozen episode duration exactly"
        )


def public_policy_progress_status(
    *,
    purpose: str,
    observe_action_count: int,
    confirmation_receipt_count: int,
    return_action_count: int,
    all_returned_home: bool,
    episode_budget_completed: bool,
    safe_completion: bool,
    deadline_miss_tick_count: int,
    adapter_failure_count: int = 0,
) -> str:
    """Return a fail-closed status without requiring a nonzero private confirmation."""

    counts = (
        observe_action_count,
        confirmation_receipt_count,
        return_action_count,
        deadline_miss_tick_count,
        adapter_failure_count,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
        raise ValidationError("public policy progress counts must be non-negative integers")
    if any(
        not isinstance(value, bool)
        for value in (all_returned_home, episode_budget_completed, safe_completion)
    ):
        raise ValidationError("public policy completion flags must be booleans")
    if purpose == SHORT_PREFLIGHT_PURPOSE:
        if adapter_failure_count:
            return "ADAPTER_FAILED"
        return "OBSERVE_REACHED" if observe_action_count > 0 else "INCOMPLETE_NO_OBSERVE"
    if purpose != COMPLETE_CALIBRATION_PURPOSE:
        raise ValidationError("public policy progress purpose is unsupported")
    complete = (
        observe_action_count > 0
        and return_action_count > 0
        and all_returned_home
        and episode_budget_completed
        and safe_completion
        and deadline_miss_tick_count == 0
        and adapter_failure_count == 0
    )
    # A zero-confirmation public method remains a valid scientific outcome.  It
    # must not be discarded, retried, or relabelled as an execution failure.
    return "CALIBRATION_EPISODE_CLOSED" if complete else "CALIBRATION_EPISODE_INCOMPLETE"


def validate_complete_calibration_summary(
    *,
    execution_mode: object,
    policy_progress: dict[str, Any],
    planning_timing: dict[str, Any],
    private_final: object,
    public_final: object,
    private_execution: dict[str, Any],
    public_execution: object,
    input_bindings: object,
    public_input_bindings: object,
    method: object,
    public_method: object,
    expected_drone_ids: set[str],
    failure_records: object,
    allow_execution_failure: bool = False,
) -> None:
    """Validate one complete, development-only G2-I calibration replay.

    The normal path is deliberately fail-closed and only accepts an episode
    that closed without an execution failure.  Measurement aggregation has a
    separate opt-in path for a *structurally complete* failed replay: its
    receipts and measured traces are still scientifically useful, but its
    outcome must remain in the denominator.  This flag never grants formal
    eligibility and never relaxes hashes, receipt bindings, duration, or
    public/private consistency checks.
    """

    if execution_mode not in {"public-policy", EXTERNAL_PROCESS_POLICY_MODE}:
        raise ValidationError("complete calibration evidence is not a public-policy run")
    if not isinstance(failure_records, list):
        raise ValidationError("complete calibration failure records are malformed")
    if not allow_execution_failure:
        if policy_progress.get("status") != "CALIBRATION_EPISODE_CLOSED":
            raise ValidationError("complete calibration episode did not close")
        if planning_timing.get("deadline_miss_tick_count") != 0:
            raise ValidationError("complete calibration episode contains a planning deadline miss")
        if failure_records:
            raise ValidationError("complete calibration episode contains failure records")
    else:
        if policy_progress.get("status") not in {
            "CALIBRATION_EPISODE_CLOSED",
            "CALIBRATION_EPISODE_INCOMPLETE",
        }:
            raise ValidationError("failed calibration episode has an invalid progress status")
        for failure in failure_records:
            if (
                not isinstance(failure, dict)
                or not isinstance(failure.get("category"), str)
                or failure["category"] not in EXECUTION_FAILURE_CATEGORIES
            ):
                raise ValidationError("failed calibration contains an unknown failure category")

    if not isinstance(private_final, dict) or not isinstance(public_final, dict):
        raise ValidationError("complete calibration replay lacks final safety evidence")
    required_final = {
        "safe_completion": True,
        "collision_detected": False,
        "out_of_bounds_detected": False,
        "all_returned_home": True,
    }
    if not allow_execution_failure:
        if any(private_final.get(key) is not value for key, value in required_final.items()):
            raise ValidationError("complete calibration final safety state is not closed")
        if any(public_final.get(key) is not value for key, value in required_final.items()):
            raise ValidationError("complete calibration public safety summary is not closed")
    else:
        for key in required_final:
            if not isinstance(private_final.get(key), bool) or not isinstance(
                public_final.get(key), bool
            ):
                raise ValidationError("failed calibration final safety flags are malformed")
    if any(public_final.get(key) is not private_final.get(key) for key in required_final):
        raise ValidationError("complete calibration public/private safety summaries differ")
    returned_home = private_final.get("returned_home_by_drone")
    if (
        not isinstance(returned_home, dict)
        or set(returned_home) != expected_drone_ids
        or any(not isinstance(value, bool) for value in returned_home.values())
    ):
        raise ValidationError("complete calibration per-drone return evidence is incomplete")
    if policy_progress.get("all_returned_home") is not private_final["all_returned_home"]:
        raise ValidationError("complete calibration progress and final return evidence differ")

    if not isinstance(public_execution, dict):
        raise ValidationError("complete calibration lacks a public execution summary")
    control_ticks = private_execution.get("control_ticks")
    control_period_s = private_execution.get("control_period_s")
    if not isinstance(control_ticks, int) or isinstance(control_ticks, bool) or control_ticks <= 0:
        raise ValidationError("complete calibration control-tick count is invalid")
    period = _finite_nonnegative(control_period_s, "complete calibration control period")
    simulated_time_s = _finite_nonnegative(
        private_execution.get("simulated_time_s"), "complete calibration simulated time"
    )
    if period <= 0.0 or not math.isclose(
        control_ticks * period, simulated_time_s, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValidationError("complete calibration tick timing does not reproduce simulated time")
    if not math.isclose(
        simulated_time_s,
        FROZEN_COMPLETE_CALIBRATION_DURATION_S,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValidationError("complete calibration evidence is not a full 300-second episode")
    if policy_progress.get("episode_budget_completed") is not True:
        raise ValidationError("complete calibration progress does not close the episode budget")
    for field in (
        "control_ticks",
        "control_period_s",
        "shared_physx_step_count",
        "simulated_time_s",
    ):
        if public_execution.get(field) != private_execution.get(field):
            raise ValidationError("complete calibration public/private execution summaries differ")
    if public_execution.get("failure_record_count") != len(failure_records):
        raise ValidationError("complete calibration failure-record count differs")

    if not isinstance(input_bindings, dict) or public_input_bindings != input_bindings:
        raise ValidationError("complete calibration input bindings differ")
    required_hashes = {
        "layout_hash",
        "stage_sha256",
        "cityspec_sha256",
        "task_spec_sha256",
        "task_spec_hash",
        "public_episode_sha256",
        "mission_sector_hash",
        "execution_contract_hash",
        "release_config_sha256",
        "cf2x_usd_sha256",
        "cf2x_schema_sha256",
        "dynamics_spec_hash",
        "controller_spec_hash",
        "baseline_source_sha256",
        "geometry_source_sha256",
        "atlas_hash",
    }
    if not required_hashes.issubset(input_bindings) or any(
        not isinstance(input_bindings[field], str)
        or not _SHA256.fullmatch(str(input_bindings[field]))
        for field in required_hashes
    ):
        raise ValidationError("complete calibration lacks immutable input hashes")
    if input_bindings.get("task_track") != "G2-I":
        raise ValidationError("complete calibration evidence is not bound to G2-I")
    if input_bindings.get("inspection_prior_level") != "full-cells":
        raise ValidationError("complete calibration evidence lacks the frozen full-cell prior")
    for field in ("layout_id", "episode_id"):
        if not isinstance(input_bindings.get(field), str) or not input_bindings[field]:
            raise ValidationError(f"complete calibration lacks its {field} binding")
    if not isinstance(method, str) or not method or public_method != method:
        raise ValidationError("complete calibration method binding differs")


_EXECUTION_RECEIPT_FIELDS = {
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
_EXECUTION_RECEIPT_FIELDS_V3 = _EXECUTION_RECEIPT_FIELDS | {"planner_invoked"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_OR_RUNTIME_URI = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\|(?:file|omniverse|nucleus)://)")
CANDIDATE_SHARED_HOLD_THRESHOLDS = {
    "minimum_duration_s": 30.0,
    "maximum_altitude_span_m": 0.02,
    "maximum_terminal_altitude_error_m": 0.01,
    "maximum_terminal_vertical_velocity_mps": 0.01,
    "maximum_late_altitude_slope_mps": 2.5e-4,
}
_ROUTE_AUDIT_SCHEMA = "org.aerocity.bench.baseline-route-budget-audit.v1"
_TIMING_SCHEMA_V1 = "org.aerocity.bench.fleet-preflight-timing.v1"
_TIMING_SCHEMA_V2 = "org.aerocity.bench.fleet-preflight-timing.v2"
_TIMING_SCHEMA_V3 = "org.aerocity.bench.fleet-preflight-timing.v3"
_TIMING_SCHEMA_V4 = "org.aerocity.bench.fleet-preflight-timing.v4"


@dataclass(frozen=True)
class FleetMember:
    """One public CF2X instance in the single shared PhysX world."""

    drone_id: str
    start_position_w_m: tuple[float, float, float]
    start_yaw_deg: float

    @property
    def prim_path(self) -> str:
        # IsaacLab treats a non-alphanumeric leaf as a prim-path regular
        # expression and then does not spawn the asset.  Benchmark drone IDs
        # deliberately allow a hyphen (``uav-00``), so isolate that user-facing
        # ABI from the low-level prim leaf.  The digest avoids collisions such
        # as ``uav-00`` versus ``uav_00`` after normalization.
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", self.drone_id)
        return f"/World/AeroCityFleetPreflight/{normalized}_{content_hash(self.drone_id)[:10]}"


def _finite_position(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a three-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must be finite")
    return result  # type: ignore[return-value]


def public_fleet_members(public_episode: dict[str, Any]) -> tuple[FleetMember, ...]:
    """Return exactly four unique public starts without opening private data."""

    starts = public_episode.get("starts")
    if not isinstance(starts, list) or len(starts) != FLEET_SIZE:
        raise ValueError(f"CF2X fleet preflight requires exactly {FLEET_SIZE} public starts")
    members: list[FleetMember] = []
    for index, item in enumerate(starts):
        if not isinstance(item, dict):
            raise ValueError(f"public start {index} must be an object")
        drone_id = str(item.get("drone_id", ""))
        if not _DRONE_ID.fullmatch(drone_id):
            raise ValueError(f"public start has an unsafe drone ID: {drone_id!r}")
        yaw = float(item.get("yaw_deg", math.nan))
        if not math.isfinite(yaw):
            raise ValueError(f"public start has a non-finite yaw: {drone_id}")
        members.append(
            FleetMember(
                drone_id=drone_id,
                start_position_w_m=_finite_position(item.get("position"), "public start position"),
                start_yaw_deg=yaw,
            )
        )
    identifiers = [member.drone_id for member in members]
    if len(set(identifiers)) != FLEET_SIZE:
        raise ValueError("CF2X fleet preflight public starts must have unique drone IDs")
    paths = [member.prim_path for member in members]
    if len(set(paths)) != FLEET_SIZE:
        raise ValueError("CF2X fleet preflight generated duplicate prim paths")
    return tuple(sorted(members, key=lambda member: member.drone_id))


def assert_action_roster_complete(
    actions: dict[str, object], members: tuple[FleetMember, ...]
) -> None:
    """Reject a control tick that omits or invents a live CF2X instance."""

    expected = {member.drone_id for member in members}
    actual = set(actions)
    if actual != expected:
        raise ValueError(
            "fleet action roster must exactly match the four live drones: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )


@dataclass
class SharedWorldStepLedger:
    """Count one shared PhysX step only after all four thrusts are written."""

    members: tuple[FleetMember, ...]
    shared_physx_step_count: int = 0

    def record_step(self, written_drone_ids: set[str]) -> None:
        expected = {member.drone_id for member in self.members}
        if written_drone_ids != expected:
            raise ValueError(
                "a shared PhysX tick requires one pending thrust target for every fleet member"
            )
        self.shared_physx_step_count += 1


def canonical_receipts(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the required sequence-then-drone ordering for L1-like evidence."""

    return sorted(
        receipts,
        key=lambda item: (int(item.get("action_sequence", -1)), str(item.get("drone_id", ""))),
    )


def assert_canonical_fleet_receipts(
    receipts: list[dict[str, Any]],
    members: tuple[FleetMember, ...],
    *,
    expected_control_ticks: int,
) -> None:
    """Validate a complete four-receipt roster at every candidate control tick."""

    if expected_control_ticks <= 0:
        raise ValueError("expected_control_ticks must be positive")
    expected_ids = {member.drone_id for member in members}
    if receipts != canonical_receipts(receipts):
        raise ValidationError("fleet execution receipts are not in canonical step/agent order")
    if len(receipts) != len(members) * expected_control_ticks:
        raise ValidationError("fleet receipt count does not equal four receipts per control tick")
    previous: dict[str, dict[str, Any] | None] = {member.drone_id: None for member in members}
    seen: set[tuple[str, int]] = set()
    for sequence in range(expected_control_ticks):
        tick = receipts[sequence * len(members) : (sequence + 1) * len(members)]
        if {str(item.get("drone_id", "")) for item in tick} != expected_ids:
            raise ValidationError("fleet receipt tick omits or duplicates a drone")
        if {int(item.get("action_sequence", -1)) for item in tick} != {sequence}:
            raise ValidationError("fleet receipt tick has an unexpected action sequence")
        for receipt in tick:
            schema = receipt.get("schema")
            expected_fields = (
                _EXECUTION_RECEIPT_FIELDS_V3
                if schema == "org.aerocity.bench.execution-receipt.v3"
                else _EXECUTION_RECEIPT_FIELDS
            )
            if set(receipt) != expected_fields:
                missing = sorted(expected_fields - set(receipt))
                extra = sorted(set(receipt) - expected_fields)
                raise ValidationError(
                    f"fleet receipt fields differ; missing={missing}, extra={extra}"
                )
            payload = dict(receipt)
            receipt_hash = payload.pop("receipt_hash", None)
            if not isinstance(receipt_hash, str) or not _SHA256.fullmatch(receipt_hash):
                raise ValidationError("fleet receipt has an invalid receipt hash")
            if content_hash(payload) != receipt_hash:
                raise ValidationError("fleet receipt content hash is corrupt")
            if schema not in {
                "org.aerocity.bench.execution-receipt.v2",
                "org.aerocity.bench.execution-receipt.v3",
            }:
                raise ValidationError("fleet receipt uses an unsupported schema")
            if schema == "org.aerocity.bench.execution-receipt.v3" and not isinstance(
                receipt.get("planner_invoked"), bool
            ):
                raise ValidationError("fleet receipt planner invocation flag is invalid")
            if receipt.get("execution_level") != "L1":
                raise ValidationError("fleet receipt must bind measured L1 state")
            drone_id = str(receipt.get("drone_id", ""))
            pair = (drone_id, sequence)
            if pair in seen:
                raise ValidationError("fleet receipt duplicates a drone/action pair")
            seen.add(pair)
            prior = previous[drone_id]
            if prior is None:
                if receipt.get("previous_receipt_hash") is not None:
                    raise ValidationError("fleet receipt chain must start without a previous hash")
                if abs(float(receipt.get("task_time_start_s", math.nan))) > 1.0e-9:
                    raise ValidationError("fleet receipt chain must start at task time zero")
            else:
                if receipt.get("previous_receipt_hash") != prior["receipt_hash"]:
                    raise ValidationError("fleet receipt chain is discontinuous for a drone")
                if receipt.get("state_before_hash") != prior["state_after_hash"]:
                    raise ValidationError("fleet receipt state chain is discontinuous for a drone")
                if (
                    abs(
                        float(receipt.get("task_time_start_s", math.nan))
                        - float(prior["task_time_end_s"])
                    )
                    > 1.0e-9
                ):
                    raise ValidationError("fleet receipt time chain is discontinuous for a drone")
            numeric = (
                "task_time_start_s",
                "task_time_end_s",
                "planning_latency_s",
                "distance_m",
                "energy_used_j",
            )
            if any(
                isinstance(receipt.get(key), bool)
                or not isinstance(receipt.get(key), (int, float))
                or not math.isfinite(float(receipt[key]))
                or float(receipt[key]) < 0.0
                for key in numeric
            ):
                raise ValidationError("fleet receipt has invalid numeric evidence")
            if float(receipt["task_time_end_s"]) < float(receipt["task_time_start_s"]):
                raise ValidationError("fleet receipt time runs backwards")
            for key in (
                "action_packet_hash",
                "source_observation_hash",
                "state_before_hash",
                "state_after_hash",
            ):
                if not isinstance(receipt.get(key), str) or not _SHA256.fullmatch(receipt[key]):
                    raise ValidationError(f"fleet receipt has invalid {key}")
            if (
                not isinstance(receipt.get("source_observation_id"), str)
                or not receipt["source_observation_id"]
            ):
                raise ValidationError("fleet receipt lacks a source observation ID")
            if not isinstance(receipt.get("confirmation_ids"), list) or len(
                receipt["confirmation_ids"]
            ) != len(set(receipt["confirmation_ids"])):
                raise ValidationError("fleet receipt confirmation IDs are invalid")
            previous[drone_id] = receipt


def assert_fleet_execution_bindings(
    receipts: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> None:
    """Bind every receipt hash to its exact public action and observation packet.

    These packets are retained only in the private engineering evidence so the
    verifier can reject a substituted action, stale observation, or summary-only
    receipt.  They contain no evaluator-private target material.
    """

    if len(bindings) != len(receipts):
        raise ValidationError("fleet execution bindings do not match receipt count")
    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValidationError("fleet execution binding must be an object")
        drone_id = binding.get("drone_id")
        sequence = binding.get("action_sequence")
        action = binding.get("action")
        observation = binding.get("source_observation")
        planner_invoked = binding.get("planner_invoked")
        planning_trigger_reasons = binding.get("planning_trigger_reasons")
        if (
            not isinstance(drone_id, str)
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 0
            or not isinstance(action, dict)
            or not isinstance(observation, dict)
        ):
            raise ValidationError("fleet execution binding has malformed public packets")
        if planner_invoked is not None or planning_trigger_reasons is not None:
            if not isinstance(planner_invoked, bool) or not isinstance(
                planning_trigger_reasons, list
            ):
                raise ValidationError("fleet execution binding has malformed planning metadata")
            allowed_reasons = {"initial", "fixed_period", *PLANNING_EVENT_TRIGGERS}
            if (
                any(not isinstance(value, str) for value in planning_trigger_reasons)
                or set(planning_trigger_reasons) - allowed_reasons
                or len(planning_trigger_reasons) != len(set(planning_trigger_reasons))
                or planner_invoked != bool(planning_trigger_reasons)
            ):
                raise ValidationError("fleet execution binding planning metadata is invalid")
        pair = (drone_id, sequence)
        if pair in by_pair:
            raise ValidationError("fleet execution bindings duplicate a drone/action pair")
        if (
            action.get("schema") != "org.aerocity.bench.action-packet.v1"
            or observation.get("schema") != "org.aerocity.bench.observation-packet.v2"
            or action.get("drone_id") != drone_id
            or observation.get("drone_id") != drone_id
            or action.get("sequence") != sequence
            or observation.get("sequence") != sequence
            or action.get("episode_id") != observation.get("episode_id")
            or action.get("issued_at_s") != observation.get("timestamp_s")
        ):
            raise ValidationError("fleet execution binding action and observation disagree")
        action_kind = action.get("kind")
        if action_kind == "OBSERVE":
            if action.get("source_observation_id") != observation.get("observation_id"):
                raise ValidationError("fleet OBSERVE action is not bound to its source observation")
        elif action.get("source_observation_id") is not None:
            raise ValidationError("fleet non-OBSERVE action binds a source observation")
        by_pair[pair] = binding
    for receipt in receipts:
        pair = (str(receipt["drone_id"]), int(receipt["action_sequence"]))
        binding = by_pair.pop(pair, None)
        if binding is None:
            raise ValidationError("fleet receipt lacks its public action/observation binding")
        action = binding["action"]
        observation = binding["source_observation"]
        if (
            receipt.get("episode_id") != action.get("episode_id")
            or receipt.get("action_requested") != action.get("kind")
            or receipt.get("action_packet_hash") != content_hash(action)
            or receipt.get("source_observation_id") != observation.get("observation_id")
            or receipt.get("source_observation_hash") != content_hash(observation)
        ):
            raise ValidationError("fleet receipt does not bind its exact action and observation")
        if receipt.get("schema") == "org.aerocity.bench.execution-receipt.v3" and (
            binding.get("planner_invoked") is not receipt.get("planner_invoked")
        ):
            raise ValidationError("fleet receipt planner flag differs from its execution binding")
    if by_pair:
        raise ValidationError("fleet execution binding has no corresponding receipt")


def assert_fleet_confirmation_bindings(
    receipts: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    observation_receipts: list[dict[str, Any]],
    confirmation_receipts: list[dict[str, Any]],
) -> None:
    """Require every confirmation to originate from an accepted OBSERVE packet.

    The confirmation payload is evaluator-private engineering evidence.  This
    check deliberately relates it only to the public action/observation packets
    retained in the private report; no target coordinate, ID, or witness is
    needed to prove the causal chain.
    """

    binding_by_observation: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        action = binding.get("action")
        observation = binding.get("source_observation")
        if not isinstance(action, dict) or not isinstance(observation, dict):
            raise ValidationError("fleet confirmation binding has malformed public packets")
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValidationError("fleet confirmation binding lacks an observation ID")
        if observation_id in binding_by_observation:
            raise ValidationError("fleet confirmation bindings reuse an observation ID")
        binding_by_observation[observation_id] = binding

    accepted_by_observation: dict[str, dict[str, Any]] = {}
    seen_observation_receipts: set[str] = set()
    for receipt in observation_receipts:
        if not isinstance(receipt, dict):
            raise ValidationError("fleet observation receipt must be an object")
        observation_id = receipt.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValidationError("fleet observation receipt lacks an observation ID")
        if observation_id in seen_observation_receipts:
            raise ValidationError("fleet observation receipts duplicate an observation ID")
        seen_observation_receipts.add(observation_id)
        binding = binding_by_observation.get(observation_id)
        if binding is None or binding["action"].get("kind") != "OBSERVE":
            raise ValidationError("fleet observation receipt is not bound to OBSERVE")
        if receipt.get("drone_id") != binding.get("drone_id"):
            raise ValidationError("fleet observation receipt drone does not match OBSERVE binding")
        if receipt.get("accepted") is True:
            accepted_by_observation[observation_id] = receipt

    receipt_by_confirmation: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        for confirmation_id in receipt["confirmation_ids"]:
            if not isinstance(confirmation_id, str) or not confirmation_id:
                raise ValidationError("fleet receipt confirmation ID is malformed")
            if confirmation_id in receipt_by_confirmation:
                raise ValidationError(
                    "fleet confirmation appears in more than one execution receipt"
                )
            binding = binding_by_observation.get(str(receipt["source_observation_id"]))
            if binding is None or binding["action"].get("kind") != "OBSERVE":
                raise ValidationError("fleet confirmation is attached to a non-OBSERVE receipt")
            receipt_by_confirmation[confirmation_id] = receipt

    confirmation_by_id: dict[str, dict[str, Any]] = {}
    for confirmation in confirmation_receipts:
        if not isinstance(confirmation, dict):
            raise ValidationError("fleet confirmation receipt must be an object")
        if confirmation.get("schema") != "org.aerocity.bench.confirmation-receipt.v1":
            raise ValidationError("fleet confirmation receipt schema is invalid")
        confirmation_id = confirmation.get("confirmation_id")
        source_observation_id = confirmation.get("source_observation_id")
        if (
            not isinstance(confirmation_id, str)
            or not confirmation_id
            or confirmation_id in confirmation_by_id
            or not isinstance(source_observation_id, str)
            or not source_observation_id
        ):
            raise ValidationError("fleet confirmation receipt identity is invalid")
        execution_receipt = receipt_by_confirmation.get(confirmation_id)
        accepted = accepted_by_observation.get(source_observation_id)
        if execution_receipt is None or accepted is None:
            raise ValidationError("fleet confirmation is not bound to an accepted OBSERVE")
        if (
            execution_receipt["source_observation_id"] != source_observation_id
            or confirmation.get("drone_id") != execution_receipt["drone_id"]
        ):
            raise ValidationError("fleet confirmation source does not match execution receipt")
        confirmation_by_id[confirmation_id] = confirmation

    if set(receipt_by_confirmation) != set(confirmation_by_id):
        raise ValidationError("fleet execution and evaluator confirmation sets differ")


def altitude_stability_metrics(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    """Measure post-warm-up altitude drift from measured native state samples.

    The slope is an ordinary least-squares fit over the latter half of the
    supplied interval.  It is deliberately separate from final altitude so a
    drone that happens to end near the requested height while continuing to
    sink cannot pass the preflight check.
    """

    if len(samples) < 4:
        raise ValueError("altitude stability needs at least four measured samples")
    values: list[tuple[float, float, float]] = []
    for sample in samples:
        time_s = float(sample.get("task_time_s", math.nan))
        position = _finite_position(sample.get("position_w_m"), "altitude sample position")
        velocity = _finite_position(sample.get("linear_velocity_w_mps"), "altitude sample velocity")
        if not math.isfinite(time_s):
            raise ValueError("altitude sample task time must be finite")
        values.append((time_s, position[2], velocity[2]))
    if any(right[0] <= left[0] for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("altitude samples must have strictly increasing task time")
    tail = values[len(values) // 2 :]
    mean_time = sum(item[0] for item in tail) / len(tail)
    mean_height = sum(item[1] for item in tail) / len(tail)
    variance = sum((item[0] - mean_time) ** 2 for item in tail)
    if variance <= 1.0e-12:
        raise ValueError("altitude sample times have zero tail variance")
    slope = sum((item[0] - mean_time) * (item[1] - mean_height) for item in tail) / variance
    heights = [item[1] for item in values]
    return {
        "sample_count": len(values),
        "duration_s": values[-1][0] - values[0][0],
        "initial_altitude_m": values[0][1],
        "final_altitude_m": values[-1][1],
        "altitude_span_m": max(heights) - min(heights),
        "late_altitude_slope_mps": slope,
        "terminal_vertical_velocity_mps": values[-1][2],
    }


def candidate_shared_hold_assessment(
    stability_by_drone: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    """Assess the documented candidate-only long-horizon shared-hold gate.

    This is deliberately an engineering regression guard, not an airworthiness
    claim.  Its limits originate in the local CF2X preflight contract and are
    reported with the result so later parameter audits can replace them rather
    than silently inheriting a hidden threshold.
    """

    if len(stability_by_drone) != FLEET_SIZE:
        raise ValueError("candidate shared-hold assessment requires exactly four drones")
    failures: dict[str, list[str]] = {}
    for drone_id, metrics in sorted(stability_by_drone.items()):
        try:
            duration = float(metrics["duration_s"])
            span = float(metrics["altitude_span_m"])
            terminal_error = abs(
                float(metrics["final_altitude_m"]) - float(metrics["initial_altitude_m"])
            )
            terminal_velocity = abs(float(metrics["terminal_vertical_velocity_mps"]))
            slope = abs(float(metrics["late_altitude_slope_mps"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate shared-hold metrics are malformed for {drone_id}") from exc
        values = (duration, span, terminal_error, terminal_velocity, slope)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"candidate shared-hold metrics are non-finite for {drone_id}")
        drone_failures: list[str] = []
        if duration + 1.0e-9 < CANDIDATE_SHARED_HOLD_THRESHOLDS["minimum_duration_s"]:
            drone_failures.append("duration")
        if span > CANDIDATE_SHARED_HOLD_THRESHOLDS["maximum_altitude_span_m"]:
            drone_failures.append("altitude_span")
        if terminal_error > CANDIDATE_SHARED_HOLD_THRESHOLDS["maximum_terminal_altitude_error_m"]:
            drone_failures.append("terminal_altitude_error")
        if (
            terminal_velocity
            > CANDIDATE_SHARED_HOLD_THRESHOLDS["maximum_terminal_vertical_velocity_mps"]
        ):
            drone_failures.append("terminal_vertical_velocity")
        if slope > CANDIDATE_SHARED_HOLD_THRESHOLDS["maximum_late_altitude_slope_mps"]:
            drone_failures.append("late_altitude_slope")
        if drone_failures:
            failures[drone_id] = drone_failures
    return {
        "status": "PASS" if not failures else "FAIL",
        "candidate_preflight_only": True,
        "thresholds": dict(CANDIDATE_SHARED_HOLD_THRESHOLDS),
        "failed_checks_by_drone": failures,
    }


def _walk_public_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            keys.append(key_text)
            keys.extend(_walk_public_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_public_keys(child))
    return keys


def _walk_public_strings(value: object) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_public_strings(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_public_strings(child))
    elif isinstance(value, str):
        values.append(value)
    return values


def assert_public_report_has_no_private_truth(public_report: dict[str, Any]) -> None:
    """Reject private targets, witnesses, traces, and private file paths in public output."""

    if public_report.get("formal_score_eligible") is not False:
        raise ValidationError("fleet preflight must explicitly remain non-formal")
    if public_report.get("evidence_scope") != FLEET_PRECHECK_SCOPE:
        raise ValidationError("fleet preflight public evidence scope is invalid")
    leaked = [
        key
        for key in _walk_public_keys(public_report)
        if key not in _ALLOWED_PUBLIC_PRIVATE_KEYS
        and any(token in key for token in _PRIVATE_KEY_TOKENS)
    ]
    if leaked:
        raise ValidationError(
            f"fleet public report leaks private truth keys: {sorted(set(leaked))}"
        )
    if any(_LOCAL_OR_RUNTIME_URI.search(value) for value in _walk_public_strings(public_report)):
        raise ValidationError("fleet public report leaks a local path or runtime URI value")


def _finite_nonnegative(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValidationError(f"{field} must be finite and non-negative")
    return number


def _validate_percentiles(value: object, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"p50_s", "p95_s", "p99_s", "max_s"}:
        raise ValidationError(f"{field} has an invalid percentile summary")
    p50 = _finite_nonnegative(value["p50_s"], f"{field}.p50_s")
    p95 = _finite_nonnegative(value["p95_s"], f"{field}.p95_s")
    p99 = _finite_nonnegative(value["p99_s"], f"{field}.p99_s")
    maximum = _finite_nonnegative(value["max_s"], f"{field}.max_s")
    if not p50 <= p95 <= p99 <= maximum:
        raise ValidationError(f"{field} percentiles are not monotonic")


def _validate_optional_substage_percentiles(value: object, field: str, max_count: int) -> None:
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be a timing summary")
    required = {"call_count", "p50_s", "p95_s", "p99_s", "max_s"}
    if set(value) != required:
        raise ValidationError(f"{field} timing summary fields are invalid")
    count = value["call_count"]
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= max_count:
        raise ValidationError(f"{field}.call_count is invalid")
    percentiles = {key: value[key] for key in ("p50_s", "p95_s", "p99_s", "max_s")}
    if count == 0:
        if any(item is not None for item in percentiles.values()):
            raise ValidationError(f"{field} must not invent samples")
        return
    _validate_percentiles(percentiles, field)


def _validate_planning_timing(
    timing: object, *, expected_control_ticks: int, execution_mode: object
) -> None:
    if not isinstance(timing, dict):
        raise ValidationError("fleet preflight lacks planning timing summary")
    required_v1 = {
        "schema",
        "control_tick_count",
        "planning_deadline_s",
        "deadline_miss_tick_count",
        "policy_call",
        "public_observation_build",
    }
    schema = timing.get("schema")
    required_external = {*required_v1, "external_process_substages"}
    required_v4 = {
        *required_external,
        "control_period_s",
        "planner_invocation_count",
        "held_action_tick_count",
        "planning_cadence",
        "planning_trigger_counts",
    }
    if schema == _TIMING_SCHEMA_V1:
        required = required_v1
    elif schema in {_TIMING_SCHEMA_V2, _TIMING_SCHEMA_V3}:
        required = required_external
    elif schema == _TIMING_SCHEMA_V4:
        required = required_v4
    else:
        raise ValidationError("fleet planning timing schema is invalid")
    if set(timing) != required:
        raise ValidationError("fleet planning timing schema is invalid")
    if int(timing["control_tick_count"]) != expected_control_ticks:
        raise ValidationError("fleet planning timing control-tick count disagrees with receipts")
    deadline = _finite_nonnegative(timing["planning_deadline_s"], "planning_deadline_s")
    if deadline <= 0.0:
        raise ValidationError("planning_deadline_s must be positive")
    deadline_misses = timing["deadline_miss_tick_count"]
    if (
        not isinstance(deadline_misses, int)
        or isinstance(deadline_misses, bool)
        or not 0 <= deadline_misses <= expected_control_ticks
    ):
        raise ValidationError("fleet planning deadline-miss count is invalid")
    if schema == _TIMING_SCHEMA_V4:
        control_period = _finite_nonnegative(timing["control_period_s"], "control_period_s")
        if control_period <= 0.0:
            raise ValidationError("control_period_s must be positive")
        invocation_count = timing["planner_invocation_count"]
        held_count = timing["held_action_tick_count"]
        if (
            not isinstance(invocation_count, int)
            or isinstance(invocation_count, bool)
            or not 0 <= invocation_count <= expected_control_ticks
            or not isinstance(held_count, int)
            or isinstance(held_count, bool)
            or held_count != expected_control_ticks - invocation_count
        ):
            raise ValidationError("fleet planner invocation counts are invalid")
        if deadline_misses > invocation_count:
            raise ValidationError("planner deadline misses exceed planner invocations")
        _validate_optional_substage_percentiles(
            timing["policy_call"], "policy_call", invocation_count
        )
        if int(timing["policy_call"]["call_count"]) != invocation_count:
            raise ValidationError("policy timing sample count differs from planner invocations")
        try:
            planning_cadence = validate_planning_cadence(
                timing["planning_cadence"],
                control_period_s=control_period,
                episode_duration_s=FROZEN_COMPLETE_CALIBRATION_DURATION_S,
            )
        except ValueError as exc:
            raise ValidationError(f"fleet planning cadence is invalid: {exc}") from exc
        trigger_counts = timing["planning_trigger_counts"]
        allowed_triggers = {"initial", "fixed_period", *PLANNING_EVENT_TRIGGERS}
        if (
            not isinstance(trigger_counts, dict)
            or set(trigger_counts) - allowed_triggers
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in trigger_counts.values()
            )
        ):
            raise ValidationError("fleet planning trigger counts are invalid")
        if execution_mode in {"public-policy", EXTERNAL_PROCESS_POLICY_MODE}:
            if invocation_count <= 0:
                raise ValidationError("public fleet run did not invoke its planner")
            interval_ticks = round(
                float(planning_cadence["period_s"]) / control_period
            )
            expected_fixed_period_count = (
                (expected_control_ticks - 1) // interval_ticks + 1
            )
            if (
                trigger_counts.get("initial", 0) != 1
                or trigger_counts.get("fixed_period", 0)
                != expected_fixed_period_count
            ):
                raise ValidationError("fleet planning fixed trigger counts are invalid")
        elif invocation_count != 0:
            raise ValidationError("internal fleet fixture reports planner invocations")
    else:
        invocation_count = expected_control_ticks
        _validate_percentiles(timing["policy_call"], "policy_call")
    _validate_percentiles(timing["public_observation_build"], "public_observation_build")
    if schema in {_TIMING_SCHEMA_V2, _TIMING_SCHEMA_V3, _TIMING_SCHEMA_V4}:
        substages = timing["external_process_substages"]
        if execution_mode != EXTERNAL_PROCESS_POLICY_MODE:
            if substages is not None:
                raise ValidationError("non-external fleet run reports external timing substages")
        else:
            if not isinstance(substages, dict):
                raise ValidationError("external fleet run lacks timing substages")
            expected_substages = (
                {
                    "bridge_act_wall_clock",
                    "bridge_act_process_cpu",
                    "fleet_arbitration_wall_clock",
                    "fleet_arbitration_process_cpu",
                    "unattributed_wall_clock",
                    "unattributed_process_cpu",
                }
                if schema == _TIMING_SCHEMA_V2
                else {
                    "bridge_act_wall_clock",
                    "projection_wall_clock",
                    "request_public_audit_wall_clock",
                    "request_json_serialize_wall_clock",
                    "request_size_check_wall_clock",
                    "request_write_flush_wall_clock",
                    "response_wait_wall_clock",
                    "response_json_decode_wall_clock",
                    "response_validate_wall_clock",
                    "action_validation_conversion_wall_clock",
                    "bridge_internal_unattributed_wall_clock",
                    "fleet_arbitration_wall_clock",
                    "unattributed_wall_clock",
                }
            )
            if set(substages) != expected_substages:
                raise ValidationError("external fleet timing substage fields are invalid")
            for field, summary in substages.items():
                _validate_optional_substage_percentiles(summary, field, invocation_count)
    if execution_mode in {"shared-hold", PRIVATE_WITNESS_FIXTURE_MODE} and deadline_misses:
        raise ValidationError("internal fleet fixtures must not report planner deadline misses")


def _validate_planning_receipt_consistency(
    timing: object,
    receipts: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> None:
    """Reconcile v4 timing summaries with every per-drone execution receipt."""

    if not isinstance(timing, dict) or timing.get("schema") != _TIMING_SCHEMA_V4:
        return
    if any(
        receipt.get("schema") != "org.aerocity.bench.execution-receipt.v3"
        for receipt in receipts
    ):
        raise ValidationError("v4 planning timing requires v3 execution receipts")
    binding_by_pair = {
        (str(binding.get("drone_id")), int(binding.get("action_sequence", -1))): binding
        for binding in bindings
    }
    by_tick: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for receipt in receipts:
        pair = (str(receipt["drone_id"]), int(receipt["action_sequence"]))
        binding = binding_by_pair.get(pair)
        if binding is None:
            raise ValidationError("v4 planning receipt lacks its execution binding")
        by_tick.setdefault(pair[1], []).append((receipt, binding))

    planner_invocations = 0
    deadline_misses = 0
    trigger_counts: Counter[str] = Counter()
    for _tick, rows in sorted(by_tick.items()):
        if len(rows) != FLEET_SIZE:
            raise ValidationError("v4 planning tick does not preserve the full fleet")
        planner_flags = {receipt.get("planner_invoked") for receipt, _ in rows}
        reasons = {
            tuple(sorted(str(value) for value in binding.get("planning_trigger_reasons", ())))
            for _, binding in rows
        }
        deadline_flags = {bool(receipt.get("deadline_miss")) for receipt, _ in rows}
        if len(planner_flags) != 1 or len(reasons) != 1 or len(deadline_flags) != 1:
            raise ValidationError("v4 planning metadata differs within a fleet tick")
        planner_invoked = planner_flags == {True}
        tick_reasons = next(iter(reasons))
        deadline_miss = deadline_flags == {True}
        if planner_invoked != bool(tick_reasons):
            raise ValidationError("v4 planner flag differs from its trigger reasons")
        if not planner_invoked and any(
            float(receipt.get("planning_latency_s", -1.0)) != 0.0
            or bool(receipt.get("deadline_miss"))
            for receipt, _ in rows
        ):
            raise ValidationError("held-action tick invents planning latency or a deadline miss")
        if deadline_miss and not planner_invoked:
            raise ValidationError("held-action tick cannot miss a planning deadline")
        planner_invocations += int(planner_invoked)
        deadline_misses += int(deadline_miss)
        trigger_counts.update(tick_reasons)

    if (
        planner_invocations != timing.get("planner_invocation_count")
        or len(by_tick) - planner_invocations != timing.get("held_action_tick_count")
        or deadline_misses != timing.get("deadline_miss_tick_count")
        or dict(sorted(trigger_counts.items())) != timing.get("planning_trigger_counts")
    ):
        raise ValidationError("v4 planning timing differs from execution receipts")


def _validate_route_budget_audit(
    audit: object, *, members: tuple[FleetMember, ...], execution_mode: object
) -> None:
    if not isinstance(audit, dict):
        raise ValidationError("fleet preflight lacks route budget audit")
    if execution_mode == "shared-hold":
        if audit != {
            "schema": _ROUTE_AUDIT_SCHEMA,
            "status": "NOT_APPLICABLE",
            "reason": "shared-hold does not execute a public search route",
        }:
            raise ValidationError("shared-hold route budget audit is invalid")
        return
    if execution_mode == PRIVATE_WITNESS_FIXTURE_MODE:
        if audit != {
            "schema": _ROUTE_AUDIT_SCHEMA,
            "status": "NOT_APPLICABLE",
            "reason": (
                "private-witness-fixture uses an evaluator-owned internal route; "
                "it is not a public search method"
            ),
        }:
            raise ValidationError("private-witness fixture route budget audit is invalid")
        return
    if execution_mode == EXTERNAL_PROCESS_POLICY_MODE:
        if audit != {
            "schema": _ROUTE_AUDIT_SCHEMA,
            "status": "NOT_APPLICABLE",
            "reason": (
                "external-process-policy owns its public route choice; "
                "the shared executor records measured deadline, safety, and return outcomes"
            ),
        }:
            raise ValidationError("external process route budget declaration is invalid")
        return
    required = {
        "schema",
        "method_id",
        "model",
        "kinematic_lower_bound_only",
        "horizontal_speed_mps",
        "vertical_speed_mps",
        "episode_duration_s",
        "by_drone",
        "status",
    }
    if set(audit) != required or audit.get("schema") != _ROUTE_AUDIT_SCHEMA:
        raise ValidationError("fleet route budget audit schema is invalid")
    if audit.get("status") != "LOWER_BOUND_FITS":
        raise ValidationError("native public-policy report cannot contain an infeasible route")
    if audit.get("kinematic_lower_bound_only") is not True:
        raise ValidationError("fleet route budget audit must remain a lower-bound declaration")
    duration = _finite_nonnegative(audit["episode_duration_s"], "route audit duration")
    if duration <= 0.0:
        raise ValidationError("fleet route audit duration must be positive")
    for field in ("horizontal_speed_mps", "vertical_speed_mps"):
        if _finite_nonnegative(audit[field], f"route audit {field}") <= 0.0:
            raise ValidationError(f"route audit {field} must be positive")
    by_drone = audit["by_drone"]
    expected_ids = {member.drone_id for member in members}
    if not isinstance(by_drone, dict) or set(by_drone) != expected_ids:
        raise ValidationError("fleet route budget audit roster differs from fleet receipts")
    required_drone = {
        "route_point_count",
        "observe_pose_count",
        "search_motion_lower_bound_s",
        "observe_dwell_lower_bound_s",
        "return_motion_lower_bound_s",
        "return_reserve_s",
        "total_required_lower_bound_s",
        "status",
    }
    for drone_id, item in by_drone.items():
        if not isinstance(item, dict) or set(item) != required_drone:
            raise ValidationError(f"fleet route budget fields are invalid for {drone_id}")
        for count_name in ("route_point_count", "observe_pose_count"):
            count = item[count_name]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValidationError(f"fleet route budget {count_name} is invalid for {drone_id}")
        pieces = [
            _finite_nonnegative(item[field], f"route audit {field} for {drone_id}")
            for field in (
                "search_motion_lower_bound_s",
                "observe_dwell_lower_bound_s",
                "return_motion_lower_bound_s",
                "return_reserve_s",
            )
        ]
        total = _finite_nonnegative(
            item["total_required_lower_bound_s"], f"route audit total for {drone_id}"
        )
        if abs(sum(pieces) - total) > 1.0e-6:
            raise ValidationError(f"fleet route budget total does not add up for {drone_id}")
        if item.get("status") != "LOWER_BOUND_FITS" or total > duration + 1.0e-9:
            raise ValidationError(f"fleet route budget does not fit for {drone_id}")


def validate_fleet_preflight_reports(
    public_path: Path,
    private_path: Path,
    *,
    allow_execution_failure: bool = False,
) -> dict[str, str | int]:
    """Verify file/content commitments without returning private run material.

    ``allow_execution_failure`` is reserved for the measurement aggregator.
    It accepts only a full-duration, receipt-complete replay whose failure is
    represented in the public/private evidence.  The default remains the
    strict closure check used by runners and readiness gates.
    """

    public = read_json(public_path)
    private = read_json(private_path)
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise ValidationError("fleet preflight evidence must be JSON objects")
    public_hash = str(public.pop("public_report_sha256", ""))
    private_hash = str(private.pop("private_report_content_sha256", ""))
    if content_hash(public) != public_hash:
        raise ValidationError("fleet public report content hash mismatch")
    if content_hash(private) != private_hash:
        raise ValidationError("fleet private report content hash mismatch")
    if public.get("schema") != "org.aerocity.bench.cf2x-l1-fleet-preflight.v4":
        raise ValidationError("fleet public report uses an unsupported schema")
    if private.get("schema") != "org.aerocity.bench.cf2x-l1-fleet-preflight-private.v4":
        raise ValidationError("fleet private report uses an unsupported schema")
    if private.get("formal_score_eligible") is not False:
        raise ValidationError("fleet private preflight attempted to claim formal status")
    if private.get("evidence_scope") != FLEET_PRIVATE_SCOPE:
        raise ValidationError("fleet private evidence scope is invalid")
    assert_public_report_has_no_private_truth(public)
    if public.get("private_report_file_sha256") != file_hash(private_path):
        raise ValidationError("fleet public report does not bind the private evidence file")
    if public.get("private_evaluator_commitment") != private.get("private_evaluator_commitment"):
        raise ValidationError("fleet public and private evaluator commitments differ")
    candidate = private.get("candidate_shared_hold")
    if not isinstance(candidate, dict) or candidate.get("candidate_preflight_only") is not True:
        raise ValidationError("fleet private report lacks a candidate stability declaration")
    execution_mode = private.get("execution_mode")
    execution_purpose = private.get("execution_purpose", SHORT_PREFLIGHT_PURPOSE)
    if public.get("execution_purpose", SHORT_PREFLIGHT_PURPOSE) != execution_purpose:
        raise ValidationError("fleet public/private execution purposes differ")
    if execution_purpose not in {
        SHORT_PREFLIGHT_PURPOSE,
        COMPLETE_CALIBRATION_PURPOSE,
    }:
        raise ValidationError("fleet private report has an unknown execution purpose")
    if execution_mode == "shared-hold":
        if candidate.get("status") not in {"PASS", "FAIL"}:
            raise ValidationError("fleet shared-hold candidate stability status is invalid")
        if candidate.get("thresholds") != CANDIDATE_SHARED_HOLD_THRESHOLDS:
            raise ValidationError("fleet shared-hold candidate thresholds are not frozen")
        failures = candidate.get("failed_checks_by_drone")
        if not isinstance(failures, dict) or (candidate["status"] == "PASS" and failures):
            raise ValidationError("fleet shared-hold candidate stability evidence is inconsistent")
    elif execution_mode in {
        "public-policy",
        EXTERNAL_PROCESS_POLICY_MODE,
        PRIVATE_WITNESS_FIXTURE_MODE,
    }:
        if candidate.get("status") != "NOT_APPLICABLE" or not isinstance(
            candidate.get("reason"), str
        ):
            raise ValidationError("fleet policy run misstates shared-hold stability status")
    else:
        raise ValidationError("fleet private report has an unknown execution mode")
    if public.get("candidate_shared_hold") != candidate:
        raise ValidationError("fleet public/private candidate stability summaries differ")
    policy_progress = private.get("policy_progress")
    if not isinstance(policy_progress, dict) or public.get("policy_progress") != policy_progress:
        raise ValidationError("fleet public/private policy-progress summaries differ")
    required_progress = {
        "status",
        "observe_action_count",
        "confirmation_receipt_count",
        "return_action_count",
        "all_returned_home",
        "episode_budget_completed",
    }
    if set(policy_progress) != required_progress:
        raise ValidationError("fleet policy-progress summary fields differ")
    if (
        not isinstance(policy_progress["observe_action_count"], int)
        or isinstance(policy_progress["observe_action_count"], bool)
        or policy_progress["observe_action_count"] < 0
        or not isinstance(policy_progress["confirmation_receipt_count"], int)
        or isinstance(policy_progress["confirmation_receipt_count"], bool)
        or policy_progress["confirmation_receipt_count"] < 0
        or not isinstance(policy_progress["return_action_count"], int)
        or isinstance(policy_progress["return_action_count"], bool)
        or policy_progress["return_action_count"] < 0
        or not isinstance(policy_progress["all_returned_home"], bool)
        or not isinstance(policy_progress["episode_budget_completed"], bool)
    ):
        raise ValidationError("fleet policy-progress summary has invalid values")
    if execution_mode == "shared-hold":
        expected_progress_status = "NOT_APPLICABLE"
    elif execution_mode == PRIVATE_WITNESS_FIXTURE_MODE:
        fixture_commitment = private.get("private_fixture_commitment")
        if not isinstance(fixture_commitment, str) or not _SHA256.fullmatch(fixture_commitment):
            raise ValidationError("private-witness fixture lacks a valid commitment")
        if public.get("private_fixture_commitment") != fixture_commitment:
            raise ValidationError("public/private private-witness commitments differ")
        closed = (
            policy_progress["observe_action_count"] > 0
            and policy_progress["confirmation_receipt_count"] > 0
            and policy_progress["return_action_count"] > 0
            and policy_progress["all_returned_home"] is True
        )
        expected_progress_status = (
            "PRIVATE_FIXTURE_CLOSED" if closed else "PRIVATE_FIXTURE_INCOMPLETE"
        )
    else:
        timing_for_progress = private.get("planning_timing")
        final_for_progress = private.get("final")
        if execution_purpose == COMPLETE_CALIBRATION_PURPOSE and (
            not isinstance(timing_for_progress, dict) or not isinstance(final_for_progress, dict)
        ):
            raise ValidationError(
                "complete calibration replay lacks timing or final safety evidence"
            )
        failure_records_for_progress = private.get("failure_records")
        adapter_failure_count = (
            sum(
                isinstance(record, dict) and record.get("category") == "external_adapter_failure"
                for record in failure_records_for_progress
            )
            if isinstance(failure_records_for_progress, list)
            else 0
        )
        expected_progress_status = public_policy_progress_status(
            purpose=str(execution_purpose),
            observe_action_count=int(policy_progress["observe_action_count"]),
            confirmation_receipt_count=int(policy_progress["confirmation_receipt_count"]),
            return_action_count=int(policy_progress["return_action_count"]),
            all_returned_home=bool(policy_progress["all_returned_home"]),
            episode_budget_completed=bool(policy_progress["episode_budget_completed"]),
            safe_completion=(
                bool(final_for_progress.get("safe_completion"))
                if isinstance(final_for_progress, dict)
                else False
            ),
            deadline_miss_tick_count=(
                int(timing_for_progress.get("deadline_miss_tick_count", -1))
                if isinstance(timing_for_progress, dict)
                else 0
            ),
            adapter_failure_count=adapter_failure_count,
        )
    if policy_progress["status"] != expected_progress_status:
        raise ValidationError("fleet policy-progress status is inconsistent with action evidence")
    execution = private.get("execution")
    if not isinstance(execution, dict):
        raise ValidationError("fleet private report lacks execution summary")
    control_ticks = int(execution.get("control_ticks", -1))
    physical_steps_per_control = int(execution.get("physical_steps_per_control", -1))
    shared_physx_step_count = int(execution.get("shared_physx_step_count", -1))
    if control_ticks <= 0 or physical_steps_per_control <= 0:
        raise ValidationError("fleet execution has invalid control or physical step counts")
    if shared_physx_step_count != control_ticks * physical_steps_per_control:
        raise ValidationError("fleet shared PhysX step count disagrees with control timing")
    members = private.get("fleet_members_private")
    if not isinstance(members, list):
        raise ValidationError("fleet private report lacks fleet roster")
    normalized_members = tuple(
        FleetMember(
            drone_id=str(item["drone_id"]),
            start_position_w_m=_finite_position(item["start_position_w_m"], "fleet member start"),
            start_yaw_deg=float(item["start_yaw_deg"]),
        )
        for item in members
        if isinstance(item, dict)
    )
    if len(normalized_members) != FLEET_SIZE:
        raise ValidationError("fleet private report does not preserve four public members")
    route_budget_audit = private.get("route_budget_audit")
    if public.get("route_budget_audit") != route_budget_audit:
        raise ValidationError("fleet public/private route budget audits differ")
    _validate_route_budget_audit(
        route_budget_audit, members=normalized_members, execution_mode=execution_mode
    )
    planning_timing = private.get("planning_timing")
    if public.get("planning_timing") != planning_timing:
        raise ValidationError("fleet public/private planning timing summaries differ")
    _validate_planning_timing(
        planning_timing, expected_control_ticks=control_ticks, execution_mode=execution_mode
    )
    if execution_purpose == COMPLETE_CALIBRATION_PURPOSE:
        validate_complete_calibration_summary(
            execution_mode=execution_mode,
            policy_progress=policy_progress,
            planning_timing=planning_timing,
            private_final=private.get("final"),
            public_final=public.get("final"),
            private_execution=execution,
            public_execution=public.get("execution"),
            input_bindings=private.get("input_bindings"),
            public_input_bindings=public.get("input_bindings"),
            method=private.get("method"),
            public_method=public.get("method"),
            expected_drone_ids={member.drone_id for member in normalized_members},
            failure_records=private.get("failure_records"),
            allow_execution_failure=allow_execution_failure,
        )
        measurement_snapshot = private.get("measurement_evidence")
        measured_state_trace = private.get("measured_state_trace_private")
        input_bindings = private.get("input_bindings")
        if not isinstance(input_bindings, dict):
            raise ValidationError("complete calibration lacks measurement input bindings")
        try:
            validate_measurement_evidence_snapshot(
                measurement_snapshot,
                measured_state_trace=measured_state_trace,
                input_bindings_hash=content_hash(input_bindings),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"complete calibration measurement evidence is invalid: {exc}"
            ) from exc
    receipts = private.get("execution_receipts")
    if not isinstance(receipts, list):
        raise ValidationError("fleet private report lacks execution receipts")
    assert_canonical_fleet_receipts(
        receipts, normalized_members, expected_control_ticks=control_ticks
    )
    observed_actions = sum(receipt["action_requested"] == "OBSERVE" for receipt in receipts)
    return_actions = sum(receipt["action_requested"] == "RETURN" for receipt in receipts)
    if policy_progress["observe_action_count"] != observed_actions:
        raise ValidationError("fleet progress OBSERVE count disagrees with execution receipts")
    if policy_progress["return_action_count"] != return_actions:
        raise ValidationError("fleet progress RETURN count disagrees with execution receipts")
    bindings = private.get("execution_bindings_public")
    if not isinstance(bindings, list):
        raise ValidationError("fleet private report lacks action and observation bindings")
    assert_fleet_execution_bindings(receipts, bindings)
    _validate_planning_receipt_consistency(planning_timing, receipts, bindings)
    observation_receipts = private.get("observation_receipts")
    confirmation_receipts = private.get("confirmation_receipts")
    if not isinstance(observation_receipts, list) or not isinstance(confirmation_receipts, list):
        raise ValidationError("fleet private report lacks evaluator receipt evidence")
    if policy_progress["confirmation_receipt_count"] != len(confirmation_receipts):
        raise ValidationError("fleet progress confirmation count disagrees with evaluator receipts")
    assert_fleet_confirmation_bindings(
        receipts, bindings, observation_receipts, confirmation_receipts
    )
    final_state = private.get("final")
    has_execution_failure = execution_purpose == COMPLETE_CALIBRATION_PURPOSE and (
        policy_progress.get("status") != "CALIBRATION_EPISODE_CLOSED"
        or int(planning_timing.get("deadline_miss_tick_count", 0)) > 0
        or bool(private.get("failure_records"))
        or not isinstance(final_state, dict)
        or not bool(final_state.get("safe_completion", False))
    )
    if has_execution_failure and not allow_execution_failure:
        # This is normally unreachable because the strict complete-summary
        # validator rejects it; retain the guard if another failure class is
        # added later.
        raise ValidationError("execution failure requires the aggregation-only validation path")
    return {
        "status": "PASS_WITH_EXECUTION_FAILURE" if has_execution_failure else "PASS",
        "control_ticks": control_ticks,
        "shared_physx_step_count": shared_physx_step_count,
        "public_report_file_sha256": file_hash(public_path),
        "private_report_file_sha256": file_hash(private_path),
        "execution_receipt_set_sha256": content_hash(receipts),
    }
