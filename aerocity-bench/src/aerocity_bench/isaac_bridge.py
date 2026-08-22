"""Isaac L1/L2 compatibility contract and native evidence validation.

This module intentionally does not import Isaac Sim at package import time.
Launching the simulator remains an explicit CLI/tool action so CPU-only schema,
generation, adapter, and evaluator tests stay usable.
"""

from __future__ import annotations

import importlib.util
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Protocol

from .canonical import content_hash, file_hash, read_json, write_json
from .contracts import ACTION_KINDS, ActionPacket, ExecutionReceipt, ObservationPacket
from .errors import ValidationError
from .geometry import distance

REQUIRED_NATIVE_CHECKS = (
    "stage_load",
    "collider_count",
    "visual_collision_isolation",
    "physics_step",
    "ray_los_agreement",
    "fov_agreement",
    "observe_dwell",
    "velocity_tracking",
    "braking_distance",
    "reset_isolation",
    "deterministic_replay_tolerance",
)
FORMAL_L1_EVIDENCE_SCOPE = "formal_l1_episode_execution_receipt_evaluator"
CAPABILITY_L1_EVIDENCE_SCOPE = "canonical_l1_dynamic_vertical_slice_not_formal_episode_score"

_TARGET_INSTANCE_PATTERN = re.compile(r"(?:^|[^a-z0-9])target[_-](\d{3})(?:[^0-9]|$)", re.I)
_START_INSTANCE_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:drone[_-]?start|start)[_-](\d{3})(?:[^0-9]|$)", re.I
)

REVIEW_OVERVIEW_FRAMES = (
    "overview_ne",
    "overview_nw",
    "overview_se",
    "overview_sw",
    "top",
)
REVIEW_LOW_FRAMES = ("north_low", "south_low", "east_low", "west_low")
REVIEW_BASE_FRAMES = (*REVIEW_OVERVIEW_FRAMES, *REVIEW_LOW_FRAMES, "starts_close")
VISUAL_REVIEW_EVIDENCE_SCOPE = (
    "scene_wide_and_per_target_local_instance_bound_visual_review_overlay_"
    "not_target_occlusion_or_formal_score"
)


def _review_labels(value: object) -> set[str]:
    """Extract stable review instance labels from Replicator metadata."""
    strings: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            for key, nested in item.items():
                strings.append(str(key))
                visit(nested)
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)

    visit(value)
    labels: set[str] = set()
    for text in strings:
        for match in _TARGET_INSTANCE_PATTERN.finditer(text):
            labels.add(f"target_{int(match.group(1)):03d}")
        for match in _START_INSTANCE_PATTERN.finditer(text):
            labels.add(f"drone_start_{int(match.group(1)):03d}")
    return labels


def aggregate_review_instance_visibility(
    frames: dict[str, dict[str, object]],
    *,
    target_count: int,
    start_count: int,
    minimum_pixels: int = 24,
    minimum_local_pixels: int = 64,
    frame_pixel_count: int | None = None,
    maximum_local_fraction: float = 0.20,
) -> dict[str, object]:
    """Require scene-wide plus identity-bound local evidence for review instances.

    The fixed overview cameras are deliberately scene-level evidence: a target
    on the hidden side of a legitimate facade need not appear in one of those
    fixed viewpoints.  Each target instead needs an identity-bound marker in
    its own private local-context frame.  The fixed-view visibility count is
    retained as a diagnostic so a sparse or obstructed global review remains
    inspectable without falsely rejecting valid geometry.
    """
    if (
        target_count <= 0
        or start_count <= 0
        or minimum_pixels <= 0
        or minimum_local_pixels <= 0
        or (frame_pixel_count is not None and frame_pixel_count <= 0)
        or not 0.0 < maximum_local_fraction < 1.0
    ):
        raise ValueError("review instance counts and minimum_pixels must be positive")
    expected_targets = [f"target_{index:03d}" for index in range(target_count)]
    expected_starts = [
        f"drone_start_{index:03d}" for index in range(start_count)
    ]
    expected = expected_targets + expected_starts
    totals = {label: 0 for label in expected}
    by_frame: dict[str, dict[str, int]] = {}
    ambiguous_ids: dict[str, list[str]] = {}
    for frame_name, frame in sorted(frames.items()):
        raw_counts = frame.get("id_pixel_counts", {})
        raw_labels = frame.get("id_to_labels", {})
        raw_semantics = frame.get("id_to_semantics", {})
        if not isinstance(raw_counts, dict) or not isinstance(raw_labels, dict):
            raise ValueError(f"invalid instance segmentation metadata for {frame_name}")
        frame_counts = {label: 0 for label in expected}
        for raw_id, raw_count in raw_counts.items():
            identifier = str(raw_id)
            labels = _review_labels(raw_labels.get(identifier, raw_labels.get(raw_id)))
            if isinstance(raw_semantics, dict):
                labels.update(
                    _review_labels(raw_semantics.get(identifier, raw_semantics.get(raw_id)))
                )
            relevant = sorted(labels & totals.keys())
            if len(relevant) > 1:
                ambiguous_ids[f"{frame_name}:{identifier}"] = relevant
                continue
            if len(relevant) == 1:
                count = int(raw_count)
                if count < 0:
                    raise ValueError("instance pixel counts cannot be negative")
                frame_counts[relevant[0]] += count
                totals[relevant[0]] += count
        by_frame[frame_name] = {key: value for key, value in frame_counts.items() if value}
    checks: dict[str, dict[str, object]] = {}
    unseen_in_scene_overviews: list[str] = []
    missing_local_targets: list[str] = []
    oversized_local_targets: list[str] = []
    missing_start_close: list[str] = []
    missing_start_local: list[str] = []
    for label in expected_targets:
        target_index = label.removeprefix("target_")
        own_local_frame = f"target_close_{target_index}"
        overview_counts = {
            name: int(by_frame.get(name, {}).get(label, 0)) for name in REVIEW_OVERVIEW_FRAMES
        }
        local_counts = {
            name: int(counts.get(label, 0))
            for name, counts in by_frame.items()
            if name in REVIEW_LOW_FRAMES or name.startswith("target_close_")
        }
        overview_best = max(overview_counts.values(), default=0)
        local_best = max(local_counts.values(), default=0)
        own_local_pixels = int(by_frame.get(own_local_frame, {}).get(label, 0))
        own_local_fraction = (
            None if frame_pixel_count is None else own_local_pixels / frame_pixel_count
        )
        if overview_best < minimum_pixels:
            unseen_in_scene_overviews.append(label)
        if own_local_pixels < minimum_local_pixels:
            missing_local_targets.append(label)
        if own_local_fraction is not None and own_local_fraction > maximum_local_fraction:
            oversized_local_targets.append(label)
        passed = (
            totals[label] >= minimum_pixels
            and own_local_pixels >= minimum_local_pixels
            and (
                own_local_fraction is None or own_local_fraction <= maximum_local_fraction
            )
        )
        checks[label] = {
            "required_total_pixels": minimum_pixels,
            "observed_pixels": totals[label],
            "scene_overview_best_pixels": overview_best,
            "scene_overview_visible": overview_best >= minimum_pixels,
            "required_local_pixels_in_one_frame": minimum_local_pixels,
            "best_local_pixels": local_best,
            "own_local_frame": own_local_frame,
            "own_local_pixels": own_local_pixels,
            "maximum_local_fraction": maximum_local_fraction,
            "own_local_fraction": (
                None if own_local_fraction is None else round(own_local_fraction, 8)
            ),
            "qualifying_frame_count": sum(
                int(counts.get(label, 0)) >= minimum_pixels for counts in by_frame.values()
            ),
            "status": "PASS" if passed else "FAIL",
        }
    for label in expected_starts:
        start_close_pixels = int(by_frame.get("starts_close", {}).get(label, 0))
        start_index = label.removeprefix("drone_start_")
        start_local_pixels = int(by_frame.get(f"start_close_{start_index}", {}).get(label, 0))
        if start_close_pixels < minimum_pixels and start_local_pixels < minimum_pixels:
            missing_start_close.append(label)
        if start_local_pixels < minimum_pixels:
            missing_start_local.append(label)
        passed = totals[label] >= minimum_pixels and max(
            start_close_pixels, start_local_pixels
        ) >= minimum_pixels
        checks[label] = {
            "required_total_pixels": minimum_pixels,
            "observed_pixels": totals[label],
            "required_starts_close_pixels": minimum_pixels,
            "starts_close_pixels": start_close_pixels,
            "required_start_local_pixels": minimum_pixels,
            "start_local_pixels": start_local_pixels,
            "status": "PASS" if passed else "FAIL",
        }
    missing = [label for label, check in checks.items() if check["status"] != "PASS"]
    status = "PASS" if not missing and not ambiguous_ids else "FAIL"
    return {
        "status": status,
        "method": "replicator_instance_segmentation_per_review_identity",
        "scope": (
            "scene_wide_and_per_instance_local_review_overlay_visibility_"
            "not_target_occlusion_or_formal_score"
        ),
        "minimum_pixels": minimum_pixels,
        "minimum_local_pixels": minimum_local_pixels,
        "frame_pixel_count": frame_pixel_count,
        "maximum_local_fraction": maximum_local_fraction,
        "expected_target_count": target_count,
        "expected_start_count": start_count,
        "verified_instance_count": len(expected) - len(missing),
        "totals": totals,
        "by_frame": by_frame,
        "checks": checks,
        "missing_instances": missing,
        "unseen_in_scene_overviews": unseen_in_scene_overviews,
        "missing_local_targets": missing_local_targets,
        "oversized_local_targets": oversized_local_targets,
        "missing_starts_close": missing_start_close,
        "missing_start_local": missing_start_local,
        "ambiguous_instance_ids": ambiguous_ids,
    }


def probe_isaac_runtime() -> dict[str, Any]:
    return {
        "isaacsim_package": importlib.util.find_spec("isaacsim") is not None,
        "isaaclab_package": importlib.util.find_spec("isaaclab") is not None,
        "omni_loaded_without_simulation_app": importlib.util.find_spec("omni") is not None,
        "launch_required_for_native_modules": True,
    }


class NativeIsaacBackend(Protocol):
    """Boundary an Isaac extension must implement for formal L1 execution."""

    def load_stage(self, stage_path: Path) -> dict[str, Any]: ...

    def reset_episode(self, public_episode: dict[str, Any]) -> dict[str, Any]: ...

    def step(
        self, action_packets: list[dict[str, Any]], control_period_s: float
    ) -> dict[str, Any]: ...

    def raycast(
        self, origin: tuple[float, float, float], target: tuple[float, float, float]
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class NativeEvidence:
    report_path: Path
    report_hash: str
    execution_level: str
    runtime_fingerprint: dict[str, str]
    input_bindings: dict[str, str]
    formal_score_eligible: bool
    evidence_scope: str


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class FormalExecutionContext:
    """Trusted, in-memory binding between validated native evidence and one run."""

    episode_id: str
    layout_id: str
    execution_contract_hash: str
    native_gate_hash: str
    runtime_fingerprint_hash: str
    execution_receipt_set_hash: str


def formal_execution_context(
    evidence: NativeEvidence, execution_receipt_set_hash: str
) -> FormalExecutionContext:
    if evidence.execution_level != "L1":
        raise ValidationError("formal geometry scoring requires validated L1 native evidence")
    if (
        not evidence.formal_score_eligible
        or evidence.evidence_scope != FORMAL_L1_EVIDENCE_SCOPE
    ):
        raise ValidationError("native evidence is a capability gate, not formal episode evidence")
    if not _is_sha256(execution_receipt_set_hash):
        raise ValidationError("formal execution receipt-set hash is invalid")
    return FormalExecutionContext(
        episode_id=str(evidence.input_bindings["episode_id"]),
        layout_id=str(evidence.input_bindings["layout_id"]),
        execution_contract_hash=str(evidence.input_bindings["execution_contract_hash"]),
        native_gate_hash=evidence.report_hash,
        runtime_fingerprint_hash=content_hash(evidence.runtime_fingerprint),
        execution_receipt_set_hash=execution_receipt_set_hash,
    )


def build_l1_execution_receipt(
    *,
    action: ActionPacket,
    source_observation: ObservationPacket,
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    task_time_start_s: float,
    task_time_end_s: float,
    planning_latency_s: float,
    action_executed: str,
    status: str,
    energy_used_j: float,
    minimum_clearance_m: float | None,
    collision: bool,
    out_of_bounds: bool,
    safety_intervention: bool,
    deadline_miss: bool,
    previous_receipt_hash: str | None,
    planner_invoked: bool = True,
    confirmation_ids: tuple[str, ...] = (),
) -> ExecutionReceipt:
    """Create one L1 receipt from measured native before/after state snapshots."""

    if (
        action.episode_id != source_observation.episode_id
        or action.drone_id != source_observation.drone_id
        or action.sequence != source_observation.sequence
        or abs(float(action.issued_at_s) - float(source_observation.timestamp_s)) > 1.0e-9
    ):
        raise ValidationError("native action is not bound to its source observation")
    if action_executed not in ACTION_KINDS:
        raise ValidationError("native receipt names an invalid executed action")
    try:
        before_position = tuple(float(value) for value in state_before["position"])
        after_position = tuple(float(value) for value in state_after["position"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("native state snapshot lacks a numeric position") from exc
    if len(before_position) != 3 or len(after_position) != 3:
        raise ValidationError("native state position must be a three-vector")
    return ExecutionReceipt(
        episode_id=action.episode_id,
        drone_id=action.drone_id,
        action_sequence=action.sequence,
        task_time_start_s=task_time_start_s,
        task_time_end_s=task_time_end_s,
        planning_latency_s=planning_latency_s,
        action_requested=action.kind,
        action_executed=action_executed,
        status=status,
        distance_m=distance(before_position, after_position),
        energy_used_j=energy_used_j,
        minimum_clearance_m=minimum_clearance_m,
        collision=collision,
        out_of_bounds=out_of_bounds,
        safety_intervention=safety_intervention,
        deadline_miss=deadline_miss,
        execution_level="L1",
        action_packet_hash=content_hash(action.to_dict()),
        source_observation_id=source_observation.observation_id,
        source_observation_hash=content_hash(source_observation.to_dict()),
        state_before_hash=content_hash(state_before),
        state_after_hash=content_hash(state_after),
        previous_receipt_hash=previous_receipt_hash,
        confirmation_ids=confirmation_ids,
        planner_invoked=planner_invoked,
    )


def write_native_gate_report(
    destination: Path,
    *,
    stage_path: Path,
    execution_level: str,
    runtime_fingerprint: dict[str, str],
    checks: dict[str, dict[str, Any]],
    input_bindings: dict[str, str] | None = None,
    formal_score_eligible: bool = False,
    evidence_scope: str = CAPABILITY_L1_EVIDENCE_SCOPE,
) -> dict[str, Any]:
    if execution_level not in {"L1", "L2", "L1-preflight"}:
        raise ValueError("native gate evidence must be L1, L2, or L1-preflight")
    if set(checks) != set(REQUIRED_NATIVE_CHECKS):
        missing = sorted(set(REQUIRED_NATIVE_CHECKS) - set(checks))
        extra = sorted(set(checks) - set(REQUIRED_NATIVE_CHECKS))
        raise ValueError(f"native check fields differ; missing={missing}, extra={extra}")
    report = {
        "schema": "org.aerocity.bench.native-isaac-gate.v1",
        "execution_level": execution_level,
        "stage_path": str(stage_path.resolve()),
        "stage_sha256": file_hash(stage_path),
        "runtime_fingerprint": runtime_fingerprint,
        "checks": checks,
        "formal_score_eligible": formal_score_eligible,
        "evidence_scope": evidence_scope,
    }
    if input_bindings is not None:
        from .native_gate_contract import NATIVE_INPUT_BINDING_KEYS

        if set(input_bindings) != NATIVE_INPUT_BINDING_KEYS or any(
            not isinstance(value, str) or not value for value in input_bindings.values()
        ):
            raise ValueError("native gate input bindings are incomplete")
        report["input_bindings"] = dict(sorted(input_bindings.items()))
    report["native_gate_hash"] = content_hash(report)
    write_json(destination, report)
    return report


def validate_native_gate_report(
    path: Path,
    expected_stage: Path | None = None,
    expected_input_bindings: dict[str, str] | None = None,
) -> NativeEvidence:
    report = read_json(path)
    expected_hash = str(report.pop("native_gate_hash", ""))
    if content_hash(report) != expected_hash:
        raise ValidationError("native Isaac gate report hash mismatch")
    if report.get("execution_level") not in {"L1", "L2"}:
        raise ValidationError("native gate report is not L1/L2")
    if not isinstance(report.get("formal_score_eligible"), bool):
        raise ValidationError("native gate formal eligibility flag is invalid")
    if not isinstance(report.get("evidence_scope"), str) or not report["evidence_scope"]:
        raise ValidationError("native gate evidence scope is invalid")
    if report["formal_score_eligible"] and report["evidence_scope"] != FORMAL_L1_EVIDENCE_SCOPE:
        raise ValidationError("formal native gate uses a non-formal evidence scope")
    checks = report.get("checks", {})
    if set(checks) != set(REQUIRED_NATIVE_CHECKS):
        raise ValidationError("native gate report does not contain every required check")
    failed = sorted(name for name, result in checks.items() if result.get("status") != "PASS")
    if failed:
        raise ValidationError(f"native Isaac gate contains failed checks: {failed}")
    input_bindings = report.get("input_bindings", {})
    if not isinstance(input_bindings, dict):
        raise ValidationError("native Isaac gate input bindings are invalid")
    normalized_bindings = {str(key): str(value) for key, value in input_bindings.items()}
    if expected_input_bindings is not None and normalized_bindings != expected_input_bindings:
        raise ValidationError("native Isaac gate belongs to different public inputs")
    reported_stage = Path(str(report["stage_path"]))
    if reported_stage.is_absolute():
        stage = reported_stage.resolve()
        if expected_stage is not None and stage != expected_stage.resolve():
            raise ValidationError("native gate report belongs to another stage")
    elif expected_stage is not None:
        stage = expected_stage.resolve()
        reported_parts = PurePath(reported_stage.as_posix()).parts
        if tuple(stage.parts[-len(reported_parts) :]) != reported_parts:
            raise ValidationError("relative native gate stage does not match the expected stage")
    else:
        raise ValidationError("relative native gate evidence requires an expected package stage")
    if not stage.is_file() or file_hash(stage) != report.get("stage_sha256"):
        raise ValidationError("native gate stage is absent or changed")
    return NativeEvidence(
        report_path=path.resolve(),
        report_hash=expected_hash,
        execution_level=str(report["execution_level"]),
        runtime_fingerprint={
            str(key): str(value) for key, value in report["runtime_fingerprint"].items()
        },
        input_bindings=normalized_bindings,
        formal_score_eligible=bool(report["formal_score_eligible"]),
        evidence_scope=str(report["evidence_scope"]),
    )


def assert_formal_receipts(
    receipts: list[dict[str, Any]],
    *,
    context: FormalExecutionContext,
    expected_drone_ids: set[str],
    expected_confirmation_ids: set[str],
    expected_task_time_s: float,
    ledger: dict[str, Any],
) -> None:
    """Validate the complete L1 receipt chain against trusted in-memory evidence."""

    if not receipts:
        raise ValidationError("formal scoring requires execution receipts")
    if content_hash(receipts) != context.execution_receipt_set_hash:
        raise ValidationError("formal execution receipts differ from the trusted receipt set")
    required_v2 = {
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
    canonical_order = sorted(
        receipts, key=lambda item: (int(item.get("action_sequence", -1)), str(item.get("drone_id")))
    )
    if receipts != canonical_order:
        raise ValidationError("formal execution receipts are not in canonical step/agent order")
    seen_pairs: set[tuple[str, int]] = set()
    last_by_drone: dict[str, dict[str, Any]] = {}
    receipt_drone_ids: set[str] = set()
    confirmation_ids: list[str] = []
    receipt_hashes: dict[tuple[str, int], str] = {}
    for receipt in receipts:
        payload = dict(receipt)
        expected_hash = str(payload.pop("receipt_hash", ""))
        schema = receipt.get("schema")
        required = (
            required_v2 | {"planner_invoked"}
            if schema == "org.aerocity.bench.execution-receipt.v3"
            else required_v2
        )
        if set(receipt) != required:
            missing = sorted(required - set(receipt))
            extra = sorted(set(receipt) - required)
            raise ValidationError(
                f"formal execution receipt fields differ; missing={missing}, extra={extra}"
            )
        if content_hash(payload) != expected_hash:
            raise ValidationError(
                "formal scoring received a corrupt execution receipt: "
                f"{receipt.get('drone_id')}/{receipt.get('action_sequence')}"
            )
        if schema not in {
            "org.aerocity.bench.execution-receipt.v2",
            "org.aerocity.bench.execution-receipt.v3",
        }:
            raise ValidationError("formal scoring requires a supported execution receipt")
        if schema == "org.aerocity.bench.execution-receipt.v3" and not isinstance(
            receipt.get("planner_invoked"), bool
        ):
            raise ValidationError("formal receipt planner invocation flag is invalid")
        if receipt.get("execution_level") != "L1":
            raise ValidationError("formal scoring received a non-L1 execution receipt")
        if receipt.get("episode_id") != context.episode_id:
            raise ValidationError("formal execution receipt belongs to another episode")
        drone_id = str(receipt.get("drone_id", ""))
        if drone_id not in expected_drone_ids:
            raise ValidationError(f"formal execution receipt names an unknown drone: {drone_id}")
        receipt_drone_ids.add(drone_id)
        sequence = receipt.get("action_sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValidationError("formal execution receipt sequence is invalid")
        pair = (drone_id, sequence)
        if pair in seen_pairs:
            raise ValidationError(f"duplicate formal execution receipt: {drone_id}/{sequence}")
        seen_pairs.add(pair)
        if receipt.get("action_requested") not in ACTION_KINDS or receipt.get(
            "action_executed"
        ) not in ACTION_KINDS:
            raise ValidationError("formal execution receipt contains an invalid action kind")
        numeric_keys = (
            "task_time_start_s",
            "task_time_end_s",
            "planning_latency_s",
            "distance_m",
            "energy_used_j",
        )
        if any(
            not isinstance(receipt.get(key), (int, float))
            or isinstance(receipt.get(key), bool)
            or not math.isfinite(float(receipt[key]))
            or float(receipt[key]) < 0.0
            for key in numeric_keys
        ):
            raise ValidationError("formal execution receipt contains invalid numeric evidence")
        if float(receipt["task_time_end_s"]) < float(receipt["task_time_start_s"]):
            raise ValidationError("formal execution receipt time runs backwards")
        clearance = receipt.get("minimum_clearance_m")
        if clearance is not None and (
            not isinstance(clearance, (int, float))
            or isinstance(clearance, bool)
            or not math.isfinite(float(clearance))
            or float(clearance) < 0.0
        ):
            raise ValidationError("formal execution receipt clearance is invalid")
        for key in (
            "action_packet_hash",
            "source_observation_hash",
            "state_before_hash",
            "state_after_hash",
        ):
            if not _is_sha256(receipt.get(key)):
                raise ValidationError(f"formal execution receipt has invalid {key}")
        if not isinstance(receipt.get("source_observation_id"), str) or not receipt[
            "source_observation_id"
        ]:
            raise ValidationError("formal execution receipt lacks a source observation ID")
        previous = last_by_drone.get(drone_id)
        if previous is None:
            if sequence != 0 or receipt.get("previous_receipt_hash") is not None:
                raise ValidationError(
                    "formal execution receipt chain does not start at sequence zero"
                )
            if abs(float(receipt["task_time_start_s"])) > 1.0e-9:
                raise ValidationError(
                    "formal execution receipt chain does not start at task time zero"
                )
        else:
            if sequence != int(previous["action_sequence"]) + 1:
                raise ValidationError("formal execution receipt chain skips an action sequence")
            previous_pair = (drone_id, int(previous["action_sequence"]))
            if receipt.get("previous_receipt_hash") != receipt_hashes[previous_pair]:
                raise ValidationError("formal execution receipt previous hash is broken")
            if receipt.get("state_before_hash") != previous.get("state_after_hash"):
                raise ValidationError("formal execution state hash chain is broken")
            if abs(
                float(receipt["task_time_start_s"]) - float(previous["task_time_end_s"])
            ) > 1.0e-9:
                raise ValidationError("formal execution receipt time chain is discontinuous")
        current_confirmation_ids = receipt.get("confirmation_ids")
        if not isinstance(current_confirmation_ids, list) or any(
            not isinstance(value, str) or not value for value in current_confirmation_ids
        ):
            raise ValidationError("formal execution receipt confirmation IDs are invalid")
        confirmation_ids.extend(current_confirmation_ids)
        receipt_hashes[pair] = expected_hash
        last_by_drone[drone_id] = receipt
    if receipt_drone_ids != expected_drone_ids:
        raise ValidationError("formal execution receipt set omits one or more expected drones")
    if len(confirmation_ids) != len(set(confirmation_ids)):
        raise ValidationError("formal execution receipts duplicate a confirmation ID")
    if set(confirmation_ids) != expected_confirmation_ids:
        raise ValidationError("formal execution receipts do not bind the evaluator confirmations")
    maximum_end = max(float(receipt["task_time_end_s"]) for receipt in receipts)
    if abs(maximum_end - float(expected_task_time_s)) > 1.0e-9:
        raise ValidationError("formal execution receipts do not cover the reported task time")
    required_ledger_keys = {
        "path_distance_m",
        "energy_used_j",
        "planning_time_s",
        "collisions",
        "out_of_bounds_actions",
        "safety_interventions",
        "deadline_misses",
    }
    missing_ledger = sorted(key for key in required_ledger_keys if key not in ledger)
    if missing_ledger:
        raise ValidationError(
            f"formal execution budget ledger omits required fields: {missing_ledger}"
        )
    for key in ("path_distance_m", "energy_used_j", "planning_time_s"):
        value = ledger[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValidationError(
                f"formal execution budget ledger has invalid numeric field: {key}"
            )
    sums = {
        "path_distance_m": sum(float(receipt["distance_m"]) for receipt in receipts),
        "energy_used_j": sum(float(receipt["energy_used_j"]) for receipt in receipts),
        "planning_time_s": sum(float(receipt["planning_latency_s"]) for receipt in receipts),
    }
    for key, value in sums.items():
        if abs(value - float(ledger.get(key, math.nan))) > 1.0e-7:
            raise ValidationError(f"formal execution receipts disagree with budget ledger: {key}")
    counts = {
        "collisions": sum(bool(receipt["collision"]) for receipt in receipts),
        "out_of_bounds_actions": sum(bool(receipt["out_of_bounds"]) for receipt in receipts),
        "safety_interventions": sum(
            bool(receipt["safety_intervention"]) for receipt in receipts
        ),
        "deadline_misses": sum(bool(receipt["deadline_miss"]) for receipt in receipts),
    }
    for key, value in counts.items():
        if value != int(ledger.get(key, -1)):
            raise ValidationError(f"formal execution receipts disagree with budget ledger: {key}")
