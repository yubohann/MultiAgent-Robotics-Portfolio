"""Condition requests and evidence checks for collection-bound captures.

The collection protocol names a cell, but a cell is not evidence that Isaac
actually applied its conditions.  This module keeps the bridge deliberately
small: it reuses the existing protocol binding and receipt, records the cell's
public condition IDs, and evaluates only evidence that can be recomputed from
the raw capture.  Unsupported axes remain unavailable instead of being
counted from labels.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

from .citylite_scene import (
    PUBLIC_ROUTE_FAMILIES_W_M,
    START_ANCHOR_IDS_BY_ROUTE_FAMILY,
    TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M,
    canonical_payload_sha256,
)


CONDITION_REALIZATION_SCHEMA = "org.rivermark.isaac-condition-realization.v1"
CONDITION_AXES = (
    "layout",
    "route",
    "route_family",
    "target_count",
    "height",
    "region",
    "occlusion",
    "density",
    "appearance",
    "dynamics",
    "lighting",
    "weather",
    "initial_condition",
    "start_anchor",
    "target_region",
    "visibility_bucket",
    "communication",
    "control_latency",
    "agent_dropout",
)

# These are the only condition IDs that the current native capture already
# exposes enough raw evidence to check.  The names are intentionally stable
# public IDs, not user-facing claims about a broader condition matrix.
SUPPORTED_CONDITION_VALUES = {
    "layout": "citylite-v1",
    "route": "fixed-public-route-v1",
    "target_count": "object-count-4-v1",
    "height": "citylite-command-altitude-v1",
    "region": "citylite-command-volume-v1",
    "dynamics": "cf2x-nominal-v1",
    "initial_condition": "public-route-anchor-v1",
    "communication": "synchronous-public-broadcast-v1",
    "control_latency": "one-step-command-latency-v1",
    "agent_dropout": "no-agent-dropout-v1",
}
SUPPORTED_CONDITION_VALUE_SETS = {
    **{axis: frozenset((value,)) for axis, value in SUPPORTED_CONDITION_VALUES.items()},
    "route_family": frozenset(
        ("citylite-route-family-a-v1", "citylite-route-family-b-v1")
    ),
    "start_anchor": frozenset(
        ("citylite-start-anchor-a-v1", "citylite-start-anchor-b-v1")
    ),
    "target_region": frozenset(
        ("citylite-target-region-a-v1", "citylite-target-region-b-v1")
    ),
    "visibility_bucket": frozenset(("direct-visible-v1", "partial-visible-v1")),
}

# Keep this ordered like CONDITION_AXES.  The order is part of the human and
# machine-readable audit surface; a set would make receipt diffs unstable.
UNSUPPORTED_CONDITION_AXES = tuple(
    axis for axis in CONDITION_AXES if axis not in SUPPORTED_CONDITION_VALUE_SETS
)
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def condition_request_from_protocol(
    protocol: Mapping[str, Any],
    *,
    protocol_id: str,
    protocol_sha256: str,
    cell_id: str,
) -> dict[str, Any]:
    """Build the public condition request for an already resolved binding."""

    cells = protocol.get("cells")
    cell = next(
        (item for item in cells if isinstance(item, Mapping) and item.get("cell_id") == cell_id),
        None,
    ) if isinstance(cells, list) else None
    if not isinstance(cell, Mapping):
        raise ValueError(f"unknown collection cell: {cell_id}")
    conditions = cell.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("collection cell conditions must be an object")
    declared_axes = tuple(axis for axis in CONDITION_AXES if axis in conditions)
    if not declared_axes or set(conditions) != set(declared_axes):
        raise ValueError("collection cell conditions must be a non-empty subset of known axes")
    axis_support = {
        axis: (
            "supported_profile"
            if conditions.get(axis) in SUPPORTED_CONDITION_VALUE_SETS.get(axis, ())
            else "unavailable"
        )
        for axis in declared_axes
    }
    return {
        "schema": CONDITION_REALIZATION_SCHEMA,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "cell_id": cell_id,
        "conditions": {axis: conditions[axis] for axis in declared_axes},
        "axis_support": axis_support,
        "status": "pending_independent_check",
    }


def validate_condition_request(
    request: Any,
    *,
    binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], ...]:
    """Validate receipt metadata without trusting it as realization evidence."""

    issues: list[dict[str, str]] = []
    if not isinstance(request, Mapping):
        return (_issue("condition_request_type", "condition_request", "must be an object"),)
    expected = {"schema", "protocol_id", "protocol_sha256", "cell_id", "conditions", "axis_support", "status"}
    for key in sorted(set(request) - expected):
        issues.append(_issue("condition_request_unknown_field", f"condition_request.{key}", "field is not supported"))
    if request.get("schema") != CONDITION_REALIZATION_SCHEMA:
        issues.append(_issue("condition_request_schema", "condition_request.schema", "unsupported condition realization schema"))
    for key in ("protocol_id", "cell_id"):
        value = request.get(key)
        if not isinstance(value, str) or not _ID.fullmatch(value):
            issues.append(_issue("condition_request_id", f"condition_request.{key}", "must be a public identifier"))
    if not isinstance(request.get("protocol_sha256"), str) or not _SHA256.fullmatch(request.get("protocol_sha256", "")):
        issues.append(_issue("condition_request_hash", "condition_request.protocol_sha256", "must be SHA-256"))
    conditions = request.get("conditions")
    condition_axes = set(conditions) if isinstance(conditions, Mapping) else set()
    if (
        not isinstance(conditions, Mapping)
        or not condition_axes
        or not condition_axes.issubset(CONDITION_AXES)
    ):
        issues.append(_issue("condition_request_axes", "condition_request.conditions", "must contain a non-empty subset of known public axes"))
    else:
        for axis, value in conditions.items():
            if not isinstance(value, str) or not _ID.fullmatch(value):
                issues.append(_issue("condition_request_value", f"condition_request.conditions.{axis}", "must be a public condition identifier"))
    support = request.get("axis_support")
    if not isinstance(support, Mapping) or set(support) != condition_axes:
        issues.append(_issue("condition_request_support", "condition_request.axis_support", "must classify every declared axis"))
    else:
        for axis, status in support.items():
            if status not in {"supported_profile", "unavailable"}:
                issues.append(_issue("condition_request_support", f"condition_request.axis_support.{axis}", "unknown support status"))
                continue
            expected_status = (
                "supported_profile"
                if conditions.get(axis) in SUPPORTED_CONDITION_VALUE_SETS.get(axis, ())
                else "unavailable"
            )
            if status != expected_status:
                issues.append(_issue("condition_request_support", f"condition_request.axis_support.{axis}", "does not match the declared condition value"))
    if request.get("status") != "pending_independent_check":
        issues.append(_issue("condition_request_status", "condition_request.status", "must remain pending until independent verification"))
    if binding is not None:
        for key in ("protocol_id", "protocol_sha256", "cell_id"):
            if request.get(key) != binding.get(key):
                issues.append(_issue("condition_request_binding", f"condition_request.{key}", "does not match collection binding"))
    return tuple(issues)


def _finite_array(value: Any, shape_tail: tuple[int, ...] | None = None) -> bool:
    try:
        if shape_tail is not None and tuple(value.shape[1:]) != shape_tail:
            return False
        return bool(value.size and value.dtype.kind in "fiu" and np.isfinite(value).all())
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _public_route_evidence_ok(public_task: Mapping[str, Any] | None) -> bool:
    """Check the public route payload without trusting route metadata."""

    if not isinstance(public_task, Mapping) or public_task.get("route_conditioning") != "public_only":
        return False
    if public_task.get("agent_count") != 8:
        return False
    routes = public_task.get("routes_w_m")
    if not isinstance(routes, list) or len(routes) != 8:
        return False
    waypoint_count: int | None = None
    for route in routes:
        if not isinstance(route, list) or len(route) < 2:
            return False
        try:
            points = np.asarray(route, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            return False
        if waypoint_count is None:
            waypoint_count = int(points.shape[0])
        elif points.shape[0] != waypoint_count:
            return False
    return waypoint_count is not None


def _route_family_evidence_ok(
    public_task: Mapping[str, Any] | None,
    requested_family: Any,
) -> bool:
    if not isinstance(requested_family, str) or requested_family not in PUBLIC_ROUTE_FAMILIES_W_M:
        return False
    if not _public_route_evidence_ok(public_task) or not isinstance(public_task, Mapping):
        return False
    routes = public_task.get("routes_w_m")
    contract = public_task.get("route_contract")
    expected_routes = PUBLIC_ROUTE_FAMILIES_W_M[requested_family]
    expected_hash = canonical_payload_sha256(expected_routes)
    return bool(
        public_task.get("route_family_id") == requested_family
        and canonical_payload_sha256(routes) == expected_hash
        and isinstance(contract, Mapping)
        and contract.get("routes_sha256") == expected_hash
    )


def _start_anchor_evidence_ok(
    scene: Mapping[str, Any] | None,
    public_task: Mapping[str, Any] | None,
    requested_anchor: Any,
) -> bool:
    if not isinstance(public_task, Mapping) or public_task.get("start_anchor_id") != requested_anchor:
        return False
    family = public_task.get("route_family_id")
    if (
        not isinstance(family, str)
        or family not in START_ANCHOR_IDS_BY_ROUTE_FAMILY
        or START_ANCHOR_IDS_BY_ROUTE_FAMILY[family] != requested_anchor
        or not isinstance(scene, Mapping)
    ):
        return False
    try:
        poses = np.asarray(scene.get("initial_root_poses_wxyz"), dtype=np.float64)
        starts = np.asarray(TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[family], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        poses.shape == (8, 7)
        and starts.shape == (8, 3)
        and np.isfinite(poses).all()
        and np.allclose(poses[:, :3], starts, rtol=0.0, atol=1.0e-6)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _initial_condition_evidence_ok(
    scene: Mapping[str, Any] | None,
    checks: Mapping[str, Any],
) -> bool:
    if not isinstance(scene, Mapping) or checks.get("literal_fleet_spawn_verified") is not True:
        return False
    try:
        poses = np.asarray(scene.get("initial_root_poses_wxyz"), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return poses.shape == (8, 7) and bool(np.isfinite(poses).all())


def _synchronous_messages_ok(messages: Any) -> bool:
    if not isinstance(messages, Mapping):
        return False
    try:
        sender = np.asarray(messages.get("sender_agent_id"))
        flags = np.asarray(messages.get("message_flags"))
    except (TypeError, ValueError):
        return False
    if (
        sender.ndim != 2
        or sender.shape[1] != 8
        or sender.shape[0] < 1
        or not np.issubdtype(sender.dtype, np.integer)
        or not np.isfinite(sender).all()
        or flags.shape != sender.shape
        or not np.issubdtype(flags.dtype, np.integer)
    ):
        return False
    expected = np.broadcast_to(np.arange(8, dtype=sender.dtype), sender.shape)
    return bool(np.array_equal(sender, expected) and np.all(flags == 1))


def _one_step_latency_ok(state: Any, receipt: Mapping[str, Any]) -> bool:
    if not isinstance(state, Mapping) or "command_time_ns" not in state or "effective_time_ns" not in state:
        return False
    try:
        command = np.asarray(state["command_time_ns"])
        effective = np.asarray(state["effective_time_ns"])
    except (TypeError, ValueError):
        return False
    dt_s = receipt.get("command", {}).get("dt_s") if isinstance(receipt.get("command"), Mapping) else None
    if (
        command.ndim != 1
        or command.shape != effective.shape
        or command.size < 1
        or not np.issubdtype(command.dtype, np.integer)
        or not np.issubdtype(effective.dtype, np.integer)
        or not _finite_number(dt_s)
        or float(dt_s) <= 0.0
    ):
        return False
    expected_delta_ns = int(round(float(dt_s) * 1.0e9))
    return bool(np.all(effective - command == expected_delta_ns))


def evaluate_condition_realization(
    request: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    scene: Mapping[str, Any] | None,
    public_task: Mapping[str, Any] | None,
    state: Any,
    messages: Any,
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute condition evidence from already decoded capture artifacts."""

    conditions = request.get("conditions") if isinstance(request, Mapping) else {}
    axes: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []

    def add(axis: str, passed: bool, evidence: str, reason: str | None = None) -> None:
        value = conditions.get(axis) if isinstance(conditions, Mapping) else None
        row: dict[str, Any] = {"requested": value, "status": "verified" if passed else "unavailable", "evidence": evidence}
        if reason:
            row["reason"] = reason
            errors.append(_issue("condition_axis", f"condition_request.conditions.{axis}", reason))
        axes[axis] = row

    def requested(axis: str) -> bool:
        return isinstance(conditions, Mapping) and axis in conditions

    scene_ok = isinstance(scene, Mapping) and scene.get("environment_id") == "RIVERMARK_CITY_LITE_v1" and scene.get("fresh_stage") is True
    if requested("layout"):
        add("layout", conditions.get("layout") == SUPPORTED_CONDITION_VALUES["layout"] and scene_ok, "scene.json.environment_id/fresh_stage", "City-Lite layout evidence is missing or unsupported" if not scene_ok else None)
    route_ok = _public_route_evidence_ok(public_task)
    if requested("route"):
        add("route", conditions.get("route") == SUPPORTED_CONDITION_VALUES["route"] and route_ok, "public_task.json route_conditioning/routes_w_m", "public route evidence is missing or unsupported" if not route_ok else None)
    if requested("route_family"):
        family_ok = _route_family_evidence_ok(public_task, conditions.get("route_family"))
        add("route_family", family_ok, "public_task.json route_family_id/routes_w_m/route_contract.routes_sha256", "route family is not exactly realized by the public trajectory contract" if not family_ok else None)
    object_ok = (
        isinstance(public_task, Mapping)
        and public_task.get("nominal_object_count") == 4
        and isinstance(scene, Mapping)
        and scene.get("search_object_prim_count") == 4
    )
    if requested("target_count"):
        add("target_count", conditions.get("target_count") == SUPPORTED_CONDITION_VALUES["target_count"] and object_ok, "public_task.json.nominal_object_count + scene.json.search_object_prim_count", "aggregate object-count evidence is missing or unsupported" if not object_ok else None)

    position_ok = _finite_array(state.get("root_pos_w_m"), (8, 3)) if isinstance(state, Mapping) else False
    if position_ok:
        positions = state["root_pos_w_m"]
        altitude_ok = bool(((positions[..., 2] >= 9.0) & (positions[..., 2] <= 14.25)).all())
        region_ok = bool(((positions[..., 0] >= -46.0) & (positions[..., 0] <= 46.0) & (positions[..., 1] >= -48.0) & (positions[..., 1] <= 44.0)).all())
    else:
        altitude_ok = region_ok = False
    if requested("height"):
        add("height", conditions.get("height") == SUPPORTED_CONDITION_VALUES["height"] and altitude_ok, "streams/state_action.npz.root_pos_w_m.z + City-Lite command volume", "trajectory altitude evidence is missing or outside the command volume" if not altitude_ok else None)
    if requested("region"):
        add("region", conditions.get("region") == SUPPORTED_CONDITION_VALUES["region"] and region_ok, "streams/state_action.npz.root_pos_w_m.xy + City-Lite command volume", "trajectory region evidence is missing or outside the command volume" if not region_ok else None)

    physics = receipt.get("physics") if isinstance(receipt, Mapping) else None
    trim = physics.get("cf2x_hover_trim") if isinstance(physics, Mapping) else None
    dynamics_ok = (
        isinstance(trim, Mapping)
        and all(_finite_number(trim.get(key)) for key in ("hover_thrust_per_rotor_n", "initial_hover_rps"))
        and checks.get("literal_fleet_spawn_verified") is True
    )
    if requested("dynamics"):
        add("dynamics", conditions.get("dynamics") == SUPPORTED_CONDITION_VALUES["dynamics"] and dynamics_ok, "capture_receipt.json.physics + literal_fleet_spawn_verified", "CF2X dynamics evidence is missing or unverified" if not dynamics_ok else None)
    initial_ok = _initial_condition_evidence_ok(scene, checks)
    if requested("initial_condition"):
        add("initial_condition", conditions.get("initial_condition") == SUPPORTED_CONDITION_VALUES["initial_condition"] and initial_ok, "scene.json.initial_root_poses_wxyz + literal fleet reset audit", "initial-condition evidence is missing or unverified" if not initial_ok else None)
    if requested("start_anchor"):
        start_ok = _start_anchor_evidence_ok(
            scene, public_task, conditions.get("start_anchor")
        ) and checks.get("literal_fleet_spawn_verified") is True
        add("start_anchor", start_ok, "scene.json.initial_root_poses_wxyz + public_task.json route family", "start anchor is not the independently reconstructed route-family anchor" if not start_ok else None)
    if requested("target_region"):
        target_region_ok = (
            checks.get("private_target_region_verified") is True
            and checks.get("private_target_region_id") == conditions.get("target_region")
        )
        add("target_region", target_region_ok, "external evaluator targets + City-Lite target-region bounds", "private target positions do not independently realize the requested target region" if not target_region_ok else None)
    if requested("visibility_bucket"):
        visibility_ok = (
            checks.get("private_target_visibility_verified") is True
            and checks.get("private_target_visibility_bucket")
            == conditions.get("visibility_bucket")
        )
        add("visibility_bucket", visibility_ok, "external evaluator targets + public routes + structural AABB visibility recomputation", "target visibility bucket is unavailable or differs from independently recomputed geometry" if not visibility_ok else None)

    message_ok = _synchronous_messages_ok(messages)
    if requested("communication"):
        add("communication", conditions.get("communication") == SUPPORTED_CONDITION_VALUES["communication"] and message_ok, "streams/public_messages.npz sender/message_flags", "synchronous communication evidence is missing or incomplete" if not message_ok else None)
    latency_ok = _one_step_latency_ok(state, receipt)
    if requested("control_latency"):
        add("control_latency", conditions.get("control_latency") == SUPPORTED_CONDITION_VALUES["control_latency"] and latency_ok, "streams/state_action.npz command/effective timestamps", "command latency is missing or not exactly one physics step" if not latency_ok else None)
    dropout_ok = position_ok and message_ok
    if requested("agent_dropout"):
        add("agent_dropout", conditions.get("agent_dropout") == SUPPORTED_CONDITION_VALUES["agent_dropout"] and dropout_ok, "state_action/public_messages agent dimensions", "agent presence cannot be reconstructed for every sample" if not dropout_ok else None)

    for axis in UNSUPPORTED_CONDITION_AXES:
        if requested(axis):
            add(axis, False, "no raw Isaac realization stream", "axis executor and independent realization evidence are not implemented")

    verified = [axis for axis, row in axes.items() if row["status"] == "verified"]
    unavailable = [axis for axis, row in axes.items() if row["status"] != "verified"]
    return {
        "schema": CONDITION_REALIZATION_SCHEMA,
        "status": "passed" if not unavailable else "unavailable",
        "verified_axes": verified,
        "unavailable_axes": unavailable,
        "axes": axes,
        "issues": errors,
    }
