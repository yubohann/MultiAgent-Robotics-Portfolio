"""T1 data-collection protocol and the frozen City-Lite split certificate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from .common import (
    CollectionProtocolIssue,
    _canonical_bytes,
    _contains_private_token,
    _issue,
    _unknown_keys,
)
from .constants import (
    _AXIS_KEYS,
    _AXIS_SET,
    _CELL_KEYS,
    _ID,
    _RANDOMIZATION_KEYS,
    _SEMVER,
    _T1_GEOMETRY_HOLDOUT_AXES,
    _T1_PROTOCOL_KEYS,
    _T1_QUALITY_GATES,
    _T1_REQUIRED_AXES,
    _VALUE,
    SEED_DERIVATION,
    T1_COLLECTION_PROTOCOL_SCHEMA,
)


def _segment_distance_m(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> float:
    """Return the minimum Euclidean distance between two finite 3D segments."""

    def subtract(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(left[axis]) - float(right[axis]) for axis in range(3))

    def dot(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(float(left[axis]) * float(right[axis]) for axis in range(3))

    def clipped(value: float) -> float:
        return min(1.0, max(0.0, value))

    first = subtract(first_end, first_start)
    second = subtract(second_end, second_start)
    offset = subtract(first_start, second_start)
    first_length_sq = dot(first, first)
    second_length_sq = dot(second, second)
    cross = dot(first, second)
    first_offset = dot(first, offset)
    second_offset = dot(second, offset)
    epsilon = 1.0e-12

    if first_length_sq <= epsilon and second_length_sq <= epsilon:
        first_parameter = second_parameter = 0.0
    elif first_length_sq <= epsilon:
        first_parameter = 0.0
        second_parameter = clipped(second_offset / second_length_sq)
    elif second_length_sq <= epsilon:
        second_parameter = 0.0
        first_parameter = clipped(-first_offset / first_length_sq)
    else:
        denominator = first_length_sq * second_length_sq - cross * cross
        first_parameter = (
            clipped((cross * second_offset - second_length_sq * first_offset) / denominator)
            if denominator > epsilon
            else 0.0
        )
        second_parameter = (cross * first_parameter + second_offset) / second_length_sq
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = clipped(-first_offset / first_length_sq)
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = clipped((cross - first_offset) / first_length_sq)

    separation = tuple(
        offset[axis]
        + first_parameter * first[axis]
        - second_parameter * second[axis]
        for axis in range(3)
    )
    return math.sqrt(dot(separation, separation))

def citylite_t1_split_certificate() -> dict[str, Any]:
    """Return the split certificate derived from frozen public City-Lite geometry."""

    from ..citylite_scene import (
        CITY_LITE_ROUTE_FAMILY_A_ID,
        CITY_LITE_ROUTE_FAMILY_B_ID,
        CITY_LITE_START_ANCHOR_A_ID,
        CITY_LITE_START_ANCHOR_B_ID,
        CITY_LITE_TARGET_REGION_A_ID,
        CITY_LITE_TARGET_REGION_B_ID,
        ENVIRONMENT_ID,
        PUBLIC_ROUTE_FAMILIES_W_M,
        SCENE_CONTRACT_SHA256,
        TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M,
        TARGET_REGIONS_W_M,
        canonical_payload_sha256,
    )

    train_routes = PUBLIC_ROUTE_FAMILIES_W_M[CITY_LITE_ROUTE_FAMILY_A_ID]
    validation_routes = PUBLIC_ROUTE_FAMILIES_W_M[CITY_LITE_ROUTE_FAMILY_B_ID]
    train_starts = TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[
        CITY_LITE_ROUTE_FAMILY_A_ID
    ]
    validation_starts = TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[
        CITY_LITE_ROUTE_FAMILY_B_ID
    ]
    train_region = TARGET_REGIONS_W_M[CITY_LITE_TARGET_REGION_B_ID]
    validation_region = TARGET_REGIONS_W_M[CITY_LITE_TARGET_REGION_A_ID]

    def waypoints(routes: Sequence[Sequence[Sequence[float]]]) -> set[tuple[float, ...]]:
        return {tuple(float(value) for value in point) for route in routes for point in route}

    def segments(
        routes: Sequence[Sequence[Sequence[float]]],
    ) -> set[tuple[tuple[float, ...], tuple[float, ...]]]:
        result: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
        for route in routes:
            for start, end in pairwise(route):
                endpoints = sorted(
                    (
                        tuple(float(value) for value in start),
                        tuple(float(value) for value in end),
                    )
                )
                result.add((endpoints[0], endpoints[1]))
        return result

    shared_waypoints = waypoints(train_routes) & waypoints(validation_routes)
    shared_segments = segments(train_routes) & segments(validation_routes)
    cross_route_segment_distances = tuple(
        _segment_distance_m(train_start, train_end, validation_start, validation_end)
        for train_route in train_routes
        for train_start, train_end in pairwise(train_route)
        for validation_route in validation_routes
        for validation_start, validation_end in pairwise(validation_route)
    )
    route_intersection_count = sum(
        distance <= 1.0e-9 for distance in cross_route_segment_distances
    )
    minimum_route_distance = min(cross_route_segment_distances)
    minimum_start_distance = min(
        math.dist(tuple(float(value) for value in train), tuple(float(value) for value in validation))
        for train in train_starts
        for validation in validation_starts
    )
    overlap_extents = tuple(
        max(
            0.0,
            min(train_region.maximum[axis], validation_region.maximum[axis])
            - max(train_region.minimum[axis], validation_region.minimum[axis]),
        )
        for axis in range(3)
    )
    overlap_volume = math.prod(overlap_extents)
    region_axis_gaps = tuple(
        max(
            0.0,
            validation_region.minimum[axis] - train_region.maximum[axis],
            train_region.minimum[axis] - validation_region.maximum[axis],
        )
        for axis in range(3)
    )
    minimum_region_distance = math.sqrt(sum(gap * gap for gap in region_axis_gaps))

    def split_entry(
        *,
        route_family_id: str,
        start_anchor_id: str,
        target_region_id: str,
    ) -> dict[str, Any]:
        routes = PUBLIC_ROUTE_FAMILIES_W_M[route_family_id]
        starts = TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[route_family_id]
        region = TARGET_REGIONS_W_M[target_region_id]
        region_payload = {
            "minimum_w_m": list(region.minimum),
            "maximum_w_m": list(region.maximum),
        }
        return {
            "route_family_id": route_family_id,
            "route_geometry_sha256": canonical_payload_sha256(routes),
            "start_anchor_id": start_anchor_id,
            "start_geometry_sha256": canonical_payload_sha256(starts),
            "target_region_id": target_region_id,
            "target_region": region_payload,
            "target_region_sha256": canonical_payload_sha256(region_payload),
        }

    return {
        "schema": "org.rivermark.benchmark.citylite-t1-split-certificate.v1",
        "scene_identity": ENVIRONMENT_ID,
        "scene_contract_sha256": SCENE_CONTRACT_SHA256,
        "claim": "same_layout_route_family_start_region_holdout",
        "train": split_entry(
            route_family_id=CITY_LITE_ROUTE_FAMILY_A_ID,
            start_anchor_id=CITY_LITE_START_ANCHOR_A_ID,
            target_region_id=CITY_LITE_TARGET_REGION_B_ID,
        ),
        "validation": split_entry(
            route_family_id=CITY_LITE_ROUTE_FAMILY_B_ID,
            start_anchor_id=CITY_LITE_START_ANCHOR_B_ID,
            target_region_id=CITY_LITE_TARGET_REGION_A_ID,
        ),
        "geometry_checks": {
            "shared_route_waypoint_count": len(shared_waypoints),
            "shared_route_segment_count": len(shared_segments),
            "route_segment_intersection_count": route_intersection_count,
            "minimum_cross_split_route_distance_m": round(minimum_route_distance, 9),
            "route_geometry_disjoint": route_intersection_count == 0,
            "minimum_cross_split_start_distance_m": round(minimum_start_distance, 9),
            "target_region_overlap_volume_m3": round(overlap_volume, 9),
            "minimum_cross_split_target_region_distance_m": round(
                minimum_region_distance, 9
            ),
            "route_family_start_region_holdout_passed": bool(
                not shared_waypoints
                and not shared_segments
                and minimum_start_distance >= 4.0
                and overlap_volume == 0.0
                and minimum_region_distance >= 4.0
            ),
        },
    }

def _validate_t1_collection_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Validate the T1 data protocol without importing T2 scoring assumptions."""

    issues: list[CollectionProtocolIssue] = []
    if not isinstance(payload, Mapping):
        return (CollectionProtocolIssue("type", "$", "protocol must be an object"),)
    for key in _unknown_keys(payload, _T1_PROTOCOL_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of T1 protocol v2")
    for key in sorted(_T1_PROTOCOL_KEYS - set(payload)):
        _issue(issues, "required", f"$.{key}", "required T1 protocol field is missing")
    if payload.get("schema") != T1_COLLECTION_PROTOCOL_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {T1_COLLECTION_PROTOCOL_SCHEMA!r}")
    for key in ("protocol_id", "version", "dataset_version"):
        value = payload.get(key)
        pattern = _ID if key == "protocol_id" else _SEMVER
        if not isinstance(value, str) or not pattern.fullmatch(value):
            _issue(issues, key, f"$.{key}", "invalid identifier or semantic version")
        elif key == "protocol_id" and _contains_private_token(value):
            _issue(issues, "private_value", f"$.{key}", "private identifiers are forbidden")
    exact_top = {
        "scene_identity": "RIVERMARK_CITY_LITE_v1",
        "track": "t1-expert-coverage-multisensor-v1",
        "purpose": "expert_coverage_dataset",
        "scoring_status": "not_scored",
        "agent_count": 8,
    }
    for key, expected in exact_top.items():
        if payload.get(key) != expected:
            _issue(issues, key, f"$.{key}", f"must equal {expected!r}")

    statistical_unit = payload.get("statistical_unit")
    expected_statistical_unit = {
        "unit": "episode",
        "resampling_unit": "episode_id",
        "frames_independent": False,
        "agents_independent": False,
        "repeated_placements_independent": False,
    }
    if statistical_unit != expected_statistical_unit:
        _issue(
            issues,
            "statistical_unit",
            "$.statistical_unit",
            "must declare episode-level inference without frame/agent pseudo-replication",
        )

    scope = payload.get("scope")
    expected_scope = {
        "layout_scope": "same_layout_route_condition_holdout",
        "cross_scene_generalization": False,
        "policy_ranking": False,
        "closed_loop_search": False,
    }
    if scope != expected_scope:
        _issue(
            issues,
            "t1_scope",
            "$.scope",
            "T1 must be same-layout data collection with no policy-ranking claim",
        )

    axes_value = payload.get("axes")
    axis_values: dict[str, set[str]] = {}
    axis_roles: dict[str, str] = {}
    if not isinstance(axes_value, list) or not axes_value:
        _issue(issues, "axes", "$.axes", "must declare condition axes")
    else:
        for index, axis in enumerate(axes_value):
            path = f"$.axes[{index}]"
            if not isinstance(axis, Mapping):
                _issue(issues, "type", path, "axis must be an object")
                continue
            for key in _unknown_keys(axis, _AXIS_KEYS):
                _issue(issues, "unknown_field", f"{path}.{key}", "unknown axis field")
            axis_id = axis.get("axis_id")
            values = axis.get("values")
            role = axis.get("split_role")
            if not isinstance(axis_id, str) or axis_id not in _AXIS_SET:
                _issue(issues, "axis_id", f"{path}.axis_id", "unknown condition axis")
                continue
            if axis_id in axis_values:
                _issue(issues, "duplicate_axis", f"{path}.axis_id", "axis must be unique")
                continue
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not _VALUE.fullmatch(value) for value in values)
                or len(set(values)) != len(values)
            ):
                _issue(issues, "axis_values", f"{path}.values", "values must be unique public identifiers")
                values = []
            if role not in {"scene", "condition", "episode", "holdout"}:
                _issue(issues, "split_role", f"{path}.split_role", "unknown split role")
            axis_values[axis_id] = set(values)
            if isinstance(role, str):
                axis_roles[axis_id] = role
    missing_axes = sorted(_T1_REQUIRED_AXES - set(axis_values))
    if missing_axes:
        _issue(issues, "missing_t1_axis", "$.axes", "missing T1 axes: " + ", ".join(missing_axes))
    for axis_id in sorted(_T1_GEOMETRY_HOLDOUT_AXES):
        if axis_roles.get(axis_id) != "holdout":
            _issue(issues, "holdout_role", f"$.axes.{axis_id}", "geometry split axis must be holdout")
    if axis_roles.get("visibility_bucket") != "condition":
        _issue(
            issues,
            "visibility_role",
            "$.axes.visibility_bucket",
            "visibility is a coverage condition, not proof of geometric holdout",
        )

    expected_certificate = citylite_t1_split_certificate()
    certificate = payload.get("split_certificate")
    if certificate != expected_certificate:
        _issue(
            issues,
            "split_certificate",
            "$.split_certificate",
            "certificate does not match frozen public route/start/region geometry",
        )

    cells_value = payload.get("cells")
    cell_ids: set[str] = set()
    cell_signatures: set[bytes] = set()
    split_axis_values: dict[str, dict[str, set[str]]] = {}
    covered_axis_values: dict[str, set[str]] = {axis: set() for axis in axis_values}
    total_minimum_admitted = 0
    split_values: set[str] = set()
    if not isinstance(cells_value, list) or not cells_value:
        _issue(issues, "cells", "$.cells", "must contain quota cells")
    else:
        for index, cell in enumerate(cells_value):
            path = f"$.cells[{index}]"
            if not isinstance(cell, Mapping):
                _issue(issues, "type", path, "cell must be an object")
                continue
            for key in _unknown_keys(cell, _CELL_KEYS):
                _issue(issues, "unknown_field", f"{path}.{key}", "unknown cell field")
            cell_id = cell.get("cell_id")
            if not isinstance(cell_id, str) or not _ID.fullmatch(cell_id):
                _issue(issues, "cell_id", f"{path}.cell_id", "invalid cell identifier")
            elif cell_id in cell_ids:
                _issue(issues, "duplicate_cell", f"{path}.cell_id", "cell ID must be unique")
            else:
                cell_ids.add(cell_id)
            split = cell.get("split")
            if split not in {"train", "validation"}:
                _issue(issues, "split", f"{path}.split", "initial T1 profile allows train/validation only")
            else:
                split_values.add(str(split))
            conditions = cell.get("conditions")
            if not isinstance(conditions, Mapping):
                _issue(issues, "conditions", f"{path}.conditions", "must be an object")
                conditions = {}
            if set(conditions) != set(axis_values):
                _issue(issues, "condition_coverage", f"{path}.conditions", "every axis must be fixed")
            for axis_id, value in conditions.items():
                if (
                    not isinstance(axis_id, str)
                    or not isinstance(value, str)
                    or axis_id not in axis_values
                    or value not in axis_values.get(axis_id, set())
                ):
                    _issue(issues, "axis_value", f"{path}.conditions.{axis_id}", "undeclared axis value")
                elif isinstance(split, str):
                    covered_axis_values[axis_id].add(value)
                    split_axis_values.setdefault(split, {}).setdefault(axis_id, set()).add(value)
            if isinstance(split, str) and split in {"train", "validation"}:
                expected_split = expected_certificate[split]
                for axis_id, certificate_key in (
                    ("route_family", "route_family_id"),
                    ("start_anchor", "start_anchor_id"),
                    ("target_region", "target_region_id"),
                ):
                    if conditions.get(axis_id) != expected_split[certificate_key]:
                        _issue(
                            issues,
                            "split_geometry_binding",
                            f"{path}.conditions.{axis_id}",
                            "cell does not match the independently derived split certificate",
                        )
            if all(
                isinstance(axis_id, str) and isinstance(value, str)
                for axis_id, value in conditions.items()
            ):
                signature = _canonical_bytes({"split": split, "conditions": dict(conditions)})
                if signature in cell_signatures:
                    _issue(issues, "duplicate_condition_cell", path, "split/conditions must be unique")
                cell_signatures.add(signature)
            minimum_attempts = cell.get("minimum_attempts")
            minimum_admitted = cell.get("minimum_admitted")
            for key, value in (("minimum_attempts", minimum_attempts), ("minimum_admitted", minimum_admitted)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    _issue(issues, key, f"{path}.{key}", "must be a positive integer")
            if isinstance(minimum_admitted, int) and not isinstance(minimum_admitted, bool):
                total_minimum_admitted += minimum_admitted
            if (
                isinstance(minimum_attempts, int)
                and not isinstance(minimum_attempts, bool)
                and isinstance(minimum_admitted, int)
                and not isinstance(minimum_admitted, bool)
                and minimum_admitted > minimum_attempts
            ):
                _issue(issues, "quota_order", path, "minimum_admitted cannot exceed attempts")
    if split_values != {"train", "validation"}:
        _issue(issues, "split_coverage", "$.cells", "T1 requires train and validation cells")
    for axis_id in sorted(_T1_GEOMETRY_HOLDOUT_AXES & set(axis_values)):
        train_values = split_axis_values.get("train", {}).get(axis_id, set())
        validation_values = split_axis_values.get("validation", {}).get(axis_id, set())
        if not train_values or not validation_values or train_values & validation_values:
            _issue(
                issues,
                "holdout_overlap",
                f"$.cells.{axis_id}",
                "train/validation holdout identifiers must not overlap",
            )
    for axis_id, values in sorted(axis_values.items()):
        if values - covered_axis_values.get(axis_id, set()):
            _issue(issues, "uncovered_axis_value", f"$.axes.{axis_id}", "declared value has no quota cell")
    if not 8 <= total_minimum_admitted <= 12:
        _issue(issues, "initial_quota", "$.cells", "initial T1 target must total 8-12 admitted episodes")

    randomization = payload.get("randomization")
    if not isinstance(randomization, Mapping) or set(randomization) != _RANDOMIZATION_KEYS:
        _issue(issues, "randomization", "$.randomization", "randomization fields are incomplete or unknown")
    else:
        if randomization.get("seed_derivation") != SEED_DERIVATION:
            _issue(issues, "seed_derivation", "$.randomization.seed_derivation", "unsupported derivation")
        seed_start = randomization.get("episode_seed_start")
        if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
            _issue(issues, "episode_seed_start", "$.randomization.episode_seed_start", "must be non-negative")
        if randomization.get("paired_initial_conditions") is not False:
            _issue(issues, "paired_initial_conditions", "$.randomization.paired_initial_conditions", "T1 is not a paired method comparison")

    analysis = payload.get("analysis_plan")
    expected_analysis = {
        "quota_basis": "coverage_quality_cost_pilot",
        "initial_admitted_episode_target": total_minimum_admitted,
        "sample_size_reestimate_after_admitted": total_minimum_admitted,
        "policy_ranking": False,
        "power_analysis_status": "not_applicable_to_t1_policy_ranking",
        "active_visibility_strata": ["direct-visible-v1"],
        "unsupported_visibility_strata": ["partial-visible-v1"],
        "primary_outputs": [
            "sensor_quality",
            "route_coverage",
            "visibility_stratum_realization",
            "failure_rate",
            "episode_bytes",
        ],
    }
    if analysis != expected_analysis:
        _issue(issues, "analysis_plan", "$.analysis_plan", "T1 quota/analysis plan is not frozen")
    if axis_values.get("visibility_bucket") != {"direct-visible-v1"}:
        _issue(issues, "visibility_scope", "$.axes.visibility_bucket", "only the CPU-feasible direct stratum is active")

    overview = payload.get("overview_retention")
    expected_overview = {
        "selection_rule": "first_each_fixed_retained_frame_stride_and_final",
        "frame_index_stride": 10,
        "fixed_world_camera": True,
        "outcome_independent": True,
        "stored_modalities": ["rgb", "semantic", "world_pose"],
        "runtime_only_modalities": ["depth"],
    }
    if overview != expected_overview:
        _issue(issues, "overview_retention", "$.overview_retention", "overview evidence schedule is not frozen")

    quality = payload.get("quality_acceptance")
    if (
        not isinstance(quality, list)
        or any(not isinstance(gate, str) for gate in quality)
        or len(quality) != len(set(quality))
        or set(quality) != _T1_QUALITY_GATES
    ):
        _issue(issues, "quality_acceptance", "$.quality_acceptance", "T1 quality gates are incomplete")

    exclusions = payload.get("exclusion_rules")
    if (
        not isinstance(exclusions, list)
        or not exclusions
        or any(not isinstance(rule, str) or not _VALUE.fullmatch(rule) for rule in exclusions)
        or len(exclusions) != len(set(exclusions))
    ):
        _issue(issues, "exclusion_rules", "$.exclusion_rules", "must be unique public identifiers")
    return tuple(issues)
