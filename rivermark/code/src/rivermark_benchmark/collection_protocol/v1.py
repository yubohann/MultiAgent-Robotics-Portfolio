"""Validation of the immutable v1 collection protocol."""

from __future__ import annotations

from collections.abc import Mapping
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
    _EVALUATION_SPLITS,
    _ID,
    _POWER_KEYS,
    _PROTOCOL_KEYS,
    _RANDOMIZATION_KEYS,
    _SAME_LAYOUT_HOLDOUT_AXES,
    _SEMVER,
    _SPLITS,
    _VALUE,
    COLLECTION_PROTOCOL_SCHEMA,
    POWER_METHOD,
    SEED_DERIVATION,
)
from .power import _valid_number, required_paired_episodes


def _validate_v1_collection_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    issues: list[CollectionProtocolIssue] = []
    if not isinstance(payload, Mapping):
        return (CollectionProtocolIssue("type", "$", "protocol must be an object"),)
    for key in _unknown_keys(payload, _PROTOCOL_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of collection protocol v1")
    if payload.get("schema") != COLLECTION_PROTOCOL_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {COLLECTION_PROTOCOL_SCHEMA!r}")
    for key in ("protocol_id", "version", "dataset_version"):
        value = payload.get(key)
        pattern = _ID if key == "protocol_id" else _SEMVER
        if not isinstance(value, str) or not pattern.fullmatch(value):
            _issue(issues, key, f"$.{key}", "invalid identifier or semantic version")
        elif key == "protocol_id" and _contains_private_token(value):
            _issue(issues, "private_value", f"$.{key}", "private/evaluator identifiers are forbidden")
    if payload.get("scene_identity") != "RIVERMARK_CITY_LITE_v1":
        _issue(issues, "scene_identity", "$.scene_identity", "only approved City-Lite v1 is allowed")
    if payload.get("track") != "multi_uav_search3d":
        _issue(issues, "track", "$.track", "protocol must target the multi-UAV Search3D track")
    if payload.get("agent_count") != 8:
        _issue(issues, "agent_count", "$.agent_count", "the formal track requires exactly eight agents")

    axes_value = payload.get("axes")
    axis_values: dict[str, set[str]] = {}
    axis_roles: dict[str, str] = {}
    if not isinstance(axes_value, list) or not axes_value:
        _issue(issues, "axes", "$.axes", "must declare at least one known condition axis")
    else:
        for index, axis in enumerate(axes_value):
            path = f"$.axes[{index}]"
            if not isinstance(axis, Mapping):
                _issue(issues, "type", path, "axis must be an object")
                continue
            for key in _unknown_keys(axis, _AXIS_KEYS):
                _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of axis v1")
            axis_id = axis.get("axis_id")
            if not isinstance(axis_id, str) or axis_id not in _AXIS_SET:
                _issue(issues, "axis_id", f"{path}.axis_id", "unknown condition axis")
                continue
            if axis_id in axis_values:
                _issue(issues, "duplicate_axis", f"{path}.axis_id", "axis_id must be unique")
                continue
            values = axis.get("values")
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not _VALUE.fullmatch(value) for value in values)
                or len(set(values)) != len(values)
            ):
                _issue(issues, "axis_values", f"{path}.values", "must be unique public identifiers")
                values = []
            split_role = axis.get("split_role")
            if split_role not in {"scene", "condition", "episode", "holdout"}:
                _issue(issues, "split_role", f"{path}.split_role", "unknown split role")
            if _contains_private_token(axis_id) or any(_contains_private_token(value) for value in values):
                _issue(issues, "private_value", path, "private/evaluator values are forbidden")
            axis_values[axis_id] = set(values)
            if isinstance(split_role, str):
                axis_roles[axis_id] = split_role

    cells_value = payload.get("cells")
    cell_ids: set[str] = set()
    cell_splits: dict[str, str] = {}
    cell_signatures: set[bytes] = set()
    split_values: set[str] = set()
    split_axis_values: dict[str, dict[str, set[str]]] = {}
    covered_axis_values: dict[str, set[str]] = {axis_id: set() for axis_id in axis_values}
    if not isinstance(cells_value, list) or not cells_value:
        _issue(issues, "cells", "$.cells", "must contain at least one quota cell")
    else:
        for index, cell in enumerate(cells_value):
            path = f"$.cells[{index}]"
            if not isinstance(cell, Mapping):
                _issue(issues, "type", path, "cell must be an object")
                continue
            for key in _unknown_keys(cell, _CELL_KEYS):
                _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of cell v1")
            cell_id = cell.get("cell_id")
            valid_cell_id = (
                isinstance(cell_id, str)
                and bool(_ID.fullmatch(cell_id))
                and not _contains_private_token(cell_id)
            )
            if not valid_cell_id:
                _issue(issues, "cell_id", f"{path}.cell_id", "invalid cell identifier")
            elif cell_id in cell_ids:
                _issue(issues, "duplicate_cell", f"{path}.cell_id", "cell_id must be unique")
                valid_cell_id = False
            else:
                cell_ids.add(cell_id)
            split = cell.get("split")
            if not isinstance(split, str) or split not in _SPLITS:
                _issue(issues, "split", f"{path}.split", "unknown benchmark split")
            else:
                split_values.add(split)
                if valid_cell_id:
                    cell_splits[str(cell_id)] = split
            conditions = cell.get("conditions")
            if not isinstance(conditions, Mapping):
                _issue(issues, "conditions", f"{path}.conditions", "must be an object")
                conditions = {}
            condition_keys = {key for key in conditions if isinstance(key, str)}
            if len(condition_keys) != len(conditions) or condition_keys != set(axis_values):
                _issue(issues, "condition_coverage", f"{path}.conditions", "every declared axis must be fixed in every cell")
            valid_conditions = True
            for axis_id, value in conditions.items():
                condition_path = f"{path}.conditions.{axis_id}"
                if not isinstance(axis_id, str) or axis_id not in axis_values:
                    _issue(issues, "unknown_axis", condition_path, "condition axis is not declared")
                    valid_conditions = False
                    continue
                if not isinstance(value, str) or value not in axis_values[axis_id]:
                    _issue(issues, "axis_value", condition_path, "value is not declared for this axis")
                    valid_conditions = False
                else:
                    covered_axis_values[axis_id].add(value)
                    if isinstance(split, str) and split in _SPLITS:
                        split_axis_values.setdefault(split, {}).setdefault(axis_id, set()).add(value)
                if _contains_private_token(axis_id) or _contains_private_token(value):
                    _issue(issues, "private_value", condition_path, "private/evaluator values are forbidden")
                    valid_conditions = False
            if isinstance(split, str) and split in _SPLITS and valid_conditions and len(conditions) == len(axis_values):
                signature = _canonical_bytes({"split": split, "conditions": dict(conditions)})
                if signature in cell_signatures:
                    _issue(issues, "duplicate_condition_cell", path, "split and condition assignment must be unique")
                cell_signatures.add(signature)
            minimum_attempts = cell.get("minimum_attempts")
            minimum_admitted = cell.get("minimum_admitted")
            for key, value in (("minimum_attempts", minimum_attempts), ("minimum_admitted", minimum_admitted)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    _issue(issues, key, f"{path}.{key}", "must be a positive integer")
            if (
                isinstance(minimum_attempts, int)
                and not isinstance(minimum_attempts, bool)
                and isinstance(minimum_admitted, int)
                and not isinstance(minimum_admitted, bool)
                and minimum_admitted > minimum_attempts
            ):
                _issue(issues, "quota_order", path, "minimum_admitted cannot exceed minimum_attempts")
    if "train" not in split_values:
        _issue(issues, "split_coverage", "$.cells", "a formal collection protocol requires a train split")
    if "train" in split_values and split_values & _EVALUATION_SPLITS:
        for axis_id in sorted(_SAME_LAYOUT_HOLDOUT_AXES):
            if axis_id not in axis_values:
                _issue(
                    issues,
                    "missing_holdout_axis",
                    "$.axes",
                    f"same-layout Search3D requires {axis_id} for route holdout",
                )
            elif axis_roles.get(axis_id) != "holdout":
                _issue(
                    issues,
                    "holdout_role",
                    f"$.axes.{axis_id}",
                    "same-layout route holdout axes must use split_role=holdout",
                )
        for evaluation_split in sorted(split_values & _EVALUATION_SPLITS):
            for axis_id in sorted(_SAME_LAYOUT_HOLDOUT_AXES & set(axis_values)):
                train_values = split_axis_values.get("train", {}).get(axis_id, set())
                evaluation_values = split_axis_values.get(evaluation_split, {}).get(axis_id, set())
                if not train_values or not evaluation_values or train_values & evaluation_values:
                    _issue(
                        issues,
                        "holdout_overlap",
                        f"$.cells.{evaluation_split}.conditions.{axis_id}",
                        "train and evaluation cells must use disjoint same-layout holdout values",
                    )
    for axis_id, values in sorted(axis_values.items()):
        uncovered = sorted(values - covered_axis_values.get(axis_id, set()))
        if uncovered:
            _issue(
                issues,
                "uncovered_axis_value",
                f"$.axes.{axis_id}",
                "declared values have no quota cell: " + ", ".join(uncovered),
            )

    randomization = payload.get("randomization")
    if not isinstance(randomization, Mapping):
        _issue(issues, "randomization", "$.randomization", "must be an object")
    else:
        for key in _unknown_keys(randomization, _RANDOMIZATION_KEYS):
            _issue(issues, "unknown_field", f"$.randomization.{key}", "field is not part of randomization v1")
        if randomization.get("seed_derivation") != SEED_DERIVATION:
            _issue(issues, "seed_derivation", "$.randomization.seed_derivation", "unsupported seed derivation")
        seed_start = randomization.get("episode_seed_start")
        if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
            _issue(issues, "episode_seed_start", "$.randomization.episode_seed_start", "must be a non-negative integer")
        if randomization.get("paired_initial_conditions") is not True:
            _issue(issues, "paired_initial_conditions", "$.randomization.paired_initial_conditions", "must be true")

    power = payload.get("power_analysis")
    if not isinstance(power, Mapping):
        _issue(issues, "power_analysis", "$.power_analysis", "must be an object")
    else:
        for key in _unknown_keys(power, _POWER_KEYS):
            _issue(issues, "unknown_field", f"$.power_analysis.{key}", "field is not part of power analysis v1")
        if power.get("method") != POWER_METHOD:
            _issue(issues, "power_method", "$.power_analysis.method", "unsupported power analysis method")
        if power.get("primary_metric") != "normalized_confirmed_auc":
            _issue(issues, "primary_metric", "$.power_analysis.primary_metric", "must use the frozen primary metric")
        if not _valid_number(power.get("familywise_alpha"), minimum=0.0, maximum=1.0):
            _issue(issues, "familywise_alpha", "$.power_analysis.familywise_alpha", "must be in (0, 1)")
        if not _valid_number(power.get("power"), minimum=0.5, maximum=1.0):
            _issue(issues, "power", "$.power_analysis.power", "must be in (0.5, 1)")
        effect_size = power.get("minimum_effect_size")
        if not _valid_number(effect_size, minimum=0.0) or (
            isinstance(effect_size, (int, float)) and float(effect_size) > 1.0
        ):
            _issue(issues, "minimum_effect_size", "$.power_analysis.minimum_effect_size", "must be in (0, 1]")
        if not _valid_number(power.get("difference_standard_deviation"), minimum=0.0):
            _issue(
                issues,
                "difference_standard_deviation",
                "$.power_analysis.difference_standard_deviation",
                "must be positive and finite",
            )
        comparison_count = power.get("comparison_count")
        if isinstance(comparison_count, bool) or not isinstance(comparison_count, int) or comparison_count < 1:
            _issue(issues, "comparison_count", "$.power_analysis.comparison_count", "must be a positive integer")
        evaluation_split = power.get("evaluation_split")
        if not isinstance(evaluation_split, str) or evaluation_split not in _EVALUATION_SPLITS:
            _issue(issues, "evaluation_split", "$.power_analysis.evaluation_split", "must be a held-out evaluation split")
        elif evaluation_split not in split_values:
            _issue(issues, "evaluation_split", "$.power_analysis.evaluation_split", "split has no declared quota cell")
        numeric_power_fields_valid = (
            _valid_number(power.get("familywise_alpha"), minimum=0.0, maximum=1.0)
            and _valid_number(power.get("power"), minimum=0.5, maximum=1.0)
            and _valid_number(effect_size, minimum=0.0)
            and isinstance(effect_size, (int, float))
            and float(effect_size) <= 1.0
            and _valid_number(power.get("difference_standard_deviation"), minimum=0.0)
            and isinstance(comparison_count, int)
            and not isinstance(comparison_count, bool)
            and comparison_count >= 1
        )
        if numeric_power_fields_valid:
            expected = required_paired_episodes(
                familywise_alpha=float(power["familywise_alpha"]),
                power=float(power["power"]),
                minimum_effect_size=float(power["minimum_effect_size"]),
                difference_standard_deviation=float(power["difference_standard_deviation"]),
                comparison_count=comparison_count,
            )
            required = power.get("required_evaluation_episodes")
            if isinstance(required, bool) or not isinstance(required, int) or required != expected:
                _issue(
                    issues,
                    "power_count",
                    "$.power_analysis.required_evaluation_episodes",
                    "does not match the declared paired calculation",
                )

    exclusions = payload.get("exclusion_rules")
    if (
        not isinstance(exclusions, list)
        or not exclusions
        or any(not isinstance(rule, str) or not _VALUE.fullmatch(rule) or _contains_private_token(rule) for rule in exclusions)
        or len(set(exclusions)) != len(exclusions)
    ):
        _issue(issues, "exclusion_rules", "$.exclusion_rules", "must be unique public rule identifiers")
    return tuple(issues)
