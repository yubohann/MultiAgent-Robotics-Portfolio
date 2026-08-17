"""Native T2 canary protocols and immutable motion contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import (
    CollectionProtocolError,
    CollectionProtocolIssue,
    _issue,
    _unknown_keys,
)
from .constants import (
    _T2_CANARY_AXIS_ROLES,
    _T2_CANARY_CELL_KEYS,
    _T2_CANARY_CLAIM_BOUNDARY,
    _T2_CANARY_CONDITIONS,
    _T2_CANARY_EXCLUSION_RULES,
    _T2_CANARY_EXECUTION_CONTRACT,
    _T2_CANARY_OVERVIEW_RETENTION,
    _T2_CANARY_PROTOCOL_KEYS,
    _T2_CANARY_QUALITY_GATES,
    _T2_CANARY_V2_MOTION_CONTRACT,
    _T2_CANARY_V2_PROTOCOL_KEYS,
    _T2_CANARY_V3_MOTION_CONTRACT,
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_PROTOCOL_SCHEMAS,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    SEED_DERIVATION,
)


def _t2_canary_axes() -> list[dict[str, Any]]:
    """Return the one frozen public condition surface for native T2 canaries."""

    return [
        {
            "axis_id": axis_id,
            "values": [value],
            "split_role": _T2_CANARY_AXIS_ROLES.get(axis_id, "condition"),
        }
        for axis_id, value in _T2_CANARY_CONDITIONS.items()
    ]

def _validate_t2_native_canary_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Validate a development-only T2 route contract, never a data protocol.

    A native T2 canary needs a public route/condition commitment so its private
    manifest can be replayed.  It must not borrow a completed T1 quota or
    claim a train/validation split, episode admission, or method score.
    """

    issues: list[CollectionProtocolIssue] = []
    if not isinstance(payload, Mapping):
        return (CollectionProtocolIssue("type", "$", "protocol must be an object"),)
    for key in _unknown_keys(payload, _T2_CANARY_PROTOCOL_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of native T2 canary protocol v1")
    for key in sorted(_T2_CANARY_PROTOCOL_KEYS - set(payload)):
        _issue(issues, "required", f"$.{key}", "required native T2 canary field is missing")

    expected_top = {
        "schema": NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
        "protocol_id": "citylite-native-t2-canary-v1",
        "version": "1.0.0",
        "scene_identity": "RIVERMARK_CITY_LITE_v1",
        "track": "native-t2-closed-loop-development-v1",
        "purpose": "calibrated_native_closed_loop_canary",
        "scoring_status": "not_scored",
        "agent_count": 8,
        "claim_boundary": _T2_CANARY_CLAIM_BOUNDARY,
        "execution_contract": _T2_CANARY_EXECUTION_CONTRACT,
        "axes": _t2_canary_axes(),
        "randomization": {
            "seed_derivation": SEED_DERIVATION,
            "episode_seed_start": 20260728,
            "paired_initial_conditions": False,
        },
        "overview_retention": _T2_CANARY_OVERVIEW_RETENTION,
        "quality_acceptance": _T2_CANARY_QUALITY_GATES,
        "exclusion_rules": _T2_CANARY_EXCLUSION_RULES,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            _issue(issues, key, f"$.{key}", "does not match the frozen native T2 canary contract")

    expected_cell = {
        "cell_id": "native-t2-canary-inner-dev-v1",
        "split": "inner_dev",
        "conditions": _T2_CANARY_CONDITIONS,
        "planned_independent_attempts": 2,
    }
    cells = payload.get("cells")
    if cells != [expected_cell]:
        _issue(
            issues,
            "cells",
            "$.cells",
            "must contain exactly the frozen inner_dev native T2 canary cell",
        )
    else:
        for key in _unknown_keys(cells[0], _T2_CANARY_CELL_KEYS):
            _issue(issues, "unknown_field", f"$.cells[0].{key}", "unknown native T2 canary cell field")
    return tuple(issues)

def _validate_t2_native_canary_v2_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Validate the first motion-feasible native-T2 revision.

    v2 is a new protocol rather than a patch to v1 because private target
    placement now depends on route yaw and bounded action timing.  The exact
    public motion contract lets the collector, sampler, and validator reject a
    mismatch before an Isaac stage is launched.
    """

    issues: list[CollectionProtocolIssue] = []
    if not isinstance(payload, Mapping):
        return (CollectionProtocolIssue("type", "$", "protocol must be an object"),)
    for key in _unknown_keys(payload, _T2_CANARY_V2_PROTOCOL_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of native T2 canary protocol v2")
    for key in sorted(_T2_CANARY_V2_PROTOCOL_KEYS - set(payload)):
        _issue(issues, "required", f"$.{key}", "required native T2 canary v2 field is missing")

    expected_top = {
        "schema": NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
        "protocol_id": "citylite-native-t2-canary-v2",
        "version": "2.0.0",
        "scene_identity": "RIVERMARK_CITY_LITE_v1",
        "track": "native-t2-closed-loop-development-v2",
        "purpose": "motion-feasible-calibrated-native-closed-loop-canary",
        "scoring_status": "not_scored",
        "agent_count": 8,
        "claim_boundary": _T2_CANARY_CLAIM_BOUNDARY,
        "execution_contract": {
            "control_mode": "native_t2_canary",
            "task_variant_id": "isaac-eight-agent-native-t2-search-canary-v2",
            "requires_cf2x_runtime_calibration": True,
            "requires_full_sensor_smoke": True,
            "required_independent_passes": 2,
        },
        "motion_contract": _T2_CANARY_V2_MOTION_CONTRACT,
        "axes": _t2_canary_axes(),
        "randomization": {
            "seed_derivation": SEED_DERIVATION,
            "episode_seed_start": 20260728,
            "paired_initial_conditions": False,
        },
        "overview_retention": _T2_CANARY_OVERVIEW_RETENTION,
        "quality_acceptance": _T2_CANARY_QUALITY_GATES
        + ["route_timing_feasibility_passed"],
        "exclusion_rules": _T2_CANARY_EXCLUSION_RULES,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            _issue(issues, key, f"$.{key}", "does not match the frozen native T2 v2 canary contract")

    expected_cell = {
        "cell_id": "native-t2-canary-inner-dev-v2",
        "split": "inner_dev",
        "conditions": _T2_CANARY_CONDITIONS,
        "planned_independent_attempts": 2,
    }
    cells = payload.get("cells")
    if cells != [expected_cell]:
        _issue(
            issues,
            "cells",
            "$.cells",
            "must contain exactly the frozen inner_dev native T2 v2 canary cell",
        )
    elif any(
        key not in _T2_CANARY_CELL_KEYS for key in cells[0]
    ):
        _issue(issues, "cells", "$.cells[0]", "unknown native T2 v2 canary cell field")
    return tuple(issues)

def _validate_t2_native_canary_v3_protocol(payload: Any) -> tuple[CollectionProtocolIssue, ...]:
    """Validate the immutable time-scaled successor to failed v2.

    v2 remains readable historical evidence.  v3 deliberately changes only
    the route clock and matching rollout duration: all sensor cadence and
    bounded action limits remain identical, so a successful native run is
    evidence of the conservative envelope rather than an unvalidated speed
    increase.
    """

    issues: list[CollectionProtocolIssue] = []
    if not isinstance(payload, Mapping):
        return (CollectionProtocolIssue("type", "$", "protocol must be an object"),)
    for key in _unknown_keys(payload, _T2_CANARY_V2_PROTOCOL_KEYS):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of native T2 canary protocol v3")
    for key in sorted(_T2_CANARY_V2_PROTOCOL_KEYS - set(payload)):
        _issue(issues, "required", f"$.{key}", "required native T2 canary v3 field is missing")

    expected_top = {
        "schema": NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
        "protocol_id": "citylite-native-t2-canary-v3",
        "version": "3.0.0",
        "scene_identity": "RIVERMARK_CITY_LITE_v1",
        "track": "native-t2-closed-loop-development-v3",
        "purpose": "time-scaled-conservative-native-closed-loop-canary",
        "scoring_status": "not_scored",
        "agent_count": 8,
        "claim_boundary": _T2_CANARY_CLAIM_BOUNDARY,
        "execution_contract": {
            "control_mode": "native_t2_canary",
            "task_variant_id": "isaac-eight-agent-native-t2-search-canary-v3",
            "requires_cf2x_runtime_calibration": True,
            "requires_full_sensor_smoke": True,
            "required_independent_passes": 2,
        },
        "motion_contract": _T2_CANARY_V3_MOTION_CONTRACT,
        "axes": _t2_canary_axes(),
        "randomization": {
            "seed_derivation": SEED_DERIVATION,
            "episode_seed_start": 20260728,
            "paired_initial_conditions": False,
        },
        "overview_retention": _T2_CANARY_OVERVIEW_RETENTION,
        "quality_acceptance": _T2_CANARY_QUALITY_GATES
        + ["route_timing_feasibility_passed"],
        "exclusion_rules": _T2_CANARY_EXCLUSION_RULES,
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            _issue(issues, key, f"$.{key}", "does not match the frozen native T2 v3 canary contract")

    expected_cell = {
        "cell_id": "native-t2-canary-inner-dev-v3",
        "split": "inner_dev",
        "conditions": _T2_CANARY_CONDITIONS,
        "planned_independent_attempts": 2,
    }
    cells = payload.get("cells")
    if cells != [expected_cell]:
        _issue(
            issues,
            "cells",
            "$.cells",
            "must contain exactly the frozen inner_dev native T2 v3 canary cell",
        )
    elif any(key not in _T2_CANARY_CELL_KEYS for key in cells[0]):
        _issue(issues, "cells", "$.cells[0]", "unknown native T2 v3 canary cell field")
    return tuple(issues)

def is_native_t2_canary_protocol(protocol: Mapping[str, Any]) -> bool:
    """Return whether a validated protocol belongs to the T2 development track."""

    return protocol.get("schema") in NATIVE_T2_CANARY_PROTOCOL_SCHEMAS


def native_t2_motion_contract(protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the revision-specific public motion contract after validation.

    Legacy v1 contains no motion contract and therefore deliberately returns
    ``None``.  It can be read for historical evidence but cannot borrow v2's
    speed or camera assumptions.
    """

    if protocol.get("schema") == NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA:
        return dict(_T2_CANARY_V2_MOTION_CONTRACT)
    if protocol.get("schema") == NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA:
        return dict(_T2_CANARY_V3_MOTION_CONTRACT)
    if protocol.get("schema") == NATIVE_T2_CANARY_PROTOCOL_SCHEMA:
        return None
    raise CollectionProtocolError("protocol is not a native T2 canary protocol")


def native_t2_v2_motion_contract() -> dict[str, Any]:
    """Return the immutable v2 motion contract used by independent validation.

    Receipts bind to this contract but do not get to define it.  Any future
    motion change requires a distinct protocol revision rather than a silent
    mutation of the v2 canary.
    """

    return dict(_T2_CANARY_V2_MOTION_CONTRACT)


def native_t2_v3_motion_contract() -> dict[str, Any]:
    """Return v3's immutable, time-scaled motion contract."""

    return dict(_T2_CANARY_V3_MOTION_CONTRACT)
