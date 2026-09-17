from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.collection_protocol import (
    T1_COLLECTION_PROTOCOL_SCHEMA,
    T1_COVERAGE_REPORT_SCHEMA,
    coverage_report,
    load_collection_protocol,
    resolve_collection_binding,
    validate_collection_protocol,
)

PROTOCOL_PATH = ROOT / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json"
V1_PATH = ROOT / "config" / "collection_protocol.citylite_minimal_v1.json"


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _codes(payload: object) -> set[str]:
    return {issue.code for issue in validate_collection_protocol(payload)}


def test_active_t1_protocol_and_legacy_v1_are_both_valid() -> None:
    protocol = load_collection_protocol(PROTOCOL_PATH)
    assert protocol["schema"] == T1_COLLECTION_PROTOCOL_SCHEMA
    assert validate_collection_protocol(protocol) == ()

    legacy = load_collection_protocol(V1_PATH)
    assert validate_collection_protocol(legacy) == ()


def test_missing_statistical_unit_and_policy_ranking_strict() -> None:
    missing = _protocol()
    del missing["statistical_unit"]
    assert "statistical_unit" in _codes(missing)

    ranking = _protocol()
    ranking["scope"]["policy_ranking"] = True
    assert "t1_scope" in _codes(ranking)


def test_altered_or_overlapping_split_geometry_fails_closed() -> None:
    overlapping = _protocol()
    validation = overlapping["cells"][1]["conditions"]
    validation["route_family"] = "citylite-route-family-a-v1"
    validation["start_anchor"] = "citylite-start-anchor-a-v1"
    assert "holdout_overlap" in _codes(overlapping)


def test_unsupported_visibility_stratum_cannot_be_activated_silently() -> None:
    protocol = _protocol()
    visibility = next(axis for axis in protocol["axes"] if axis["axis_id"] == "visibility_bucket")
    visibility["values"].append("partial-visible-v1")
    assert "visibility_scope" in _codes(protocol)


def test_t1_binding_and_empty_coverage_have_no_search_power_claim() -> None:
    protocol = load_collection_protocol(PROTOCOL_PATH)
    first = resolve_collection_binding(
        protocol, cell_id="train-citylite-direct-v2", episode_index=0
    )
    second = resolve_collection_binding(
        protocol, cell_id="train-citylite-direct-v2", episode_index=0
    )
    assert first == second
    assert first["split"] == "train"

    report = coverage_report(protocol, [])
    assert report["schema"] == T1_COVERAGE_REPORT_SCHEMA
    assert report["complete"] is False
    assert report["quota_analysis"]["policy_ranking"] is False
    assert report["quota_analysis"]["initial_admitted_episode_target"] == 8
    assert "power_analysis" not in report


def test_t1_protocol_rejects_unknown_top_level_fields() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["normalized_confirmed_auc"] = 1.0
    assert "unknown_field" in _codes(protocol)


def test_t1_protocol_malformed_nested_values_return_issues_instead_of_raising() -> None:
    cases: list[tuple[str, dict[str, object], str]] = []

    axis_value = copy.deepcopy(_protocol())
    axis_value["axes"][0]["values"].append([])
    cases.append(("axis value", axis_value, "axis_values"))

    condition_value = copy.deepcopy(_protocol())
    condition_value["cells"][0]["conditions"]["layout"] = []
    cases.append(("condition value", condition_value, "axis_value"))

    quality_gate = copy.deepcopy(_protocol())
    quality_gate["quality_acceptance"].append({})
    cases.append(("quality gate", quality_gate, "quality_acceptance"))

    exclusion = copy.deepcopy(_protocol())
    exclusion["exclusion_rules"].append([])
    cases.append(("exclusion", exclusion, "exclusion_rules"))

    for label, protocol, expected_code in cases:
        codes = _codes(protocol)
        assert expected_code in codes, label
