from __future__ import annotations

import copy

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.statistical_protocol import build_statistical_planning_report


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema": "org.aerocity.bench.g2-i-statistical-protocol.v1",
        "formal_score_eligible": False,
        "primary_metric": "mean_final_confirmed_recall",
        "independent_unit": "layout_ancestor",
        "pairing_key": "layout_hash",
        "reference_method_id": "reference",
        "comparator_method_ids": ["candidate"],
        "alpha_two_sided": 0.05,
        "desired_power": 0.8,
        "minimum_detectable_effect": 0.1,
        "minimum_formal_ancestor_count": 12,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 7,
        "multiple_comparison_control": "holm_within_primary_family",
        "failure_denominator_policy": "retain_all_completed_ancestor_rows",
    }
    protocol["protocol_hash"] = content_hash(protocol)
    return protocol


def _method(method_id: str, values: list[float]) -> dict[str, object]:
    return {
        "method_id": method_id,
        "requires_private_truth": False,
        "ancestors": [
            {
                "layout_hash": f"layout-{index}",
                "episode_count": 3,
                "mean_final_confirmed_recall": value,
                "all_returned_home": index != 3,
                "collision_count": 1 if index == 2 else 0,
                "deadline_misses": 2 if index == 1 else 0,
            }
            for index, value in enumerate(values)
        ],
    }


def _report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.g2-i-l0-searchability-calibration.v1",
        "formal_score_eligible": False,
        "method_reports": [
            _method("reference", [0.05, 0.10, 0.15, 0.20, 0.25]),
            _method("candidate", [0.10, 0.20, 0.20, 0.35, 0.30]),
        ],
    }
    report["report_hash"] = content_hash(report)
    return report


def test_planning_uses_layout_ancestors_and_retains_failure_denominators() -> None:
    report = build_statistical_planning_report(_protocol(), _report())

    comparison = report["comparisons"][0]
    assert report["formal_score_eligible"] is False
    assert report["gate_checks"]["episode_rows_not_treated_as_independent"] is True
    assert comparison["layout_ancestor_count"] == 5
    assert comparison["included_failure_rows"] == {
        "ancestors_with_any_collision": 1,
        "ancestors_with_any_deadline_miss": 1,
        "ancestors_with_any_nonreturn": 1,
    }
    assert comparison["estimated_required_ancestors_for_mde"] >= 12
    assert comparison["planning_status"] == "INSUFFICIENT_CURRENT_ANCESTORS"


def test_planning_rejects_unpaired_ancestor_sets() -> None:
    report = _report()
    report["method_reports"][1]["ancestors"].pop()
    report["report_hash"] = content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    with pytest.raises(ValueError, match="different layout-ancestor sets"):
        build_statistical_planning_report(_protocol(), report)


def test_planning_rejects_private_oracle_from_primary_family() -> None:
    report = _report()
    report["method_reports"][1]["requires_private_truth"] = True
    report["report_hash"] = content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    with pytest.raises(ValueError, match="private-truth method"):
        build_statistical_planning_report(_protocol(), report)


def test_protocol_hash_is_fail_closed() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["protocol_hash"] = "0" * 64

    with pytest.raises(ValueError, match="protocol hash mismatch"):
        build_statistical_planning_report(protocol, _report())
