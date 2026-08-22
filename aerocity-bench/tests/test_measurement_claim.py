from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.measurement_claim import build_measurement_claim_report


def _protocol() -> dict[str, object]:
    protocol: dict[str, object] = {
        "schema": "org.aerocity.bench.g2-i-measurement-claim-protocol.v1",
        "formal_score_eligible": False,
        "outcome_metric": "mean_final_confirmed_recall",
        "independent_unit": "layout_ancestor",
        "cross_validation": "leave_one_layout_ancestor_out",
        "coverage_only_features": ["free_space_coverage_auc"],
        "augmented_features": ["free_space_coverage_auc", "inspection_footprint_auc"],
        "method_fixed_effects": True,
        "minimum_ancestor_count": 5,
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 23,
        "failure_denominator_policy": "retain_all_completed_ancestor_rows",
        "panel_manifest_schema": "org.aerocity.bench.g2-i-measurement-claim-panel.v1",
    }
    protocol["protocol_hash"] = content_hash(protocol)
    return protocol


def _panel(protocol: dict[str, object]) -> dict[str, object]:
    panel: dict[str, object] = {
        "schema": "org.aerocity.bench.g2-i-measurement-claim-panel.v1",
        "formal_score_eligible": False,
        "purpose": "precommitted_calibration_measurement_panel",
        "protocol_hash": protocol["protocol_hash"],
        "precommitted_before_execution": True,
        "layout_ancestors": [f"ancestor-{index}" for index in range(6)],
        "method_ids": ["allocation", "systematic"],
    }
    panel["panel_hash"] = content_hash(panel)
    return panel


def _records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ancestor_index in range(6):
        coverage = 0.15 + 0.11 * (ancestor_index % 3)
        inspection = 0.10 + 0.12 * ancestor_index
        for method_id, offset in (("systematic", 0.00), ("allocation", 0.08)):
            rows.append(
                {
                    "layout_ancestor": f"ancestor-{ancestor_index}",
                    "method_id": method_id,
                    "method_uses_private_truth": False,
                    "episode_count": 3,
                    "source_run_report_hashes": sorted(
                        content_hash(["run", ancestor_index, method_id, episode_index])
                        for episode_index in range(3)
                    ),
                    "failure_included": True,
                    "terminal_status_counts": (
                        {"completed": 2, "deadline_exhausted": 1}
                        if ancestor_index == 2 and method_id == "allocation"
                        else {"completed": 3}
                    ),
                    "mean_final_confirmed_recall": min(
                        1.0, 0.05 + 0.08 * coverage + 0.75 * inspection + offset
                    ),
                    "free_space_coverage_auc": coverage,
                    "inspection_footprint_auc": inspection,
                }
            )
    return rows


def test_measurement_claim_uses_ancestor_level_holdout_and_detects_incremental_footprint() -> None:
    protocol = _protocol()
    report = build_measurement_claim_report(protocol, _records(), _panel(protocol))

    assert report["formal_score_eligible"] is False
    assert report["overall_status"] == "CALIBRATION_ANALYSIS_ONLY"
    assert report["layout_ancestor_count"] == 6
    assert report["episode_rows_are_not_independent"] is True
    assert report["precommitted_method_by_ancestor_panel_complete"] is True
    assert report["terminal_status_counts"]["deadline_exhausted"] == 1
    assert report["gate_checks"]["complete_method_by_ancestor_panel"] is True
    assert (
        report["models"]["coverage_plus_legal_inspection"]["ancestor_equal_oos_rmse"]
        < report["models"]["coverage_only"]["ancestor_equal_oos_rmse"]
    )
    assert report["incremental_prediction"]["mean_ancestor_equal_mse_reduction"] > 0.0
    assert len(report["report_hash"]) == 64


def test_measurement_claim_rejects_pseudoreplication_and_private_truth() -> None:
    protocol = _protocol()
    panel = _panel(protocol)
    duplicated = _records()
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_measurement_claim_report(protocol, duplicated, panel)

    incomplete = _records()
    incomplete.pop()
    with pytest.raises(ValueError, match="differ from the precommitted panel"):
        build_measurement_claim_report(protocol, incomplete, panel)

    dropped_ancestor = [row for row in _records() if row["layout_ancestor"] != "ancestor-2"]
    with pytest.raises(ValueError, match="differ from the precommitted panel"):
        build_measurement_claim_report(protocol, dropped_ancestor, panel)

    private_truth = _records()
    private_truth[0]["method_uses_private_truth"] = True
    with pytest.raises(ValueError, match="private-truth"):
        build_measurement_claim_report(protocol, private_truth, panel)

    untraceable = _records()
    untraceable[0]["source_run_report_hashes"] = ["0" * 64]
    with pytest.raises(ValueError, match="source run-report hashes"):
        build_measurement_claim_report(protocol, untraceable, panel)

    incomplete_status_ledger = _records()
    incomplete_status_ledger[0]["terminal_status_counts"] = {"completed": 2}
    with pytest.raises(ValueError, match="terminal-status counts"):
        build_measurement_claim_report(protocol, incomplete_status_ledger, panel)


def test_measurement_claim_rejects_dropped_failure_and_mutated_protocol() -> None:
    protocol = _protocol()
    panel = _panel(protocol)
    dropped = _records()
    dropped[0]["failure_included"] = False
    with pytest.raises(ValueError, match="drop"):
        build_measurement_claim_report(protocol, dropped, panel)

    protocol["protocol_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_measurement_claim_report(protocol, _records(), panel)

    panel = _panel(_protocol())
    panel["precommitted_before_execution"] = False
    panel["panel_hash"] = content_hash(
        {key: value for key, value in panel.items() if key != "panel_hash"}
    )
    with pytest.raises(ValueError, match="not precommitted"):
        build_measurement_claim_report(_protocol(), _records(), panel)


def test_measurement_claim_tool_writes_once_and_refuses_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    tool_path = Path(__file__).parents[1] / "tools" / "analyze_g2_i_measurement_claim.py"
    spec = importlib.util.spec_from_file_location("measurement_claim_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol_path = tmp_path / "protocol.json"
    panel_path = tmp_path / "panel.json"
    evidence_manifest_path = tmp_path / "evidence-manifest.json"
    output_path = tmp_path / "report.json"
    protocol = _protocol()
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    panel_path.write_text(json.dumps(_panel(protocol)), encoding="utf-8")
    evidence_manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "aggregate_measurement_records",
        lambda *_args, **_kwargs: {"records": _records()},
    )

    assert (
        module.main(
            [
                "--protocol",
                str(protocol_path),
                "--panel-manifest",
                str(panel_path),
                "--evidence-manifest",
                str(evidence_manifest_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["formal_score_eligible"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        module.run(protocol_path, panel_path, evidence_manifest_path, output_path)


def test_checked_in_measurement_protocol_remains_hash_valid() -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / "g2i-measurement-claim-protocol-v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = build_measurement_claim_report(protocol, _records(), _panel(protocol))
    assert report["layout_ancestor_count"] == 6
    assert report["gate_checks"]["minimum_ancestor_count_met"] is False
