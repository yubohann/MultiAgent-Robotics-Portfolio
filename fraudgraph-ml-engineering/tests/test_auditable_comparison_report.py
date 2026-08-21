from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_auditable_comparison_report.py"
SPEC = importlib.util.spec_from_file_location("auditable_comparison_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record_payload(*, dataset: str = "comp", data_revision: str = "release-a") -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": "SplitGNN + Transformer",
        "data_revision": data_revision,
        "split_policy": "fixed-public-masks",
        "selection_policy": "validation_only_auc_then_f1macro",
        "seed": 30,
        "metrics": {
            "validation": {"auc": 0.7},
            "test": {"auc": 0.68, "pr_auc": 0.5, "f1_macro": 0.6},
        },
    }


def test_explicit_record_is_validated_and_rendered_without_ranking(tmp_path) -> None:
    record_path = tmp_path / "comp_splitgnn_seed30.json"
    record_path.write_text(json.dumps(_record_payload()), encoding="utf-8")

    record = MODULE.validate_record(record_path, MODULE._read_record(record_path))
    report = MODULE.build_report("Comparison", [record])
    markdown = MODULE.render_markdown(report)

    assert report["report_policy"]["ranking"] == "not_performed"
    assert record["source_sha256"]
    assert "SplitGNN + Transformer" in markdown
    assert "ranking: `not_performed`" in markdown


def test_invalid_policy_temporary_artifact_and_mixed_revision_are_rejected(tmp_path) -> None:
    invalid_path = tmp_path / "comp_splitgnn_seed30.json"
    invalid_payload = _record_payload()
    invalid_payload["selection_policy"] = "test_auc"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_only"):
        MODULE.validate_record(invalid_path, MODULE._read_record(invalid_path))

    smoke_path = tmp_path / "comp_smoke_seed30.json"
    smoke_path.write_text(json.dumps(_record_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="temporary or smoke"):
        MODULE.validate_record(smoke_path, MODULE._read_record(smoke_path))

    first_path = tmp_path / "comp_a_seed30.json"
    second_path = tmp_path / "comp_b_seed31.json"
    first_path.write_text(json.dumps(_record_payload(data_revision="release-a")), encoding="utf-8")
    second_path.write_text(json.dumps(_record_payload(data_revision="release-b")), encoding="utf-8")
    first = MODULE.validate_record(first_path, MODULE._read_record(first_path))
    second = MODULE.validate_record(second_path, MODULE._read_record(second_path))
    with pytest.raises(ValueError, match="mixed data revisions"):
        MODULE.build_report("Comparison", [first, second])


def test_cli_writes_auditable_json_and_markdown_outputs(tmp_path) -> None:
    record_path = tmp_path / "comp_splitgnn_seed30.json"
    output_root = tmp_path / "report"
    record_path.write_text(json.dumps(_record_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--record",
            str(record_path),
            "--output_root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    emitted = json.loads(completed.stdout)
    report = json.loads((output_root / "auditable_comparison.json").read_text(encoding="utf-8"))
    markdown = (output_root / "auditable_comparison.md").read_text(encoding="utf-8")
    assert Path(emitted["json_path"]).is_file()
    assert report["records"][0]["source_record"] == record_path.name
    assert "test_metric_policy: `reported_after_validation_only_selection`" in markdown
