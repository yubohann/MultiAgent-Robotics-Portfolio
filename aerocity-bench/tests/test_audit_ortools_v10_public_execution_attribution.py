from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aerocity_bench.errors import ValidationError
from aerocity_bench.public_boundary import assert_public_fields

ROOT = Path(
    "reason/g2-i-risk-audit-20260731/"
    "c-gate-ortools-grouped-safe-sky-replication-20260804-v1"
)


def _load_module():
    path = Path("tools/audit_ortools_v10_public_execution_attribution.py")
    spec = importlib.util.spec_from_file_location("ortools_v10_public_execution_attribution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reports() -> dict[str, Path]:
    root = ROOT / "replays"
    return {
        f"g2-i-calibration-ancestor-{ancestor}": root
        / f"g2-i-calibration-ancestor-{ancestor}__ortools-public-atlas-routing-baseline.public.json"
        for ancestor in ("00", "03", "05")
    }


def test_public_execution_attribution_retains_zero_without_claiming_hidden_cause(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "attribution.json"

    attribution = module.build(
        calibration_audit_path=ROOT / "replication-public-calibration-audit.json",
        reports=_reports(),
        output=output,
    )

    assert output.is_file()
    assert attribution["formal_score_eligible"] is False
    retained = attribution["retained_zero_confirmation"]
    assert retained["layout_ancestor"] == "g2-i-calibration-ancestor-05"
    assert retained["public_execution_closed"] is True
    assert retained["status"] == "PUBLIC_EXECUTION_CAUSE_UNIDENTIFIABLE"
    comparison = retained["comparison_to_nonzero_replays"]
    assert comparison["zero_assigned_cell_count"] == 75
    assert comparison["zero_observation_receipt_count"] == 160
    assert comparison["observation_receipt_count_range"] == [98, 116]
    assert_public_fields(attribution)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build(
            calibration_audit_path=ROOT / "replication-public-calibration-audit.json",
            reports=_reports(),
            output=output,
        )


def test_public_execution_attribution_rejects_leaked_target_field(tmp_path: Path) -> None:
    module = _load_module()
    reports = _reports()
    leaked = json.loads(reports["g2-i-calibration-ancestor-05"].read_text(encoding="utf-8"))
    leaked["target_coordinate"] = [0.0, 0.0, 0.0]
    leaked_path = tmp_path / "leaked-public-report.json"
    leaked_path.write_text(json.dumps(leaked), encoding="utf-8")
    reports["g2-i-calibration-ancestor-05"] = leaked_path

    with pytest.raises(ValidationError, match="private truth keys"):
        module.build(
            calibration_audit_path=ROOT / "replication-public-calibration-audit.json",
            reports=reports,
            output=tmp_path / "attribution.json",
        )
