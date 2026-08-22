from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(
    "reason/g2-i-risk-audit-20260731/"
    "c-gate-ortools-grouped-safe-sky-repair-20260804-v1"
)
LAYOUT_ROOT = Path(
    "reason/g2-i-risk-audit-20260731/"
    "c-gate-ortools-common-l1-20260804-v2/layouts"
)


def _load_audit_module():
    path = Path("tools/audit_ortools_v10_grouped_safe_sky_calibration.py")
    spec = importlib.util.spec_from_file_location("audit_ortools_v10_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[Path, dict[str, Path], dict[str, Path]]:
    aggregate = ROOT / "ortools-v10-three-ancestor-l1-calibration-aggregate.json"
    layouts = {
        "g2-i-calibration-ancestor-00": LAYOUT_ROOT
        / "ancestor-00/splits/calibration/city-ca15aabd44bb0d3d",
        "g2-i-calibration-ancestor-03": LAYOUT_ROOT
        / "ancestor-03/splits/calibration/city-eb394ee3e0f415be",
        "g2-i-calibration-ancestor-05": LAYOUT_ROOT
        / "ancestor-05/splits/calibration/city-28fb9135e27bc7db",
    }
    reports = {
        "g2-i-calibration-ancestor-00": ROOT
        / "smoke-ancestor-00/replays/"
        "g2-i-calibration-ancestor-00__ortools-public-atlas-routing-v10-grouped-safe-sky.public.json",
        "g2-i-calibration-ancestor-03": ROOT
        / "rerun-ancestor-03/replays/"
        "g2-i-calibration-ancestor-03__ortools-public-atlas-routing-v10-grouped-safe-sky.public.json",
        "g2-i-calibration-ancestor-05": ROOT
        / "rerun-ancestor-05/replays/"
        "g2-i-calibration-ancestor-05__ortools-public-atlas-routing-v10-grouped-safe-sky.public.json",
    }
    return aggregate, layouts, reports


def test_v10_calibration_audit_retains_zero_confirmation_and_stays_development_only(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    aggregate, layouts, reports = _inputs()
    output = tmp_path / "audit.json"

    audit = module.build(
        aggregate_path=aggregate,
        layouts=layouts,
        reports=reports,
        output=output,
    )

    assert output.is_file()
    assert audit["formal_score_eligible"] is False
    assert audit["summary"] == {
        "independent_layout_ancestor_count": 3,
        "nonzero_confirmation_ancestor_count": 2,
        "safe_completion_ancestor_count": 3,
        "total_anonymous_confirmation_receipt_count": 2,
        "total_observation_receipt_count": 374,
    }
    assert audit["interpretation"]["private_truth_read"] is False
    assert audit["interpretation"]["zero_confirmation_is_retained"] is True
    assert audit["rows"][2]["layout_ancestor"] == "g2-i-calibration-ancestor-05"
    assert audit["rows"][2]["anonymous_confirmation_receipt_count"] == 0

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build(
            aggregate_path=aggregate,
            layouts=layouts,
            reports=reports,
            output=output,
        )
