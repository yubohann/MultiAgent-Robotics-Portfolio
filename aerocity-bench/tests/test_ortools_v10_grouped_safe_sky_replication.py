from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPAIR_ROOT = Path(
    "reason/g2-i-risk-audit-20260731/"
    "c-gate-ortools-grouped-safe-sky-repair-20260804-v1"
)
LAYOUT_ROOT = Path(
    "reason/g2-i-risk-audit-20260731/"
    "c-gate-ortools-common-l1-20260804-v2/layouts"
)


def _load_runner():
    path = Path("tools/run_ortools_v10_grouped_safe_sky_replication.py")
    spec = importlib.util.spec_from_file_location("ortools_v10_replication_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layouts() -> dict[str, Path]:
    return {
        "g2-i-calibration-ancestor-00": LAYOUT_ROOT
        / "ancestor-00/splits/calibration/city-ca15aabd44bb0d3d",
        "g2-i-calibration-ancestor-03": LAYOUT_ROOT
        / "ancestor-03/splits/calibration/city-eb394ee3e0f415be",
        "g2-i-calibration-ancestor-05": LAYOUT_ROOT
        / "ancestor-05/splits/calibration/city-28fb9135e27bc7db",
    }


def test_v10_replication_plan_is_outcome_blind_and_hash_binds_external_inputs(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    repository = Path(__file__).parents[1]
    cf2x_usd = repository.parents[1] / "assets/new/cf2x.usd"
    plan_path = tmp_path / "plan.json"

    plan = runner.build_plan(
        layouts=_layouts(),
        release_config=repository / "configs/releases/ordinary-v1-mini.json",
        cf2x_usd=cf2x_usd,
        isaac_python=Path(sys.executable),
        adapter_manifest_path=(
            REPAIR_ROOT / "external/ortools-adapter-manifest-v10-grouped-safe-sky.json"
        ),
        output=plan_path,
    )

    assert runner._hash_bound_plan(plan_path)["plan_hash"] == plan["plan_hash"]
    assert plan["formal_score_eligible"] is False
    assert plan["status"] == "PRECOMMITTED_UNRUN"
    assert plan["layout_ancestors"] == list(runner.ANCESTORS)
    assert [record["layout_ancestor"] for record in plan["records"]] == list(runner.ANCESTORS)
    assert all("confirmation" not in record for record in plan["records"])
    assert "cf2x_usd_path" not in plan
    assert "isaac_python_path" not in plan
    assert plan["execution"]["retry_decision_reads_outcome"] is False

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.build_plan(
            layouts=_layouts(),
            release_config=repository / "configs/releases/ordinary-v1-mini.json",
            cf2x_usd=cf2x_usd,
            isaac_python=Path(sys.executable),
            adapter_manifest_path=(
                REPAIR_ROOT / "external/ortools-adapter-manifest-v10-grouped-safe-sky.json"
            ),
            output=plan_path,
        )
