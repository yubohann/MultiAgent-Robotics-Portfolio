from __future__ import annotations

from pathlib import Path

import pytest

from aerocity_bench.atlas.atlas_audit import audit_inspection_atlas
from aerocity_bench.core.errors import GenerationRejected
from aerocity_bench.generation.compiler import compile_g2i_task_spec
from aerocity_bench.generation.generator_v3 import generate_city_v3
from aerocity_bench.generation.ordinary_config import load_ordinary_config
from aerocity_bench.release.fidelity_audit import compare_l0_l1_rankings

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def ordinary_config():
    return load_ordinary_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def development_city(ordinary_config):
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    for attempt in range(8):
        try:
            return generate_city_v3(ordinary_config, "train", 0, attempt, assets)
        except GenerationRejected:
            continue
    raise AssertionError("expected an admitted deterministic development city")


def test_atlas_geometry_audit_reports_pass_cpu(ordinary_config, development_city) -> None:
    task_spec = compile_g2i_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )
    contract = ordinary_config.raw["execution_contract"]
    report = audit_inspection_atlas(
        development_city,
        task_spec["inspection_atlas"],
        contract,
        fleet_count=ordinary_config.fleet_count,
        episode_duration_s=float(contract["episode"]["duration_s"]),
    )

    assert report["layout_id"] == development_city["layout_id"]
    assert report["formal_score_eligible"] is False
    assert report["cpu_geometry_status"] in {"PASS_CPU", "FAIL"}
    assert report["aggregate"]["cell_count"] > 0
    assert report["aggregate"]["transit_node_count"] > 0
    assert report["budget_bracket"]["fleet_count"] == ordinary_config.fleet_count


def test_fidelity_audit_weights_ancestors_equally() -> None:
    l0_records = []
    l1_records = []
    for method_index, method in enumerate(("a", "b", "c")):
        for ancestor_index in range(3):
            score = 0.1 * (method_index + 1) + 0.01 * ancestor_index
            l0_records.append(
                {
                    "method_id": method,
                    "layout_ancestor": f"layout-{ancestor_index}",
                    "score": score,
                    "execution_level": "L0",
                }
            )
            l1_records.append(
                {
                    "method_id": method,
                    "layout_ancestor": f"layout-{ancestor_index}",
                    "score": score + 0.05 * method_index,
                    "execution_level": "L1",
                }
            )
    report = compare_l0_l1_rankings(l0_records, l1_records)
    assert report["status"] == "MEASURED_NOT_FROZEN"
    assert report["independent_ancestor_count"] == 3
    assert report["spearman_rank_correlation"] == pytest.approx(1.0)
    assert report["contract_freeze_allowed"] is False


def test_fidelity_audit_requires_identical_method_ancestor_pairs() -> None:
    records = [
        {
            "method_id": "a",
            "layout_ancestor": "layout-0",
            "score": 0.5,
            "execution_level": "L0",
        }
    ]
    with pytest.raises(ValueError, match="identical method/ancestor pairs"):
        compare_l0_l1_rankings(records, [])


def test_fidelity_audit_rejects_wrong_execution_level() -> None:
    records = [
        {
            "method_id": "a",
            "layout_ancestor": "layout-0",
            "score": 0.5,
            "execution_level": "L2",
        }
    ]
    with pytest.raises(ValueError, match="wrong execution level"):
        compare_l0_l1_rankings(records, records)
