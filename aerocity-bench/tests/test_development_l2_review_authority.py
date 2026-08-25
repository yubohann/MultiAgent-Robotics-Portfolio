from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.scene_audit import DEVELOPMENT_AUDIT_SPLITS, SCENE_AUDIT_COHORT_SCHEMA
from tools.build_development_l2_review_authority import (
    _cohort_members,
    development_l2_review_config,
)


def _summary(per_split: int = 4) -> dict[str, object]:
    members: list[dict[str, object]] = []
    for split in DEVELOPMENT_AUDIT_SPLITS:
        for index in range(per_split):
            members.append(
                {
                    "split": split,
                    "index": index,
                    "layout_hash": f"{split[0]}{index}".ljust(64, "0"),
                }
            )
    return {
        "schema": SCENE_AUDIT_COHORT_SCHEMA,
        "status": "PASS",
        "formal_score_eligible": False,
        "sampling": {"layouts_per_split": per_split},
        "members": members,
        "report_hash": content_hash(members),
    }


def test_development_l2_config_preserves_task_contract_and_scales_only_development() -> None:
    root = Path(__file__).parents[1]
    base = load_ordinary_config(root / "configs" / "releases" / "ordinary-v1-mini.json")

    derived = development_l2_review_config(base, layouts_per_split=4)

    assert derived.raw["release_kind"] == "CUSTOM"
    assert derived.raw["execution_contract"] == base.raw["execution_contract"]
    assert derived.raw["fleet"] == base.raw["fleet"]
    assert [derived.count(split) for split in DEVELOPMENT_AUDIT_SPLITS] == [4, 4, 4]
    assert derived.count("test_iid") == base.count("test_iid")
    with pytest.raises(ValueError, match="12--18"):
        development_l2_review_config(base, layouts_per_split=3)


def test_cohort_binding_rejects_duplicate_or_nonmatching_members() -> None:
    summary = _summary()

    members = _cohort_members(summary, layouts_per_split=4)

    assert len(members) == 12
    duplicate = copy.deepcopy(summary)
    duplicate["members"].append(copy.deepcopy(duplicate["members"][0]))
    with pytest.raises(ValueError, match="repeats"):
        _cohort_members(duplicate, layouts_per_split=4)

    missing = copy.deepcopy(summary)
    missing["members"].pop()
    with pytest.raises(ValueError, match="do not match"):
        _cohort_members(missing, layouts_per_split=4)
