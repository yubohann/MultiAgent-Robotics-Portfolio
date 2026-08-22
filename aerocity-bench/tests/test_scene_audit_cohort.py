from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.scene_audit import (
    SCENE_AUDIT_SCHEMA,
    development_scene_audit_plan,
    summarize_development_scene_audit_cohort,
)


def _receipt(split: str, index: int, layout_hash: str) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": SCENE_AUDIT_SCHEMA,
        "status": "PASS",
        "split": split,
        "layout_id": f"city-{index}",
        "layout_hash": layout_hash,
        "task_geometry_hash": f"geometry-{index}",
        "generator_version": "aerocity-generator-ordinary-v3.1",
        "scene_counts": {"buildings": 1, "episodes": 3},
        "generation_rejections_before_acceptance": 0,
        "error_categories": [],
    }
    report["report_hash"] = content_hash(report)
    return report


def _failed_receipt(split: str) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": SCENE_AUDIT_SCHEMA,
        "status": "FAIL",
        "split": split,
        "index": 0,
        "max_attempts": 8,
        "error_categories": ["no_complete_city_episode_candidate"],
        "generation_rejection_count": 8,
    }
    report["report_hash"] = content_hash(report)
    return report


def test_development_scene_audit_plan_is_balanced_and_deterministic() -> None:
    plan = development_scene_audit_plan(4)

    assert len(plan) == 12
    assert plan[:4] == (("train", 0), ("train", 1), ("train", 2), ("train", 3))
    assert plan[-1] == ("calibration", 3)
    with pytest.raises(ValueError, match="positive"):
        development_scene_audit_plan(0)


def test_cohort_summary_is_private_safe_and_detects_duplicate_layouts() -> None:
    root = Path(__file__).parents[1]
    ordinary_config = load_ordinary_config(root / "configs" / "releases" / "ordinary-v1-mini.json")
    receipts = {
        (split, index): _receipt(split, index, f"layout-{split}-{index}")
        for split, index in development_scene_audit_plan(1)
    }
    report = summarize_development_scene_audit_cohort(
        ordinary_config,
        receipts,
        per_split=1,
    )

    assert report["status"] == "PASS"
    assert report["formal_score_eligible"] is False
    assert report["sampling"]["layout_count"] == 3
    rendered = str(report)
    assert "targets" not in rendered
    assert "witness" not in rendered

    duplicate = copy.deepcopy(receipts)
    duplicate[("validation", 0)]["layout_hash"] = duplicate[("train", 0)]["layout_hash"]
    duplicate[("validation", 0)]["report_hash"] = content_hash(
        {key: value for key, value in duplicate[("validation", 0)].items() if key != "report_hash"}
    )
    duplicate_report = summarize_development_scene_audit_cohort(
        ordinary_config,
        duplicate,
        per_split=1,
    )
    assert duplicate_report["status"] == "FAIL"
    assert duplicate_report["all_layout_hashes_unique"] is False


def test_cohort_summary_retains_a_generation_failure_instead_of_crashing() -> None:
    root = Path(__file__).parents[1]
    ordinary_config = load_ordinary_config(root / "configs" / "releases" / "ordinary-v1-mini.json")
    receipts = {
        (split, index): _receipt(split, index, f"layout-{split}-{index}")
        for split, index in development_scene_audit_plan(1)
    }
    receipts[("train", 0)] = _failed_receipt("train")

    report = summarize_development_scene_audit_cohort(
        ordinary_config,
        receipts,
        per_split=1,
    )

    assert report["status"] == "FAIL"
    assert report["failed_members"] == [
        {
            "split": "train",
            "index": 0,
            "error_categories": ["no_complete_city_episode_candidate"],
        }
    ]
