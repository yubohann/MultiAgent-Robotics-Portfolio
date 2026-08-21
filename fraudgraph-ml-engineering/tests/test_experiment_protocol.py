from __future__ import annotations

import json

import pytest

from fraud_ml_engineering.experiment_protocol import (
    aggregate_metric,
    dedupe_float_values,
    hybrid_checkpoint_path,
    hybrid_summary_path,
    label_fraction_slug,
    load_hybrid_summary,
    load_summary_payload,
    mean_std_metric,
    resolve_checkpoint_mode,
    resolve_seeds,
)


def test_hybrid_artifact_paths_follow_the_canonical_names() -> None:
    result_root = "artifacts/experiments/fusion_ablation"

    assert hybrid_summary_path(result_root, "comp").as_posix().endswith("comp/comp_hybrid_summary.json")
    assert hybrid_checkpoint_path(result_root, "comp").as_posix().endswith("comp/comp_hybrid_fraudgraph.pt")


def test_summary_loading_accepts_utf8_bom_and_nested_summary(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"summary": {"completed": True}}), encoding="utf-8-sig")

    assert load_summary_payload(summary_path) == {"summary": {"completed": True}}
    assert load_hybrid_summary(summary_path) == {"completed": True}
    assert load_hybrid_summary(tmp_path / "missing.json") is None


def test_checkpoint_seed_fraction_and_metric_helpers_are_deterministic() -> None:
    assert resolve_checkpoint_mode(True, "reuse") == "fresh"
    assert resolve_checkpoint_mode(False, "CONTINUE") == "continue"
    assert resolve_checkpoint_mode(False, "invalid") == "reuse"
    assert resolve_seeds(deprecated_seed=-1, seeds=[30, 30, 31]) == [30, 31]
    assert resolve_seeds(deprecated_seed=42, seeds=[30, 31]) == [42]
    assert dedupe_float_values([0.1, 0.10, 0.05]) == [0.1, 0.05]
    assert label_fraction_slug(0.10) == "0p1"
    assert label_fraction_slug(-0.05) == "m0p05"
    assert aggregate_metric([]) == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    aggregate = aggregate_metric([0.2, 0.4])
    assert aggregate == pytest.approx({"mean": 0.3, "std": 0.1, "min": 0.2, "max": 0.4})
    assert mean_std_metric([0.2, 0.4]) == pytest.approx({"mean": 0.3, "std": 0.1})
