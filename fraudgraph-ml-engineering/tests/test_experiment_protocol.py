import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_ml_engineering.experiment_protocol import (
    aggregate_metric,
    dedupe_float_values,
    label_fraction_slug,
    mean_std_metric,
    resolve_seeds,
)


def test_dedupe_float_values():
    assert dedupe_float_values([0.1, 0.1, 0.2]) == [0.1, 0.2]


def test_label_fraction_slug_is_filesystem_safe():
    assert label_fraction_slug(0.1) == "0p1"
    assert label_fraction_slug(0.25) == "0p25"


def test_aggregate_metric_summary():
    summary = aggregate_metric([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["std"] > 0


def test_aggregate_metric_empty():
    summary = aggregate_metric([])
    assert summary == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_mean_std_metric():
    summary = mean_std_metric([1.0, 2.0, 3.0])
    assert summary["mean"] == 2.0
    assert summary["std"] > 0


def test_resolve_seeds_respects_deprecated_override():
    seeds = resolve_seeds(deprecated_seed=7, seeds=[1, 2, 3])
    assert seeds == [7]


def test_resolve_seeds_falls_back_to_defaults():
    seeds = resolve_seeds(deprecated_seed=-1, seeds=[])
    assert isinstance(seeds, list)
    assert len(seeds) >= 1