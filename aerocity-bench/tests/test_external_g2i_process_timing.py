from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _tool():
    path = Path("tools/diagnose_external_g2i_process_timing.py")
    spec = importlib.util.spec_from_file_location("external_g2i_timing_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_summary_is_monotonic_and_reports_call_count() -> None:
    tool = _tool()
    summary = tool._summary([0.003, 0.001, 0.002])
    assert summary["call_count"] == 3
    assert summary["p50_s"] <= summary["p95_s"] <= summary["p99_s"] <= summary["max_s"]


def test_timing_summary_rejects_invalid_samples() -> None:
    tool = _tool()
    with pytest.raises(ValueError, match="non-empty"):
        tool._summary([])
    with pytest.raises(ValueError, match="finite"):
        tool._summary([float("nan")])
