from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_panel_builder():
    path = Path("tools/build_aco3d_g2i_l0_panel.py")
    spec = importlib.util.spec_from_file_location("aco3d_g2i_l0_panel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(city: str, *, returned: bool = True) -> dict[str, object]:
    return {
        "schema": "org.aerocity.bench.aco3d-public-atlas-smoke.v1",
        "scope": "calibration_only_source_locked_translation_smoke",
        "pass_semantics": "safety_and_abi_integrity",
        "formal_score_eligible": False,
        "return_closure_required": True,
        "pass": True,
        "upstream": {
            "url": "https://example.invalid/aco",
            "commit": "a" * 40,
            "license": "MIT",
            "source_lock_sha256": "b" * 64,
            "adapter_version": "translation-v1",
            "source_checkout_verified": True,
            "upstream_runtime_executed": False,
        },
        "adapter": {
            "adapter_source_sha256": "c" * 64,
            "runner_source_sha256": "d" * 64,
        },
        "public_input_hashes": {"city": city, "release_config": "e" * 64},
        "execution": {
            "formal_score_eligible": False,
            "all_returned_home": returned,
            "failure_categories": [],
            "collision_count": 0,
            "out_of_bounds_actions": 0,
            "deadline_miss_tick_count": 0,
            "task_time_s": 300.0,
            "receipt_count": 6000,
            "observe_request_count": 20,
            "confirmation_count": 1,
            "inspection_coverage": {"area_fraction": 0.8, "cell_fraction": 0.75},
        },
    }


def test_panel_requires_three_matching_safe_returning_calibration_reports(
    tmp_path: Path,
) -> None:
    module = _load_panel_builder()
    paths: dict[str, Path] = {}
    for index, label in enumerate(("ancestor-00", "ancestor-01", "ancestor-02")):
        path = tmp_path / f"{label}.json"
        module.write_json(path, _report(f"{index:064x}"))
        paths[label] = path

    panel = module.build_panel(paths)

    assert panel["status"] == "L0_CALIBRATION_PANEL_PASS_NOT_GATE_C"
    assert panel["formal_score_eligible"] is False
    assert panel["gate_c_eligible"] is False
    assert panel["aggregate"]["all_returned_home"] is True
    assert panel["aggregate"]["total_anonymous_confirmation_count"] == 3


def test_panel_rejects_a_nonreturning_replay(tmp_path: Path) -> None:
    module = _load_panel_builder()
    paths: dict[str, Path] = {}
    for index, label in enumerate(("ancestor-00", "ancestor-01", "ancestor-02")):
        path = tmp_path / f"{label}.json"
        module.write_json(path, _report(f"{index:064x}", returned=label != "ancestor-02"))
        paths[label] = path

    with pytest.raises(ValueError, match="did not return"):
        module.build_panel(paths)
