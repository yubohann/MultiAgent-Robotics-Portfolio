from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_audit_module():
    path = Path("tools/audit_external_candidate_semantics.py")
    spec = importlib.util.spec_from_file_location("audit_external_candidate_semantics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_keeps_current_candidates_out_of_gate_c() -> None:
    module = _load_audit_module()

    report = module.audit(Path("external/candidate-input-semantics.json"))

    assert report["candidate_count"] == 13
    assert report["c_gate_eligible"] == []
    assert report["status"] == "BLOCKED_NO_SUBSTANTIVE_EXTERNAL_G2_I_METHOD"


def test_eligible_candidate_requires_closed_semantics_and_license(tmp_path: Path) -> None:
    module = _load_audit_module()
    registry_path = Path("external/candidate-input-semantics.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = registry["candidates"][0]
    candidate["c_gate_eligible"] = True
    candidate["task_class"] = "three-dimensional geometry-search inspection routing"
    invalid_path = tmp_path / "bad-registry.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="three-city L0 and L1 closed integration"):
        module.audit(invalid_path)


def test_candidate_registry_rejects_ambiguous_schema(tmp_path: Path) -> None:
    module = _load_audit_module()
    registry_path = Path("external/candidate-input-semantics.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["candidates"][0]["undocumented_claim"] = "external baseline"
    registry_path = tmp_path / "ambiguous-registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="fields differ"):
        module.audit(registry_path)


def test_eligible_candidate_requires_a_full_upstream_revision(tmp_path: Path) -> None:
    module = _load_audit_module()
    registry_path = Path("external/candidate-input-semantics.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = registry["candidates"][0]
    candidate["c_gate_eligible"] = True
    candidate["task_class"] = "three-dimensional geometry-search inspection routing"
    candidate["upstream_commit"] = "main"
    invalid_path = tmp_path / "unlocked-revision.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="locked full upstream revision"):
        module.audit(invalid_path)


def test_eligible_perception_candidate_is_rejected_from_geometry_search_gate(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    registry = json.loads(
        Path("external/candidate-input-semantics.json").read_text(encoding="utf-8")
    )
    candidate = registry["candidates"][0]
    candidate.update(
        {
            "c_gate_eligible": True,
            "task_class": "three-dimensional perception geometry-search method",
            "missing_or_forbidden_inputs": [],
            "license_status": "verified",
            "integration_status": "three-city L0 and L1 closed integration",
        }
    )
    invalid_path = tmp_path / "perception-registry.json"
    invalid_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="perception-search method"):
        module.audit(invalid_path)
