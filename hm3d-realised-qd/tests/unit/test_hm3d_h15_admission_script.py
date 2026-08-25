from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "admit_hm3d_h15_sensor_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hm3d_h15_admission", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h15_admission_rejects_nonpassing_assembly_audit(tmp_path: Path):
    module = _load_module()
    p06 = {"selected_profile": {"mode": "sparse_range_3d"}}
    audit = {"status": "H15_ASSEMBLY_INCOMPLETE"}

    with pytest.raises(ValueError, match="did not pass"):
        module._validate_assembly(p06, audit)


def test_h15_admission_rejects_row_hash_drift(tmp_path: Path):
    module = _load_module()
    row = tmp_path / "row.json"
    row.write_text(json.dumps({"original": True}), encoding="utf-8")
    p06 = {"selected_profile": {"mode": "sparse_range_3d"}}
    audit = {
        "status": "H15_ASSEMBLY_PASS",
        "matrix": {"status": "PASS", "rows": 6},
        "selected_profile": p06["selected_profile"],
        "row_files": [{"path": str(row), "sha256": "0" * 64} for _ in range(6)],
    }

    with pytest.raises(ValueError, match="hash mismatch"):
        module._validate_assembly(p06, audit)
