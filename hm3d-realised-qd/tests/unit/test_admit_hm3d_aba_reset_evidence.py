from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "admit_hm3d_aba_reset_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("admit_hm3d_aba_reset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_admission_rejects_missing_p01_scene(tmp_path: Path):
    module = _module()
    p01 = tmp_path / "p01.json"
    p01.write_text(
        '{"phase_id":"P01","kind":"asset_lock","origin":"source_license_audit",'
        '"payload":{"scenes":[]}}',
        encoding="utf-8",
    )
    development = tmp_path / "development.json"
    development.write_text(
        '{"status":"DEVELOPMENT_ABA_RESET_PASSED_NOT_FORMAL_P02",'
        '"measured":true,"synthetic":false,"p02_candidate":{},"raw_probes":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="a1, b, and a2"):
        module._verified_p02_payload(
            development_evidence_path=development,
            p01_artifact_path=p01,
        )
