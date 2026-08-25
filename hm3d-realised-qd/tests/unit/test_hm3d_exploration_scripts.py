from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p09_freeze_refuses_unfinished_p08_and_synthetic_evidence(tmp_path: Path):
    protocol = ROOT / "configs" / "external" / "hm3d_multi_uav_exploration_protocol.json"
    p07 = tmp_path / "p07.json"
    p08 = tmp_path / "p08.json"
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "p09.json"
    _write_json(p07, {"status": "P07_EXPLORATION_TASK_VALID"})
    _write_json(p08, {"status": "P08_NOT_READY"})
    _write_json(
        evidence,
        {
            "synthetic": True,
            "failure_denominators": {"complete": False},
            "test_scene_accessed_before_freeze": False,
        },
    )
    result = _run_script(
        "freeze_hm3d_p09_protocol.py",
        "--protocol",
        str(protocol),
        "--p07-summary",
        str(p07),
        "--p08-admission",
        str(p08),
        "--runtime-evidence",
        str(evidence),
        "--output",
        str(output),
    )
    assert result.returncode == 2
    payload = _read_json(output)
    assert payload["status"] == "P09_FREEZE_REFUSED"
    assert "P08_NOT_COMPLETE" in payload["reasons"]
    assert "P08_REAL_RUNTIME_EVIDENCE_REQUIRED" in payload["reasons"]
    assert "SYNTHETIC_OR_MOCK_EVIDENCE_FORBIDDEN" in payload["reasons"]


def test_p08_admission_rejects_the_legacy_qd_control_chain(tmp_path: Path):
    p07 = tmp_path / "p07.json"
    matrix = tmp_path / "matrix.json"
    output = tmp_path / "p08.json"
    _write_json(p07, {"status": "P07_EXPLORATION_TASK_VALID"})
    _write_json(
        matrix,
        {
            "schema_version": "hm3d-exploration-mechanism-matrix-v1",
            "status": "NOT_FORMAL_RESULT",
            "adjacent_ablation_chain": ["a", "b", "c", "d"],
            "qd_controls": ["no_qd", "planned_descriptor_archive"],
        },
    )

    result = _run_script(
        "run_hm3d_p08_mechanism_matrix.py",
        "--p07-summary",
        str(p07),
        "--mechanism-matrix",
        str(matrix),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    payload = _read_json(output)
    assert payload["status"] == "P08_NOT_READY"
    assert "QD_MECHANISM_MATRIX_SCHEMA_OUTDATED" in payload["reasons"]
    assert "QD_CONTROL_CHAIN_INCOMPLETE" in payload["reasons"]
    assert "QD_ADMISSION_RULE_MISSING" in payload["reasons"]
