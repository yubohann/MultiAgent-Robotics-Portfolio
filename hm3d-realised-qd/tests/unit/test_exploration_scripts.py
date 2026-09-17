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


def test_mechanism_admission_rejects_the_legacy_qd_control_chain(tmp_path: Path):
    exploration = tmp_path / "exploration.json"
    matrix = tmp_path / "matrix.json"
    output = tmp_path / "mechanism.json"
    _write_json(exploration, {"status": "EXPLORATION_TASK_VALID"})
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
        "mechanism/run_mechanism_matrix.py",
        "--exploration-summary",
        str(exploration),
        "--mechanism-matrix",
        str(matrix),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    payload = _read_json(output)
    assert payload["status"] == "MECHANISM_NOT_READY"
    assert "QD_MECHANISM_MATRIX_SCHEMA_OUTDATED" in payload["reasons"]
    assert "QD_CONTROL_CHAIN_INCOMPLETE" in payload["reasons"]
    assert "QD_ADMISSION_RULE_MISSING" in payload["reasons"]
