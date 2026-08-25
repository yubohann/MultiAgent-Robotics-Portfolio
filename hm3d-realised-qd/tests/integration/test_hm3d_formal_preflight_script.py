from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_hm3d_preflight_script_separates_contract_from_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "audit_hm3d_formal_preflight.py"
    contract_output = tmp_path / "contract.json"
    contract = subprocess.run(
        [
            sys.executable,
            str(script),
            "--contract-only",
            "--output",
            str(contract_output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert contract.returncode == 0, contract.stderr
    assert json.loads(contract_output.read_text(encoding="utf-8"))["status"] == "CONTRACT_PASS"

    runtime_output = tmp_path / "runtime.json"
    runtime = subprocess.run(
        [sys.executable, str(script), "--output", str(runtime_output)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert runtime.returncode == 2, runtime.stderr
    report = json.loads(runtime_output.read_text(encoding="utf-8"))
    assert report["status"] == "RUNTIME_NOT_READY"
    assert report["contract"]["status"] == "CONTRACT_PASS"
    assert report["formal_experiment_start_authorized"] is False
    assert report["formal_results_authorized"] is False
