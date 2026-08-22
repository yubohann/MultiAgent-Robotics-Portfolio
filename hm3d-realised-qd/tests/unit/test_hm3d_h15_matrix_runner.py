from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATRIX_RUNNER = ROOT / "scripts" / "run_hm3d_h15_sensor_matrix.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("hm3d_h15_matrix_runner", MATRIX_RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_runner_has_complete_serial_matrix_plan(tmp_path: Path):
    module = _load_runner()
    assert tuple(module.FORMAL_H15_SENSOR_PILOT_MODES) == (
        "physics_only",
        "sparse_range_3d",
    )
    assert module._row_path(tmp_path, "sparse_range_3d").name == "row_N4_sparse_range_3d_v3.json"


def test_matrix_runner_rejects_invalid_runtime_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _load_runner()
    missing = tmp_path / "missing.exe"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_hm3d_h15_sensor_matrix.py",
            "--isaac-python",
            str(missing),
            "--scene-id",
            "scene",
            "--collision-usd",
            str(missing),
            "--receiver-positions-json",
            str(missing),
            "--rows-dir",
            str(tmp_path / "rows"),
            "--p06-output",
            str(tmp_path / "p06.json"),
            "--audit-output",
            str(tmp_path / "audit.json"),
            "--ledger-output",
            str(tmp_path / "ledger.json"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="Isaac Python"):
        module.main()
