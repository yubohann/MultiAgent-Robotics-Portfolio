from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.cpu_test_runner import (
    CPU_TEST_RUNNER_SCHEMA,
    CpuTestRunnerError,
    build_test_chunks,
    run_cpu_test_chunks,
    verify_run_report,
)


def _tests(root: Path, count: int = 3) -> Path:
    tests = root / "tests"
    tests.mkdir()
    for index in range(count):
        tests.joinpath(f"test_{index:02d}.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return tests


def test_chunk_plan_is_sorted_and_complete(tmp_path: Path) -> None:
    tests = _tests(tmp_path, count=5)
    chunks = build_test_chunks(tests, chunk_size=2)
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-001", "chunk-002", "chunk-003"]
    assert [item for chunk in chunks for item in chunk.test_files] == [
        "test_00.py",
        "test_01.py",
        "test_02.py",
        "test_03.py",
        "test_04.py",
    ]


def test_runner_records_all_chunk_exit_codes_and_hash_bound_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests = _tests(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        code = 1 if command[-1].endswith("test_01.py") else 0
        return subprocess.CompletedProcess(command, code, stdout=f"{command[-1]}\n".encode())

    monkeypatch.setattr("rivermark_benchmark.cpu_test_runner.subprocess.run", fake_run)
    output = tmp_path.parent / "cpu-test-report"
    report_path = run_cpu_test_chunks(test_root=tests, output_dir=output, chunk_size=1)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == CPU_TEST_RUNNER_SCHEMA
    assert report["status"] == "failed"
    assert [row["return_code"] for row in report["chunks"]] == [0, 1, 0]
    assert len(calls) == 3
    assert calls[0][-1] == "tests/test_00.py"
    assert verify_run_report(report_path) == ()


def test_runner_report_detects_log_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tests = _tests(tmp_path, count=1)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=b"passed\n")

    monkeypatch.setattr("rivermark_benchmark.cpu_test_runner.subprocess.run", fake_run)
    output = tmp_path.parent / "cpu-test-tamper"
    report_path = run_cpu_test_chunks(test_root=tests, output_dir=output, chunk_size=1)
    output.joinpath("logs", "chunk-001.txt").write_bytes(b"changed\n")
    assert "log hash does not match" in verify_run_report(report_path)[0]


def test_runner_rejects_existing_or_repository_output(tmp_path: Path) -> None:
    tests = _tests(tmp_path, count=1)
    existing = tmp_path.parent / "existing-cpu-report"
    existing.mkdir()
    with pytest.raises(CpuTestRunnerError, match="must not already exist"):
        run_cpu_test_chunks(test_root=tests, output_dir=existing)
    with pytest.raises(CpuTestRunnerError, match="outside the repository"):
        run_cpu_test_chunks(test_root=tests, output_dir=ROOT / "generated-test-report")
