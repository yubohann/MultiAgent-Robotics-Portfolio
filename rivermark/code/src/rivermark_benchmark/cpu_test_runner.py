"""Run the complete CPU test suite in auditable, independently recorded chunks.

Some Windows execution hosts can lose the final stdout from a long-lived test
process even though its child process continues and exits.  This runner starts
one pytest child at a time and atomically updates a small report after every
chunk.  The report is test evidence only: it never imports Isaac, Torch, a
dataset payload, or a GPU runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

CPU_TEST_RUNNER_SCHEMA = "org.rivermark.cpu-test-chunk-run.v1"


class CpuTestRunnerError(ValueError):
    """Raised when a chunked CPU test run cannot be audited safely."""


@dataclass(frozen=True)
class TestChunk:
    """A deterministic group of relative pytest file paths."""

    chunk_id: str
    test_files: tuple[str, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_test_chunks(test_root: Path, *, chunk_size: int) -> tuple[TestChunk, ...]:
    """List every checked test file in deterministic, bounded groups."""

    root = Path(test_root).expanduser().resolve()
    if not root.is_dir():
        raise CpuTestRunnerError("test root must be an existing directory")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise CpuTestRunnerError("chunk size must be a positive integer")
    files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("test_*.py"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not files:
        raise CpuTestRunnerError("test root contains no test_*.py files")
    return tuple(
        TestChunk(
            chunk_id=f"chunk-{index:03d}",
            test_files=files[start : start + chunk_size],
        )
        for index, start in enumerate(range(0, len(files), chunk_size), start=1)
    )


def _new_report(chunks: Sequence[TestChunk], *, test_root: Path, chunk_size: int) -> dict[str, Any]:
    return {
        "schema": CPU_TEST_RUNNER_SCHEMA,
        "status": "running",
        "test_root": test_root.name,
        "chunk_size": chunk_size,
        "started_wall_time_ns": time.time_ns(),
        "finished_wall_time_ns": None,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "test_files": list(chunk.test_files),
                "status": "pending",
                "return_code": None,
                "duration_s": None,
                "log_relative_path": None,
                "log_sha256": None,
                "log_bytes": None,
            }
            for chunk in chunks
        ],
    }


def report_sha256(report: Mapping[str, Any]) -> str:
    """Hash a report while excluding its self-reference."""

    payload = dict(report)
    payload.pop("report_sha256", None)
    return _sha256_bytes(_canonical_json(payload))


def verify_run_report(path: Path) -> tuple[str, ...]:
    """Verify a completed report and its exact chunk logs without pytest."""

    report_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"cannot read report: {exc}",)
    if not isinstance(report, Mapping):
        return ("report must be a JSON object",)
    issues: list[str] = []
    if report.get("schema") != CPU_TEST_RUNNER_SCHEMA:
        issues.append("report schema is invalid")
    if report.get("status") not in {"passed", "failed"}:
        issues.append("report is not terminal")
    if report.get("report_sha256") != report_sha256(report):
        issues.append("report self-hash does not match")
    chunks = report.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return (*issues, "report chunks are missing")
    seen_files: set[str] = set()
    return_codes: list[int] = []
    for index, chunk in enumerate(chunks):
        label = f"chunks[{index}]"
        if not isinstance(chunk, Mapping):
            issues.append(f"{label} is not an object")
            continue
        files = chunk.get("test_files")
        def valid_relative_test_path(item: Any) -> bool:
            if not isinstance(item, str):
                return False
            path = PurePosixPath(item)
            return (
                not path.is_absolute()
                and ".." not in path.parts
                and path.name.startswith("test_")
                and path.suffix == ".py"
            )

        if not isinstance(files, list) or not files or any(not valid_relative_test_path(item) for item in files):
            issues.append(f"{label} has invalid test files")
        elif seen_files.intersection(files):
            issues.append(f"{label} repeats a test file")
        else:
            seen_files.update(files)
        return_code = chunk.get("return_code")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            issues.append(f"{label} lacks a final return code")
        else:
            return_codes.append(return_code)
        relative_log = chunk.get("log_relative_path")
        if not isinstance(relative_log, str) or not relative_log.startswith("logs/"):
            issues.append(f"{label} log path is invalid")
            continue
        log_path = report_path.parent / relative_log
        if not _is_relative_to(log_path.resolve(), report_path.parent):
            issues.append(f"{label} log path escapes report directory")
            continue
        try:
            content = log_path.read_bytes()
        except OSError:
            issues.append(f"{label} log is missing")
            continue
        if chunk.get("log_bytes") != len(content) or chunk.get("log_sha256") != _sha256_bytes(content):
            issues.append(f"{label} log hash does not match")
    expected_status = "passed" if return_codes and all(code == 0 for code in return_codes) else "failed"
    if report.get("status") in {"passed", "failed"} and report.get("status") != expected_status:
        issues.append("terminal report status disagrees with chunk return codes")
    return tuple(issues)


def run_cpu_test_chunks(
    *,
    test_root: Path,
    output_dir: Path,
    chunk_size: int = 8,
    pytest_args: Sequence[str] = ("-q",),
    python_executable: str | None = None,
) -> Path:
    """Run every test file sequentially and atomically record each result."""

    root = Path(test_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if destination.exists():
        raise CpuTestRunnerError("CPU test output directory must not already exist")
    if _is_relative_to(destination, repository_root):
        raise CpuTestRunnerError("CPU test output directory must stay outside the repository")
    if not destination.parent.is_dir():
        raise CpuTestRunnerError("CPU test output parent must exist")
    chunks = build_test_chunks(root, chunk_size=chunk_size)
    destination.mkdir()
    report_path = destination / "cpu_test_report.json"
    report = _new_report(chunks, test_root=root, chunk_size=chunk_size)
    report["report_sha256"] = report_sha256(report)
    _write_json_atomic(report_path, report)

    environment = os.environ.copy()
    source_root = repository_root / "src"
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    executable = python_executable or sys.executable
    for index, chunk in enumerate(chunks):
        # Run one level above ``tests`` so imports such as ``tests.test_evaluator``
        # resolve exactly as they do from a clean repository checkout.  Keep paths
        # in the report relative to ``test_root``; only the subprocess arguments
        # need the root directory prefix.
        pytest_paths = [(PurePosixPath(root.name) / item).as_posix() for item in chunk.test_files]
        command = [executable, "-m", "pytest", *pytest_args, *pytest_paths]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=root.parent,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log = bytes(completed.stdout)
        log_relative_path = f"logs/{chunk.chunk_id}.txt"
        _write_bytes_atomic(destination / log_relative_path, log)
        record = report["chunks"][index]
        record.update(
            {
                "status": "passed" if completed.returncode == 0 else "failed",
                "return_code": int(completed.returncode),
                "duration_s": round(time.monotonic() - started, 6),
                "log_relative_path": log_relative_path,
                "log_sha256": _sha256_bytes(log),
                "log_bytes": len(log),
            }
        )
        report["report_sha256"] = report_sha256(report)
        _write_json_atomic(report_path, report)
    report["status"] = "passed" if all(row["return_code"] == 0 for row in report["chunks"]) else "failed"
    report["finished_wall_time_ns"] = time.time_ns()
    report["report_sha256"] = report_sha256(report)
    _write_json_atomic(report_path, report)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("tests"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--pytest-arg", action="append", default=["-q"])
    parser.add_argument("--verify", type=Path, help="Verify an existing terminal report and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify is not None:
        if args.output_dir is not None:
            raise SystemExit("--output-dir cannot be combined with --verify")
        issues = verify_run_report(args.verify)
        print(json.dumps({"status": "passed" if not issues else "failed", "issues": list(issues)}))
        return 0 if not issues else 1
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --verify is used")
    report = run_cpu_test_chunks(
        test_root=args.test_root,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        pytest_args=tuple(args.pytest_arg),
    )
    issues = verify_run_report(report)
    print(json.dumps({"report": str(report), "status": "passed" if not issues else "failed", "issues": list(issues)}))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
