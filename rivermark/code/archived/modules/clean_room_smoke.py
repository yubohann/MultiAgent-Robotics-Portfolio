"""Run the public CPU smoke from a temporary clean Git clone.

This command is a reproducibility check for the public researcher entry path.
It does not provide second-machine evidence, an Isaac reproduction, or a
formal dataset episode. The temporary clone and fixture are removed after the
report is written; only a path-free, hash-bound report remains.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from .provenance import repository_root


CLEAN_ROOM_SMOKE_SCHEMA = "org.rivermark.benchmark.clean-room-smoke.v1"
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/Users/|/home/|/tmp/|\\\\)")
_FORBIDDEN_PUBLIC_FIELDS = (
    "evaluator_truth_sha256",
    "private_manifest",
    "hidden_target",
    "target_coordinates",
)


class CleanRoomSmokeError(ValueError):
    """Raised when a clean-room smoke request would overwrite user data."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, cwd: Path | None = None, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
    """Run a bounded command while keeping command output out of the report."""

    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )


def _git_value(root: Path, *arguments: str) -> tuple[str | None, str | None]:
    try:
        result = _run(["git", "-C", str(root), *arguments])
    except (OSError, subprocess.TimeoutExpired):
        return None, "git_unavailable"
    if result.returncode != 0:
        return None, "git_command_failed"
    return result.stdout.strip(), None


def _empty_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise CleanRoomSmokeError(f"refusing to write into a non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _remove_tree_with_retry(path: Path, *, timeout_s: float = 5.0) -> bool:
    """Remove a disposable Windows worktree after short-lived handle release.

    Git and a completed child Python process can leave a directory handle open
    briefly on Windows.  Retrying only the directory cleanup avoids treating a
    clean-room pass as failed while still refusing to silently accumulate clone
    trees when a handle remains permanently open.
    """

    deadline = time.monotonic() + timeout_s
    last_error_path: str | None = None

    def onerror(function: Any, name: str, _exc_info: Any) -> None:
        nonlocal last_error_path
        last_error_path = name
        # Git packfiles are commonly read-only on Windows.  Clear only the
        # write-protection bit and retry the same unlink/rmdir operation; an
        # actual open-handle error remains fail-closed and is retried above.
        os.chmod(name, stat.S_IWRITE)
        function(name)

    while True:
        try:
            shutil.rmtree(path, onerror=onerror)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if time.monotonic() >= deadline:
                if path.exists() and last_error_path is not None:
                    raise CleanRoomSmokeError(
                        f"temporary clean-room path remains locked: {last_error_path}"
                    )
                return not path.exists()
            time.sleep(0.1)


@contextlib.contextmanager
def _temporary_directory_with_retry(*, prefix: str) -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        if not _remove_tree_with_retry(path):
            raise CleanRoomSmokeError("temporary clean-room clone could not be removed")


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_ABSOLUTE_PATH.search(value))
    if isinstance(value, dict):
        return any(_has_absolute_path(key) or _has_absolute_path(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _contains_forbidden_field(value: Any) -> bool:
    if isinstance(value, dict):
        if any(field in value for field in _FORBIDDEN_PUBLIC_FIELDS):
            return True
        return any(_contains_forbidden_field(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def _failed_report(*, source_revision: str | None, code: str, elapsed_ms: float) -> dict[str, Any]:
    return {
        "schema": CLEAN_ROOM_SMOKE_SCHEMA,
        "status": "failed",
        "claim_boundary": "clean_clone_cpu_entry_smoke_only",
        "formal_benchmark_admission": False,
        "failure_code": code,
        "source": {"revision": source_revision, "worktree_dirty": code == "source_worktree_dirty"},
        "checks": {
            "source_clean": code != "source_worktree_dirty",
            "clone_revision": False,
            "clone_worktree_clean": False,
            "cpu_researcher_smoke": False,
            "no_private_truth": True,
            "no_absolute_paths": True,
            "formal_benchmark_admission": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "wall_time_ms": round(elapsed_ms, 3),
            "temporary_clone_retained": False,
        },
    }


def run_clean_room_smoke(
    output_root: Path,
    *,
    source_root: Path | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run the CPU entry smoke in a disposable clone and write its report."""

    root = output_root.resolve()
    _empty_destination(root)
    started = time.perf_counter()
    source = (source_root or repository_root()).resolve()
    revision, revision_error = _git_value(source, "rev-parse", "HEAD")
    status, status_error = _git_value(source, "status", "--porcelain", "--untracked-files=all")
    if revision_error is not None:
        report = _failed_report(source_revision=None, code=revision_error, elapsed_ms=(time.perf_counter() - started) * 1000.0)
        _write_json(root / "clean_room_report.json", report)
        return report
    if status_error is not None:
        report = _failed_report(source_revision=revision, code=status_error, elapsed_ms=(time.perf_counter() - started) * 1000.0)
        _write_json(root / "clean_room_report.json", report)
        return report
    if not _REVISION.fullmatch(revision or ""):
        report = _failed_report(source_revision=revision, code="malformed_source_revision", elapsed_ms=(time.perf_counter() - started) * 1000.0)
        _write_json(root / "clean_room_report.json", report)
        return report
    if status:
        report = _failed_report(source_revision=revision, code="source_worktree_dirty", elapsed_ms=(time.perf_counter() - started) * 1000.0)
        _write_json(root / "clean_room_report.json", report)
        return report

    clone_revision: str | None = None
    smoke_report: dict[str, Any] | None = None
    child_returncode: int | None = None
    failure_code: str | None = None
    with _temporary_directory_with_retry(prefix="rivermark-clean-room-") as temporary:
        clone = temporary / "clone"
        try:
            clone_result = _run(
                ["git", "clone", "--no-local", "--no-hardlinks", "--quiet", str(source), str(clone)],
                timeout_s=timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired):
            clone_result = None
            failure_code = "clone_failed"
        if clone_result is not None:
            if clone_result.returncode != 0:
                failure_code = "clone_failed"
            else:
                clone_revision, clone_revision_error = _git_value(clone, "rev-parse", "HEAD")
                clone_status, clone_status_error = _git_value(clone, "status", "--porcelain", "--untracked-files=all")
                if clone_revision_error or clone_status_error:
                    failure_code = "clone_audit_failed"
                elif clone_revision != revision or clone_status:
                    failure_code = "clone_not_clean_or_mismatched"
                else:
                    # Keep generated fixture bytes outside the clone. The
                    # researcher smoke records source cleanliness after it
                    # writes its output, so placing it under the checkout
                    # would create a false dirty-worktree failure.
                    smoke_output = temporary / "smoke-output"
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = str(clone / "src")
                    environment["PYTHONNOUSERSITE"] = "1"
                    environment["PYTHONDONTWRITEBYTECODE"] = "1"
                    command = [sys.executable, "-m", "rivermark_benchmark.researcher_entry", str(smoke_output)]
                    try:
                        smoke_result = subprocess.run(
                            command,
                            # Keep the child process out of the clone's current
                            # directory.  Windows can retain a cwd handle for
                            # a short interval after interpreter shutdown,
                            # which otherwise prevents deterministic cleanup.
                            cwd=str(temporary),
                            env=environment,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=timeout_s,
                            check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        smoke_result = None
                        failure_code = "researcher_smoke_failed"
                    if smoke_result is not None:
                        child_returncode = smoke_result.returncode
                        report_path = smoke_output / "researcher_smoke_report.json"
                        if smoke_result.returncode != 0 or not report_path.is_file():
                            failure_code = "researcher_smoke_failed"
                        else:
                            try:
                                smoke_report = json.loads(report_path.read_text(encoding="utf-8"))
                            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                                failure_code = "researcher_report_invalid"
                            else:
                                if not isinstance(smoke_report, dict):
                                    failure_code = "researcher_report_invalid"
                                elif (
                                    smoke_report.get("status") != "passed"
                                    or smoke_report.get("formal_benchmark_admission") is not False
                                    or smoke_report.get("checks", {}).get("private_truth_present") is not False
                                    or smoke_report.get("checks", {}).get("isaac_started") is not False
                                    or smoke_report.get("source", {}).get("revision") != clone_revision
                                    or smoke_report.get("source", {}).get("worktree_dirty") is not False
                                ):
                                    failure_code = "researcher_smoke_boundary_failed"
                                elif _contains_forbidden_field(smoke_report) or _has_absolute_path(smoke_report):
                                    failure_code = "researcher_report_not_public"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if failure_code is not None or smoke_report is None or clone_revision is None:
        report = _failed_report(source_revision=revision, code=failure_code or "clean_room_failed", elapsed_ms=elapsed_ms)
        report["checks"]["source_clean"] = True
        report["clone"] = {"strategy": "git_clone_no_local_no_hardlinks", "revision": clone_revision}
        report["runtime"].update({"child_returncode": child_returncode, "temporary_clone_retained": False})
        _write_json(root / "clean_room_report.json", report)
        return report

    smoke_digest = hashlib.sha256(
        (json.dumps(smoke_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()
    report = {
        "schema": CLEAN_ROOM_SMOKE_SCHEMA,
        "status": "passed",
        "claim_boundary": "clean_clone_cpu_entry_smoke_only",
        "formal_benchmark_admission": False,
        "source": {"revision": revision, "worktree_dirty": False},
        "clone": {
            "strategy": "git_clone_no_local_no_hardlinks",
            "revision": clone_revision,
            "worktree_clean": True,
            "temporary_clone_retained": False,
        },
        "cpu_smoke": {
            "status": "passed",
            "schema": smoke_report.get("schema"),
            "report_sha256": smoke_digest,
            "fixture_frame_count": smoke_report.get("fixture", {}).get("frame_count"),
            "fixture_agent_count": smoke_report.get("fixture", {}).get("agent_count"),
            "episode_manifest_sha256": smoke_report.get("fixture", {}).get("episode_manifest_sha256"),
        },
        "checks": {
            "source_clean": True,
            "clone_revision": clone_revision == revision,
            "clone_worktree_clean": True,
            "cpu_researcher_smoke": True,
            "no_private_truth": True,
            "no_absolute_paths": True,
            "formal_benchmark_admission": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "wall_time_ms": round(elapsed_ms, 3),
            "child_returncode": child_returncode,
            "temporary_clone_retained": False,
        },
    }
    if not _SHA256.fullmatch(smoke_digest):
        raise CleanRoomSmokeError("internal smoke report digest failure")
    _write_json(root / "clean_room_report.json", report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="new or empty directory for the redacted report")
    parser.add_argument("--source-root", type=Path, default=None, help="clean Git repository to clone")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_clean_room_smoke(args.output_root, source_root=args.source_root, timeout_s=args.timeout_s)
    except (OSError, CleanRoomSmokeError, ValueError) as exc:
        print(json.dumps({"schema": CLEAN_ROOM_SMOKE_SCHEMA, "status": "failed", "error": type(exc).__name__}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
