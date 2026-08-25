"""Fail-closed host guards for long-running Isaac subprocess batches."""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import HostGuardError

WINDOWS_START_COMMIT_LIMIT = 0.65
WINDOWS_RUNTIME_COMMIT_LIMIT = 0.82
HOST_1344_SIGNATURE = "settokeninformation(tokendefaultdacl): 1344"
HOST_GUARD_SCHEMA = "org.aerocity.bench.isaac-host-guard.v3"


@dataclass(frozen=True)
class HostSnapshot:
    captured_at: str
    platform: str
    commit_limit_bytes: int | None
    commit_available_bytes: int | None
    commit_used_bytes: int | None
    commit_fraction: float | None


@dataclass(frozen=True)
class GuardedProcessResult:
    returncode: int
    elapsed_s: float
    maximum_commit_fraction: float | None
    fatal_1344: bool
    snapshot_before: HostSnapshot
    snapshot_after: HostSnapshot


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def host_snapshot() -> HostSnapshot:
    captured_at = dt.datetime.now(dt.UTC).isoformat()
    if os.name != "nt":
        return HostSnapshot(captured_at, platform.platform(), None, None, None, None)
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise HostGuardError("GlobalMemoryStatusEx failed")
    limit = int(status.ullTotalPageFile)
    available = int(status.ullAvailPageFile)
    used = max(0, limit - available)
    fraction = used / limit if limit else None
    return HostSnapshot(captured_at, platform.platform(), limit, available, used, fraction)


def validate_start_snapshot(
    snapshot: HostSnapshot,
    *,
    maximum_commit_fraction: float = WINDOWS_START_COMMIT_LIMIT,
) -> None:
    if not 0.0 < maximum_commit_fraction < 1.0:
        raise ValueError("maximum commit fraction must be between zero and one")
    if os.name == "nt" and commit_limit_exceeded(snapshot, maximum_commit_fraction):
        raise HostGuardError(
            "Windows commit is too high for a new Isaac process: "
            f"{snapshot.commit_fraction:.3f} >= {maximum_commit_fraction:.3f}"
        )


def commit_limit_exceeded(snapshot: HostSnapshot, maximum_commit_fraction: float) -> bool:
    if not 0.0 < maximum_commit_fraction < 1.0:
        raise ValueError("maximum commit fraction must be between zero and one")
    return (
        snapshot.commit_fraction is not None and snapshot.commit_fraction >= maximum_commit_fraction
    )


def is_host_1344(returncode: int, log_text: str) -> bool:
    return returncode == 1344 or HOST_1344_SIGNATURE in log_text.casefold()


def _is_isaac_process_record(name: str, command_line: str) -> bool:
    normalized_name = name.casefold()
    normalized_command = command_line.casefold()
    if normalized_name in {"kit.exe", "isaac-sim.exe"}:
        return True
    if normalized_name not in {"python.exe", "pythonw.exe"}:
        return False
    # IsaacLab launchers need not share a fixed script basename.  Its Python
    # environment together with headless execution is a narrower and more
    # durable signature than maintaining an incomplete script-name allowlist.
    if "isaaclab" in normalized_command and "--headless" in normalized_command:
        return True
    return any(
        marker in normalized_command
        for marker in (
            "cf2x_l1_fleet_preflight.py",
            "isaac_capture.py",
            "isaac_native_gate.py",
            "quadrotor_l1_vertical_slice.py",
            "quadrotor_physics_preflight.py",
            "run_hm3d_p07_execution_smoke.py",
            "replay_hm3d_cf2x_collision.py",
            "run_hm3d_",
            "omni.kit.app",
        )
    )


def _is_process_tree_member(
    pid: int,
    *,
    root_pid: int,
    parent_by_pid: dict[int, int],
) -> bool:
    """Return whether *pid* belongs to the process tree rooted at *root_pid*."""

    current = pid
    visited: set[int] = set()
    while current > 0 and current not in visited:
        if current == root_pid:
            return True
        visited.add(current)
        current = parent_by_pid.get(current, -1)
    return False


def foreign_isaac_processes(*, owned_root_pid: int | None = None) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = (
        "$OutputEncoding = [Console]::OutputEncoding = "
        "[System.Text.UTF8Encoding]::new($false); "
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20.0,
        )
        payload = json.loads(result.stdout or "[]")
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError) as exc:
        raise HostGuardError("cannot complete the Windows Isaac process census") from exc
    records = payload if isinstance(payload, list) else [payload]
    parent_by_pid = {
        int(record.get("ProcessId") or -1): int(record.get("ParentProcessId") or -1)
        for record in records
        if isinstance(record, dict)
    }
    matches = []
    for record in records:
        pid = int(record.get("ProcessId") or -1)
        if pid == os.getpid():
            continue
        if owned_root_pid is not None and _is_process_tree_member(
            pid,
            root_pid=owned_root_pid,
            parent_by_pid=parent_by_pid,
        ):
            continue
        name = str(record.get("Name") or "").casefold()
        command_line = str(record.get("CommandLine") or "").casefold()
        if _is_isaac_process_record(name, command_line):
            matches.append(
                {
                    "pid": pid,
                    "parent_pid": parent_by_pid.get(pid, -1),
                    "name": name,
                    "command_line": command_line,
                }
            )
    return matches


@contextlib.contextmanager
def isaac_host_lock() -> Any:
    if os.name != "nt":
        yield
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel32.ReleaseMutex.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    handle = kernel32.CreateMutexW(None, True, "Local\\AeroCityBenchIsaacRuntime")
    if not handle:
        raise HostGuardError("cannot create the AeroCityBench Isaac host mutex")
    already_exists = ctypes.get_last_error() == 183
    if already_exists:
        kernel32.CloseHandle(handle)
        raise HostGuardError("another AeroCityBench Isaac owner already holds the host mutex")
    try:
        yield
    finally:
        kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Terminate only the process tree created for this attempt.  Isaac may
        # retain Kit/render children, so killing only the Python parent can
        # poison every later batch job on the same host.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
        except (OSError, subprocess.SubprocessError):
            # The monitor must still regain control if taskkill itself cannot
            # start or times out.  Killing the parent is weaker than a tree
            # kill, but it prevents the guard from losing its failure receipt.
            process.kill()
    else:
        process.terminate()
    try:
        process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20.0)


def _write_guard_report(
    report_path: Path,
    *,
    status: str,
    returncode: int | None,
    elapsed_s: float,
    maximum_commit_fraction: float | None,
    trigger: str | None,
    foreign_runtime_count_before: int,
    snapshot_before: HostSnapshot,
    snapshot_after: HostSnapshot,
    child_pid: int | None,
    evidence_binding: str | None,
    foreign_runtime_count_after: int | None = None,
) -> None:
    report = {
        "schema": HOST_GUARD_SCHEMA,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 6),
        "maximum_commit_fraction": maximum_commit_fraction,
        "start_commit_limit": WINDOWS_START_COMMIT_LIMIT,
        "runtime_commit_limit": WINDOWS_RUNTIME_COMMIT_LIMIT,
        "trigger": trigger,
        "foreign_runtime_count_before": foreign_runtime_count_before,
        "foreign_runtime_count_after": foreign_runtime_count_after,
        "child_pid": child_pid,
        "evidence_binding": evidence_binding,
        "process_tree_policy": "terminate_owned_attempt_tree_only",
        "snapshot_before": asdict(snapshot_before),
        "snapshot_after": asdict(snapshot_after),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, report_path)


def validate_host_guard_pass_receipt(
    path: Path,
    *,
    expected_evidence_binding: str | None = None,
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"completed replay host guard receipt is invalid: {path}") from exc
    required = {
        "schema": HOST_GUARD_SCHEMA,
        "status": "PASS",
        "returncode": 0,
        "trigger": None,
        "foreign_runtime_count_before": 0,
        "foreign_runtime_count_after": 0,
    }
    if not isinstance(receipt, dict):
        raise ValueError(f"completed replay host guard receipt is invalid: {path}")
    for field, expected in required.items():
        actual = receipt.get(field)
        if field == "returncode" and (isinstance(actual, bool) or not isinstance(actual, int)):
            raise ValueError(f"completed replay host guard {field} is invalid: {path}")
        if actual != expected:
            raise ValueError(f"completed replay host guard {field} is invalid: {path}")
    if (
        expected_evidence_binding is not None
        and receipt.get("evidence_binding") != expected_evidence_binding
    ):
        raise ValueError(f"completed replay host guard evidence_binding is invalid: {path}")
    return receipt


def run_guarded_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    report_path: Path,
    timeout_s: float,
    poll_interval_s: float = 1.0,
    evidence_binding: str | None = None,
) -> GuardedProcessResult:
    if timeout_s <= 0.0:
        raise ValueError("guarded process timeout must be positive")
    if poll_interval_s <= 0.0:
        raise ValueError("guarded process poll interval must be positive")
    before = host_snapshot()
    try:
        validate_start_snapshot(before)
    except HostGuardError:
        _write_guard_report(
            report_path,
            status="FAIL",
            returncode=None,
            elapsed_s=0.0,
            maximum_commit_fraction=before.commit_fraction,
            trigger="start_commit_limit",
            foreign_runtime_count_before=0,
            snapshot_before=before,
            snapshot_after=before,
            child_pid=None,
            evidence_binding=evidence_binding,
        )
        raise
    try:
        foreign = foreign_isaac_processes()
    except HostGuardError:
        _write_guard_report(
            report_path,
            status="FAIL",
            returncode=None,
            elapsed_s=0.0,
            maximum_commit_fraction=before.commit_fraction,
            trigger="process_census_failure",
            foreign_runtime_count_before=-1,
            snapshot_before=before,
            snapshot_after=before,
            child_pid=None,
            evidence_binding=evidence_binding,
        )
        raise
    if foreign:
        _write_guard_report(
            report_path,
            status="FAIL",
            returncode=None,
            elapsed_s=0.0,
            maximum_commit_fraction=before.commit_fraction,
            trigger="foreign_runtime",
            foreign_runtime_count_before=len(foreign),
            snapshot_before=before,
            snapshot_after=before,
            child_pid=None,
            evidence_binding=evidence_binding,
        )
        raise HostGuardError(f"foreign Isaac runtime already active: {foreign}")
    started = time.monotonic()
    maximum_commit = before.commit_fraction
    trigger: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[Any] | None = None
    monitor_error: BaseException | None = None
    foreign_during: list[dict[str, Any]] = []
    returncode: int | None = None
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > timeout_s:
                        trigger = "timeout"
                        _stop_process(process)
                        break
                    snapshot = host_snapshot()
                    if snapshot.commit_fraction is not None:
                        maximum_commit = max(maximum_commit or 0.0, snapshot.commit_fraction)
                        if (
                            os.name == "nt"
                            and snapshot.commit_fraction >= WINDOWS_RUNTIME_COMMIT_LIMIT
                        ):
                            trigger = "runtime_commit_limit"
                            _stop_process(process)
                            break
                    foreign_during = foreign_isaac_processes(owned_root_pid=process.pid)
                    if foreign_during:
                        trigger = "foreign_runtime_during_attempt"
                        _stop_process(process)
                        break
                    time.sleep(poll_interval_s)
            except BaseException as exc:
                trigger = "monitor_failure"
                monitor_error = exc
                if process.poll() is None:
                    _stop_process(process)
            returncode = int(process.wait())
    except OSError:
        trigger = "launch_error"
        elapsed = time.monotonic() - started
        _write_guard_report(
            report_path,
            status="FAIL",
            returncode=None,
            elapsed_s=elapsed,
            maximum_commit_fraction=maximum_commit,
            trigger=trigger,
            foreign_runtime_count_before=len(foreign),
            snapshot_before=before,
            snapshot_after=before,
            child_pid=process.pid if process is not None else None,
            evidence_binding=evidence_binding,
        )
        raise
    finally:
        try:
            after = host_snapshot()
        except HostGuardError:
            after = before
    elapsed = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assert returncode is not None
    fatal_1344 = is_host_1344(returncode, log_text)
    foreign_after: list[dict[str, Any]] | None
    post_census_trigger: str | None = None
    try:
        foreign_after = foreign_isaac_processes()
    except HostGuardError:
        foreign_after = None
        post_census_trigger = "post_process_census_failure"
    if foreign_after:
        post_census_trigger = "residual_runtime"
    final_trigger = "windows_1344" if fatal_1344 else (trigger or post_census_trigger)
    _write_guard_report(
        report_path,
        status="PASS" if returncode == 0 and final_trigger is None else "FAIL",
        returncode=returncode,
        elapsed_s=elapsed,
        maximum_commit_fraction=maximum_commit,
        trigger=final_trigger,
        foreign_runtime_count_before=len(foreign),
        foreign_runtime_count_after=(None if foreign_after is None else len(foreign_after)),
        snapshot_before=before,
        snapshot_after=after,
        child_pid=process.pid if process is not None else None,
        evidence_binding=evidence_binding,
    )
    result = GuardedProcessResult(
        returncode,
        elapsed,
        maximum_commit,
        fatal_1344,
        before,
        after,
    )
    if fatal_1344:
        raise HostGuardError("Windows execution host failed with TokenDefaultDacl error 1344")
    if trigger == "runtime_commit_limit":
        raise HostGuardError("Windows commit crossed the 82% Isaac runtime limit")
    if trigger == "timeout":
        raise TimeoutError(f"Isaac process exceeded {timeout_s} seconds")
    if trigger == "foreign_runtime_during_attempt":
        raise HostGuardError(
            "foreign Isaac runtime started during the owned attempt; the owned process "
            f"tree was stopped and the foreign runtime was left untouched: {foreign_during}"
        )
    if monitor_error is not None:
        if isinstance(monitor_error, (KeyboardInterrupt, SystemExit)):
            raise monitor_error
        raise HostGuardError(
            "Isaac host monitoring failed; the owned process tree was stopped"
        ) from monitor_error
    if post_census_trigger == "post_process_census_failure":
        raise HostGuardError(
            "Isaac post-process census failed; refusing to start another batch attempt"
        )
    if post_census_trigger == "residual_runtime":
        raise HostGuardError(
            "Isaac runtime remained after the owned child exited; refusing to start another "
            f"batch attempt: {foreign_after}"
        )
    return result
