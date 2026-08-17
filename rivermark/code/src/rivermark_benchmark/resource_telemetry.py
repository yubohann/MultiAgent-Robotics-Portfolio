"""Low-frequency in-process resource telemetry for bounded Isaac runs."""

from __future__ import annotations

import ctypes
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


RESOURCE_TELEMETRY_SCHEMA = "org.rivermark.resource-telemetry.v1"
FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA = "org.rivermark.foreign-native-process-census.v1"

# These values are the shared Windows resource policy for native geometry
# scans, target-free smoke tests, and protocol-bound captures.  They leave
# headroom above the measured 69.29% successful-capture peak while still
# allowing a clean host to start at its normal idle commit level.
DEFAULT_PREFLIGHT_COMMIT_PERCENT = 65.0
DEFAULT_ABORT_COMMIT_PERCENT = 82.0

# A generic browser or editor can legitimately retain a large address space.
# These executable names identify processes that can independently create an
# Isaac/Kit runtime or a CUDA-heavy Python workload.  The census deliberately
# publishes no process identifiers, image paths, or command lines.
_NATIVE_RUNTIME_EXECUTABLE_NAMES = frozenset(
    ("python.exe", "pythonw.exe", "kit.exe", "isaac-sim.exe", "isaacsim.exe")
)


def _process_memory_snapshot() -> dict[str, int] | None:
    if os.name == "nt":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                # PROCESS_MEMORY_COUNTERS_EX extends the base structure with
                # PrivateUsage.  This is the process private-commit figure;
                # PagefileUsage is retained separately because it is broader.
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_uint32]
        api.restype = ctypes.c_int
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.GetCurrentProcess()
        if not api(handle, ctypes.byref(counters), counters.cb):
            return None
        return {
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "pagefile_usage_bytes": int(counters.PagefileUsage),
            "peak_pagefile_usage_bytes": int(counters.PeakPagefileUsage),
            "private_commit_bytes": int(counters.PrivateUsage),
        }
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss)
        if platform.system() != "Darwin":
            rss *= 1024
        return {
            "working_set_bytes": rss,
            "peak_working_set_bytes": rss,
            "private_commit_bytes": rss,
        }
    except (ImportError, OSError, AttributeError):
        return None


def _system_commit_snapshot() -> dict[str, int | float] | None:
    if os.name != "nt":
        return None
    class _PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", ctypes.c_uint32),
            ("ProcessCount", ctypes.c_uint32),
            ("ThreadCount", ctypes.c_uint32),
        ]

    info = _PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    try:
        api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_uint32]
        api.restype = ctypes.c_int
        if not api(ctypes.byref(info), info.cb) or info.CommitLimit <= 0:
            return None
    except (AttributeError, OSError):
        return None
    total = int(info.CommitTotal * info.PageSize)
    limit = int(info.CommitLimit * info.PageSize)
    peak = int(info.CommitPeak * info.PageSize)
    return {
        "commit_total_bytes": total,
        "commit_limit_bytes": limit,
        "commit_peak_bytes": peak,
        "commit_percent": 100.0 * total / limit,
        # These are host-wide counters.  They make a reset-time commit spike
        # diagnosable without recording unrelated process names, command lines,
        # or private data in a capture receipt.
        "physical_total_bytes": int(info.PhysicalTotal * info.PageSize),
        "physical_available_bytes": int(info.PhysicalAvailable * info.PageSize),
        "system_cache_bytes": int(info.SystemCache * info.PageSize),
        "kernel_total_bytes": int(info.KernelTotal * info.PageSize),
        "kernel_paged_bytes": int(info.KernelPaged * info.PageSize),
        "kernel_nonpaged_bytes": int(info.KernelNonpaged * info.PageSize),
        "process_count": int(info.ProcessCount),
        "thread_count": int(info.ThreadCount),
    }


def _windows_native_process_rows() -> tuple[dict[str, int | str], ...] | None:
    """Return private-commit rows for other possible native-runtime owners.

    The function is Windows-only and best effort.  Access-denied and
    short-lived processes are skipped rather than treated as a reason to
    prevent all collection.  Callers receive only an aggregated census.
    """

    if os.name != "nt":
        return None
    try:
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        process_query = 0x0400 | 0x1000 | 0x0010
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        enum_processes = psapi.EnumProcesses
        enum_processes.argtypes = [
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        enum_processes.restype = ctypes.c_int
        open_process = kernel.OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        close_handle = kernel.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        query_image = kernel.QueryFullProcessImageNameW
        query_image.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        query_image.restype = ctypes.c_int
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_uint32]
        get_memory.restype = ctypes.c_int

        capacity = 512
        while True:
            identifiers = (ctypes.c_ulong * capacity)()
            byte_count = ctypes.c_ulong()
            if not enum_processes(
                identifiers, ctypes.sizeof(identifiers), ctypes.byref(byte_count)
            ):
                return None
            count = int(byte_count.value // ctypes.sizeof(ctypes.c_ulong))
            if count < capacity:
                break
            capacity *= 2
            if capacity > 65536:
                return None

        rows: list[dict[str, int | str]] = []
        for raw_pid in identifiers[:count]:
            pid = int(raw_pid)
            if pid <= 0:
                continue
            handle = open_process(process_query, False, pid)
            if not handle:
                continue
            try:
                image = ctypes.create_unicode_buffer(32768)
                image_length = ctypes.c_uint32(len(image))
                if not query_image(handle, 0, image, ctypes.byref(image_length)):
                    continue
                executable = os.path.basename(image.value).casefold()
                if executable not in _NATIVE_RUNTIME_EXECUTABLE_NAMES:
                    continue
                counters = _Counters()
                counters.cb = ctypes.sizeof(counters)
                if not get_memory(handle, ctypes.byref(counters), counters.cb):
                    continue
                rows.append(
                    {
                        "pid": pid,
                        "executable": executable,
                        "private_commit_bytes": int(counters.PrivateUsage),
                    }
                )
            finally:
                close_handle(handle)
        return tuple(rows)
    except (AttributeError, OSError):
        return None


def _owned_process_ids_from_parent_rows(
    parent_rows: Iterable[tuple[int, int]], *, root_pid: int
) -> frozenset[int]:
    """Return ``root_pid`` and every descendant in a process-parent snapshot."""

    if root_pid <= 0:
        raise ValueError("root_pid must be positive")
    parents = {
        int(pid): int(parent_pid)
        for pid, parent_pid in parent_rows
        if isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and isinstance(parent_pid, int)
        and not isinstance(parent_pid, bool)
        and parent_pid >= 0
    }
    owned = {root_pid}
    while True:
        descendants = {
            pid for pid, parent_pid in parents.items() if parent_pid in owned and pid not in owned
        }
        if not descendants:
            return frozenset(owned)
        owned.update(descendants)


def _windows_owned_process_ids(root_pid: int) -> frozenset[int] | None:
    """Best-effort owner process tree for an Isaac AppLauncher invocation.

    Isaac Sim can retain the renderer in a Kit child process.  Counting that
    child as foreign makes a full-sensor smoke reject itself after launch.  A
    failed process-tree query intentionally falls back to the root PID only in
    the caller, preserving the conservative foreign-process gate.
    """

    if os.name != "nt":
        return frozenset({root_pid})
    try:
        class _ProcessEntry32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_uint32),
                ("cntUsage", ctypes.c_uint32),
                ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32),
                ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_uint32),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create_snapshot = kernel.CreateToolhelp32Snapshot
        create_snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        create_snapshot.restype = ctypes.c_void_p
        process_first = kernel.Process32FirstW
        process_first.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry32W)]
        process_first.restype = ctypes.c_int
        process_next = kernel.Process32NextW
        process_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry32W)]
        process_next.restype = ctypes.c_int
        close_handle = kernel.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int

        snapshot = create_snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
        invalid_handle = ctypes.c_void_p(-1).value
        if not snapshot or snapshot == invalid_handle:
            return None
        try:
            entry = _ProcessEntry32W()
            entry.dwSize = ctypes.sizeof(entry)
            if not process_first(snapshot, ctypes.byref(entry)):
                return None
            parent_rows: list[tuple[int, int]] = []
            while True:
                parent_rows.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
                entry.dwSize = ctypes.sizeof(entry)
                if not process_next(snapshot, ctypes.byref(entry)):
                    break
            return _owned_process_ids_from_parent_rows(parent_rows, root_pid=root_pid)
        finally:
            close_handle(snapshot)
    except (AttributeError, OSError):
        return None


def _summarize_foreign_native_process_rows(
    rows: tuple[dict[str, int | str], ...],
    *,
    current_pid: int,
    minimum_private_commit_bytes: int,
    excluded_pids: Iterable[int] = (),
) -> dict[str, int | str]:
    """Project local process rows to a privacy-preserving admission census."""

    if minimum_private_commit_bytes < 0:
        raise ValueError("minimum_private_commit_bytes must be non-negative")
    excluded = {current_pid}
    excluded.update(
        pid
        for pid in excluded_pids
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
    )
    candidates: list[int] = []
    for row in rows:
        pid = row.get("pid")
        executable = row.get("executable")
        private_commit = row.get("private_commit_bytes")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid in excluded
            or not isinstance(executable, str)
            or executable.casefold() not in _NATIVE_RUNTIME_EXECUTABLE_NAMES
            or not isinstance(private_commit, int)
            or isinstance(private_commit, bool)
            or private_commit < minimum_private_commit_bytes
        ):
            continue
        candidates.append(private_commit)
    return {
        "schema": FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA,
        "enumerated_native_process_count": len(rows),
        "minimum_private_commit_bytes": minimum_private_commit_bytes,
        "candidate_count": len(candidates),
        "candidate_private_commit_bytes": sum(candidates),
        "maximum_candidate_private_commit_bytes": max(candidates, default=0),
    }


def foreign_native_process_census(
    *, minimum_private_commit_bytes: int, current_pid: int | None = None
) -> dict[str, int | str] | None:
    """Return an anonymous census of conflicting native-runtime processes.

    A collection owner is excluded by PID.  ``None`` means the host could not
    provide a reliable Windows process census and must be recorded as such by
    the caller rather than fabricated as an empty result.
    """

    rows = _windows_native_process_rows()
    if rows is None:
        return None
    owner_pid = os.getpid() if current_pid is None else current_pid
    owned_pids = _windows_owned_process_ids(owner_pid)
    return _summarize_foreign_native_process_rows(
        rows,
        current_pid=owner_pid,
        minimum_private_commit_bytes=minimum_private_commit_bytes,
        excluded_pids=owned_pids if owned_pids is not None else (),
    )


def _gpu_memory_snapshot(torch_module: Any | None) -> dict[str, int | float] | None:
    if torch_module is None:
        return None
    try:
        cuda = torch_module.cuda
        if not cuda.is_available():
            return None
        return {
            "device_count": int(cuda.device_count()),
            "allocated_bytes": int(cuda.memory_allocated()),
            "reserved_bytes": int(cuda.memory_reserved()),
            "peak_allocated_bytes": int(cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(cuda.max_memory_reserved()),
        }
    except (AttributeError, RuntimeError, TypeError):
        return None


@dataclass
class ResourceTelemetry:
    """Explicitly sampled resource observations; no worker thread is created."""

    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, phase: str, *, torch_module: Any | None = None) -> dict[str, Any]:
        if not isinstance(phase, str) or not phase:
            raise ValueError("telemetry phase must be a non-empty string")
        process = _process_memory_snapshot()
        system_commit = _system_commit_snapshot()
        if isinstance(process, Mapping) and isinstance(system_commit, dict):
            total = system_commit.get("commit_total_bytes")
            private_commit = process.get("private_commit_bytes")
            if (
                isinstance(total, int)
                and not isinstance(total, bool)
                and isinstance(private_commit, int)
                and not isinstance(private_commit, bool)
                and 0 <= private_commit <= total
            ):
                # This is deliberately an aggregate.  It includes all other
                # process, kernel, and driver commit without exposing their
                # identities in a development or public receipt.
                system_commit["commit_outside_current_process_bytes"] = total - private_commit
        row: dict[str, Any] = {
            "wall_time_ns": time.time_ns(),
            "phase": phase,
            "process": process,
            "system_commit": system_commit,
            "gpu": _gpu_memory_snapshot(torch_module),
        }
        self.samples.append(row)
        return row

    def as_dict(self) -> dict[str, Any]:
        maxima: dict[str, int | float] = {}
        for row in self.samples:
            for section in ("process", "system_commit", "gpu"):
                values = row.get(section)
                if not isinstance(values, Mapping):
                    continue
                for key, value in values.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        maxima[key] = max(maxima.get(key, value), value)
        return {
            "schema": RESOURCE_TELEMETRY_SCHEMA,
            "sample_count": len(self.samples),
            "samples": list(self.samples),
            "maxima": maxima,
            "sampling": "explicit_in_process_phase_boundaries",
        }
