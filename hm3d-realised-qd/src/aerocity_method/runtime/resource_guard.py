"""Fail-closed resource admission for long-running local processes."""

from __future__ import annotations

import csv
import ctypes
import errno
import io
import os
import subprocess
from dataclasses import dataclass
from typing import Any

RESOURCE_FAILURE_CLASS = "RESOURCE_ADMISSION_FAILURE"
WINDOWS_RESOURCE_ERROR_CODES = frozenset({8, 14, 1450, 1344})


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    available_memory_bytes: int
    process_count: int
    active_heavy_jobs: int = 0

    def __post_init__(self) -> None:
        for name in ("available_memory_bytes", "process_count", "active_heavy_jobs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    memory_bytes: int
    process_slots: int = 1
    heavy_jobs: int = 0

    def __post_init__(self) -> None:
        if self.memory_bytes < 0 or self.process_slots < 1 or self.heavy_jobs < 0:
            raise ValueError("invalid resource request")


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    memory_reserve_bytes: int = 2 * 1024**3
    maximum_processes: int = 2048
    maximum_heavy_jobs: int = 1

    def __post_init__(self) -> None:
        if self.memory_reserve_bytes < 0:
            raise ValueError("memory reserve must be non-negative")
        if self.maximum_processes < 1 or self.maximum_heavy_jobs < 1:
            raise ValueError("resource maxima must be positive")


@dataclass(frozen=True, slots=True)
class ResourceAdmissionDecision:
    admitted: bool
    reasons: tuple[str, ...]
    snapshot: ResourceSnapshot
    request: ResourceRequest

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reasons": list(self.reasons),
            "snapshot": {
                "available_memory_bytes": self.snapshot.available_memory_bytes,
                "process_count": self.snapshot.process_count,
                "active_heavy_jobs": self.snapshot.active_heavy_jobs,
            },
            "request": {
                "memory_bytes": self.request.memory_bytes,
                "process_slots": self.request.process_slots,
                "heavy_jobs": self.request.heavy_jobs,
            },
        }


def evaluate_resource_admission(
    snapshot: ResourceSnapshot,
    request: ResourceRequest,
    limits: ResourceLimits | None = None,
) -> ResourceAdmissionDecision:
    limits = limits or ResourceLimits()
    reasons: list[str] = []
    if snapshot.available_memory_bytes - request.memory_bytes < limits.memory_reserve_bytes:
        reasons.append("MEMORY_RESERVE_WOULD_BE_VIOLATED")
    if snapshot.process_count + request.process_slots > limits.maximum_processes:
        reasons.append("PROCESS_LIMIT_WOULD_BE_VIOLATED")
    if snapshot.active_heavy_jobs + request.heavy_jobs > limits.maximum_heavy_jobs:
        reasons.append("HEAVY_JOB_LIMIT_WOULD_BE_VIOLATED")
    return ResourceAdmissionDecision(not reasons, tuple(reasons), snapshot, request)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _windows_available_memory() -> int:
    status = _MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.available_physical)


def _linux_available_memory() -> int:
    with open("/proc/meminfo", encoding="ascii") as handle:
        values = {
            fields[0].rstrip(":"): int(fields[1]) * 1024
            for line in handle
            if (fields := line.split()) and len(fields) >= 2
        }
    if "MemAvailable" not in values:
        raise OSError("MemAvailable is absent from /proc/meminfo")
    return values["MemAvailable"]


def _process_count() -> int:
    if os.name != "nt":
        return sum(name.isdigit() for name in os.listdir("/proc"))
    completed = subprocess.run(
        ("tasklist.exe", "/FO", "CSV", "/NH"),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    return sum(1 for row in csv.reader(io.StringIO(completed.stdout)) if row)


def sample_local_resources(*, active_heavy_jobs: int = 0) -> ResourceSnapshot:
    available = _windows_available_memory() if os.name == "nt" else _linux_available_memory()
    return ResourceSnapshot(available, _process_count(), active_heavy_jobs)


def classify_process_start_failure(error: BaseException) -> dict[str, Any]:
    winerror = getattr(error, "winerror", None)
    error_number = getattr(error, "errno", None)
    resource_failure = winerror in WINDOWS_RESOURCE_ERROR_CODES or error_number in {
        errno.ENOMEM,
        errno.EAGAIN,
    }
    return {
        "failure_class": RESOURCE_FAILURE_CLASS if resource_failure else "PROCESS_START_FAILURE",
        "error_type": type(error).__name__,
        "winerror": winerror,
        "errno": error_number,
        "message": str(error),
        "retry_allowed": False,
    }
