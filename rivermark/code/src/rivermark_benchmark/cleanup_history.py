"""Conservative cleanup of completed Isaac run directories.

Large capture artifacts are moved to the Windows Recycle Bin instead of being
unlinked. This is reversible archival, not guaranteed capacity reclamation:
the Recycle Bin may continue to consume space on the same volume until an
operator explicitly empties it. Automatic cleanup considers only orphaned,
crash-left runs with no terminal receipt; terminal evidence requires explicit
operator opt-in. Eligible directories must exceed the retention age and size
thresholds. The active output directory is always protected, and every
attempted move is recorded in a small JSONL ledger in the run root.
"""

from __future__ import annotations

import ctypes
import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


TERMINAL_STATUSES = frozenset({"captured", "failed", "aborted", "orphaned"})
# Capture receipts, including failures, are scientific evidence.  Automatic
# housekeeping may only recycle an interrupted run that never obtained a final
# receipt.  An operator must explicitly opt in before moving terminal evidence.
AUTOMATIC_CLEANUP_STATUSES = frozenset({"orphaned"})
DEFAULT_MIN_SIZE_BYTES = 1 * 1024**3
DEFAULT_MIN_AGE_HOURS = 24.0


@dataclass(frozen=True)
class CleanupRecord:
    path: str
    status: str
    size_bytes: int
    action: str
    reason: str
    observed_wall_time_ns: int


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _receipt_status(path: Path) -> str | None:
    receipt = path / "capture_receipt.json"
    try:
        value = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = value.get("status")
    return status if isinstance(status, str) else None


def _orphan_progress_status(path: Path) -> str | None:
    """Classify a crash-left run that never reached receipt finalization.

    Isaac writes ``capture_progress.json`` at every meaningful stage.  A
    schema-valid progress file without a receipt is therefore evidence of a
    previously-started run, while a directory with neither file remains an
    unknown directory and is deliberately protected.  Age is checked by the
    caller using every file's newest mtime, so a live run cannot be selected
    merely because it has not written a receipt yet.
    """

    progress = path / "capture_progress.json"
    try:
        value = json.loads(progress.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema") != "org.rivermark.isaac-capture-progress.v1":
        return None
    if not isinstance(value.get("stage"), str) or not value["stage"]:
        return None
    return "orphaned"


def _orphan_start_status(path: Path) -> str | None:
    """Classify a start marker left before preflight could write progress."""

    marker = path / "capture_start.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema") != "org.rivermark.isaac-capture-start.v1":
        return None
    if not isinstance(value.get("attempt_id"), str) or not re.fullmatch(
        r"attempt-[a-f0-9]{32}", value["attempt_id"]
    ):
        return None
    if not isinstance(value.get("started_wall_time_ns"), int) or value["started_wall_time_ns"] <= 0:
        return None
    if not isinstance(value.get("source_revision"), str) or not re.fullmatch(
        r"[0-9a-f]{7,64}", value["source_revision"]
    ):
        return None
    if not isinstance(value.get("source_tree_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["source_tree_sha256"]
    ):
        return None
    if not isinstance(value.get("source_worktree_dirty"), bool):
        return None
    if not isinstance(value.get("task_kind"), str) or not value["task_kind"]:
        return None
    if not isinstance(value.get("control_mode"), str) or not value["control_mode"]:
        return None
    if (
        isinstance(value.get("agent_count_requested"), bool)
        or not isinstance(value.get("agent_count_requested"), int)
        or value["agent_count_requested"] < 1
    ):
        return None
    return "orphaned"


def _move_to_recycle_bin(path: Path) -> None:
    if os.name != "nt":
        raise OSError("Recycle Bin cleanup is only implemented on Windows")
    # SHFileOperation accepts a double-NUL-terminated UTF-16 path list and
    # performs a reversible FO_DELETE with the FOF_ALLOWUNDO flag.
    class _SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", ctypes.c_int),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    source = str(path.resolve()) + "\0\0"
    operation = _SHFILEOPSTRUCTW(
        None,
        3,  # FO_DELETE
        source,
        None,
        0x0040 | 0x0010 | 0x0004 | 0x0002,  # ALLOWUNDO|NOCONFIRMATION|SILENT|NOERRORUI
        0,
        None,
        None,
    )
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise OSError(f"SHFileOperationW failed for {path} with code {result}")


def cleanup_completed_runs(
    root: Path,
    *,
    keep_paths: Iterable[Path] = (),
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    min_size_bytes: int = DEFAULT_MIN_SIZE_BYTES,
    dry_run: bool = False,
    include_terminal_receipts: bool = False,
    now_ns: int | None = None,
) -> tuple[CleanupRecord, ...]:
    """Move eligible sibling Isaac runs to the Recycle Bin.

    Automatic cleanup considers only a schema-valid progress/start marker left
    by a crashed/interrupted run. A completed, failed, or aborted capture
    receipt is retained as evidence unless an operator explicitly sets
    ``include_terminal_receipts``. Eligible directories must also exceed the
    age and size thresholds. Unknown and active/running directories are always
    retained. The function never deletes files permanently and must not be
    treated as proof that volume free space increased.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return ()
    if min_age_hours < 0 or min_size_bytes < 0:
        raise ValueError("cleanup thresholds must be non-negative")
    keep = {Path(path).expanduser().resolve() for path in keep_paths}
    now = time.time_ns() if now_ns is None else int(now_ns)
    age_ns = int(min_age_hours * 3600.0 * 1_000_000_000)
    records: list[CleanupRecord] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.is_symlink() or candidate.resolve() in keep:
            continue
        status = _receipt_status(candidate)
        if status is None:
            status = _orphan_progress_status(candidate)
        if status is None:
            status = _orphan_start_status(candidate)
        if status not in TERMINAL_STATUSES:
            continue
        if not include_terminal_receipts and status not in AUTOMATIC_CLEANUP_STATUSES:
            continue
        try:
            files = [item for item in candidate.rglob("*") if item.is_file()]
            newest_ns = max((item.stat().st_mtime_ns for item in files), default=0)
            size_bytes = _directory_size(candidate)
        except OSError as error:
            records.append(
                CleanupRecord(str(candidate), status, 0, "skipped", f"stat_failed:{error}", now)
            )
            continue
        if size_bytes < min_size_bytes:
            continue
        if now - newest_ns < age_ns:
            continue
        action = "dry_run" if dry_run else "recycle_bin"
        reason = f"terminal_receipt_older_than_{min_age_hours:g}h"
        try:
            if not dry_run:
                _move_to_recycle_bin(candidate)
        except OSError as error:
            action = "skipped"
            reason = f"recycle_failed:{error}"
        records.append(CleanupRecord(str(candidate), status, size_bytes, action, reason, now))
    ledger = root / "cleanup_history.jsonl"
    with ledger.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return tuple(records)


__all__ = ["CleanupRecord", "cleanup_completed_runs"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="parent directory containing Isaac run folders")
    parser.add_argument("--keep", action="append", type=Path, default=[])
    parser.add_argument("--min-age-hours", type=float, default=DEFAULT_MIN_AGE_HOURS)
    parser.add_argument("--min-size-gib", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-terminal-receipts",
        action="store_true",
        help="Allow explicit archival of captured/failed/aborted evidence after dry-run review.",
    )
    args = parser.parse_args(argv)
    records = cleanup_completed_runs(
        args.root,
        keep_paths=args.keep,
        min_age_hours=args.min_age_hours,
        min_size_bytes=max(0, int(args.min_size_gib * 1024**3)),
        dry_run=args.dry_run,
        include_terminal_receipts=args.include_terminal_receipts,
    )
    print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
