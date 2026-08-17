"""Public, redacted accounting for every collection attempt.

The ledger is an append-only JSONL control-plane artifact.  It contains
counts and non-sensitive failure categories, never evaluator truth or local
source paths.  It is intentionally independent from the formal dataset index:
failed attempts remain countable without becoming training episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FAILURE_LEDGER_SCHEMA = "org.rivermark.benchmark.failure-ledger.v1"
CAPTURE_START_SCHEMA = "org.rivermark.isaac-capture-start.v1"
FAILURE_CATEGORIES = frozenset(
    {
        "none",
        "capture_failure",
        "sensor_failure",
        "physics_safety_abort",
        "task_failure",
        "evaluator_failure",
        "infrastructure_failure",
        "quality_failure",
        "license_failure",
    }
)
OUTCOMES = frozenset({"admitted", "quarantined", "failed"})
SPLITS = frozenset({"pilot", "train", "inner_dev", "validation", "blind_test", "ood_test"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_PRIVATE_TOKENS = ("evaluator", "private", "hidden_target", "target_truth")
_CAPTURE_TERMINAL_STATUSES = frozenset({"captured", "failed", "aborted"})
DEFAULT_CRASH_LEFT_MIN_AGE_HOURS = 24.0
_COLLECTION_BINDING_KEYS = frozenset(
    {"protocol_id", "protocol_sha256", "cell_id", "split", "episode_index", "episode_seed"}
)


@dataclass(frozen=True)
class FailureLedgerIssue:
    code: str
    path: str
    message: str


class FailureLedgerError(ValueError):
    """Raised when a public ledger record is malformed or unsafe."""


@dataclass(frozen=True)
class CrashLeftRecovery:
    """Public result for one crash-left marker considered by recovery."""

    attempt_id: str
    status: str
    stage: str
    reason_code: str


@dataclass(frozen=True)
class FailureRecord:
    attempt_id: str
    outcome: str
    category: str
    stage: str
    recorded_at: str
    split: str | None = None
    episode_id: str | None = None
    source_capture_sha256: str | None = None
    receipt_sha256: str | None = None
    reason_code: str | None = None
    collection_protocol_id: str | None = None
    collection_protocol_sha256: str | None = None
    collection_cell_id: str | None = None
    collection_episode_index: int | None = None
    episode_seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"schema": FAILURE_LEDGER_SCHEMA, **asdict(self)}


def _public_text(value: Any, *, path: str, issues: list[FailureLedgerIssue], required: bool = False) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not value or "\x00" in value:
        issues.append(FailureLedgerIssue("text", path, "must be a non-empty public string"))
        return
    lowered = value.lower()
    if any(token in lowered for token in _PRIVATE_TOKENS) or re.search(r"(?:[A-Za-z]:[\\/]|\\\\)", value):
        issues.append(FailureLedgerIssue("private_text", path, "private/evaluator paths and truth names are forbidden"))


def validate_failure_record(record: Any) -> tuple[FailureLedgerIssue, ...]:
    issues: list[FailureLedgerIssue] = []
    if not isinstance(record, Mapping):
        return (FailureLedgerIssue("type", "$", "record must be an object"),)
    allowed = {
        "schema",
        "attempt_id",
        "outcome",
        "category",
        "stage",
        "recorded_at",
        "split",
        "episode_id",
        "source_capture_sha256",
        "receipt_sha256",
        "reason_code",
        "collection_protocol_id",
        "collection_protocol_sha256",
        "collection_cell_id",
        "collection_episode_index",
        "episode_seed",
    }
    for key in sorted((key for key in record if not isinstance(key, str) or key not in allowed), key=str):
        issues.append(FailureLedgerIssue("unknown_field", f"$.{key}", "field is not part of ledger v1"))
    if record.get("schema") != FAILURE_LEDGER_SCHEMA:
        issues.append(FailureLedgerIssue("schema", "$.schema", f"expected {FAILURE_LEDGER_SCHEMA!r}"))
    for key in ("attempt_id", "stage", "recorded_at"):
        _public_text(record.get(key), path=f"$.{key}", issues=issues, required=True)
    recorded_at = record.get("recorded_at")
    if isinstance(recorded_at, str):
        try:
            parsed_time = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except ValueError:
            parsed_time = None
        if parsed_time is None or parsed_time.tzinfo is None:
            issues.append(
                FailureLedgerIssue(
                    "recorded_at",
                    "$.recorded_at",
                    "must be an ISO-8601 timestamp with an explicit timezone",
                )
            )
    attempt_id = record.get("attempt_id")
    if isinstance(attempt_id, str) and not _ID.fullmatch(attempt_id):
        issues.append(FailureLedgerIssue("attempt_id", "$.attempt_id", "invalid attempt identifier"))
    if record.get("outcome") not in OUTCOMES:
        issues.append(FailureLedgerIssue("outcome", "$.outcome", "unknown outcome"))
    if record.get("category") not in FAILURE_CATEGORIES:
        issues.append(FailureLedgerIssue("category", "$.category", "unknown failure category"))
    if record.get("outcome") == "admitted" and record.get("category") != "none":
        issues.append(FailureLedgerIssue("outcome_category", "$.category", "admitted records must use category 'none'"))
    if record.get("outcome") != "admitted" and record.get("category") == "none":
        issues.append(FailureLedgerIssue("outcome_category", "$.category", "failed/quarantined records need a failure category"))
    for key in ("split", "episode_id", "reason_code"):
        _public_text(record.get(key), path=f"$.{key}", issues=issues)
    if record.get("split") is not None and record.get("split") not in SPLITS:
        issues.append(FailureLedgerIssue("split", "$.split", "unknown benchmark split"))
    episode_id = record.get("episode_id")
    if episode_id is not None and isinstance(episode_id, str) and not _ID.fullmatch(episode_id):
        issues.append(FailureLedgerIssue("episode_id", "$.episode_id", "invalid episode identifier"))
    for key in ("source_capture_sha256", "receipt_sha256"):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
            issues.append(FailureLedgerIssue("sha256", f"$.{key}", "must be 64 lowercase hexadecimal characters"))
    protocol_fields = (
        "collection_protocol_id",
        "collection_protocol_sha256",
        "collection_cell_id",
        "collection_episode_index",
        "episode_seed",
    )
    present_protocol_fields = [key for key in protocol_fields if record.get(key) is not None]
    if present_protocol_fields and len(present_protocol_fields) != len(protocol_fields):
        issues.append(
            FailureLedgerIssue(
                "collection_binding",
                "$.collection_protocol_id",
                "collection protocol ID, hash, cell, and episode seed must be declared together",
            )
        )
    for key in ("collection_protocol_id", "collection_cell_id"):
        value = record.get(key)
        _public_text(value, path=f"$.{key}", issues=issues)
        if value is not None and isinstance(value, str) and not _ID.fullmatch(value):
            issues.append(FailureLedgerIssue(key, f"$.{key}", "invalid public identifier"))
    protocol_hash = record.get("collection_protocol_sha256")
    if protocol_hash is not None and (not isinstance(protocol_hash, str) or not _SHA256.fullmatch(protocol_hash)):
        issues.append(
            FailureLedgerIssue(
                "sha256",
                "$.collection_protocol_sha256",
                "must be 64 lowercase hexadecimal characters",
            )
        )
    episode_seed = record.get("episode_seed")
    episode_index = record.get("collection_episode_index")
    if episode_index is not None and (
        isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0
    ):
        issues.append(
            FailureLedgerIssue(
                "collection_episode_index",
                "$.collection_episode_index",
                "must be a non-negative integer",
            )
        )
    if episode_seed is not None and (
        isinstance(episode_seed, bool)
        or not isinstance(episode_seed, int)
        or not 0 <= episode_seed <= 0xFFFFFFFF
    ):
        issues.append(FailureLedgerIssue("episode_seed", "$.episode_seed", "must be an unsigned 32-bit integer"))
    return tuple(issues)


def _validated_collection_binding(value: Any) -> dict[str, Any] | None:
    """Validate a path-free capture marker binding and return a copy."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _COLLECTION_BINDING_KEYS:
        return None
    protocol_id = value.get("protocol_id")
    cell_id = value.get("cell_id")
    protocol_sha256 = value.get("protocol_sha256")
    split = value.get("split")
    episode_index = value.get("episode_index")
    episode_seed = value.get("episode_seed")
    if (
        not isinstance(protocol_id, str)
        or not _ID.fullmatch(protocol_id)
        or any(token in protocol_id.lower() for token in _PRIVATE_TOKENS)
        or not isinstance(cell_id, str)
        or not _ID.fullmatch(cell_id)
        or any(token in cell_id.lower() for token in _PRIVATE_TOKENS)
        or not isinstance(protocol_sha256, str)
        or not _SHA256.fullmatch(protocol_sha256)
        or split not in SPLITS
        or isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
        or isinstance(episode_seed, bool)
        or not isinstance(episode_seed, int)
        or not 0 <= episode_seed <= 0xFFFFFFFF
    ):
        return None
    return {
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "cell_id": cell_id,
        "split": split,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
    }


def _raise_if_invalid(record: Mapping[str, Any]) -> None:
    issues = validate_failure_record(record)
    if issues:
        formatted = "; ".join(f"{issue.code}:{issue.path}" for issue in issues)
        raise FailureLedgerError(formatted)


def append_failure_record(path: Path, record: FailureRecord | Mapping[str, Any]) -> None:
    """Append one validated public record without rewriting existing history."""

    payload = record.as_dict() if isinstance(record, FailureRecord) else dict(record)
    _raise_if_invalid(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_failure_record_once(path: Path, record: FailureRecord | Mapping[str, Any]) -> str:
    """Append a terminal record once, rejecting a conflicting retry.

    Capture finalization can be retried after a process interruption or an
    ambiguous control-plane error.  Rewriting the ledger would hide history,
    while blindly appending would make a single physical attempt invalid due
    to a duplicate ID.  The terminal capture path is single-owner, so a
    validated read-before-append is sufficient here; concurrent writers are
    still outside the capture contract.
    """

    payload = record.as_dict() if isinstance(record, FailureRecord) else dict(record)
    _raise_if_invalid(payload)
    attempt_id = str(payload["attempt_id"])
    if path.is_file():
        for existing in load_failure_ledger(path):
            if existing["attempt_id"] != attempt_id:
                continue
            comparable_keys = set(payload) | set(existing)
            comparable_keys.discard("recorded_at")
            if any(existing.get(key) != payload.get(key) for key in comparable_keys):
                raise FailureLedgerError(
                    f"conflicting terminal record for existing attempt_id: {attempt_id}"
                )
            return "already_recorded"
    append_failure_record(path, payload)
    return "appended"


def load_failure_ledger(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load an append-only public ledger and reject malformed duplicates."""

    if not path.is_file():
        raise FailureLedgerError(f"failure ledger is missing: {path}")
    records: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FailureLedgerError(f"invalid JSON at line {line_number}: {exc}") from exc
        _raise_if_invalid(record)
        attempt_id = record["attempt_id"]
        if attempt_id in seen:
            raise FailureLedgerError(f"duplicate attempt_id at line {line_number}: {attempt_id}")
        seen.add(attempt_id)
        records.append(record)
    return tuple(records)


def summarize_failure_ledger(path: Path) -> dict[str, Any]:
    """Validate and aggregate an append-only ledger, rejecting duplicates."""

    records = load_failure_ledger(path)
    seen = {str(record["attempt_id"]) for record in records}
    outcomes = Counter(record["outcome"] for record in records)
    categories = Counter(record["category"] for record in records if record["category"] != "none")
    return {
        "schema": FAILURE_LEDGER_SCHEMA,
        "attempt_count": len(records),
        "admitted_count": outcomes.get("admitted", 0),
        "quarantined_count": outcomes.get("quarantined", 0),
        "failed_count": outcomes.get("failed", 0),
        "failure_categories": dict(sorted(categories.items())),
        "attempt_ids_sha256": hashlib.sha256("\n".join(sorted(seen)).encode("utf-8")).hexdigest(),
    }


def _load_capture_start(path: Path) -> Mapping[str, Any] | None:
    """Load and validate the private-on-disk, public-field start marker."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    allowed = {
        "schema",
        "attempt_id",
        "started_wall_time_ns",
        "source_revision",
        "source_tree_sha256",
        "source_worktree_dirty",
        "task_kind",
        "control_mode",
        "agent_count_requested",
        "collection_binding",
    }
    if value.get("schema") != CAPTURE_START_SCHEMA or set(value) - allowed:
        return None
    attempt_id = value.get("attempt_id")
    if not isinstance(attempt_id, str) or not _ID.fullmatch(attempt_id):
        return None
    started = value.get("started_wall_time_ns")
    if isinstance(started, bool) or not isinstance(started, int) or started <= 0:
        return None
    source_revision = value.get("source_revision")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", source_revision):
        return None
    if not isinstance(value.get("source_tree_sha256"), str) or not _SHA256.fullmatch(value["source_tree_sha256"]):
        return None
    if not isinstance(value.get("source_worktree_dirty"), bool):
        return None
    if not isinstance(value.get("task_kind"), str) or not value["task_kind"]:
        return None
    if not isinstance(value.get("control_mode"), str) or not value["control_mode"]:
        return None
    agent_count = value.get("agent_count_requested")
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count < 1:
        return None
    binding = _validated_collection_binding(value.get("collection_binding"))
    if value.get("collection_binding") is not None and binding is None:
        return None
    return value


def _capture_receipt_status(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value.get("status") if isinstance(value, Mapping) and isinstance(value.get("status"), str) else None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _terminal_recovery_record(
    attempt_id: str,
    receipt_path: Path,
    receipt_status: str,
    recorded_at: str,
    collection_binding: Mapping[str, Any] | None = None,
) -> FailureRecord:
    receipt_hash = _sha256_file(receipt_path)
    binding_kwargs = _failure_binding_kwargs(collection_binding)
    split = collection_binding.get("split") if isinstance(collection_binding, Mapping) else "pilot"
    if receipt_status == "captured":
        return FailureRecord(
            attempt_id=attempt_id,
            outcome="quarantined",
            category="quality_failure",
            stage="isaac_capture_recovery",
            recorded_at=recorded_at,
            split=split if isinstance(split, str) else "pilot",
            receipt_sha256=receipt_hash,
            reason_code="development_evidence_not_formal",
            **binding_kwargs,
        )
    return FailureRecord(
        attempt_id=attempt_id,
        outcome="failed",
        category="capture_failure",
        stage="isaac_capture_recovery",
        recorded_at=recorded_at,
        split=split if isinstance(split, str) else "pilot",
        receipt_sha256=receipt_hash,
        reason_code="capture_not_completed",
        **binding_kwargs,
    )


def _failure_binding_kwargs(binding: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        return {}
    return {
        "collection_protocol_id": binding.get("protocol_id"),
        "collection_protocol_sha256": binding.get("protocol_sha256"),
        "collection_cell_id": binding.get("cell_id"),
        "collection_episode_index": binding.get("episode_index"),
        "episode_seed": binding.get("episode_seed"),
    }


def _capture_progress_stage(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "unknown"
    stage = value.get("stage") if isinstance(value, Mapping) else None
    if not isinstance(stage, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", stage):
        return "unknown"
    return stage.lower()


def recover_crash_left_attempts(
    root: Path,
    *,
    ledger_path: Path | None = None,
    min_age_hours: float = DEFAULT_CRASH_LEFT_MIN_AGE_HOURS,
    now_ns: int | None = None,
    dry_run: bool = False,
) -> tuple[CrashLeftRecovery, ...]:
    """Record stale start markers whose capture process never wrote a final receipt.

    Recovery is deliberately conservative: only direct child directories of a
    directory named ``rivermark-runs`` are considered, schema-invalid markers
    are ignored, and the newest file must exceed the age threshold.  A running
    or recently-updated capture therefore remains untouched.  Re-running this
    function is idempotent because the marker's attempt ID is checked against
    the validated ledger before an append.
    """

    if min_age_hours < 0:
        raise ValueError("min_age_hours must be non-negative")
    root = Path(root).expanduser().resolve()
    if root.name.lower() != "rivermark-runs" or not root.is_dir():
        return ()
    ledger = (ledger_path or (root / "failure_ledger.jsonl")).expanduser().resolve()
    seen: set[str] = set()
    if ledger.is_file():
        summary = summarize_failure_ledger(ledger)
        del summary
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(str(json.loads(line)["attempt_id"]))
    now = time.time_ns() if now_ns is None else int(now_ns)
    age_ns = int(float(min_age_hours) * 3600.0 * 1_000_000_000)
    recovered: list[CrashLeftRecovery] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        marker = candidate / "capture_start.json"
        start = _load_capture_start(marker)
        if start is None:
            continue
        attempt_id = str(start["attempt_id"])
        if attempt_id in seen:
            recovered.append(CrashLeftRecovery(attempt_id, "already_recorded", "unknown", "already_recorded"))
            continue
        collection_binding = _validated_collection_binding(start.get("collection_binding"))
        receipt_path = candidate / "capture_receipt.json"
        receipt_status = _capture_receipt_status(receipt_path)
        if receipt_status in _CAPTURE_TERMINAL_STATUSES:
            record = _terminal_recovery_record(
                attempt_id,
                receipt_path,
                receipt_status,
                datetime.fromtimestamp(now / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                collection_binding,
            )
            if not dry_run:
                append_failure_record(ledger, record)
                seen.add(attempt_id)
            recovered.append(
                CrashLeftRecovery(
                    attempt_id,
                    "recovered_terminal" if not dry_run else "would_recover_terminal",
                    receipt_status,
                    record.reason_code or "terminal_receipt",
                )
            )
            continue
        try:
            files = [item for item in candidate.rglob("*") if item.is_file()]
            newest_ns = max((item.stat().st_mtime_ns for item in files), default=0)
        except OSError:
            continue
        if newest_ns <= 0 or now - newest_ns < age_ns:
            recovered.append(CrashLeftRecovery(attempt_id, "protected_recent", "unknown", "recent_or_active"))
            continue
        stage = _capture_progress_stage(candidate / "capture_progress.json")
        reason_code = f"crash_left_after_{stage}"
        record = FailureRecord(
            attempt_id=attempt_id,
            outcome="failed",
            category="infrastructure_failure",
            stage="isaac_capture_recovery",
            recorded_at=datetime.fromtimestamp(now / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            split=(collection_binding.get("split") if collection_binding else "pilot"),
            reason_code=reason_code,
            **_failure_binding_kwargs(collection_binding),
        )
        if not dry_run:
            append_failure_record(ledger, record)
            seen.add(attempt_id)
        recovered.append(CrashLeftRecovery(attempt_id, "recovered" if not dry_run else "would_recover", stage, reason_code))
    return tuple(recovered)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    return parser.parse_args(argv)


def recovery_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover stale crash-left Isaac capture attempts")
    parser.add_argument("root", type=Path, help="rivermark-runs directory")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--min-age-hours", type=float, default=DEFAULT_CRASH_LEFT_MIN_AGE_HOURS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        records = recover_crash_left_attempts(
            args.root,
            ledger_path=args.ledger,
            min_age_hours=args.min_age_hours,
            dry_run=args.dry_run,
        )
    except (OSError, UnicodeDecodeError, FailureLedgerError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"schema": CAPTURE_START_SCHEMA, "status": "ok", "records": [asdict(record) for record in records]}, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        print(json.dumps(summarize_failure_ledger(args.ledger), indent=2, sort_keys=True))
        return 0
    except (OSError, UnicodeDecodeError, FailureLedgerError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
