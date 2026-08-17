"""Validate and selectively download immutable Rivermark release shards.

The release manifest is deliberately separate from a native Isaac capture.  It
contains only public shard metadata and opaque source-capture commitments; it
never contains evaluator-private targets.  Downloads are sequential, hashed
while written, and promoted atomically so a failed transfer cannot become a
valid-looking dataset file.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RELEASE_MANIFEST_SCHEMA = "org.rivermark.benchmark.release-manifest.v1"
DOWNLOAD_PLAN_SCHEMA = "org.rivermark.benchmark.download-plan.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_SPLITS = frozenset({"pilot", "train", "inner_dev", "validation", "blind_test", "ood_test"})
_TOP_KEYS = frozenset(
    {
        "schema",
        "dataset_version",
        "release_id",
        "license_status",
        "created_at",
        "source_revision",
        "supply_chain_manifest_sha256",
        "metadata_uri",
        "accounting",
        "defects",
        "shards",
    }
)
_SHARD_KEYS = frozenset(
    {
        "shard_id",
        "episode_id",
        "split",
        "modality",
        "agent_id",
        "frame_start",
        "frame_end",
        "media_type",
        "compression",
        "schema",
        "path",
        "url",
        "size_bytes",
        "sha256",
        "source_capture_sha256",
        "license_status",
    }
)
_PRIVATE_TOKENS = ("evaluator", "private", "hidden_target", "target_truth")
_FORMAL_RELEASE_SPLITS = ("blind_test", "inner_dev", "ood_test", "train", "validation")
_GENERIC_STREAM_SCHEMA = "org.rivermark.benchmark.episode-stream.v1"
_FAILURE_LEDGER_SCHEMA = "org.rivermark.benchmark.failure-ledger.v1"
_ACCOUNTING_KEYS = frozenset({"failure_ledger", "failure_summary"})
_ACCOUNTING_LEDGER_KEYS = frozenset(
    {"path", "url", "size_bytes", "sha256", "schema", "media_type", "compression", "license_status"}
)
_ACCOUNTING_SUMMARY_KEYS = frozenset(
    {
        "schema",
        "attempt_count",
        "admitted_count",
        "quarantined_count",
        "failed_count",
        "failure_categories",
        "attempt_ids_sha256",
    }
)
_DEFECT_KEYS = frozenset(
    {
        "issue_id",
        "status",
        "severity",
        "summary",
        "affected_shards",
        "correction_mapping",
        "version_bump_policy",
        "deprecation_window",
        "tombstone",
    }
)
_DEFECT_SHARD_KEYS = frozenset({"shard_id", "episode_id", "path", "frame_start", "frame_end", "original_sha256"})
_CORRECTION_KEYS = frozenset({"old_shard_id", "new_shard_id", "new_release_id", "new_dataset_version", "new_sha256"})
_DEPRECATION_KEYS = frozenset({"grace_releases", "replacement_required"})
_TOMBSTONE_KEYS = frozenset({"kind", "reason", "replacement_release_id", "replacement_shard_ids"})
_DEFECT_STATUSES = frozenset({"open", "resolved", "withdrawn"})
_DEFECT_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_VERSION_BUMP_POLICIES = frozenset({"patch", "minor", "major"})
_TOMBSTONE_KINDS = frozenset({"withdrawn", "superseded", "invalid"})
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)


def _contains_private_token(value: Any) -> bool:
    return isinstance(value, str) and any(token in value.lower() for token in _PRIVATE_TOKENS)


@dataclass(frozen=True)
class ReleaseManifestIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class DownloadResult:
    shard_id: str
    path: Path
    status: str
    size_bytes: int
    sha256: str


class ReleaseManifestError(ValueError):
    """Raised when a manifest or selected shard is unsafe or malformed."""


class DownloadError(RuntimeError):
    """Raised when a shard cannot be downloaded and hash-verified."""


class ReleaseBuildError(ReleaseManifestError):
    """Raised when a public release manifest cannot be built safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    path = value.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return False
    return all(part not in {"", ".", ".."} for part in path.split("/"))


def _issue(issues: list[ReleaseManifestIssue], code: str, path: str, message: str) -> None:
    issues.append(ReleaseManifestIssue(code, path, message))


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _validate_url(
    value: Any,
    *,
    path: str,
    require_https: bool,
    issues: list[ReleaseManifestIssue],
) -> None:
    if not isinstance(value, str) or not value:
        _issue(issues, "url", path, "must be a non-empty URL")
        return
    parsed = urllib.parse.urlparse(value)
    allowed = {"https"} if require_https else {"https", "http", "file"}
    if parsed.scheme not in allowed:
        _issue(issues, "url_scheme", path, f"scheme must be one of {sorted(allowed)}")
        return
    if parsed.scheme in {"http", "https"}:
        hostname = parsed.hostname
        if not hostname:
            _issue(issues, "url_host", path, "HTTP(S) URL must include a host")
        else:
            lowered_host = hostname.lower().rstrip(".")
            if lowered_host == "localhost" or lowered_host.endswith(".local"):
                _issue(issues, "private_host", path, "release URL must not target a local host")
            try:
                address = ipaddress.ip_address(lowered_host)
            except ValueError:
                address = None
            if address is not None and (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_unspecified
            ):
                _issue(issues, "private_host", path, "release URL must not target a private or local address")
    elif parsed.scheme == "file" and not parsed.path:
        _issue(issues, "url_path", path, "file URL must include a path")
    lowered = value.lower()
    if any(token in lowered for token in _PRIVATE_TOKENS):
        _issue(issues, "private_url", path, "release URL must not contain private/evaluator payload names")


def validate_release_manifest(
    payload: Any,
    *,
    require_https: bool = False,
) -> tuple[ReleaseManifestIssue, ...]:
    """Validate the public release-manifest v1 contract without downloading it."""

    issues: list[ReleaseManifestIssue] = []
    if not isinstance(payload, Mapping):
        return (ReleaseManifestIssue("type", "$", "manifest must be an object"),)
    unknown = set(payload) - _TOP_KEYS
    for key in sorted(unknown):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of release-manifest v1")
    for key in ("schema", "dataset_version", "release_id", "license_status", "shards"):
        if key not in payload:
            _issue(issues, "required", f"$.{key}", "field is required")
    if payload.get("schema") != RELEASE_MANIFEST_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {RELEASE_MANIFEST_SCHEMA!r}")
    if not isinstance(payload.get("dataset_version"), str) or not _SEMVER.fullmatch(payload["dataset_version"]):
        _issue(issues, "dataset_version", "$.dataset_version", "must be a semantic version")
    if not isinstance(payload.get("release_id"), str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", str(payload.get("release_id", ""))
    ):
        _issue(issues, "release_id", "$.release_id", "invalid release identifier")
    if payload.get("license_status") != "redistribution_cleared":
        _issue(
            issues,
            "license_status",
            "$.license_status",
            "public release requires redistribution_cleared",
        )
    source_revision = payload.get("source_revision")
    if source_revision is not None and (
        not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", source_revision)
    ):
        _issue(issues, "source_revision", "$.source_revision", "must be a lowercase Git revision")
    if "supply_chain_manifest_sha256" in payload and not _valid_sha(payload["supply_chain_manifest_sha256"]):
        _issue(
            issues,
            "supply_chain_manifest_sha256",
            "$.supply_chain_manifest_sha256",
            "must be 64 lowercase hexadecimal characters",
        )
    defects = payload.get("defects")
    defect_records: list[Mapping[str, Any]] = []
    if defects is not None:
        if not isinstance(defects, list):
            _issue(issues, "defects", "$.defects", "must be an array")
        else:
            seen_issue_ids: set[str] = set()
            for index, defect in enumerate(defects):
                path = f"$.defects[{index}]"
                if not isinstance(defect, Mapping):
                    _issue(issues, "type", path, "defect must be an object")
                    continue
                defect_records.append(defect)
                for key in sorted(set(defect) - _DEFECT_KEYS):
                    _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of defect index v1")
                for key in ("issue_id", "status", "severity", "summary", "affected_shards", "version_bump_policy", "deprecation_window"):
                    if key not in defect:
                        _issue(issues, "required", f"{path}.{key}", "field is required")
                issue_id = defect.get("issue_id")
                if not isinstance(issue_id, str) or not re.fullmatch(r"[A-Z][A-Z0-9._-]{2,63}", issue_id):
                    _issue(issues, "issue_id", f"{path}.issue_id", "must be a stable uppercase identifier")
                elif issue_id in seen_issue_ids:
                    _issue(issues, "duplicate_issue_id", f"{path}.issue_id", "issue IDs must be unique")
                else:
                    seen_issue_ids.add(issue_id)
                status = defect.get("status")
                if status not in _DEFECT_STATUSES:
                    _issue(issues, "defect_status", f"{path}.status", "unknown defect status")
                if defect.get("severity") not in _DEFECT_SEVERITIES:
                    _issue(issues, "defect_severity", f"{path}.severity", "unknown defect severity")
                if not isinstance(defect.get("summary"), str) or not defect.get("summary"):
                    _issue(issues, "defect_summary", f"{path}.summary", "must be a non-empty public summary")
                elif _contains_private_token(defect.get("summary")):
                    _issue(issues, "private_field", f"{path}.summary", "must not mention private/evaluator payloads")
                affected = defect.get("affected_shards")
                if not isinstance(affected, list) or not affected:
                    _issue(issues, "affected_shards", f"{path}.affected_shards", "must contain at least one shard")
                else:
                    seen_affected: set[str] = set()
                    for affected_index, entry in enumerate(affected):
                        entry_path = f"{path}.affected_shards[{affected_index}]"
                        if not isinstance(entry, Mapping):
                            _issue(issues, "type", entry_path, "affected shard must be an object")
                            continue
                        for key in sorted(set(entry) - _DEFECT_SHARD_KEYS):
                            _issue(issues, "unknown_field", f"{entry_path}.{key}", "field is not part of defect shard reference v1")
                        for key in ("shard_id", "episode_id", "path", "original_sha256"):
                            if key not in entry:
                                _issue(issues, "required", f"{entry_path}.{key}", "field is required")
                        shard_id = entry.get("shard_id")
                        if not isinstance(shard_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", shard_id):
                            _issue(issues, "shard_id", f"{entry_path}.shard_id", "invalid shard identifier")
                        elif shard_id in seen_affected:
                            _issue(issues, "duplicate_shard_id", f"{entry_path}.shard_id", "affected shard IDs must be unique")
                        else:
                            seen_affected.add(shard_id)
                        episode_id = entry.get("episode_id")
                        if not isinstance(episode_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", episode_id):
                            _issue(issues, "episode_id", f"{entry_path}.episode_id", "invalid episode identifier")
                        if not _is_safe_relative_path(entry.get("path")) or any(token in str(entry.get("path", "")).lower() for token in _PRIVATE_TOKENS):
                            _issue(issues, "unsafe_path", f"{entry_path}.path", "must be a public safe relative path")
                        if not _valid_sha(entry.get("original_sha256")):
                            _issue(issues, "sha256", f"{entry_path}.original_sha256", "must be 64 lowercase hexadecimal characters")
                        start = entry.get("frame_start")
                        end = entry.get("frame_end")
                        if (start is None) != (end is None):
                            _issue(issues, "frame_range", entry_path, "frame_start and frame_end must be provided together")
                        if start is not None and (not isinstance(start, int) or isinstance(start, bool) or start < 0):
                            _issue(issues, "frame_range", f"{entry_path}.frame_start", "must be a non-negative integer")
                        if end is not None and (not isinstance(end, int) or isinstance(end, bool) or end <= (start if isinstance(start, int) else 0)):
                            _issue(issues, "frame_range", f"{entry_path}.frame_end", "must be greater than frame_start")
                if defect.get("version_bump_policy") not in _VERSION_BUMP_POLICIES:
                    _issue(issues, "version_bump_policy", f"{path}.version_bump_policy", "must be patch, minor, or major")
                window = defect.get("deprecation_window")
                if not isinstance(window, Mapping):
                    _issue(issues, "deprecation_window", f"{path}.deprecation_window", "must be an object")
                else:
                    for key in sorted(set(window) - _DEPRECATION_KEYS):
                        _issue(issues, "unknown_field", f"{path}.deprecation_window.{key}", "field is not part of deprecation window v1")
                    grace = window.get("grace_releases")
                    if not isinstance(grace, int) or isinstance(grace, bool) or grace < 0:
                        _issue(issues, "deprecation_window", f"{path}.deprecation_window.grace_releases", "must be a non-negative integer")
                    if not isinstance(window.get("replacement_required"), bool):
                        _issue(issues, "deprecation_window", f"{path}.deprecation_window.replacement_required", "must be boolean")
                correction = defect.get("correction_mapping")
                if correction is not None:
                    if not isinstance(correction, list) or not correction:
                        _issue(issues, "correction_mapping", f"{path}.correction_mapping", "must be a non-empty array when present")
                    else:
                        for correction_index, mapping in enumerate(correction):
                            mapping_path = f"{path}.correction_mapping[{correction_index}]"
                            if not isinstance(mapping, Mapping):
                                _issue(issues, "type", mapping_path, "correction mapping must be an object")
                                continue
                            for key in sorted(set(mapping) - _CORRECTION_KEYS):
                                _issue(issues, "unknown_field", f"{mapping_path}.{key}", "field is not part of correction mapping v1")
                            for key in ("old_shard_id", "new_release_id", "new_dataset_version", "new_sha256"):
                                if key not in mapping:
                                    _issue(issues, "required", f"{mapping_path}.{key}", "field is required")
                            if not isinstance(mapping.get("old_shard_id"), str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", str(mapping.get("old_shard_id", ""))):
                                _issue(issues, "shard_id", f"{mapping_path}.old_shard_id", "invalid shard identifier")
                            if not isinstance(mapping.get("new_release_id"), str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", str(mapping.get("new_release_id", ""))):
                                _issue(issues, "release_id", f"{mapping_path}.new_release_id", "invalid release identifier")
                            if not isinstance(mapping.get("new_dataset_version"), str) or not _SEMVER.fullmatch(str(mapping.get("new_dataset_version", ""))):
                                _issue(issues, "dataset_version", f"{mapping_path}.new_dataset_version", "must be a semantic version")
                            if not _valid_sha(mapping.get("new_sha256")):
                                _issue(issues, "sha256", f"{mapping_path}.new_sha256", "must be 64 lowercase hexadecimal characters")
                            new_shard_id = mapping.get("new_shard_id")
                            if new_shard_id is not None and (not isinstance(new_shard_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", new_shard_id)):
                                _issue(issues, "shard_id", f"{mapping_path}.new_shard_id", "invalid shard identifier")
                elif status == "resolved":
                    _issue(issues, "correction_mapping", f"{path}.correction_mapping", "resolved defects require an immutable correction mapping")
                tombstone = defect.get("tombstone")
                if tombstone is not None:
                    if not isinstance(tombstone, Mapping):
                        _issue(issues, "tombstone", f"{path}.tombstone", "must be an object")
                    else:
                        for key in sorted(set(tombstone) - _TOMBSTONE_KEYS):
                            _issue(issues, "unknown_field", f"{path}.tombstone.{key}", "field is not part of tombstone v1")
                        for key in ("kind", "reason"):
                            if key not in tombstone:
                                _issue(issues, "required", f"{path}.tombstone.{key}", "field is required")
                        if tombstone.get("kind") not in _TOMBSTONE_KINDS:
                            _issue(issues, "tombstone", f"{path}.tombstone.kind", "unknown tombstone kind")
                        if not isinstance(tombstone.get("reason"), str) or not tombstone.get("reason"):
                            _issue(issues, "tombstone", f"{path}.tombstone.reason", "must be a non-empty public reason")
                        elif _contains_private_token(tombstone.get("reason")):
                            _issue(issues, "private_field", f"{path}.tombstone.reason", "must not mention private/evaluator payloads")
                        replacement_ids = tombstone.get("replacement_shard_ids")
                        if replacement_ids is not None and (not isinstance(replacement_ids, list) or any(not isinstance(value, str) for value in replacement_ids)):
                            _issue(issues, "tombstone", f"{path}.tombstone.replacement_shard_ids", "must be an array of shard IDs")
                        replacement_release = tombstone.get("replacement_release_id")
                        if replacement_release is not None and (not isinstance(replacement_release, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", replacement_release)):
                            _issue(issues, "release_id", f"{path}.tombstone.replacement_release_id", "invalid release identifier")
                elif status == "withdrawn":
                    _issue(issues, "tombstone", f"{path}.tombstone", "withdrawn defects require a machine-readable tombstone")
    accounting = payload.get("accounting")
    if accounting is not None:
        accounting_path = "$.accounting"
        if not isinstance(accounting, Mapping):
            _issue(issues, "accounting", accounting_path, "must be an object")
        else:
            for key in sorted(set(accounting) - _ACCOUNTING_KEYS):
                _issue(issues, "unknown_field", f"{accounting_path}.{key}", "field is not part of accounting v1")
            ledger = accounting.get("failure_ledger")
            if not isinstance(ledger, Mapping):
                _issue(issues, "accounting_ledger", f"{accounting_path}.failure_ledger", "must be an object")
            else:
                for key in sorted(set(ledger) - _ACCOUNTING_LEDGER_KEYS):
                    _issue(
                        issues,
                        "unknown_field",
                        f"{accounting_path}.failure_ledger.{key}",
                        "field is not part of accounting ledger v1",
                    )
                for key in ("path", "url", "size_bytes", "sha256", "schema", "media_type"):
                    if key not in ledger:
                        _issue(
                            issues,
                            "required",
                            f"{accounting_path}.failure_ledger.{key}",
                            "field is required",
                        )
                if not _is_safe_relative_path(ledger.get("path")):
                    _issue(
                        issues,
                        "unsafe_path",
                        f"{accounting_path}.failure_ledger.path",
                        "must be a safe relative path",
                    )
                _validate_url(
                    ledger.get("url"),
                    path=f"{accounting_path}.failure_ledger.url",
                    require_https=require_https,
                    issues=issues,
                )
                size = ledger.get("size_bytes")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    _issue(
                        issues,
                        "size_bytes",
                        f"{accounting_path}.failure_ledger.size_bytes",
                        "must be a non-negative integer",
                    )
                if not _valid_sha(ledger.get("sha256")):
                    _issue(
                        issues,
                        "sha256",
                        f"{accounting_path}.failure_ledger.sha256",
                        "must be 64 lowercase hexadecimal characters",
                    )
                if ledger.get("schema") != _FAILURE_LEDGER_SCHEMA:
                    _issue(
                        issues,
                        "accounting_schema",
                        f"{accounting_path}.failure_ledger.schema",
                        f"expected {_FAILURE_LEDGER_SCHEMA!r}",
                    )
                for key in ("media_type",):
                    if not isinstance(ledger.get(key), str) or not ledger[key]:
                        _issue(
                            issues,
                            key,
                            f"{accounting_path}.failure_ledger.{key}",
                            "must be a non-empty string",
                        )
                if "compression" in ledger and (
                    not isinstance(ledger["compression"], str) or not ledger["compression"]
                ):
                    _issue(
                        issues,
                        "compression",
                        f"{accounting_path}.failure_ledger.compression",
                        "must be a non-empty string",
                    )
                if "license_status" in ledger and ledger["license_status"] != "redistribution_cleared":
                    _issue(
                        issues,
                        "license_status",
                        f"{accounting_path}.failure_ledger.license_status",
                        "accounting file is not redistribution-cleared",
                    )
            summary = accounting.get("failure_summary")
            if not isinstance(summary, Mapping):
                _issue(issues, "accounting_summary", f"{accounting_path}.failure_summary", "must be an object")
            else:
                for key in sorted(set(summary) - _ACCOUNTING_SUMMARY_KEYS):
                    _issue(
                        issues,
                        "unknown_field",
                        f"{accounting_path}.failure_summary.{key}",
                        "field is not part of accounting summary v1",
                    )
                for key in (
                    "schema",
                    "attempt_count",
                    "admitted_count",
                    "quarantined_count",
                    "failed_count",
                    "failure_categories",
                    "attempt_ids_sha256",
                ):
                    if key not in summary:
                        _issue(
                            issues,
                            "required",
                            f"{accounting_path}.failure_summary.{key}",
                            "field is required",
                        )
                if summary.get("schema") != _FAILURE_LEDGER_SCHEMA:
                    _issue(
                        issues,
                        "accounting_schema",
                        f"{accounting_path}.failure_summary.schema",
                        f"expected {_FAILURE_LEDGER_SCHEMA!r}",
                    )
                counts: list[int] = []
                for key in ("attempt_count", "admitted_count", "quarantined_count", "failed_count"):
                    value = summary.get(key)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        _issue(
                            issues,
                            "accounting_count",
                            f"{accounting_path}.failure_summary.{key}",
                            "must be a non-negative integer",
                        )
                    else:
                        counts.append(value)
                if len(counts) == 4 and counts[0] != sum(counts[1:]):
                    _issue(
                        issues,
                        "accounting_count_mismatch",
                        f"{accounting_path}.failure_summary",
                        "attempt_count must equal admitted + quarantined + failed",
                    )
                categories = summary.get("failure_categories")
                if not isinstance(categories, Mapping):
                    _issue(
                        issues,
                        "accounting_categories",
                        f"{accounting_path}.failure_summary.failure_categories",
                        "must be an object",
                    )
                else:
                    for category, count in categories.items():
                        if (
                            not isinstance(category, str)
                            or not category
                            or not isinstance(count, int)
                            or isinstance(count, bool)
                            or count < 0
                        ):
                            _issue(
                                issues,
                                "accounting_categories",
                                f"{accounting_path}.failure_summary.failure_categories",
                                "category names and counts must be public non-negative values",
                            )
                if not _valid_sha(summary.get("attempt_ids_sha256")):
                    _issue(
                        issues,
                        "sha256",
                        f"{accounting_path}.failure_summary.attempt_ids_sha256",
                        "must be 64 lowercase hexadecimal characters",
                    )
    for key in ("created_at", "metadata_uri"):
        if key in payload:
            if key == "created_at" and (not isinstance(payload[key], str) or not payload[key]):
                _issue(issues, "created_at", "$.created_at", "must be a non-empty string")
            elif key == "metadata_uri":
                _validate_url(payload[key], path="$.metadata_uri", require_https=require_https, issues=issues)
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        _issue(issues, "shards", "$.shards", "must contain at least one shard")
        return tuple(issues)

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, shard in enumerate(shards):
        path = f"$.shards[{index}]"
        if not isinstance(shard, Mapping):
            _issue(issues, "type", path, "shard must be an object")
            continue
        for key in sorted(set(shard) - _SHARD_KEYS):
            _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of shard v1")
        required = (
            "shard_id",
            "episode_id",
            "split",
            "modality",
            "media_type",
            "schema",
            "path",
            "url",
            "size_bytes",
            "sha256",
            "source_capture_sha256",
        )
        for key in required:
            if key not in shard:
                _issue(issues, "required", f"{path}.{key}", "field is required")
        shard_id = shard.get("shard_id")
        if not isinstance(shard_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", shard_id):
            _issue(issues, "shard_id", f"{path}.shard_id", "invalid shard identifier")
        elif shard_id in seen_ids:
            _issue(issues, "duplicate_shard_id", f"{path}.shard_id", "shard_id must be unique")
        else:
            seen_ids.add(shard_id)
        episode_id = shard.get("episode_id")
        if not isinstance(episode_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", episode_id):
            _issue(issues, "episode_id", f"{path}.episode_id", "invalid episode identifier")
        if shard.get("split") not in _SPLITS:
            _issue(issues, "split", f"{path}.split", "unknown benchmark split")
        for key in ("modality", "media_type", "schema"):
            value = shard.get(key)
            if not isinstance(value, str) or not value:
                _issue(issues, key, f"{path}.{key}", "must be a non-empty string")
        relative = shard.get("path")
        if not _is_safe_relative_path(relative):
            _issue(issues, "unsafe_path", f"{path}.path", "must be a safe relative path")
        elif relative in seen_paths:
            _issue(issues, "duplicate_path", f"{path}.path", "shard paths must be unique")
        else:
            seen_paths.add(relative)
        _validate_url(shard.get("url"), path=f"{path}.url", require_https=require_https, issues=issues)
        size = shard.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _issue(issues, "size_bytes", f"{path}.size_bytes", "must be a non-negative integer")
        for key in ("sha256", "source_capture_sha256"):
            if not _valid_sha(shard.get(key)):
                _issue(issues, "sha256", f"{path}.{key}", "must be 64 lowercase hexadecimal characters")
        if "agent_id" in shard and (
            not isinstance(shard["agent_id"], int) or isinstance(shard["agent_id"], bool) or shard["agent_id"] < 0
        ):
            _issue(issues, "agent_id", f"{path}.agent_id", "must be a non-negative integer")
        start = shard.get("frame_start")
        end = shard.get("frame_end")
        if (start is None) != (end is None):
            _issue(issues, "frame_range", path, "frame_start and frame_end must be provided together")
        if start is not None and (
            not isinstance(start, int) or isinstance(start, bool) or start < 0
        ):
            _issue(issues, "frame_range", f"{path}.frame_start", "must be a non-negative integer")
        if end is not None and (
            not isinstance(end, int) or isinstance(end, bool) or end <= (start if isinstance(start, int) else 0)
        ):
            _issue(issues, "frame_range", f"{path}.frame_end", "must be greater than frame_start")
        if "license_status" in shard and shard["license_status"] != "redistribution_cleared":
            _issue(issues, "license_status", f"{path}.license_status", "shard is not redistribution-cleared")
    if isinstance(accounting, Mapping):
        ledger = accounting.get("failure_ledger")
        ledger_path = ledger.get("path") if isinstance(ledger, Mapping) else None
        if isinstance(ledger_path, str) and ledger_path in seen_paths:
            _issue(
                issues,
                "duplicate_path",
                "$.accounting.failure_ledger.path",
                "accounting path must not collide with a release shard",
            )
    shard_by_id = {
        shard.get("shard_id"): shard
        for shard in shards
        if isinstance(shard, Mapping) and isinstance(shard.get("shard_id"), str)
    }
    for defect_index, defect in enumerate(defect_records):
        affected = defect.get("affected_shards")
        if not isinstance(affected, list):
            continue
        for affected_index, reference in enumerate(affected):
            if not isinstance(reference, Mapping):
                continue
            shard_id = reference.get("shard_id")
            actual = shard_by_id.get(shard_id)
            reference_path = f"$.defects[{defect_index}].affected_shards[{affected_index}]"
            if actual is None:
                _issue(issues, "unknown_shard", f"{reference_path}.shard_id", "defect must reference a shard in this manifest")
                continue
            for key in ("episode_id", "path"):
                if reference.get(key) != actual.get(key):
                    _issue(issues, "defect_binding", f"{reference_path}.{key}", f"must match manifest shard {shard_id!r}")
            if reference.get("original_sha256") != actual.get("sha256"):
                _issue(issues, "defect_binding", f"{reference_path}.original_sha256", f"must match immutable shard hash for {shard_id!r}")
            for key in ("frame_start", "frame_end"):
                if key in reference and reference.get(key) != actual.get(key):
                    _issue(issues, "defect_binding", f"{reference_path}.{key}", f"must match manifest shard {shard_id!r}")
        correction = defect.get("correction_mapping")
        if isinstance(correction, list):
            for correction_index, mapping in enumerate(correction):
                if not isinstance(mapping, Mapping):
                    continue
                old_id = mapping.get("old_shard_id")
                if old_id not in shard_by_id:
                    _issue(issues, "unknown_shard", f"$.defects[{defect_index}].correction_mapping[{correction_index}].old_shard_id", "correction must reference an affected shard in this manifest")
                new_id = mapping.get("new_shard_id")
                if new_id in shard_by_id and mapping.get("new_sha256") != shard_by_id[new_id].get("sha256"):
                    _issue(issues, "correction_binding", f"$.defects[{defect_index}].correction_mapping[{correction_index}].new_sha256", "must match the replacement shard hash")
    return tuple(issues)


def _release_episode_roots(dataset_root: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for split in _FORMAL_RELEASE_SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for episode_root in sorted(split_root.iterdir(), key=lambda path: path.name):
            if episode_root.is_dir():
                roots.append(episode_root)
    return tuple(roots)


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ReleaseBuildError(f"{label} must be a JSON object: {path}")
    return value


def _release_url(base_url: str, relative: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"https", "http", "file"}:
        raise ReleaseBuildError("base_url must use https, http, or file")
    if parsed.scheme in {"https", "http"} and not parsed.netloc:
        raise ReleaseBuildError("base_url must include a host")
    if parsed.scheme == "file" and not parsed.path:
        raise ReleaseBuildError("file base_url must include a path")
    root = base_url if base_url.endswith("/") else base_url + "/"
    return urllib.parse.urljoin(root, urllib.parse.quote(relative, safe="/._-"))


def _add_release_shard(
    shards: dict[str, dict[str, Any]],
    *,
    dataset_root: Path,
    episode_root: Path,
    split: str,
    episode_id: str,
    source_capture_sha256: str,
    base_url: str,
    stream_id: str,
    modality: str,
    media_type: str,
    schema: str,
    relative: str,
    agent_id: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> None:
    if not _is_safe_relative_path(relative) or any(token in relative.lower() for token in _PRIVATE_TOKENS):
        raise ReleaseBuildError(f"refusing unsafe or private payload path: {relative!r}")
    payload = (episode_root / relative).resolve()
    root = dataset_root.resolve()
    if not payload.is_file() or not payload.is_relative_to(root):
        raise ReleaseBuildError(f"manifest-bound payload is missing: {relative}")
    path = (Path(split) / episode_id / Path(relative)).as_posix()
    shard_id = f"{split}-{episode_id}-{stream_id}"
    if agent_id is not None:
        shard_id += f"-agent-{agent_id}"
    if len(shard_id) > 128:
        suffix = hashlib.sha256(shard_id.encode("utf-8")).hexdigest()[:16]
        shard_id = shard_id[:111] + "-" + suffix
    if shard_id in shards:
        raise ReleaseBuildError(f"duplicate release shard id: {shard_id}")
    record: dict[str, Any] = {
        "shard_id": shard_id,
        "episode_id": episode_id,
        "split": split,
        "modality": modality,
        "media_type": media_type,
        "compression": "as-stored",
        "schema": schema or _GENERIC_STREAM_SCHEMA,
        "path": path,
        "url": _release_url(base_url, path),
        "size_bytes": payload.stat().st_size,
        "sha256": sha256_file(payload),
        "source_capture_sha256": source_capture_sha256,
    }
    # Frame ranges are emitted only when the source manifest explicitly
    # declares them.  The builder never infers a frame count from bytes.
    if frame_start is not None or frame_end is not None:
        if (
            not isinstance(frame_start, int)
            or isinstance(frame_start, bool)
            or not isinstance(frame_end, int)
            or isinstance(frame_end, bool)
            or frame_start < 0
            or frame_end <= frame_start
        ):
            raise ReleaseBuildError(f"invalid declared frame range for {relative}")
        record.update(frame_start=frame_start, frame_end=frame_end)
    if agent_id is not None:
        record["agent_id"] = agent_id
    shards[shard_id] = record


def build_release_manifest(
    dataset_root: Path,
    *,
    release_id: str,
    base_url: str,
    source_revision: str,
    dataset_version: str | None = None,
    output_path: Path | None = None,
    created_at: str | None = None,
    supply_chain_manifest: Path | None = None,
    require_https: bool = True,
) -> dict[str, Any]:
    """Build a hash-bound manifest from an already verified public dataset.

    This function does not admit captures or copy payloads.  It refuses an
    empty/invalid dataset, reads only public stream bindings from each formal
    release episode, and never infers frame ranges from file size or names.
    ``base_url`` is explicit so a local path cannot accidentally be published
    as a public download location.
    """

    root = dataset_root.resolve()
    if not root.is_dir():
        raise ReleaseBuildError(f"dataset root is not a directory: {dataset_root}")
    if not isinstance(release_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", release_id):
        raise ReleaseBuildError("release_id is not a valid lowercase identifier")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{7,64}", source_revision):
        raise ReleaseBuildError("source_revision must be a lowercase Git revision")
    base_issues: list[ReleaseManifestIssue] = []
    _validate_url(base_url, path="$.base_url", require_https=require_https, issues=base_issues)
    if base_issues:
        raise ReleaseBuildError("invalid base_url: " + "; ".join(issue.message for issue in base_issues))

    episode_roots = _release_episode_roots(root)
    if not episode_roots:
        raise ReleaseBuildError("refusing to build a release manifest from an empty formal dataset")

    # Import lazily to keep the release downloader dependency-light and avoid
    # importing dataset-admission code for ordinary manifest downloads.
    from .formal_dataset import verify_dataset_integrity

    integrity = verify_dataset_integrity(root)
    if integrity.issues:
        raise ReleaseBuildError(
            "public dataset failed integrity verification: "
            + "; ".join(issue.code for issue in integrity.issues)
        )

    ledger_path = root / "manifests" / "failure_ledger.jsonl"
    try:
        from .failure_ledger import FailureLedgerError, summarize_failure_ledger

        failure_summary = summarize_failure_ledger(ledger_path)
    except (FailureLedgerError, OSError, UnicodeDecodeError) as exc:
        raise ReleaseBuildError(f"public dataset lacks a valid failure ledger: {exc}") from exc
    if failure_summary["admitted_count"] != len(episode_roots):
        raise ReleaseBuildError(
            "failure ledger admitted_count does not match formal episode count: "
            f"{failure_summary['admitted_count']} != {len(episode_roots)}"
        )
    accounting = {
        "failure_ledger": {
            "path": "manifests/failure_ledger.jsonl",
            "url": _release_url(base_url, "manifests/failure_ledger.jsonl"),
            "size_bytes": ledger_path.stat().st_size,
            "sha256": sha256_file(ledger_path),
            "schema": _FAILURE_LEDGER_SCHEMA,
            "media_type": "application/x-ndjson",
            "compression": "none",
            "license_status": "redistribution_cleared",
        },
        "failure_summary": failure_summary,
    }

    if supply_chain_manifest is None:
        raise ReleaseBuildError(
            "public release requires a supply-chain manifest with explicit license, SBOM, and signature evidence"
        )
    from .supply_chain import (
        load_supply_chain_manifest,
        supply_chain_sha256,
        verify_supply_chain_manifest,
    )

    supply_chain_report = verify_supply_chain_manifest(supply_chain_manifest, require_release=True)
    if supply_chain_report["status"] != "valid":
        raise ReleaseBuildError(
            "supply-chain manifest failed release validation: "
            + "; ".join(issue["code"] for issue in supply_chain_report["issues"])
        )
    if supply_chain_report.get("release_id") != release_id:
        raise ReleaseBuildError(
            "supply-chain manifest release_id does not match requested release_id"
        )
    supply_chain_payload = load_supply_chain_manifest(supply_chain_manifest)
    if supply_chain_sha256(supply_chain_payload) != supply_chain_report["manifest_sha256"]:
        raise ReleaseBuildError("supply-chain manifest changed after release verification")
    supply_chain_assets = supply_chain_payload.get("assets")
    if not isinstance(supply_chain_assets, list):
        raise ReleaseBuildError("supply-chain manifest has no usable asset inventory")
    required_asset_kinds = {"code", "scene_layer", "robot_asset", "data"}
    asset_kinds = {
        asset.get("kind")
        for asset in supply_chain_assets
        if isinstance(asset, Mapping)
    }
    missing_asset_kinds = sorted(required_asset_kinds - asset_kinds)
    if missing_asset_kinds:
        raise ReleaseBuildError(
            f"supply-chain manifest is missing dataset release surfaces: {missing_asset_kinds}"
        )
    cleared_data_hashes = {
        asset.get("sha256")
        for asset in supply_chain_assets
        if isinstance(asset, Mapping)
        and asset.get("kind") == "data"
        and asset.get("license_status") == "redistribution_cleared"
        and asset.get("redistributable") is True
    }

    shards: dict[str, dict[str, Any]] = {}
    versions: set[str] = set()
    for episode_root in episode_roots:
        manifest = _read_object(episode_root / "episode_manifest.json", label="episode manifest")
        admission = _read_object(episode_root / "admission.json", label="release admission")
        split = manifest.get("split")
        episode_id = manifest.get("episode_id")
        if split not in _FORMAL_RELEASE_SPLITS or not isinstance(episode_id, str):
            raise ReleaseBuildError(f"release episode has invalid split or episode_id: {episode_root}")
        if admission.get("formal_benchmark_admission") is not True:
            raise ReleaseBuildError(f"release episode is not formally admitted: {episode_root}")
        if admission.get("supply_chain_manifest_sha256") != supply_chain_report["manifest_sha256"]:
            raise ReleaseBuildError(f"release admission binds a different supply-chain manifest: {episode_root}")
        if admission.get("supply_chain_release_id") != release_id:
            raise ReleaseBuildError(f"release admission binds a different supply-chain release_id: {episode_root}")
        source_capture = admission.get("formal_capture_receipt_sha256")
        if not _valid_sha(source_capture):
            raise ReleaseBuildError(f"release admission lacks a valid capture receipt hash: {episode_root}")
        if source_capture not in cleared_data_hashes:
            raise ReleaseBuildError(f"supply-chain data assets do not bind release episode: {episode_root}")
        version = manifest.get("dataset_version")
        if not isinstance(version, str) or not _SEMVER.fullmatch(version):
            raise ReleaseBuildError(f"release episode has invalid dataset_version: {episode_root}")
        versions.add(version)
        streams = manifest.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ReleaseBuildError(f"release episode has no public streams: {episode_root}")
        for metadata_name, metadata_schema in (
            ("episode_manifest.json", "org.rivermark.benchmark.episode.v1"),
            ("lineage.json", "org.rivermark.benchmark.episode-lineage.v1"),
            ("formal_capture_receipt.json", "org.rivermark.benchmark.formal-capture-receipt.v1"),
            ("admission.json", "org.rivermark.benchmark.release-admission.v1"),
        ):
            _add_release_shard(
                shards,
                dataset_root=root,
                episode_root=episode_root,
                split=split,
                episode_id=episode_id,
                source_capture_sha256=source_capture,
                base_url=base_url,
                stream_id="metadata-" + metadata_name.removesuffix(".json").replace("_", "-"),
                modality="metadata",
                media_type="application/json",
                schema=metadata_schema,
                relative=metadata_name,
            )
        abi_ref = manifest.get("observation_abi")
        if not isinstance(abi_ref, Mapping) or not isinstance(abi_ref.get("path"), str):
            raise ReleaseBuildError(f"release episode lacks a bound observation ABI: {episode_root}")
        _add_release_shard(
            shards,
            dataset_root=root,
            episode_root=episode_root,
            split=split,
            episode_id=episode_id,
            source_capture_sha256=source_capture,
            base_url=base_url,
            stream_id="metadata-observation-abi",
            modality="metadata",
            media_type="application/json",
            schema="org.rivermark.benchmark.observation-abi.v1",
            relative=abi_ref["path"],
        )
        for stream in streams:
            if not isinstance(stream, Mapping):
                raise ReleaseBuildError(f"release episode has malformed stream: {episode_root}")
            if stream.get("partition") == "evaluator_private":
                raise ReleaseBuildError(f"evaluator-private stream reached release root: {episode_root}")
            if stream.get("partition") not in {"policy_visible", "learning_labels"}:
                raise ReleaseBuildError(f"unknown public stream partition: {episode_root}")
            stream_id = stream.get("stream_id")
            modality = stream.get("modality")
            if not isinstance(stream_id, str) or not isinstance(modality, str):
                raise ReleaseBuildError(f"stream lacks stable id/modality: {episode_root}")
            stream_schema = stream.get("schema")
            schema = stream_schema if isinstance(stream_schema, str) else _GENERIC_STREAM_SCHEMA
            if isinstance(stream.get("path"), str):
                _add_release_shard(
                    shards,
                    dataset_root=root,
                    episode_root=episode_root,
                    split=split,
                    episode_id=episode_id,
                    source_capture_sha256=source_capture,
                    base_url=base_url,
                    stream_id=stream_id,
                    modality=modality,
                    media_type=str(stream.get("media_type", "application/octet-stream")),
                    schema=schema,
                    relative=stream["path"],
                )
                continue
            template = stream.get("path_template")
            index_relative = stream.get("content_hash_index_path")
            if not isinstance(template, str) or not isinstance(index_relative, str):
                raise ReleaseBuildError(f"stream has no concrete payload binding: {episode_root}/{stream_id}")
            index = _read_object(episode_root / index_relative, label="content-hash index")
            files = index.get("files")
            if not isinstance(files, list):
                raise ReleaseBuildError(f"content-hash index files must be a list: {episode_root}/{index_relative}")
            _add_release_shard(
                shards,
                dataset_root=root,
                episode_root=episode_root,
                split=split,
                episode_id=episode_id,
                source_capture_sha256=source_capture,
                base_url=base_url,
                stream_id=stream_id + "-index",
                modality=modality + "__index",
                media_type="application/json",
                schema="org.rivermark.benchmark.content-hash-index.v1",
                relative=index_relative,
            )
            for entry in files:
                if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                    raise ReleaseBuildError(f"malformed content-hash entry: {episode_root}/{index_relative}")
                agent_id = entry.get("agent_id")
                if not isinstance(agent_id, int) or isinstance(agent_id, bool) or agent_id < 0:
                    raise ReleaseBuildError(f"invalid content-hash agent id: {episode_root}/{index_relative}")
                expected = template.replace("{agent_id}", str(agent_id))
                if entry["path"] != expected:
                    raise ReleaseBuildError(f"content-hash path does not match template: {episode_root}/{stream_id}")
                _add_release_shard(
                    shards,
                    dataset_root=root,
                    episode_root=episode_root,
                    split=split,
                    episode_id=episode_id,
                    source_capture_sha256=source_capture,
                    base_url=base_url,
                    stream_id=stream_id,
                    modality=modality,
                    media_type=str(stream.get("media_type", "application/octet-stream")),
                    schema=schema,
                    relative=entry["path"],
                    agent_id=agent_id,
                )
    if len(versions) != 1:
        raise ReleaseBuildError("formal release episodes must use one dataset_version")
    version = next(iter(versions))
    if dataset_version is not None and dataset_version != version:
        raise ReleaseBuildError(f"dataset_version {dataset_version!r} does not match episodes ({version!r})")
    payload: dict[str, Any] = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "dataset_version": version,
        "release_id": release_id,
        "license_status": "redistribution_cleared",
        "source_revision": source_revision,
        "supply_chain_manifest_sha256": supply_chain_report["manifest_sha256"],
        "accounting": accounting,
        "shards": sorted(shards.values(), key=lambda shard: shard["shard_id"]),
    }
    if created_at is not None:
        payload["created_at"] = created_at
    issues = validate_release_manifest(payload, require_https=require_https)
    if issues:
        raise ReleaseBuildError("generated release manifest failed validation: " + "; ".join(issue.code for issue in issues))
    if output_path is not None:
        output = output_path.resolve()
        if output.exists():
            raise ReleaseBuildError(f"refusing to overwrite existing release manifest: {output}")
        _write_json_atomic(output, payload)
    return payload


def load_release_manifest(path: Path, *, require_https: bool = False) -> dict[str, Any]:
    """Load and validate a release manifest, failing before any transfer."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"cannot read release manifest {path}: {exc}") from exc
    issues = validate_release_manifest(payload, require_https=require_https)
    if issues:
        formatted = "; ".join(f"{issue.code}:{issue.path}" for issue in issues)
        raise ReleaseManifestError(f"invalid release manifest {path}: {formatted}")
    return dict(payload)


def select_shards(
    payload: Mapping[str, Any],
    *,
    episode_ids: Iterable[str] = (),
    splits: Iterable[str] = (),
    modalities: Iterable[str] = (),
    agent_ids: Iterable[int] = (),
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Select complete pre-sharded frame ranges without changing ordering.

    Frame bounds use the half-open interval ``[frame_start, frame_end)``.
    A request only returns shards whose declared range is fully contained in
    the request.  It never slices or rewrites a shard, so a request that cuts
    through a shard fails closed in :func:`download_shards` rather than
    silently downloading extra frames.
    """

    requested_frame_range = _normalise_requested_frame_range(frame_start, frame_end)

    episodes = frozenset(episode_ids)
    requested_splits = frozenset(splits)
    requested_modalities = frozenset(modalities)
    requested_agents = frozenset(agent_ids)
    selected: list[Mapping[str, Any]] = []
    for shard in payload.get("shards", []):
        if episodes and shard.get("episode_id") not in episodes:
            continue
        if requested_splits and shard.get("split") not in requested_splits:
            continue
        if requested_modalities and shard.get("modality") not in requested_modalities:
            continue
        if requested_agents and shard.get("agent_id") not in requested_agents:
            continue
        if requested_frame_range is not None:
            shard_start = shard.get("frame_start")
            shard_end = shard.get("frame_end")
            if (
                not isinstance(shard_start, int)
                or isinstance(shard_start, bool)
                or not isinstance(shard_end, int)
                or isinstance(shard_end, bool)
                or shard_start < requested_frame_range[0]
                or shard_end > requested_frame_range[1]
            ):
                continue
        selected.append(shard)
    return tuple(selected)


def _normalise_requested_frame_range(
    frame_start: int | None,
    frame_end: int | None,
) -> tuple[int, int] | None:
    if frame_start is None and frame_end is None:
        return None
    if (
        not isinstance(frame_start, int)
        or isinstance(frame_start, bool)
        or not isinstance(frame_end, int)
        or isinstance(frame_end, bool)
        or frame_start < 0
        or frame_end <= frame_start
    ):
        raise ReleaseManifestError(
            "frame_start and frame_end must be non-negative integers with frame_end > frame_start"
        )
    return frame_start, frame_end


def _download_targets(
    payload: Mapping[str, Any],
    *,
    episode_ids: Iterable[str] = (),
    splits: Iterable[str] = (),
    modalities: Iterable[str] = (),
    agent_ids: Iterable[int] = (),
    frame_start: int | None = None,
    frame_end: int | None = None,
    include_accounting: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Resolve public shard targets once for planning and downloading."""

    selected = list(
        select_shards(
            payload,
            episode_ids=episode_ids,
            splits=splits,
            modalities=modalities,
            agent_ids=agent_ids,
            frame_start=frame_start,
            frame_end=frame_end,
        )
    )
    withdrawn_ids = {
        reference.get("shard_id")
        for defect in payload.get("defects", [])
        if isinstance(defect, Mapping) and defect.get("status") == "withdrawn"
        for reference in defect.get("affected_shards", [])
        if isinstance(reference, Mapping)
    }
    selected_withdrawn = [shard.get("shard_id") for shard in selected if shard.get("shard_id") in withdrawn_ids]
    if selected_withdrawn:
        raise ReleaseManifestError(
            "selection includes withdrawn shard(s); use the correction mapping and a newer release: "
            + ", ".join(sorted(str(value) for value in selected_withdrawn))
        )
    if include_accounting:
        accounting = payload.get("accounting")
        ledger = accounting.get("failure_ledger") if isinstance(accounting, Mapping) else None
        if not isinstance(ledger, Mapping):
            raise ReleaseManifestError("release manifest has no downloadable failure ledger accounting")
        selected.append({"shard_id": "release-failure-ledger", **ledger})
    if not selected:
        raise ReleaseManifestError("selection matched no release shards")
    return tuple(selected)


def plan_download(
    manifest_path: Path,
    *,
    episode_ids: Iterable[str] = (),
    splits: Iterable[str] = (),
    modalities: Iterable[str] = (),
    agent_ids: Iterable[int] = (),
    frame_start: int | None = None,
    frame_end: int | None = None,
    include_accounting: bool = False,
    require_https: bool = False,
) -> dict[str, Any]:
    """Return a byte-level download plan without creating a cache or fetching data.

    The plan uses only manifest-declared sizes and complete shard ranges. It is
    intentionally separate from :func:`download_shards` so a researcher can
    approve disk space before any network or local-file transfer begins.
    """

    episode_ids = tuple(episode_ids)
    splits = tuple(splits)
    modalities = tuple(modalities)
    agent_ids = tuple(agent_ids)
    payload = load_release_manifest(manifest_path, require_https=require_https)
    targets = _download_targets(
        payload,
        episode_ids=episode_ids,
        splits=splits,
        modalities=modalities,
        agent_ids=agent_ids,
        frame_start=frame_start,
        frame_end=frame_end,
        include_accounting=include_accounting,
    )
    public_fields = (
        "shard_id",
        "episode_id",
        "split",
        "modality",
        "agent_id",
        "frame_start",
        "frame_end",
        "path",
        "size_bytes",
        "sha256",
    )
    shards = [{field: shard[field] for field in public_fields if field in shard} for shard in targets]
    total_bytes = sum(int(shard["size_bytes"]) for shard in targets)
    return {
        "schema": DOWNLOAD_PLAN_SCHEMA,
        "status": "planned",
        "release_id": payload["release_id"],
        "dataset_version": payload["dataset_version"],
        "filters": {
            "episode_ids": sorted(set(episode_ids)),
            "splits": sorted(set(splits)),
            "modalities": sorted(set(modalities)),
            "agent_ids": sorted(set(agent_ids)),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "include_accounting": include_accounting,
        },
        "shard_count": len(shards),
        "total_bytes": total_bytes,
        "shards": shards,
    }


def verify_shard_file(shard: Mapping[str, Any], path: Path) -> None:
    """Verify one downloaded shard against its declared size and SHA-256."""

    if not path.is_file():
        raise DownloadError(f"downloaded shard is missing: {path}")
    expected_size = shard["size_bytes"]
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DownloadError(f"size mismatch for {shard['shard_id']}: expected {expected_size}, got {actual_size}")
    actual_hash = sha256_file(path)
    if actual_hash != shard["sha256"]:
        raise DownloadError(f"SHA-256 mismatch for {shard['shard_id']}: expected {shard['sha256']}, got {actual_hash}")


def _open_download(url: str, offset: int):
    headers = {"User-Agent": "rivermark-benchmark/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def _range_response_starts_at(response: Any, offset: int, expected_size: int) -> bool:
    """Require a compliant 206 response before appending to a partial shard."""

    if getattr(response, "status", None) != 206:
        return False
    header = response.headers.get("Content-Range") if getattr(response, "headers", None) is not None else None
    if not isinstance(header, str):
        return False
    match = _CONTENT_RANGE.fullmatch(header.strip())
    if match is None:
        return False
    start, end, total = match.groups()
    if int(start) != offset or int(end) != expected_size - 1:
        return False
    if total != "*" and int(total) != expected_size:
        return False
    return True


def _download_one(shard: Mapping[str, Any], destination: Path) -> DownloadResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_shard_file(shard, destination)
        return DownloadResult(shard["shard_id"], destination, "already_verified", destination.stat().st_size, shard["sha256"])
    partial = destination.with_name(destination.name + ".part")
    expected_size = int(shard["size_bytes"])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    elif offset == expected_size and partial.exists():
        try:
            verify_shard_file(shard, partial)
        except DownloadError:
            partial.unlink()
            offset = 0
        else:
            os.replace(partial, destination)
            return DownloadResult(shard["shard_id"], destination, "resumed_verified", expected_size, shard["sha256"])
    response = None
    try:
        response = _open_download(shard["url"], offset)
        append = offset > 0 and _range_response_starts_at(response, offset, expected_size)
        if offset > 0 and not append:
            # A proxy/server may ignore or mis-handle Range. Never append an
            # unverified body to a partial shard; restart from byte zero.
            response.close()
            response = _open_download(shard["url"], 0)
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        verify_shard_file(shard, partial)
        os.replace(partial, destination)
        return DownloadResult(shard["shard_id"], destination, "downloaded", expected_size, shard["sha256"])
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, DownloadError) as exc:
        if isinstance(exc, DownloadError) and partial.exists():
            partial.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {shard['shard_id']}: {exc}") from exc
    finally:
        if response is not None:
            response.close()


def download_shards(
    manifest_path: Path,
    destination: Path,
    *,
    episode_ids: Iterable[str] = (),
    splits: Iterable[str] = (),
    modalities: Iterable[str] = (),
    agent_ids: Iterable[int] = (),
    frame_start: int | None = None,
    frame_end: int | None = None,
    include_accounting: bool = False,
    require_https: bool = False,
) -> tuple[DownloadResult, ...]:
    """Download selected public shards sequentially and verify atomically.

    Frame selection operates on complete manifest-declared shards.  Arbitrary
    frame extraction is intentionally outside this byte-level downloader.
    """

    payload = load_release_manifest(manifest_path, require_https=require_https)
    targets = _download_targets(
        payload,
        episode_ids=episode_ids,
        splits=splits,
        modalities=modalities,
        agent_ids=agent_ids,
        frame_start=frame_start,
        frame_end=frame_end,
        include_accounting=include_accounting,
    )
    results: list[DownloadResult] = []
    root = destination.resolve()
    for shard in targets:
        relative = Path(shard["path"])
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            raise ReleaseManifestError(f"shard path escapes destination: {shard['path']}")
        results.append(_download_one(shard, target))
    return tuple(results)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="validate a release manifest without downloading")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--require-https", action="store_true")
    download = subparsers.add_parser("download", help="download and hash-verify selected shards")
    download.add_argument("manifest", type=Path)
    download.add_argument("destination", type=Path)
    download.add_argument("--episode", action="append", default=[])
    download.add_argument("--split", action="append", default=[])
    download.add_argument("--modality", action="append", default=[])
    download.add_argument("--agent-id", action="append", type=int, default=[])
    download.add_argument("--frame-start", type=int)
    download.add_argument("--frame-end", type=int)
    download.add_argument(
        "--include-accounting",
        action="store_true",
        help="also download the hash-bound public failure ledger",
    )
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="plan selected bytes and shards without creating a cache or fetching data",
    )
    download.add_argument("--require-https", action="store_true")
    build = subparsers.add_parser("build", help="build a release manifest from a verified formal dataset")
    build.add_argument("dataset_root", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--release-id", required=True)
    build.add_argument("--base-url", required=True)
    build.add_argument("--source-revision", required=True)
    build.add_argument("--dataset-version")
    build.add_argument("--created-at")
    build.add_argument(
        "--supply-chain-manifest",
        type=Path,
        required=True,
        help="cleared supply-chain manifest required for a public release",
    )
    build.add_argument(
        "--allow-non-https",
        action="store_true",
        help="allow http/file URLs for local fixtures; never use for a public release",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "verify":
            payload = load_release_manifest(args.manifest, require_https=args.require_https)
            print(json.dumps({"status": "valid", "shard_count": len(payload["shards"])}, indent=2))
            return 0
        if args.command == "build":
            payload = build_release_manifest(
                args.dataset_root,
                release_id=args.release_id,
                base_url=args.base_url,
                source_revision=args.source_revision,
                dataset_version=args.dataset_version,
                output_path=args.output,
                created_at=args.created_at,
                supply_chain_manifest=args.supply_chain_manifest,
                require_https=not args.allow_non_https,
            )
            print(json.dumps({"status": "built", "output": str(args.output), "shard_count": len(payload["shards"])}, indent=2))
            return 0
        if args.dry_run:
            plan = plan_download(
                args.manifest,
                episode_ids=args.episode,
                splits=args.split,
                modalities=args.modality,
                agent_ids=args.agent_id,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                include_accounting=args.include_accounting,
                require_https=args.require_https,
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        results = download_shards(
            args.manifest,
            args.destination,
            episode_ids=args.episode,
            splits=args.split,
            modalities=args.modality,
            agent_ids=args.agent_id,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            include_accounting=args.include_accounting,
            require_https=args.require_https,
        )
        print(json.dumps({"status": "downloaded", "shards": [asdict(result) for result in results]}, indent=2, default=str))
        return 0
    except (ReleaseManifestError, DownloadError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
