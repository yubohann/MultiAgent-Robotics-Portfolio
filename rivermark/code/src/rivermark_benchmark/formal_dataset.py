"""Fail-closed formal dataset admission, indexing, and integrity tooling.

This module deliberately does *not* turn a pilot recording into a formal
benchmark episode.  It accepts only a capture that already has a separately
produced formal-capture receipt, whose exact receipt hash has been approved by
the release operator.  The operator approval is the local trust root: a JSON
file can prove content integrity, but cannot prove that an untrusted process
actually captured data in Isaac Lab or on hardware.

The public release projection is intentionally smaller than a capture
directory.  It contains policy-visible data, eligible learning labels, and
opaque evaluator commitments, but never evaluator-private payloads.  Invalid
captures are recorded in ``quarantine/`` without deleting or modifying their
source directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .abi import observation_abi_sha256, validate_formal_observation_abi
from .collection_protocol import validate_collection_binding
from .failure_ledger import FailureRecord, append_failure_record
from .schema import is_safe_relative_path, is_sha256
from .supply_chain import (
    SupplyChainError,
    load_supply_chain_manifest,
    supply_chain_sha256,
    verify_supply_chain_manifest,
)
from .validate import validate_episode_manifest

FORMAL_CAPTURE_RECEIPT_SCHEMA = "org.rivermark.benchmark.formal-capture-receipt.v1"
LINEAGE_SCHEMA = "org.rivermark.benchmark.episode-lineage.v1"
CONTENT_HASH_INDEX_SCHEMA = "org.rivermark.benchmark.content-hash-index.v1"
RELEASE_ADMISSION_SCHEMA = "org.rivermark.benchmark.release-admission.v1"
DATASET_INDEX_SCHEMA = "org.rivermark.benchmark.dataset-index.v1"
SPLIT_AUTHORITY_SCHEMA = "org.rivermark.benchmark.split-authority.v1"
QUARANTINE_SCHEMA = "org.rivermark.benchmark.quarantine-record.v1"

FORMAL_SPLITS = frozenset({"train", "inner_dev", "validation", "blind_test", "ood_test"})
BLIND_SPLITS = frozenset({"blind_test", "ood_test"})
LINEAGE_AXES = (
    "layout_lineage",
    "task_manifest",
    "episode",
    "appearance_domain",
    "dynamics_domain",
    "instruction_family",
    "instruction_annotator",
    "trajectory_lineage",
    "asset_lineage",
    "behavior_policy_checkpoint_family",
)
_GROUPING_AXES = tuple(axis for axis in LINEAGE_AXES if axis != "episode")
_RECEIPT_PATH = "formal_capture_receipt.json"
_LINEAGE_PATH = "lineage.json"
_ADMISSION_PATH = "admission.json"
_MANIFEST_PATH = "episode_manifest.json"
_SAFE_NAME = re.compile(r"[^a-z0-9._-]+")
_RELEASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_REQUIRED_RELEASE_ASSET_KINDS = frozenset({"code", "scene_layer", "robot_asset", "data"})
_RESERVED_RELEASE_PATH_PARTS = frozenset(
    {
        "evaluator_private",
        "evaluator-private",
        "hidden",
        "hidden_truth",
        "private",
        "target_truth",
    }
)


@dataclass(frozen=True)
class DatasetIssue:
    """A non-sensitive reason why a candidate or release is invalid."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class CandidateIntegrityReport:
    episode_root: Path
    episode_id: str | None
    manifest: Mapping[str, Any] | None
    lineage: Mapping[str, Any] | None
    manifest_sha256: str | None
    lineage_sha256: str | None
    receipt_sha256: str | None
    issues: tuple[DatasetIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CollectionResult:
    admitted: bool
    episode_root: Path | None
    quarantine_record: Path | None
    issues: tuple[DatasetIssue, ...]


@dataclass(frozen=True)
class DatasetIntegrityReport:
    dataset_root: Path
    episode_count: int
    issues: tuple[DatasetIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write one canonical JSON file without exposing a partial final file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _issue(issues: list[DatasetIssue], code: str, path: str, message: str) -> None:
    issues.append(DatasetIssue(code=code, path=path, message=message))


def _read_json(path: Path, *, issues: list[DatasetIssue], issue_path: str) -> Mapping[str, Any] | None:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _issue(issues, "invalid_json", issue_path, str(exc))
        return None
    if not isinstance(value, Mapping):
        _issue(issues, "json_type", issue_path, "expected a JSON object")
        return None
    return value


def _canonical_relative(value: object) -> str | None:
    if not is_safe_relative_path(value):
        return None
    return PurePosixPath(str(value).replace("\\", "/")).as_posix()


def _contained_file(root: Path, relative: object) -> Path | None:
    canonical = _canonical_relative(relative)
    if canonical is None:
        return None
    try:
        resolved_root = root.resolve()
        candidate = (resolved_root / canonical).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            return None
        return candidate
    except OSError:
        return None


def _path_has_reserved_partition(value: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(value).parts}
    return bool(parts & _RESERVED_RELEASE_PATH_PARTS)


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    path: str,
    issues: list[DatasetIssue],
) -> None:
    expected_set = set(expected)
    for key in value:
        if key not in expected_set:
            _issue(issues, "unknown_field", f"{path}.{key}", "field is not permitted by this contract")
    for key in expected_set:
        if key not in value:
            _issue(issues, "required", f"{path}.{key}", "required field is missing")


def _nonempty_string(value: object, *, maximum: int = 256) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum


def _validate_lineage(
    lineage: Mapping[str, Any] | None,
    *,
    manifest: Mapping[str, Any] | None,
    path: str,
    issues: list[DatasetIssue],
) -> None:
    if lineage is None:
        return
    _expect_exact_keys(lineage, {"schema", "episode_id", "axes"}, path=path, issues=issues)
    if lineage.get("schema") != LINEAGE_SCHEMA:
        _issue(issues, "lineage_schema", f"{path}.schema", f"expected {LINEAGE_SCHEMA!r}")
    episode_id = lineage.get("episode_id")
    if not _nonempty_string(episode_id, maximum=128):
        _issue(issues, "lineage_episode_id", f"{path}.episode_id", "must be a non-empty identifier")
    axes = lineage.get("axes")
    if not isinstance(axes, Mapping):
        _issue(issues, "lineage_axes", f"{path}.axes", "must be an object of opaque SHA-256 values")
        return
    _expect_exact_keys(axes, LINEAGE_AXES, path=f"{path}.axes", issues=issues)
    for axis in LINEAGE_AXES:
        if not is_sha256(axes.get(axis)):
            _issue(issues, "lineage_axis", f"{path}.axes.{axis}", "must be a SHA-256 commitment")
    if manifest is None:
        return
    manifest_id = manifest.get("episode_id")
    if episode_id != manifest_id:
        _issue(issues, "lineage_episode_mismatch", f"{path}.episode_id", "does not match episode manifest")
    layout = manifest.get("layout")
    task = manifest.get("task")
    if isinstance(layout, Mapping) and axes.get("layout_lineage") != layout.get("layout_lineage_hash"):
        _issue(
            issues,
            "lineage_layout_mismatch",
            f"{path}.axes.layout_lineage",
            "must equal layout.layout_lineage_hash",
        )
    if isinstance(task, Mapping) and axes.get("task_manifest") != task.get("task_spec_sha256"):
        _issue(
            issues,
            "lineage_task_mismatch",
            f"{path}.axes.task_manifest",
            "must equal task.task_spec_sha256",
        )
    if isinstance(manifest_id, str) and axes.get("episode") != _sha256_text(manifest_id):
        _issue(
            issues,
            "lineage_episode_hash",
            f"{path}.axes.episode",
            "must be SHA-256 of the public episode_id",
        )


def _validate_capture_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    manifest_sha256: str | None,
    lineage_sha256: str | None,
    manifest: Mapping[str, Any] | None,
    path: str,
    issues: list[DatasetIssue],
) -> None:
    if receipt is None:
        return
    receipt_keys = {
        "schema",
        "status",
        "formal_benchmark_admission",
        "episode_manifest_sha256",
        "lineage_sha256",
        "observation_abi_sha256",
        "capture_backend",
        "integrity",
        "partitions",
    }
    if "collection_binding" in receipt:
        receipt_keys.add("collection_binding")
    _expect_exact_keys(
        receipt,
        receipt_keys,
        path=path,
        issues=issues,
    )
    if receipt.get("schema") != FORMAL_CAPTURE_RECEIPT_SCHEMA:
        _issue(issues, "receipt_schema", f"{path}.schema", f"expected {FORMAL_CAPTURE_RECEIPT_SCHEMA!r}")
    if receipt.get("status") != "admitted":
        _issue(issues, "receipt_status", f"{path}.status", "formal receipt status must be 'admitted'")
    if receipt.get("formal_benchmark_admission") is not True:
        _issue(issues, "formal_admission", f"{path}.formal_benchmark_admission", "must be true")
    if not is_sha256(receipt.get("episode_manifest_sha256")):
        _issue(issues, "receipt_manifest_hash", f"{path}.episode_manifest_sha256", "must be SHA-256")
    elif manifest_sha256 is not None and receipt.get("episode_manifest_sha256") != manifest_sha256:
        _issue(issues, "receipt_manifest_mismatch", f"{path}.episode_manifest_sha256", "does not bind this manifest")
    if not is_sha256(receipt.get("lineage_sha256")):
        _issue(issues, "receipt_lineage_hash", f"{path}.lineage_sha256", "must be SHA-256")
    elif lineage_sha256 is not None and receipt.get("lineage_sha256") != lineage_sha256:
        _issue(issues, "receipt_lineage_mismatch", f"{path}.lineage_sha256", "does not bind this lineage")
    abi_hash = receipt.get("observation_abi_sha256")
    if not is_sha256(abi_hash):
        _issue(issues, "receipt_abi_hash", f"{path}.observation_abi_sha256", "must be SHA-256")
    elif isinstance(manifest, Mapping):
        abi_ref = manifest.get("observation_abi")
        if isinstance(abi_ref, Mapping) and abi_hash != abi_ref.get("sha256"):
            _issue(
                issues,
                "receipt_abi_mismatch",
                f"{path}.observation_abi_sha256",
                "does not bind the manifest observation ABI",
            )

    collection_binding = receipt.get("collection_binding")
    manifest_binding = manifest.get("collection_binding") if isinstance(manifest, Mapping) else None
    if collection_binding is not None:
        for binding_issue in validate_collection_binding(collection_binding):
            suffix = binding_issue.path.lstrip("$").lstrip(".")
            _issue(
                issues,
                f"receipt_collection_{binding_issue.code}",
                f"{path}.collection_binding.{suffix}" if suffix else f"{path}.collection_binding",
                binding_issue.message,
            )
    if collection_binding != manifest_binding:
        _issue(
            issues,
            "receipt_collection_mismatch",
            f"{path}.collection_binding",
            "must exactly match the episode manifest collection binding",
        )

    backend = receipt.get("capture_backend")
    if not isinstance(backend, Mapping):
        _issue(issues, "receipt_backend", f"{path}.capture_backend", "must be an object")
    else:
        _expect_exact_keys(
            backend,
            {"kind", "build", "sensor_physics_smoke_receipt_sha256"},
            path=f"{path}.capture_backend",
            issues=issues,
        )
        if backend.get("kind") not in {"isaaclab", "hardware"}:
            _issue(issues, "receipt_backend", f"{path}.capture_backend.kind", "only isaaclab or hardware is admissible")
        if not _nonempty_string(backend.get("build")):
            _issue(issues, "receipt_backend", f"{path}.capture_backend.build", "must be a non-empty build identifier")
        if not is_sha256(backend.get("sensor_physics_smoke_receipt_sha256")):
            _issue(
                issues,
                "receipt_backend",
                f"{path}.capture_backend.sensor_physics_smoke_receipt_sha256",
                "must be SHA-256",
            )

    integrity = receipt.get("integrity")
    integrity_keys = {
        "online_capture",
        "queue_overflow",
        "silent_frame_drop",
        "timestamp_audit_passed",
        "pose_closure_audit_passed",
        "action_causality_audit_passed",
        "sensor_decode_audit_passed",
        "policy_leakage_audit_passed",
        "independent_validator_id",
        "independent_validator_sha256",
        "pose_closure_threshold_m",
    }
    if collection_binding is not None:
        integrity_keys.add("condition_realization_verified")
    if not isinstance(integrity, Mapping):
        _issue(issues, "receipt_integrity", f"{path}.integrity", "must be an object")
    else:
        _expect_exact_keys(integrity, integrity_keys, path=f"{path}.integrity", issues=issues)
        for key in (
            "online_capture",
            "timestamp_audit_passed",
            "pose_closure_audit_passed",
            "action_causality_audit_passed",
            "sensor_decode_audit_passed",
            "policy_leakage_audit_passed",
        ):
            if integrity.get(key) is not True:
                _issue(issues, "receipt_integrity", f"{path}.integrity.{key}", "must be true")
        for key in ("queue_overflow", "silent_frame_drop"):
            if integrity.get(key) is not False:
                _issue(issues, "receipt_integrity", f"{path}.integrity.{key}", "must be false")
        if not _nonempty_string(integrity.get("independent_validator_id")):
            _issue(issues, "receipt_integrity", f"{path}.integrity.independent_validator_id", "must be non-empty")
        if not is_sha256(integrity.get("independent_validator_sha256")):
            _issue(issues, "receipt_integrity", f"{path}.integrity.independent_validator_sha256", "must be SHA-256")
        if collection_binding is not None and integrity.get("condition_realization_verified") is not True:
            _issue(
                issues,
                "receipt_condition_realization",
                f"{path}.integrity.condition_realization_verified",
                "collection-bound formal receipts require independent condition realization",
            )
        threshold = integrity.get("pose_closure_threshold_m")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or threshold <= 0.0
        ):
            _issue(issues, "receipt_integrity", f"{path}.integrity.pose_closure_threshold_m", "must be finite and positive")
        elif isinstance(manifest, Mapping):
            quality = manifest.get("quality")
            if isinstance(quality, Mapping):
                closure = quality.get("pose_closure_max_error_m")
                if isinstance(closure, (int, float)) and not isinstance(closure, bool) and closure > threshold:
                    _issue(
                        issues,
                        "pose_closure_threshold",
                        "$.quality.pose_closure_max_error_m",
                        "manifest pose closure exceeds the independently attested threshold",
                    )

    partitions = receipt.get("partitions")
    partition_keys = {
        "policy_visible_audit_sha256",
        "learning_labels_release_allowed",
        "evaluator_private_distributed",
        "evaluator_private_server_only",
    }
    if not isinstance(partitions, Mapping):
        _issue(issues, "receipt_partitions", f"{path}.partitions", "must be an object")
    else:
        _expect_exact_keys(partitions, partition_keys, path=f"{path}.partitions", issues=issues)
        if not is_sha256(partitions.get("policy_visible_audit_sha256")):
            _issue(issues, "receipt_partitions", f"{path}.partitions.policy_visible_audit_sha256", "must be SHA-256")
        for key, expected in (
            ("learning_labels_release_allowed", None),
            ("evaluator_private_distributed", False),
            ("evaluator_private_server_only", True),
        ):
            value = partitions.get(key)
            if not isinstance(value, bool):
                _issue(issues, "receipt_partitions", f"{path}.partitions.{key}", "must be boolean")
            elif expected is not None and value is not expected:
                _issue(issues, "receipt_partitions", f"{path}.partitions.{key}", f"must be {expected}")
        if isinstance(manifest, Mapping):
            learning = manifest.get("learning_labels")
            private = manifest.get("evaluator_private")
            if isinstance(learning, Mapping) and partitions.get("learning_labels_release_allowed") != learning.get("distributed"):
                _issue(
                    issues,
                    "receipt_partition_mismatch",
                    f"{path}.partitions.learning_labels_release_allowed",
                    "must match manifest learning_labels.distributed",
                )
            if isinstance(private, Mapping) and (
                private.get("distributed") is not False or private.get("server_only") is not True
            ):
                _issue(issues, "private_distribution", "$.evaluator_private", "must be server-only and not distributed")


def _validate_formal_manifest_rules(manifest: Mapping[str, Any], issues: list[DatasetIssue]) -> None:
    """Apply the stricter admission rules not expressible in manifest v1 alone."""

    split = manifest.get("split")
    if split not in FORMAL_SPLITS:
        _issue(issues, "formal_split", "$.split", "formal admission excludes the pilot split")
    abi_ref = manifest.get("observation_abi")
    if not isinstance(abi_ref, Mapping):
        _issue(issues, "observation_abi_required", "$.observation_abi", "formal admission requires a bound observation ABI")
    elif not is_sha256(abi_ref.get("sha256")) or _canonical_relative(abi_ref.get("path")) is None:
        _issue(issues, "observation_abi_required", "$.observation_abi", "formal admission requires a safe path and SHA-256 ABI binding")
    quality = manifest.get("quality")
    if not isinstance(quality, Mapping):
        return
    if quality.get("recording_valid") is not True:
        _issue(issues, "recording_invalid", "$.quality.recording_valid", "formal admission requires a valid recording")
    if quality.get("timestamp_monotonic") is not True:
        _issue(issues, "timestamp_monotonic", "$.quality.timestamp_monotonic", "formal admission requires monotonic timestamps")
    if quality.get("frame_completeness_ratio") != 1.0:
        _issue(
            issues,
            "frame_completeness",
            "$.quality.frame_completeness_ratio",
            "v1 formal admission currently requires a complete, non-interpolated capture",
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("scene_asset_license_status") != "redistribution_cleared":
        _issue(
            issues,
            "asset_license",
            "$.provenance.scene_asset_license_status",
            "public dataset admission requires redistribution_cleared assets",
        )

    streams = manifest.get("streams")
    policy = manifest.get("policy_visible")
    learning = manifest.get("learning_labels")
    if not isinstance(streams, list):
        return
    policy_modalities = set(policy.get("modalities", ())) if isinstance(policy, Mapping) else set()
    policy_streams = {
        stream.get("modality")
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("partition") == "policy_visible"
    }
    missing_policy_streams = sorted(modality for modality in policy_modalities if modality not in policy_streams)
    if missing_policy_streams:
        _issue(
            issues,
            "missing_policy_stream",
            "$.streams",
            f"every declared policy modality needs a bound stream: {missing_policy_streams}",
        )
    private_streams = [
        stream.get("stream_id")
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("partition") == "evaluator_private"
    ]
    if private_streams:
        _issue(
            issues,
            "evaluator_private_stream",
            "$.streams",
            "evaluator-private payloads must stay outside an admissible capture directory",
        )
    label_modalities = set(learning.get("modalities", ())) if isinstance(learning, Mapping) else set()
    label_streams = {
        stream.get("modality")
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("partition") == "learning_labels"
    }
    labels_distributed = learning.get("distributed") if isinstance(learning, Mapping) else None
    if labels_distributed is True and not label_modalities.issubset(label_streams):
        _issue(
            issues,
            "missing_learning_label_stream",
            "$.streams",
            "every distributed learning-label modality needs a bound stream",
        )
    if split in BLIND_SPLITS and (labels_distributed is not False or label_modalities or label_streams):
        _issue(
            issues,
            "blind_learning_labels",
            "$.learning_labels",
            "blind/OOD release candidates cannot carry learning labels",
        )


def _content_index_files(
    root: Path,
    stream: Mapping[str, Any],
    *,
    agent_count: int,
    issues: list[DatasetIssue],
    issue_path: str,
) -> dict[str, str]:
    """Validate a template binding and return every payload path/hash it covers."""

    index_relative = _canonical_relative(stream.get("content_hash_index_path"))
    expected_index_hash = stream.get("content_hash_index_sha256")
    if index_relative is None or not is_sha256(expected_index_hash):
        return {}
    index_path = _contained_file(root, index_relative)
    if index_path is None:
        _issue(issues, "missing_file", f"{issue_path}.content_hash_index_path", "content-hash index is missing")
        return {}
    if sha256_file(index_path) != expected_index_hash:
        _issue(issues, "file_hash", f"{issue_path}.content_hash_index_path", "content-hash index digest does not match")
        return {}
    index = _read_json(index_path, issues=issues, issue_path=f"{issue_path}.content_hash_index")
    if index is None:
        return {}
    _expect_exact_keys(index, {"schema", "stream_id", "files"}, path=f"{issue_path}.content_hash_index", issues=issues)
    if index.get("schema") != CONTENT_HASH_INDEX_SCHEMA:
        _issue(issues, "content_index_schema", f"{issue_path}.content_hash_index.schema", "unsupported content-hash index schema")
    if index.get("stream_id") != stream.get("stream_id"):
        _issue(issues, "content_index_stream", f"{issue_path}.content_hash_index.stream_id", "does not bind this stream")
    files = index.get("files")
    if not isinstance(files, list):
        _issue(issues, "content_index_files", f"{issue_path}.content_hash_index.files", "must be a list")
        return {}
    template = _canonical_relative(stream.get("path_template"))
    if template is None:
        return {}
    result: dict[str, str] = {index_relative: str(expected_index_hash)}
    seen_agents: set[int] = set()
    for file_index, entry in enumerate(files):
        entry_path = f"{issue_path}.content_hash_index.files[{file_index}]"
        if not isinstance(entry, Mapping):
            _issue(issues, "content_index_entry", entry_path, "must be an object")
            continue
        _expect_exact_keys(entry, {"agent_id", "path", "sha256"}, path=entry_path, issues=issues)
        agent_id = entry.get("agent_id")
        if not isinstance(agent_id, int) or isinstance(agent_id, bool) or not 0 <= agent_id < agent_count:
            _issue(issues, "content_index_agent", f"{entry_path}.agent_id", "agent id is outside the task range")
            continue
        if agent_id in seen_agents:
            _issue(issues, "content_index_agent", f"{entry_path}.agent_id", "agent id occurs more than once")
            continue
        seen_agents.add(agent_id)
        relative = _canonical_relative(entry.get("path"))
        if relative is None:
            _issue(issues, "unsafe_path", f"{entry_path}.path", "payload path is unsafe")
            continue
        expected = template.replace("{agent_id}", str(agent_id))
        if relative != expected:
            _issue(issues, "content_index_path", f"{entry_path}.path", "does not match stream path_template")
            continue
        digest = entry.get("sha256")
        if not is_sha256(digest):
            _issue(issues, "sha256", f"{entry_path}.sha256", "must be SHA-256")
            continue
        payload = _contained_file(root, relative)
        if payload is None:
            _issue(issues, "missing_file", f"{entry_path}.path", "template payload is missing")
            continue
        if sha256_file(payload) != digest:
            _issue(issues, "file_hash", f"{entry_path}.path", "template payload digest does not match")
            continue
        previous = result.setdefault(relative, str(digest))
        if previous != digest:
            _issue(issues, "path_hash_conflict", f"{entry_path}.path", "one path has incompatible digests")
    expected_agents = set(range(agent_count))
    if seen_agents != expected_agents:
        _issue(
            issues,
            "content_index_coverage",
            f"{issue_path}.content_hash_index.files",
            "template bindings must cover every agent exactly once",
        )
    return result


def _bound_files(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    include_learning_labels: bool,
    issues: list[DatasetIssue],
) -> dict[str, str]:
    """Return all public release files referenced by a manifest, with hashes."""

    result: dict[str, str] = {}

    def bind(relative_value: object, digest: object, path: str) -> None:
        relative = _canonical_relative(relative_value)
        if relative is None:
            _issue(issues, "unsafe_path", path, "release path is unsafe")
            return
        if _path_has_reserved_partition(relative):
            _issue(issues, "reserved_release_path", path, "public release path uses a private partition name")
            return
        if not is_sha256(digest):
            _issue(issues, "sha256", path, "release binding requires SHA-256")
            return
        file_path = _contained_file(root, relative)
        if file_path is None:
            _issue(issues, "missing_file", path, "bound file is missing")
            return
        actual = sha256_file(file_path)
        if actual != digest:
            _issue(issues, "file_hash", path, "bound file hash does not match")
            return
        previous = result.setdefault(relative, str(digest))
        if previous != digest:
            _issue(issues, "path_hash_conflict", path, "one path has incompatible digests")

    def bind_observation_abi(relative_value: object, digest: object, path: str) -> None:
        relative = _canonical_relative(relative_value)
        if relative is None:
            _issue(issues, "unsafe_path", path, "release path is unsafe")
            return
        if _path_has_reserved_partition(relative):
            _issue(issues, "reserved_release_path", path, "public release path uses a private partition name")
            return
        if not is_sha256(digest):
            _issue(issues, "sha256", path, "observation ABI binding requires SHA-256")
            return
        file_path = _contained_file(root, relative)
        if file_path is None:
            _issue(issues, "missing_file", path, "observation ABI file is missing")
            return
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _issue(issues, "abi_json", path, f"cannot read observation ABI: {exc}")
            return
        abi_issues = validate_formal_observation_abi(payload)
        for abi_issue in abi_issues:
            _issue(issues, f"abi_{abi_issue.code}", f"{path}.{abi_issue.path.lstrip('$').lstrip('.')}", abi_issue.message)
        if abi_issues:
            return
        try:
            actual = observation_abi_sha256(payload)
        except Exception as exc:  # the structural validation above should make this unreachable
            _issue(issues, "abi_hash", path, f"cannot canonicalize observation ABI: {exc}")
            return
        if actual != digest:
            _issue(issues, "abi_hash", f"{path}.sha256", "canonical observation ABI hash does not match")
            return
        actual_file_hash = sha256_file(file_path)
        previous = result.setdefault(relative, actual_file_hash)
        if previous != actual_file_hash:
            _issue(issues, "path_hash_conflict", path, "one path has incompatible file digests")

    observation_abi = manifest.get("observation_abi")
    if isinstance(observation_abi, Mapping):
        bind_observation_abi(observation_abi.get("path"), observation_abi.get("sha256"), "$.observation_abi.path")
    layout = manifest.get("layout")
    if isinstance(layout, Mapping):
        bind(layout.get("scene_manifest_ref"), layout.get("scene_manifest_sha256"), "$.layout.scene_manifest_ref")
    task = manifest.get("task")
    if isinstance(task, Mapping):
        bind(task.get("task_spec_ref"), task.get("task_spec_sha256"), "$.task.task_spec_ref")
        agent_count = task.get("agent_count")
    else:
        agent_count = None
    if not isinstance(agent_count, int) or isinstance(agent_count, bool) or agent_count < 1:
        _issue(issues, "agent_count", "$.task.agent_count", "cannot bind per-agent streams without a valid agent count")
        return result
    streams = manifest.get("streams")
    if not isinstance(streams, list):
        return result
    for stream_index, raw_stream in enumerate(streams):
        stream_path = f"$.streams[{stream_index}]"
        if not isinstance(raw_stream, Mapping):
            continue
        partition = raw_stream.get("partition")
        if partition == "evaluator_private":
            _issue(issues, "evaluator_private_stream", stream_path, "private stream cannot be in a release projection")
            continue
        if partition == "learning_labels" and not include_learning_labels:
            continue
        if partition not in {"policy_visible", "learning_labels"}:
            continue
        if "path" in raw_stream or "sha256" in raw_stream:
            bind(raw_stream.get("path"), raw_stream.get("sha256"), f"{stream_path}.path")
        else:
            files = _content_index_files(
                root,
                raw_stream,
                agent_count=agent_count,
                issues=issues,
                issue_path=stream_path,
            )
            for relative, digest in files.items():
                if _path_has_reserved_partition(relative):
                    _issue(issues, "reserved_release_path", stream_path, "public release path uses a private partition name")
                    continue
                previous = result.setdefault(relative, digest)
                if previous != digest:
                    _issue(issues, "path_hash_conflict", stream_path, "one path has incompatible digests")
    return result


def verify_candidate_episode(
    episode_root: Path,
    *,
    trusted_receipt_hashes: Iterable[str] = (),
    require_trusted_receipt: bool = True,
) -> CandidateIntegrityReport:
    """Verify a source capture before it can be projected into a release.

    ``trusted_receipt_hashes`` is intentionally explicit.  A valid-looking
    receipt without an operator-approved digest is quarantined rather than
    being promoted into a benchmark release.
    """

    root = episode_root.resolve()
    issues: list[DatasetIssue] = []
    manifest_path = root / _MANIFEST_PATH
    lineage_path = root / _LINEAGE_PATH
    receipt_path = root / _RECEIPT_PATH
    manifest = _read_json(manifest_path, issues=issues, issue_path=_MANIFEST_PATH)
    lineage = _read_json(lineage_path, issues=issues, issue_path=_LINEAGE_PATH)
    receipt = _read_json(receipt_path, issues=issues, issue_path=_RECEIPT_PATH)
    manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    lineage_hash = sha256_file(lineage_path) if lineage_path.is_file() else None
    receipt_hash = sha256_file(receipt_path) if receipt_path.is_file() else None
    if manifest is not None:
        for issue in validate_episode_manifest(manifest, base_dir=root, check_files=True):
            _issue(issues, f"manifest_{issue.code}", issue.path, issue.message)
        _validate_formal_manifest_rules(manifest, issues)
    _validate_lineage(lineage, manifest=manifest, path="$lineage", issues=issues)
    _validate_capture_receipt(
        receipt,
        manifest_sha256=manifest_hash,
        lineage_sha256=lineage_hash,
        manifest=manifest,
        path="$formal_capture_receipt",
        issues=issues,
    )
    trusted = set(trusted_receipt_hashes)
    if require_trusted_receipt:
        if receipt_hash is None or receipt_hash not in trusted:
            _issue(
                issues,
                "untrusted_capture_receipt",
                _RECEIPT_PATH,
                "formal receipt hash is not in the explicit operator allowlist",
            )
    if manifest is not None:
        # Source captures may retain non-distributed learning labels for an
        # internal training workflow.  They must still be manifest-bound and
        # pass integrity checks; only the public projection omits them.
        bound = _bound_files(root, manifest, include_learning_labels=True, issues=issues)
        _validate_file_inventory(
            root,
            expected_files=set(bound) | {_MANIFEST_PATH, _LINEAGE_PATH, _RECEIPT_PATH},
            scope="candidate",
            issues=issues,
        )
    episode_id = manifest.get("episode_id") if isinstance(manifest, Mapping) and isinstance(manifest.get("episode_id"), str) else None
    return CandidateIntegrityReport(
        episode_root=root,
        episode_id=episode_id,
        manifest=manifest,
        lineage=lineage,
        manifest_sha256=manifest_hash,
        lineage_sha256=lineage_hash,
        receipt_sha256=receipt_hash,
        issues=tuple(issues),
    )


def _project_manifest_for_release(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Drop non-distributed learning payload references from a public release."""

    projected = copy.deepcopy(dict(manifest))
    learning = projected.get("learning_labels")
    split = projected.get("split")
    include_labels = isinstance(learning, Mapping) and learning.get("distributed") is True and split not in BLIND_SPLITS
    streams = projected.get("streams")
    if isinstance(streams, list):
        projected["streams"] = [
            stream
            for stream in streams
            if isinstance(stream, Mapping)
            and stream.get("partition") != "evaluator_private"
            and (stream.get("partition") != "learning_labels" or include_labels)
        ]
    if not include_labels:
        projected["learning_labels"] = {"distributed": False, "modalities": []}
    return projected


def _safe_copy(source_root: Path, relative: str, destination_root: Path) -> None:
    source = _contained_file(source_root, relative)
    if source is None:
        raise RuntimeError(f"refusing to copy unavailable source file {relative!r}")
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _safe_episode_label(value: str | None, fallback: str) -> str:
    raw = value if isinstance(value, str) and value else fallback
    result = _SAFE_NAME.sub("-", raw.lower()).strip(".-")
    return result[:96] or "unknown-episode"


def quarantine_candidate(
    dataset_root: Path,
    candidate_root: Path,
    issues: Iterable[DatasetIssue],
    *,
    episode_id: str | None = None,
    manifest_sha256: str | None = None,
    receipt_sha256: str | None = None,
) -> Path:
    """Retain an immutable, non-sensitive failure record without moving data."""

    normalized_issues = tuple(issues)
    fingerprint = hashlib.sha256(
        _canonical_json_bytes(
            {
                "episode_id": episode_id,
                "manifest_sha256": manifest_sha256,
                "receipt_sha256": receipt_sha256,
                "issues": [asdict(issue) for issue in normalized_issues],
            }
        )
    ).hexdigest()
    label = _safe_episode_label(episode_id, candidate_root.name)
    record_path = dataset_root.resolve() / "quarantine" / f"{label}-{fingerprint[:16]}.json"
    payload = {
        "schema": QUARANTINE_SCHEMA,
        "episode_id": episode_id,
        "candidate_directory_name": _safe_episode_label(candidate_root.name, "candidate"),
        "source_retained": True,
        "source_manifest_sha256": manifest_sha256,
        "source_capture_receipt_sha256": receipt_sha256,
        "reasons": [asdict(issue) for issue in normalized_issues],
    }
    if record_path.exists():
        existing_issues: list[DatasetIssue] = []
        existing = _read_json(record_path, issues=existing_issues, issue_path=str(record_path))
        if existing != payload:
            raise RuntimeError(f"quarantine record collision: {record_path}")
        return record_path
    _write_json_atomic(record_path, payload)
    return record_path


def _lineage_axes(lineage: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    if not isinstance(lineage, Mapping):
        return None
    axes = lineage.get("axes")
    if not isinstance(axes, Mapping) or not all(is_sha256(axes.get(axis)) for axis in LINEAGE_AXES):
        return None
    return {axis: str(axes[axis]) for axis in LINEAGE_AXES}


def _split_conflicts(entries: Sequence[tuple[str, str, Mapping[str, str]]]) -> tuple[DatasetIssue, ...]:
    """Find any split overlap using transitive lineage grouping across all axes."""

    issues: list[DatasetIssue] = []
    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seen: dict[tuple[str, str], int] = {}
    for index, (_, _, axes) in enumerate(entries):
        for axis in _GROUPING_AXES:
            marker = (axis, axes[axis])
            previous = seen.get(marker)
            if previous is None:
                seen[marker] = index
            else:
                union(previous, index)
    groups: dict[int, list[int]] = {}
    for index in range(len(entries)):
        groups.setdefault(find(index), []).append(index)
    for member_indices in groups.values():
        splits = {entries[index][1] for index in member_indices}
        if len(splits) > 1:
            episode_ids = sorted(entries[index][0] for index in member_indices)
            _issue(
                issues,
                "split_lineage_overlap",
                "$.lineage",
                "connected lineage group spans multiple splits: " + ", ".join(episode_ids),
            )
    trajectories: dict[str, str] = {}
    for episode_id, _, axes in entries:
        trajectory = axes["trajectory_lineage"]
        previous = trajectories.setdefault(trajectory, episode_id)
        if previous != episode_id:
            _issue(
                issues,
                "duplicate_trajectory_lineage",
                "$.lineage.axes.trajectory_lineage",
                f"episodes {previous!r} and {episode_id!r} declare the same trajectory lineage",
            )
    return tuple(issues)


def _split_authority_payload(entries: Sequence[tuple[str, str, Mapping[str, str]]], dataset_version: str) -> dict[str, Any]:
    """Build a deterministic, opaque split authority from validated entries."""

    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seen: dict[tuple[str, str], int] = {}
    for index, (_, _, axes) in enumerate(entries):
        for axis in _GROUPING_AXES:
            key = (axis, axes[axis])
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(entries)):
        groups.setdefault(find(index), []).append(index)
    authority_groups: list[dict[str, Any]] = []
    for indices in groups.values():
        episode_ids = sorted(entries[index][0] for index in indices)
        split = entries[indices[0]][1]
        group_id = hashlib.sha256(_canonical_json_bytes({"episode_ids": episode_ids, "split": split})).hexdigest()
        authority_groups.append({"group_id": group_id, "split": split, "episode_ids": episode_ids})
    return {
        "schema": SPLIT_AUTHORITY_SCHEMA,
        "dataset_version": dataset_version,
        "grouping_axes": list(_GROUPING_AXES),
        "groups": sorted(authority_groups, key=lambda group: (group["split"], group["group_id"])),
    }


def plan_split_authority(candidate_roots: Iterable[Path]) -> tuple[dict[str, Any] | None, tuple[DatasetIssue, ...]]:
    """Validate predeclared split assignments before formal collection.

    Split assignment is intentionally not mutated after capture: changing
    ``manifest.split`` would invalidate the independently bound receipt.  This
    function checks the assignments encoded in candidate manifests and emits a
    deterministic authority document only if all lineage groups are disjoint.
    """

    reports = [
        verify_candidate_episode(root, require_trusted_receipt=False)
        for root in candidate_roots
    ]
    issues = [issue for report in reports for issue in report.issues]
    entries: list[tuple[str, str, Mapping[str, str]]] = []
    versions: set[str] = set()
    for report in reports:
        if report.manifest is None or report.episode_id is None:
            continue
        axes = _lineage_axes(report.lineage)
        split = report.manifest.get("split")
        version = report.manifest.get("dataset_version")
        if axes is None or not isinstance(split, str) or not isinstance(version, str):
            continue
        entries.append((report.episode_id, split, axes))
        versions.add(version)
    if len(versions) != 1:
        _issue(issues, "dataset_version", "$.dataset_version", "a split authority requires exactly one dataset version")
    episode_ids = [entry[0] for entry in entries]
    if len(set(episode_ids)) != len(episode_ids):
        _issue(issues, "duplicate_episode_id", "$.episode_id", "split authority cannot contain duplicate episode ids")
    issues.extend(_split_conflicts(entries))
    if issues or not entries or len(versions) != 1:
        return None, tuple(issues)
    return _split_authority_payload(entries, next(iter(versions))), ()


def _release_files(root: Path) -> set[str]:
    files: set[str] = set()
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_symlink():
            files.add(f"__symlink__:{path.relative_to(root).as_posix()}")
        elif path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _validate_file_inventory(
    root: Path,
    *,
    expected_files: set[str],
    scope: str,
    issues: list[DatasetIssue],
) -> None:
    """Reject unbound files, symlinks, and private-looking directories.

    A manifest hash is not enough if a directory can silently carry additional
    content.  Formal captures and public releases are therefore closed worlds:
    every regular file must be either metadata required by this module or a
    manifest-bound payload.  Evaluator-private material belongs in the
    separately controlled evaluator store, never beside a candidate episode.
    """

    if not root.exists():
        return
    present_files: set[str] = set()
    symlinks: list[str] = []
    reserved: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        # Metadata contains the opaque evaluator commitment, but no private
        # payload.  Partition names are forbidden for user-provided paths and
        # directories, not for this fixed filename.
        if _path_has_reserved_partition(relative) and relative not in expected_files:
            reserved.append(relative)
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file():
            present_files.add(relative)
    if symlinks:
        _issue(
            issues,
            f"{scope}_symlink",
            f"${scope}",
            "directories cannot contain symlinks: " + ", ".join(sorted(symlinks)),
        )
    if reserved:
        _issue(
            issues,
            f"{scope}_private_partition",
            f"${scope}",
            "directories cannot contain evaluator-private partition names: " + ", ".join(sorted(reserved)),
        )
    extra = sorted(present_files - expected_files)
    missing = sorted(expected_files - present_files)
    if extra:
        _issue(
            issues,
            f"unexpected_{scope}_file",
            f"${scope}",
            "unbound files are present: " + ", ".join(extra),
        )
    if missing:
        _issue(
            issues,
            f"missing_{scope}_file",
            f"${scope}",
            "expected files are missing: " + ", ".join(missing),
        )


def _validate_admission_record(
    admission: Mapping[str, Any] | None,
    *,
    manifest: Mapping[str, Any] | None,
    manifest_sha256: str | None,
    receipt_sha256: str | None,
    lineage_sha256: str | None,
    receipt: Mapping[str, Any] | None,
    issues: list[DatasetIssue],
) -> None:
    if admission is None:
        return
    admission_keys = {
        "schema",
        "episode_id",
        "split",
        "formal_benchmark_admission",
        "source_episode_manifest_sha256",
        "release_episode_manifest_sha256",
        "formal_capture_receipt_sha256",
        "lineage_sha256",
        "supply_chain_manifest_sha256",
        "supply_chain_release_id",
        "included_partitions",
        "withheld_learning_modalities",
        "evaluator_private_payload_included",
    }
    if "collection_binding" in admission:
        admission_keys.add("collection_binding")
    _expect_exact_keys(
        admission,
        admission_keys,
        path="$admission",
        issues=issues,
    )
    if admission.get("schema") != RELEASE_ADMISSION_SCHEMA:
        _issue(issues, "admission_schema", "$admission.schema", "unsupported release admission schema")
    if admission.get("formal_benchmark_admission") is not True:
        _issue(issues, "admission_formal", "$admission.formal_benchmark_admission", "must be true")
    if manifest is not None:
        if admission.get("episode_id") != manifest.get("episode_id"):
            _issue(issues, "admission_episode", "$admission.episode_id", "does not match release manifest")
        if admission.get("split") != manifest.get("split"):
            _issue(issues, "admission_split", "$admission.split", "does not match release manifest")
    if admission.get("release_episode_manifest_sha256") != manifest_sha256:
        _issue(issues, "admission_manifest_hash", "$admission.release_episode_manifest_sha256", "does not bind release manifest")
    if admission.get("formal_capture_receipt_sha256") != receipt_sha256:
        _issue(issues, "admission_receipt_hash", "$admission.formal_capture_receipt_sha256", "does not bind formal receipt")
    if admission.get("lineage_sha256") != lineage_sha256:
        _issue(issues, "admission_lineage_hash", "$admission.lineage_sha256", "does not bind lineage")
    if not is_sha256(admission.get("supply_chain_manifest_sha256")):
        _issue(
            issues,
            "admission_supply_chain_hash",
            "$admission.supply_chain_manifest_sha256",
            "must bind a release-verified supply-chain manifest",
        )
    release_id = admission.get("supply_chain_release_id")
    if not isinstance(release_id, str) or not _RELEASE_ID.fullmatch(release_id):
        _issue(
            issues,
            "admission_supply_chain_release_id",
            "$admission.supply_chain_release_id",
            "must be a valid release identifier",
        )
    source_hash = admission.get("source_episode_manifest_sha256")
    if not is_sha256(source_hash):
        _issue(issues, "admission_source_hash", "$admission.source_episode_manifest_sha256", "must be SHA-256")
    elif isinstance(receipt, Mapping) and receipt.get("episode_manifest_sha256") != source_hash:
        _issue(issues, "admission_source_hash", "$admission.source_episode_manifest_sha256", "does not match capture receipt")
    if admission.get("evaluator_private_payload_included") is not False:
        _issue(issues, "admission_private_payload", "$admission.evaluator_private_payload_included", "must be false")
    admission_binding = admission.get("collection_binding")
    manifest_binding = manifest.get("collection_binding") if isinstance(manifest, Mapping) else None
    receipt_binding = receipt.get("collection_binding") if isinstance(receipt, Mapping) else None
    if admission_binding is not None:
        for binding_issue in validate_collection_binding(admission_binding):
            suffix = binding_issue.path.lstrip("$").lstrip(".")
            _issue(
                issues,
                f"admission_collection_{binding_issue.code}",
                f"$admission.collection_binding.{suffix}" if suffix else "$admission.collection_binding",
                binding_issue.message,
            )
    if admission_binding != manifest_binding or admission_binding != receipt_binding:
        _issue(
            issues,
            "admission_collection_mismatch",
            "$admission.collection_binding",
            "must exactly match the manifest and formal capture receipt",
        )
    included = admission.get("included_partitions")
    if not isinstance(included, list) or not included or any(value not in {"policy_visible", "learning_labels"} for value in included):
        _issue(issues, "admission_partitions", "$admission.included_partitions", "must list only public partitions")
    withheld = admission.get("withheld_learning_modalities")
    if not isinstance(withheld, list) or not all(isinstance(value, str) for value in withheld):
        _issue(issues, "admission_withheld", "$admission.withheld_learning_modalities", "must be a string list")


def _verify_release_episode(episode_root: Path) -> tuple[dict[str, Any] | None, Mapping[str, str] | None, tuple[DatasetIssue, ...]]:
    root = episode_root.resolve()
    issues: list[DatasetIssue] = []
    manifest_path = root / _MANIFEST_PATH
    receipt_path = root / _RECEIPT_PATH
    lineage_path = root / _LINEAGE_PATH
    admission_path = root / _ADMISSION_PATH
    manifest = _read_json(manifest_path, issues=issues, issue_path=_MANIFEST_PATH)
    receipt = _read_json(receipt_path, issues=issues, issue_path=_RECEIPT_PATH)
    lineage = _read_json(lineage_path, issues=issues, issue_path=_LINEAGE_PATH)
    admission = _read_json(admission_path, issues=issues, issue_path=_ADMISSION_PATH)
    manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
    receipt_hash = sha256_file(receipt_path) if receipt_path.is_file() else None
    lineage_hash = sha256_file(lineage_path) if lineage_path.is_file() else None
    if manifest is not None:
        for issue in validate_episode_manifest(manifest, base_dir=root, check_files=True):
            _issue(issues, f"manifest_{issue.code}", issue.path, issue.message)
        _validate_formal_manifest_rules(manifest, issues)
    _validate_lineage(lineage, manifest=manifest, path="$lineage", issues=issues)
    _validate_capture_receipt(
        receipt,
        manifest_sha256=(admission.get("source_episode_manifest_sha256") if isinstance(admission, Mapping) else None),
        lineage_sha256=lineage_hash,
        manifest=manifest,
        path="$formal_capture_receipt",
        issues=issues,
    )
    _validate_admission_record(
        admission,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        receipt_sha256=receipt_hash,
        lineage_sha256=lineage_hash,
        receipt=receipt,
        issues=issues,
    )
    expected_files = {_MANIFEST_PATH, _RECEIPT_PATH, _LINEAGE_PATH, _ADMISSION_PATH}
    if manifest is not None:
        learning = manifest.get("learning_labels")
        include_labels = isinstance(learning, Mapping) and learning.get("distributed") is True
        for relative in _bound_files(root, manifest, include_learning_labels=include_labels, issues=issues):
            expected_files.add(relative)
    _validate_file_inventory(root, expected_files=expected_files, scope="release", issues=issues)
    axes = _lineage_axes(lineage)
    return (dict(manifest) if manifest is not None else None), axes, tuple(issues)


def _index_record(
    episode_root: Path,
    manifest: Mapping[str, Any],
    lineage: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = episode_root / _MANIFEST_PATH
    receipt_path = episode_root / _RECEIPT_PATH
    admission_path = episode_root / _ADMISSION_PATH
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    return {
        "episode_id": manifest["episode_id"],
        "split": manifest["split"],
        "dataset_version": manifest["dataset_version"],
        "episode_manifest_path": (Path(manifest["split"]) / manifest["episode_id"] / _MANIFEST_PATH).as_posix(),
        "release_episode_manifest_sha256": sha256_file(manifest_path),
        "source_episode_manifest_sha256": admission["source_episode_manifest_sha256"],
        "formal_capture_receipt_sha256": sha256_file(receipt_path),
        "lineage_sha256": sha256_file(episode_root / _LINEAGE_PATH),
        "supply_chain_manifest_sha256": admission["supply_chain_manifest_sha256"],
        "supply_chain_release_id": admission["supply_chain_release_id"],
        "layout_id": manifest["layout"]["layout_id"],
        "layout_hash": manifest["layout"]["layout_hash"],
        "layout_lineage_hash": manifest["layout"]["layout_lineage_hash"],
        "task_variant_id": manifest["task"]["task_variant_id"],
        "task_spec_sha256": manifest["task"]["task_spec_sha256"],
        "information_profile": manifest["task"]["information_profile"],
        "agent_count": manifest["task"]["agent_count"],
        "policy_modalities": sorted(manifest["policy_visible"]["modalities"]),
        "learning_labels_distributed": manifest["learning_labels"]["distributed"],
        "learning_label_modalities": sorted(manifest["learning_labels"]["modalities"]),
        "evaluator_private_manifest_sha256": manifest["evaluator_private"]["manifest_sha256"],
        "lineage_group_commitment": hashlib.sha256(
            _canonical_json_bytes({axis: lineage[axis] for axis in _GROUPING_AXES})
        ).hexdigest(),
    }


def _discover_release_roots(dataset_root: Path) -> list[Path]:
    result: list[Path] = []
    for split in sorted(FORMAL_SPLITS):
        split_root = dataset_root / split
        if not split_root.is_dir():
            continue
        for child in sorted(split_root.iterdir(), key=lambda path: path.name):
            if child.is_dir():
                result.append(child)
    return result


def rebuild_dataset_index(dataset_root: Path, *, write: bool = True) -> DatasetIntegrityReport:
    """Verify every release episode and atomically rebuild public index files."""

    root = dataset_root.resolve()
    issues: list[DatasetIssue] = []
    records: list[dict[str, Any]] = []
    entries: list[tuple[str, str, Mapping[str, str]]] = []
    for episode_root in _discover_release_roots(root):
        manifest, axes, episode_issues = _verify_release_episode(episode_root)
        issues.extend(episode_issues)
        if manifest is None or axes is None or episode_issues:
            continue
        if episode_root.parent.name != manifest.get("split") or episode_root.name != manifest.get("episode_id"):
            _issue(
                issues,
                "release_location",
                str(episode_root),
                "release path must be <split>/<episode_id>",
            )
            continue
        entries.append((str(manifest["episode_id"]), str(manifest["split"]), axes))
        records.append(_index_record(episode_root, manifest, axes))
    ids = [record["episode_id"] for record in records]
    if len(set(ids)) != len(ids):
        _issue(issues, "duplicate_episode_id", "$.episodes", "release index has duplicate episode ids")
    versions = {record["dataset_version"] for record in records}
    if len(versions) > 1:
        _issue(issues, "dataset_version", "$.dataset_version", "one release root cannot mix dataset versions")
    supply_chain_hashes = {record["supply_chain_manifest_sha256"] for record in records}
    supply_chain_release_ids = {record["supply_chain_release_id"] for record in records}
    if len(supply_chain_hashes) > 1 or len(supply_chain_release_ids) > 1:
        _issue(
            issues,
            "supply_chain_mismatch",
            "$.episodes",
            "one release root cannot mix supply-chain decisions",
        )
    issues.extend(_split_conflicts(entries))
    if issues:
        return DatasetIntegrityReport(root, len(records), tuple(issues))
    if write:
        version = next(iter(versions), "0.0.0-empty")
        payload = {
            "schema": DATASET_INDEX_SCHEMA,
            "dataset_version": version,
            "episode_count": len(records),
            "episodes": sorted(records, key=lambda record: (record["split"], record["episode_id"])),
        }
        _write_json_atomic(root / "manifests" / "dataset_index.json", payload)
        _write_json_atomic(root / "manifests" / "split_authority.json", _split_authority_payload(entries, version))
    return DatasetIntegrityReport(root, len(records), ())


def verify_dataset_integrity(dataset_root: Path) -> DatasetIntegrityReport:
    """Check release files and ensure the checked index is exact and current."""

    root = dataset_root.resolve()
    report = rebuild_dataset_index(root, write=False)
    issues = list(report.issues)
    index_path = root / "manifests" / "dataset_index.json"
    authority_path = root / "manifests" / "split_authority.json"
    index = _read_json(index_path, issues=issues, issue_path="manifests/dataset_index.json")
    authority = _read_json(authority_path, issues=issues, issue_path="manifests/split_authority.json")
    if index is not None:
        expected: list[dict[str, Any]] = []
        for episode_root in _discover_release_roots(root):
            manifest, axes, episode_issues = _verify_release_episode(episode_root)
            if not episode_issues and manifest is not None and axes is not None:
                expected.append(_index_record(episode_root, manifest, axes))
        versions = {record["dataset_version"] for record in expected}
        expected_payload = {
            "schema": DATASET_INDEX_SCHEMA,
            "dataset_version": next(iter(versions), "0.0.0-empty"),
            "episode_count": len(expected),
            "episodes": sorted(expected, key=lambda record: (record["split"], record["episode_id"])),
        }
        if index != expected_payload:
            _issue(issues, "dataset_index_stale", "manifests/dataset_index.json", "index is absent, malformed, or does not match releases")
    if authority is not None:
        entries: list[tuple[str, str, Mapping[str, str]]] = []
        versions: set[str] = set()
        for episode_root in _discover_release_roots(root):
            manifest, axes, episode_issues = _verify_release_episode(episode_root)
            if not episode_issues and manifest is not None and axes is not None:
                entries.append((str(manifest["episode_id"]), str(manifest["split"]), axes))
                versions.add(str(manifest["dataset_version"]))
        expected_authority = _split_authority_payload(entries, next(iter(versions), "0.0.0-empty"))
        if authority != expected_authority:
            _issue(issues, "split_authority_stale", "manifests/split_authority.json", "authority does not match release lineages")
    return DatasetIntegrityReport(root, report.episode_count, tuple(issues))


class DatasetCollector:
    """Project independently admitted source captures into a public release root."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        trusted_receipt_hashes: Iterable[str],
        supply_chain_manifest: Path,
        failure_ledger_path: Path | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.trusted_receipt_hashes = frozenset(trusted_receipt_hashes)
        self.supply_chain_manifest = supply_chain_manifest.resolve()
        self.failure_ledger_path = (
            failure_ledger_path.resolve()
            if failure_ledger_path is not None
            else self.dataset_root / "manifests" / "failure_ledger.jsonl"
        )

    def _verify_supply_chain(
        self,
        report: CandidateIntegrityReport,
    ) -> tuple[str | None, str | None, tuple[DatasetIssue, ...]]:
        """Verify release clearance and bind it to this exact candidate receipt."""

        try:
            supply_report = verify_supply_chain_manifest(
                self.supply_chain_manifest,
                require_release=True,
            )
            payload = load_supply_chain_manifest(self.supply_chain_manifest)
        except (OSError, UnicodeDecodeError, SupplyChainError) as exc:
            return (
                None,
                None,
                (DatasetIssue("supply_chain_manifest", "$supply_chain", str(exc)),),
            )
        issues = tuple(
            DatasetIssue(
                f"supply_chain_{item.get('code', 'invalid')}",
                f"$supply_chain{str(item.get('path', '$'))[1:]}",
                str(item.get("message", "release supply-chain validation failed")),
            )
            for item in supply_report.get("issues", ())
            if isinstance(item, Mapping)
        )
        if supply_report.get("status") != "valid":
            return None, None, issues or (
                DatasetIssue("supply_chain_invalid", "$supply_chain", "release validation failed"),
            )
        if supply_chain_sha256(payload) != supply_report.get("manifest_sha256"):
            return (
                None,
                None,
                (
                    DatasetIssue(
                        "supply_chain_changed",
                        "$supply_chain",
                        "manifest changed after release verification",
                    ),
                ),
            )

        assets = payload.get("assets")
        assert isinstance(assets, list)  # Release validation above guarantees this.
        kinds = {
            asset.get("kind")
            for asset in assets
            if isinstance(asset, Mapping)
        }
        missing_kinds = sorted(_REQUIRED_RELEASE_ASSET_KINDS - kinds)
        binding_issues: list[DatasetIssue] = []
        if missing_kinds:
            binding_issues.append(
                DatasetIssue(
                    "supply_chain_surface",
                    "$supply_chain.assets",
                    f"dataset release is missing asset decisions for {missing_kinds}",
                )
            )
        data_hashes = {
            asset.get("sha256")
            for asset in assets
            if isinstance(asset, Mapping) and asset.get("kind") == "data"
        }
        if report.receipt_sha256 not in data_hashes:
            binding_issues.append(
                DatasetIssue(
                    "supply_chain_candidate_binding",
                    "$supply_chain.assets",
                    "a cleared data asset must bind this formal capture receipt SHA-256",
                )
            )
        release_id = supply_report.get("release_id")
        manifest_sha256 = supply_report.get("manifest_sha256")
        if not isinstance(release_id, str) or not _RELEASE_ID.fullmatch(release_id):
            binding_issues.append(
                DatasetIssue(
                    "supply_chain_release_id",
                    "$supply_chain.release_id",
                    "release identifier is invalid",
                )
            )
        if not is_sha256(manifest_sha256):
            binding_issues.append(
                DatasetIssue(
                    "supply_chain_hash",
                    "$supply_chain",
                    "canonical supply-chain hash is invalid",
                )
            )
        if binding_issues:
            return None, None, tuple(binding_issues)
        return str(manifest_sha256), str(release_id), ()

    @staticmethod
    def _ledger_category(issues: Sequence[DatasetIssue]) -> str:
        if not issues:
            return "none"
        codes = {issue.code for issue in issues}
        if codes & {"destination_exists", "split_authority", "split_authority_stale"}:
            return "infrastructure_failure"
        if codes & {"receipt_backend", "sensor_decode_audit", "pose_closure_threshold", "frame_completeness"}:
            return "sensor_failure"
        if codes & {"private_distribution", "evaluator_private_stream", "policy_truth_leak"}:
            return "quality_failure"
        return "quality_failure"

    def _record_ledger(
        self,
        report: CandidateIntegrityReport,
        *,
        outcome: str,
        issues: Sequence[DatasetIssue],
    ) -> None:
        # The random suffix distinguishes a repeated attempt over the same
        # capture hash while keeping all public identifiers path-free.
        attempt_seed = f"{report.receipt_sha256 or report.manifest_sha256 or 'unknown'}:{uuid.uuid4().hex}"
        attempt_id = "attempt-" + hashlib.sha256(attempt_seed.encode("utf-8")).hexdigest()[:32]
        split = report.manifest.get("split") if isinstance(report.manifest, Mapping) else None
        if not isinstance(split, str) or split not in FORMAL_SPLITS | {"pilot"}:
            split = None
        episode_id = (
            report.episode_id
            if isinstance(report.episode_id, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", report.episode_id)
            else None
        )
        reason_code = issues[0].code if issues else None
        collection_binding = (
            report.manifest.get("collection_binding")
            if isinstance(report.manifest, Mapping)
            else None
        )
        binding_kwargs: dict[str, Any] = {}
        if isinstance(collection_binding, Mapping) and not validate_collection_binding(collection_binding):
            binding_kwargs = {
                "collection_protocol_id": collection_binding.get("protocol_id"),
                "collection_protocol_sha256": collection_binding.get("protocol_sha256"),
                "collection_cell_id": collection_binding.get("cell_id"),
                "collection_episode_index": collection_binding.get("episode_index"),
                "episode_seed": collection_binding.get("episode_seed"),
            }
            split = collection_binding.get("split")
        record = FailureRecord(
            attempt_id=attempt_id,
            outcome=outcome,
            category=self._ledger_category(issues),
            stage="formal_admission",
            recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            split=split,
            episode_id=episode_id,
            source_capture_sha256=report.receipt_sha256,
            receipt_sha256=report.receipt_sha256,
            reason_code=reason_code,
            **binding_kwargs,
        )
        append_failure_record(self.failure_ledger_path, record)

    def collect(self, candidate_root: Path) -> CollectionResult:
        report = verify_candidate_episode(
            candidate_root,
            trusted_receipt_hashes=self.trusted_receipt_hashes,
            require_trusted_receipt=True,
        )
        if not report.valid or report.manifest is None or report.lineage is None or report.episode_id is None:
            quarantine = quarantine_candidate(
                self.dataset_root,
                candidate_root,
                report.issues,
                episode_id=report.episode_id,
                manifest_sha256=report.manifest_sha256,
                receipt_sha256=report.receipt_sha256,
            )
            self._record_ledger(report, outcome="quarantined", issues=report.issues)
            return CollectionResult(False, None, quarantine, report.issues)

        supply_chain_sha256, supply_chain_release_id, supply_chain_issues = self._verify_supply_chain(report)
        if supply_chain_issues:
            quarantine = quarantine_candidate(
                self.dataset_root,
                candidate_root,
                supply_chain_issues,
                episode_id=report.episode_id,
                manifest_sha256=report.manifest_sha256,
                receipt_sha256=report.receipt_sha256,
            )
            self._record_ledger(report, outcome="quarantined", issues=supply_chain_issues)
            return CollectionResult(False, None, quarantine, supply_chain_issues)
        assert supply_chain_sha256 is not None
        assert supply_chain_release_id is not None

        manifest = report.manifest
        split = manifest.get("split")
        if not isinstance(split, str):
            issue = DatasetIssue("split", "$.split", "candidate has no usable split")
            quarantine = quarantine_candidate(self.dataset_root, candidate_root, (issue,), episode_id=report.episode_id)
            self._record_ledger(report, outcome="quarantined", issues=(issue,))
            return CollectionResult(False, None, quarantine, (issue,))
        destination = self.dataset_root / split / report.episode_id
        if destination.exists():
            issue = DatasetIssue("destination_exists", str(destination), "refusing to overwrite an existing release episode")
            quarantine = quarantine_candidate(
                self.dataset_root,
                candidate_root,
                (issue,),
                episode_id=report.episode_id,
                manifest_sha256=report.manifest_sha256,
                receipt_sha256=report.receipt_sha256,
            )
            self._record_ledger(report, outcome="quarantined", issues=(issue,))
            return CollectionResult(False, None, quarantine, (issue,))

        existing_report = rebuild_dataset_index(self.dataset_root, write=False)
        if existing_report.issues:
            raise RuntimeError(
                "dataset root is not internally consistent; repair it before collecting: "
                + "; ".join(issue.code for issue in existing_report.issues)
            )
        candidate_axes = _lineage_axes(report.lineage)
        if candidate_axes is None:
            raise RuntimeError("candidate lineage unexpectedly failed after validation")
        entries = [(report.episode_id, split, candidate_axes)]
        existing_supply_issues: list[DatasetIssue] = []
        for existing_root in _discover_release_roots(self.dataset_root):
            existing_manifest, existing_axes, existing_issues = _verify_release_episode(existing_root)
            if existing_issues or existing_manifest is None or existing_axes is None:
                raise RuntimeError(f"existing release episode became invalid: {existing_root}")
            entries.append((str(existing_manifest["episode_id"]), str(existing_manifest["split"]), existing_axes))
            existing_admission = json.loads(
                (existing_root / _ADMISSION_PATH).read_text(encoding="utf-8")
            )
            if (
                existing_admission.get("supply_chain_manifest_sha256") != supply_chain_sha256
                or existing_admission.get("supply_chain_release_id") != supply_chain_release_id
            ):
                existing_supply_issues.append(
                    DatasetIssue(
                        "supply_chain_mismatch",
                        "$dataset.episodes",
                        "candidate uses a different supply-chain decision from existing episodes",
                    )
                )
        admission_issues = tuple(existing_supply_issues) + _split_conflicts(entries)
        if admission_issues:
            quarantine = quarantine_candidate(
                self.dataset_root,
                candidate_root,
                admission_issues,
                episode_id=report.episode_id,
                manifest_sha256=report.manifest_sha256,
                receipt_sha256=report.receipt_sha256,
            )
            self._record_ledger(report, outcome="quarantined", issues=admission_issues)
            return CollectionResult(False, None, quarantine, admission_issues)

        self._project(
            report,
            destination,
            supply_chain_manifest_sha256=supply_chain_sha256,
            supply_chain_release_id=supply_chain_release_id,
        )
        rebuilt = rebuild_dataset_index(self.dataset_root, write=True)
        if rebuilt.issues:
            raise RuntimeError("release projection completed but index rebuild failed: " + "; ".join(issue.code for issue in rebuilt.issues))
        self._record_ledger(report, outcome="admitted", issues=())
        return CollectionResult(True, destination, None, ())

    def _project(
        self,
        report: CandidateIntegrityReport,
        destination: Path,
        *,
        supply_chain_manifest_sha256: str,
        supply_chain_release_id: str,
    ) -> None:
        assert report.manifest is not None
        assert report.manifest_sha256 is not None
        assert report.lineage_sha256 is not None
        assert report.receipt_sha256 is not None
        source_root = report.episode_root
        projected = _project_manifest_for_release(report.manifest)
        learning = report.manifest.get("learning_labels")
        include_labels = isinstance(learning, Mapping) and learning.get("distributed") is True
        withheld = sorted(learning.get("modalities", ())) if isinstance(learning, Mapping) and not include_labels else []
        bind_issues: list[DatasetIssue] = []
        bindings = _bound_files(source_root, report.manifest, include_learning_labels=include_labels, issues=bind_issues)
        if bind_issues:
            raise RuntimeError("candidate changed after verification: " + "; ".join(issue.code for issue in bind_issues))
        stage = self.dataset_root / ".staging" / f"{destination.name}-{uuid.uuid4().hex}"
        try:
            for relative in sorted(bindings):
                _safe_copy(source_root, relative, stage)
            _safe_copy(source_root, _LINEAGE_PATH, stage)
            _safe_copy(source_root, _RECEIPT_PATH, stage)
            manifest_path = stage / _MANIFEST_PATH
            _write_json_atomic(manifest_path, projected)
            admission = {
                "schema": RELEASE_ADMISSION_SCHEMA,
                "episode_id": projected["episode_id"],
                "split": projected["split"],
                "formal_benchmark_admission": True,
                "source_episode_manifest_sha256": report.manifest_sha256,
                "release_episode_manifest_sha256": sha256_file(manifest_path),
                "formal_capture_receipt_sha256": report.receipt_sha256,
                "lineage_sha256": report.lineage_sha256,
                "supply_chain_manifest_sha256": supply_chain_manifest_sha256,
                "supply_chain_release_id": supply_chain_release_id,
                "included_partitions": ["policy_visible"] + (["learning_labels"] if include_labels else []),
                "withheld_learning_modalities": withheld,
                "evaluator_private_payload_included": False,
            }
            collection_binding = projected.get("collection_binding")
            if isinstance(collection_binding, Mapping):
                admission["collection_binding"] = dict(collection_binding)
            _write_json_atomic(stage / _ADMISSION_PATH, admission)
            _, _, stage_issues = _verify_release_episode(stage)
            if stage_issues:
                raise RuntimeError("release projection is invalid: " + "; ".join(issue.code for issue in stage_issues))
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-candidate", help="validate one formal source capture")
    verify.add_argument("episode_root", type=Path)
    verify.add_argument("--trusted-receipt-sha256", action="append", default=[])
    verify.add_argument("--allow-untrusted-receipt", action="store_true")
    verify.add_argument("--json", action="store_true", dest="as_json")
    collect = subparsers.add_parser("collect", help="admit a trusted capture or write a quarantine record")
    collect.add_argument("episode_root", type=Path)
    collect.add_argument("dataset_root", type=Path)
    collect.add_argument("--trusted-receipt-sha256", action="append", default=[], required=True)
    collect.add_argument("--supply-chain-manifest", type=Path, required=True)
    collect.add_argument("--json", action="store_true", dest="as_json")
    plan = subparsers.add_parser("split-plan", help="validate predeclared, lineage-safe split assignments")
    plan.add_argument("episode_roots", type=Path, nargs="+")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--json", action="store_true", dest="as_json")
    index = subparsers.add_parser("index", help="verify releases and rebuild deterministic public indices")
    index.add_argument("dataset_root", type=Path)
    index.add_argument("--json", action="store_true", dest="as_json")
    verify_dataset = subparsers.add_parser("verify-dataset", help="verify public payload and index integrity")
    verify_dataset.add_argument("dataset_root", type=Path)
    verify_dataset.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def _report_payload(report: CandidateIntegrityReport | DatasetIntegrityReport | CollectionResult) -> dict[str, Any]:
    if isinstance(report, CandidateIntegrityReport):
        return {
            "status": "valid" if report.valid else "invalid",
            "episode_root": str(report.episode_root),
            "episode_id": report.episode_id,
            "manifest_sha256": report.manifest_sha256,
            "formal_capture_receipt_sha256": report.receipt_sha256,
            "issues": [asdict(issue) for issue in report.issues],
        }
    if isinstance(report, DatasetIntegrityReport):
        return {
            "status": "valid" if report.valid else "invalid",
            "dataset_root": str(report.dataset_root),
            "episode_count": report.episode_count,
            "issues": [asdict(issue) for issue in report.issues],
        }
    return {
        "status": "admitted" if report.admitted else "quarantined",
        "episode_root": str(report.episode_root) if report.episode_root else None,
        "quarantine_record": str(report.quarantine_record) if report.quarantine_record else None,
        "issues": [asdict(issue) for issue in report.issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "verify-candidate":
        report = verify_candidate_episode(
            args.episode_root,
            trusted_receipt_hashes=args.trusted_receipt_sha256,
            require_trusted_receipt=not args.allow_untrusted_receipt,
        )
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    if args.command == "collect":
        result = DatasetCollector(
            args.dataset_root,
            trusted_receipt_hashes=args.trusted_receipt_sha256,
            supply_chain_manifest=args.supply_chain_manifest,
        ).collect(args.episode_root)
        print(json.dumps(_report_payload(result), indent=2, sort_keys=True))
        return 0 if result.admitted else 1
    if args.command == "split-plan":
        payload, issues = plan_split_authority(args.episode_roots)
        result = {"status": "valid" if not issues else "invalid", "issues": [asdict(issue) for issue in issues]}
        if payload is not None:
            _write_json_atomic(args.output, payload)
            result["output"] = str(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if payload is not None else 1
    if args.command == "index":
        report = rebuild_dataset_index(args.dataset_root, write=True)
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    if args.command == "verify-dataset":
        report = verify_dataset_integrity(args.dataset_root)
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
