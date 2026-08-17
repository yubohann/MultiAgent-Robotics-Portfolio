"""Package an independently validated Isaac Search3D capture for admission.

The packer is a projection boundary, not an evaluator and not an admission
authority.  It copies only explicitly selected policy-visible artifacts into a
new closed-world candidate, commits to evaluator truth held elsewhere, and
binds the resulting formal receipt to an independent validation receipt.  A
release operator must still approve the formal receipt hash before
``DatasetCollector`` can admit the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from . import isaac_validate as _isaac_validate
from .abi import observation_abi_sha256, validate_formal_observation_abi
from .collection_protocol import (
    CollectionProtocolError,
    load_collection_protocol,
    resolve_collection_binding,
    validate_collection_binding,
)
from .condition_realization import (
    condition_request_from_protocol,
    validate_condition_request,
)
from .formal_dataset import (
    FORMAL_CAPTURE_RECEIPT_SCHEMA,
    LINEAGE_AXES,
    LINEAGE_SCHEMA,
    sha256_file,
    verify_candidate_episode,
)
from .isaac_public_manifest import (
    PublicManifestError,
    build_public_scene_manifest,
)
from .isaac_validate import VALIDATION_SCHEMA, validate_isaac_capture
from .policy_projection import (
    PolicyProjectionError,
    inspect_candidate_pack_streams,
    validate_candidate_abi_sources,
)
from .schema import (
    EPISODE_SCHEMA,
    forbidden_policy_key,
    forbidden_policy_value_token,
    is_safe_relative_path,
    is_sha256,
    iter_tree,
)

PACK_SPEC_SCHEMA = "org.rivermark.benchmark.isaac-pack-spec.v1"
PACK_SPEC_SCHEMA_V2 = "org.rivermark.benchmark.isaac-pack-spec.v2"
_PACK_SPEC_SCHEMAS = frozenset({PACK_SPEC_SCHEMA, PACK_SPEC_SCHEMA_V2})
_ONBOARD_CHUNKED_SOURCE = "sensors/onboard_rgbd.npz"
_REQUIRED_VALIDATION_CHECKS = {
    "online_capture": True,
    "queue_overflow": False,
    "silent_frame_drop": False,
    "timestamp_audit_passed": True,
    "pose_closure_audit_passed": True,
    "action_causality_audit_passed": True,
    "sensor_decode_audit_passed": True,
    "policy_leakage_audit_passed": True,
}
_LINEAGE_VALUE_AXES = frozenset(
    {
        "appearance_domain",
        "dynamics_domain",
        "instruction_family",
        "instruction_annotator",
        "asset_lineage",
        "behavior_policy_checkpoint_family",
    }
)
_FORBIDDEN_SOURCE_PARTS = frozenset(
    {
        "evaluator-private",
        "evaluator_private",
        "hidden",
        "hidden_truth",
        "learning_labels",
        "private",
        "target_truth",
    }
)


@dataclass(frozen=True)
class IsaacPackIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class IsaacPackResult:
    candidate_root: Path | None
    formal_receipt_sha256: str | None
    issues: tuple[IsaacPackIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues and self.candidate_root is not None


class _PackError(RuntimeError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.issue = IsaacPackIssue(code, path, message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _PackError("invalid_json", label, str(exc)) from exc
    if not isinstance(value, Mapping):
        raise _PackError("json_type", label, "expected a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise _PackError("spec_keys", path, f"missing={missing}, extra={extra}")


def _safe_source(
    root: Path,
    relative: object,
    *,
    path: str,
    source_scope: str = "capture",
) -> Path:
    if not is_safe_relative_path(relative):
        raise _PackError(
            "unsafe_source",
            path,
            f"source must be a safe {source_scope}-relative path",
        )
    canonical = PurePosixPath(str(relative).replace("\\", "/"))
    lowered = {part.lower() for part in canonical.parts}
    if lowered & _FORBIDDEN_SOURCE_PARTS or any("overview" in part for part in lowered):
        raise _PackError("private_or_overview_source", path, "private truth and overview artifacts cannot be packed")
    candidate = (root / canonical).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise _PackError("missing_source", path, "selected source is not a contained file")
    return candidate


def _safe_destination(relative: object, *, path: str) -> str:
    if not is_safe_relative_path(relative):
        raise _PackError("unsafe_destination", path, "destination must be a safe relative path")
    canonical = PurePosixPath(str(relative).replace("\\", "/")).as_posix()
    lowered = {part.lower() for part in PurePosixPath(canonical).parts}
    if lowered & _FORBIDDEN_SOURCE_PARTS or any("overview" in part for part in lowered):
        raise _PackError("private_or_overview_destination", path, "private truth and overview artifacts cannot be packed")
    return canonical


def _scan_public_json(path: Path, *, label: str) -> None:
    values: list[Any] = []
    try:
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            values = [json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise _PackError("public_json", label, str(exc)) from exc
    for record_index, value in enumerate(values):
        for tree_path, key, child in iter_tree(value):
            if key is not None and forbidden_policy_key(key):
                raise _PackError(
                    "policy_truth_leak",
                    f"{label}[{record_index}]{tree_path[1:]}",
                    f"policy-visible key {key!r} is forbidden",
                )
            if isinstance(child, str):
                token = forbidden_policy_value_token(child)
                if token is not None:
                    raise _PackError(
                        "policy_truth_leak",
                        f"{label}[{record_index}]{tree_path[1:]}",
                        f"policy-visible string references forbidden provenance token {token!r}",
                    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=6)


def _project_stream(source: Path, destination: Path, stream: Mapping[str, Any], *, path: str) -> int:
    fields = stream.get("fields")
    timestamp_field = stream.get("timestamp_field")
    if not isinstance(timestamp_field, str) or not timestamp_field:
        raise _PackError("timestamp_field", f"{path}.timestamp_field", "must be a non-empty string")
    if fields is not None:
        if source.suffix.lower() != ".npz" or destination.suffix.lower() != ".npz":
            raise _PackError("npz_projection", f"{path}.fields", "field projection requires NPZ source and destination")
        if not isinstance(fields, list) or not fields or not all(isinstance(item, str) and item for item in fields):
            raise _PackError("npz_fields", f"{path}.fields", "must be a non-empty list of field names")
        if len(fields) != len(set(fields)) or timestamp_field not in fields:
            raise _PackError("npz_fields", f"{path}.fields", "fields must be unique and include timestamp_field")
        for field in fields:
            if forbidden_policy_key(field):
                raise _PackError("policy_truth_leak", f"{path}.fields", f"field {field!r} is forbidden")
        try:
            with np.load(source, allow_pickle=False) as payload:
                missing = sorted(set(fields) - set(payload.files))
                if missing:
                    raise _PackError("npz_fields", f"{path}.fields", f"source is missing {missing}")
                arrays = {field: payload[field].copy() for field in fields}
        except _PackError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise _PackError("npz_decode", f"{path}.source", str(exc)) from exc
        timestamps = arrays[timestamp_field]
        if timestamps.dtype != np.int64 or timestamps.ndim != 1 or not len(timestamps):
            raise _PackError("timestamps", f"{path}.timestamp_field", "must select a non-empty int64 [T] array")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise _PackError("timestamps", f"{path}.timestamp_field", "timestamps must be strictly increasing")
        _write_deterministic_npz(destination, arrays)
        return len(timestamps)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _scan_public_json(destination, label=path)
    if destination.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise _PackError("sample_count", path, "JSONL stream is empty")
        times: list[int] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise _PackError("stream_record", f"{path}[{index}]", "must be a JSON object")
            timestamp = record.get(timestamp_field)
            if not isinstance(timestamp, int) or isinstance(timestamp, bool):
                raise _PackError("timestamps", f"{path}[{index}].{timestamp_field}", "must be an integer")
            times.append(timestamp)
        if any(current < previous for previous, current in pairwise(times)):
            raise _PackError("timestamps", path, "JSONL timestamps must be monotonic")
        return len(records)
    if destination.suffix.lower() == ".json":
        return 1
    sample_count = stream.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise _PackError("sample_count", f"{path}.sample_count", "binary copy streams require a positive sample_count")
    return sample_count


def _validation_contract(
    capture_root: Path,
    validation_path: Path,
    evaluator_manifest: Path,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any]]:
    evaluator_sha256 = sha256_file(evaluator_manifest)
    validation = _read_object(validation_path, label="independent_validation")
    if validation.get("schema") != VALIDATION_SCHEMA or validation.get("status") != "passed":
        raise _PackError("validation_status", "independent_validation", "independent validation did not pass")
    if validation.get("formal_benchmark_admission") is not False or validation.get("issues") != []:
        raise _PackError("validation_boundary", "independent_validation", "validator must not self-admit or retain issues")
    capture_receipt_path = capture_root / "capture_receipt.json"
    if not capture_receipt_path.is_file():
        raise _PackError("capture_receipt", "capture_receipt.json", "raw capture receipt is missing")
    capture_sha256 = sha256_file(capture_receipt_path)
    if validation.get("capture_receipt_sha256") != capture_sha256:
        raise _PackError("validation_binding", "independent_validation.capture_receipt_sha256", "does not bind this capture")
    validator_id = validation.get("validator_id")
    if not isinstance(validator_id, str) or not validator_id.strip():
        raise _PackError("validator_id", "independent_validation.validator_id", "must be non-empty")
    validator_source = validation.get("validator_source_sha256")
    if not is_sha256(validator_source):
        raise _PackError("validator_source", "independent_validation.validator_source_sha256", "must be SHA-256")
    if validator_source != sha256_file(Path(_isaac_validate.__file__).resolve()):
        raise _PackError(
            "validator_source",
            "independent_validation.validator_source_sha256",
            "does not match the validator source used by this packer",
        )
    checks = validation.get("checks")
    if not isinstance(checks, Mapping):
        raise _PackError("validation_checks", "independent_validation.checks", "must be an object")
    for key, expected in _REQUIRED_VALIDATION_CHECKS.items():
        if checks.get(key) is not expected:
            raise _PackError("validation_check", f"independent_validation.checks.{key}", f"must be {expected}")
    if checks.get("evaluator_manifest_sha256") != evaluator_sha256:
        raise _PackError("evaluator_binding", "independent_validation.checks.evaluator_manifest_sha256", "does not bind external evaluator truth")
    report = validate_isaac_capture(
        capture_root,
        evaluator_manifest=evaluator_manifest,
        require_clean_source=True,
    )
    if not report.valid or report.receipt_sha256 != capture_sha256:
        detail = ", ".join(issue.code for issue in report.issues)
        raise _PackError("revalidation", str(capture_root), f"independent validator rerun failed: {detail}")
    for key, expected in _REQUIRED_VALIDATION_CHECKS.items():
        if report.checks.get(key) is not expected or report.checks.get(key) != checks.get(key):
            raise _PackError("revalidation_check", f"revalidation.checks.{key}", "does not reproduce validation receipt")
    if report.checks.get("evaluator_manifest_sha256") != evaluator_sha256:
        raise _PackError(
            "revalidation_check",
            "revalidation.checks.evaluator_manifest_sha256",
            "does not reproduce the evaluator commitment",
        )
    return validation, sha256_file(validation_path), checks


def _validate_spec(spec: Mapping[str, Any]) -> None:
    _exact_keys(
        spec,
        {
            "schema",
            "dataset_version",
            "episode_id",
            "split",
            "layout",
            "task",
            "timebase",
            "coordinate_frames",
            "observation_abi",
            "streams",
            "provenance",
            "quality",
            "lineage_values",
            "capture_backend",
        },
        path="$spec",
    )
    if spec.get("schema") not in _PACK_SPEC_SCHEMAS:
        raise _PackError(
            "spec_schema",
            "$spec.schema",
            f"expected one of {sorted(_PACK_SPEC_SCHEMAS)}",
        )
    lineage = spec.get("lineage_values")
    if not isinstance(lineage, Mapping) or set(lineage) != _LINEAGE_VALUE_AXES:
        raise _PackError("lineage_values", "$spec.lineage_values", f"must contain exactly {sorted(_LINEAGE_VALUE_AXES)}")
    streams = spec.get("streams")
    if not isinstance(streams, list) or not streams:
        raise _PackError("streams", "$spec.streams", "must be a non-empty list")


def validate_isaac_pack_spec(spec: Mapping[str, Any]) -> tuple[IsaacPackIssue, ...]:
    """Validate the closed top-level pack-spec contract without writing data."""

    try:
        _validate_spec(spec)
    except _PackError as exc:
        return (exc.issue,)
    return ()


def _validate_v2_stream_contract(
    streams: object,
    candidate_streams: Mapping[str, Any],
) -> None:
    if not isinstance(streams, list):
        raise _PackError("streams", "$spec.streams", "must be a list")
    by_id = {
        stream.get("stream_id"): stream
        for stream in streams
        if isinstance(stream, Mapping) and isinstance(stream.get("stream_id"), str)
    }
    if len(by_id) != len(streams) or set(by_id) != set(candidate_streams):
        raise _PackError(
            "candidate_stream_set",
            "$spec.streams",
            "v2 stream IDs must exactly match the audited eight-stream contract",
        )
    rgbd_destinations = {by_id[stream_id].get("path") for stream_id in ("rgb", "depth")}
    if len(rgbd_destinations) != 1:
        raise _PackError(
            "chunked_archive_destination",
            "$spec.streams",
            "RGB and depth must share one destination for the complete RGB-D archive",
        )
    for stream_id, expected in candidate_streams.items():
        stream = by_id[stream_id]
        assert isinstance(stream, Mapping)
        path = f"$spec.streams.{stream_id}"
        for spec_key, contract_key in (
            ("source", "path"),
            ("modality", "modality"),
            ("timestamp_field", "timestamp_field"),
        ):
            if stream.get(spec_key) != expected.get(contract_key):
                raise _PackError(
                    "candidate_stream_mismatch",
                    f"{path}.{spec_key}",
                    "does not match the audited candidate stream contract",
                )
        arrays = expected.get("arrays")
        source = str(stream.get("source", ""))
        is_chunked_rgbd = source == _ONBOARD_CHUNKED_SOURCE
        if is_chunked_rgbd:
            if not isinstance(arrays, Mapping) or not arrays:
                raise _PackError("candidate_stream_contract", path, "chunked stream arrays are missing")
            first = next(iter(arrays.values()))
            shape = first.get("shape") if isinstance(first, Mapping) else None
            expected_count = shape[0] if isinstance(shape, list) and shape else None
            if "fields" in stream or stream.get("sample_count") != expected_count:
                raise _PackError(
                    "chunked_archive_projection",
                    path,
                    "RGB-D must copy the complete chunked archive with its audited frame count",
                )
        elif stream.get("fields") != expected.get("fields"):
            raise _PackError(
                "candidate_stream_fields",
                f"{path}.fields",
                "must exactly match the audited field selection",
            )


def pack_isaac_capture(
    capture_root: Path,
    validation_receipt: Path,
    evaluator_manifest: Path,
    pack_spec: Path,
    destination: Path,
    collection_protocol: Path | None = None,
) -> IsaacPackResult:
    """Build one closed-world formal candidate without admitting it."""

    capture_root = capture_root.resolve()
    destination = destination.resolve()
    temporary: Path | None = None
    try:
        if destination.exists():
            raise _PackError("destination_exists", str(destination), "destination must not already exist")
        evaluator_manifest = evaluator_manifest.resolve()
        if not evaluator_manifest.is_file():
            raise _PackError("evaluator_manifest", str(evaluator_manifest), "external evaluator manifest is missing")
        evaluator_sha256 = sha256_file(evaluator_manifest)
        validation, validation_sha256, checks = _validation_contract(
            capture_root, validation_receipt.resolve(), evaluator_manifest
        )
        pack_spec_path = pack_spec.resolve()
        spec = _read_object(pack_spec_path, label="pack_spec")
        _validate_spec(spec)
        raw_receipt = _read_object(capture_root / "capture_receipt.json", label="capture_receipt.json")
        raw_capture_sha256 = sha256_file(capture_root / "capture_receipt.json")
        if raw_receipt.get("task_kind") != "search3d":
            raise _PackError("task_kind", "capture_receipt.json.task_kind", "only a Search3D capture can be packed")
        if raw_receipt.get("source_worktree_dirty") is not False:
            raise _PackError("dirty_source", "capture_receipt.json.source_worktree_dirty", "formal candidates require a clean source revision")
        if raw_receipt.get("evaluator_manifest_sha256") != evaluator_sha256:
            raise _PackError(
                "evaluator_binding",
                "capture_receipt.json.evaluator_manifest_sha256",
                "raw capture does not commit to the external evaluator manifest",
            )
        raw_collection_binding = raw_receipt.get("collection_binding")
        raw_condition_request = raw_receipt.get("condition_request")
        resolved_collection_binding: dict[str, Any] | None = None
        if raw_collection_binding is not None:
            binding_issues = validate_collection_binding(raw_collection_binding)
            if binding_issues:
                detail = "; ".join(f"{item.code}:{item.path}" for item in binding_issues)
                raise _PackError("collection_binding", "capture_receipt.json.collection_binding", detail)
            if (
                checks.get("collection_binding_present") is not True
                or checks.get("collection_binding_verified") is not True
            ):
                raise _PackError(
                    "collection_validation",
                    "independent_validation.checks.collection_binding_verified",
                    "independent validation did not attest the capture binding and runtime seed",
                )
            if collection_protocol is None:
                raise _PackError(
                    "collection_protocol",
                    "collection_protocol",
                    "a capture-bound collection protocol file is required for packing",
                )
            assert isinstance(raw_collection_binding, Mapping)
            try:
                protocol = load_collection_protocol(collection_protocol.expanduser().resolve())
                resolved_collection_binding = resolve_collection_binding(
                    protocol,
                    cell_id=str(raw_collection_binding["cell_id"]),
                    episode_index=int(raw_collection_binding["episode_index"]),
                )
                expected_condition_request = None
                if isinstance(protocol, Mapping) and isinstance(protocol.get("cells"), list):
                    expected_condition_request = condition_request_from_protocol(
                        protocol,
                        protocol_id=str(resolved_collection_binding["protocol_id"]),
                        protocol_sha256=str(resolved_collection_binding["protocol_sha256"]),
                        cell_id=str(resolved_collection_binding["cell_id"]),
                    )
            except (OSError, CollectionProtocolError, ValueError, TypeError, KeyError) as exc:
                raise _PackError("collection_protocol", "collection_protocol", str(exc)) from exc
            if dict(raw_collection_binding) != resolved_collection_binding:
                raise _PackError(
                    "collection_binding",
                    "capture_receipt.json.collection_binding",
                    "raw binding does not match the public protocol hash, split, or deterministic seed",
                )
            if spec.get("split") != resolved_collection_binding["split"]:
                raise _PackError(
                    "collection_split",
                    "$spec.split",
                    "pack split must match the predeclared collection cell",
                )
            if expected_condition_request is not None:
                condition_issues = validate_condition_request(
                    raw_condition_request,
                    binding=resolved_collection_binding,
                )
                if condition_issues:
                    detail = "; ".join(f"{item['code']}:{item['path']}" for item in condition_issues)
                    raise _PackError("condition_request", "capture_receipt.json.condition_request", detail)
                if dict(raw_condition_request) != expected_condition_request:
                    raise _PackError(
                        "condition_request",
                        "capture_receipt.json.condition_request",
                        "raw condition request does not match the public protocol cell",
                    )
                if checks.get("condition_realization_verified") is not True:
                    raise _PackError(
                        "condition_realization",
                        "independent_validation.checks.condition_realization_verified",
                        "independent validation did not verify every requested condition from raw evidence",
                    )
        elif collection_protocol is not None:
            raise _PackError(
                "collection_binding",
                "capture_receipt.json.collection_binding",
                "a collection protocol cannot be attached after capture",
            )
        revision = raw_receipt.get("source_revision")
        if not isinstance(revision, str) or not 7 <= len(revision) <= 64 or any(ch not in "0123456789abcdef" for ch in revision):
            raise _PackError("source_revision", "capture_receipt.json.source_revision", "must be a Git hex revision")
        provenance = spec.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("code_commit") != revision:
            raise _PackError("source_revision", "$spec.provenance.code_commit", "must equal the capture source revision")

        raw_backend = raw_receipt.get("capture_backend")
        if not isinstance(raw_backend, Mapping):
            raise _PackError(
                "capture_backend_commitment",
                "capture_receipt.json.capture_backend",
                "formal packing requires a capture-bound backend and sensor-physics smoke commitment",
            )
        _exact_keys(
            raw_backend,
            {"kind", "build", "sensor_physics_smoke_receipt_sha256"},
            path="capture_receipt.json.capture_backend",
        )
        if raw_backend.get("kind") != "isaaclab":
            raise _PackError(
                "capture_backend_kind",
                "capture_receipt.json.capture_backend.kind",
                "native Isaac packing requires kind='isaaclab'",
            )
        if (
            not isinstance(raw_backend.get("build"), str)
            or not raw_backend["build"].strip()
            or not is_sha256(raw_backend.get("sensor_physics_smoke_receipt_sha256"))
        ):
            raise _PackError(
                "capture_backend_commitment",
                "capture_receipt.json.capture_backend",
                "backend build and sensor-physics smoke SHA-256 must be capture-bound",
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        layout_spec = spec.get("layout")
        task_spec = spec.get("task")
        if not isinstance(layout_spec, Mapping) or not isinstance(task_spec, Mapping):
            raise _PackError("spec_type", "$spec", "layout and task must be objects")
        _exact_keys(layout_spec, {"layout_id", "layout_hash", "layout_lineage_hash", "source"}, path="$spec.layout")
        _exact_keys(
            task_spec,
            {"task_id", "task_variant_id", "information_profile", "observation_scope", "agent_count", "source"},
            path="$spec.task",
        )
        abi_spec = spec.get("observation_abi")
        if not isinstance(abi_spec, Mapping):
            raise _PackError("observation_abi", "$spec.observation_abi", "must be an object")
        if spec.get("schema") == PACK_SPEC_SCHEMA:
            _exact_keys(abi_spec, {"source", "path"}, path="$spec.observation_abi")
            abi_source = _safe_source(
                capture_root,
                abi_spec["source"],
                path="$spec.observation_abi.source",
            )
        else:
            _exact_keys(
                abi_spec,
                {
                    "source",
                    "source_scope",
                    "path",
                    "sha256",
                    "capture_receipt_sha256",
                },
                path="$spec.observation_abi",
            )
            if abi_spec.get("source_scope") != "pack_spec":
                raise _PackError(
                    "observation_abi_scope",
                    "$spec.observation_abi.source_scope",
                    "v2 external ABI must be relative to the pack-spec directory",
                )
            if abi_spec.get("capture_receipt_sha256") != raw_capture_sha256:
                raise _PackError(
                    "observation_abi_capture_binding",
                    "$spec.observation_abi.capture_receipt_sha256",
                    "external ABI descriptor does not bind this capture receipt",
                )
            abi_source = _safe_source(
                pack_spec_path.parent,
                abi_spec["source"],
                path="$spec.observation_abi.source",
                source_scope="pack-spec",
            )
        abi_relative = _safe_destination(abi_spec["path"], path="$spec.observation_abi.path")
        try:
            abi_payload = json.loads(abi_source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise _PackError("observation_abi_json", "$spec.observation_abi.source", str(exc)) from exc
        abi_issues = validate_formal_observation_abi(abi_payload)
        if abi_issues:
            detail = "; ".join(f"{item.code}:{item.path}" for item in abi_issues)
            raise _PackError("observation_abi", "$spec.observation_abi.source", detail)
        abi_hash = observation_abi_sha256(abi_payload)
        if spec.get("schema") == PACK_SPEC_SCHEMA_V2:
            if abi_spec.get("sha256") != abi_hash:
                raise _PackError(
                    "observation_abi_hash",
                    "$spec.observation_abi.sha256",
                    "external ABI canonical SHA-256 does not match the descriptor",
                )
            try:
                candidate_streams = inspect_candidate_pack_streams(capture_root)
            except PolicyProjectionError as exc:
                raise _PackError(
                    "candidate_stream_contract",
                    str(capture_root),
                    str(exc),
                ) from exc
            source_issues = validate_candidate_abi_sources(abi_payload, candidate_streams)
            if source_issues:
                detail = "; ".join(
                    f"{item.code}:{item.path}" for item in source_issues
                )
                raise _PackError(
                    "observation_abi_source_contract",
                    "$spec.observation_abi.source",
                    detail,
                )
            _validate_v2_stream_contract(spec.get("streams"), candidate_streams)

        scene_source = _safe_source(capture_root, layout_spec["source"], path="$spec.layout.source")
        scene_destination = temporary / "scenes" / "scene.json"
        scene_destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.get("schema") == PACK_SPEC_SCHEMA_V2:
            try:
                public_scene = build_public_scene_manifest(
                    _read_object(scene_source, label="$spec.layout.source")
                )
            except PublicManifestError as exc:
                raise _PackError(
                    "public_scene_projection", "$spec.layout.source", str(exc)
                ) from exc
            _write_json(scene_destination, public_scene)
            if layout_spec.get("layout_hash") != sha256_file(scene_destination):
                raise _PackError(
                    "layout_hash",
                    "$spec.layout.layout_hash",
                    "must bind the deterministic public scene projection",
                )
        else:
            shutil.copyfile(scene_source, scene_destination)
        _scan_public_json(scene_destination, label="$spec.layout.source")
        task_source = _safe_source(capture_root, task_spec["source"], path="$spec.task.source")
        task_destination = temporary / "tasks" / "task.json"
        task_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task_source, task_destination)
        _scan_public_json(task_destination, label="$spec.task.source")

        manifest_streams: list[dict[str, Any]] = []
        projected_outputs: dict[str, tuple[str, tuple[str, ...] | None, int, str]] = {}
        if abi_relative in projected_outputs:
            raise _PackError("observation_abi_collision", "$spec.observation_abi.path", "ABI destination collides with a stream")
        abi_destination = temporary / abi_relative
        abi_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(abi_source, abi_destination)
        raw_streams = spec["streams"]
        assert isinstance(raw_streams, list)
        for index, raw_stream in enumerate(raw_streams):
            path = f"$spec.streams[{index}]"
            if not isinstance(raw_stream, Mapping):
                raise _PackError("stream_type", path, "must be an object")
            allowed = {
                "stream_id", "partition", "modality", "media_type", "timestamp_field",
                "source", "path", "fields", "sample_count",
            }
            if set(raw_stream) - allowed:
                raise _PackError("stream_keys", path, f"extra fields: {sorted(set(raw_stream) - allowed)}")
            required = {"stream_id", "partition", "modality", "media_type", "timestamp_field", "source", "path"}
            if required - set(raw_stream):
                raise _PackError("stream_keys", path, f"missing fields: {sorted(required - set(raw_stream))}")
            if raw_stream.get("partition") != "policy_visible":
                raise _PackError("stream_partition", f"{path}.partition", "v1 Isaac packer publishes policy-visible streams only")
            source = _safe_source(capture_root, raw_stream["source"], path=f"{path}.source")
            relative = _safe_destination(raw_stream["path"], path=f"{path}.path")
            if relative == abi_relative:
                raise _PackError("observation_abi_collision", f"{path}.path", "stream destination collides with observation ABI")
            selection = tuple(raw_stream["fields"]) if isinstance(raw_stream.get("fields"), list) else None
            output_key = (source.as_posix(), selection)
            previous = projected_outputs.get(relative)
            if previous is None:
                count = _project_stream(source, temporary / relative, raw_stream, path=path)
                projected_outputs[relative] = (*output_key, count, sha256_file(temporary / relative))
            else:
                previous_source, previous_fields, count, _ = previous
                if (previous_source, previous_fields) != output_key:
                    raise _PackError("stream_collision", f"{path}.path", "one destination has incompatible projections")
            _, _, count, digest = projected_outputs[relative]
            manifest_streams.append(
                {
                    "stream_id": raw_stream["stream_id"],
                    "partition": "policy_visible",
                    "modality": raw_stream["modality"],
                    "media_type": raw_stream["media_type"],
                    "sample_count": count,
                    "timestamp_field": raw_stream["timestamp_field"],
                    "path": relative,
                    "sha256": digest,
                }
            )

        profile = task_spec.get("information_profile")
        quality_spec = spec.get("quality")
        if not isinstance(quality_spec, Mapping):
            raise _PackError("spec_type", "$spec.quality", "must be an object")
        _exact_keys(quality_spec, {"task_success", "invalid_reasons"}, path="$spec.quality")
        pose_error = checks.get("pose_closure_max_error_m")
        if not isinstance(pose_error, (int, float)) or isinstance(pose_error, bool) or not math.isfinite(float(pose_error)):
            raise _PackError("pose_closure", "independent_validation.checks.pose_closure_max_error_m", "must be finite")
        manifest = {
            "schema": EPISODE_SCHEMA,
            "dataset_version": spec["dataset_version"],
            "episode_id": spec["episode_id"],
            "split": spec["split"],
            "layout": {
                "layout_id": layout_spec["layout_id"],
                "layout_hash": layout_spec["layout_hash"],
                "layout_lineage_hash": layout_spec["layout_lineage_hash"],
                "scene_manifest_ref": "scenes/scene.json",
                "scene_manifest_sha256": sha256_file(scene_destination),
            },
            "task": {
                "task_id": task_spec["task_id"],
                "task_variant_id": task_spec["task_variant_id"],
                "task_spec_ref": "tasks/task.json",
                "task_spec_sha256": sha256_file(task_destination),
                "information_profile": profile,
                "observation_scope": task_spec["observation_scope"],
                "agent_count": task_spec["agent_count"],
            },
            "timebase": spec["timebase"],
            "coordinate_frames": spec["coordinate_frames"],
            "observation_abi": {
                "path": abi_relative,
                "sha256": abi_hash,
            },
            "streams": manifest_streams,
            "policy_visible": {
                "information_profile": profile,
                "modalities": sorted({stream["modality"] for stream in manifest_streams}),
            },
            "learning_labels": {"distributed": False, "modalities": []},
            "evaluator_private": {
                "distributed": False,
                "server_only": True,
                "manifest_sha256": evaluator_sha256,
            },
            "provenance": dict(provenance),
            "quality": {
                "recording_valid": True,
                "task_success": quality_spec["task_success"],
                "invalid_reasons": quality_spec["invalid_reasons"],
                "frame_completeness_ratio": 1.0,
                "timestamp_monotonic": True,
                "pose_closure_max_error_m": float(pose_error),
            },
        }
        if resolved_collection_binding is not None:
            manifest["collection_binding"] = dict(resolved_collection_binding)
        manifest_path = temporary / "episode_manifest.json"
        _write_json(manifest_path, manifest)

        lineage_values = spec["lineage_values"]
        assert isinstance(lineage_values, Mapping)
        capture_sha256 = raw_capture_sha256
        axes = {
            "layout_lineage": layout_spec["layout_lineage_hash"],
            "task_manifest": sha256_file(task_destination),
            "episode": hashlib.sha256(str(spec["episode_id"]).encode("utf-8")).hexdigest(),
            "trajectory_lineage": capture_sha256,
        }
        axes.update(
            {
                axis: _digest_value({"axis": axis, "value": lineage_values[axis]})
                for axis in sorted(_LINEAGE_VALUE_AXES)
            }
        )
        if set(axes) != set(LINEAGE_AXES) or not all(is_sha256(value) for value in axes.values()):
            raise _PackError("lineage", "$spec.lineage_values", "could not construct all ten lineage commitments")
        lineage_path = temporary / "lineage.json"
        _write_json(lineage_path, {"schema": LINEAGE_SCHEMA, "episode_id": spec["episode_id"], "axes": axes})

        public_inventory = {
            relative: digest for relative, (_, _, _, digest) in sorted(projected_outputs.items())
        }
        public_inventory[abi_relative] = sha256_file(abi_destination)
        public_inventory["scenes/scene.json"] = sha256_file(scene_destination)
        public_inventory["tasks/task.json"] = sha256_file(task_destination)
        backend = spec.get("capture_backend")
        if not isinstance(backend, Mapping):
            raise _PackError("spec_type", "$spec.capture_backend", "must be an object")
        _exact_keys(backend, {"build", "sensor_physics_smoke_receipt_sha256"}, path="$spec.capture_backend")
        if backend != {
            "build": raw_backend["build"],
            "sensor_physics_smoke_receipt_sha256": raw_backend[
                "sensor_physics_smoke_receipt_sha256"
            ],
        }:
            raise _PackError(
                "capture_backend_mismatch",
                "$spec.capture_backend",
                "pack spec backend must exactly match the capture-bound commitment",
            )
        receipt = {
            "schema": FORMAL_CAPTURE_RECEIPT_SCHEMA,
            "status": "admitted",
            "formal_benchmark_admission": True,
            "episode_manifest_sha256": sha256_file(manifest_path),
            "lineage_sha256": sha256_file(lineage_path),
            "observation_abi_sha256": abi_hash,
            "capture_backend": {
                "kind": "isaaclab",
                "build": backend["build"],
                "sensor_physics_smoke_receipt_sha256": backend["sensor_physics_smoke_receipt_sha256"],
            },
            "integrity": {
                "online_capture": True,
                "queue_overflow": False,
                "silent_frame_drop": False,
                "timestamp_audit_passed": True,
                "pose_closure_audit_passed": True,
                "action_causality_audit_passed": True,
                "sensor_decode_audit_passed": True,
                "policy_leakage_audit_passed": True,
                "independent_validator_id": validation["validator_id"],
                "independent_validator_sha256": validation_sha256,
                "pose_closure_threshold_m": float(checks.get("pose_closure_threshold_m", 0.01)),
            },
            "partitions": {
                "policy_visible_audit_sha256": _digest_value(public_inventory),
                "learning_labels_release_allowed": False,
                "evaluator_private_distributed": False,
                "evaluator_private_server_only": True,
            },
        }
        if resolved_collection_binding is not None:
            receipt["integrity"]["condition_realization_verified"] = True
            receipt["collection_binding"] = dict(resolved_collection_binding)
        receipt_path = temporary / "formal_capture_receipt.json"
        _write_json(receipt_path, receipt)
        verification = verify_candidate_episode(temporary, require_trusted_receipt=False)
        if not verification.valid:
            detail = "; ".join(f"{issue.code}:{issue.path}" for issue in verification.issues)
            raise _PackError("candidate_verification", str(temporary), detail)
        os.replace(temporary, destination)
        temporary = None
        return IsaacPackResult(destination, sha256_file(destination / "formal_capture_receipt.json"), ())
    except _PackError as exc:
        return IsaacPackResult(None, None, (exc.issue,))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return IsaacPackResult(None, None, (IsaacPackIssue("pack_error", str(destination), str(exc)),))
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("validation_receipt", type=Path)
    parser.add_argument("evaluator_manifest", type=Path)
    parser.add_argument("pack_spec", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--collection-protocol",
        type=Path,
        help="Public protocol required when the raw capture contains a collection binding.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = pack_isaac_capture(
        args.capture_root,
        args.validation_receipt,
        args.evaluator_manifest,
        args.pack_spec,
        args.destination,
        args.collection_protocol,
    )
    print(
        json.dumps(
            {
                "valid": result.valid,
                "candidate_root": str(result.candidate_root) if result.candidate_root else None,
                "formal_receipt_sha256": result.formal_receipt_sha256,
                "issues": [asdict(issue) for issue in result.issues],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
