"""Read-only readiness audit for packaging a native Isaac T1 capture.

The audit does not create a candidate, copy payload bytes, admit an episode,
or disclose evaluator-private truth.  It turns the packer's implicit
prerequisites into explicit, machine-readable blockers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
from .condition_realization import condition_request_from_protocol
from .formal_dataset import sha256_file
from .isaac_pack import (
    PACK_SPEC_SCHEMA,
    PACK_SPEC_SCHEMA_V2,
    validate_isaac_pack_spec,
)
from .isaac_public_manifest import (
    build_public_scene_manifest,
    public_manifest_sha256,
    validate_public_payload,
)
from .isaac_validate import validate_isaac_capture
from .policy_projection import (
    PolicyProjectionError,
    PolicySourceInspection,
    inspect_candidate_pack_streams,
    inspect_policy_observation_sources,
    validate_candidate_abi_sources,
)
from .schema import (
    INFORMATION_PROFILE_MODALITIES,
    forbidden_policy_key,
    is_safe_relative_path,
    is_sha256,
)
from .supply_chain import SupplyChainError, verify_supply_chain_manifest

PACK_READINESS_SCHEMA = "org.rivermark.benchmark.isaac-pack-readiness.v1"
_CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
_VALIDATION_SCHEMA = "org.rivermark.isaac-independent-validation.v1"
_FORBIDDEN_PARTS = frozenset(
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
class PackReadinessIssue:
    scope: str
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PackReadinessReport:
    capture_receipt_sha256: str | None
    independent_validation_sha256: str | None
    source_revision: str | None
    expected_evaluator_manifest_sha256: str | None
    policy_source_inspection: Mapping[str, Any] | None
    candidate_streams: Mapping[str, Any] | None
    checks: Mapping[str, bool]
    issues: tuple[PackReadinessIssue, ...]

    @property
    def candidate_pack_ready(self) -> bool:
        return not any(issue.scope == "candidate_pack" for issue in self.issues)

    @property
    def supply_chain_release_ready(self) -> bool:
        return not any(issue.scope == "formal_release" for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PACK_READINESS_SCHEMA,
            "status": "ready_for_candidate_pack" if self.candidate_pack_ready else "blocked",
            "candidate_pack_ready": self.candidate_pack_ready,
            "supply_chain_release_ready": self.supply_chain_release_ready,
            "formal_benchmark_admission": False,
            "capture_receipt_sha256": self.capture_receipt_sha256,
            "independent_validation_sha256": self.independent_validation_sha256,
            "source_revision": self.source_revision,
            "expected_evaluator_manifest_sha256": self.expected_evaluator_manifest_sha256,
            "policy_source_inspection": self.policy_source_inspection,
            "candidate_streams": self.candidate_streams,
            "checks": dict(self.checks),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class _AuditedAbi:
    path: Path
    capture_relative: str | None
    canonical_sha256: str


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _add(
    issues: list[PackReadinessIssue],
    code: str,
    path: str,
    message: str,
    *,
    scope: str = "candidate_pack",
) -> None:
    issues.append(PackReadinessIssue(scope, code, path, message))


def _safe_capture_relative(capture: Path, path: Path) -> str | None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not resolved.is_relative_to(capture):
        return None
    relative = resolved.relative_to(capture).as_posix()
    return relative if is_safe_relative_path(relative) else None


def _private_or_overview(relative: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(relative).parts}
    return bool(parts & _FORBIDDEN_PARTS) or any("overview" in part for part in parts)


def _inspection_payload(inspection: PolicySourceInspection) -> dict[str, Any]:
    return {
        "frame_count": inspection.frame_count,
        "state_sample_count": inspection.state_sample_count,
        "collection_binding": dict(inspection.collection_binding),
        "source_artifacts": [dict(item) for item in inspection.source_artifacts],
        "streams": dict(inspection.streams),
    }


def _audit_protocol(
    receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    protocol_path: Path | None,
    issues: list[PackReadinessIssue],
) -> bool:
    binding = receipt.get("collection_binding")
    binding_issues = validate_collection_binding(binding)
    if binding_issues:
        detail = "; ".join(f"{item.code}:{item.path}" for item in binding_issues)
        _add(issues, "collection_binding", "capture_receipt.json.collection_binding", detail)
        return False
    if protocol_path is None:
        _add(issues, "collection_protocol_missing", "collection_protocol", "bound public protocol was not supplied")
        return False
    try:
        protocol = load_collection_protocol(protocol_path.expanduser().resolve())
        assert isinstance(binding, Mapping)
        resolved = resolve_collection_binding(
            protocol,
            cell_id=str(binding["cell_id"]),
            episode_index=int(binding["episode_index"]),
        )
        expected_condition = condition_request_from_protocol(
            protocol,
            protocol_id=str(resolved["protocol_id"]),
            protocol_sha256=str(resolved["protocol_sha256"]),
            cell_id=str(resolved["cell_id"]),
        )
    except (OSError, CollectionProtocolError, KeyError, TypeError, ValueError) as exc:
        _add(issues, "collection_protocol", "collection_protocol", str(exc))
        return False
    checks = validation.get("checks")
    if dict(binding) != resolved:
        _add(issues, "collection_binding_mismatch", "capture_receipt.json.collection_binding", "does not resolve from the supplied protocol")
        return False
    if receipt.get("condition_request") != expected_condition:
        _add(issues, "condition_request_mismatch", "capture_receipt.json.condition_request", "does not equal the frozen protocol request")
        return False
    if not isinstance(checks, Mapping) or checks.get("condition_realization_verified") is not True:
        _add(issues, "condition_realization", "independent_validation.checks.condition_realization_verified", "independent validation did not attest condition realization")
        return False
    return True


def _audit_evaluator(
    capture: Path,
    receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    evaluator_manifest: Path | None,
    issues: list[PackReadinessIssue],
) -> bool:
    expected = receipt.get("evaluator_manifest_sha256")
    checks = validation.get("checks")
    validation_expected = checks.get("evaluator_manifest_sha256") if isinstance(checks, Mapping) else None
    if not is_sha256(expected) or validation_expected != expected:
        _add(issues, "evaluator_commitment", "capture_receipt.json.evaluator_manifest_sha256", "capture and validation do not share one evaluator SHA-256")
        return False
    if evaluator_manifest is None:
        _add(issues, "evaluator_manifest_missing", "evaluator_manifest", "external evaluator-private manifest was not supplied")
        return False
    manifest = evaluator_manifest.expanduser().resolve()
    if not manifest.is_file():
        _add(
            issues,
            "evaluator_manifest_missing",
            "evaluator_manifest",
            "external evaluator-private manifest is missing",
        )
        return False
    if sha256_file(manifest) != expected:
        _add(issues, "evaluator_manifest_mismatch", "evaluator_manifest", "external manifest bytes do not match the committed SHA-256")
        return False
    report = validate_isaac_capture(capture, evaluator_manifest=manifest, require_clean_source=True)
    if not report.valid or report.receipt_sha256 != sha256_file(capture / "capture_receipt.json"):
        detail = ", ".join(issue.code for issue in report.issues)
        _add(issues, "private_revalidation_failed", "evaluator_manifest", detail or "validator receipt binding failed")
        return False
    return True


def _audit_abi(
    capture: Path,
    abi_path: Path | None,
    source_streams: Mapping[str, Any] | None,
    issues: list[PackReadinessIssue],
) -> tuple[bool, _AuditedAbi | None]:
    if abi_path is None:
        _add(issues, "observation_abi_missing", "observation_abi", "formal ABI was not supplied")
        return False, None
    resolved = abi_path.expanduser().resolve()
    if not resolved.is_file():
        _add(issues, "observation_abi_missing", "observation_abi", "formal ABI file is missing")
        return False, None
    relative = _safe_capture_relative(capture, abi_path)
    try:
        abi = _read_object(resolved, label="observation ABI")
    except (TypeError, ValueError) as exc:
        _add(issues, "observation_abi_json", "observation_abi", str(exc))
        return False, None
    abi_issues = validate_formal_observation_abi(abi)
    if abi_issues:
        detail = "; ".join(f"{item.code}:{item.path}" for item in abi_issues)
        _add(issues, "observation_abi", "observation_abi", detail)
        return False, None
    source_issues = (
        validate_candidate_abi_sources(abi, source_streams)
        if source_streams is not None
        else ()
    )
    for issue in source_issues:
        _add(issues, issue.code, issue.path, issue.message)
    audited = _AuditedAbi(
        path=resolved,
        capture_relative=relative,
        canonical_sha256=observation_abi_sha256(abi),
    )
    return source_streams is not None and not source_issues, audited


def _audit_pack_spec(
    capture: Path,
    spec_path: Path | None,
    *,
    source_revision: str | None,
    split: object,
    audited_abi: _AuditedAbi | None,
    capture_receipt_sha256: str | None,
    source_streams: Mapping[str, Any] | None,
    capture_backend: Mapping[str, Any] | None,
    issues: list[PackReadinessIssue],
) -> bool:
    if spec_path is None:
        _add(issues, "pack_spec_missing", "pack_spec", "closed-world pack spec was not supplied")
        return False
    try:
        spec = _read_object(spec_path.expanduser().resolve(), label="pack spec")
    except (TypeError, ValueError) as exc:
        _add(issues, "pack_spec_json", "pack_spec", str(exc))
        return False
    structural = validate_isaac_pack_spec(spec)
    if structural:
        for issue in structural:
            _add(issues, issue.code, issue.path, issue.message)
        return False
    valid = True
    provenance = spec.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("code_commit") != source_revision:
        _add(issues, "source_revision_mismatch", "$spec.provenance.code_commit", "must equal the capture source revision")
        valid = False
    if spec.get("split") != split:
        _add(issues, "pack_split_mismatch", "$spec.split", "must equal the bound collection split")
        valid = False
    spec_backend = spec.get("capture_backend")
    expected_backend = (
        {
            "build": capture_backend.get("build"),
            "sensor_physics_smoke_receipt_sha256": capture_backend.get(
                "sensor_physics_smoke_receipt_sha256"
            ),
        }
        if isinstance(capture_backend, Mapping)
        else None
    )
    if not isinstance(spec_backend, Mapping) or spec_backend != expected_backend:
        _add(
            issues,
            "capture_backend_mismatch",
            "$spec.capture_backend",
            "must exactly match the capture-bound backend commitment",
        )
        valid = False
    abi_spec = spec.get("observation_abi")
    if not isinstance(abi_spec, Mapping) or audited_abi is None:
        _add(issues, "pack_abi_mismatch", "$spec.observation_abi", "must select the audited ABI")
        valid = False
    elif spec.get("schema") == PACK_SPEC_SCHEMA:
        if (
            audited_abi.capture_relative is None
            or abi_spec.get("source") != audited_abi.capture_relative
        ):
            _add(
                issues,
                "pack_abi_mismatch",
                "$spec.observation_abi.source",
                "v1 pack specs require the audited capture-relative ABI",
            )
            valid = False
    elif spec.get("schema") == PACK_SPEC_SCHEMA_V2:
        source = abi_spec.get("source")
        spec_root = spec_path.expanduser().resolve().parent
        resolved_source = (
            (spec_root / PurePosixPath(str(source).replace("\\", "/"))).resolve()
            if is_safe_relative_path(source)
            else None
        )
        if (
            abi_spec.get("source_scope") != "pack_spec"
            or resolved_source != audited_abi.path
            or abi_spec.get("sha256") != audited_abi.canonical_sha256
            or abi_spec.get("capture_receipt_sha256") != capture_receipt_sha256
        ):
            _add(
                issues,
                "pack_abi_mismatch",
                "$spec.observation_abi",
                "v2 external ABI source, canonical hash, or capture binding differs",
            )
            valid = False
        layout_spec = spec.get("layout")
        task_spec = spec.get("task")
        if not isinstance(layout_spec, Mapping) or not isinstance(task_spec, Mapping):
            _add(
                issues,
                "public_control_plane",
                "$spec",
                "layout and task must be objects",
            )
            valid = False
        else:
            raw_scene: Mapping[str, Any] | None = None
            public_scene: Mapping[str, Any] | None = None
            scene_source = layout_spec.get("source")
            if not is_safe_relative_path(scene_source):
                _add(
                    issues,
                    "public_scene_source",
                    "$spec.layout.source",
                    "must be a safe capture-relative JSON source",
                )
                valid = False
            else:
                scene_path = (capture / str(scene_source)).resolve()
                if not scene_path.is_relative_to(capture) or not scene_path.is_file():
                    _add(
                        issues,
                        "public_scene_source",
                        "$spec.layout.source",
                        "selected scene source is missing",
                    )
                    valid = False
                else:
                    try:
                        raw_scene = _read_object(scene_path, label="public scene source")
                        public_scene = build_public_scene_manifest(raw_scene)
                    except (TypeError, ValueError) as exc:
                        _add(
                            issues,
                            "public_scene_projection",
                            "$spec.layout.source",
                            str(exc),
                        )
                        valid = False
            if (
                public_scene is not None
                and layout_spec.get("layout_hash") != public_manifest_sha256(public_scene)
            ):
                _add(
                    issues,
                    "layout_hash",
                    "$spec.layout.layout_hash",
                    "must bind the deterministic public scene projection",
                )
                valid = False

            public_task: Mapping[str, Any] | None = None
            task_source = task_spec.get("source")
            if not is_safe_relative_path(task_source):
                _add(
                    issues,
                    "public_task_source",
                    "$spec.task.source",
                    "must be a safe capture-relative JSON source",
                )
                valid = False
            else:
                task_path = (capture / str(task_source)).resolve()
                if not task_path.is_relative_to(capture) or not task_path.is_file():
                    _add(
                        issues,
                        "public_task_source",
                        "$spec.task.source",
                        "selected public task source is missing",
                    )
                    valid = False
                else:
                    try:
                        public_task = _read_object(task_path, label="public task source")
                        validate_public_payload(public_task)
                    except (TypeError, ValueError) as exc:
                        _add(
                            issues,
                            "public_task_projection",
                            "$spec.task.source",
                            str(exc),
                        )
                        valid = False
            if (
                public_scene is not None
                and public_task is not None
                and (
                    task_spec.get("agent_count") != public_scene.get("agent_count")
                    or public_task.get("agent_count") != public_scene.get("agent_count")
                )
            ):
                _add(
                    issues,
                    "public_agent_count",
                    "$spec.task.agent_count",
                    "spec, public task, and public scene agent counts must match",
                )
                valid = False
    streams = spec.get("streams")
    assert isinstance(streams, list)
    stream_ids = [
        stream.get("stream_id")
        for stream in streams
        if isinstance(stream, Mapping) and isinstance(stream.get("stream_id"), str)
    ]
    if source_streams is None or len(stream_ids) != len(set(stream_ids)) or set(stream_ids) != set(source_streams):
        _add(
            issues,
            "candidate_stream_set",
            "$spec.streams",
            "stream IDs must exactly match the audited candidate stream set",
        )
        valid = False
    modalities = {
        stream.get("modality")
        for stream in streams
        if isinstance(stream, Mapping) and isinstance(stream.get("modality"), str)
    }
    task_spec = spec.get("task")
    profile = task_spec.get("information_profile") if isinstance(task_spec, Mapping) else None
    expected_modalities = INFORMATION_PROFILE_MODALITIES.get(profile) if isinstance(profile, str) else None
    if expected_modalities is None or modalities != expected_modalities:
        _add(
            issues,
            "information_profile_modalities",
            "$spec.streams",
            "stream modalities must exactly match the frozen information profile",
        )
        valid = False
    for index, stream in enumerate(streams):
        path = f"$spec.streams[{index}]"
        if not isinstance(stream, Mapping):
            continue
        stream_id = stream.get("stream_id")
        expected_stream = source_streams.get(stream_id) if source_streams is not None else None
        if isinstance(expected_stream, Mapping):
            for key in ("source", "modality", "timestamp_field"):
                expected_key = "path" if key == "source" else key
                if stream.get(key) != expected_stream.get(expected_key):
                    _add(
                        issues,
                        "candidate_stream_mismatch",
                        f"{path}.{key}",
                        "does not match the audited candidate stream contract",
                    )
                    valid = False
        destination = stream.get("path")
        if (
            not isinstance(destination, str)
            or not is_safe_relative_path(destination)
            or _private_or_overview(destination)
        ):
            _add(
                issues,
                "private_or_unsafe_destination",
                f"{path}.path",
                "destination must be public, relative, and not overview",
            )
            valid = False
        relative = stream.get("source")
        if not isinstance(relative, str) or not is_safe_relative_path(relative) or _private_or_overview(relative):
            _add(issues, "private_or_unsafe_source", f"{path}.source", "source must be public, capture-relative, and not overview")
            valid = False
            continue
        source = (capture / relative).resolve()
        if not source.is_relative_to(capture) or not source.is_file():
            _add(issues, "missing_source", f"{path}.source", "selected source does not exist in the capture")
            valid = False
            continue
        fields = stream.get("fields")
        chunked = False
        available: set[str] = set()
        if source.suffix.lower() == ".npz":
            try:
                with np.load(source, allow_pickle=False) as payload:
                    available = set(payload.files)
            except (OSError, ValueError, EOFError) as exc:
                _add(issues, "npz_decode", f"{path}.source", str(exc))
                valid = False
            else:
                chunked = "__rivermark_chunked_frame_archive_v1__" in available
        if chunked:
            expected_arrays = expected_stream.get("arrays") if isinstance(expected_stream, Mapping) else None
            first_shape = next(iter(expected_arrays.values())).get("shape") if isinstance(expected_arrays, Mapping) and expected_arrays else None
            expected_count = first_shape[0] if isinstance(first_shape, list) and first_shape else None
            if fields is not None or stream.get("sample_count") != expected_count:
                _add(
                    issues,
                    "chunked_archive_projection",
                    path,
                    "chunked RGB-D must be copied whole with the audited frame count",
                )
                valid = False
            continue
        if isinstance(fields, list):
            if (
                not fields
                or not all(isinstance(field, str) and field for field in fields)
                or len(fields) != len(set(fields))
                or stream.get("timestamp_field") not in fields
            ):
                _add(
                    issues,
                    "npz_fields",
                    f"{path}.fields",
                    "must be unique strings containing timestamp_field",
                )
                valid = False
                continue
            forbidden = [field for field in fields if isinstance(field, str) and forbidden_policy_key(field)]
            if forbidden:
                _add(issues, "policy_truth_leak", f"{path}.fields", f"forbidden fields selected: {sorted(forbidden)}")
                valid = False
            if source.suffix.lower() == ".npz":
                missing = sorted(set(fields) - available)
                if missing:
                    _add(issues, "npz_fields", f"{path}.fields", f"source is missing {missing}")
                    valid = False
                if isinstance(expected_stream, Mapping):
                    expected_fields = expected_stream.get("fields")
                    if isinstance(expected_fields, list) and set(fields) != set(expected_fields):
                        _add(
                            issues,
                            "candidate_stream_fields",
                            f"{path}.fields",
                            "selected fields do not match the audited candidate stream contract",
                        )
                        valid = False
        elif isinstance(expected_stream, Mapping):
            _add(
                issues,
                "candidate_stream_fields",
                f"{path}.fields",
                "non-chunked NPZ streams require the exact audited field selection",
            )
            valid = False
    rgbd_destinations = {
        stream.get("path")
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("stream_id") in {"rgb", "depth"}
    }
    if len(rgbd_destinations) != 1:
        _add(
            issues,
            "chunked_archive_destination",
            "$spec.streams",
            "RGB and depth must share one destination for the complete RGB-D archive",
        )
        valid = False
    return valid


def audit_isaac_pack_readiness(
    capture_root: Path,
    *,
    evaluator_manifest: Path | None = None,
    observation_abi: Path | None = None,
    pack_spec: Path | None = None,
    collection_protocol: Path | None = None,
    supply_chain_manifest: Path | None = None,
) -> PackReadinessReport:
    """Audit one capture without writing candidate or dataset payloads."""

    capture = capture_root.expanduser().resolve()
    issues: list[PackReadinessIssue] = []
    checks: dict[str, bool] = {}
    receipt: Mapping[str, Any] = {}
    validation: Mapping[str, Any] = {}
    receipt_hash: str | None = None
    validation_hash: str | None = None
    source_revision: str | None = None
    expected_evaluator: str | None = None
    inspection: PolicySourceInspection | None = None
    inspection_payload: Mapping[str, Any] | None = None
    candidate_streams: Mapping[str, Any] | None = None
    capture_backend: Mapping[str, Any] | None = None

    if not capture.is_dir():
        _add(issues, "capture_missing", "capture_root", "capture directory does not exist")
    else:
        try:
            receipt = _read_object(capture / "capture_receipt.json", label="capture receipt")
            validation = _read_object(capture / "independent_validation.json", label="independent validation")
            receipt_hash = sha256_file(capture / "capture_receipt.json")
            validation_hash = sha256_file(capture / "independent_validation.json")
            source_revision = receipt.get("source_revision") if isinstance(receipt.get("source_revision"), str) else None
            expected_evaluator = receipt.get("evaluator_manifest_sha256") if is_sha256(receipt.get("evaluator_manifest_sha256")) else None
        except (OSError, TypeError, ValueError) as exc:
            _add(issues, "capture_metadata", "capture_root", str(exc))
        else:
            checks["capture_receipt_schema"] = receipt.get("schema") == _CAPTURE_SCHEMA
            checks["validation_schema"] = validation.get("schema") == _VALIDATION_SCHEMA
            checks["validation_receipt_binding"] = validation.get("capture_receipt_sha256") == receipt_hash
            checks["validator_source_current"] = validation.get("validator_source_sha256") == sha256_file(Path(_isaac_validate.__file__).resolve())
            raw_backend = receipt.get("capture_backend")
            capture_backend = raw_backend if isinstance(raw_backend, Mapping) else None
            checks["capture_backend_commitment"] = bool(
                isinstance(capture_backend, Mapping)
                and set(capture_backend)
                == {"kind", "build", "sensor_physics_smoke_receipt_sha256"}
                and capture_backend.get("kind") == "isaaclab"
                and isinstance(capture_backend.get("build"), str)
                and bool(capture_backend["build"].strip())
                and is_sha256(capture_backend.get("sensor_physics_smoke_receipt_sha256"))
            )
            for name, ok in tuple(checks.items()):
                if not ok:
                    _add(issues, name, name, "required capture/validator binding is false")
            try:
                inspection = inspect_policy_observation_sources(capture)
                inspection_payload = _inspection_payload(inspection)
                checks["policy_source_contract"] = True
            except (OSError, PolicyProjectionError, ValueError) as exc:
                checks["policy_source_contract"] = False
                checks["candidate_stream_contract"] = False
                _add(issues, "policy_source_contract", "capture_root", str(exc))
            else:
                try:
                    candidate_streams = inspect_candidate_pack_streams(
                        capture,
                        inspection=inspection,
                    )
                    checks["candidate_stream_contract"] = True
                except (OSError, ValueError) as exc:
                    checks["candidate_stream_contract"] = False
                    _add(issues, "candidate_stream_contract", "capture_root", str(exc))

    if receipt:
        checks["collection_protocol"] = _audit_protocol(receipt, validation, collection_protocol, issues)
        checks["evaluator_private_revalidation"] = _audit_evaluator(
            capture, receipt, validation, evaluator_manifest, issues
        )
    else:
        checks["collection_protocol"] = False
        checks["evaluator_private_revalidation"] = False

    abi_ok, audited_abi = _audit_abi(capture, observation_abi, candidate_streams, issues)
    checks["formal_observation_abi"] = abi_ok
    if audited_abi is not None:
        checks["observation_abi_sha256"] = is_sha256(audited_abi.canonical_sha256)
    binding = receipt.get("collection_binding") if isinstance(receipt, Mapping) else None
    split = binding.get("split") if isinstance(binding, Mapping) else None
    checks["pack_spec"] = _audit_pack_spec(
        capture,
        pack_spec,
        source_revision=source_revision,
        split=split,
        audited_abi=audited_abi,
        capture_receipt_sha256=receipt_hash,
        source_streams=candidate_streams,
        capture_backend=capture_backend,
        issues=issues,
    )

    if supply_chain_manifest is None:
        _add(issues, "supply_chain_manifest_missing", "supply_chain_manifest", "release-level supply-chain proof was not supplied", scope="formal_release")
        checks["supply_chain_release"] = False
    else:
        try:
            supply = verify_supply_chain_manifest(
                supply_chain_manifest.expanduser().resolve(),
                require_release=True,
                verify_artifacts=True,
            )
        except (OSError, SupplyChainError, ValueError) as exc:
            _add(issues, "supply_chain_manifest", "supply_chain_manifest", str(exc), scope="formal_release")
            checks["supply_chain_release"] = False
        else:
            checks["supply_chain_release"] = supply.get("status") == "valid"
            if not checks["supply_chain_release"]:
                codes = sorted({str(item.get("code")) for item in supply.get("issues", []) if isinstance(item, Mapping)})
                _add(issues, "supply_chain_not_release_ready", "supply_chain_manifest", f"release checks failed: {codes}", scope="formal_release")

    return PackReadinessReport(
        capture_receipt_sha256=receipt_hash,
        independent_validation_sha256=validation_hash,
        source_revision=source_revision,
        expected_evaluator_manifest_sha256=expected_evaluator,
        policy_source_inspection=inspection_payload,
        candidate_streams=candidate_streams,
        checks=checks,
        issues=tuple(issues),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--evaluator-manifest", type=Path)
    parser.add_argument("--observation-abi", type=Path)
    parser.add_argument("--pack-spec", type=Path)
    parser.add_argument("--collection-protocol", type=Path)
    parser.add_argument("--supply-chain-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_isaac_pack_readiness(
        args.capture_root,
        evaluator_manifest=args.evaluator_manifest,
        observation_abi=args.observation_abi,
        pack_spec=args.pack_spec,
        collection_protocol=args.collection_protocol,
        supply_chain_manifest=args.supply_chain_manifest,
    )
    payload = report.as_dict()
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite readiness report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report.candidate_pack_ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
