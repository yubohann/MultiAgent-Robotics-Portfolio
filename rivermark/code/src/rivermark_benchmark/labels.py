"""Validate the versioned Rivermark object/obstacle/landmark label ABI.

The ontology is a contract for future label payloads, not a source of labels.
Public validation accepts only the learning-label partition and rejects private
evaluator vocabulary before a record can be written to a public projection.
Cross-frame validation keeps an instance identity bound to one class for the
whole episode and rejects duplicate observations in a frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import forbidden_policy_key, forbidden_policy_value_token, is_safe_relative_path, iter_tree


LABEL_ONTOLOGY_SCHEMA = "org.rivermark.benchmark.label-ontology.v1"
LABEL_RECORD_SCHEMA = "org.rivermark.benchmark.label-record.v1"
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_ID = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
_CATEGORIES = frozenset({"object", "obstacle", "landmark"})
_GEOMETRY_TYPES = frozenset({"point3d", "bbox3d", "bbox2d", "mask2d", "polyline3d"})
_FRAME_IDS = frozenset({"world", "camera_optical", "body"})
_VISIBILITY = frozenset({"visible", "partially_occluded", "occluded", "out_of_view"})
_ANNOTATION_SOURCES = frozenset({"simulator_semantic", "simulator_geometry", "human_annotated", "derived"})
_ANNOTATION_STATUS = frozenset({"unreviewed", "reviewed", "adjudicated"})


@dataclass(frozen=True)
class LabelIssue:
    code: str
    path: str
    message: str


class LabelValidationError(ValueError):
    """Raised by the CLI when a label contract is invalid."""


def _issue(issues: list[LabelIssue], code: str, path: str, message: str) -> None:
    issues.append(LabelIssue(code, path, message))


def _required(value: Mapping[str, Any], names: Sequence[str], path: str, issues: list[LabelIssue]) -> None:
    for name in names:
        if name not in value:
            _issue(issues, "required", f"{path}.{name}", "required field is missing")


def _unknown(value: Mapping[str, Any], allowed: frozenset[str], path: str, issues: list[LabelIssue]) -> None:
    for name in sorted(set(value) - allowed):
        _issue(issues, "unknown_field", f"{path}.{name}", "field is not part of the label contract")


def _id(value: Any, path: str, issues: list[LabelIssue]) -> bool:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        _issue(issues, "identifier", path, "must be a lowercase path-free identifier")
        return False
    return True


def _version(value: Any, path: str, issues: list[LabelIssue]) -> bool:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        _issue(issues, "version", path, "must be a three-part version")
        return False
    return True


def _finite(value: Any, path: str, issues: list[LabelIssue]) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _issue(issues, "finite_number", path, "must be a finite number")
        return False
    return True


def _vector(value: Any, width: int, path: str, issues: list[LabelIssue], *, positive: bool = False) -> bool:
    if not isinstance(value, list) or len(value) != width:
        _issue(issues, "vector", path, f"must contain exactly {width} numeric values")
        return False
    valid = all(_finite(item, f"{path}[{index}]", issues) for index, item in enumerate(value))
    if positive and valid and any(float(item) <= 0 for item in value):
        _issue(issues, "positive_vector", path, "all dimensions must be greater than zero")
        valid = False
    return valid


def _optional_vector(value: Any, width: int, path: str, issues: list[LabelIssue], *, positive: bool = False) -> None:
    if value is not None:
        _vector(value, width, path, issues, positive=positive)


def _sha256(value: Any, path: str, issues: list[LabelIssue]) -> bool:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _issue(issues, "sha256", path, "must be a lowercase SHA-256")
        return False
    return True


def _scan_public_vocabulary(value: Any, path: str, issues: list[LabelIssue]) -> None:
    """Reject hidden truth and private paths even when nested in a label."""

    for child_path, key, child in iter_tree(value, path):
        if key is not None and forbidden_policy_key(key):
            _issue(issues, "private_key", child_path, "evaluator/private truth is not allowed in public labels")
        if isinstance(child, str) and forbidden_policy_value_token(child) is not None:
            _issue(issues, "private_value", child_path, "evaluator/private vocabulary is not allowed in public labels")


def validate_ontology(payload: Any) -> tuple[LabelIssue, ...]:
    """Validate one ontology manifest without optional dependencies."""

    issues: list[LabelIssue] = []
    if not isinstance(payload, Mapping):
        return (LabelIssue("type", "$", "ontology must be an object"),)
    allowed = frozenset({"schema", "ontology_id", "version", "coordinate_frame", "instance_policy", "distribution", "classes"})
    _unknown(payload, allowed, "$", issues)
    _required(payload, tuple(sorted(allowed)), "$", issues)
    if payload.get("schema") != LABEL_ONTOLOGY_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {LABEL_ONTOLOGY_SCHEMA!r}")
    _id(payload.get("ontology_id"), "$.ontology_id", issues)
    _version(payload.get("version"), "$.version", issues)
    if payload.get("coordinate_frame") != "world_rivermark_v1":
        _issue(issues, "coordinate_frame", "$.coordinate_frame", "must be world_rivermark_v1")
    policy = payload.get("instance_policy")
    if not isinstance(policy, Mapping):
        _issue(issues, "type", "$.instance_policy", "must be an object")
    else:
        allowed_policy = frozenset({"identity_key", "scope", "class_immutable"})
        _unknown(policy, allowed_policy, "$.instance_policy", issues)
        _required(policy, tuple(sorted(allowed_policy)), "$.instance_policy", issues)
        expected = {"identity_key": "instance_id", "scope": "episode", "class_immutable": True}
        for key, expected_value in expected.items():
            if policy.get(key) != expected_value:
                _issue(issues, "instance_policy", f"$.instance_policy.{key}", f"must be {expected_value!r}")
    if payload.get("distribution") not in {"development_only", "cleared_public"}:
        _issue(issues, "distribution", "$.distribution", "must be development_only or cleared_public")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        _issue(issues, "classes", "$.classes", "must contain at least one class")
        classes = []
    seen: set[str] = set()
    for index, item in enumerate(classes):
        path = f"$.classes[{index}]"
        if not isinstance(item, Mapping):
            _issue(issues, "type", path, "class must be an object")
            continue
        allowed_class = frozenset({"class_id", "category", "geometry_types"})
        _unknown(item, allowed_class, path, issues)
        _required(item, tuple(sorted(allowed_class)), path, issues)
        class_id = item.get("class_id")
        if _id(class_id, f"{path}.class_id", issues):
            if class_id in seen:
                _issue(issues, "duplicate_class", f"{path}.class_id", "class_id must be unique")
            seen.add(class_id)
        if item.get("category") not in _CATEGORIES:
            _issue(issues, "category", f"{path}.category", "must be object, obstacle, or landmark")
        geometry_types = item.get("geometry_types")
        if not isinstance(geometry_types, list) or not geometry_types:
            _issue(issues, "geometry_types", f"{path}.geometry_types", "must contain at least one geometry type")
        elif len(set(geometry_types)) != len(geometry_types) or any(value not in _GEOMETRY_TYPES for value in geometry_types):
            _issue(issues, "geometry_types", f"{path}.geometry_types", "contains a duplicate or unsupported geometry type")
    _scan_public_vocabulary(payload, "$", issues)
    return tuple(issues)


def _validate_geometry(
    geometry: Any,
    *,
    path: str,
    allowed_types: frozenset[str],
    issues: list[LabelIssue],
) -> None:
    if not isinstance(geometry, Mapping):
        _issue(issues, "type", path, "geometry must be an object")
        return
    allowed = frozenset(
        {
            "type",
            "frame_id",
            "position_m",
            "center_m",
            "dimensions_m",
            "orientation_wxyz",
            "xywh_px",
            "mask_path",
            "mask_sha256",
            "width_px",
            "height_px",
            "points_m",
        }
    )
    _unknown(geometry, allowed, path, issues)
    _required(geometry, ("type", "frame_id"), path, issues)
    geometry_type = geometry.get("type")
    if geometry_type not in allowed_types:
        _issue(issues, "geometry_type", f"{path}.type", "geometry type is not declared for this class")
        return
    frame_id = geometry.get("frame_id")
    if frame_id not in _FRAME_IDS:
        _issue(issues, "frame_id", f"{path}.frame_id", "must be world, camera_optical, or body")
    if geometry_type == "point3d":
        _vector(geometry.get("position_m"), 3, f"{path}.position_m", issues)
        if set(geometry) - {"type", "frame_id", "position_m"}:
            _issue(issues, "geometry_fields", path, "point3d has fields for another geometry type")
    elif geometry_type == "bbox3d":
        _vector(geometry.get("center_m"), 3, f"{path}.center_m", issues)
        _vector(geometry.get("dimensions_m"), 3, f"{path}.dimensions_m", issues, positive=True)
        if _vector(geometry.get("orientation_wxyz"), 4, f"{path}.orientation_wxyz", issues):
            norm = math.sqrt(sum(float(value) ** 2 for value in geometry["orientation_wxyz"]))
            if abs(norm - 1.0) > 1e-3:
                _issue(issues, "quaternion", f"{path}.orientation_wxyz", "must be normalized in wxyz order")
        if set(geometry) - {"type", "frame_id", "center_m", "dimensions_m", "orientation_wxyz"}:
            _issue(issues, "geometry_fields", path, "bbox3d has fields for another geometry type")
    elif geometry_type == "bbox2d":
        if _vector(geometry.get("xywh_px"), 4, f"{path}.xywh_px", issues, positive=False):
            if float(geometry["xywh_px"][2]) < 0 or float(geometry["xywh_px"][3]) < 0:
                _issue(issues, "bbox2d", f"{path}.xywh_px", "width and height must be non-negative")
        if set(geometry) - {"type", "frame_id", "xywh_px"}:
            _issue(issues, "geometry_fields", path, "bbox2d has fields for another geometry type")
    elif geometry_type == "mask2d":
        _sha256(geometry.get("mask_sha256"), f"{path}.mask_sha256", issues)
        if not is_safe_relative_path(geometry.get("mask_path")):
            _issue(issues, "mask_path", f"{path}.mask_path", "must be a safe relative path")
        for key in ("width_px", "height_px"):
            value = geometry.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                _issue(issues, "mask_size", f"{path}.{key}", "must be a positive integer")
        if set(geometry) - {"type", "frame_id", "mask_path", "mask_sha256", "width_px", "height_px"}:
            _issue(issues, "geometry_fields", path, "mask2d has fields for another geometry type")
    elif geometry_type == "polyline3d":
        points = geometry.get("points_m")
        if not isinstance(points, list) or len(points) < 2:
            _issue(issues, "polyline", f"{path}.points_m", "must contain at least two points")
        else:
            for index, point in enumerate(points):
                _vector(point, 3, f"{path}.points_m[{index}]", issues)
        if set(geometry) - {"type", "frame_id", "points_m"}:
            _issue(issues, "geometry_fields", path, "polyline3d has fields for another geometry type")


def _validate_label(
    label: Any,
    *,
    path: str,
    classes: Mapping[str, frozenset[str]],
    issues: list[LabelIssue],
) -> str | None:
    if not isinstance(label, Mapping):
        _issue(issues, "type", path, "label must be an object")
        return None
    allowed = frozenset({"instance_id", "class_id", "partition", "geometry", "visibility", "visible_fraction", "uncertainty", "annotation"})
    _unknown(label, allowed, path, issues)
    _required(label, tuple(sorted(allowed)), path, issues)
    instance_id = label.get("instance_id")
    _id(instance_id, f"{path}.instance_id", issues)
    class_id = label.get("class_id")
    _id(class_id, f"{path}.class_id", issues)
    if class_id not in classes:
        _issue(issues, "unknown_class", f"{path}.class_id", "class_id is not declared by the ontology")
        allowed_types = frozenset()
    else:
        allowed_types = classes[class_id]
    if label.get("partition") != "learning_labels":
        _issue(issues, "partition", f"{path}.partition", "public labels must use learning_labels")
    _validate_geometry(label.get("geometry"), path=f"{path}.geometry", allowed_types=allowed_types, issues=issues)
    if label.get("visibility") not in _VISIBILITY:
        _issue(issues, "visibility", f"{path}.visibility", "unsupported visibility state")
    visible_fraction = label.get("visible_fraction")
    if not _finite(visible_fraction, f"{path}.visible_fraction", issues) or not 0 <= float(visible_fraction) <= 1:
        _issue(issues, "visible_fraction", f"{path}.visible_fraction", "must lie in [0, 1]")
    elif label.get("visibility") == "out_of_view" and float(visible_fraction) != 0:
        _issue(issues, "visibility_consistency", f"{path}.visible_fraction", "out_of_view labels must have zero visible fraction")
    uncertainty = label.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        _issue(issues, "type", f"{path}.uncertainty", "uncertainty must be an object")
    else:
        allowed_uncertainty = frozenset({"status", "confidence", "position_std_m", "extent_std_m", "pixel_std_px", "orientation_std_rad"})
        _unknown(uncertainty, allowed_uncertainty, f"{path}.uncertainty", issues)
        _required(uncertainty, ("status",), f"{path}.uncertainty", issues)
        if uncertainty.get("status") not in {"not_available", "estimated", "measured"}:
            _issue(issues, "uncertainty_status", f"{path}.uncertainty.status", "unsupported uncertainty status")
        if "confidence" in uncertainty and (
            not _finite(uncertainty.get("confidence"), f"{path}.uncertainty.confidence", issues)
            or not 0 <= float(uncertainty["confidence"]) <= 1
        ):
            _issue(issues, "uncertainty_confidence", f"{path}.uncertainty.confidence", "must lie in [0, 1]")
        _optional_vector(uncertainty.get("position_std_m"), 3, f"{path}.uncertainty.position_std_m", issues, positive=True)
        _optional_vector(uncertainty.get("extent_std_m"), 3, f"{path}.uncertainty.extent_std_m", issues, positive=True)
        _optional_vector(uncertainty.get("pixel_std_px"), 4, f"{path}.uncertainty.pixel_std_px", issues, positive=True)
        if "orientation_std_rad" in uncertainty and (
            not _finite(uncertainty.get("orientation_std_rad"), f"{path}.uncertainty.orientation_std_rad", issues)
            or float(uncertainty["orientation_std_rad"]) < 0
        ):
            _issue(issues, "uncertainty_std", f"{path}.uncertainty.orientation_std_rad", "must be non-negative")
    annotation = label.get("annotation")
    if not isinstance(annotation, Mapping):
        _issue(issues, "type", f"{path}.annotation", "annotation must be an object")
    else:
        allowed_annotation = frozenset({"source", "status", "annotator_count", "agreement", "known_error_ids"})
        _unknown(annotation, allowed_annotation, f"{path}.annotation", issues)
        _required(annotation, ("source", "status", "annotator_count"), f"{path}.annotation", issues)
        if annotation.get("source") not in _ANNOTATION_SOURCES:
            _issue(issues, "annotation_source", f"{path}.annotation.source", "unsupported annotation source")
        if annotation.get("status") not in _ANNOTATION_STATUS:
            _issue(issues, "annotation_status", f"{path}.annotation.status", "unsupported annotation status")
        count = annotation.get("annotator_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _issue(issues, "annotator_count", f"{path}.annotation.annotator_count", "must be a non-negative integer")
        if "agreement" in annotation and (
            not _finite(annotation.get("agreement"), f"{path}.annotation.agreement", issues)
            or not 0 <= float(annotation["agreement"]) <= 1
        ):
            _issue(issues, "agreement", f"{path}.annotation.agreement", "must lie in [0, 1]")
        known = annotation.get("known_error_ids", [])
        if not isinstance(known, list) or len(set(known)) != len(known):
            _issue(issues, "known_error_ids", f"{path}.annotation.known_error_ids", "must be a list of unique identifiers")
        else:
            for index, error_id in enumerate(known):
                _id(error_id, f"{path}.annotation.known_error_ids[{index}]", issues)
    return instance_id if isinstance(instance_id, str) else None


def validate_label_record(
    payload: Any,
    ontology: Mapping[str, Any],
    *,
    public: bool = True,
) -> tuple[LabelIssue, ...]:
    """Validate one frame record against an ontology manifest."""

    issues: list[LabelIssue] = list(validate_ontology(ontology))
    if not isinstance(payload, Mapping):
        return tuple(issues + [LabelIssue("type", "$", "label record must be an object")])
    allowed = frozenset(
        {
            "schema",
            "ontology_id",
            "ontology_version",
            "episode_id",
            "frame_index",
            "timestamp_ns",
            "source_capture_receipt_sha256",
            "source_revision",
            "labels",
        }
    )
    _unknown(payload, allowed, "$", issues)
    _required(payload, tuple(sorted(allowed)), "$", issues)
    if payload.get("schema") != LABEL_RECORD_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {LABEL_RECORD_SCHEMA!r}")
    ontology_id = ontology.get("ontology_id")
    ontology_version = ontology.get("version")
    if payload.get("ontology_id") != ontology_id:
        _issue(issues, "ontology_binding", "$.ontology_id", "does not match the ontology manifest")
    if payload.get("ontology_version") != ontology_version:
        _issue(issues, "ontology_binding", "$.ontology_version", "does not match the ontology manifest")
    _id(payload.get("episode_id"), "$.episode_id", issues)
    for key in ("frame_index", "timestamp_ns"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _issue(issues, "integer", f"$.{key}", "must be a non-negative integer")
    if "source_capture_receipt_sha256" in payload:
        _sha256(payload["source_capture_receipt_sha256"], "$.source_capture_receipt_sha256", issues)
    if "source_revision" in payload and (not isinstance(payload["source_revision"], str) or not _REVISION.fullmatch(payload["source_revision"])):
        _issue(issues, "source_revision", "$.source_revision", "must be a lowercase Git revision")
    classes = {
        item["class_id"]: frozenset(item["geometry_types"])
        for item in ontology.get("classes", [])
        if isinstance(item, Mapping) and isinstance(item.get("class_id"), str) and isinstance(item.get("geometry_types"), list)
    }
    labels = payload.get("labels")
    seen: set[str] = set()
    if not isinstance(labels, list):
        _issue(issues, "labels", "$.labels", "must be an array")
    else:
        for index, label in enumerate(labels):
            instance_id = _validate_label(label, path=f"$.labels[{index}]", classes=classes, issues=issues)
            if instance_id is not None:
                if instance_id in seen:
                    _issue(issues, "duplicate_instance", f"$.labels[{index}].instance_id", "an instance may occur once per frame")
                seen.add(instance_id)
    if public:
        _scan_public_vocabulary(payload, "$", issues)
    return tuple(issues)


def validate_label_sequence(
    records: Iterable[Mapping[str, Any]],
    ontology: Mapping[str, Any],
    *,
    public: bool = True,
) -> tuple[LabelIssue, ...]:
    """Validate frame ordering and instance identity across one episode."""

    issues: list[LabelIssue] = []
    seen_classes: dict[str, str] = {}
    seen_frames: set[int] = set()
    previous_timestamp: int | None = None
    episode_id: str | None = None
    for index, record in enumerate(records):
        issues.extend(validate_label_record(record, ontology, public=public))
        if not isinstance(record, Mapping):
            continue
        frame = record.get("frame_index")
        timestamp = record.get("timestamp_ns")
        current_episode = record.get("episode_id")
        if episode_id is None and isinstance(current_episode, str):
            episode_id = current_episode
        elif isinstance(current_episode, str) and episode_id != current_episode:
            _issue(issues, "episode_binding", f"$[{index}].episode_id", "all records must belong to one episode")
        if isinstance(frame, int) and not isinstance(frame, bool):
            if frame in seen_frames:
                _issue(issues, "duplicate_frame", f"$[{index}].frame_index", "frame_index must be unique")
            seen_frames.add(frame)
        if isinstance(timestamp, int) and not isinstance(timestamp, bool):
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                _issue(issues, "timestamp_order", f"$[{index}].timestamp_ns", "timestamps must be strictly increasing")
            previous_timestamp = timestamp
        labels = record.get("labels")
        if isinstance(labels, list):
            for label_index, label in enumerate(labels):
                if not isinstance(label, Mapping):
                    continue
                instance_id = label.get("instance_id")
                class_id = label.get("class_id")
                if not isinstance(instance_id, str) or not isinstance(class_id, str):
                    continue
                old_class = seen_classes.setdefault(instance_id, class_id)
                if old_class != class_id:
                    _issue(
                        issues,
                        "instance_class_change",
                        f"$[{index}].labels[{label_index}].class_id",
                        "an instance_id cannot change class within an episode",
                    )
    return tuple(issues)


def canonical_ontology_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def ontology_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_ontology_bytes(payload)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelValidationError("cannot read ontology JSON input") from exc


def _iter_records(path: Path) -> Iterable[Mapping[str, Any]]:
    """Read one JSON object, an object with ``records``, or JSONL incrementally."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            first = stream.read(1)
            if not first:
                raise LabelValidationError("annotation input is empty")
            stream.seek(0)
            # JSONL is selected by its conventional extension. A regular JSON
            # object/list is parsed as one document; this avoids guessing from
            # a leading '{', which is also the first byte of every JSONL row.
            if path.suffix.lower() not in {".jsonl", ".ndjson"}:
                payload = json.load(stream)
                if isinstance(payload, list):
                    records = payload
                elif isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
                    records = payload["records"]
                else:
                    records = [payload]
                for record in records:
                    if not isinstance(record, Mapping):
                        raise LabelValidationError("annotation JSON must contain objects")
                    yield record
                return
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LabelValidationError(f"invalid JSONL record at line {line_number}") from exc
                if not isinstance(record, Mapping):
                    raise LabelValidationError(f"JSONL record at line {line_number} must be an object")
                yield record
    except LabelValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LabelValidationError("cannot read annotation input") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Rivermark label ontology and annotation records")
    parser.add_argument("ontology", type=Path)
    parser.add_argument("records", type=Path)
    args = parser.parse_args(argv)
    try:
        ontology = _load_json(args.ontology)
        record_count = 0

        def counted_records() -> Iterable[Mapping[str, Any]]:
            nonlocal record_count
            for record in _iter_records(args.records):
                record_count += 1
                yield record

        records = counted_records()
    except LabelValidationError as exc:
        print(
            json.dumps(
                {
                    "schema": "org.rivermark.benchmark.label-validation-report.v1",
                    "status": "invalid",
                    "issues": [{"code": "input", "path": "$", "message": str(exc)}],
                    "claim_boundary": "label ABI validation only; no formal episode or redistribution claim",
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
        )
        return 1
    ontology_issues = validate_ontology(ontology)
    try:
        issues = tuple(ontology_issues) + (() if ontology_issues else validate_label_sequence(records, ontology))
    except LabelValidationError as exc:
        print(
            json.dumps(
                {
                    "schema": "org.rivermark.benchmark.label-validation-report.v1",
                    "status": "invalid",
                    "issues": [{"code": "input", "path": "$", "message": str(exc)}],
                    "claim_boundary": "label ABI validation only; no formal episode or redistribution claim",
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
        )
        return 1
    report = {
        "schema": "org.rivermark.benchmark.label-validation-report.v1",
        "status": "passed" if not issues else "failed",
        "ontology_sha256": ontology_sha256(ontology) if isinstance(ontology, Mapping) else None,
        "record_count": record_count,
        "issues": [issue.__dict__ for issue in issues],
        "claim_boundary": "label ABI validation only; no formal episode or redistribution claim",
    }
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
