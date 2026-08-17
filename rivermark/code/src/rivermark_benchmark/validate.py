"""Zero-dependency manifest and policy-leakage validator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .abi import observation_abi_sha256, validate_observation_abi
from .collection_protocol import validate_collection_binding
from .schema import (
    ALLOWED_OBSERVATION_SCOPES,
    ALLOWED_SPLITS,
    EPISODE_SCHEMA,
    INFORMATION_PROFILE_MODALITIES,
    forbidden_policy_key,
    forbidden_policy_value_token,
    is_safe_relative_path,
    is_sha256,
    iter_tree,
)


_ROOT_KEYS = frozenset(
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
        "policy_visible",
        "learning_labels",
        "evaluator_private",
        "provenance",
        "quality",
        "collection_binding",
    }
)

_STREAM_KEYS = frozenset(
    {
        "stream_id",
        "partition",
        "modality",
        "media_type",
        "sample_count",
        "timestamp_field",
        "agent_id",
        "path",
        "sha256",
        "path_template",
        "content_hash_index_path",
        "content_hash_index_sha256",
    }
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


def _issue(issues: list[ValidationIssue], code: str, path: str, message: str) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message))


def _mapping(
    value: Any, *, path: str, issues: list[ValidationIssue]
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        _issue(issues, "type", path, "expected an object")
        return None
    return value


def _required(
    value: Mapping[str, Any], names: Sequence[str], *, path: str, issues: list[ValidationIssue]
) -> None:
    for name in names:
        if name not in value:
            _issue(issues, "required", f"{path}.{name}", "required field is missing")


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str] | frozenset[str], *, path: str,
    issues: list[ValidationIssue]
) -> None:
    for name in value:
        if name not in allowed:
            _issue(issues, "unknown_field", f"{path}.{name}", "field is not part of manifest v1")


def _string_list(
    value: Any, *, path: str, allow_empty: bool, issues: list[ValidationIssue]
) -> list[str] | None:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and bool(item.strip()) for item in value)
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        _issue(issues, "string_list", path, f"must be a {qualifier} list of non-empty strings")
        return None
    if len(set(value)) != len(value):
        _issue(issues, "duplicate_value", path, "values must be unique")
    return value


def _contained_path(base_dir: Path, relative: object) -> Path | None:
    """Resolve an episode-relative path without following it outside the episode."""

    if not is_safe_relative_path(relative):
        return None
    try:
        root = base_dir.resolve()
        candidate = (root / str(relative)).resolve()
        return candidate if candidate.is_relative_to(root) else None
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_bound_file(
    relative: object, digest: object, *, path: str, base_dir: Path | None,
    check_files: bool, issues: list[ValidationIssue]
) -> None:
    if not check_files:
        return
    if base_dir is None:
        _issue(issues, "missing_base_dir", path, "file checking requires an episode base directory")
        return
    file_path = _contained_path(base_dir, relative)
    if file_path is None:
        _issue(issues, "path_escape", path, "resolved path escapes the episode directory")
    elif not file_path.is_file():
        _issue(issues, "missing_file", path, f"missing stream file: {file_path}")
    elif is_sha256(digest) and _sha256_file(file_path) != digest:
        _issue(issues, "file_hash", path, "stream file hash does not match")


def _scan_policy_value(value: Any, *, root_path: str, issues: list[ValidationIssue]) -> None:
    for path, raw_key, child in iter_tree(value, root_path):
        if raw_key is not None and forbidden_policy_key(raw_key):
            _issue(
                issues,
                "policy_truth_leak",
                path,
                f"policy-visible key {raw_key!r} exposes evaluator, future, reward, seed, or target truth",
            )
        if isinstance(child, str):
            token = forbidden_policy_value_token(child)
            if token is not None:
                _issue(
                    issues,
                    "policy_truth_reference",
                    path,
                    f"policy-visible string references forbidden provenance token {token!r}",
                )


def _validate_policy_visible(value: Any, issues: list[ValidationIssue]) -> None:
    policy = _mapping(value, path="$.policy_visible", issues=issues)
    if policy is None:
        return
    _scan_policy_value(policy, root_path="$.policy_visible", issues=issues)


def _validate_streams(
    value: Any, *, policy_modalities: set[str] | None, base_dir: Path | None,
    check_files: bool, issues: list[ValidationIssue]
) -> None:
    if not isinstance(value, list) or not value:
        _issue(issues, "streams", "$.streams", "streams must be a non-empty list")
        return
    seen: set[str] = set()
    for index, raw in enumerate(value):
        path = f"$.streams[{index}]"
        stream = _mapping(raw, path=path, issues=issues)
        if stream is None:
            continue
        _reject_unknown(stream, _STREAM_KEYS, path=path, issues=issues)
        _required(
            stream,
            ("stream_id", "partition", "modality", "media_type", "sample_count", "timestamp_field"),
            path=path,
            issues=issues,
        )
        stream_id = stream.get("stream_id")
        if not isinstance(stream_id, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{1,127}", stream_id
        ):
            _issue(issues, "stream_id", f"{path}.stream_id", "stream_id has invalid syntax")
        elif stream_id in seen:
            _issue(issues, "duplicate_stream", f"{path}.stream_id", "stream_id must be unique")
        else:
            seen.add(stream_id)
        partition = stream.get("partition")
        if not isinstance(partition, str) or partition not in {
            "policy_visible", "learning_labels", "evaluator_private"
        }:
            _issue(issues, "stream_partition", f"{path}.partition", "unknown access partition")
        modality = stream.get("modality")
        for key in ("modality", "media_type", "timestamp_field"):
            field = stream.get(key)
            if not isinstance(field, str) or not field.strip() or len(field) > 128:
                _issue(issues, "stream_field", f"{path}.{key}", "must be a 1-128 character string")
        agent_id = stream.get("agent_id")
        if agent_id is not None and (
            not isinstance(agent_id, int) or isinstance(agent_id, bool) or not 0 <= agent_id <= 31
        ):
            _issue(issues, "agent_id", f"{path}.agent_id", "agent_id must be in [0, 31]")
        if partition == "policy_visible":
            _scan_policy_value(stream, root_path=path, issues=issues)
            if isinstance(modality, str) and forbidden_policy_key(modality):
                _issue(
                    issues,
                    "policy_truth_leak",
                    f"{path}.modality",
                    "policy-visible modality denotes evaluator, reward, seed, or target truth",
                )
            if policy_modalities is not None and modality not in policy_modalities:
                _issue(
                    issues,
                    "undeclared_policy_modality",
                    f"{path}.modality",
                    "policy-visible stream modality is absent from policy_visible.modalities",
                )
        count = stream.get("sample_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            _issue(issues, "sample_count", f"{path}.sample_count", "sample_count must be a non-negative integer")

        concrete_keys = {"path", "sha256"}
        template_keys = {
            "path_template",
            "content_hash_index_path",
            "content_hash_index_sha256",
        }
        has_concrete = bool(concrete_keys & stream.keys())
        has_template = bool(template_keys & stream.keys())
        if has_concrete == has_template:
            _issue(
                issues,
                "stream_binding",
                path,
                "stream must use exactly one concrete path/hash or template/hash-index binding",
            )
            continue
        if has_concrete:
            if not concrete_keys <= stream.keys():
                _issue(issues, "stream_binding", path, "concrete binding requires path and sha256")
            relative = stream.get("path")
            digest = stream.get("sha256")
            if not is_safe_relative_path(relative):
                _issue(issues, "unsafe_path", f"{path}.path", "stream path must be relative without '..'")
            if not is_sha256(digest):
                _issue(issues, "sha256", f"{path}.sha256", "stream sha256 must be 64 lowercase hex digits")
            _check_bound_file(
                relative,
                digest,
                path=f"{path}.path",
                base_dir=base_dir,
                check_files=check_files,
                issues=issues,
            )
        else:
            if not template_keys <= stream.keys():
                _issue(
                    issues,
                    "stream_binding",
                    path,
                    "template binding requires path_template and complete content-hash index",
                )
            for key in ("path_template", "content_hash_index_path"):
                if not is_safe_relative_path(stream.get(key)):
                    _issue(issues, "unsafe_path", f"{path}.{key}", "template binding path is unsafe")
            if "{agent_id}" not in str(stream.get("path_template", "")):
                _issue(
                    issues,
                    "path_template",
                    f"{path}.path_template",
                    "per-agent path_template must include '{agent_id}'",
                )
            if not is_sha256(stream.get("content_hash_index_sha256")):
                _issue(
                    issues,
                    "sha256",
                    f"{path}.content_hash_index_sha256",
                    "content hash index sha256 must be 64 lowercase hex digits",
                )
            _check_bound_file(
                stream.get("content_hash_index_path"),
                stream.get("content_hash_index_sha256"),
                path=f"{path}.content_hash_index_path",
                base_dir=base_dir,
                check_files=check_files,
                issues=issues,
            )


def _validate_observation_abi_ref(
    value: Any,
    *,
    base_dir: Path | None,
    check_files: bool,
    issues: list[ValidationIssue],
) -> None:
    path = "$.observation_abi"
    abi = _mapping(value, path=path, issues=issues)
    if abi is None:
        return
    _reject_unknown(abi, frozenset({"path", "sha256"}), path=path, issues=issues)
    _required(abi, ("path", "sha256"), path=path, issues=issues)
    if not is_safe_relative_path(abi.get("path")):
        _issue(issues, "unsafe_path", f"{path}.path", "observation ABI path is unsafe")
    if not is_sha256(abi.get("sha256")):
        _issue(issues, "sha256", f"{path}.sha256", "observation ABI hash must be SHA-256")
    if not check_files or base_dir is None:
        return
    candidate = _contained_path(base_dir, abi.get("path"))
    if candidate is None or not candidate.is_file():
        _issue(issues, "missing_file", f"{path}.path", "missing observation ABI file")
        return
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _issue(issues, "abi_json", path, f"cannot read observation ABI: {exc}")
        return
    abi_issues = validate_observation_abi(payload)
    for abi_issue in abi_issues:
        suffix = abi_issue.path.lstrip("$").lstrip(".")
        _issue(issues, f"abi_{abi_issue.code}", f"{path}.{suffix}" if suffix else path, abi_issue.message)
    if not abi_issues and is_sha256(abi.get("sha256")):
        try:
            actual = observation_abi_sha256(payload)
        except Exception as exc:  # validation above should make this unreachable; fail closed if it is not
            _issue(issues, "abi_hash", path, f"cannot canonicalize observation ABI: {exc}")
        else:
            if actual != abi.get("sha256"):
                _issue(issues, "abi_hash", f"{path}.sha256", "canonical observation ABI hash does not match")


def validate_episode_manifest(
    manifest: Any, *, base_dir: Path | None = None, check_files: bool = False
) -> tuple[ValidationIssue, ...]:
    """Validate the v1 metadata ABI, provenance, and policy information boundary."""

    issues: list[ValidationIssue] = []
    root = _mapping(manifest, path="$", issues=issues)
    if root is None:
        return tuple(issues)
    _reject_unknown(root, _ROOT_KEYS, path="$", issues=issues)
    _required(
        root,
        (
            "schema",
            "dataset_version",
            "episode_id",
            "split",
            "layout",
            "task",
            "timebase",
            "coordinate_frames",
            "streams",
            "policy_visible",
            "learning_labels",
            "evaluator_private",
            "provenance",
            "quality",
        ),
        path="$",
        issues=issues,
    )
    if root.get("schema") != EPISODE_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {EPISODE_SCHEMA!r}")
    dataset_version = root.get("dataset_version")
    if not isinstance(dataset_version, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", dataset_version
    ):
        _issue(issues, "dataset_version", "$.dataset_version", "expected a semantic version")
    episode_id = root.get("episode_id")
    if not isinstance(episode_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", episode_id
    ):
        _issue(issues, "episode_id", "$.episode_id", "episode_id has invalid syntax")
    split = root.get("split")
    if not isinstance(split, str) or split not in ALLOWED_SPLITS:
        _issue(issues, "split", "$.split", "unknown benchmark split")
    collection_binding = root.get("collection_binding")
    if collection_binding is not None:
        for binding_issue in validate_collection_binding(collection_binding):
            suffix = binding_issue.path.lstrip("$").lstrip(".")
            _issue(
                issues,
                f"collection_{binding_issue.code}",
                f"$.collection_binding.{suffix}" if suffix else "$.collection_binding",
                binding_issue.message,
            )
        if isinstance(collection_binding, Mapping) and collection_binding.get("split") != split:
            _issue(
                issues,
                "collection_split",
                "$.collection_binding.split",
                "must match the episode split",
            )

    layout = _mapping(root.get("layout"), path="$.layout", issues=issues)
    if layout is not None:
        _reject_unknown(
            layout,
            frozenset(
                {
                    "layout_id",
                    "layout_hash",
                    "layout_lineage_hash",
                    "scene_manifest_ref",
                    "scene_manifest_sha256",
                }
            ),
            path="$.layout",
            issues=issues,
        )
        _required(
            layout,
            (
                "layout_id",
                "layout_hash",
                "layout_lineage_hash",
                "scene_manifest_ref",
                "scene_manifest_sha256",
            ),
            path="$.layout",
            issues=issues,
        )
        for key in ("layout_hash", "layout_lineage_hash"):
            if not is_sha256(layout.get(key)):
                _issue(issues, "sha256", f"$.layout.{key}", "expected 64 lowercase hex digits")
        if not is_safe_relative_path(layout.get("scene_manifest_ref")):
            _issue(issues, "unsafe_path", "$.layout.scene_manifest_ref", "scene manifest path is unsafe")
        if not is_sha256(layout.get("scene_manifest_sha256")):
            _issue(issues, "sha256", "$.layout.scene_manifest_sha256", "invalid scene manifest hash")
        _check_bound_file(
            layout.get("scene_manifest_ref"),
            layout.get("scene_manifest_sha256"),
            path="$.layout.scene_manifest_ref",
            base_dir=base_dir,
            check_files=check_files,
            issues=issues,
        )
        layout_id = layout.get("layout_id")
        if not isinstance(layout_id, str) or not 1 <= len(layout_id) <= 128:
            _issue(issues, "layout_id", "$.layout.layout_id", "layout_id must be 1-128 characters")

    task = _mapping(root.get("task"), path="$.task", issues=issues)
    profile = None
    if task is not None:
        _reject_unknown(
            task,
            frozenset(
                {
                    "task_id",
                    "task_variant_id",
                    "task_spec_ref",
                    "task_spec_sha256",
                    "information_profile",
                    "observation_scope",
                    "agent_count",
                }
            ),
            path="$.task",
            issues=issues,
        )
        _required(
            task,
            (
                "task_id",
                "task_variant_id",
                "task_spec_ref",
                "task_spec_sha256",
                "information_profile",
                "observation_scope",
                "agent_count",
            ),
            path="$.task",
            issues=issues,
        )
        profile = task.get("information_profile")
        if not isinstance(profile, str) or profile not in INFORMATION_PROFILE_MODALITIES:
            _issue(issues, "information_profile", "$.task.information_profile", "unknown profile")
        observation_scope = task.get("observation_scope")
        if not isinstance(observation_scope, str) or observation_scope not in ALLOWED_OBSERVATION_SCOPES:
            _issue(issues, "observation_scope", "$.task.observation_scope", "unknown observation scope")
        if not is_safe_relative_path(task.get("task_spec_ref")):
            _issue(issues, "unsafe_path", "$.task.task_spec_ref", "task spec path is unsafe")
        if not is_sha256(task.get("task_spec_sha256")):
            _issue(issues, "sha256", "$.task.task_spec_sha256", "invalid task spec hash")
        _check_bound_file(
            task.get("task_spec_ref"),
            task.get("task_spec_sha256"),
            path="$.task.task_spec_ref",
            base_dir=base_dir,
            check_files=check_files,
            issues=issues,
        )
        count = task.get("agent_count")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32:
            _issue(issues, "agent_count", "$.task.agent_count", "agent_count must be in [1, 32]")
        for key, pattern in (
            ("task_id", r"[a-z0-9][a-z0-9._-]{1,63}"),
            ("task_variant_id", r"[a-z0-9][a-z0-9._-]{2,127}"),
        ):
            field = task.get(key)
            if not isinstance(field, str) or not re.fullmatch(pattern, field):
                _issue(issues, key, f"$.task.{key}", f"{key} has invalid syntax")

    timebase = _mapping(root.get("timebase"), path="$.timebase", issues=issues)
    if timebase is not None:
        _reject_unknown(
            timebase,
            frozenset({"unit", "physics_dt_ns", "proprioception_period_ns", "camera_period_ns"}),
            path="$.timebase",
            issues=issues,
        )
        _required(
            timebase,
            ("unit", "physics_dt_ns", "proprioception_period_ns", "camera_period_ns"),
            path="$.timebase",
            issues=issues,
        )
        if timebase.get("unit") != "ns":
            _issue(issues, "time_unit", "$.timebase.unit", "canonical time unit must be ns")
        for key in ("physics_dt_ns", "proprioception_period_ns", "camera_period_ns"):
            value = timebase.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                _issue(issues, "period", f"$.timebase.{key}", "period must be a positive integer")

    frames = _mapping(root.get("coordinate_frames"), path="$.coordinate_frames", issues=issues)
    expected_frames = {
        "handedness": "right",
        "world_up_axis": "+z",
        "world_frame_convention": "x_east_y_north_z_up",
        "body_frame_convention": "flu",
        "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "m",
        "angle_unit": "rad",
        "quaternion_order": "wxyz",
        "transform_notation": "T_parent_child",
    }
    if frames is not None:
        _reject_unknown(frames, frozenset(expected_frames), path="$.coordinate_frames", issues=issues)
        _required(frames, tuple(expected_frames), path="$.coordinate_frames", issues=issues)
        for key, expected in expected_frames.items():
            if frames.get(key) != expected:
                _issue(issues, "coordinate_frame", f"$.coordinate_frames.{key}", f"expected {expected!r}")

    if "observation_abi" in root:
        _validate_observation_abi_ref(
            root.get("observation_abi"),
            base_dir=base_dir,
            check_files=check_files,
            issues=issues,
        )

    policy = _mapping(root.get("policy_visible"), path="$.policy_visible", issues=issues)
    policy_modalities: set[str] | None = None
    if policy is not None:
        _required(policy, ("information_profile", "modalities"), path="$.policy_visible", issues=issues)
        if profile is not None and policy.get("information_profile") != profile:
            _issue(
                issues,
                "profile_mismatch",
                "$.policy_visible.information_profile",
                "task and policy-visible information profiles differ",
            )
        modalities = policy.get("modalities")
        checked_modalities = _string_list(
            modalities,
            path="$.policy_visible.modalities",
            allow_empty=False,
            issues=issues,
        )
        if checked_modalities is not None:
            policy_modalities = set(checked_modalities)
        if (
            policy_modalities is not None
            and isinstance(profile, str)
            and profile in INFORMATION_PROFILE_MODALITIES
        ):
            expected = INFORMATION_PROFILE_MODALITIES[profile]
            missing = expected - policy_modalities
            extra = policy_modalities - expected
            if missing:
                _issue(
                    issues,
                    "profile_modalities",
                    "$.policy_visible.modalities",
                    f"profile is missing required modalities: {sorted(missing)}",
                )
            if extra:
                _issue(
                    issues,
                    "profile_modalities",
                    "$.policy_visible.modalities",
                    f"profile contains undeclared extra modalities: {sorted(extra)}",
                )
    _validate_policy_visible(root.get("policy_visible"), issues)
    _validate_streams(
        root.get("streams"),
        policy_modalities=policy_modalities,
        base_dir=base_dir,
        check_files=check_files,
        issues=issues,
    )

    provenance = _mapping(root.get("provenance"), path="$.provenance", issues=issues)
    if provenance is not None:
        provenance_keys = frozenset(
            {
                "route_conditioning",
                "observation_generation",
                "collector_type",
                "policy_id",
                "code_commit",
                "simulator_build",
                "scene_asset_license_status",
            }
        )
        _reject_unknown(provenance, provenance_keys, path="$.provenance", issues=issues)
        _required(provenance, tuple(provenance_keys), path="$.provenance", issues=issues)
        if provenance.get("route_conditioning") != "public_only":
            _issue(
                issues,
                "target_conditioned_route",
                "$.provenance.route_conditioning",
                "formal data must be generated from public-only routes",
            )
        if provenance.get("observation_generation") != "online_runtime":
            _issue(
                issues,
                "posthoc_observation",
                "$.provenance.observation_generation",
                "formal observations must be captured online, not rendered post-hoc",
            )
        collector_type = provenance.get("collector_type")
        if not isinstance(collector_type, str) or collector_type not in {
            "classical", "random", "qd", "rl", "human", "scripted", "mixed"
        }:
            _issue(issues, "collector_type", "$.provenance.collector_type", "unknown collector type")
        for key in ("policy_id", "simulator_build"):
            value = provenance.get(key)
            if not isinstance(value, str) or not 1 <= len(value) <= 256:
                _issue(issues, key, f"$.provenance.{key}", "must be 1-256 characters")
        commit = provenance.get("code_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
            _issue(issues, "code_commit", "$.provenance.code_commit", "invalid source commit")
        license_status = provenance.get("scene_asset_license_status")
        if not isinstance(license_status, str) or license_status not in {
            "pending", "internal_only", "redistribution_cleared"
        }:
            _issue(
                issues,
                "license_status",
                "$.provenance.scene_asset_license_status",
                "unknown asset license status",
            )

    learning = _mapping(root.get("learning_labels"), path="$.learning_labels", issues=issues)
    private = _mapping(root.get("evaluator_private"), path="$.evaluator_private", issues=issues)
    if learning is not None:
        learning_keys = frozenset({"distributed", "modalities"})
        _reject_unknown(learning, learning_keys, path="$.learning_labels", issues=issues)
        _required(learning, tuple(learning_keys), path="$.learning_labels", issues=issues)
        if not isinstance(learning.get("distributed"), bool):
            _issue(issues, "distributed", "$.learning_labels.distributed", "must be boolean")
        _string_list(
            learning.get("modalities"),
            path="$.learning_labels.modalities",
            allow_empty=True,
            issues=issues,
        )
    if private is not None:
        private_keys = frozenset({"distributed", "server_only", "manifest_sha256"})
        _reject_unknown(private, private_keys, path="$.evaluator_private", issues=issues)
        _required(private, tuple(private_keys), path="$.evaluator_private", issues=issues)
        for key in ("distributed", "server_only"):
            if not isinstance(private.get(key), bool):
                _issue(issues, key, f"$.evaluator_private.{key}", "must be boolean")
        if not is_sha256(private.get("manifest_sha256")):
            _issue(issues, "sha256", "$.evaluator_private.manifest_sha256", "invalid evaluator manifest hash")
        if isinstance(split, str) and split in {"blind_test", "ood_test"} and (
            private.get("distributed") is not False or private.get("server_only") is not True
        ):
            _issue(
                issues,
                "blind_truth_distribution",
                "$.evaluator_private",
                "blind/OOD evaluator truth must be server-only and not distributed",
            )
    if (
        learning is not None
        and isinstance(split, str)
        and split in {"blind_test", "ood_test"}
        and learning.get("distributed") is True
    ):
        _issue(
            issues,
            "blind_label_distribution",
            "$.learning_labels.distributed",
            "blind/OOD learning labels cannot be distributed",
        )

    quality = _mapping(root.get("quality"), path="$.quality", issues=issues)
    if quality is not None:
        quality_keys = frozenset(
            {
                "recording_valid",
                "task_success",
                "invalid_reasons",
                "frame_completeness_ratio",
                "timestamp_monotonic",
                "pose_closure_max_error_m",
            }
        )
        _reject_unknown(quality, quality_keys, path="$.quality", issues=issues)
        _required(quality, tuple(quality_keys), path="$.quality", issues=issues)
        valid = quality.get("recording_valid")
        reasons = quality.get("invalid_reasons")
        if not isinstance(valid, bool):
            _issue(issues, "recording_valid", "$.quality.recording_valid", "must be boolean")
        if not isinstance(reasons, list) or not all(isinstance(item, str) and item for item in reasons):
            _issue(issues, "invalid_reasons", "$.quality.invalid_reasons", "must be a list of strings")
        elif len(set(reasons)) != len(reasons):
            _issue(issues, "duplicate_value", "$.quality.invalid_reasons", "values must be unique")
        elif valid and reasons:
            _issue(issues, "quality_contradiction", "$.quality", "valid recording cannot have invalid reasons")
        elif valid is False and not reasons:
            _issue(issues, "quality_contradiction", "$.quality", "invalid recording needs at least one reason")
        ratio = quality.get("frame_completeness_ratio")
        if (
            not isinstance(ratio, (int, float))
            or isinstance(ratio, bool)
            or not math.isfinite(ratio)
            or not 0.0 <= ratio <= 1.0
        ):
            _issue(issues, "frame_completeness", "$.quality.frame_completeness_ratio", "must be in [0, 1]")
        for key in ("task_success", "timestamp_monotonic"):
            if not isinstance(quality.get(key), bool):
                _issue(issues, key, f"$.quality.{key}", "must be boolean")
        closure = quality.get("pose_closure_max_error_m")
        if (
            not isinstance(closure, (int, float))
            or isinstance(closure, bool)
            or not math.isfinite(closure)
            or closure < 0.0
        ):
            _issue(
                issues,
                "pose_closure",
                "$.quality.pose_closure_max_error_m",
                "must be a non-negative number",
            )

    return tuple(issues)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="root for all manifest-relative paths (defaults to manifest directory)",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 2
    issues = validate_episode_manifest(
        payload,
        base_dir=(args.dataset_root.resolve() if args.dataset_root else manifest_path.parent),
        check_files=bool(args.check_files),
    )
    report = {
        "status": "valid" if not issues else "invalid",
        "manifest": str(manifest_path),
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if args.as_json or issues:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"VALID {manifest_path}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
