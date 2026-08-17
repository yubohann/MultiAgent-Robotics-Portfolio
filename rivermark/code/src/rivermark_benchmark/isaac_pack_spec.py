"""Build a closed-world pack specification without packing or admitting data.

The output is an external, capture-bound control-plane artifact. It selects the
exact eight public streams audited by ``policy_projection`` and binds an
external observation ABI. It never copies capture payloads, writes a formal
receipt, or changes the dataset index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .abi import observation_abi_sha256, validate_formal_observation_abi
from .isaac_pack import PACK_SPEC_SCHEMA_V2, validate_isaac_pack_spec
from .isaac_public_manifest import (
    build_public_scene_manifest,
    public_manifest_sha256,
    validate_public_payload,
)
from .policy_projection import (
    PolicyProjectionError,
    inspect_candidate_pack_streams,
    validate_candidate_abi_sources,
)

_EPISODE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_DATASET_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STREAM_IDS = frozenset(
    {"actions", "state", "task", "messages", "rgb", "depth", "lidar", "imu"}
)
_COORDINATE_FRAME_KEYS = frozenset(
    {
        "handedness",
        "world_up_axis",
        "world_frame_convention",
        "body_frame_convention",
        "camera_optical_frame_convention",
        "length_unit",
        "angle_unit",
        "quaternion_order",
        "transform_notation",
    }
)


class IsaacPackSpecError(ValueError):
    """Raised when public capture evidence cannot support a truthful pack spec."""


def _object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsaacPackSpecError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise IsaacPackSpecError(f"{label} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IsaacPackSpecError(f"{label} must be an object")
    return value


def _require_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IsaacPackSpecError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IsaacPackSpecError(f"{label} must be SHA-256")
    return value


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    if not result:
        raise IsaacPackSpecError("could not derive a safe episode identifier")
    return result


def _default_episode_id(
    layout_id: str,
    split: str,
    cell_id: str,
    episode_index: int,
    capture_receipt_sha256: str,
) -> str:
    prefix = _slug(
        f"rivermark-{layout_id}-{split}-{cell_id}-{episode_index:04d}"
    )
    suffix = capture_receipt_sha256[:12]
    available = 127 - len(suffix)
    return f"{prefix[:available].rstrip('-._')}-{suffix}"


def _stream_specs(source_streams: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(source_streams) != _STREAM_IDS:
        raise IsaacPackSpecError("source contract must contain the exact eight T1 streams")
    packed: list[dict[str, Any]] = []
    for stream_id in sorted(source_streams):
        stream = _require_mapping(
            source_streams[stream_id], label=f"source stream {stream_id}"
        )
        source = _require_text(stream.get("path"), label=f"{stream_id}.path")
        fields = stream.get("fields")
        arrays = _require_mapping(stream.get("arrays"), label=f"{stream_id}.arrays")
        entry: dict[str, Any] = {
            "stream_id": stream_id,
            "partition": "policy_visible",
            "modality": _require_text(
                stream.get("modality"), label=f"{stream_id}.modality"
            ),
            "media_type": "application/x-npz",
            "timestamp_field": _require_text(
                stream.get("timestamp_field"), label=f"{stream_id}.timestamp_field"
            ),
            "source": source,
            "path": (
                "streams/onboard_rgbd.npz"
                if source == "sensors/onboard_rgbd.npz"
                else f"streams/{stream_id}.npz"
            ),
        }
        if source == "sensors/onboard_rgbd.npz":
            timestamps = _require_mapping(
                arrays.get("timestamps_ns"), label=f"{stream_id}.arrays.timestamps_ns"
            )
            shape = timestamps.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or not isinstance(shape[0], int)
                or isinstance(shape[0], bool)
                or shape[0] < 1
            ):
                raise IsaacPackSpecError(
                    f"{stream_id}.arrays.timestamps_ns.shape has no frame count"
                )
            entry["sample_count"] = shape[0]
        else:
            if (
                not isinstance(fields, list)
                or not fields
                or not all(isinstance(field, str) and field for field in fields)
                or len(fields) != len(set(fields))
            ):
                raise IsaacPackSpecError(f"{stream_id}.fields must be unique strings")
            entry["fields"] = list(fields)
        packed.append(entry)
    return packed


def build_isaac_pack_spec(
    *,
    capture_receipt: Mapping[str, Any],
    scene: Mapping[str, Any],
    public_task: Mapping[str, Any],
    observation_abi: Mapping[str, Any],
    source_streams: Mapping[str, Any],
    capture_receipt_sha256: str,
    observation_abi_source: str,
    dataset_version: str,
    episode_id: str | None = None,
    scene_asset_license_status: str = "pending",
) -> dict[str, Any]:
    """Derive and self-check one v2 pack spec from immutable public evidence."""

    if _DATASET_VERSION.fullmatch(dataset_version) is None:
        raise IsaacPackSpecError("dataset_version is not a semantic data version")
    _require_sha256(capture_receipt_sha256, label="capture_receipt_sha256")
    if scene_asset_license_status not in {
        "pending",
        "internal_only",
        "redistribution_cleared",
    }:
        raise IsaacPackSpecError("scene_asset_license_status is invalid")
    if capture_receipt.get("status") != "captured" or capture_receipt.get("ok") is not True:
        raise IsaacPackSpecError("capture receipt is not a completed capture")
    if capture_receipt.get("source_worktree_dirty") is not False:
        raise IsaacPackSpecError("pack specs require a clean source capture")
    if capture_receipt.get("task_kind") != "search3d":
        raise IsaacPackSpecError("pack specs require a Search3D capture")

    binding = _require_mapping(
        capture_receipt.get("collection_binding"), label="collection_binding"
    )
    condition_request = _require_mapping(
        capture_receipt.get("condition_request"), label="condition_request"
    )
    conditions = _require_mapping(
        condition_request.get("conditions"), label="condition_request.conditions"
    )
    split = _require_text(binding.get("split"), label="collection_binding.split")
    cell_id = _require_text(binding.get("cell_id"), label="collection_binding.cell_id")
    episode_index = binding.get("episode_index")
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise IsaacPackSpecError("collection_binding.episode_index must be non-negative")
    if condition_request.get("cell_id") != cell_id:
        raise IsaacPackSpecError("condition request does not match collection cell")
    layout_id = _require_text(conditions.get("layout"), label="conditions.layout")

    source_revision = _require_text(
        capture_receipt.get("source_revision"), label="source_revision"
    )
    if re.fullmatch(r"[0-9a-f]{7,64}", source_revision) is None:
        raise IsaacPackSpecError("source_revision must be a Git hex revision")
    backend = _require_mapping(
        capture_receipt.get("capture_backend"), label="capture_backend"
    )
    if backend.get("kind") != "isaaclab":
        raise IsaacPackSpecError("capture_backend.kind must be isaaclab")
    backend_build = _require_text(backend.get("build"), label="capture_backend.build")
    smoke_sha256 = _require_sha256(
        backend.get("sensor_physics_smoke_receipt_sha256"),
        label="capture_backend.sensor_physics_smoke_receipt_sha256",
    )

    scene_contract = _require_mapping(scene.get("scene_contract"), label="scene.scene_contract")
    layout_lineage_hash = _require_sha256(
        scene_contract.get("payload_sha256"),
        label="scene.scene_contract.payload_sha256",
    )
    inventory = _require_mapping(
        scene.get("rivermark_layer_inventory"),
        label="scene.rivermark_layer_inventory",
    )
    asset_lineage = _require_sha256(
        inventory.get("inventory_sha256"),
        label="scene.rivermark_layer_inventory.inventory_sha256",
    )
    public_scene = build_public_scene_manifest(scene)
    validate_public_payload(public_task)

    abi_issues = validate_formal_observation_abi(observation_abi)
    source_issues = validate_candidate_abi_sources(observation_abi, source_streams)
    if abi_issues or source_issues:
        detail = "; ".join(
            f"{issue.code}:{issue.path}" for issue in (*abi_issues, *source_issues)
        )
        raise IsaacPackSpecError(f"observation ABI failed closed: {detail}")
    coordinate_frames = _require_mapping(
        observation_abi.get("coordinate_frames"), label="ABI coordinate_frames"
    )
    if set(coordinate_frames) != _COORDINATE_FRAME_KEYS:
        raise IsaacPackSpecError("ABI coordinate frame contract is not exact")

    command = _require_mapping(capture_receipt.get("command"), label="command")
    dt_s = command.get("dt_s")
    capture_stride = command.get("capture_stride")
    if (
        not isinstance(dt_s, (int, float))
        or isinstance(dt_s, bool)
        or float(dt_s) <= 0.0
        or not isinstance(capture_stride, int)
        or isinstance(capture_stride, bool)
        or capture_stride < 1
    ):
        raise IsaacPackSpecError("command timebase is invalid")
    physics_dt_ns = round(float(dt_s) * 1_000_000_000)
    if physics_dt_ns < 1:
        raise IsaacPackSpecError("physics dt rounds below one nanosecond")

    task_variant_id = _require_text(
        public_task.get("task_variant_id"), label="public_task.task_variant_id"
    )
    agent_count = public_task.get("agent_count")
    if (
        not isinstance(agent_count, int)
        or isinstance(agent_count, bool)
        or not 1 <= agent_count <= 32
    ):
        raise IsaacPackSpecError("public_task.agent_count is invalid")
    if public_task.get("route_conditioning") != "public_only":
        raise IsaacPackSpecError("only public-route conditioning can be packed")
    if agent_count != public_scene["agent_count"]:
        raise IsaacPackSpecError(
            "public task agent_count does not match the public scene projection"
        )
    information_profile = _require_text(
        capture_receipt.get("information_profile"), label="information_profile"
    )

    resolved_episode_id = episode_id or _default_episode_id(
        layout_id, split, cell_id, episode_index, capture_receipt_sha256
    )
    if _EPISODE_ID.fullmatch(resolved_episode_id) is None:
        raise IsaacPackSpecError("episode_id is not a safe formal identifier")
    if not observation_abi_source or observation_abi_source.startswith(("/", "\\")):
        raise IsaacPackSpecError("observation ABI source must be relative to the spec")
    if ":" in observation_abi_source or ".." in Path(observation_abi_source).parts:
        raise IsaacPackSpecError("observation ABI source escapes the spec directory")

    payload = {
        "schema": PACK_SPEC_SCHEMA_V2,
        "dataset_version": dataset_version,
        "episode_id": resolved_episode_id,
        "split": split,
        "layout": {
            "layout_id": layout_id,
            "layout_hash": public_manifest_sha256(public_scene),
            "layout_lineage_hash": layout_lineage_hash,
            "source": "scene.json",
        },
        "task": {
            "task_id": "multi_uav_search3d",
            "task_variant_id": task_variant_id,
            "information_profile": information_profile,
            "observation_scope": "decentralized_explicit_comm",
            "agent_count": agent_count,
            "source": "public_task.json",
        },
        "timebase": {
            "unit": "ns",
            "physics_dt_ns": physics_dt_ns,
            "proprioception_period_ns": physics_dt_ns,
            "camera_period_ns": physics_dt_ns * capture_stride,
        },
        "coordinate_frames": dict(coordinate_frames),
        "observation_abi": {
            "source": observation_abi_source.replace("\\", "/"),
            "source_scope": "pack_spec",
            "path": "metadata/observation_abi.json",
            "sha256": observation_abi_sha256(observation_abi),
            "capture_receipt_sha256": capture_receipt_sha256,
        },
        "streams": _stream_specs(source_streams),
        "provenance": {
            "route_conditioning": "public_only",
            "observation_generation": "online_runtime",
            "collector_type": "scripted",
            "policy_id": "fixed-public-route-expert-coverage-v1",
            "code_commit": source_revision,
            "simulator_build": backend_build,
            "scene_asset_license_status": scene_asset_license_status,
        },
        "quality": {"task_success": False, "invalid_reasons": []},
        "lineage_values": {
            "appearance_domain": f"{layout_id}:{conditions.get('visibility_bucket', 'unknown')}",
            "dynamics_domain": _require_text(
                conditions.get("dynamics"), label="conditions.dynamics"
            ),
            "instruction_family": "none",
            "instruction_annotator": "none",
            "asset_lineage": asset_lineage,
            "behavior_policy_checkpoint_family": _require_text(
                conditions.get("route"), label="conditions.route"
            ),
        },
        "capture_backend": {
            "build": backend_build,
            "sensor_physics_smoke_receipt_sha256": smoke_sha256,
        },
    }
    structural = validate_isaac_pack_spec(payload)
    if structural:
        detail = "; ".join(f"{issue.code}:{issue.path}" for issue in structural)
        raise IsaacPackSpecError(f"generated pack spec failed closed: {detail}")
    return payload


def pack_spec_for_capture(
    capture_root: Path,
    observation_abi_path: Path,
    output_path: Path,
    *,
    dataset_version: str,
    episode_id: str | None = None,
    scene_asset_license_status: str = "pending",
) -> dict[str, Any]:
    capture = capture_root.expanduser().resolve()
    abi_path = observation_abi_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not abi_path.is_relative_to(output.parent):
        raise IsaacPackSpecError("observation ABI must be inside the pack-spec directory")
    abi_source = abi_path.relative_to(output.parent).as_posix()
    receipt_path = capture / "capture_receipt.json"
    scene_path = capture / "scene.json"
    return build_isaac_pack_spec(
        capture_receipt=_object(receipt_path, label="capture_receipt.json"),
        scene=_object(scene_path, label="scene.json"),
        public_task=_object(capture / "public_task.json", label="public_task.json"),
        observation_abi=_object(abi_path, label="observation ABI"),
        source_streams=inspect_candidate_pack_streams(capture),
        capture_receipt_sha256=_sha256_file(receipt_path),
        observation_abi_source=abi_source,
        dataset_version=dataset_version,
        episode_id=episode_id,
        scene_asset_license_status=scene_asset_license_status,
    )


def write_pack_spec(
    capture_root: Path,
    observation_abi_path: Path,
    output_path: Path,
    *,
    dataset_version: str,
    episode_id: str | None = None,
    scene_asset_license_status: str = "pending",
) -> str:
    """Atomically write a new external pack spec and return its file SHA-256."""

    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise IsaacPackSpecError(f"refusing to overwrite pack spec: {destination}")
    payload = pack_spec_for_capture(
        capture_root,
        observation_abi_path,
        destination,
        dataset_version=dataset_version,
        episode_id=episode_id,
        scene_asset_license_status=scene_asset_license_status,
    )
    serialized = (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256_file(destination)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("observation_abi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--scene-asset-license-status",
        choices=("pending", "internal_only", "redistribution_cleared"),
        default="pending",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        digest = write_pack_spec(
            args.capture_root,
            args.observation_abi,
            args.output,
            dataset_version=args.dataset_version,
            episode_id=args.episode_id,
            scene_asset_license_status=args.scene_asset_license_status,
        )
    except (IsaacPackSpecError, PolicyProjectionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "written",
                "formal_benchmark_admission": False,
                "output": str(args.output.expanduser().resolve()),
                "pack_spec_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IsaacPackSpecError",
    "build_isaac_pack_spec",
    "pack_spec_for_capture",
    "write_pack_spec",
]
