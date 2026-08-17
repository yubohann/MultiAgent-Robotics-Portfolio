"""Build evaluator-private City-Lite target manifests outside the repository.

The generator is deliberately Isaac-free.  It consumes a public collection
cell and a trusted native geometry scan, then writes selected target truth
only to an operator-controlled path.  The returned manifest must never be
placed in Git, a capture payload, or a public task description.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .citylite_scene import (
    AABB,
    CITY_LITE_TARGET_REGION_A_ID,
    CITY_LITE_TARGET_REGION_B_ID,
    ENVIRONMENT_ID,
    ROUTE_CLEARANCE_M,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SHA256,
    aabb_geometry_sha256,
    resolve_public_route_family,
)
from .citylite_task import (
    TARGET_VISIBILITY_BUCKETS,
    sample_private_targets,
    target_visibility_execution_window,
    target_visibility_geometry_contract,
)
from .collection_protocol import (
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    CollectionProtocolError,
    is_native_t2_canary_protocol,
    load_collection_protocol,
    native_t2_motion_contract,
    resolve_collection_binding,
)

PRIVATE_EVALUATOR_SCHEMA = "org.rivermark.evaluator-private-search-manifest.v1"
PRIVATE_MANIFEST_GENERATOR_SCHEMA = "org.rivermark.private-manifest-generator.v1"
NATIVE_GEOMETRY_SCAN_SCHEMA = "org.rivermark.native-geometry-scan.v1"
NATIVE_GEOMETRY_SCAN_GENERATOR = "org.rivermark.isaac-geometry-scan.v1"
NATIVE_GEOMETRY_SCAN_TOOL_PATH = "src/rivermark_benchmark/isaac_geometry_scan.py"
NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND = "native_citylite_stage_aabb_v1"
PRIVATE_TARGET_ORIGIN = "external_private_evaluator"
PRIVATE_TARGET_PLACEMENT_SCHEMA = "org.rivermark.private-target-placement.v1"
TASK_VARIANT_ID = "isaac-eight-agent-public-waypoint-search-v1"
NATIVE_T2_TASK_VARIANT_ID = "isaac-eight-agent-native-t2-search-canary-v1"
NATIVE_T2_V2_TASK_VARIANT_ID = "isaac-eight-agent-native-t2-search-canary-v2"
NATIVE_T2_V3_TASK_VARIANT_ID = "isaac-eight-agent-native-t2-search-canary-v3"
SUPPORTED_TASK_VARIANT_IDS = frozenset(
    (
        TASK_VARIANT_ID,
        NATIVE_T2_TASK_VARIANT_ID,
        NATIVE_T2_V2_TASK_VARIANT_ID,
        NATIVE_T2_V3_TASK_VARIANT_ID,
    )
)
TARGET_COUNT = 4
PRIVATE_TARGET_RADIUS_M = 0.30
PRIVATE_TARGET_OBSTACLE_CLEARANCE_M = ROUTE_CLEARANCE_M
PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M = 2.0
PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M = 1.5
PRIVATE_TARGET_MAX_RADIUS_M = 0.75
PRIVATE_MANIFEST_RETENTION_KIND = "external_content_addressed_v1"
PRIVATE_MANIFEST_RETENTION_MAX_BYTES = 4 * 1024 * 1024


class PrivateManifestGenerationError(ValueError):
    """Raised when a private target manifest cannot be safely constructed."""


@dataclass(frozen=True)
class NativeGeometryCatalog:
    """Trusted structural AABBs reconstructed from a native geometry scan."""

    structural_aabbs: tuple[AABB, ...]
    aabb_geometry_sha256: str
    scan_sha256: str


@dataclass(frozen=True)
class RetainedPrivateManifest:
    """A private manifest snapshot retained outside public capture evidence."""

    path: Path
    sha256: str
    byte_count: int


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PrivateManifestGenerationError(
            f"cannot read native geometry scan {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PrivateManifestGenerationError("native geometry scan must be a JSON object")
    return payload


def _sha256(value: Path) -> str:
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retain_private_evaluator_manifest(
    source_path: Path,
    retention_root: Path,
    *,
    forbidden_roots: Sequence[Path],
) -> RetainedPrivateManifest:
    """Persist and re-open an exact evaluator-owned manifest snapshot.

    The manifest is content-addressed under an operator-provided private root.
    The function never writes to a capture or source repository, never
    overwrites an existing object, and returns the exact retained path that a
    subsequent capture must use. A crash can at worst leave a temporary file;
    a colliding or incomplete destination always fails closed.
    """

    source = Path(source_path).expanduser().resolve()
    root = Path(retention_root).expanduser().resolve()
    if source.suffix.lower() != ".json":
        raise PrivateManifestGenerationError(
            "private evaluator manifest must name a .json file"
        )
    if not source.is_file():
        raise FileNotFoundError(f"private evaluator manifest is missing: {source}")
    if not root.is_dir():
        raise PrivateManifestGenerationError(
            "private manifest retention root must be an existing directory"
        )
    for forbidden_root in forbidden_roots:
        resolved_forbidden = Path(forbidden_root).expanduser().resolve()
        if _is_within(root, resolved_forbidden):
            raise PrivateManifestGenerationError(
                "private manifest retention root must be outside capture and repository roots"
            )
        if _is_within(source, resolved_forbidden):
            raise PrivateManifestGenerationError(
                "private evaluator manifest must be outside capture and repository roots"
            )

    payload = source.read_bytes()
    if not payload:
        raise PrivateManifestGenerationError("private evaluator manifest is empty")
    if len(payload) > PRIVATE_MANIFEST_RETENTION_MAX_BYTES:
        raise PrivateManifestGenerationError(
            "private evaluator manifest exceeds the retention size limit"
        )
    digest = hashlib.sha256(payload).hexdigest()
    destination = root / f"{digest}.json"

    def _verify_destination() -> None:
        if not destination.is_file() or not _is_within(destination.resolve(), root):
            raise PrivateManifestGenerationError(
                "retained private manifest is missing or escapes its retention root"
            )
        if destination.stat().st_size != len(payload) or destination.read_bytes() != payload:
            raise PrivateManifestGenerationError(
                "existing retained private manifest differs from the source snapshot"
            )

    if destination.exists():
        _verify_destination()
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{digest}.",
                suffix=".tmp",
                dir=root,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                _verify_destination()
            except OSError as exc:
                raise PrivateManifestGenerationError(
                    f"cannot atomically retain private manifest: {exc}"
                ) from exc
            else:
                _verify_destination()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return RetainedPrivateManifest(
        path=destination,
        sha256=digest,
        byte_count=len(payload),
    )


def native_geometry_scan_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of a scan excluding its self-reference."""

    canonical = dict(payload)
    canonical.pop("scan_sha256", None)
    try:
        encoded = (
            json.dumps(
                canonical,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PrivateManifestGenerationError(
            "native geometry scan cannot be canonicalized"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PrivateManifestGenerationError(f"geometry scan {field} must be a lowercase SHA-256")
    return value


def load_native_geometry_catalog(path: Path) -> NativeGeometryCatalog:
    """Load the minimal trustworthy geometry needed for private sampling.

    A scan is accepted only when it identifies the exact approved City-Lite
    payload.  The returned AABB digest is recomputed instead of trusting a
    scanner-supplied summary hash.
    """

    scan_path = Path(path).expanduser().resolve()
    payload = _strict_json(scan_path)
    if payload.get("schema") != NATIVE_GEOMETRY_SCAN_SCHEMA:
        raise PrivateManifestGenerationError("geometry scan has an unsupported schema")
    if payload.get("status") != "passed" or payload.get("formal") is not False:
        raise PrivateManifestGenerationError("geometry scan must be a passed non-formal native artifact")
    if payload.get("generator") != NATIVE_GEOMETRY_SCAN_GENERATOR:
        raise PrivateManifestGenerationError("geometry scan has an unsupported generator")
    if payload.get("geometry_evidence_kind") != NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND:
        raise PrivateManifestGenerationError("geometry scan has an unsupported evidence kind")
    if payload.get("tool_path") != NATIVE_GEOMETRY_SCAN_TOOL_PATH:
        raise PrivateManifestGenerationError("geometry scan tool path is invalid")
    _require_sha256(payload.get("tool_sha256"), field="tool_sha256")
    if not isinstance(payload.get("source_revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", payload["source_revision"]
    ):
        raise PrivateManifestGenerationError("geometry scan source_revision must be a full Git revision")
    _require_sha256(payload.get("source_tree_sha256"), field="source_tree_sha256")
    if payload.get("source_worktree_dirty") is not False:
        raise PrivateManifestGenerationError("geometry scan must be generated from a clean worktree")
    runtime_lock = payload.get("runtime_lock")
    if not isinstance(runtime_lock, Mapping):
        raise PrivateManifestGenerationError("geometry scan runtime lock binding is missing")
    if set(runtime_lock) != {"sha256", "profile_id", "audit_status"}:
        raise PrivateManifestGenerationError("geometry scan runtime lock binding is invalid")
    _require_sha256(runtime_lock.get("sha256"), field="runtime_lock.sha256")
    if not isinstance(runtime_lock.get("profile_id"), str) or not runtime_lock["profile_id"]:
        raise PrivateManifestGenerationError("geometry scan runtime lock profile is invalid")
    if runtime_lock.get("audit_status") != "passed":
        raise PrivateManifestGenerationError("geometry scan runtime lock audit did not pass")
    if payload.get("scene_id") != ENVIRONMENT_ID:
        raise PrivateManifestGenerationError("geometry scan is not for approved City-Lite")
    if payload.get("scene_content_sha256") != SCENE_CONTRACT_PAYLOAD_SHA256:
        raise PrivateManifestGenerationError(
            "geometry scan scene payload does not match the approved City-Lite contract"
        )
    if payload.get("scene_contract_sha256") != SCENE_CONTRACT_SHA256:
        raise PrivateManifestGenerationError(
            "geometry scan scene contract does not match the approved City-Lite contract"
        )
    raw_scan_sha256 = _require_sha256(payload.get("scan_sha256"), field="scan_sha256")
    if raw_scan_sha256 != native_geometry_scan_sha256(payload):
        raise PrivateManifestGenerationError("geometry scan SHA-256 does not match its canonical payload")
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains:
        raise PrivateManifestGenerationError("geometry scan must contain structural domains")

    boxes: list[AABB] = []
    source_prims: set[str] = set()
    for index, domain in enumerate(domains):
        if not isinstance(domain, Mapping):
            raise PrivateManifestGenerationError(f"geometry scan domains[{index}] must be an object")
        raw_aabb = domain.get("aabb")
        if not isinstance(raw_aabb, Mapping):
            raise PrivateManifestGenerationError(f"geometry scan domains[{index}].aabb is missing")
        path_value = raw_aabb.get("path")
        source_kind = raw_aabb.get("source_kind")
        if not isinstance(path_value, str) or not path_value.startswith("/World/"):
            raise PrivateManifestGenerationError(
                f"geometry scan domains[{index}].aabb.path must be a world prim"
            )
        if not isinstance(source_kind, str) or not source_kind:
            raise PrivateManifestGenerationError(
                f"geometry scan domains[{index}].aabb.source_kind is missing"
            )
        if path_value in source_prims:
            raise PrivateManifestGenerationError("geometry scan AABB paths must be unique")
        source_prims.add(path_value)
        try:
            box = AABB(
                tuple(raw_aabb["min"]),
                tuple(raw_aabb["max"]),
                source_prim=path_value,
                category=source_kind,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PrivateManifestGenerationError(
                f"geometry scan domains[{index}].aabb is invalid: {exc}"
            ) from exc
        boxes.append(box)
    return NativeGeometryCatalog(
        structural_aabbs=tuple(boxes),
        aabb_geometry_sha256=aabb_geometry_sha256(boxes),
        scan_sha256=raw_scan_sha256,
    )


def _protocol_cell(protocol: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    for cell in protocol.get("cells", []):
        if isinstance(cell, Mapping) and cell.get("cell_id") == cell_id:
            return cell
    raise PrivateManifestGenerationError(f"unknown protocol cell: {cell_id}")


def _target_count(condition: Any) -> int:
    if condition != "object-count-4-v1":
        raise PrivateManifestGenerationError(
            "private target generation only supports object-count-4-v1"
        )
    return TARGET_COUNT


def build_private_evaluator_manifest(
    *,
    protocol_path: Path,
    cell_id: str,
    episode_index: int,
    geometry_scan_path: Path,
    target_seed: int,
    task_variant_id: str = TASK_VARIANT_ID,
) -> dict[str, Any]:
    """Generate one private manifest bound to a frozen public protocol cell.

    ``target_seed`` is evaluator-private entropy.  It must not be derived from
    the public episode seed and is never persisted in the returned manifest.
    """

    if isinstance(target_seed, bool) or not isinstance(target_seed, int):
        raise PrivateManifestGenerationError("target_seed must be an integer")
    if not 0 <= target_seed <= 0xFFFFFFFF:
        raise PrivateManifestGenerationError("target_seed must be an unsigned 32-bit integer")
    if task_variant_id not in SUPPORTED_TASK_VARIANT_IDS:
        raise PrivateManifestGenerationError("task_variant_id is not a supported private-evaluator task")

    try:
        protocol = load_collection_protocol(Path(protocol_path).expanduser().resolve())
        binding = resolve_collection_binding(
            protocol, cell_id=cell_id, episode_index=episode_index
        )
    except (OSError, CollectionProtocolError, ValueError, TypeError, KeyError) as exc:
        raise PrivateManifestGenerationError(f"collection protocol binding rejected: {exc}") from exc
    native_t2_variant_by_schema = {
        NATIVE_T2_CANARY_PROTOCOL_SCHEMA: NATIVE_T2_TASK_VARIANT_ID,
        NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA: NATIVE_T2_V2_TASK_VARIANT_ID,
        NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA: NATIVE_T2_V3_TASK_VARIANT_ID,
    }
    expected_t2_variant = native_t2_variant_by_schema.get(protocol.get("schema"))
    if (
        task_variant_id
        in (NATIVE_T2_TASK_VARIANT_ID, NATIVE_T2_V2_TASK_VARIANT_ID, NATIVE_T2_V3_TASK_VARIANT_ID)
        and expected_t2_variant is None
    ):
        raise PrivateManifestGenerationError(
            "native T2 private manifests require the dedicated native T2 canary protocol"
        )
    if expected_t2_variant is not None and task_variant_id != expected_t2_variant:
        raise PrivateManifestGenerationError(
            "development native T2 protocol cannot generate a non-T2 private manifest "
            "or a different native T2 protocol revision"
        )
    cell = _protocol_cell(protocol, cell_id)
    conditions = cell.get("conditions")
    if not isinstance(conditions, Mapping):
        raise PrivateManifestGenerationError("collection cell conditions are missing")

    route_family_id = conditions.get("route_family")
    target_region_id = conditions.get("target_region")
    visibility_bucket = conditions.get("visibility_bucket")
    if not isinstance(route_family_id, str):
        raise PrivateManifestGenerationError("collection cell route_family is missing")
    if target_region_id not in {
        CITY_LITE_TARGET_REGION_A_ID,
        CITY_LITE_TARGET_REGION_B_ID,
    }:
        raise PrivateManifestGenerationError("collection cell target_region is not executable")
    if visibility_bucket not in TARGET_VISIBILITY_BUCKETS:
        raise PrivateManifestGenerationError("collection cell visibility_bucket is not executable")
    try:
        routes_w_m = resolve_public_route_family(route_family_id)
    except ValueError as exc:
        raise PrivateManifestGenerationError("collection cell route_family is not executable") from exc
    catalog = load_native_geometry_catalog(geometry_scan_path)
    target_count = _target_count(conditions.get("target_count"))
    motion = native_t2_motion_contract(protocol) if is_native_t2_canary_protocol(protocol) else None
    execution_window = None
    heading_kwargs: dict[str, Any] = {}
    if motion is not None:
        try:
            execution_window = target_visibility_execution_window(
                dt_s=float(motion["dt_s"]),
                warmup_steps=int(motion["warmup_steps"]),
                rollout_steps=int(motion["rollout_steps"]),
                capture_stride=int(motion["capture_stride"]),
                waypoint_segment_seconds=float(motion["waypoint_segment_seconds"]),
            )
            heading_kwargs = {
                "camera_heading_model": str(motion["camera_heading_model"]),
                "max_yaw_rate_rad_s": float(motion["max_yaw_rate_rad_s"]),
                "yaw_feedback_gain": float(motion["yaw_feedback_gain"]),
                "yaw_stability_error_rad": float(motion["yaw_stability_error_rad"]),
                "yaw_settle_margin_s": float(motion["yaw_settle_margin_s"]),
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise PrivateManifestGenerationError(
                "native T2 motion contract cannot form a target visibility execution window"
            ) from exc
    try:
        sampled = sample_private_targets(
            seed=target_seed,
            target_count=target_count,
            target_region_id=target_region_id,
            visibility_bucket=visibility_bucket,
            routes_w_m=routes_w_m,
            structural_aabbs=catalog.structural_aabbs,
            radius_m=PRIVATE_TARGET_RADIUS_M,
            obstacle_clearance_m=PRIVATE_TARGET_OBSTACLE_CLEARANCE_M,
            minimum_route_separation_m=PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M,
            minimum_pairwise_separation_m=PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M,
            execution_window=execution_window,
            **heading_kwargs,
        )
    except ValueError as exc:
        raise PrivateManifestGenerationError(
            "private target sampling failed without relaxing the frozen geometry gates: "
            f"{exc}"
        ) from exc
    targets = [
        {
            "target_id": str(target["target_id"]),
            "position_w_m": list(target["position_w_m"]),
            "radius_m": float(target["radius_m"]),
            "visibility_bucket": str(target["visibility_bucket"]),
        }
        for target in sampled
    ]
    return {
        "schema": PRIVATE_EVALUATOR_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "city_lite_scene_contract_sha256": SCENE_CONTRACT_SHA256,
        "city_lite_scene_payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
        "task_variant_id": task_variant_id,
        "sampled_before_policy_start": True,
        "route_conditioning": "public_only",
        "collection_binding": binding,
        "target_origin": {
            "kind": PRIVATE_TARGET_ORIGIN,
            "candidate_pool_released": False,
            "seed_released": False,
            "coordinates_released": False,
        },
        "target_placement_contract": {
            "schema": PRIVATE_TARGET_PLACEMENT_SCHEMA,
            "obstacle_clearance_m": PRIVATE_TARGET_OBSTACLE_CLEARANCE_M,
            "minimum_route_separation_m": PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M,
            "minimum_pairwise_separation_m": PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M,
        },
        "target_visibility_contract": target_visibility_geometry_contract(
            route_family_id=route_family_id,
            routes_w_m=routes_w_m,
            aabb_geometry_sha256=catalog.aabb_geometry_sha256,
            target_region_id=target_region_id,
            visibility_bucket=visibility_bucket,
            execution_window=execution_window,
            **heading_kwargs,
        ),
        "geometry_evidence": {
            "schema": PRIVATE_MANIFEST_GENERATOR_SCHEMA,
            "native_scan_sha256": catalog.scan_sha256,
            "aabb_geometry_sha256": catalog.aabb_geometry_sha256,
        },
        "targets": targets,
    }


def write_private_evaluator_manifest(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    """Atomically write private truth and reject a path inside the source tree."""

    output_path = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    try:
        output_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise PrivateManifestGenerationError(
            "private evaluator manifest output must be outside the repository"
        )
    if output_path.suffix.lower() != ".json":
        raise PrivateManifestGenerationError("private evaluator manifest output must be a .json file")
    if output_path.exists():
        raise PrivateManifestGenerationError(
            f"refusing to overwrite existing private manifest: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(output_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    entropy = parser.add_mutually_exclusive_group(required=True)
    entropy.add_argument(
        "--target-seed",
        type=int,
        help="explicit evaluator-private uint32 entropy; never use the public episode seed",
    )
    entropy.add_argument(
        "--generate-target-seed",
        action="store_true",
        help="generate evaluator-private uint32 entropy inside this process",
    )
    parser.add_argument("--geometry-scan", type=Path, required=True)
    parser.add_argument(
        "--task-variant-id",
        choices=tuple(sorted(SUPPORTED_TASK_VARIANT_IDS)),
        default=TASK_VARIANT_ID,
        help="private-evaluator task variant; the default is the T1 expert-coverage task",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target_seed = secrets.randbits(32) if args.generate_target_seed else args.target_seed
    try:
        manifest = build_private_evaluator_manifest(
            protocol_path=args.protocol,
            cell_id=args.cell_id,
            episode_index=args.episode_index,
            geometry_scan_path=args.geometry_scan,
            target_seed=target_seed,
            task_variant_id=args.task_variant_id,
        )
        digest = write_private_evaluator_manifest(
            args.output, manifest, repository_root=args.repository_root
        )
    except (OSError, PrivateManifestGenerationError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=True))
        return 1
    binding = manifest["collection_binding"]
    print(
        json.dumps(
            {
                "status": "written",
                "manifest_sha256": digest,
                "target_count": len(manifest["targets"]),
                "collection_binding": {
                    key: binding[key]
                    for key in ("protocol_id", "protocol_sha256", "cell_id", "split", "episode_index")
                },
                "aabb_geometry_sha256": manifest["geometry_evidence"]["aabb_geometry_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
