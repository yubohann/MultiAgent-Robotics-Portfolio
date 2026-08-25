"""Runtime preparation for official HM3D assets.

The helpers in this module deliberately distinguish the public three-scene
example from the formal HM3D train/validation/test release.  An official
example can exercise conversion, collision, sensing and control, but it can
never authorize a formal split, P09 freeze or paper result.

Heavy geometry dependencies are imported inside runtime functions so the core
contracts and their unit tests remain usable without Isaac Sim or trimesh.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerocity_method.contracts.io import canonical_sha256, require_identifier, require_sha256

HM3D_RUNTIME_SCHEMA_VERSION = "hm3d-runtime-preparation-v1"
OFFICIAL_EXAMPLE_TIER = "official_example_v0.2"
FORMAL_SPLIT_TIER = "official_formal_split_v0.2"
ENGINEERING_EXAMPLE_STATUS = "ENGINEERING_EXAMPLE_ONLY"
FORMAL_ASSET_STATUS = "FORMAL_ASSET_CANDIDATE"


def file_sha256(path: Path) -> str:
    """Hash a potentially large asset without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HM3DAssetRecord:
    """One source-locked HM3D geometry or annotation artifact."""

    scene_id: str
    split: str
    asset_tier: str
    asset_kind: str
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        require_identifier(self.scene_id, "scene_id")
        if self.split not in {"example", "train", "validation", "test"}:
            raise ValueError("unsupported HM3D split")
        if self.asset_tier not in {OFFICIAL_EXAMPLE_TIER, FORMAL_SPLIT_TIER}:
            raise ValueError("unsupported HM3D asset tier")
        if self.asset_tier == OFFICIAL_EXAMPLE_TIER and self.split != "example":
            raise ValueError("official examples cannot be represented as a formal split")
        require_identifier(self.asset_kind, "asset_kind")
        require_identifier(self.path, "path")
        require_sha256(self.sha256, "sha256")
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes <= 0:
            raise ValueError("asset byte count must be a positive integer")

    @property
    def status(self) -> str:
        if self.asset_tier == OFFICIAL_EXAMPLE_TIER:
            return ENGINEERING_EXAMPLE_STATUS
        return FORMAL_ASSET_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "split": self.split,
            "asset_tier": self.asset_tier,
            "asset_kind": self.asset_kind,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "status": self.status,
        }


def lock_asset(
    path: Path,
    *,
    scene_id: str,
    split: str,
    asset_tier: str,
    asset_kind: str,
) -> HM3DAssetRecord:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return HM3DAssetRecord(
        scene_id=scene_id,
        split=split,
        asset_tier=asset_tier,
        asset_kind=asset_kind,
        path=str(resolved),
        sha256=file_sha256(resolved),
        bytes=resolved.stat().st_size,
    )


def audit_asset_scope(records: Iterable[HM3DAssetRecord]) -> dict[str, Any]:
    """Return permissions implied by the exact locked assets."""

    rows = tuple(records)
    if not rows:
        raise ValueError("HM3D asset audit needs at least one artifact")
    tiers = {row.asset_tier for row in rows}
    scene_ids = sorted({row.scene_id for row in rows})
    example_only = tiers == {OFFICIAL_EXAMPLE_TIER}
    split_by_scene: dict[str, str] = {}
    for row in rows:
        previous = split_by_scene.setdefault(row.scene_id, row.split)
        if previous != row.split:
            raise ValueError("one HM3D scene cannot belong to multiple splits")
    formal_splits = {row.split for row in rows if row.asset_tier == FORMAL_SPLIT_TIER}
    formal_split_complete = formal_splits == {"train", "validation", "test"}
    return {
        "schema_version": HM3D_RUNTIME_SCHEMA_VERSION,
        "status": (ENGINEERING_EXAMPLE_STATUS if example_only else FORMAL_ASSET_STATUS),
        "scene_ids": scene_ids,
        "asset_tiers": sorted(tiers),
        "engineering_runtime_authorized": True,
        "formal_split_complete": formal_split_complete,
        "formal_training_authorized": formal_split_complete,
        "p09_freeze_authorized": False,
        "formal_results_authorized": False,
        "forbidden_claims": (
            [
                "formal HM3D train/validation/test result",
                "P09 protocol freeze",
                "paper main-table result",
            ]
            if example_only
            else ["P09 protocol freeze before P01-P08 evidence"]
        ),
    }


def load_official_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "scene",
        "split",
        "num_floors",
        "num_rooms",
        "navigable_area",
        "floor_space",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Matterport metadata.csv does not have the expected HM3D fields")
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = math.floor((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_official_metadata(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Summarize official scene metrics without confusing area with side length."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("HM3D metadata summary needs rows")
    result: dict[str, Any] = {
        "schema_version": HM3D_RUNTIME_SCHEMA_VERSION,
        "scene_rows": len(materialized),
        "note": "area values are square metres, not scene side lengths",
        "metrics": {},
    }
    for name in ("navigable_area", "floor_space", "num_rooms", "num_floors"):
        values = [float(row[name]) for row in materialized]
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"official metadata contains invalid {name}")
        result["metrics"][name] = {
            "mean": sum(values) / len(values),
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "max": max(values),
        }
    largest = sorted(
        materialized,
        key=lambda row: float(row["navigable_area"]),
        reverse=True,
    )[:10]
    result["largest_by_navigable_area"] = [
        {
            "scene_id": row["scene"],
            "split": row["split"],
            "num_floors": int(row["num_floors"]),
            "num_rooms": int(row["num_rooms"]),
            "navigable_area_m2": float(row["navigable_area"]),
            "floor_space_m2": float(row["floor_space"]),
        }
        for row in largest
    ]
    result["summary_sha256"] = canonical_sha256(result)
    return result


def load_z_up_mesh(path: Path) -> Any:
    """Load the source GLB through trimesh's scene graph in Z-up coordinates."""

    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError("HM3D GLB did not yield a non-empty triangle mesh")
    return mesh


def geometry_audit(path: Path) -> tuple[Any, dict[str, Any]]:
    """Measure the exact source geometry used for voxelization."""

    import numpy as np

    mesh = load_z_up_mesh(path)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    if bounds.shape != (2, 3) or np.any(~np.isfinite(bounds)) or np.any(extents <= 0.0):
        raise ValueError("HM3D geometry bounds are invalid")
    report = {
        "schema_version": HM3D_RUNTIME_SCHEMA_VERSION,
        "source_path": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bounds_min_m": bounds[0].tolist(),
        "bounds_max_m": bounds[1].tolist(),
        "extents_m": extents.tolist(),
        "vertical_span_m": float(extents[2]),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "axis_convention": "trimesh-z-up-metres",
    }
    report["geometry_audit_sha256"] = canonical_sha256(report)
    return mesh, report


def _axis_enclosed(surface: Any, axis: int) -> Any:
    import numpy as np

    forward = np.maximum.accumulate(surface, axis=axis)
    backward = np.flip(
        np.maximum.accumulate(np.flip(surface, axis=axis), axis=axis),
        axis=axis,
    )
    return forward & backward


def build_enclosed_esdf(
    mesh: Any,
    *,
    resolution_m: float,
    vehicle_clearance_m: float,
    min_component_voxels: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Voxelize collision geometry and derive conservative enclosed free space.

    The free-space mask is not the Habitat ground navmesh.  A voxel must have
    collision surfaces in both directions along all three world axes, exceed
    the vehicle-clearance distance, and belong to a non-trivial 3D component.
    This rejects the unbounded space surrounding an indoor scan.  Isaac/PhysX
    collision replay remains a separate admission requirement.
    """

    import numpy as np
    from scipy import ndimage

    resolution = float(resolution_m)
    clearance = float(vehicle_clearance_m)
    if not math.isfinite(resolution) or not 0.05 <= resolution <= 0.5:
        raise ValueError("ESDF resolution must be in [0.05, 0.5] metres")
    if not math.isfinite(clearance) or clearance < resolution / 2.0:
        raise ValueError("vehicle clearance must be at least half a voxel")
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be positive")

    voxel_grid = mesh.voxelized(pitch=resolution)
    surface = np.asarray(voxel_grid.matrix, dtype=bool)
    if surface.ndim != 3 or min(surface.shape) < 3 or not surface.any():
        raise ValueError("surface voxelization did not produce a valid 3D grid")
    enclosed = ~surface
    for axis in range(3):
        enclosed &= _axis_enclosed(surface, axis)
    collision_distance = ndimage.distance_transform_edt(~surface) * resolution
    candidate_free = enclosed & (collision_distance >= clearance)
    labels, component_count = ndimage.label(
        candidate_free,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    counts = np.bincount(labels.reshape(-1))
    keep_ids = np.flatnonzero(counts >= min_component_voxels)
    keep_ids = keep_ids[keep_ids != 0]
    free = np.isin(labels, keep_ids)
    if not free.any():
        raise ValueError("no enclosed vehicle-clear free-space component survived")

    free_indices = np.argwhere(free)
    free_points = voxel_grid.indices_to_points(free_indices)
    origin = voxel_grid.indices_to_points(np.asarray([[0, 0, 0]], dtype=int))[0]
    z_counts = np.bincount(free_indices[:, 2], minlength=surface.shape[2])
    # A multi-floor building can be one connected 3D component through its
    # stairs.  Count substantial horizontal free-space bands instead of only
    # completely empty Z slices; narrow stairwell slices then separate floors.
    substantial_threshold = max(8, int(math.ceil(float(z_counts.max()) * 0.05)))
    active_z = np.flatnonzero(z_counts >= substantial_threshold)
    band_count = 1 + int(np.count_nonzero(np.diff(active_z) > 1))
    component_rows = sorted((int(counts[index]) for index in keep_ids), reverse=True)
    arrays = {
        "surface_occupancy": surface,
        "free_mask": free,
        "component_labels": labels.astype(np.int32),
        "collision_distance_m": collision_distance.astype(np.float32),
        "origin_center_m": np.asarray(origin, dtype=np.float64),
        "resolution_m": np.asarray(resolution, dtype=np.float64),
    }
    report = {
        "schema_version": HM3D_RUNTIME_SCHEMA_VERSION,
        "representation": "voxel_esdf_3d",
        "generation_method": "surface-voxelization+six-axis-enclosure+clearance-v1",
        "navmesh_authorizes_flight": False,
        "resolution_m": resolution,
        "vehicle_clearance_m": clearance,
        "grid_shape": list(surface.shape),
        "origin_center_m": np.asarray(origin, dtype=float).tolist(),
        "surface_voxels": int(surface.sum()),
        "enclosed_candidate_voxels": int(candidate_free.sum()),
        "free_voxels": int(free.sum()),
        "free_flight_volume_m3": float(free.sum() * resolution**3),
        "free_bounds_min_m": np.min(free_points, axis=0).tolist(),
        "free_bounds_max_m": np.max(free_points, axis=0).tolist(),
        "vertical_span_m": float(np.ptp(free_points[:, 2])),
        "connected_component_count_before_filter": int(component_count),
        "retained_component_count": len(component_rows),
        "retained_component_voxels": component_rows,
        "connected_height_band_count": band_count,
        "height_band_minimum_voxels_per_slice": substantial_threshold,
        "outside_space_rejected": True,
        "physx_collision_replay_required": True,
    }
    report["flight_space_manifest_hash"] = canonical_sha256(report)
    return arrays, report


def grid_points(arrays: dict[str, Any], mask_name: str = "free_mask") -> Any:
    """Convert selected voxel centres to world-space points."""

    import numpy as np

    mask = np.asarray(arrays[mask_name], dtype=bool)
    origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    resolution = float(np.asarray(arrays["resolution_m"]).item())
    return origin + np.argwhere(mask).astype(np.float64) * resolution


def reachable_component_mask(
    arrays: dict[str, Any],
    *,
    start_positions_m: Iterable[tuple[float, float, float]],
) -> tuple[Any, dict[str, Any]]:
    """Return the evaluator-only free-space union reachable from episode starts.

    P03 deliberately records the complete retained indoor flight space.  An
    exploration episode, however, may only score components containing one of
    its frozen physical resets.  This helper keeps that distinction explicit:
    it never exposes the mask to method code and returns enough provenance to
    make the episode-level denominator independently auditable.
    """

    import numpy as np

    free = np.asarray(arrays["free_mask"], dtype=bool)
    labels = np.asarray(arrays["component_labels"], dtype=np.int32)
    origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
    resolution = float(np.asarray(arrays["resolution_m"]).item())
    if free.ndim != 3 or labels.shape != free.shape:
        raise ValueError("reachable denominator requires aligned 3D free-space labels")
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(resolution):
        raise ValueError("reachable denominator grid metadata is invalid")
    if resolution <= 0.0:
        raise ValueError("reachable denominator resolution must be positive")

    starts = tuple(tuple(float(value) for value in point) for point in start_positions_m)
    if not starts:
        raise ValueError("reachable denominator requires at least one physical start")
    component_ids: list[int] = []
    start_indices: list[tuple[int, int, int]] = []
    for start_index, start in enumerate(starts):
        if len(start) != 3 or not all(math.isfinite(value) for value in start):
            raise ValueError(f"reachable denominator start[{start_index}] is invalid")
        continuous = (np.asarray(start, dtype=np.float64) - origin) / resolution
        voxel = np.rint(continuous).astype(np.int64)
        if (
            any(voxel[axis] < 0 or voxel[axis] >= free.shape[axis] for axis in range(3))
            or np.max(np.abs(continuous - voxel)) > 0.5 + 1.0e-6
        ):
            raise ValueError(f"reachable denominator start[{start_index}] is outside the ESDF grid")
        voxel_index = tuple(int(value) for value in voxel)
        component_id = int(labels[voxel_index])
        if component_id <= 0 or not bool(free[voxel_index]):
            raise ValueError(
                f"reachable denominator start[{start_index}] is not in retained free flight space"
            )
        start_indices.append(voxel_index)
        component_ids.append(component_id)

    selected_ids = tuple(sorted(set(component_ids)))
    mask = free & np.isin(labels, selected_ids)
    if not mask.any():
        raise RuntimeError("reachable denominator selected no free-flight voxels")
    component_voxel_counts = {
        str(component_id): int(np.count_nonzero(free & (labels == component_id)))
        for component_id in selected_ids
    }
    voxel_count = int(mask.sum())
    mask_sha256 = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    metadata = {
        "schema_version": "hm3d-reachable-evaluation-denominator-v1",
        "connectivity": 26,
        "origin_center_m": origin.tolist(),
        "resolution_m": resolution,
        "start_positions_m": [list(point) for point in starts],
        "start_voxel_indices": [list(index) for index in start_indices],
        "start_component_ids": list(component_ids),
        "component_ids": list(selected_ids),
        "component_voxel_counts": component_voxel_counts,
        "reachable_voxel_count": voxel_count,
        "reachable_volume_m3": float(voxel_count * resolution**3),
        "mask_sha256": mask_sha256,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return mask, metadata


def select_spread_points(
    arrays: dict[str, Any],
    *,
    count: int,
    minimum_clearance_m: float,
    seed: int,
    largest_component_only: bool = True,
) -> Any:
    """Select deterministic farthest-spread public waypoints from free voxels."""

    import numpy as np

    if count < 1:
        raise ValueError("spread-point count must be positive")
    free = np.asarray(arrays["free_mask"], dtype=bool)
    if largest_component_only:
        labels = np.asarray(arrays.get("component_labels"))
        if labels.shape != free.shape:
            raise ValueError("component labels are required for route waypoint selection")
        counts = np.bincount(labels[free].reshape(-1))
        counts[0] = 0
        free &= labels == int(np.argmax(counts))
    distance = np.asarray(arrays["collision_distance_m"], dtype=np.float32)
    eligible = free & (distance >= float(minimum_clearance_m))
    points = grid_points({**arrays, "eligible": eligible}, "eligible")
    if len(points) < count:
        raise ValueError("not enough vehicle-clear free voxels for requested points")
    rng = np.random.default_rng(seed)
    first_pool = np.flatnonzero(
        np.isclose(
            distance[eligible],
            float(distance[eligible].max()),
            atol=1.0e-6,
        )
    )
    selected = [int(rng.choice(first_pool))]
    nearest_sq = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest_sq))
        selected.append(index)
        nearest_sq = np.minimum(
            nearest_sq,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return points[np.asarray(selected, dtype=int)]


__all__ = [
    "ENGINEERING_EXAMPLE_STATUS",
    "FORMAL_ASSET_STATUS",
    "FORMAL_SPLIT_TIER",
    "HM3DAssetRecord",
    "HM3D_RUNTIME_SCHEMA_VERSION",
    "OFFICIAL_EXAMPLE_TIER",
    "audit_asset_scope",
    "build_enclosed_esdf",
    "file_sha256",
    "geometry_audit",
    "grid_points",
    "load_official_metadata",
    "lock_asset",
    "select_spread_points",
    "summarize_official_metadata",
]
