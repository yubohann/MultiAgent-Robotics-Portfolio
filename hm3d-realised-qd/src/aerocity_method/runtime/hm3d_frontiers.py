"""3D frontier extraction from public HM3D exploration beliefs."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from aerocity_method.contracts.exploration import FrontierCluster
from aerocity_method.runtime.hm3d_belief import FREE, UNKNOWN, SparseVoxelBelief, VoxelKey

_NEIGHBORS_6: tuple[VoxelKey, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# Public range rays are not restricted to axis-aligned sensor directions.  A
# route graph that follows received free space therefore needs the full local
# voxel neighbourhood.  Frontier extraction deliberately continues to use
# ``_NEIGHBORS_6`` below: face adjacency is the intended frontier topology.
_NEIGHBORS_26: tuple[VoxelKey, ...] = tuple(
    (delta_x, delta_y, delta_z)
    for delta_x in (-1, 0, 1)
    for delta_y in (-1, 0, 1)
    for delta_z in (-1, 0, 1)
    if (delta_x, delta_y, delta_z) != (0, 0, 0)
)


def neighbors_6(key: VoxelKey) -> tuple[VoxelKey, ...]:
    return tuple(
        (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2]) for delta in _NEIGHBORS_6
    )


def neighbors_26(key: VoxelKey) -> tuple[VoxelKey, ...]:
    """Return the deterministic full local neighbourhood for route connectivity."""

    return tuple(
        (key[0] + delta[0], key[1] + delta[1], key[2] + delta[2])
        for delta in _NEIGHBORS_26
    )


@dataclass(frozen=True, slots=True)
class FrontierExtractionConfig:
    min_cluster_voxels: int = 1
    max_viewpoints_per_cluster: int = 4
    height_band_m: float = 1.5

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_cluster_voxels, int)
            or isinstance(self.min_cluster_voxels, bool)
            or self.min_cluster_voxels < 1
        ):
            raise ValueError("min_cluster_voxels must be a positive integer")
        if (
            not isinstance(self.max_viewpoints_per_cluster, int)
            or isinstance(self.max_viewpoints_per_cluster, bool)
            or self.max_viewpoints_per_cluster < 1
        ):
            raise ValueError("max_viewpoints_per_cluster must be a positive integer")
        if not math.isfinite(self.height_band_m) or self.height_band_m <= 0.0:
            raise ValueError("height_band_m must be positive")


def _frontier_unknown_count(belief: SparseVoxelBelief, key: VoxelKey) -> int:
    return sum(belief.state(neighbor) == UNKNOWN for neighbor in neighbors_6(key))


def _is_frontier_free(belief: SparseVoxelBelief, key: VoxelKey) -> bool:
    return belief.state(key) == FREE and _frontier_unknown_count(belief, key) > 0


def _normal_for_cluster(
    belief: SparseVoxelBelief, keys: tuple[VoxelKey, ...]
) -> tuple[float, float, float]:
    vector = [0.0, 0.0, 0.0]
    for key in keys:
        for delta, neighbor in zip(_NEIGHBORS_6, neighbors_6(key), strict=True):
            if belief.state(neighbor) == UNKNOWN:
                vector[0] += delta[0]
                vector[1] += delta[1]
                vector[2] += delta[2]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 1.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _representative_viewpoints(
    centers: tuple[tuple[float, float, float], ...],
    *,
    maximum_count: int,
) -> tuple[tuple[float, float, float], ...]:
    """Cover a frontier cluster deterministically instead of slicing voxel order."""

    if len(centers) <= maximum_count:
        return centers
    centroid = tuple(sum(point[axis] for point in centers) / len(centers) for axis in range(3))
    first = min(centers, key=lambda point: (math.dist(point, centroid), point))
    selected = [first]
    remaining = set(centers)
    remaining.remove(first)
    while remaining and len(selected) < maximum_count:
        chosen = max(
            remaining,
            key=lambda point: (
                min(math.dist(point, prior) for prior in selected),
                math.dist(point, centroid),
                point,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def extract_frontier_clusters(
    belief: SparseVoxelBelief,
    *,
    config: FrontierExtractionConfig | None = None,
) -> tuple[FrontierCluster, ...]:
    """Return public free/unknown frontier clusters.

    A vertical frontier is not a separate special case here: unknown cells
    above or below free cells contribute to the same 6-neighbor criterion and
    appear in the cluster normal/height band.
    """

    cfg = config or FrontierExtractionConfig()
    candidates = {key for key in belief.free_keys() if _is_frontier_free(belief, key)}
    clusters: list[tuple[VoxelKey, ...]] = []
    while candidates:
        root = min(candidates)
        queue: deque[VoxelKey] = deque((root,))
        candidates.remove(root)
        cluster: list[VoxelKey] = []
        while queue:
            key = queue.popleft()
            cluster.append(key)
            for neighbor in neighbors_6(key):
                if neighbor in candidates:
                    candidates.remove(neighbor)
                    queue.append(neighbor)
        if len(cluster) >= cfg.min_cluster_voxels:
            clusters.append(tuple(sorted(cluster)))

    version = belief.version()
    rows: list[FrontierCluster] = []
    for index, keys in enumerate(sorted(clusters, key=lambda row: row[0])):
        centers = tuple(belief.voxel_center(key) for key in keys)
        centroid = tuple(sum(point[axis] for point in centers) / len(centers) for axis in range(3))
        unknown_count = sum(_frontier_unknown_count(belief, key) for key in keys)
        viewpoints = _representative_viewpoints(
            centers,
            maximum_count=cfg.max_viewpoints_per_cluster,
        )
        relative_height_band = math.floor((centroid[2] - belief.origin_m[2]) / cfg.height_band_m)
        rows.append(
            FrontierCluster(
                frontier_id=f"{belief.agent_id}-frontier-{index}",
                belief_version_sha256=version.digest,
                centroid_m=centroid,  # type: ignore[arg-type]
                outward_normal=_normal_for_cluster(belief, keys),
                viewpoint_candidates_m=viewpoints,
                unknown_voxel_count=unknown_count,
                expected_gain_m3=unknown_count * belief.resolution_m**3,
                relative_height_band=relative_height_band,
            )
        )
    return tuple(rows)


__all__ = [
    "FrontierExtractionConfig",
    "extract_frontier_clusters",
    "neighbors_26",
    "neighbors_6",
]
