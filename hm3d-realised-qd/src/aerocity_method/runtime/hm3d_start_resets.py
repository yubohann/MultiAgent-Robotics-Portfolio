"""Deterministic environment-side initial-state selection for HM3D P07."""

from __future__ import annotations

import numpy as np

P07_START_RESET_SCHEMA_VERSION = "hm3d-p07-start-reset-candidates-v1"
# A P0 qualification reset must leave enough room for the first interior
# sample of a guarded movement command.  The numeric threshold is deliberately
# supplied by the execution layer when the manifest is generated or admitted;
# this constant only freezes the reset-selection semantics in the artifact.
P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE = (
    "largest-component-route-sample-departure-witness-local-farthest-spread-v4"
)
P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION = "hm3d-p07-start-departure-witness-v1"


def _largest_component_grid(
    arrays: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Validate P03 arrays and return their largest retained component."""

    try:
        free = np.asarray(arrays["free_mask"], dtype=bool)
        labels = np.asarray(arrays["component_labels"], dtype=np.int32)
        distance = np.asarray(arrays["collision_distance_m"], dtype=np.float64)
        origin = np.asarray(arrays["origin_center_m"], dtype=np.float64)
        resolution = float(np.asarray(arrays["resolution_m"]).item())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ESDF arrays lack a valid clearance-filtered reset grid") from error
    if free.ndim != 3 or labels.shape != free.shape or distance.shape != free.shape:
        raise ValueError("ESDF reset arrays must have aligned three-dimensional masks")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("ESDF reset origin must be a finite three-vector")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("ESDF reset resolution must be finite and positive")

    counts = np.bincount(labels[free])
    if len(counts):
        counts[0] = 0
    component = int(np.argmax(counts)) if len(counts) else 0
    if component == 0:
        raise ValueError("P03 ESDF has no retained free-flight component")
    return free, labels, distance, origin, resolution, component


def largest_component_clearance_points(
    arrays: dict[str, object],
    *,
    minimum_clearance_m: float,
) -> np.ndarray:
    """Return largest-component voxel centres that meet a reset mobility margin.

    The P03 free-flight mask only proves the physical body-clearance contract.
    A P07 reset also needs enough static clearance to depart through the first
    interior sample of a subsequent guarded exploration leg. This
    environment-side filter never enters the public map or selector state;
    PhysX still verifies the chosen fleet and every actual route at runtime.
    """

    if not np.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")
    free, labels, distance, origin, resolution, component = _largest_component_grid(arrays)
    eligible = free & (labels == component) & (distance >= minimum_clearance_m)
    if not np.any(eligible):
        raise ValueError("largest P03 component has no reset points at the mobility clearance")
    return origin + np.argwhere(eligible).astype(np.float64) * resolution


def largest_component_departure_witnesses(
    arrays: dict[str, object],
    *,
    minimum_route_sample_clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return starts and one nonzero six-neighbour departure witness each.

    Reset selection cannot mistake a collision-admitted dead-end voxel for a
    usable launch point.  Every returned start and its witness endpoint carry
    an extra half-voxel clearance.  Because distance to the voxelised static
    surface is 1-Lipschitz, that leaves the requested route-sample margin
    along the short, axis-aligned first hop.  The build script subsequently
    checks the same samples against the exact collision mesh; PhysX remains
    the final runtime authority.
    """

    if (
        not np.isfinite(minimum_route_sample_clearance_m)
        or minimum_route_sample_clearance_m <= 0.0
    ):
        raise ValueError("minimum_route_sample_clearance_m must be finite and positive")
    free, labels, distance, origin, resolution, component = _largest_component_grid(arrays)
    grid_tube_clearance_m = float(minimum_route_sample_clearance_m + resolution / 2.0)
    core = free & (labels == component) & (distance >= grid_tube_clearance_m)
    directions = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.int64,
    )
    best_score = np.full(core.shape, -np.inf, dtype=np.float64)
    best_direction = np.full(core.shape, -1, dtype=np.int8)

    for direction_index, direction in enumerate(directions):
        source_slices = tuple(
            slice(max(0, -int(delta)), size - max(0, int(delta)))
            for delta, size in zip(direction, core.shape, strict=True)
        )
        target_slices = tuple(
            slice(max(0, int(delta)), size - max(0, -int(delta)))
            for delta, size in zip(direction, core.shape, strict=True)
        )
        source_valid = core[source_slices]
        target_valid = core[target_slices]
        local_indices = np.argwhere(source_valid & target_valid)
        if not len(local_indices):
            continue
        offsets = np.asarray(
            [source_slice.start for source_slice in source_slices], dtype=np.int64
        )
        source_indices = local_indices + offsets
        target_indices = source_indices + direction
        source_key = tuple(source_indices[:, axis] for axis in range(3))
        target_key = tuple(target_indices[:, axis] for axis in range(3))
        scores = distance[target_key]
        current = best_score[source_key]
        # Direction order is fixed, so exact ties keep the first direction.
        replace = scores > current + 1.0e-12
        if np.any(replace):
            replace_key = tuple(source_indices[replace, axis] for axis in range(3))
            best_score[replace_key] = scores[replace]
            best_direction[replace_key] = direction_index

    start_indices = np.argwhere(best_direction >= 0)
    if not len(start_indices):
        raise ValueError("largest P03 component has no route-sample-clear first departure")
    endpoint_indices = start_indices + directions[best_direction[tuple(start_indices.T)]]
    starts = origin + start_indices.astype(np.float64) * resolution
    endpoints = origin + endpoint_indices.astype(np.float64) * resolution
    return starts, endpoints, grid_tube_clearance_m


def select_local_spread_positions(
    points: np.ndarray,
    *,
    count: int,
    seed: int,
    cluster_radius_m: float,
    minimum_separation_m: float,
) -> np.ndarray:
    """Select one deterministic, separated local reset candidate cluster.

    The caller supplies evaluator-side, collision-admitted free-space points.
    The returned positions define an environment reset distribution, never a
    policy feature, frontier, or evaluator denominator.  A later PhysX worker
    must still verify the selected fleet's actual range/LOS connectivity.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("points must be a non-empty (N, 3) array")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if count > len(values):
        raise ValueError("points contain fewer positions than requested")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if cluster_radius_m <= 0.0 or minimum_separation_m <= 0.0:
        raise ValueError("cluster radius and separation must be positive")
    if minimum_separation_m > 2.0 * cluster_radius_m:
        raise ValueError("minimum separation exceeds the local cluster diameter")

    generator = np.random.default_rng(seed)
    for anchor_index in generator.permutation(len(values)):
        anchor = values[int(anchor_index)]
        distances = np.linalg.norm(values - anchor, axis=1)
        local_indices = np.flatnonzero(distances <= cluster_radius_m + 1.0e-12)
        if len(local_indices) < count:
            continue
        selected = [int(anchor_index)]
        remaining = [int(index) for index in local_indices if int(index) != int(anchor_index)]
        while remaining and len(selected) < count:
            selected_points = values[np.asarray(selected, dtype=np.int64)]
            next_index = max(
                remaining,
                key=lambda index: (
                    float(np.min(np.sum((values[index] - selected_points) ** 2, axis=1))),
                    -index,
                ),
            )
            nearest = float(np.min(np.linalg.norm(values[next_index] - selected_points, axis=1)))
            if nearest < minimum_separation_m - 1.0e-12:
                break
            selected.append(next_index)
            remaining.remove(next_index)
        if len(selected) == count:
            return values[np.asarray(selected, dtype=np.int64)].copy()
    raise ValueError(
        "no local collision-admitted cluster satisfies the requested count and separation"
    )


__all__ = [
    "P07_START_RESET_SCHEMA_VERSION",
    "P07_START_RESET_ROUTE_SAMPLE_SELECTION_RULE",
    "P07_START_RESET_DEPARTURE_WITNESS_SCHEMA_VERSION",
    "largest_component_clearance_points",
    "largest_component_departure_witnesses",
    "select_local_spread_positions",
]
