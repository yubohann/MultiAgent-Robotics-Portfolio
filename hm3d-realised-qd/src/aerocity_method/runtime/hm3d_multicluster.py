"""Pure multi-cluster indexing and isolation helpers for HM3D collection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE

Point3 = tuple[float, float, float]


def _point(raw: Sequence[float], label: str) -> Point3:
    if len(raw) != 3:
        raise ValueError(f"{label} must have three coordinates")
    point = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{label} must be finite")
    return point  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class HM3DClusterLayout:
    """Map local HM3D team coordinates to isolated Isaac environment origins."""

    env_origins_m: tuple[Point3, ...]
    fleet_size: int = FORMAL_FLEET_SIZE

    def __post_init__(self) -> None:
        if not self.env_origins_m:
            raise ValueError("multi-cluster layout requires at least one environment")
        if self.fleet_size != FORMAL_FLEET_SIZE:
            raise ValueError(f"each cluster must contain exactly {FORMAL_FLEET_SIZE} CF2X")
        normalized = tuple(_point(origin, "environment origin") for origin in self.env_origins_m)
        if len(set(normalized)) != len(normalized):
            raise ValueError("cluster environment origins must be unique")
        object.__setattr__(self, "env_origins_m", normalized)

    @property
    def cluster_count(self) -> int:
        return len(self.env_origins_m)

    @property
    def total_agent_count(self) -> int:
        return self.cluster_count * self.fleet_size

    def flat_agent_index(self, cluster_id: int, agent_index: int) -> int:
        if not 0 <= cluster_id < self.cluster_count:
            raise IndexError("cluster_id is outside the layout")
        if not 0 <= agent_index < self.fleet_size:
            raise IndexError("agent_index is outside the four-CF2X team")
        return cluster_id * self.fleet_size + agent_index

    def cluster_slice(self, cluster_id: int) -> slice:
        start = self.flat_agent_index(cluster_id, 0)
        return slice(start, start + self.fleet_size)

    def to_world(self, cluster_id: int, local_point_m: Sequence[float]) -> Point3:
        point = _point(local_point_m, "local point")
        origin = self.env_origins_m[cluster_id]
        return tuple(point[axis] + origin[axis] for axis in range(3))  # type: ignore[return-value]

    def to_local(self, cluster_id: int, world_point_m: Sequence[float]) -> Point3:
        point = _point(world_point_m, "world point")
        origin = self.env_origins_m[cluster_id]
        return tuple(point[axis] - origin[axis] for axis in range(3))  # type: ignore[return-value]

    def local_team_from_flat_world(
        self, cluster_id: int, flat_world_points_m: Sequence[Sequence[float]]
    ) -> tuple[Point3, ...]:
        if len(flat_world_points_m) != self.total_agent_count:
            raise ValueError("flat world state does not match cluster_count * fleet_size")
        rows = flat_world_points_m[self.cluster_slice(cluster_id)]
        return tuple(self.to_local(cluster_id, row) for row in rows)


def cluster_seed(
    *, scene_id: str, cluster_id: int, episode_id: str, base_seed: int
) -> int:
    """Derive a stable independent 63-bit random stream identifier."""

    if not scene_id or not episode_id:
        raise ValueError("scene_id and episode_id must be non-empty")
    if cluster_id < 0 or base_seed < 0:
        raise ValueError("cluster_id and base_seed must be non-negative")
    digest = hashlib.sha256(
        f"{scene_id}\0{cluster_id}\0{episode_id}\0{base_seed}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def validate_cluster_start_sets(
    start_sets_m: Sequence[Sequence[Sequence[float]]],
    *,
    allow_identical_for_isolation_probe: bool = False,
) -> tuple[tuple[Point3, ...], ...]:
    """Validate one independent four-CF2X reset set per vectorized cluster."""

    if not start_sets_m:
        raise ValueError("multi-cluster execution requires at least one start set")
    normalized: list[tuple[Point3, ...]] = []
    canonical_sets: list[tuple[Point3, ...]] = []
    for cluster_id, raw_set in enumerate(start_sets_m):
        if len(raw_set) != FORMAL_FLEET_SIZE:
            raise ValueError(
                f"cluster {cluster_id} requires exactly {FORMAL_FLEET_SIZE} starts"
            )
        rows = tuple(
            _point(point, f"cluster {cluster_id} start {agent_id}")
            for agent_id, point in enumerate(raw_set)
        )
        if len(set(rows)) != len(rows):
            raise ValueError(f"cluster {cluster_id} contains duplicate UAV starts")
        normalized.append(rows)
        canonical_sets.append(tuple(sorted(rows)))
    if not allow_identical_for_isolation_probe and len(set(canonical_sets)) != len(
        canonical_sets
    ):
        raise ValueError(
            "vectorized training clusters must use distinct local start sets; identical "
            "sets are permitted only for an explicit isolation probe"
        )
    return tuple(normalized)


def ordered_qd_outcome_updates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Freeze one archive snapshot, then merge outcomes in a deterministic order."""

    required = ("scene_id", "cluster_id", "episode_id", "decision_id")
    normalized: list[Mapping[str, Any]] = []
    keys: set[tuple[str, int, str, str]] = set()
    for row in rows:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"QD update lacks ordering fields: {', '.join(missing)}")
        key = (
            str(row["scene_id"]),
            int(row["cluster_id"]),
            str(row["episode_id"]),
            str(row["decision_id"]),
        )
        if key in keys:
            raise ValueError("duplicate cluster QD outcome update")
        keys.add(key)
        normalized.append(row)
    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                str(row["scene_id"]),
                int(row["cluster_id"]),
                str(row["episode_id"]),
                str(row["decision_id"]),
            ),
        )
    )


def audit_reference_cluster_invariance(
    reference: Mapping[str, Any],
    perturbed_peer: Mapping[str, Any],
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Check that changing cluster B did not alter cluster A's local execution."""

    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")
    reasons: list[str] = []
    for field in ("selected_candidate_ids", "action_hashes", "outcome_hashes"):
        if reference.get(field) != perturbed_peer.get(field):
            reasons.append(f"REFERENCE_CLUSTER_{field.upper()}_CHANGED")
    ref_trace = reference.get("local_root_trace_m")
    peer_trace = perturbed_peer.get("local_root_trace_m")
    maximum_error_m = math.inf
    if (
        isinstance(ref_trace, list)
        and isinstance(peer_trace, list)
        and len(ref_trace) == len(peer_trace)
    ):
        maximum_error_m = 0.0
        for ref_step, peer_step in zip(ref_trace, peer_trace, strict=True):
            if not isinstance(ref_step, list) or not isinstance(peer_step, list):
                maximum_error_m = math.inf
                break
            if len(ref_step) != len(peer_step):
                maximum_error_m = math.inf
                break
            for ref_point, peer_point in zip(ref_step, peer_step, strict=True):
                maximum_error_m = max(
                    maximum_error_m,
                    math.dist(
                        _point(ref_point, "reference trace"),
                        _point(peer_point, "peer trace"),
                    ),
                )
    if maximum_error_m > tolerance_m:
        reasons.append("REFERENCE_CLUSTER_LOCAL_TRACE_CHANGED")
    for payload, label in ((reference, "REFERENCE"), (perturbed_peer, "PERTURBED")):
        for field in (
            "cross_cluster_contact_count",
            "cross_cluster_message_count",
            "cross_cluster_map_delta_count",
        ):
            if payload.get(field) != 0:
                reasons.append(f"{label}_{field.upper()}_NONZERO")
    return {
        "schema_version": "hm3d-multicluster-invariance-audit-v1",
        "passed": not reasons,
        "maximum_local_trace_error_m": maximum_error_m,
        "tolerance_m": tolerance_m,
        "reasons": sorted(set(reasons)),
    }


__all__ = [
    "HM3DClusterLayout",
    "audit_reference_cluster_invariance",
    "cluster_seed",
    "ordered_qd_outcome_updates",
    "validate_cluster_start_sets",
]
