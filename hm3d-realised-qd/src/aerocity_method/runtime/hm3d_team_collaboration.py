"""Public audits that distinguish team exploration from translated path copies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aerocity_method.contracts.io import finite_number, require_identifier

Point3 = tuple[float, float, float]

HM3D_TEAM_TRAJECTORY_AUDIT_SCHEMA_VERSION = "hm3d-team-trajectory-diversity-v2"
DEFAULT_MOVEMENT_THRESHOLD_M = 0.10
DEFAULT_TRANSLATED_RMSE_THRESHOLD_M = 0.125
DEFAULT_TRANSLATED_MAX_DEVIATION_THRESHOLD_M = 0.25
DEFAULT_RESAMPLE_COUNT = 33


def _point3(values: Sequence[float], label: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    return tuple(
        finite_number(value, f"{label}[{axis}]") for axis, value in enumerate(values)
    )  # type: ignore[return-value]


def _path(values: Sequence[Sequence[float]], label: str) -> tuple[Point3, ...]:
    points = tuple(_point3(point, f"{label}[{index}]") for index, point in enumerate(values))
    if not points:
        raise ValueError(f"{label} must contain at least one point")
    return points


def _path_length_m(path: Sequence[Point3]) -> float:
    return sum(math.dist(left, right) for left, right in zip(path, path[1:], strict=False))


def _translation_normalised_arc_samples(
    path: Sequence[Point3], *, sample_count: int
) -> tuple[Point3, ...]:
    """Remove the start translation and resample geometry by normalized arc length.

    Arc-length sampling removes controller timing differences, but deliberately
    does not rotate or scale a route. Two vehicles flying the same curve at
    different world positions therefore remain comparable as translated copies.
    """

    if sample_count < 3:
        raise ValueError("trajectory audit requires at least three resampled points")
    points = tuple(path)
    origin = points[0]
    translated = tuple(
        tuple(point[axis] - origin[axis] for axis in range(3)) for point in points
    )
    segment_lengths = tuple(
        math.dist(left, right) for left, right in zip(translated, translated[1:], strict=False)
    )
    total = sum(segment_lengths)
    if total <= 1.0e-12:
        return tuple((0.0, 0.0, 0.0) for _ in range(sample_count))
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    rows: list[Point3] = []
    segment_index = 0
    for sample_index in range(sample_count):
        target = total * sample_index / (sample_count - 1)
        while (
            segment_index + 1 < len(cumulative) - 1
            and cumulative[segment_index + 1] < target
        ):
            segment_index += 1
        segment_start = cumulative[segment_index]
        segment_end = cumulative[segment_index + 1]
        if segment_end - segment_start <= 1.0e-12:
            rows.append(translated[segment_index + 1])
            continue
        fraction = (target - segment_start) / (segment_end - segment_start)
        left = translated[segment_index]
        right = translated[segment_index + 1]
        rows.append(
            tuple(left[axis] + fraction * (right[axis] - left[axis]) for axis in range(3))
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class TranslationInvariantTrajectoryPairAudit:
    left_agent_id: str
    right_agent_id: str
    left_path_length_m: float
    right_path_length_m: float
    translated_rmse_m: float
    translated_max_deviation_m: float
    translated_endpoint_deviation_m: float
    duplicate_after_translation: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "left_agent_id": self.left_agent_id,
            "right_agent_id": self.right_agent_id,
            "left_path_length_m": self.left_path_length_m,
            "right_path_length_m": self.right_path_length_m,
            "translated_rmse_m": self.translated_rmse_m,
            "translated_max_deviation_m": self.translated_max_deviation_m,
            "translated_endpoint_deviation_m": self.translated_endpoint_deviation_m,
            "duplicate_after_translation": self.duplicate_after_translation,
        }


@dataclass(frozen=True, slots=True)
class TeamTrajectoryDiversityAudit:
    scope: str
    agent_count: int
    moving_explorer_agent_ids: tuple[str, ...]
    excluded_from_explorer_pair_audit_agent_ids: tuple[str, ...]
    pair_audits: tuple[TranslationInvariantTrajectoryPairAudit, ...]
    duplicate_pair_agent_ids: tuple[tuple[str, str], ...]
    movement_threshold_m: float
    translated_rmse_threshold_m: float
    translated_max_deviation_threshold_m: float
    resample_count: int
    status: str
    reasons: tuple[str, ...]
    schema_version: str = HM3D_TEAM_TRAJECTORY_AUDIT_SCHEMA_VERSION

    @property
    def has_translated_duplicate(self) -> bool:
        return bool(self.duplicate_pair_agent_ids)

    @property
    def observable(self) -> bool:
        return len(self.moving_explorer_agent_ids) >= 2

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "agent_count": self.agent_count,
            "moving_explorer_agent_ids": list(self.moving_explorer_agent_ids),
            "excluded_from_explorer_pair_audit_agent_ids": list(
                self.excluded_from_explorer_pair_audit_agent_ids
            ),
            "moving_explorer_count": len(self.moving_explorer_agent_ids),
            "pair_audits": [row.to_dict() for row in self.pair_audits],
            "duplicate_pair_agent_ids": [list(pair) for pair in self.duplicate_pair_agent_ids],
            "duplicate_pair_count": len(self.duplicate_pair_agent_ids),
            "movement_threshold_m": self.movement_threshold_m,
            "translated_rmse_threshold_m": self.translated_rmse_threshold_m,
            "translated_max_deviation_threshold_m": (
                self.translated_max_deviation_threshold_m
            ),
            "resample_count": self.resample_count,
            "observable": self.observable,
            "status": self.status,
            "reasons": list(self.reasons),
            "claim_limit": (
                "Translation and traversal timing are removed; rotation and geometric scale "
                "are retained. This is a path-copy audit, not a task-performance metric."
            ),
        }


def audit_translation_invariant_team_trajectories(
    paths_by_agent: Mapping[str, Sequence[Sequence[float]]],
    *,
    roles_by_agent: Mapping[str, str] | None = None,
    scope: str,
    movement_threshold_m: float = DEFAULT_MOVEMENT_THRESHOLD_M,
    translated_rmse_threshold_m: float = DEFAULT_TRANSLATED_RMSE_THRESHOLD_M,
    translated_max_deviation_threshold_m: float = (
        DEFAULT_TRANSLATED_MAX_DEVIATION_THRESHOLD_M
    ),
    resample_count: int = DEFAULT_RESAMPLE_COUNT,
) -> TeamTrajectoryDiversityAudit:
    """Reject moving explorer paths that become the same after translation.

    Non-explorers and explorers below the movement threshold are excluded from
    pairwise copy detection. This audit does not infer why an agent is
    stationary: execution outcomes own relay, collision and controller-failure
    attribution. Fewer than two moving explorers is reported as unobservable
    rather than fabricated as either diverse or duplicated.
    """

    if len(paths_by_agent) < 2:
        raise ValueError("team trajectory audit requires at least two agents")
    scope = require_identifier(scope, "scope")
    movement_threshold = finite_number(movement_threshold_m, "movement_threshold_m")
    rmse_threshold = finite_number(
        translated_rmse_threshold_m, "translated_rmse_threshold_m"
    )
    maximum_threshold = finite_number(
        translated_max_deviation_threshold_m,
        "translated_max_deviation_threshold_m",
    )
    if movement_threshold <= 0.0 or rmse_threshold <= 0.0 or maximum_threshold <= 0.0:
        raise ValueError("team trajectory thresholds must be positive")
    if rmse_threshold > maximum_threshold:
        raise ValueError("translated RMSE threshold cannot exceed maximum-deviation threshold")
    if resample_count < 3:
        raise ValueError("trajectory audit requires at least three resampled points")
    agent_ids = tuple(sorted(paths_by_agent))
    paths = {agent_id: _path(paths_by_agent[agent_id], agent_id) for agent_id in agent_ids}
    if roles_by_agent is None:
        roles = {agent_id: "explore" for agent_id in agent_ids}
    else:
        if set(roles_by_agent) != set(agent_ids):
            raise ValueError("trajectory roles must cover exactly the audited agents")
        roles = {
            agent_id: require_identifier(roles_by_agent[agent_id], f"role[{agent_id}]")
            for agent_id in agent_ids
        }
    lengths = {agent_id: _path_length_m(paths[agent_id]) for agent_id in agent_ids}
    moving_explorers = tuple(
        agent_id
        for agent_id in agent_ids
        if roles[agent_id] == "explore" and lengths[agent_id] >= movement_threshold
    )
    holds = tuple(agent_id for agent_id in agent_ids if agent_id not in moving_explorers)
    resampled = {
        agent_id: _translation_normalised_arc_samples(
            paths[agent_id], sample_count=resample_count
        )
        for agent_id in moving_explorers
    }
    pair_rows: list[TranslationInvariantTrajectoryPairAudit] = []
    duplicate_pairs: list[tuple[str, str]] = []
    for left_index, left_agent_id in enumerate(moving_explorers):
        for right_agent_id in moving_explorers[left_index + 1 :]:
            point_distances = tuple(
                math.dist(left, right)
                for left, right in zip(
                    resampled[left_agent_id], resampled[right_agent_id], strict=True
                )
            )
            rmse = math.sqrt(
                sum(distance * distance for distance in point_distances)
                / len(point_distances)
            )
            maximum = max(point_distances)
            endpoint = point_distances[-1]
            duplicate = (
                rmse <= rmse_threshold + 1.0e-12
                and maximum <= maximum_threshold + 1.0e-12
            )
            pair_rows.append(
                TranslationInvariantTrajectoryPairAudit(
                    left_agent_id=left_agent_id,
                    right_agent_id=right_agent_id,
                    left_path_length_m=lengths[left_agent_id],
                    right_path_length_m=lengths[right_agent_id],
                    translated_rmse_m=rmse,
                    translated_max_deviation_m=maximum,
                    translated_endpoint_deviation_m=endpoint,
                    duplicate_after_translation=duplicate,
                )
            )
            if duplicate:
                duplicate_pairs.append((left_agent_id, right_agent_id))
    reasons: list[str] = []
    if len(moving_explorers) < 2:
        reasons.append("FEWER_THAN_TWO_MOVING_EXPLORERS")
    if duplicate_pairs:
        reasons.append("TRANSLATED_EXPLORER_TRAJECTORY_COPY")
    if duplicate_pairs:
        status = "TEAM_TRAJECTORY_DIVERSITY_NOT_ADMITTED"
    elif len(moving_explorers) < 2:
        status = "TEAM_TRAJECTORY_DIVERSITY_UNOBSERVABLE"
    else:
        status = "TEAM_TRAJECTORY_DIVERSITY_ADMITTED"
    return TeamTrajectoryDiversityAudit(
        scope=scope,
        agent_count=len(agent_ids),
        moving_explorer_agent_ids=moving_explorers,
        excluded_from_explorer_pair_audit_agent_ids=holds,
        pair_audits=tuple(pair_rows),
        duplicate_pair_agent_ids=tuple(duplicate_pairs),
        movement_threshold_m=movement_threshold,
        translated_rmse_threshold_m=rmse_threshold,
        translated_max_deviation_threshold_m=maximum_threshold,
        resample_count=resample_count,
        status=status,
        reasons=tuple(reasons),
    )


__all__ = [
    "DEFAULT_MOVEMENT_THRESHOLD_M",
    "DEFAULT_RESAMPLE_COUNT",
    "DEFAULT_TRANSLATED_MAX_DEVIATION_THRESHOLD_M",
    "DEFAULT_TRANSLATED_RMSE_THRESHOLD_M",
    "HM3D_TEAM_TRAJECTORY_AUDIT_SCHEMA_VERSION",
    "TeamTrajectoryDiversityAudit",
    "TranslationInvariantTrajectoryPairAudit",
    "audit_translation_invariant_team_trajectories",
]
