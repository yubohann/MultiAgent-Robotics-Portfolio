"""Metrics for the HM3D target-free online 3D exploration task.

The primary score is time-integrated explored free-flight volume.  This module
has no target-confirmation semantics and is suitable for the P07 validity
matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier

EXPLORATION_METRIC_SCHEMA_VERSION = "hm3d-exploration-metrics-v2"


def evaluation_denominator_sha256(scene_rows: tuple[dict[str, object], ...]) -> str:
    """Return the stable evaluator-only denominator identity for a P03 cohort.

    The raw ESDF and voxel membership remain evaluator-private.  This digest
    binds every later P04--P10 report to the scene-level volume, clearance,
    resolution, and flight-space provenance that define its denominator.
    """

    required = {
        "scene_id",
        "source_geometry_sha256",
        "flight_space_manifest_hash",
        "collision_geometry_sha256",
        "resolution_m",
        "vehicle_clearance_m",
        "free_flight_volume_m3",
    }
    if not scene_rows:
        raise ValueError("evaluation denominator requires at least one P03 scene")
    manifest: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(scene_rows):
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"P03 denominator row[{index}] is incomplete")
        scene_id = require_identifier(raw["scene_id"], "denominator scene_id")
        if scene_id in seen:
            raise ValueError("evaluation denominator contains duplicate scene IDs")
        seen.add(scene_id)
        for key in (
            "source_geometry_sha256",
            "flight_space_manifest_hash",
            "collision_geometry_sha256",
        ):
            value = raw[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{key} must be a SHA-256 digest")
            int(value, 16)
        row = {
            "scene_id": scene_id,
            "source_geometry_sha256": raw["source_geometry_sha256"],
            "flight_space_manifest_hash": raw["flight_space_manifest_hash"],
            "collision_geometry_sha256": raw["collision_geometry_sha256"],
            "resolution_m": finite_number(raw["resolution_m"], "resolution_m"),
            "vehicle_clearance_m": finite_number(raw["vehicle_clearance_m"], "vehicle_clearance_m"),
            "free_flight_volume_m3": finite_number(
                raw["free_flight_volume_m3"], "free_flight_volume_m3"
            ),
        }
        if (
            row["resolution_m"] <= 0.0
            or row["vehicle_clearance_m"] <= 0.0
            or row["free_flight_volume_m3"] <= 0.0
        ):
            raise ValueError("evaluation denominator geometry values must be positive")
        manifest.append(row)
    return canonical_sha256(sorted(manifest, key=lambda row: str(row["scene_id"])))


@dataclass(frozen=True, slots=True)
class ExplorationMetricSample:
    timestamp_s: float
    explored_free_volume_m3: float
    true_free_volume_m3: float
    predicted_free_volume_m3: float
    hallucinated_free_volume_m3: float = 0.0
    occupied_precision: float | None = None
    occupied_recall: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "timestamp_s",
            "explored_free_volume_m3",
            "true_free_volume_m3",
            "predicted_free_volume_m3",
            "hallucinated_free_volume_m3",
        ):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.true_free_volume_m3 <= 0.0:
            raise ValueError("true_free_volume_m3 must be positive")
        if self.explored_free_volume_m3 - self.true_free_volume_m3 > 1.0e-9:
            raise ValueError("explored volume cannot exceed the evaluator denominator")
        if self.hallucinated_free_volume_m3 - self.predicted_free_volume_m3 > 1.0e-9:
            raise ValueError("hallucinated free volume cannot exceed predicted free volume")
        for name in ("occupied_precision", "occupied_recall"):
            value = getattr(self, name)
            if value is None:
                continue
            resolved = finite_number(value, name)
            if not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, resolved)

    @property
    def coverage_fraction(self) -> float:
        return self.explored_free_volume_m3 / self.true_free_volume_m3

    @property
    def hallucinated_free_rate(self) -> float:
        if self.predicted_free_volume_m3 <= 1.0e-12:
            return 0.0
        return self.hallucinated_free_volume_m3 / self.predicted_free_volume_m3


@dataclass(frozen=True, slots=True)
class ExplorationMetricReport:
    episode_id: str
    horizon_s: float
    explored_free_flight_volume_auc_time: float
    final_coverage_at_budget: float
    final_explored_free_volume_m3: float
    evaluator_reachable_free_flight_volume_m3: float
    mean_explored_free_volume_rate_m3_per_s: float
    mean_hallucinated_free_rate: float
    mean_occupied_precision: float | None
    mean_occupied_recall: float | None
    collision_count: int
    energy_j: float
    communication_delivery_ratio: float | None
    schema_version: str = EXPLORATION_METRIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPLORATION_METRIC_SCHEMA_VERSION:
            raise ValueError("exploration metric schema version mismatch")
        require_identifier(self.episode_id, "episode_id")
        horizon = finite_number(self.horizon_s, "horizon_s")
        if horizon <= 0.0:
            raise ValueError("horizon_s must be positive")
        object.__setattr__(self, "horizon_s", horizon)
        for name in (
            "explored_free_flight_volume_auc_time",
            "final_coverage_at_budget",
            "final_explored_free_volume_m3",
            "evaluator_reachable_free_flight_volume_m3",
            "mean_explored_free_volume_rate_m3_per_s",
            "mean_hallucinated_free_rate",
            "energy_j",
        ):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "final_coverage_at_budget",
            "explored_free_flight_volume_auc_time",
            "mean_hallucinated_free_rate",
        ):
            value = getattr(self, name)
            if value > 1.0 + 1.0e-9:
                raise ValueError(f"{name} must be normalized to [0, 1]")
        if self.evaluator_reachable_free_flight_volume_m3 <= 0.0:
            raise ValueError("evaluator_reachable_free_flight_volume_m3 must be positive")
        if (
            self.final_explored_free_volume_m3
            > self.evaluator_reachable_free_flight_volume_m3 + 1.0e-9
        ):
            raise ValueError("final explored volume cannot exceed the evaluator denominator")
        expected_coverage = (
            self.final_explored_free_volume_m3 / self.evaluator_reachable_free_flight_volume_m3
        )
        if not math.isclose(
            self.final_coverage_at_budget, expected_coverage, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError("final coverage disagrees with the reported volume denominator")
        if (
            not isinstance(self.collision_count, int)
            or isinstance(self.collision_count, bool)
            or self.collision_count < 0
        ):
            raise ValueError("collision_count must be a non-negative integer")
        for name in (
            "mean_occupied_precision",
            "mean_occupied_recall",
            "communication_delivery_ratio",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            resolved = finite_number(value, name)
            if not 0.0 <= resolved <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, resolved)

    @property
    def report_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "horizon_s": self.horizon_s,
            "explored_free_flight_volume_auc_time": self.explored_free_flight_volume_auc_time,
            "final_coverage_at_budget": self.final_coverage_at_budget,
            "final_explored_free_volume_m3": self.final_explored_free_volume_m3,
            "evaluator_reachable_free_flight_volume_m3": (
                self.evaluator_reachable_free_flight_volume_m3
            ),
            "mean_explored_free_volume_rate_m3_per_s": self.mean_explored_free_volume_rate_m3_per_s,
            "mean_hallucinated_free_rate": self.mean_hallucinated_free_rate,
            "mean_occupied_precision": self.mean_occupied_precision,
            "mean_occupied_recall": self.mean_occupied_recall,
            "collision_count": self.collision_count,
            "energy_j": self.energy_j,
            "communication_delivery_ratio": self.communication_delivery_ratio,
        }
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def _normalized_auc(samples: tuple[ExplorationMetricSample, ...], horizon_s: float) -> float:
    if samples[0].timestamp_s != 0.0:
        raise ValueError("coverage curve must start at t=0")
    if samples[-1].timestamp_s > horizon_s + 1.0e-9:
        raise ValueError("coverage curve exceeds the episode horizon")
    area = 0.0
    previous = samples[0]
    for current in samples[1:]:
        if current.timestamp_s <= previous.timestamp_s:
            raise ValueError("metric sample timestamps must strictly increase")
        dt = current.timestamp_s - previous.timestamp_s
        area += 0.5 * dt * (previous.coverage_fraction + current.coverage_fraction)
        previous = current
    if samples[-1].timestamp_s < horizon_s:
        area += (horizon_s - samples[-1].timestamp_s) * samples[-1].coverage_fraction
    return min(1.0, area / horizon_s)


def score_exploration_episode(
    *,
    episode_id: str,
    samples: tuple[ExplorationMetricSample, ...],
    horizon_s: float,
    collision_count: int = 0,
    energy_j: float = 0.0,
    delivered_messages: int | None = None,
    attempted_messages: int | None = None,
) -> ExplorationMetricReport:
    """Compute normalized exploration metrics with failure denominators intact."""

    require_identifier(episode_id, "episode_id")
    horizon = finite_number(horizon_s, "horizon_s")
    if horizon <= 0.0:
        raise ValueError("horizon_s must be positive")
    if not samples:
        raise ValueError("exploration metrics require at least one sample")
    rows = tuple(samples)
    denominator = rows[0].true_free_volume_m3
    previous_explored = -1.0
    for sample in rows:
        if not math.isclose(sample.true_free_volume_m3, denominator, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "evaluator free-flight denominator must remain frozen within an episode"
            )
        if sample.explored_free_volume_m3 + 1.0e-9 < previous_explored:
            raise ValueError("unique explored free volume must be monotone non-decreasing")
        previous_explored = sample.explored_free_volume_m3
    auc = _normalized_auc(rows, horizon)
    final_coverage = rows[-1].coverage_fraction
    precision_values = tuple(
        sample.occupied_precision for sample in rows if sample.occupied_precision is not None
    )
    recall_values = tuple(
        sample.occupied_recall for sample in rows if sample.occupied_recall is not None
    )
    if (delivered_messages is None) != (attempted_messages is None):
        raise ValueError("message delivery ratio needs both numerator and denominator")
    delivery_ratio = None
    if delivered_messages is not None and attempted_messages is not None:
        if (
            not isinstance(delivered_messages, int)
            or isinstance(delivered_messages, bool)
            or not isinstance(attempted_messages, int)
            or isinstance(attempted_messages, bool)
            or delivered_messages < 0
            or attempted_messages < 1
            or delivered_messages > attempted_messages
        ):
            raise ValueError("message delivery counts are invalid")
        delivery_ratio = delivered_messages / attempted_messages
    return ExplorationMetricReport(
        episode_id=episode_id,
        horizon_s=horizon,
        explored_free_flight_volume_auc_time=auc,
        final_coverage_at_budget=final_coverage,
        final_explored_free_volume_m3=rows[-1].explored_free_volume_m3,
        evaluator_reachable_free_flight_volume_m3=denominator,
        mean_explored_free_volume_rate_m3_per_s=rows[-1].explored_free_volume_m3 / horizon,
        mean_hallucinated_free_rate=mean(sample.hallucinated_free_rate for sample in rows),
        mean_occupied_precision=None if not precision_values else mean(precision_values),
        mean_occupied_recall=None if not recall_values else mean(recall_values),
        collision_count=collision_count,
        energy_j=energy_j,
        communication_delivery_ratio=delivery_ratio,
    )


@dataclass(frozen=True, slots=True)
class SceneMacroAggregate:
    method_id: str
    reports: tuple[ExplorationMetricReport, ...]

    def __post_init__(self) -> None:
        require_identifier(self.method_id, "method_id")
        if not self.reports:
            raise ValueError("macro aggregate requires reports")

    def to_dict(self) -> dict[str, object]:
        reports = tuple(self.reports)
        auc_values = [row.explored_free_flight_volume_auc_time for row in reports]
        final_values = [row.final_coverage_at_budget for row in reports]
        worst_decile_index = max(0, math.ceil(0.1 * len(auc_values)) - 1)
        return {
            "schema_version": EXPLORATION_METRIC_SCHEMA_VERSION,
            "method_id": self.method_id,
            "episode_count": len(reports),
            "macro_auc_time": mean(auc_values),
            "macro_final_coverage": mean(final_values),
            "macro_final_explored_free_volume_m3": mean(
                row.final_explored_free_volume_m3 for row in reports
            ),
            "macro_mean_explored_free_volume_rate_m3_per_s": mean(
                row.mean_explored_free_volume_rate_m3_per_s for row in reports
            ),
            "worst_decile_auc_time": sorted(auc_values)[worst_decile_index],
            "collision_count": sum(row.collision_count for row in reports),
            "energy_j": sum(row.energy_j for row in reports),
        }


__all__ = [
    "EXPLORATION_METRIC_SCHEMA_VERSION",
    "ExplorationMetricReport",
    "ExplorationMetricSample",
    "SceneMacroAggregate",
    "evaluation_denominator_sha256",
    "score_exploration_episode",
]
