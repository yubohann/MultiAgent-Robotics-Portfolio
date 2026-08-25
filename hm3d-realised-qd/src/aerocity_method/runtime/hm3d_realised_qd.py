"""Outcome-grounded QD descriptors and richness audits for HM3D exploration.

The descriptor deliberately contains behaviour, not outcome quality.  Every
input is available to the method after execution: applied trajectories and
public sparse-range outcomes.  Evaluator mesh, ESDF and denominator voxels are
not accepted here.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import pvariance

from aerocity_method.archives.qd import (
    AdmissionDecision,
    ArchiveSpec,
    DescriptorAxis,
    Elite,
    QDArchive,
)
from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.models import CandidateFragmentManifest
from aerocity_method.runtime.hm3d_belief import PublicRangeRayOutcome, SparseVoxelBelief

Point3 = tuple[float, float, float]
VoxelKey = tuple[int, int, int]

HM3D_REALISED_QD_SCHEMA_VERSION = "hm3d-outcome-grounded-qd-v5"
HM3D_QD_FEATURE_VECTOR_SCHEMA_VERSION = "hm3d-outcome-qd-feature-vector-v1"
HM3D_PUBLIC_EXPLORATION_NEED_SCHEMA_VERSION = "hm3d-public-exploration-need-v1"
HM3D_REALISED_QD_ARCHIVE_SPEC = ArchiveSpec(
    (
        DescriptorAxis("vertical_motion_ratio", 0.0, 1.0, 4),
        DescriptorAxis("team_spatial_dispersion", 0.0, 1.0, 4),
        DescriptorAxis("public_observation_complementarity", 0.0, 1.0, 4),
    )
)
# A 4 x 4 x 4 grid has 64 nominal cells.  Six occupied cells only establish
# that an archive is not constant; they do not give an online selector enough
# independent outcome modes to claim a repertoire.  These floors are fixed
# before validation and deliberately describe admission, not performance.
MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION = 6
MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION = 12
MINIMUM_REALISED_QD_JOINT_CELLS = 6
MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS = 4.0
MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION = 0.90
MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT = 0.10
# A rich repertoire is useful only when the current public map contains a
# detectable reason to select one of its modes.  These fixed floors keep the
# selector from manufacturing a QD intervention after the public exploration
# deficits have already vanished.
MINIMUM_PUBLIC_EXPLORATION_NEED_STRENGTH = 0.15
MINIMUM_QD_NEED_ALIGNMENT_IMPROVEMENT = 0.0
# A selected mode is only trusted on validation when its realised response is
# within the uncertainty declared before execution.  This is an accountability
# check for a selected QD intervention, not a task-performance threshold; the
# paired AUC test in P08 remains the evidence for task benefit.
MINIMUM_QD_NEED_REALISATION_FIDELITY_RATE = 0.0
HM3D_QD_CALIBRATION_INTENT_MODES = (
    "vertical_low",
    "vertical_high",
    "dispersion_low",
    "dispersion_high",
    "complementarity_low",
    "complementarity_high",
)
_CALIBRATION_MODE_CONTRASTS = (
    ("vertical_motion_ratio", "vertical_low", "vertical_high", 0),
    ("team_spatial_dispersion", "dispersion_low", "dispersion_high", 1),
    (
        "public_observation_complementarity",
        "complementarity_low",
        "complementarity_high",
        2,
    ),
)
HM3D_CANDIDATE_INTENT_SPEC = ArchiveSpec(
    (
        DescriptorAxis("vertical_motion_intent", 0.0, 1.0, 4),
        DescriptorAxis("endpoint_dispersion_intent", 0.0, 1.0, 4),
        DescriptorAxis("directional_complementarity_intent", 0.0, 1.0, 4),
    )
)
QD_PUBLIC_VALUE_BACKBONE_ID = "public-quality-hint-per-cost-v1"

# The deployed archive remains three-dimensional.  These four pre-registered
# families let train-only calibration falsify the current hand-picked axes
# instead of treating them as correct by construction.  They deliberately
# differ only in the vertical and team-allocation coordinate: complementarity
# is kept in every family because it is the direct public evidence of whether
# agents observed overlapping space.
HM3D_QD_DESCRIPTOR_FAMILIES: tuple[tuple[str, tuple[str, str, str]], ...] = (
    (
        "v4_motion_dispersion_complementarity",
        (
            "vertical_motion_ratio",
            "team_spatial_dispersion",
            "public_observation_complementarity",
        ),
    ),
    (
        "observed_vertical_span_dispersion_complementarity",
        (
            "public_vertical_observation_span",
            "team_spatial_dispersion",
            "public_observation_complementarity",
        ),
    ),
    (
        "motion_unique_balance_complementarity",
        (
            "vertical_motion_ratio",
            "public_unique_contribution_balance",
            "public_observation_complementarity",
        ),
    ),
    (
        "observed_vertical_span_unique_balance_complementarity",
        (
            "public_vertical_observation_span",
            "public_unique_contribution_balance",
            "public_observation_complementarity",
        ),
    ),
)
HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID = "v4_motion_dispersion_complementarity"


def qd_selector_backbone_sha256(*, utility_slack: float) -> str:
    """Identify the common public value layer used by a QD comparison.

    P08 has to prove that no-QD, planned-QD and realised-QD faced the same
    candidate value signal.  The current development worker uses transparent
    public gain/cost hints; a trained RB-SF-SAC provider can replace it later,
    but must emit a different, shared digest for all three controls.
    """

    slack = finite_number(utility_slack, "QD utility_slack")
    if not 0.0 <= slack <= 1.0:
        raise ValueError("QD utility_slack must lie in [0, 1]")
    return canonical_sha256(
        {
            "candidate_value_provider": QD_PUBLIC_VALUE_BACKBONE_ID,
            "utility_slack": slack,
            "selector_schema": HM3D_REALISED_QD_SCHEMA_VERSION,
        }
    )


def _distance(left: Point3, right: Point3) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _checked_point(point: Sequence[float], label: str) -> Point3:
    if len(point) != 3:
        raise ValueError(f"{label} must be three-dimensional")
    return tuple(finite_number(value, label) for value in point)  # type: ignore[return-value]


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _checked_descriptor(values: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    descriptor = tuple(finite_number(value, label) for value in values)
    if any(value < 0.0 or value > 1.0 for value in descriptor):
        raise ValueError(f"{label} values must lie in [0, 1]")
    return descriptor  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PublicExplorationNeed:
    """Current exploration deficit computed only from public sparse-range state.

    The three coordinates deliberately use the same semantic order as the
    realised-QD descriptor: vertical exploration, team spatial separation,
    and duplicate-observation reduction.  They are a *selection context*, not
    an archive coordinate and not an evaluator score.  This prevents the
    archive from being filled merely because a behaviour is rare when that
    behaviour is irrelevant to the map state currently visible to the method.
    """

    vertical_exploration_deficit: float
    spatial_dispersion_deficit: float
    duplicate_observation_deficit: float
    source_public_belief_sha256: str
    source_agent_footprints_sha256: str
    source_public_outcome_count: int
    schema_version: str = HM3D_PUBLIC_EXPLORATION_NEED_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HM3D_PUBLIC_EXPLORATION_NEED_SCHEMA_VERSION:
            raise ValueError("public exploration need schema version mismatch")
        values = _checked_descriptor(self.values, "public exploration need")
        object.__setattr__(self, "vertical_exploration_deficit", values[0])
        object.__setattr__(self, "spatial_dispersion_deficit", values[1])
        object.__setattr__(self, "duplicate_observation_deficit", values[2])
        require_sha256(self.source_public_belief_sha256, "public need belief hash")
        require_sha256(self.source_agent_footprints_sha256, "public need footprint hash")
        if (
            not isinstance(self.source_public_outcome_count, int)
            or isinstance(self.source_public_outcome_count, bool)
            or self.source_public_outcome_count < 0
        ):
            raise ValueError("public need outcome count must be a non-negative integer")

    @property
    def values(self) -> tuple[float, float, float]:
        return (
            self.vertical_exploration_deficit,
            self.spatial_dispersion_deficit,
            self.duplicate_observation_deficit,
        )

    @property
    def strength(self) -> float:
        return math.sqrt(sum(value * value for value in self.values) / 3.0)

    @property
    def active(self) -> bool:
        return self.strength >= MINIMUM_PUBLIC_EXPLORATION_NEED_STRENGTH

    def alignment(self, descriptor: Sequence[float]) -> float:
        """Return deficit-weighted fulfilment by a predicted realised mode.

        A zero deficit is a zero *priority*, not a request to repeat the
        opposite behaviour.  For example, after a team has already separated
        spatially, the selector must not prefer a compact formation simply to
        match a zero-valued coordinate.  Only dimensions with a public
        exploration deficit contribute to the score.
        """

        mode = _checked_descriptor(descriptor, "predicted realised descriptor")
        total_priority = sum(self.values)
        if total_priority <= 1.0e-12:
            return 0.0
        return sum(self.values[index] * mode[index] for index in range(3)) / total_priority

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "vertical_exploration_deficit": self.vertical_exploration_deficit,
            "spatial_dispersion_deficit": self.spatial_dispersion_deficit,
            "duplicate_observation_deficit": self.duplicate_observation_deficit,
            "values": list(self.values),
            "strength": self.strength,
            "active": self.active,
            "minimum_active_strength": MINIMUM_PUBLIC_EXPLORATION_NEED_STRENGTH,
            "source_public_belief_sha256": self.source_public_belief_sha256,
            "source_agent_footprints_sha256": self.source_agent_footprints_sha256,
            "source_public_outcome_count": self.source_public_outcome_count,
        }


def public_exploration_need_from_public_belief(
    belief: SparseVoxelBelief,
    *,
    agent_free_voxel_keys: Mapping[str, Sequence[VoxelKey]],
    agent_ids: Sequence[str],
    spatial_reference_m: float,
    height_band_m: float = 1.0,
) -> PublicExplorationNeed:
    """Derive a current QD demand vector without evaluator-side geometry.

    ``belief`` and per-agent voxel sets are both reconstructed from delivered
    sparse-range outcomes.  No scene mesh, ESDF, floor annotation, target, or
    remaining-free-volume denominator appears in this calculation.  The
    result is intentionally conservative: an absent public observation gives
    a high deficit, while already broad, non-overlapping public evidence makes
    QD abstain rather than continue filling behaviour cells for its own sake.
    """

    if not isinstance(belief, SparseVoxelBelief):
        raise TypeError("public exploration need requires a SparseVoxelBelief")
    identifiers = tuple(sorted(agent_ids))
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("public exploration need requires unique agent IDs")
    for agent_id in identifiers:
        require_identifier(agent_id, "public need agent_id")
        if agent_id not in agent_free_voxel_keys:
            raise ValueError("public exploration need lacks an agent footprint")
    reference = finite_number(spatial_reference_m, "public need spatial reference")
    band_m = finite_number(height_band_m, "public need height band")
    if reference <= 0.0 or band_m <= 0.0:
        raise ValueError("public exploration need scales must be positive")

    free_keys = belief.free_keys()
    band_voxels = max(1, int(round(band_m / belief.resolution_m)))
    if free_keys:
        band_counts = Counter(key[2] // band_voxels for key in free_keys)
        total = sum(band_counts.values())
        entropy = -sum((count / total) * math.log(count / total) for count in band_counts.values())
        effective_band_count = math.exp(entropy)
        vertical_coverage = _clamp_unit((effective_band_count - 1.0) / 3.0)
        vertical_deficit = 1.0 - vertical_coverage

        centers = tuple(belief.voxel_center(key) for key in free_keys)
        mean_x = sum(point[0] for point in centers) / len(centers)
        mean_y = sum(point[1] for point in centers) / len(centers)
        radial_spread = math.sqrt(
            sum((point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2 for point in centers)
            / len(centers)
        )
        # A radius of half the public communication reference is already
        # sufficient evidence that the observed map is not one local cluster.
        spatial_coverage = _clamp_unit(2.0 * radial_spread / reference)
        spatial_deficit = 1.0 - spatial_coverage
    else:
        vertical_deficit = 1.0
        spatial_deficit = 1.0

    footprints = {
        agent_id: tuple(sorted({tuple(key) for key in agent_free_voxel_keys[agent_id]}))
        for agent_id in identifiers
    }
    if len(identifiers) < 2 or any(not footprint for footprint in footprints.values()):
        duplicate_deficit = 1.0
    else:
        overlaps = []
        for index, left_id in enumerate(identifiers):
            left = set(footprints[left_id])
            for right_id in identifiers[index + 1 :]:
                right = set(footprints[right_id])
                union = left | right
                overlaps.append(0.0 if not union else len(left & right) / len(union))
        duplicate_deficit = sum(overlaps) / len(overlaps) if overlaps else 1.0
    footprint_hash = canonical_sha256(
        {"agent_free_voxel_keys": {agent_id: footprints[agent_id] for agent_id in identifiers}}
    )
    return PublicExplorationNeed(
        vertical_exploration_deficit=vertical_deficit,
        spatial_dispersion_deficit=spatial_deficit,
        duplicate_observation_deficit=duplicate_deficit,
        source_public_belief_sha256=belief.content_sha256,
        source_agent_footprints_sha256=footprint_hash,
        source_public_outcome_count=belief.outcome_count,
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("correlation inputs must have matching lengths")
    if len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale <= 1.0e-12 or right_scale <= 1.0e-12:
        return 0.0
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_scale * right_scale)


def _shannon_effective_cell_count(cells: Sequence[tuple[int, ...]]) -> float:
    """Return exp(entropy) so occupancy cannot be faked by singleton cells."""

    if not cells:
        return 0.0
    total = len(cells)
    entropy = -sum((count / total) * math.log(count / total) for count in Counter(cells).values())
    return math.exp(entropy)


@dataclass(frozen=True, slots=True)
class RealisedQDDescriptor:
    """Three public, realised dimensions that classify team behaviour."""

    vertical_motion_ratio: float
    team_spatial_dispersion: float
    public_observation_complementarity: float
    schema_version: str = HM3D_REALISED_QD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HM3D_REALISED_QD_SCHEMA_VERSION:
            raise ValueError("unsupported realised-QD descriptor schema")
        for name in (
            "vertical_motion_ratio",
            "team_spatial_dispersion",
            "public_observation_complementarity",
        ):
            value = finite_number(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)

    @property
    def values(self) -> tuple[float, float, float]:
        return (
            self.vertical_motion_ratio,
            self.team_spatial_dispersion,
            self.public_observation_complementarity,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "vertical_motion_ratio": self.vertical_motion_ratio,
            "team_spatial_dispersion": self.team_spatial_dispersion,
            "public_observation_complementarity": self.public_observation_complementarity,
        }


def _per_agent_public_free_voxels(
    *,
    scene_id: str,
    agent_ids: Sequence[str],
    range_outcomes: Sequence[PublicRangeRayOutcome],
    resolution_m: float,
) -> tuple[tuple[str, ...], dict[str, SparseVoxelBelief], dict[str, frozenset[VoxelKey]]]:
    """Reconstruct each agent's free voxels solely from delivered public outcomes."""

    require_identifier(scene_id, "scene_id")
    ids = tuple(sorted(set(agent_ids)))
    if not ids:
        raise ValueError("public QD evidence requires at least one agent")
    for agent_id in ids:
        require_identifier(agent_id, "agent_id")
    resolution = finite_number(resolution_m, "resolution_m")
    if resolution <= 0.0:
        raise ValueError("resolution_m must be positive")
    beliefs = {agent_id: SparseVoxelBelief(scene_id, agent_id, resolution) for agent_id in ids}
    for outcome in range_outcomes:
        if outcome.agent_id not in beliefs:
            raise ValueError("range outcome belongs to an unknown QD agent")
        beliefs[outcome.agent_id].integrate_ray(outcome)
    return (
        ids,
        beliefs,
        {agent_id: frozenset(belief.free_keys()) for agent_id, belief in beliefs.items()},
    )


@dataclass(frozen=True, slots=True)
class OutcomeQDFeatureVector:
    """All pre-registered outcome-only candidates for the three QD axes.

    This is deliberately a *diagnostic feature vector*, not a four- or
    five-dimensional archive.  Calibration may compare the fixed three-axis
    families below, but deployment must still freeze exactly one family before
    validation.  Every value is derived from post-execution public evidence.
    """

    vertical_motion_ratio: float
    public_vertical_observation_span: float
    team_spatial_dispersion: float
    public_unique_contribution_balance: float
    public_observation_complementarity: float
    schema_version: str = HM3D_QD_FEATURE_VECTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HM3D_QD_FEATURE_VECTOR_SCHEMA_VERSION:
            raise ValueError("unsupported outcome-QD feature-vector schema")
        for name in (
            "vertical_motion_ratio",
            "public_vertical_observation_span",
            "team_spatial_dispersion",
            "public_unique_contribution_balance",
            "public_observation_complementarity",
        ):
            value = finite_number(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)

    def value(self, name: str) -> float:
        if name not in {
            "vertical_motion_ratio",
            "public_vertical_observation_span",
            "team_spatial_dispersion",
            "public_unique_contribution_balance",
            "public_observation_complementarity",
        }:
            raise ValueError(f"unknown outcome-QD feature: {name}")
        return float(getattr(self, name))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "vertical_motion_ratio": self.vertical_motion_ratio,
            "public_vertical_observation_span": self.public_vertical_observation_span,
            "team_spatial_dispersion": self.team_spatial_dispersion,
            "public_unique_contribution_balance": self.public_unique_contribution_balance,
            "public_observation_complementarity": self.public_observation_complementarity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> OutcomeQDFeatureVector:
        return cls(
            vertical_motion_ratio=payload.get("vertical_motion_ratio"),  # type: ignore[arg-type]
            public_vertical_observation_span=payload.get(  # type: ignore[arg-type]
                "public_vertical_observation_span"
            ),
            team_spatial_dispersion=payload.get("team_spatial_dispersion"),  # type: ignore[arg-type]
            public_unique_contribution_balance=payload.get(  # type: ignore[arg-type]
                "public_unique_contribution_balance"
            ),
            public_observation_complementarity=payload.get(  # type: ignore[arg-type]
                "public_observation_complementarity"
            ),
            schema_version=payload.get("schema_version", ""),  # type: ignore[arg-type]
        )


def descriptor_values_for_qd_family(
    features: OutcomeQDFeatureVector, family_id: str
) -> tuple[float, float, float]:
    """Return one pre-registered three-axis descriptor family.

    The returned tuple has no implicit archive semantics.  Callers must bind
    its family ID and the axis names into their immutable train artifact before
    a selector is allowed to use it.
    """

    for registered_family_id, axis_names in HM3D_QD_DESCRIPTOR_FAMILIES:
        if family_id == registered_family_id:
            return tuple(features.value(name) for name in axis_names)  # type: ignore[return-value]
    raise ValueError(f"unsupported outcome-QD descriptor family: {family_id}")


def outcome_qd_feature_vector_from_public_outcomes(
    *,
    scene_id: str,
    agent_ids: Sequence[str],
    applied_paths_by_agent: Mapping[str, Sequence[Sequence[float]]],
    range_outcomes: Sequence[PublicRangeRayOutcome],
    resolution_m: float,
    spatial_reference_m: float,
) -> OutcomeQDFeatureVector:
    """Measure all pre-registered descriptor candidates from public outcomes.

    ``vertical_motion_ratio`` exposes realised climbing effort, while
    ``public_vertical_observation_span`` requires that execution actually
    created observations at separated heights and therefore cannot be raised
    by vertical oscillation in empty space.  ``team_spatial_dispersion`` and
    ``public_unique_contribution_balance`` are intentionally both recorded:
    the former is geometric separation; the latter measures whether each
    member supplied a comparable amount of non-overlapping public evidence.
    Train calibration decides whether either is redundant with observation
    complementarity.  Neither evaluator geometry nor unseen voxels are read.
    """

    require_identifier(scene_id, "scene_id")
    ids = tuple(sorted(set(agent_ids)))
    if not ids:
        raise ValueError("outcome-QD features require at least one agent")
    for agent_id in ids:
        require_identifier(agent_id, "agent_id")
        if agent_id not in applied_paths_by_agent:
            raise ValueError(f"missing applied path for {agent_id}")
    resolution = finite_number(resolution_m, "resolution_m")
    if resolution <= 0.0:
        raise ValueError("resolution_m must be positive")
    spatial_reference = finite_number(spatial_reference_m, "spatial_reference_m")
    if spatial_reference <= 0.0:
        raise ValueError("spatial_reference_m must be positive")

    total_path_m = 0.0
    vertical_path_m = 0.0
    for agent_id in ids:
        path = tuple(
            _checked_point(point, f"{agent_id}.applied_path")
            for point in applied_paths_by_agent[agent_id]
        )
        if not path:
            raise ValueError(f"applied path for {agent_id} is empty")
        for start, end in zip(path, path[1:], strict=False):
            total_path_m += _distance(start, end)
            vertical_path_m += abs(end[2] - start[2])
    vertical_motion_ratio = (
        0.0 if total_path_m <= 1.0e-12 else _clamp_unit(vertical_path_m / total_path_m)
    )

    _, beliefs, per_agent_free = _per_agent_public_free_voxels(
        scene_id=scene_id,
        agent_ids=ids,
        range_outcomes=range_outcomes,
        resolution_m=resolution,
    )
    centers_by_agent = {
        agent_id: tuple(beliefs[agent_id].voxel_center(key) for key in per_agent_free[agent_id])
        for agent_id in ids
    }
    public_z = tuple(point[2] for centers in centers_by_agent.values() for point in centers)
    vertical_span = 0.0 if not public_z else max(public_z) - min(public_z)

    centroids = tuple(
        tuple(sum(point[axis] for point in centers) / len(centers) for axis in range(3))
        for centers in centers_by_agent.values()
        if centers
    )
    centroid_distances = tuple(
        _distance(left, right)
        for index, left in enumerate(centroids)
        for right in centroids[index + 1 :]
    )
    mean_centroid_distance_m = (
        0.0 if not centroid_distances else sum(centroid_distances) / len(centroid_distances)
    )

    key_owners: Counter[VoxelKey] = Counter(key for keys in per_agent_free.values() for key in keys)
    unique_contributions = tuple(
        sum(key_owners[key] == 1 for key in per_agent_free[agent_id]) for agent_id in ids
    )
    unique_total = sum(unique_contributions)
    if unique_total == 0:
        unique_balance = 0.0
    elif len(ids) == 1:
        unique_balance = 1.0
    else:
        entropy = -sum(
            (contribution / unique_total) * math.log(contribution / unique_total)
            for contribution in unique_contributions
            if contribution
        )
        unique_balance = entropy / math.log(len(ids))

    complementarity: list[float] = []
    for left_index, left_agent_id in enumerate(ids):
        left = per_agent_free[left_agent_id]
        for right_agent_id in ids[left_index + 1 :]:
            right = per_agent_free[right_agent_id]
            if not left or not right:
                complementarity.append(0.0)
                continue
            union = left | right
            complementarity.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return OutcomeQDFeatureVector(
        vertical_motion_ratio=vertical_motion_ratio,
        public_vertical_observation_span=_clamp_unit(vertical_span / spatial_reference),
        team_spatial_dispersion=_clamp_unit(mean_centroid_distance_m / spatial_reference),
        public_unique_contribution_balance=_clamp_unit(unique_balance),
        public_observation_complementarity=_clamp_unit(
            0.0 if not complementarity else sum(complementarity) / len(complementarity)
        ),
    )


def realised_descriptor_from_public_outcomes(
    *,
    scene_id: str,
    agent_ids: Sequence[str],
    applied_paths_by_agent: Mapping[str, Sequence[Sequence[float]]],
    range_outcomes: Sequence[PublicRangeRayOutcome],
    resolution_m: float,
    spatial_reference_m: float,
) -> RealisedQDDescriptor:
    """Measure realised behaviour only from public post-execution evidence.

    ``vertical_motion_ratio`` captures actual vertical travel rather than the
    difference between team endpoints.  ``team_spatial_dispersion`` measures
    the separation of the agents' public observed-free-voxel centroids relative
    to the frozen public communication radius.  The old ``d / (d + 1 m)``
    transform saturated in ordinary indoor rooms.  The third axis,
    ``public_observation_complementarity``, is the mean pairwise Jaccard
    dissimilarity of per-agent public free-voxel sets.  A drone with no public
    contribution receives zero complementarity, so an idle drone cannot look
    diverse merely because its footprint is empty.  No evaluator geometry is
    read by any axis.
    """

    features = outcome_qd_feature_vector_from_public_outcomes(
        scene_id=scene_id,
        agent_ids=agent_ids,
        applied_paths_by_agent=applied_paths_by_agent,
        range_outcomes=range_outcomes,
        resolution_m=resolution_m,
        spatial_reference_m=spatial_reference_m,
    )
    return RealisedQDDescriptor(
        vertical_motion_ratio=features.vertical_motion_ratio,
        team_spatial_dispersion=features.team_spatial_dispersion,
        public_observation_complementarity=features.public_observation_complementarity,
    )


def public_observation_workload_balance_from_range_outcomes(
    *,
    scene_id: str,
    agent_ids: Sequence[str],
    range_outcomes: Sequence[PublicRangeRayOutcome],
    resolution_m: float,
) -> float:
    """Return workload balance as a diagnostic rather than an archive axis."""

    ids, _, per_agent_free = _per_agent_public_free_voxels(
        scene_id=scene_id,
        agent_ids=agent_ids,
        range_outcomes=range_outcomes,
        resolution_m=resolution_m,
    )
    contributions = tuple(len(per_agent_free[agent_id]) for agent_id in ids)
    maximum_contribution = max(contributions)
    return 0.0 if maximum_contribution == 0 else min(contributions) / maximum_contribution


def public_free_footprint_from_range_outcomes(
    *,
    scene_id: str,
    agent_ids: Sequence[str],
    range_outcomes: Sequence[PublicRangeRayOutcome],
    resolution_m: float,
) -> frozenset[VoxelKey]:
    """Return the fragment's public, newly observed free-voxel footprint.

    This is intentionally not an evaluator coverage score.  It is the union
    of the free voxels that the method itself can reconstruct from this
    fragment's delivered sparse-range outcomes.  P08 uses it only to test
    whether different QD cells actually stand for distinguishable exploration
    behaviours, rather than for three arbitrary scalar coordinates.
    """

    _, _, per_agent_free = _per_agent_public_free_voxels(
        scene_id=scene_id,
        agent_ids=agent_ids,
        range_outcomes=range_outcomes,
        resolution_m=resolution_m,
    )
    return frozenset(key for keys in per_agent_free.values() for key in keys)


def _normalise_public_footprint(values: Sequence[Sequence[int]], label: str) -> frozenset[VoxelKey]:
    keys: set[VoxelKey] = set()
    for index, raw_key in enumerate(values):
        if isinstance(raw_key, (str, bytes)) or len(raw_key) != 3:
            raise ValueError(f"{label}[{index}] must be a three-dimensional voxel key")
        key: list[int] = []
        for axis, value in enumerate(raw_key):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label}[{index}][{axis}] must be an integer voxel coordinate")
            key.append(value)
        keys.add((key[0], key[1], key[2]))
    return frozenset(keys)


@dataclass(frozen=True, slots=True)
class RealisedQDFootprintSeparationAudit:
    """Check that realised-QD cells denote distinct public observation modes.

    Archive occupancy alone is a weak test: unrelated scalar descriptors can
    fill many cells even when all candidate fragments observe the same local
    space.  This audit compares Jaccard dissimilarity of public free-voxel
    footprints inside and across realised archive cells.  It does not look at
    the evaluator's mesh, ESDF, or final coverage denominator.
    """

    sample_count: int
    minimum_footprint_voxel_count: int
    smallest_footprint_voxel_count: int
    same_cell_pair_count: int
    different_cell_pair_count: int
    mean_same_cell_jaccard_dissimilarity: float
    mean_different_cell_jaccard_dissimilarity: float
    footprint_separation_margin: float
    minimum_same_cell_pairs: int
    minimum_different_cell_pairs: int
    minimum_footprint_separation_margin: float
    status: str
    reasons: tuple[str, ...]
    schema_version: str = HM3D_REALISED_QD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "minimum_footprint_voxel_count": self.minimum_footprint_voxel_count,
            "smallest_footprint_voxel_count": self.smallest_footprint_voxel_count,
            "same_cell_pair_count": self.same_cell_pair_count,
            "different_cell_pair_count": self.different_cell_pair_count,
            "mean_same_cell_jaccard_dissimilarity": self.mean_same_cell_jaccard_dissimilarity,
            "mean_different_cell_jaccard_dissimilarity": (
                self.mean_different_cell_jaccard_dissimilarity
            ),
            "footprint_separation_margin": self.footprint_separation_margin,
            "minimum_same_cell_pairs": self.minimum_same_cell_pairs,
            "minimum_different_cell_pairs": self.minimum_different_cell_pairs,
            "minimum_footprint_separation_margin": self.minimum_footprint_separation_margin,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_realised_qd_footprint_separation(
    descriptors: Sequence[RealisedQDDescriptor],
    public_free_footprints: Sequence[Sequence[Sequence[int]]],
    *,
    spec: ArchiveSpec = HM3D_REALISED_QD_ARCHIVE_SPEC,
    minimum_footprint_voxel_count: int = 1,
    minimum_same_cell_pairs: int = 4,
    minimum_different_cell_pairs: int = 12,
    minimum_footprint_separation_margin: float = 0.05,
) -> RealisedQDFootprintSeparationAudit:
    """Fail closed when QD cell labels do not separate public footprints.

    The small 0.05 floor is an *admission* effect-size floor on a [0, 1]
    Jaccard scale, selected before validation data.  It is deliberately not a
    replacement for the paired task-level effect test in P08.
    """

    rows = tuple(descriptors)
    if len(rows) != len(public_free_footprints):
        raise ValueError("QD descriptors and public footprints must have matching counts")
    if spec != HM3D_REALISED_QD_ARCHIVE_SPEC:
        raise ValueError("footprint-separation audit requires the frozen realised-QD spec")
    if (
        minimum_footprint_voxel_count < 1
        or minimum_same_cell_pairs < 1
        or minimum_different_cell_pairs < 1
        or not 0.0 <= minimum_footprint_separation_margin <= 1.0
    ):
        raise ValueError("invalid QD footprint-separation evidence floor")

    footprints = tuple(
        _normalise_public_footprint(footprint, "public_free_footprint")
        for footprint in public_free_footprints
    )
    smallest = min((len(footprint) for footprint in footprints), default=0)
    cells = tuple(spec.cell(row.values) for row in rows)
    same: list[float] = []
    different: list[float] = []
    for left_index, left in enumerate(footprints):
        for right_index in range(left_index + 1, len(footprints)):
            right = footprints[right_index]
            union = left | right
            dissimilarity = 0.0 if not union else 1.0 - len(left & right) / len(union)
            if cells[left_index] == cells[right_index]:
                same.append(dissimilarity)
            else:
                different.append(dissimilarity)
    same_mean = 0.0 if not same else sum(same) / len(same)
    different_mean = 0.0 if not different else sum(different) / len(different)
    margin = different_mean - same_mean
    reasons: list[str] = []
    if smallest < minimum_footprint_voxel_count:
        reasons.append("QD_PUBLIC_FOOTPRINT_EMPTY_OR_TRUNCATED")
    if len(same) < minimum_same_cell_pairs:
        reasons.append("INSUFFICIENT_WITHIN_CELL_FOOTPRINT_PAIRS")
    if len(different) < minimum_different_cell_pairs:
        reasons.append("INSUFFICIENT_CROSS_CELL_FOOTPRINT_PAIRS")
    if (
        len(same) >= minimum_same_cell_pairs
        and len(different) >= minimum_different_cell_pairs
        and margin + 1.0e-12 < minimum_footprint_separation_margin
    ):
        reasons.append("QD_CELLS_DO_NOT_SEPARATE_PUBLIC_EXPLORATION_FOOTPRINTS")
    return RealisedQDFootprintSeparationAudit(
        sample_count=len(rows),
        minimum_footprint_voxel_count=minimum_footprint_voxel_count,
        smallest_footprint_voxel_count=smallest,
        same_cell_pair_count=len(same),
        different_cell_pair_count=len(different),
        mean_same_cell_jaccard_dissimilarity=same_mean,
        mean_different_cell_jaccard_dissimilarity=different_mean,
        footprint_separation_margin=margin,
        minimum_same_cell_pairs=minimum_same_cell_pairs,
        minimum_different_cell_pairs=minimum_different_cell_pairs,
        minimum_footprint_separation_margin=minimum_footprint_separation_margin,
        status=(
            "QD_FOOTPRINT_SEPARATION_ADMITTED"
            if not reasons
            else "QD_FOOTPRINT_SEPARATION_NOT_ADMITTED"
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class RealisedQDReproducibilityAudit:
    """Audit repeatability of outcome-grounded cells in train-only replays.

    A rich archive can still be unusable if the same legal public candidate
    moves between unrelated cells whenever PhysX, communication timing, or
    the controller is replayed.  This audit deliberately uses only repeated
    candidate-manifest identities and their realised public descriptors.  It
    neither reads evaluator geometry nor reuses validation outcomes.
    """

    repeated_manifest_group_count: int
    repeated_pair_count: int
    stable_cell_pair_count: int
    cell_stability_rate: float
    mean_normalized_descriptor_l2: float
    minimum_repeated_manifest_groups: int
    minimum_repeated_pairs: int
    minimum_cell_stability_rate: float
    maximum_mean_normalized_descriptor_l2: float
    status: str
    reasons: tuple[str, ...]
    schema_version: str = HM3D_REALISED_QD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "repeated_manifest_group_count": self.repeated_manifest_group_count,
            "repeated_pair_count": self.repeated_pair_count,
            "stable_cell_pair_count": self.stable_cell_pair_count,
            "cell_stability_rate": self.cell_stability_rate,
            "mean_normalized_descriptor_l2": self.mean_normalized_descriptor_l2,
            "minimum_repeated_manifest_groups": self.minimum_repeated_manifest_groups,
            "minimum_repeated_pairs": self.minimum_repeated_pairs,
            "minimum_cell_stability_rate": self.minimum_cell_stability_rate,
            "maximum_mean_normalized_descriptor_l2": self.maximum_mean_normalized_descriptor_l2,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_realised_qd_reproducibility(
    descriptors_by_manifest_sha256: Mapping[str, Sequence[RealisedQDDescriptor]],
    *,
    spec: ArchiveSpec = HM3D_REALISED_QD_ARCHIVE_SPEC,
    minimum_repeated_manifest_groups: int = 3,
    minimum_repeated_pairs: int = 3,
    minimum_cell_stability_rate: float = 0.70,
    maximum_mean_normalized_descriptor_l2: float = 0.25,
) -> RealisedQDReproducibilityAudit:
    """Reject a train archive whose cells are not repeatable under replay.

    A manifest SHA-256 binds one public context and guarded candidate.  A
    repeated group contains two or more independent real executions of that
    exact public candidate.  Pairwise cell stability is intentionally checked
    before P08: a selector can abstain on local uncertainty later, but it must
    not begin with a globally non-repeatable repertoire.
    """

    if spec != HM3D_REALISED_QD_ARCHIVE_SPEC:
        raise ValueError("reproducibility audit requires the frozen realised-QD spec")
    if (
        minimum_repeated_manifest_groups < 1
        or minimum_repeated_pairs < 1
        or not 0.0 <= minimum_cell_stability_rate <= 1.0
        or not 0.0 <= maximum_mean_normalized_descriptor_l2 <= 1.0
    ):
        raise ValueError("invalid QD reproducibility evidence floor")

    repeated_groups = 0
    pair_count = 0
    stable_pair_count = 0
    normalized_l2: list[float] = []
    for manifest_sha256, raw_descriptors in descriptors_by_manifest_sha256.items():
        require_sha256(manifest_sha256, "QD replay manifest hash")
        descriptors = tuple(raw_descriptors)
        if len(descriptors) < 2:
            continue
        repeated_groups += 1
        cells = tuple(spec.cell(descriptor.values) for descriptor in descriptors)
        for left_index, left in enumerate(descriptors):
            for right_index in range(left_index + 1, len(descriptors)):
                right = descriptors[right_index]
                pair_count += 1
                stable_pair_count += cells[left_index] == cells[right_index]
                normalized_l2.append(
                    math.sqrt(
                        sum((left.values[axis] - right.values[axis]) ** 2 for axis in range(3))
                    )
                    / math.sqrt(3.0)
                )
    stability_rate = 0.0 if pair_count == 0 else stable_pair_count / pair_count
    mean_l2 = 0.0 if not normalized_l2 else sum(normalized_l2) / len(normalized_l2)
    reasons: list[str] = []
    if repeated_groups < minimum_repeated_manifest_groups:
        reasons.append("INSUFFICIENT_REPEATED_PUBLIC_CANDIDATE_GROUPS")
    if pair_count < minimum_repeated_pairs:
        reasons.append("INSUFFICIENT_REPEATED_OUTCOME_PAIRS")
    if pair_count and stability_rate + 1.0e-12 < minimum_cell_stability_rate:
        reasons.append("QD_CELLS_NOT_REPRODUCIBLE_UNDER_PUBLIC_REPLAY")
    if pair_count and mean_l2 - 1.0e-12 > maximum_mean_normalized_descriptor_l2:
        reasons.append("QD_DESCRIPTOR_REPLAY_VARIANCE_TOO_HIGH")
    return RealisedQDReproducibilityAudit(
        repeated_manifest_group_count=repeated_groups,
        repeated_pair_count=pair_count,
        stable_cell_pair_count=stable_pair_count,
        cell_stability_rate=stability_rate,
        mean_normalized_descriptor_l2=mean_l2,
        minimum_repeated_manifest_groups=minimum_repeated_manifest_groups,
        minimum_repeated_pairs=minimum_repeated_pairs,
        minimum_cell_stability_rate=minimum_cell_stability_rate,
        maximum_mean_normalized_descriptor_l2=maximum_mean_normalized_descriptor_l2,
        status=(
            "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED"
            if not reasons
            else "QD_DESCRIPTOR_REPRODUCIBILITY_NOT_ADMITTED"
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class RealisedQDModeContrastAudit:
    """Verify that each frozen calibration intent controls its stated axis.

    Archive richness can be faked when all three descriptor coordinates rise
    and fall together.  The train-only calibration deliberately asks the
    emitter for a low and high public intent on every axis.  Each pair must
    produce a one-cell realised contrast on that pair's intended axis across
    independent train scenes.  This tests controllability, not task reward.
    """

    sample_count: int
    mode_sample_counts: tuple[tuple[str, int], ...]
    mode_scene_counts: tuple[tuple[str, int], ...]
    axis_mean_realised_values: tuple[tuple[str, float, float], ...]
    axis_mean_cell_gaps: tuple[tuple[str, float], ...]
    contrast_effect_vectors: tuple[tuple[str, float, float, float], ...]
    contrast_target_alignment: tuple[tuple[str, float], ...]
    maximum_pairwise_contrast_cosine: float
    contrast_matrix_absolute_determinant: float
    minimum_samples_per_mode: int
    minimum_scenes_per_mode: int
    minimum_mean_cell_gap: float
    minimum_target_alignment: float
    maximum_pairwise_contrast_cosine_allowed: float
    minimum_contrast_matrix_absolute_determinant: float
    status: str
    reasons: tuple[str, ...]
    schema_version: str = HM3D_REALISED_QD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "mode_sample_counts": [list(row) for row in self.mode_sample_counts],
            "mode_scene_counts": [list(row) for row in self.mode_scene_counts],
            "axis_mean_realised_values": [list(row) for row in self.axis_mean_realised_values],
            "axis_mean_cell_gaps": [list(row) for row in self.axis_mean_cell_gaps],
            "contrast_effect_vectors": [list(row) for row in self.contrast_effect_vectors],
            "contrast_target_alignment": [list(row) for row in self.contrast_target_alignment],
            "maximum_pairwise_contrast_cosine": self.maximum_pairwise_contrast_cosine,
            "contrast_matrix_absolute_determinant": self.contrast_matrix_absolute_determinant,
            "minimum_samples_per_mode": self.minimum_samples_per_mode,
            "minimum_scenes_per_mode": self.minimum_scenes_per_mode,
            "minimum_mean_cell_gap": self.minimum_mean_cell_gap,
            "minimum_target_alignment": self.minimum_target_alignment,
            "maximum_pairwise_contrast_cosine_allowed": (
                self.maximum_pairwise_contrast_cosine_allowed
            ),
            "minimum_contrast_matrix_absolute_determinant": (
                self.minimum_contrast_matrix_absolute_determinant
            ),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_realised_qd_calibration_mode_contrasts(
    mode_labels: Sequence[str],
    descriptors: Sequence[RealisedQDDescriptor],
    scene_ids: Sequence[str],
    *,
    spec: ArchiveSpec = HM3D_REALISED_QD_ARCHIVE_SPEC,
    minimum_samples_per_mode: int = 2,
    minimum_scenes_per_mode: int = 2,
    minimum_mean_cell_gap: float = 1.0,
    minimum_target_alignment: float = 0.60,
    maximum_pairwise_contrast_cosine: float = 0.90,
    minimum_contrast_matrix_absolute_determinant: float = 0.20,
) -> RealisedQDModeContrastAudit:
    """Fail closed unless low/high train calibrations separate each QD axis.

    A gap is measured in frozen archive-cell coordinates, rather than by an
    arbitrary raw-value threshold.  One cell is the smallest contrast that
    can influence the MAP-Elites archive used by the selector.  The three
    high-minus-low effect vectors must also be non-collinear: otherwise the
    archive has nominally three axes but only one controllable behaviour.
    """

    if spec != HM3D_REALISED_QD_ARCHIVE_SPEC:
        raise ValueError("mode-contrast audit requires the frozen realised-QD spec")
    if (
        minimum_samples_per_mode < 1
        or minimum_scenes_per_mode < 1
        or not 0.0 < minimum_mean_cell_gap <= len(spec.axes) + 1.0
        or not 0.0 < minimum_target_alignment <= 1.0
        or not 0.0 <= maximum_pairwise_contrast_cosine < 1.0
        or not 0.0 < minimum_contrast_matrix_absolute_determinant <= 1.0
    ):
        raise ValueError("invalid QD calibration mode-contrast evidence floor")
    labels = tuple(mode_labels)
    rows = tuple(descriptors)
    scenes = tuple(scene_ids)
    if len(labels) != len(rows) or len(rows) != len(scenes):
        raise ValueError("QD calibration labels, descriptors, and scenes must align")
    invalid = sorted(set(labels) - set(HM3D_QD_CALIBRATION_INTENT_MODES))
    if invalid:
        raise ValueError(f"unknown QD calibration intent mode: {invalid[0]}")
    for scene_id in scenes:
        require_identifier(scene_id, "scene_id")

    mode_descriptors = {
        mode: tuple(
            descriptor for label, descriptor in zip(labels, rows, strict=True) if label == mode
        )
        for mode in HM3D_QD_CALIBRATION_INTENT_MODES
    }
    mode_scenes = {
        mode: {scene_id for label, scene_id in zip(labels, scenes, strict=True) if label == mode}
        for mode in HM3D_QD_CALIBRATION_INTENT_MODES
    }
    reasons: list[str] = []
    for mode in HM3D_QD_CALIBRATION_INTENT_MODES:
        if len(mode_descriptors[mode]) < minimum_samples_per_mode:
            reasons.append(f"INSUFFICIENT_QD_CALIBRATION_SAMPLES_{mode.upper()}")
        if len(mode_scenes[mode]) < minimum_scenes_per_mode:
            reasons.append(f"INSUFFICIENT_QD_CALIBRATION_SCENES_{mode.upper()}")

    axis_means: list[tuple[str, float, float]] = []
    axis_cell_gaps: list[tuple[str, float]] = []
    effect_vectors: list[tuple[str, float, float, float]] = []
    target_alignments: list[tuple[str, float]] = []
    raw_effect_vectors: list[tuple[float, float, float]] = []
    for axis_name, low_mode, high_mode, axis in _CALIBRATION_MODE_CONTRASTS:
        low = mode_descriptors[low_mode]
        high = mode_descriptors[high_mode]
        low_mean = 0.0 if not low else sum(row.values[axis] for row in low) / len(low)
        high_mean = 0.0 if not high else sum(row.values[axis] for row in high) / len(high)
        axis_means.append((axis_name, low_mean, high_mean))
        low_cells = tuple(spec.cell(row.values)[axis] for row in low)
        high_cells = tuple(spec.cell(row.values)[axis] for row in high)
        low_cell_mean = 0.0 if not low_cells else sum(low_cells) / len(low_cells)
        high_cell_mean = 0.0 if not high_cells else sum(high_cells) / len(high_cells)
        cell_gap = high_cell_mean - low_cell_mean
        axis_cell_gaps.append((axis_name, cell_gap))
        if (
            len(low) >= minimum_samples_per_mode
            and len(high) >= minimum_samples_per_mode
            and cell_gap + 1.0e-12 < minimum_mean_cell_gap
        ):
            reasons.append(f"QD_CALIBRATION_AXIS_NOT_CONTROLLABLE_{axis_name.upper()}")
        effect = tuple(
            (0.0 if not high else sum(row.values[dimension] for row in high) / len(high))
            - (0.0 if not low else sum(row.values[dimension] for row in low) / len(low))
            for dimension in range(len(spec.axes))
        )
        effect_vectors.append((axis_name, *effect))
        raw_effect_vectors.append(effect)
        norm = math.sqrt(sum(value * value for value in effect))
        alignment = 0.0 if norm <= 1.0e-12 else effect[axis] / norm
        target_alignments.append((axis_name, alignment))
        if (
            len(low) >= minimum_samples_per_mode
            and len(high) >= minimum_samples_per_mode
            and alignment + 1.0e-12 < minimum_target_alignment
        ):
            reasons.append(f"QD_CALIBRATION_AXIS_NOT_SPECIFIC_{axis_name.upper()}")

    normalized_effects: list[tuple[float, float, float]] = []
    for effect in raw_effect_vectors:
        norm = math.sqrt(sum(value * value for value in effect))
        normalized_effects.append(
            (0.0, 0.0, 0.0) if norm <= 1.0e-12 else tuple(value / norm for value in effect)
        )
    pairwise_cosines = [
        abs(sum(left[axis] * right[axis] for axis in range(3)))
        for index, left in enumerate(normalized_effects)
        for right in normalized_effects[index + 1 :]
    ]
    maximum_cosine = max(pairwise_cosines, default=1.0)
    left, middle, right = normalized_effects
    determinant = abs(
        left[0] * (middle[1] * right[2] - middle[2] * right[1])
        - left[1] * (middle[0] * right[2] - middle[2] * right[0])
        + left[2] * (middle[0] * right[1] - middle[1] * right[0])
    )
    all_modes_complete = all(
        len(mode_descriptors[mode]) >= minimum_samples_per_mode
        and len(mode_scenes[mode]) >= minimum_scenes_per_mode
        for mode in HM3D_QD_CALIBRATION_INTENT_MODES
    )
    if all_modes_complete and maximum_cosine - 1.0e-12 > maximum_pairwise_contrast_cosine:
        reasons.append("QD_CALIBRATION_CONTRASTS_COLLINEAR")
    if all_modes_complete and determinant + 1.0e-12 < minimum_contrast_matrix_absolute_determinant:
        reasons.append("QD_CALIBRATION_CONTRAST_MATRIX_RANK_DEFICIENT")
    return RealisedQDModeContrastAudit(
        sample_count=len(rows),
        mode_sample_counts=tuple(
            (mode, len(mode_descriptors[mode])) for mode in HM3D_QD_CALIBRATION_INTENT_MODES
        ),
        mode_scene_counts=tuple(
            (mode, len(mode_scenes[mode])) for mode in HM3D_QD_CALIBRATION_INTENT_MODES
        ),
        axis_mean_realised_values=tuple(axis_means),
        axis_mean_cell_gaps=tuple(axis_cell_gaps),
        contrast_effect_vectors=tuple(effect_vectors),
        contrast_target_alignment=tuple(target_alignments),
        maximum_pairwise_contrast_cosine=maximum_cosine,
        contrast_matrix_absolute_determinant=determinant,
        minimum_samples_per_mode=minimum_samples_per_mode,
        minimum_scenes_per_mode=minimum_scenes_per_mode,
        minimum_mean_cell_gap=minimum_mean_cell_gap,
        minimum_target_alignment=minimum_target_alignment,
        maximum_pairwise_contrast_cosine_allowed=maximum_pairwise_contrast_cosine,
        minimum_contrast_matrix_absolute_determinant=(minimum_contrast_matrix_absolute_determinant),
        status=(
            "QD_CALIBRATION_MODE_CONTRAST_ADMITTED"
            if not reasons
            else "QD_CALIBRATION_MODE_CONTRAST_NOT_ADMITTED"
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class RealisedQDRichnessAudit:
    """A small, falsifiable admission report for outcome-grounded QD."""

    sample_count: int
    axis_variances: tuple[float, float, float]
    axis_pairwise_correlations: tuple[float, float, float]
    maximum_absolute_axis_correlation: float
    axis_correlation_absolute_determinant: float
    axis_occupied_bin_counts: tuple[int, int, int]
    joint_effective_cells: int
    joint_shannon_effective_cells: float
    archive_capacity: int
    minimum_samples: int
    minimum_axis_bins: int
    minimum_joint_cells: int
    minimum_joint_shannon_effective_cells: float
    maximum_absolute_axis_correlation_allowed: float
    minimum_axis_correlation_absolute_determinant: float
    status: str
    reasons: tuple[str, ...]
    schema_version: str = HM3D_REALISED_QD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "axis_variances": list(self.axis_variances),
            "axis_pairwise_correlations": list(self.axis_pairwise_correlations),
            "maximum_absolute_axis_correlation": self.maximum_absolute_axis_correlation,
            "axis_correlation_absolute_determinant": (self.axis_correlation_absolute_determinant),
            "axis_occupied_bin_counts": list(self.axis_occupied_bin_counts),
            "joint_effective_cells": self.joint_effective_cells,
            "joint_shannon_effective_cells": self.joint_shannon_effective_cells,
            "archive_capacity": self.archive_capacity,
            "minimum_samples": self.minimum_samples,
            "minimum_axis_bins": self.minimum_axis_bins,
            "minimum_joint_cells": self.minimum_joint_cells,
            "minimum_joint_shannon_effective_cells": self.minimum_joint_shannon_effective_cells,
            "maximum_absolute_axis_correlation_allowed": (
                self.maximum_absolute_axis_correlation_allowed
            ),
            "minimum_axis_correlation_absolute_determinant": (
                self.minimum_axis_correlation_absolute_determinant
            ),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_realised_qd_richness(
    descriptors: Sequence[RealisedQDDescriptor],
    *,
    spec: ArchiveSpec = HM3D_REALISED_QD_ARCHIVE_SPEC,
    minimum_samples: int = MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION,
    minimum_axis_bins: int = 2,
    minimum_joint_cells: int = MINIMUM_REALISED_QD_JOINT_CELLS,
    minimum_joint_shannon_effective_cells: float = MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS,
    maximum_absolute_axis_correlation: float = MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION,
    minimum_axis_correlation_absolute_determinant: float = (
        MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT
    ),
) -> RealisedQDRichnessAudit:
    """Reject a sparse, dominated, or effectively one-dimensional archive.

    The archive is intentionally only three-dimensional: adding a fourth
    axis would multiply the sparse 64-cell grid to 256 cells.  Instead, this
    audit requires adequate *real* outcome support and rejects a history whose
    three reported axes vary only along the same line.  It remains an
    admission test, not evidence that QD outperforms no-QD.
    """

    if (
        minimum_samples < 1
        or minimum_axis_bins < 2
        or minimum_joint_cells < 2
        or minimum_joint_cells > spec.capacity
        or not 1.0 <= minimum_joint_shannon_effective_cells <= minimum_joint_cells
        or not 0.0 <= maximum_absolute_axis_correlation < 1.0
        or not 0.0 < minimum_axis_correlation_absolute_determinant <= 1.0
    ):
        raise ValueError("QD richness thresholds are below the minimum evidence floor")
    rows = tuple(descriptors)
    if len(spec.axes) != 3:
        raise ValueError("HM3D realised-QD audit requires exactly three archive axes")
    cells = tuple(spec.cell(row.values) for row in rows)
    variances = tuple(
        0.0 if not rows else pvariance(tuple(row.values[axis] for row in rows)) for axis in range(3)
    )
    axis_values = tuple(tuple(row.values[axis] for row in rows) for axis in range(3))
    correlations = (
        _pearson(axis_values[0], axis_values[1]),
        _pearson(axis_values[0], axis_values[2]),
        _pearson(axis_values[1], axis_values[2]),
    )
    maximum_correlation = max((abs(value) for value in correlations), default=0.0)
    first_second, first_third, second_third = correlations
    correlation_determinant = abs(
        1.0
        + 2.0 * first_second * first_third * second_third
        - first_second**2
        - first_third**2
        - second_third**2
    )
    occupied = tuple(len({cell[axis] for cell in cells}) for axis in range(3))
    reasons: list[str] = []
    if len(rows) < minimum_samples:
        reasons.append("INSUFFICIENT_OUTCOME_GROUNDED_SAMPLES")
    for axis, count in zip(spec.axes, occupied, strict=True):
        if count < minimum_axis_bins:
            reasons.append(f"DEGENERATE_{axis.name.upper()}")
    joint_cells = len(set(cells))
    if joint_cells < minimum_joint_cells:
        reasons.append("INSUFFICIENT_JOINT_ARCHIVE_CELLS")
    shannon_effective_cells = _shannon_effective_cell_count(cells)
    if shannon_effective_cells + 1.0e-12 < minimum_joint_shannon_effective_cells:
        reasons.append("JOINT_ARCHIVE_DOMINATED_BY_TOO_FEW_MODES")
    if maximum_correlation - 1.0e-12 > maximum_absolute_axis_correlation:
        reasons.append("QD_DESCRIPTOR_AXES_COLLINEAR")
    if correlation_determinant + 1.0e-12 < minimum_axis_correlation_absolute_determinant:
        reasons.append("QD_DESCRIPTOR_EFFECTIVE_DIMENSION_TOO_LOW")
    return RealisedQDRichnessAudit(
        sample_count=len(rows),
        axis_variances=variances,
        axis_pairwise_correlations=correlations,
        maximum_absolute_axis_correlation=maximum_correlation,
        axis_correlation_absolute_determinant=correlation_determinant,
        axis_occupied_bin_counts=occupied,
        joint_effective_cells=joint_cells,
        joint_shannon_effective_cells=shannon_effective_cells,
        archive_capacity=spec.capacity,
        minimum_samples=minimum_samples,
        minimum_axis_bins=minimum_axis_bins,
        minimum_joint_cells=minimum_joint_cells,
        minimum_joint_shannon_effective_cells=minimum_joint_shannon_effective_cells,
        maximum_absolute_axis_correlation_allowed=maximum_absolute_axis_correlation,
        minimum_axis_correlation_absolute_determinant=(
            minimum_axis_correlation_absolute_determinant
        ),
        status="QD_DESCRIPTOR_ADMITTED" if not reasons else "QD_DESCRIPTOR_NOT_ADMITTED",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class QDDescriptorFamilyScreenRow:
    """One pre-registered three-axis family evaluated on train outcomes."""

    family_id: str
    axis_names: tuple[str, str, str]
    richness_audit: RealisedQDRichnessAudit
    footprint_separation_audit: RealisedQDFootprintSeparationAudit
    cross_scene_support_audit: QDDescriptorFamilyCrossSceneSupportAudit
    eligibility_score: float | None
    admitted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "axis_names": list(self.axis_names),
            "richness_audit": self.richness_audit.to_dict(),
            "footprint_separation_audit": self.footprint_separation_audit.to_dict(),
            "cross_scene_support_audit": self.cross_scene_support_audit.to_dict(),
            "eligibility_score": self.eligibility_score,
            "admitted": self.admitted,
        }


@dataclass(frozen=True, slots=True)
class QDDescriptorFamilyScreen:
    """Falsify a hand-picked descriptor family before it reaches validation."""

    feature_schema_version: str
    current_family_id: str
    recommended_family_id: str | None
    family_rows: tuple[QDDescriptorFamilyScreenRow, ...]
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "current_family_id": self.current_family_id,
            "recommended_family_id": self.recommended_family_id,
            "family_rows": [row.to_dict() for row in self.family_rows],
            "status": self.status,
            "reasons": list(self.reasons),
            "selection_rule": (
                "Pre-registered train-only families are eligible only when both outcome "
                "richness and public-footprint separation pass. Among eligible families, "
                "maximize the fixed reliability score; lexical family_id breaks exact ties."
            ),
        }


@dataclass(frozen=True, slots=True)
class QDDescriptorFamilyCrossSceneSupportAudit:
    """Reject a descriptor that is rich only because of one train scene.

    HM3D scenes differ sharply in vertical opportunity and room topology.  A
    pooled archive can consequently appear non-collinear when one building
    contributes all climbing behaviour and another contributes all horizontal
    separation.  That is not a reusable team repertoire.  This low-cost audit
    asks every selected train scene to support all three axes before a family
    can compete; task reward remains deliberately absent from the test.
    """

    scene_count: int
    scene_sample_counts: tuple[tuple[str, int], ...]
    scene_axis_occupied_bin_counts: tuple[tuple[str, tuple[int, int, int]], ...]
    scene_joint_effective_cells: tuple[tuple[str, int], ...]
    minimum_scene_count: int
    minimum_samples_per_scene: int
    minimum_axis_bins_per_scene: int
    minimum_joint_cells_per_scene: int
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_count": self.scene_count,
            "scene_sample_counts": [list(row) for row in self.scene_sample_counts],
            "scene_axis_occupied_bin_counts": [
                [scene_id, list(counts)] for scene_id, counts in self.scene_axis_occupied_bin_counts
            ],
            "scene_joint_effective_cells": [list(row) for row in self.scene_joint_effective_cells],
            "minimum_scene_count": self.minimum_scene_count,
            "minimum_samples_per_scene": self.minimum_samples_per_scene,
            "minimum_axis_bins_per_scene": self.minimum_axis_bins_per_scene,
            "minimum_joint_cells_per_scene": self.minimum_joint_cells_per_scene,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_qd_descriptor_family_cross_scene_support(
    descriptors: Sequence[RealisedQDDescriptor],
    scene_ids: Sequence[str],
    *,
    spec: ArchiveSpec = HM3D_REALISED_QD_ARCHIVE_SPEC,
    minimum_scene_count: int = 2,
    minimum_samples_per_scene: int = 6,
    minimum_axis_bins_per_scene: int = 2,
    minimum_joint_cells_per_scene: int = 3,
) -> QDDescriptorFamilyCrossSceneSupportAudit:
    """Require each axis to be supported in every train calibration scene.

    The floor is intentionally lower than the pooled 12-outcome admission:
    this is a portability check, not a separate performance experiment.  It
    catches the specific failure where one scene creates apparent three-axis
    variety while the other scene contributes a constant descriptor.
    """

    rows = tuple(descriptors)
    scenes = tuple(scene_ids)
    if len(rows) != len(scenes):
        raise ValueError("QD descriptors and scene IDs must have matching counts")
    if (
        minimum_scene_count < 2
        or minimum_samples_per_scene < 1
        or minimum_axis_bins_per_scene < 2
        or minimum_joint_cells_per_scene < 2
        or minimum_joint_cells_per_scene > spec.capacity
    ):
        raise ValueError("invalid QD descriptor cross-scene support floor")

    by_scene: dict[str, list[RealisedQDDescriptor]] = {}
    for scene_id, descriptor in zip(scenes, rows, strict=True):
        require_identifier(scene_id, "QD descriptor scene_id")
        by_scene.setdefault(scene_id, []).append(descriptor)
    sample_counts: list[tuple[str, int]] = []
    axis_counts: list[tuple[str, tuple[int, int, int]]] = []
    joint_counts: list[tuple[str, int]] = []
    reasons: list[str] = []
    for scene_id in sorted(by_scene):
        scene_rows = tuple(by_scene[scene_id])
        cells = tuple(spec.cell(row.values) for row in scene_rows)
        occupied = tuple(len({cell[axis] for cell in cells}) for axis in range(3))
        joint = len(set(cells))
        sample_counts.append((scene_id, len(scene_rows)))
        axis_counts.append((scene_id, occupied))
        joint_counts.append((scene_id, joint))
        if len(scene_rows) < minimum_samples_per_scene:
            reasons.append(f"QD_DESCRIPTOR_SCENE_TOO_FEW_OUTCOMES_{scene_id}")
        if any(count < minimum_axis_bins_per_scene for count in occupied):
            reasons.append(f"QD_DESCRIPTOR_SCENE_AXIS_DEGENERATE_{scene_id}")
        if joint < minimum_joint_cells_per_scene:
            reasons.append(f"QD_DESCRIPTOR_SCENE_JOINT_SPACE_SPARSE_{scene_id}")
    if len(by_scene) < minimum_scene_count:
        reasons.append("QD_DESCRIPTOR_FAMILY_INSUFFICIENT_CROSS_SCENE_SUPPORT")
    return QDDescriptorFamilyCrossSceneSupportAudit(
        scene_count=len(by_scene),
        scene_sample_counts=tuple(sample_counts),
        scene_axis_occupied_bin_counts=tuple(axis_counts),
        scene_joint_effective_cells=tuple(joint_counts),
        minimum_scene_count=minimum_scene_count,
        minimum_samples_per_scene=minimum_samples_per_scene,
        minimum_axis_bins_per_scene=minimum_axis_bins_per_scene,
        minimum_joint_cells_per_scene=minimum_joint_cells_per_scene,
        status=(
            "QD_DESCRIPTOR_FAMILY_CROSS_SCENE_ADMITTED"
            if not reasons
            else "QD_DESCRIPTOR_FAMILY_CROSS_SCENE_NOT_ADMITTED"
        ),
        reasons=tuple(reasons),
    )


def audit_pre_registered_qd_descriptor_families(
    features: Sequence[OutcomeQDFeatureVector],
    public_free_footprints: Sequence[Sequence[Sequence[int]]],
    scene_ids: Sequence[str],
) -> QDDescriptorFamilyScreen:
    """Screen the fixed QD families without opening a higher-dimensional archive.

    This is a train-only *falsification* step.  It cannot establish task gain,
    and it never reads validation outcomes.  It does make the current v4
    family fail closed when an already pre-registered outcome-only alternative
    provides a richer, less redundant and semantically separable repertoire.
    """

    rows = tuple(features)
    scenes = tuple(scene_ids)
    if len(rows) != len(public_free_footprints) or len(rows) != len(scenes):
        raise ValueError("QD feature vectors, public footprints, and scene IDs must align")
    family_rows: list[QDDescriptorFamilyScreenRow] = []
    for family_id, axis_names in HM3D_QD_DESCRIPTOR_FAMILIES:
        descriptors = tuple(
            RealisedQDDescriptor(*descriptor_values_for_qd_family(feature, family_id))
            for feature in rows
        )
        richness = audit_realised_qd_richness(descriptors)
        footprint_separation = audit_realised_qd_footprint_separation(
            descriptors, public_free_footprints
        )
        cross_scene_support = audit_qd_descriptor_family_cross_scene_support(descriptors, scenes)
        admitted = (
            richness.status == "QD_DESCRIPTOR_ADMITTED"
            and footprint_separation.status == "QD_FOOTPRINT_SEPARATION_ADMITTED"
            and cross_scene_support.status == "QD_DESCRIPTOR_FAMILY_CROSS_SCENE_ADMITTED"
        )
        # The score only ranks already-admitted families.  It rewards even
        # support across cells, effective independent dimensions and semantic
        # footprint separation; it has no task reward or validation term.
        score = None
        if admitted:
            score = (
                richness.joint_shannon_effective_cells
                / richness.minimum_joint_shannon_effective_cells
                + richness.axis_correlation_absolute_determinant
                / richness.minimum_axis_correlation_absolute_determinant
                + footprint_separation.footprint_separation_margin
                / footprint_separation.minimum_footprint_separation_margin
                + (1.0 - richness.maximum_absolute_axis_correlation)
            )
        family_rows.append(
            QDDescriptorFamilyScreenRow(
                family_id=family_id,
                axis_names=axis_names,
                richness_audit=richness,
                footprint_separation_audit=footprint_separation,
                cross_scene_support_audit=cross_scene_support,
                eligibility_score=score,
                admitted=admitted,
            )
        )
    admitted_rows = tuple(row for row in family_rows if row.admitted)
    recommended = (
        min(admitted_rows, key=lambda row: (-float(row.eligibility_score), row.family_id))
        if admitted_rows
        else None
    )
    reasons: list[str] = []
    if recommended is None:
        reasons.append("NO_PRE_REGISTERED_OUTCOME_DESCRIPTOR_FAMILY_IS_ADMITTED")
    elif recommended.family_id != HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID:
        reasons.append("CURRENT_QD_DESCRIPTOR_FAMILY_IS_REDUNDANT_OR_WEAKER_ON_TRAIN_OUTCOMES")
    return QDDescriptorFamilyScreen(
        feature_schema_version=HM3D_QD_FEATURE_VECTOR_SCHEMA_VERSION,
        current_family_id=HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
        recommended_family_id=None if recommended is None else recommended.family_id,
        family_rows=tuple(family_rows),
        status=(
            "QD_DESCRIPTOR_FAMILY_CURRENT_ADMITTED"
            if not reasons
            else "QD_DESCRIPTOR_FAMILY_REDESIGN_REQUIRED"
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class CandidateIntentRichnessAudit:
    """Pre-execution audit of the public modes offered to a QD selector.

    ``planned_descriptor`` is called *intent* here to prevent a category error:
    it may index historical outcome evidence, but it must never be placed in a
    realised-QD archive as though it were the executed behaviour.
    """

    feasible_candidate_count: int
    axis_occupied_bin_counts: tuple[int, int, int]
    joint_effective_cells: int
    joint_shannon_effective_cells: float
    minimum_feasible_candidates: int
    minimum_axis_bins: int
    minimum_joint_cells: int
    minimum_joint_shannon_effective_cells: float
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible_candidate_count": self.feasible_candidate_count,
            "axis_occupied_bin_counts": list(self.axis_occupied_bin_counts),
            "joint_effective_cells": self.joint_effective_cells,
            "joint_shannon_effective_cells": self.joint_shannon_effective_cells,
            "minimum_feasible_candidates": self.minimum_feasible_candidates,
            "minimum_axis_bins": self.minimum_axis_bins,
            "minimum_joint_cells": self.minimum_joint_cells,
            "minimum_joint_shannon_effective_cells": self.minimum_joint_shannon_effective_cells,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_public_candidate_intent_richness(
    candidates: Sequence[CandidateFragmentManifest],
    *,
    spec: ArchiveSpec = HM3D_CANDIDATE_INTENT_SPEC,
    minimum_feasible_candidates: int = 6,
    minimum_axis_bins: int = 2,
    minimum_joint_cells: int = 6,
    minimum_joint_shannon_effective_cells: float = 4.0,
) -> CandidateIntentRichnessAudit:
    """Reject an emitter that offers only one public behaviour mode.

    This audit assesses candidate *availability*, not performance.  It avoids
    mistaking a selector that always sees the same behaviour for a failed QD
    algorithm.  The later realised-descriptor audit remains authoritative for
    archive admission.
    """

    if (
        minimum_feasible_candidates < 2
        or minimum_axis_bins < 2
        or minimum_joint_cells < 2
        or not 1.0 <= minimum_joint_shannon_effective_cells <= minimum_joint_cells
    ):
        raise ValueError("candidate-intent thresholds are below the minimum evidence floor")
    if spec != HM3D_CANDIDATE_INTENT_SPEC:
        raise ValueError("HM3D candidate-intent audit needs the frozen intent schema")
    if len(spec.axes) != 3:
        raise ValueError("HM3D candidate-intent audit requires exactly three axes")
    legal = tuple(candidate for candidate in candidates if candidate.feasible)
    intents = tuple(
        _checked_descriptor(candidate.planned_descriptor, "candidate intent") for candidate in legal
    )
    cells = tuple(spec.cell(intent) for intent in intents)
    occupied = tuple(len({cell[axis] for cell in cells}) for axis in range(3))
    reasons: list[str] = []
    if len(legal) < minimum_feasible_candidates:
        reasons.append("INSUFFICIENT_FEASIBLE_CANDIDATES")
    for axis, count in zip(spec.axes, occupied, strict=True):
        if count < minimum_axis_bins:
            reasons.append(f"DEGENERATE_INTENT_{axis.name.upper()}")
    joint_cells = len(set(cells))
    if joint_cells < minimum_joint_cells:
        reasons.append("INSUFFICIENT_INTENT_JOINT_CELLS")
    shannon_effective_cells = _shannon_effective_cell_count(cells)
    if shannon_effective_cells + 1.0e-12 < minimum_joint_shannon_effective_cells:
        reasons.append("INTENT_JOINT_SPACE_DOMINATED_BY_TOO_FEW_MODES")
    return CandidateIntentRichnessAudit(
        feasible_candidate_count=len(legal),
        axis_occupied_bin_counts=occupied,
        joint_effective_cells=joint_cells,
        joint_shannon_effective_cells=shannon_effective_cells,
        minimum_feasible_candidates=minimum_feasible_candidates,
        minimum_axis_bins=minimum_axis_bins,
        minimum_joint_cells=minimum_joint_cells,
        minimum_joint_shannon_effective_cells=minimum_joint_shannon_effective_cells,
        status="QD_CANDIDATE_INTENT_ADMITTED"
        if not reasons
        else "QD_CANDIDATE_INTENT_NOT_ADMITTED",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class ValueProtectedCandidateDiversityAudit:
    """Check whether QD has a legitimate choice among near-value candidates.

    A candidate pool may be globally varied while every alternative behaviour
    is much worse than the public value-best action.  A conservative selector
    must not select those alternatives merely to fill an archive.  This audit
    therefore measures diversity only inside the same value-protected set used
    by the QD selectors; it is an opportunity diagnostic, never a realised
    archive coordinate.
    """

    feasible_candidate_count: int
    value_protected_candidate_count: int
    value_protected_joint_cells: int
    best_public_utility: float | None
    public_utility_floor: float | None
    utility_slack: float
    minimum_value_protected_candidates: int
    minimum_value_protected_joint_cells: int
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feasible_candidate_count": self.feasible_candidate_count,
            "value_protected_candidate_count": self.value_protected_candidate_count,
            "value_protected_joint_cells": self.value_protected_joint_cells,
            "best_public_utility": self.best_public_utility,
            "public_utility_floor": self.public_utility_floor,
            "utility_slack": self.utility_slack,
            "minimum_value_protected_candidates": self.minimum_value_protected_candidates,
            "minimum_value_protected_joint_cells": self.minimum_value_protected_joint_cells,
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_value_protected_candidate_diversity(
    candidates: Sequence[CandidateFragmentManifest],
    *,
    base_utilities: Mapping[str, float] | None = None,
    utility_slack: float = 0.10,
    spec: ArchiveSpec = HM3D_CANDIDATE_INTENT_SPEC,
    minimum_value_protected_candidates: int = 2,
    minimum_value_protected_joint_cells: int = 2,
) -> ValueProtectedCandidateDiversityAudit:
    """Audit whether QD can choose behaviour without sacrificing public value.

    The utility band exactly matches :class:`PlannedQDSelector` and
    :class:`OutcomeGroundedQDSelector`: QD may only break ties within
    ``utility_slack`` of the observed public utility range.  A failed audit is
    recorded for failure attribution; the online selector safely abstains to
    the value-best candidate instead of making the episode invalid.
    """

    if spec != HM3D_CANDIDATE_INTENT_SPEC or len(spec.axes) != 3:
        raise ValueError("value-protected QD audit needs the frozen intent schema")
    if not 0.0 <= finite_number(utility_slack, "QD utility_slack") <= 1.0:
        raise ValueError("QD utility_slack must lie in [0, 1]")
    if minimum_value_protected_candidates < 2 or minimum_value_protected_joint_cells < 2:
        raise ValueError("value-protected QD evidence floor is too low")

    legal = tuple(candidate for candidate in candidates if candidate.feasible)
    reasons: list[str] = []
    if not legal:
        reasons.append("QD_NO_FEASIBLE_CANDIDATE_FOR_VALUE_PROTECTED_DIVERSITY")
        return ValueProtectedCandidateDiversityAudit(
            feasible_candidate_count=0,
            value_protected_candidate_count=0,
            value_protected_joint_cells=0,
            best_public_utility=None,
            public_utility_floor=None,
            utility_slack=utility_slack,
            minimum_value_protected_candidates=minimum_value_protected_candidates,
            minimum_value_protected_joint_cells=minimum_value_protected_joint_cells,
            status="QD_VALUE_PROTECTED_DIVERSITY_NOT_ADMITTED",
            reasons=tuple(reasons),
        )
    if base_utilities is None:
        utilities = tuple(
            candidate.quality_hint / max(1.0e-9, candidate.cost_hint) for candidate in legal
        )
    else:
        legal_ids = {candidate.candidate_id for candidate in legal}
        if set(base_utilities) != legal_ids:
            raise ValueError("candidate value provider must score exactly the legal public pool")
        utilities = tuple(
            finite_number(base_utilities[candidate.candidate_id], "candidate base utility")
            for candidate in legal
        )
    best = max(utilities)
    floor = best - utility_slack * (best - min(utilities))
    protected = tuple(
        candidate
        for candidate, utility in zip(legal, utilities, strict=True)
        if utility >= floor - 1.0e-12
    )
    cells = {
        spec.cell(_checked_descriptor(candidate.planned_descriptor, "candidate intent"))
        for candidate in protected
    }
    if len(protected) < minimum_value_protected_candidates:
        reasons.append("QD_TOO_FEW_VALUE_PROTECTED_CANDIDATES")
    if len(cells) < minimum_value_protected_joint_cells:
        reasons.append("QD_NO_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY")
    return ValueProtectedCandidateDiversityAudit(
        feasible_candidate_count=len(legal),
        value_protected_candidate_count=len(protected),
        value_protected_joint_cells=len(cells),
        best_public_utility=best,
        public_utility_floor=floor,
        utility_slack=utility_slack,
        minimum_value_protected_candidates=minimum_value_protected_candidates,
        minimum_value_protected_joint_cells=minimum_value_protected_joint_cells,
        status=(
            "QD_VALUE_PROTECTED_DIVERSITY_ADMITTED"
            if not reasons
            else "QD_VALUE_PROTECTED_DIVERSITY_NOT_ADMITTED"
        ),
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class IntentRealisedAlignmentAudit:
    """Whether public candidate intent has usable predictive relation to outcomes."""

    sample_count: int
    axis_correlations: tuple[float, float, float]
    mean_l2_gap: float
    aligned_axis_count: int
    global_mean_mse: float
    leave_one_out_knn_mse: float
    relative_prediction_mse_reduction: float
    scene_count: int | None
    cross_scene_global_mean_mse: float | None
    cross_scene_knn_mse: float | None
    cross_scene_relative_prediction_mse_reduction: float | None
    minimum_scene_count: int | None
    minimum_samples: int
    minimum_axis_correlation: float
    minimum_aligned_axes: int
    minimum_relative_prediction_mse_reduction: float
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "axis_correlations": list(self.axis_correlations),
            "mean_l2_gap": self.mean_l2_gap,
            "aligned_axis_count": self.aligned_axis_count,
            "global_mean_mse": self.global_mean_mse,
            "leave_one_out_knn_mse": self.leave_one_out_knn_mse,
            "relative_prediction_mse_reduction": self.relative_prediction_mse_reduction,
            "scene_count": self.scene_count,
            "cross_scene_global_mean_mse": self.cross_scene_global_mean_mse,
            "cross_scene_knn_mse": self.cross_scene_knn_mse,
            "cross_scene_relative_prediction_mse_reduction": (
                self.cross_scene_relative_prediction_mse_reduction
            ),
            "minimum_scene_count": self.minimum_scene_count,
            "minimum_samples": self.minimum_samples,
            "minimum_axis_correlation": self.minimum_axis_correlation,
            "minimum_aligned_axes": self.minimum_aligned_axes,
            "minimum_relative_prediction_mse_reduction": (
                self.minimum_relative_prediction_mse_reduction
            ),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def audit_intent_realised_alignment(
    intents: Sequence[Sequence[float]],
    descriptors: Sequence[RealisedQDDescriptor],
    *,
    minimum_samples: int = 12,
    minimum_axis_correlation: float = 0.15,
    minimum_aligned_axes: int = 2,
    minimum_relative_prediction_mse_reduction: float = 0.05,
    scene_ids: Sequence[str] | None = None,
    minimum_scene_count: int = 2,
) -> IntentRealisedAlignmentAudit:
    """Require evidence that emitted public modes survive real execution.

    Correlation alone can be high because both fields drift with time or scene
    scale.  We therefore also require a leave-one-out nearest-intent predictor
    to beat the global realised-descriptor mean.  This verifies the property
    the selector actually needs: public intent must carry some out-of-sample
    information about realised behaviour.  When ``scene_ids`` is supplied,
    the stronger leave-one-scene-out check prevents a descriptor that only
    tracks one building's scale from entering the train archive.  The later
    paired P08 result, not this admission audit, establishes task-level benefit.
    """

    if minimum_samples < 2 or not 0.0 < minimum_axis_correlation <= 1.0:
        raise ValueError("invalid intent-realised alignment threshold")
    if not 1 <= minimum_aligned_axes <= 3:
        raise ValueError("minimum_aligned_axes must be in [1, 3]")
    if not 0.0 <= minimum_relative_prediction_mse_reduction < 1.0:
        raise ValueError("minimum relative prediction improvement must be in [0, 1)")
    if minimum_scene_count < 2:
        raise ValueError("minimum_scene_count must be at least two")
    intent_rows = tuple(_checked_descriptor(values, "candidate intent") for values in intents)
    realised_rows = tuple(descriptor.values for descriptor in descriptors)
    if len(intent_rows) != len(realised_rows):
        raise ValueError("intent and realised descriptor samples must have matching counts")
    correlations = tuple(
        _pearson(
            tuple(intent[axis] for intent in intent_rows),
            tuple(realised[axis] for realised in realised_rows),
        )
        for axis in range(3)
    )
    mean_l2_gap = (
        0.0
        if not intent_rows
        else sum(
            math.sqrt(sum((intent[axis] - realised[axis]) ** 2 for axis in range(3)))
            for intent, realised in zip(intent_rows, realised_rows, strict=True)
        )
        / len(intent_rows)
    )
    aligned_axes = sum(correlation >= minimum_axis_correlation for correlation in correlations)
    global_mean = tuple(
        sum(realised[axis] for realised in realised_rows) / len(realised_rows)
        if realised_rows
        else 0.0
        for axis in range(3)
    )
    global_mean_mse = (
        0.0
        if not realised_rows
        else sum(
            sum((value - global_mean[axis]) ** 2 for axis, value in enumerate(realised))
            for realised in realised_rows
        )
        / (len(realised_rows) * 3)
    )
    prediction_errors: list[float] = []
    for held_out, intent in enumerate(intent_rows):
        neighbours = sorted(
            (
                math.sqrt(sum((intent[axis] - other_intent[axis]) ** 2 for axis in range(3))),
                other_index,
            )
            for other_index, other_intent in enumerate(intent_rows)
            if other_index != held_out
        )[:3]
        if not neighbours:
            continue
        weights = tuple(1.0 / max(distance, 1.0e-6) for distance, _ in neighbours)
        normalizer = sum(weights)
        prediction = tuple(
            sum(
                weight * realised_rows[index][axis]
                for weight, (_, index) in zip(weights, neighbours, strict=True)
            )
            / normalizer
            for axis in range(3)
        )
        prediction_errors.extend(
            (prediction[axis] - realised_rows[held_out][axis]) ** 2 for axis in range(3)
        )
    leave_one_out_knn_mse = (
        0.0 if not prediction_errors else sum(prediction_errors) / len(prediction_errors)
    )
    relative_reduction = (
        0.0 if global_mean_mse <= 1.0e-12 else 1.0 - leave_one_out_knn_mse / global_mean_mse
    )
    labels: tuple[str, ...] | None = None
    scene_count: int | None = None
    cross_scene_global_mean_mse: float | None = None
    cross_scene_knn_mse: float | None = None
    cross_scene_relative_reduction: float | None = None
    if scene_ids is not None:
        labels = tuple(scene_ids)
        if len(labels) != len(intent_rows):
            raise ValueError("scene_ids must match intent and realised descriptor samples")
        for scene_id in labels:
            require_identifier(scene_id, "scene_id")
        scene_count = len(set(labels))
        cross_baseline_errors: list[float] = []
        cross_prediction_errors: list[float] = []
        for held_out, intent in enumerate(intent_rows):
            train_indices = tuple(
                index for index, label in enumerate(labels) if label != labels[held_out]
            )
            if not train_indices:
                continue
            train_mean = tuple(
                sum(realised_rows[index][axis] for index in train_indices) / len(train_indices)
                for axis in range(3)
            )
            neighbours = sorted(
                (
                    math.sqrt(
                        sum((intent[axis] - intent_rows[index][axis]) ** 2 for axis in range(3))
                    ),
                    index,
                )
                for index in train_indices
            )[:3]
            weights = tuple(1.0 / max(distance, 1.0e-6) for distance, _ in neighbours)
            normalizer = sum(weights)
            prediction = tuple(
                sum(
                    weight * realised_rows[index][axis]
                    for weight, (_, index) in zip(weights, neighbours, strict=True)
                )
                / normalizer
                for axis in range(3)
            )
            cross_baseline_errors.extend(
                (train_mean[axis] - realised_rows[held_out][axis]) ** 2 for axis in range(3)
            )
            cross_prediction_errors.extend(
                (prediction[axis] - realised_rows[held_out][axis]) ** 2 for axis in range(3)
            )
        cross_scene_global_mean_mse = (
            0.0
            if not cross_baseline_errors
            else sum(cross_baseline_errors) / len(cross_baseline_errors)
        )
        cross_scene_knn_mse = (
            0.0
            if not cross_prediction_errors
            else sum(cross_prediction_errors) / len(cross_prediction_errors)
        )
        cross_scene_relative_reduction = (
            0.0
            if cross_scene_global_mean_mse <= 1.0e-12
            else 1.0 - cross_scene_knn_mse / cross_scene_global_mean_mse
        )
    reasons: list[str] = []
    if len(intent_rows) < minimum_samples:
        reasons.append("INSUFFICIENT_INTENT_OUTCOME_PAIRS")
    if aligned_axes < minimum_aligned_axes:
        reasons.append("INTENT_DOES_NOT_PREDICT_REALISED_BEHAVIOUR")
    if relative_reduction < minimum_relative_prediction_mse_reduction:
        reasons.append("INTENT_PREDICTOR_DOES_NOT_OUTPERFORM_GLOBAL_MEAN")
    if scene_ids is not None and scene_count is not None:
        if scene_count < minimum_scene_count:
            reasons.append("INSUFFICIENT_CROSS_SCENE_INTENT_OUTCOME_EVIDENCE")
        elif (
            cross_scene_relative_reduction is None
            or cross_scene_relative_reduction < minimum_relative_prediction_mse_reduction
        ):
            reasons.append("INTENT_PREDICTOR_DOES_NOT_TRANSFER_ACROSS_TRAIN_SCENES")
    return IntentRealisedAlignmentAudit(
        sample_count=len(intent_rows),
        axis_correlations=correlations,
        mean_l2_gap=mean_l2_gap,
        aligned_axis_count=aligned_axes,
        global_mean_mse=global_mean_mse,
        leave_one_out_knn_mse=leave_one_out_knn_mse,
        relative_prediction_mse_reduction=relative_reduction,
        scene_count=scene_count,
        cross_scene_global_mean_mse=cross_scene_global_mean_mse,
        cross_scene_knn_mse=cross_scene_knn_mse,
        cross_scene_relative_prediction_mse_reduction=cross_scene_relative_reduction,
        minimum_scene_count=minimum_scene_count if scene_ids is not None else None,
        minimum_samples=minimum_samples,
        minimum_axis_correlation=minimum_axis_correlation,
        minimum_aligned_axes=minimum_aligned_axes,
        minimum_relative_prediction_mse_reduction=minimum_relative_prediction_mse_reduction,
        status="QD_INTENT_OUTCOME_ALIGNMENT_ADMITTED"
        if not reasons
        else "QD_INTENT_OUTCOME_ALIGNMENT_NOT_ADMITTED",
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class OutcomeGroundedQDExperience:
    """One post-execution observation used to predict future realised modes."""

    intent: tuple[float, float, float]
    realised: RealisedQDDescriptor
    public_quality: float
    public_cost: float
    execution_outcome_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _checked_descriptor(self.intent, "candidate intent"))
        quality = finite_number(self.public_quality, "public_quality")
        cost = finite_number(self.public_cost, "public_cost")
        if quality < 0.0 or cost < 0.0:
            raise ValueError("public outcome quality and cost must be non-negative")
        object.__setattr__(self, "public_quality", quality)
        object.__setattr__(self, "public_cost", cost)
        require_sha256(self.execution_outcome_sha256, "execution outcome hash")


@dataclass(frozen=True, slots=True)
class OutcomeGroundedQDSelection:
    """Auditable selection result; no archived trajectory is re-executed."""

    selected_candidate_id: str
    selected_manifest_hash: str
    scores: tuple[tuple[str, float], ...]
    predicted_descriptors: tuple[tuple[str, tuple[float, float, float]], ...]
    evidence_count: int
    base_best_candidate_id: str
    diversity_changed_selection: bool
    eligible_candidate_count: int
    uncertainty_abstained_candidate_count: int
    qd_abstained: bool
    archive_entry_count: int
    archive_revision: int
    public_exploration_need: PublicExplorationNeed
    selected_predicted_descriptor: tuple[float, float, float]
    selected_prediction_uncertainty: float
    selected_need_alignment: float
    base_best_need_alignment: float
    need_changed_selection: bool
    minimum_need_alignment_improvement: float
    qd_abstention_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_manifest_hash": self.selected_manifest_hash,
            "scores": list(self.scores),
            "predicted_descriptors": [
                {"candidate_id": candidate_id, "descriptor": list(descriptor)}
                for candidate_id, descriptor in self.predicted_descriptors
            ],
            "evidence_count": self.evidence_count,
            "base_best_candidate_id": self.base_best_candidate_id,
            "diversity_changed_selection": self.diversity_changed_selection,
            "eligible_candidate_count": self.eligible_candidate_count,
            "uncertainty_abstained_candidate_count": self.uncertainty_abstained_candidate_count,
            "qd_abstained": self.qd_abstained,
            "archive_entry_count": self.archive_entry_count,
            "archive_revision": self.archive_revision,
            "public_exploration_need": self.public_exploration_need.to_dict(),
            "selected_predicted_descriptor": list(self.selected_predicted_descriptor),
            "selected_prediction_uncertainty": self.selected_prediction_uncertainty,
            "selected_need_alignment": self.selected_need_alignment,
            "base_best_need_alignment": self.base_best_need_alignment,
            "need_changed_selection": self.need_changed_selection,
            "minimum_need_alignment_improvement": self.minimum_need_alignment_improvement,
            "qd_abstention_reason": self.qd_abstention_reason,
        }


@dataclass(frozen=True, slots=True)
class PlannedQDSelection:
    """Diagnostic selection record for the intentionally flawed planned-QD control."""

    selected_candidate_id: str
    selected_manifest_hash: str
    scores: tuple[tuple[str, float], ...]
    archive_entry_count: int
    base_best_candidate_id: str
    diversity_changed_selection: bool
    eligible_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_manifest_hash": self.selected_manifest_hash,
            "scores": list(self.scores),
            "archive_entry_count": self.archive_entry_count,
            "archive_semantics": "planned_intent_diagnostic_only",
            "base_best_candidate_id": self.base_best_candidate_id,
            "diversity_changed_selection": self.diversity_changed_selection,
            "eligible_candidate_count": self.eligible_candidate_count,
        }


class PlannedQDSelector:
    """Planned-intent archive control used only to test outcome grounding.

    This control reproduces the category error that the proposed mechanism is
    intended to avoid: it records a candidate's intended descriptor before it
    is executed.  It must therefore remain an explicit P08 diagnostic and
    never be exposed as the realised-QD method or a P10 ranked row.
    """

    def __init__(
        self,
        archive: QDArchive,
        *,
        diversity_weight: float = 0.25,
        utility_slack: float = 0.10,
    ) -> None:
        if archive.spec != HM3D_REALISED_QD_ARCHIVE_SPEC:
            raise ValueError("planned-QD selector needs the HM3D archive spec")
        if diversity_weight < 0.0 or not 0.0 <= utility_slack <= 1.0:
            raise ValueError("planned-QD selector weights are invalid")
        self.archive = archive
        self.diversity_weight = diversity_weight
        self.utility_slack = utility_slack
        self._admission_index = 0

    def observe_intent(
        self,
        intent: Sequence[float],
        *,
        public_quality: float,
        public_cost: float,
        source_id: str,
    ) -> None:
        """Admit an intended mode without looking at its realised outcome."""

        descriptor = _checked_descriptor(intent, "planned candidate intent")
        quality = finite_number(public_quality, "planned public_quality")
        cost = finite_number(public_cost, "planned public_cost")
        if quality < 0.0 or cost < 0.0:
            raise ValueError("planned public quality and cost must be non-negative")
        require_identifier(source_id, "planned-QD source_id")
        index = self._admission_index
        self._admission_index += 1
        synthetic_hash = hashlib.sha256(
            f"planned-qd-diagnostic:{source_id}:{index}".encode()
        ).hexdigest()
        self.archive.add_or_update(
            Elite(
                candidate_id=f"planned-qd-{index}",
                manifest_hash=synthetic_hash,
                behavior_hash=synthetic_hash,
                realised_descriptor=descriptor,
                quality=quality,
                cost=cost,
                feasible=True,
                source="hm3d-planned-qd-diagnostic-only",
            )
        )

    def select(
        self,
        candidates: Sequence[CandidateFragmentManifest],
        *,
        base_utilities: Mapping[str, float] | None = None,
    ) -> tuple[CandidateFragmentManifest, PlannedQDSelection]:
        legal = tuple(candidate for candidate in candidates if candidate.feasible)
        if not legal:
            raise ValueError("planned-QD selector requires at least one feasible candidate")
        if base_utilities is None:
            utilities = tuple(
                candidate.quality_hint / max(1.0e-9, candidate.cost_hint) for candidate in legal
            )
        else:
            legal_ids = {candidate.candidate_id for candidate in legal}
            if set(base_utilities) != legal_ids:
                raise ValueError(
                    "candidate value provider must score exactly the legal public pool"
                )
            utilities = tuple(
                finite_number(base_utilities[candidate.candidate_id], "candidate base utility")
                for candidate in legal
            )
        base_low = min(utilities)
        base_high = max(utilities)
        base_best_index = min(
            range(len(legal)), key=lambda index: (-utilities[index], legal[index].manifest_hash)
        )
        base_best = legal[base_best_index]
        utility_floor = base_high - self.utility_slack * (base_high - base_low)
        eligible = tuple(
            index for index, utility in enumerate(utilities) if utility >= utility_floor - 1.0e-12
        )
        scores: list[tuple[float, CandidateFragmentManifest]] = []
        for index in eligible:
            candidate = legal[index]
            base_utility = utilities[index]
            descriptor = _checked_descriptor(candidate.planned_descriptor, "candidate intent")
            cell = self.archive.spec.cell(descriptor)
            if self.archive.get(cell) is None:
                novelty = 1.0
            else:
                nearest = min(
                    math.sqrt(
                        sum(
                            (descriptor[axis] - elite.realised_descriptor[axis]) ** 2
                            for axis in range(3)
                        )
                    )
                    / math.sqrt(3.0)
                    for _, elite in self.archive.items()
                )
                novelty = nearest
            normalized_base = (
                1.0
                if base_high - base_low <= 1.0e-12
                else (base_utility - base_low) / (base_high - base_low)
            )
            scores.append((normalized_base + self.diversity_weight * novelty, candidate))
        scores.sort(key=lambda row: (-row[0], row[1].manifest_hash))
        _, selected = scores[0]
        self.observe_intent(
            selected.planned_descriptor,
            public_quality=selected.quality_hint,
            public_cost=selected.cost_hint,
            source_id=selected.candidate_id,
        )
        return selected, PlannedQDSelection(
            selected_candidate_id=selected.candidate_id,
            selected_manifest_hash=selected.manifest_hash,
            scores=tuple((candidate.candidate_id, score) for score, candidate in scores),
            archive_entry_count=len(tuple(self.archive.items())),
            base_best_candidate_id=base_best.candidate_id,
            diversity_changed_selection=selected.candidate_id != base_best.candidate_id,
            eligible_candidate_count=len(eligible),
        )


class OutcomeGroundedQDSelector:
    """Select current candidates using a conservative outcome-derived mode model.

    It records a candidate's public *intent* before execution and only learns
    its realised outcome afterwards.  Consequently old manifests are never
    replayed from the archive, and a guard-rewritten or failed plan cannot
    occupy an archive cell merely because it was intended to do so.
    """

    def __init__(
        self,
        archive: QDArchive,
        *,
        minimum_evidence: int = 12,
        neighbours: int = 3,
        diversity_weight: float = 0.25,
        repertoire_novelty_weight: float = 0.05,
        uncertainty_weight: float = 0.10,
        maximum_prediction_uncertainty: float = 0.20,
        utility_slack: float = 0.10,
    ) -> None:
        if archive.spec != HM3D_REALISED_QD_ARCHIVE_SPEC:
            raise ValueError("outcome-grounded selector needs the HM3D realised-QD archive spec")
        if minimum_evidence < neighbours or neighbours < 1:
            raise ValueError("selector evidence and neighbour counts are inconsistent")
        if (
            diversity_weight < 0.0
            or repertoire_novelty_weight < 0.0
            or uncertainty_weight < 0.0
            or not 0.0 <= maximum_prediction_uncertainty <= 1.0
            or not 0.0 <= utility_slack <= 1.0
        ):
            raise ValueError("selector weights must be non-negative")
        self.archive = archive
        self.minimum_evidence = minimum_evidence
        self.neighbours = neighbours
        self.diversity_weight = diversity_weight
        self.repertoire_novelty_weight = repertoire_novelty_weight
        self.uncertainty_weight = uncertainty_weight
        self.maximum_prediction_uncertainty = maximum_prediction_uncertainty
        self.utility_slack = utility_slack
        self._experiences: list[OutcomeGroundedQDExperience] = []

    @property
    def evidence_count(self) -> int:
        return len(self._experiences)

    @property
    def qualified(self) -> bool:
        return (
            self.evidence_count >= self.minimum_evidence
            and len(self.archive) >= MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION
        )

    def observe(
        self,
        candidate: CandidateFragmentManifest,
        realised: RealisedQDDescriptor,
        *,
        public_quality: float,
        public_cost: float,
        execution_outcome_sha256: str,
        execution_feasible: bool,
    ) -> AdmissionDecision:
        """Admit one *executed* candidate into both the predictor and archive.

        The archive is deliberately updated here rather than by a parallel
        caller.  This makes it impossible for the online realised-QD selector
        to learn from a outcome while accidentally leaving its novelty archive
        empty.  ``execution_outcome_sha256`` is the immutable digest of the
        real executor outcome, never a planned route digest.  A collision,
        out-of-bounds, clearance violation, or incomplete fragment ledger
        sets ``execution_feasible`` false and is excluded from both the
        predictor and archive.
        """

        if not candidate.feasible:
            raise ValueError("infeasible candidates cannot train the realised-QD selector")
        require_sha256(execution_outcome_sha256, "execution outcome hash")
        if not isinstance(execution_feasible, bool):
            raise ValueError("execution_feasible must be a boolean")
        if execution_feasible is not True:
            return AdmissionDecision(
                False,
                "EXECUTION_NOT_QD_FEASIBLE",
                None,
                None,
                self.archive.revision,
            )
        self.observe_intent(
            candidate.planned_descriptor,
            realised,
            public_quality=public_quality,
            public_cost=public_cost,
            execution_outcome_sha256=execution_outcome_sha256,
        )
        return self.archive.add_or_update(
            Elite(
                candidate_id=candidate.candidate_id,
                manifest_hash=candidate.manifest_hash,
                behavior_hash=execution_outcome_sha256,
                realised_descriptor=realised.values,
                quality=public_quality,
                cost=public_cost,
                feasible=True,
                source=HM3D_REALISED_QD_SCHEMA_VERSION,
            )
        )

    def observe_intent(
        self,
        intent: Sequence[float],
        realised: RealisedQDDescriptor,
        *,
        public_quality: float,
        public_cost: float,
        execution_outcome_sha256: str,
    ) -> None:
        """Record a outcome-backed history row without retaining a trajectory.

        This narrower method exists for loading an already-admitted train
        archive.  It must still carry the executor-outcome digest: a planned
        descriptor alone is not an experience.
        """

        self._experiences.append(
            OutcomeGroundedQDExperience(
                intent=_checked_descriptor(intent, "candidate intent"),
                realised=realised,
                public_quality=public_quality,
                public_cost=public_cost,
                execution_outcome_sha256=execution_outcome_sha256,
            )
        )

    def _prediction(
        self, intent: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], float]:
        ranked = sorted(
            (
                math.sqrt(sum((intent[axis] - item.intent[axis]) ** 2 for axis in range(3))),
                index,
                item,
            )
            for index, item in enumerate(self._experiences)
        )[: self.neighbours]
        weights = tuple(1.0 / max(1.0e-6, distance) for distance, _, _ in ranked)
        weight_sum = sum(weights)
        prediction = tuple(
            sum(
                weight * item.realised.values[axis]
                for weight, (_, _, item) in zip(weights, ranked, strict=True)
            )
            / weight_sum
            for axis in range(3)
        )
        uncertainty = math.sqrt(
            sum(
                weight
                * sum((item.realised.values[axis] - prediction[axis]) ** 2 for axis in range(3))
                for weight, (_, _, item) in zip(weights, ranked, strict=True)
            )
            / weight_sum
        ) / math.sqrt(3.0)
        return _checked_descriptor(prediction, "predicted realised descriptor"), uncertainty

    def select(
        self,
        candidates: Sequence[CandidateFragmentManifest],
        *,
        base_utilities: Mapping[str, float] | None = None,
        public_exploration_need: PublicExplorationNeed,
    ) -> tuple[CandidateFragmentManifest, OutcomeGroundedQDSelection]:
        if not self.qualified:
            raise ValueError(
                "outcome-grounded QD selector is not qualified; "
                "collect more real execution outcomes and fill "
                f"{MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION} realised archive cells"
            )
        legal = tuple(candidate for candidate in candidates if candidate.feasible)
        if not legal:
            raise ValueError("outcome-grounded selector requires at least one feasible candidate")
        if base_utilities is None:
            utilities = tuple(
                candidate.quality_hint / max(1.0e-9, candidate.cost_hint) for candidate in legal
            )
        else:
            legal_ids = {candidate.candidate_id for candidate in legal}
            if set(base_utilities) != legal_ids:
                raise ValueError(
                    "candidate value provider must score exactly the legal public pool"
                )
            utilities = tuple(
                finite_number(base_utilities[candidate.candidate_id], "candidate base utility")
                for candidate in legal
            )
        base_low = min(utilities)
        base_high = max(utilities)
        base_best_index = min(
            range(len(legal)), key=lambda index: (-utilities[index], legal[index].manifest_hash)
        )
        base_best = legal[base_best_index]
        base_intent = _checked_descriptor(base_best.planned_descriptor, "candidate intent")
        base_prediction, base_uncertainty = self._prediction(base_intent)
        base_need_alignment = public_exploration_need.alignment(base_prediction)
        utility_floor = base_high - self.utility_slack * (base_high - base_low)
        eligible = tuple(
            index for index, utility in enumerate(utilities) if utility >= utility_floor - 1.0e-12
        )
        scores: list[
            tuple[
                float,
                CandidateFragmentManifest,
                tuple[float, float, float],
                float,
                float,
            ]
        ] = []
        uncertainty_abstentions = 0
        qd_abstention_reason: str | None = None
        if not public_exploration_need.active:
            qd_abstention_reason = "PUBLIC_EXPLORATION_NEED_BELOW_ACTIVE_FLOOR"
        for index in eligible:
            candidate = legal[index]
            base_utility = utilities[index]
            intent = _checked_descriptor(candidate.planned_descriptor, "candidate intent")
            prediction, uncertainty = self._prediction(intent)
            if uncertainty > self.maximum_prediction_uncertainty:
                uncertainty_abstentions += 1
                continue
            need_alignment = public_exploration_need.alignment(prediction)
            # QD is only permitted to move away from the public-value optimum
            # when the predicted realised mode has a material advantage for a
            # current, outcome-derived exploration deficit.  Otherwise an
            # archive novelty bonus would merely fill cells.
            if (
                not public_exploration_need.active
                or need_alignment < base_need_alignment + MINIMUM_QD_NEED_ALIGNMENT_IMPROVEMENT
            ):
                continue
            cell = self.archive.spec.cell(prediction)
            if self.archive.get(cell) is None:
                novelty = 1.0
            else:
                elite_descriptors = tuple(
                    elite.realised_descriptor for _, elite in self.archive.items()
                )
                nearest = min(
                    math.sqrt(sum((prediction[axis] - descriptor[axis]) ** 2 for axis in range(3)))
                    / math.sqrt(3.0)
                    for descriptor in elite_descriptors
                )
                novelty = nearest
            normalized_base = (
                1.0
                if base_high - base_low <= 1.0e-12
                else (base_utility - base_low) / (base_high - base_low)
            )
            score = (
                normalized_base
                + self.diversity_weight * need_alignment
                + self.repertoire_novelty_weight * novelty
                - self.uncertainty_weight * uncertainty
            )
            scores.append((score, candidate, prediction, uncertainty, need_alignment))
        scores.sort(key=lambda row: (-row[0], row[1].manifest_hash))
        if scores:
            (
                selected_score,
                selected,
                selected_prediction,
                selected_uncertainty,
                selected_need_alignment,
            ) = scores[0]
            del selected_score
        else:
            selected = base_best
            selected_prediction = base_prediction
            selected_uncertainty = base_uncertainty
            selected_need_alignment = base_need_alignment
            if qd_abstention_reason is None:
                qd_abstention_reason = "NO_VALUE_PROTECTED_CANDIDATE_IMPROVES_CURRENT_PUBLIC_NEED"
        qd_abstained = not scores
        need_changed_selection = (
            selected.candidate_id != base_best.candidate_id
            and selected_need_alignment
            >= base_need_alignment + MINIMUM_QD_NEED_ALIGNMENT_IMPROVEMENT
        )
        selection = OutcomeGroundedQDSelection(
            selected_candidate_id=selected.candidate_id,
            selected_manifest_hash=selected.manifest_hash,
            scores=tuple((candidate.candidate_id, score) for score, candidate, _, _, _ in scores),
            predicted_descriptors=tuple(
                (candidate.candidate_id, prediction) for _, candidate, prediction, _, _ in scores
            ),
            evidence_count=self.evidence_count,
            base_best_candidate_id=base_best.candidate_id,
            diversity_changed_selection=need_changed_selection,
            eligible_candidate_count=len(eligible),
            uncertainty_abstained_candidate_count=uncertainty_abstentions,
            qd_abstained=qd_abstained,
            archive_entry_count=len(self.archive),
            archive_revision=self.archive.revision,
            public_exploration_need=public_exploration_need,
            selected_predicted_descriptor=selected_prediction,
            selected_prediction_uncertainty=selected_uncertainty,
            selected_need_alignment=selected_need_alignment,
            base_best_need_alignment=base_need_alignment,
            need_changed_selection=need_changed_selection,
            minimum_need_alignment_improvement=MINIMUM_QD_NEED_ALIGNMENT_IMPROVEMENT,
            qd_abstention_reason=qd_abstention_reason,
        )
        return selected, selection


__all__ = [
    "CandidateIntentRichnessAudit",
    "HM3D_CANDIDATE_INTENT_SPEC",
    "HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID",
    "HM3D_QD_DESCRIPTOR_FAMILIES",
    "HM3D_QD_FEATURE_VECTOR_SCHEMA_VERSION",
    "HM3D_PUBLIC_EXPLORATION_NEED_SCHEMA_VERSION",
    "HM3D_REALISED_QD_ARCHIVE_SPEC",
    "HM3D_REALISED_QD_SCHEMA_VERSION",
    "HM3D_QD_CALIBRATION_INTENT_MODES",
    "IntentRealisedAlignmentAudit",
    "MAXIMUM_REALISED_QD_AXIS_ABSOLUTE_CORRELATION",
    "MINIMUM_PUBLIC_EXPLORATION_NEED_STRENGTH",
    "MINIMUM_QD_NEED_ALIGNMENT_IMPROVEMENT",
    "MINIMUM_QD_NEED_REALISATION_FIDELITY_RATE",
    "MINIMUM_REALISED_QD_AXIS_CORRELATION_DETERMINANT",
    "MINIMUM_REALISED_QD_JOINT_CELLS",
    "MINIMUM_REALISED_QD_OUTCOMES_FOR_ADMISSION",
    "MINIMUM_REALISED_QD_SHANNON_EFFECTIVE_CELLS",
    "PlannedQDSelection",
    "PlannedQDSelector",
    "PublicExplorationNeed",
    "MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION",
    "RealisedQDDescriptor",
    "RealisedQDFootprintSeparationAudit",
    "RealisedQDModeContrastAudit",
    "RealisedQDReproducibilityAudit",
    "RealisedQDRichnessAudit",
    "QDDescriptorFamilyScreen",
    "QDDescriptorFamilyScreenRow",
    "QDDescriptorFamilyCrossSceneSupportAudit",
    "OutcomeQDFeatureVector",
    "OutcomeGroundedQDExperience",
    "OutcomeGroundedQDSelection",
    "OutcomeGroundedQDSelector",
    "QD_PUBLIC_VALUE_BACKBONE_ID",
    "audit_intent_realised_alignment",
    "audit_pre_registered_qd_descriptor_families",
    "audit_qd_descriptor_family_cross_scene_support",
    "audit_realised_qd_calibration_mode_contrasts",
    "audit_public_candidate_intent_richness",
    "audit_value_protected_candidate_diversity",
    "audit_realised_qd_footprint_separation",
    "audit_realised_qd_reproducibility",
    "audit_realised_qd_richness",
    "public_free_footprint_from_range_outcomes",
    "public_exploration_need_from_public_belief",
    "public_observation_workload_balance_from_range_outcomes",
    "outcome_qd_feature_vector_from_public_outcomes",
    "realised_descriptor_from_public_outcomes",
    "descriptor_values_for_qd_family",
    "qd_selector_backbone_sha256",
    "ValueProtectedCandidateDiversityAudit",
]
