from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    GuardedPath,
    build_public_candidate_pool,
)
from aerocity_method.archives.qd import Elite, QDArchive
from aerocity_method.runtime.hm3d_belief import FREE, PublicRangeRayOutcome, SparseVoxelBelief
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    PlannedQDSelector,
    PublicExplorationNeed,
    RealisedQDDescriptor,
    OutcomeGroundedQDSelector,
    OutcomeQDFeatureVector,
    audit_intent_realised_alignment,
    audit_pre_registered_qd_descriptor_families,
    audit_public_candidate_intent_richness,
    audit_qd_descriptor_family_cross_scene_support,
    audit_realised_qd_calibration_mode_contrasts,
    audit_realised_qd_footprint_separation,
    audit_realised_qd_reproducibility,
    audit_realised_qd_richness,
    audit_value_protected_candidate_diversity,
    descriptor_values_for_qd_family,
    public_exploration_need_from_public_belief,
    public_free_footprint_from_range_outcomes,
    public_observation_workload_balance_from_range_outcomes,
    realised_descriptor_from_public_outcomes,
    outcome_qd_feature_vector_from_public_outcomes,
)


def _candidate_pool() -> tuple:
    from test_hm3d_public_baselines import _state

    return build_public_candidate_pool(
        _state(),
        lambda agent_id, path: GuardedPath(True, path),
        candidate_limit=3,
    )


def _outcome(
    outcome_id: str,
    agent_id: str,
    origin: tuple[float, float, float],
    endpoint: tuple[float, float, float],
) -> PublicRangeRayOutcome:
    return PublicRangeRayOutcome(
        observation_id=outcome_id,
        agent_id=agent_id,
        timestamp_s=1.0,
        origin_m=origin,
        endpoint_m=endpoint,
        hit_occupied=False,
    )


def _execution_hash(index: int) -> str:
    return f"{index:064x}"


def _need(values: tuple[float, float, float]) -> PublicExplorationNeed:
    return PublicExplorationNeed(
        vertical_exploration_deficit=values[0],
        spatial_dispersion_deficit=values[1],
        duplicate_observation_deficit=values[2],
        source_public_belief_sha256="a" * 64,
        source_agent_footprints_sha256="b" * 64,
        source_public_outcome_count=4,
    )


def _seed_outcome_archive(archive: QDArchive) -> None:
    for index, descriptor in enumerate(
        (
            (0.1, 0.1, 0.1),
            (0.4, 0.1, 0.1),
            (0.1, 0.4, 0.1),
            (0.1, 0.1, 0.4),
            (0.6, 0.6, 0.1),
            (0.1, 0.4, 0.4),
            (0.4, 0.1, 0.4),
            (0.7, 0.1, 0.1),
            (0.1, 0.7, 0.1),
            (0.1, 0.1, 0.7),
            (0.7, 0.7, 0.4),
            (0.9, 0.9, 0.9),
        )
    ):
        archive.add_or_update(
            Elite(
                candidate_id=f"outcome-history-{index}",
                manifest_hash=_execution_hash(100 + index),
                behavior_hash=_execution_hash(200 + index),
                realised_descriptor=descriptor,
                quality=1.0,
                cost=1.0,
                feasible=True,
                source="unit-test-outcome-history",
            )
        )


def test_realised_qd_uses_actual_vertical_motion_and_public_team_outcomes() -> None:
    descriptor = realised_descriptor_from_public_outcomes(
        scene_id="scene0",
        agent_ids=("uav0", "uav1"),
        applied_paths_by_agent={
            "uav0": ((0.0, 0.0, 1.0), (0.0, 0.0, 3.0)),
            "uav1": ((2.0, 0.0, 1.0), (4.0, 0.0, 1.0)),
        },
        range_outcomes=(
            _outcome("uav0-ray", "uav0", (0.0, 0.0, 1.0), (0.0, 2.0, 1.0)),
            _outcome("uav1-ray", "uav1", (2.0, 0.0, 1.0), (2.0, 2.0, 1.0)),
        ),
        resolution_m=0.5,
        spatial_reference_m=10.0,
    )

    assert descriptor.vertical_motion_ratio == pytest.approx(0.5)
    assert descriptor.team_spatial_dispersion > 0.0
    assert descriptor.public_observation_complementarity == pytest.approx(1.0)


def test_outcome_feature_vector_separates_vertical_observation_from_empty_climbing() -> None:
    features = outcome_qd_feature_vector_from_public_outcomes(
        scene_id="scene0",
        agent_ids=("uav0", "uav1"),
        applied_paths_by_agent={
            "uav0": ((0.0, 0.0, 1.0), (0.0, 0.0, 3.0)),
            "uav1": ((2.0, 0.0, 1.0), (2.0, 2.0, 1.0)),
        },
        # The team climbed, but its delivered public observations are all at
        # one altitude.  A motion-only coordinate must not be mistaken for
        # evidence of vertical exploration.
        range_outcomes=(
            _outcome("uav0-ray", "uav0", (0.0, 0.0, 1.0), (0.0, 2.0, 1.0)),
            _outcome("uav1-ray", "uav1", (2.0, 0.0, 1.0), (2.0, 2.0, 1.0)),
        ),
        resolution_m=0.5,
        spatial_reference_m=10.0,
    )

    assert features.vertical_motion_ratio == pytest.approx(0.5)
    assert features.public_vertical_observation_span == pytest.approx(0.0)
    assert features.public_unique_contribution_balance == pytest.approx(1.0)
    assert features.public_observation_complementarity == pytest.approx(1.0)
    assert descriptor_values_for_qd_family(features, HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID) == (
        features.vertical_motion_ratio,
        features.team_spatial_dispersion,
        features.public_observation_complementarity,
    )


def test_descriptor_family_screen_rejects_current_redundant_axes() -> None:
    features: list[OutcomeQDFeatureVector] = []
    footprints: list[tuple[tuple[int, int, int], ...]] = []
    scene_ids: list[str] = []
    # 12 train-only behaviour cells, each replayed once.  The deployed v4
    # family has vertical motion and spatial dispersion on the same line;
    # pre-registered observed-height alternatives have three independent axes.
    for motion_index in range(4):
        for span_index in range(3):
            motion = motion_index / 3.0
            span = 0.1 + 0.3 * span_index
            complementarity = ((motion_index + span_index) % 4) / 3.0
            footprint = (
                (motion_index, span_index, int(complementarity * 30)),
                (motion_index, span_index, int(complementarity * 30) + 100),
            )
            for _ in range(2):
                features.append(
                    OutcomeQDFeatureVector(
                        vertical_motion_ratio=motion,
                        public_vertical_observation_span=span,
                        team_spatial_dispersion=motion,
                        public_unique_contribution_balance=motion,
                        public_observation_complementarity=complementarity,
                    )
                )
                footprints.append(footprint)
                scene_ids.append("scene_a" if len(scene_ids) % 2 == 0 else "scene_b")

    screen = audit_pre_registered_qd_descriptor_families(features, footprints, scene_ids)

    assert screen.current_family_id == HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID
    assert screen.status == "QD_DESCRIPTOR_FAMILY_REDESIGN_REQUIRED"
    assert screen.recommended_family_id is not None
    assert screen.recommended_family_id != HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID


def test_descriptor_family_screen_rejects_richness_concentrated_in_one_scene() -> None:
    descriptors: list[RealisedQDDescriptor] = []
    scene_ids: list[str] = []
    # The pooled archive covers many cells, but every outcome from scene_b has
    # the same descriptor.  A QD family that only exists in one layout is not
    # allowed to become the frozen HM3D repertoire.
    patterns = (
        (0.10, 0.10, 0.10),
        (0.10, 0.40, 0.70),
        (0.40, 0.70, 0.10),
        (0.70, 0.10, 0.40),
        (0.70, 0.70, 0.70),
        (0.40, 0.40, 0.70),
    )
    descriptors.extend(RealisedQDDescriptor(*pattern) for pattern in patterns * 2)
    scene_ids.extend("scene_a" for _ in patterns * 2)
    descriptors.extend(RealisedQDDescriptor(0.50, 0.50, 0.50) for _ in range(12))
    scene_ids.extend("scene_b" for _ in range(12))

    audit = audit_qd_descriptor_family_cross_scene_support(descriptors, scene_ids)

    assert audit.status == "QD_DESCRIPTOR_FAMILY_CROSS_SCENE_NOT_ADMITTED"
    assert "QD_DESCRIPTOR_SCENE_AXIS_DEGENERATE_scene_b" in audit.reasons


def test_realised_qd_represents_repeated_sensing_without_rewarding_idle_agents() -> None:
    descriptor = realised_descriptor_from_public_outcomes(
        scene_id="scene0",
        agent_ids=("uav0", "uav1"),
        applied_paths_by_agent={
            "uav0": ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)),
            "uav1": ((0.0, 0.0, 1.0),),
        },
        range_outcomes=(
            _outcome("uav0-ray", "uav0", (0.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
            _outcome("uav1-ray", "uav1", (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        ),
        resolution_m=0.5,
        spatial_reference_m=10.0,
    )

    assert descriptor.vertical_motion_ratio == pytest.approx(0.0)
    assert descriptor.team_spatial_dispersion <= 0.1
    assert descriptor.public_observation_complementarity < 1.0
    assert (
        public_observation_workload_balance_from_range_outcomes(
            scene_id="scene0",
            agent_ids=("uav0", "uav1"),
            range_outcomes=(
                _outcome("uav0-ray", "uav0", (0.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
                _outcome("uav1-ray", "uav1", (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
            ),
            resolution_m=0.5,
        )
        < 1.0
    )


def test_team_dispersion_uses_public_communication_scale_not_one_metre_curve() -> None:
    def descriptor_at(distance_m: float):
        return realised_descriptor_from_public_outcomes(
            scene_id="scene0",
            agent_ids=("uav0", "uav1"),
            applied_paths_by_agent={
                "uav0": ((0.0, 0.0, 1.0),),
                "uav1": ((distance_m, 0.0, 1.0),),
            },
            range_outcomes=(
                _outcome("uav0-ray", "uav0", (0.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
                _outcome(
                    "uav1-ray",
                    "uav1",
                    (distance_m, 0.0, 1.0),
                    (distance_m, 1.0, 1.0),
                ),
            ),
            resolution_m=0.5,
            spatial_reference_m=10.0,
        )

    near, far = descriptor_at(2.0), descriptor_at(8.0)

    assert near.team_spatial_dispersion == pytest.approx(0.2)
    assert far.team_spatial_dispersion == pytest.approx(0.8)


def test_qd_richness_audit_rejects_constant_descriptor_even_with_many_samples() -> None:
    rows = [RealisedQDDescriptor(0.0, 1.0, 1.0) for _ in range(8)]

    audit = audit_realised_qd_richness(rows)

    assert audit.status == "QD_DESCRIPTOR_NOT_ADMITTED"
    assert audit.axis_occupied_bin_counts == (1, 1, 1)
    assert "DEGENERATE_VERTICAL_MOTION_RATIO" in audit.reasons
    assert "INSUFFICIENT_JOINT_ARCHIVE_CELLS" in audit.reasons


def test_qd_richness_audit_accepts_publicly_distinct_realised_modes() -> None:
    patterns = (
        (0.05, 0.05, 0.05),
        (0.05, 0.05, 0.30),
        (0.05, 0.30, 0.55),
        (0.05, 0.55, 0.05),
        (0.30, 0.05, 0.55),
        (0.30, 0.30, 0.05),
        (0.30, 0.55, 0.30),
        (0.30, 0.55, 0.55),
        (0.55, 0.05, 0.30),
        (0.55, 0.30, 0.05),
        (0.55, 0.30, 0.55),
        (0.55, 0.55, 0.30),
    )
    rows = tuple(RealisedQDDescriptor(*values) for values in patterns * 2)

    audit = audit_realised_qd_richness(rows, spec=HM3D_REALISED_QD_ARCHIVE_SPEC)

    assert audit.status == "QD_DESCRIPTOR_ADMITTED"
    assert audit.joint_effective_cells >= 6
    assert audit.joint_shannon_effective_cells >= 4
    assert all(count >= 2 for count in audit.axis_occupied_bin_counts)
    assert audit.axis_correlation_absolute_determinant >= 0.10


def test_qd_richness_audit_rejects_single_dominant_mode_with_token_cells() -> None:
    rows = (
        *(RealisedQDDescriptor(0.05, 0.05, 0.05) for _ in range(7)),
        RealisedQDDescriptor(0.30, 0.30, 0.30),
        RealisedQDDescriptor(0.55, 0.55, 0.55),
        RealisedQDDescriptor(0.80, 0.80, 0.80),
        RealisedQDDescriptor(0.30, 0.55, 0.80),
        RealisedQDDescriptor(0.55, 0.80, 0.30),
    )

    audit = audit_realised_qd_richness(rows)

    assert audit.joint_effective_cells == 6
    assert audit.joint_shannon_effective_cells < 4.0
    assert "JOINT_ARCHIVE_DOMINATED_BY_TOO_FEW_MODES" in audit.reasons


def test_qd_richness_audit_rejects_a_nominally_full_but_collinear_archive() -> None:
    rows = tuple(
        RealisedQDDescriptor(value, value, value)
        for value in (0.05, 0.10, 0.30, 0.35, 0.55, 0.60, 0.80, 0.85) * 3
    )

    audit = audit_realised_qd_richness(rows)

    assert audit.status == "QD_DESCRIPTOR_NOT_ADMITTED"
    assert "QD_DESCRIPTOR_AXES_COLLINEAR" in audit.reasons
    assert "QD_DESCRIPTOR_EFFECTIVE_DIMENSION_TOO_LOW" in audit.reasons


def test_public_footprint_is_derived_only_from_the_current_public_range_outcomes() -> None:
    footprint = public_free_footprint_from_range_outcomes(
        scene_id="scene0",
        agent_ids=("uav0", "uav1"),
        range_outcomes=(
            _outcome("r0", "uav0", (0.0, 0.0, 1.0), (2.0, 0.0, 1.0)),
            _outcome("r1", "uav1", (0.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
        ),
        resolution_m=0.5,
    )

    assert footprint
    assert all(len(key) == 3 for key in footprint)


def test_qd_footprint_separation_rejects_rich_scalar_cells_that_observe_the_same_space() -> None:
    descriptors = tuple(
        RealisedQDDescriptor(*values)
        for values in (
            (0.05, 0.05, 0.05),
            (0.05, 0.05, 0.05),
            (0.30, 0.30, 0.30),
            (0.30, 0.30, 0.30),
            (0.55, 0.55, 0.55),
            (0.55, 0.55, 0.55),
            (0.80, 0.80, 0.80),
            (0.80, 0.80, 0.80),
            (0.30, 0.55, 0.80),
            (0.30, 0.55, 0.80),
            (0.55, 0.80, 0.30),
            (0.55, 0.80, 0.30),
        )
    )
    identical = tuple(((0, 0, 0), (0, 0, 1), (0, 0, 2)) for _ in descriptors)

    audit = audit_realised_qd_footprint_separation(descriptors, identical)

    assert audit.status == "QD_FOOTPRINT_SEPARATION_NOT_ADMITTED"
    assert "QD_CELLS_DO_NOT_SEPARATE_PUBLIC_EXPLORATION_FOOTPRINTS" in audit.reasons


def test_qd_footprint_separation_accepts_cells_with_distinct_public_observation_modes() -> None:
    descriptors = tuple(
        RealisedQDDescriptor(*values)
        for values in (
            (0.05, 0.05, 0.05),
            (0.05, 0.05, 0.05),
            (0.30, 0.30, 0.30),
            (0.30, 0.30, 0.30),
            (0.55, 0.55, 0.55),
            (0.55, 0.55, 0.55),
            (0.80, 0.80, 0.80),
            (0.80, 0.80, 0.80),
            (0.30, 0.55, 0.80),
            (0.30, 0.55, 0.80),
            (0.55, 0.80, 0.30),
            (0.55, 0.80, 0.30),
        )
    )
    footprints = tuple(
        tuple((100 * (index // 2) + voxel, index // 2, 0) for voxel in range(4))
        for index in range(len(descriptors))
    )

    audit = audit_realised_qd_footprint_separation(descriptors, footprints)

    assert audit.status == "QD_FOOTPRINT_SEPARATION_ADMITTED"
    assert audit.footprint_separation_margin >= 0.05


def test_qd_reproducibility_admits_stable_independent_public_replays() -> None:
    modes = (
        (0.10, 0.20, 0.30),
        (0.35, 0.50, 0.65),
        (0.75, 0.80, 0.15),
    )
    audit = audit_realised_qd_reproducibility(
        {
            _execution_hash(500 + index): (
                RealisedQDDescriptor(*mode),
                RealisedQDDescriptor(*mode),
            )
            for index, mode in enumerate(modes)
        }
    )

    assert audit.status == "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED"
    assert audit.repeated_manifest_group_count == 3
    assert audit.repeated_pair_count == 3
    assert audit.cell_stability_rate == 1.0
    assert audit.mean_normalized_descriptor_l2 == 0.0


def test_qd_reproducibility_rejects_cell_drift_under_public_replay() -> None:
    audit = audit_realised_qd_reproducibility(
        {
            _execution_hash(600 + index): (
                RealisedQDDescriptor(0.05, 0.05, 0.05),
                RealisedQDDescriptor(0.95, 0.95, 0.95),
            )
            for index in range(3)
        }
    )

    assert audit.status == "QD_DESCRIPTOR_REPRODUCIBILITY_NOT_ADMITTED"
    assert "QD_CELLS_NOT_REPRODUCIBLE_UNDER_PUBLIC_REPLAY" in audit.reasons
    assert "QD_DESCRIPTOR_REPLAY_VARIANCE_TOO_HIGH" in audit.reasons


def test_qd_reproducibility_rejects_history_without_repeated_candidate_manifests() -> None:
    audit = audit_realised_qd_reproducibility(
        {
            _execution_hash(700 + index): (RealisedQDDescriptor(0.2, 0.4, 0.6),)
            for index in range(12)
        }
    )

    assert audit.status == "QD_DESCRIPTOR_REPRODUCIBILITY_NOT_ADMITTED"
    assert "INSUFFICIENT_REPEATED_PUBLIC_CANDIDATE_GROUPS" in audit.reasons
    assert "INSUFFICIENT_REPEATED_OUTCOME_PAIRS" in audit.reasons


def test_qd_calibration_requires_each_public_intent_to_control_its_axis() -> None:
    labels = (
        "vertical_low",
        "vertical_low",
        "vertical_high",
        "vertical_high",
        "dispersion_low",
        "dispersion_low",
        "dispersion_high",
        "dispersion_high",
        "complementarity_low",
        "complementarity_low",
        "complementarity_high",
        "complementarity_high",
    )
    scenes = tuple("scene_a" if index % 2 == 0 else "scene_b" for index in range(len(labels)))
    descriptors = (
        RealisedQDDescriptor(0.10, 0.50, 0.50),
        RealisedQDDescriptor(0.15, 0.50, 0.50),
        RealisedQDDescriptor(0.80, 0.50, 0.50),
        RealisedQDDescriptor(0.85, 0.50, 0.50),
        RealisedQDDescriptor(0.50, 0.10, 0.50),
        RealisedQDDescriptor(0.50, 0.15, 0.50),
        RealisedQDDescriptor(0.50, 0.80, 0.50),
        RealisedQDDescriptor(0.50, 0.85, 0.50),
        RealisedQDDescriptor(0.50, 0.50, 0.10),
        RealisedQDDescriptor(0.50, 0.50, 0.15),
        RealisedQDDescriptor(0.50, 0.50, 0.80),
        RealisedQDDescriptor(0.50, 0.50, 0.85),
    )

    audit = audit_realised_qd_calibration_mode_contrasts(labels, descriptors, scenes)

    assert audit.status == "QD_CALIBRATION_MODE_CONTRAST_ADMITTED"
    assert all(gap >= 1.0 for _, gap in audit.axis_mean_cell_gaps)


def test_qd_calibration_rejects_named_intents_that_collapse_to_one_mode() -> None:
    labels = (
        "vertical_low",
        "vertical_low",
        "vertical_high",
        "vertical_high",
        "dispersion_low",
        "dispersion_low",
        "dispersion_high",
        "dispersion_high",
        "complementarity_low",
        "complementarity_low",
        "complementarity_high",
        "complementarity_high",
    )
    descriptors = tuple(RealisedQDDescriptor(0.50, 0.50, 0.50) for _ in labels)
    scenes = tuple("scene_a" if index % 2 == 0 else "scene_b" for index in range(len(labels)))

    audit = audit_realised_qd_calibration_mode_contrasts(labels, descriptors, scenes)

    assert audit.status == "QD_CALIBRATION_MODE_CONTRAST_NOT_ADMITTED"
    assert "QD_CALIBRATION_AXIS_NOT_CONTROLLABLE_VERTICAL_MOTION_RATIO" in audit.reasons
    assert "QD_CALIBRATION_AXIS_NOT_CONTROLLABLE_TEAM_SPATIAL_DISPERSION" in audit.reasons
    assert (
        "QD_CALIBRATION_AXIS_NOT_CONTROLLABLE_PUBLIC_OBSERVATION_COMPLEMENTARITY" in audit.reasons
    )


def test_qd_calibration_rejects_three_named_axes_that_are_one_effect_direction() -> None:
    labels = (
        "vertical_low",
        "vertical_low",
        "vertical_high",
        "vertical_high",
        "dispersion_low",
        "dispersion_low",
        "dispersion_high",
        "dispersion_high",
        "complementarity_low",
        "complementarity_low",
        "complementarity_high",
        "complementarity_high",
    )
    low = (RealisedQDDescriptor(0.10, 0.10, 0.10), RealisedQDDescriptor(0.15, 0.15, 0.15))
    high = (RealisedQDDescriptor(0.80, 0.80, 0.80), RealisedQDDescriptor(0.85, 0.85, 0.85))
    descriptors = (*low, *high, *low, *high, *low, *high)
    scenes = tuple("scene_a" if index % 2 == 0 else "scene_b" for index in range(len(labels)))

    audit = audit_realised_qd_calibration_mode_contrasts(labels, descriptors, scenes)

    assert audit.status == "QD_CALIBRATION_MODE_CONTRAST_NOT_ADMITTED"
    assert "QD_CALIBRATION_AXIS_NOT_SPECIFIC_VERTICAL_MOTION_RATIO" in audit.reasons
    assert "QD_CALIBRATION_CONTRASTS_COLLINEAR" in audit.reasons
    assert "QD_CALIBRATION_CONTRAST_MATRIX_RANK_DEFICIENT" in audit.reasons


def test_candidate_intent_audit_rejects_a_pool_that_only_offers_one_mode() -> None:
    pool = _candidate_pool()
    constant = tuple(
        replace(candidate, planned_descriptor=(0.25, 0.25, 0.25)) for candidate in pool
    )

    audit = audit_public_candidate_intent_richness(
        constant,
        minimum_feasible_candidates=3,
    )

    assert audit.status == "QD_CANDIDATE_INTENT_NOT_ADMITTED"
    assert "DEGENERATE_INTENT_VERTICAL_MOTION_INTENT" in audit.reasons
    assert "INSUFFICIENT_INTENT_JOINT_CELLS" in audit.reasons


def test_candidate_intent_audit_rejects_a_pool_dominated_by_one_mode() -> None:
    pool = _candidate_pool()
    patterns = (
        *((0.05, 0.05, 0.05) for _ in range(7)),
        (0.30, 0.30, 0.30),
        (0.55, 0.55, 0.55),
        (0.80, 0.80, 0.80),
        (0.30, 0.55, 0.80),
        (0.55, 0.80, 0.30),
    )
    candidates = tuple(
        replace(
            pool[index % len(pool)],
            candidate_id=f"mode{index}",
            planned_descriptor=descriptor,
        )
        for index, descriptor in enumerate(patterns)
    )

    audit = audit_public_candidate_intent_richness(candidates)

    assert audit.joint_effective_cells == 6
    assert audit.joint_shannon_effective_cells < 4.0
    assert "INTENT_JOINT_SPACE_DOMINATED_BY_TOO_FEW_MODES" in audit.reasons


def test_value_protected_diversity_requires_near_value_distinct_modes() -> None:
    low, high, third = _candidate_pool()
    value_best_only = (
        replace(
            low,
            planned_descriptor=(0.10, 0.10, 0.10),
            quality_hint=10.0,
            cost_hint=1.0,
        ),
        replace(
            high,
            planned_descriptor=(0.90, 0.90, 0.90),
            quality_hint=1.0,
            cost_hint=1.0,
        ),
        replace(
            third,
            planned_descriptor=(0.50, 0.50, 0.50),
            quality_hint=0.5,
            cost_hint=1.0,
        ),
    )

    no_opportunity = audit_value_protected_candidate_diversity(value_best_only)

    assert no_opportunity.status == "QD_VALUE_PROTECTED_DIVERSITY_NOT_ADMITTED"
    assert "QD_NO_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY" in no_opportunity.reasons

    near_value_diverse = (
        value_best_only[0],
        replace(value_best_only[1], quality_hint=9.5),
        value_best_only[2],
    )
    admitted = audit_value_protected_candidate_diversity(near_value_diverse)

    assert admitted.status == "QD_VALUE_PROTECTED_DIVERSITY_ADMITTED"
    assert admitted.value_protected_candidate_count == 2
    assert admitted.value_protected_joint_cells == 2


def test_intent_outcome_alignment_requires_realised_modes_to_follow_emitter_intent() -> None:
    intents = tuple((index / 11.0, index / 11.0, index / 11.0) for index in range(12))
    descriptors = tuple(
        RealisedQDDescriptor(
            0.1 + 0.8 * intent[0],
            0.1 + 0.8 * intent[1],
            0.1 + 0.8 * intent[2],
        )
        for intent in intents
    )

    audit = audit_intent_realised_alignment(intents, descriptors)

    assert audit.status == "QD_INTENT_OUTCOME_ALIGNMENT_ADMITTED"
    assert audit.aligned_axis_count == 3
    assert all(correlation > 0.9 for correlation in audit.axis_correlations)


def test_intent_alignment_rejects_correlation_without_out_of_sample_prediction_gain() -> None:
    intents = tuple((index / 11.0, index / 11.0, index / 11.0) for index in range(12))
    values = tuple(
        0.10 + 0.25 * index / 11.0 if index % 2 == 0 else 0.65 + 0.25 * index / 11.0
        for index in range(12)
    )
    descriptors = tuple(RealisedQDDescriptor(value, value, value) for value in values)

    audit = audit_intent_realised_alignment(intents, descriptors)

    assert audit.aligned_axis_count == 3
    assert audit.relative_prediction_mse_reduction < 0.0
    assert audit.status == "QD_INTENT_OUTCOME_ALIGNMENT_NOT_ADMITTED"
    assert "INTENT_PREDICTOR_DOES_NOT_OUTPERFORM_GLOBAL_MEAN" in audit.reasons


def test_outcome_grounded_selector_refuses_warmup_then_prefers_an_active_public_need() -> None:
    low, high, *_ = _candidate_pool()
    low = replace(low, planned_descriptor=(0.2, 0.2, 0.2), quality_hint=1.0, cost_hint=1.0)
    high = replace(high, planned_descriptor=(0.9, 0.9, 0.9), quality_hint=1.0, cost_hint=1.0)
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    _seed_outcome_archive(archive)
    selector = OutcomeGroundedQDSelector(archive, minimum_evidence=3, neighbours=1)

    with pytest.raises(ValueError, match="not qualified"):
        selector.select((low, high), public_exploration_need=_need((0.2, 0.2, 0.2)))

    selector.observe(
        low,
        RealisedQDDescriptor(0.2, 0.2, 0.2),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(1),
        execution_feasible=True,
    )
    selector.observe(
        high,
        RealisedQDDescriptor(0.9, 0.9, 0.9),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(2),
        execution_feasible=True,
    )
    selector.observe(
        high,
        RealisedQDDescriptor(0.9, 0.9, 0.9),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(3),
        execution_feasible=True,
    )

    selected, selection = selector.select(
        (low, high), public_exploration_need=_need((0.2, 0.2, 0.2))
    )

    assert selected.manifest_hash == high.manifest_hash
    assert selection.evidence_count == 3
    assert selection.selected_manifest_hash == high.manifest_hash
    assert selection.archive_entry_count == 12
    assert archive.get(archive.spec.cell((0.2, 0.2, 0.2))) is not None


def test_outcome_grounded_selector_refuses_an_underpopulated_archive() -> None:
    low, *_ = _candidate_pool()
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    selector = OutcomeGroundedQDSelector(archive, minimum_evidence=1, neighbours=1)
    selector.observe(
        low,
        RealisedQDDescriptor(0.1, 0.1, 0.1),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(10),
        execution_feasible=True,
    )

    with pytest.raises(ValueError, match="fill 6 realised archive cells"):
        selector.select((low,), public_exploration_need=_need((0.1, 0.1, 0.1)))


def test_outcome_grounded_selector_refuses_a_non_outcome_digest() -> None:
    low, *_ = _candidate_pool()
    selector = OutcomeGroundedQDSelector(
        QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC),
        minimum_evidence=1,
        neighbours=1,
    )

    with pytest.raises(ValueError, match="execution outcome hash"):
        selector.observe(
            low,
            RealisedQDDescriptor(0.1, 0.1, 0.1),
            public_quality=1.0,
            public_cost=1.0,
            execution_outcome_sha256="planned-route-hash",
            execution_feasible=True,
        )

    assert len(selector.archive) == 0
    assert selector.evidence_count == 0


def test_outcome_grounded_selector_excludes_incomplete_or_unsafe_execution() -> None:
    low, *_ = _candidate_pool()
    selector = OutcomeGroundedQDSelector(
        QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC),
        minimum_evidence=1,
        neighbours=1,
    )

    decision = selector.observe(
        low,
        RealisedQDDescriptor(0.1, 0.1, 0.1),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(11),
        execution_feasible=False,
    )

    assert decision.admitted is False
    assert decision.reason == "EXECUTION_NOT_QD_FEASIBLE"
    assert len(selector.archive) == 0
    assert selector.evidence_count == 0


def test_outcome_grounded_selector_never_trades_large_public_value_for_novelty() -> None:
    low, high, *_ = _candidate_pool()
    low = replace(low, planned_descriptor=(0.1, 0.1, 0.1), quality_hint=0.1, cost_hint=1.0)
    high = replace(high, planned_descriptor=(0.9, 0.9, 0.9), quality_hint=1.0, cost_hint=1.0)
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    _seed_outcome_archive(archive)
    selector = OutcomeGroundedQDSelector(
        archive,
        minimum_evidence=3,
        neighbours=1,
        utility_slack=0.10,
    )
    selector.observe(
        low,
        RealisedQDDescriptor(0.1, 0.1, 0.1),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(4),
        execution_feasible=True,
    )
    selector.observe(
        high,
        RealisedQDDescriptor(0.9, 0.9, 0.9),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(5),
        execution_feasible=True,
    )
    selector.observe(
        high,
        RealisedQDDescriptor(0.9, 0.9, 0.9),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(6),
        execution_feasible=True,
    )

    selected, selection = selector.select(
        (low, high),
        base_utilities={low.candidate_id: 0.1, high.candidate_id: 1.0},
        public_exploration_need=_need((0.1, 0.1, 0.1)),
    )

    assert selected.candidate_id == high.candidate_id
    assert selection.base_best_candidate_id == high.candidate_id
    assert selection.eligible_candidate_count == 1
    assert selection.diversity_changed_selection is False


def test_outcome_grounded_selector_abstains_when_outcomes_are_too_uncertain() -> None:
    low, high, *_ = _candidate_pool()
    low = replace(low, planned_descriptor=(0.1, 0.1, 0.1), quality_hint=1.0, cost_hint=1.0)
    high = replace(high, planned_descriptor=(0.9, 0.9, 0.9), quality_hint=2.0, cost_hint=1.0)
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    _seed_outcome_archive(archive)
    selector = OutcomeGroundedQDSelector(
        archive,
        minimum_evidence=3,
        neighbours=3,
        maximum_prediction_uncertainty=0.15,
    )
    selector.observe(
        low,
        RealisedQDDescriptor(0.0, 0.0, 0.0),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(7),
        execution_feasible=True,
    )
    selector.observe(
        low,
        RealisedQDDescriptor(0.7, 0.7, 0.7),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(8),
        execution_feasible=True,
    )
    selector.observe(
        low,
        RealisedQDDescriptor(1.0, 1.0, 1.0),
        public_quality=1.0,
        public_cost=1.0,
        execution_outcome_sha256=_execution_hash(9),
        execution_feasible=True,
    )

    selected, selection = selector.select(
        (low, high),
        base_utilities={low.candidate_id: 1.0, high.candidate_id: 2.0},
        public_exploration_need=_need((0.1, 0.1, 0.1)),
    )

    assert selected.candidate_id == high.candidate_id
    assert selection.qd_abstained is True
    assert selection.uncertainty_abstained_candidate_count == 1
    assert selection.diversity_changed_selection is False


def test_public_exploration_need_uses_only_public_sparse_range_beliefs() -> None:
    belief = SparseVoxelBelief("scene0", "team", 1.0)
    for key in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        belief.set_state(key, FREE)

    need = public_exploration_need_from_public_belief(
        belief,
        agent_free_voxel_keys={
            "uav0": ((0, 0, 0), (1, 0, 0)),
            "uav1": ((0, 0, 0), (0, 1, 0)),
        },
        agent_ids=("uav0", "uav1"),
        spatial_reference_m=10.0,
        height_band_m=1.0,
    )

    assert need.vertical_exploration_deficit == pytest.approx(1.0)
    assert need.spatial_dispersion_deficit > 0.7
    assert need.duplicate_observation_deficit > 0.0
    assert need.source_public_belief_sha256 == belief.content_sha256
    assert "ESDF" not in str(need.to_dict())


def test_outcome_grounded_qd_changes_only_for_a_better_current_public_need() -> None:
    first, second, *_ = _candidate_pool()
    candidates = tuple(
        replace(candidate, planned_descriptor=descriptor, quality_hint=1.0, cost_hint=1.0)
        for candidate, descriptor in zip(
            (first, second), ((0.9, 0.2, 0.2), (0.2, 0.9, 0.2)), strict=True
        )
    )
    base, alternative = sorted(candidates, key=lambda candidate: candidate.manifest_hash)
    alternative_axis = max(
        range(3),
        key=lambda axis: alternative.planned_descriptor[axis] - base.planned_descriptor[axis],
    )
    assert (
        alternative.planned_descriptor[alternative_axis] > base.planned_descriptor[alternative_axis]
    )
    need_values = tuple(1.0 if axis == alternative_axis else 0.0 for axis in range(3))
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    _seed_outcome_archive(archive)
    selector = OutcomeGroundedQDSelector(archive, minimum_evidence=2, neighbours=1)
    for index, candidate in enumerate(candidates):
        selector.observe(
            candidate,
            RealisedQDDescriptor(*candidate.planned_descriptor),
            public_quality=1.0,
            public_cost=1.0,
            execution_outcome_sha256=_execution_hash(50 + index),
            execution_feasible=True,
        )

    selected, selection = selector.select(
        candidates,
        public_exploration_need=_need(need_values),
    )

    assert selection.base_best_candidate_id == base.candidate_id
    assert selected.candidate_id == alternative.candidate_id
    assert selection.need_changed_selection is True
    assert selection.selected_need_alignment > selection.base_best_need_alignment
    assert selection.qd_abstained is False


def test_outcome_grounded_qd_abstains_when_current_public_need_is_closed() -> None:
    low, high, *_ = _candidate_pool()
    low = replace(low, planned_descriptor=(0.1, 0.1, 0.1), quality_hint=1.0, cost_hint=1.0)
    high = replace(high, planned_descriptor=(0.9, 0.9, 0.9), quality_hint=1.0, cost_hint=1.0)
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    _seed_outcome_archive(archive)
    selector = OutcomeGroundedQDSelector(archive, minimum_evidence=2, neighbours=1)
    for index, candidate in enumerate((low, high)):
        selector.observe(
            candidate,
            RealisedQDDescriptor(*candidate.planned_descriptor),
            public_quality=1.0,
            public_cost=1.0,
            execution_outcome_sha256=_execution_hash(60 + index),
            execution_feasible=True,
        )

    selected, selection = selector.select(
        (low, high), public_exploration_need=_need((0.01, 0.01, 0.01))
    )

    assert selected.candidate_id == selection.base_best_candidate_id
    assert selection.qd_abstained is True
    assert selection.qd_abstention_reason == "PUBLIC_EXPLORATION_NEED_BELOW_ACTIVE_FLOOR"
    assert selection.need_changed_selection is False


def test_planned_qd_is_explicitly_a_diagnostic_intent_archive() -> None:
    low, high, *_ = _candidate_pool()
    low = replace(low, planned_descriptor=(0.1, 0.1, 0.1), quality_hint=1.0, cost_hint=1.0)
    high = replace(high, planned_descriptor=(0.9, 0.9, 0.9), quality_hint=1.0, cost_hint=1.0)
    archive = QDArchive(HM3D_REALISED_QD_ARCHIVE_SPEC)
    selector = PlannedQDSelector(archive)
    selector.observe_intent(
        high.planned_descriptor,
        public_quality=1.0,
        public_cost=1.0,
        source_id="history0",
    )

    selected, selection = selector.select((low, high))

    assert selected.manifest_hash == low.manifest_hash
    assert selection.to_dict()["archive_semantics"] == "planned_intent_diagnostic_only"
