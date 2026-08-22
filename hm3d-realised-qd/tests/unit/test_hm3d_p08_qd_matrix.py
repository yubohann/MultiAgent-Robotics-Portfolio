from __future__ import annotations

from copy import deepcopy

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.hm3d_public_schema import public_schema_fields
from aerocity_method.evaluation.hm3d_p08_qd_matrix import (
    P08QDUnit,
    assemble_p08_qd_paired_evidence,
)
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    HM3D_REALISED_QD_SCHEMA_VERSION,
    RealisedQDDescriptor,
    audit_realised_qd_calibration_mode_contrasts,
    audit_realised_qd_reproducibility,
    audit_realised_qd_richness,
)

_DESCRIPTOR_PATTERNS = (
    (0.05, 0.20, 0.30),
    (0.25, 0.45, 0.55),
    (0.45, 0.65, 0.75),
    (0.65, 0.85, 0.15),
    (0.85, 0.15, 0.45),
    (0.15, 0.75, 0.85),
)

_RICH_DESCRIPTOR_PATTERNS = (
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


def _digest(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("runtime_record_sha256", None)
    payload["runtime_record_sha256"] = canonical_sha256(payload)
    return payload


def _train_descriptor_admission() -> dict[str, object]:
    richness = audit_realised_qd_richness(
        tuple(RealisedQDDescriptor(*descriptor) for descriptor in _RICH_DESCRIPTOR_PATTERNS * 2)
    )
    assert richness.status == "QD_DESCRIPTOR_ADMITTED"
    replay = audit_realised_qd_reproducibility(
        {
            f"{index + 31:064x}": (
                RealisedQDDescriptor(*descriptor),
                RealisedQDDescriptor(*descriptor),
            )
            for index, descriptor in enumerate(_DESCRIPTOR_PATTERNS[:3])
        }
    )
    assert replay.status == "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED"
    mode_descriptors = {
        "vertical_low": RealisedQDDescriptor(0.10, 0.50, 0.50),
        "vertical_high": RealisedQDDescriptor(0.80, 0.50, 0.50),
        "dispersion_low": RealisedQDDescriptor(0.50, 0.10, 0.50),
        "dispersion_high": RealisedQDDescriptor(0.50, 0.80, 0.50),
        "complementarity_low": RealisedQDDescriptor(0.50, 0.50, 0.10),
        "complementarity_high": RealisedQDDescriptor(0.50, 0.50, 0.80),
    }
    mode_labels = tuple(mode for mode in mode_descriptors for _ in range(2))
    mode_contrast = audit_realised_qd_calibration_mode_contrasts(
        mode_labels,
        tuple(mode_descriptors[mode] for mode in mode_labels),
        tuple("train_scene0" if index % 2 == 0 else "train_scene1" for index in range(12)),
    )
    assert mode_contrast.status == "QD_CALIBRATION_MODE_CONTRAST_ADMITTED"
    admission: dict[str, object] = {
        "status": "QD_TRAIN_DESCRIPTOR_ADMITTED",
        "descriptor_schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
        "archive_spec_sha256": HM3D_REALISED_QD_ARCHIVE_SPEC.digest,
        "outcome_count": richness.sample_count,
        "scene_ids": ["train_scene0", "train_scene1"],
        "split_manifest_sha256": "3" * 64,
        "source_runtime_record_sha256s": ["1" * 64, "2" * 64],
        "richness_audit": richness.to_dict(),
        "intent_outcome_alignment": {
            "status": "QD_INTENT_OUTCOME_ALIGNMENT_ADMITTED",
            "scene_count": 2,
            "cross_scene_relative_prediction_mse_reduction": 0.2,
        },
        "footprint_separation_audit": {"status": "QD_FOOTPRINT_SEPARATION_ADMITTED"},
        "descriptor_family_screen": {
            "status": "QD_DESCRIPTOR_FAMILY_CURRENT_ADMITTED",
            "current_family_id": HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
            "recommended_family_id": HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
            "family_rows": [
                {
                    "family_id": HM3D_CURRENT_QD_DESCRIPTOR_FAMILY_ID,
                    "admitted": True,
                    "cross_scene_support_audit": {
                        "status": "QD_DESCRIPTOR_FAMILY_CROSS_SCENE_ADMITTED",
                        "scene_count": 2,
                    },
                }
            ],
        },
        "reproducibility_audit": replay.to_dict(),
        "calibration_mode_contrast_audit": mode_contrast.to_dict(),
    }
    admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
    return admission


def _record(
    *,
    strategy: str,
    unit_index: int,
    metric: float,
    selection_changed: bool = True,
    archive_entry_count: int = 12,
    value_protected_opportunity: bool = True,
) -> dict[str, object]:
    scene_id = f"scene{unit_index % 2}"
    fleet_size = 4
    random_key = unit_index // 6
    public_context = {
        "context_id": f"context-{unit_index}",
        "episode_id": f"episode-{unit_index}",
        "decision_id": "decision0",
    }
    admissions = []
    for offset in range(2):
        descriptor = _DESCRIPTOR_PATTERNS[(2 * unit_index + offset) % len(_DESCRIPTOR_PATTERNS)]
        pattern_index = (2 * unit_index + offset) % len(_DESCRIPTOR_PATTERNS)
        admissions.append(
            {
                "feasible": True,
                "executed": True,
                "candidate_id": f"candidate-{unit_index}-{offset}",
                "execution_outcome_sha256": "f" * 64,
                "public_candidate_intent": list(descriptor),
                "descriptor": {
                    "schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
                    "vertical_motion_ratio": descriptor[0],
                    "team_spatial_dispersion": descriptor[1],
                    "public_observation_complementarity": descriptor[2],
                },
                # Identical descriptor modes retain overlapping public
                # footprints; different modes expose disjoint ones.  The P08
                # mechanism test must reject records that cannot establish
                # this semantic link from real execution artifacts.
                "public_new_free_voxel_keys": [
                    [100 * pattern_index + voxel, pattern_index, offset] for voxel in range(4)
                ],
            }
        )
    payload: dict[str, object] = {
        "schema_version": "hm3d-p07-exploration-execution-v1",
        "status": "P07_EXECUTION_SMOKE_COMPLETE",
        "synthetic": False,
        "formal_result": False,
        "p07_task_validity_closed": False,
        "selection_partition": "validation",
        "strategy": strategy,
        "scene_id": scene_id,
        "fleet_size": fleet_size,
        "random_key": random_key,
        "public_episode_id": f"episode-{unit_index}",
        "public_context_hash": canonical_sha256(public_context),
        "public_candidate_pool_hash": f"{unit_index:064x}",
        **public_schema_fields(),
        "sensor_profile_sha256": "a" * 64,
        "public_contract_sha256": "b" * 64,
        "evaluation_denominator_sha256": "c" * 64,
        "communication_contract_sha256": "d" * 64,
        "action_budget_s": 40.0,
        "candidate_limit": 8,
        "physics_dt_s": 1.0 / 120.0,
        "outcome_time_tolerance_s": 0.25,
        "selector_backbone_sha256": "e" * 64,
        "metric_report": {"explored_free_flight_volume_auc_time": metric},
        "decisions": [
            {
                "selection": {
                    "diversity_changed_selection": selection_changed,
                    "qd_abstained": not selection_changed,
                    "archive_entry_count": archive_entry_count,
                    "archive_revision": 12,
                    "public_exploration_need": {
                        "schema_version": "hm3d-public-exploration-need-v1",
                        "values": [0.9, 0.9, 0.9],
                        "strength": 0.9,
                        "active": True,
                        "minimum_active_strength": 0.15,
                        "source_public_belief_sha256": "9" * 64,
                        "source_agent_footprints_sha256": "8" * 64,
                        "source_public_outcome_count": 12,
                    },
                    "selected_need_alignment": 0.9 if selection_changed else 0.5,
                    "base_best_need_alignment": 0.5,
                    "need_changed_selection": selection_changed,
                    "minimum_need_alignment_improvement": 0.02,
                    "selected_predicted_descriptor": [0.9, 0.9, 0.9],
                    "selected_prediction_uncertainty": 0.10,
                    "realised_descriptor": {
                        "schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
                        "vertical_motion_ratio": 0.9,
                        "team_spatial_dispersion": 0.9,
                        "public_observation_complementarity": 0.9,
                    },
                    "realised_need_alignment": 0.9,
                    "need_alignment_prediction_error": 0.0,
                    "qd_abstention_reason": (
                        None
                        if selection_changed
                        else "NO_VALUE_PROTECTED_CANDIDATE_IMPROVES_CURRENT_PUBLIC_NEED"
                    ),
                }
            }
        ],
        "realised_qd": {
            "history": {"train_descriptor_admission": _train_descriptor_admission()},
            "candidate_intent_audits": [
                {
                    "minimum_feasible_candidates": 6,
                    "minimum_axis_bins": 2,
                    "minimum_joint_cells": 6,
                    "minimum_joint_shannon_effective_cells": 4.0,
                    "status": "QD_CANDIDATE_INTENT_ADMITTED",
                }
            ],
            "value_protected_candidate_diversity_audits": [
                {
                    "minimum_value_protected_candidates": 2,
                    "minimum_value_protected_joint_cells": 2,
                    "utility_slack": 0.10,
                    "status": (
                        "QD_VALUE_PROTECTED_DIVERSITY_ADMITTED"
                        if value_protected_opportunity
                        else "QD_VALUE_PROTECTED_DIVERSITY_NOT_ADMITTED"
                    ),
                }
            ],
            "admissions": admissions,
        },
    }
    return _digest(payload)


def _units(
    *,
    selection_changed: bool = True,
    realised_metric: float = 0.35,
    value_protected_opportunity: bool = True,
) -> tuple[P08QDUnit, ...]:
    return tuple(
        P08QDUnit(
            unit_id=f"unit{index}",
            no_qd=_record(strategy="no_qd", unit_index=index, metric=0.20),
            planned_qd=_record(
                strategy="planned_qd",
                unit_index=index,
                metric=0.25,
                value_protected_opportunity=value_protected_opportunity,
            ),
            realised_qd=_record(
                strategy="realised_qd",
                unit_index=index,
                metric=realised_metric,
                selection_changed=selection_changed,
                value_protected_opportunity=value_protected_opportunity,
            ),
        )
        for index in range(12)
    )


def test_p08_qd_matrix_validates_only_a_paired_effect_with_admitted_mechanism() -> None:
    evidence = assemble_p08_qd_paired_evidence(_units())

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert evidence["reasons"] == []
    mechanism = evidence["qd_mechanism_evidence"]
    assert mechanism["candidate_intent_admission"]["admitted_pool_count"] == 24
    assert mechanism["value_protected_diversity_opportunity"]["opportunity_rate"] == 1.0
    assert mechanism["paired_effect"]["qd_active_decision_rate"] == 1.0
    assert mechanism["paired_effect"]["selection_change_rate"] == 1.0


def test_p08_qd_matrix_rejects_a_selector_that_never_changes_selection() -> None:
    evidence = assemble_p08_qd_paired_evidence(_units(selection_changed=False))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert "QD_ALWAYS_ABSTAINED_FOR_UNCERTAIN_REALISATION" in evidence["reasons"]


def test_p08_qd_matrix_records_an_unfaithful_qd_prediction_without_blocking_p08() -> None:
    units = list(_units())
    for unit in units:
        selection = unit.realised_qd["decisions"][0]["selection"]
        selection["realised_descriptor"] = {
            "schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
            "vertical_motion_ratio": 0.0,
            "team_spatial_dispersion": 0.0,
            "public_observation_complementarity": 0.0,
        }
        selection["realised_need_alignment"] = 0.0
        selection["need_alignment_prediction_error"] = 0.9
        _digest(unit.realised_qd)

    evidence = assemble_p08_qd_paired_evidence(tuple(units))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert evidence["reasons"] == []


def test_p08_qd_matrix_reports_when_qd_had_no_near_value_diverse_option() -> None:
    evidence = assemble_p08_qd_paired_evidence(_units(value_protected_opportunity=False))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert "QD_VALUE_PROTECTED_DIVERSITY_OPPORTUNITY_INSUFFICIENT" in evidence["reasons"]


def test_p08_qd_matrix_rejects_non_positive_realised_qd_effect() -> None:
    evidence = assemble_p08_qd_paired_evidence(_units(realised_metric=0.20))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert "REALISED_QD_HAS_NO_SIGNIFICANT_PAIRED_ADVANTAGE" in evidence["reasons"]


def test_p08_qd_matrix_rejects_a_statistically_positive_but_trivial_qd_effect() -> None:
    evidence = assemble_p08_qd_paired_evidence(_units(realised_metric=0.251))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert "REALISED_QD_HAS_NO_SIGNIFICANT_PAIRED_ADVANTAGE" in evidence["reasons"]


def test_p08_qd_matrix_requires_a_train_only_descriptor_admission() -> None:
    units = list(_units())
    for unit in units:
        admission = unit.realised_qd["realised_qd"]["history"]["train_descriptor_admission"]
        admission["richness_audit"] = {"status": "QD_DESCRIPTOR_NOT_ADMITTED"}
        admission.pop("train_descriptor_admission_sha256")
        admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
        _digest(unit.realised_qd)

    with pytest.raises(ValueError, match="richness evidence"):
        assemble_p08_qd_paired_evidence(tuple(units))


def test_p08_qd_matrix_keeps_train_replay_stability_as_a_diagnostic() -> None:
    units = list(_units())
    for unit in units:
        admission = unit.realised_qd["realised_qd"]["history"]["train_descriptor_admission"]
        replay = admission["reproducibility_audit"]
        replay["cell_stability_rate"] = 0.5
        admission.pop("train_descriptor_admission_sha256")
        admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
        _digest(unit.realised_qd)

    evidence = assemble_p08_qd_paired_evidence(tuple(units))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"


def test_p08_qd_matrix_keeps_train_mode_calibration_as_a_diagnostic() -> None:
    units = list(_units())
    for unit in units:
        admission = unit.realised_qd["realised_qd"]["history"]["train_descriptor_admission"]
        admission["calibration_mode_contrast_audit"] = {
            "status": "QD_CALIBRATION_MODE_CONTRAST_NOT_ADMITTED"
        }
        admission.pop("train_descriptor_admission_sha256")
        admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
        _digest(unit.realised_qd)

    evidence = assemble_p08_qd_paired_evidence(tuple(units))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"


def test_p08_qd_matrix_allows_a_diagnostic_calibration_to_vary() -> None:
    units = list(_units())
    for unit in units:
        admission = unit.realised_qd["realised_qd"]["history"]["train_descriptor_admission"]
        contrast = admission["calibration_mode_contrast_audit"]
        contrast["minimum_target_alignment"] = 0.10
        admission.pop("train_descriptor_admission_sha256")
        admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
        _digest(unit.realised_qd)

    evidence = assemble_p08_qd_paired_evidence(tuple(units))

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"


def test_p08_qd_matrix_rejects_a_train_admission_without_frozen_split_provenance() -> None:
    units = list(_units())
    for unit in units:
        admission = unit.realised_qd["realised_qd"]["history"]["train_descriptor_admission"]
        admission.pop("split_manifest_sha256")
        admission.pop("train_descriptor_admission_sha256")
        admission["train_descriptor_admission_sha256"] = canonical_sha256(admission)
        _digest(unit.realised_qd)

    with pytest.raises(ValueError, match="split manifest hash"):
        assemble_p08_qd_paired_evidence(tuple(units))


def test_p08_qd_matrix_rejects_active_qd_without_a_populated_outcome_archive() -> None:
    units = tuple(
        P08QDUnit(
            unit_id=f"unit{index}",
            no_qd=_record(strategy="no_qd", unit_index=index, metric=0.20),
            planned_qd=_record(strategy="planned_qd", unit_index=index, metric=0.25),
            realised_qd=_record(
                strategy="realised_qd",
                unit_index=index,
                metric=0.35,
                archive_entry_count=5,
            ),
        )
        for index in range(12)
    )

    evidence = assemble_p08_qd_paired_evidence(units)

    assert evidence["status"] == "P08_QD_PILOT_COMPLETE"
    assert "QD_ACTIVE_WITHOUT_A_RICH_OUTCOME_GROUNDED_ARCHIVE" in evidence["reasons"]


def test_p08_qd_matrix_rejects_pairing_or_backbone_drift() -> None:
    no_qd = _record(strategy="no_qd", unit_index=0, metric=0.20)
    planned_qd = _record(strategy="planned_qd", unit_index=0, metric=0.25)
    realised_qd = deepcopy(_record(strategy="realised_qd", unit_index=0, metric=0.35))
    realised_qd["selector_backbone_sha256"] = "f" * 64
    _digest(realised_qd)

    with pytest.raises(ValueError, match="candidate-value backbone"):
        P08QDUnit("unit0", no_qd, planned_qd, realised_qd)

    realised_qd = deepcopy(_record(strategy="realised_qd", unit_index=0, metric=0.35))
    realised_qd["public_candidate_pool_hash"] = "f" * 64
    _digest(realised_qd)
    with pytest.raises(ValueError, match="pair drift"):
        P08QDUnit("unit0", no_qd, planned_qd, realised_qd)
