from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.evaluation.hm3d_exploration_contract import (
    load_exploration_observation_contract,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    evaluation_denominator_sha256,
)
from aerocity_method.evaluation.hm3d_preflight import (
    FORMAL_MATRIX_METHODS,
    MECHANISM_VARIANTS,
    METHOD_CORE,
    PHASE_SPECS,
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    TASK_VALIDITY_METHODS,
    HM3DFormalPreflightEvidence,
    audit_hm3d_formal_preflight,
    audit_preflight_contract,
    load_preflight_protocol,
)
from aerocity_method.runtime.hm3d_realised_qd import (
    HM3D_REALISED_QD_ARCHIVE_SPEC,
    HM3D_REALISED_QD_SCHEMA_VERSION,
    RealisedQDDescriptor,
    audit_realised_qd_richness,
)
from aerocity_method.runtime.sensors import FORMAL_H15_SENSOR_PILOT_MODES, SensorProfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _admitted_richness_audit() -> dict[str, object]:
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
    audit = audit_realised_qd_richness(
        tuple(RealisedQDDescriptor(*values) for values in patterns * 2)
    )
    assert audit.status == "QD_DESCRIPTOR_ADMITTED"
    return audit.to_dict()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return _sha(path)


def _profile(mode: str) -> SensorProfile:
    if mode == "physics_only":
        return SensorProfile("physics", mode, 0.0, (), ())
    if mode == "sparse_range_3d":
        return SensorProfile(
            "range",
            mode,
            10.0,
            ("transit", "observe", "dwell", "map_update"),
            ("range_points", "source_observation_id"),
            range_enabled=True,
        )
    raise ValueError(f"unsupported formal H15 mode: {mode}")


def _sensor_row(fleet_size: int, mode: str) -> dict[str, Any]:
    profile = _profile(mode)
    observations = [0] * fleet_size if mode == "physics_only" else [10] * fleet_size
    return {
        "comparison_id": "h15-real-run",
        "scene_id": "scene-train",
        "episode_id": "episode-h15",
        "fleet_size": fleet_size,
        "profile": profile.to_dict(),
        "physics_dt_s": 0.01,
        "planned_episodes": 1,
        "executed_episodes": 1,
        "failed_episodes": 0,
        "physics_real_time_factor": 1.0,
        "environment_steps_per_s": 100.0,
        "sensor_frames_per_s": 0.0 if mode == "physics_only" else 20.0,
        "render_time_s": 0.0 if mode == "physics_only" else 0.2,
        "transfer_time_s": 0.0 if mode == "physics_only" else 0.1,
        "gpu_memory_mb": 100.0,
        "cpu_memory_mb": 200.0,
        "dropped_frames": 0,
        "observations_per_agent": observations,
        "measurement_scope": "throughput_only",
        "wall_clock_s": 2.0,
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": "hm3d-run-20260731-001",
        "runtime_command_sha256": SHA_A,
    }


def _denominator() -> dict[str, int]:
    return {
        "planned": 1,
        "executed": 1,
        "failed": 0,
        "timeout": 0,
        "oom": 0,
        "other_failed": 0,
    }


def _artifact(
    root: Path,
    phase_id: str,
    kind: str,
    origin: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    path = root / "artifacts" / f"{phase_id}.json"
    envelope = {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": phase_id,
        "kind": kind,
        "origin": origin,
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return {
        "phase_id": phase_id,
        "kind": kind,
        "origin": origin,
        "path": str(path.relative_to(root)),
        "sha256": _sha(path),
    }


def _build_evidence(tmp_path: Path, *, include_results: bool = False) -> dict[str, Any]:
    protocol = load_preflight_protocol(PROTOCOL_PATH)
    license_path = tmp_path / "private_assets" / "HM3D_LICENSE.txt"
    license_hash = _write(license_path, "HM3D access and license audit record")
    scene_files: dict[str, Path] = {}
    scene_splits = {
        "scene-train": "train",
        "scene-validation": "validation",
        "scene-test": "test",
    }
    for scene_id in scene_splits:
        path = tmp_path / "private_assets" / f"{scene_id}.glb"
        _write(path, f"official HM3D bytes for {scene_id}")
        scene_files[scene_id] = path
    scene_rows = [
        {
            "scene_id": scene_id,
            "split": split,
            "asset_origin": "official_hm3d",
            "path": str(scene_files[scene_id].relative_to(tmp_path)),
            "sha256": _sha(scene_files[scene_id]),
        }
        for scene_id, split in scene_splits.items()
    ]
    assignments = [
        {
            "scene_id": row["scene_id"],
            "split": row["split"],
            "asset_sha256": row["sha256"],
        }
        for row in scene_rows
    ]
    assignments.sort(key=lambda row: row["scene_id"])
    split_hash = canonical_sha256(assignments)
    flight_hashes = {
        scene_id: hashlib.sha256(f"flight-{scene_id}".encode()).hexdigest()
        for scene_id in scene_splits
    }
    public_contract_hash = load_exploration_observation_contract().digest
    metric_registry_hash = SHA_A

    p01 = {
        "evidence_class": "source_license_audit",
        "dataset_version": "hm3d-v0.2",
        "license_id": "HM3D-DATA-TERMS",
        "license_record_path": str(license_path.relative_to(tmp_path)),
        "license_sha256": license_hash,
        "source_url": "https://aihabitat.org/datasets/hm3d/",
        "raw_assets_redistributed": False,
        "repository_included": False,
        "conversion_tool": {"tool_id": "habitat-sim", "version": "0.3.3", "sha256": SHA_C},
        "scenes": scene_rows,
    }
    p02 = {
        **_runtime_identity(),
        "length_unit_m": 1.0,
        "source_up_axis": "Y",
        "runtime_up_axis": "Z",
        "coordinate_transform_sha256": SHA_A,
        "gravity_m_s2": [0.0, 0.0, -9.81],
        "vehicle_envelope_m": [0.11, 0.11, 0.04],
        "simulator_id": "habitat-sim-isaac-bridge",
        "simulator_version": "runtime-v1",
        "controller_sha256": SHA_B,
        "dynamics_sha256": SHA_C,
        "vehicle_collider_sha256": SHA_A,
        "aba_reset": {
            "scene_a_id": "scene-train",
            "scene_b_id": "scene-validation",
            "a1_fingerprint": SHA_A,
            "b_fingerprint": SHA_B,
            "a2_fingerprint": SHA_A,
            "components": [
                "scene",
                "collider",
                "contact",
                "sensor",
                "rng",
                "controller",
                "reset_state",
            ],
            "passed": True,
        },
    }
    p03_scenes = []
    for row in scene_rows:
        if row["split"] == "test":
            continue
        p03_scenes.append(
            {
                "scene_id": row["scene_id"],
                "source_geometry_sha256": row["sha256"],
                "flight_space_manifest_hash": flight_hashes[row["scene_id"]],
                "representation": "voxel_esdf_3d",
                "dimension": 3,
                "resolution_m": 0.05,
                "collision_geometry_sha256": SHA_A,
                "free_flight_validated": True,
                "generator_version": "flight-space-v1",
                "vehicle_clearance_m": 0.1,
                "vertical_span_m": 3.0,
                "free_flight_volume_m3": 100.0,
                "connected_height_band_count": 3,
                "vertical_opportunity_fraction": 0.2,
                "fixed_altitude_control_run": True,
                "fixed_altitude_control_delta": 0.1,
                "fixed_altitude_control_relative_gain": 0.2,
                "vertical_counterfactual_sha256": SHA_B,
                "collision_replay_passed": True,
                "flight_space_evidence_sha256": SHA_A,
                "collision_replay_evidence_sha256": SHA_B,
                "collision_derivative_sha256": SHA_C,
            }
        )
    p03 = {
        **_runtime_identity(),
        "navmesh_authorizes_flight": False,
        "admission_scope": "stratified_development_cohort",
        "scenes": p03_scenes,
    }
    denominator_hash = evaluation_denominator_sha256(tuple(p03_scenes))
    p04_episodes = []
    for scene_id in ("scene-train", "scene-validation"):
        p04_episodes.append(
            {
                "episode_id": f"opportunity-{scene_id}",
                "scene_id": scene_id,
                "source_geometry_sha256": _sha(scene_files[scene_id]),
                "flight_space_manifest_hash": flight_hashes[scene_id],
                "source_observation_ids_total": 32,
                "observed_free_voxels_total": 128,
                "observation_voxel_resolution_m": 0.25,
                "source_observation_binding": True,
                "method_private_truth_fields": [],
            }
        )
    p04 = {
        **_runtime_identity(),
        "public_contract_sha256": public_contract_hash,
        "evaluation_denominator_sha256": denominator_hash,
        "split_manifest_sha256": split_hash,
        "episodes": p04_episodes,
    }
    p05 = {
        "evidence_class": "source_license_audit",
        "official_split_provenance": "official-hm3d-scene-split-v0.2",
        "dataset_version": "hm3d-v0.2",
        "scene_assignments": assignments,
        "split_manifest_sha256": split_hash,
        "public_contract_sha256": public_contract_hash,
        "evaluation_denominator_sha256": denominator_hash,
        "episode_seed_manifest_sha256": SHA_A,
        "difficulty_distribution_sha256": SHA_B,
        "run_partition": "development",
        "test_used_for_development": False,
        "test_access_count_before_freeze": 0,
    }
    selected = _profile("sparse_range_3d")
    comparison_methods = set(FORMAL_MATRIX_METHODS) | set(MECHANISM_VARIANTS)
    p06 = {
        **_runtime_identity(),
        "source_observation_binding": True,
        "selection_partition": "validation",
        "records": [
            _sensor_row(fleet, mode)
            for fleet in (4,)
            for mode in FORMAL_H15_SENSOR_PILOT_MODES
        ],
        "selected_profile": selected.to_dict(),
        "entitlements": [
            {"method_id": method, "profile_hash": selected.entitlement_hash}
            for method in sorted(comparison_methods)
        ],
    }
    p07_rows = []
    for index, method in enumerate(TASK_VALIDITY_METHODS):
        p07_rows.append(
            {
                "method_id": method,
                "deployed": True,
                "reads_private_truth": False,
                "oracle_only": False,
                "ranked": True,
                "budget_sha256": SHA_A,
                "sensor_profile_sha256": selected.entitlement_hash,
                "public_contract_sha256": public_contract_hash,
                "evaluation_denominator_sha256": denominator_hash,
                "evaluation_geometry_denominator_sha256": denominator_hash,
                **_denominator(),
                "explored_free_flight_volume_auc_time": 0.1 + 0.05 * index,
                "final_coverage_at_budget": 0.1 + 0.1 * index,
                "collision_count": 0,
                "communication_failure_count": 0,
                "energy_used_j": 10.0,
            }
        )
    p07 = {
        **_runtime_identity(),
        "partition": "validation",
        "budget_sha256": SHA_A,
        "sensor_profile_sha256": selected.entitlement_hash,
        "public_contract_sha256": public_contract_hash,
        "evaluation_denominator_sha256": denominator_hash,
        "evaluation_geometry_denominator_sha256": denominator_hash,
        "primary_metric": "Explored-Free-Flight-Volume-AUC_time",
        "rows": p07_rows,
        "task_validity_passed": True,
    }
    variant_auc = {
        "no_qd": 0.30,
        "planned_qd": 0.36,
        "realised_qd": 0.42,
        "no_ogfr": 0.34,
        "ogfr": 0.46,
        "rb_sf_sac_reference": 0.39,
        "rb_sf_sac_selected": 0.50,
    }
    method_core_hash = SHA_C
    p08_rows = [
        {
            "variant_id": variant,
            "configuration_sha256": hashlib.sha256(variant.encode()).hexdigest(),
            "selector_backbone_sha256": method_core_hash,
            "budget_sha256": SHA_A,
            "sensor_profile_sha256": selected.entitlement_hash,
            **_denominator(),
            "public_contract_sha256": public_contract_hash,
            "evaluation_denominator_sha256": denominator_hash,
            "explored_free_flight_volume_auc_time": variant_auc[variant],
            "final_coverage_at_budget": variant_auc[variant] + 0.03,
            "outcome_only_supervision": variant in {"ogfr", "rb_sf_sac_selected"},
            "fragment_outcome_count": 2 if variant in {"ogfr", "rb_sf_sac_selected"} else 0,
            "accepted_fragment_outcome_count": 2 if variant in {"ogfr", "rb_sf_sac_selected"} else 0,
            "outcome_gated_fragment_credit_count": 2
            if variant in {"ogfr", "rb_sf_sac_selected"}
            else 0,
            "archive_effective_cells": (
                12 if variant in {"realised_qd", "rb_sf_sac_selected"} else 1
            ),
            "archive_coverage": 0.25 if variant in {"realised_qd", "rb_sf_sac_selected"} else 0.05,
            "selector_history_mode": "recurrent_public_outcomes"
            if variant == "rb_sf_sac_selected"
            else "feedforward_public_outcomes",
        }
        for variant in MECHANISM_VARIANTS
    ]
    train_descriptor_admission = {
        "status": "QD_TRAIN_DESCRIPTOR_ADMITTED",
        "descriptor_schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
        "archive_spec_sha256": HM3D_REALISED_QD_ARCHIVE_SPEC.digest,
        "outcome_count": 24,
        "scene_ids": ["scene-train-a", "scene-train-b"],
        "split_manifest_sha256": split_hash,
        "source_runtime_record_sha256s": [SHA_A, SHA_B],
        "richness_audit": _admitted_richness_audit(),
        "intent_outcome_alignment": {
            "status": "QD_INTENT_OUTCOME_ALIGNMENT_ADMITTED",
            "scene_count": 2,
            "cross_scene_relative_prediction_mse_reduction": 0.20,
        },
        "footprint_separation_audit": {"status": "QD_FOOTPRINT_SEPARATION_ADMITTED"},
        "reproducibility_audit": {
            "schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
            "repeated_manifest_group_count": 3,
            "repeated_pair_count": 3,
            "stable_cell_pair_count": 3,
            "cell_stability_rate": 1.0,
            "mean_normalized_descriptor_l2": 0.0,
            "minimum_repeated_manifest_groups": 3,
            "minimum_repeated_pairs": 3,
            "minimum_cell_stability_rate": 0.70,
            "maximum_mean_normalized_descriptor_l2": 0.25,
            "status": "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED",
            "reasons": [],
        },
        "calibration_mode_contrast_audit": {
            "schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
            "sample_count": 12,
            "mode_sample_counts": [
                ["vertical_low", 2],
                ["vertical_high", 2],
                ["dispersion_low", 2],
                ["dispersion_high", 2],
                ["complementarity_low", 2],
                ["complementarity_high", 2],
            ],
            "mode_scene_counts": [
                ["vertical_low", 2],
                ["vertical_high", 2],
                ["dispersion_low", 2],
                ["dispersion_high", 2],
                ["complementarity_low", 2],
                ["complementarity_high", 2],
            ],
            "axis_mean_realised_values": [
                ["vertical_motion_ratio", 0.10, 0.80],
                ["team_spatial_dispersion", 0.10, 0.80],
                ["public_observation_complementarity", 0.10, 0.80],
            ],
            "axis_mean_cell_gaps": [
                ["vertical_motion_ratio", 2.0],
                ["team_spatial_dispersion", 2.0],
                ["public_observation_complementarity", 2.0],
            ],
            "contrast_effect_vectors": [
                ["vertical_motion_ratio", 0.70, 0.0, 0.0],
                ["team_spatial_dispersion", 0.0, 0.70, 0.0],
                ["public_observation_complementarity", 0.0, 0.0, 0.70],
            ],
            "contrast_target_alignment": [
                ["vertical_motion_ratio", 1.0],
                ["team_spatial_dispersion", 1.0],
                ["public_observation_complementarity", 1.0],
            ],
            "maximum_pairwise_contrast_cosine": 0.0,
            "contrast_matrix_absolute_determinant": 1.0,
            "minimum_samples_per_mode": 2,
            "minimum_scenes_per_mode": 2,
            "minimum_mean_cell_gap": 1.0,
            "minimum_target_alignment": 0.60,
            "maximum_pairwise_contrast_cosine_allowed": 0.90,
            "minimum_contrast_matrix_absolute_determinant": 0.20,
            "status": "QD_CALIBRATION_MODE_CONTRAST_ADMITTED",
            "reasons": [],
        },
    }
    train_descriptor_admission["train_descriptor_admission_sha256"] = canonical_sha256(
        train_descriptor_admission
    )
    p08 = {
        **_runtime_identity(),
        "partition": "validation",
        "budget_sha256": SHA_A,
        "sensor_profile_sha256": selected.entitlement_hash,
        "public_contract_sha256": public_contract_hash,
        "evaluation_denominator_sha256": denominator_hash,
        "primary_metric": "Explored-Free-Flight-Volume-AUC_time",
        "paired_independent_units": 12,
        "rows": p08_rows,
        "selection_gain": 0.12,
        "selection_regret": 0.03,
        "qd_mechanism_evidence": {
            "descriptor_schema_version": HM3D_REALISED_QD_SCHEMA_VERSION,
            "selector_backbone_sha256": method_core_hash,
            "candidate_intent_admission": {
                "status": "QD_CANDIDATE_INTENT_ADMITTED",
                "assessed_pool_count": 12,
                "admitted_pool_count": 12,
                "minimum_feasible_candidates": 6,
                "minimum_axis_bins": 2,
                "minimum_joint_cells": 6,
                "minimum_joint_shannon_effective_cells": 4.0,
            },
            "train_descriptor_admission": train_descriptor_admission,
            "validation_outcome_schema": {
                "status": "QD_VALIDATION_OUTCOMES_SCHEMA_VALID",
                "policy": "validation_outcomes_do_not_tune_or_admit_descriptor_axes",
            },
            "paired_effect": {
                "unit_rows": [
                    {
                        "unit_id": f"validation-unit-{index:02d}",
                        "scene_id": f"scene-validation-{index // 6}",
                        "fleet_size": 4,
                        "seed": 11 + index // 6,
                        "initial_public_candidate_pool_sha256": SHA_A,
                        "no_qd_auc": 0.30 + 0.001 * (index % 3),
                        "planned_qd_auc": 0.36 + 0.001 * (index % 3),
                        "realised_qd_auc": 0.42 + 0.001 * (index % 3),
                    }
                    for index in range(12)
                ],
                "qd_active_decision_rate": 0.90,
                "selection_change_rate": 0.25,
                "minimum_selection_change_rate": 0.20,
                "minimum_archive_entries_for_active_selection": 12,
                "minimum_practical_relative_auc_gain": 0.05,
                "minimum_effect_denominator_auc": 0.01,
                "underpopulated_active_selection_count": 0,
            },
        },
        "archive_build_time_s": 10.0,
        "archive_bytes": 1024,
        "outcome_utilization": 0.7,
        "negative_transfer_rate": 0.05,
        "negative_transfer_limit": 0.1,
        "calibration_error": 0.08,
        "tuning_iterations": 1,
        "adjustment_log": [
            {
                "iteration": 1,
                "change": "validation-only descriptor and fragment audit",
                "partition": "validation",
                "configuration_sha256": method_core_hash,
            }
        ],
        "selected_method_core_sha256": method_core_hash,
        "mainline_components_removed": [],
        "recurrent_history_selector_exercised": True,
        "fragment_outcome_schema_sha256": SHA_B,
        "mechanism_pilot_passed": True,
    }
    p09 = {
        "evidence_class": "frozen_protocol",
        "frozen": True,
        "freeze_timestamp": "2026-07-31T23:59:00+08:00",
        "protocol_hash": protocol.protocol_hash,
        "code_snapshot_sha256": SHA_A,
        "scene_manifest_sha256": split_hash,
        "public_contract_sha256": public_contract_hash,
        "evaluation_denominator_sha256": denominator_hash,
        "metric_registry_sha256": metric_registry_hash,
        "sensor_profile_sha256": selected.entitlement_hash,
        "dynamics_sha256": SHA_C,
        "controller_sha256": SHA_B,
        "method_core_sha256": method_core_hash,
        "budget_sha256": SHA_A,
        "physical_time_s": 120.0,
        "planner_calls": 120,
        "candidate_count": 16,
        "compute_cap_s": 1.0,
        "memory_cap_mb": 12000.0,
        "seeds": [11, 22, 33],
        "primary_metric": "Explored-Free-Flight-Volume-AUC_time",
        "coverage_role": "primary_task_quality",
        "statistical_test": "paired-bootstrap-with-holm-correction",
        "alpha": 0.05,
        "target_power": 0.8,
        "test_access_count_before_freeze": 0,
        "aerocity_bench_accesses": [],
    }
    p09["freeze_sha256"] = canonical_sha256(p09)
    payloads = [p01, p02, p03, p04, p05, p06, p07, p08, p09]
    artifacts = [
        _artifact(tmp_path, phase, kind, origin, payload)
        for (phase, kind, origin), payload in zip(PHASE_SPECS[:9], payloads, strict=True)
    ]
    if include_results:
        result_rows = []
        for method in FORMAL_MATRIX_METHODS:
            for fleet in (4,):
                for seed in p09["seeds"]:
                    raw_result = (
                        tmp_path / "raw_results" / f"{method}-{fleet}-scene-test-{seed}.json"
                    )
                    raw_payload = {
                        "method_id": method,
                        "fleet_size": fleet,
                        "scene_id": "scene-test",
                        "seed": seed,
                        "explored_free_flight_volume_auc_time": 0.5,
                        "final_coverage_at_budget": 0.55,
                        "status": "SUCCESS",
                    }
                    raw_hash = _write(raw_result, json.dumps(raw_payload))
                    result_rows.append(
                        {
                            "method_id": method,
                            "fleet_size": fleet,
                            "scene_id": "scene-test",
                            "seed": seed,
                            "budget_sha256": SHA_A,
                            "sensor_profile_sha256": selected.entitlement_hash,
                            "public_contract_sha256": public_contract_hash,
                            "evaluation_denominator_sha256": denominator_hash,
                            "reads_private_truth": False,
                            "ranked": True,
                            **_denominator(),
                            "explored_free_flight_volume_auc_time": 0.5,
                            "final_coverage_at_budget": 0.55,
                            "collision_count": 0,
                            "communication_failure_count": 0,
                            "energy_used_j": 100.0,
                            "raw_result_path": str(raw_result.relative_to(tmp_path)),
                            "raw_result_sha256": raw_hash,
                        }
                    )
        p10 = {
            **_runtime_identity(),
            "freeze_sha256": p09["freeze_sha256"],
            "run_partition": "test",
            "scene_ids": ["scene-test"],
            "seeds": p09["seeds"],
            "fleet_size": 4,
            "method_ids": list(FORMAL_MATRIX_METHODS),
            "test_scene_admissions": [
                {
                    "scene_id": "scene-test",
                    "source_geometry_sha256": _sha(scene_files["scene-test"]),
                    "representation": "voxel_esdf_3d",
                    "dimension": 3,
                    "free_flight_validated": True,
                    "vertical_span_m": 3.0,
                    "free_flight_volume_m3": 100.0,
                    "connected_height_band_count": 3,
                    "flight_space_manifest_hash": SHA_A,
                    "collision_geometry_sha256": SHA_B,
                    "collision_derivative_sha256": SHA_C,
                    "collision_replay_passed": True,
                    "collision_replay_evidence_sha256": SHA_A,
                }
            ],
            "rows": result_rows,
        }
        artifacts.append(_artifact(tmp_path, *PHASE_SPECS[9], p10))
    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "protocol_hash": protocol.protocol_hash,
        "requested_gate": "formal_results" if include_results else "formal_experiment_start",
        "method_core": METHOD_CORE,
        "aerocity_bench_accesses": [],
        "artifacts": artifacts,
    }


def _audit(tmp_path: Path, evidence_payload: dict[str, Any]) -> dict[str, Any]:
    protocol = load_preflight_protocol(PROTOCOL_PATH)
    evidence = HM3DFormalPreflightEvidence.from_dict(evidence_payload)
    return audit_hm3d_formal_preflight(protocol, evidence, evidence_root=tmp_path)


def _mutate_artifact(
    tmp_path: Path,
    evidence: dict[str, Any],
    phase_id: str,
    mutate: Any,
) -> None:
    reference = next(row for row in evidence["artifacts"] if row["phase_id"] == phase_id)
    path = tmp_path / reference["path"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    mutate(envelope)
    path.write_text(json.dumps(envelope), encoding="utf-8")
    reference["sha256"] = _sha(path)


def test_contract_is_complete_without_claiming_runtime():
    protocol = load_preflight_protocol(PROTOCOL_PATH)
    report = audit_preflight_contract(protocol)
    assert report["status"] == "CONTRACT_PASS"
    assert report["formal_experiment_start_requires"] == [row[0] for row in PHASE_SPECS[:9]]
    assert report["formal_results_require"] == [row[0] for row in PHASE_SPECS]


def test_missing_runtime_evidence_is_not_ready(tmp_path):
    protocol = load_preflight_protocol(PROTOCOL_PATH)
    report = audit_hm3d_formal_preflight(protocol, None, evidence_root=tmp_path)
    assert report["status"] == "RUNTIME_NOT_READY"
    assert report["formal_experiment_start_authorized"] is False
    assert report["formal_results_authorized"] is False
    assert report["failure_attribution_status"] == "ATTRIBUTION_UNRESOLVED"


def test_p01_through_p09_authorize_start_but_not_results(tmp_path):
    report = _audit(tmp_path, _build_evidence(tmp_path))
    assert report["status"] == "FORMAL_EXPERIMENT_READY"
    assert report["formal_experiment_start_authorized"] is True
    assert report["formal_results_authorized"] is False
    assert report["phases"][-1]["reasons"] == ["EVIDENCE_MISSING"]


def test_complete_holdout_matrix_authorizes_results(tmp_path):
    report = _audit(tmp_path, _build_evidence(tmp_path, include_results=True))
    assert report["status"] == "FORMAL_RESULTS_READY"
    assert report["formal_results_authorized"] is True


@pytest.mark.parametrize(
    ("phase_id", "mutate", "message"),
    [
        ("P01", lambda row: row["payload"].update(raw_assets_redistributed=True), "redistributed"),
        ("P02", lambda row: row["payload"].update(length_unit_m=0.01), "meters"),
        (
            "P02",
            lambda row: row["payload"]["aba_reset"].update(a2_fingerprint=SHA_C),
            "fingerprints differ",
        ),
        ("P03", lambda row: row["payload"].update(navmesh_authorizes_flight=True), "navmesh"),
        (
            "P04",
            lambda row: row["payload"]["episodes"][0].update(zero_opportunity_targets=1),
            "fields mismatch",
        ),
        (
            "P04",
            lambda row: row["payload"]["episodes"][0].update(source_observation_binding=False),
            "source_observation_id",
        ),
        ("P05", lambda row: row["payload"].update(test_used_for_development=True), "test scenes"),
        ("P06", lambda row: row["payload"]["records"].pop(), "H15 pilot"),
        ("P07", lambda row: row["payload"]["rows"][0].update(deployed=False), "undeployed"),
        (
            "P08",
            lambda row: row["payload"].update(mainline_components_removed=["ogfr"]),
            "cannot delete",
        ),
        (
            "P09",
            lambda row: row["payload"].update(test_access_count_before_freeze=1),
            "test scenes",
        ),
    ],
)
def test_preflight_fails_closed_for_invalid_phase_evidence(tmp_path, phase_id, mutate, message):
    evidence = _build_evidence(tmp_path)
    _mutate_artifact(tmp_path, evidence, phase_id, mutate)
    report = _audit(tmp_path, evidence)
    phase = next(row for row in report["phases"] if row["phase_id"] == phase_id)
    assert phase["status"] == "RUNTIME_NOT_READY"
    assert message.casefold() in " ".join(phase["reasons"]).casefold()
    assert report["formal_experiment_start_authorized"] is False


def test_asset_hash_is_recomputed_from_the_real_file(tmp_path):
    evidence = _build_evidence(tmp_path)
    (tmp_path / "private_assets" / "scene-train.glb").write_text("changed", encoding="utf-8")
    report = _audit(tmp_path, evidence)
    p01 = report["phases"][0]
    assert p01["status"] == "RUNTIME_NOT_READY"
    assert "hash mismatch" in " ".join(p01["reasons"])


def test_synthetic_envelope_and_fixture_runtime_identity_are_rejected(tmp_path):
    evidence = _build_evidence(tmp_path)
    _mutate_artifact(tmp_path, evidence, "P03", lambda row: row.update(synthetic=True))
    report = _audit(tmp_path, evidence)
    assert "synthetic" in " ".join(report["phases"][2]["reasons"])

    evidence = _build_evidence(tmp_path / "second")
    _mutate_artifact(
        tmp_path / "second",
        evidence,
        "P03",
        lambda row: row["payload"].update(runtime_run_id="fixture-run"),
    )
    report = _audit(tmp_path / "second", evidence)
    assert "fixture" in " ".join(report["phases"][2]["reasons"])


def test_sensor_entitlement_mismatch_is_rejected(tmp_path):
    evidence = _build_evidence(tmp_path)
    _mutate_artifact(
        tmp_path,
        evidence,
        "P06",
        lambda row: row["payload"]["entitlements"][0].update(profile_hash=SHA_A),
    )
    report = _audit(tmp_path, evidence)
    assert "unequal sensor" in " ".join(report["phases"][5]["reasons"])
    assert report["failure_attribution_status"] == "TASK_INVALID_OR_UNCALIBRATED"


def test_tampered_freeze_hash_and_incomplete_holdout_denominator_are_rejected(tmp_path):
    evidence = _build_evidence(tmp_path, include_results=True)
    _mutate_artifact(
        tmp_path,
        evidence,
        "P09",
        lambda row: row["payload"].update(physical_time_s=121.0),
    )
    report = _audit(tmp_path, evidence)
    assert "tampered" in " ".join(report["phases"][8]["reasons"])
    assert report["formal_results_authorized"] is False

    evidence = _build_evidence(tmp_path / "denominator", include_results=True)
    _mutate_artifact(
        tmp_path / "denominator",
        evidence,
        "P10",
        lambda row: row["payload"]["rows"][0].update(planned=2),
    )
    report = _audit(tmp_path / "denominator", evidence)
    assert "denominator" in " ".join(report["phases"][9]["reasons"])


def test_evidence_manifest_forbids_aerocity_bench_access(tmp_path):
    payload = _build_evidence(tmp_path)
    payload["aerocity_bench_accesses"] = ["read-scene"]
    with pytest.raises(ValueError, match="AeroCityBench"):
        HM3DFormalPreflightEvidence.from_dict(payload)


def test_p08_reports_realised_qd_without_a_paired_gain_over_planned_qd(tmp_path):
    evidence = _build_evidence(tmp_path)
    _mutate_artifact(
        tmp_path,
        evidence,
        "P08",
        lambda row: [
            record.update(realised_qd_auc=0.20)
            for record in row["payload"]["qd_mechanism_evidence"]["paired_effect"]["unit_rows"]
        ],
    )

    report = _audit(tmp_path, evidence)

    phase = next(row for row in report["phases"] if row["phase_id"] == "P08")
    assert phase["status"] == "READY"
    assert phase["reasons"] == []
