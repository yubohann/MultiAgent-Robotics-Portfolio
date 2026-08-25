from __future__ import annotations

import copy

from aerocity_bench.canonical import content_hash
from tools.verify_g2_i_a_gate import (
    DENSITY_POLICIES,
    TARGET_PROCESSES,
    verify_a_gate,
)


def _safe(**values):
    return {
        "all_returned_home": True,
        "collision_count": 0,
        "out_of_bounds_actions": 0,
        "deadline_misses": 0,
        **values,
    }


def _inputs():
    manifest_hash = "a" * 64
    implementation_hash = "b" * 64
    public_reports = []
    for method_index, method_id in enumerate(
        ("atlas-surface-inspector", "atlas-region-greedy")
    ):
        ancestors = [
            _safe(
                mean_confirmation_count=float(index + method_index),
                mean_final_confirmed_recall=0.2 + 0.05 * method_index,
            )
            for index in range(5)
        ]
        public_reports.append(
            {
                "method_id": method_id,
                "ancestor_count": 5,
                "nonzero_ancestor_count": 5,
                "mean_final_confirmed_recall": 0.2 + 0.05 * method_index,
                "ancestors": ancestors,
            }
        )
    oracle = {
        "method_id": "centralized-oracle",
        "ancestor_count": 5,
        "nonzero_ancestor_count": 5,
        "mean_final_confirmed_recall": 0.8,
        "ancestors": [_safe(mean_confirmation_count=4.0) for _ in range(5)],
    }
    searchability = {
        "contract": {
            "calibration_manifest_hash": manifest_hash,
            "calibration_implementation_hash": implementation_hash,
            "episode_duration_s": 300.0,
            "max_steps": None,
            "target_count_visible_to_method": False,
            "l0_not_a_native_or_formal_score": True,
            "mission_sector_required": True,
            "mission_sector_frozen_before_private_sampling": True,
            "methods": [
                "atlas-surface-inspector",
                "atlas-region-greedy",
                "centralized-oracle",
            ],
        },
        "searchability_gate": {"complete_method_set": True},
        "method_reports": [*public_reports, oracle],
        "report_hash": "1" * 64,
    }
    density = {
        "source_manifest_hash": manifest_hash,
        "base_calibration_implementation_hash": implementation_hash,
        "contract": {
            "complete_condition_set": True,
            "declared_policy_ids": list(DENSITY_POLICIES),
            "selected_policy_ids": list(DENSITY_POLICIES),
            "selected_method_ids": [
                "atlas-surface-inspector",
                "atlas-region-greedy",
            ],
            "target_truth_visible_to_methods": False,
            "l0_not_a_native_or_formal_score": True,
            "selected_city_panel_matched_across_density_conditions": True,
            "public_region_cohort_matched_across_density_conditions": True,
            "private_target_realizations_matched_across_density_conditions": True,
        },
        "aggregate": {"condition_count": 6},
        "method_reports": [
            _safe(
                sampling_policy_id=policy,
                method_id=method,
                independent_ancestor_count=5,
                nonzero_ancestor_count=4,
            )
            for policy in DENSITY_POLICIES
            for method in ("atlas-surface-inspector", "atlas-region-greedy")
        ],
        "report_hash": "2" * 64,
    }
    target_process = {
        "manifest_hash": manifest_hash,
        "base_calibration_implementation_hash": implementation_hash,
        "contract": {
            "complete_method_set": True,
            "declared_method_ids": [
                "atlas-surface-inspector",
                "atlas-region-greedy",
            ],
            "selected_method_ids": [
                "atlas-surface-inspector",
                "atlas-region-greedy",
            ],
            "target_truth_visible_to_methods": False,
            "l0_not_a_native_or_formal_score": True,
            "paired_by_layout_ancestor": True,
            "frozen_private_episode_replayed": True,
        },
        "aggregate": {"target_processes": list(TARGET_PROCESSES)},
        "method_reports": [
            {
                "method_id": method,
                "by_target_process": [
                    _safe(independent_ancestor_count=5, target_process=process)
                    for process in TARGET_PROCESSES
                ],
                "paired_final_recall_deltas": [
                    {"comparison": "clustered_surface_minus_uniform_surface"},
                    {"comparison": "height_stratified_minus_uniform_surface"},
                ],
            }
            for method in ("atlas-surface-inspector", "atlas-region-greedy")
        ],
        "report_hash": "3" * 64,
    }
    prior = {
        "manifest_hash": manifest_hash,
        "base_calibration_implementation_hash": implementation_hash,
        "contract": {
            "complete_prior_set": True,
            "coarse_policy_receives_cells_or_poses": False,
            "declared_prior_levels": ["coarse-regions", "full-cells"],
            "selected_prior_levels": ["coarse-regions", "full-cells"],
            "target_truth_visible_to_public_policy": False,
            "l0_not_a_native_or_formal_score": True,
            "same_private_episode_per_pair": True,
            "same_strict_full_atlas_evaluator_per_pair": True,
        },
        "aggregate": [
            _safe(
                prior_level="coarse-regions",
                method_id="atlas-coarse-region-inspector",
                nonzero_ancestor_count=1,
            ),
            _safe(
                prior_level="full-cells",
                method_id="atlas-region-greedy",
                nonzero_ancestor_count=5,
            ),
        ],
        "report_hash": "4" * 64,
    }
    scientific = {
        "manifest_hash": manifest_hash,
        "aggregate": {"independent_ancestor_count": 5},
        "geometry_reports": [{"execution_contract_hash": "e" * 64}],
        "gate_checks": {
            "cpu_geometry_all_pass": True,
            "paired_leakage_probe_pass": True,
            "sector_process_leakage_probe_pass": True,
            "sampling_policy_frozen": True,
        },
        "leakage_report": {},
        "report_hash": "5" * 64,
    }
    split = copy.deepcopy(scientific)
    split["manifest_hash"] = "c" * 64
    split["aggregate"]["independent_ancestor_count"] = 9
    split["leakage_report"] = {
        "split_label_probe": {"status": "PASS_NO_DETECTED_SIGNAL"},
        "sector_split_label_probe": {"status": "PASS_NO_DETECTED_SIGNAL"},
    }
    split["report_hash"] = "6" * 64
    return searchability, density, target_process, prior, scientific, split


def test_a_gate_verifies_only_complete_bound_evidence() -> None:
    inputs = _inputs()
    report = verify_a_gate(
        searchability=inputs[0],
        density=inputs[1],
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert report["status"] == "VERIFIED"
    assert report["failure_count"] == 0
    assert report["authorizes_next_gate"] is True
    assert report["report_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    mismatched = copy.deepcopy(inputs[2])
    mismatched["base_calibration_implementation_hash"] = "d" * 64
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=inputs[1],
        target_process=mismatched,
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert blocked["checks"]["bindings"]["same_base_calibration_implementation"] is False


def test_a_gate_rejects_condition_identity_substitution_and_duplicates() -> None:
    inputs = _inputs()

    density = copy.deepcopy(inputs[1])
    density["method_reports"][-1] = copy.deepcopy(density["method_reports"][0])
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=density,
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["checks"]["density"]["complete_three_by_two_condition_grid"]
        is False
    )

    target_process = copy.deepcopy(inputs[2])
    target_process["method_reports"][1]["method_id"] = "unknown-public-method"
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=inputs[1],
        target_process=target_process,
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["checks"]["target_process"]["complete_paired_process_panel"]
        is False
    )

    prior = copy.deepcopy(inputs[3])
    prior["aggregate"].append(copy.deepcopy(prior["aggregate"][0]))
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=inputs[1],
        target_process=inputs[2],
        prior=prior,
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert blocked["checks"]["prior"]["complete_coarse_full_pair"] is False


def test_a_gate_fails_closed_on_missing_safety_or_ancestor_evidence() -> None:
    inputs = _inputs()

    density = copy.deepcopy(inputs[1])
    density["method_reports"][2].pop("out_of_bounds_actions")
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=density,
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["checks"]["density"]["nominal_density_conditions_safe_and_return"]
        is False
    )

    searchability = copy.deepcopy(inputs[0])
    searchability["method_reports"][2]["ancestors"] = []
    blocked = verify_a_gate(
        searchability=searchability,
        density=inputs[1],
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["checks"]["searchability"][
            "oracle_five_ancestor_feasible_and_returns"
        ]
        is False
    )


def test_a_gate_rejects_mutated_task_or_execution_contract() -> None:
    inputs = _inputs()

    searchability = copy.deepcopy(inputs[0])
    searchability["contract"]["episode_duration_s"] = 301.0
    blocked = verify_a_gate(
        searchability=searchability,
        density=inputs[1],
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=inputs[5],
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["checks"]["searchability"]["canonical_task_contract_used"]
        is False
    )

    split = copy.deepcopy(inputs[5])
    split["geometry_reports"][0]["execution_contract_hash"] = "f" * 64
    blocked = verify_a_gate(
        searchability=inputs[0],
        density=inputs[1],
        target_process=inputs[2],
        prior=inputs[3],
        scientific_audit=inputs[4],
        split_audit=split,
    )
    assert blocked["status"] == "NO_GO"
    assert blocked["checks"]["bindings"]["same_nonempty_execution_contract"] is False
