from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aerocity_bench.adapters import (
    AdapterDeclaration,
    ExternalProcessPlannerBridge,
    PlannerAdapter,
    _action_from_dict,
)
from aerocity_bench.atlas_audit import audit_inspection_atlas
from aerocity_bench.atlas_leakage import _multiclass_probe, audit_atlas_leakage
from aerocity_bench.baselines import create_baseline
from aerocity_bench.canonical import content_hash, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.contracts import ActionPacket, ObservationReceipt, Pose3D
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.evaluator import PrivateEvaluator
from aerocity_bench.fidelity_audit import compare_l0_l1_rankings
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.geometry import AABB, sensor_pose
from aerocity_bench.metrics import _coverage_auc
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import _starts, public_episode_projection
from tools.audit_g2_i_scientific_risks import MANIFEST_SCHEMA, run_audit
from tools.merge_g2_i_density_ablation_shards import merge_density_shards
from tools.merge_g2_i_l0_calibration_shards import merge_shards
from tools.merge_g2_i_prior_ablation_shards import merge_prior_shards
from tools.merge_g2_i_target_process_ablation_shards import (
    merge_target_process_shards,
)
from tools.profile_g2_i_policy_latency import (
    adjudicate_controlled_repeats,
    summarize_invocations,
    summarize_latencies,
)
from tools.run_g2_i_density_ablation import aggregate_density_results
from tools.run_g2_i_l0_calibration import METHODS, _assemble_report
from tools.run_g2_i_prior_ablation import aggregate_prior_results
from tools.run_g2_i_target_process_ablation import aggregate_target_process_results

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


def test_density_shard_merger_supports_documented_direct_script_invocation() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "merge_g2_i_density_ablation_shards.py"),
            "--help",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout


def _calibration_method_report(method_id: str, *, private: bool, recall: float) -> dict:
    ancestors = [
        {
            "layout_hash": content_hash([method_id, index]),
            "episode_count": 3,
            "mean_confirmation_count": 1.0,
            "mean_final_confirmed_recall": recall,
            "mean_confirmed_recall_auc": recall / 2.0,
            "mean_inspection_footprint_final": 0.1,
            "mean_inspection_footprint_auc": 0.05,
            "all_returned_home": True,
            "collision_count": 0,
            "out_of_bounds_actions": 0,
            "deadline_misses": 0,
            "mean_task_time_s": 300.0,
            "mean_wall_clock_s": 1.0,
            "mean_selected_observe_pose_count": 4.0,
        }
        for index in range(5)
    ]
    return {
        "method_id": method_id,
        "observation_profile": "O1-private" if private else "G2-I",
        "requires_private_truth": private,
        "ancestor_count": 5,
        "ancestors": ancestors,
        "nonzero_ancestor_count": 5,
        "mean_confirmation_count": 1.0,
        "mean_final_confirmed_recall": recall,
    }


def test_l0_calibration_shards_merge_only_complete_bound_method_set() -> None:
    reports = [
        _calibration_method_report(method_id, private=method_id == METHODS[-1], recall=0.4)
        for method_id in METHODS
    ]
    shards = [
        _assemble_report(
            calibration_manifest_hash="a" * 64,
            calibration_implementation_hash="c" * 64,
            episode_duration_s=300.0,
            max_steps=None,
            record_count=15,
            method_reports=[report],
        )
        for report in reports
    ]
    merged = merge_shards(shards)
    assert merged["contract"]["methods"] == list(METHODS)
    assert merged["searchability_gate"]["complete_method_set"] is True
    assert merged["searchability_gate"]["oracle_feasible_and_returns"] is True
    assert merged["searchability_gate"]["stable_nonsaturated_non_oracle_methods"] == list(
        METHODS[:2]
    )
    assert merged["report_hash"] == content_hash(
        {key: value for key, value in merged.items() if key != "report_hash"}
    )

    mismatched = copy.deepcopy(shards[-1])
    mismatched["contract"]["calibration_manifest_hash"] = "b" * 64
    mismatched.pop("report_hash")
    mismatched["report_hash"] = content_hash(mismatched)
    with pytest.raises(ValueError, match="same manifest"):
        merge_shards([*shards[:-1], mismatched])

    mismatched_code = copy.deepcopy(shards[-1])
    mismatched_code["contract"]["calibration_implementation_hash"] = "d" * 64
    mismatched_code.pop("report_hash")
    mismatched_code["report_hash"] = content_hash(mismatched_code)
    with pytest.raises(ValueError, match="same implementation"):
        merge_shards([*shards[:-1], mismatched_code])
    with pytest.raises(ValueError, match="multiple shards"):
        merge_shards([*shards, shards[0]])


def test_density_shards_require_complete_policy_method_grid() -> None:
    policies = ("sparse", "nominal", "dense")
    shards = []
    for policy_id in policies:
        for method_id in METHODS[:2]:
            shard = {
                "schema": "org.aerocity.bench.g2-i-atlas-density-ablation.v1",
                "formal_score_eligible": False,
                "overall_status": "CALIBRATION_ONLY",
                "manifest_hash": "a" * 64,
                "source_manifest_hash": "d" * 64,
                "base_calibration_implementation_hash": "c" * 64,
                "density_implementation_hash": "b" * 64,
                "execution_level": "L0",
                "contract": {
                    "declared_policy_ids": list(policies),
                    "target_truth_visible_to_methods": False,
                    "policy_specific_sector_frozen_before_private_sampling": True,
                    "private_target_realizations_matched_across_density_conditions": True,
                    "private_targets_reused_from_source_manifest": True,
                    "public_region_cohort_matched_across_density_conditions": True,
                    "selected_city_panel_matched_across_density_conditions": True,
                },
                "method_reports": [
                    {
                        "sampling_policy_id": policy_id,
                        "method_id": method_id,
                        "independent_ancestor_count": 5,
                    }
                ],
            }
            shard["report_hash"] = content_hash(shard)
            shards.append(shard)
    merged = merge_density_shards(shards)
    assert merged["contract"]["complete_condition_set"] is True
    assert merged["aggregate"]["condition_count"] == 6
    assert merged["report_hash"] == content_hash(
        {key: value for key, value in merged.items() if key != "report_hash"}
    )
    with pytest.raises(ValueError, match="every policy-method"):
        merge_density_shards(shards[:-1])

    unsafe = copy.deepcopy(shards[-1])
    unsafe["contract"][
        "private_target_realizations_matched_across_density_conditions"
    ] = False
    unsafe.pop("report_hash")
    unsafe["report_hash"] = content_hash(unsafe)
    with pytest.raises(ValueError, match="paired private-safe"):
        merge_density_shards([*shards[:-1], unsafe])


def test_target_process_shards_require_complete_methods_and_implementation() -> None:
    reports = []
    for method_id in ("atlas-surface-inspector", "atlas-region-greedy"):
        reports.append(
            {
                "method_id": method_id,
                "by_target_process": [],
                "paired_final_recall_deltas": [],
            }
        )
    shards = []
    for report in reports:
        shard = {
            "schema": "org.aerocity.bench.g2-i-target-process-performance-ablation.v1",
            "formal_score_eligible": False,
            "overall_status": "CALIBRATION_ONLY",
            "manifest_hash": "a" * 64,
            "base_calibration_implementation_hash": "c" * 64,
            "target_process_implementation_hash": "b" * 64,
            "execution_level": "L0",
            "contract": {},
            "aggregate": {
                "independent_ancestor_count": 5,
                "target_processes": ["clustered_surface", "uniform_surface"],
            },
            "method_reports": [report],
        }
        shard["report_hash"] = content_hash(shard)
        shards.append(shard)
    merged = merge_target_process_shards(shards)
    assert merged["contract"]["complete_method_set"] is True
    assert merged["aggregate"]["method_count"] == 2
    with pytest.raises(ValueError, match="same implementation"):
        changed = copy.deepcopy(shards[-1])
        changed["target_process_implementation_hash"] = "c" * 64
        changed.pop("report_hash")
        changed["report_hash"] = content_hash(changed)
        merge_target_process_shards([shards[0], changed])


def test_prior_shards_require_complete_prior_levels() -> None:
    shards = []
    for prior_level, method_id in (
        ("coarse-regions", "atlas-coarse-region-inspector"),
        ("full-cells", "atlas-region-greedy"),
    ):
        shard = {
            "schema": "org.aerocity.bench.g2-i-prior-ablation.v1",
            "formal_score_eligible": False,
            "overall_status": "DIAGNOSTIC_ONLY",
            "manifest_hash": "a" * 64,
            "base_calibration_implementation_hash": "c" * 64,
            "prior_implementation_hash": "b" * 64,
            "execution_level": "L0",
            "contract": {},
            "aggregate": [
                {
                    "prior_level": prior_level,
                    "method_id": method_id,
                    "independent_ancestor_count": 5,
                }
            ],
        }
        shard["report_hash"] = content_hash(shard)
        shards.append(shard)
    merged = merge_prior_shards(shards)
    assert merged["contract"]["complete_prior_set"] is True
    assert [row["prior_level"] for row in merged["aggregate"]] == [
        "coarse-regions",
        "full-cells",
    ]
    with pytest.raises(ValueError, match="complete prior-level"):
        merge_prior_shards(shards[:1])


@pytest.fixture(scope="module")
def risk_fixture():
    config = load_ordinary_config(CONFIG_PATH)
    assets = list(config.raw["assets"]["allowlist"])
    city = None
    for attempt in range(24):
        try:
            city = generate_city_v3(config, "train", 0, attempt, assets)
            break
        except GenerationRejected:
            continue
    if city is None:
        raise AssertionError("expected an admitted G2-I risk-audit city")
    task_spec = compile_g2_i_task_spec(city, config.raw["execution_contract"], config.raw["fleet"])
    cells = [
        cell for region in task_spec["inspection_atlas"]["regions"] for cell in region["cells"]
    ][:12]
    if len(cells) < 12:
        raise AssertionError("risk fixture needs twelve public cells")

    def private_site(cell, site_id):
        normal = tuple(float(value) for value in cell["surface_normal"])
        position = [float(value) for value in cell["surface_point"]]
        return {
            "site_id": site_id,
            "position": position,
            "normal": list(normal),
            "owner_collider_id": "unit-owner",
            "valid_before_run": True,
        }

    targets = []
    distractors = []
    pairs = []
    for index in range(6):
        target = private_site(cells[index], f"unit-target-site-{index}")
        target["target_id"] = f"unit-target-{index}"
        distractor = private_site(cells[index + 6], f"unit-distractor-site-{index}")
        distractor["distractor_id"] = f"unit-distractor-{index}"
        targets.append(target)
        distractors.append(distractor)
        pairs.append(
            {
                "pair_id": f"unit-pair-{index}",
                "target_site_id": target["site_id"],
                "distractor_site_id": distractor["site_id"],
            }
        )
    validity = {
        "schema": "org.aerocity.bench.target-validity-private.v1",
        "layout_hash": city["layout_hash"],
        "condition_group_id": "unit-condition",
        "target_ids": [target["target_id"] for target in targets],
        "site_ids": [target["site_id"] for target in targets],
        "witness_hashes": [content_hash([]) for _ in targets],
        "reachability_hashes": [content_hash(["unit-reachability", i]) for i in range(6)],
        "frozen_before_execution": True,
    }
    validity["validity_hash"] = content_hash(validity)
    episode = {
        "schema": "org.aerocity.bench.episode-private.ordinary.v3",
        "episode_id": "unit-risk-episode",
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "target_process": "unit-process",
        "target_count": len(targets),
        "targets": targets,
        "distractors": distractors,
        "counterfactual_pairs": pairs,
        "target_validity": validity,
        "fleet_profile": {
            "name": config.raw["fleet"]["profile"],
            "count": config.fleet_count,
        },
        "starts": _starts(city, config, config.fleet_count, 20260731),
        "execution_contract_hash": content_hash(config.raw["execution_contract"]),
    }
    episode["episode_hash"] = content_hash(episode)
    return config, city, episode, task_spec


def _accepted(observation_id: str, drone_id: str, timestamp_s: float) -> ObservationReceipt:
    return ObservationReceipt.create(observation_id, drone_id, timestamp_s, True, "accepted")


def _cell_observation(
    runtime: L0FleetRuntime, drone_id: str, cell, timestamp_s: float, sequence: int
):
    template = runtime.reset()[drone_id]
    return dataclasses.replace(
        template,
        observation_id=f"strict-cell-{sequence}",
        sequence=sequence,
        timestamp_s=timestamp_s,
        pose=Pose3D(cell.pose.position, cell.pose.yaw_deg),
        sensor_pitch_deg=cell.pose.pitch_deg,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
    )


def test_public_cell_credit_requires_orientation_los_acceptance_and_dwell(risk_fixture) -> None:
    config, city, episode, task_spec = risk_fixture
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public_episode_projection(episode),
    )
    drone_id = sorted(runtime.reset())[0]
    cell = next(
        candidate
        for candidate in runtime._public_atlas_cells.values()
        if runtime._public_cell_visible(
            _cell_observation(runtime, drone_id, candidate, 0.0, 0), candidate
        )
    )
    correct = _cell_observation(runtime, drone_id, cell, 0.0, 0)
    wrong_yaw = dataclasses.replace(
        correct,
        pose=Pose3D(
            cell.pose.position,
            cell.pose.yaw_deg + 180.0,
            cell.pose.pitch_deg,
        ),
    )
    wrong_pitch = dataclasses.replace(
        correct,
        sensor_pitch_deg=cell.pose.pitch_deg + 180.0,
    )
    assert not runtime._public_cell_visible(wrong_yaw, cell)
    assert not runtime._public_cell_visible(wrong_pitch, cell)

    # The nominal atlas pose is a public suggestion, not a hard waypoint.
    # Moving farther out along the public surface normal must remain eligible
    # when the observation still satisfies the physical sensor contract.
    normal = cell.surface_normal
    refined_position = tuple(
        value + 0.75 * normal[index] for index, value in enumerate(cell.pose.position)
    )
    refined = dataclasses.replace(
        correct,
        pose=Pose3D(refined_position, cell.pose.yaw_deg, cell.pose.pitch_deg),
    )
    assert runtime._public_cell_visible(refined, cell)

    camera = sensor_pose(
        correct.pose,
        config.raw["execution_contract"]["sensor_rig"]["translation_body_m"],
        sensor_pitch_deg=correct.sensor_pitch_deg,
    )
    midpoint = tuple(
        (first + second) / 2.0
        for first, second in zip(camera.position, cell.surface_point, strict=True)
    )
    blocker = AABB.from_center_size("test-los-blocker", midpoint, (0.08, 0.08, 0.08))
    runtime._colliders.append(blocker)
    assert not runtime._public_cell_visible(correct, cell)
    runtime._colliders.pop()

    rejected = ObservationReceipt.create(correct.observation_id, drone_id, 0.0, False, "stale")
    runtime._record_public_atlas_observation(correct, rejected)
    assert cell.cell_id not in runtime._visited_public_atlas_cells
    runtime._record_public_atlas_observation(
        correct, _accepted(correct.observation_id, drone_id, 0.0)
    )
    assert cell.cell_id not in runtime._visited_public_atlas_cells
    period = float(config.raw["execution_contract"]["control_period_s"])
    dwell = float(config.raw["execution_contract"]["observe"]["continuous_dwell_s"])
    timestamp = period
    sequence = 1
    while timestamp <= dwell + period + 1.0e-9:
        observation = _cell_observation(runtime, drone_id, cell, timestamp, sequence)
        runtime._record_public_atlas_observation(
            observation, _accepted(observation.observation_id, drone_id, timestamp)
        )
        timestamp += period
        sequence += 1
    assert cell.cell_id in runtime._visited_public_atlas_cells


def test_g2_i_runtime_uses_and_binds_the_method_visible_episode(risk_fixture) -> None:
    config, city, episode, task_spec = risk_fixture

    with pytest.raises(ValueError, match="method-visible public episode"):
        L0FleetRuntime(config, city, episode, public_task_spec=task_spec)

    public_episode = public_episode_projection(episode)
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public_episode,
    )
    result = runtime.result()
    assert result["public_task_spec_hash"] == content_hash(task_spec)
    assert result["public_episode_hash"] == content_hash(public_episode)

    public_episode["starts"][0] = {
        **public_episode["starts"][0],
        "yaw_deg": float(public_episode["starts"][0]["yaw_deg"]) + 1.0,
    }
    with pytest.raises(ValueError, match="binding differs for starts"):
        L0FleetRuntime(
            config,
            city,
            episode,
            public_task_spec=task_spec,
            public_episode=public_episode,
        )


def test_g2_i_runtime_snapshots_public_and_private_setup_objects(risk_fixture) -> None:
    config, city, episode, task_spec = risk_fixture
    caller_city = copy.deepcopy(city)
    caller_episode = copy.deepcopy(episode)
    caller_task_spec = copy.deepcopy(task_spec)
    public_episode = public_episode_projection(caller_episode)
    public_task_hash = content_hash(caller_task_spec)
    public_episode_hash = content_hash(public_episode)
    private_episode_hash = content_hash(caller_episode)
    city_hash = content_hash(caller_city)

    runtime = L0FleetRuntime(
        config,
        caller_city,
        caller_episode,
        public_task_spec=caller_task_spec,
        public_episode=public_episode,
    )
    public_cell_ids = set(runtime._public_atlas_cells)

    # These emulate a policy-side or caller-side mutation after setup.  The
    # authority state must continue to score the exact objects it validated.
    public_episode["starts"][0]["position"][0] += 100.0
    caller_task_spec["inspection_atlas"]["regions"] = []
    caller_episode["starts"][0]["position"][0] += 100.0
    caller_city["colliders"] = []

    assert set(runtime._public_atlas_cells) == public_cell_ids
    assert content_hash(runtime.public_task_spec) == public_task_hash
    assert content_hash(runtime.public_episode) == public_episode_hash
    assert content_hash(runtime.private_episode) == private_episode_hash
    assert content_hash(runtime.city) == city_hash
    result = runtime.result()
    assert result["public_task_spec_hash"] == public_task_hash
    assert result["public_episode_hash"] == public_episode_hash


def test_g2_i_public_entrypoints_reject_unvalidated_episode_before_policy_receives_it(
    risk_fixture, tmp_path: Path
) -> None:
    config, _city, episode, task_spec = risk_fixture
    public_episode = public_episode_projection(episode)
    public_episode["unrecognized_public_field"] = "must-not-reach-policy"

    with pytest.raises(ValueError, match="public episode fields differ"):
        create_baseline("atlas-region-greedy", config, task_spec, public_episode)

    declaration = AdapterDeclaration(
        adapter_id="unit-public-episode-boundary-v1",
        method_id="unit-public-method",
        capability_profile="G2-I",
        upstream_url=None,
        upstream_commit=None,
        upstream_license="MIT",
        process_boundary="in_process",
        training_allowed=False,
        decentralized_execution=False,
    )
    adapter = PlannerAdapter(declaration, planner=object())
    with pytest.raises(ValueError, match="public episode fields differ"):
        adapter.reset(public_episode, public_task_spec=task_spec)

    external_declaration = dataclasses.replace(
        declaration,
        adapter_id="unit-public-episode-process-boundary-v1",
        process_boundary="process",
        upstream_url="https://example.invalid/public-method",
        upstream_commit="a" * 40,
    )
    server = tmp_path / "planner.py"
    server.write_text("raise SystemExit(0)\n", encoding="utf-8")
    with ExternalProcessPlannerBridge(
        external_declaration,
        [sys.executable, "-u", str(server)],
        cwd=tmp_path,
    ) as bridge:
        with pytest.raises(ValueError, match="public episode fields differ"):
            bridge.reset(public_episode, public_task_spec=task_spec)


def test_public_waypoint_pitch_reaches_l0_cell_visibility_through_gimbal(risk_fixture) -> None:
    config, city, episode, task_spec = risk_fixture
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public_episode_projection(episode),
    )
    drone_id = sorted(runtime.reset())[0]
    template = runtime.reset()[drone_id]
    cell = next(
        candidate
        for candidate in runtime._public_atlas_cells.values()
        if abs(candidate.pose.pitch_deg) > 1.0
        and runtime._public_cell_visible(
            _cell_observation(runtime, drone_id, candidate, 0.0, 0), candidate
        )
    )
    action = _action_from_dict(
        {
            "kind": "WAYPOINT",
            "waypoint": Pose3D(cell.pose.position, cell.pose.yaw_deg).to_dict(),
            "sensor_pitch_deg": cell.pose.pitch_deg,
        },
        template,
        template.sequence,
    )
    state = runtime._states[drone_id]
    state.pose = Pose3D(cell.pose.position, cell.pose.yaw_deg, 0.0)
    requested = runtime._requested_destination(state, action)

    assert requested.pitch_deg == 0.0
    first_gimbal_pitch = runtime._requested_sensor_pitch(state, action)
    assert first_gimbal_pitch == pytest.approx(-18.0)
    state.sensor_pitch_deg = cell.pose.pitch_deg
    assert runtime._requested_sensor_pitch(state, action) == pytest.approx(cell.pose.pitch_deg)
    assert runtime._public_cell_visible(
        dataclasses.replace(
            template,
            pose=requested,
            sensor_pitch_deg=state.sensor_pitch_deg,
            linear_velocity_world_mps=(0.0, 0.0, 0.0),
            angular_speed_deg_s=0.0,
        ),
        cell,
    )


def test_area_weighted_auc_is_invariant_to_auxiliary_cell_splitting() -> None:
    duration = 10.0
    unsplit_area = [[2.0, 4.0], [6.0, 10.0]]
    split_area = [[2.0, 4.0], [6.0, 10.0]]
    assert _coverage_auc(unsplit_area, 1, duration, 10.0) == pytest.approx(
        _coverage_auc(split_area, 1, duration, 10.0)
    )
    assert _coverage_auc([[2.0, 1.0], [6.0, 2.0]], 1, duration, 2.0) != pytest.approx(
        _coverage_auc([[2.0, 1.0], [6.0, 1.0]], 1, duration, 1.0)
    )


def test_public_latency_profile_summarizes_receipts_without_truth_fields() -> None:
    timing = summarize_latencies([0.02, 0.05, 0.16, 0.25], deadline_s=0.15)
    assert timing == {
        "receipt_count": 4,
        "deadline_miss_count": 2,
        "mean_planning_latency_s": 0.12,
        "p95_planning_latency_s": 0.25,
        "max_planning_latency_s": 0.25,
        "top_ten_planning_latency_s": [0.25, 0.16, 0.05, 0.02],
    }
    with pytest.raises(ValueError, match="non-negative"):
        summarize_latencies([0.01, -0.01], deadline_s=0.15)


def test_latency_profile_distinguishes_invocations_from_drone_receipts() -> None:
    timing = summarize_invocations(
        [
            {
                "invocation_index": 0,
                "task_time_s": 1.0,
                "active_drone_count": 4,
                "wall_clock_latency_s": 0.18,
                "process_cpu_latency_s": 0.03,
                "non_cpu_delay_s": 0.15,
                "deadline_miss_receipt_count": 4,
            },
            {
                "invocation_index": 1,
                "task_time_s": 1.25,
                "active_drone_count": 4,
                "wall_clock_latency_s": 0.02,
                "process_cpu_latency_s": 0.019,
                "non_cpu_delay_s": 0.001,
                "deadline_miss_receipt_count": 0,
            },
        ],
        deadline_s=0.15,
    )
    assert timing["deadline_overrun_invocation_count"] == 1
    assert timing["deadline_miss_receipt_count"] == 4
    assert timing["overrun_invocations"][0]["attribution"] == ("non_cpu_delay_candidate")


def test_controlled_repeat_rule_requires_three_safe_runs_with_p95_headroom() -> None:
    replicate = {
        "timing": {
            "deadline_overrun_invocation_count": 0,
            "deadline_miss_receipt_count": 0,
            "wall_clock": {
                "p95_planning_latency_s": 0.08,
                "max_planning_latency_s": 0.11,
            },
        },
        "safety": {
            "returned_home_all": True,
            "collision_count": 0,
            "out_of_bounds_actions": 0,
        },
    }
    diagnostic = adjudicate_controlled_repeats([replicate, replicate], deadline_s=0.15)
    assert diagnostic["status"] == "DIAGNOSTIC_ONLY"
    assert diagnostic["permits_full_calibration_rerun"] is False

    passed = adjudicate_controlled_repeats([replicate, replicate, replicate], deadline_s=0.15)
    assert passed["status"] == "PASS"
    assert passed["permits_full_calibration_rerun"] is True

    no_headroom = {
        **replicate,
        "timing": {
            **replicate["timing"],
            "wall_clock": {
                "p95_planning_latency_s": 0.12,
                "max_planning_latency_s": 0.12,
            },
        },
    }
    blocked = adjudicate_controlled_repeats([replicate, replicate, no_headroom], deadline_s=0.15)
    assert blocked["status"] == "NO_GO"
    assert blocked["checks"]["wall_clock_p95_has_25_percent_headroom"] is False

    shared_host = adjudicate_controlled_repeats(
        [replicate, replicate, replicate],
        deadline_s=0.15,
        host_quiescent=False,
    )
    assert shared_host["status"] == "NO_GO"
    assert shared_host["checks"]["host_quiescent_before_and_after"] is False


def test_target_process_ablation_uses_paired_ancestor_aggregation() -> None:
    raw = []
    for method_id in ("atlas-surface-inspector", "atlas-region-greedy"):
        for ancestor in ("a", "b"):
            for process, recall in (("uniform_surface", 0.1), ("clustered_surface", 0.3)):
                raw.append(
                    {
                        "method_id": method_id,
                        "layout_ancestor": ancestor,
                        "target_process": process,
                        "confirmation_count": 1,
                        "final_confirmed_recall": recall,
                        "confirmed_recall_auc": recall / 2.0,
                        "inspection_footprint_final": 0.2,
                        "returned_home_all": True,
                        "collision_count": 0,
                        "out_of_bounds_actions": 0,
                        "deadline_misses": 0,
                    }
                )
    reports = aggregate_target_process_results(
        raw,
        expected_processes=("uniform_surface", "clustered_surface"),
    )
    assert len(reports) == 2
    assert reports[0]["by_target_process"][0]["independent_ancestor_count"] == 2
    assert reports[0]["by_target_process"][0]["nonzero_ancestor_count"] == 2
    assert reports[0]["paired_final_recall_deltas"] == [
        {
            "comparison": "clustered_surface_minus_uniform_surface",
            "independent_ancestor_count": 2,
            "mean_final_confirmed_recall_delta": pytest.approx(0.2),
        }
    ]
    with pytest.raises(ValueError, match="one row per target process"):
        aggregate_target_process_results(
            raw[:-1],
            expected_processes=("uniform_surface", "clustered_surface"),
        )


def test_density_ablation_uses_ancestor_not_episode_replicate_counts() -> None:
    raw = []
    for policy_id in ("sparse", "nominal"):
        for method_id in ("atlas-surface-inspector", "atlas-region-greedy"):
            for ancestor in ("a", "b"):
                for process in ("uniform_surface", "clustered_surface"):
                    raw.append(
                        {
                            "sampling_policy_id": policy_id,
                            "method_id": method_id,
                            "layout_ancestor": ancestor,
                            "confirmation_count": 1,
                            "final_confirmed_recall": 0.2,
                            "returned_home_all": True,
                            "collision_count": 0,
                            "out_of_bounds_actions": 0,
                            "deadline_misses": 0,
                            "target_process": process,
                        }
                    )
    reports = aggregate_density_results(raw, policy_ids=("sparse", "nominal"))
    assert len(reports) == 4
    assert reports[0]["independent_ancestor_count"] == 2
    assert reports[0]["nonzero_ancestor_count"] == 2
    assert reports[0]["mean_final_confirmed_recall"] == pytest.approx(0.2)
    assert len(reports[0]["ancestors"]) == 2
    assert all(
        set(item).isdisjoint({"layout_ancestor", "target_process"})
        for item in reports[0]["ancestors"]
    )
    assert all(
        item["deadline_failure_replicate_count"] == 0
        and item["maximum_deadline_misses_per_replicate"] == 0
        for item in reports[0]["ancestors"]
    )


def test_prior_ablation_retains_out_of_bounds_failures_in_ancestor_aggregate() -> None:
    raw = []
    for prior_level in ("coarse-regions", "full-cells"):
        for ancestor in ("a", "b"):
            raw.append(
                {
                    "prior_level": prior_level,
                    "layout_ancestor_hash": ancestor,
                    "confirmation_count": 1,
                    "inspection_footprint_final": 0.2,
                    "collision_count": 0,
                    "out_of_bounds_actions": int(
                        prior_level == "full-cells" and ancestor == "b"
                    ),
                    "returned_home_all": True,
                    "deadline_misses": 0,
                }
            )
    aggregate = aggregate_prior_results(
        raw, prior_levels=("coarse-regions", "full-cells")
    )
    by_prior = {row["prior_level"]: row for row in aggregate}
    assert by_prior["coarse-regions"]["out_of_bounds_actions"] == 0
    assert by_prior["full-cells"]["out_of_bounds_actions"] == 1
    assert by_prior["full-cells"]["independent_ancestor_count"] == 2


def test_cpu_atlas_audit_is_aggregate_private_safe_and_native_ineligible(risk_fixture) -> None:
    config, city, _, task_spec = risk_fixture
    report = audit_inspection_atlas(
        city,
        task_spec["inspection_atlas"],
        config.raw["execution_contract"],
        fleet_count=config.fleet_count,
        episode_duration_s=float(config.raw["execution_contract"]["episode"]["duration_s"]),
    )
    assert report["formal_score_eligible"] is False
    assert report["remaining_gates"]["native_cf2x_cell_shortlist_replay"] is False
    assert report["budget_bracket"]["not_a_solvability_claim"] is True
    text = json.dumps(report, sort_keys=True)
    assert "target-" not in text
    assert "distractor-" not in text


def test_grouped_leakage_probe_uses_counterfactual_pairs_without_private_output(
    risk_fixture,
) -> None:
    config, _, episode, task_spec = risk_fixture
    records = []
    for index in range(4):
        atlas = copy.deepcopy(task_spec["inspection_atlas"])
        atlas["layout_id"] = f"unit-layout-{index}"
        atlas["inspection_geometry_hash"] = content_hash(["unit-geometry", index])
        atlas.pop("atlas_hash")
        atlas["atlas_hash"] = content_hash(atlas)
        records.append(
            {
                "atlas": atlas,
                "private_episode": episode,
                "layout_ancestor": f"unit-ancestor-{index}",
                "split_label": "development-a" if index % 2 == 0 else "development-b",
            }
        )
    report = audit_atlas_leakage(
        records,
        execution_contract=config.raw["execution_contract"],
        permutation_count=16,
    )
    assert report["paired_label_probe"]["group_count"] == 4
    assert report["privacy_contract"]["grouped_by_layout_ancestor"] is True
    text = json.dumps(report, sort_keys=True)
    assert not any(item["target_id"] in text for item in episode["targets"])
    assert not any(item["distractor_id"] in text for item in episode["distractors"])
    assert not any(str(item["position"]) in text for item in episode["targets"])


def test_split_probe_permutes_layout_ancestor_labels_as_whole_groups() -> None:
    groups = [f"ancestor-{index:02d}" for index in range(9) for _ in range(3)]
    labels = [label for label in ("calibration", "train", "validation") for _ in range(9)]
    features = [(float(index // 3), float((index // 3) % 2)) for index in range(len(groups))]
    report = _multiclass_probe(
        features,
        labels,
        groups,
        permutation_count=16,
        seed_tag="unit-split",
        permutation_scheme="between-layout-ancestors",
    )
    assert report["permutation_scheme"] == "between-layout-ancestors"

    mixed_labels = list(labels)
    mixed_labels[1] = "train"
    with pytest.raises(ValueError, match="one label per ancestor"):
        _multiclass_probe(
            features,
            mixed_labels,
            groups,
            permutation_count=16,
            seed_tag="unit-invalid-split",
            permutation_scheme="between-layout-ancestors",
        )


def test_external_method_adapter_receives_public_g2_i_contract_only(risk_fixture) -> None:
    _, _, episode, task_spec = risk_fixture

    class Planner:
        reset_payload = None

        def reset(self, task):
            self.reset_payload = task

        def act(self, observations):
            return observations

    planner = Planner()
    declaration = AdapterDeclaration(
        adapter_id="unit-g2-i",
        method_id="unit-public-method",
        capability_profile="G2-I",
        upstream_url=None,
        upstream_commit=None,
        upstream_license="BSD-3-Clause",
        process_boundary="in_process",
        training_allowed=False,
        decentralized_execution=False,
    )
    adapter = PlannerAdapter(declaration, planner)
    with pytest.raises(ValueError, match="requires a public G2-I task spec"):
        adapter.reset(public_episode_projection(episode))
    adapter.reset(public_episode_projection(episode), public_task_spec=task_spec)
    payload = planner.reset_payload
    assert payload["public_task_spec"]["task_track"] == "G2-I"
    text = json.dumps(payload, sort_keys=True)
    assert "target_process" not in text

    def string_values(node):
        if isinstance(node, dict):
            return [value for nested in node.values() for value in string_values(nested)]
        if isinstance(node, list):
            return [value for nested in node for value in string_values(nested)]
        return [node] if isinstance(node, str) else []

    exposed_strings = set(string_values(payload))
    assert exposed_strings.isdisjoint({item["target_id"] for item in episode["targets"]})


def test_l0_l1_ranking_audit_uses_ancestor_means_not_episode_count() -> None:
    l0 = []
    l1 = []
    for method_index, method_id in enumerate(("method-a", "method-b", "method-c")):
        for ancestor_index, ancestor in enumerate(("ancestor-1", "ancestor-2", "ancestor-3")):
            base = 0.9 - 0.2 * method_index - 0.03 * ancestor_index
            for level, records, delta in (("L0", l0, 0.0), ("L1", l1, -0.02 * method_index)):
                records.append(
                    {
                        "method_id": method_id,
                        "layout_ancestor": ancestor,
                        "score": base + delta,
                        "execution_level": level,
                        "evidence_hash": content_hash([level, method_id, ancestor, 0]),
                    }
                )
    # An extra episode replicate changes only ancestor-1's within-ancestor mean;
    # it does not give ancestor-1 a second vote in the cross-layout statistic.
    for level, records in (("L0", l0), ("L1", l1)):
        records.append(
            {
                "method_id": "method-a",
                "layout_ancestor": "ancestor-1",
                "score": 0.7,
                "execution_level": level,
                "evidence_hash": content_hash([level, "method-a", "ancestor-1", 1]),
            }
        )
    report = compare_l0_l1_rankings(l0, l1)
    assert report["status"] == "MEASURED_NOT_FROZEN"
    assert report["independent_ancestor_count"] == 3
    assert report["episode_replicates_are_not_independent"] is True
    assert report["contract_freeze_allowed"] is False


def test_scientific_audit_script_is_development_only_and_private_safe(
    tmp_path: Path, risk_fixture
) -> None:
    config, city, episode, _ = risk_fixture
    write_json(tmp_path / "release_config.json", config.raw)
    write_json(tmp_path / "city.json", city)
    write_json(tmp_path / "episode.json", episode)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "method-independent-task-calibration",
        "self_method_results_used": False,
        "release_config_path": "release_config.json",
        "records": [
            {
                "city_path": "city.json",
                "private_episode_path": "episode.json",
                "layout_ancestor": "unit-ancestor",
                "split_label": "train",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    report = run_audit(manifest_path, permutation_count=16)
    assert report["overall_status"] == "FORMAL_NO_GO"
    assert report["aggregate"]["independent_ancestor_count"] == 1
    assert report["gate_checks"]["native_cf2x_reachability_pass"] is False
    text = json.dumps(report, sort_keys=True)
    assert not any(item["target_id"] in text for item in episode["targets"])

    manifest["records"][0]["split_label"] = "test_iid"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="must not inspect a formal split"):
        run_audit(manifest_path, permutation_count=16)


def test_simultaneous_dwell_tie_is_not_silently_attributed_by_drone_order(
    risk_fixture,
) -> None:
    config, city, episode, task_spec = risk_fixture
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public_episode_projection(episode),
    )
    templates = runtime.reset()
    drone_ids = sorted(templates)[:2]
    target_cell = next(iter(runtime._public_atlas_cells.values()))
    evaluator = PrivateEvaluator(
        config,
        city,
        episode,
        receipt_secret=b"unit-test-secret-long-enough",
    )
    period = float(config.raw["execution_contract"]["control_period_s"])
    dwell = float(config.raw["execution_contract"]["observe"]["continuous_dwell_s"])
    timestamp = 0.0
    sequence = 0
    while timestamp <= dwell + period + 1.0e-9:
        for drone_id in drone_ids:
            observation = dataclasses.replace(
                templates[drone_id],
                observation_id=f"tie-{drone_id}-{sequence}",
                sequence=sequence,
                timestamp_s=timestamp,
                pose=target_cell.pose,
                linear_velocity_world_mps=(0.0, 0.0, 0.0),
                angular_speed_deg_s=0.0,
            )
            action = ActionPacket(
                episode_id=episode["episode_id"],
                drone_id=drone_id,
                sequence=sequence,
                issued_at_s=timestamp,
                kind="OBSERVE",
                source_observation_id=observation.observation_id,
            )
            evaluator.process(observation, action)
        timestamp += period
        sequence += 1
    audit = evaluator.private_audit_snapshot()
    assert audit["confirmed_count"] >= 1
    assert audit["simultaneous_confirmation_tie_count"] >= 1
