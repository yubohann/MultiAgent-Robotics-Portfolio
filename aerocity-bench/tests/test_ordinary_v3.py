from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

import aerocity_bench.baselines as baseline_module
from aerocity_bench.adapters import (
    AdapterDeclaration,
    DecentralizedPlannerAdapter,
    ExternalProcessPlannerBridge,
    PettingZooParallelWrapper,
    PlannerAdapter,
    ReplayWriter,
    project_g1,
    validate_replay,
)
from aerocity_bench.baselines import (
    BASELINES,
    _anisotropic_motion_lower_bound_s,
    _route_respects_public_prior,
    create_baseline,
)
from aerocity_bench.blind import submission_spec
from aerocity_bench.builder_v3 import (
    MAX_ATTEMPTS_PER_LAYOUT,
    build_ordinary_release,
    export_public_release,
    promote_ordinary_release,
    validate_ordinary_release,
    validate_public_release,
)
from aerocity_bench.canonical import content_hash, file_hash, write_json, write_json_atomic
from aerocity_bench.cli import (
    _capture_failure_detail,
    _capture_review_batch,
    _native_gate,
    _review_layout_authority,
    _verified_prepared_attempt,
    _verified_review_attempt,
)
from aerocity_bench.cli import main as cli_main
from aerocity_bench.compiler import (
    compile_coarse_prior,
    compile_g2_i_task_spec,
    compile_method_task_spec,
    compile_scene,
)
from aerocity_bench.contracts import (
    ActionPacket,
    BudgetLedger,
    MessagePacket,
    ObservationPacket,
    Pose3D,
)
from aerocity_bench.errors import (
    AssetRegistryError,
    GenerationRejected,
    HostGuardError,
    ValidationError,
)
from aerocity_bench.evaluator import PrivateEvaluator
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.geometry import (
    AABB,
    distance,
    line_of_sight,
    minimum_segment_clearance,
    review_camera_pose,
    segment_aabb_clearance,
    segment_intersects_expanded_aabb,
    segment_segment_distance,
)
from aerocity_bench.host_guard import (
    HOST_GUARD_SCHEMA,
    GuardedProcessResult,
    HostSnapshot,
    commit_limit_exceeded,
    host_snapshot,
    is_host_1344,
    isaac_host_lock,
    run_guarded_process,
)
from aerocity_bench.isaac_bridge import (
    CAPABILITY_L1_EVIDENCE_SCOPE,
    FORMAL_L1_EVIDENCE_SCOPE,
    REQUIRED_NATIVE_CHECKS,
    REVIEW_BASE_FRAMES,
    VISUAL_REVIEW_EVIDENCE_SCOPE,
    aggregate_review_instance_visibility,
    assert_formal_receipts,
    build_l1_execution_receipt,
    formal_execution_context,
    validate_native_gate_report,
    write_native_gate_report,
)
from aerocity_bench.metrics import confirmed_recall_auc, evaluate_run, time_to_recall
from aerocity_bench.native_gate_contract import (
    build_native_action_transcript,
    commanded_braking_distance,
    compare_native_replays,
    evaluate_native_dwell_samples,
    load_native_gate_inputs,
    select_native_test_directions,
)
from aerocity_bench.ordinary_config import (
    FORMAL_SPLITS,
    ORDINARY_SPLITS,
    OrdinaryReleaseConfig,
    load_ordinary_config,
    load_public_runtime_contract,
    public_execution_contract,
    validate_public_execution_contract,
)
from aerocity_bench.resources import preset, schema, write_preset
from aerocity_bench.ros_bridge import ROSWireCodec
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.scene_audit import audit_development_layout, audit_generated_city
from aerocity_bench.supply_chain import load_official_cc0_lock, validate_bundle_root
from aerocity_bench.targets_v3 import (
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
    sample_visual_review_episode_v3,
)
from tools.audit_baseline_opportunity import (
    _arguments as opportunity_audit_arguments,
)
from tools.audit_baseline_opportunity import (
    _execute_policy_with_trace,
    _load_derived_development_inputs,
    _visibility_summary,
    audit_method,
)
from tools.materialize_g2_i_l1_layout import materialize as materialize_g2_i_l1_layout

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def ordinary_config() -> OrdinaryReleaseConfig:
    return load_ordinary_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def city_and_sites(ordinary_config: OrdinaryReleaseConfig):
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    city = _first_admitted_city(ordinary_config, "train", 0, assets)
    sites = derive_support_sites_v3(city, ordinary_config)
    return city, sites


def _first_admitted_city(
    config: OrdinaryReleaseConfig, split: str, index: int, assets: list[str]
) -> dict[str, object]:
    """Match the builder's deterministic rejection-and-resample contract."""

    for attempt in range(MAX_ATTEMPTS_PER_LAYOUT):
        try:
            return generate_city_v3(config, split, index, attempt, assets)
        except GenerationRejected:
            continue
    raise AssertionError(f"no admitted city for {split}[{index}]")


@pytest.fixture()
def episode(ordinary_config: OrdinaryReleaseConfig, city_and_sites):
    city, sites = city_and_sites
    return sample_episode_v3(ordinary_config, city, sites, 0)


def test_config_freezes_calibration_and_formal_splits(
    ordinary_config: OrdinaryReleaseConfig,
) -> None:
    assert ordinary_config.fleet_count == 4
    assert ordinary_config.raw["governance"]["calibration_split"] == "calibration"
    assert tuple(ordinary_config.raw["governance"]["formal_splits"]) == FORMAL_SPLITS
    assert set(ordinary_config.target_processes("test_process_ood")).isdisjoint(
        ordinary_config.target_processes("train")
    )


def test_public_execution_contract_removes_private_target_invariant(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, tmp_path: Path
) -> None:
    """Every public method surface must bind the same redacted contract."""

    city, _ = city_and_sites
    public_contract = public_execution_contract(ordinary_config.raw["execution_contract"])
    validate_public_execution_contract(public_contract)
    assert "fixed_target_count_private" not in public_contract["episode"]

    task = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    assert task["execution_contract"] == public_contract
    assert task["public_execution_contract_hash"] == content_hash(public_contract)
    assert "fixed_target_count_private" not in json.dumps(task, sort_keys=True)

    runtime_contract = {
        "schema": "org.aerocity.bench.runtime-contract-public.ordinary.v1",
        "release_version": ordinary_config.version,
        "generator_version": ordinary_config.generator_version,
        "fleet": ordinary_config.raw["fleet"],
        "execution_contract": public_contract,
        "authority_release_commitment": "a" * 64,
    }
    runtime_contract["contract_hash"] = content_hash(runtime_contract)
    runtime_path = tmp_path / "benchmark_contract.json"
    write_json(runtime_path, runtime_contract)
    assert load_public_runtime_contract(runtime_path).raw["execution_contract"] == public_contract


def test_public_execution_contract_rejects_private_fields(
    ordinary_config: OrdinaryReleaseConfig,
) -> None:
    public_contract = public_execution_contract(ordinary_config.raw["execution_contract"])
    public_contract["episode"]["fixed_target_count_private"] = True
    with pytest.raises(ValueError, match="non-public"):
        validate_public_execution_contract(public_contract)


def test_config_object_order_has_no_semantics(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["split_counts"] = dict(reversed(list(raw["split_counts"].items())))
    raw["episodes_per_layout"] = dict(reversed(list(raw["episodes_per_layout"].items())))
    path = tmp_path / "reordered.json"
    write_json(path, raw)
    assert load_ordinary_config(path).total_layouts == 6


def test_config_rejects_method_tuning_formal_test(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["governance"]["method_results_may_tune_benchmark"] = True
    path = tmp_path / "invalid.json"
    write_json(path, raw)
    with pytest.raises(ValueError, match="cannot tune"):
        load_ordinary_config(path)


def test_config_requires_full_hover_endurance_without_land_action(tmp_path: Path) -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["execution_contract"]["vehicle"]["energy_budget_j"] = 9000.0
    path = tmp_path / "insufficient-endurance.json"
    write_json(path, raw)
    with pytest.raises(ValueError, match="full-episode hover"):
        load_ordinary_config(path)


def test_generator_is_deterministic_and_topology_ood_is_disjoint(
    ordinary_config: OrdinaryReleaseConfig,
) -> None:
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    first = _first_admitted_city(ordinary_config, "train", 0, assets)
    second = _first_admitted_city(ordinary_config, "train", 0, assets)
    holdout = _first_admitted_city(ordinary_config, "test_topology", 0, assets)
    assert first == second
    assert first["layout_hash"] != holdout["layout_hash"]
    assert first["family_private"] in ordinary_config.raw["city_grammar"]["development_families"]
    assert holdout["family_private"] in ordinary_config.raw["city_grammar"]["topology_ood_families"]


def test_scene_detail_is_bounded_and_visual_variants_do_not_change_targets(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, sites = city_and_sites
    details = [
        (str(building["id"]), component)
        for building in city["buildings"]
        for component in building["components"]
        if component.get("structural_role") in {"roof_parapet", "entrance_canopy", "roof_equipment"}
    ]
    detail_owners = {f"{building_id}/{component['id']}" for building_id, component in details}
    assert details
    assert any(component["structural_role"] == "roof_parapet" for _, component in details)
    assert all(component.get("target_support") is False for _, component in details)
    assert all(site["owner_collider_id"] not in detail_owners for site in sites)
    assert all(
        sum(
            component.get("structural_role")
            in {"roof_parapet", "entrance_canopy", "roof_equipment"}
            for component in building["components"]
        )
        <= 10
        for building in city["buildings"]
    )
    assert city["metrics"]["architectural_detail_count"] == len(details)
    assert sum(city["metrics"]["road_surface_histogram"].values()) == len(city["roads"])
    assert len(city["task_geometry_hash"]) == 64
    assert city["visual_detail_profile"] == "procedural-urban-detail-v1"
    assert city["visual_facade_accents"]
    assert city["metrics"]["visual_facade_accent_count"] == len(
        city["visual_facade_accents"]
    )
    per_building = Counter(
        str(accent["building_id"]) for accent in city["visual_facade_accents"]
    )
    assert max(per_building.values()) <= 4
    assert all(
        accent["physics_role"] == "visual_only"
        for accent in city["visual_facade_accents"]
    )

    visual_variant = copy.deepcopy(city)
    visual_variant["layout_hash"] = "f" * 64
    visual_variant["roads"][0]["surface_style"] = "paved_local"
    visual_variant["decorations"] = []
    visual_variant["visual_facade_accents"] = []
    variant_sites = derive_support_sites_v3(visual_variant, ordinary_config)
    assert variant_sites == sites
    original_episode = sample_episode_v3(ordinary_config, city, sites, 0)
    variant_episode = sample_episode_v3(ordinary_config, visual_variant, variant_sites, 0)
    assert variant_episode["starts"] == original_episode["starts"]
    assert variant_episode["targets"] == original_episode["targets"]
    assert variant_episode["distractors"] == original_episode["distractors"]
    assert compile_coarse_prior(visual_variant) == compile_coarse_prior(city)


@pytest.mark.parametrize("split", ("train", "test_topology"))
def test_scene_details_attach_to_real_building_components(
    ordinary_config: OrdinaryReleaseConfig, split: str
) -> None:
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    city = _first_admitted_city(ordinary_config, split, 0, assets)
    for building in city["buildings"]:
        hosts = {
            str(component["id"]): component
            for component in building["components"]
            if component.get("target_support", True) is True
        }
        for component in building["components"]:
            role = component.get("structural_role")
            if role not in {"entrance_canopy", "roof_equipment"}:
                continue
            host = hosts[str(component["host_component_id"])]
            host_x, host_y, host_z = (float(value) for value in host["center"])
            host_width, host_depth, host_height = (float(value) for value in host["size"])
            detail_x, detail_y, detail_z = (float(value) for value in component["center"])
            detail_width, detail_depth, detail_height = (
                float(value) for value in component["size"]
            )
            if role == "roof_equipment":
                assert detail_x - detail_width / 2.0 >= host_x - host_width / 2.0
                assert detail_x + detail_width / 2.0 <= host_x + host_width / 2.0
                assert detail_y - detail_depth / 2.0 >= host_y - host_depth / 2.0
                assert detail_y + detail_depth / 2.0 <= host_y + host_depth / 2.0
                roof_gap = detail_z - detail_height / 2.0 - (host_z + host_height / 2.0)
                assert 0.0 <= roof_gap <= 0.02
                continue
            side = str(component["attachment_side"])
            if side in {"south", "north"}:
                assert detail_x - detail_width / 2.0 >= host_x - host_width / 2.0
                assert detail_x + detail_width / 2.0 <= host_x + host_width / 2.0
                outward = 1.0 if side == "north" else -1.0
                host_face = host_y + outward * host_depth / 2.0
                canopy_face = detail_y - outward * detail_depth / 2.0
            else:
                assert detail_y - detail_depth / 2.0 >= host_y - host_depth / 2.0
                assert detail_y + detail_depth / 2.0 <= host_y + host_depth / 2.0
                outward = 1.0 if side == "east" else -1.0
                host_face = host_x + outward * host_width / 2.0
                canopy_face = detail_x - outward * detail_width / 2.0
            wall_gap = (canopy_face - host_face) * outward
            assert 0.0 <= wall_gap <= 0.02


@pytest.mark.parametrize("split", ("train", "test_topology"))
def test_building_components_have_no_interior_collider_intersections(
    ordinary_config: OrdinaryReleaseConfig, split: str
) -> None:
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    city = _first_admitted_city(ordinary_config, split, 0, assets)
    for building in city["buildings"]:
        boxes = [
            AABB.from_center_size(
                str(component["id"]), component["center"], component["size"], "building"
            )
            for component in building["components"]
        ]
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                assert not all(
                    low_a < high_b and high_a > low_b
                    for low_a, high_a, low_b, high_b in zip(
                        box.minimum, box.maximum, other.minimum, other.maximum, strict=True
                    )
                ), f"{building['id']}: {box.collider_id} intersects {other.collider_id}"


def test_semantic_obstacles_do_not_intersect_buildings(city_and_sites) -> None:
    city, _ = city_and_sites
    buildings = [
        AABB.from_center_size(
            f"{building['id']}/{component['id']}", component["center"], component["size"]
        )
        for building in city["buildings"]
        for component in building["components"]
    ]
    for obstacle in city["obstacles"]:
        box = AABB.from_center_size(obstacle["id"], obstacle["center"], obstacle["size"])
        assert all(
            not all(
                low_a < high_b and high_a > low_b
                for low_a, high_a, low_b, high_b in zip(
                    box.minimum, box.maximum, other.minimum, other.maximum, strict=True
                )
            )
            for other in buildings
        )
        assert obstacle["semantic_anchor"].startswith("building-")


def test_sites_are_3d_contextual_and_have_legal_witnesses(city_and_sites) -> None:
    _, sites = city_and_sites
    assert len(sites) > 100
    assert len({site["altitude_band"] for site in sites}) >= 4
    assert len({round(site["position"][2], 2) for site in sites}) > 12
    assert all(site["context_collider_count"] >= 2 for site in sites)
    assert all(site["surrounding_collider_count"] >= 1 for site in sites)
    assert all(site["owner_collider_id"] in site["context_collider_ids"] for site in sites)
    assert all(site["legal_witnesses"] for site in sites)


def test_private_safe_scene_audit_covers_complete_development_episode_contract(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, _ = city_and_sites
    report = audit_generated_city(ordinary_config, city)
    assert report["status"] == "PASS"
    assert report["split"] == "train"
    assert report["scene_counts"]["episodes"] == ordinary_config.episodes("train")
    assert report["scene_counts"]["structural_details"]
    assert report["scene_counts"]["structural_details"]["roof_parapet"] > 0
    assert report["generation_rejections_before_acceptance"] == 0
    rendered = json.dumps(report, sort_keys=True)
    assert "target_count" not in rendered
    assert "targets_private_count" not in rendered
    assert "legal_support_sites_private_count" not in rendered
    assert "legal_witnesses" not in rendered
    assert '"position"' not in rendered
    assert '"site_id"' not in rendered
    assert "target layout permits" not in rendered


def test_scene_audit_rejects_formal_split_sampling(ordinary_config: OrdinaryReleaseConfig) -> None:
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    with pytest.raises(ValueError, match="only train, validation, or calibration"):
        audit_development_layout(ordinary_config, "test_iid", 0, assets)


def test_target_processes_preserve_fixed_paired_count_and_ood(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, sites = city_and_sites
    episodes = [sample_episode_v3(ordinary_config, city, sites, index) for index in range(3)]
    assert {item["target_count"] for item in episodes} == {episodes[0]["target_count"]}
    assert {item["condition_group_id"] for item in episodes} == {"process-pair-0000"}
    assert {item["target_process"] for item in episodes} == {
        "uniform_surface",
        "clustered_surface",
        "height_stratified",
    }
    assert all(len(item["targets"]) == len(item["distractors"]) for item in episodes)
    assert all(
        target["reachability_hash"]
        and all("reachability_proof" in witness for witness in target["legal_witnesses"])
        for item in episodes
        for target in item["targets"]
    )
    margin = float(ordinary_config.raw["execution_contract"]["vehicle"]["radius_m"]) + float(
        ordinary_config.raw["execution_contract"]["vehicle"]["minimum_clearance_m"]
    )
    lower = [value + margin for value in city["flight_bounds"]["minimum"]]
    upper = [value - margin for value in city["flight_bounds"]["maximum"]]
    assert all(
        all(
            low <= value <= high
            for low, value, high in zip(lower, start["position"], upper, strict=True)
        )
        for item in episodes
        for start in item["starts"]
    )


def test_paired_processes_reuse_detached_reachability_proofs(
    ordinary_config: OrdinaryReleaseConfig,
    city_and_sites,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aerocity_bench.targets_v3 as target_module

    city, sites = city_and_sites
    target_module._clear_reachability_cache_for_tests()
    original = target_module._episode_reachable_sites
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(target_module, "_episode_reachable_sites", counted)
    first = sample_episode_v3(ordinary_config, city, sites, 0)
    sample_episode_v3(ordinary_config, city, sites, 1)
    sample_episode_v3(ordinary_config, city, sites, 2)
    assert calls == 1

    first["targets"][0]["legal_witnesses"][0]["clearance_m"] = -1.0
    replay = sample_episode_v3(ordinary_config, city, sites, 0)
    assert replay["targets"][0]["legal_witnesses"][0]["clearance_m"] >= 0.0


def test_visual_review_episode_has_32_contextual_3d_targets(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, sites = city_and_sites
    review = sample_visual_review_episode_v3(
        ordinary_config, city, sites, episode["starts"], target_count=32
    )
    assert review["target_count"] == len(review["targets"]) == 32
    assert review["formal_score_eligible"] is False
    assert review["audit"]["vertical_span_m"] >= 20.0
    assert len(review["audit"]["altitude_histogram"]) >= 3
    assert review["audit"]["all_targets_have_surrounding_colliders"] is True
    assert review["audit"]["all_targets_have_legal_witnesses"] is True
    assert all("local_review_pose" in target for target in review["targets"])
    assert all("local_context_review_pose" in target for target in review["targets"])
    assert all("local_context_review_look_at" in target for target in review["targets"])
    assert review["audit"]["all_targets_have_private_l2_context_pose"] is True
    assert review["audit"]["all_private_l2_context_poses_within_vertical_flight_bounds"] is True
    assert review["audit"]["minimum_private_l2_context_camera_height_m"] >= float(
        city["flight_bounds"]["minimum"][2]
    )
    assert review["audit"]["maximum_private_l2_context_camera_height_m"] <= float(
        city["flight_bounds"]["maximum"][2]
    )
    assert review["audit"]["minimum_private_l2_context_distance_m"] > float(
        ordinary_config.raw["execution_contract"]["observe"]["max_range_m"]
    )
    assert review["episode_hash"] == content_hash(
        {key: value for key, value in review.items() if key != "episode_hash"}
    )
    ood_city = dict(city)
    ood_city["split"] = "test_process_ood"
    ood = sample_episode_v3(ordinary_config, ood_city, sites, 0)
    assert ood["target_process"] == "anisotropic_clustered_surface"


def test_l2_context_review_camera_rejects_subground_facade_candidates(
    ordinary_config: OrdinaryReleaseConfig,
) -> None:
    """A near-ground facade must not select the lower oblique camera ring."""
    import aerocity_bench.targets_v3 as target_module

    city = {
        "size_m": 80.0,
        "flight_bounds": {
            "minimum": [-40.0, -40.0, 1.0],
            "maximum": [40.0, 40.0, 70.0],
        },
        "buildings": [
            {
                "id": "building-0",
                "components": [
                    {
                        "id": "body",
                        "center": [5.0, 0.0, 2.0],
                        "size": [4.0, 4.0, 4.0],
                    }
                ],
            }
        ],
        "obstacles": [],
    }
    near_ground_facade = {
        "site_id": "site-near-ground-facade",
        "position": [2.9, 0.0, 2.25],
        "normal": [-1.0, 0.0, 0.0],
        "owner_collider_id": "building-0/body",
        "surrounding_collider_ids": [],
    }

    review = target_module._context_review_pose(city, near_ground_facade, ordinary_config)

    assert review is not None
    camera_height = float(review["pose"]["position"][2])
    assert camera_height >= float(city["flight_bounds"]["minimum"][2])
    assert camera_height <= float(city["flight_bounds"]["maximum"][2])


def test_visual_review_deterministically_resamples_covisible_draw(
    ordinary_config: OrdinaryReleaseConfig,
    city_and_sites,
    episode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aerocity_bench.targets_v3 as target_module

    city, sites = city_and_sites
    distance_calls = 0

    def fixed_candidates(
        candidates: list[dict[str, object]],
        count: int,
        _separation: float,
        _process_name: str,
        _profile: dict[str, object],
        _rng: object,
    ) -> list[dict[str, object]]:
        return candidates[:count]

    def first_draw_covisible(_first: dict[str, object], _second: dict[str, object]) -> float:
        nonlocal distance_calls
        distance_calls += 1
        return 0.0 if distance_calls <= 64 else 100.0

    monkeypatch.setattr(target_module, "_sample_targets", fixed_candidates)
    monkeypatch.setattr(target_module, "_distance", first_draw_covisible)
    reachable_sites = [{**site, "reachability_hash": "a" * 64} for site in sites]
    monkeypatch.setattr(
        target_module,
        "_episode_reachable_sites",
        lambda *_args, **_kwargs: copy.deepcopy(reachable_sites),
    )
    monkeypatch.setattr(
        target_module,
        "_context_review_pose",
        lambda _city, site, _config: {
            "pose": site["legal_witnesses"][0]["pose"],
            "target_distance_m": 4.0,
            "clearance_m": 1.0,
            "oblique_lateral_ratio": 0.5,
            "look_at": site["position"],
            "visible_context_collider_ids": [],
        },
    )
    review = sample_visual_review_episode_v3(
        ordinary_config,
        city,
        sites,
        episode["starts"],
        target_count=8,
    )
    assert review["audit"]["deterministic_sampling_attempt"] == 1
    assert review["audit"]["rejected_sampling_attempts"] == 1
    assert review["audit"]["sampling_rejection_histogram"] == {"excessive_co_visibility": 1}


def test_public_episode_projection_has_no_target_truth(episode) -> None:
    public = public_episode_projection(episode)
    text = json.dumps(public).lower()
    assert "target-000" not in text
    assert "site-" not in text
    assert "witness-" not in text
    assert public["target_count_public"] is False
    assert public["target_process_public"] is False


def _observation(
    episode_id: str, pose: Pose3D, sequence: int, timestamp: float
) -> ObservationPacket:
    return ObservationPacket(
        episode_id=episode_id,
        observation_id=f"observation-{sequence}",
        drone_id="uav-00",
        sequence=sequence,
        timestamp_s=timestamp,
        pose=pose,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=8000.0,
    )


def test_evaluator_requires_continuous_observation_dwell(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    evaluator = PrivateEvaluator(
        ordinary_config, city, episode, receipt_secret=b"evaluator-test-secret-32-bytes!!"
    )
    pose = Pose3D.from_dict(episode["targets"][0]["legal_witnesses"][0]["pose"])
    all_confirmations = []
    for sequence, timestamp in enumerate((0.0, 0.2, 0.4, 0.6)):
        observation = _observation(episode["episode_id"], pose, sequence, timestamp)
        action = ActionPacket(
            episode["episode_id"],
            "uav-00",
            sequence,
            timestamp,
            "OBSERVE",
            source_observation_id=observation.observation_id,
        )
        receipt, confirmations = evaluator.process(observation, action)
        assert receipt.accepted
        all_confirmations.extend(confirmations)
    assert len(all_confirmations) >= 1
    assert evaluator.verify_confirmation(all_confirmations[0])
    assert all_confirmations[0].anonymous_target_handle.startswith("found-")


def test_evaluator_rejects_replayed_source(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    evaluator = PrivateEvaluator(
        ordinary_config, city, episode, receipt_secret=b"evaluator-test-secret-32-bytes!!"
    )
    pose = Pose3D.from_dict(episode["targets"][0]["legal_witnesses"][0]["pose"])
    observation = _observation(episode["episode_id"], pose, 0, 0.0)
    action = ActionPacket(
        episode["episode_id"],
        "uav-00",
        0,
        0.0,
        "OBSERVE",
        source_observation_id=observation.observation_id,
    )
    assert evaluator.process(observation, action)[0].accepted
    replay_receipt, confirmations = evaluator.process(observation, action)
    assert not replay_receipt.accepted
    assert replay_receipt.reason == "source_observation_replayed"
    assert confirmations == ()


def test_evaluator_enforces_observe_session_cooldown(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    evaluator = PrivateEvaluator(
        ordinary_config, city, episode, receipt_secret=b"evaluator-test-secret-32-bytes!!"
    )
    pose = Pose3D.from_dict(episode["targets"][0]["legal_witnesses"][0]["pose"])
    first = _observation(episode["episode_id"], pose, 0, 0.0)
    first_action = ActionPacket(
        episode["episode_id"],
        "uav-00",
        0,
        0.0,
        "OBSERVE",
        source_observation_id=first.observation_id,
    )
    assert evaluator.process(first, first_action)[0].accepted
    evaluator.end_observe("uav-00", 0.2)
    second = _observation(episode["episode_id"], pose, 1, 0.4)
    second_action = ActionPacket(
        episode["episode_id"],
        "uav-00",
        1,
        0.4,
        "OBSERVE",
        source_observation_id=second.observation_id,
    )
    receipt, confirmations = evaluator.process(second, second_action)
    assert not receipt.accepted
    assert receipt.reason == "observe_cooldown_active"
    assert confirmations == ()


def test_geometry_line_of_sight_blocks_intermediate_collider() -> None:
    wall = AABB.from_center_size("wall", (1.0, 0.0, 0.0), (0.2, 2.0, 2.0))
    assert line_of_sight((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), [wall]) == (False, "wall")
    assert line_of_sight((0.0, 2.0, 0.0), (2.0, 2.0, 0.0), [wall]) == (True, None)
    assert segment_aabb_clearance((0.0, 2.0, 0.0), (2.0, 2.0, 0.0), wall) == pytest.approx(1.0)
    assert segment_segment_distance(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
    ) == pytest.approx(0.0)


def test_runtime_advances_clock_on_planning_overrun_and_never_scores_l0(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    actions = {
        drone_id: ActionPacket(
            episode["episode_id"], drone_id, packet.sequence, packet.timestamp_s, "HOVER"
        )
        for drone_id, packet in observations.items()
    }
    result = runtime.step(actions, planning_latencies_s={drone_id: 0.35 for drone_id in actions})
    assert result.task_time_s > ordinary_config.raw["execution_contract"]["control_period_s"]
    run = runtime.result()
    report = evaluate_run(run, episode, 300.0)
    assert not run["formal_score_eligible"]
    assert not report["formal_score_eligible"]
    assert report["compute"]["deadline_misses"] == 4
    assert report["coverage_diagnostics"]["coverage_2d_auc"] < 0.01
    assert report["safety"]["minimum_clearance_m"] is not None
    assert report["private_group_metrics"]["worst_group_recall"] == 0.0


def test_execution_receipt_v2_binds_actions_observations_states_and_chain(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    first_actions = {
        drone_id: ActionPacket(
            episode["episode_id"], drone_id, packet.sequence, packet.timestamp_s, "HOVER"
        )
        for drone_id, packet in observations.items()
    }
    first = runtime.step(first_actions)
    second_actions = {
        drone_id: ActionPacket(
            episode["episode_id"], drone_id, packet.sequence, packet.timestamp_s, "HOVER"
        )
        for drone_id, packet in first.observations.items()
    }
    runtime.step(second_actions)
    receipts = runtime.result()["execution_receipts"]
    for drone_id in sorted(observations):
        chain = [receipt for receipt in receipts if receipt["drone_id"] == drone_id]
        assert [receipt["action_sequence"] for receipt in chain] == [0, 1]
        assert chain[0]["schema"] == "org.aerocity.bench.execution-receipt.v3"
        assert chain[0]["previous_receipt_hash"] is None
        assert chain[1]["previous_receipt_hash"] == chain[0]["receipt_hash"]
        assert chain[1]["state_before_hash"] == chain[0]["state_after_hash"]
        assert chain[0]["action_packet_hash"] == content_hash(first_actions[drone_id].to_dict())
        assert chain[0]["source_observation_hash"] == content_hash(observations[drone_id].to_dict())


def test_formal_receipt_validator_rejects_deleted_and_rehashed_steps(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode, tmp_path: Path
) -> None:
    city, _ = city_and_sites
    start = episode["starts"][0]
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="obs-first",
        drone_id=str(start["drone_id"]),
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D.from_dict(start),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=24000.0,
    )
    action = ActionPacket(
        episode["episode_id"],
        observation.drone_id,
        0,
        0.0,
        "HOVER",
    )
    before = {
        "position": list(observation.pose.position),
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_mps": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }
    first = build_l1_execution_receipt(
        action=action,
        source_observation=observation,
        state_before=before,
        state_after=before,
        task_time_start_s=0.0,
        task_time_end_s=0.2,
        planning_latency_s=0.01,
        action_executed="HOVER",
        status="executed",
        energy_used_j=8.0,
        minimum_clearance_m=2.0,
        collision=False,
        out_of_bounds=False,
        safety_intervention=False,
        deadline_miss=False,
        previous_receipt_hash=None,
    ).to_dict()
    second_observation = ObservationPacket(
        episode_id=observation.episode_id,
        observation_id="obs-second",
        drone_id=observation.drone_id,
        sequence=1,
        timestamp_s=0.2,
        pose=observation.pose,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=23992.0,
    )
    second_action = ActionPacket(episode["episode_id"], observation.drone_id, 1, 0.2, "HOVER")
    second = build_l1_execution_receipt(
        action=second_action,
        source_observation=second_observation,
        state_before=before,
        state_after=before,
        task_time_start_s=0.2,
        task_time_end_s=0.4,
        planning_latency_s=0.02,
        action_executed="HOVER",
        status="executed",
        energy_used_j=8.0,
        minimum_clearance_m=2.0,
        collision=False,
        out_of_bounds=False,
        safety_intervention=False,
        deadline_miss=False,
        previous_receipt_hash=str(first["receipt_hash"]),
    ).to_dict()
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    checks = {name: {"status": "PASS"} for name in REQUIRED_NATIVE_CHECKS}
    report_path = tmp_path / "formal-native.json"
    input_bindings = {
        "release_config_sha256": "a" * 64,
        "task_spec_sha256": "b" * 64,
        "public_episode_sha256": "c" * 64,
        "cityspec_sha256": "d" * 64,
        "execution_contract_hash": content_hash(ordinary_config.raw["execution_contract"]),
        "layout_id": str(city["layout_id"]),
        "episode_id": str(episode["episode_id"]),
    }
    write_native_gate_report(
        report_path,
        stage_path=stage,
        execution_level="L1",
        runtime_fingerprint={"isaac_sim": "test"},
        checks=checks,
        input_bindings=input_bindings,
        formal_score_eligible=True,
        evidence_scope=FORMAL_L1_EVIDENCE_SCOPE,
    )
    evidence = validate_native_gate_report(report_path, stage, input_bindings)
    receipts = [first, second]
    context = formal_execution_context(evidence, content_hash(receipts))
    ledger = BudgetLedger(
        energy_used_j=16.0,
        planning_time_s=0.03,
        minimum_clearance_m=2.0,
    ).to_dict()
    assert_formal_receipts(
        receipts,
        context=context,
        expected_drone_ids={observation.drone_id},
        expected_confirmation_ids=set(),
        expected_task_time_s=0.4,
        ledger=ledger,
    )

    incomplete_ledger = dict(ledger)
    incomplete_ledger.pop("planning_time_s")
    with pytest.raises(ValidationError, match="omits required fields"):
        assert_formal_receipts(
            receipts,
            context=context,
            expected_drone_ids={observation.drone_id},
            expected_confirmation_ids=set(),
            expected_task_time_s=0.4,
            ledger=incomplete_ledger,
        )

    deleted = [second]
    deleted_context = formal_execution_context(evidence, content_hash(deleted))
    with pytest.raises(ValidationError, match="sequence zero"):
        assert_formal_receipts(
            deleted,
            context=deleted_context,
            expected_drone_ids={observation.drone_id},
            expected_confirmation_ids=set(),
            expected_task_time_s=0.4,
            ledger=ledger,
        )

    rehashed = [dict(first), dict(second)]
    rehashed[1]["previous_receipt_hash"] = "f" * 64
    payload = dict(rehashed[1])
    payload.pop("receipt_hash")
    rehashed[1]["receipt_hash"] = content_hash(payload)
    rehashed_context = formal_execution_context(evidence, content_hash(rehashed))
    with pytest.raises(ValidationError, match="previous hash"):
        assert_formal_receipts(
            rehashed,
            context=rehashed_context,
            expected_drone_ids={observation.drone_id},
            expected_confirmation_ids=set(),
            expected_task_time_s=0.4,
            ledger=ledger,
        )


def test_capability_native_report_cannot_promote_to_formal_context(tmp_path: Path) -> None:
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    report_path = tmp_path / "capability-native.json"
    write_native_gate_report(
        report_path,
        stage_path=stage,
        execution_level="L1",
        runtime_fingerprint={"isaac_sim": "test"},
        checks={name: {"status": "PASS"} for name in REQUIRED_NATIVE_CHECKS},
        formal_score_eligible=False,
        evidence_scope=CAPABILITY_L1_EVIDENCE_SCOPE,
    )
    evidence = validate_native_gate_report(report_path, stage)
    with pytest.raises(ValidationError, match="capability gate"):
        formal_execution_context(evidence, "a" * 64)


def test_formal_metric_requires_trusted_native_context(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode, tmp_path: Path
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    actions = {
        drone_id: ActionPacket(
            episode["episode_id"], drone_id, packet.sequence, packet.timestamp_s, "HOVER"
        )
        for drone_id, packet in observations.items()
    }
    runtime.step(actions, planning_latencies_s={drone_id: 0.01 for drone_id in actions})
    run = runtime.result()
    receipts = []
    for source in run["execution_receipts"]:
        receipt = dict(source)
        receipt["execution_level"] = "L1"
        payload = dict(receipt)
        payload.pop("receipt_hash")
        receipt["receipt_hash"] = content_hash(payload)
        receipts.append(receipt)
    receipt_set_hash = content_hash(receipts)

    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    input_bindings = {
        "release_config_sha256": "a" * 64,
        "task_spec_sha256": "b" * 64,
        "public_episode_sha256": "c" * 64,
        "cityspec_sha256": "d" * 64,
        "execution_contract_hash": content_hash(ordinary_config.raw["execution_contract"]),
        "layout_id": str(city["layout_id"]),
        "episode_id": str(episode["episode_id"]),
    }
    report_path = tmp_path / "formal-native.json"
    write_native_gate_report(
        report_path,
        stage_path=stage,
        execution_level="L1",
        runtime_fingerprint={"isaac_sim": "test", "driver": "test"},
        checks={name: {"status": "PASS"} for name in REQUIRED_NATIVE_CHECKS},
        input_bindings=input_bindings,
        formal_score_eligible=True,
        evidence_scope=FORMAL_L1_EVIDENCE_SCOPE,
    )
    evidence = validate_native_gate_report(report_path, stage, input_bindings)
    context = formal_execution_context(evidence, receipt_set_hash)
    run.update(
        {
            "execution_level": "L1",
            "formal_score_eligible": True,
            "layout_id": context.layout_id,
            "execution_receipts": receipts,
            "execution_receipt_set_hash": receipt_set_hash,
            "execution_contract_hash": context.execution_contract_hash,
            "native_gate_hash": context.native_gate_hash,
            "runtime_fingerprint_hash": context.runtime_fingerprint_hash,
        }
    )
    with pytest.raises(ValidationError, match="trusted in-memory"):
        evaluate_run(run, episode, 300.0)
    report = evaluate_run(run, episode, 300.0, formal_context=context)
    assert report["execution_level"] == "L1"
    assert report["formal_score_eligible"]

    tampered_roster = dict(run)
    tampered_roster["returned_home"] = {
        next(iter(run["returned_home"])): True,
    }
    with pytest.raises(ValidationError, match="returned-home roster"):
        evaluate_run(tampered_roster, episode, 300.0, formal_context=context)

    tampered_episode = dict(episode)
    tampered_episode["layout_id"] = "layout-not-bound"
    with pytest.raises(ValidationError, match="does not match private episode"):
        evaluate_run(run, tampered_episode, 300.0, formal_context=context)


def test_runtime_rejects_unbound_action_time_and_crossing_trajectories(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    invalid = {
        drone_id: ActionPacket(
            episode["episode_id"],
            drone_id,
            packet.sequence,
            packet.timestamp_s + (0.1 if index == 0 else 0.0),
            "HOVER",
        )
        for index, (drone_id, packet) in enumerate(observations.items())
    }
    with pytest.raises(ValueError, match="timestamp"):
        runtime.step(invalid)

    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    first_id, second_id = sorted(observations)[:2]
    first_pose = observations[first_id].pose
    second_pose = observations[second_id].pose
    actions = {}
    for drone_id, packet in observations.items():
        waypoint = None
        kind = "HOVER"
        if drone_id == first_id:
            kind, waypoint = "WAYPOINT", second_pose
        elif drone_id == second_id:
            kind, waypoint = "WAYPOINT", first_pose
        actions[drone_id] = ActionPacket(
            episode["episode_id"],
            drone_id,
            packet.sequence,
            packet.timestamp_s,
            kind,
            waypoint=waypoint,
        )
    result = runtime.step(actions)
    collided = {failure.drone_id for failure in result.failures if failure.category == "collision"}
    assert {first_id, second_id} <= collided


def test_runtime_enforces_message_bandwidth_and_duplicate_ids(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    source_id = sorted(observations)[0]
    destination = sorted(observations)[1]
    messages = tuple(
        MessagePacket(
            message_id=f"message-{index}",
            source_drone_id=source_id,
            destination_drone_ids=(destination,),
            created_at_s=0.0,
            expires_at_s=0.8,
            payload=b"x" * 1024,
        )
        for index in range(4)
    )
    actions = {
        drone_id: ActionPacket(
            episode["episode_id"],
            drone_id,
            packet.sequence,
            packet.timestamp_s,
            "HOVER",
            messages=messages if drone_id == source_id else (),
        )
        for drone_id, packet in observations.items()
    }
    result = runtime.step(actions)
    assert runtime.ledger.bandwidth_messages_rejected == 1
    assert runtime.ledger.communication_packets_sent == 3
    observations = runtime._latest_observations
    duplicate = MessagePacket(
        message_id="message-0",
        source_drone_id=source_id,
        destination_drone_ids=(destination,),
        created_at_s=runtime.task_time_s,
        expires_at_s=runtime.task_time_s + 0.8,
        payload=b"x",
    )
    actions = {
        drone_id: ActionPacket(
            episode["episode_id"],
            drone_id,
            packet.sequence,
            packet.timestamp_s,
            "HOVER",
            messages=(duplicate,) if drone_id == source_id else (),
        )
        for drone_id, packet in observations.items()
    }
    result = runtime.step(actions)
    assert runtime.ledger.duplicate_messages_rejected == 1
    received = result.observations[destination].received_messages
    assert (
        runtime.ledger.communication_packets_delivered
        + runtime.ledger.communication_packets_dropped
        - runtime.ledger.bandwidth_messages_rejected
        - runtime.ledger.duplicate_messages_rejected
        == runtime.ledger.communication_packets_sent
    )
    assert len(received) == runtime.ledger.communication_packets_delivered
    assert runtime.ledger.out_of_bounds_actions == 0


def test_planner_and_ros_codecs_bind_message_identity(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    observation = next(iter(L0FleetRuntime(ordinary_config, city, episode).reset().values()))
    destination = next(
        start["drone_id"]
        for start in episode["starts"]
        if start["drone_id"] != observation.drone_id
    )
    method_action = {
        "kind": "HOVER",
        "messages": [
            {
                "message_id": "adapter-message-1",
                "destination_drone_ids": [destination],
                "expires_at_s": observation.timestamp_s + 0.5,
                "payload_hex": "00ff",
            }
        ],
    }

    declaration = AdapterDeclaration(
        adapter_id="centralized-g1-v1",
        method_id="test-planner",
        capability_profile="G1",
        upstream_url=None,
        upstream_commit=None,
        upstream_license="BSD-3-Clause",
        process_boundary="in_process",
        training_allowed=True,
        decentralized_execution=False,
    )

    class Planner:
        def reset(self, task_spec):
            del task_spec

        def act(self, observations):
            return {drone_id: method_action for drone_id in observations}

    adapter = PlannerAdapter(declaration, Planner())
    actions, _ = adapter.act({observation.drone_id: observation})
    message = actions[observation.drone_id].messages[0]
    assert message.source_drone_id == observation.drone_id
    assert message.created_at_s == observation.timestamp_s
    assert message.payload == b"\x00\xff"

    encoded = json.dumps(method_action).encode("utf-8")
    ros_action = ROSWireCodec.action(encoded, observation)
    assert ros_action.messages[0].source_drone_id == observation.drone_id
    assert ros_action.messages[0].payload == b"\x00\xff"


def test_metrics_auc_and_right_censoring() -> None:
    assert confirmed_recall_auc([2.0, 6.0], 2, 10.0) == pytest.approx(0.6)
    reached = time_to_recall([2.0, 6.0], 4, 0.5, 10.0)
    assert reached == {"time_s": 6.0, "right_censored": False, "required_count": 2}
    censored = time_to_recall([2.0], 4, 0.5, 10.0)
    assert censored["right_censored"] and censored["time_s"] == 10.0


def test_g1_projection_excludes_private_truth(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    observation = next(iter(L0FleetRuntime(ordinary_config, city, episode).reset().values()))
    projection = project_g1(observation)
    serialized = json.dumps(projection).lower()
    assert "target_process" not in serialized
    assert "witness" not in serialized
    assert "site_id" not in serialized
    assert projection["schema"] == "org.aerocity.bench.g1-observation.v2"
    assert projection["local_occupancy_resolution_m"] > 0.0
    assert projection["local_occupancy_radius_m"] > 0.0
    assert len(projection["local_occupancy_origin_world_m"]) == 3


def test_g1_local_occupancy_is_bounded_dense_voxel_data(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    observed_nonempty = False
    for observation in observations.values():
        assert observation.local_occupancy_resolution_m == pytest.approx(2.0)
        assert observation.local_occupancy_radius_m == pytest.approx(14.0)
        assert all(len(cell) == 3 for cell in observation.local_occupancy)
        assert all(max(abs(value) for value in cell) <= 7 for cell in observation.local_occupancy)
        if observation.local_occupancy:
            observed_nonempty = True
            origin = observation.local_occupancy_origin_world_m
            resolution = observation.local_occupancy_resolution_m
            world_centers = [
                tuple(origin[axis] + cell[axis] * resolution for axis in range(3))
                for cell in observation.local_occupancy
            ]
            assert all(
                distance(observation.pose.position, center)
                <= observation.local_occupancy_radius_m + math.sqrt(3.0) * resolution / 2.0
                for center in world_centers
            )
    assert observed_nonempty


def test_adapter_license_boundary() -> None:
    declaration = AdapterDeclaration(
        adapter_id="fuel-json-v1",
        method_id="fuel-upstream",
        capability_profile="G1",
        upstream_url="https://example.invalid/fuel",
        upstream_commit="a" * 40,
        upstream_license="GPL-3.0",
        process_boundary="in_process",
        training_allowed=False,
        decentralized_execution=True,
    )
    with pytest.raises(ValueError, match="independent boundary"):
        declaration.validate()


def test_external_adapter_declaration_requires_full_revision_and_container_digest() -> None:
    invalid_revision = AdapterDeclaration(
        adapter_id="external-v1",
        method_id="external",
        capability_profile="G1",
        upstream_url="https://github.com/example/project",
        upstream_commit="a" * 12,
        upstream_license="MIT",
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=True,
    )
    with pytest.raises(ValueError, match="full 40- or 64-character"):
        invalid_revision.validate()

    missing_digest = AdapterDeclaration(
        adapter_id="external-container-v1",
        method_id="external-container",
        capability_profile="G1",
        upstream_url="https://github.com/example/project",
        upstream_commit="a" * 40,
        upstream_license="GPL-3.0-only",
        process_boundary="container",
        training_allowed=False,
        decentralized_execution=True,
    )
    with pytest.raises(ValueError, match="pinned sha256 OCI image digest"):
        missing_digest.validate()

    valid_container = dataclasses.replace(missing_digest, runtime_image_digest="sha256:" + "b" * 64)
    assert valid_container.to_dict()["runtime_image_digest"] == "sha256:" + "b" * 64

    class Planner:
        def reset(self, task_spec):
            del task_spec

        def act(self, observations):
            del observations
            return {}

    with pytest.raises(ValueError, match="real external bridge"):
        PlannerAdapter(valid_container, Planner())


def test_external_process_bridge_binds_public_requests_and_canonical_actions(
    tmp_path: Path, ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    server = tmp_path / "external_planner.py"
    server.write_text(
        """import json
import sys

REQUEST_SCHEMA = 'org.aerocity.bench.external-planner-request.v1'
RESPONSE_SCHEMA = 'org.aerocity.bench.external-planner-response.v1'

for line in sys.stdin:
    request = json.loads(line)
    assert request['schema'] == REQUEST_SCHEMA
    assert 'targets' not in str(request).lower()
    if request['kind'] == 'reset':
        def contains_target_key(value):
            if isinstance(value, dict):
                return any(
                    'target' in key.lower() or contains_target_key(nested)
                    for key, nested in value.items()
                )
            if isinstance(value, list):
                return any(contains_target_key(nested) for nested in value)
            return False
        assert not contains_target_key(request['public_episode'])
    response = {
        'schema': RESPONSE_SCHEMA,
        'request_id': request['request_id'],
        'status': 'ok',
    }
    if request['kind'] == 'act':
        response['actions'] = {
            drone_id: {'kind': 'HOVER'}
            for drone_id in request['observations']
        }
    print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    declaration = AdapterDeclaration(
        adapter_id="external-jsonl-v1",
        method_id="external-upstream",
        capability_profile="G1",
        upstream_url="https://github.com/example/external-upstream",
        upstream_commit="c" * 40,
        upstream_license="GPL-3.0-only",
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    public_episode = public_episode_projection(episode)
    public_episode["nested_audit"] = {"target_count_public": False}
    with ExternalProcessPlannerBridge(
        declaration,
        [sys.executable, "-u", str(server)],
        cwd=tmp_path,
        response_timeout_s=2.0,
    ) as bridge:
        bridge.reset(public_episode)
        actions, latencies = bridge.act(runtime.reset())
        assert set(actions) == {str(start["drone_id"]) for start in episode["starts"]}
        assert all(action.kind == "HOVER" for action in actions.values())
        assert set(latencies) == set(actions)
        assert bridge.adapter_tax_report()["call_count"] == 1
        timing = bridge.last_act_timing()
        assert timing is not None
        assert set(timing) == {
            "projection_wall_clock_s",
            "request_public_audit_wall_clock_s",
            "request_json_serialize_wall_clock_s",
            "request_size_check_wall_clock_s",
            "request_write_flush_wall_clock_s",
            "response_wait_wall_clock_s",
            "response_json_decode_wall_clock_s",
            "response_validate_wall_clock_s",
            "action_validation_conversion_wall_clock_s",
            "bridge_act_wall_clock_s",
            "bridge_internal_unattributed_wall_clock_s",
        }
        assert all(isinstance(value, float) and value >= 0.0 for value in timing.values())
        assert timing["bridge_act_wall_clock_s"] >= sum(
            value
            for field, value in timing.items()
            if field != "bridge_act_wall_clock_s"
        )


def test_external_process_bridge_rejects_private_wire_payload_and_false_boundaries(
    tmp_path: Path,
) -> None:
    declaration = AdapterDeclaration(
        adapter_id="external-process-v1",
        method_id="external",
        capability_profile="G1",
        upstream_url="https://github.com/example/external",
        upstream_commit="d" * 40,
        upstream_license="MIT",
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    server = tmp_path / "server.py"
    server.write_text("raise SystemExit(0)\n", encoding="utf-8")
    with ExternalProcessPlannerBridge(
        declaration,
        [sys.executable, "-u", str(server)],
        cwd=tmp_path,
    ) as bridge:
        with pytest.raises(ValueError, match="private fields"):
            bridge.reset({"targets": [{"position": [0, 0, 0]}]})
        with pytest.raises(ValueError, match="private fields"):
            bridge.reset({"nested": {"Target-ID": "private-target"}})
        with pytest.raises(ValueError, match="private fields"):
            bridge.reset({"evaluator-seed": "private-seed"})
        with pytest.raises(ValueError, match="non-string object key"):
            bridge.reset({1: "not-a-json-object-key"})
        with pytest.raises(ValueError, match="non-ASCII object key"):
            bridge.reset({"Ｔarget-ID": "private-target"})

    container = dataclasses.replace(
        declaration,
        process_boundary="container",
        runtime_image_digest="sha256:" + "f" * 64,
    )
    with pytest.raises(ValueError, match="only runs a real process"):
        ExternalProcessPlannerBridge(container, [sys.executable, "-u", str(server)])


def test_external_process_bridge_rejects_mismatched_response_and_timeout(
    tmp_path: Path, ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    declaration = AdapterDeclaration(
        adapter_id="external-protocol-v1",
        method_id="external-protocol",
        capability_profile="G1",
        upstream_url="https://github.com/example/external-protocol",
        upstream_commit="e" * 40,
        upstream_license="MIT",
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    mismatch_server = tmp_path / "mismatch_server.py"
    mismatch_server.write_text(
        """import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        'schema': 'org.aerocity.bench.external-planner-response.v1',
        'request_id': 'wrong-request-id',
        'status': 'ok',
    }
    print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    with ExternalProcessPlannerBridge(
        declaration,
        [sys.executable, "-u", str(mismatch_server)],
        cwd=tmp_path,
    ) as bridge:
        with pytest.raises(ValueError, match="does not bind the active request"):
            bridge.reset(public_episode_projection(episode))

    timeout_server = tmp_path / "timeout_server.py"
    timeout_server.write_text(
        """import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    if request['kind'] == 'reset':
        print(json.dumps({
            'schema': 'org.aerocity.bench.external-planner-response.v1',
            'request_id': request['request_id'],
            'status': 'ok',
        }), flush=True)
    else:
        time.sleep(5.0)
""",
        encoding="utf-8",
    )
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    with ExternalProcessPlannerBridge(
        declaration,
        [sys.executable, "-u", str(timeout_server)],
        cwd=tmp_path,
        response_timeout_s=0.1,
    ) as bridge:
        bridge.reset(public_episode_projection(episode))
        with pytest.raises(TimeoutError, match="before its deadline"):
            bridge.act(runtime.reset())
        assert bridge._process.poll() is not None


def test_adapter_cannot_claim_decentralized_execution_with_fleet_visibility() -> None:
    declaration = AdapterDeclaration(
        adapter_id="local-g1-v1",
        method_id="local-planner",
        capability_profile="G1",
        upstream_url=None,
        upstream_commit=None,
        upstream_license="BSD-3-Clause",
        process_boundary="in_process",
        training_allowed=True,
        decentralized_execution=True,
    )

    class Planner:
        def reset(self, task_spec):
            del task_spec

        def act(self, observations):
            del observations
            return {}

    with pytest.raises(ValueError, match="centralized observation"):
        PlannerAdapter(declaration, Planner())
    with pytest.raises(ValueError, match="at least one"):
        DecentralizedPlannerAdapter(declaration, {})


def test_replay_hash_chain_detects_tampering(
    tmp_path: Path, ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    runtime = L0FleetRuntime(ordinary_config, city, episode)
    observations = runtime.reset()
    actions = {
        drone_id: ActionPacket(
            episode["episode_id"], drone_id, packet.sequence, packet.timestamp_s, "HOVER"
        )
        for drone_id, packet in observations.items()
    }
    result = runtime.step(actions)
    path = tmp_path / "replay.json"
    writer = ReplayWriter(path, {"execution_level": "L0"})
    writer.append(observations, actions, result)
    writer.close()
    assert validate_replay(path)["status"] == "PASS"
    replay = json.loads(path.read_text(encoding="utf-8"))
    replay["records"][0]["task_time_s"] += 1.0
    path.write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(ValueError, match="replay content hash"):
        validate_replay(path)


def test_pettingzoo_wrapper_surface(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    wrapper = PettingZooParallelWrapper(L0FleetRuntime(ordinary_config, city, episode))
    observations, infos = wrapper.reset()
    assert set(observations) == set(wrapper.possible_agents)
    assert set(infos) == set(wrapper.possible_agents)


def test_baseline_registry_separates_oracle_and_substantive_methods(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    public = public_episode_projection(episode)
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    assert BASELINES["centralized-oracle"].role == "diagnostic"
    assert BASELINES["nearest-frontier"].substantive_method
    with pytest.raises(ValueError, match="private diagnostic"):
        create_baseline("centralized-oracle", ordinary_config, task_spec, public)


def test_centralized_oracle_follows_safe_routes_and_confirms(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    public = public_episode_projection(episode)
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "centralized-oracle",
        ordinary_config,
        task_spec,
        public,
        private_episode=episode,
    )
    result = L0FleetRuntime(ordinary_config, city, episode).run_policy(policy, max_steps=450)
    assert result["confirmations"]
    assert result["budget_ledger"]["clearance_interventions"] == 0
    assert result["evaluator_private_audit"]["observation_count"] >= 4
    assert result["evaluator_private_audit"]["visibility_diagnostics"].get("moving", 0) == 0


@pytest.mark.parametrize(
    "method_id",
    (
        "random-safe",
        "sweep-2d",
        "sweep-3d",
        "nearest-frontier",
        "information-frontier",
        "decentralized-auction",
    ),
)
def test_public_baselines_do_not_immediately_self_collide_or_leave_bounds(
    method_id: str,
    ordinary_config: OrdinaryReleaseConfig,
    city_and_sites,
    episode,
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public_episode = public_episode_projection(episode)
    policy = create_baseline(method_id, ordinary_config, task_spec, public_episode)
    result = L0FleetRuntime(ordinary_config, city, episode).run_policy(policy, max_steps=20)
    assert result["budget_ledger"]["collisions"] == 0, result["failure_records"]
    assert result["budget_ledger"]["out_of_bounds_actions"] == 0


def test_public_surface_scan_routes_are_finite_3d_and_target_private(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public = public_episode_projection(episode)
    policy = create_baseline("information-frontier", ordinary_config, task_spec, public)
    assert not BASELINES["information-frontier"].requires_private_truth
    assert all(route for route in policy.routes.values())
    assert all(
        policy.observe_indices[drone_id] and max(policy.observe_indices[drone_id]) < len(route)
        for drone_id, route in policy.routes.items()
    )
    observation_altitudes = {
        round(route[index].position[2], 2)
        for drone_id, route in policy.routes.items()
        for index in policy.observe_indices[drone_id]
    }
    assert len(observation_altitudes) >= 3
    assert all(policy.indices[drone_id] == 0 for drone_id in policy.routes)
    assert not any(target["target_id"] in repr(policy.routes) for target in episode["targets"])


def test_g2_i_atlas_inspector_consumes_only_public_cells_and_fits_l0_bracket(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    """The first G2-I method must be executable before any long RL training."""

    city, _ = city_and_sites
    task_spec = compile_g2_i_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public = public_episode_projection(episode)
    policy = create_baseline("atlas-surface-inspector", ordinary_config, task_spec, public)
    assert BASELINES["atlas-surface-inspector"].observation_profile == "G2-I"
    assert all(route for route in policy.routes.values())
    assert sum(len(indices) for indices in policy.observe_indices.values()) > len(public["starts"])
    assert all(len(indices) >= 1 for indices in policy.observe_indices.values())
    selection = policy.public_selection_contract
    assert selection["target_independent"] is True
    assert selection["maximum_regions_per_drone"] == 1
    assert selection["maximum_evenly_spaced_cells_per_region"] > 1
    assert selection["selected_observe_pose_count"] == sum(
        len(indices) for indices in policy.observe_indices.values()
    )
    assert all(
        item["status"] == "LOWER_BOUND_FITS"
        for item in policy.route_budget_audit(horizontal_speed_mps=1.5, vertical_speed_mps=1.0)[
            "by_drone"
        ].values()
    )
    route_text = repr(policy.routes)
    assert not any(target["target_id"] in route_text for target in episode["targets"])
    result = L0FleetRuntime(
        ordinary_config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public,
    ).run_policy(policy)
    rejected_receipts = [
        receipt
        for receipt in result["execution_receipts"]
        if receipt["collision"] or receipt["out_of_bounds"] or receipt["safety_intervention"]
    ]
    assert result["budget_ledger"]["collisions"] == 0, [
        {
            key: receipt[key]
            for key in (
                "drone_id",
                "action_sequence",
                "action_requested",
                "action_executed",
                "status",
                "collision",
                "out_of_bounds",
                "safety_intervention",
                "minimum_clearance_m",
                "source_observation_id",
            )
        }
        for receipt in rejected_receipts
    ]
    assert result["budget_ledger"]["out_of_bounds_actions"] == 0
    assert result["coverage_denominators"]["inspection_atlas_cells"] > 0
    # One sensor-valid OBSERVE may cover multiple neighboring surface cells.
    # Credit is therefore bounded by the public denominator, not by the number
    # of drones or OBSERVE actions; the risk-gate tests separately falsify
    # wrong-yaw, wrong-pitch, blocked-LOS, and insufficient-dwell credit.
    assert (
        0
        < result["inspection_cell_count_trace"][-1][1]
        <= result["coverage_denominators"]["inspection_atlas_cells"]
    )
    assert (
        0
        < result["inspection_coverage_trace"][-1][1]
        <= result["coverage_denominators"]["inspection_atlas_area_m2"]
    )
    assert all(result["returned_home"].values())
    metrics = evaluate_run(
        result,
        episode,
        float(ordinary_config.raw["execution_contract"]["episode"]["duration_s"]),
    )
    assert metrics["coverage_diagnostics"]["inspection_footprint_auc"] is not None
    assert metrics["coverage_diagnostics"]["inspection_footprint_final"] > 0.0


def test_g2_i_region_greedy_derives_capacity_from_public_route_budget(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_g2_i_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public = public_episode_projection(episode)
    policy = create_baseline("atlas-region-greedy", ordinary_config, task_spec, public)
    contract = policy.public_selection_contract
    assert contract is not None
    assert contract["target_independent"] is True
    assert contract["selection_objective"] == (
        "maximize_public_region_breadth_then_observe_cells_under_budget"
    )
    assert (
        1
        <= contract["maximum_regions_per_drone"]
        <= math.ceil(contract["candidate_region_count"] / ordinary_config.fleet_count)
    )
    assert contract["maximum_evenly_spaced_cells_per_region"] >= 1
    assert contract["selected_observe_pose_count"] == sum(
        len(indices) for indices in policy.observe_indices.values()
    )
    assert contract["selected_observe_pose_count"] > len(public["starts"])
    assert contract["route_budget_audit"]["status"] == "LOWER_BOUND_FITS"
    assert all(
        item["status"] == "LOWER_BOUND_FITS"
        for item in contract["route_budget_audit"]["by_drone"].values()
    )
    route_text = repr(policy.routes)
    assert not any(target["target_id"] in route_text for target in episode["targets"])


def test_public_sweep_route_has_an_explicit_budget_audit(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    assert task_spec["public_transit_contract"]["height_envelope_basis"] == (
        "aggregate_city_collider_ceiling_plus_vehicle_margin"
    )
    safe_sky = float(task_spec["public_transit_contract"]["safe_sky_altitude_m"])
    vehicle = ordinary_config.raw["execution_contract"]["vehicle"]
    assert safe_sky <= float(city["flight_bounds"]["maximum"][2]) - (
        float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    audit = policy.route_budget_audit(horizontal_speed_mps=1.5, vertical_speed_mps=1.0)
    assert audit["status"] == "LOWER_BOUND_FITS"
    assert set(audit["by_drone"]) == set(policy.routes)
    assert all(
        item["total_required_lower_bound_s"] <= audit["episode_duration_s"]
        for item in audit["by_drone"].values()
    )
    # This is intentionally a sparse coverage diagnostic, not an exhaustive
    # facade sweep.  One G1-screened scan pose per drone leaves execution
    # margin for the candidate CF2X controller inside the frozen 300 s task.
    assert all(item["observe_pose_count"] == 1 for item in audit["by_drone"].values())
    assert all(
        min(policy.observe_indices[drone_id]) > 0
        and policy.routes[drone_id][min(policy.observe_indices[drone_id]) - 1].position
        == policy.routes[drone_id][min(policy.observe_indices[drone_id])].position
        and min(policy.observe_indices[drone_id]) - 1 not in policy.observe_indices[drone_id]
        for drone_id in policy.routes
    )
    vehicle_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    starts = {
        str(item["drone_id"]): tuple(float(value) for value in item["position"])
        for item in public_episode_projection(episode)["starts"]
    }
    assert all(
        _route_respects_public_prior(
            policy.routes[drone_id],
            start=starts[drone_id],
            home=policy.homes[drone_id].position,
            task_spec=task_spec,
            body_margin_m=vehicle_margin,
        )
        for drone_id in policy.routes
    )

    infeasible = policy.route_budget_audit(horizontal_speed_mps=0.01, vertical_speed_mps=0.01)
    assert infeasible["status"] == "BUDGET_INFEASIBLE"
    assert all(item["status"] == "BUDGET_INFEASIBLE" for item in infeasible["by_drone"].values())

    no_lane_city = copy.deepcopy(city)
    no_lane_city["buildings"][0]["components"][0]["center"][2] = 69.0
    with pytest.raises(ValueError, match="safe-sky transit lane"):
        compile_method_task_spec(
            no_lane_city,
            ordinary_config.raw["execution_contract"],
            ordinary_config.raw["fleet"],
        )


def test_route_budget_motion_lower_bound_allows_concurrent_horizontal_and_vertical_motion() -> None:
    # The audit rejects only routes that cannot fit even with simultaneous
    # horizontal and vertical references. Summing these terms would overstate
    # the lower bound and unfairly reject an otherwise compatible method.
    assert _anisotropic_motion_lower_bound_s(
        ((0.0, 0.0, 0.0), (6.0, 8.0, 10.0)),
        horizontal_speed_mps=1.0,
        vertical_speed_mps=1.0,
    ) == pytest.approx(10.0)


def test_public_route_does_not_skip_an_unreached_waypoint_after_short_hover(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    """Deadline hovers are executor evidence, not permission to skip a route leg."""

    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    observations = L0FleetRuntime(ordinary_config, city, episode).reset()
    initial_indices = dict(policy.indices)
    first_actions = policy(observations)
    for _ in range(5):
        repeated_actions = policy(observations)
        assert {drone_id: action.waypoint for drone_id, action in repeated_actions.items()} == {
            drone_id: action.waypoint for drone_id, action in first_actions.items()
        }
    assert policy.indices == initial_indices


def test_local_occupancy_cannot_skip_a_distant_transit_waypoint(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="distant-transit",
        drone_id="uav-00",
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D((0.0, 0.0, 20.0), 0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=((0, 0, 0),),
        local_occupancy_origin_world_m=(0.0, 0.0, 20.0),
        local_occupancy_resolution_m=2.0,
        local_occupancy_radius_m=14.0,
    )
    assert not policy._local_occupancy_blocks_scan(  # noqa: SLF001
        observation, Pose3D((0.0, 0.0, 2.5), 0.0)
    )


def test_opportunity_audit_separates_full_plan_from_executed_prefix(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public = public_episode_projection(episode)
    policy = create_baseline("sweep-3d", ordinary_config, task_spec, public)
    assert sum(len(indices) for indices in policy.observe_indices.values()) > 0

    observations, run_result = _execute_policy_with_trace(
        policy,
        config=ordinary_config,
        city=city,
        private_episode=episode,
        public_task_spec=task_spec,
        public_episode=public,
        max_steps=1,
    )
    evaluator = PrivateEvaluator(
        ordinary_config,
        city,
        episode,
        receipt_secret=b"audit-regression-test-secret",
    )
    summary = _visibility_summary(
        observations,
        evaluator=evaluator,
        private_episode=episode,
    )
    assert observations == []
    assert summary["observation_count"] == 0
    assert summary["visible_target_count"] == 0
    assert run_result["task_time_s"] >= float(
        ordinary_config.raw["execution_contract"]["control_period_s"]
    )


def test_opportunity_audit_requires_explicit_public_method_and_step_cap(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    """Development opportunity diagnostics must not silently scan every method."""

    with pytest.raises(SystemExit):
        opportunity_audit_arguments(
            [
                "authority-root",
                "--split",
                "calibration",
                "--method",
                "centralized-oracle",
                "--max-steps",
                "10",
                "--output",
                "audit.json",
            ]
        )
    with pytest.raises(SystemExit):
        opportunity_audit_arguments(
            [
                "authority-root",
                "--split",
                "calibration",
                "--method",
                "sweep-3d",
                "--max-steps",
                "0",
                "--output",
                "audit.json",
            ]
        )
    arguments = opportunity_audit_arguments(
        [
            "authority-root",
            "--split",
            "calibration",
            "--method",
            "sweep-3d",
            "--max-steps",
            "10",
            "--output",
            "audit.json",
        ]
    )
    assert arguments.method == ["sweep-3d"]
    assert arguments.max_steps == 10

    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    with pytest.raises(ValueError, match="max_steps must be positive"):
        audit_method(
            "sweep-3d",
            config=ordinary_config,
            task_spec=task_spec,
            public_episode=public_episode_projection(episode),
            private_episode=episode,
            city=city,
            max_steps=0,
        )


def test_derived_opportunity_inputs_are_receipt_bound_and_nonformal(tmp_path: Path) -> None:
    """Only a deterministic public-task recompile may use the local compatibility path."""

    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("calibration",),
        source_commit="d" * 40,
    )
    index = json.loads((authority / "release_index.json").read_text(encoding="utf-8"))
    layout = index["layouts"][0]
    relative = Path("splits") / "calibration" / layout["layout_id"]
    source_layout = authority / relative
    derived = tmp_path / "derived"
    derived_layout = derived / relative
    shutil.copytree(source_layout, derived_layout)

    source_task = source_layout / "method_public" / "task_spec.json"
    derived_task = derived_layout / "method_public" / "task_spec.json"
    source_city = source_layout / "scene_authority" / "cityspec.json"
    derived_city = derived_layout / "scene_authority" / "cityspec.json"
    source_private = source_layout / "evaluator_private" / "episodes" / "episode-0000.json"
    derived_private = derived_layout / "evaluator_private" / "episodes" / "episode-0000.json"
    receipt_payload = {
        "schema": "org.aerocity.bench.cf2x-v4-derived-input.v1",
        "source_release_root": str(authority.resolve()),
        "source_layout_relative_path": relative.as_posix(),
        "derived_layout_relative_path": relative.as_posix(),
        "release_config_sha256": file_hash(authority / "authority_private" / "release_config.json"),
        "cityspec_sha256_before_and_after": file_hash(source_city),
        "private_episode_0000_sha256_before_and_after": file_hash(source_private),
        "source_task_spec_sha256": file_hash(source_task),
        "derived_task_spec_sha256": file_hash(derived_task),
        "formal_score_eligible": False,
    }
    receipt = {**receipt_payload, "receipt_hash": content_hash(receipt_payload)}
    write_json(derived / "DERIVATION_RECEIPT.json", receipt)

    loaded = _load_derived_development_inputs(
        authority,
        layout=layout,
        split="calibration",
        episode_index=0,
        derived_root=derived,
    )
    assert loaded[1]["public_transit_contract"]
    assert loaded[-1]["derivation_receipt_hash"] == receipt["receipt_hash"]
    assert file_hash(derived_city) == file_hash(source_city)
    assert file_hash(derived_private) == file_hash(source_private)

    derived_task.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="derived task-spec hash mismatch"):
        _load_derived_development_inputs(
            authority,
            layout=layout,
            split="calibration",
            episode_index=0,
            derived_root=derived,
        )


def test_public_scan_refinement_uses_outermost_occupancy_boundary(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
    )
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="occupancy-refinement",
        drone_id="uav-00",
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D((-3.0, 0.0, 0.0), 0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=((0, 0, 0), (1, 0, 0)),
        local_occupancy_origin_world_m=(0.0, 0.0, 0.0),
        local_occupancy_resolution_m=2.0,
        local_occupancy_radius_m=14.0,
    )
    refined = policy._refine_scan_pose(  # noqa: SLF001 - reference-policy contract test
        observation,
        Pose3D((-2.0, 0.0, 0.0), 0.0),
    )
    body_margin = float(ordinary_config.raw["execution_contract"]["vehicle"]["radius_m"]) + float(
        ordinary_config.raw["execution_contract"]["vehicle"]["minimum_clearance_m"]
    )
    assert abs(refined.position[0] - -2.5) < 1.0e-9
    assert (
        min(
            box.point_distance(refined.position)
            for box in policy._local_occupancy_boxes(observation)
        )
        > body_margin
    )

    drone_id = next(iter(policy.routes))
    route_index = min(policy.observe_indices[drone_id])
    route_pose = policy.routes[drone_id][route_index]
    policy.indices[drone_id] = route_index
    route_observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="single-refinement",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=0.0,
        pose=route_pose,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )
    policy._next_pose(drone_id, route_observation)  # noqa: SLF001
    first_route_pose = policy.routes[drone_id][route_index]
    policy._next_pose(drone_id, route_observation)  # noqa: SLF001
    assert policy.refined_scan_indices[drone_id] == {route_index}
    assert policy.routes[drone_id][route_index] == first_route_pose


def test_expanded_aabb_broad_phase_never_rejects_an_exact_margin_violation() -> None:
    boxes = (
        AABB("origin", (-1.0, -2.0, -0.5), (1.0, 2.0, 0.5)),
        AABB("offset", (3.0, -4.0, 2.0), (5.0, -1.0, 6.0)),
    )
    coordinates = (-7.0, -3.0, -0.25, 0.0, 2.5, 6.0, 9.0)
    endpoints = [
        (x_value, y_value, z_value)
        for x_value in coordinates
        for y_value in coordinates[::2]
        for z_value in coordinates[1::2]
    ]
    for box in boxes:
        for start, end in zip(endpoints, reversed(endpoints), strict=True):
            exact = segment_aabb_clearance(start, end, box)
            for margin in (0.0, 0.1, 0.75, 2.0):
                if exact <= margin + 1.0e-9:
                    assert segment_intersects_expanded_aabb(start, end, box, margin)
    with pytest.raises(ValueError, match="finite and non-negative"):
        segment_intersects_expanded_aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), boxes[0], -0.1)


def test_public_scan_refinement_broad_phase_limits_exact_voxel_checks(
    ordinary_config: OrdinaryReleaseConfig,
    city_and_sites,
    episode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
    )
    irrelevant = tuple(
        (x_index, y_index, z_index)
        for x_index in range(-8, 9)
        for y_index in range(10, 17)
        for z_index in range(-4, 5)
    )
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="broad-phase-work-bound",
        drone_id="uav-00",
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D((-3.0, 0.0, 0.0), 0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=((0, 0, 0), *irrelevant),
        local_occupancy_origin_world_m=(0.0, 0.0, 0.0),
        local_occupancy_resolution_m=0.5,
        local_occupancy_radius_m=14.0,
    )
    exact_call_count = 0
    original = baseline_module.segment_aabb_clearance

    def counted_exact_clearance(start, end, box):
        nonlocal exact_call_count
        exact_call_count += 1
        return original(start, end, box)

    monkeypatch.setattr(baseline_module, "segment_aabb_clearance", counted_exact_clearance)
    refined = policy._refine_scan_pose(  # noqa: SLF001 - reference-policy work-bound test
        observation,
        Pose3D((-2.0, 0.0, 0.0), 0.0),
    )
    assert refined.position[0] <= -1.5
    assert len(observation.local_occupancy) > 1000
    assert exact_call_count < 50


def test_public_scan_refinement_waits_for_local_occupancy_visibility(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    """A public scan cell is refined once, only after it enters local sensing."""

    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
    )
    drone_id = next(iter(policy.routes))
    route_index = min(policy.observe_indices[drone_id])
    target = policy.routes[drone_id][route_index]
    policy.indices[drone_id] = route_index

    distant = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="distant-public-cell",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D(
            (target.position[0] - 20.0, target.position[1], target.position[2]),
            target.yaw_deg,
            target.pitch_deg,
        ),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=((0, 0, 0),),
        local_occupancy_origin_world_m=(
            target.position[0] - 20.0,
            target.position[1],
            target.position[2],
        ),
        local_occupancy_resolution_m=2.0,
        local_occupancy_radius_m=14.0,
    )
    assert policy._next_pose(drone_id, distant) == target  # noqa: SLF001
    assert policy.refined_scan_indices[drone_id] == set()

    nearby = dataclasses.replace(
        distant,
        observation_id="nearby-public-cell",
        sequence=1,
        pose=target,
        local_occupancy=(),
        local_occupancy_origin_world_m=target.position,
    )
    policy._next_pose(drone_id, nearby)  # noqa: SLF001 - reference-policy contract test
    first_refined_pose = policy.routes[drone_id][route_index]
    policy._next_pose(drone_id, nearby)  # noqa: SLF001 - must not re-run refinement
    assert policy.refined_scan_indices[drone_id] == {route_index}
    assert policy.routes[drone_id][route_index] == first_refined_pose


def test_public_scan_refinement_relocates_around_side_obstacle(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
    )
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="side-obstacle-refinement",
        drone_id="uav-00",
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D((-3.0, 0.0, 5.0), 0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=((0, 0, 1), (-1, 1, 1)),
        local_occupancy_origin_world_m=(0.0, 0.0, 0.0),
        local_occupancy_resolution_m=2.0,
        local_occupancy_radius_m=14.0,
    )
    base = Pose3D((-2.0, 0.0, 2.2), 0.0)
    refined = policy._refine_scan_pose(  # noqa: SLF001 - reference-policy contract test
        observation,
        base,
    )
    occupied = policy._local_occupancy_boxes(observation)  # noqa: SLF001
    body_margin = float(ordinary_config.raw["execution_contract"]["vehicle"]["radius_m"]) + float(
        ordinary_config.raw["execution_contract"]["vehicle"]["minimum_clearance_m"]
    )
    assert refined != base
    assert min(box.point_distance(refined.position) for box in occupied) > body_margin
    segment_clearance, _ = minimum_segment_clearance(
        observation.pose.position,
        refined.position,
        occupied,
    )
    assert segment_clearance > body_margin


def test_public_route_policy_enters_return_state_before_deadline(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    public = public_episode_projection(episode)
    policy = create_baseline("sweep-3d", ordinary_config, task_spec, public)
    observations = L0FleetRuntime(ordinary_config, city, episode).reset()
    duration = float(ordinary_config.raw["execution_contract"]["episode"]["duration_s"])
    late_observations = {
        drone_id: ObservationPacket(
            episode_id=packet.episode_id,
            observation_id=f"late-{packet.observation_id}",
            drone_id=packet.drone_id,
            sequence=packet.sequence,
            timestamp_s=duration - 20.0,
            pose=packet.pose,
            linear_velocity_world_mps=packet.linear_velocity_world_mps,
            angular_speed_deg_s=packet.angular_speed_deg_s,
            energy_remaining_j=packet.energy_remaining_j,
            local_occupancy=packet.local_occupancy,
            local_occupancy_origin_world_m=packet.local_occupancy_origin_world_m,
            local_occupancy_resolution_m=packet.local_occupancy_resolution_m,
            local_occupancy_radius_m=packet.local_occupancy_radius_m,
            teammate_states=packet.teammate_states,
            received_messages=packet.received_messages,
            health=packet.health,
        )
        for drone_id, packet in observations.items()
    }
    actions = policy(late_observations)
    assert all(policy.return_phases[drone_id] != "search" for drone_id in actions)
    assert all(action.kind in {"WAYPOINT", "RETURN", "HOVER"} for action in actions.values())
    assert all(action.kind != "OBSERVE" for action in actions.values())


def test_public_return_trigger_uses_the_audited_execution_speed_envelope(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    drone_id = max(policy.routes, key=lambda item: len(policy.routes[item]))
    home = policy.homes[drone_id]
    current = policy.routes[drone_id][-1]
    transit_z = policy.transit_altitudes[drone_id]
    horizontal_distance = math.hypot(
        current.position[0] - home.position[0], current.position[1] - home.position[1]
    )
    vertical_distance = abs(transit_z - current.position[2]) + abs(transit_z - home.position[2])
    vehicle = ordinary_config.raw["execution_contract"]["vehicle"]
    reserve = float(ordinary_config.raw["execution_contract"]["episode"]["return_reserve_s"])

    def trigger_threshold(horizontal_speed: float, vertical_speed: float) -> float:
        motion = vertical_distance / vertical_speed + horizontal_distance / horizontal_speed + 3.0
        return max(reserve + 12.0, motion * 1.25 + 8.0)

    release_threshold = trigger_threshold(
        float(vehicle["horizontal_speed_mps"]), float(vehicle["vertical_speed_mps"])
    )
    audited_threshold = trigger_threshold(1.5, 1.0)
    assert audited_threshold > release_threshold
    remaining = (release_threshold + audited_threshold) / 2.0
    duration = float(ordinary_config.raw["execution_contract"]["episode"]["duration_s"])
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="audited-return-envelope",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=duration - remaining,
        pose=current,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )
    assert policy._return_triggered(drone_id, observation) is False  # noqa: SLF001
    policy.route_budget_audit(horizontal_speed_mps=1.5, vertical_speed_mps=1.0)
    assert policy._return_triggered(drone_id, observation) is True  # noqa: SLF001


def test_public_return_retreat_rejects_locally_blocked_segments(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    drone_id = sorted(policy.routes)[0]
    start = policy.homes[drone_id]
    occupied_ring = tuple(
        (x_index, y_index, 0)
        for x_index, y_index in (
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 1),
        )
    )
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="blocked-retreat",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=280.0,
        pose=start,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
        local_occupancy=occupied_ring,
        local_occupancy_origin_world_m=(
            start.position[0] - 2.0,
            start.position[1] - 2.0,
            start.position[2],
        ),
        local_occupancy_resolution_m=1.0,
        local_occupancy_radius_m=5.0,
    )
    retreat = policy._best_retreat_pose(observation)  # noqa: SLF001
    assert retreat.position == start.position
    action = policy._return_action(drone_id, observation)  # noqa: SLF001
    assert policy.return_phases[drone_id] == "ascend"
    assert action.kind == "WAYPOINT"
    assert action.waypoint is not None
    assert action.waypoint.position[2] == policy.transit_altitudes[drone_id]


def test_public_return_retreat_does_not_freeze_controller_derived_pitch(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_g2_i_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "atlas-surface-inspector",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
    )
    drone_id = sorted(policy.routes)[1]
    start = policy.homes[drone_id]
    moving = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="return-retreat-moving-pitch",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=167.6,
        pose=Pose3D(
            start.position,
            yaw_deg=-161.89547,
            pitch_deg=-2.94583,
        ),
        linear_velocity_world_mps=(0.3, 0.4, 0.1),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )

    retreat_action = policy._return_action(drone_id, moving)  # noqa: SLF001
    assert retreat_action.kind == "WAYPOINT"
    assert retreat_action.waypoint is not None
    assert retreat_action.waypoint.pitch_deg == 0.0
    assert policy.return_phases[drone_id] == "retreat"

    arrived = dataclasses.replace(
        moving,
        observation_id="return-retreat-settled-level",
        sequence=1,
        timestamp_s=174.0,
        pose=Pose3D(
            retreat_action.waypoint.position,
            yaw_deg=retreat_action.waypoint.yaw_deg,
            pitch_deg=0.0,
        ),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
    )
    ascend_action = policy._return_action(drone_id, arrived)  # noqa: SLF001

    assert policy.return_phases[drone_id] == "ascend"
    assert ascend_action.kind == "WAYPOINT"
    assert ascend_action.waypoint is not None
    assert ascend_action.waypoint.position[2] == policy.transit_altitudes[drone_id]


def test_public_return_retreat_scores_each_segment_once(
    monkeypatch: pytest.MonkeyPatch,
    ordinary_config: OrdinaryReleaseConfig,
    city_and_sites,
    episode,
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    drone_id = sorted(policy.routes)[0]
    start = policy.homes[drone_id]
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="retreat-clearance-call-count",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=280.0,
        pose=start,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )
    calls = 0
    original = baseline_module.minimum_segment_clearance

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(baseline_module, "minimum_segment_clearance", counted)
    policy._best_retreat_pose(observation)  # noqa: SLF001
    assert calls == 8


def test_reference_policy_yields_on_crossing_public_fleet_segments(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    observations = {
        drone_id: ObservationPacket(
            episode_id=episode["episode_id"],
            observation_id=f"crossing-{drone_id}",
            drone_id=drone_id,
            sequence=0,
            timestamp_s=0.0,
            pose=Pose3D(position, 0.0),
            linear_velocity_world_mps=(0.0, 0.0, 0.0),
            angular_speed_deg_s=0.0,
            energy_remaining_j=1000.0,
        )
        for drone_id, position in {
            "uav-00": (0.0, 0.0, 0.0),
            "uav-01": (1.0, 0.0, 0.0),
        }.items()
    }
    actions = {
        "uav-00": ActionPacket(
            episode["episode_id"],
            "uav-00",
            0,
            0.0,
            "WAYPOINT",
            waypoint=Pose3D((1.0, 1.0, 0.0), 0.0),
        ),
        "uav-01": ActionPacket(
            episode["episode_id"],
            "uav-01",
            0,
            0.0,
            "WAYPOINT",
            waypoint=Pose3D((0.0, 1.0, 0.0), 0.0),
        ),
    }
    arbitrated = policy._arbitrate_teammate_trajectories(  # noqa: SLF001
        actions, observations
    )
    assert arbitrated["uav-00"].kind == "WAYPOINT"
    assert arbitrated["uav-01"].kind == "HOVER"


def test_public_transit_holds_measured_attitude_until_scan_pose(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    drone_id = sorted(policy.routes)[0]
    observation = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="transit-attitude",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D(
            tuple(
                float(value)
                for value in public_episode_projection(episode)["starts"][0]["position"]
            ),
            37.0,
            4.0,
            -2.0,
        ),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )
    action = policy({drone_id: observation})[drone_id]
    assert action.kind == "WAYPOINT"
    assert action.waypoint is not None
    assert action.waypoint.yaw_deg == pytest.approx(37.0)
    assert action.waypoint.pitch_deg == pytest.approx(4.0)
    assert action.waypoint.roll_deg == pytest.approx(-2.0)


def test_public_return_descent_binds_an_explicit_home_waypoint(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites, episode
) -> None:
    city, _ = city_and_sites
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "sweep-3d", ordinary_config, task_spec, public_episode_projection(episode)
    )
    drone_id = sorted(policy.routes)[0]
    home = policy.homes[drone_id]
    transit_z = policy.transit_altitudes[drone_id]
    policy.return_phases[drone_id] = "descend"
    descending = ObservationPacket(
        episode_id=episode["episode_id"],
        observation_id="return-descent",
        drone_id=drone_id,
        sequence=0,
        timestamp_s=0.0,
        pose=Pose3D((home.position[0], home.position[1], transit_z), home.yaw_deg),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1000.0,
    )
    action = policy._return_action(drone_id, descending)  # noqa: SLF001 - return ABI test
    assert action.kind == "RETURN"
    assert action.waypoint == home

    landed = dataclasses.replace(
        descending,
        observation_id="return-home",
        sequence=1,
        pose=home,
    )
    assert policy._return_action(drone_id, landed).kind == "HOVER"  # noqa: SLF001
    assert policy.return_phases[drone_id] == "home"


def test_blind_submission_requires_digest_and_private_free_mount(tmp_path: Path) -> None:
    declaration = {
        "schema": "org.aerocity.bench.adapter-declaration.v1",
        "adapter_id": "test",
    }
    path = tmp_path / "adapter.json"
    write_json(path, declaration)
    spec = submission_spec(
        team_id="team-a",
        submission_id="submission-1",
        image=f"registry.example/aerocity/method@sha256:{'a' * 64}",
        adapter_declaration_path=path,
    )
    mounted = {item["source_role"] for item in spec["sandbox"]["mounts"]}
    assert "evaluator_private" not in mounted
    with pytest.raises(ValueError, match="immutable"):
        submission_spec(
            team_id="team-a",
            submission_id="submission-2",
            image="method:latest",
            adapter_declaration_path=path,
        )


def test_native_gate_requires_all_pass_and_stage_hash(tmp_path: Path) -> None:
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    checks = {name: {"status": "PASS", "evidence": "unit-test"} for name in REQUIRED_NATIVE_CHECKS}
    report_path = tmp_path / "native.json"
    write_native_gate_report(
        report_path,
        stage_path=stage,
        execution_level="L1",
        runtime_fingerprint={"isaac_sim": "test", "driver": "test"},
        checks=checks,
    )
    assert validate_native_gate_report(report_path, stage).execution_level == "L1"
    with pytest.raises(ValidationError, match="different public inputs"):
        validate_native_gate_report(
            report_path,
            stage,
            {"layout_id": "different-layout"},
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checks"]["ray_los_agreement"]["status"] = "FAIL"
    report_without_hash = {key: value for key, value in report.items() if key != "native_gate_hash"}
    report["native_gate_hash"] = content_hash(report_without_hash)
    write_json(report_path, report)
    with pytest.raises(ValidationError, match="failed checks"):
        validate_native_gate_report(report_path, stage)

    preflight_path = tmp_path / "preflight.json"
    write_native_gate_report(
        preflight_path,
        stage_path=stage,
        execution_level="L1-preflight",
        runtime_fingerprint={"isaac_sim": "test", "driver": "test"},
        checks={name: {"status": "PASS"} for name in REQUIRED_NATIVE_CHECKS},
    )
    with pytest.raises(ValidationError, match="not L1/L2"):
        validate_native_gate_report(preflight_path, stage)


def test_native_dynamic_contract_builds_safe_acceleration_limited_transcript(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, sites = city_and_sites
    private_episode = sample_episode_v3(ordinary_config, city, sites, 0)
    directions = select_native_test_directions(
        city,
        list(private_episode["starts"]),
        travel_distance_m=5.2,
        clearance_m=1.07,
        body_radius_m=0.32,
    )
    assert set(directions) == {f"uav-{index:02d}" for index in range(4)}
    contract = ordinary_config.raw["execution_contract"]
    transcript = build_native_action_transcript(contract, directions)
    assert len(transcript) == 36
    assert commanded_braking_distance(transcript, 0.2) == pytest.approx(1.5)
    previous = {drone_id: (0.0, 0.0, 0.0) for drone_id in directions}
    for step in transcript:
        for drone_id, command in step["commands"].items():
            velocity = tuple(command["linear_velocity_world_mps"])
            delta = distance(previous[drone_id], velocity)
            assert delta <= 0.5 + 1.0e-9
            previous[drone_id] = velocity


def test_native_dwell_and_quaternion_replay_contracts() -> None:
    samples = []
    for timestamp in (0.0, 0.2, 0.4, 0.6):
        samples.append(
            {
                "observe_case": "positive",
                "linear_speed_mps": 0.0,
                "angular_speed_deg_s": 0.0,
                "position": [0.0, 0.0, 2.5],
                "task_time_s": timestamp,
            }
        )
    interrupted_trace = (
        (0.8, 0.0),
        (1.0, 0.0),
        (1.2, 0.5),
        (1.4, 0.0),
        (1.6, 0.0),
        (1.8, 0.0),
    )
    for timestamp, speed in interrupted_trace:
        samples.append(
            {
                "observe_case": "interrupted",
                "linear_speed_mps": speed,
                "angular_speed_deg_s": 0.0,
                "position": [0.0, 0.0, 2.5],
                "task_time_s": timestamp,
            }
        )
    dwell = evaluate_native_dwell_samples(
        samples,
        {
            "continuous_dwell_s": 0.5,
            "max_linear_speed_mps": 0.25,
            "max_angular_speed_deg_s": 8.0,
            "max_pose_drift_m": 0.12,
        },
    )
    assert dwell["status"] == "PASS"
    first = [
        {
            "command_index": 0,
            "drone_id": "uav-00",
            "position": [1.0, 2.0, 3.0],
            "linear_velocity_mps": [0.5, 0.0, 0.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    second = json.loads(json.dumps(first))
    second[0]["orientation_wxyz"] = [-1.0, 0.0, 0.0, 0.0]
    replay = compare_native_replays(
        first,
        second,
        position_tolerance_m=1.0e-4,
        velocity_tolerance_mps=1.0e-4,
        orientation_tolerance=1.0e-5,
    )
    assert replay["status"] == "PASS"


def test_wheel_resources_are_available_without_repository_paths(tmp_path: Path) -> None:
    assert preset("ordinary-v1-mini")["schema"] == ("org.aerocity.bench.release.ordinary.v3")
    assert schema("ordinary-v3")["title"].startswith("AeroCityBench")
    output = tmp_path / "ordinary.json"
    assert write_preset("ordinary-v1-mini", output)["status"] == "PASS"
    assert load_ordinary_config(output).fleet_count == 4


def test_native_gate_command_keeps_preflight_ineligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train",),
        source_commit="3" * 40,
    )

    def fake_guard(command: list[str], **kwargs: object) -> GuardedProcessResult:
        stage = Path(command[command.index("--stage") + 1])
        output = Path(command[command.index("--output") + 1])
        _, _, _, _, input_bindings = load_native_gate_inputs(
            Path(command[command.index("--release-config") + 1]),
            Path(command[command.index("--task-spec") + 1]),
            Path(command[command.index("--public-episode") + 1]),
            Path(command[command.index("--cityspec") + 1]),
        )
        checks = {name: {"status": "PASS"} for name in REQUIRED_NATIVE_CHECKS}
        checks["velocity_tracking"] = {"status": "FAIL", "reason": "unit-test"}
        report = write_native_gate_report(
            output / "native_gate.json",
            stage_path=stage,
            execution_level="L1-preflight",
            runtime_fingerprint={
                "isaac_sim": "test",
                "native_gate_script_sha256": file_hash(Path(command[1])),
            },
            checks=checks,
            input_bindings=input_bindings,
        )
        report["formal_score_eligible"] = False
        report["evidence_scope"] = "native_stage_physics_preflight_not_formal_geometry_score"
        report.pop("native_gate_hash")
        report["native_gate_hash"] = content_hash(report)
        write_json(output / "native_gate.json", report)
        snapshot = _safe_host_snapshot()
        return GuardedProcessResult(0, 1.0, 0.2, False, snapshot, snapshot)

    monkeypatch.setattr("aerocity_bench.cli.run_guarded_process", fake_guard)
    result = _native_gate(
        argparse.Namespace(
            authority_root=authority,
            split="train",
            layout_id=None,
            output=tmp_path / "native",
            isaac_python=Path(sys.executable),
            timeout_s=30.0,
            step_count=3,
        )
    )
    assert result["status"] == "FAIL"
    assert result["execution_level"] == "L1-preflight"
    assert result["formal_score_eligible"] is False
    assert result["failed_checks"] == ["velocity_tracking"]


def test_schema_accepts_preset_and_rejects_shortcuts() -> None:
    release_schema = schema("ordinary-v3")
    Draft202012Validator.check_schema(release_schema)
    validator = Draft202012Validator(release_schema)
    raw = preset("ordinary-v1-mini")
    validator.validate(raw)

    extra = json.loads(json.dumps(raw))
    extra["unregistered_tuning_knob"] = True
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(extra)

    leaked_process_parameter = json.loads(json.dumps(raw))
    leaked_process_parameter["target_processes"]["profiles"]["clustered_surface"][
        "vertical_scale"
    ] = 2.0
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(leaked_process_parameter)

    wrong_formal_split = json.loads(json.dumps(raw))
    wrong_formal_split["governance"]["formal_splits"] = [
        "test_iid",
        "test_process_ood",
        "test_topology",
    ]
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(wrong_formal_split)


def test_cli_preset_commands_are_installation_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli_main(["list-presets", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing == {"presets": ["ordinary-v1-mini"]}

    assert cli_main(["show-preset", "ordinary-v1-mini"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["schema"] == "org.aerocity.bench.release.ordinary.v3"

    destination = tmp_path / "editable.json"
    assert (
        cli_main(
            [
                "init-config",
                "--preset",
                "ordinary-v1-mini",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS"
    assert load_ordinary_config(destination).fleet_count == 4


def _fake_asset_bundle(tmp_path: Path) -> tuple[Path, str, str]:
    asset_root = tmp_path / "assets"
    bundle_name = "test_cc0_bundle"
    bundle = asset_root / bundle_name
    model = bundle / "models" / "simple.usda"
    model.parent.mkdir(parents=True)
    model.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    license_snapshot = bundle / "provenance" / "snapshots" / "license.html"
    license_snapshot.parent.mkdir(parents=True)
    license_snapshot.write_text("<html>CC0 test evidence</html>\n", encoding="utf-8")
    official_snapshots = {}
    for evidence_name, suffix, content in (
        ("source_page", ".html", "<html>official asset page</html>\n"),
        ("info_api", ".json", '{"authors":{"Test Creator":"All"}}\n'),
        ("files_api", ".json", '{"blend":{"url":"https://example.invalid/simple.usda"}}\n'),
    ):
        snapshot = bundle / "provenance" / "snapshots" / "simple_asset" / (evidence_name + suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(content, encoding="utf-8")
        official_snapshots[evidence_name] = snapshot
    registry = {
        "assets": [
            {
                "asset_id": "simple_asset",
                "kind": "usd_model",
                "role": "visual_decoration",
                "source_page": "https://example.invalid/simple",
                "spdx": "CC0-1.0",
                "redistribution_allowed": True,
                "files": [
                    {
                        "path": "models/simple.usda",
                        "source_url": "https://example.invalid/simple.usda",
                        "sha256": file_hash(model),
                        "bytes": model.stat().st_size,
                    }
                ],
            }
        ]
    }
    registry_path = bundle / "ASSET_REGISTRY.json"
    write_json(registry_path, registry)
    manifest = {
        "failures": [],
        "verification_summary": {"all_file_and_url_checks_passed": True},
        "global_evidence": {
            "polyhaven_license": {
                "snapshot_path": str(license_snapshot),
                "sha256": file_hash(license_snapshot),
            }
        },
        "assets": [
            {
                "asset_id": "simple_asset",
                "creator_names": ["Test Creator"],
                "official_evidence": {
                    "source_page": {
                        "snapshot_path": str(official_snapshots["source_page"]),
                        "sha256": file_hash(official_snapshots["source_page"]),
                        "requested_url": "https://polyhaven.com/a/simple_asset",
                        "retrieved_at_utc": "2026-07-29T00:00:00Z",
                        "http_status": 200,
                    },
                    "info_api": {
                        "snapshot_path": str(official_snapshots["info_api"]),
                        "sha256": file_hash(official_snapshots["info_api"]),
                        "requested_url": "https://api.polyhaven.com/info/simple_asset",
                        "retrieved_at_utc": "2026-07-29T00:00:00Z",
                        "http_status": 200,
                    },
                    "files_api": {
                        "snapshot_path": str(official_snapshots["files_api"]),
                        "sha256": file_hash(official_snapshots["files_api"]),
                        "requested_url": "https://api.polyhaven.com/files/simple_asset",
                        "retrieved_at_utc": "2026-07-29T00:00:00Z",
                        "http_status": 200,
                    },
                },
            }
        ],
    }
    manifest_path = bundle / "provenance" / "PROVENANCE_MANIFEST.json"
    write_json(manifest_path, manifest)
    (bundle / "provenance" / "PROVENANCE_MANIFEST.sha256").write_text(
        f"{file_hash(manifest_path)}  PROVENANCE_MANIFEST.json\n", encoding="ascii"
    )
    return asset_root, bundle_name, "simple_asset"


def _fake_config(
    tmp_path: Path,
    bundle: str,
    asset_id: str,
    *,
    maximum_single_observation_target_fraction: float | None = None,
) -> OrdinaryReleaseConfig:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["release_version"] = "test-ordinary-v3"
    raw["assets"]["bundle"] = bundle
    raw["assets"]["allowlist"] = [asset_id]
    if maximum_single_observation_target_fraction is not None:
        raw["admission"]["maximum_single_observation_target_fraction"] = (
            maximum_single_observation_target_fraction
        )
    for split in ORDINARY_SPLITS:
        raw["split_counts"][split] = 1
    path = tmp_path / "ordinary.json"
    write_json(path, raw)
    return load_ordinary_config(path)


def test_compiler_keeps_structural_detail_collidable_and_ground_detail_visual_only(
    tmp_path: Path,
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    lock, _, _ = load_official_cc0_lock(asset_root, bundle, [asset_id])
    city = _first_admitted_city(config, "train", 0, [asset_id])
    output = tmp_path / "compiled"
    compile_scene(city, output, lock)
    scene = (output / "scene.usda").read_text(encoding="utf-8")
    collision = (output / "collision.usda").read_text(encoding="utf-8")
    expected_collision_count = (
        1
        + sum(len(building["components"]) for building in city["buildings"])
        + len(city["obstacles"])
    )
    assert 'def Xform "UrbanGroundDetail"' in scene
    assert "sidewalk" in scene
    assert "mark_" in scene
    assert 'def Xform "ProceduralVisualDetail"' in scene
    assert "facade_" in scene
    assert "bool physics:collisionEnabled = false" in scene
    assert collision.count('"PhysicsCollisionAPI"') == expected_collision_count
    assert "UrbanGroundDetail" not in collision
    assert "VisualDecorations" not in collision
    assert "ProceduralVisualDetail" not in collision


def test_supply_chain_rejects_escaping_bundle(tmp_path: Path) -> None:
    with pytest.raises(AssetRegistryError, match="simple controlled name"):
        validate_bundle_root(tmp_path, "../nucleus")


def test_supply_chain_and_builder_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    lock, evidence, closure = load_official_cc0_lock(asset_root, bundle, [asset_id])
    assert set(lock.records) == {asset_id}
    assert evidence.asset_creators[asset_id] == ("Test Creator",)
    assert closure["unresolved_dependencies"] == 0
    output = tmp_path / "authority"
    report = build_ordinary_release(
        config,
        asset_root,
        output,
        ("train",),
        source_commit="a" * 40,
    )
    assert report["status"] == "PASS"
    assert report["episode_count"] == 3
    assert validate_ordinary_release(output)["status"] == "PASS"
    run_dir = tmp_path / "run"
    assert (
        cli_main(
            [
                "run-baseline",
                str(output),
                "--method",
                "random-safe",
                "--split",
                "train",
                "--max-steps",
                "1",
                "--output",
                str(run_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    episode_path = next((output / "splits" / "train").glob("*/evaluator_private/episodes/*.json"))
    metric_path = tmp_path / "metrics.json"
    assert (
        cli_main(
            [
                "evaluate",
                "--run",
                str(run_dir),
                "--episode",
                str(episode_path),
                "--duration-s",
                "300",
                "--output",
                str(metric_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["execution_level"] == "L0"
    assert metric_path.is_file()
    with pytest.raises(ValidationError, match="native Isaac"):
        export_public_release(output, tmp_path / "public")
    with pytest.raises(ValidationError, match="complete"):
        promote_ordinary_release(
            output,
            tmp_path / "promoted",
            native_report_dir=tmp_path / "absent-native",
            scientific_report_path=tmp_path / "absent-scientific.json",
        )
    (output / "unmanifested-file.txt").write_text("must be rejected\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="manifest file set differs"):
        validate_ordinary_release(output)


@pytest.mark.parametrize("prior_level", ["full-cells", "coarse-regions"])
def test_g2_i_l1_materializer_stages_assets_in_release_relative_layout(
    tmp_path: Path, prior_level: str
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(
        tmp_path,
        bundle,
        asset_id,
        maximum_single_observation_target_fraction=1.0,
    )
    city = _first_admitted_city(config, "calibration", 0, [asset_id])
    city_path = tmp_path / "city.json"
    write_json(city_path, city)
    output = tmp_path / f"g2-i-{prior_level}"

    manifest = materialize_g2_i_l1_layout(
        city_path,
        tmp_path / "ordinary.json",
        asset_root,
        output,
        episode_index=0,
        prior_level=prior_level,
    )

    layout_root = output / manifest["layout_relative_root"]
    scene_path = layout_root / "scene_authority" / "scene.usda"
    task = json.loads((layout_root / "method_public" / "task_spec.json").read_text())
    public_episode = json.loads(
        (layout_root / "method_public" / "episodes" / "episode-0000.json").read_text()
    )
    authority_task = json.loads(
        (layout_root / "evaluator_private" / "task_spec_authority.json").read_text()
    )
    assert manifest["formal_score_eligible"] is False
    assert manifest["inspection_prior_level"] == prior_level
    assert manifest["private_episode_source"] == "development-resample"
    assert scene_path.is_file()
    assert (output / "_assets" / "asset_lock.json").is_file()
    assert (output / "_assets" / bundle / "models" / "simple.usda").is_file()
    if prior_level == "full-cells":
        assert manifest["atlas_hash"] == task["inspection_atlas"]["atlas_hash"]
        assert public_episode["mission_sector"]["selected_cell_ids"]
    else:
        assert manifest["atlas_hash"] == task["inspection_atlas_projection"]["source_atlas_hash"]
        assert "mission_sector" not in public_episode
        assert "selected_cell_ids" not in json.dumps(public_episode)
    assert manifest["authority_task_spec_hash"] == authority_task["task_spec_hash"]
    assert authority_task["inspection_atlas"]["atlas_hash"] == manifest["atlas_hash"]


def test_g2_i_l1_materializer_replays_frozen_private_episode_fail_closed(
    tmp_path: Path,
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(
        tmp_path,
        bundle,
        asset_id,
        maximum_single_observation_target_fraction=1.0,
    )
    city = _first_admitted_city(config, "calibration", 0, [asset_id])
    city_path = tmp_path / "city.json"
    write_json(city_path, city)
    authority_task = compile_g2_i_task_spec(
        city, config.raw["execution_contract"], config.raw["fleet"]
    )
    episode = sample_episode_v3(
        config,
        city,
        derive_support_sites_v3(city, config),
        0,
        public_task_spec=authority_task,
    )
    episode_path = tmp_path / "frozen-private-episode.json"
    write_json(episode_path, episode)

    output = tmp_path / "g2-i-frozen"
    manifest = materialize_g2_i_l1_layout(
        city_path,
        tmp_path / "ordinary.json",
        asset_root,
        output,
        episode_index=0,
        prior_level="full-cells",
        private_episode_path=episode_path,
    )
    materialized = json.loads(
        (
            output
            / manifest["layout_relative_root"]
            / "evaluator_private"
            / "episodes"
            / "episode-0000.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["private_episode_source"] == "frozen-calibration-input"
    assert manifest["private_episode_sha256"] == content_hash(episode)
    assert materialized == episode

    tampered = copy.deepcopy(episode)
    tampered["targets"][0]["position"][0] += 0.01
    tampered_path = tmp_path / "tampered-private-episode.json"
    write_json(tampered_path, tampered)
    rejected_output = tmp_path / "g2-i-tampered"
    with pytest.raises(ValueError, match="hash does not match"):
        materialize_g2_i_l1_layout(
            city_path,
            tmp_path / "ordinary.json",
            asset_root,
            rejected_output,
            episode_index=0,
            prior_level="full-cells",
            private_episode_path=tampered_path,
        )
    assert not rejected_output.exists()


def test_g2_i_l1_materializer_requires_explicit_split_for_staged_public_cityspec(
    tmp_path: Path,
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(
        tmp_path,
        bundle,
        asset_id,
        maximum_single_observation_target_fraction=1.0,
    )
    city = _first_admitted_city(config, "calibration", 0, [asset_id])
    staged_public_city = {key: value for key, value in city.items() if key != "split"}
    city_path = tmp_path / "staged-public-city.json"
    write_json(city_path, staged_public_city)

    with pytest.raises(ValueError, match="supply --development-split"):
        materialize_g2_i_l1_layout(
            city_path,
            tmp_path / "ordinary.json",
            asset_root,
            tmp_path / "missing-split",
            episode_index=0,
            prior_level="full-cells",
        )

    manifest = materialize_g2_i_l1_layout(
        city_path,
        tmp_path / "ordinary.json",
        asset_root,
        tmp_path / "explicit-calibration",
        episode_index=0,
        prior_level="full-cells",
        development_split="calibration",
    )
    assert manifest["split"] == "calibration"

    with pytest.raises(ValueError, match="formal split"):
        materialize_g2_i_l1_layout(
            city_path,
            tmp_path / "ordinary.json",
            asset_root,
            tmp_path / "formal-split",
            episode_index=0,
            prior_level="full-cells",
            development_split="test_iid",
        )


def test_complete_release_can_only_export_after_evidence_promotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority-complete"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        source_commit="c" * 40,
    )
    source_index = json.loads((authority / "release_index.json").read_text(encoding="utf-8"))
    native_reports = tmp_path / "native-reports"
    checks = {
        name: {"status": "PASS", "evidence": "integration-test"} for name in REQUIRED_NATIVE_CHECKS
    }
    for layout in source_index["layouts"]:
        stage = (
            authority
            / "splits"
            / layout["split"]
            / layout["layout_id"]
            / "scene_authority"
            / "stage.usda"
        )
        layout_root = authority / "splits" / layout["split"] / layout["layout_id"]
        _, _, _, _, input_bindings = load_native_gate_inputs(
            authority / "authority_private" / "release_config.json",
            layout_root / "method_public" / "task_spec.json",
            sorted((layout_root / "method_public" / "episodes").glob("*.json"))[0],
            layout_root / "scene_authority" / "cityspec.json",
        )
        write_native_gate_report(
            native_reports / f"{layout['layout_id']}.json",
            stage_path=stage,
            execution_level="L1",
            runtime_fingerprint={"isaac_sim": "test", "driver": "test"},
            checks=checks,
            input_bindings=input_bindings,
            formal_score_eligible=True,
            evidence_scope=FORMAL_L1_EVIDENCE_SCOPE,
        )
    scientific = {
        "schema": "org.aerocity.bench.scientific-gate.ordinary.v1",
        "status": "PASS",
        "source_release_index_hash": source_index["release_index_hash"],
        "execution_contract_hash": source_index["execution_contract_hash"],
        "calibration_split": "calibration",
        "formal_results_accessed": False,
        "gates": {
            name: {"status": "PASS", "evidence_hash": content_hash([name, "evidence"])}
            for name in (
                "three_dimensionality",
                "difficulty_calibration",
                "shortcut_red_team",
                "coverage_to_search_pilot",
                "baseline_vertical_slice",
            )
        },
        "approved_by": "integration-test",
        "approved_at": "2026-07-29T00:00:00Z",
    }
    scientific["scientific_gate_hash"] = content_hash(scientific)
    scientific_path = tmp_path / "scientific-gate.json"
    write_json(scientific_path, scientific)
    promoted = tmp_path / "authority-promoted"
    report = promote_ordinary_release(
        authority,
        promoted,
        native_report_dir=native_reports,
        scientific_report_path=scientific_path,
    )
    assert report["native_isaac_gate"] == "verified"
    assert report["scientific_status"] == "release_candidate"
    public = tmp_path / "public"
    assert export_public_release(promoted, public)["status"] == "PASS"
    public_report = validate_public_release(public)
    assert public_report["development_layout_count"] == 3
    assert public_report["formal_blind_layout_count"] == 3
    assert not list(public.rglob("evaluator_private"))
    assert not list(public.rglob("authority_private"))
    assert list(public.rglob("development_evaluator"))
    assert not (public / "release_config.json").exists()
    assert not any((public / "splits" / split).exists() for split in FORMAL_SPLITS)
    assert (public / "benchmark_contract.json").is_file()

    assert cli_main(["validate", str(public)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    public_run = tmp_path / "public-calibration-run"
    assert (
        cli_main(
            [
                "run-baseline",
                str(public),
                "--method",
                "random-safe",
                "--split",
                "calibration",
                "--max-steps",
                "2",
                "--output",
                str(public_run),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["formal_score_eligible"] is False
    assert (
        cli_main(
            [
                "run-baseline",
                str(public),
                "--method",
                "random-safe",
                "--split",
                "test_iid",
                "--max-steps",
                "1",
                "--output",
                str(tmp_path / "forbidden-formal-run"),
            ]
        )
        == 2
    )
    assert "blind evaluator" in capsys.readouterr().err


def test_builder_detects_private_episode_tampering(tmp_path: Path) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    output = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        output,
        ("train",),
        source_commit="b" * 40,
    )
    episode_path = next(output.glob("splits/train/*/evaluator_private/episodes/*.json"))
    data = json.loads(episode_path.read_text(encoding="utf-8"))
    data["targets"][0]["position"][0] += 1.0
    write_json(episode_path, data)
    with pytest.raises(ValidationError):
        validate_ordinary_release(output)


def test_review_camera_avoids_buildings_and_keeps_starts_visible() -> None:
    city = {
        "size_m": 80.0,
        "metrics": {"height_max_m": 42.0},
        "buildings": [
            {
                "id": "building-blocking-old-camera",
                "components": [
                    {
                        "id": "tower",
                        "center": [20.0, -16.0, 20.0],
                        "size": [12.0, 12.0, 40.0],
                    }
                ],
            }
        ],
        "obstacles": [],
    }
    starts = [
        [3.25, 1.65, 2.5],
        [2.0, 2.9, 2.5],
        [0.75, 1.65, 2.5],
        [2.0, 0.4, 2.5],
    ]

    camera, look_at = review_camera_pose(starts, city)

    blocker = AABB.from_center_size("tower", [20.0, -16.0, 20.0], [12.0, 12.0, 40.0])
    assert not blocker.contains(camera, margin=2.0)
    assert camera != pytest.approx((20.0, -16.0, 18.5))
    assert look_at == pytest.approx((2.0, 1.65, 2.5))


def test_review_camera_validates_input() -> None:
    city = {"size_m": 20.0, "metrics": {"height_max_m": 4.0}, "buildings": [], "obstacles": []}
    with pytest.raises(ValueError, match="at least one"):
        review_camera_pose([], city)
    with pytest.raises(ValueError, match="three-dimensional"):
        review_camera_pose([[1.0, 2.0]], city)


def test_capture_review_one_command_prepares_private_overlay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train",),
        source_commit="e" * 40,
    )
    output = tmp_path / "review"

    assert (
        cli_main(
            [
                "capture-review",
                str(authority),
                "--split",
                "train",
                "--target-count",
                "8",
                "--output",
                str(output),
                "--prepare-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    episode = json.loads((output / "review_episode_8.json").read_text(encoding="utf-8"))
    assert report["status"] == "PREPARED"
    assert report["formal_score_eligible"] is False
    assert episode["target_count"] == 8
    assert episode["formal_score_eligible"] is False


def test_capture_review_batch_is_resumable_and_never_uses_formal_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train", "calibration", "test_iid"),
        source_commit="f" * 40,
    )
    output = tmp_path / "batch"
    command = [
        "capture-review-batch",
        str(authority),
        "--target-count",
        "8",
        "--output",
        str(output),
        "--prepare-only",
    ]

    assert cli_main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "PASS"
    assert first["layout_count"] == 2
    assert {job["split"] for job in first["jobs"]} == {"train", "calibration"}
    assert not (output / "scenes" / "test_iid").exists()
    contract = json.loads((output / "batch_contract.json").read_text(encoding="utf-8"))
    assert contract["schema"] == "org.aerocity.bench.visual-review-batch-contract.v6"
    pipeline = dict(contract["review_pipeline"])
    pipeline_hash = pipeline.pop("pipeline_hash")
    assert content_hash(pipeline) == pipeline_hash
    assert set(pipeline["function_source_hashes"]) == {
        "instance_visibility_aggregator",
        "layout_authority",
        "prepared_attempt_verifier",
        "review_frame_mode_verifier",
        "review_frame_resolution_verifier",
        "review_attempt_verifier",
        "review_sampler",
    }
    progress = json.loads((output / "batch_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "PASS"
    assert progress["report_hash"] == first["report_hash"]

    assert cli_main([*command, "--resume"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["report_hash"] == first["report_hash"]

    first_episode = Path(second["jobs"][0]["verified"]["episode"])
    corrupted = json.loads(first_episode.read_text(encoding="utf-8"))
    corrupted["target_count"] = 7
    write_json(first_episode, corrupted)
    assert cli_main([*command, "--resume"]) == 0
    recovered = json.loads(capsys.readouterr().out)
    assert recovered["jobs"][0]["verified"]["attempt"] == "attempt-02"
    assert recovered["report_hash"] != first["report_hash"]

    assert cli_main([*command, "--timeout-s", "601", "--resume"]) == 2
    assert "resume contract differs" in capsys.readouterr().err


def test_capture_review_rejects_formal_target_disclosure(tmp_path: Path) -> None:
    args = argparse.Namespace(
        authority_root=tmp_path,
        split="test_iid",
        layout_id=None,
        target_count=32,
        process="height_stratified",
        output=tmp_path / "review",
        isaac_python=None,
        width=960,
        height=640,
        timeout_s=10.0,
        prepare_only=True,
    )
    with pytest.raises(ValueError, match="formal test targets"):
        from aerocity_bench.cli import _capture_review

        _capture_review(args)


def test_capture_review_rejects_noncanonical_l2_resolution(tmp_path: Path) -> None:
    from aerocity_bench.cli import _capture_review

    args = argparse.Namespace(
        authority_root=tmp_path,
        split="train",
        layout_id=None,
        target_count=8,
        process="height_stratified",
        output=tmp_path / "review",
        isaac_python=None,
        width=480,
        height=320,
        timeout_s=10.0,
        prepare_only=False,
    )
    with pytest.raises(ValueError, match="frozen 960x640"):
        _capture_review(args)


def test_windows_1344_and_commit_pressure_are_host_failures() -> None:
    from aerocity_bench.host_guard import _is_isaac_process_record, _is_process_tree_member

    safe = HostSnapshot("now", "test", 1000, 400, 600, 0.60)
    threshold = HostSnapshot("now", "test", 1000, 350, 650, 0.65)
    current = host_snapshot()

    assert not commit_limit_exceeded(safe, 0.65)
    assert commit_limit_exceeded(threshold, 0.65)
    assert is_host_1344(1, "SetTokenInformation(TokenDefaultDacl): 1344")
    assert is_host_1344(1344, "")
    assert not is_host_1344(1, "ordinary scene validation failure")
    assert _is_isaac_process_record("python.exe", "python isaac_native_gate.py")
    assert _is_isaac_process_record("python.exe", "python quadrotor_physics_preflight.py")
    assert _is_isaac_process_record("python.exe", "python cf2x_l1_fleet_preflight.py")
    assert _is_isaac_process_record("python.exe", "python quadrotor_l1_vertical_slice.py")
    assert _is_isaac_process_record("python.exe", "python run_hm3d_p07_execution_smoke.py")
    assert _is_isaac_process_record("kit.exe", "")
    assert not _is_isaac_process_record("powershell.exe", "ruff check tools/isaac_native_gate.py")
    parent_by_pid = {101: 100, 102: 101, 777: 1}
    assert _is_process_tree_member(100, root_pid=100, parent_by_pid=parent_by_pid)
    assert _is_process_tree_member(102, root_pid=100, parent_by_pid=parent_by_pid)
    assert not _is_process_tree_member(777, root_pid=100, parent_by_pid=parent_by_pid)
    if current.commit_fraction is not None:
        assert 0.0 <= current.commit_fraction <= 1.0


def _safe_host_snapshot() -> HostSnapshot:
    return HostSnapshot("now", "test", 1000, 800, 200, 0.20)


def test_guarded_process_classifies_exit_timeout_commit_and_1344(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aerocity_bench.host_guard.foreign_isaac_processes", lambda **_kwargs: [])
    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", _safe_host_snapshot)
    ordinary_report = tmp_path / "ordinary-host.json"
    ordinary = run_guarded_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        environment=os.environ.copy(),
        log_path=tmp_path / "ordinary.log",
        report_path=ordinary_report,
        timeout_s=5.0,
        poll_interval_s=0.01,
    )
    assert ordinary.returncode == 7
    assert json.loads(ordinary_report.read_text(encoding="utf-8"))["trigger"] is None

    with pytest.raises(TimeoutError, match="exceeded"):
        run_guarded_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "timeout.log",
            report_path=tmp_path / "timeout-host.json",
            timeout_s=0.05,
            poll_interval_s=0.01,
        )
    timeout_report = json.loads((tmp_path / "timeout-host.json").read_text(encoding="utf-8"))
    assert timeout_report["trigger"] == "timeout"
    assert timeout_report["process_tree_policy"] == "terminate_owned_attempt_tree_only"

    calls = 0

    def pressure_snapshot() -> HostSnapshot:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _safe_host_snapshot()
        return HostSnapshot("now", "test", 1000, 100, 900, 0.90)

    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", pressure_snapshot)
    with pytest.raises(HostGuardError, match="82%"):
        run_guarded_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "commit.log",
            report_path=tmp_path / "commit-host.json",
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
    assert (
        json.loads((tmp_path / "commit-host.json").read_text(encoding="utf-8"))["trigger"]
        == "runtime_commit_limit"
    )

    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", _safe_host_snapshot)
    with pytest.raises(HostGuardError, match="1344"):
        run_guarded_process(
            [
                sys.executable,
                "-c",
                "print('SetTokenInformation(TokenDefaultDacl): 1344'); raise SystemExit(1)",
            ],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "1344.log",
            report_path=tmp_path / "1344-host.json",
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
    assert (
        json.loads((tmp_path / "1344-host.json").read_text(encoding="utf-8"))["trigger"]
        == "windows_1344"
    )


def test_guarded_process_rejects_a_residual_isaac_runtime_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    census_calls = 0

    def staged_census(*, owned_root_pid: int | None = None) -> list[dict[str, object]]:
        nonlocal census_calls
        if owned_root_pid is not None:
            return []
        census_calls += 1
        return (
            []
            if census_calls == 1
            else [{"pid": 777, "name": "kit.exe", "command_line": "isaac-sim"}]
        )

    monkeypatch.setattr("aerocity_bench.host_guard.foreign_isaac_processes", staged_census)
    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", _safe_host_snapshot)
    report_path = tmp_path / "residual-host.json"
    with pytest.raises(HostGuardError, match="remained after the owned child exited"):
        run_guarded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "residual.log",
            report_path=report_path,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["trigger"] == "residual_runtime"
    assert report["foreign_runtime_count_before"] == 0
    assert report["foreign_runtime_count_after"] == 1


def test_guarded_process_fails_fast_on_foreign_runtime_started_during_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aerocity_bench import host_guard as host_guard_module

    monitor_census_count = 0

    def staged_census(*, owned_root_pid: int | None = None) -> list[dict[str, object]]:
        nonlocal monitor_census_count
        if owned_root_pid is None:
            return []
        monitor_census_count += 1
        if monitor_census_count < 2:
            return []
        return [
            {
                "pid": 777,
                "parent_pid": 1,
                "name": "kit.exe",
                "command_line": "foreign isaac-sim",
            }
        ]

    stopped_owned_pids: list[int] = []
    original_stop = host_guard_module._stop_process

    def recording_stop(process: subprocess.Popen[object]) -> None:
        assert process.pid != 777
        stopped_owned_pids.append(process.pid)
        original_stop(process)

    monkeypatch.setattr("aerocity_bench.host_guard.foreign_isaac_processes", staged_census)
    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", _safe_host_snapshot)
    monkeypatch.setattr("aerocity_bench.host_guard._stop_process", recording_stop)
    report_path = tmp_path / "foreign-during-host.json"
    with pytest.raises(HostGuardError, match="started during"):
        run_guarded_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "foreign-during.log",
            report_path=report_path,
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stopped_owned_pids == [report["child_pid"]]
    assert report["status"] == "FAIL"
    assert report["trigger"] == "foreign_runtime_during_attempt"
    assert report["foreign_runtime_count_before"] == 0


def test_cli_module_entrypoint_executes_main() -> None:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "aerocity_bench.cli", "list-presets", "--json"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ordinary-v1-mini" in json.loads(completed.stdout)["presets"]


def test_guarded_process_writes_preflight_and_monitor_failure_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    high_commit = HostSnapshot("now", "test", 1000, 300, 700, 0.70)
    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", lambda: high_commit)
    with pytest.raises(HostGuardError, match="too high"):
        run_guarded_process(
            [sys.executable, "-c", "print('must not launch')"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "start.log",
            report_path=tmp_path / "start-host.json",
            timeout_s=5.0,
        )
    start_report = json.loads((tmp_path / "start-host.json").read_text(encoding="utf-8"))
    assert start_report["trigger"] == "start_commit_limit"
    assert start_report["child_pid"] is None

    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", _safe_host_snapshot)
    monkeypatch.setattr(
        "aerocity_bench.host_guard.foreign_isaac_processes",
        lambda **_kwargs: [{"pid": 123, "name": "kit.exe", "command_line": "isaac-sim"}],
    )
    with pytest.raises(HostGuardError, match="already active"):
        run_guarded_process(
            [sys.executable, "-c", "print('must not launch')"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "foreign.log",
            report_path=tmp_path / "foreign-host.json",
            timeout_s=5.0,
        )
    foreign_report = json.loads((tmp_path / "foreign-host.json").read_text(encoding="utf-8"))
    assert foreign_report["trigger"] == "foreign_runtime"
    assert foreign_report["foreign_runtime_count_before"] == 1

    calls = 0

    def failing_monitor_snapshot() -> HostSnapshot:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _safe_host_snapshot()
        raise HostGuardError("monitor probe failed")

    monkeypatch.setattr("aerocity_bench.host_guard.foreign_isaac_processes", lambda **_kwargs: [])
    monkeypatch.setattr("aerocity_bench.host_guard.host_snapshot", failing_monitor_snapshot)
    with pytest.raises(HostGuardError, match="monitoring failed"):
        run_guarded_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=tmp_path / "monitor.log",
            report_path=tmp_path / "monitor-host.json",
            timeout_s=5.0,
            poll_interval_s=0.01,
        )
    monitor_report = json.loads((tmp_path / "monitor-host.json").read_text(encoding="utf-8"))
    assert monitor_report["trigger"] == "monitor_failure"
    assert monitor_report["status"] == "FAIL"


def test_windows_tree_stop_falls_back_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aerocity_bench.host_guard import _stop_process

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 1

    process = FakeProcess()
    monkeypatch.setattr("aerocity_bench.host_guard.os.name", "nt")
    monkeypatch.setattr(
        "aerocity_bench.host_guard.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("taskkill", 30)),
    )
    _stop_process(process)  # type: ignore[arg-type]
    assert process.killed


def test_host_mutex_is_exclusive_and_releasable() -> None:
    with isaac_host_lock():
        with pytest.raises(HostGuardError, match="already holds"):
            with isaac_host_lock():
                pass
    with isaac_host_lock():
        pass


def test_review_resume_recomputes_all_artifact_hashes(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-01"
    views = attempt / "isaac_views"
    views.mkdir(parents=True)
    episode_payload = {
        "schema": "org.aerocity.bench.visual-review-private.v1",
        "layout_id": "city-test",
        "layout_hash": "1" * 64,
        "target_count": 2,
        "formal_score_eligible": False,
    }
    episode = {**episode_payload, "episode_hash": content_hash(episode_payload)}
    write_json(attempt / "review_episode_2.json", episode)
    frames = {}
    frame_names = [*REVIEW_BASE_FRAMES, "target_close_000", "target_close_001"]
    for index, name in enumerate(frame_names):
        rgb = views / f"{name}_rgb.png"
        depth = views / f"{name}_depth.png"
        mask = views / f"{name}_instance_segmentation.npz"
        labels = views / f"{name}_instance_labels.json"
        rgb.write_bytes(f"rgb-{index}".encode())
        depth.write_bytes(f"depth-{index}".encode())
        mask.write_bytes(f"mask-{index}".encode())
        labels.write_text(json.dumps({"idToLabels": {"1": "/World/Target_000"}}), encoding="utf-8")
        frames[name] = {
            "review_overlay_mode": (
                "local_context" if name.startswith("target_close_") else "overview_highlight"
            ),
            "rgb": {"sha256": file_hash(rgb), "shape": [640, 960, 3]},
            "depth": {"sha256": file_hash(depth), "shape": [640, 960]},
            "instance_segmentation": {
                "mask_sha256": file_hash(mask),
                "labels_sha256": file_hash(labels),
                "shape": [640, 960],
            },
        }
    contact = views / "review_contact_sheet.png"
    target_contact = views / "target_review_contact_sheet.png"
    contact.write_bytes(b"contact")
    target_contact.write_bytes(b"target-contact")
    authority_payload = {
        "schema": "org.aerocity.bench.review-layout-authority.v1",
        "split": "train",
        "layout_id": "city-test",
        "layout_hash": "1" * 64,
        "authority_manifest_hash": "2" * 64,
        "authority_manifest_sha256": "3" * 64,
        "cityspec_sha256": "4" * 64,
        "stage_sha256": "5" * 64,
        "scene_sha256": "6" * 64,
        "collision_sha256": "7" * 64,
    }
    authority = {
        **authority_payload,
        "authority_record_hash": content_hash(authority_payload),
    }
    write_json(attempt / "review_authority.json", authority)
    health = {
        "schema": "org.aerocity.bench.isaac-scene-health.v6",
        "status": "passed",
        "layout_id": "city-test",
        "layout_hash": "1" * 64,
        "stage_sha256": "5" * 64,
        "scene_sha256": "6" * 64,
        "collision_sha256": "7" * 64,
        "evidence_scope": VISUAL_REVIEW_EVIDENCE_SCOPE,
        "authority_record": authority,
        "private_target_audit": {
            "target_count": 2,
            "formal_score_eligible": False,
            "episode_hash": episode["episode_hash"],
            "start_markers_overlap_free": True,
        },
        "frames": frames,
        "contact_sheet": {"sha256": file_hash(contact), "shape": [1706, 960, 3]},
        "target_contact_sheet": {
            "sha256": file_hash(target_contact),
            "shape": [106 + 240, 1280, 3],
        },
        "review_marker_visibility": {"status": "PASS", "scope": "review_overlay_visibility"},
        "instance_visibility": {"status": "PASS"},
        "frame_diversity": {"status": "PASS"},
        "review_overlay_collision_prim_count": 0,
        "review_overlay_rigid_body_prim_count": 0,
    }
    health["health_report_hash"] = content_hash(health)
    write_json(views / "isaac_scene_health_review.json", health)
    write_json(
        attempt / "host_guard.json",
        {"schema": HOST_GUARD_SCHEMA, "status": "PASS", "returncode": 0, "trigger": None},
    )
    assert _verified_review_attempt(attempt, 2, authority) is not None
    health["review_marker_visibility"] = {"status": "FAIL", "scope": "review_overlay_visibility"}
    health_payload = dict(health)
    health_payload.pop("health_report_hash", None)
    health["health_report_hash"] = content_hash(health_payload)
    write_json(views / "isaac_scene_health_review.json", health)
    assert _verified_review_attempt(attempt, 2, authority) is None
    health["review_marker_visibility"] = {"status": "PASS", "scope": "review_overlay_visibility"}
    health_payload = dict(health)
    health_payload.pop("health_report_hash", None)
    health["health_report_hash"] = content_hash(health_payload)
    write_json(views / "isaac_scene_health_review.json", health)
    assert _verified_review_attempt(attempt, 2, authority) is not None
    health["frames"]["target_close_000"]["rgb"]["shape"] = [320, 480, 3]
    health_payload = dict(health)
    health_payload.pop("health_report_hash", None)
    health["health_report_hash"] = content_hash(health_payload)
    write_json(views / "isaac_scene_health_review.json", health)
    assert _verified_review_attempt(attempt, 2, authority) is None
    health["frames"]["target_close_000"]["rgb"]["shape"] = [640, 960, 3]
    health_payload = dict(health)
    health_payload.pop("health_report_hash", None)
    health["health_report_hash"] = content_hash(health_payload)
    write_json(views / "isaac_scene_health_review.json", health)
    assert _verified_review_attempt(attempt, 2, authority) is not None
    (views / "target_close_000_rgb.png").write_bytes(b"tampered")
    assert _verified_review_attempt(attempt, 2, authority) is None
    (views / "target_close_000_rgb.png").write_bytes(b"rgb-10")
    (views / "target_close_000_instance_segmentation.npz").write_bytes(b"tampered mask")
    assert _verified_review_attempt(attempt, 2, authority) is None
    (views / "target_close_000_instance_segmentation.npz").write_bytes(b"mask-10")
    (views / "target_close_000_instance_labels.json").write_text("{}", encoding="utf-8")
    assert _verified_review_attempt(attempt, 2, authority) is None
    (views / "target_close_000_instance_labels.json").write_text(
        json.dumps({"idToLabels": {"1": "/World/Target_000"}}), encoding="utf-8"
    )
    host_report = json.loads((attempt / "host_guard.json").read_text(encoding="utf-8"))
    host_report.update({"status": "FAIL", "returncode": 1, "trigger": "runtime_commit_limit"})
    write_json(attempt / "host_guard.json", host_report)
    assert _verified_review_attempt(attempt, 2, authority) is None


def test_capture_failure_detail_prefers_machine_progress_exception(tmp_path: Path) -> None:
    views = tmp_path / "isaac_views"
    views.mkdir()
    (views / "review_progress.log").write_text(
        "before_camera\nexception=RuntimeError: marker audit failed\ntraceback\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "isaac_capture.log"
    log_path.write_text("noisy Kit shutdown output", encoding="utf-8")
    assert _capture_failure_detail(views, log_path) == "RuntimeError: marker audit failed"


def test_instance_visibility_requires_every_target_local_context_and_start() -> None:
    complete = {
        "overview_ne": {
            "id_pixel_counts": {"11": 40, "12": 31},
            "id_to_labels": {
                "11": "/World/EvaluatorPrivateAudit/Target_000",
                "12": "/World/EvaluatorPrivateAudit/Target_001",
            },
            "id_to_semantics": {},
        },
    }
    complete["target_close_000"] = {
        "id_pixel_counts": {"11": 80},
        "id_to_labels": {"11": "/World/EvaluatorPrivateAudit/Target_000"},
        "id_to_semantics": {},
    }
    complete["target_close_001"] = {
        "id_pixel_counts": {"12": 80},
        "id_to_labels": {"12": "/World/EvaluatorPrivateAudit/Target_001"},
        "id_to_semantics": {},
    }
    complete["starts_close"] = {
        "id_pixel_counts": {"21": 30, "22": 30},
        "id_to_labels": {
            "21": "/World/EvaluatorPrivateAudit/DroneStart_000",
            "22": "/World/EvaluatorPrivateAudit/DroneStart_001",
        },
        "id_to_semantics": {},
    }
    report = aggregate_review_instance_visibility(
        complete, target_count=2, start_count=2, minimum_pixels=24
    )
    assert report["status"] == "PASS"
    assert report["verified_instance_count"] == 4

    hidden_last_target = json.loads(json.dumps(complete))
    hidden_last_target["overview_ne"]["id_pixel_counts"]["12"] = 0
    report = aggregate_review_instance_visibility(
        hidden_last_target, target_count=2, start_count=2, minimum_pixels=24
    )
    assert report["status"] == "PASS"
    assert report["unseen_in_scene_overviews"] == ["target_001"]

    cross_target_views = json.loads(json.dumps(complete))
    cross_target_views["target_close_000"]["id_to_labels"]["11"] = (
        "/World/EvaluatorPrivateAudit/Target_001"
    )
    cross_target_views["target_close_001"]["id_to_labels"]["12"] = (
        "/World/EvaluatorPrivateAudit/Target_000"
    )
    report = aggregate_review_instance_visibility(
        cross_target_views, target_count=2, start_count=2, minimum_pixels=24
    )
    assert report["status"] == "FAIL"
    assert report["missing_local_targets"] == ["target_000", "target_001"]

    report = aggregate_review_instance_visibility(
        complete,
        target_count=2,
        start_count=2,
        minimum_pixels=24,
        frame_pixel_count=100,
        maximum_local_fraction=0.20,
    )
    assert report["status"] == "FAIL"
    assert report["oversized_local_targets"] == ["target_000", "target_001"]


def test_prepare_resume_rejects_cross_layout_attempt_copy(tmp_path: Path) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train", "calibration"),
        source_commit="3" * 40,
    )
    output = tmp_path / "prepare-batch"
    args = argparse.Namespace(
        authority_root=authority,
        splits=["train", "calibration"],
        target_count=8,
        process="height_stratified",
        output=output,
        isaac_python=None,
        width=960,
        height=640,
        timeout_s=600.0,
        max_attempts=2,
        limit=None,
        resume=False,
        prepare_only=True,
    )
    first = _capture_review_batch(args)
    assert first["status"] == "PASS"
    index = json.loads((authority / "release_index.json").read_text(encoding="utf-8"))
    by_split = {layout["split"]: layout for layout in index["layouts"]}
    train = by_split["train"]
    calibration = by_split["calibration"]
    train_attempt = output / "scenes" / "train" / train["layout_id"] / "attempt-01"
    copied = tmp_path / "copied-attempt"
    shutil.copytree(train_attempt, copied)
    calibration_authority = _review_layout_authority(authority, calibration)
    assert _verified_prepared_attempt(copied, 8, calibration_authority) is None

    calibration_attempt = (
        output / "scenes" / "calibration" / calibration["layout_id"] / "attempt-01"
    )
    shutil.rmtree(calibration_attempt)
    shutil.copytree(train_attempt, calibration_attempt)
    resumed = _capture_review_batch(argparse.Namespace(**{**vars(args), "resume": True}))
    assert resumed["status"] == "PASS"
    calibration_job = next(job for job in resumed["jobs"] if job["split"] == "calibration")
    assert calibration_job["verified"]["attempt"] == "attempt-02"


def test_atomic_json_receipt_never_leaves_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    write_json_atomic(path, {"status": "first"})
    write_json_atomic(path, {"status": "second"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "second"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_batch_stops_after_host_failure_without_retrying_other_layouts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train", "calibration"),
        source_commit="1" * 40,
    )
    calls = 0

    def fail_host(_args: argparse.Namespace) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise HostGuardError("SetTokenInformation(TokenDefaultDacl): 1344")

    monkeypatch.setattr("aerocity_bench.cli._capture_review", fail_host)
    code = cli_main(
        [
            "capture-review-batch",
            str(authority),
            "--target-count",
            "8",
            "--output",
            str(tmp_path / "batch"),
            "--max-attempts",
            "3",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert calls == 1
    assert report["host_abort"] is True
    assert [job["status"] for job in report["jobs"]] == [
        "FAIL",
        "ABORTED_HOST_FAILURE",
    ]
    assert report["jobs"][0]["errors"][0]["category"] == (
        "execution_host_failure_not_scientific_failure"
    )


def test_batch_does_not_retry_deterministic_generation_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aerocity_bench.errors import GenerationRejected

    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train",),
        source_commit="4" * 40,
    )
    calls = 0

    def reject(_args: argparse.Namespace) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise GenerationRejected("deterministic invalid review draw")

    monkeypatch.setattr("aerocity_bench.cli._capture_review", reject)
    report = _capture_review_batch(
        argparse.Namespace(
            authority_root=authority,
            splits=["train"],
            target_count=8,
            process="height_stratified",
            output=tmp_path / "batch",
            isaac_python=None,
            width=640,
            height=480,
            timeout_s=600.0,
            max_attempts=3,
            limit=None,
            resume=False,
            prepare_only=True,
        )
    )
    assert report["status"] == "FAIL"
    assert calls == 1
    assert report["jobs"][0]["errors"][0]["category"] == (
        "deterministic_generation_rejection_not_retried"
    )


def test_capture_batch_writes_interrupt_progress_and_checks_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_root, bundle, asset_id = _fake_asset_bundle(tmp_path)
    config = _fake_config(tmp_path, bundle, asset_id)
    authority = tmp_path / "authority"
    build_ordinary_release(
        config,
        asset_root,
        authority,
        ("train",),
        source_commit="2" * 40,
    )
    base_args = argparse.Namespace(
        authority_root=authority,
        splits=["train"],
        target_count=8,
        process="height_stratified",
        output=tmp_path / "disk-batch",
        isaac_python=None,
        width=960,
        height=640,
        timeout_s=600.0,
        max_attempts=2,
        limit=None,
        resume=False,
        prepare_only=True,
    )
    monkeypatch.setattr(
        "aerocity_bench.cli.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=1, used=1, free=0),
    )
    with pytest.raises(ValueError, match="insufficient free space"):
        _capture_review_batch(base_args)

    interrupt_args = argparse.Namespace(
        **{**vars(base_args), "output": tmp_path / "interrupt-batch"}
    )
    monkeypatch.setattr(
        "aerocity_bench.cli.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=10**12, used=0, free=10**12),
    )
    monkeypatch.setattr(
        "aerocity_bench.cli._capture_review",
        lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _capture_review_batch(interrupt_args)
    progress = json.loads(
        (interrupt_args.output / "batch_progress.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "INTERRUPTED"
    assert progress["active_job"]["attempt"] == "attempt-01"
