"""Regression gates for the target-independent G2-I mission sector."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aerocity_bench.atlas_leakage import audit_atlas_leakage
from aerocity_bench.baselines import create_baseline
from aerocity_bench.canonical import content_hash
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.contracts import ActionPacket
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.g2_i_rl import (
    G2_I_CONFIRMATION_ONLY_REWARD_V1,
    G2_I_INSPECTION_SHAPED_REWARD_V1,
    G2_I_RL_CONTEXT_SCHEMA,
    G2IGymnasiumFleetWrapper,
)
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    MISSION_SECTOR_SCHEMA,
    compile_inspection_atlas,
    compile_public_mission_sector,
    inspection_sampling_policy,
    validate_public_mission_sector,
)
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import (
    _episode_condition,
    _mission_sector_support_sites,
    _starts,
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
)
from tools.build_g2_i_density_ablation_manifest import _matched_region_sectors

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def sector_fixture():
    config = load_ordinary_config(CONFIG_PATH)
    assets = list(config.raw["assets"]["allowlist"])
    for attempt in range(32):
        try:
            city = generate_city_v3(config, "calibration", 0, attempt, assets)
            task = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            sites = derive_support_sites_v3(city, config)
            episode = sample_episode_v3(
                config, city, sites, 0, public_task_spec=task
            )
            return config, city, task, episode
        except GenerationRejected:
            continue
    raise AssertionError("expected a calibration city with an admissible G2-I sector")


def test_sector_is_versioned_public_and_has_budget_certificate(sector_fixture) -> None:
    config, city, task, episode = sector_fixture
    sector = episode["mission_sector"]
    assert sector["schema"] == MISSION_SECTOR_SCHEMA
    assert sector["truth_independent"] is True
    assert sector["frozen_before_sampling"] is True
    assert sector["cell_count"] > 0
    assert len(sector["altitude_bands"]) >= 3
    assert sector["capacity_certificate"]["all_lower_bounds_fit"] is True
    certificate = sector["capacity_certificate"]
    assert certificate["calibration_status"] == "frozen"
    assert certificate["capacity_fraction"] == pytest.approx(1.0)
    assert certificate["capacity_limit_s"] == pytest.approx(
        config.raw["execution_contract"]["episode"]["duration_s"]
    )
    assert all(
        value <= certificate["capacity_limit_s"]
        for value in certificate["per_drone_required_lower_bound_s"].values()
    )
    validate_public_mission_sector(
        sector,
        task["inspection_atlas"],
        episode["starts"],
        config.raw["execution_contract"],
    )
    public = public_episode_projection(episode)
    assert public["mission_sector"] == sector
    assert "target_count" not in public
    assert "targets" not in public
    assert city["layout_id"] == episode["layout_id"]


def test_g2_i_sector_episode_cannot_use_the_legacy_runtime_entrypoint(sector_fixture) -> None:
    config, city, _task, episode = sector_fixture
    with pytest.raises(ValueError, match="method-visible public task spec"):
        L0FleetRuntime(config, city, episode)


def _hover_actions(observations: dict[str, dict[str, object]]) -> dict[str, ActionPacket]:
    actions = {}
    for drone_id, projected in observations.items():
        agent = projected["agent"]
        assert isinstance(agent, dict)
        actions[drone_id] = ActionPacket(
            episode_id=str(agent["episode_id"]),
            drone_id=drone_id,
            sequence=int(agent["sequence"]),
            issued_at_s=float(agent["timestamp_s"]),
            kind="HOVER",
        )
    return actions


def _g2_i_wrapper(sector_fixture, reward_contract=G2_I_INSPECTION_SHAPED_REWARD_V1):
    config, city, task, episode = sector_fixture
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        public_task_spec=task,
        public_episode=public_episode_projection(episode),
    )
    return runtime, G2IGymnasiumFleetWrapper(runtime, reward_contract=reward_contract)


def test_g2_i_rl_wrapper_exposes_only_fixed_public_high_level_state(sector_fixture) -> None:
    runtime, wrapper = _g2_i_wrapper(sector_fixture)
    observations, info = wrapper.reset()
    context = info["public_context"]
    assert context["schema"] == G2_I_RL_CONTEXT_SCHEMA
    assert context["context_hash"] == content_hash(
        {key: value for key, value in context.items() if key != "context_hash"}
    )
    assert len(context["cell_handles"]) == len(
        runtime.public_episode["mission_sector"]["selected_cell_ids"]
    )
    assert len(set(context["cell_handles"])) == len(context["cell_handles"])
    serialized = json.dumps({"context": context, "observations": observations})
    for forbidden in ("targets", "target_count", "target_process", "witness", "evaluator"):
        assert forbidden not in serialized
    assert all(
        len(observation["inspection_history"]["visited_cell_mask"])
        == len(context["cell_handles"])
        for observation in observations.values()
    )


def test_g2_i_rl_step_batches_control_ticks_and_audits_reward(sector_fixture) -> None:
    runtime, wrapper = _g2_i_wrapper(sector_fixture)
    observations, _ = wrapper.reset()
    next_observations, rewards, terminated, truncated, info = wrapper.step(
        _hover_actions(observations)
    )
    fleet_count = len(observations)
    assert info["control_ticks_executed"] == 5
    assert runtime.task_time_s == pytest.approx(1.0)
    receipts = runtime.result()["execution_receipts"]
    assert len(receipts) == 5 * fleet_count
    assert sum(receipt["planner_invoked"] for receipt in receipts) == fleet_count
    assert all(
        receipt["planning_latency_s"] == 0.0
        for receipt in receipts
        if not receipt["planner_invoked"]
    )
    assert set(info["reward_components"]) == set(
        G2_I_INSPECTION_SHAPED_REWARD_V1.weights
    )
    assert info["reward_contract"]["uses_private_truth"] is False
    assert info["reward_contract"]["is_benchmark_score"] is False
    assert len(set(rewards.values())) == 1
    assert not any(terminated.values())
    assert not any(truncated.values())
    assert all(
        observation["agent"]["sequence"] == 5
        for observation in next_observations.values()
    )


def test_g2_i_rl_reset_clears_episode_local_cadence_and_history(sector_fixture) -> None:
    _, wrapper = _g2_i_wrapper(sector_fixture)
    observations, _ = wrapper.reset()
    wrapper.step(_hover_actions(observations))
    reset_observations, _ = wrapper.reset()
    _, _, _, _, info = wrapper.step(_hover_actions(reset_observations))
    assert info["control_ticks_executed"] == 5
    assert info["planning_trigger_reasons"] == ["fixed_period", "initial"]


def test_g2_i_rl_reward_ablation_keeps_components_and_changes_only_weights(
    sector_fixture,
) -> None:
    _, shaped = _g2_i_wrapper(
        sector_fixture, reward_contract=G2_I_INSPECTION_SHAPED_REWARD_V1
    )
    shaped_observations, _ = shaped.reset()
    _, shaped_rewards, _, _, shaped_info = shaped.step(
        _hover_actions(shaped_observations)
    )

    _, confirmation_only = _g2_i_wrapper(
        sector_fixture, reward_contract=G2_I_CONFIRMATION_ONLY_REWARD_V1
    )
    confirmation_observations, _ = confirmation_only.reset()
    _, confirmation_rewards, _, _, confirmation_info = confirmation_only.step(
        _hover_actions(confirmation_observations)
    )

    assert shaped_info["reward_components"] == confirmation_info["reward_components"]
    assert shaped_info["reward_contract"]["weights"] != confirmation_info[
        "reward_contract"
    ]["weights"]
    assert next(iter(shaped_rewards.values())) < next(
        iter(confirmation_rewards.values())
    )


def test_g2_i_rl_hover_transition_is_invariant_to_private_target_process(
    sector_fixture,
) -> None:
    config, city, task, episode = sector_fixture
    variant = copy.deepcopy(episode)
    variant["target_process"] = "private-counterfactual-process"
    variant.pop("episode_hash")
    variant["episode_hash"] = content_hash(variant)
    assert variant["episode_hash"] != episode["episode_hash"]
    assert public_episode_projection(variant) == public_episode_projection(episode)

    wrappers = []
    for private_episode in (episode, variant):
        runtime = L0FleetRuntime(
            config,
            city,
            private_episode,
            public_task_spec=task,
            public_episode=public_episode_projection(private_episode),
        )
        wrappers.append(G2IGymnasiumFleetWrapper(runtime))
    original_observations, original_info = wrappers[0].reset()
    variant_observations, variant_info = wrappers[1].reset()
    assert original_observations == variant_observations
    assert original_info == variant_info

    original_step = wrappers[0].step(_hover_actions(original_observations))
    variant_step = wrappers[1].step(_hover_actions(variant_observations))
    assert original_step == variant_step


def test_sector_capacity_is_recomputed_from_public_assignment(sector_fixture) -> None:
    config, _, task, episode = sector_fixture
    sector = copy.deepcopy(episode["mission_sector"])
    assert set(sector["cell_assignment_by_drone"]) == {
        start["drone_id"] for start in episode["starts"]
    }

    # Re-hashing a forged certificate must not bypass the route-budget gate.
    first_drone = sorted(sector["cell_assignment_by_drone"])[0]
    sector["capacity_certificate"]["per_drone_required_lower_bound_s"][first_drone] -= 1.0
    sector.pop("sector_hash")
    sector["sector_hash"] = content_hash(sector)
    with pytest.raises(ValueError, match="capacity certificate is not reproducible"):
        validate_public_mission_sector(
            sector,
            task["inspection_atlas"],
            episode["starts"],
            config.raw["execution_contract"],
        )


def test_sector_assignment_cannot_drop_a_cell_after_rehashing(sector_fixture) -> None:
    config, _, task, episode = sector_fixture
    sector = copy.deepcopy(episode["mission_sector"])
    first_drone = sorted(sector["cell_assignment_by_drone"])[0]
    removed = sector["cell_assignment_by_drone"][first_drone].pop()
    assert removed in sector["selected_cell_ids"]
    sector.pop("sector_hash")
    sector["sector_hash"] = content_hash(sector)
    with pytest.raises(ValueError, match="assignment does not cover selected cells"):
        validate_public_mission_sector(
            sector,
            task["inspection_atlas"],
            episode["starts"],
            config.raw["execution_contract"],
        )


def test_legacy_mission_sector_schema_is_rejected_after_rehashing(sector_fixture) -> None:
    config, _, task, episode = sector_fixture
    sector = copy.deepcopy(episode["mission_sector"])
    sector["schema"] = "org.aerocity.bench.inspection-mission-sector-public.v1"
    sector.pop("sector_hash")
    sector["sector_hash"] = content_hash(sector)
    with pytest.raises(ValueError, match="mission sector schema differs"):
        validate_public_mission_sector(
            sector,
            task["inspection_atlas"],
            episode["starts"],
            config.raw["execution_contract"],
        )


def test_sector_selection_does_not_change_with_private_target_metadata(sector_fixture) -> None:
    config, city, task, episode = sector_fixture
    _, group_seed, _ = _episode_condition(config, city, 0)
    starts = _starts(city, config, config.fleet_count, group_seed)
    original = compile_public_mission_sector(
        task["inspection_atlas"], starts, config.raw["execution_contract"]
    )
    variant = copy.deepcopy(city)
    variant["targets"] = [{"target_id": "private-only", "position": [1.0, 2.0, 3.0]}]
    variant["target_process"] = "private-process-only"
    revised = compile_public_mission_sector(
        compile_inspection_atlas(variant, config.raw["execution_contract"]),
        starts,
        config.raw["execution_contract"],
    )
    assert revised == original
    assert episode["mission_sector"] == original


def test_precomputed_private_sector_sites_preserve_frozen_episode_bytes(sector_fixture) -> None:
    config, city, task, episode = sector_fixture
    sites = _mission_sector_support_sites(
        city,
        task,
        episode["mission_sector"],
        episode["starts"],
        config,
    )
    replay = sample_episode_v3(
        config,
        city,
        [],
        0,
        public_task_spec=task,
        precomputed_mission_sector=episode["mission_sector"],
        precomputed_mission_sector_sites=sites,
    )
    assert replay == episode


def test_counterfactual_distractors_share_the_public_mission_region(sector_fixture) -> None:
    config, city, task, episode = sector_fixture
    sites = _mission_sector_support_sites(
        city,
        task,
        episode["mission_sector"],
        episode["starts"],
        config,
    )
    region_by_site = {
        str(site["site_id"]): str(site["_mission_region_id"]) for site in sites
    }
    assert all(
        region_by_site[str(pair["target_site_id"])]
        == region_by_site[str(pair["distractor_site_id"])]
        for pair in episode["counterfactual_pairs"]
    )


def test_density_conditions_freeze_one_common_public_region_cohort(sector_fixture) -> None:
    config, city, _, episode = sector_fixture
    policy_ids = (
        "g2-i-geometric-sampling-density-sparse-v1",
        "g2-i-geometric-sampling-calibration-candidate-v2",
        "g2-i-geometric-sampling-density-dense-v1",
    )
    task_specs = {
        policy_id: compile_g2_i_task_spec(
            city,
            config.raw["execution_contract"],
            config.raw["fleet"],
            inspection_sampling_policy=inspection_sampling_policy(policy_id),
        )
        for policy_id in policy_ids
    }
    sectors, common_regions = _matched_region_sectors(
        task_specs,
        episode["starts"],
        config.raw["execution_contract"],
        set(episode["mission_sector"]["selected_region_ids"]),
    )
    assert common_regions == set(episode["mission_sector"]["selected_region_ids"])
    assert all(
        set(sector["selected_region_ids"]) == common_regions
        for sector in sectors.values()
    )


def test_coarse_region_inspector_cannot_consume_full_cell_sector(sector_fixture) -> None:
    config, city, _, episode = sector_fixture
    coarse_task = compile_g2_i_task_spec(
        city,
        config.raw["execution_contract"],
        config.raw["fleet"],
        inspection_prior_level=ATLAS_PRIOR_COARSE,
    )
    public = public_episode_projection(episode)
    with pytest.raises(ValueError, match="public mission sector requires the full"):
        create_baseline(
            "atlas-coarse-region-inspector", config, coarse_task, public
        )
    coarse_public = {
        key: value
        for key, value in public.items()
        if key not in {"mission_sector", "mission_sector_hash"}
    }
    coarse_public["coarse_region_ids"] = list(
        episode["mission_sector"]["selected_region_ids"]
    )
    policy = create_baseline(
        "atlas-coarse-region-inspector", config, coarse_task, coarse_public
    )
    assert any(policy.routes.values())
    serialized = json.dumps(coarse_task, sort_keys=True)
    for forbidden in ('"cells"', '"surface_point"', '"surface_normal"', '"pose_envelope"'):
        assert forbidden not in serialized


def test_sector_only_leakage_probe_is_grouped_and_private_safe(sector_fixture) -> None:
    config, _, task, episode = sector_fixture
    records = []
    processes = ("uniform_surface", "clustered_surface", "height_stratified")
    for ancestor_index in range(4):
        atlas = copy.deepcopy(task["inspection_atlas"])
        atlas["layout_id"] = f"sector-unit-layout-{ancestor_index}"
        atlas["inspection_geometry_hash"] = content_hash(
            ["sector-unit-geometry", ancestor_index]
        )
        atlas.pop("atlas_hash")
        atlas["atlas_hash"] = content_hash(atlas)
        for process in processes:
            private = copy.deepcopy(episode)
            private["target_process"] = process
            private["mission_sector"]["atlas_hash"] = atlas["atlas_hash"]
            private["mission_sector"].pop("sector_hash")
            private["mission_sector"]["sector_hash"] = content_hash(
                private["mission_sector"]
            )
            private["mission_sector_hash"] = private["mission_sector"]["sector_hash"]
            private.pop("episode_hash")
            private["episode_hash"] = content_hash(private)
            records.append(
                {
                    "atlas": atlas,
                    "private_episode": private,
                    "layout_ancestor": f"sector-unit-ancestor-{ancestor_index}",
                    "split_label": "development",
                }
            )
    report = audit_atlas_leakage(
        records,
        execution_contract=config.raw["execution_contract"],
        permutation_count=16,
    )
    assert report["sector_process_label_probe"]["status"] == "PASS_NO_DETECTED_SIGNAL"
    assert report["sector_contract"]["sector_record_count"] == 12
    assert report["sector_contract"][
        "target_distractor_pairs_checked_inside_sector"
    ] == 12 * len(episode["counterfactual_pairs"])
    serialized = json.dumps(report, sort_keys=True)
    assert not any(target["target_id"] in serialized for target in episode["targets"])


def test_leakage_probe_rejects_rehashed_forged_sector_certificate(sector_fixture) -> None:
    config, _, task, episode = sector_fixture
    private = copy.deepcopy(episode)
    private["mission_sector"]["capacity_certificate"]["capacity_fraction"] = 1.35
    private["mission_sector"].pop("sector_hash")
    private["mission_sector"]["sector_hash"] = content_hash(private["mission_sector"])
    private["mission_sector_hash"] = private["mission_sector"]["sector_hash"]
    private.pop("episode_hash")
    private["episode_hash"] = content_hash(private)
    with pytest.raises(ValueError, match="mission sector capacity policy differs"):
        audit_atlas_leakage(
            [
                {
                    "atlas": task["inspection_atlas"],
                    "private_episode": private,
                    "layout_ancestor": "forged-sector-ancestor",
                    "split_label": "calibration",
                }
            ],
            execution_contract=config.raw["execution_contract"],
            permutation_count=16,
        )
