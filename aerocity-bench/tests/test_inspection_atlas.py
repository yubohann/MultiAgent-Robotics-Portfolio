from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash
from aerocity_bench.compiler import compile_g2_i_task_spec, compile_method_task_spec
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    ATLAS_PRIOR_FULL,
    ATLAS_SCHEMA,
    TASK_TRACK_G1_U,
    TASK_TRACK_G2_I,
    compile_inspection_atlas,
    inspection_sampling_policy,
    project_inspection_atlas,
    validate_inspection_atlas_projection,
    validate_public_inspection_atlas,
)
from aerocity_bench.ordinary_config import load_ordinary_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def ordinary_config():
    return load_ordinary_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def development_city(ordinary_config):
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    for attempt in range(8):
        try:
            return generate_city_v3(ordinary_config, "train", 0, attempt, assets)
        except GenerationRejected:
            continue
    raise AssertionError("expected an admitted deterministic development city")


def test_g2_i_is_explicit_and_does_not_reclassify_g1_u(ordinary_config, development_city) -> None:
    g1 = compile_method_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )
    g2 = compile_g2_i_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )

    assert g1["schema"] == "org.aerocity.bench.task-spec-public.ordinary.v1"
    assert g1["task_track"] == TASK_TRACK_G1_U
    assert "inspection_atlas" not in g1
    assert g2["schema"] == "org.aerocity.bench.task-spec-public.g2-i.v1"
    assert g2["task_track"] == TASK_TRACK_G2_I
    assert g2["inspection_atlas"]["schema"] == ATLAS_SCHEMA
    assert g2["inspection_atlas"]["atlas_hash"]


def test_atlas_is_deterministic_and_ignores_private_episode_metadata(
    ordinary_config, development_city
) -> None:
    original = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    private_variant = copy.deepcopy(development_city)
    private_variant.update(
        {
            "target_process": "anisotropic_clustered_surface",
            "target_count": 999,
            "targets": [{"target_id": "target-999", "position": [1.0, 2.0, 3.0]}],
            "support_sites": [{"site_id": "site-private"}],
            "evaluator_private": {"legal_witnesses": [{"witness_id": "witness-private"}]},
        }
    )
    private_variant["buildings"][0]["components"][0]["target_support"] = False
    private_variant["obstacles"][0]["support_domain"] = False
    assert (
        compile_inspection_atlas(private_variant, ordinary_config.raw["execution_contract"])
        == original
    )


def test_atlas_contains_public_structure_classes_without_target_truth(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    text = json.dumps(atlas, sort_keys=True).lower()
    assert {region["region_class"] for region in atlas["regions"]} == {
        "roof",
        "facade",
        "entrance",
        "rubble",
    }
    assert sum(len(region["cells"]) for region in atlas["regions"]) > 0
    assert atlas["transit_graph"]["nodes"]
    for forbidden in ("target-", "site-", "witness", "evaluator", "distractor", "target_process"):
        assert forbidden not in text


def test_atlas_cells_reserve_public_cf2x_vertical_tracking_margin(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    vehicle = ordinary_config.raw["execution_contract"]["vehicle"]
    bound_buffer = float(atlas["sampling_policy"]["flight_bound_buffer_m"])
    expected_margin = (
        float(vehicle["radius_m"])
        + float(vehicle["minimum_clearance_m"])
        + bound_buffer
    )
    assert atlas["geometric_admission"]["flight_bound_margin_m"] == pytest.approx(
        expected_margin
    )
    minimum_z = float(development_city["flight_bounds"]["minimum"][2]) + expected_margin
    maximum_z = float(development_city["flight_bounds"]["maximum"][2]) - expected_margin
    for region in atlas["regions"]:
        for cell in region["cells"]:
            altitude = float(cell["pose"]["position"][2])
            assert minimum_z - 1.0e-9 <= altitude <= maximum_z + 1.0e-9


def test_transit_nodes_reserve_the_same_flight_bound_margin(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(
        development_city, ordinary_config.raw["execution_contract"]
    )
    vehicle = ordinary_config.raw["execution_contract"]["vehicle"]
    margin = (
        float(vehicle["radius_m"])
        + float(vehicle["minimum_clearance_m"])
        + float(atlas["sampling_policy"]["flight_bound_buffer_m"])
    )
    minimum = development_city["flight_bounds"]["minimum"]
    maximum = development_city["flight_bounds"]["maximum"]
    for node in atlas["transit_graph"]["nodes"]:
        position = node["position"]
        assert all(
            float(low) + margin - 1.0e-9
            <= float(coordinate)
            <= float(high) - margin + 1.0e-9
            for coordinate, low, high in zip(position, minimum, maximum, strict=True)
        )


def test_atlas_rejects_nested_private_field_and_hash_tampering(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    leaked = copy.deepcopy(atlas)
    leaked["regions"][0]["cells"][0]["pose"]["target_coordinates"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="prohibited key"):
        validate_public_inspection_atlas(leaked)

    tampered = copy.deepcopy(atlas)
    tampered["regions"][0]["represented_area_m2"] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_public_inspection_atlas(tampered)


def test_atlas_hash_changes_for_public_observation_contract(
    ordinary_config, development_city
) -> None:
    original = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    revised_contract = copy.deepcopy(ordinary_config.raw["execution_contract"])
    revised_contract["observe"]["horizontal_fov_deg"] += 4.0
    revised = compile_inspection_atlas(development_city, revised_contract)
    assert revised["atlas_hash"] != original["atlas_hash"]
    assert (
        revised["observation_contract"]["horizontal_cell_spacing_m"]
        != original["observation_contract"]["horizontal_cell_spacing_m"]
    )


def test_density_candidates_are_explicit_validated_and_do_not_mutate_default(
    ordinary_config, development_city
) -> None:
    default = compile_inspection_atlas(
        development_city, ordinary_config.raw["execution_contract"]
    )
    nominal_policy = inspection_sampling_policy(
        "g2-i-geometric-sampling-calibration-candidate-v2"
    )
    assert nominal_policy["calibration_status"] == "frozen"
    assert compile_inspection_atlas(
        development_city,
        ordinary_config.raw["execution_contract"],
        sampling_policy=nominal_policy,
    ) == default

    sparse = compile_inspection_atlas(
        development_city,
        ordinary_config.raw["execution_contract"],
        sampling_policy=inspection_sampling_policy(
            "g2-i-geometric-sampling-density-sparse-v1"
        ),
    )
    dense = compile_inspection_atlas(
        development_city,
        ordinary_config.raw["execution_contract"],
        sampling_policy=inspection_sampling_policy(
            "g2-i-geometric-sampling-density-dense-v1"
        ),
    )
    validate_public_inspection_atlas(sparse)
    validate_public_inspection_atlas(dense)
    assert sparse["sampling_policy"]["calibration_status"] == "ablation-only"
    assert dense["sampling_policy"]["calibration_status"] == "ablation-only"
    assert sparse["atlas_hash"] != dense["atlas_hash"]
    assert (
        sparse["observation_contract"]["horizontal_cell_spacing_m"]
        > default["observation_contract"]["horizontal_cell_spacing_m"]
        > dense["observation_contract"]["horizontal_cell_spacing_m"]
    )

    unregistered = inspection_sampling_policy(
        "g2-i-geometric-sampling-density-dense-v1"
    )
    unregistered["footprint_spacing_fraction"] = 0.49
    with pytest.raises(ValueError, match="recognized candidate"):
        compile_inspection_atlas(
            development_city,
            ordinary_config.raw["execution_contract"],
            sampling_policy=unregistered,
        )


def test_atlas_uses_a_target_independent_geometry_hash(ordinary_config, development_city) -> None:
    original = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    target_variant = copy.deepcopy(development_city)
    target_variant["task_geometry_hash"] = "f" * 64
    target_variant["buildings"][0]["components"][0]["target_support"] = False
    target_variant["obstacles"][0]["support_domain"] = False

    revised = compile_inspection_atlas(target_variant, ordinary_config.raw["execution_contract"])
    assert revised == original


def test_coarse_and_full_prior_projections_have_explicit_information_limits(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    coarse = project_inspection_atlas(atlas, ATLAS_PRIOR_COARSE)
    full = project_inspection_atlas(atlas, ATLAS_PRIOR_FULL)
    validate_inspection_atlas_projection(coarse)
    validate_inspection_atlas_projection(full)
    assert coarse == project_inspection_atlas(atlas, ATLAS_PRIOR_COARSE)
    assert full == project_inspection_atlas(atlas, ATLAS_PRIOR_FULL)
    assert "transit_graph" not in coarse
    assert "transit_graph" in full
    coarse_text = json.dumps(coarse, sort_keys=True)
    for forbidden_key in (
        '"cells"',
        '"pose"',
        '"surface_point"',
        '"surface_normal"',
        '"pose_envelope"',
    ):
        assert forbidden_key not in coarse_text
    assert '"pose"' in json.dumps(full, sort_keys=True)

    coarse_task = compile_g2_i_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
        inspection_prior_level=ATLAS_PRIOR_COARSE,
    )
    assert coarse_task["inspection_prior_level"] == ATLAS_PRIOR_COARSE
    assert "inspection_atlas" not in coarse_task
    assert coarse_task["inspection_atlas_projection"] == coarse


def test_disconnected_transit_graph_is_rejected_even_with_a_valid_hash(
    ordinary_config, development_city
) -> None:
    atlas = compile_inspection_atlas(development_city, ordinary_config.raw["execution_contract"])
    disconnected = copy.deepcopy(atlas)
    disconnected["transit_graph"]["edges"] = []
    disconnected.pop("atlas_hash")
    disconnected["atlas_hash"] = content_hash(disconnected)
    with pytest.raises(ValueError, match="disconnected"):
        validate_public_inspection_atlas(disconnected)
