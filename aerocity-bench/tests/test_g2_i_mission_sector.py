from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aerocity_bench.compiler import compile_g2_i_task_spec, compile_method_task_spec
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    TASK_TRACK_G2_I,
    project_inspection_atlas,
    validate_inspection_atlas_projection,
    validate_public_inspection_atlas,
    validate_public_mission_sector,
)
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.public_boundary import validate_public_episode
from aerocity_bench.targets_v3 import (
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def ordinary_config(tmp_path_factory):
    import json

    from aerocity_bench.canonical import write_json

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["admission"]["maximum_single_observation_target_fraction"] = 1.0
    path = tmp_path_factory.mktemp("g2i-config") / "ordinary.json"
    write_json(path, raw)
    return load_ordinary_config(path)


@pytest.fixture(scope="module")
def development_city(ordinary_config):
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    for attempt in range(8):
        try:
            return generate_city_v3(ordinary_config, "train", 0, attempt, assets)
        except GenerationRejected:
            continue
    raise AssertionError("expected an admitted deterministic development city")


@pytest.fixture(scope="module")
def g2_i_episode(ordinary_config, development_city):
    task_spec = compile_g2_i_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )
    sites = derive_support_sites_v3(development_city, ordinary_config)
    episode = sample_episode_v3(
        ordinary_config,
        development_city,
        sites,
        0,
        public_task_spec=task_spec,
    )
    return task_spec, episode


def test_g2_i_task_compiles_a_full_atlas_and_mission_sector(
    ordinary_config, development_city, g2_i_episode
) -> None:
    task_spec, episode = g2_i_episode
    atlas = task_spec["inspection_atlas"]
    sector = episode["mission_sector"]

    assert task_spec["task_track"] == TASK_TRACK_G2_I
    assert atlas["layout_id"] == development_city["layout_id"]
    validate_public_inspection_atlas(atlas)
    validate_public_mission_sector(
        sector,
        atlas,
        episode["starts"],
        ordinary_config.raw["execution_contract"],
    )
    assert sector["layout_id"] == atlas["layout_id"]
    assert sector["truth_independent"] is True
    assert sector["cell_count"] == len(sector["selected_cell_ids"])
    assigned = [
        cell_id
        for cell_ids in sector["cell_assignment_by_drone"].values()
        for cell_id in cell_ids
    ]
    assert sorted(assigned) == sorted(sector["selected_cell_ids"])
    assert set(sector["cell_assignment_by_drone"]) == {
        str(start["drone_id"]) for start in episode["starts"]
    }


def test_public_episode_projection_carries_the_sector_without_private_truth(
    ordinary_config, g2_i_episode
) -> None:
    task_spec, episode = g2_i_episode
    projection = public_episode_projection(episode)
    dumped = json.dumps(projection, sort_keys=True)

    assert "mission_sector" in projection
    assert "targets" not in dumped
    assert "site_id" not in dumped
    assert "witness" not in dumped
    validate_public_episode(projection, task_spec)


def test_sector_must_be_bound_to_its_atlas_layout(ordinary_config, g2_i_episode) -> None:
    task_spec, episode = g2_i_episode
    atlas = task_spec["inspection_atlas"]
    forged = copy.deepcopy(episode["mission_sector"])
    forged["layout_id"] = "city-other-000-a0"

    with pytest.raises(ValueError, match="target-independent atlas"):
        validate_public_mission_sector(
            forged,
            atlas,
            episode["starts"],
            ordinary_config.raw["execution_contract"],
        )


def test_sector_rejects_an_assignment_that_drops_a_cell(ordinary_config, g2_i_episode) -> None:
    task_spec, episode = g2_i_episode
    atlas = task_spec["inspection_atlas"]
    forged = copy.deepcopy(episode["mission_sector"])
    first_drone = sorted(forged["cell_assignment_by_drone"])[0]
    forged["cell_assignment_by_drone"][first_drone] = forged["cell_assignment_by_drone"][
        first_drone
    ][:-1]

    with pytest.raises(ValueError, match="does not cover"):
        validate_public_mission_sector(
            forged,
            atlas,
            episode["starts"],
            ordinary_config.raw["execution_contract"],
        )


def test_sector_rejects_unknown_or_duplicate_obligations(ordinary_config, g2_i_episode) -> None:
    task_spec, episode = g2_i_episode
    atlas = task_spec["inspection_atlas"]
    unknown = copy.deepcopy(episode["mission_sector"])
    unknown["selected_cell_ids"] = ["atlas-cell-unknown"]
    unknown["cell_count"] = 1
    with pytest.raises(ValueError, match="unknown or duplicate"):
        validate_public_mission_sector(
            unknown,
            atlas,
            episode["starts"],
            ordinary_config.raw["execution_contract"],
        )

    duplicated = copy.deepcopy(episode["mission_sector"])
    duplicated["selected_cell_ids"] = duplicated["selected_cell_ids"] + [
        duplicated["selected_cell_ids"][0]
    ]
    duplicated["cell_count"] = len(duplicated["selected_cell_ids"])
    with pytest.raises(ValueError, match="unknown or duplicate"):
        validate_public_mission_sector(
            duplicated,
            atlas,
            episode["starts"],
            ordinary_config.raw["execution_contract"],
        )


def test_coarse_projection_is_versioned_and_withholds_full_geometry(
    ordinary_config, development_city
) -> None:
    task_spec = compile_method_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )
    assert "inspection_atlas" not in task_spec

    full_task = compile_g2_i_task_spec(
        development_city,
        ordinary_config.raw["execution_contract"],
        ordinary_config.raw["fleet"],
    )
    atlas = full_task["inspection_atlas"]
    coarse = project_inspection_atlas(atlas, ATLAS_PRIOR_COARSE)
    validate_inspection_atlas_projection(coarse)
    dumped = json.dumps(coarse, sort_keys=True)
    for forbidden in ('"cells"', '"pose"', '"surface_point"', '"transit_graph"'):
        assert forbidden not in dumped
    assert coarse["layout_id"] == atlas["layout_id"]
