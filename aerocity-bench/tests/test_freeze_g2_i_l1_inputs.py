from __future__ import annotations

from pathlib import Path

from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.targets_v3 import derive_support_sites_v3, sample_episode_v3
from tools.freeze_g2_i_l1_inputs import FROZEN_INPUT_SOURCE, freeze_inputs


def _calibration_city(config, index: int) -> dict:
    for attempt in range(64):
        try:
            city = generate_city_v3(
                config, "calibration", index, attempt, config.raw["assets"]["allowlist"]
            )
            task_spec = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            sample_episode_v3(
                config,
                city,
                derive_support_sites_v3(city, config),
                0,
                public_task_spec=task_spec,
            )
            return city
        except (GenerationRejected, ValueError):
            continue
    raise AssertionError("failed to generate an admitted calibration CitySpec")


def test_freeze_l1_inputs_precommits_three_distinct_calibration_episodes(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config_path = root / "configs" / "releases" / "ordinary-v1-mini.json"
    config = load_ordinary_config(config_path)
    source_root = tmp_path / "source"
    manifest_root = tmp_path / "manifest"
    records = []
    for index in range(3):
        city_path = source_root / f"city-{index:02d}.json"
        city = _calibration_city(config, index)
        private_city_path = source_root / f"private-city-{index:02d}.json"
        write_json(private_city_path, city)
        public_city = dict(city)
        public_city.pop("split")
        public_city.pop("spawn_grammar")
        write_json(city_path, public_city)
        records.append(
            {
                "layout_ancestor": f"g2-i-calibration-ancestor-{index:02d}",
                "city_path": city_path.name,
                "private_city_source_path": private_city_path.name,
                "private_city_source_sha256": content_hash(city),
                "private_episode_path": f"retired-episode-{index:02d}.json",
                "split_label": "calibration",
            }
        )
    source = {
        "schema": "org.aerocity.bench.g2-i-scientific-audit-manifest.v1",
        "purpose": "test-source",
        "formal_score_eligible": False,
        "self_method_results_used": False,
        "development_splits": ["calibration"],
        "accepted_ancestor_count": 3,
        "records": records,
    }
    source["manifest_hash"] = content_hash(source)
    source_path = manifest_root / "source-manifest.json"
    write_json(source_path, source)

    report = freeze_inputs(
        source_path,
        config_path,
        tmp_path / "frozen",
        source_root=source_root,
    )

    assert report["formal_score_eligible"] is False
    assert report["private_episode_source"] == FROZEN_INPUT_SOURCE
    assert report["requested_condition_index"] is None
    assert report["process_count"] == 3
    assert len(report["records"]) == 3
    assert len({item["layout_hash"] for item in report["records"]}) == 3
    assert report["manifest_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    for record in report["records"]:
        episode = read_json(tmp_path / "frozen" / record["private_episode_path"])
        assert content_hash(episode) == record["private_episode_sha256"]
        assert record["private_city_source_sha256"] != record["city_source_sha256"]
        assert record["episode_index"] % report["process_count"] == record["condition_index"]
