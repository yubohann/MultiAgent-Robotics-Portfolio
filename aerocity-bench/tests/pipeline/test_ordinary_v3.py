from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerocity_bench.cli import main as cli_main
from aerocity_bench.core.canonical import write_json
from aerocity_bench.core.errors import GenerationRejected, ValidationError
from aerocity_bench.generation.builder_v3 import (
    MAX_ATTEMPTS_PER_LAYOUT,
    build_ordinary_release,
    export_public_release,
    validate_ordinary_release,
    validate_public_release,
)
from aerocity_bench.generation.compiler import (
    compile_method_task_spec,
    compile_scene,
    public_cityspec_v3,
)
from aerocity_bench.generation.generator_v3 import generate_city_v3
from aerocity_bench.generation.ordinary_config import (
    FORMAL_SPLITS,
    ORDINARY_SPLITS,
    OrdinaryReleaseConfig,
    load_ordinary_config,
    load_public_runtime_contract,
    public_execution_contract,
    validate_public_execution_contract,
)
from aerocity_bench.generation.targets_v3 import (
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
)
from aerocity_bench.release.scene_audit import audit_generated_city
from aerocity_bench.release.supply_chain import load_official_cc0_lock
from aerocity_bench.runtime.baselines import BASELINES, create_baseline
from aerocity_bench.runtime.metrics import evaluate_run
from aerocity_bench.runtime.runtime import L0FleetRuntime

ROOT = Path(__file__).resolve().parents[2]
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
    for attempt in range(MAX_ATTEMPTS_PER_LAYOUT):
        try:
            return generate_city_v3(config, split, index, attempt, assets)
        except GenerationRejected:
            continue
    raise AssertionError(f"no admitted city for {split}[{index}]")


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
    city, _ = city_and_sites
    public_contract = public_execution_contract(ordinary_config.raw["execution_contract"])
    validate_public_execution_contract(public_contract)
    assert "fixed_target_count_private" not in public_contract["episode"]

    task = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    assert task["execution_contract"] == public_contract
    assert "fixed_target_count_private" not in json.dumps(task, sort_keys=True)

    runtime_contract = {
        "schema": "org.aerocity.bench.runtime-contract-public.ordinary.v1",
        "release_version": ordinary_config.version,
        "generator_version": ordinary_config.generator_version,
        "fleet": ordinary_config.raw["fleet"],
        "execution_contract": public_contract,
    }
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


def test_generate_city_v3_is_deterministic_with_explicit_ids(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, _ = city_and_sites
    assets = list(ordinary_config.raw["assets"]["allowlist"])
    rebased = generate_city_v3(ordinary_config, "train", 0, 0, assets)
    assert rebased["layout_id"] == city["layout_id"]
    assert rebased["task_geometry_id"] == city["task_geometry_id"]
    assert rebased["roads"] == city["roads"]
    assert rebased["buildings"] == city["buildings"]
    assert city["layout_id"].startswith("city-train-000")
    assert city["task_geometry_id"].endswith("-task")


def test_public_cityspec_projection_hides_private_fields(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, _ = city_and_sites
    public = public_cityspec_v3(city)
    dumped = json.dumps(public, sort_keys=True)
    for private_key in ("generation_seed", "split", "family_private", "spawn_grammar"):
        assert private_key not in dumped
    assert public["layout_id"] == city["layout_id"]
    assert public["task_geometry_id"] == city["task_geometry_id"]
    assert public["asset_set_id"] == city["asset_set_id"]


def test_support_sites_and_episode_contract(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, sites = city_and_sites
    assert sites
    assert all(site["legal_witness_count"] >= 1 for site in sites)
    episode = sample_episode_v3(ordinary_config, city, sites, 0)
    assert episode["layout_id"] == city["layout_id"]
    assert int(episode["target_count"]) == len(episode["targets"])
    assert len(episode["distractors"]) == len(episode["targets"])
    projection = public_episode_projection(episode)
    dumped = json.dumps(projection, sort_keys=True)
    assert "targets" not in dumped
    assert "site_id" not in dumped
    assert projection["target_count_public"] is False


def test_scene_audit_reports_private_safe_admission(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, _ = city_and_sites
    report = audit_generated_city(ordinary_config, city)
    assert report["status"] in {"PASS", "FAIL"}
    dumped = json.dumps(report, sort_keys=True)
    assert "targets" not in dumped
    assert "site_id" not in dumped


def test_l0_baseline_runtime_and_metrics(
    ordinary_config: OrdinaryReleaseConfig, city_and_sites
) -> None:
    city, sites = city_and_sites
    episode = sample_episode_v3(ordinary_config, city, sites, 0)
    task_spec = compile_method_task_spec(
        city, ordinary_config.raw["execution_contract"], ordinary_config.raw["fleet"]
    )
    policy = create_baseline(
        "random-safe",
        ordinary_config,
        task_spec,
        public_episode_projection(episode),
        private_episode=episode,
    )
    runtime = L0FleetRuntime(
        ordinary_config,
        city,
        episode,
        public_task_spec=task_spec,
        public_episode=public_episode_projection(episode),
    )
    run_result = runtime.run_policy(policy, max_steps=3)
    assert run_result["execution_level"] == "L0"
    assert run_result["execution_receipts"]
    assert "planning_latency_s" in run_result["execution_receipts"][0]
    duration = float(ordinary_config.raw["execution_contract"]["episode"]["duration_s"])
    report = evaluate_run(run_result, episode, duration)
    assert 0.0 <= report["quality"]["final_confirmed_recall"] <= 1.0


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
                        "bytes": model.stat().st_size,
                    }
                ],
            }
        ]
    }
    write_json(bundle / "ASSET_REGISTRY.json", registry)
    evidence_urls = {
        "source_page": "https://polyhaven.com/a/simple_asset",
        "info_api": "https://api.polyhaven.com/info/simple_asset",
        "files_api": "https://api.polyhaven.com/files/simple_asset",
    }
    manifest = {
        "failures": [],
        "verification_summary": {"all_file_and_url_checks_passed": True},
        "global_evidence": {
            "polyhaven_license": {"snapshot_path": str(license_snapshot)}
        },
        "assets": [
            {
                "asset_id": "simple_asset",
                "creator_names": ["Test Creator"],
                "official_evidence": {
                    evidence_name: {
                        "snapshot_path": str(snapshot),
                        "requested_url": evidence_urls[evidence_name],
                        "retrieved_at_utc": "2026-07-29T00:00:00Z",
                        "http_status": 200,
                    }
                    for evidence_name, snapshot in official_snapshots.items()
                },
            }
        ],
    }
    write_json(bundle / "provenance" / "PROVENANCE_MANIFEST.json", manifest)
    return asset_root, bundle_name, "simple_asset"


def _fake_config(tmp_path: Path, bundle: str, asset_id: str) -> OrdinaryReleaseConfig:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["release_version"] = "test-ordinary-v3"
    raw["assets"]["bundle"] = bundle
    raw["assets"]["allowlist"] = [asset_id]
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
    assert 'def Xform "ProceduralVisualDetail"' in scene
    assert "bool physics:collisionEnabled = false" in scene
    assert collision.count('"PhysicsCollisionAPI"') == expected_collision_count
    assert "UrbanGroundDetail" not in collision


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
    public = tmp_path / "public"
    export_public_release(output, public)
    assert validate_public_release(public)["status"] == "PASS"
    assert (
        cli_main(
            [
                "run-baseline",
                str(public),
                "--method",
                "random-safe",
                "--split",
                "train",
                "--max-steps",
                "1",
                "--output",
                str(tmp_path / "run-public"),
            ]
        )
        == 0
    )
    (output / "unmanifested-file.txt").write_text("must be rejected\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="manifest file set differs"):
        validate_ordinary_release(output)


def test_baselines_are_registered() -> None:
    assert "random-safe" in BASELINES
    assert BASELINES["random-safe"].requires_private_truth is False
