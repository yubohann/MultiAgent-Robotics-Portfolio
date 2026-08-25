from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerocity_bench.assets import ACCEPTED_SPDX
from aerocity_bench.audit import FORBIDDEN_PUBLIC_KEYS, validate_release
from aerocity_bench.builder import build_release
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.config import EXPECTED_SPLITS, OBSERVATION_TIERS, load_release_config
from aerocity_bench.errors import ValidationError
from aerocity_bench.generator import generate_city
from aerocity_bench.targets import derive_support_sites, sample_episode

ASSET_IDS = {
    "street_lamp_01",
    "concrete_road_barrier",
    "tree_small_02",
    "street_lamp_02",
    "concrete_road_barrier_02",
    "pine_tree_01",
}


def _walk_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(str(key))
            result.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_walk_keys(child))
    return result


class BenchmarkReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.asset_root = cls.root / "assets"
        bundle = cls.asset_root / "open_city_cc0_assets_20260729"
        records = []
        for asset_id in sorted(ASSET_IDS):
            relative = f"usd_models/{asset_id}/{asset_id}.usda"
            path = bundle.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '#usda 1.0\n( defaultPrim = "Asset" )\ndef Xform "Asset" {}\n',
                encoding="utf-8",
            )
            records.append(
                {
                    "asset_id": asset_id,
                    "role": "test_visual",
                    "kind": "usd_model",
                    "spdx": "CC0-1.0",
                    "redistribution_allowed": True,
                    "files": [
                        {"path": relative, "sha256": file_hash(path), "bytes": path.stat().st_size}
                    ],
                }
            )
        write_json(
            bundle / "ASSET_REGISTRY.json",
            {"registry_version": "test-v1", "assets": records},
        )
        repo_root = Path(__file__).resolve().parents[1]
        cls.config = load_release_config(repo_root / "configs" / "releases" / "smoke.json")
        cls.release = cls.root / "release"
        cls.report = build_release(cls.config, cls.asset_root, cls.release)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.workspace.cleanup()

    def test_release_passes_integrity_privacy_process_and_fault_audits(self) -> None:
        report = validate_release(self.release, write_report=False)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["layout_count"], self.config.total_layouts)
        self.assertEqual(report["scientific_status"], "pilot_only")
        self.assertEqual(report["target_process_gate"], "passed")
        self.assertEqual(report["fault_pairing_gate"], "passed")

    def test_city_and_episode_generation_are_deterministic(self) -> None:
        first = generate_city(self.config, "train", 0, 0, ["street_lamp_01"])
        second = generate_city(self.config, "train", 0, 0, ["street_lamp_01"])
        self.assertEqual(first, second)
        changed = generate_city(self.config, "train", 0, 1, ["street_lamp_01"])
        self.assertNotEqual(first["layout_hash"], changed["layout_hash"])
        sites = derive_support_sites(first)
        self.assertEqual(
            sample_episode(self.config, first, sites, 0),
            sample_episode(self.config, first, sites, 0),
        )

    def test_public_partition_contains_no_private_or_split_truth(self) -> None:
        for path in self.release.glob("splits/*/*/public/*.json"):
            leaked = FORBIDDEN_PUBLIC_KEYS & _walk_keys(read_json(path))
            self.assertFalse(leaked, f"{path}: {sorted(leaked)}")

    def test_target_processes_remain_three_dimensional_without_fake_universal_quotas(self) -> None:
        processes: set[str] = set()
        for path in self.release.glob("splits/*/*/evaluator_private/episodes/*.json"):
            episode = read_json(path)
            targets = episode["targets"]
            processes.add(episode["target_process"])
            self.assertEqual(len(targets), episode["target_count"])
            self.assertTrue(all(len(item["position"]) == 3 for item in targets))
            self.assertTrue(all(item["support_class"] != "facade" for item in targets))
            if episode["target_process"] == "height_stratified":
                self.assertGreaterEqual(len({item["altitude_band"] for item in targets}), 3)
        self.assertEqual(
            processes,
            {"uniform_surface", "clustered_surface", "height_stratified"},
        )

    def test_target_process_interventions_control_nonprocess_factors(self) -> None:
        paths = sorted(
            self.release.glob("splits/test_target_process/*/evaluator_private/episodes/*.json")
        )
        episodes = [read_json(path) for path in paths]
        self.assertEqual(len(episodes), 3)
        for key in (
            "episode_seed",
            "condition_group_id",
            "target_count",
            "starts",
            "smoke",
            "communication",
            "energy_budget_j",
        ):
            self.assertEqual(len({content_hash(episode[key]) for episode in episodes}), 1, key)
        self.assertEqual(len({episode["target_process"] for episode in episodes}), 3)
        self.assertEqual(len({content_hash(episode["targets"]) for episode in episodes}), 3)

    def test_opening_sites_are_on_real_component_faces(self) -> None:
        city = generate_city(self.config, "train", 0, 0, ["street_lamp_01"])
        components = {
            f"{building['id']}/{component['id']}": component
            for building in city["buildings"]
            for component in building["components"]
        }
        openings = [
            site for site in derive_support_sites(city) if site["support_class"] == "opening"
        ]
        self.assertTrue(openings)
        for site in openings:
            component = components[site["owner_id"]]
            cx, cy, cz = [float(value) for value in component["center"]]
            sx, sy, sz = [float(value) for value in component["size"]]
            px, py, pz = [float(value) for value in site["position"]]
            on_x_face = math.isclose(abs(px - cx), sx / 2 + 0.2, abs_tol=1e-4)
            on_y_face = math.isclose(abs(py - cy), sy / 2 + 0.2, abs_tol=1e-4)
            self.assertTrue(on_x_face ^ on_y_face)
            self.assertGreaterEqual(pz, cz - sz / 2)
            self.assertLessEqual(pz, cz + sz / 2)

    def test_official_fleet_and_resilience_interventions_are_paired(self) -> None:
        paths = list(
            self.release.glob("splits/test_resilience/*/evaluator_private/episodes/*.json")
        )
        episodes = [read_json(path) for path in sorted(paths)]
        self.assertTrue(episodes)
        payload_hashes = set()
        profiles = set()
        faults_by_profile = {}
        start_ids: set[str] = set()
        for episode in episodes:
            self.assertEqual(episode["fleet_profile"]["count"], 8)
            self.assertEqual(len(episode["starts"]), 8)
            start_ids = {start["drone_id"] for start in episode["starts"]}
            fault = episode["fault_spec"]
            self.assertLessEqual(set(fault["affected_drone_ids"]), start_ids)
            profiles.add(fault["profile"])
            faults_by_profile[fault["profile"]] = fault
            payload = {
                key: episode[key]
                for key in (
                    "episode_seed",
                    "condition_group_id",
                    "target_process",
                    "targets",
                    "starts",
                    "smoke",
                    "communication",
                    "energy_budget_j",
                )
            }
            payload_hashes.add(content_hash(payload))
        self.assertEqual(len(payload_hashes), 1)
        self.assertEqual(profiles, set(self.config.fault_profiles("test_resilience")))
        self.assertEqual(start_ids, {f"uav-{index:02d}" for index in range(8)})
        hard_one = faults_by_profile["hard_loss_1"]
        hard_two = faults_by_profile["hard_loss_2"]
        self.assertLessEqual(
            set(hard_one["affected_drone_ids"]), set(hard_two["affected_drone_ids"])
        )
        self.assertEqual(hard_one["onset_fraction"], hard_two["onset_fraction"])

    def test_core_fleet_remains_four_and_fleet_tracks_are_explicit(self) -> None:
        core_episode_path = next(
            self.release.glob("splits/test_iid/*/evaluator_private/episodes/*.json")
        )
        episode = read_json(core_episode_path)
        self.assertEqual(episode["fleet_profile"]["count"], 4)
        self.assertEqual(len(episode["starts"]), 4)
        fleet = self.config.raw["evaluation_tracks"]["fleet"]
        self.assertEqual(fleet["scaling_counts"], [2, 4, 8])
        self.assertEqual(fleet["resilience_count"], 8)
        self.assertEqual(fleet["resilience_loss_counts"], [0, 1, 2])
        self.assertEqual(fleet["conditional_stress_counts"], [12])

    def test_observation_contract_separates_capability_and_confirmation(self) -> None:
        contract = self.config.raw["observation_contract"]
        self.assertEqual(tuple(contract["tiers"]), OBSERVATION_TIERS)
        self.assertEqual(
            contract["leaderboards"]["perception-search-3d"],
            ["T4_full_perception"],
        )
        confirmation = contract["geometry_confirmation"]
        self.assertFalse(confirmation["allow_distance_only"])
        self.assertTrue(confirmation["require_line_of_sight"])
        self.assertTrue(confirmation["require_surface_facing"])
        self.assertTrue(confirmation["require_source_observation_id"])

    def test_distance_only_confirmation_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.config.raw)
        mutated["observation_contract"]["geometry_confirmation"]["allow_distance_only"] = True
        path = self.root / "invalid-distance-confirmation.json"
        write_json(path, mutated)
        with self.assertRaises(ValueError):
            load_release_config(path)

    def test_observation_channel_loss_is_modality_neutral(self) -> None:
        path = next(
            episode_path
            for episode_path in self.release.glob(
                "splits/test_resilience/*/evaluator_private/episodes/*.json"
            )
            if read_json(episode_path)["fault_spec"]["type"] == "observation_channel_loss"
        )
        fault = read_json(path)["fault_spec"]
        self.assertEqual(fault["channel"], "local_geometry")
        self.assertNotIn("sensor", fault)

    def test_invalid_target_process_training_leak_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.config.raw)
        mutated["target_processes"]["by_split"]["train"].append("height_stratified")
        path = self.root / "invalid-release.json"
        write_json(path, mutated)
        with self.assertRaises(ValueError):
            load_release_config(path)

    def test_tamper_is_rejected(self) -> None:
        tampered = self.root / "tampered"
        shutil.copytree(self.release, tampered)
        path = next(tampered.glob("splits/*/*/public/cityspec.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["size_m"] += 1
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaises(ValidationError):
            validate_release(tampered, write_report=False)

    def test_split_and_asset_contracts(self) -> None:
        index = read_json(self.release / "release_index.json")
        self.assertEqual(set(index["selected_splits"]), set(EXPECTED_SPLITS))
        self.assertEqual(
            len({item["generation_seed"] for item in index["layouts"]}),
            len(index["layouts"]),
        )
        self.assertEqual(
            ACCEPTED_SPDX,
            frozenset({"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"}),
        )


if __name__ == "__main__":
    unittest.main()
