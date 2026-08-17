from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.validate import validate_episode_manifest


FIXTURE = ROOT / "tests" / "fixtures" / "episode_manifest_fixture.json"


def load_manifest() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def issue_codes(manifest: dict) -> set[str]:
    return {issue.code for issue in validate_episode_manifest(manifest)}


class EpisodeManifestValidationTests(unittest.TestCase):
    def test_minimal_manifest_is_structurally_valid(self) -> None:
        self.assertEqual(validate_episode_manifest(load_manifest()), ())

    def test_hidden_target_field_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["policy_visible"]["hiddenTargetCoordinates"] = [1.0, 2.0, 3.0]
        self.assertIn("policy_truth_leak", issue_codes(manifest))

    def test_hidden_target_stream_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["streams"][0]["modality"] = "hidden_target_xyz"
        manifest["policy_visible"]["modalities"].append("hidden_target_xyz")
        self.assertIn("policy_truth_leak", issue_codes(manifest))

    def test_profile_is_closed_to_extra_modalities(self) -> None:
        manifest = load_manifest()
        manifest["policy_visible"]["modalities"].append("lidar")
        self.assertIn("profile_modalities", issue_codes(manifest))

    def test_multisensor_profile_is_closed_and_valid(self) -> None:
        manifest = load_manifest()
        manifest["task"]["information_profile"] = "multisensor_rgbd_lidar_radar_state"
        manifest["policy_visible"]["information_profile"] = "multisensor_rgbd_lidar_radar_state"
        manifest["policy_visible"]["modalities"] = [
            "rgb",
            "distance_to_image_plane",
            "lidar",
            "radar",
            "imu",
            "proprioception",
            "public_task_state",
            "public_team_messages",
            "high_level_action_history",
        ]
        manifest["streams"].extend(
            [
                {
                    "stream_id": "pilot-lidar",
                    "partition": "policy_visible",
                    "modality": "lidar",
                    "media_type": "application/x-npz",
                    "sample_count": 0,
                    "timestamp_field": "sensor_time_ns",
                    "path": "payloads/lidar.npz",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                },
                {
                    "stream_id": "pilot-radar",
                    "partition": "policy_visible",
                    "modality": "radar",
                    "media_type": "application/x-npz",
                    "sample_count": 0,
                    "timestamp_field": "sensor_time_ns",
                    "path": "payloads/radar.npz",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                },
                {
                    "stream_id": "pilot-imu",
                    "partition": "policy_visible",
                    "modality": "imu",
                    "media_type": "application/x-npz",
                    "sample_count": 0,
                    "timestamp_field": "sensor_time_ns",
                    "path": "payloads/imu.npz",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                },
            ]
        )
        self.assertEqual(validate_episode_manifest(manifest), ())

    def test_no_radar_multisensor_profile_is_closed_and_valid(self) -> None:
        manifest = load_manifest()
        profile = "multisensor_rgbd_lidar_imu_state"
        manifest["task"]["information_profile"] = profile
        manifest["policy_visible"]["information_profile"] = profile
        manifest["policy_visible"]["modalities"] = [
            "rgb",
            "distance_to_image_plane",
            "lidar",
            "imu",
            "proprioception",
            "public_task_state",
            "public_team_messages",
            "high_level_action_history",
        ]
        manifest["streams"].extend(
            [
                {
                    "stream_id": f"isaac-{modality}",
                    "partition": "policy_visible",
                    "modality": modality,
                    "media_type": "application/x-npz",
                    "sample_count": 0,
                    "timestamp_field": "sensor_time_ns",
                    "path": f"payloads/{modality}.npz",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                }
                for modality in ("lidar", "imu")
            ]
        )
        self.assertEqual(validate_episode_manifest(manifest), ())

        manifest["policy_visible"]["modalities"].append("semantic_segmentation")
        self.assertIn("profile_modalities", issue_codes(manifest))

        manifest["policy_visible"]["modalities"].remove("semantic_segmentation")

        manifest["policy_visible"]["modalities"].append("radar")
        self.assertIn("profile_modalities", issue_codes(manifest))

    def test_victim_truth_alias_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["policy_visible"]["victim_positions"] = [[1.0, 2.0, 3.0]]
        self.assertIn("policy_truth_leak", issue_codes(manifest))

    def test_legal_target_waypoint_is_not_mistaken_for_target_truth(self) -> None:
        manifest = load_manifest()
        manifest["policy_visible"]["latest_action"] = {
            "target_waypoint_xyz": [1.0, 2.0, 3.0]
        }
        self.assertNotIn("policy_truth_leak", issue_codes(manifest))

    def test_target_conditioned_route_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["provenance"]["route_conditioning"] = "target_conditioned"
        self.assertIn("target_conditioned_route", issue_codes(manifest))

    def test_posthoc_observation_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["provenance"]["observation_generation"] = "post_hoc_render"
        self.assertIn("posthoc_observation", issue_codes(manifest))

    def test_absolute_and_parent_paths_are_rejected(self) -> None:
        for unsafe in ("C:\\private\\scene.json", "C:private\\scene.json", "../scene.json"):
            with self.subTest(path=unsafe):
                manifest = load_manifest()
                manifest["layout"]["scene_manifest_ref"] = unsafe
                self.assertIn("unsafe_path", issue_codes(manifest))

    def test_blind_labels_cannot_be_distributed(self) -> None:
        manifest = load_manifest()
        manifest["split"] = "blind_test"
        self.assertIn("blind_label_distribution", issue_codes(manifest))

    def test_blind_truth_must_be_server_only(self) -> None:
        manifest = load_manifest()
        manifest["split"] = "ood_test"
        manifest["learning_labels"]["distributed"] = False
        manifest["evaluator_private"]["distributed"] = True
        manifest["evaluator_private"]["server_only"] = False
        self.assertIn("blind_truth_distribution", issue_codes(manifest))

    def test_missing_nested_required_field_is_rejected(self) -> None:
        manifest = load_manifest()
        del manifest["provenance"]["code_commit"]
        self.assertIn("required", issue_codes(manifest))

    def test_unknown_root_field_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["legacy_oracle_blob"] = {}
        self.assertIn("unknown_field", issue_codes(manifest))

    def test_stream_id_must_be_a_string(self) -> None:
        manifest = load_manifest()
        manifest["streams"][0]["stream_id"] = 12
        self.assertIn("stream_id", issue_codes(manifest))

    def test_malformed_enum_types_never_crash(self) -> None:
        mutations = (
            lambda value: value.__setitem__("split", []),
            lambda value: value["streams"][0].__setitem__("stream_id", []),
            lambda value: value["streams"][0].__setitem__("partition", []),
            lambda value: value["task"].__setitem__("information_profile", []),
            lambda value: value["task"].__setitem__("observation_scope", []),
            lambda value: value["provenance"].__setitem__("collector_type", []),
            lambda value: value["provenance"].__setitem__("scene_asset_license_status", []),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                manifest = load_manifest()
                mutate(manifest)
                self.assertTrue(validate_episode_manifest(manifest))

    def test_stream_metadata_length_is_bounded(self) -> None:
        manifest = load_manifest()
        manifest["streams"][0]["media_type"] = "x" * 129
        self.assertIn("stream_field", issue_codes(manifest))

    def test_non_finite_quality_number_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["quality"]["pose_closure_max_error_m"] = float("nan")
        self.assertIn("pose_closure", issue_codes(manifest))

    def test_invalid_reasons_must_be_unique(self) -> None:
        manifest = load_manifest()
        manifest["quality"]["invalid_reasons"] = ["missing", "missing"]
        self.assertIn("duplicate_value", issue_codes(manifest))

    def test_manifest_mutations_do_not_change_fixture(self) -> None:
        manifest = load_manifest()
        baseline = copy.deepcopy(manifest)
        manifest["quality"]["recording_valid"] = True
        self.assertNotEqual(manifest, baseline)
        self.assertEqual(load_manifest(), baseline)


if __name__ == "__main__":
    unittest.main()
