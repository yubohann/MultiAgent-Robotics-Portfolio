from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_task import (
    LIDAR_RAY_COUNT,
    ONBOARD_IMAGE_HEIGHT,
    ONBOARD_IMAGE_WIDTH,
)
from rivermark_benchmark.formal_dataset import sha256_file
from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.policy_projection import (
    POLICY_OBSERVATION_SCHEMA,
    POLICY_PROJECTION_SCHEMA,
    PolicyProjectionError,
    inspect_candidate_pack_streams,
    inspect_policy_observation_sources,
    project_policy_observations,
    validate_candidate_abi_sources,
)

AGENTS = 8
TIMESTAMPS = np.asarray([100, 200], dtype=np.int64)
SOURCE_PATHS = (
    "sensors/onboard_rgbd.npz",
    "sensors/lidar.npz",
    "sensors/imu.npz",
    "streams/state_action.npz",
    "streams/public_task.npz",
    "streams/public_messages.npz",
    "learning_labels/semantic_metadata.json",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _rebind_capture(root: Path) -> None:
    receipt_path = root / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_hashes"] = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in SOURCE_PATHS
    }
    _write_json(receipt_path, receipt)
    _write_json(
        root / "independent_validation.json",
        {
            "schema": "org.rivermark.isaac-independent-validation.v1",
            "status": "passed",
            "issues": [],
            "capture_receipt_sha256": sha256_file(receipt_path),
        },
    )


def _capture_fixture(
    root: Path,
    *,
    revision: str = "a" * 40,
    rgb_dtype: np.dtype | type = np.uint8,
    rgb_channels: int = 3,
    lidar_dtype: np.dtype | type = np.float32,
    lidar_ray_count: int = LIDAR_RAY_COUNT,
) -> Path:
    (root / "sensors").mkdir(parents=True)
    (root / "streams").mkdir()
    write_chunked_frame_archive(
        root / "sensors/onboard_rgbd.npz",
        timestamps_ns=TIMESTAMPS,
        inline_fields={},
        frame_fields={
            "rgb": np.zeros(
                (2, AGENTS, ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH, rgb_channels),
                dtype=rgb_dtype,
            ),
            "distance_to_image_plane_m": np.ones(
                (2, AGENTS, ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH, 1),
                dtype=np.float32,
            ),
        },
    )
    np.savez_compressed(
        root / "sensors/lidar.npz",
        timestamps_ns=TIMESTAMPS,
        pos_w_m=np.zeros((2, AGENTS, 3), dtype=np.float32),
        quat_wxyz=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (2, AGENTS, 1)),
        ranges_m=np.ones((2, AGENTS, lidar_ray_count), dtype=lidar_dtype),
    )
    np.savez_compressed(
        root / "sensors/imu.npz",
        timestamps_ns=TIMESTAMPS,
        pos_w_m=np.zeros((2, AGENTS, 3), dtype=np.float32),
        quat_wxyz=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (2, AGENTS, 1)),
        linear_acceleration_b_mps2=np.zeros((2, AGENTS, 3), dtype=np.float32),
        angular_velocity_b_radps=np.zeros((2, AGENTS, 3), dtype=np.float32),
    )
    state_times = np.asarray([100, 150, 200], dtype=np.int64)
    np.savez_compressed(
        root / "streams/state_action.npz",
        command_time_ns=state_times - 1,
        effective_time_ns=state_times,
        root_pos_w_m=np.zeros((3, AGENTS, 3), dtype=np.float32),
        root_quat_wxyz=np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (3, AGENTS, 1)),
        root_lin_vel_w_mps=np.zeros((3, AGENTS, 3), dtype=np.float32),
        root_ang_vel_b_radps=np.zeros((3, AGENTS, 3), dtype=np.float32),
        desired_pos_w_m=np.zeros((3, AGENTS, 3), dtype=np.float32),
        desired_vel_w_mps=np.zeros((3, AGENTS, 3), dtype=np.float32),
        target_thrust_n=np.ones((3, AGENTS, 4), dtype=np.float32),
        applied_thrust_n=np.ones((3, AGENTS, 4), dtype=np.float32),
    )
    np.savez_compressed(
        root / "streams/public_task.npz",
        timestamps_ns=TIMESTAMPS,
        waypoint_index=np.zeros((2, AGENTS), dtype=np.int64),
        waypoint_progress=np.zeros((2, AGENTS), dtype=np.float32),
        desired_waypoint_w_m=np.zeros((2, AGENTS, 3), dtype=np.float32),
        distance_to_waypoint_m=np.ones((2, AGENTS), dtype=np.float32),
        waypoint_reached=np.zeros((2, AGENTS), dtype=bool),
        action_mode=np.zeros((2, AGENTS), dtype=np.int8),
        coverage_cell_id=np.arange(AGENTS, dtype=np.int64)[None, :].repeat(2, axis=0),
        task_time_s=np.zeros((2, AGENTS), dtype=np.float32),
    )
    np.savez_compressed(
        root / "streams/public_messages.npz",
        timestamps_ns=TIMESTAMPS,
        sender_agent_id=np.arange(AGENTS, dtype=np.int64)[None, :].repeat(2, axis=0),
        message_sequence=np.zeros((2, AGENTS), dtype=np.int64),
        message_waypoint_index=np.zeros((2, AGENTS), dtype=np.int64),
        message_position_w_m=np.zeros((2, AGENTS, 3), dtype=np.float32),
        message_velocity_w_mps=np.zeros((2, AGENTS, 3), dtype=np.float32),
        message_flags=np.ones((2, AGENTS), dtype=np.uint8),
    )
    _write_json(
        root / "learning_labels/semantic_metadata.json",
        {
            "schema": "org.rivermark.isaac-semantic-metadata.v1",
            "partition": "learning_labels",
            "policy_visible": False,
        },
    )
    _write_json(
        root / "capture_receipt.json",
        {
            "schema": "org.rivermark.isaac-swarm-capture.v1",
            "status": "captured",
            "ok": True,
            "source_worktree_dirty": False,
            "source_revision": revision,
            "claim_boundary": {"formal_benchmark_admission": False},
            "task": {
                "task_kind": "expert_coverage_dataset",
                "track": "t1-expert-coverage-multisensor-v1",
                "scoring_status": "not_scored",
            },
            "physics": {"same_world_agent_count": AGENTS},
            "collection_binding": {
                "protocol_id": "citylite-t1-projection-v1",
                "protocol_sha256": "b" * 64,
                "cell_id": "train-route-a",
                "split": "train",
                "episode_index": 0,
                "episode_seed": 42,
                "private_evaluator_path": "C:/not-copied/manifest.json",
            },
            "artifact_hashes": {},
        },
    )
    _rebind_capture(root)
    return root


class PolicyProjectionTests(unittest.TestCase):
    def test_candidate_pack_inspection_covers_exact_eight_streams_and_abi_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _capture_fixture(Path(temporary) / "capture")
            streams = inspect_candidate_pack_streams(capture)

            self.assertEqual(
                set(streams),
                {"actions", "state", "task", "messages", "rgb", "depth", "lidar", "imu"},
            )
            self.assertEqual(streams["actions"]["timestamp_field"], "command_time_ns")
            self.assertEqual(streams["rgb"]["path"], streams["depth"]["path"])
            abi = {
                "streams": [
                    {
                        "stream_id": stream_id,
                        "modality": stream["modality"],
                        "partition": "policy_visible",
                        "fields": [
                            {
                                "name": field,
                                "dtype": np.dtype(descriptor["dtype"]).name,
                                "shape": [
                                    (
                                        "physics_step"
                                        if stream_id in {"actions", "state"}
                                        else "sensor_frame"
                                    ),
                                    *descriptor["shape"][1:],
                                ],
                                "timestamp_field": stream["timestamp_field"],
                            }
                            for field, descriptor in stream["arrays"].items()
                        ],
                    }
                    for stream_id, stream in streams.items()
                ]
            }
            self.assertEqual(validate_candidate_abi_sources(abi, streams), ())

            abi["streams"][0]["fields"][0]["dtype"] = "float64"
            self.assertIn(
                "abi_dtype_mismatch",
                {issue.code for issue in validate_candidate_abi_sources(abi, streams)},
            )

            extra_stream = copy.deepcopy(abi)
            extra_stream["streams"].append(
                {
                    "stream_id": "invented",
                    "modality": "invented",
                    "partition": "policy_visible",
                    "fields": [],
                }
            )
            self.assertIn(
                "abi_stream_set",
                {
                    issue.code
                    for issue in validate_candidate_abi_sources(extra_stream, streams)
                },
            )

            extra_field = copy.deepcopy(abi)
            extra_field["streams"][0]["fields"].append(
                {
                    "name": "invented",
                    "dtype": "float32",
                    "shape": ["physics_step"],
                    "timestamp_field": "command_time_ns",
                }
            )
            self.assertIn(
                "abi_field_set",
                {
                    issue.code
                    for issue in validate_candidate_abi_sources(extra_field, streams)
                },
            )

            wrong_shape = copy.deepcopy(abi)
            wrong_shape["streams"][0]["fields"][0]["shape"][0] = "sensor_frame"
            self.assertIn(
                "abi_shape_mismatch",
                {
                    issue.code
                    for issue in validate_candidate_abi_sources(wrong_shape, streams)
                },
            )

            wrong_metadata = copy.deepcopy(abi)
            wrong_metadata["streams"][0]["modality"] = "invented"
            wrong_metadata["streams"][0]["fields"][0]["timestamp_field"] = "timestamps_ns"
            codes = {
                issue.code
                for issue in validate_candidate_abi_sources(wrong_metadata, streams)
            }
            self.assertIn("abi_modality_mismatch", codes)
            self.assertIn("abi_timestamp_mismatch", codes)

    def test_read_only_inspection_reuses_complete_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _capture_fixture(Path(temporary) / "capture")
            before = sorted(path.relative_to(capture) for path in capture.rglob("*"))
            inspection = inspect_policy_observation_sources(capture)
            after = sorted(path.relative_to(capture) for path in capture.rglob("*"))

            self.assertEqual(before, after)
            self.assertEqual(inspection.frame_count, 2)
            self.assertEqual(inspection.state_sample_count, 3)
            self.assertEqual(set(inspection.streams), {"onboard_rgbd", "lidar", "imu", "state"})
            self.assertEqual(inspection.source_revision, "a" * 40)

    def test_projection_is_external_hash_bound_and_contains_only_allow_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "capture")
            output = root / "projection"
            result = project_policy_observations(capture, output)

            self.assertEqual(result.observation_count, 16)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"observations.jsonl", "projection_manifest.json"},
            )
            manifest = json.loads((output / "projection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], POLICY_PROJECTION_SCHEMA)
            self.assertFalse(manifest["formal_benchmark_admission"])
            self.assertFalse(manifest["t2_score_permitted"])
            self.assertEqual(manifest["observation_count"], 16)
            self.assertEqual(
                set(manifest["streams"]),
                {"onboard_rgbd", "lidar", "imu", "state"},
            )
            self.assertEqual(
                manifest["streams"]["onboard_rgbd"]["arrays"]["rgb"],
                {
                    "dtype": "|u1",
                    "shape": [
                        2,
                        AGENTS,
                        ONBOARD_IMAGE_HEIGHT,
                        ONBOARD_IMAGE_WIDTH,
                        3,
                    ],
                },
            )
            self.assertEqual(
                manifest["streams"]["lidar"]["arrays"]["ranges_m"]["shape"],
                [2, AGENTS, LIDAR_RAY_COUNT],
            )
            exposed_paths = {artifact["path"] for artifact in manifest["source_artifacts"]}
            self.assertNotIn("streams/public_task.npz", exposed_paths)
            self.assertNotIn("streams/public_messages.npz", exposed_paths)
            records = [json.loads(line) for line in (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(record["schema"] == POLICY_OBSERVATION_SCHEMA for record in records))
            self.assertEqual(len({record["observation_id"] for record in records}), 16)
            serialized = json.dumps({"manifest": manifest, "records": records}, sort_keys=True).lower()
            for token in ("semantic", "learning_label", "target", "evaluator", "private", "ground_truth", "overview", "camera_pose", "contact"):
                self.assertNotIn(token, serialized)

    def test_timestamp_and_agent_dimension_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            time_capture = _capture_fixture(root / "time")
            imu_path = time_capture / "sensors/imu.npz"
            imu = _arrays(imu_path)
            imu["timestamps_ns"] = np.asarray([100, 201], dtype=np.int64)
            np.savez_compressed(imu_path, **imu)
            _rebind_capture(time_capture)
            with self.assertRaisesRegex(PolicyProjectionError, "timestamps do not match"):
                project_policy_observations(time_capture, root / "time-output")

            agent_capture = _capture_fixture(root / "agent")
            lidar_path = agent_capture / "sensors/lidar.npz"
            lidar = _arrays(lidar_path)
            lidar["ranges_m"] = np.ones((2, 7, 16), dtype=np.float32)
            np.savez_compressed(lidar_path, **lidar)
            _rebind_capture(agent_capture)
            with self.assertRaisesRegex(PolicyProjectionError, "must begin with"):
                project_policy_observations(agent_capture, root / "agent-output")

    def test_sensor_dtypes_and_complete_shapes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_rgb_dtype = _capture_fixture(root / "rgb-dtype", rgb_dtype=np.float32)
            with self.assertRaisesRegex(PolicyProjectionError, "rgb must use uint8"):
                project_policy_observations(wrong_rgb_dtype, root / "rgb-dtype-output")

            wrong_rgb_channels = _capture_fixture(root / "rgb-channels", rgb_channels=4)
            with self.assertRaisesRegex(PolicyProjectionError, "rgb must end with"):
                project_policy_observations(wrong_rgb_channels, root / "rgb-channels-output")

            wrong_lidar_dtype = _capture_fixture(root / "lidar-dtype", lidar_dtype=np.float64)
            with self.assertRaisesRegex(PolicyProjectionError, "lidar ranges_m must use float32"):
                project_policy_observations(wrong_lidar_dtype, root / "lidar-dtype-output")

            wrong_lidar_rays = _capture_fixture(
                root / "lidar-rays", lidar_ray_count=LIDAR_RAY_COUNT - 1
            )
            with self.assertRaisesRegex(PolicyProjectionError, "lidar ranges must end with"):
                project_policy_observations(wrong_lidar_rays, root / "lidar-rays-output")

    def test_later_chunked_frame_shape_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "capture")
            onboard_path = capture / "sensors/onboard_rgbd.npz"
            arrays = _arrays(onboard_path)
            arrays["rgb__frame__000001"] = np.zeros(
                (AGENTS, ONBOARD_IMAGE_HEIGHT, ONBOARD_IMAGE_WIDTH, 4),
                dtype=np.uint8,
            )
            np.savez(onboard_path, **arrays)
            _rebind_capture(capture)
            with self.assertRaisesRegex(PolicyProjectionError, "disagrees with frame 0"):
                project_policy_observations(capture, root / "output")

    def test_extra_label_like_field_and_stale_validation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "field")
            state_path = capture / "streams/state_action.npz"
            state = _arrays(state_path)
            state["target_hint"] = np.ones((3, AGENTS, 3), dtype=np.float32)
            np.savez_compressed(state_path, **state)
            _rebind_capture(capture)
            with self.assertRaisesRegex(PolicyProjectionError, "differ.*allow-list"):
                project_policy_observations(capture, root / "field-output")

            stale = _capture_fixture(root / "stale")
            receipt = json.loads((stale / "capture_receipt.json").read_text(encoding="utf-8"))
            receipt["source_revision"] = "c" * 40
            _write_json(stale / "capture_receipt.json", receipt)
            with self.assertRaisesRegex(PolicyProjectionError, "validation is absent, failed, or stale"):
                project_policy_observations(stale, root / "stale-output")

    def test_noncanonical_npz_members_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("sensors/lidar.npz", "sensors/onboard_rgbd.npz"):
                with self.subTest(relative=relative):
                    capture = _capture_fixture(root / relative.replace("/", "-"))
                    with zipfile.ZipFile(capture / relative, mode="a") as archive:
                        archive.writestr("hidden.bin", b"must-not-be-policy-visible")
                    _rebind_capture(capture)
                    with self.assertRaisesRegex(
                        PolicyProjectionError, "differ.*allow-list"
                    ):
                        project_policy_observations(capture, root / f"{capture.name}-output")

    def test_output_inside_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _capture_fixture(Path(temporary) / "capture")
            with self.assertRaisesRegex(PolicyProjectionError, "outside the source capture"):
                project_policy_observations(capture, capture / "projection")

    def test_unsafe_public_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "capture")
            receipt_path = capture / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["collection_binding"]["cell_id"] = "../escape"
            _write_json(receipt_path, receipt)
            _rebind_capture(capture)
            with self.assertRaisesRegex(PolicyProjectionError, "cell_id is not public-safe"):
                project_policy_observations(capture, root / "output")

    def test_source_binding_changes_observation_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _capture_fixture(root / "first", revision="a" * 40)
            second = _capture_fixture(root / "second", revision="c" * 40)
            first_result = project_policy_observations(first, root / "first-output")
            second_result = project_policy_observations(second, root / "second-output")
            self.assertNotEqual(first_result.observations_sha256, second_result.observations_sha256)
            self.assertNotEqual(first_result.manifest_sha256, second_result.manifest_sha256)

    def test_invalid_source_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "capture", revision="not-a-git-revision")
            with self.assertRaisesRegex(
                PolicyProjectionError, "source revision is not a Git commit hash"
            ):
                project_policy_observations(capture, root / "output")


if __name__ == "__main__":
    unittest.main()
