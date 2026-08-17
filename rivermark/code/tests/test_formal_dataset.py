from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.abi import OBSERVATION_ABI_SCHEMA, observation_abi_sha256
from rivermark_benchmark.failure_ledger import (
    load_failure_ledger,
    summarize_failure_ledger,
)
from rivermark_benchmark.formal_dataset import (
    CONTENT_HASH_INDEX_SCHEMA,
    FORMAL_CAPTURE_RECEIPT_SCHEMA,
    LINEAGE_AXES,
    LINEAGE_SCHEMA,
    DatasetCollector,
    plan_split_authority,
    sha256_file,
    verify_candidate_episode,
    verify_dataset_integrity,
)
from rivermark_benchmark.release_manifest import (
    ReleaseBuildError,
    build_release_manifest,
    validate_release_manifest,
)
from rivermark_benchmark.supply_chain import (
    canonical_supply_chain_bytes,
    supply_chain_sha256,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    return sha256_file(path)


def _release_supply_chain(data_receipt_hashes: tuple[str, ...] = ()) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "org.rivermark.benchmark.supply-chain.v1",
        "manifest_version": "1.0.0",
        "release_id": "release-test-001",
        "created_at": "2026-07-24T00:00:00Z",
        "assets": [
            {
                "asset_id": "project-code",
                "kind": "code",
                "source_uri": "https://github.com/yubohann/rivermark-benchmark",
                "sha256": "a" * 64,
                "license_spdx": "Apache-2.0",
                "license_status": "redistribution_cleared",
                "redistributable": True,
                "attribution": "Rivermark Benchmark maintainers",
                "decision_record": {
                    "record_id": "test-clearance-project-code",
                    "approved_by": "Test release approver",
                    "approved_at": "2026-07-24T00:00:00Z",
                    "evidence_sha256": "d" * 64,
                },
            },
            {
                "asset_id": "city-lite-scene-input",
                "kind": "scene_layer",
                "source_uri": "https://example.org/rivermark/city-lite",
                "sha256": "b" * 64,
                "license_spdx": "CC-BY-4.0",
                "license_status": "redistribution_cleared",
                "redistributable": True,
                "attribution": "Synthetic City-Lite test fixture",
                "decision_record": {
                    "record_id": "test-clearance-city-lite",
                    "approved_by": "Test release approver",
                    "approved_at": "2026-07-24T00:00:00Z",
                    "evidence_sha256": "e" * 64,
                },
            },
            {
                "asset_id": "cf2x-robot-input",
                "kind": "robot_asset",
                "source_uri": "https://example.org/rivermark/cf2x",
                "sha256": "c" * 64,
                "license_spdx": "Apache-2.0",
                "license_status": "redistribution_cleared",
                "redistributable": True,
                "attribution": "Synthetic CF2X test fixture",
                "decision_record": {
                    "record_id": "test-clearance-cf2x",
                    "approved_by": "Test release approver",
                    "approved_at": "2026-07-24T00:00:00Z",
                    "evidence_sha256": "f" * 64,
                },
            },
        ],
        "runtime_dependencies": [
            {
                "name": "numpy",
                "version": "1.24.0",
                "license_spdx": "BSD-3-Clause",
                "source_uri": "https://pypi.org/project/numpy/",
            }
        ],
        "sbom": {
            "format": "cyclonedx-json",
            "status": "verified",
            "path": "artifacts/sbom.cdx.json",
            "uri": "https://example.org/rivermark/sbom.json",
            "sha256": "0" * 64,
            "spec_version": "1.7",
            "generator": "cyclonedx-py 7.3.1",
        },
        "signature": {
            "status": "cryptographically_verified",
            "algorithm": "ed25519",
            "key_id": "test-release-key",
            "path": "artifacts/supply-chain.ed25519.sig",
            "uri": "https://example.org/rivermark/release.sig",
            "sha256": "0" * 64,
            "manifest_sha256": "0" * 64,
            "public_key_path": "artifacts/release.ed25519.pub",
            "public_key_uri": "https://example.org/rivermark/release.ed25519.pub",
            "public_key_sha256": "0" * 64,
        },
    }
    assets = payload["assets"]
    assert isinstance(assets, list)
    for index, receipt_hash in enumerate(data_receipt_hashes):
        assets.append(
            {
                "asset_id": f"formal-episode-data-{index:03d}",
                "kind": "data",
                "source_uri": f"https://example.org/rivermark/data/{index:03d}",
                "sha256": receipt_hash,
                "license_spdx": "LicenseRef-Rivermark-Derived-Data",
                "license_status": "redistribution_cleared",
                "redistributable": True,
                "attribution": "Synthetic formal episode test fixture",
                "decision_record": {
                    "record_id": f"test-clearance-data-{index:03d}",
                    "approved_by": "Test release approver",
                    "approved_at": "2026-07-24T00:00:00Z",
                    "evidence_sha256": hashlib.sha256(
                        f"data-decision-{index}".encode()
                    ).hexdigest(),
                },
            }
        )
    return payload


def _write_release_supply_chain(
    root: Path,
    *,
    data_receipt_hashes: tuple[str, ...] = (),
) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    payload = _release_supply_chain(data_receipt_hashes)
    sbom_path = root / "artifacts/sbom.cdx.json"
    signature_path = root / "artifacts/supply-chain.ed25519.sig"
    public_key_path = root / "artifacts/release.ed25519.pub"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        sbom_path,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "serialNumber": "urn:uuid:00000000-0000-0000-0000-000000000001",
            "version": 1,
            "metadata": {"component": {"type": "application", "name": "rivermark-benchmark", "version": "0.1.0"}},
            "components": [{"type": "library", "name": "numpy", "version": "1.24.0"}],
        },
    )
    sbom = payload["sbom"]
    assert isinstance(sbom, dict)
    sbom["sha256"] = sha256_file(sbom_path)
    private_key = Ed25519PrivateKey.generate()
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    signature = payload["signature"]
    assert isinstance(signature, dict)
    signature["public_key_sha256"] = sha256_file(public_key_path)
    signature_path.write_bytes(private_key.sign(canonical_supply_chain_bytes(payload)))
    signature["sha256"] = sha256_file(signature_path)
    signature["manifest_sha256"] = supply_chain_sha256(payload)
    manifest_path = root / "supply-chain.json"
    _write_json(manifest_path, payload)
    return manifest_path


def _observation_abi() -> dict[str, object]:
    field = {
        "name": "sample",
        "dtype": "float32",
        "shape": ["frame", "agent", 4],
        "units": "m",
        "frame_id": "body",
        "agent_id_field": "agent_id",
        "timestamp_field": "sensor_time_ns",
        "missing": {"policy": "not_applicable", "sentinel": None, "mask_field": None},
        "valid_range": {"min": None, "max": None, "inclusive": True},
        "compression": "npz_deflate",
        "time_semantics": "sensor_sample",
    }
    action = dict(field)
    action["name"] = "command"
    action["time_semantics"] = "command_before_step"
    return {
        "schema": OBSERVATION_ABI_SCHEMA,
        "version": "1.1.0",
        "action_timing": {
            "command_write": "before_simulation_step",
            "simulation_step": "after_command_write",
            "state_update": "after_simulation_step",
            "sensor_read": "after_state_update",
            "storage": "after_sensor_read",
        },
        "coordinate_frames": {
            "handedness": "right",
            "world_up_axis": "+z",
            "world_frame_convention": "x_east_y_north_z_up",
            "body_frame_convention": "flu",
            "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "transform_notation": "T_parent_child",
        },
        "calibration": {
            "camera": {
                "status": "recorded",
                "source": "formal-fixture",
                "intrinsics": {"model": "pinhole", "width_px": 8, "height_px": 8, "fx_px": 8.0, "fy_px": 8.0, "cx_px": 4.0, "cy_px": 4.0},
                "extrinsics": {"formula": "T_world_camera = T_world_body * T_body_camera", "quaternion_order": "wxyz"},
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "lidar": {"status": "unavailable", "source": "formal-fixture"},
            "imu": {"status": "unavailable", "source": "formal-fixture"},
        },
        "streams": [
            {"stream_id": "state", "modality": "proprioception", "partition": "policy_visible", "encoding": "npz", "fidelity": "simulator_consistent", "fidelity_limitations": ["no_hardware_sensor_noise"], "fields": [field]},
            {"stream_id": "actions", "modality": "high_level_action_history", "partition": "policy_visible", "encoding": "npz", "fidelity": "simulator_consistent", "fidelity_limitations": ["no_actuator_identification"], "fields": [action]},
        ],
    }


def _candidate(
    root: Path,
    *,
    episode_id: str,
    split: str = "train",
    layout_lineage: str | None = None,
    use_template_stream: bool = False,
    collection_binding: dict[str, object] | None = None,
) -> tuple[Path, str]:
    """Create a small independent-capture fixture, never a pilot promotion."""

    episode_root = root / episode_id
    scene_hash = _write_text(episode_root / "scenes" / "scene.json", '{"public":true}\n')
    task_hash = _write_text(episode_root / "tasks" / "task.json", '{"task":"search"}\n')
    state_hash = _write_text(episode_root / "streams" / "state.jsonl", '{"sim_time_ns":0}\n')
    action_hash = _write_text(episode_root / "streams" / "actions.jsonl", '{"sim_time_ns":0}\n')
    mission_hash = _write_text(episode_root / "streams" / "mission.json", '{"remaining_budget_ns":1}\n')
    message_hash = _write_text(episode_root / "streams" / "messages.jsonl", '{"sim_time_ns":0}\n')
    abi_payload = _observation_abi()
    abi_path = episode_root / "metadata" / "observation_abi.json"
    _write_json(abi_path, abi_payload)
    abi_hash = observation_abi_sha256(abi_payload)
    layout_lineage = layout_lineage or _digest(f"layout-lineage:{episode_id}")
    streams: list[dict[str, object]] = [
        {
            "stream_id": "state",
            "partition": "policy_visible",
            "modality": "proprioception",
            "media_type": "application/x-ndjson",
            "sample_count": 1,
            "timestamp_field": "sim_time_ns",
            "path": "streams/state.jsonl",
            "sha256": state_hash,
        },
        {
            "stream_id": "actions",
            "partition": "policy_visible",
            "modality": "high_level_action_history",
            "media_type": "application/x-ndjson",
            "sample_count": 1,
            "timestamp_field": "sim_time_ns",
            "path": "streams/actions.jsonl",
            "sha256": action_hash,
        },
        {
            "stream_id": "mission",
            "partition": "policy_visible",
            "modality": "public_task_state",
            "media_type": "application/json",
            "sample_count": 1,
            "timestamp_field": "sim_time_ns",
            "path": "streams/mission.json",
            "sha256": mission_hash,
        },
        {
            "stream_id": "messages",
            "partition": "policy_visible",
            "modality": "public_team_messages",
            "media_type": "application/x-ndjson",
            "sample_count": 1,
            "timestamp_field": "sim_time_ns",
            "path": "streams/messages.jsonl",
            "sha256": message_hash,
        },
    ]
    if use_template_stream:
        rgb_files: list[dict[str, object]] = []
        for agent_id in range(2):
            relative = f"cameras/{agent_id}/rgb.bin"
            rgb_files.append(
                {
                    "agent_id": agent_id,
                    "path": relative,
                    "sha256": _write_text(episode_root / relative, f"agent-{agent_id}\n"),
                }
            )
        index_path = episode_root / "cameras" / "rgb_hashes.json"
        _write_json(
            index_path,
            {
                "schema": CONTENT_HASH_INDEX_SCHEMA,
                "stream_id": "rgb",
                "files": rgb_files,
            },
        )
        streams.append(
            {
                "stream_id": "rgb",
                "partition": "policy_visible",
                "modality": "rgb",
                "media_type": "application/octet-stream",
                "sample_count": 2,
                "timestamp_field": "sensor_time_ns",
                "path_template": "cameras/{agent_id}/rgb.bin",
                "content_hash_index_path": "cameras/rgb_hashes.json",
                "content_hash_index_sha256": sha256_file(index_path),
            }
        )
    profile = "egocentric_rgb_state" if use_template_stream else "state_only"
    modalities = [
        "high_level_action_history",
        "proprioception",
        "public_task_state",
        "public_team_messages",
    ]
    if use_template_stream:
        modalities.append("rgb")
    manifest = {
        "schema": "org.rivermark.benchmark.episode.v1",
        "dataset_version": "1.0.0",
        "episode_id": episode_id,
        "split": split,
        "layout": {
            "layout_id": "formal-layout-a",
            "layout_hash": _digest("layout-a"),
            "layout_lineage_hash": layout_lineage,
            "scene_manifest_ref": "scenes/scene.json",
            "scene_manifest_sha256": scene_hash,
        },
        "task": {
            "task_id": "multi_uav_search3d",
            "task_variant_id": "formal-search-a",
            "task_spec_ref": "tasks/task.json",
            "task_spec_sha256": task_hash,
            "information_profile": profile,
            "observation_scope": "decentralized_explicit_comm",
            "agent_count": 2,
        },
        "timebase": {
            "unit": "ns",
            "physics_dt_ns": 5_000_000,
            "proprioception_period_ns": 20_000_000,
            "camera_period_ns": 100_000_000,
        },
        "coordinate_frames": {
            "handedness": "right",
            "world_up_axis": "+z",
            "world_frame_convention": "x_east_y_north_z_up",
            "body_frame_convention": "flu",
            "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "transform_notation": "T_parent_child",
        },
        "observation_abi": {
            "path": "metadata/observation_abi.json",
            "sha256": abi_hash,
        },
        "streams": streams,
        "policy_visible": {"information_profile": profile, "modalities": modalities},
        "learning_labels": {"distributed": False, "modalities": []},
        "evaluator_private": {
            "distributed": False,
            "server_only": True,
            "manifest_sha256": _digest(f"evaluator:{episode_id}"),
        },
        "provenance": {
            "route_conditioning": "public_only",
            "observation_generation": "online_runtime",
            "collector_type": "classical",
            "policy_id": "independent-formal-capture-fixture",
            "code_commit": "0123456789abcdef",
            "simulator_build": "isaaclab-4.5.0",
            "scene_asset_license_status": "redistribution_cleared",
        },
        "quality": {
            "recording_valid": True,
            "task_success": False,
            "invalid_reasons": [],
            "frame_completeness_ratio": 1.0,
            "timestamp_monotonic": True,
            "pose_closure_max_error_m": 0.002,
        },
    }
    if collection_binding is not None:
        manifest["collection_binding"] = copy.deepcopy(collection_binding)
    manifest_path = episode_root / "episode_manifest.json"
    _write_json(manifest_path, manifest)
    lineage_axes = {axis: _digest(f"{axis}:{episode_id}") for axis in LINEAGE_AXES}
    lineage_axes["layout_lineage"] = layout_lineage
    lineage_axes["task_manifest"] = task_hash
    lineage_axes["episode"] = _digest(episode_id)
    lineage_path = episode_root / "lineage.json"
    _write_json(
        lineage_path,
        {"schema": LINEAGE_SCHEMA, "episode_id": episode_id, "axes": lineage_axes},
    )
    receipt_path = episode_root / "formal_capture_receipt.json"
    receipt = {
        "schema": FORMAL_CAPTURE_RECEIPT_SCHEMA,
        "status": "admitted",
        "formal_benchmark_admission": True,
        "episode_manifest_sha256": sha256_file(manifest_path),
        "lineage_sha256": sha256_file(lineage_path),
        "observation_abi_sha256": abi_hash,
        "capture_backend": {
            "kind": "isaaclab",
            "build": "isaaclab-4.5.0",
            "sensor_physics_smoke_receipt_sha256": _digest("isaac-smoke"),
        },
        "integrity": {
            "online_capture": True,
            "queue_overflow": False,
            "silent_frame_drop": False,
            "timestamp_audit_passed": True,
            "pose_closure_audit_passed": True,
            "action_causality_audit_passed": True,
            "sensor_decode_audit_passed": True,
            "policy_leakage_audit_passed": True,
            "independent_validator_id": "independent-isaac-validation-fixture",
            "independent_validator_sha256": _digest("independent-validator"),
            "pose_closure_threshold_m": 0.01,
        },
        "partitions": {
            "policy_visible_audit_sha256": _digest("policy-audit"),
            "learning_labels_release_allowed": False,
            "evaluator_private_distributed": False,
            "evaluator_private_server_only": True,
        },
    }
    if collection_binding is not None:
        receipt["integrity"]["condition_realization_verified"] = True
        receipt["collection_binding"] = copy.deepcopy(collection_binding)
    _write_json(receipt_path, receipt)
    return episode_root, sha256_file(receipt_path)


class DatasetPipelineTests(unittest.TestCase):
    def test_collector_preserves_collection_binding_in_admission_and_ledger(self) -> None:
        binding: dict[str, object] = {
            "protocol_id": "citylite-coverage-v1",
            "protocol_sha256": "c" * 64,
            "cell_id": "train-route-0",
            "split": "train",
            "episode_index": 3,
            "episode_seed": 42,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(
                root / "captures",
                episode_id="formal-episode-collection-binding",
                collection_binding=binding,
            )
            dataset_root = root / "dataset"
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=supply_chain,
            ).collect(candidate)
            self.assertTrue(result.admitted, result.issues)
            admission = json.loads(
                (result.episode_root / "admission.json").read_text(encoding="utf-8")
            )
            self.assertEqual(admission["collection_binding"], binding)
            self.assertEqual(
                admission["supply_chain_manifest_sha256"],
                supply_chain_sha256(
                    json.loads(supply_chain.read_text(encoding="utf-8"))
                ),
            )
            self.assertEqual(admission["supply_chain_release_id"], "release-test-001")
            records = load_failure_ledger(dataset_root / "manifests" / "failure_ledger.jsonl")
            self.assertEqual(records[0]["collection_protocol_id"], binding["protocol_id"])
            self.assertEqual(records[0]["collection_cell_id"], binding["cell_id"])
            self.assertEqual(records[0]["collection_episode_index"], binding["episode_index"])
            self.assertEqual(records[0]["episode_seed"], binding["episode_seed"])

    def test_collector_records_admission_denominator_in_public_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            admitted, admitted_hash = _candidate(root / "captures", episode_id="formal-episode-ledger-a")
            rejected, rejected_hash = _candidate(root / "captures", episode_id="formal-episode-ledger-b")
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(admitted_hash, rejected_hash),
            )
            collector = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[admitted_hash],
                supply_chain_manifest=supply_chain,
            )
            self.assertTrue(collector.collect(admitted).admitted)
            self.assertFalse(collector.collect(rejected).admitted)
            summary = summarize_failure_ledger(dataset_root / "manifests" / "failure_ledger.jsonl")
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["admitted_count"], 1)
            self.assertEqual(summary["quarantined_count"], 1)

    def test_collector_rejects_pending_or_unbound_supply_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(
                root / "captures",
                episode_id="formal-episode-license-gate",
            )
            pending = _release_supply_chain((receipt_hash,))
            first_asset = pending["assets"][0]
            assert isinstance(first_asset, dict)
            first_asset["license_status"] = "pending"
            first_asset["redistributable"] = False
            del first_asset["decision_record"]
            pending["signature"] = {"status": "unsigned"}
            pending_path = root / "pending-supply-chain.json"
            _write_json(pending_path, pending)
            pending_result = DatasetCollector(
                root / "pending-dataset",
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=pending_path,
            ).collect(candidate)
            self.assertFalse(pending_result.admitted)
            self.assertIn(
                "supply_chain_license_closure",
                {issue.code for issue in pending_result.issues},
            )
            self.assertFalse((root / "pending-dataset" / "train").exists())

            unbound_path = _write_release_supply_chain(root / "unbound")
            unbound_result = DatasetCollector(
                root / "unbound-dataset",
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=unbound_path,
            ).collect(candidate)
            self.assertFalse(unbound_result.admitted)
            self.assertEqual(
                {issue.code for issue in unbound_result.issues},
                {"supply_chain_candidate_binding", "supply_chain_surface"},
            )
            self.assertFalse((root / "unbound-dataset" / "train").exists())

    def test_collector_rejects_mixed_supply_chain_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, first_hash = _candidate(
                root / "captures",
                episode_id="formal-episode-supply-a",
            )
            second, second_hash = _candidate(
                root / "captures",
                episode_id="formal-episode-supply-b",
            )
            first_supply = _write_release_supply_chain(
                root / "supply-a",
                data_receipt_hashes=(first_hash, second_hash),
            )
            second_supply = _write_release_supply_chain(
                root / "supply-b",
                data_receipt_hashes=(first_hash, second_hash),
            )
            dataset_root = root / "dataset"
            first_result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[first_hash],
                supply_chain_manifest=first_supply,
            ).collect(first)
            self.assertTrue(first_result.admitted, first_result.issues)
            second_result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[second_hash],
                supply_chain_manifest=second_supply,
            ).collect(second)
            self.assertFalse(second_result.admitted)
            self.assertIn(
                "supply_chain_mismatch",
                {issue.code for issue in second_result.issues},
            )
            self.assertFalse((dataset_root / "train" / "formal-episode-supply-b").exists())
            self.assertEqual(verify_dataset_integrity(dataset_root).episode_count, 1)

    def test_collector_rejects_supply_chain_changed_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(
                root / "captures",
                episode_id="formal-episode-supply-race",
            )
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            changed = json.loads(supply_chain.read_text(encoding="utf-8"))
            changed["assets"][0]["sha256"] = "9" * 64
            with patch(
                "rivermark_benchmark.formal_dataset.load_supply_chain_manifest",
                return_value=changed,
            ):
                result = DatasetCollector(
                    root / "dataset",
                    trusted_receipt_hashes=[receipt_hash],
                    supply_chain_manifest=supply_chain,
                ).collect(candidate)
            self.assertFalse(result.admitted)
            self.assertEqual(
                {issue.code for issue in result.issues},
                {"supply_chain_changed"},
            )
            self.assertFalse((root / "dataset" / "train").exists())

    def test_formal_admission_requires_observation_abi_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-abi-required")
            manifest_path = candidate / "episode_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["observation_abi"]
            _write_json(manifest_path, manifest)
            report = verify_candidate_episode(candidate, require_trusted_receipt=False)
            self.assertIn("observation_abi_required", {issue.code for issue in report.issues})

    def test_formal_receipt_must_bind_observation_abi_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-abi-receipt")
            receipt_path = candidate / "formal_capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["observation_abi_sha256"] = "0" * 64
            _write_json(receipt_path, receipt)
            report = verify_candidate_episode(candidate, require_trusted_receipt=False)
            self.assertIn("receipt_abi_mismatch", {issue.code for issue in report.issues})

    def test_collection_bound_receipt_requires_condition_realization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = {
                "protocol_id": "citylite-coverage-v1",
                "protocol_sha256": "c" * 64,
                "cell_id": "train-route-0",
                "split": "train",
                "episode_index": 0,
                "episode_seed": 42,
            }
            candidate, _ = _candidate(
                root / "captures",
                episode_id="formal-episode-condition-realization-required",
                collection_binding=binding,
            )
            receipt_path = candidate / "formal_capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            del receipt["integrity"]["condition_realization_verified"]
            _write_json(receipt_path, receipt)
            report = verify_candidate_episode(candidate, require_trusted_receipt=False)
            self.assertIn("receipt_condition_realization", {issue.code for issue in report.issues})

    def test_release_manifest_builder_binds_verified_public_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(
                root / "captures", episode_id="formal-episode-release", use_template_stream=True
            )
            dataset_root = root / "dataset"
            supply_chain_path = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=supply_chain_path,
            ).collect(candidate)
            self.assertTrue(result.admitted, result.issues)
            output = root / "release-manifest.json"
            payload = build_release_manifest(
                dataset_root,
                release_id="release-test-001",
                base_url="https://cdn.example.test/rivermark/",
                source_revision="0123456789abcdef",
                output_path=output,
                supply_chain_manifest=supply_chain_path,
            )
            self.assertEqual(validate_release_manifest(payload, require_https=True), ())
            self.assertGreaterEqual(len(payload["shards"]), 11)
            self.assertEqual(
                {shard["source_capture_sha256"] for shard in payload["shards"]},
                {receipt_hash},
            )
            paths = {shard["path"] for shard in payload["shards"]}
            self.assertIn("train/formal-episode-release/cameras/0/rgb.bin", paths)
            self.assertIn("train/formal-episode-release/cameras/rgb_hashes.json", paths)
            self.assertIn("train/formal-episode-release/episode_manifest.json", paths)
            self.assertIn("train/formal-episode-release/metadata/observation_abi.json", paths)
            self.assertEqual(
                {shard["media_type"] for shard in payload["shards"] if shard["modality"] == "metadata"},
                {"application/json"},
            )
            self.assertEqual(payload["accounting"]["failure_summary"]["attempt_count"], 1)
            self.assertEqual(payload["accounting"]["failure_summary"]["admitted_count"], 1)
            self.assertEqual(
                payload["accounting"]["failure_ledger"]["path"],
                "manifests/failure_ledger.jsonl",
            )
            self.assertTrue(output.is_file())
            alternate_supply_chain = _write_release_supply_chain(
                root / "alternate-supply-chain",
                data_receipt_hashes=(receipt_hash,),
            )
            with self.assertRaisesRegex(ReleaseBuildError, "different supply-chain manifest"):
                build_release_manifest(
                    dataset_root,
                    release_id="release-test-001",
                    base_url="https://cdn.example.test/rivermark/",
                    source_revision="0123456789abcdef",
                    supply_chain_manifest=alternate_supply_chain,
                )
            with self.assertRaisesRegex(ReleaseBuildError, "supply-chain manifest"):
                build_release_manifest(
                    dataset_root,
                    release_id="release-test-001-no-audit",
                    base_url="https://cdn.example.test/rivermark/",
                    source_revision="0123456789abcdef",
                )

            (dataset_root / "manifests" / "failure_ledger.jsonl").unlink()
            with self.assertRaisesRegex(ReleaseBuildError, "failure ledger"):
                build_release_manifest(
                    dataset_root,
                    release_id="release-test-001",
                    base_url="https://cdn.example.test/rivermark/",
                    source_revision="0123456789abcdef",
                    supply_chain_manifest=supply_chain_path,
                )

    def test_release_manifest_builder_refuses_empty_formal_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "empty"
            dataset_root.mkdir()
            with self.assertRaisesRegex(ReleaseBuildError, "empty formal dataset"):
                build_release_manifest(
                    dataset_root,
                    release_id="release-test-002",
                    base_url="https://cdn.example.test/rivermark/",
                    source_revision="0123456789abcdef",
                )

    def test_trusted_candidate_is_projected_and_indexed_without_private_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-a", use_template_stream=True)
            dataset_root = root / "dataset"
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=supply_chain,
            ).collect(candidate)
            self.assertTrue(result.admitted, result.issues)
            self.assertIsNotNone(result.episode_root)
            release = result.episode_root
            assert release is not None
            self.assertTrue((candidate / "episode_manifest.json").is_file(), "source capture must be retained")
            self.assertFalse((release / "evaluator_private").exists())
            self.assertEqual(verify_dataset_integrity(dataset_root).issues, ())
            index = json.loads((dataset_root / "manifests" / "dataset_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["episode_count"], 1)
            self.assertEqual(index["episodes"][0]["episode_id"], "formal-episode-a")
            self.assertTrue((release / "cameras" / "0" / "rgb.bin").is_file())
            self.assertTrue((release / "cameras" / "1" / "rgb.bin").is_file())

    def test_untrusted_or_pilot_like_capture_is_quarantined_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-b")
            dataset_root = root / "dataset"
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[],
                supply_chain_manifest=supply_chain,
            ).collect(candidate)
            self.assertFalse(result.admitted)
            self.assertIsNotNone(result.quarantine_record)
            codes = {issue.code for issue in result.issues}
            self.assertIn("untrusted_capture_receipt", codes)
            quarantine = json.loads(result.quarantine_record.read_text(encoding="utf-8"))
            self.assertTrue(quarantine["source_retained"])
            self.assertTrue((candidate / "episode_manifest.json").is_file())
            self.assertFalse((dataset_root / "train" / "formal-episode-b").exists())

    def test_lineage_overlap_between_splits_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_layout = _digest("same-layout-lineage")
            first, first_hash = _candidate(
                root / "captures", episode_id="formal-episode-c", split="train", layout_lineage=common_layout
            )
            second, second_hash = _candidate(
                root / "captures", episode_id="formal-episode-d", split="validation", layout_lineage=common_layout
            )
            dataset_root = root / "dataset"
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(first_hash, second_hash),
            )
            collector = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[first_hash, second_hash],
                supply_chain_manifest=supply_chain,
            )
            self.assertTrue(collector.collect(first).admitted)
            rejected = collector.collect(second)
            self.assertFalse(rejected.admitted)
            self.assertIn("split_lineage_overlap", {issue.code for issue in rejected.issues})
            payload, issues = plan_split_authority([first, second])
            self.assertIsNone(payload)
            self.assertIn("split_lineage_overlap", {issue.code for issue in issues})

    def test_tampered_release_payload_and_stale_index_fail_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-e")
            dataset_root = root / "dataset"
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                dataset_root,
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=supply_chain,
            ).collect(candidate)
            self.assertTrue(result.admitted)
            release = result.episode_root
            assert release is not None
            (release / "streams" / "state.jsonl").write_text("tampered\n", encoding="utf-8")
            report = verify_dataset_integrity(dataset_root)
            self.assertFalse(report.valid)
            self.assertIn("manifest_file_hash", {issue.code for issue in report.issues})
            self.assertIn("dataset_index_stale", {issue.code for issue in report.issues})

    def test_candidate_requires_complete_template_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-f", use_template_stream=True)
            index_path = candidate / "cameras" / "rgb_hashes.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["files"].pop()
            _write_json(index_path, index)
            report = verify_candidate_episode(candidate, trusted_receipt_hashes=[receipt_hash])
            codes = {issue.code for issue in report.issues}
            self.assertIn("manifest_file_hash", codes)
            self.assertIn("file_hash", codes)

    def test_non_distributed_label_is_verified_but_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-g")
            label_path = candidate / "labels" / "semantic.json"
            label_hash = _write_text(label_path, '{"class":1}\n')
            manifest_path = candidate / "episode_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learning_labels"] = {"distributed": False, "modalities": ["semantic_segmentation"]}
            manifest["streams"].append(
                {
                    "stream_id": "semantic",
                    "partition": "learning_labels",
                    "modality": "semantic_segmentation",
                    "media_type": "application/json",
                    "sample_count": 1,
                    "timestamp_field": "sensor_time_ns",
                    "path": "labels/semantic.json",
                    "sha256": label_hash,
                }
            )
            _write_json(manifest_path, manifest)
            receipt_path = candidate / "formal_capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["episode_manifest_sha256"] = sha256_file(manifest_path)
            _write_json(receipt_path, receipt)
            receipt_hash = sha256_file(receipt_path)
            report = verify_candidate_episode(candidate, trusted_receipt_hashes=[receipt_hash])
            self.assertTrue(report.valid, report.issues)
            supply_chain = _write_release_supply_chain(
                root,
                data_receipt_hashes=(receipt_hash,),
            )
            result = DatasetCollector(
                root / "dataset",
                trusted_receipt_hashes=[receipt_hash],
                supply_chain_manifest=supply_chain,
            ).collect(candidate)
            self.assertTrue(result.admitted, result.issues)
            assert result.episode_root is not None
            self.assertFalse((result.episode_root / "labels" / "semantic.json").exists())
            release_manifest = json.loads((result.episode_root / "episode_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(release_manifest["learning_labels"], {"distributed": False, "modalities": []})
            self.assertEqual(verify_dataset_integrity(root / "dataset").issues, ())

    def test_unbound_private_directory_is_rejected_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, receipt_hash = _candidate(root / "captures", episode_id="formal-episode-h")
            _write_text(candidate / "evaluator_private" / "truth.json", '{"not":"released"}\n')
            report = verify_candidate_episode(candidate, trusted_receipt_hashes=[receipt_hash])
            codes = {issue.code for issue in report.issues}
            self.assertIn("candidate_private_partition", codes)
            self.assertIn("unexpected_candidate_file", codes)


if __name__ == "__main__":
    unittest.main()
