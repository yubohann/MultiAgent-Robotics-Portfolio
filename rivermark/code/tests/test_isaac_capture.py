from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("PyTorch is required for Isaac capture boundary tests") from error


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.isaac_capture import (
    AGENT_COUNT,
    CaptureStorageBudget,
    PRIVATE_TARGET_ORIGIN,
    PRIVATE_TARGET_PLACEMENT_SCHEMA,
    HOVER_THRUST_PER_ROTOR_N,
    IDENTITY_MARKER_RADIUS_M,
    INITIAL_HOVER_RPS,
    MAX_CF2X_ANGULAR_VELOCITY_RADPS,
    MAX_CF2X_LINEAR_VELOCITY_MPS,
    ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD,
    ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M,
    ONBOARD_CAMERA_PITCH_DOWN_RAD,
    OVERVIEW_WITNESS_FOCAL_LENGTH_MM,
    OVERVIEW_WITNESS_IMAGE_HEIGHT,
    OVERVIEW_WITNESS_IMAGE_WIDTH,
    PrivateEvaluatorManifestError,
    RadarUnavailableError,
    SensorPhysicsSmokeReceiptError,
    SWARM_AGENT_LITERAL_PRIM_PATHS,
    _onboard_visual_intrusion_evidence,
    _onboard_scene_content_evidence,
    _onboard_semantic_frame_evidence,
    _redact_private_target_metadata,
    _target_semantic_visibility_evidence,
    _target_semantic_slots,
    _target_visibility_checkpoint_summary,
    _target_visibility_rollout_summary,
    _capture_target_visibility_execution_window,
    _public_follow_view_from_body_pose,
    _public_route_witness_schedule,
    _public_route_witness_view_at_time_ns,
    _require_onboard_visual_integrity,
    _require_onboard_scene_content,
    _require_overview_tracked_agent_visibility,
    _set_public_follow_overview_view,
    _set_public_route_witness_overview_view,
    _LiteralUsdWorldPose,
    _RuntimeTargetUsdObservation,
    _audit_literal_city_lite_usd_spawn_poses,
    _audit_runtime_target_usd_authoring,
    _runtime_target_sphere_prim,
    _runtime_target_class_labels,
    _city_lite_spawn_states,
    _city_lite_initial_root_states,
    _city_lite_initial_thruster_rps,
    _verify_literal_city_lite_spawn,
    _make_multirotor_cfgs,
    _artifact_hashes,
    _capture_quality_observations,
    _camera_mount_quat_wxyz,
    _captured_frame_count,
    _capture_storage_budget,
    _native_t2_motion_contract_for_capture,
    _bind_sensor_physics_smoke_receipt,
    _public_capture_failure,
    _resolve_collection_binding,
    _run_capture_preflight,
    _overview_archive_frame_indices,
    _acquire_capture_app_launcher_lease,
    _close_capture_resources,
    _evaluate_and_record_runtime_safety,
    _enforce_system_commit_guard,
    _enforce_foreign_native_process_guard,
    _enforce_runtime_storage_guard,
    _persist_receipt_snapshot,
    _persist_terminal_capture_state,
    _failure_ledger_classification,
    _record_raw_capture_attempt,
    _sha256,
    _initial_route_heading_yaws_rad,
    _look_at_quat_wxyz,
    _make_sensors,
    _onboard_semantic_metadata,
    _overview_city_content_evidence,
    _persist_initial_overview_failure_diagnostics,
    _overview_tracked_agent_visibility_evidence,
    _require_overview_city_content,
    _overview_view_spec,
    _camera_pose_closure_from_usd,
    _onboard_camera_fabric_pose_diagnostic,
    _onboard_camera_frame_counter,
    _onboard_camera_mount_diagnostics,
    _prepare_onboard_camera_local_mount,
    _require_onboard_camera_render_read_fence,
    _SensorUpdateTimeline,
    _to_numpy,
    _quat_rotate,
    _world_camera_quat_from_usd_axes,
    _velocity_yaw_controller_target,
    _validate_args,
    _validate_private_manifest_input,
    build_parser,
    main,
    validate_external_private_evaluator_manifest,
    validate_private_target_execution_window,
    validate_private_target_geometry,
)
from rivermark_benchmark.citylite_scene import (
    AABB,
    CITY_LITE_ROUTE_FAMILY_A_ID,
    CITY_LITE_ROUTE_FAMILY_B_ID,
    CITY_LITE_TARGET_REGION_A_ID,
    PUBLIC_ROUTES_W_M,
    aabb_geometry_sha256,
    resolve_public_route_family,
)
from rivermark_benchmark.citylite_task import (
    sample_private_targets,
    target_visibility_geometry_contract,
)
from rivermark_benchmark.isaac_runtime_safety import (
    CONTACT_ABORT_FORCE_N,
    INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
    RuntimeSafetyAbort,
    physics_time_ns,
    runtime_safety_receipt_template,
)
from rivermark_benchmark.failure_ledger import load_failure_ledger, summarize_failure_ledger
from rivermark_benchmark.preflight import PreflightCheck, PreflightReport


class IsaacCaptureBoundaryTests(unittest.TestCase):
    def test_onboard_camera_mount_is_explicit_downward_pitch_in_world_convention(self) -> None:
        pitch = ONBOARD_CAMERA_PITCH_DOWN_RAD
        self.assertAlmostEqual(pitch, math.radians(15.0))
        quaternion = _camera_mount_quat_wxyz(pitch)
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0)

        # IsaacLab's Camera ``world`` convention uses +X as the optical axis.
        # Rotate +X with the WXYZ quaternion and verify the documented +Y
        # positive rotation points it below the horizontal plane.
        w, x, y, z = quaternion
        vx, vy, vz = 1.0, 0.0, 0.0
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        observed = (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )
        self.assertGreater(observed[0], 0.95)
        self.assertAlmostEqual(observed[1], 0.0, places=7)
        self.assertLess(observed[2], -0.20)
        self.assertAlmostEqual(observed[2], -math.sin(pitch), places=7)

    def test_onboard_camera_mount_rejects_invalid_pitch(self) -> None:
        with self.assertRaises(ValueError):
            _camera_mount_quat_wxyz(math.pi / 2.0)
        with self.assertRaises(ValueError):
            _camera_mount_quat_wxyz(float("nan"))

    def test_captured_frame_count_matches_final_partial_stride(self) -> None:
        self.assertEqual(_captured_frame_count(2400, 1), 2400)
        self.assertEqual(_captured_frame_count(2400, 10), 240)
        self.assertEqual(_captured_frame_count(21, 10), 3)
        with self.assertRaises(ValueError):
            _captured_frame_count(0, 10)

    def test_capture_storage_budget_tracks_retained_frames_and_overview_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            default_args = build_parser().parse_args(
                ["--output-dir", str(Path(temporary) / "default")]
            )
            budget = _capture_storage_budget(default_args)
            self.assertIsInstance(budget, CaptureStorageBudget)
            self.assertEqual(budget.sensor_frame_count, 240)
            self.assertEqual(budget.overview_frame_count, 25)
            self.assertGreater(budget.required_bytes, budget.finalization_peak_bytes)

            full_rate_args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(Path(temporary) / "full-rate"),
                    "--steps",
                    "2400",
                    "--capture-stride",
                    "1",
                    "--overview-width",
                    "16",
                    "--overview-height",
                    "16",
                ]
            )
            full_rate = _capture_storage_budget(full_rate_args)
            self.assertEqual(full_rate.sensor_frame_count, 2400)
            self.assertGreater(full_rate.required_bytes, budget.required_bytes)

    def test_capture_defaults_use_the_shared_measured_commit_guard(self) -> None:
        args = build_parser().parse_args(["--output-dir", "capture"])
        self.assertEqual(args.preflight_commit_percent, 65.0)
        self.assertEqual(args.abort_commit_percent, 82.0)

    def test_preflight_rejects_a_declared_storage_budget_below_derived_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    str(output),
                    "--estimated-capture-gib",
                    "0.001",
                ]
            )
            receipt: dict[str, object] = {}
            with self.assertRaisesRegex(RuntimeError, "below the derived capture storage reservation"):
                _run_capture_preflight(args, output, receipt)
            self.assertIn("capture_storage_budget", receipt)

    def test_private_route_failure_receipt_redacts_exception_details(self) -> None:
        error = ValueError("private-object-00 at [1.0, 2.0, 3.0]")
        public = _public_capture_failure(error, private_route=True)
        serialized = json.dumps(public, sort_keys=True)
        self.assertNotIn("private-object-00", serialized)
        self.assertNotIn("1.0, 2.0, 3.0", serialized)
        self.assertTrue(public["traceback_redacted"])
        self.assertTrue(public["private_inputs_redacted"])

        development = _public_capture_failure(error, private_route=False)
        self.assertIn("private-object-00", str(development))

    def test_runtime_storage_guard_fails_closed_before_external_space_is_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capture"
            output.mkdir()
            (output / "capture_progress.json").write_text("{}\n", encoding="utf-8")
            args = build_parser().parse_args(["--output-dir", str(output)])
            budget = _capture_storage_budget(args)
            receipt: dict[str, object] = {}
            with patch(
                "rivermark_benchmark.isaac_capture.shutil.disk_usage",
                return_value=SimpleNamespace(free=0),
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime storage reservation"):
                    _enforce_runtime_storage_guard(
                        args,
                        receipt,
                        phase="before_sensor_spool",
                        output_dir=output,
                        budget=budget,
                    )
            guard = receipt["runtime_storage_guard"]
            self.assertIsInstance(guard, dict)
            self.assertFalse(guard["events"][0]["passed"])

    def test_overview_archive_schedule_is_index_only_and_covers_both_boundaries(self) -> None:
        self.assertEqual(_overview_archive_frame_indices(1), (0,))
        self.assertEqual(_overview_archive_frame_indices(10), (0, 9))
        self.assertEqual(_overview_archive_frame_indices(11), (0, 10))
        self.assertEqual(_overview_archive_frame_indices(21), (0, 10, 20))
        self.assertEqual(_overview_archive_frame_indices(22), (0, 10, 20, 21))
        with self.assertRaises(ValueError):
            _overview_archive_frame_indices(0)
        with self.assertRaises(ValueError):
            _overview_archive_frame_indices(2, 0)

    def test_sensor_timeline_does_not_double_advance_retained_contact_reads(self) -> None:
        class FakeSensor:
            def __init__(self) -> None:
                self.total_dt_s = 0.0
                self.calls: list[tuple[float, bool]] = []

            def update(self, dt_s: float, *, force_recompute: bool) -> None:
                self.total_dt_s += float(dt_s)
                self.calls.append((float(dt_s), bool(force_recompute)))

        sensor = FakeSensor()
        timeline = _SensorUpdateTimeline()
        dt_s = 0.005
        timeline.update(sensor, time_ns=physics_time_ns(0, dt_s))
        for step in range(1, 6):
            timestamp = physics_time_ns(step, dt_s)
            timeline.update(sensor, time_ns=timestamp)
            timeline.update(sensor, time_ns=timestamp)
        self.assertAlmostEqual(sensor.total_dt_s, 5 * dt_s, places=12)
        self.assertEqual(len(sensor.calls), 1 + 2 * 5)
        self.assertEqual(sensor.calls[0][0], 0.0)
        for step in range(5):
            self.assertAlmostEqual(sensor.calls[1 + 2 * step][0], dt_s, places=12)
            self.assertEqual(sensor.calls[2 + 2 * step][0], 0.0)
        self.assertTrue(all(force for _, force in sensor.calls))

    @staticmethod
    def _external_manifest() -> dict[str, object]:
        structural_aabbs = (AABB((-4.0, -4.0, 0.0), (4.0, 4.0, 19.0)),)
        sampled_targets = sample_private_targets(
            seed=17,
            target_count=4,
            target_region_id=CITY_LITE_TARGET_REGION_A_ID,
            visibility_bucket="direct-visible-v1",
            routes_w_m=PUBLIC_ROUTES_W_M,
            structural_aabbs=structural_aabbs,
            radius_m=0.30,
            obstacle_clearance_m=0.85,
            minimum_route_separation_m=2.0,
            minimum_pairwise_separation_m=1.5,
        )
        return {
            "schema": "org.rivermark.evaluator-private-search-manifest.v1",
            "environment_id": "RIVERMARK_CITY_LITE_v1",
            "city_lite_scene_contract_sha256": "a" * 64,
            "city_lite_scene_payload_sha256": "b" * 64,
            "task_variant_id": "isaac-eight-agent-public-waypoint-search-v1",
            "sampled_before_policy_start": True,
            "route_conditioning": "public_only",
            "target_origin": {
                "kind": PRIVATE_TARGET_ORIGIN,
                "candidate_pool_released": False,
                "seed_released": False,
                "coordinates_released": False,
            },
            "target_placement_contract": {
                "schema": PRIVATE_TARGET_PLACEMENT_SCHEMA,
                "obstacle_clearance_m": 0.85,
                "minimum_route_separation_m": 2.0,
                "minimum_pairwise_separation_m": 1.5,
            },
            "target_visibility_contract": target_visibility_geometry_contract(
                route_family_id=CITY_LITE_ROUTE_FAMILY_A_ID,
                routes_w_m=PUBLIC_ROUTES_W_M,
                aabb_geometry_sha256=aabb_geometry_sha256(structural_aabbs),
                target_region_id=CITY_LITE_TARGET_REGION_A_ID,
                visibility_bucket="direct-visible-v1",
            ),
            "targets": [
                {
                    "target_id": f"fixture-private-{index}",
                    "position_w_m": target["position_w_m"],
                    "radius_m": target["radius_m"],
                    "visibility_bucket": "direct-visible-v1",
                }
                for index, target in enumerate(sampled_targets)
            ],
        }

    def _args(self, *extra: str):
        return build_parser().parse_args(
            [
                "--output-dir",
                "unused",
                "--evaluator-private-manifest",
                "unused-private.json",
                "--evaluator-private-manifest-retention-root",
                "unused-private-retention",
                *extra,
            ]
        )

    def test_radar_requirement_fails_closed_before_isaac_import(self) -> None:
        with self.assertRaisesRegex(RadarUnavailableError, "No validated RTX radar or hardware radar"):
            _validate_args(self._args("--require-radar"))

    def test_nonfinite_and_out_of_range_capture_parameters_are_rejected(self) -> None:
        invalid = (
            ("--steps", "0"),
            ("--warmup-steps", "-1"),
            ("--capture-stride", "0"),
            ("--dt", "0"),
            ("--dt", str(math.nan)),
            ("--maximum-foreign-native-private-commit-gib", "0"),
            ("--maximum-foreign-native-private-commit-gib", str(math.nan)),
            ("--base-thrust", "0"),
            ("--base-thrust", "0.180001"),
            ("--overview-width", "15"),
        )
        for option, value in invalid:
            with self.subTest(option=option, value=value):
                with self.assertRaises(ValueError):
                    _validate_args(self._args(option, value))
        with self.assertRaisesRegex(ValueError, "upstream CF2X hover trim"):
            _validate_args(self._args("--base-thrust", "0.085"))

    def test_fixed_public_route_requires_locked_overview_witness_resolution(self) -> None:
        defaults = self._args()
        self.assertEqual(
            (defaults.overview_width, defaults.overview_height),
            (OVERVIEW_WITNESS_IMAGE_WIDTH, OVERVIEW_WITNESS_IMAGE_HEIGHT),
        )
        _validate_args(defaults)
        with self.assertRaisesRegex(ValueError, "locked 1920x1080 overview witness"):
            _validate_args(
                self._args("--overview-width", "640", "--overview-height", "360")
            )

    def test_fixed_public_route_requires_external_manifest_retention_root(self) -> None:
        args = build_parser().parse_args(
            [
                "--output-dir",
                "unused",
                "--evaluator-private-manifest",
                "unused-private.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "retention-root is required"):
            _validate_args(args)

    def test_initial_overview_gate_failure_persists_raw_native_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            output.mkdir()
            rgb = np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3)
            depth = np.full((1, 2, 4, 1), 17.5, dtype=np.float32)
            semantic = np.full((1, 2, 4, 1), 16, dtype=np.int32)
            root_pos = np.arange(24, dtype=np.float32).reshape(8, 3)
            root_quat = np.zeros((8, 4), dtype=np.float32)
            root_quat[:, 0] = 1.0
            root_velocity = np.zeros((8, 3), dtype=np.float32)
            result = _persist_initial_overview_failure_diagnostics(
                output,
                rgb=rgb,
                depth=depth,
                semantic=semantic,
                semantic_metadata={"idToLabels": {"16": {"class": "agent_identity", "agent_id": "2"}}},
                content_evidence={"passed": True},
                agent_visibility_evidence={"passed": False, "tracked_agent_pixel_count": 26},
                root_pos_w_m=root_pos,
                root_quat_wxyz=root_quat,
                root_lin_vel_w_mps=root_velocity,
                np=np,
            )
            archive = output / result["archive_relative_path"]
            metadata = output / result["metadata_relative_path"]
            self.assertTrue(archive.is_file())
            self.assertTrue(metadata.is_file())
            self.assertEqual(result["archive_sha256"], _sha256(archive))
            self.assertEqual(result["metadata_sha256"], _sha256(metadata))
            with np.load(archive, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["rgb"], rgb)
                np.testing.assert_array_equal(stored["distance_to_image_plane"], depth)
                np.testing.assert_array_equal(stored["semantic_segmentation"], semantic)
                np.testing.assert_array_equal(stored["root_pos_w_m"], root_pos)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertFalse(payload["private_evaluator_coordinates_included"])
            self.assertEqual(payload["overview_agent_visibility_evidence"]["tracked_agent_pixel_count"], 26)

    def test_collection_protocol_arguments_are_atomic_and_indexed(self) -> None:
        incomplete = (
            ("--collection-protocol", "protocol.json"),
            ("--collection-cell-id", "train-route-0"),
            ("--collection-episode-index", "0"),
        )
        for option, value in incomplete:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "must be provided together"):
                    _validate_args(self._args(option, value))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_args(
                self._args(
                    "--collection-protocol",
                    "protocol.json",
                    "--collection-cell-id",
                    "train-route-0",
                    "--collection-episode-index",
                    "-1",
                )
            )
        _validate_args(
            self._args(
                "--collection-protocol",
                "protocol.json",
                "--collection-cell-id",
                "train-route-0",
                "--collection-episode-index",
                "0",
                "--runtime-lock",
                "runtime-lock.json",
                "--isaaclab-source",
                "isaaclab-source",
                "--sensor-physics-smoke-receipt",
                "smoke/isaac_smoke_receipt.json",
            )
        )
        with self.assertRaisesRegex(ValueError, "requires --runtime-lock"):
            _validate_args(
                self._args(
                    "--collection-protocol",
                    "protocol.json",
                    "--collection-cell-id",
                    "train-route-0",
                    "--collection-episode-index",
                    "0",
                )
            )
        with self.assertRaisesRegex(ValueError, "requires --sensor-physics-smoke-receipt"):
            _validate_args(
                self._args(
                    "--collection-protocol",
                    "protocol.json",
                    "--collection-cell-id",
                    "train-route-0",
                    "--collection-episode-index",
                    "0",
                    "--runtime-lock",
                    "runtime-lock.json",
                    "--isaaclab-source",
                    "isaaclab-source",
                )
            )
        with self.assertRaisesRegex(ValueError, "provided together"):
            _validate_args(self._args("--runtime-lock", "runtime-lock.json"))
        with self.assertRaisesRegex(ValueError, "requires --runtime-lock"):
            _validate_args(
                self._args(
                    "--sensor-physics-smoke-receipt",
                    "smoke/isaac_smoke_receipt.json",
                )
            )
        with self.assertRaisesRegex(ValueError, "only valid for fixed_public_route or native_t2_canary"):
            _validate_args(
                self._args(
                    "--control-mode",
                    "sb3_state_only_transfer",
                    "--collection-protocol",
                    "protocol.json",
                    "--collection-cell-id",
                    "train-route-0",
                    "--collection-episode-index",
                    "0",
                )
            )

    def test_native_t2_canary_requires_and_binds_the_dedicated_protocol(self) -> None:
        t2_protocol = ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v1.json"
        t1_protocol = ROOT / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json"
        with self.assertRaisesRegex(ValueError, "native_t2_canary requires --collection-protocol"):
            _validate_args(self._args("--control-mode", "native_t2_canary"))
        with tempfile.TemporaryDirectory() as temporary:
            calibration = Path(temporary) / "calibration.json"
            calibration.write_text("{}\n", encoding="utf-8")
            args = self._args(
                "--control-mode",
                "native_t2_canary",
                "--collection-protocol",
                str(t2_protocol),
                "--collection-cell-id",
                "native-t2-canary-inner-dev-v1",
                "--collection-episode-index",
                "0",
                "--runtime-lock",
                "runtime-lock.json",
                "--isaaclab-source",
                "isaaclab-source",
                "--sensor-physics-smoke-receipt",
                "smoke/isaac_smoke_receipt.json",
                "--cf2x-runtime-calibration",
                str(calibration),
            )
            _validate_args(args)
            binding = _resolve_collection_binding(args)
            self.assertIsNotNone(binding)
            self.assertEqual(binding["protocol_id"], "citylite-native-t2-canary-v1")

            args.collection_protocol = t1_protocol
            with self.assertRaisesRegex(ValueError, "requires the dedicated native T2 canary protocol"):
                _resolve_collection_binding(args)

    def test_native_t2_v3_preflight_uses_its_own_feasible_clock_and_variant(self) -> None:
        t2_protocol = ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v3.json"
        with tempfile.TemporaryDirectory() as temporary:
            calibration = Path(temporary) / "calibration.json"
            calibration.write_text("{}\n", encoding="utf-8")
            args = self._args(
                "--control-mode",
                "native_t2_canary",
                "--collection-protocol",
                str(t2_protocol),
                "--collection-cell-id",
                "native-t2-canary-inner-dev-v3",
                "--collection-episode-index",
                "0",
                "--runtime-lock",
                "runtime-lock.json",
                "--isaaclab-source",
                "isaaclab-source",
                "--sensor-physics-smoke-receipt",
                "smoke/isaac_smoke_receipt.json",
                "--cf2x-runtime-calibration",
                str(calibration),
                "--steps",
                "4800",
            )
            _validate_args(args)
            binding = _resolve_collection_binding(args)
            assert binding is not None
            contract = _native_t2_motion_contract_for_capture(args, collection_binding=binding)
            assert contract is not None
            self.assertEqual(
                contract["task_variant_id"], "isaac-eight-agent-native-t2-search-canary-v3"
            )
            self.assertEqual(contract["motion_contract"]["waypoint_segment_seconds"], 12.0)
            self.assertEqual(
                contract["route_timing_feasibility"]["vertical_speed_budget_mps"], 0.36
            )

    def test_sensor_physics_smoke_binding_is_external_exact_and_fail_closed(self) -> None:
        lock_sha256 = "a" * 64
        source_revision = "b" * 40
        source_tree_sha256 = "c" * 64
        assets = {
            "city_lite_contract_sha256": "d" * 64,
            "cf2x_usd_sha256": "e" * 64,
        }
        runtime_lock = {
            "profile_id": "fixture-runtime",
            "assets": dict(assets),
        }
        payload = {
            "status": "passed",
            "resource_probe_profile": "full",
            "runtime_lock_sha256": lock_sha256,
            "runtime_profile_id": "fixture-runtime",
            "source": {
                "source_revision": source_revision,
                "source_tree_sha256": source_tree_sha256,
                "source_worktree_dirty": False,
            },
            "runtime_audit": {"observed": {"assets": dict(assets)}},
            "scene": {"contract_sha256": assets["city_lite_contract_sha256"]},
        }
        receipt = {
            "source_revision": source_revision,
            "source_tree_sha256": source_tree_sha256,
            "source_worktree_dirty": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "capture"
            output.mkdir()
            smoke = root / "smoke" / "isaac_smoke_receipt.json"
            smoke.parent.mkdir()

            def write_smoke(value: dict, *, stale_sidecar: bool = False) -> None:
                smoke.write_text(
                    json.dumps(value, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                digest = "f" * 64 if stale_sidecar else _sha256(smoke)
                smoke.with_suffix(".sha256").write_text(
                    f"{digest}  isaac_smoke_receipt.json\n",
                    encoding="ascii",
                )

            write_smoke(payload)
            args = self._args(
                "--runtime-lock",
                "runtime-lock.json",
                "--isaaclab-source",
                "isaaclab-source",
                "--sensor-physics-smoke-receipt",
                str(smoke),
            )
            with patch(
                "rivermark_benchmark.isaac_smoke.validate_smoke_receipt",
                return_value=(),
            ), patch(
                "rivermark_benchmark.runtime_lock.runtime_lock_sha256",
                return_value=lock_sha256,
            ):
                _bind_sensor_physics_smoke_receipt(
                    args,
                    output,
                    receipt,
                    runtime_lock,
                )
            self.assertEqual(
                receipt["capture_backend"],
                {
                    "kind": "isaaclab",
                    "build": f"isaaclab:fixture-runtime@sha256:{lock_sha256}",
                    "sensor_physics_smoke_receipt_sha256": _sha256(smoke),
                },
            )
            self.assertFalse((output / "isaac_smoke_receipt.json").exists())

            missing_args = self._args(
                "--runtime-lock",
                "runtime-lock.json",
                "--isaaclab-source",
                "isaaclab-source",
                "--sensor-physics-smoke-receipt",
                str(root / "missing" / "isaac_smoke_receipt.json"),
            )
            with self.assertRaises(SensorPhysicsSmokeReceiptError):
                _bind_sensor_physics_smoke_receipt(
                    missing_args,
                    output,
                    dict(receipt),
                    runtime_lock,
                )

            rejected = (
                ("invalid", payload, ("injected smoke failure",), False),
                ("failed", {**payload, "status": "failed"}, (), False),
                (
                    "revision",
                    {
                        **payload,
                        "source": {**payload["source"], "source_revision": "0" * 40},
                    },
                    (),
                    False,
                ),
                (
                    "runtime",
                    {**payload, "runtime_lock_sha256": "0" * 64},
                    (),
                    False,
                ),
                (
                    "assets",
                    {
                        **payload,
                        "runtime_audit": {
                            "observed": {
                                "assets": {**assets, "cf2x_usd_sha256": "0" * 64}
                            }
                        },
                    },
                    (),
                    False,
                ),
                (
                    "scene_contract",
                    {
                        **payload,
                        "scene": {"contract_sha256": "0" * 64},
                    },
                    (),
                    False,
                ),
                ("tampered", payload, (), True),
            )
            for name, candidate, validation_errors, stale_sidecar in rejected:
                with self.subTest(name=name):
                    write_smoke(candidate, stale_sidecar=stale_sidecar)
                    candidate_receipt = dict(receipt)
                    candidate_receipt.pop("capture_backend", None)
                    with patch(
                        "rivermark_benchmark.isaac_smoke.validate_smoke_receipt",
                        return_value=validation_errors,
                    ), patch(
                        "rivermark_benchmark.runtime_lock.runtime_lock_sha256",
                        return_value=lock_sha256,
                    ), self.assertRaises(SensorPhysicsSmokeReceiptError):
                        _bind_sensor_physics_smoke_receipt(
                            args,
                            output,
                            candidate_receipt,
                            runtime_lock,
                        )
                    self.assertNotIn("capture_backend", candidate_receipt)

            write_smoke(payload)
            dirty_capture_receipt = {
                **receipt,
                "source_worktree_dirty": True,
            }
            dirty_capture_receipt.pop("capture_backend", None)
            with patch(
                "rivermark_benchmark.isaac_smoke.validate_smoke_receipt",
                return_value=(),
            ), patch(
                "rivermark_benchmark.runtime_lock.runtime_lock_sha256",
                return_value=lock_sha256,
            ), self.assertRaisesRegex(
                SensorPhysicsSmokeReceiptError,
                "clean capture source tree",
            ):
                _bind_sensor_physics_smoke_receipt(
                    args,
                    output,
                    dirty_capture_receipt,
                    runtime_lock,
                )
            self.assertNotIn("capture_backend", dirty_capture_receipt)

    def test_windows_commit_guard_rejects_pressure_before_app_launcher(self) -> None:
        args = SimpleNamespace(preflight_commit_percent=65.0, abort_commit_percent=82.0)
        snapshot = {
            "commit_total_bytes": 44 * 1024**3,
            "commit_limit_bytes": 46 * 1024**3,
            "commit_peak_bytes": 45 * 1024**3,
            "commit_percent": 95.65,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            output.mkdir()
            receipt: dict[str, object] = {}
            with patch(
                "rivermark_benchmark.isaac_capture._windows_system_commit_snapshot",
                return_value=snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "95.65%"):
                    _enforce_system_commit_guard(
                        args, receipt, phase="before_app_launcher", output_dir=output
                    )
            guard = receipt["system_commit_guard"]
            self.assertEqual(guard["status"], "active")
            self.assertEqual(guard["last_snapshot"], snapshot)
            self.assertTrue((output / "capture_progress.json").is_file())

    def test_foreign_native_process_guard_rejects_before_app_launcher(self) -> None:
        args = SimpleNamespace(maximum_foreign_native_private_commit_gib=8.0)
        census = {
            "schema": "org.rivermark.foreign-native-process-census.v1",
            "enumerated_native_process_count": 2,
            "minimum_private_commit_bytes": 8 * 1024**3,
            "candidate_count": 1,
            "candidate_private_commit_bytes": 19 * 1024**3,
            "maximum_candidate_private_commit_bytes": 19 * 1024**3,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            output.mkdir()
            receipt: dict[str, object] = {}
            with self.assertRaisesRegex(
                RuntimeError, "another high-commit Python/Kit/Isaac process"
            ):
                _enforce_foreign_native_process_guard(
                    args,
                    receipt,
                    phase="after_app_launcher",
                    output_dir=output,
                    census=census,
                )
            guard = receipt["foreign_native_process_guard"]
            self.assertEqual(guard["status"], "rejected")
            self.assertEqual(guard["last_census"], census)
            self.assertEqual(guard["sample_count"], 1)
            self.assertEqual(guard["last_phase"], "after_app_launcher")
            self.assertEqual(guard["maximum_candidate_count"], 1)
            self.assertEqual(
                guard["maximum_candidate_private_commit_bytes"], 19 * 1024**3
            )
            progress = json.loads((output / "capture_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["stage"], "foreign_native_process_guard_rejected")
            self.assertEqual(progress["phase"], "after_app_launcher")

    def test_foreign_native_process_guard_tracks_zero_candidate_runtime_samples(self) -> None:
        args = SimpleNamespace(maximum_foreign_native_private_commit_gib=8.0)
        census = {
            "schema": "org.rivermark.foreign-native-process-census.v1",
            "enumerated_native_process_count": 1,
            "minimum_private_commit_bytes": 8 * 1024**3,
            "candidate_count": 0,
            "candidate_private_commit_bytes": 0,
            "maximum_candidate_private_commit_bytes": 0,
        }
        receipt: dict[str, object] = {}

        for phase in ("preflight", "sensors_constructed", "rollout_step_0"):
            _enforce_foreign_native_process_guard(
                args,
                receipt,
                phase=phase,
                census=census,
            )

        guard = receipt["foreign_native_process_guard"]
        self.assertEqual(guard["status"], "active")
        self.assertEqual(guard["sample_count"], 3)
        self.assertEqual(guard["last_phase"], "rollout_step_0")
        self.assertEqual(guard["maximum_candidate_count"], 0)
        self.assertEqual(guard["maximum_candidate_private_commit_bytes"], 0)

    def test_foreign_native_process_guard_fails_closed_when_census_unavailable(self) -> None:
        args = SimpleNamespace(maximum_foreign_native_private_commit_gib=8.0)
        receipt: dict[str, object] = {}

        with self.assertRaisesRegex(RuntimeError, "unavailable at simulation_reset"):
            _enforce_foreign_native_process_guard(
                args,
                receipt,
                phase="simulation_reset",
                census=None,
            )

        guard = receipt["foreign_native_process_guard"]
        self.assertEqual(guard["status"], "rejected")
        self.assertEqual(guard["last_census_status"], "unavailable")
        self.assertEqual(guard["sample_count"], 1)

    def test_foreign_native_process_guard_fails_closed_on_malformed_census(self) -> None:
        args = SimpleNamespace(maximum_foreign_native_private_commit_gib=8.0)
        malformed = {
            "schema": "org.rivermark.foreign-native-process-census.v1",
            "enumerated_native_process_count": 0,
            "minimum_private_commit_bytes": 8 * 1024**3,
            "candidate_count": 1,
            "candidate_private_commit_bytes": 19 * 1024**3,
            "maximum_candidate_private_commit_bytes": 19 * 1024**3,
        }
        receipt: dict[str, object] = {}

        with self.assertRaisesRegex(RuntimeError, "malformed at warmup_step_0"):
            _enforce_foreign_native_process_guard(
                args,
                receipt,
                phase="warmup_step_0",
                census=malformed,
            )

        guard = receipt["foreign_native_process_guard"]
        self.assertEqual(guard["status"], "rejected")
        self.assertEqual(guard["last_census_status"], "malformed")
        self.assertEqual(guard["last_census_error"], "candidate_count")

    def test_windows_commit_guard_rejects_after_reset_and_persists_checksum(self) -> None:
        args = SimpleNamespace(preflight_commit_percent=65.0, abort_commit_percent=82.0)
        snapshot = {
            "commit_total_bytes": 44 * 1024**3,
            "commit_limit_bytes": 46 * 1024**3,
            "commit_peak_bytes": 45 * 1024**3,
            "commit_percent": 82.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            output.mkdir()
            receipt: dict[str, object] = {"status": "running"}
            with patch(
                "rivermark_benchmark.isaac_capture._windows_system_commit_snapshot",
                return_value=snapshot,
            ):
                with self.assertRaisesRegex(RuntimeError, "limit 82.00%"):
                    _enforce_system_commit_guard(
                        args, receipt, phase="after_reset", output_dir=output
                    )
            guard = receipt["system_commit_guard"]
            self.assertEqual(guard["last_phase"], "after_reset")
            self.assertEqual(guard["maximum_observed_percent"], 82.0)
            self.assertEqual(guard["maximum_phase"], "after_reset")
            self.assertEqual(guard["maximum_snapshot"], snapshot)
            progress = json.loads((output / "capture_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["stage"], "system_commit_guard_rejected")
            self.assertEqual(progress["phase"], "after_reset")

            receipt["status"] = "failed"
            _persist_receipt_snapshot(output, receipt)
            receipt_path = output / "capture_receipt.json"
            self.assertEqual(
                (output / "capture_receipt.sha256").read_text(encoding="ascii"),
                f"{_sha256(receipt_path)}  capture_receipt.json\n",
            )

    def test_windows_commit_guard_uses_provided_snapshot_without_fabricating_peak(self) -> None:
        args = SimpleNamespace(preflight_commit_percent=65.0, abort_commit_percent=82.0)
        snapshot = {
            "commit_total_bytes": 30 * 1024**3,
            "commit_limit_bytes": 64 * 1024**3,
            "commit_peak_bytes": 31 * 1024**3,
            "commit_percent": 46.875,
        }
        receipt: dict[str, object] = {}
        with patch(
            "rivermark_benchmark.isaac_capture._windows_system_commit_snapshot",
            side_effect=AssertionError("guard must use the telemetry snapshot"),
        ):
            _enforce_system_commit_guard(
                args,
                receipt,
                phase="after_reset",
                snapshot=snapshot,
            )
        guard = receipt["system_commit_guard"]
        self.assertEqual(guard["last_snapshot"], snapshot)
        self.assertEqual(guard["maximum_snapshot"], snapshot)
        self.assertEqual(guard["maximum_phase"], "after_reset")

        unavailable_receipt: dict[str, object] = {}
        with patch(
            "rivermark_benchmark.isaac_capture._windows_system_commit_snapshot",
            side_effect=AssertionError("unavailable telemetry must not be resampled"),
        ):
            _enforce_system_commit_guard(
                args,
                unavailable_receipt,
                phase="after_reset",
                snapshot=None,
            )
        unavailable_guard = unavailable_receipt["system_commit_guard"]
        self.assertEqual(unavailable_guard["status"], "unavailable")
        self.assertNotIn("maximum_observed_percent", unavailable_guard)

    def test_main_preflight_rejection_binds_commit_snapshot_to_telemetry(self) -> None:
        snapshot = {
            "commit_total_bytes": 44 * 1024**3,
            "commit_limit_bytes": 46 * 1024**3,
            "commit_peak_bytes": 45 * 1024**3,
            "commit_percent": 95.65,
        }

        class Telemetry:
            def __init__(self) -> None:
                self.samples: list[dict[str, object]] = []

            def sample(self, phase: str, **_: object) -> dict[str, object]:
                row: dict[str, object] = {
                    "wall_time_ns": 1,
                    "phase": phase,
                    "process": None,
                    "system_commit": snapshot,
                    "gpu": None,
                }
                self.samples.append(row)
                return row

            def as_dict(self) -> dict[str, object]:
                return {
                    "schema": "org.rivermark.resource-telemetry.v1",
                    "sample_count": len(self.samples),
                    "samples": list(self.samples),
                    "maxima": {},
                    "sampling": "explicit_in_process_phase_boundaries",
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            drone = Path(temporary) / "cf2x.usd"
            drone.write_bytes(b"fixture")
            private_manifest = Path(temporary) / "private.json"
            private_manifest.write_text("{}\n", encoding="utf-8")
            telemetry = Telemetry()
            with patch(
                "rivermark_benchmark.isaac_capture.ResourceTelemetry",
                return_value=telemetry,
            ), patch(
                "rivermark_benchmark.isaac_capture._windows_system_commit_snapshot",
                side_effect=AssertionError("guard must use the telemetry snapshot"),
            ), patch("rivermark_benchmark.isaac_capture._capture") as capture:
                result = main(
                    [
                        "--output-dir",
                        str(root),
                        "--drone-usd",
                        str(drone),
                        "--evaluator-private-manifest",
                        str(private_manifest),
                        "--evaluator-private-manifest-retention-root",
                        str(private_manifest.parent),
                    ]
                )
            self.assertEqual(result, 1)
            capture.assert_not_called()
            receipt = json.loads(
                (root / "capture_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["failure"]["type"], "RuntimeError")
            telemetry_receipt = receipt["resource_telemetry"]
            self.assertEqual(telemetry_receipt["sample_count"], 1)
            self.assertEqual(telemetry_receipt["samples"][0]["phase"], "preflight")
            self.assertEqual(telemetry_receipt["samples"][0]["system_commit"], snapshot)
            guard = receipt["system_commit_guard"]
            self.assertEqual(guard["last_phase"], "preflight")
            self.assertEqual(guard["last_snapshot"], snapshot)

    def test_close_failure_still_releases_capture_app_launcher_lease(self) -> None:
        class App:
            def close(self, **_: object) -> None:
                raise RuntimeError("Kit close failed")

        class Lease:
            released = False

            def release(self) -> None:
                self.released = True

        lease = Lease()
        with self.assertRaisesRegex(RuntimeError, "Kit close failed"):
            _close_capture_resources(App(), lease)
        self.assertTrue(lease.released)

    def test_lease_conflict_records_acquiring_state_before_rejection(self) -> None:
        class Lease:
            def acquire(self) -> None:
                raise RuntimeError("lease conflict")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture"
            output.mkdir()
            receipt: dict[str, object] = {}
            with self.assertRaisesRegex(RuntimeError, "lease conflict"):
                _acquire_capture_app_launcher_lease(Lease(), output, receipt)
            self.assertEqual(receipt["app_launcher_lease"]["state"], "acquiring")
            progress = json.loads(
                (output / "capture_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["stage"], "app_launcher_lease_acquiring")

    def test_state_only_transfer_requires_checkpoint_and_forbids_private_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "policy.zip"
            checkpoint.write_bytes(b"fixture")
            transfer_args = build_parser().parse_args(
                [
                    "--output-dir",
                    "unused",
                    "--control-mode",
                    "sb3_state_only_transfer",
                    "--sb3-checkpoint",
                    str(checkpoint),
                ]
            )
            _validate_args(transfer_args)
            self.assertEqual(transfer_args.sb3_max_vertical_speed_mps, 0.05)

            with_private = build_parser().parse_args(
                [
                    "--output-dir",
                    "unused",
                    "--control-mode",
                    "sb3_state_only_transfer",
                    "--sb3-checkpoint",
                    str(checkpoint),
                    "--evaluator-private-manifest",
                    "private.json",
                ]
            )
            with self.assertRaisesRegex(ValueError, "forbids --evaluator-private-manifest"):
                _validate_args(with_private)

        missing = build_parser().parse_args(
            ["--output-dir", "unused", "--control-mode", "sb3_state_only_transfer"]
        )
        with self.assertRaisesRegex(ValueError, "requires --sb3-checkpoint"):
            _validate_args(missing)

    def test_velocity_yaw_lowerer_clips_a_real_eight_cf2x_command(self) -> None:
        allocation = torch.tensor(
            (
                (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
                (-0.046, 0.046, 0.046, -0.046),
                (-0.046, -0.046, 0.046, 0.046),
                (0.006, -0.006, 0.006, -0.006),
            ),
            dtype=torch.float32,
        )

        class Robot:
            device = "cpu"
            allocation_matrix = allocation
            data = SimpleNamespace(
                root_pos_w=torch.tensor(
                    [(-40.0, -12.0, 9.25)] * 8, dtype=torch.float32
                ),
                root_lin_vel_w=torch.zeros((8, 3), dtype=torch.float32),
                root_quat_w=torch.tensor([(1.0, 0.0, 0.0, 0.0)] * 8),
                root_ang_vel_b=torch.zeros((8, 3), dtype=torch.float32),
            )

        math_utils = SimpleNamespace(
            euler_xyz_from_quat=lambda quat: (
                torch.zeros(8),
                torch.zeros(8),
                torch.zeros(8),
            )
        )
        target, position, velocity, yaw_rate = _velocity_yaw_controller_target(
            Robot(),
            torch.tensor([(4.0, 0.0, 2.0)] * 8, dtype=torch.float32),
            torch.full((8,), 3.0, dtype=torch.float32),
            torch.full((8,), 9.25, dtype=torch.float32),
            HOVER_THRUST_PER_ROTOR_N,
            0.2,
            torch,
            math_utils,
        )
        self.assertEqual(tuple(target.shape), (8, 4))
        self.assertTrue(torch.all((target >= 0.0) & (target <= 0.18)))
        self.assertTrue(torch.all(torch.linalg.vector_norm(velocity[:, :2], dim=-1) <= 2.30001))
        self.assertTrue(torch.all(torch.abs(velocity[:, 2]) <= 1.25001))
        self.assertTrue(torch.all(torch.abs(yaw_rate) <= 1.40001))
        self.assertTrue(torch.all(position[:, 2] >= 9.25))

        with self.assertRaisesRegex(ValueError, "world velocity"):
            _velocity_yaw_controller_target(
                Robot(),
                torch.zeros((7, 3), dtype=torch.float32),
                torch.zeros((8,), dtype=torch.float32),
                torch.zeros((8,), dtype=torch.float32),
                HOVER_THRUST_PER_ROTOR_N,
                0.2,
                torch,
                math_utils,
            )

    def test_cf2x_factory_uses_upstream_hover_trim_and_velocity_limits(self) -> None:
        class Config:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        class MultirotorConfig(Config):
            InitialStateCfg = Config

        sim_utils = SimpleNamespace(
            UsdFileCfg=Config,
            RigidBodyPropertiesCfg=Config,
            ArticulationRootPropertiesCfg=Config,
        )
        args = self._args()
        configs = _make_multirotor_cfgs(args, sim_utils, MultirotorConfig, Config)
        self.assertEqual(len(configs), 8)
        self.assertEqual(
            tuple(config.prim_path for config in configs),
            SWARM_AGENT_LITERAL_PRIM_PATHS,
        )
        for config, (position, quaternion) in zip(
            configs, _city_lite_spawn_states(), strict=True
        ):
            rigid = config.spawn.rigid_props
            self.assertEqual(rigid.max_linear_velocity, MAX_CF2X_LINEAR_VELOCITY_MPS)
            self.assertEqual(rigid.max_angular_velocity, MAX_CF2X_ANGULAR_VELOCITY_RADPS)
            self.assertEqual(config.init_state.pos, position)
            self.assertEqual(config.init_state.rot, quaternion)
            self.assertEqual(config.init_state.lin_vel, (0.0, 0.0, 0.0))
            self.assertEqual(config.init_state.ang_vel, (0.0, 0.0, 0.0))
            self.assertEqual(config.init_state.rps, {
                "m1_prop": INITIAL_HOVER_RPS,
                "m2_prop": INITIAL_HOVER_RPS,
                "m3_prop": INITIAL_HOVER_RPS,
                "m4_prop": INITIAL_HOVER_RPS,
            })
            thrusters = config.actuators["thrusters"]
            self.assertEqual(thrusters.thrust_range, (0.0, 0.18))
            self.assertEqual(thrusters.thrust_const_range, (1.0e-6, 1.0e-6))
        self.assertEqual(args.base_thrust, HOVER_THRUST_PER_ROTOR_N)

    def test_onboard_camera_cfg_requests_documented_post_render_pose_read(self) -> None:
        class Config:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.__dict__.update(kwargs)

        class CameraCfg(Config):
            OffsetCfg = Config

        class TiledCameraCfg(CameraCfg):
            pass

        class MultiMeshRayCasterCfg(Config):
            OffsetCfg = Config
            RaycastTargetCfg = Config

        class Sensor:
            def __init__(self, cfg: object) -> None:
                self.cfg = cfg

        class TiledSensor(Sensor):
            pass

        isaaclab_module = ModuleType("isaaclab")
        sensors_module = ModuleType("isaaclab.sensors")
        ray_caster_module = ModuleType("isaaclab.sensors.ray_caster")
        sensors_module.Camera = Sensor
        sensors_module.CameraCfg = CameraCfg
        sensors_module.TiledCamera = TiledSensor
        sensors_module.TiledCameraCfg = TiledCameraCfg
        sensors_module.ContactSensor = Sensor
        sensors_module.ContactSensorCfg = Config
        sensors_module.Imu = Sensor
        sensors_module.ImuCfg = Config
        ray_caster_module.MultiMeshRayCaster = Sensor
        ray_caster_module.MultiMeshRayCasterCfg = MultiMeshRayCasterCfg
        ray_caster_module.patterns = SimpleNamespace(LidarPatternCfg=Config)
        isaaclab_module.sensors = sensors_module
        sensors_module.ray_caster = ray_caster_module
        sim_utils = SimpleNamespace(PinholeCameraCfg=Config)

        with patch.dict(
            sys.modules,
            {
                "isaaclab": isaaclab_module,
                "isaaclab.sensors": sensors_module,
                "isaaclab.sensors.ray_caster": ray_caster_module,
            },
        ):
            _, _, _, _, _, onboard_cfg, overview_cfg, _ = _make_sensors(
                self._args(), sim_utils, []
            )
            tiled_onboard, tiled_overview, *_ = _make_sensors(
                self._args(),
                sim_utils,
                [],
                use_tiled_onboard_camera=True,
                use_tiled_overview_camera=True,
            )

        self.assertIs(onboard_cfg.update_latest_camera_pose, True)
        self.assertIs(overview_cfg.update_latest_camera_pose, True)
        self.assertEqual(onboard_cfg.offset.pos, (0.12, 0.0, 0.04))
        self.assertEqual(onboard_cfg.offset.rot, _camera_mount_quat_wxyz())
        self.assertEqual(onboard_cfg.offset.convention, "world")
        self.assertIsInstance(tiled_onboard, TiledSensor)
        self.assertIsInstance(tiled_overview, TiledSensor)

    def test_literal_usd_spawn_audit_requires_ordered_rigid_route_anchors(self) -> None:
        observed = tuple(
            _LiteralUsdWorldPose(
                prim_path=path,
                position_w_m=position,
                quaternion_wxyz=quaternion,
                rigid_transform_determinant=1.0,
                basis_axis_lengths=(1.0, 1.0, 1.0),
            )
            for path, (position, quaternion) in zip(
                SWARM_AGENT_LITERAL_PRIM_PATHS, _city_lite_spawn_states(), strict=True
            )
        )
        receipt = _audit_literal_city_lite_usd_spawn_poses(observed)
        self.assertEqual(receipt["source"], "fresh_stage_usd_xform_cache_before_sim_reset")
        self.assertEqual(len(receipt["per_agent"]), 8)
        self.assertLessEqual(receipt["max_position_error_m"], 1.0e-6)
        self.assertLessEqual(receipt["max_orientation_error_rad"], 1.0e-6)

        sign_flipped = list(observed)
        first = sign_flipped[0]
        sign_flipped[0] = _LiteralUsdWorldPose(
            prim_path=first.prim_path,
            position_w_m=first.position_w_m,
            quaternion_wxyz=tuple(-value for value in first.quaternion_wxyz),
            rigid_transform_determinant=first.rigid_transform_determinant,
            basis_axis_lengths=first.basis_axis_lengths,
        )
        self.assertLessEqual(
            _audit_literal_city_lite_usd_spawn_poses(tuple(sign_flipped))["max_orientation_error_rad"],
            1.0e-6,
        )

        mutations = (
            tuple(reversed(observed)),
            (
                _LiteralUsdWorldPose(
                    prim_path=observed[0].prim_path,
                    position_w_m=(
                        observed[0].position_w_m[0] + 0.001,
                        observed[0].position_w_m[1],
                        observed[0].position_w_m[2],
                    ),
                    quaternion_wxyz=observed[0].quaternion_wxyz,
                    rigid_transform_determinant=1.0,
                    basis_axis_lengths=(1.0, 1.0, 1.0),
                ),
                *observed[1:],
            ),
            (
                _LiteralUsdWorldPose(
                    prim_path=observed[0].prim_path,
                    position_w_m=observed[0].position_w_m,
                    quaternion_wxyz=(0.0, 0.0, 0.0, 0.0),
                    rigid_transform_determinant=1.0,
                    basis_axis_lengths=(1.0, 1.0, 1.0),
                ),
                *observed[1:],
            ),
            (
                _LiteralUsdWorldPose(
                    prim_path=observed[0].prim_path,
                    position_w_m=observed[0].position_w_m,
                    quaternion_wxyz=observed[0].quaternion_wxyz,
                    rigid_transform_determinant=0.5,
                    basis_axis_lengths=(1.0, 1.0, 1.0),
                ),
                *observed[1:],
            ),
            (
                _LiteralUsdWorldPose(
                    prim_path=observed[0].prim_path,
                    position_w_m=observed[0].position_w_m,
                    quaternion_wxyz=observed[0].quaternion_wxyz,
                    rigid_transform_determinant=1.0,
                    basis_axis_lengths=(2.0, 0.5, 1.0),
                ),
                *observed[1:],
            ),
        )
        for mutated in mutations:
            with self.subTest(mutation=mutated[0]):
                with self.assertRaises(RuntimeError):
                    _audit_literal_city_lite_usd_spawn_poses(mutated)

    def test_runtime_target_usd_audit_is_strict_and_receipt_safe(self) -> None:
        private_manifest = {
            "targets": [
                {
                    "target_id": "private-target-alpha",
                    "position_w_m": [1.25, -2.5, 7.75],
                    "radius_m": 0.30,
                },
                {
                    "target_id": "private-target-beta",
                    "position_w_m": [-3.5, 4.25, 8.5],
                    "radius_m": 0.45,
                },
            ]
        }
        observed = (
            _RuntimeTargetUsdObservation(
                prim_path="/World/SearchTargets/Target_0",
                position_w_m=(1.25, -2.5, 7.75),
                radius_m=0.30,
                bound_extents_m=(0.60, 0.60, 0.60),
                active=True,
                visible=True,
                renderable=True,
                semantic_class_labels=("search_target_slot_000",),
                rigid_transform_determinant=1.0,
                basis_axis_lengths=(1.0, 1.0, 1.0),
            ),
            _RuntimeTargetUsdObservation(
                prim_path="/World/SearchTargets/Target_1",
                position_w_m=(-3.5, 4.25, 8.5),
                radius_m=0.45,
                bound_extents_m=(0.90, 0.90, 0.90),
                active=True,
                visible=True,
                renderable=True,
                semantic_class_labels=("search_target_slot_001",),
                rigid_transform_determinant=1.0,
                basis_axis_lengths=(1.0, 1.0, 1.0),
            ),
        )
        receipt = _audit_runtime_target_usd_authoring(observed, private_manifest)
        self.assertEqual(receipt["target_count"], 2)
        self.assertTrue(receipt["all_targets_active"])
        self.assertTrue(receipt["all_targets_visible"])
        self.assertTrue(receipt["all_targets_renderable"])
        self.assertTrue(receipt["all_targets_have_expected_class_label"])
        self.assertEqual(receipt["maximum_world_position_error_m"], 0.0)
        encoded = json.dumps(receipt, sort_keys=True)
        for forbidden in (
            "private-target-alpha",
            "private-target-beta",
            "position_w_m",
            "SearchTargets/Target_0",
            "1.25",
            "-2.5",
        ):
            self.assertNotIn(forbidden, encoded)

        mutations = (
            (
                "missing target",
                observed[:1],
                "count differs",
            ),
            (
                "wrong path",
                (
                    _RuntimeTargetUsdObservation(
                        prim_path="/World/SearchTargets/Other",
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "prim_path"},
                    ),
                    observed[1],
                ),
                "unstable target-path order",
            ),
            (
                "inactive",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "active"},
                        active=False,
                    ),
                    observed[1],
                ),
                "inactive target",
            ),
            (
                "invisible",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "visible"},
                        visible=False,
                    ),
                    observed[1],
                ),
                "invisible target",
            ),
            (
                "unrenderable",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "renderable"},
                        renderable=False,
                    ),
                    observed[1],
                ),
                "unrenderable target",
            ),
            (
                "missing class label",
                (
                    _RuntimeTargetUsdObservation(
                        **{
                            field: getattr(observed[0], field)
                            for field in observed[0].__dataclass_fields__
                            if field != "semantic_class_labels"
                        },
                        semantic_class_labels=(),
                    ),
                    observed[1],
                ),
                "class semantic label",
            ),
            (
                "wrong class label",
                (
                    _RuntimeTargetUsdObservation(
                        **{
                            field: getattr(observed[0], field)
                            for field in observed[0].__dataclass_fields__
                            if field != "semantic_class_labels"
                        },
                        semantic_class_labels=("search_target_slot_001",),
                    ),
                    observed[1],
                ),
                "mismatched class semantic label",
            ),
            (
                "wrong position",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "position_w_m"},
                        position_w_m=(1.251, -2.5, 7.75),
                    ),
                    observed[1],
                ),
                "position differs",
            ),
            (
                "wrong radius",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "radius_m"},
                        radius_m=0.31,
                    ),
                    observed[1],
                ),
                "radius differs",
            ),
            (
                "wrong bound",
                (
                    _RuntimeTargetUsdObservation(
                        **{field: getattr(observed[0], field) for field in observed[0].__dataclass_fields__ if field != "bound_extents_m"},
                        bound_extents_m=(0.61, 0.60, 0.60),
                    ),
                    observed[1],
                ),
                "bound differs",
            ),
        )
        for name, mutated, message in mutations:
            with self.subTest(mutation=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    _audit_runtime_target_usd_authoring(mutated, private_manifest)

    def test_runtime_target_sphere_prim_accepts_isaaclab_shape_root(self) -> None:
        class FakePrim:
            def __init__(self, kind: str, children: tuple["FakePrim", ...] = ()) -> None:
                self.kind = kind
                self.children = children

            def IsA(self, prim_type: str) -> bool:
                return self.kind == prim_type

        direct = FakePrim("Sphere")
        wrapped_geometry = FakePrim("Sphere")
        wrapped = FakePrim("Xform", (FakePrim("Xform", (wrapped_geometry,)),))
        wrapped_descendants = (wrapped, wrapped.children[0], wrapped_geometry)
        self.assertIs(_runtime_target_sphere_prim(direct, "Sphere"), direct)
        self.assertIs(
            _runtime_target_sphere_prim(
                wrapped, "Sphere", descendants=wrapped_descendants
            ),
            wrapped_geometry,
        )

        for malformed in (
            FakePrim("Xform"),
            FakePrim("Xform", (FakePrim("Sphere"), FakePrim("Sphere"))),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one sphere geometry"):
                _runtime_target_sphere_prim(
                    malformed,
                    "Sphere",
                    descendants=(malformed, *malformed.children),
                )

    def test_runtime_target_class_labels_accepts_root_or_geometry_and_rejects_conflicts(self) -> None:
        class FakeAttr:
            def __init__(self, value):
                self.value = value

            def Get(self):
                return self.value

        class FakeLabelsApi:
            def __init__(self, prim, instance):
                self.prim = prim
                self.instance = instance

            def GetLabelsAttr(self):
                return FakeAttr(self.prim.labels.get(self.instance))

        class FakePrim:
            def __init__(self, labels=None):
                self.labels = dict(labels or {})

            def GetAppliedSchemas(self):
                return [f"SemanticsLabelsAPI:{name}" for name in self.labels]

        class FakeTokenArray:
            """Minimal iterable stand-in for OpenUSD's non-list token arrays."""

            def __init__(self, values):
                self.values = tuple(values)

            def __iter__(self):
                return iter(self.values)

        root = FakePrim({"class": ["search_target_slot_000"]})
        self.assertEqual(
            _runtime_target_class_labels(root, FakeLabelsApi, descendants=(root,)),
            ("search_target_slot_000",),
        )
        token_array_root = FakePrim(
            {"class": FakeTokenArray(["search_target_slot_000"])}
        )
        self.assertEqual(
            _runtime_target_class_labels(
                token_array_root, FakeLabelsApi, descendants=(token_array_root,)
            ),
            ("search_target_slot_000",),
        )
        geometry = FakePrim({"class": ["search_target_slot_001"]})
        wrapped = FakePrim()
        self.assertEqual(
            _runtime_target_class_labels(
                wrapped, FakeLabelsApi, descendants=(wrapped, geometry)
            ),
            ("search_target_slot_001",),
        )
        duplicate_root = FakePrim({"class": ["search_target_slot_000"]})
        duplicate_geometry = FakePrim({"class": ["search_target_slot_000"]})
        self.assertEqual(
            _runtime_target_class_labels(
                duplicate_root,
                FakeLabelsApi,
                descendants=(duplicate_root, duplicate_geometry),
            ),
            ("search_target_slot_000",),
        )
        for malformed in (
            FakePrim(),
            FakePrim({"class": ["search_target_slot_000", "search_target_slot_001"]}),
            FakePrim({"target_id": ["private-id"]}),
        ):
            with self.assertRaisesRegex(RuntimeError, "semantic label"):
                _runtime_target_class_labels(malformed, FakeLabelsApi, descendants=(malformed,))

    def test_literal_city_lite_spawn_proves_authored_defaults_and_records_settling(self) -> None:
        expected_root_states = _city_lite_initial_root_states(torch, "cpu")
        expected_thruster_rps = _city_lite_initial_thruster_rps(torch, "cpu")
        self.assertEqual(tuple(expected_root_states.shape), (8, 13))
        self.assertEqual(tuple(expected_thruster_rps.shape), (8, 4))
        torch.testing.assert_close(
            expected_root_states[:, :3],
            torch.tensor(
                [state[0] for state in _city_lite_spawn_states()], dtype=torch.float32
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected_root_states[:, 3:7],
            torch.tensor(
                [state[1] for state in _city_lite_spawn_states()], dtype=torch.float32
            ),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected_root_states[:, 7:],
            torch.zeros((8, 6), dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected_thruster_rps,
            torch.full((8, 4), INITIAL_HOVER_RPS, dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

        observed_position = expected_root_states[:, :3].clone()
        # This is the observed Isaac reset-scale physical displacement from
        # the City-Lite smoke run.  It is evidence, not a root-state rewrite.
        observed_position[0, 2] -= 0.002452
        observed_linear_velocity = torch.zeros((8, 3), dtype=torch.float32)
        observed_linear_velocity[0, 2] = -0.196151
        robot = SimpleNamespace(
            data=SimpleNamespace(
                default_root_state=expected_root_states.clone(),
                default_thruster_rps=expected_thruster_rps.clone(),
                thrust_target=torch.full(
                    (8, 4), HOVER_THRUST_PER_ROTOR_N, dtype=torch.float32
                ),
                root_pos_w=observed_position,
                root_quat_w=expected_root_states[:, 3:7].clone(),
                root_lin_vel_w=observed_linear_velocity,
                root_ang_vel_b=torch.zeros((8, 3), dtype=torch.float32),
            )
        )
        receipt = _verify_literal_city_lite_spawn(
            robot, expected_root_states, expected_thruster_rps, torch
        )
        self.assertEqual(receipt["literal_prim_paths"], list(SWARM_AGENT_LITERAL_PRIM_PATHS))
        self.assertFalse(receipt["post_reset_root_pose_rewrite"])
        self.assertFalse(receipt["post_reset_root_velocity_rewrite"])
        authored = receipt["authored_defaults"]
        self.assertLessEqual(authored["root_state_max_abs_error"], 1.0e-6)
        self.assertLessEqual(authored["thruster_rps_max_abs_error"], 1.0e-4)
        self.assertLessEqual(authored["thrust_target_max_abs_error_n"], 1.0e-6)
        settling = receipt["post_reset_physics_settling"]
        self.assertEqual(
            settling["classification"], "observed_after_sim_reset_before_first_command"
        )
        self.assertAlmostEqual(settling["max_position_delta_m"], 0.002452, places=5)
        self.assertAlmostEqual(settling["max_orientation_delta_rad"], 0.0, places=6)
        self.assertAlmostEqual(settling["max_linear_velocity_mps"], 0.196151, places=5)
        self.assertAlmostEqual(settling["max_angular_velocity_radps"], 0.0, places=6)

    def test_literal_city_lite_spawn_rejects_unproven_defaults_or_unsafe_live_state(self) -> None:
        expected_root_states = _city_lite_initial_root_states(torch, "cpu")
        expected_thruster_rps = _city_lite_initial_thruster_rps(torch, "cpu")

        def make_robot() -> SimpleNamespace:
            return SimpleNamespace(
                data=SimpleNamespace(
                    default_root_state=expected_root_states.clone(),
                    default_thruster_rps=expected_thruster_rps.clone(),
                    thrust_target=torch.full(
                        (8, 4), HOVER_THRUST_PER_ROTOR_N, dtype=torch.float32
                    ),
                    root_pos_w=expected_root_states[:, :3].clone(),
                    root_quat_w=expected_root_states[:, 3:7].clone(),
                    root_lin_vel_w=torch.zeros((8, 3), dtype=torch.float32),
                    root_ang_vel_b=torch.zeros((8, 3), dtype=torch.float32),
                )
            )

        corruptions = (
            (
                "default_root_state",
                lambda robot: robot.data.default_root_state.__setitem__((2, 0), -99.0),
            ),
            (
                "default_thruster_rps",
                lambda robot: robot.data.default_thruster_rps.__setitem__((4, 3), 0.0),
            ),
            (
                "reset_thrust_target",
                lambda robot: robot.data.thrust_target.__setitem__((6, 1), 0.0),
            ),
            ("live_pose", lambda robot: robot.data.root_pos_w.__setitem__((3, 2), 99.0)),
            (
                "live_orientation",
                lambda robot: robot.data.root_quat_w.__setitem__(
                    (3, slice(None)), torch.tensor((0.0, 1.0, 0.0, 0.0))
                ),
            ),
            (
                "nonfinite_live_velocity",
                lambda robot: robot.data.root_lin_vel_w.__setitem__((1, 0), math.nan),
            ),
        )
        for label, corrupt in corruptions:
            with self.subTest(corruption=label):
                robot = make_robot()
                corrupt(robot)
                with self.assertRaises(RuntimeError):
                    _verify_literal_city_lite_spawn(
                        robot, expected_root_states, expected_thruster_rps, torch
                    )

    def test_post_reset_settling_never_bypasses_contact_or_inter_agent_guard(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)

        def samples() -> dict[str, list[object]]:
            return {
                "physics_step": [],
                "physics_time_ns": [],
                "phase_code": [],
                "frame_outcome_code": [],
                "root_pos_w_m": [],
                "net_contact_forces_w_n": [],
                "max_contact_force_n": [],
            }

        safe_positions = np.asarray(
            [[-30.0 + agent_id * 4.0, 0.0, 11.0] for agent_id in range(8)],
            dtype=np.float32,
        )
        contact_forces = np.zeros((8, 1, 3), dtype=np.float32)
        contact_forces[5, 0, 2] = CONTACT_ABORT_FORCE_N
        contact_guard = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        with self.assertRaises(RuntimeSafetyAbort) as contact_raised:
            _evaluate_and_record_runtime_safety(
                samples(),
                contact_guard,
                previous_positions_w_m=None,
                current_positions_w_m=safe_positions,
                net_contact_forces_w_n=contact_forces,
                structural_aabbs=boxes,
                phase="post_reset",
                physics_step=0,
                physics_dt_s=0.005,
            )
        self.assertEqual(contact_raised.exception.violation["kind"], "contact_force_violation")

        crowded_positions = safe_positions.copy()
        crowded_positions[0] = (0.0, 0.0, 11.0)
        crowded_positions[1] = (
            INTER_AGENT_MINIMUM_CENTER_SEPARATION_M - 1.0e-3,
            0.0,
            11.0,
        )
        separation_guard = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        with self.assertRaises(RuntimeSafetyAbort) as separation_raised:
            _evaluate_and_record_runtime_safety(
                samples(),
                separation_guard,
                previous_positions_w_m=None,
                current_positions_w_m=crowded_positions,
                net_contact_forces_w_n=np.zeros((8, 1, 3), dtype=np.float32),
                structural_aabbs=boxes,
                phase="post_reset",
                physics_step=0,
                physics_dt_s=0.005,
            )
        self.assertEqual(
            separation_raised.exception.violation["kind"],
            "inter_agent_swept_separation_violation",
        )

    def test_runtime_safety_snapshot_is_not_aliased_to_mutable_state(self) -> None:
        boxes = (AABB((-0.1, -0.1, 10.0), (0.1, 0.1, 12.0)),)
        guard = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        samples = {
            "physics_step": [],
            "physics_time_ns": [],
            "phase_code": [],
            "frame_outcome_code": [],
            "root_pos_w_m": [],
            "net_contact_forces_w_n": [],
            "max_contact_force_n": [],
        }
        current = np.asarray(
            [[-2.0 + agent_id * 3.0, 0.0, 11.0] for agent_id in range(8)],
            dtype=np.float32,
        )
        forces = np.zeros((8, 1, 3), dtype=np.float32)
        previous = _evaluate_and_record_runtime_safety(
            samples,
            guard,
            previous_positions_w_m=None,
            current_positions_w_m=current,
            net_contact_forces_w_n=forces,
            structural_aabbs=boxes,
            phase="post_reset",
            physics_step=0,
            physics_dt_s=0.005,
        )
        current[0, 0] = 2.0
        with self.assertRaises(RuntimeSafetyAbort) as raised:
            _evaluate_and_record_runtime_safety(
                samples,
                guard,
                previous_positions_w_m=previous,
                current_positions_w_m=current,
                net_contact_forces_w_n=forces,
                structural_aabbs=boxes,
                phase="rollout",
                physics_step=1,
                physics_dt_s=0.005,
            )
        self.assertEqual(raised.exception.violation["kind"], "structural_aabb_clearance_violation")
        self.assertEqual(samples["frame_outcome_code"], [0, 1])
        self.assertEqual(samples["physics_time_ns"], [0, 5_000_000])

    def test_missing_usd_writes_a_failure_receipt_without_starting_isaac(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            missing_usd = Path(temporary) / "missing-cf2x.usd"
            private_manifest = Path(temporary) / "private" / "evaluator.json"
            private_manifest.parent.mkdir(parents=True)
            private_manifest.write_text("{}\n", encoding="utf-8")
            result = main(
                [
                    "--output-dir",
                    str(root),
                    "--drone-usd",
                    str(missing_usd),
                    "--evaluator-private-manifest",
                    str(private_manifest),
                    "--evaluator-private-manifest-retention-root",
                    str(private_manifest.parent),
                ]
            )
            self.assertEqual(result, 1)
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["ok"])
            self.assertEqual(receipt["failure"]["type"], "FileNotFoundError")
            self.assertFalse(receipt["claim_boundary"]["formal_benchmark_admission"])
            self.assertFalse(receipt["claim_boundary"]["radar_profile_eligible"])
            self.assertFalse(receipt["claim_boundary"]["semantic_labels_policy_visible"])
            self.assertEqual(receipt["task_kind"], "search3d")
            self.assertEqual(
                receipt["capture_integrity"],
                {
                    "online_capture": True,
                    "queue_used": False,
                    "queue_overflow": False,
                    "silent_frame_drop": False,
                    "synchronous_sensor_reads": True,
                    "sensor_step_order": [
                        "command_write",
                        "simulation_step",
                        "state_update",
                        "safety_contact_read",
                        "camera_pose_update",
                        "render",
                        "rgbd_lidar_imu_read",
                        "retained_contact_read",
                        "storage",
                    ],
                    "per_physics_step_safety_contact_reads": True,
                    "retained_contact_read_in_synchronous_sensor_phase": True,
                },
            )
            self.assertEqual(private_manifest.read_text(encoding="utf-8"), "{}\n")
            checksum = (root / "capture_receipt.sha256").read_text(encoding="ascii")
            self.assertTrue(checksum.endswith("  capture_receipt.json\n"))

    def test_existing_empty_output_directory_is_never_reused(self) -> None:
        """The capture root is atomically claimed before any receipt is written."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must not already exist"):
                main(["--output-dir", str(root)])
            self.assertEqual(tuple(root.iterdir()), ())

    def test_rivermark_runs_failure_is_added_to_public_attempt_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rivermark-runs" / "capture"
            missing_usd = Path(temporary) / "missing-cf2x.usd"
            private_manifest = Path(temporary) / "private" / "evaluator.json"
            private_manifest.parent.mkdir(parents=True)
            private_manifest.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "--output-dir",
                        str(root),
                        "--drone-usd",
                        str(missing_usd),
                        "--evaluator-private-manifest",
                        str(private_manifest),
                        "--evaluator-private-manifest-retention-root",
                        str(private_manifest.parent),
                    ]
                ),
                1,
            )
            summary = summarize_failure_ledger(root.parent / "failure_ledger.jsonl")
            self.assertEqual(summary["attempt_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            receipt = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8"))
            _record_raw_capture_attempt(root, receipt)
            self.assertEqual(summarize_failure_ledger(root.parent / "failure_ledger.jsonl")["attempt_count"], 1)

    def test_failure_ledger_classifies_resource_guards_separately_from_capture_errors(self) -> None:
        foreign_guard_receipt = {
            "status": "failed",
            "failure": {"type": "RuntimeError"},
            "foreign_native_process_guard": {"status": "rejected"},
        }
        system_commit_receipt = {
            "status": "failed",
            "failure": {"type": "RuntimeError"},
            "system_commit_guard": {"status": "rejected"},
        }
        ordinary_failure_receipt = {
            "status": "failed",
            "failure": {"type": "RuntimeError"},
        }
        self.assertEqual(
            _failure_ledger_classification(foreign_guard_receipt),
            ("failed", "infrastructure_failure", "foreign_native_process_guard_rejected"),
        )
        self.assertEqual(
            _failure_ledger_classification(system_commit_receipt),
            ("failed", "infrastructure_failure", "system_commit_guard_rejected"),
        )
        self.assertEqual(
            _failure_ledger_classification(ordinary_failure_receipt),
            ("failed", "capture_failure", "runtimeerror"),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rivermark-runs" / "foreign-guard-rejection"
            root.mkdir(parents=True)
            (root / "capture_start.json").write_text(
                json.dumps({"attempt_id": "attempt-" + "c" * 32}),
                encoding="utf-8",
            )
            _persist_receipt_snapshot(root, foreign_guard_receipt)
            _record_raw_capture_attempt(root, foreign_guard_receipt)
            records = load_failure_ledger(root.parent / "failure_ledger.jsonl")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["category"], "infrastructure_failure")
            self.assertEqual(records[0]["reason_code"], "foreign_native_process_guard_rejected")

    def test_terminal_state_persists_receipt_and_ledger_before_resource_close(self) -> None:
        """The inner capture finalizer can call this before Kit shutdown."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rivermark-runs" / "terminal-capture"
            root.mkdir(parents=True)
            diagnostic = root / "failure_diagnostics" / "initial_overview_native.npz"
            diagnostic.parent.mkdir()
            diagnostic.write_bytes(b"native-overview-diagnostics")
            (root / "capture_start.json").write_text(
                json.dumps({"attempt_id": "attempt-" + "a" * 32}),
                encoding="utf-8",
            )
            receipt: dict[str, object] = {
                "status": "captured",
                "ok": True,
                "collection_binding": {
                    "protocol_id": "citylite-coverage-v1",
                    "protocol_sha256": "b" * 64,
                    "cell_id": "train-route-0",
                    "split": "train",
                    "episode_index": 0,
                    "episode_seed": 42,
                },
            }
            _persist_terminal_capture_state(root, receipt)

            stored = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "captured")
            self.assertTrue((root / "capture_receipt.sha256").is_file())
            self.assertEqual(
                stored["artifact_hashes"]["failure_diagnostics/initial_overview_native.npz"],
                {
                    "bytes": len(b"native-overview-diagnostics"),
                    "sha256": _sha256(diagnostic),
                },
            )
            summary = summarize_failure_ledger(root.parent / "failure_ledger.jsonl")
            self.assertEqual(summary["attempt_count"], 1)
            self.assertEqual(summary["quarantined_count"], 1)

            _persist_terminal_capture_state(root, receipt)
            self.assertEqual(summarize_failure_ledger(root.parent / "failure_ledger.jsonl")["attempt_count"], 1)

    def test_receipt_snapshot_is_atomic_when_replacement_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "capture_receipt.json"
            receipt_path.write_text('{"previous": true}\n', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("injected replacement failure")):
                with self.assertRaisesRegex(OSError, "injected replacement failure"):
                    _persist_receipt_snapshot(root, {"status": "captured"})
            self.assertEqual(receipt_path.read_text(encoding="utf-8"), '{"previous": true}\n')
            self.assertEqual(tuple(root.glob(".capture_receipt.json.*.tmp")), ())

    def test_preflight_rejection_persists_receipt_without_entering_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            drone = Path(temporary) / "cf2x.usd"
            drone.write_bytes(b"fixture")
            private_manifest = Path(temporary) / "private.json"
            private_manifest.write_text("{}\n", encoding="utf-8")
            report = PreflightReport(
                checks=(
                    PreflightCheck(
                        "gpu_capacity",
                        False,
                        {"capable_gpus": []},
                        "one capable GPU",
                        "refusing launch",
                    ),
                ),
                source=None,
            )
            with patch("rivermark_benchmark.preflight.run_preflight", return_value=report):
                with patch(
                    "rivermark_benchmark.isaac_capture.foreign_native_process_census",
                    return_value={
                        "schema": "org.rivermark.foreign-native-process-census.v1",
                        "enumerated_native_process_count": 0,
                        "minimum_private_commit_bytes": 8 * 1024**3,
                        "candidate_count": 0,
                        "candidate_private_commit_bytes": 0,
                        "maximum_candidate_private_commit_bytes": 0,
                    },
                ):
                    with patch("rivermark_benchmark.isaac_capture._capture") as capture:
                        result = main(
                            [
                                "--output-dir",
                                str(root),
                                "--drone-usd",
                                str(drone),
                                "--evaluator-private-manifest",
                                str(private_manifest),
                                "--evaluator-private-manifest-retention-root",
                                str(private_manifest.parent),
                            ]
                        )
            self.assertEqual(result, 1)
            capture.assert_not_called()
            receipt = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["preflight"]["valid"])
            self.assertEqual(receipt["preflight"]["checks"][0]["name"], "gpu_capacity")
            self.assertEqual(receipt["failure"]["type"], "RuntimeError")
            self.assertTrue((root / "capture_progress.json").is_file())

    def test_private_evaluator_manifest_must_preexist_outside_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = (root / "capture").resolve()
            with self.assertRaisesRegex(ValueError, "must be outside"):
                _validate_private_manifest_input(output, output / "evaluator-private.json")
            with self.assertRaisesRegex(FileNotFoundError, "external private evaluator manifest is missing"):
                _validate_private_manifest_input(output, root / "missing.json")

    def test_external_private_manifest_must_be_json_and_existing_inputs_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = (root / "capture").resolve()
            existing = root / "private.json"
            existing.write_text("{}", encoding="utf-8")
            _validate_private_manifest_input(output, existing.resolve())
            with self.assertRaisesRegex(ValueError, "must name a .json"):
                _validate_private_manifest_input(output, (root / "private.bin").resolve())

    def test_external_manifest_contract_and_placement_are_fail_closed(self) -> None:
        manifest = self._external_manifest()
        targets = validate_external_private_evaluator_manifest(
            manifest,
            city_lite_scene_contract_sha256="a" * 64,
            city_lite_scene_payload_sha256="b" * 64,
        )
        self.assertEqual(len(targets), 4)
        report = validate_private_target_geometry(
            manifest,
            structural_aabbs=(AABB((-4.0, -4.0, 0.0), (4.0, 4.0, 19.0)),),
            public_routes_w_m=PUBLIC_ROUTES_W_M,
            city_lite_scene_contract_sha256="a" * 64,
            city_lite_scene_payload_sha256="b" * 64,
        )
        self.assertEqual(report["target_count"], 4)

        origin = manifest["target_origin"]
        assert isinstance(origin, dict)
        origin["candidate_pool_released"] = True
        with self.assertRaisesRegex(PrivateEvaluatorManifestError, "external private evaluator"):
            validate_external_private_evaluator_manifest(
                manifest,
                city_lite_scene_contract_sha256="a" * 64,
                city_lite_scene_payload_sha256="b" * 64,
            )

        missing_visibility = self._external_manifest()
        del missing_visibility["target_visibility_contract"]
        with self.assertRaisesRegex(PrivateEvaluatorManifestError, "target_visibility_contract"):
            validate_external_private_evaluator_manifest(
                missing_visibility,
                city_lite_scene_contract_sha256="a" * 64,
                city_lite_scene_payload_sha256="b" * 64,
            )

    def test_private_target_visibility_uses_per_camera_slot_labels_and_redacts_ids(self) -> None:
        target_ids = [f"fixture-private-{index}" for index in range(4)]
        target_slots = _target_semantic_slots(4)
        semantic = np.zeros((AGENT_COUNT, 8, 8, 1), dtype=np.int32)
        metadata = {"per_camera": [{"id_to_labels": {}} for _ in range(AGENT_COUNT)]}
        for index, (target_id, target_slot) in enumerate(zip(target_ids, target_slots), start=1):
            semantic[index - 1, 0:3, 0:3, 0] = index
            labels = metadata["per_camera"][index - 1]["id_to_labels"]
            labels[str(index)] = {"class": target_slot}

        evidence = _target_semantic_visibility_evidence(
            semantic,
            metadata,
            target_slots,
            minimum_pixels=8,
        )

        self.assertTrue(evidence["passed"], evidence)
        self.assertEqual(
            {row["visible_sensor_frames"] for row in evidence["per_target_slot"].values()},
            {1},
        )
        metadata["unrelated_private_text"] = target_ids[0]
        redacted = _redact_private_target_metadata(
            metadata, private_target_ids=target_ids
        )
        self.assertNotIn("target_id", repr(redacted))
        self.assertNotIn("fixture-private", repr(redacted))

        semantic[3] = 0
        missing = _target_semantic_visibility_evidence(
            semantic,
            metadata,
            target_slots,
            minimum_pixels=8,
        )
        self.assertFalse(missing["passed"])

    def test_private_target_visibility_does_not_merge_camera_local_semantic_ids(self) -> None:
        target_slot = _target_semantic_slots(1)[0]
        semantic = np.zeros((AGENT_COUNT, 8, 8, 1), dtype=np.int32)
        semantic[0, 0:3, 0:3, 0] = 17
        semantic[1, :, :, 0] = 17
        metadata = {
            "per_camera": [
                {"id_to_labels": {"17": {"class": target_slot}}},
                {"id_to_labels": {"17": {"class": "prop_structure"}}},
                *({"id_to_labels": {}} for _ in range(AGENT_COUNT - 2)),
            ]
        }
        evidence = _target_semantic_visibility_evidence(
            semantic, metadata, (target_slot,), minimum_pixels=8
        )
        self.assertTrue(evidence["passed"], evidence)
        self.assertEqual(
            evidence["per_target_slot"][target_slot]["maximum_pixels_in_one_camera"], 9
        )

    def test_onboard_semantic_frame_evidence_uses_one_frame_local_mapping_for_both_gates(self) -> None:
        target_slots = _target_semantic_slots(4)
        depth = np.full((AGENT_COUNT, 8, 8, 1), 10.0, dtype=np.float32)
        semantic = np.ones((AGENT_COUNT, 8, 8, 1), dtype=np.int32)
        metadata = {
            "per_camera": [
                {"id_to_labels": {"0": {"class": "background"}, "1": {"class": "building"}}}
                for _ in range(AGENT_COUNT)
            ]
        }
        for agent_id, target_slot in enumerate(target_slots):
            semantic[agent_id, 0:3, 0:3, 0] = agent_id + 2
            metadata["per_camera"][agent_id]["id_to_labels"][str(agent_id + 2)] = {
                "class": target_slot
            }

        scene_content, target_visibility = _onboard_semantic_frame_evidence(
            depth, semantic, metadata, target_slots
        )

        self.assertTrue(scene_content["passed"], scene_content)
        self.assertIsNotNone(target_visibility)
        assert target_visibility is not None
        self.assertTrue(target_visibility["passed"], target_visibility)
        no_targets_scene_content, no_targets_visibility = _onboard_semantic_frame_evidence(
            depth, semantic, metadata
        )
        self.assertTrue(no_targets_scene_content["passed"])
        self.assertIsNone(no_targets_visibility)

    def test_private_target_visibility_rejects_legacy_flat_or_generic_metadata(self) -> None:
        target_slot = _target_semantic_slots(1)[0]
        semantic = np.zeros((AGENT_COUNT, 8, 8, 1), dtype=np.int32)
        semantic[0, 0:3, 0:3, 0] = 1
        legacy = {"id_to_labels": {"1": {"class": "search_target", "instance": target_slot}}}
        evidence = _target_semantic_visibility_evidence(
            semantic, legacy, (target_slot,), minimum_pixels=8
        )
        self.assertFalse(evidence["passed"])
        self.assertIn("one ID mapping per rendered camera", evidence["failures"][0])

    def test_target_visibility_checkpoint_summary_uses_capture_slot_abi(self) -> None:
        evidence = {
            "schema": "org.rivermark.isaac-target-visibility-evidence.v1",
            "passed": False,
            "per_target_slot": {
                "search_target_slot_000": {"visible_sensor_frames": 1},
                "search_target_slot_001": {"visible_sensor_frames": 0},
            },
        }
        self.assertEqual(
            _target_visibility_checkpoint_summary(evidence),
            {
                "schema": "org.rivermark.isaac-target-visibility-evidence.v1",
                "passed": False,
                "visible_target_count": 1,
                "target_count": 2,
            },
        )
        self.assertIsNone(_target_visibility_checkpoint_summary(None))
        with self.assertRaisesRegex(ValueError, "per_target_slot"):
            _target_visibility_checkpoint_summary({"schema": "test", "passed": False})

    def test_target_visibility_rollout_summary_is_redacted_and_persistable(self) -> None:
        summary = _target_visibility_rollout_summary(
            _target_semantic_slots(2),
            (
                {
                    "per_target_slot": {
                        "search_target_slot_000": {
                            "maximum_pixels_in_one_camera": 13,
                            "visible_sensor_frames": 1,
                            "private_target_id": "never-persist-this",
                        },
                        "search_target_slot_001": {
                            "maximum_pixels_in_one_camera": 5,
                            "visible_sensor_frames": 0,
                        },
                    }
                },
            ),
        )
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["failed_target_slots"], ["search_target_slot_001"])
        self.assertEqual(summary["per_target_slot"]["search_target_slot_000"]["max_pixels"], 13)
        self.assertNotIn("never-persist-this", repr(summary))

    def test_private_target_execution_window_must_match_capture_arguments(self) -> None:
        manifest = self._external_manifest()
        args = self._args()
        execution_window = _capture_target_visibility_execution_window(args)
        validate_private_target_execution_window(
            manifest, execution_window=execution_window
        )
        args.steps += 1
        with self.assertRaisesRegex(PrivateEvaluatorManifestError, "does not match capture arguments"):
            validate_private_target_execution_window(
                manifest,
                execution_window=_capture_target_visibility_execution_window(args),
            )

    def test_private_placement_rejects_aabb_overlap_and_public_route_targets(self) -> None:
        manifest = self._external_manifest()
        targets = manifest["targets"]
        assert isinstance(targets, list)
        first = targets[0]
        assert isinstance(first, dict)
        first["position_w_m"] = [0.0, 0.0, 12.0]
        with self.assertRaisesRegex(PrivateEvaluatorManifestError, "overlaps protected"):
            validate_private_target_geometry(
                manifest,
                structural_aabbs=(AABB((-4.0, -4.0, 0.0), (4.0, 4.0, 19.0)),),
                public_routes_w_m=PUBLIC_ROUTES_W_M,
                city_lite_scene_contract_sha256="a" * 64,
                city_lite_scene_payload_sha256="b" * 64,
            )
        first["position_w_m"] = [0.0, 30.0, 12.0]
        with self.assertRaisesRegex(PrivateEvaluatorManifestError, "too close to a public route"):
            validate_private_target_geometry(
                manifest,
                structural_aabbs=(AABB((-4.0, -4.0, 0.0), (4.0, 4.0, 19.0)),),
                public_routes_w_m=PUBLIC_ROUTES_W_M,
                city_lite_scene_contract_sha256="a" * 64,
                city_lite_scene_payload_sha256="b" * 64,
            )

    def test_artifact_inventory_excludes_self_referential_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload.bin").write_bytes(b"evidence")
            (root / "capture_receipt.json").write_text("{}", encoding="utf-8")
            (root / "capture_receipt.sha256").write_text("stale", encoding="ascii")
            hashes = _artifact_hashes(root)
            self.assertEqual(set(hashes), {"payload.bin"})
            self.assertEqual(hashes["payload.bin"]["bytes"], len(b"evidence"))
            self.assertEqual(len(hashes["payload.bin"]["sha256"]), 64)

    def test_sensor_value_conversion_copies_numpy_values_and_tensor_values(self) -> None:
        source = np.asarray([1.0, 2.0], dtype=np.float32)
        converted = _to_numpy(source)
        self.assertIsInstance(converted, np.ndarray)
        self.assertIsNot(converted, source)
        np.testing.assert_array_equal(converted, source)

    def test_onboard_semantic_metadata_accepts_tiled_and_batched_camera_api_shapes(self) -> None:
        tiled = SimpleNamespace(
            data=SimpleNamespace(info={"semantic_segmentation": {"id_to_labels": {"1": "building"}}})
        )
        batched = SimpleNamespace(
            data=SimpleNamespace(
                info=[
                    {"semantic_segmentation": {"id_to_labels": {"1": "building"}}},
                    {"rgb": None},
                ]
            )
        )
        self.assertEqual(_onboard_semantic_metadata(tiled), {"id_to_labels": {"1": "building"}})
        self.assertEqual(
            _onboard_semantic_metadata(batched),
            {"per_camera": [{"id_to_labels": {"1": "building"}}, {}]},
        )
        empty = SimpleNamespace(data=SimpleNamespace(info=[{"semantic_segmentation": {}}, {}]))
        self.assertEqual(_onboard_semantic_metadata(empty), {})

    def test_fabric_pose_is_diagnostic_not_render_acceptance_gate(self) -> None:
        self.assertEqual(ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M, 1.0e-4)
        self.assertEqual(ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD, 2.0e-3)
        closure = {
            "position_error_m": torch.tensor(
                [ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M * 1.01]
            ),
            "orientation_error_rad": torch.tensor(
                [ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD * 1.01]
            ),
        }
        diagnostic = _onboard_camera_fabric_pose_diagnostic(closure, torch)
        self.assertEqual(diagnostic["authority"], "diagnostic_only_camera_fabric_cache")
        self.assertEqual(diagnostic["acceptance_authority"], "render_facing_usd_hierarchy")
        self.assertFalse(diagnostic["within_reference_tolerance"])
        self.assertEqual(diagnostic["status"], "lag_or_unverified_non_authoritative")
        nonfinite = _onboard_camera_fabric_pose_diagnostic(
            {
                "position_error_m": torch.tensor([float("nan")]),
                "orientation_error_rad": torch.tensor([0.0]),
            },
            torch,
        )
        self.assertFalse(nonfinite["finite"])
        self.assertIsNone(nonfinite["max_position_error_m"])

    def test_usd_camera_axes_produce_authoritative_world_pose(self) -> None:
        quaternion = _world_camera_quat_from_usd_axes(
            forward=(0.0, 0.0, -1.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        quat = torch.tensor([quaternion], dtype=torch.float64)
        self.assertTrue(
            torch.allclose(
                _quat_rotate(quat, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64), torch),
                torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                _quat_rotate(quat, torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64), torch),
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            )
        )
        expected_pos = torch.zeros((AGENT_COUNT, 3), dtype=torch.float64)
        expected_quat = quat.repeat(AGENT_COUNT, 1)
        closure = {
            "per_agent": [
                {
                    "observed_pos_w_m": [0.0, 0.0, 0.0],
                    "observed_quat_wxyz": list(quaternion),
                }
                for _ in range(AGENT_COUNT)
            ]
        }
        render_pose = _camera_pose_closure_from_usd(
            closure, expected_pos, expected_quat, torch
        )
        self.assertEqual(render_pose["authority"], "render_facing_usd_hierarchy")
        self.assertTrue(torch.equal(render_pose["position_error_m"], torch.zeros(AGENT_COUNT, dtype=torch.float64)))

    def test_overview_city_content_gate_uses_semantics_when_available_and_geometry_when_not(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, ::2, 0] = 255
        depth = np.broadcast_to(
            np.linspace(12.0, 60.0, 32, dtype=np.float32)[None, :, None],
            (32, 32, 1),
        ).copy()
        structural_metadata = {
            "id_to_labels": {
                "9": {"class": "prop_structure", "prop_structure_type": "building"}
            }
        }
        semantic = np.zeros((32, 32, 1), dtype=np.int32)
        semantic[8:24, 8:24] = 9
        valid = _overview_city_content_evidence(
            rgb, depth, semantic, structural_metadata
        )
        self.assertTrue(valid["passed"], valid)
        self.assertTrue(valid["city_evidence_passed"], valid)
        self.assertTrue(valid["structural_evidence_passed"], valid)
        self.assertTrue(valid["structural_semantics_required"])
        self.assertGreater(valid["structural_pixel_fraction"], 0.001)

        fallback = _overview_city_content_evidence(rgb, depth, None, {})
        self.assertTrue(fallback["passed"], fallback)
        self.assertTrue(fallback["city_evidence_passed"], fallback)
        self.assertTrue(fallback["structural_evidence_passed"], fallback)
        self.assertFalse(fallback["structural_semantics_required"])

        all_background = _overview_city_content_evidence(
            rgb,
            np.full((32, 32, 1), 200.0, dtype=np.float32),
            semantic,
            structural_metadata,
        )
        self.assertFalse(all_background["passed"])
        self.assertIn(
            "overview depth has insufficient non-background geometry",
            all_background["failures"],
        )
        self.assertFalse(all_background["city_evidence_passed"])

        no_structural_pixels = _overview_city_content_evidence(
            rgb, depth, np.zeros((32, 32, 1), dtype=np.int32), structural_metadata
        )
        self.assertFalse(no_structural_pixels["passed"])
        self.assertTrue(no_structural_pixels["city_evidence_passed"])
        self.assertFalse(no_structural_pixels["structural_evidence_passed"])
        with self.assertRaisesRegex(RuntimeError, "no labelled structural"):
            _require_overview_city_content(no_structural_pixels)

    def test_overview_city_content_gate_rejects_camera_embedded_in_geometry(self) -> None:
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, ::2, 1] = 255
        evidence = _overview_city_content_evidence(
            rgb,
            np.full((32, 32, 1), 0.25, dtype=np.float32),
            None,
            {},
        )
        self.assertFalse(evidence["passed"])
        self.assertIn(
            "overview camera is dominated by near-surface geometry",
            evidence["failures"],
        )

    def test_fixed_overview_view_is_a_non_degenerate_public_city_frame(self) -> None:
        spec = _overview_view_spec()
        self.assertEqual(spec["eye_w_m"], [60.0, -78.0, 42.0])
        self.assertEqual(spec["target_w_m"], [0.0, -1.0, 8.0])
        self.assertGreater(spec["view_distance_m"], 100.0)
        self.assertLess(spec["position_tolerance_m"], 0.1)
        self.assertGreater(spec["forward_cosine_min"], 0.99)

    def test_initial_aircraft_yaws_face_first_public_horizontal_route_segments(self) -> None:
        headings = _initial_route_heading_yaws_rad()
        self.assertEqual(len(headings), 8)
        expected = (
            math.pi / 2.0,
            0.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi / 2.0,
        )
        for actual, target in zip(headings, expected, strict=True):
            self.assertAlmostEqual(actual, target)

    def test_overview_look_at_quaternion_points_usd_negative_z_at_target(self) -> None:
        spec = _overview_view_spec()
        w, x, y, z = _look_at_quat_wxyz(spec["eye_w_m"], spec["target_w_m"])
        # Quaternion rotation of USD camera's local -Z optical axis.
        vx, vy, vz = 0.0, 0.0, -1.0
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        observed = (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )
        expected_raw = tuple(
            spec["target_w_m"][axis] - spec["eye_w_m"][axis] for axis in range(3)
        )
        expected_norm = math.sqrt(sum(component * component for component in expected_raw))
        expected = tuple(component / expected_norm for component in expected_raw)
        self.assertAlmostEqual(sum(component * component for component in (w, x, y, z)), 1.0)
        self.assertGreater(sum(observed[axis] * expected[axis] for axis in range(3)), 0.999999)

    def test_public_follow_view_is_finite_and_tracks_the_cf2x_body_pose(self) -> None:
        view = _public_follow_view_from_body_pose(
            (3.0, -4.0, 10.0),
            (1.0, 0.0, 0.0, 0.0),
        )

        self.assertEqual(view["tracked_agent_id"], 0)
        self.assertTrue(np.isfinite(np.asarray(view["eye_w_m"], dtype=np.float64)).all())
        self.assertTrue(np.isfinite(np.asarray(view["target_w_m"], dtype=np.float64)).all())
        self.assertTrue(
            np.isfinite(np.asarray(view["orientation_wxyz"], dtype=np.float64)).all()
        )
        self.assertGreater(
            np.linalg.norm(
                np.asarray(view["target_w_m"], dtype=np.float64)
                - np.asarray(view["eye_w_m"], dtype=np.float64)
            ),
            1.0,
        )

    def test_public_route_witness_is_one_frozen_world_pose(self) -> None:
        schedule = _public_route_witness_schedule()

        self.assertEqual(schedule["mode"], "public_fixed_route_witness_schedule")
        self.assertEqual(schedule["selection"], "single_frozen_public_world_pose")
        self.assertTrue(schedule["selection_state_independent"])
        self.assertEqual(schedule["tracked_agent_id"], 2)
        self.assertGreaterEqual(schedule["minimum_tracked_agent_displacement_m"], 3.0)
        self.assertGreaterEqual(schedule["minimum_tracked_agent_pixels"], 32)
        self.assertEqual(len(schedule["shots"]), 1)
        self.assertEqual(schedule["shots"][0]["start_time_ns"], 0)
        self.assertIsNone(schedule["shots"][0]["end_time_ns"])
        self.assertEqual(schedule["shots"][0]["eye_w_m"], [0.0, -95.0, 30.0])
        self.assertEqual(schedule["shots"][0]["target_w_m"], [0.0, 0.0, 13.0])

        view = _public_route_witness_view_at_time_ns(7_000_000_000)
        for timestamp_ns in (0, 7_000_000_000, 9_500_000_000, 2**63 - 1):
            candidate = _public_route_witness_view_at_time_ns(timestamp_ns)
            self.assertEqual(candidate["shot_index"], 0)
            self.assertEqual(candidate["eye_w_m"], view["eye_w_m"])
            self.assertEqual(candidate["target_w_m"], view["target_w_m"])
            self.assertEqual(candidate["orientation_wxyz"], view["orientation_wxyz"])
        w, x, y, z = view["orientation_wxyz"]
        # Quaternion rotation of USD camera's local -Z optical axis.
        vx, vy, vz = 0.0, 0.0, -1.0
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        observed = (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )
        expected_raw = tuple(
            view["target_w_m"][axis] - view["eye_w_m"][axis]
            for axis in range(3)
        )
        expected_norm = math.sqrt(sum(component * component for component in expected_raw))
        expected = tuple(component / expected_norm for component in expected_raw)
        self.assertGreater(sum(observed[axis] * expected[axis] for axis in range(3)), 0.999999)

    def test_public_route_witness_pose_keeps_agent_two_route_in_frame(self) -> None:
        """Guard the native canary's single-camera frustum in CPU-only tests."""

        view = _public_route_witness_view_at_time_ns(0)
        eye = np.asarray(view["eye_w_m"], dtype=np.float64)
        target = np.asarray(view["target_w_m"], dtype=np.float64)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        fx = OVERVIEW_WITNESS_FOCAL_LENGTH_MM * 1920.0 / 36.0
        fy = fx
        projected: list[tuple[float, float, float]] = []
        for route_family_id in (CITY_LITE_ROUTE_FAMILY_A_ID, CITY_LITE_ROUTE_FAMILY_B_ID):
            route = resolve_public_route_family(route_family_id)[2]
            for index in range(len(route) - 1):
                start, end = route[index], route[index + 1]
                start_np = np.asarray(start, dtype=np.float64)
                end_np = np.asarray(end, dtype=np.float64)
                for fraction in np.linspace(0.0, 1.0, 21):
                    point = start_np + (end_np - start_np) * fraction
                    relative = point - eye
                    depth = float(np.dot(relative, forward))
                    self.assertGreater(depth, 0.0)
                    projected.append(
                        (
                            960.0 + fx * float(np.dot(relative, right)) / depth,
                            540.0 + fy * float(np.dot(relative, up)) / depth,
                            depth,
                        )
                    )
        us = [row[0] for row in projected]
        vs = [row[1] for row in projected]
        self.assertGreaterEqual(min(us), 30.0)
        self.assertLessEqual(max(us), 1890.0)
        self.assertGreaterEqual(min(vs), 30.0)
        self.assertLessEqual(max(vs), 1050.0)
        farthest_depth = max(row[2] for row in projected)
        projected_marker_radius = fx * IDENTITY_MARKER_RADIUS_M / farthest_depth
        self.assertGreaterEqual(
            math.pi * projected_marker_radius * projected_marker_radius,
            32.0,
            "the farthest route marker must have at least 32 projected pixels",
        )

    def test_public_route_witness_rejects_a_second_pose(self) -> None:
        two_poses = (
            (0, None, (0.0, -60.0, 30.0), (10.0, -28.0, 13.0)),
            (0, None, (0.0, -50.0, 17.0), (0.0, -42.0, 11.5)),
        )
        with patch(
            "rivermark_benchmark.isaac_capture.OVERVIEW_WITNESS_SHOTS", two_poses
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one frozen world pose"):
                _public_route_witness_schedule()

    def test_receipt_quality_observations_capture_fixed_witness_camera_displacement(self) -> None:
        timestamps = np.asarray([1, 6_999_999_999, 7_000_000_000, 9_500_000_000], dtype=np.int64)
        positions = np.asarray(
            [
                _public_route_witness_view_at_time_ns(int(timestamp))["eye_w_m"]
                for timestamp in timestamps
            ],
            dtype=np.float64,
        )

        observations = _capture_quality_observations(
            timestamps_ns=timestamps,
            camera_position_errors_m=np.zeros((len(timestamps), AGENT_COUNT), dtype=np.float64),
            camera_orientation_errors_rad=np.zeros((len(timestamps), AGENT_COUNT), dtype=np.float64),
            onboard_usd_max_position_error_m=0.0,
            onboard_usd_min_forward_alignment_cosine=1.0,
            onboard_usd_max_orientation_error_rad=0.0,
            overview_closure={"position_error_m": 0.0, "forward_alignment_cosine": 1.0},
            overview_camera_positions_w_m=positions,
            overview_first_rgb=np.asarray([[0, 1]], dtype=np.uint8),
            target_thrust_n=np.ones((1, AGENT_COUNT, 4), dtype=np.float64),
            applied_thrust_n=np.ones((1, AGENT_COUNT, 4), dtype=np.float64),
            np=np,
        )

        self.assertTrue(observations["timestamps_strictly_monotonic"])
        self.assertEqual(observations["overview_camera_max_displacement_m"], 0.0)
        self.assertTrue(observations["overview_first_frame_nonconstant"])

    def test_route_witness_visibility_requires_the_named_cf2x_marker_at_usable_scale(self) -> None:
        semantic = np.zeros((8, 8), dtype=np.int32)
        semantic[1:7, 1:7] = 17
        metadata = {
            "id_to_labels": {
                "17": {"class": "agent_identity,cf2x", "agent_id": "2"}
            }
        }
        evidence = _overview_tracked_agent_visibility_evidence(semantic, metadata)
        self.assertTrue(evidence["passed"], evidence)
        self.assertEqual(evidence["tracked_agent_pixel_count"], 36)

        semantic[1:7, 1:7] = 0
        semantic[1:6, 1:6] = 17
        evidence = _overview_tracked_agent_visibility_evidence(semantic, metadata)
        self.assertFalse(evidence["passed"])
        with self.assertRaisesRegex(RuntimeError, "does not visibly contain"):
            _require_overview_tracked_agent_visibility(evidence)

    def test_onboard_visual_intrusion_gate_accepts_clear_frames_and_rejects_near_geometry(self) -> None:
        clear_depth = np.full((AGENT_COUNT, 4, 4, 1), 2.0, dtype=np.float32)
        clear_lidar = np.full((AGENT_COUNT, 32), 2.0, dtype=np.float32)
        clean = _onboard_visual_intrusion_evidence(
            clear_depth,
            clear_lidar,
            lidar_max_distance_m=35.0,
        )
        self.assertTrue(clean["passed"], clean)

        embedded_depth = clear_depth.copy()
        embedded_lidar = clear_lidar.copy()
        embedded_depth[1] = 0.20
        embedded_lidar[1, :10] = 0.20
        invalid = _onboard_visual_intrusion_evidence(
            embedded_depth,
            embedded_lidar,
            lidar_max_distance_m=35.0,
        )
        self.assertFalse(invalid["passed"])
        self.assertEqual(invalid["per_agent"][1]["agent_id"], 1)
        self.assertTrue(invalid["per_agent"][1]["failures"])
        with self.assertRaisesRegex(RuntimeError, "visual intrusion"):
            _require_onboard_visual_integrity(invalid)

    def test_onboard_scene_content_gate_rejects_background_dominated_frames(self) -> None:
        clear_depth = np.full((AGENT_COUNT, 4, 4, 1), 20.0, dtype=np.float32)
        clear_labels = np.ones((AGENT_COUNT, 4, 4, 1), dtype=np.int32)
        metadata = {
            "per_camera": [
                {
                    "id_to_labels": {
                        "0": {"class": "BACKGROUND"},
                        "1": {"class": "building"},
                    }
                }
                for _ in range(AGENT_COUNT)
            ]
        }
        clean = _onboard_scene_content_evidence(clear_depth, clear_labels, metadata)
        self.assertTrue(clean["passed"], clean)
        self.assertEqual(clean["per_agent"][0]["background_fraction"], 0.0)

        background_depth = np.full_like(clear_depth, 100.0)
        background_labels = np.zeros_like(clear_labels)
        invalid = _onboard_scene_content_evidence(background_depth, background_labels, metadata)
        self.assertFalse(invalid["passed"])
        self.assertTrue(invalid["per_agent"][2]["failures"])
        with self.assertRaisesRegex(RuntimeError, "scene-content"):
            _require_onboard_scene_content(invalid)

    def test_onboard_scene_content_gate_rejects_malformed_or_unlabelled_input(self) -> None:
        depth = np.ones((AGENT_COUNT, 4, 4, 1), dtype=np.float32)
        labels = np.ones((AGENT_COUNT, 4, 4), dtype=np.int32)
        malformed = _onboard_scene_content_evidence(depth, labels, {})
        self.assertFalse(malformed["passed"])
        self.assertTrue(malformed["failures"])

    def test_public_follow_camera_authors_the_usd_render_transform(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[3.0, -4.0, 10.0]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        overview = Mock(name="overview")
        overview.cfg.prim_path = "/World/OverviewCamera"
        closure = {"position_error_m": 0.0, "forward_alignment_cosine": 1.0}
        with (
            patch("rivermark_benchmark.isaac_capture._author_camera_usd_look_at") as author,
            patch(
                "rivermark_benchmark.isaac_capture._camera_usd_pose_closure_for_view",
                return_value=closure,
            ),
        ):
            result = _set_public_follow_overview_view("stage", overview, robot, torch)

        self.assertEqual(result["pose_closure"], closure)
        author.assert_called_once_with(
            "stage",
            "/World/OverviewCamera",
            eye=result["eye_w_m"],
            target=result["target_w_m"],
        )
        overview.set_world_poses.assert_not_called()

    def test_public_route_witness_authors_the_declared_usd_render_transform(self) -> None:
        overview = Mock(name="overview")
        overview.cfg.prim_path = "/World/OverviewCamera"
        closure = {"position_error_m": 0.0, "forward_alignment_cosine": 1.0}
        with (
            patch("rivermark_benchmark.isaac_capture._author_camera_usd_look_at") as author,
            patch(
                "rivermark_benchmark.isaac_capture._camera_usd_pose_closure_for_view",
                return_value=closure,
            ),
        ):
            result = _set_public_route_witness_overview_view("stage", overview)

        self.assertEqual(result["pose_closure"], closure)
        author.assert_called_once_with(
            "stage",
            "/World/OverviewCamera",
            eye=result["eye_w_m"],
            target=result["target_w_m"],
        )
        overview.set_world_poses.assert_not_called()

    def test_onboard_render_preparation_uses_only_native_parent_relative_mount(self) -> None:
        expected_positions = Mock(name="expected_positions")
        expected_world_quats = Mock(name="expected_world_quats")
        sim = Mock(name="sim")
        torch = Mock(name="torch")
        usd_closure = {"usd": "closed"}
        events: list[str] = []

        def flush_fabric() -> None:
            events.append("flush_dynamic_fabric")

        def usd_pose(*_args: object) -> dict[str, str]:
            events.append("read_usd")
            return usd_closure

        def require_usd(*_args: object) -> None:
            events.append("require_usd")

        sim.forward.side_effect = flush_fabric
        with (
            patch(
                "rivermark_benchmark.isaac_capture._expected_onboard_camera_world_poses",
                return_value=(expected_positions, expected_world_quats),
            ),
            patch(
                "rivermark_benchmark.isaac_capture._synchronize_onboard_camera_fabric_local_transforms",
                side_effect=AssertionError("onboard render preparation must not write a Fabric local override"),
                create=True,
            ) as local_override,
            patch(
                "rivermark_benchmark.isaac_capture._author_onboard_camera_usd_transforms",
                side_effect=AssertionError("onboard render preparation must not author a manual USD matrix"),
                create=True,
            ) as usd_author,
            patch("rivermark_benchmark.isaac_capture._onboard_camera_usd_pose_closure", side_effect=usd_pose),
            patch("rivermark_benchmark.isaac_capture._require_onboard_camera_usd_pose", side_effect=require_usd),
        ):
            result = _prepare_onboard_camera_local_mount(
                sim, "stage", "robot", torch
            )

        self.assertEqual(
            result,
            (expected_positions, expected_world_quats, usd_closure),
        )
        local_override.assert_not_called()
        usd_author.assert_not_called()
        self.assertEqual(
            events,
            [
                "flush_dynamic_fabric",
                "read_usd",
                "require_usd",
            ],
        )

    def test_onboard_render_read_fence_requires_one_counter_increment(self) -> None:
        camera = SimpleNamespace(_frame=torch.arange(AGENT_COUNT, dtype=torch.int64))
        before = _onboard_camera_frame_counter(camera, torch)
        camera._frame.add_(1)
        fence = _require_onboard_camera_render_read_fence(camera, before, torch)
        self.assertTrue(torch.equal(fence["pre_frame_index"], torch.arange(AGENT_COUNT)))
        self.assertTrue(torch.equal(fence["post_frame_index"], torch.arange(AGENT_COUNT) + 1))

        camera._frame.add_(2)
        with self.assertRaisesRegex(RuntimeError, "exactly one buffer update"):
            _require_onboard_camera_render_read_fence(camera, fence["post_frame_index"], torch)

    def test_onboard_camera_mount_diagnostic_distinguishes_parent_and_previous_step(self) -> None:
        root_pos = torch.zeros((AGENT_COUNT, 3), dtype=torch.float32)
        root_quat = torch.zeros((AGENT_COUNT, 4), dtype=torch.float32)
        root_quat[:, 0] = 1.0
        body_pos = root_pos[:, None, :].repeat(1, 5, 1)
        body_pos[:, 0, 2] = 0.001
        body_quat = root_quat[:, None, :].repeat(1, 5, 1)
        observed_pos = root_pos + torch.tensor((0.12, 0.0, 0.041), dtype=torch.float32)
        observed_quat = root_quat.clone()
        robot = SimpleNamespace(
            body_names=["body", "m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            data=SimpleNamespace(
                body_link_pos_w=body_pos,
                body_link_quat_w=body_quat,
            ),
        )
        camera = SimpleNamespace(
            data=SimpleNamespace(pos_w=observed_pos, quat_w_world=observed_quat)
        )
        root_expected = root_pos + torch.tensor((0.12, 0.0, 0.04), dtype=torch.float32)
        diagnostic = _onboard_camera_mount_diagnostics(
            robot,
            camera,
            root_expected_pos_w=root_expected,
            root_expected_quat_wxyz=root_quat,
            previous_root_expected_pos_w=root_expected - torch.tensor((0.0, 0.0, 0.001)),
            previous_root_expected_phase="post_reset_state_update",
            torch=torch,
        )

        self.assertEqual(diagnostic["literal_parent_link"], {"name": "body", "index": 0})
        self.assertEqual(len(diagnostic["per_agent"]), AGENT_COUNT)
        self.assertAlmostEqual(diagnostic["maximum_root_residual_norm_m"], 0.001, places=7)
        self.assertAlmostEqual(diagnostic["maximum_body_link_residual_norm_m"], 0.0, places=7)
        self.assertEqual(diagnostic["per_agent"][0]["previous_root_expected_phase"], "post_reset_state_update")
        self.assertAlmostEqual(diagnostic["per_agent"][0]["previous_root_residual_norm_m"], 0.002, places=7)

        observed_pos[0, 0] = float("nan")
        nonfinite = _onboard_camera_mount_diagnostics(
            robot,
            camera,
            root_expected_pos_w=root_expected,
            root_expected_quat_wxyz=root_quat,
            previous_root_expected_pos_w=None,
            previous_root_expected_phase=None,
            torch=torch,
        )
        self.assertFalse(nonfinite["per_agent"][0]["fabric_pose_finite"])
        self.assertIsNone(nonfinite["per_agent"][0]["root_residual_norm_m"])
        self.assertIsNone(nonfinite["maximum_root_residual_norm_m"])
        json.dumps(nonfinite, allow_nan=False)

    def test_onboard_camera_mount_diagnostic_rejects_unlabeled_previous_pose(self) -> None:
        root_pos = torch.zeros((AGENT_COUNT, 3), dtype=torch.float32)
        root_quat = torch.zeros((AGENT_COUNT, 4), dtype=torch.float32)
        root_quat[:, 0] = 1.0
        robot = SimpleNamespace(
            body_names=["body", "m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            data=SimpleNamespace(
                body_link_pos_w=root_pos[:, None, :].repeat(1, 5, 1),
                body_link_quat_w=root_quat[:, None, :].repeat(1, 5, 1),
            ),
        )
        camera = SimpleNamespace(data=SimpleNamespace(pos_w=root_pos, quat_w_world=root_quat))
        with self.assertRaisesRegex(RuntimeError, "requires an auditable phase"):
            _onboard_camera_mount_diagnostics(
                robot,
                camera,
                root_expected_pos_w=root_pos,
                root_expected_quat_wxyz=root_quat,
                previous_root_expected_pos_w=root_pos,
                previous_root_expected_phase=None,
                torch=torch,
            )


if __name__ == "__main__":
    unittest.main()
