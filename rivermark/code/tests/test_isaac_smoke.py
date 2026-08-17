from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.isaac_smoke import (  # noqa: E402
    ISAAC_SMOKE_SCHEMA,
    RUNTIME_AUDIT_SCHEMA,
    SMOKE_SENSOR_NAMES,
    STEP_ORDER,
    IsaacSmokeError,
    _check_commit,
    _close_smoke_app,
    _close_smoke_resources,
    _write_receipt,
    main,
    run_target_free_smoke,
    validate_smoke_receipt,
)


def _receipt() -> dict:
    commit_total = 26 * 1024**3
    commit_limit = 68 * 1024**3
    commit_snapshot = {
        "commit_total_bytes": commit_total,
        "commit_limit_bytes": commit_limit,
        "commit_peak_bytes": 27 * 1024**3,
        "commit_percent": 100.0 * commit_total / commit_limit,
    }
    telemetry_sample = {
        "wall_time_ns": 1,
        "phase": "preflight",
        "process": None,
        "system_commit": commit_snapshot,
        "gpu": None,
    }
    return {
        "schema": ISAAC_SMOKE_SCHEMA,
        "status": "passed",
        "claim_boundary": {
            "formal_episode": False,
            "benchmark_score": False,
            "private_targets_present": False,
            "evaluator_truth_used": False,
            "sensor_payload_retained": False,
        },
        "step_order": list(STEP_ORDER),
        "agent_count": 8,
        "resource_probe_profile": "full",
        "search_target_prim_count": 0,
        "sensors": {name: True for name in ("rgb", "depth", "semantic", "lidar", "imu", "contact")},
        "runtime_lock_sha256": "a" * 64,
        "runtime_profile_id": "test-profile",
        "runtime_audit": {
            "schema": RUNTIME_AUDIT_SCHEMA,
            "status": "passed",
            "profile_id": "test-profile",
            "runtime_lock_sha256": "a" * 64,
            "configuration_observation": "public_runtime_environment_and_locked_assets",
            "observed": {
                "configuration_observation": "public_runtime_environment_and_locked_assets",
            },
            "issues": [],
        },
        "launcher": {
            "headless": True,
            "enable_cameras": True,
            "device": "cuda:0",
            "rendering_mode": "balanced",
            "livestream": 0,
            "xr": False,
            "distributed": False,
            "kit_args": "",
            "experience": {"path": "apps/smoke.kit", "sha256": "b" * 64},
        },
        "simulation": {
            "device": "cuda:0",
            "dt_s": 0.005,
            "gravity_w_mps2": [0.0, 0.0, -9.81],
            "agent_count": 8,
            "render_interval": 1,
            "use_fabric": True,
            "config_digests": {
                name: {"settings": {}, "sha256": value}
                for name, value in (("render", "c" * 64), ("fabric", "d" * 64), ("physx", "e" * 64))
            },
        },
        "runtime_observed": {
            "device": "cuda:0",
            "physics_dt_s": 0.005,
            "rendering_dt_s": 0.005,
            "gravity_w_mps2": [0.0, 0.0, -9.81],
            "rendering_mode": "balanced",
            "render_interval": 1,
            "use_fabric": True,
            "rtx_sensors_active": True,
            "config_digests": {"render": "c" * 64, "fabric": "d" * 64, "physx": "e" * 64},
            "configuration_observation": "public_simulation_context_and_locked_cfg",
        },
        "sensor_last_frame_sha256": {
            name: "f" * 64
            for name in ("rgb", "depth", "semantic", "lidar", "imu", "contact")
        },
        "source": {"source_worktree_dirty": False},
        "foreign_native_process_guard": {
            "schema": "org.rivermark.foreign-native-process-guard.v2",
            "maximum_private_commit_gib": 8.0,
            "status": "active",
            "sample_count": 6,
            "last_phase": "before_render_step_2",
            "last_census_status": "available",
            "maximum_candidate_count": 0,
            "maximum_candidate_count_phase": None,
            "maximum_candidate_private_commit_bytes": 0,
            "maximum_candidate_private_commit_phase": None,
        },
        "system_commit": {
            "preflight": commit_snapshot,
            "maximum_observed_percent": commit_snapshot["commit_percent"],
            "maximum_phase": "preflight",
            "maximum_snapshot": commit_snapshot,
            "last_phase": "preflight",
            "last_snapshot": commit_snapshot,
        },
        "resource_telemetry": {
            "schema": "org.rivermark.resource-telemetry.v1",
            "sample_count": 1,
            "samples": [telemetry_sample],
            "maxima": {},
            "sampling": "explicit_in_process_phase_boundaries",
        },
        "physics_steps": 2,
        "step_trace": [
            {"step": 1, "events": list(STEP_ORDER)},
            {"step": 2, "events": list(STEP_ORDER)},
        ],
        "onboard_camera_render_read_fences": [
            {
                "pre_frame_index": [0] * 8,
                "post_frame_index": [1] * 8,
            },
            {
                "pre_frame_index": [1] * 8,
                "post_frame_index": [2] * 8,
            },
        ],
    }


class IsaacSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._zero_foreign_census = {
            "schema": "org.rivermark.foreign-native-process-census.v1",
            "enumerated_native_process_count": 0,
            "minimum_private_commit_bytes": 8 * 1024**3,
            "candidate_count": 0,
            "candidate_private_commit_bytes": 0,
            "maximum_candidate_private_commit_bytes": 0,
        }
        census_patch = patch(
            "rivermark_benchmark.isaac_capture.foreign_native_process_census",
            return_value=self._zero_foreign_census,
        )
        census_patch.start()
        self.addCleanup(census_patch.stop)

    def test_full_smoke_defaults_use_production_route_witness_resolution(self) -> None:
        """Keep the smoke's fixed-world witness physically comparable to capture."""

        with patch(
            "rivermark_benchmark.isaac_smoke.run_target_free_smoke", return_value={}
        ) as run_smoke:
            self.assertEqual(
                main(
                    [
                        "--output-dir",
                        "smoke",
                        "--runtime-lock",
                        "runtime-lock.json",
                        "--isaaclab-source",
                        "isaaclab",
                        "--scene-contract",
                        "scene-contract.json",
                        "--drone-usd",
                        "cf2x.usd",
                    ]
                ),
                0,
            )

        args = run_smoke.call_args.args[0]
        self.assertEqual((args.overview_width, args.overview_height), (1920, 1080))
        self.assertEqual((args.preflight_commit_percent, args.abort_commit_percent), (65.0, 82.0))
        self.assertEqual(args.maximum_foreign_native_private_commit_gib, 8.0)

    def test_passing_receipt_binds_target_free_ordered_smoke(self) -> None:
        self.assertEqual(validate_smoke_receipt(_receipt()), ())

    def test_targets_dirty_source_missing_sensor_and_order_fail(self) -> None:
        receipt = _receipt()
        receipt["search_target_prim_count"] = 1
        receipt["source"]["source_worktree_dirty"] = True
        receipt["sensors"]["lidar"] = False
        receipt["step_trace"][1]["events"] = list(reversed(STEP_ORDER))
        errors = validate_smoke_receipt(receipt)
        self.assertGreaterEqual(len(errors), 4)

    def test_passing_receipt_requires_active_runtime_foreign_process_guard(self) -> None:
        for mutation in (
            lambda guard: guard.update(status="rejected"),
            lambda guard: guard.update(sample_count=3),
            lambda guard: guard.update(maximum_candidate_count=1),
            lambda guard: guard.update(last_census_status="unavailable"),
        ):
            with self.subTest(mutation=mutation):
                receipt = _receipt()
                mutation(receipt["foreign_native_process_guard"])
                self.assertIn(
                    "runtime foreign-native process guard is not bound",
                    validate_smoke_receipt(receipt),
                )

    def test_smoke_rejects_disabled_foreign_process_guard_before_isaac(self) -> None:
        args = SimpleNamespace(
            output_dir=Path("unused"),
            resource_probe_profile="full",
            steps=1,
            preflight_commit_percent=65.0,
            abort_commit_percent=82.0,
            maximum_foreign_native_private_commit_gib=0.0,
            minimum_free_gib=0.0,
        )
        with self.assertRaisesRegex(IsaacSmokeError, "finite and positive"):
            from rivermark_benchmark.isaac_smoke import _run_target_free_smoke_checked

            _run_target_free_smoke_checked(args, Path("unused"), {}, object(), {})

    def test_reset_only_camera_profiles_are_explicit_and_cannot_claim_frames(self) -> None:
        profiles = {
            "no_cameras": (
                {"rgb": False, "depth": False, "semantic": False, "lidar": True, "imu": True, "contact": True},
                False,
                {"onboard": False, "overview": False},
            ),
            "onboard_only": (
                {name: True for name in SMOKE_SENSOR_NAMES},
                True,
                {"onboard": True, "overview": False},
            ),
            "overview_only": (
                {name: True for name in SMOKE_SENSOR_NAMES},
                True,
                {"onboard": False, "overview": True},
            ),
            "onboard_tiled_only": (
                {name: True for name in SMOKE_SENSOR_NAMES},
                True,
                {"onboard": True, "overview": False},
            ),
            "overview_tiled_only": (
                {name: True for name in SMOKE_SENSOR_NAMES},
                True,
                {"onboard": False, "overview": True},
            ),
        }
        for profile, (sensors, rtx_active, render_products) in profiles.items():
            with self.subTest(profile=profile):
                receipt = _receipt()
                receipt["resource_probe_profile"] = profile
                receipt["physics_steps"] = 0
                receipt["step_trace"] = []
                receipt["sensors"] = sensors
                receipt["sensor_last_frame_sha256"] = {}
                receipt["runtime_observed"]["rtx_sensors_active"] = rtx_active
                implementations = {
                    "onboard": "tiled_camera" if profile == "onboard_tiled_only" else (
                        "camera" if render_products["onboard"] else "none"
                    ),
                    "overview": "tiled_camera" if profile == "overview_tiled_only" else (
                        "camera" if render_products["overview"] else "none"
                    ),
                }
                receipt["resource_probe"] = {
                    "kind": "reset_only",
                    "camera_render_products": render_products,
                    "camera_sensor_implementations": implementations,
                    "not_capture_evidence": True,
                }
                receipt["resource_probe_request"] = copy.deepcopy(receipt["resource_probe"])
                self.assertEqual(validate_smoke_receipt(receipt), ())

        receipt["sensor_last_frame_sha256"] = {"rgb": "f" * 64}
        self.assertIn(
            "reset-only resource probes must not retain sensor-frame digests",
            validate_smoke_receipt(receipt),
        )

        receipt["sensor_last_frame_sha256"] = {}
        receipt["resource_probe"]["camera_render_products"]["overview"] = False
        self.assertIn(
            "resource probe render-product contract is not bound",
            validate_smoke_receipt(receipt),
        )

        receipt["resource_probe"]["camera_render_products"]["overview"] = True
        receipt["resource_probe_request"]["camera_sensor_implementations"]["overview"] = "camera"
        self.assertIn(
            "tiled resource probe implementation request is not bound",
            validate_smoke_receipt(receipt),
        )

        receipt = _receipt()
        receipt["resource_probe_profile"] = "onboard_tiled_only"
        receipt["physics_steps"] = 0
        receipt["step_trace"] = []
        receipt["sensors"] = {name: True for name in SMOKE_SENSOR_NAMES}
        receipt["sensor_last_frame_sha256"] = {}
        receipt["runtime_observed"]["rtx_sensors_active"] = True
        receipt["resource_probe"] = {
            "kind": "reset_only",
            "camera_render_products": {"onboard": True, "overview": False},
            "camera_sensor_implementations": {"onboard": "tiled_camera", "overview": "none"},
            "not_capture_evidence": True,
        }
        receipt["resource_probe_request"] = copy.deepcopy(receipt["resource_probe"])
        self.assertEqual(validate_smoke_receipt(receipt), ())
        receipt["resource_probe_request"]["camera_sensor_implementations"]["onboard"] = "camera"
        self.assertIn(
            "tiled resource probe implementation request is not bound",
            validate_smoke_receipt(receipt),
        )

        receipt = _receipt()
        receipt["resource_probe_profile"] = "no_cameras"
        receipt["physics_steps"] = 0
        receipt["step_trace"] = []
        receipt["sensors"] = {
            "rgb": False,
            "depth": False,
            "semantic": False,
            "lidar": True,
            "imu": True,
            "contact": True,
        }
        receipt["sensor_last_frame_sha256"] = {}
        receipt["runtime_observed"]["rtx_sensors_active"] = False
        receipt["resource_probe"] = {
            "kind": "reset_only",
            "camera_render_products_constructed": False,
            "not_capture_evidence": True,
        }
        self.assertEqual(validate_smoke_receipt(receipt), ())

        receipt = _receipt()
        receipt["resource_probe_profile"] = "unknown"
        self.assertIn("resource probe profile is unsupported", validate_smoke_receipt(receipt))

    def test_runtime_rendering_and_simulation_bindings_are_required(self) -> None:
        receipt = _receipt()
        receipt["launcher"]["rendering_mode"] = "unbound"
        receipt["simulation"]["dt_s"] = 0.01
        errors = validate_smoke_receipt(receipt)
        self.assertIn("launcher configuration is not bound", errors)
        self.assertIn("live renderer or simulation configuration is not bound", errors)

    def test_live_runtime_observation_must_match_the_lock(self) -> None:
        receipt = _receipt()
        receipt["runtime_observed"]["device"] = "cuda:1"
        receipt["runtime_observed"]["rtx_sensors_active"] = False
        errors = validate_smoke_receipt(receipt)
        self.assertIn("live simulation device is not bound", errors)

    def test_runtime_audit_must_bind_schema_profile_and_lock_hash(self) -> None:
        receipt = _receipt()
        receipt["runtime_audit"]["schema"] = "wrong-schema"
        receipt["runtime_audit"]["profile_id"] = "other-profile"
        receipt["runtime_audit"]["runtime_lock_sha256"] = "b" * 64
        errors = validate_smoke_receipt(receipt)
        self.assertIn("runtime lock audit schema is not bound", errors)
        self.assertIn("runtime lock audit hash is not bound", errors)
        self.assertIn("runtime lock audit profile is not bound", errors)

    def test_runtime_audit_and_sensor_digests_are_not_self_report_only(self) -> None:
        receipt = _receipt()
        receipt["runtime_audit"]["issues"] = [{"path": "$.assets", "message": "drift"}]
        receipt["sensor_last_frame_sha256"].pop("lidar")
        errors = validate_smoke_receipt(receipt)
        self.assertIn("runtime lock audit contains unresolved issues", errors)
        self.assertIn("last-frame sensor digests are incomplete or malformed", errors)

    def test_system_commit_peak_must_bind_a_coherent_exact_snapshot(self) -> None:
        receipt = _receipt()
        receipt["system_commit"]["maximum_snapshot"] = {
            **receipt["system_commit"]["maximum_snapshot"],
            "commit_percent": 42.0,
        }
        self.assertIn(
            "system commit guard evidence is incomplete or incoherent",
            validate_smoke_receipt(receipt),
        )

        receipt = _receipt()
        receipt["system_commit"].pop("last_snapshot")
        self.assertIn(
            "system commit guard evidence is incomplete or incoherent",
            validate_smoke_receipt(receipt),
        )

        receipt = _receipt()
        receipt["resource_telemetry"]["samples"][0]["phase"] = "after_reset"
        self.assertIn(
            "system commit guard evidence is incomplete or incoherent",
            validate_smoke_receipt(receipt),
        )

    def test_runtime_observation_provenance_is_required(self) -> None:
        receipt = _receipt()
        receipt["runtime_audit"]["observed"].pop("configuration_observation")
        receipt["runtime_observed"].pop("configuration_observation")
        errors = validate_smoke_receipt(receipt)
        self.assertIn("runtime lock audit observed environment is not provenance-bound", errors)
        self.assertIn("live runtime observation provenance is missing", errors)

    def test_non_mapping_trace_row_fails_closed(self) -> None:
        receipt = _receipt()
        receipt["step_trace"][1] = None
        self.assertIn(
            "step trace contains an invalid or out-of-order event",
            validate_smoke_receipt(receipt),
        )

    def test_camera_render_read_fence_fails_closed(self) -> None:
        receipt = _receipt()
        receipt["onboard_camera_render_read_fences"][1]["post_frame_index"][3] = 5
        self.assertIn(
            "onboard camera render/read frame fence is invalid",
            validate_smoke_receipt(receipt),
        )

    def test_failed_receipt_still_requires_honest_claim_boundary(self) -> None:
        receipt = _receipt()
        receipt["status"] = "failed"
        receipt.pop("step_trace")
        self.assertEqual(validate_smoke_receipt(receipt), ())
        receipt["claim_boundary"]["formal_episode"] = True
        self.assertIn("claim boundary is incomplete", validate_smoke_receipt(receipt))

    def test_float32_gravity_readback_passes_receipt_validation(self) -> None:
        receipt = _receipt()
        receipt["runtime_observed"]["gravity_w_mps2"] = [0.0, 0.0, -9.8100004196167]
        self.assertEqual(validate_smoke_receipt(receipt), ())

        receipt["runtime_observed"]["gravity_w_mps2"] = [0.0, 0.0, -9.81001]
        self.assertIn(
            "live renderer or simulation configuration is not bound",
            validate_smoke_receipt(receipt),
        )

        receipt = _receipt()
        receipt["simulation"]["gravity_w_mps2"] = [0.0, 0.0, float("nan")]
        self.assertIn("simulation configuration is not bound", validate_smoke_receipt(receipt))

    def test_standalone_smoke_uses_immediate_shutdown_after_receipt(self) -> None:
        class App:
            def __init__(self) -> None:
                self.calls: list[tuple[bool, bool]] = []

            def close(self, *, wait_for_replicator: bool, skip_cleanup: bool) -> None:
                self.calls.append((wait_for_replicator, skip_cleanup))

        app = App()
        _close_smoke_app(app)
        self.assertEqual(app.calls, [(False, True)])

    def test_close_failure_still_releases_the_app_launcher_lease(self) -> None:
        class App:
            def close(self, **_: object) -> None:
                raise RuntimeError("Kit close failed")

        class Lease:
            released = False

            def release(self) -> None:
                self.released = True

        lease = Lease()
        with self.assertRaisesRegex(RuntimeError, "Kit close failed"):
            _close_smoke_resources(App(), lease)
        self.assertTrue(lease.released)

    def test_running_smoke_receipt_is_terminalized_after_prelaunch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "smoke"
            args = type("Args", (), {"output_dir": output_dir})()

            def fail_after_reservation(
                _args: object,
                reserved_output: Path,
                early_receipt: dict,
                _resource_telemetry: object,
                _system_commit: object,
            ) -> dict:
                _write_receipt(
                    reserved_output,
                    {**early_receipt, "status": "running", "prelaunch_marker": True},
                )
                raise IsaacSmokeError("pre-launch setup failed")

            with patch(
                "rivermark_benchmark.isaac_smoke._run_target_free_smoke_checked",
                side_effect=fail_after_reservation,
            ):
                with self.assertRaisesRegex(IsaacSmokeError, "pre-launch setup failed"):
                    run_target_free_smoke(args)

            receipt_path = output_dir / "isaac_smoke_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["prelaunch_marker"])
            self.assertEqual(receipt["failure"]["type"], "IsaacSmokeError")
            self.assertEqual(
                (output_dir / "isaac_smoke_receipt.sha256").read_text(encoding="ascii"),
                f"{hashlib.sha256(receipt_path.read_bytes()).hexdigest()}  isaac_smoke_receipt.json\n",
            )

    def test_system_exit_terminalizes_a_reserved_smoke_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "smoke"
            args = type("Args", (), {"output_dir": output_dir})()

            def exit_after_reservation(
                _args: object,
                reserved_output: Path,
                early_receipt: dict,
                _resource_telemetry: object,
                _system_commit: object,
            ) -> dict:
                _write_receipt(reserved_output, {**early_receipt, "status": "running"})
                raise SystemExit(23)

            with patch(
                "rivermark_benchmark.isaac_smoke._run_target_free_smoke_checked",
                side_effect=exit_after_reservation,
            ):
                with self.assertRaises(SystemExit):
                    run_target_free_smoke(args)

            receipt = json.loads(
                (output_dir / "isaac_smoke_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["failure"]["type"], "SystemExit")

    def test_foreign_process_rejection_terminalizes_smoke_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "smoke"
            args = type(
                "Args",
                (),
                {
                    "output_dir": output_dir,
                    "maximum_foreign_native_private_commit_gib": 8.0,
                },
            )()
            foreign_census = {
                **self._zero_foreign_census,
                "enumerated_native_process_count": 1,
                "candidate_count": 1,
                "candidate_private_commit_bytes": 20 * 1024**3,
                "maximum_candidate_private_commit_bytes": 20 * 1024**3,
            }

            with patch(
                "rivermark_benchmark.isaac_capture.foreign_native_process_census",
                return_value=foreign_census,
            ), self.assertRaisesRegex(RuntimeError, "refusing native Isaac run"):
                run_target_free_smoke(args)

            receipt_path = output_dir / "isaac_smoke_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(
                receipt["foreign_native_process_guard"]["status"], "rejected"
            )
            self.assertEqual(
                receipt["foreign_native_process_guard"]["last_phase"], "smoke_start"
            )
            self.assertEqual(receipt["failure"]["type"], "RuntimeError")
            self.assertEqual(
                (output_dir / "isaac_smoke_receipt.sha256").read_text(encoding="ascii"),
                f"{hashlib.sha256(receipt_path.read_bytes()).hexdigest()}  isaac_smoke_receipt.json\n",
            )

    def test_prelaunch_commit_rejection_binds_telemetry_and_blocks_source_activation(self) -> None:
        class Telemetry:
            def __init__(self, snapshots: dict[str, dict[str, float | int]]) -> None:
                self._snapshots = snapshots
                self.samples: list[dict[str, object]] = []

            def sample(self, phase: str, **_: object) -> dict[str, object]:
                row: dict[str, object] = {
                    "wall_time_ns": len(self.samples) + 1,
                    "phase": phase,
                    "process": None,
                    "system_commit": self._snapshots[phase],
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

        source = SimpleNamespace(
            source_worktree_dirty=False,
            source_revision="test-revision",
            as_dict=lambda: {"source_worktree_dirty": False},
        )
        lock = {
            "profile_id": "test-profile",
            "launcher": {},
            "simulation": {},
        }
        args = SimpleNamespace(
            preflight_commit_percent=70.0,
            abort_commit_percent=85.0,
            minimum_free_gib=0.0,
            steps=1,
            runtime_lock=Path("runtime-lock.json"),
            isaaclab_source=Path("isaaclab"),
            scene_contract=Path("scene.json"),
            drone_usd=Path("cf2x.usd"),
        )
        low = {
            "commit_total_bytes": 26 * 1024**3,
            "commit_limit_bytes": 68 * 1024**3,
            "commit_peak_bytes": 27 * 1024**3,
            "commit_percent": 38.0,
        }
        high = {
            "commit_total_bytes": 49 * 1024**3,
            "commit_limit_bytes": 68 * 1024**3,
            "commit_peak_bytes": 50 * 1024**3,
            "commit_percent": 72.0,
        }
        for expected_phase, snapshots, expected_phases in (
            (
                "preflight",
                {"smoke_start": low, "preflight": high},
                ["smoke_start", "preflight"],
            ),
            (
                "before_app_launcher",
                {
                    "smoke_start": low,
                    "preflight": low,
                    "before_app_launcher": high,
                },
                ["smoke_start", "preflight", "before_app_launcher"],
            ),
        ):
            with self.subTest(phase=expected_phase), tempfile.TemporaryDirectory() as temporary:
                args.output_dir = Path(temporary) / "smoke"
                rejected_snapshot = snapshots[expected_phase]
                with patch("rivermark_benchmark.isaac_smoke.ResourceTelemetry", return_value=Telemetry(snapshots)), patch(
                    "rivermark_benchmark.isaac_smoke.detect_source_provenance",
                    return_value=source,
                ), patch(
                    "rivermark_benchmark.isaac_smoke.load_runtime_lock",
                    return_value=lock,
                ), patch(
                    "rivermark_benchmark.isaac_smoke.runtime_lock_sha256",
                    return_value="a" * 64,
                ), patch(
                    "rivermark_benchmark.isaac_smoke.audit_runtime_lock",
                    return_value={"status": "passed"},
                ), patch(
                    "rivermark_benchmark.isaac_smoke._windows_system_commit_snapshot",
                    side_effect=AssertionError("provided telemetry snapshot must be used"),
                ), patch(
                    "rivermark_benchmark.isaac_smoke._activate_local_isaaclab_source"
                ) as activate_source:
                    with self.assertRaisesRegex(IsaacSmokeError, f"at {expected_phase}"):
                        run_target_free_smoke(args)

                activate_source.assert_not_called()
                receipt = json.loads(
                    (args.output_dir / "isaac_smoke_receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(
                    receipt["system_commit"]["maximum_snapshot"], rejected_snapshot
                )
                self.assertEqual(receipt["system_commit"]["maximum_phase"], expected_phase)
                telemetry = receipt["resource_telemetry"]
                self.assertEqual(
                    [sample["phase"] for sample in telemetry["samples"]],
                    expected_phases,
                )
                self.assertEqual(
                    telemetry["samples"][-1]["system_commit"], rejected_snapshot
                )

    def test_after_reset_commit_rejection_preserves_peak_for_failed_receipt(self) -> None:
        preflight = {
            "commit_total_bytes": 26 * 1024**3,
            "commit_limit_bytes": 68 * 1024**3,
            "commit_peak_bytes": 27 * 1024**3,
            "commit_percent": 38.0,
        }
        after_reset = {
            "commit_total_bytes": 58 * 1024**3,
            "commit_limit_bytes": 68 * 1024**3,
            "commit_peak_bytes": 59 * 1024**3,
            "commit_percent": 85.3,
        }
        system_commit: dict[str, object] = {
            "preflight": preflight,
            "maximum_observed_percent": preflight["commit_percent"],
            "maximum_phase": "preflight",
            "maximum_snapshot": preflight,
        }
        with patch(
            "rivermark_benchmark.isaac_smoke._windows_system_commit_snapshot",
            return_value=after_reset,
        ):
            with self.assertRaisesRegex(IsaacSmokeError, "at after_reset"):
                _check_commit(
                    threshold_percent=85.0,
                    phase="after_reset",
                    system_commit=system_commit,
                )

        # The exception path expands the same receipt mapping, so a terminal
        # failure cannot regress to its lower preflight-only observation.
        failed_receipt = {"status": "failed", "system_commit": system_commit}
        self.assertEqual(
            failed_receipt["system_commit"]["maximum_observed_percent"],
            after_reset["commit_percent"],
        )
        self.assertEqual(failed_receipt["system_commit"]["maximum_phase"], "after_reset")
        self.assertEqual(failed_receipt["system_commit"]["maximum_snapshot"], after_reset)
        self.assertEqual(failed_receipt["system_commit"]["last_phase"], "after_reset")
        self.assertEqual(failed_receipt["system_commit"]["last_snapshot"], after_reset)

    def test_unavailable_commit_snapshot_does_not_invent_peak(self) -> None:
        system_commit: dict[str, object] = {
            "preflight": None,
            "maximum_observed_percent": None,
        }
        with patch(
            "rivermark_benchmark.isaac_smoke._windows_system_commit_snapshot",
            side_effect=AssertionError("unavailable telemetry must not be resampled"),
        ):
            self.assertIsNone(
                _check_commit(
                    threshold_percent=85.0,
                    phase="after_reset",
                    system_commit=system_commit,
                    snapshot=None,
                )
            )
        self.assertIsNone(system_commit["maximum_observed_percent"])
        self.assertNotIn("maximum_phase", system_commit)
        self.assertNotIn("maximum_snapshot", system_commit)


if __name__ == "__main__":
    unittest.main()
