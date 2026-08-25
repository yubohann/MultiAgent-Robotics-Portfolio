from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
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

from rivermark_benchmark.runtime_lock import (  # noqa: E402
    RUNTIME_LOCK_SCHEMA,
    config_sha256,
    compare_live_simulation,
    compare_runtime,
    configure_simulation_cfg,
    environment_lock_sha256,
    load_runtime_lock,
    locked_launcher_kwargs,
    observe_live_simulation,
    runtime_lock_sha256,
    resolve_locked_environment_lock,
    source_tree_sha256,
    RuntimeLockError,
    validate_runtime_lock,
    validate_locked_launcher_environment,
)


def _lock() -> dict:
    render = {"rendering_mode": "balanced"}
    fabric = {"enable_scene_query_support": False, "use_fabric": True}
    physx = {"solver_type": 1}
    return {
        "schema": RUNTIME_LOCK_SCHEMA,
        "profile_id": "citylite-windows-isaac-5.1",
        "python": {"implementation": "CPython", "version": "3.11.14"},
        "host": {"system": "Windows", "machine": "AMD64", "minimum_windows_build": 26100},
        "distributions": {"isaacsim": "5.1.0.0", "isaaclab": "2.3.0", "isaaclab-contrib": "0.0.2"},
        "isaaclab_source": {"relative_path": "isaaclab", "tree_sha256": "a" * 64, "file_count": 2, "byte_count": 8, "version_file": "2.3.2", "extension_version": "0.54.3"},
        "isaaclab_contrib_source": {"relative_path": "isaaclab_contrib", "tree_sha256": "b" * 64, "file_count": 2, "byte_count": 8, "version_file": "2.3.2", "extension_version": "0.0.2"},
        "environment_lock": {"repository_relative_path": "requirements-isaac-capture.lock", "sha256": "f" * 64},
        "gpu": {"vendor": "NVIDIA", "minimum_driver_version": "576.80", "minimum_vram_bytes": 8 * 1024**3},
        "launcher": {
            "headless": True, "enable_cameras": True, "device": "cuda:0",
            "rendering_mode": "balanced", "livestream": 0, "xr": False,
            "distributed": False, "kit_args": "",
            "experience": {"path": "apps/isaaclab.python.headless.rendering.kit", "sha256": "c" * 64},
        },
        "simulation": {
            "device": "cuda:0", "dt_s": 0.005,
            "gravity_w_mps2": [0.0, 0.0, -9.81], "agent_count": 8,
            "render_interval": 1, "use_fabric": True,
            "config_digests": {
                "render": {"settings": render, "sha256": config_sha256(render)},
                "fabric": {"settings": fabric, "sha256": config_sha256(fabric)},
                "physx": {"settings": physx, "sha256": config_sha256(physx)},
            },
        },
        "assets": {"city_lite_contract_sha256": "d" * 64, "cf2x_usd_sha256": "e" * 64},
    }


def _observed(lock: dict) -> dict:
    return {
        "python": dict(lock["python"]),
        "host": {"system": "Windows", "machine": "AMD64", "windows_build": 26100},
        "distributions": dict(lock["distributions"]),
        "isaaclab_source": dict(lock["isaaclab_source"]),
        "isaaclab_contrib_source": dict(lock["isaaclab_contrib_source"]),
        "environment_lock": dict(lock["environment_lock"]),
        "launcher": {
            "experience": dict(lock["launcher"]["experience"]),
            "isaaclab_source_path_matches": True,
        },
        "gpu": {"devices": [{"name": "GPU", "vram_bytes": 24 * 1024**3, "driver_version": "576.80"}], "probe_error": None},
        "assets": dict(lock["assets"]),
    }


class RuntimeLockTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional jsonschema dependency is not installed")
    def test_checked_in_runtime_lock_matches_json_schema(self) -> None:
        from jsonschema import Draft202012Validator

        schema = json.loads((ROOT / "schemas" / "isaac_runtime_lock_v2.schema.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "config" / "isaac_runtime.windows-5.1.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(lock)

    def test_checked_in_runtime_lock_binds_opencv_and_dependency_lock(self) -> None:
        lock_path = ROOT / "config" / "isaac_runtime.windows-5.1.json"
        lock = load_runtime_lock(lock_path)
        dependency_lock = ROOT / "requirements-isaac-capture.lock"
        dependencies = {
            line.split("==", 1)[0]: line.split("==", 1)[1]
            for line in dependency_lock.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(dependencies["opencv-python"], "4.11.0.86")
        self.assertEqual(lock["distributions"]["opencv-python"], dependencies["opencv-python"])
        self.assertEqual(
            lock["environment_lock"]["sha256"],
            environment_lock_sha256(dependency_lock),
        )

    def test_valid_lock_and_observation_pass(self) -> None:
        lock = _lock()
        self.assertEqual(validate_runtime_lock(lock), ())
        self.assertEqual(compare_runtime(lock, _observed(lock)), ())
        self.assertEqual(runtime_lock_sha256(lock), runtime_lock_sha256(copy.deepcopy(lock)))

    def test_version_asset_source_and_gpu_drift_fail(self) -> None:
        lock = _lock()
        observed = _observed(lock)
        observed["distributions"]["isaacsim"] = "5.2.0.0"
        observed["assets"]["cf2x_usd_sha256"] = "d" * 64
        observed["isaaclab_source"]["tree_sha256"] = "e" * 64
        observed["isaaclab_contrib_source"]["tree_sha256"] = "f" * 64
        observed["gpu"]["devices"] = []
        observed["environment_lock"]["sha256"] = "0" * 64
        paths = {issue.path for issue in compare_runtime(lock, observed)}
        self.assertIn("$.distributions.isaacsim", paths)
        self.assertIn("$.assets.cf2x_usd_sha256", paths)
        self.assertIn("$.isaaclab_source.tree_sha256", paths)
        self.assertIn("$.isaaclab_contrib_source.tree_sha256", paths)
        self.assertIn("$.environment_lock.sha256", paths)
        self.assertIn("$.gpu", paths)

    def test_environment_lock_is_required_and_resolves_inside_repository(self) -> None:
        lock = _lock()
        del lock["environment_lock"]
        self.assertIn("environment_lock", {issue.code for issue in validate_runtime_lock(lock)})
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            (repository / "config").mkdir()
            dependency_lock = repository / "requirements-isaac-capture.lock"
            dependency_lock.write_text("numpy==1.26.0\n", encoding="utf-8")
            runtime_path = repository / "config" / "runtime.json"
            runtime_path.write_text("{}", encoding="utf-8")
            resolved = resolve_locked_environment_lock(runtime_path, _lock())
            self.assertEqual(resolved, dependency_lock.resolve())

    def test_tree_hash_is_path_independent_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp, tempfile.TemporaryDirectory() as second_temp:
            first = Path(first_temp)
            second = Path(second_temp)
            for root in (first, second):
                (root / "pkg").mkdir()
                (root / "pkg" / "a.py").write_text("a = 1\n", encoding="utf-8")
                (root / "config.toml").write_text("x = 2\n", encoding="utf-8")
                (root / "ignored.bin").write_bytes(b"ignored")
            self.assertEqual(source_tree_sha256(first), source_tree_sha256(second))
            (second / "pkg" / "a.py").write_text("a = 3\n", encoding="utf-8")
            self.assertNotEqual(source_tree_sha256(first)[0], source_tree_sha256(second)[0])

    def test_tree_hash_ignores_unreadable_non_source_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pkg").mkdir()
            (root / "pkg" / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "pkg" / "libfastcdr.so").write_bytes(b"binary")

            original_is_file = Path.is_file

            def is_file_without_binary_links(path: Path) -> bool:
                if path.suffix.lower() == ".so":
                    raise OSError("unresolvable non-source link")
                return original_is_file(path)

            with patch.object(Path, "is_file", is_file_without_binary_links):
                digest, file_count, byte_count = source_tree_sha256(root)

            self.assertEqual(file_count, 1)
            self.assertEqual(byte_count, (root / "pkg" / "a.py").stat().st_size)
            self.assertEqual(digest, source_tree_sha256(root)[0])

    def test_unknown_fields_fail_closed(self) -> None:
        lock = _lock()
        lock["guess"] = True
        self.assertIn("fields", {issue.code for issue in validate_runtime_lock(lock)})

    def test_legacy_v1_and_cross_section_drift_fail_closed(self) -> None:
        lock = _lock()
        lock["schema"] = "org.rivermark.benchmark.isaac-runtime-lock.v1"
        self.assertIn("schema", {issue.code for issue in validate_runtime_lock(lock)})
        lock = _lock()
        lock["launcher"]["device"] = "cpu"
        paths = {issue.path for issue in validate_runtime_lock(lock)}
        self.assertIn("$.launcher.device", paths)

    def test_locked_launcher_cfg_and_live_observation_use_only_bound_values(self) -> None:
        lock = _lock()
        source = Path(r"C:\IsaacLab\source\isaaclab")
        kwargs = locked_launcher_kwargs(lock, source)
        self.assertEqual(kwargs["experience"], str(Path(r"C:\IsaacLab\apps\isaaclab.python.headless.rendering.kit")))
        self.assertEqual(kwargs["kit_args"], "")

        cfg = SimpleNamespace(
            device="cpu",
            dt=1.0,
            gravity=(0.0, 0.0, 0.0),
            render_interval=3,
            use_fabric=False,
            enable_scene_query_support=True,
            render=SimpleNamespace(rendering_mode="quality"),
            physx=SimpleNamespace(solver_type=0),
        )
        configure_simulation_cfg(cfg, lock)
        self.assertEqual(cfg.device, "cuda:0")
        self.assertEqual(cfg.dt, 0.005)
        self.assertEqual(cfg.render.rendering_mode, "balanced")
        self.assertEqual(cfg.physx.solver_type, 1)

        class PhysicsContext:
            @staticmethod
            def get_gravity():
                return [0.0, 0.0, -1.0], 9.81

        class Simulation:

            @staticmethod
            def get_physics_dt():
                return 0.005

            @staticmethod
            def get_rendering_dt():
                return 0.005

            @staticmethod
            def get_physics_context():
                return PhysicsContext()

            @staticmethod
            def is_fabric_enabled():
                return True

            @staticmethod
            def has_rtx_sensors():
                return True

        simulation = Simulation()
        simulation.device = cfg.device
        simulation.cfg = cfg
        simulation.carb_settings = SimpleNamespace(get=lambda _path: "balanced")
        observed = observe_live_simulation(lock, simulation)
        self.assertEqual(compare_live_simulation(lock, observed), ())

    def test_xr_environment_conflict_fails_before_launcher(self) -> None:
        with patch.dict("os.environ", {"XR": "1"}, clear=False):
            with self.assertRaisesRegex(ValueError, "conflicts"):
                validate_locked_launcher_environment(_lock())

    def test_live_gravity_accepts_only_float32_readback_error(self) -> None:
        lock = _lock()
        observed = {
            "device": "cuda:0",
            "physics_dt_s": 0.005,
            "rendering_dt_s": 0.005,
            "gravity_w_mps2": [0.0, 0.0, -9.8100004196167],
            "render_interval": 1,
            "use_fabric": True,
            "rendering_mode": "balanced",
            "config_digests": {
                name: payload["sha256"]
                for name, payload in lock["simulation"]["config_digests"].items()
            },
        }
        self.assertEqual(compare_live_simulation(lock, observed), ())

        for invalid_gravity in (
            [0.0, 0.0, -9.81001],
            [0.0, 0.0, float("nan")],
            [0.0, 0.0],
            [0.0, 0.0, "-9.81"],
        ):
            with self.subTest(gravity=invalid_gravity):
                invalid_observed = dict(observed, gravity_w_mps2=invalid_gravity)
                self.assertEqual(
                    {issue.path for issue in compare_live_simulation(lock, invalid_observed)},
                    {"$.simulation.gravity_w_mps2"},
                )

    def test_live_simulation_invalid_timing_is_a_structured_mismatch(self) -> None:
        lock = _lock()
        observed = {
            "device": "cuda:0",
            "physics_dt_s": "wrong",
            "rendering_dt_s": math.inf,
            "gravity_w_mps2": [0.0, 0.0, -9.81],
            "render_interval": 1,
            "use_fabric": True,
            "rendering_mode": "balanced",
            "config_digests": {
                name: payload["sha256"]
                for name, payload in lock["simulation"]["config_digests"].items()
            },
        }
        paths = {issue.path for issue in compare_live_simulation(lock, observed)}
        self.assertIn("$.simulation.dt_s", paths)
        self.assertIn("$.simulation.render_interval", paths)

    def test_nonfinite_locked_gravity_fails_schema_validation(self) -> None:
        lock = _lock()
        lock["simulation"]["gravity_w_mps2"][2] = float("inf")
        self.assertIn(
            "$.simulation.gravity_w_mps2",
            {issue.path for issue in validate_runtime_lock(lock)},
        )

    def test_lock_validation_rejects_unsafe_scalar_values(self) -> None:
        cases = (
            ("nan_dt", lambda lock: lock["simulation"].__setitem__("dt_s", math.nan), "$.simulation.dt_s"),
            ("infinite_dt", lambda lock: lock["simulation"].__setitem__("dt_s", math.inf), "$.simulation.dt_s"),
            ("zero_vram", lambda lock: lock["gpu"].__setitem__("minimum_vram_bytes", 0), "$.gpu.minimum_vram_bytes"),
            ("negative_vram", lambda lock: lock["gpu"].__setitem__("minimum_vram_bytes", -1), "$.gpu.minimum_vram_bytes"),
            ("string_vram", lambda lock: lock["gpu"].__setitem__("minimum_vram_bytes", "8GiB"), "$.gpu.minimum_vram_bytes"),
            ("zero_windows_build", lambda lock: lock["host"].__setitem__("minimum_windows_build", 0), "$.host.minimum_windows_build"),
            ("string_windows_build", lambda lock: lock["host"].__setitem__("minimum_windows_build", "26100"), "$.host.minimum_windows_build"),
            ("non_windows_host", lambda lock: lock["host"].__setitem__("system", "Linux"), "$.host.system"),
            ("bool_agent_count", lambda lock: lock["simulation"].__setitem__("agent_count", True), "$.simulation.agent_count"),
            ("bool_render_interval", lambda lock: lock["simulation"].__setitem__("render_interval", True), "$.simulation.render_interval"),
            ("bool_livestream", lambda lock: lock["launcher"].__setitem__("livestream", True), "$.launcher.livestream"),
            ("bool_source_count", lambda lock: lock["isaaclab_source"].__setitem__("file_count", True), "$.isaaclab_source.file_count"),
        )
        for name, mutate, expected_path in cases:
            with self.subTest(name=name):
                lock = _lock()
                mutate(lock)
                self.assertIn(expected_path, {issue.path for issue in validate_runtime_lock(lock)})

    def test_lock_validation_rejects_nonfinite_settings_without_raising(self) -> None:
        lock = _lock()
        lock["simulation"]["config_digests"]["render"]["settings"]["exposure"] = math.nan
        self.assertIn(
            "$.simulation.config_digests.render.settings",
            {issue.path for issue in validate_runtime_lock(lock)},
        )

    def test_lock_validation_rejects_scalar_digest_settings_without_raising(self) -> None:
        lock = _lock()
        lock["simulation"]["config_digests"]["render"]["settings"] = "not-a-settings-object"
        self.assertIn(
            "$.simulation.config_digests.render.settings",
            {issue.path for issue in validate_runtime_lock(lock)},
        )

    def test_load_runtime_lock_rejects_nonstandard_or_overflowing_json_numbers(self) -> None:
        serialized = json.dumps(_lock(), sort_keys=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime-lock.json"
            for literal in ("NaN", "Infinity", "1e999"):
                with self.subTest(literal=literal):
                    path.write_text(
                        serialized.replace('"dt_s": 0.005', f'"dt_s": {literal}'),
                        encoding="utf-8",
                    )
                    with self.assertRaises(RuntimeLockError):
                        load_runtime_lock(path)


if __name__ == "__main__":
    unittest.main()
