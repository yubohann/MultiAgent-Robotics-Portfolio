from __future__ import annotations

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

from rivermark_benchmark.preflight import (
    RuntimePreflightRequirements,
    _probe_nvidia_smi,
    run_preflight,
    sha256_file,
)


class PreflightTests(unittest.TestCase):
    def test_nested_missing_output_directory_uses_existing_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new" / "nested" / "run"
            report = run_preflight(output_dir=output, minimum_free_bytes=1)
            self.assertTrue(report.valid, report.as_dict())

    def test_storage_and_asset_hash_pass_without_isaac(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "cf2x.usd"
            asset.write_bytes(b"asset")
            import hashlib

            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            report = run_preflight(
                output_dir=root / "runs",
                minimum_free_bytes=1,
                estimated_capture_bytes=1,
                required_assets=[(asset, digest)],
                source_root=None,
            )
            self.assertTrue(report.valid, report.as_dict())

    def test_missing_or_mismatched_asset_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = run_preflight(
                output_dir=root,
                minimum_free_bytes=0,
                required_assets=[(root / "missing.usd", None)],
            )
            self.assertFalse(report.valid)
            self.assertTrue(any(check.name.startswith("asset:") and not check.passed for check in report.checks))

    def test_negative_storage_budget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                run_preflight(output_dir=Path(temporary), minimum_free_bytes=-1)

    def test_existing_file_as_output_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-a-directory"
            output.write_bytes(b"occupied")
            report = run_preflight(output_dir=output, minimum_free_bytes=0)
            self.assertFalse(report.valid)
            self.assertFalse(next(check for check in report.checks if check.name == "output_directory").passed)

    def test_runtime_gpu_probe_parses_and_enforces_vram_and_driver(self) -> None:
        gpu = {"name": "Test GPU", "vram_bytes": 16 * 1024**3, "driver_version": "550.54.14"}
        with tempfile.TemporaryDirectory() as temporary:
            with patch("rivermark_benchmark.preflight._probe_nvidia_smi", return_value=([gpu], None)):
                report = run_preflight(
                    output_dir=Path(temporary),
                    minimum_free_bytes=0,
                    runtime=RuntimePreflightRequirements(
                        require_gpu=True,
                        minimum_gpu_vram_bytes=8 * 1024**3,
                        minimum_driver_version="545.0",
                    ),
                )
            self.assertTrue(report.valid, report.as_dict())
            self.assertTrue(next(check for check in report.checks if check.name == "gpu_capacity").passed)

            with patch("rivermark_benchmark.preflight._probe_nvidia_smi", return_value=([gpu], None)):
                failed = run_preflight(
                    output_dir=Path(temporary),
                    minimum_free_bytes=0,
                    runtime=RuntimePreflightRequirements(
                        require_gpu=True,
                        minimum_gpu_vram_bytes=32 * 1024**3,
                        minimum_driver_version="551.0",
                    ),
                )
        self.assertFalse(failed.valid)
        self.assertFalse(next(check for check in failed.checks if check.name == "gpu_capacity").passed)

    def test_nvidia_smi_probe_is_non_shell_and_bounded(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="Test GPU, 16384, 550.54.14\n",
            stderr="",
        )
        with patch("rivermark_benchmark.preflight.shutil.which", return_value="nvidia-smi.exe") as which:
            with patch("rivermark_benchmark.preflight.subprocess.run", return_value=completed) as run:
                records, error = _probe_nvidia_smi()
        self.assertIsNone(error)
        self.assertEqual(records[0]["name"], "Test GPU")
        self.assertEqual(records[0]["vram_bytes"], 16 * 1024**3)
        which.assert_called_once_with("nvidia-smi")
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_scene_contract_file_and_authority_are_checked_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "scene_contract.json"
            contract.write_text('{"scene_id":"RIVERMARK_CITY_LITE_v1"}\n', encoding="utf-8")
            authority = SimpleNamespace(
                contract_sha256=sha256_file(contract),
                asset_paths={"scene.usda": contract},
            )
            with patch("rivermark_benchmark.preflight.resolve_city_lite_authority", return_value=authority):
                report = run_preflight(
                    output_dir=root / "runs",
                    minimum_free_bytes=0,
                    runtime=RuntimePreflightRequirements(
                        scene_contract=contract,
                        scene_contract_sha256=sha256_file(contract),
                    ),
                )
        self.assertTrue(next(check for check in report.checks if check.name == "scene_contract_file").passed)
        self.assertTrue(next(check for check in report.checks if check.name == "scene_contract_authority").passed)

    def test_scene_contract_hash_mismatch_fails_closed_without_resolving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = Path(temporary) / "scene_contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            with patch("rivermark_benchmark.preflight.resolve_city_lite_authority") as resolver:
                report = run_preflight(
                    output_dir=Path(temporary),
                    minimum_free_bytes=0,
                    runtime=RuntimePreflightRequirements(
                        scene_contract=contract,
                        scene_contract_sha256="0" * 64,
                    ),
                )
        self.assertFalse(report.valid)
        resolver.assert_not_called()
        self.assertFalse(next(check for check in report.checks if check.name == "scene_contract_file").passed)

    def test_runtime_lock_is_checked_inside_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "runtime-lock.json"
            source = root / "isaaclab" / "source"
            contract = root / "scene-contract.json"
            drone = root / "cf2x.usd"
            source.mkdir(parents=True)
            lock.write_text("{}\n", encoding="utf-8")
            contract.write_text('{"scene_id":"RIVERMARK_CITY_LITE_v1"}\n', encoding="utf-8")
            drone.write_bytes(b"cf2x")
            authority = SimpleNamespace(
                contract_sha256=sha256_file(contract),
                asset_paths={"scene.usda": contract},
            )
            with patch("rivermark_benchmark.preflight.resolve_city_lite_authority", return_value=authority):
                with patch(
                    "rivermark_benchmark.runtime_lock.audit_runtime_lock",
                    return_value={"status": "passed", "profile_id": "test"},
                ) as audit:
                    report = run_preflight(
                        output_dir=root / "runs",
                        minimum_free_bytes=0,
                        runtime=RuntimePreflightRequirements(
                            runtime_lock=lock,
                            isaaclab_source=source,
                            scene_contract=contract,
                            cf2x_usd=drone,
                        ),
                    )
        runtime_check = next(check for check in report.checks if check.name == "runtime_lock")
        self.assertTrue(runtime_check.passed, runtime_check)
        audit.assert_called_once_with(
            lock.resolve(),
            isaaclab_source=source,
            scene_contract=contract,
            cf2x_usd=drone,
        )

    def test_runtime_lock_without_all_bound_paths_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "runtime-lock.json"
            lock.write_text("{}\n", encoding="utf-8")
            report = run_preflight(
                output_dir=Path(temporary) / "runs",
                minimum_free_bytes=0,
                runtime=RuntimePreflightRequirements(runtime_lock=lock),
            )
        runtime_check = next(check for check in report.checks if check.name == "runtime_lock")
        self.assertFalse(runtime_check.passed)
        self.assertEqual(runtime_check.value["status"], "failed")


if __name__ == "__main__":
    unittest.main()
