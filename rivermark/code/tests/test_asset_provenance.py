from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.asset_provenance import (  # noqa: E402
    LOCAL_ASSETS_SCHEMA,
    AssetProvenanceError,
    audit_local_assets_config,
    inspect_usd,
    main,
)


class AssetProvenanceTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _local_assets_config(self, root: Path) -> tuple[Path, dict[str, object]]:
        assets = root / "installed-assets"
        assets.mkdir()
        isaaclab = root / "IsaacLab"
        isaaclab.mkdir()
        python = isaaclab / "python.exe"
        python.write_bytes(b"local launcher placeholder")
        package_manifest = assets / "install-inventory.json"
        package_manifest.write_text('{"package":"Isaac Sim 5.1.x"}\n', encoding="utf-8")
        contract = assets / "rivermark_city_lite_scene_contract_v1.json"
        contract.write_text('{"schema":"city-lite"}\n', encoding="utf-8")
        layer = assets / "rivermark_city_lite.usda"
        layer.write_text('#usda 1.0\n@./dsready_content/nv_content/rivermark.usd@\n', encoding="utf-8")
        cf2x = assets / "cf2x.usd"
        cf2x.write_bytes(b"PXR-USDC\x00isaac-dev.ov.nvidia.com/Robots/Crazyflie/cf2x.usd")
        payload: dict[str, object] = {
            "schema": LOCAL_ASSETS_SCHEMA,
            "asset_root": str(assets),
            "asset_package_id": "official_isaacsim_assets_5_1",
            "asset_package_version": "Isaac Sim 5.1.x",
            "asset_package_manifest": str(package_manifest),
            "asset_package_sha256": self._sha256(package_manifest),
            "isaaclab_root": str(isaaclab),
            "isaac_python": str(python),
            "city_lite_contract": str(contract),
            "city_lite_contract_sha256": self._sha256(contract),
            "city_lite_layer": str(layer),
            "city_lite_layer_sha256": self._sha256(layer),
            "cf2x_usd": str(cf2x),
            "cf2x_usd_sha256": self._sha256(cf2x),
            "cf2x_source_provenance": "Isaac Sim runtime asset; distribution unresolved",
            "license_status": "internal_only",
            "public_redistribution": {
                "raw_nvidia_assets": False,
                "cf2x_usd": False,
                "city_lite_composed_layer": False,
                "rendered_video": False,
                "derived_sensor_payload": False,
            },
        }
        config = root / "local-assets.json"
        config.write_text(json.dumps(payload), encoding="utf-8")
        return config, payload

    def test_byoa_audit_binds_external_local_assets_without_clearing_them(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._local_assets_config(root)
            repository_root = root / "rivermark-benchmark"
            repository_root.mkdir()
            report = audit_local_assets_config(config, repository_root=repository_root)
            self.assertEqual(report["status"], "passed", report["issues"])
            self.assertEqual(report["schema"], LOCAL_ASSETS_SCHEMA)
            self.assertEqual(len(report["reports"]), 2)
            self.assertTrue(any(item["has_external_references"] for item in report["reports"]))

    def test_byoa_cli_accepts_external_asset_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _ = self._local_assets_config(root)
            repository_root = root / "rivermark-benchmark"
            repository_root.mkdir()
            self.assertEqual(
                main(["--config", str(config), "--repository-root", str(repository_root)]),
                0,
            )

    def test_byoa_audit_rejects_hash_mismatch_and_public_redistribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, payload = self._local_assets_config(root)
            payload["cf2x_usd_sha256"] = "0" * 64
            redistribution = payload["public_redistribution"]
            assert isinstance(redistribution, dict)
            redistribution["rendered_video"] = True
            config.write_text(json.dumps(payload), encoding="utf-8")
            report = audit_local_assets_config(config, repository_root=root / "repository")
            self.assertEqual(report["status"], "blocked")
            codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("hash_mismatch", codes)
            self.assertIn("redistribution", codes)

    def test_byoa_audit_rejects_runtime_asset_in_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, payload = self._local_assets_config(root)
            repository_root = root / "repository"
            repository_root.mkdir()
            copied_asset = repository_root / "cf2x.usd"
            copied_asset.write_bytes(Path(str(payload["cf2x_usd"])).read_bytes())
            payload["cf2x_usd"] = str(copied_asset)
            payload["cf2x_usd_sha256"] = self._sha256(copied_asset)
            config.write_text(json.dumps(payload), encoding="utf-8")
            report = audit_local_assets_config(config, repository_root=repository_root)
            self.assertEqual(report["status"], "blocked")
            self.assertIn("repository_asset", {issue["code"] for issue in report["issues"]})

    def test_binary_usdc_marker_is_detected_and_hash_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cf2x.usd"
            payload = b"PXR-USDC\x00metadata isaac-dev.ov.nvidia.com/I/Robots/Bitcraze/cf2x.usd\x00"
            path.write_bytes(payload)
            report = inspect_usd(path)
            self.assertEqual(report.usd_format, "usdc")
            self.assertTrue(report.has_external_references)
            self.assertEqual(report.classification, "nvidia_or_external_runtime_reference")
            self.assertEqual(report.license_status, "unresolved")
            self.assertEqual(report.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(report.references[0]["kind"], "nvidia_isaac_nucleus")

    def test_ascii_dsready_reference_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rivermark.usda"
            path.write_text(
                '#usda 1.0\n\n"./dsready_content/nv_content/rivermark/plaza.usd"\n',
                encoding="utf-8",
            )
            report = inspect_usd(path)
            self.assertEqual(report.usd_format, "usda")
            self.assertIn(
                "nvidia_dsready_content",
                {reference["kind"] for reference in report.references},
            )

    def test_usdc_split_marker_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rivermark.usd"
            path.write_bytes(b"PXR-USDC\x00dsready_\xB1\x02eent/nv\x0Bcontent")
            report = inspect_usd(path)
            self.assertTrue(report.has_external_references)
            self.assertIn(
                "nvidia_dsready_marker",
                {reference["kind"] for reference in report.references},
            )

    def test_truncated_scan_is_not_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.usd"
            path.write_bytes(b"#usda 1.0\n" + b"x" * 32 + b"dsready_content/late.usd")
            report = inspect_usd(path, max_scan_bytes=16)
            self.assertFalse(report.scan_complete)
            self.assertEqual(report.classification, "scan_truncated_unknown")
            self.assertEqual(report.license_status, "unresolved")

    def test_cli_blocks_external_reference_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scene.usd"
            path.write_bytes(b"PXR-USDC\x00dsready_content/nv_content/scene.usd")
            # main prints JSON; the return code is the contract under test.
            self.assertEqual(main([str(path), "--require-no-external-references"]), 2)

    def test_invalid_suffix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"data")
            with self.assertRaises(AssetProvenanceError):
                inspect_usd(path)


if __name__ == "__main__":
    unittest.main()
