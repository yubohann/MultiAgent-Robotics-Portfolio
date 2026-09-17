from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.supply_chain import (
    validate_supply_chain_manifest,
)


class LicenseScopeTests(unittest.TestCase):
    def test_source_license_is_bound_but_external_assets_remain_blocked(self) -> None:
        license_path = ROOT / "LICENSE"
        notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
        policy = (ROOT / "docs" / "asset-policy.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        payload = json.loads((ROOT / "config" / "supply_chain.pending.json").read_text(encoding="utf-8"))

        self.assertTrue(license_path.read_text(encoding="utf-8").lstrip().startswith("Apache License"))
        self.assertIn("does not grant rights in NVIDIA Isaac Sim", notice)
        self.assertIn("vendor NVIDIA Isaac Sim", policy)
        self.assertIn('license: "Apache-2.0"', citation)

        by_id = {asset["asset_id"]: asset for asset in payload["assets"]}
        source = by_id["rivermark-benchmark-source-license"]
        self.assertEqual(source["path"], "LICENSE")
        self.assertEqual(source["license_spdx"], "Apache-2.0")
        self.assertEqual(source["license_status"], "redistribution_cleared")
        self.assertTrue(source["redistributable"])
        self.assertIsInstance(source["identity"], str)
        self.assertEqual(source["decision_record"]["evidence_identity"], source["identity"])

        for asset_id in (
            "nvidia-rivermark-composition-usd",
            "city-lite-combined-layer",
            "cf2x-usd",
            "derived-recordings-and-labels",
            "public-demo-video",
        ):
            self.assertFalse(by_id[asset_id]["redistributable"], asset_id)
            self.assertNotEqual(by_id[asset_id]["license_status"], "redistribution_cleared", asset_id)

        release_codes = {issue.code for issue in validate_supply_chain_manifest(payload, require_release=True)}
        self.assertIn("license_closure", release_codes)
        self.assertIn("signature_required", release_codes)


if __name__ == "__main__":
    unittest.main()
