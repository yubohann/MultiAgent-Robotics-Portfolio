import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CommunityContractTests(unittest.TestCase):
    def test_stability_policy_declares_compatibility_and_deprecation(self) -> None:
        text = (ROOT / "docs" / "api-schema-stability.md").read_text(encoding="utf-8")
        for marker in ("Compatibility rules", "Deprecation", "ABI compatibility", "never replaces bytes"):
            self.assertIn(marker, text)

