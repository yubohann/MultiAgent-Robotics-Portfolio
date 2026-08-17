import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CommunityContractTests(unittest.TestCase):
    def test_stability_policy_declares_compatibility_and_deprecation(self) -> None:
        text = (ROOT / "docs" / "api-schema-stability.md").read_text(encoding="utf-8")
        for marker in ("Compatibility rules", "Deprecation", "ABI compatibility", "never replaces bytes"):
            self.assertIn(marker, text)

    def test_security_policy_keeps_private_truth_out_of_public_issues(self) -> None:
        text = (ROOT / "docs" / "security-and-integrity.md").read_text(encoding="utf-8")
        self.assertIn("private GitHub security-advisory channel", text)
        self.assertIn("private target manifests", text)
        self.assertIn("hash-bound defect and tombstone", text)

