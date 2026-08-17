from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DocumentedNativeCaptureTests(unittest.TestCase):
    def test_completed_readme_disables_duplicate_native_capture(self) -> None:
        """A completed frozen cohort must not expose another live capture binding."""

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        protocol = json.loads(
            (ROOT / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json").read_text(
                encoding="utf-8"
            )
        )
        normalized = " ".join(readme.split())
        self.assertEqual({cell["minimum_attempts"] for cell in protocol["cells"]}, {4})
        self.assertIn(
            "No further collection binding is permitted under active protocol v2", normalized
        )
        self.assertIn("4 train + 4 validation unique-candidate sequence is complete", normalized)
        self.assertNotIn("& $py -m rivermark_benchmark.isaac_capture", readme)
        self.assertNotIn("$cellId =", readme)
        self.assertNotIn("$episodeIndex =", readme)
        self.assertNotIn("run a second capture", readme.lower())


if __name__ == "__main__":
    unittest.main()
