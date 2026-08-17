from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.researcher_entry import (  # noqa: E402
    RESEARCHER_SMOKE_SCHEMA,
    ResearcherEntryError,
    run_researcher_smoke,
)


class ResearcherEntryTests(unittest.TestCase):
    def test_smoke_is_hash_bound_and_non_formal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke"
            result = run_researcher_smoke(root)
            self.assertEqual(result["schema"], RESEARCHER_SMOKE_SCHEMA)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["formal_benchmark_admission"])
            self.assertFalse(result["checks"]["private_truth_present"])
            self.assertFalse(result["checks"]["isaac_started"])
            report = json.loads((root / "researcher_smoke_report.json").read_text())
            self.assertEqual(report["fixture"]["frame_count"], 5)
            self.assertGreater(report["runtime"]["artifact_bytes_before_report"], 0)
            self.assertGreaterEqual(report["runtime"]["peak_python_allocated_bytes"], 0)
            self.assertEqual(report["checks"]["selective_loader"], "passed")

    def test_non_empty_destination_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke"
            root.mkdir()
            marker = root / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(ResearcherEntryError):
                run_researcher_smoke(root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
