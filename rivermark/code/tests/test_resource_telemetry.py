from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rivermark_benchmark.resource_telemetry import (
    FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA,
    RESOURCE_TELEMETRY_SCHEMA,
    ResourceTelemetry,
    _owned_process_ids_from_parent_rows,
    _summarize_foreign_native_process_rows,
    foreign_native_process_census,
)


class ResourceTelemetryTests(unittest.TestCase):
    def test_explicit_samples_have_bounded_summary(self) -> None:
        telemetry = ResourceTelemetry()
        first = telemetry.sample("before_app_launcher")
        second = telemetry.sample("after_reset")
        payload = telemetry.as_dict()
        self.assertEqual(payload["schema"], RESOURCE_TELEMETRY_SCHEMA)
        self.assertEqual(payload["sample_count"], 2)
        self.assertEqual([row["phase"] for row in payload["samples"]], ["before_app_launcher", "after_reset"])
        self.assertIn("maxima", payload)
        self.assertEqual(first["phase"], "before_app_launcher")
        self.assertEqual(second["phase"], "after_reset")

    def test_invalid_phase_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResourceTelemetry().sample("")

    def test_windows_private_commit_has_a_distinct_semantic_field(self) -> None:
        telemetry = ResourceTelemetry()
        with mock.patch(
            "rivermark_benchmark.resource_telemetry._process_memory_snapshot",
            return_value={
                "working_set_bytes": 10,
                "private_commit_bytes": 7,
                "pagefile_usage_bytes": 9,
                "peak_pagefile_usage_bytes": 11,
            },
        ):
            row = telemetry.sample("private_commit_check")
        self.assertEqual(row["process"]["private_commit_bytes"], 7)
        self.assertEqual(row["process"]["pagefile_usage_bytes"], 9)

    def test_commit_attribution_is_aggregate_and_does_not_require_process_identity(self) -> None:
        telemetry = ResourceTelemetry()
        with mock.patch(
            "rivermark_benchmark.resource_telemetry._process_memory_snapshot",
            return_value={"private_commit_bytes": 7},
        ), mock.patch(
            "rivermark_benchmark.resource_telemetry._system_commit_snapshot",
            return_value={
                "commit_total_bytes": 19,
                "commit_limit_bytes": 32,
                "commit_peak_bytes": 20,
                "commit_percent": 59.375,
                "kernel_total_bytes": 3,
            },
        ):
            row = telemetry.sample("after_reset")
        system_commit = row["system_commit"]
        self.assertEqual(system_commit["commit_outside_current_process_bytes"], 12)
        self.assertEqual(system_commit["kernel_total_bytes"], 3)
        self.assertNotIn("processes", system_commit)

    def test_commit_attribution_omits_incoherent_snapshot(self) -> None:
        telemetry = ResourceTelemetry()
        with mock.patch(
            "rivermark_benchmark.resource_telemetry._process_memory_snapshot",
            return_value={"private_commit_bytes": 20},
        ), mock.patch(
            "rivermark_benchmark.resource_telemetry._system_commit_snapshot",
            return_value={"commit_total_bytes": 19},
        ):
            row = telemetry.sample("after_reset")
        self.assertNotIn("commit_outside_current_process_bytes", row["system_commit"])

    def test_foreign_native_census_excludes_self_and_non_native_processes(self) -> None:
        gib = 1024**3
        census = _summarize_foreign_native_process_rows(
            (
                {"pid": 11, "executable": "python.exe", "private_commit_bytes": 19 * gib},
                {"pid": 22, "executable": "kit.exe", "private_commit_bytes": 8 * gib},
                {"pid": 33, "executable": "msedge.exe", "private_commit_bytes": 12 * gib},
                {"pid": 44, "executable": "python.exe", "private_commit_bytes": 7 * gib},
            ),
            current_pid=22,
            minimum_private_commit_bytes=8 * gib,
        )
        self.assertEqual(census["schema"], FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA)
        self.assertEqual(census["candidate_count"], 1)
        self.assertEqual(census["candidate_private_commit_bytes"], 19 * gib)
        self.assertEqual(census["maximum_candidate_private_commit_bytes"], 19 * gib)

    def test_owned_process_tree_includes_all_descendants(self) -> None:
        owned = _owned_process_ids_from_parent_rows(
            ((2, 1), (3, 2), (4, 3), (5, 1), (6, 5)), root_pid=2
        )
        self.assertEqual(owned, frozenset({2, 3, 4}))

    def test_foreign_native_census_excludes_owned_kit_child_but_not_foreign_owner(self) -> None:
        gib = 1024**3
        rows = (
            {"pid": 22, "executable": "python.exe", "private_commit_bytes": 9 * gib},
            {"pid": 23, "executable": "kit.exe", "private_commit_bytes": 12 * gib},
            {"pid": 31, "executable": "python.exe", "private_commit_bytes": 14 * gib},
        )
        with mock.patch(
            "rivermark_benchmark.resource_telemetry._windows_native_process_rows",
            return_value=rows,
        ), mock.patch(
            "rivermark_benchmark.resource_telemetry._windows_owned_process_ids",
            return_value=frozenset({22, 23}),
        ):
            census = foreign_native_process_census(
                current_pid=22, minimum_private_commit_bytes=8 * gib
            )
        self.assertEqual(census["candidate_count"], 1)
        self.assertEqual(census["candidate_private_commit_bytes"], 14 * gib)


if __name__ == "__main__":
    unittest.main()
