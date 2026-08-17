from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.collection_protocol import (  # noqa: E402
    COLLECTION_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    POWER_METHOD,
    SEED_DERIVATION,
    CollectionProtocolError,
    coverage_report,
    derive_episode_seed,
    load_collection_protocol,
    native_t2_motion_contract,
    native_t2_v2_motion_contract,
    native_t2_v3_motion_contract,
    protocol_sha256,
    resolve_collection_binding,
    required_paired_episodes,
    validate_collection_protocol,
    validate_collection_binding,
)
from rivermark_benchmark.failure_ledger import FailureRecord  # noqa: E402


AXES = (
    "layout",
    "route",
    "route_family",
    "target_count",
    "height",
    "region",
    "occlusion",
    "density",
    "appearance",
    "dynamics",
    "lighting",
    "weather",
    "initial_condition",
    "start_anchor",
    "target_region",
    "visibility_bucket",
    "communication",
    "control_latency",
    "agent_dropout",
)
SPLITS = ("train", "inner_dev", "validation", "blind_test", "ood_test")
NATIVE_T2_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v1.json"
)
NATIVE_T2_V2_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v2.json"
)
NATIVE_T2_V3_PROTOCOL_PATH = (
    ROOT / "config" / "collection_protocol.citylite_native_t2_canary_v3.json"
)


def _protocol() -> dict:
    required = required_paired_episodes(
        familywise_alpha=0.05,
        power=0.8,
        minimum_effect_size=0.05,
        difference_standard_deviation=0.15,
        comparison_count=2,
    )
    routes = [f"route-{index}" for index in range(len(SPLITS))]
    holdout_values = {
        "route_family": [f"route-family-{index}" for index in range(len(SPLITS))],
        "start_anchor": [f"start-anchor-{index}" for index in range(len(SPLITS))],
        "target_region": [f"target-region-{index}" for index in range(len(SPLITS))],
        "visibility_bucket": [f"visibility-bucket-{index}" for index in range(len(SPLITS))],
    }
    axes = []
    for axis in AXES:
        values = routes if axis == "route" else holdout_values.get(axis, [f"{axis}-nominal"])
        split_role = (
            "holdout"
            if axis in holdout_values
            else "scene"
            if axis == "layout"
            else "episode"
            if axis == "initial_condition"
            else "condition"
        )
        axes.append({"axis_id": axis, "values": values, "split_role": split_role})
    base_conditions = {axis: f"{axis}-nominal" for axis in AXES}
    cells = []
    for index, split in enumerate(SPLITS):
        conditions = {
            **base_conditions,
            "route": routes[index],
            **{axis: values[index] for axis, values in holdout_values.items()},
        }
        cells.append(
            {
                "cell_id": f"{split}-route-{index}",
                "split": split,
                "conditions": conditions,
                "minimum_attempts": 1,
                "minimum_admitted": 1,
            }
        )
    return {
        "schema": COLLECTION_PROTOCOL_SCHEMA,
        "protocol_id": "citylite-coverage-v1",
        "version": "1.0.0",
        "dataset_version": "0.1.0",
        "scene_identity": "RIVERMARK_CITY_LITE_v1",
        "track": "multi_uav_search3d",
        "agent_count": 8,
        "axes": axes,
        "cells": cells,
        "randomization": {
            "seed_derivation": SEED_DERIVATION,
            "episode_seed_start": 20260724,
            "paired_initial_conditions": True,
        },
        "power_analysis": {
            "method": POWER_METHOD,
            "primary_metric": "normalized_confirmed_auc",
            "familywise_alpha": 0.05,
            "power": 0.8,
            "minimum_effect_size": 0.05,
            "difference_standard_deviation": 0.15,
            "comparison_count": 2,
            "evaluation_split": "validation",
            "required_evaluation_episodes": required,
        },
        "exclusion_rules": [
            "reject_missing_sensor_frame",
            "reject_pose_closure_failure",
            "retain_failed_attempt_in_ledger",
        ],
    }


def _record(
    protocol: dict,
    *,
    cell_id: str,
    episode_index: int,
    serial: int,
    outcome: str = "admitted",
    episode_id: str | None = None,
    reason_code: str | None = None,
) -> dict:
    cell = next(cell for cell in protocol["cells"] if cell["cell_id"] == cell_id)
    seed = derive_episode_seed(
        protocol_id=protocol["protocol_id"],
        cell_id=cell_id,
        episode_seed_start=protocol["randomization"]["episode_seed_start"],
        episode_index=episode_index,
    )
    return FailureRecord(
        attempt_id=f"attempt-{serial:05d}",
        outcome=outcome,
        category="none" if outcome == "admitted" else "quality_failure",
        stage="formal_admission",
        recorded_at="2026-07-24T00:00:00Z",
        split=cell["split"],
        episode_id=episode_id if episode_id is not None else (f"episode-{serial:05d}" if outcome == "admitted" else None),
        reason_code=reason_code,
        collection_protocol_id=protocol["protocol_id"],
        collection_protocol_sha256=protocol_sha256(protocol),
        collection_cell_id=cell_id,
        collection_episode_index=episode_index,
        episode_seed=seed,
    ).as_dict()


def _complete_records(protocol: dict) -> list[dict]:
    records = []
    serial = 0
    validation_cell = next(cell["cell_id"] for cell in protocol["cells"] if cell["split"] == "validation")
    required = protocol["power_analysis"]["required_evaluation_episodes"]
    for episode_index in range(required):
        serial += 1
        records.append(_record(protocol, cell_id=validation_cell, episode_index=episode_index, serial=serial))
    for cell in protocol["cells"]:
        if cell["cell_id"] == validation_cell:
            continue
        serial += 1
        records.append(_record(protocol, cell_id=cell["cell_id"], episode_index=0, serial=serial))
    return records


class CollectionProtocolTests(unittest.TestCase):
    def test_protocol_power_and_seed_are_deterministic(self) -> None:
        payload = _protocol()
        self.assertEqual(validate_collection_protocol(payload), ())
        self.assertEqual(
            required_paired_episodes(
                familywise_alpha=0.05,
                power=0.8,
                minimum_effect_size=0.05,
                difference_standard_deviation=0.15,
                comparison_count=2,
            ),
            payload["power_analysis"]["required_evaluation_episodes"],
        )
        self.assertEqual(payload["power_analysis"]["required_evaluation_episodes"], 86)
        first = derive_episode_seed(
            protocol_id=payload["protocol_id"],
            cell_id=payload["cells"][0]["cell_id"],
            episode_seed_start=payload["randomization"]["episode_seed_start"],
            episode_index=0,
        )
        self.assertEqual(first, 3504686111)
        self.assertNotEqual(
            first,
            derive_episode_seed(
                protocol_id=payload["protocol_id"],
                cell_id=payload["cells"][0]["cell_id"],
                episode_seed_start=payload["randomization"]["episode_seed_start"],
                episode_index=1,
            ),
        )
        binding = resolve_collection_binding(
            payload,
            cell_id=payload["cells"][0]["cell_id"],
            episode_index=0,
        )
        self.assertEqual(validate_collection_binding(binding), ())
        self.assertEqual(binding["protocol_sha256"], protocol_sha256(payload))
        self.assertEqual(binding["episode_seed"], first)
        private_binding = {**binding, "cell_id": "private-route-0"}
        self.assertIn(
            "cell_id",
            {issue.code for issue in validate_collection_binding(private_binding)},
        )
        with self.assertRaisesRegex(CollectionProtocolError, "unknown collection cell"):
            resolve_collection_binding(payload, cell_id="missing-cell", episode_index=0)

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional jsonschema dependency is not installed")
    def test_checked_in_minimal_profile_has_explicit_empty_coverage(self) -> None:
        from jsonschema import Draft202012Validator

        path = ROOT / "config" / "collection_protocol.citylite_minimal_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_collection_protocol(payload), ())
        self.assertEqual(len(payload["axes"]), 14)
        self.assertEqual({cell["split"] for cell in payload["cells"]}, {"train", "validation"})
        report = coverage_report(payload, [])
        self.assertFalse(report["complete"])
        self.assertEqual(report["attempt_count"], 0)
        checked_in_report = json.loads(
            (ROOT / "config" / "collection_coverage.citylite_minimal_v1.empty.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(checked_in_report, report)
        schema = json.loads((ROOT / "schemas" / "coverage_report_v1.schema.json").read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.path))
        self.assertEqual(errors, [], [error.message for error in errors])

    def test_coverage_requires_quota_and_held_out_power_target(self) -> None:
        payload = _protocol()
        records = _complete_records(payload)
        report = coverage_report(payload, records)
        self.assertTrue(report["complete"])
        self.assertEqual(report["protocol_sha256"], protocol_sha256(payload))
        self.assertTrue(report["power_analysis"]["power_target_met"])
        self.assertEqual({cell["status"] for cell in report["cells"]}, {"passed"})

        train_cell = next(cell["cell_id"] for cell in payload["cells"] if cell["split"] == "train")
        train_heavy = [
            _record(payload, cell_id=cell["cell_id"], episode_index=0, serial=index)
            for index, cell in enumerate(payload["cells"], 1)
        ]
        start = len(train_heavy) + 1
        for offset in range(payload["power_analysis"]["required_evaluation_episodes"]):
            train_heavy.append(
                _record(payload, cell_id=train_cell, episode_index=offset + 1, serial=start + offset)
            )
        incomplete = coverage_report(payload, train_heavy)
        self.assertFalse(incomplete["complete"])
        self.assertFalse(incomplete["power_analysis"]["power_target_met"])

    def test_report_accounts_for_public_exclusion_reasons(self) -> None:
        payload = _protocol()
        records = _complete_records(payload)
        cell_id = payload["cells"][0]["cell_id"]
        records.append(
            _record(
                payload,
                cell_id=cell_id,
                episode_index=1000,
                serial=1000,
                outcome="failed",
                reason_code="pose_closure_failure",
            )
        )
        report = coverage_report(payload, records)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["exclusion_reasons"], {"pose_closure_failure": 1})

    def test_private_unknown_and_incomplete_protocol_values_fail_closed(self) -> None:
        payload = _protocol()
        payload["cells"][0]["conditions"]["route"] = "hidden_target_route"
        codes = {issue.code for issue in validate_collection_protocol(payload)}
        self.assertIn("axis_value", codes)
        self.assertIn("private_value", codes)

        unknown = copy.deepcopy(_protocol())
        unknown["axes"][0]["axis_id"] = []
        codes = {issue.code for issue in validate_collection_protocol(unknown)}
        self.assertIn("axis_id", codes)
        self.assertIn("unknown_axis", codes)
        self.assertIn("condition_coverage", codes)

        missing_train = copy.deepcopy(_protocol())
        missing_train["cells"] = [
            cell for cell in missing_train["cells"] if cell["split"] != "train"
        ]
        self.assertIn("split_coverage", {issue.code for issue in validate_collection_protocol(missing_train)})

    def test_same_layout_holdout_rejects_seed_only_train_validation_split(self) -> None:
        payload = _protocol()
        validation = next(cell for cell in payload["cells"] if cell["split"] == "validation")
        train = next(cell for cell in payload["cells"] if cell["split"] == "train")
        for axis in ("route_family", "start_anchor", "target_region", "visibility_bucket"):
            validation["conditions"][axis] = train["conditions"][axis]
        codes = {issue.code for issue in validate_collection_protocol(payload)}
        self.assertIn("holdout_overlap", codes)

    def test_same_layout_holdout_axes_are_mandatory_and_explicit(self) -> None:
        payload = _protocol()
        payload["axes"] = [axis for axis in payload["axes"] if axis["axis_id"] != "visibility_bucket"]
        for cell in payload["cells"]:
            del cell["conditions"]["visibility_bucket"]
        self.assertIn(
            "missing_holdout_axis",
            {issue.code for issue in validate_collection_protocol(payload)},
        )

    def test_pilot_protocol_may_omit_unimplemented_blind_and_ood_splits(self) -> None:
        payload = _protocol()
        payload["cells"] = [
            cell for cell in payload["cells"] if cell["split"] in {"train", "validation"}
        ]
        for axis in payload["axes"]:
            used_values = {cell["conditions"][axis["axis_id"]] for cell in payload["cells"]}
            axis["values"] = [value for value in axis["values"] if value in used_values]
        self.assertEqual(validate_collection_protocol(payload), ())

        no_evaluation = copy.deepcopy(payload)
        no_evaluation["cells"] = [
            cell for cell in no_evaluation["cells"] if cell["split"] == "train"
        ]
        self.assertIn(
            "evaluation_split",
            {issue.code for issue in validate_collection_protocol(no_evaluation)},
        )

    def test_protocol_may_omit_axes_without_runtime_executors(self) -> None:
        payload = _protocol()
        omitted = {"occlusion", "density", "appearance", "lighting", "weather"}
        payload["axes"] = [axis for axis in payload["axes"] if axis["axis_id"] not in omitted]
        for cell in payload["cells"]:
            cell["conditions"] = {
                axis: value for axis, value in cell["conditions"].items() if axis not in omitted
            }
        self.assertEqual(validate_collection_protocol(payload), ())

        incomplete = copy.deepcopy(payload)
        del incomplete["cells"][0]["conditions"]["route"]
        self.assertIn(
            "condition_coverage",
            {issue.code for issue in validate_collection_protocol(incomplete)},
        )

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "optional jsonschema dependency is not installed")
    def test_normalized_coverage_report_matches_public_schema(self) -> None:
        from jsonschema import Draft202012Validator

        payload = _protocol()
        omitted = {"occlusion", "density", "appearance", "lighting", "weather"}
        payload["axes"] = [axis for axis in payload["axes"] if axis["axis_id"] not in omitted]
        for cell in payload["cells"]:
            cell["conditions"] = {
                axis: value for axis, value in cell["conditions"].items() if axis not in omitted
            }
        payload["cells"] = [cell for cell in payload["cells"] if cell["split"] in {"train", "validation"}]
        for axis in payload["axes"]:
            used_values = {cell["conditions"][axis["axis_id"]] for cell in payload["cells"]}
            axis["values"] = [value for value in axis["values"] if value in used_values]
        report = coverage_report(payload, [])
        schema = json.loads((ROOT / "schemas" / "coverage_report_v1.schema.json").read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.path))
        self.assertEqual(errors, [], [error.message for error in errors])

    def test_protocol_hash_seed_and_duplicate_bindings_are_rejected(self) -> None:
        payload = _protocol()
        record = _record(payload, cell_id=payload["cells"][0]["cell_id"], episode_index=0, serial=1)
        with self.assertRaisesRegex(CollectionProtocolError, "duplicate attempt_id"):
            coverage_report(payload, [record, dict(record)])

        duplicate_episode = _record(payload, cell_id=payload["cells"][1]["cell_id"], episode_index=0, serial=2)
        duplicate_episode["episode_id"] = record["episode_id"]
        with self.assertRaisesRegex(CollectionProtocolError, "duplicate admitted episode_id"):
            coverage_report(payload, [record, duplicate_episode])

        duplicate_index = _record(payload, cell_id=payload["cells"][0]["cell_id"], episode_index=0, serial=3)
        with self.assertRaisesRegex(CollectionProtocolError, "repeats an admitted collection cell episode index"):
            coverage_report(payload, [record, duplicate_index])

        failed_retry = _record(
            payload,
            cell_id=payload["cells"][0]["cell_id"],
            episode_index=0,
            serial=4,
            outcome="failed",
            reason_code="sensor_frame_missing",
        )
        retry_report = coverage_report(payload, [failed_retry, record])
        self.assertEqual(retry_report["attempt_count"], 2)
        self.assertEqual(retry_report["admitted_count"], 1)

        wrong_hash = dict(record, collection_protocol_sha256="0" * 64)
        stale_report = coverage_report(payload, [wrong_hash])
        self.assertEqual(stale_report["attempt_count"], 0)
        self.assertEqual(stale_report["excluded_protocol_hash_count"], 1)

        wrong_seed = dict(record, episode_seed=(record["episode_seed"] + 1) % (2**32))
        with self.assertRaisesRegex(CollectionProtocolError, "deterministic episode seed"):
            coverage_report(payload, [wrong_seed])

    def test_coverage_ignores_legacy_and_other_protocol_ledger_records(self) -> None:
        payload = _protocol()
        current = _record(
            payload,
            cell_id=payload["cells"][0]["cell_id"],
            episode_index=0,
            serial=1,
        )
        other_protocol = copy.deepcopy(payload)
        other_protocol["protocol_id"] = "other-coverage-v1"
        foreign = _record(
            other_protocol,
            cell_id=other_protocol["cells"][0]["cell_id"],
            episode_index=0,
            serial=2,
        )
        prior_revision = _record(
            payload,
            cell_id=payload["cells"][0]["cell_id"],
            episode_index=1,
            serial=3,
        )
        prior_revision["collection_protocol_sha256"] = "f" * 64
        legacy = FailureRecord(
            attempt_id="attempt-00004",
            outcome="failed",
            category="capture_failure",
            stage="isaac_capture",
            recorded_at="2026-07-24T00:00:00Z",
            split="pilot",
            reason_code="legacy_capture",
        ).as_dict()

        report = coverage_report(payload, [legacy, foreign, prior_revision, current])

        self.assertEqual(report["attempt_count"], 1)
        self.assertEqual(report["admitted_count"], 1)
        self.assertEqual(report["ledger_record_count"], 4)
        self.assertEqual(report["excluded_ledger_record_count"], 3)
        self.assertEqual(report["excluded_protocol_id_count"], 2)
        self.assertEqual(report["excluded_protocol_hash_count"], 1)
        current_cell = next(
            cell
            for cell in report["cells"]
            if cell["cell_id"] == payload["cells"][0]["cell_id"]
        )
        self.assertEqual(current_cell["attempt_count"], 1)

    def test_native_t2_canary_protocol_is_development_only_and_not_coverage_input(self) -> None:
        protocol = load_collection_protocol(NATIVE_T2_PROTOCOL_PATH)

        self.assertEqual(protocol["schema"], NATIVE_T2_CANARY_PROTOCOL_SCHEMA)
        self.assertEqual(validate_collection_protocol(protocol), ())
        self.assertEqual(protocol["claim_boundary"]["formal_episode"], False)
        self.assertEqual(protocol["claim_boundary"]["benchmark_score"], False)
        binding = resolve_collection_binding(
            protocol,
            cell_id="native-t2-canary-inner-dev-v1",
            episode_index=1,
        )
        self.assertEqual(binding["split"], "inner_dev")
        self.assertEqual(binding["protocol_id"], "citylite-native-t2-canary-v1")
        with self.assertRaisesRegex(CollectionProtocolError, "not defined for development-only"):
            coverage_report(protocol, [])

        malformed = copy.deepcopy(protocol)
        malformed["execution_contract"]["required_independent_passes"] = 1
        issues = validate_collection_protocol(malformed)
        self.assertTrue(any(issue.code == "execution_contract" for issue in issues))

    def test_native_t2_v2_binds_motion_and_rejects_timing_tampering(self) -> None:
        protocol = load_collection_protocol(NATIVE_T2_V2_PROTOCOL_PATH)

        self.assertEqual(protocol["schema"], NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA)
        motion = native_t2_motion_contract(protocol)
        assert motion is not None
        self.assertEqual(motion, native_t2_v2_motion_contract())
        self.assertEqual(motion["max_horizontal_speed_mps"], 2.0)
        self.assertEqual(motion["camera_heading_model"], "segment_horizontal_heading_yaw_limited_v1")
        binding = resolve_collection_binding(
            protocol,
            cell_id="native-t2-canary-inner-dev-v2",
            episode_index=1,
        )
        self.assertEqual(binding["protocol_id"], "citylite-native-t2-canary-v2")
        malformed = copy.deepcopy(protocol)
        malformed["motion_contract"]["max_horizontal_speed_mps"] = 0.75
        issues = validate_collection_protocol(malformed)
        self.assertTrue(any(issue.code == "motion_contract" for issue in issues))

    def test_native_t2_v3_is_a_distinct_time_scaled_motion_contract(self) -> None:
        protocol = load_collection_protocol(NATIVE_T2_V3_PROTOCOL_PATH)

        self.assertEqual(protocol["schema"], NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA)
        motion = native_t2_motion_contract(protocol)
        assert motion is not None
        self.assertEqual(motion, native_t2_v3_motion_contract())
        self.assertEqual(motion["waypoint_segment_seconds"], 12.0)
        self.assertEqual(motion["rollout_steps"], 4800)
        self.assertEqual(motion["max_vertical_speed_mps"], 0.4)
        binding = resolve_collection_binding(
            protocol,
            cell_id="native-t2-canary-inner-dev-v3",
            episode_index=1,
        )
        self.assertEqual(binding["protocol_id"], "citylite-native-t2-canary-v3")
        malformed = copy.deepcopy(protocol)
        malformed["motion_contract"]["waypoint_segment_seconds"] = 6.0
        issues = validate_collection_protocol(malformed)
        self.assertTrue(any(issue.code == "motion_contract" for issue in issues))


if __name__ == "__main__":
    unittest.main()
