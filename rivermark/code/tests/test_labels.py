from __future__ import annotations

import copy
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.labels import (  # noqa: E402
    LABEL_ONTOLOGY_SCHEMA,
    LABEL_RECORD_SCHEMA,
    main,
    ontology_sha256,
    validate_label_record,
    validate_label_sequence,
    validate_ontology,
)


def _ontology() -> dict[str, object]:
    return json.loads((ROOT / "config" / "label_ontology.citylite_v1.json").read_text(encoding="utf-8"))


def _record(*, frame_index: int = 0, timestamp_ns: int = 10) -> dict[str, object]:
    return {
        "schema": LABEL_RECORD_SCHEMA,
        "ontology_id": "citylite-search3d-v1",
        "ontology_version": "1.0.0",
        "episode_id": "episode-0001",
        "frame_index": frame_index,
        "timestamp_ns": timestamp_ns,
        "source_capture_receipt_sha256": "b" * 64,
        "source_revision": "a" * 40,
        "labels": [
            {
                "instance_id": "target-0001",
                "class_id": "search_target",
                "partition": "learning_labels",
                "geometry": {
                    "type": "bbox3d",
                    "frame_id": "world",
                    "center_m": [1.0, 2.0, 3.0],
                    "dimensions_m": [1.0, 2.0, 1.5],
                    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "visibility": "visible",
                "visible_fraction": 1.0,
                "uncertainty": {
                    "status": "not_available",
                },
                "annotation": {
                    "source": "simulator_geometry",
                    "status": "reviewed",
                    "annotator_count": 0,
                    "known_error_ids": [],
                },
            }
        ],
    }


class LabelOntologyTests(unittest.TestCase):
    def test_json_schemas_accept_checked_in_ontology_and_record(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema is not installed")
        ontology_schema = json.loads((ROOT / "schemas" / "label_ontology_v1.schema.json").read_text(encoding="utf-8"))
        record_schema = json.loads((ROOT / "schemas" / "label_record_v1.schema.json").read_text(encoding="utf-8"))
        ontology = _ontology()
        record = _record()
        self.assertEqual(list(Draft202012Validator(ontology_schema).iter_errors(ontology)), [])
        self.assertEqual(list(Draft202012Validator(record_schema).iter_errors(record)), [])

    def test_checked_in_ontology_is_valid_and_hash_stable(self) -> None:
        ontology = _ontology()
        self.assertEqual(validate_ontology(ontology), ())
        self.assertEqual(ontology["schema"], LABEL_ONTOLOGY_SCHEMA)
        self.assertEqual(len(ontology_sha256(ontology)), 64)
        self.assertEqual(ontology["distribution"], "development_only")

    def test_valid_record_and_sequence_preserve_identity(self) -> None:
        ontology = _ontology()
        first = _record()
        second = _record(frame_index=1, timestamp_ns=20)
        self.assertEqual(validate_label_record(first, ontology), ())
        self.assertEqual(validate_label_sequence([first, second], ontology), ())

    def test_geometry_and_identity_errors_fail_closed(self) -> None:
        ontology = _ontology()
        malformed = _record()
        malformed["labels"][0]["geometry"]["orientation_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        malformed["labels"][0]["geometry"]["dimensions_m"] = [0.0, 1.0, 1.0]
        codes = {issue.code for issue in validate_label_record(malformed, ontology)}
        self.assertIn("quaternion", codes)
        self.assertIn("positive_vector", codes)
        second = _record(frame_index=1, timestamp_ns=20)
        second["labels"][0]["class_id"] = "city_landmark"
        identity_codes = {issue.code for issue in validate_label_sequence([_record(), second], ontology)}
        self.assertIn("instance_class_change", identity_codes)

    def test_private_partition_and_nested_truth_are_rejected(self) -> None:
        ontology = _ontology()
        private = _record()
        private["labels"][0]["partition"] = "evaluator_private"
        private["labels"][0]["annotation"]["source"] = "evaluator_private"
        private["private_target_coordinates"] = [1, 2, 3]
        codes = {issue.code for issue in validate_label_record(private, ontology)}
        self.assertIn("partition", codes)
        self.assertIn("private_key", codes)
        self.assertIn("private_value", codes)

    def test_frame_order_and_duplicates_are_rejected(self) -> None:
        ontology = _ontology()
        first = _record(frame_index=0, timestamp_ns=20)
        duplicate = copy.deepcopy(first)
        duplicate["frame_index"] = 0
        duplicate["timestamp_ns"] = 10
        codes = {issue.code for issue in validate_label_sequence([first, duplicate], ontology)}
        self.assertIn("duplicate_frame", codes)
        self.assertIn("timestamp_order", codes)
        duplicate_label = copy.deepcopy(first)
        duplicate_label["labels"].append(copy.deepcopy(duplicate_label["labels"][0]))
        label_codes = {issue.code for issue in validate_label_record(duplicate_label, ontology)}
        self.assertIn("duplicate_instance", label_codes)

    def test_uncertainty_contract_rejects_invalid_confidence(self) -> None:
        record = _record()
        record["labels"][0]["uncertainty"] = {"status": "estimated", "confidence": 1.5}
        codes = {issue.code for issue in validate_label_record(record, _ontology())}
        self.assertIn("uncertainty_confidence", codes)

    def test_jsonl_cli_is_stream_compatible_and_input_errors_are_path_free(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            records.write_text(json.dumps(_record()) + "\n" + json.dumps(_record(frame_index=1, timestamp_ns=20)) + "\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(ROOT / "config" / "label_ontology.citylite_v1.json"), str(records)]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["record_count"], 2)

            missing = root / "missing.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(ROOT / "config" / "label_ontology.citylite_v1.json"), str(missing)]), 1)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "invalid")
            self.assertNotIn(str(missing), output.getvalue())


if __name__ == "__main__":
    unittest.main()
