from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rivermark_benchmark.cohort_audit as cohort_audit_module
from rivermark_benchmark.cohort_audit import (
    COHORT_AUDIT_SCHEMA,
    CohortAuditError,
    build_development_cohort_audit,
    main,
    verify_development_cohort_audit,
)
from rivermark_benchmark.collection_protocol import (
    load_collection_protocol,
    resolve_collection_binding,
)
from rivermark_benchmark.failure_ledger import FailureRecord

PROTOCOL = ROOT / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json"
GATE_CHECKS = {
    "timestamp_audit_passed": True,
    "sensor_phase_trace_verified": True,
    "action_causality_audit_passed": True,
    "pose_closure_audit_passed": True,
    "visual_intrusion_verified": True,
    "onboard_scene_content_verified": True,
    "contact_free": True,
    "runtime_safety_trace_verified": True,
    "trajectory_segment_clearance_verified": True,
    "condition_realization_verified": True,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_payload_sha256(payload: dict[str, object]) -> str:
    normalized = {**payload, "report_payload_sha256": ""}
    encoded = (
        json.dumps(
            normalized,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture(
    root: Path,
    *,
    protocol: dict[str, object],
    cell_id: str,
    episode_index: int,
) -> tuple[Path, FailureRecord]:
    capture = root / f"capture-{cell_id}-{episode_index}"
    capture.mkdir()
    binding = resolve_collection_binding(protocol, cell_id=cell_id, episode_index=episode_index)
    attempt_id = f"attempt-{cell_id}-{episode_index}"
    visibility = {
        "passed": True,
        "target_count": 4,
        "targets_meeting_visibility": 4,
        "failed_target_count": 0,
        "per_target_slot": {
            f"search_target_slot_{index:03d}": {
                "visible_frames": 10 + index + episode_index,
                "max_pixels": 20 + index + episode_index,
            }
            for index in range(4)
        },
    }
    outcome_path = capture / "task_outcome.json"
    _write_json(
        outcome_path,
        {
            "schema": "org.rivermark.t1-target-observability.v1",
            "scoring_status": "not_scored",
            "target_observability": visibility,
        },
    )
    payload_path = capture / "payload.bin"
    payload_path.write_bytes(b"payload" * (episode_index + 1))
    receipt = {
        "schema": "org.rivermark.isaac-swarm-capture.v1",
        "status": "captured",
        "ok": True,
        "source_worktree_dirty": False,
        "source_revision": "2" * 40,
        "source_tree_sha256": ("3" if cell_id.startswith("train") else "4") * 64,
        "capture_attempt_id": attempt_id,
        "collection_binding": binding,
        "created_wall_time_ns": 1_000_000_000,
        "finished_wall_time_ns": 7_000_000_000 + episode_index,
        "capture_storage_budget": {"required_bytes": 1000},
        "resource_telemetry": {
            "maxima": {
                "commit_percent": 40.0 + episode_index,
                "private_commit_bytes": 20_000 + episode_index,
            }
        },
        "artifact_hashes": {
            "payload.bin": {
                "sha256": _sha256(payload_path),
                "bytes": payload_path.stat().st_size,
            },
            "task_outcome.json": {
                "sha256": _sha256(outcome_path),
                "bytes": outcome_path.stat().st_size,
            }
        },
    }
    receipt_path = capture / "capture_receipt.json"
    _write_json(receipt_path, receipt)
    receipt_sha = _sha256(receipt_path)
    (capture / "capture_receipt.sha256").write_text(
        f"{receipt_sha}  capture_receipt.json\n", encoding="ascii"
    )
    checks = {
        **GATE_CHECKS,
        "target_observability": visibility,
        "minimum_agent_max_displacement_m": 11.0 + episode_index / 100,
        "route_witness_tracked_agent_max_displacement_m": 11.4 + episode_index / 100,
    }
    _write_json(
        capture / "independent_validation.json",
        {
            "schema": "org.rivermark.isaac-independent-validation.v1",
            "status": "passed",
            "issues": [],
            "capture_receipt_sha256": receipt_sha,
            "formal_benchmark_admission": False,
            "checks": checks,
        },
    )
    record = FailureRecord(
        attempt_id=attempt_id,
        outcome="quarantined",
        category="quality_failure",
        stage="isaac_capture",
        recorded_at=f"2026-07-27T00:0{episode_index}:00Z",
        split=str(binding["split"]),
        source_capture_sha256=receipt_sha,
        receipt_sha256=receipt_sha,
        reason_code="development_evidence_not_formal",
        collection_protocol_id=str(binding["protocol_id"]),
        collection_protocol_sha256=str(binding["protocol_sha256"]),
        collection_cell_id=str(binding["cell_id"]),
        collection_episode_index=int(binding["episode_index"]),
        episode_seed=int(binding["episode_seed"]),
    )
    return capture, record


def _fixture(tmp_path: Path) -> tuple[list[Path], Path]:
    protocol = dict(load_collection_protocol(PROTOCOL))
    captures: list[Path] = []
    records: list[FailureRecord] = []
    for cell in protocol["cells"]:
        cell_id = str(cell["cell_id"])
        for episode_index in range(1, 5):
            capture, record = _capture(
                tmp_path,
                protocol=protocol,
                cell_id=cell_id,
                episode_index=episode_index,
            )
            captures.append(capture)
            records.append(record)
    extra_cells = [
        "train-citylite-direct-v2",
        "validation-citylite-direct-v2",
        "validation-citylite-direct-v2",
    ]
    for suffix, cell_id in enumerate(extra_cells):
        binding = resolve_collection_binding(protocol, cell_id=cell_id, episode_index=0)
        records.append(
            FailureRecord(
                attempt_id=f"attempt-canary-{suffix}",
                outcome="quarantined",
                category="quality_failure",
                stage="isaac_capture",
                recorded_at=f"2026-07-27T01:0{suffix}:00Z",
                split=str(binding["split"]),
                source_capture_sha256=str(suffix + 5) * 64,
                receipt_sha256=str(suffix + 5) * 64,
                reason_code="development_evidence_not_formal",
                collection_protocol_id=str(binding["protocol_id"]),
                collection_protocol_sha256=str(binding["protocol_sha256"]),
                collection_cell_id=str(binding["cell_id"]),
                collection_episode_index=0,
                episode_seed=int(binding["episode_seed"]),
            )
        )
    ledger = tmp_path / "failure_ledger.jsonl"
    ledger.write_text(
        "".join(json.dumps(record.as_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return captures, ledger


def _append_non_candidate_admissions(ledger: Path) -> None:
    protocol = load_collection_protocol(PROTOCOL)
    records: list[FailureRecord] = []
    for cell in protocol["cells"]:
        cell_id = str(cell["cell_id"])
        for episode_index in range(10, 14):
            binding = resolve_collection_binding(
                protocol,
                cell_id=cell_id,
                episode_index=episode_index,
            )
            suffix = hashlib.sha256(f"{cell_id}:{episode_index}".encode()).hexdigest()
            records.append(
                FailureRecord(
                    attempt_id=f"attempt-admitted-{suffix[:16]}",
                    outcome="admitted",
                    category="none",
                    stage="formal_admission",
                    recorded_at=f"2026-07-27T02:{episode_index:02d}:00Z",
                    split=str(binding["split"]),
                    episode_id=f"episode-{suffix[:16]}",
                    source_capture_sha256=suffix,
                    receipt_sha256=suffix,
                    collection_protocol_id=str(binding["protocol_id"]),
                    collection_protocol_sha256=str(binding["protocol_sha256"]),
                    collection_cell_id=str(binding["cell_id"]),
                    collection_episode_index=int(binding["episode_index"]),
                    episode_seed=int(binding["episode_seed"]),
                )
            )
    with ledger.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")


def _rewrite_ledger_record(ledger: Path, attempt_id: str, **updates: object) -> None:
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["attempt_id"] == attempt_id:
            record.update(updates)
            break
    else:
        raise AssertionError(f"missing fixture ledger record: {attempt_id}")
    ledger.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _refresh_receipt_bindings(capture: Path, ledger: Path) -> None:
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_sha = _sha256(receipt_path)
    (capture / "capture_receipt.sha256").write_text(
        f"{receipt_sha}  capture_receipt.json\n", encoding="ascii"
    )
    validation_path = capture / "independent_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["capture_receipt_sha256"] = receipt_sha
    _write_json(validation_path, validation)
    _rewrite_ledger_record(
        ledger,
        receipt["capture_attempt_id"],
        receipt_sha256=receipt_sha,
        source_capture_sha256=receipt_sha,
    )


def _rebind_outcome(capture: Path, ledger: Path) -> None:
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outcome_path = capture / "task_outcome.json"
    receipt["artifact_hashes"]["task_outcome.json"] = {
        "sha256": _sha256(outcome_path),
        "bytes": outcome_path.stat().st_size,
    }
    _write_json(receipt_path, receipt)
    _refresh_receipt_bindings(capture, ledger)


def test_cohort_audit_binds_eight_candidates_to_eleven_attempts(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    assert report["schema"] == COHORT_AUDIT_SCHEMA
    assert report["status"] == "passed"
    assert report["audit_provenance"] == {
        "analyzer_module": "rivermark_benchmark.cohort_audit",
        "analyzer_source_sha256": _sha256(Path(cohort_audit_module.__file__)),
        "construction_scope": "source_protocol_ledger_and_capture_artifacts",
        "offline_verification_scope": "internal_structure_and_unkeyed_digest_only",
    }
    assert report["formal"] is False
    assert report["aggregate"]["candidate_count"] == 8
    assert report["aggregate"]["candidate_count_by_split"] == {"train": 4, "validation": 4}
    assert report["aggregate"]["target_count"] == 32
    assert report["aggregate"]["targets_meeting_visibility"] == 32
    assert report["protocol"]["target_quota_by_cell"] == {
        "train-citylite-direct-v2": {
            "condition_id": "object-count-4-v1",
            "target_count": 4,
        },
        "validation-citylite-direct-v2": {
            "condition_id": "object-count-4-v1",
            "target_count": 4,
        },
    }
    assert report["aggregate"]["capture_failure_rate"] == 0.0
    assert report["aggregate"]["quarantine_rate"] == 1.0
    assert report["accounting"]["protocol_attempt_count"] == 11
    assert report["accounting"]["candidate_attempt_count"] == 8
    assert report["accounting"]["noncandidate_protocol_attempt_count"] == 3
    assert report["admission_readiness"]["native_evidence_ready_for_review"] is True
    assert report["admission_readiness"]["formal_admission_complete"] is False
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "target_coordinates" not in encoded
    if importlib.util.find_spec("jsonschema") is not None:
        import jsonschema

        schema = json.loads(
            (ROOT / "schemas" / "development_cohort_audit_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator(schema).validate(report)


def test_cohort_audit_rejects_stale_validation_binding(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    validation_path = captures[0] / "independent_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["capture_receipt_sha256"] = "0" * 64
    _write_json(validation_path, validation)
    with pytest.raises(CohortAuditError, match="artifact_hash_binding_passed"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)


def test_cohort_audit_rejects_post_validation_task_outcome_tampering(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    outcome_path = captures[0] / "task_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["scoring_status"] = "tampered-after-validation"
    _write_json(outcome_path, outcome)

    with pytest.raises(CohortAuditError, match="task_outcome.json.*capture receipt"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)


def test_cohort_audit_rejects_post_validation_bound_payload_tampering(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    (captures[0] / "payload.bin").write_bytes(b"tampered-after-validation")

    with pytest.raises(CohortAuditError, match="payload.bin.*capture receipt"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)


def test_cohort_audit_capture_bytes_ignore_unbound_post_validation_files(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    before = build_development_cohort_audit(PROTOCOL, ledger, captures)
    (captures[0] / "post-validation-diagnostic.bin").write_bytes(b"diagnostic" * 100)
    after = build_development_cohort_audit(PROTOCOL, ledger, captures)

    assert after["aggregate"]["capture_bytes"] == before["aggregate"]["capture_bytes"]


def test_cohort_audit_rejects_outcome_validation_visibility_disagreement(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    outcome_path = captures[0] / "task_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["target_observability"]["targets_meeting_visibility"] = 3
    outcome["target_observability"]["failed_target_count"] = 1
    outcome["target_observability"]["passed"] = False
    _write_json(outcome_path, outcome)
    _rebind_outcome(captures[0], ledger)

    with pytest.raises(CohortAuditError, match="target observability.*disagrees"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)


def test_cohort_audit_rejects_failed_candidate_ledger_outcome(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    receipt = json.loads((captures[0] / "capture_receipt.json").read_text(encoding="utf-8"))
    _rewrite_ledger_record(ledger, receipt["capture_attempt_id"], outcome="failed")

    with pytest.raises(CohortAuditError, match="development quarantine"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)


def test_cohort_audit_rejects_duplicate_or_incomplete_binding_set(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    with pytest.raises(CohortAuditError, match="unique attempts and receipts"):
        build_development_cohort_audit(PROTOCOL, ledger, [*captures[:-1], captures[0]])


def test_development_candidates_never_inherit_non_candidate_formal_admissions(
    tmp_path: Path,
) -> None:
    captures, ledger = _fixture(tmp_path)
    _append_non_candidate_admissions(ledger)

    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    assert report["accounting"]["coverage"]["admitted_count"] == 8
    assert report["admission_readiness"]["formal_admission_complete"] is False


def test_cli_refuses_to_overwrite_a_report(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    output = tmp_path / "cohort_audit.json"
    args = [str(PROTOCOL), "--failure-ledger", str(ledger), "--output", str(output)]
    for capture in captures:
        args.extend(["--capture", str(capture)])
    assert main(args) == 0
    assert output.is_file()
    verified = verify_development_cohort_audit(output)
    assert verified["aggregate"]["candidate_count"] == 8
    assert main(["--verify-report", str(output)]) == 0
    assert main(args) == 1

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["aggregate"]["candidate_count"] = 7
    _write_json(output, payload)
    with pytest.raises(CohortAuditError, match="payload digest is stale"):
        verify_development_cohort_audit(output)


def test_verifier_fail_closes_on_resigned_malformed_structure(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    del report["candidates"][0]["binding"]["cell_id"]
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "malformed.json"
    _write_json(output, report)
    with pytest.raises(CohortAuditError, match="candidate binding is malformed"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_empty_quality_gates(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    report["candidates"][0]["quality_gates"] = {}
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "empty-gates.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="quality gates"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_failed_candidate_state(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    report["candidates"][0]["ledger"]["outcome"] = "failed"
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "failed-candidate.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="development quarantine"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_schema_invalid_split(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    report["candidates"][0]["binding"]["split"] = "pilot"
    report["aggregate"]["candidate_count_by_split"] = {
        "pilot": 1,
        "train": 3,
        "validation": 4,
    }
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "invalid-split.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="candidate binding is malformed"):
        verify_development_cohort_audit(output)


@pytest.mark.parametrize(
    "split",
    ["train", "inner_dev", "validation", "blind_test", "ood_test"],
)
def test_verifier_accepts_every_schema_split(tmp_path: Path, split: str) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    cell_id = report["candidates"][0]["binding"]["cell_id"]
    for candidate in report["candidates"]:
        if candidate["binding"]["cell_id"] == cell_id:
            candidate["binding"]["split"] = split
    for cell in report["accounting"]["coverage"]["cells"]:
        if cell["cell_id"] == cell_id:
            cell["split"] = split
    report["aggregate"]["candidate_count_by_split"] = dict(
        sorted(Counter(item["binding"]["split"] for item in report["candidates"]).items())
    )
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / f"valid-{split}.json"
    _write_json(output, report)

    verified = verify_development_cohort_audit(output)
    assert verified["aggregate"]["candidate_count_by_split"] == report["aggregate"][
        "candidate_count_by_split"
    ]


@pytest.mark.parametrize("location", ["report", "candidate"])
def test_verifier_rejects_resigned_schema_forbidden_field(
    tmp_path: Path,
    location: str,
) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    target = report if location == "report" else report["candidates"][0]
    target["unexpected_claim"] = True
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / f"unexpected-{location}.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="structure is malformed"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_false_resource_distribution(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    report["aggregate"]["capture_bytes"] = {
        "count": 8,
        "minimum": 1,
        "median": 1.0,
        "mean": 1.0,
        "maximum": 1,
    }
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "false-capture-byte-distribution.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="aggregate distributions"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_weakened_target_quota(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    for candidate in report["candidates"]:
        candidate["target_visibility"].update(
            {
                "target_count": 1,
                "targets_meeting_visibility": 1,
                "failed_target_count": 0,
            }
        )
    report["aggregate"]["target_count"] = len(report["candidates"])
    report["aggregate"]["targets_meeting_visibility"] = len(report["candidates"])
    report["aggregate"]["target_visibility_rate"] = 1.0
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "weakened-target-quota.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="target quota"):
        verify_development_cohort_audit(output)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("ledger_hash", "ledger accounting"),
        ("ledger_count", "ledger accounting"),
        ("blocking_reasons", "admission-readiness"),
    ],
)
def test_verifier_rejects_resigned_false_accounting_claim(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    if mutation == "ledger_hash":
        report["accounting"]["failure_ledger_sha256"] = "not-a-sha256"
    elif mutation == "ledger_count":
        report["accounting"]["ledger_record_count"] += 1
    else:
        report["admission_readiness"]["blocking_reason_codes"] = []
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / f"false-{mutation}.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match=error):
        verify_development_cohort_audit(output)


@pytest.mark.parametrize("mutation", ["global_counts", "cell_counts"])
def test_verifier_rejects_resigned_inconsistent_coverage(
    tmp_path: Path,
    mutation: str,
) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    coverage = report["accounting"]["coverage"]
    if mutation == "global_counts":
        coverage["quarantined_count"] = 0
        coverage["failed_count"] = coverage["attempt_count"]
        report["aggregate"]["quarantine_rate"] = 0.0
        report["aggregate"]["capture_failure_rate"] = 1.0
    else:
        coverage["cells"][0]["quarantined_count"] = 0
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / f"inconsistent-coverage-{mutation}.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="coverage"):
        verify_development_cohort_audit(output)


def test_verifier_rejects_resigned_subthreshold_route_witness(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    report = build_development_cohort_audit(PROTOCOL, ledger, captures)
    report["candidates"][0]["route_witness_displacement_m"] = 2.99
    report["aggregate"]["route_witness_displacement_m"] = cohort_audit_module._distribution(
        [candidate["route_witness_displacement_m"] for candidate in report["candidates"]]
    )
    report["report_payload_sha256"] = _report_payload_sha256(report)
    output = tmp_path / "subthreshold-route-witness.json"
    _write_json(output, report)

    with pytest.raises(CohortAuditError, match="route_witness"):
        verify_development_cohort_audit(output)


@pytest.mark.skipif(os.name != "nt", reason="case-folding alias is Windows-specific")
def test_cohort_audit_rejects_casefolded_artifact_alias(tmp_path: Path) -> None:
    captures, ledger = _fixture(tmp_path)
    receipt_path = captures[0] / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_hashes"]["PAYLOAD.BIN"] = dict(
        receipt["artifact_hashes"]["payload.bin"]
    )
    _write_json(receipt_path, receipt)
    _refresh_receipt_bindings(captures[0], ledger)

    with pytest.raises(CohortAuditError, match="duplicate artifact path"):
        build_development_cohort_audit(PROTOCOL, ledger, captures)
