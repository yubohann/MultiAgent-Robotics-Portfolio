"""Verify the development-only public CF2X calibration replay panel for gate B."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.cf2x_fleet_preflight_contract import (
    COMPLETE_CALIBRATION_PURPOSE,
    validate_fleet_preflight_reports,
)
from aerocity_bench.cf2x_l0_pairing_contract import validate_l0_pairing_header
from aerocity_bench.fidelity_audit import FIDELITY_REPORT_SCHEMA, compare_l0_l1_rankings
from aerocity_bench.host_guard import HOST_GUARD_SCHEMA, validate_host_guard_pass_receipt

A_GATE_SCHEMA = "org.aerocity.bench.g2-i-a-gate-freeze.v1"
B_GATE_MANIFEST_SCHEMA = "org.aerocity.bench.cf2x-b-gate-manifest.v1"
B_GATE_MANIFEST_SCHEMA_V2 = "org.aerocity.bench.cf2x-b-gate-manifest.v2"
B_GATE_REPORT_SCHEMA = "org.aerocity.bench.cf2x-b-gate-freeze.v1"
B_GATE_REPORT_SCHEMA_V2 = "org.aerocity.bench.cf2x-b-gate-freeze.v2"
RANKING_METHODS = (
    "sweep-3d",
    "atlas-surface-inspector",
    "atlas-region-greedy",
)
SEARCH_METHODS = ("atlas-surface-inspector", "atlas-region-greedy")
RETRY_POLICY_SCHEMA = "org.aerocity.bench.infrastructure-censoring-policy.v1"
CENSORED_ATTEMPT_SCHEMA = "org.aerocity.bench.infrastructure-censored-attempt.v1"
ALLOWED_CENSORING_TRIGGERS = (
    "foreign_runtime",
    "foreign_runtime_during_attempt",
    "residual_runtime",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_DIRECTORY = re.compile(r"^attempt-([0-9]{3})$")


def _current_evidence_pipeline_bindings() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    paths = {
        "manifest_builder_source_sha256": repository
        / "tools"
        / "build_cf2x_b_gate_manifest.py",
        "replay_runner_source_sha256": repository / "tools" / "run_cf2x_b_gate_replays.py",
        "fleet_preflight_source_sha256": repository
        / "tools"
        / "cf2x_l1_fleet_preflight.py",
        "final_verifier_source_sha256": Path(__file__).resolve(),
        "host_guard_source_sha256": repository / "src" / "aerocity_bench" / "host_guard.py",
        "behavior_audit_source_sha256": repository
        / "src"
        / "aerocity_bench"
        / "behavioral_distinctness.py",
    }
    return {field: file_hash(path) for field, path in paths.items()}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--l0-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _validated_hashed_report(path: Path, schema: str | tuple[str, ...]) -> dict[str, Any]:
    report = read_json(path.resolve())
    schemas = (schema,) if isinstance(schema, str) else schema
    if not isinstance(report, dict) or report.get("schema") not in schemas:
        raise ValueError(f"evidence schema differs: {path}")
    supplied_hash = str(report.get("report_hash", ""))
    payload = dict(report)
    payload.pop("report_hash", None)
    if content_hash(payload) != supplied_hash:
        raise ValueError(f"evidence report hash mismatch: {path}")
    return report


def _relative_evidence_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("B-gate replay evidence path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("B-gate replay evidence paths must stay relative to the manifest")
    return (root / relative).resolve()


def _representative_panel_contract(
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], bool]:
    """Validate the candidate-to-representative method mapping for B-gate replay."""

    if manifest.get("schema") == B_GATE_MANIFEST_SCHEMA:
        methods = tuple(str(value) for value in manifest.get("method_ids", ()))
        if methods != RANKING_METHODS:
            raise ValueError("legacy B-gate manifest method plan differs")
        return methods, tuple((method,) for method in methods), False
    if manifest.get("schema") != B_GATE_MANIFEST_SCHEMA_V2:
        raise ValueError("B-gate manifest schema differs")
    preflight = manifest.get("behavior_preflight")
    required = {
        "schema",
        "audit_report_hash",
        "audit_file_sha256",
        "context_count",
        "candidate_method_ids",
        "mechanism_groups",
        "l1_representative_method_ids",
        "excluded_redundant_method_ids",
        "candidate_methods_are_not_deleted",
        "redundant_methods_do_not_count_as_independent_mechanisms",
        "binding_hash",
    }
    if not isinstance(preflight, dict) or set(preflight) != required:
        raise ValueError("B-gate v2 behavior preflight fields differ")
    payload = dict(preflight)
    supplied_hash = str(payload.pop("binding_hash", ""))
    groups_value = preflight.get("mechanism_groups")
    if not isinstance(groups_value, list) or any(
        not isinstance(group, list) or not group for group in groups_value
    ):
        raise ValueError("B-gate v2 behavior mechanism groups are invalid")
    groups = tuple(tuple(str(method) for method in group) for group in groups_value)
    representatives = tuple(
        str(value) for value in preflight.get("l1_representative_method_ids", ())
    )
    excluded = sorted(
        method for group in groups for method in group if method != group[0]
    )
    manifest_methods = tuple(str(value) for value in manifest.get("method_ids", ()))
    if (
        preflight.get("schema")
        != "org.aerocity.bench.cf2x-behavior-preflight-binding.v1"
        or supplied_hash != content_hash(payload)
        or not _SHA256.fullmatch(str(preflight.get("audit_report_hash", "")))
        or not _SHA256.fullmatch(str(preflight.get("audit_file_sha256", "")))
        or not isinstance(preflight.get("context_count"), int)
        or isinstance(preflight.get("context_count"), bool)
        or int(preflight["context_count"]) < 3
        or tuple(preflight.get("candidate_method_ids", ())) != RANKING_METHODS
        or tuple(sorted(groups)) != groups
        or any(tuple(sorted(group)) != group for group in groups)
        or sorted(method for group in groups for method in group)
        != sorted(RANKING_METHODS)
        or representatives != tuple(group[0] for group in groups)
        or manifest_methods != representatives
        or preflight.get("excluded_redundant_method_ids") != excluded
        or preflight.get("candidate_methods_are_not_deleted") is not True
        or preflight.get("redundant_methods_do_not_count_as_independent_mechanisms")
        is not True
    ):
        raise ValueError("B-gate v2 behavior preflight binding is invalid")
    return representatives, groups, True


def _validated_censoring_policy(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    policy = manifest.get("infrastructure_censoring_policy")
    if policy is None:
        return None
    published_bindings = manifest.get("evidence_pipeline_bindings")
    current_bindings = _current_evidence_pipeline_bindings()
    replay_binding_fields = set(current_bindings) - {"final_verifier_source_sha256"}
    if (
        not isinstance(published_bindings, dict)
        or set(published_bindings) != set(current_bindings)
        or any(
            published_bindings[field] != current_bindings[field]
            for field in replay_binding_fields
        )
    ):
        raise ValueError("B-gate evidence pipeline differs from its precommitted source hashes")
    required_fields = {
        "schema",
        "selection_rule",
        "decision_reads_method_outcome",
        "max_attempts_per_pair",
        "retry_archive_root",
        "allowed_host_guard_triggers",
        "required_quiescence_s",
        "maximum_quiescence_wait_s",
        "method_or_safety_failure_retry_allowed",
        "preserve_every_censored_attempt",
    }
    if not isinstance(policy, dict) or set(policy) != required_fields:
        raise ValueError("B-gate infrastructure censoring policy fields differ")
    allowed = tuple(policy["allowed_host_guard_triggers"])
    maximum_attempts = policy["max_attempts_per_pair"]
    if (
        policy["schema"] != RETRY_POLICY_SCHEMA
        or policy["selection_rule"] != "first-host-isolated-complete-attempt"
        or policy["decision_reads_method_outcome"] is not False
        or policy["method_or_safety_failure_retry_allowed"] is not False
        or policy["preserve_every_censored_attempt"] is not True
        or allowed != ALLOWED_CENSORING_TRIGGERS
        or isinstance(maximum_attempts, bool)
        or int(maximum_attempts) <= 1
        or float(policy["required_quiescence_s"]) < 0.0
        or float(policy["maximum_quiescence_wait_s"])
        < float(policy["required_quiescence_s"])
    ):
        raise ValueError("B-gate infrastructure censoring policy is invalid")
    root = manifest_path.resolve().parent
    archive_root = _relative_evidence_path(root, policy["retry_archive_root"])
    replay_root = _relative_evidence_path(root, manifest.get("replay_root"))
    runtime_root = _relative_evidence_path(root, manifest.get("runtime_root"))
    if archive_root in {replay_root, runtime_root}:
        raise ValueError("B-gate censoring archive must be isolated")
    return {
        **policy,
        "allowed_host_guard_triggers": allowed,
        "max_attempts_per_pair": int(maximum_attempts),
        "archive_root": archive_root,
    }


def _load_censored_attempt_records(
    manifest_path: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    policy = _validated_censoring_policy(manifest_path, manifest)
    if policy is None:
        return []
    archive_root = Path(policy["archive_root"])
    if not archive_root.exists():
        return []
    if not archive_root.is_dir() or archive_root.is_symlink():
        raise ValueError("B-gate censoring archive root is invalid")
    representative_methods, _groups, _is_v2 = _representative_panel_contract(manifest)
    ancestors = tuple(str(value) for value in manifest.get("layout_ancestors", ()))
    pairs = {
        f"{record['layout_ancestor']}__{record['method_id']}": (
            str(record["layout_ancestor"]),
            str(record["method_id"]),
        )
        for record in manifest.get("records", ())
        if isinstance(record, dict)
    }
    if len(pairs) != len(ancestors) * len(representative_methods):
        raise ValueError("B-gate censoring archive cannot bind an incomplete replay panel")
    normalized: list[dict[str, Any]] = []
    for pair_root in sorted(archive_root.iterdir()):
        if pair_root.name not in pairs or not pair_root.is_dir() or pair_root.is_symlink():
            raise ValueError(f"unexpected B-gate censoring archive entry: {pair_root}")
        ancestor, method = pairs[pair_root.name]
        attempt_roots = sorted(pair_root.iterdir())
        if any(not path.is_dir() or path.is_symlink() for path in attempt_roots):
            raise ValueError(f"invalid censored-attempt entry below {pair_root}")
        numbers: list[int] = []
        for attempt_root in attempt_roots:
            match = _ATTEMPT_DIRECTORY.fullmatch(attempt_root.name)
            if match is None:
                raise ValueError(f"invalid censored-attempt directory name: {attempt_root}")
            numbers.append(int(match.group(1)))
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"censored attempts are not a complete sequence below {pair_root}")
        if len(numbers) >= int(policy["max_attempts_per_pair"]):
            raise ValueError(
                "a clean replay cannot follow an exhausted infrastructure attempt budget"
            )
        for attempt_number, attempt_root in zip(numbers, attempt_roots, strict=True):
            ledger_path = attempt_root / "attempt.json"
            ledger = read_json(ledger_path)
            required_ledger_fields = {
                "schema",
                "layout_ancestor",
                "method_id",
                "attempt_number",
                "evidence_binding",
                "host_guard_trigger",
                "decision_reads_method_outcome",
                "retry_authorized",
                "artifacts",
                "report_hash",
            }
            if not isinstance(ledger, dict) or set(ledger) != required_ledger_fields:
                raise ValueError(f"censored-attempt ledger fields differ: {ledger_path}")
            payload = dict(ledger)
            supplied_hash = str(payload.pop("report_hash"))
            expected_binding = content_hash(
                {
                    "manifest_report_hash": manifest["report_hash"],
                    "layout_ancestor": ancestor,
                    "method_id": method,
                }
            )
            artifacts = ledger["artifacts"]
            if (
                ledger["schema"] != CENSORED_ATTEMPT_SCHEMA
                or ledger["layout_ancestor"] != ancestor
                or ledger["method_id"] != method
                or ledger["attempt_number"] != attempt_number
                or ledger["evidence_binding"] != expected_binding
                or ledger["host_guard_trigger"] not in ALLOWED_CENSORING_TRIGGERS
                or ledger["decision_reads_method_outcome"] is not False
                or ledger["retry_authorized"] is not True
                or not isinstance(artifacts, dict)
                or not artifacts
                or content_hash(payload) != supplied_hash
            ):
                raise ValueError(f"censored-attempt ledger is invalid: {ledger_path}")
            expected_files = {"attempt.json"}
            artifact_hashes: dict[str, str] = {}
            for relative_value, expected_hash in artifacts.items():
                if not isinstance(relative_value, str) or not isinstance(expected_hash, str):
                    raise ValueError(f"censored-attempt artifact map is invalid: {ledger_path}")
                relative = Path(relative_value)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not _SHA256.fullmatch(expected_hash)
                ):
                    raise ValueError(
                        f"censored-attempt artifact path/hash is invalid: {ledger_path}"
                    )
                candidate = attempt_root / relative
                relative_parents = [
                    attempt_root.joinpath(*relative.parts[:index])
                    for index in range(1, len(relative.parts))
                ]
                if candidate.is_symlink() or any(path.is_symlink() for path in relative_parents):
                    raise ValueError(f"censored-attempt artifact uses a link: {candidate}")
                artifact = candidate.resolve()
                try:
                    artifact.relative_to(attempt_root.resolve())
                except ValueError as error:
                    raise ValueError(
                        f"censored-attempt artifact escapes its archive: {candidate}"
                    ) from error
                if (
                    not artifact.is_file()
                    or file_hash(artifact) != expected_hash
                ):
                    raise ValueError(f"censored-attempt artifact differs: {artifact}")
                expected_files.add(relative.as_posix())
                artifact_hashes[relative.as_posix()] = expected_hash
            actual_files = {
                path.relative_to(attempt_root).as_posix()
                for path in attempt_root.rglob("*")
                if path.is_file()
            }
            if actual_files != expected_files:
                raise ValueError(f"censored-attempt archive has unbound files: {attempt_root}")
            host_guard_path = attempt_root / "runtime" / "host_guard.json"
            host_guard = read_json(host_guard_path)
            if (
                not isinstance(host_guard, dict)
                or host_guard.get("schema") != HOST_GUARD_SCHEMA
                or host_guard.get("status") != "FAIL"
                or host_guard.get("trigger") != ledger["host_guard_trigger"]
                or host_guard.get("evidence_binding") != expected_binding
            ):
                raise ValueError(f"censored host-guard receipt is invalid: {host_guard_path}")
            normalized.append(
                {
                    "layout_ancestor": ancestor,
                    "method_id": method,
                    "attempt_number": attempt_number,
                    "evidence_binding": expected_binding,
                    "host_guard_trigger": str(ledger["host_guard_trigger"]),
                    "retry_authorized": True,
                    "attempt_report_hash": supplied_hash,
                    "attempt_file_sha256": file_hash(ledger_path),
                    "host_guard_report_file_sha256": file_hash(host_guard_path),
                    "artifact_hashes": artifact_hashes,
                    "host_guard_validated": True,
                }
            )
    return normalized


def _load_replay_records(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _validated_hashed_report(
        manifest_path, (B_GATE_MANIFEST_SCHEMA, B_GATE_MANIFEST_SCHEMA_V2)
    )
    if manifest.get("formal_score_eligible") is not False:
        raise ValueError("B-gate manifest must remain development-only")
    _representative_panel_contract(manifest)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("B-gate manifest records must be an array")
    root = manifest_path.resolve().parent
    runtime_root_value = manifest.get("runtime_root")
    if not isinstance(runtime_root_value, str) or not runtime_root_value:
        raise ValueError("B-gate manifest runtime root is missing")
    runtime_root = _relative_evidence_path(root, runtime_root_value)
    normalized: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "layout_ancestor",
            "method_id",
            "public_report",
            "private_report",
        }:
            raise ValueError("B-gate manifest replay fields differ")
        public_path = _relative_evidence_path(root, record["public_report"])
        private_path = _relative_evidence_path(root, record["private_report"])
        ancestor = str(record["layout_ancestor"])
        method = str(record["method_id"])
        host_guard_path = runtime_root / f"{ancestor}__{method}" / "host_guard.json"
        if (
            public_path in seen_paths
            or private_path in seen_paths
            or host_guard_path in seen_paths
            or len({public_path, private_path, host_guard_path}) != 3
        ):
            raise ValueError("B-gate manifest reuses an evidence path")
        seen_paths.update((public_path, private_path, host_guard_path))
        validation = validate_fleet_preflight_reports(public_path, private_path)
        attempt_binding = content_hash(
            {
                "manifest_report_hash": manifest["report_hash"],
                "layout_ancestor": ancestor,
                "method_id": method,
            }
        )
        validate_host_guard_pass_receipt(
            host_guard_path,
            expected_evidence_binding=attempt_binding,
        )
        public = read_json(public_path)
        if not isinstance(public, dict):
            raise ValueError("B-gate public replay report must be an object")
        normalized.append(
            {
                "layout_ancestor": ancestor,
                "method_id": method,
                "public_report_file_sha256": validation["public_report_file_sha256"],
                "private_report_file_sha256": validation["private_report_file_sha256"],
                "host_guard_report_file_sha256": file_hash(host_guard_path),
                "host_guard_passed": True,
                "formal_score_eligible": public.get("formal_score_eligible"),
                "execution_purpose": public.get("execution_purpose"),
                "complete_calibration_replay": public.get("complete_calibration_replay"),
                "method": public.get("method"),
                "input_bindings": public.get("input_bindings"),
                "policy_progress": public.get("policy_progress"),
                "planning_timing": public.get("planning_timing"),
                "execution": public.get("execution"),
                "final": public.get("final"),
            }
        )
    return manifest, normalized


def _valid_l0_records(records: object) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("B-gate L0 records must be a JSON array")
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "method_id",
            "layout_ancestor",
            "score",
            "execution_level",
            "evidence_hash",
        }:
            raise ValueError("B-gate L0 record fields differ")
        if record.get("execution_level") != "L0":
            raise ValueError("B-gate L0 record has the wrong execution level")
        if (
            not isinstance(record.get("method_id"), str)
            or not record["method_id"]
            or not isinstance(record.get("layout_ancestor"), str)
            or not record["layout_ancestor"]
            or not isinstance(record.get("evidence_hash"), str)
            or not _SHA256.fullmatch(record["evidence_hash"])
        ):
            raise ValueError("B-gate L0 record identifiers are invalid")
        if not math.isfinite(float(record["score"])):
            raise ValueError("B-gate L0 score must be finite")
        normalized.append(dict(record))
    return normalized


def _load_l0_pairing_records(
    pairing_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Accept only the signed L0 pairing report bound to this replay manifest."""

    records = validate_l0_pairing_header(
        read_json(pairing_path.resolve()),
        manifest_report_hash=str(manifest["report_hash"]),
        manifest_file_sha256=file_hash(manifest_path.resolve()),
        method_ids=tuple(manifest["method_ids"]),
        layout_ancestors=tuple(manifest["layout_ancestors"]),
        expected_input_bindings=dict(manifest["expected_input_bindings"]),
    )
    return _valid_l0_records(
        [
            {
                "method_id": record["method_id"],
                "layout_ancestor": record["layout_ancestor"],
                "score": record["score"],
                "execution_level": record["execution_level"],
                "evidence_hash": record["evidence_hash"],
            }
            for record in records
        ]
    )


def verify_b_gate(
    *,
    a_gate: dict[str, Any],
    manifest: dict[str, Any],
    replay_records: list[dict[str, Any]],
    l0_records: list[dict[str, Any]],
    censored_attempt_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate a complete replay panel without imposing a rank-correlation outcome."""

    l0_records = _valid_l0_records(l0_records)
    required_replay_fields = {
        "layout_ancestor",
        "method_id",
        "public_report_file_sha256",
        "private_report_file_sha256",
        "host_guard_report_file_sha256",
        "host_guard_passed",
        "formal_score_eligible",
        "execution_purpose",
        "complete_calibration_replay",
        "method",
        "input_bindings",
        "policy_progress",
        "planning_timing",
        "execution",
        "final",
    }
    if any(set(record) != required_replay_fields for record in replay_records):
        raise ValueError("B-gate normalized replay fields differ")
    representative_methods, mechanism_groups, is_v2 = _representative_panel_contract(
        manifest
    )
    methods = sorted({str(record["method_id"]) for record in replay_records})
    ancestors = sorted({str(record["layout_ancestor"]) for record in replay_records})
    pairs = [
        (str(record["method_id"]), str(record["layout_ancestor"])) for record in replay_records
    ]
    complete_matrix = (
        tuple(methods) == tuple(sorted(representative_methods))
        and len(ancestors) >= 3
        and len(pairs) == len(set(pairs)) == len(methods) * len(ancestors)
        and set(pairs) == {(method, ancestor) for method in methods for ancestor in ancestors}
    )
    planned_methods = tuple(manifest.get("method_ids", ()))
    planned_ancestors = tuple(manifest.get("layout_ancestors", ()))
    precommitted_plan = (
        manifest.get("precommitted_before_replays") is True
        and manifest.get("selection_policy") == "sorted-ancestor-even-quantiles-v1"
        and manifest.get("a_gate_report_hash") == a_gate.get("report_hash")
        and planned_methods == representative_methods
        and planned_ancestors == tuple(ancestors)
    )

    censoring_policy = manifest.get("infrastructure_censoring_policy")
    required_censored_fields = {
        "layout_ancestor",
        "method_id",
        "attempt_number",
        "evidence_binding",
        "host_guard_trigger",
        "retry_authorized",
        "attempt_report_hash",
        "attempt_file_sha256",
        "host_guard_report_file_sha256",
        "artifact_hashes",
        "host_guard_validated",
    }
    censoring_records = censored_attempt_records or []
    censoring_records_valid = censored_attempt_records is not None
    if censoring_policy is None:
        censoring_records_valid = not censoring_records
    elif not isinstance(censoring_policy, dict):
        censoring_records_valid = False
    else:
        maximum_attempts = int(censoring_policy.get("max_attempts_per_pair", 0))
        by_pair: dict[tuple[str, str], list[int]] = {}
        for record in censoring_records:
            if set(record) != required_censored_fields:
                censoring_records_valid = False
                continue
            pair = (str(record["layout_ancestor"]), str(record["method_id"]))
            expected_binding = content_hash(
                {
                    "manifest_report_hash": manifest.get("report_hash"),
                    "layout_ancestor": pair[0],
                    "method_id": pair[1],
                }
            )
            artifact_hashes = record["artifact_hashes"]
            if (
                pair[0] not in planned_ancestors
                or pair[1] not in planned_methods
                or record["evidence_binding"] != expected_binding
                or record["host_guard_trigger"] not in ALLOWED_CENSORING_TRIGGERS
                or record["retry_authorized"] is not True
                or record["host_guard_validated"] is not True
                or not isinstance(record["attempt_number"], int)
                or record["attempt_number"] <= 0
                or not isinstance(artifact_hashes, dict)
                or not artifact_hashes
                or any(
                    not isinstance(value, str) or not _SHA256.fullmatch(value)
                    for value in (
                        record["attempt_report_hash"],
                        record["attempt_file_sha256"],
                        record["host_guard_report_file_sha256"],
                        *artifact_hashes.values(),
                    )
                )
            ):
                censoring_records_valid = False
            by_pair.setdefault(pair, []).append(int(record["attempt_number"]))
        for numbers in by_pair.values():
            ordered = sorted(numbers)
            if (
                ordered != list(range(1, len(ordered) + 1))
                or len(ordered) >= maximum_attempts
            ):
                censoring_records_valid = False

    replay_closed = all(
        record["formal_score_eligible"] is False
        and record["execution_purpose"] == COMPLETE_CALIBRATION_PURPOSE
        and record["complete_calibration_replay"] is True
        and record["method"] == record["method_id"]
        and isinstance(record["policy_progress"], dict)
        and record["policy_progress"].get("status") == "CALIBRATION_EPISODE_CLOSED"
        and isinstance(record["planning_timing"], dict)
        and record["planning_timing"].get("deadline_miss_tick_count") == 0
        and isinstance(record["execution"], dict)
        and record["execution"].get("failure_record_count") == 0
        and isinstance(record["final"], dict)
        and record["final"].get("safe_completion") is True
        and record["final"].get("all_returned_home") is True
        for record in replay_records
    )

    global_binding_fields = {
        "release_config_sha256",
        "execution_contract_hash",
        "cf2x_usd_sha256",
        "cf2x_schema_sha256",
        "dynamics_spec_hash",
        "controller_spec_hash",
        "baseline_source_sha256",
        "geometry_source_sha256",
    }
    bindings = [record.get("input_bindings") for record in replay_records]
    bindings_valid = all(
        isinstance(binding, dict)
        and all(
            isinstance(binding.get(field), str) and _SHA256.fullmatch(binding[field])
            for field in global_binding_fields
        )
        for binding in bindings
    )
    global_bindings_frozen = bindings_valid and all(
        len({binding[field] for binding in bindings}) == 1  # type: ignore[index]
        for field in global_binding_fields
    )
    expected_input_bindings = manifest.get("expected_input_bindings")
    required_expected_binding_fields = {
        "baseline_source_sha256",
        "geometry_source_sha256",
        "controller_spec_hash",
        "cf2x_usd_sha256",
        "release_config_sha256",
    }
    reports_match_precommitted_bindings = (
        isinstance(expected_input_bindings, dict)
        and set(expected_input_bindings) == required_expected_binding_fields
        and all(
            isinstance(expected_input_bindings[field], str)
            and _SHA256.fullmatch(expected_input_bindings[field])
            for field in required_expected_binding_fields
        )
        and bindings_valid
        and all(
            binding[field] == expected_input_bindings[field]  # type: ignore[index]
            for binding in bindings
            for field in required_expected_binding_fields
        )
    )

    ancestor_binding_fields = {
        "layout_hash",
        "stage_sha256",
        "cityspec_sha256",
        "task_spec_sha256",
        "task_spec_hash",
        "public_episode_sha256",
        "mission_sector_hash",
        "atlas_hash",
    }
    ancestor_bindings_frozen = bindings_valid
    layout_hashes: dict[str, str] = {}
    if ancestor_bindings_frozen:
        for ancestor in ancestors:
            rows = [
                record["input_bindings"]
                for record in replay_records
                if record["layout_ancestor"] == ancestor
            ]
            if not rows or any(
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(field), str) or not _SHA256.fullmatch(row[field])
                    for field in ancestor_binding_fields
                )
                for row in rows
            ):
                ancestor_bindings_frozen = False
                break
            if any(len({row[field] for row in rows}) != 1 for field in ancestor_binding_fields):
                ancestor_bindings_frozen = False
                break
            layout_hashes[ancestor] = str(rows[0]["layout_hash"])
    independent_layouts = (
        ancestor_bindings_frozen
        and len(layout_hashes) == len(ancestors)
        and len(set(layout_hashes.values())) == len(ancestors)
    )

    l1_records = [
        {
            "method_id": str(record["method_id"]),
            "layout_ancestor": str(record["layout_ancestor"]),
            "score": float(record["policy_progress"]["confirmation_receipt_count"]),
            "execution_level": "L1",
            "evidence_hash": str(record["public_report_file_sha256"]),
        }
        for record in replay_records
        if isinstance(record["policy_progress"], dict)
    ]
    l0_pairs = [(str(record["method_id"]), str(record["layout_ancestor"])) for record in l0_records]
    l0_l1_pairing_complete = (
        complete_matrix
        and len(l0_pairs) == len(set(l0_pairs)) == len(pairs)
        and set(l0_pairs) == set(pairs)
    )
    if l0_l1_pairing_complete and len(methods) >= 2:
        fidelity = compare_l0_l1_rankings(l0_records, l1_records)
    else:
        fidelity = {
            "schema": FIDELITY_REPORT_SCHEMA,
            "formal_score_eligible": False,
            "status": "INSUFFICIENT_DATA",
            "method_count": len(methods),
            "independent_ancestor_count": len(ancestors),
            "reason": "L0 and L1 do not form one complete paired method/ancestor matrix",
            "contract_freeze_allowed": False,
        }
        fidelity["report_hash"] = content_hash(fidelity)
    l0_by_pair = {
        (str(record["method_id"]), str(record["layout_ancestor"])): float(record["score"])
        for record in l0_records
    }
    l1_by_pair = {
        (str(record["method_id"]), str(record["layout_ancestor"])): float(record["score"])
        for record in l1_records
    }
    l1_positive_pairs = {pair for pair, score in l1_by_pair.items() if score > 0.0}
    false_negative_pairs = sorted(
        pair for pair in l1_positive_pairs if l0_by_pair.get(pair, 0.0) <= 0.0
    )
    false_negative_rate = (
        len(false_negative_pairs) / len(l1_positive_pairs) if l1_positive_pairs else 0.0
    )
    search_mechanism_groups = [
        group for group in mechanism_groups if set(group) & set(SEARCH_METHODS)
    ]
    search_mechanisms_nonzero = all(
        any(
            l1_by_pair.get((group[0], ancestor), 0.0) > 0.0
            for ancestor in ancestors
        )
        for group in search_mechanism_groups
    )
    independent_search_mechanisms = len(search_mechanism_groups) >= 2
    report_hashes_valid = all(
        isinstance(record["public_report_file_sha256"], str)
        and _SHA256.fullmatch(record["public_report_file_sha256"])
        and isinstance(record["private_report_file_sha256"], str)
        and _SHA256.fullmatch(record["private_report_file_sha256"])
        and isinstance(record["host_guard_report_file_sha256"], str)
        and _SHA256.fullmatch(record["host_guard_report_file_sha256"])
        for record in replay_records
    )
    matrix_check = (
        "complete_representative_method_by_three_ancestor_matrix"
        if is_v2
        else "complete_three_method_by_three_ancestor_matrix"
    )
    search_check = (
        "two_distinct_search_mechanisms_have_nonzero_l1_signal"
        if is_v2
        else "two_search_methods_have_nonzero_l1_signal"
    )
    checks = {
        "a_gate_verified": a_gate.get("schema") == A_GATE_SCHEMA
        and a_gate.get("status") == "VERIFIED"
        and a_gate.get("authorizes_next_gate") is True,
        matrix_check: complete_matrix,
        "precommitted_method_and_ancestor_plan": precommitted_plan,
        "all_public_replays_closed_and_safe": replay_closed,
        "global_cf2x_controller_and_contract_bindings_frozen": global_bindings_frozen,
        "all_replays_match_precommitted_input_bindings": (
            reports_match_precommitted_bindings
        ),
        "paired_ancestor_inputs_frozen": ancestor_bindings_frozen,
        "three_independent_layout_ancestors": independent_layouts,
        search_check: independent_search_mechanisms and search_mechanisms_nonzero,
        "l0_l1_pairing_and_rank_audit_complete": l0_l1_pairing_complete
        and fidelity.get("status") == "MEASURED_NOT_FROZEN",
        "all_report_file_hashes_valid": report_hashes_valid,
        "all_host_guard_receipts_pass_and_match_precommitted_attempts": all(
            record["host_guard_passed"] is True for record in replay_records
        ),
        "all_infrastructure_censoring_attempts_preserved_and_bound": (
            censoring_records_valid
        ),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": B_GATE_REPORT_SCHEMA_V2 if is_v2 else B_GATE_REPORT_SCHEMA,
        "gate": "B_PUBLIC_CF2X_CALIBRATION_EQUIVALENCE",
        "status": "VERIFIED" if passed else "NO_GO",
        "formal_score_eligible": False,
        "authorizes_formal_test_access": False,
        "authorizes_next_gate": passed,
        "checks": checks,
        "method_ids": methods,
        "candidate_method_ids": list(RANKING_METHODS),
        "mechanism_groups": [list(group) for group in mechanism_groups],
        "distinct_search_mechanism_count": len(search_mechanism_groups),
        "layout_ancestors": ancestors,
        "replay_count": len(replay_records),
        "fidelity_audit": fidelity,
        "l0_screening": {
            "false_negative_pair_count": len(false_negative_pairs),
            "l1_positive_pair_count": len(l1_positive_pairs),
            "false_negative_rate": round(false_negative_rate, 6),
            "false_negative_pairs": [list(pair) for pair in false_negative_pairs],
            "screening_authorized": passed and not false_negative_pairs,
            "ranking_disagreement_does_not_invalidate_l1_scores": True,
        },
        "infrastructure_censoring": {
            "status": (
                "VERIFIED"
                if censoring_policy is not None and censoring_records_valid
                else "NOT_CONFIGURED"
                if censoring_policy is None and censoring_records_valid
                else "INVALID"
            ),
            "policy_configured": censoring_policy is not None,
            "selection_rule": (
                censoring_policy.get("selection_rule")
                if isinstance(censoring_policy, dict)
                else None
            ),
            "censored_attempt_count": len(censoring_records),
            "clean_attempt_count": len(replay_records),
            "total_attempt_denominator": len(censoring_records) + len(replay_records),
            "counts_by_trigger": {
                trigger: sum(
                    record["host_guard_trigger"] == trigger for record in censoring_records
                )
                for trigger in ALLOWED_CENSORING_TRIGGERS
            },
            "attempt_report_hashes": sorted(
                str(record["attempt_report_hash"]) for record in censoring_records
            ),
            "attempt_file_sha256": sorted(
                str(record["attempt_file_sha256"]) for record in censoring_records
            ),
            "host_guard_receipt_sha256": sorted(
                str(record["host_guard_report_file_sha256"])
                for record in censoring_records
            ),
            "method_outcomes_were_not_read_for_retry": (
                isinstance(censoring_policy, dict)
                and censoring_policy.get("decision_reads_method_outcome") is False
                and censoring_policy.get("method_or_safety_failure_retry_allowed") is False
            )
            if censoring_policy is not None
            else True,
        },
        "input_report_hashes": {
            "a_gate": a_gate.get("report_hash"),
            "manifest": manifest.get("report_hash"),
            "public_replays": sorted(
                str(record["public_report_file_sha256"]) for record in replay_records
            ),
            "private_replays": sorted(
                str(record["private_report_file_sha256"]) for record in replay_records
            ),
            "host_guard_receipts": sorted(
                str(record["host_guard_report_file_sha256"]) for record in replay_records
            ),
            "l0_record_set": content_hash(l0_records),
            "censored_attempt_set": content_hash(censoring_records),
        },
        "failure_count": sum(not value for value in checks.values()),
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite B-gate evidence: {args.output}")
    a_gate = _validated_hashed_report(args.a_gate, A_GATE_SCHEMA)
    manifest, replay_records = _load_replay_records(args.manifest)
    censored_attempt_records = _load_censored_attempt_records(args.manifest, manifest)
    l0_records = _load_l0_pairing_records(args.l0_records, args.manifest, manifest)
    report = verify_b_gate(
        a_gate=a_gate,
        manifest=manifest,
        replay_records=replay_records,
        l0_records=l0_records,
        censored_attempt_records=censored_attempt_records,
    )
    write_json(args.output, report)
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
