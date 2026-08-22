"""Fail-closed governance audit for every AeroCityBench experiment class.

The benchmark has legitimate engineering iterations: a controller bug, an
invalid public artifact, or a host failure must be fixed and rerun.  They are
not legitimate task-design iterations.  In particular, no result may be used
to change the public information boundary, workload, task budget, scoring, or
formal exclusion rule after that result has been observed.

This module makes that distinction machine-checkable for the experiment
registry.  It intentionally reports ``FORMAL_NO_GO`` until a separate formal
release process supplies a frozen, independently reviewed registry.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json

REGISTRY_SCHEMA = "org.aerocity.bench.experiment-governance-registry.v1"
REPORT_SCHEMA = "org.aerocity.bench.experiment-governance-audit.v1"
EXTERNAL_EVIDENCE_MANIFEST_SCHEMA = (
    "org.aerocity.bench.experiment-governance-external-evidence-manifest.v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_KINDS = frozenset(
    {
        "g1u_legacy_diagnostic",
        "g2i_task_calibration",
        "g2i_contract_ablation",
        "l1_engineering_fixture",
        "l1_public_replay",
        "external_method_bridge",
        "scene_and_license_validation",
        "release_reproducibility",
        "formal_main_matrix",
        "learning_method_training",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "id",
        "kind",
        "phase",
        "evidence_scope",
        "status",
        "public_input_boundary",
        "task_contract_status",
        "result_adaptive_change",
        "policy_information_source",
        "formal_score_eligible",
        "evidence_paths",
        "blocking_conditions",
    }
)
_PHASES = frozenset({"retired", "engineering", "calibration", "formal"})
_STATUSES = frozenset(
    {
        "RETIRED",
        "ENGINEERING_ONLY",
        "CALIBRATION_ONLY",
        "DEVELOPMENT_ONLY",
        "BLOCKED",
        "NOT_STARTED",
        "FORMAL",
    }
)
_BOUNDARIES = frozenset({"enforced", "not_applicable", "legacy_failed"})
_CONTRACT_STATES = frozenset(
    {
        "obsolete",
        "mutable_development",
        "freeze_candidate",
        "frozen_calibration_only",
        "formal_frozen",
    }
)
_RESULT_ADAPTATION = frozenset(
    {
        "method_independent_only",
        "engineering_defect_only",
        "forbidden",
        "not_applicable",
    }
)
_POLICY_SOURCES = frozenset(
    {"target_agnostic_public", "private_fixture", "not_run", "not_applicable"}
)
_CURRENT_CF2X_REPLAY_ID = "cf2x-public-three-ancestor-v16-current-boundary"
_CURRENT_CF2X_EVIDENCE = frozenset(
    {
        "reason/benchmark-external-methodology-audit-20260802/"
        "current-boundary-l1-panel-20260803/current-boundary-source-manifest-v2.json",
        "reason/benchmark-external-methodology-audit-20260802/"
        "current-boundary-l1-panel-20260803/cf2x-b-gate-manifest-v16-current-boundary.json",
        "reason/benchmark-external-methodology-audit-20260802/"
        "current-boundary-l1-panel-20260803/l0-pairing-v16-current-boundary.json",
        "reason/benchmark-external-methodology-audit-20260802/"
        "current-boundary-l1-panel-20260803/cf2x-b-gate-verification-v16-current-boundary.json",
    }
)


def _normalized_evidence_path(value: object) -> str:
    candidate = Path(str(value))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("evidence path must be relative to the repository root")
    return candidate.as_posix()


def _current_git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot resolve repository commit: {type(error).__name__}") from error
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or not _GIT_COMMIT.fullmatch(commit):
        raise ValueError("cannot resolve repository commit")
    return commit


@dataclass(frozen=True)
class _EvidenceResolver:
    """Resolve registered evidence without allowing an external path escape."""

    repository_root: Path
    external_root: Path | None = None
    external_hashes: dict[str, str] | None = None
    manifest_hash: str | None = None
    source_commit: str | None = None

    def path_for(self, value: object) -> Path:
        relative = _normalized_evidence_path(value)
        local = (self.repository_root / relative).resolve()
        if self.repository_root == local or self.repository_root in local.parents:
            if local.is_file():
                return local
        if self.external_root is None or self.external_hashes is None:
            raise FileNotFoundError(f"registered evidence does not exist: {relative}")
        expected = self.external_hashes.get(relative)
        if expected is None:
            raise FileNotFoundError(
                f"registered evidence is absent locally and not listed externally: {relative}"
            )
        external = (self.external_root / relative).resolve()
        if self.external_root != external and self.external_root not in external.parents:
            raise ValueError("external evidence path escapes the declared evidence root")
        if not external.is_file():
            raise FileNotFoundError(f"external evidence does not exist: {relative}")
        if file_hash(external) != expected:
            raise ValueError(f"external evidence hash differs: {relative}")
        return external


def _registered_evidence_paths(registry: object) -> frozenset[str]:
    if not isinstance(registry, dict) or not isinstance(registry.get("records"), list):
        raise ValueError("experiment governance registry has no records")
    paths: set[str] = set()
    for record in registry["records"]:
        if not isinstance(record, dict):
            continue
        evidence = record.get("evidence_paths")
        if not isinstance(evidence, list):
            continue
        for value in evidence:
            paths.add(_normalized_evidence_path(value))
    return frozenset(paths)


def _external_manifest_payload(
    *,
    repository_root: Path,
    registry_path: Path,
    registry: object,
    evidence_root: Path,
) -> dict[str, object]:
    root = repository_root.resolve()
    source_registry = registry_path.resolve()
    if source_registry.parent != (root / "configs").resolve() or source_registry.name != (
        "experiment-governance-v1.json"
    ):
        raise ValueError("external evidence requires configs/experiment-governance-v1.json")
    if not source_registry.is_file():
        raise FileNotFoundError("experiment governance registry does not exist")
    external_root = evidence_root.resolve()
    if not external_root.is_dir():
        raise FileNotFoundError("external evidence root does not exist")
    paths = _registered_evidence_paths(registry)
    evidence: list[dict[str, str]] = []
    for relative in sorted(paths):
        candidate = (external_root / relative).resolve()
        if external_root != candidate and external_root not in candidate.parents:
            raise ValueError("external evidence path escapes the declared evidence root")
        if not candidate.is_file():
            raise FileNotFoundError(f"external evidence does not exist: {relative}")
        evidence.append({"path": relative, "sha256": file_hash(candidate)})
    return {
        "schema": EXTERNAL_EVIDENCE_MANIFEST_SCHEMA,
        "source_commit": _current_git_commit(root),
        "registry_path": "configs/experiment-governance-v1.json",
        # Git worktrees may normalize line endings differently. Bind the JSON
        # document canonically so a byte-only CRLF conversion cannot invalidate
        # an otherwise identical release configuration.
        "registry_file_sha256": content_hash(read_json(source_registry)),
        "registry_content_hash": (
            registry.get("registry_hash") if isinstance(registry, dict) else None
        ),
        "evidence": evidence,
    }


def build_external_evidence_manifest(
    *,
    repository_root: Path,
    registry_path: Path,
    registry: object,
    evidence_root: Path,
) -> dict[str, object]:
    """Return a source-bound manifest for small, uncommitted evidence receipts.

    The manifest never records an absolute evidence-root path.  A clean checkout
    must opt in to both this manifest and an evidence root, and every fallback
    file is checked against its registered relative path and content hash.
    """

    manifest = _external_manifest_payload(
        repository_root=repository_root,
        registry_path=registry_path,
        registry=registry,
        evidence_root=evidence_root,
    )
    manifest["manifest_hash"] = content_hash(manifest)
    return manifest


def _load_external_evidence_resolver(
    *,
    repository_root: Path,
    registry_path: Path,
    registry: object,
    evidence_root: Path,
    manifest_path: Path,
) -> _EvidenceResolver:
    manifest = read_json(manifest_path)
    required = {
        "schema",
        "source_commit",
        "registry_path",
        "registry_file_sha256",
        "registry_content_hash",
        "evidence",
        "manifest_hash",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("external evidence manifest fields differ")
    if manifest.get("schema") != EXTERNAL_EVIDENCE_MANIFEST_SCHEMA:
        raise ValueError("external evidence manifest schema differs")
    declared_hash = manifest.get("manifest_hash")
    if not isinstance(declared_hash, str) or declared_hash != content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise ValueError("external evidence manifest hash mismatch")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not _GIT_COMMIT.fullmatch(source_commit):
        raise ValueError("external evidence manifest source commit is invalid")
    if source_commit != _current_git_commit(repository_root):
        raise ValueError("external evidence manifest source commit differs")
    if manifest.get("registry_path") != "configs/experiment-governance-v1.json":
        raise ValueError("external evidence manifest registry path differs")
    actual_registry = registry_path.resolve()
    expected_registry = (
        repository_root.resolve() / "configs" / "experiment-governance-v1.json"
    ).resolve()
    if actual_registry != expected_registry or not actual_registry.is_file():
        raise ValueError("external evidence registry path differs")
    registry_file_hash = manifest.get("registry_file_sha256")
    if not isinstance(registry_file_hash, str) or not _SHA256.fullmatch(registry_file_hash):
        raise ValueError("external evidence manifest registry file hash is invalid")
    if registry_file_hash != content_hash(read_json(actual_registry)):
        raise ValueError("external evidence manifest registry file hash differs")
    if manifest.get("registry_content_hash") != registry.get("registry_hash"):
        raise ValueError("external evidence manifest registry content hash differs")

    expected_paths = _registered_evidence_paths(registry)
    raw_evidence = manifest.get("evidence")
    if not isinstance(raw_evidence, list):
        raise ValueError("external evidence manifest evidence must be a list")
    hashes: dict[str, str] = {}
    for item in raw_evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("external evidence manifest entry fields differ")
        relative = _normalized_evidence_path(item.get("path"))
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("external evidence manifest entry hash is invalid")
        if relative in hashes:
            raise ValueError("external evidence manifest contains duplicate paths")
        hashes[relative] = digest
    if frozenset(hashes) != expected_paths:
        raise ValueError("external evidence manifest paths differ from the registry")

    external_root = evidence_root.resolve()
    if not external_root.is_dir():
        raise FileNotFoundError("external evidence root does not exist")
    # Validate the entire bundle up front. A file that is not needed by the
    # current check must not silently become mutable just because it is absent.
    for relative, expected_hash in hashes.items():
        candidate = (external_root / relative).resolve()
        if external_root != candidate and external_root not in candidate.parents:
            raise ValueError("external evidence path escapes the declared evidence root")
        if not candidate.is_file():
            raise FileNotFoundError(f"external evidence does not exist: {relative}")
        if file_hash(candidate) != expected_hash:
            raise ValueError(f"external evidence hash differs: {relative}")
    return _EvidenceResolver(
        repository_root=repository_root.resolve(),
        external_root=external_root,
        external_hashes=hashes,
        manifest_hash=declared_hash,
        source_commit=source_commit,
    )


def _report_hash_matches(report: object) -> bool:
    if not isinstance(report, dict) or not isinstance(report.get("report_hash"), str):
        return False
    return report["report_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )


def _manifest_hash_matches(manifest: object) -> bool:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("manifest_hash"), str):
        return False
    return manifest["manifest_hash"] == content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )


def _historical_v16_calibration_evidence_is_verified(
    record: object, resolver: _EvidenceResolver
) -> bool:
    """Bind the superseded v16 calibration history to its public summaries.

    Private replays remain local and are intentionally not read here.  The verifier,
    manifest, source manifest, and paired L0 summary are sufficient to detect an
    accidental path substitution or a rewritten historical calibration record.
    """

    if not isinstance(record, dict):
        return False
    paths = record.get("evidence_paths")
    if not isinstance(paths, list) or frozenset(map(str, paths)) != _CURRENT_CF2X_EVIDENCE:
        return False
    try:
        source_path = resolver.path_for(next(
            path for path in paths if path.endswith("source-manifest-v2.json")
        ))
        manifest_path = resolver.path_for(next(
            path
            for path in paths
            if path.endswith("cf2x-b-gate-manifest-v16-current-boundary.json")
        ))
        pairing_path = resolver.path_for(next(
            path for path in paths if path.endswith("l0-pairing-v16-current-boundary.json")
        ))
        verification_path = resolver.path_for(next(
            path
            for path in paths
            if path.endswith("cf2x-b-gate-verification-v16-current-boundary.json")
        ))
        source = read_json(source_path)
        manifest = read_json(manifest_path)
        pairing = read_json(pairing_path)
        verification = read_json(verification_path)
    except (FileNotFoundError, StopIteration, ValueError):
        return False

    checks = verification.get("checks") if isinstance(verification, dict) else None
    input_hashes = (
        verification.get("input_report_hashes") if isinstance(verification, dict) else None
    )
    fidelity_audit = verification.get("fidelity_audit") if isinstance(verification, dict) else None
    return (
        _report_hash_matches(manifest)
        and _report_hash_matches(pairing)
        and _report_hash_matches(verification)
        and isinstance(source, dict)
        and _manifest_hash_matches(source)
        and source.get("formal_score_eligible") is False
        and source.get("accepted_ancestor_count") == 3
        and isinstance(manifest, dict)
        and manifest.get("formal_score_eligible") is False
        and manifest.get("purpose") == "development-only-public-four-cf2x-l1-calibration"
        and isinstance(pairing, dict)
        and pairing.get("status") == "VERIFIED_L0_PAIRING"
        and pairing.get("formal_score_eligible") is False
        and isinstance(verification, dict)
        and verification.get("status") == "VERIFIED"
        and verification.get("failure_count") == 0
        and verification.get("formal_score_eligible") is False
        and verification.get("authorizes_formal_test_access") is False
        and verification.get("layout_ancestors") == manifest.get("layout_ancestors")
        and isinstance(verification.get("method_ids"), list)
        and isinstance(manifest.get("method_ids"), list)
        and set(verification["method_ids"]) == set(manifest["method_ids"])
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
        and isinstance(input_hashes, dict)
        and input_hashes.get("manifest") == manifest.get("report_hash")
        and isinstance(fidelity_audit, dict)
        and fidelity_audit.get("status") == "MEASURED_NOT_FROZEN"
    )


def _record_issues(record: object, resolver: _EvidenceResolver) -> list[str]:
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        return ["record fields differ from the governance schema"]

    issues: list[str] = []
    identifier = record["id"]
    kind = record["kind"]
    phase = record["phase"]
    status = record["status"]
    boundary = record["public_input_boundary"]
    contract = record["task_contract_status"]
    adaptation = record["result_adaptive_change"]
    source = record["policy_information_source"]
    eligible = record["formal_score_eligible"]
    paths = record["evidence_paths"]
    blockers = record["blocking_conditions"]

    if not isinstance(identifier, str) or not identifier:
        issues.append("id must be a non-empty string")
    if kind not in REQUIRED_KINDS:
        issues.append("unknown experiment kind")
    if phase not in _PHASES or status not in _STATUSES:
        issues.append("unknown phase or status")
    if boundary not in _BOUNDARIES or contract not in _CONTRACT_STATES:
        issues.append("unknown boundary or contract status")
    if adaptation not in _RESULT_ADAPTATION or source not in _POLICY_SOURCES:
        issues.append("unknown result-adaptation or policy-source value")
    if not isinstance(eligible, bool):
        issues.append("formal_score_eligible must be boolean")
    if not isinstance(paths, list) or not paths:
        issues.append("evidence_paths must be a non-empty list")
    elif len(paths) != len(set(map(str, paths))):
        issues.append("evidence_paths contains duplicates")
    else:
        for path in paths:
            try:
                resolver.path_for(path)
            except (FileNotFoundError, ValueError) as error:
                issues.append(str(error))
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        issues.append("blocking_conditions must be a list of non-empty strings")

    if boundary == "legacy_failed" and not (
        phase == "retired" and status == "RETIRED" and eligible is False
    ):
        issues.append("a failed legacy public boundary may only be retained as retired evidence")
    if source == "private_fixture" and not (
        phase == "engineering" and status == "ENGINEERING_ONLY" and eligible is False
    ):
        issues.append("private fixtures may only be engineering evidence")
    if kind == "g1u_legacy_diagnostic" and phase != "retired":
        issues.append("G1-U target-search diagnostics cannot remain an active main experiment")
    if kind == "l1_public_replay" and boundary != "enforced" and phase != "retired":
        issues.append("an active public L1 replay requires an enforced public boundary")
    if kind == "formal_main_matrix" and not (
        phase == "formal" and status == "FORMAL" and contract == "formal_frozen"
    ):
        if not (phase == "calibration" and status == "BLOCKED" and eligible is False):
            issues.append("the formal main matrix must be explicitly blocked or formally frozen")
    if kind == "learning_method_training" and status not in {"BLOCKED", "NOT_STARTED", "FORMAL"}:
        issues.append("learning training may not inherit calibration evidence as a result")
    if phase in {"calibration", "formal"} and adaptation not in {
        "method_independent_only",
        "forbidden",
    }:
        issues.append("calibration/formal experiments require a result-independent change rule")
    if eligible and not (
        phase == "formal"
        and status == "FORMAL"
        and boundary == "enforced"
        and contract == "formal_frozen"
        and adaptation == "forbidden"
    ):
        issues.append(
            "formal eligibility requires a frozen formal contract and no result adaptation"
        )
    if not eligible and status == "FORMAL":
        issues.append("formal status cannot coexist with formal_score_eligible=false")
    return issues


def audit_experiment_governance(
    registry: object,
    *,
    repository_root: Path,
    registry_path: Path | None = None,
    external_evidence_root: Path | None = None,
    external_evidence_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate coverage and evidence eligibility without reading private truth.

    A passing audit proves containment: legacy evidence is quarantined and no
    development result is being promoted.  It deliberately does *not* grant
    permission to run the formal matrix.
    """

    root = repository_root.resolve()
    if not isinstance(registry, dict):
        raise ValueError("experiment governance registry must be an object")
    required = {"schema", "registry_version", "records", "registry_hash"}
    if set(registry) != required or registry["schema"] != REGISTRY_SCHEMA:
        raise ValueError("experiment governance registry fields differ")
    unhashed = {key: value for key, value in registry.items() if key != "registry_hash"}
    if registry["registry_hash"] != content_hash(unhashed):
        raise ValueError("experiment governance registry hash mismatch")
    records = registry["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("experiment governance registry has no records")
    if (external_evidence_root is None) != (external_evidence_manifest is None):
        raise ValueError(
            "external evidence requires both an evidence root and an evidence manifest"
        )
    if external_evidence_root is None:
        resolver = _EvidenceResolver(repository_root=root)
        external_evidence_report: dict[str, object] = {"status": "NOT_USED"}
    else:
        if registry_path is None:
            raise ValueError("external evidence requires the registry file path")
        resolver = _load_external_evidence_resolver(
            repository_root=root,
            registry_path=registry_path,
            registry=registry,
            evidence_root=external_evidence_root,
            manifest_path=external_evidence_manifest,
        )
        external_evidence_report = {
            "status": "BOUND_EXTERNAL_RECEIPTS",
            "manifest_hash": resolver.manifest_hash,
            "source_commit": resolver.source_commit,
            "registered_path_count": len(resolver.external_hashes or {}),
        }

    issues: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    kinds: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", "<invalid>")) if isinstance(record, dict) else "<invalid>"
        if record_id in seen_ids:
            issues.append({"id": record_id, "issue": "duplicate experiment id"})
        seen_ids.add(record_id)
        if isinstance(record, dict):
            kinds.add(str(record.get("kind", "")))
        for issue in _record_issues(record, resolver):
            issues.append({"id": record_id, "issue": issue})
        if isinstance(record, dict) and set(record) == _RECORD_FIELDS:
            normalized.append(
                {
                    "id": record["id"],
                    "kind": record["kind"],
                    "phase": record["phase"],
                    "status": record["status"],
                    "formal_score_eligible": record["formal_score_eligible"],
                    "public_input_boundary": record["public_input_boundary"],
                }
            )
    missing_kinds = sorted(REQUIRED_KINDS - kinds)
    if missing_kinds:
        issues.append(
            {"id": "registry", "issue": f"missing experiment coverage: {', '.join(missing_kinds)}"}
        )

    by_kind = {record["kind"]: record for record in records if isinstance(record, dict)}
    by_id = {record["id"]: record for record in records if isinstance(record, dict)}
    legacy_l1_replay = by_id.get("cf2x-public-three-ancestor-v12", {})
    stale_l1_replay = by_id.get("cf2x-public-three-ancestor-v15", {})
    historical_v16_replay = by_id.get(_CURRENT_CF2X_REPLAY_ID, {})
    formal_matrix = by_kind.get("formal_main_matrix", {})
    training = by_kind.get("learning_method_training", {})
    containment_checks = {
        "legacy_g1u_retired": by_kind.get("g1u_legacy_diagnostic", {}).get("status") == "RETIRED",
        "legacy_cf2x_replay_retired": (
            legacy_l1_replay.get("public_input_boundary") == "legacy_failed"
            and legacy_l1_replay.get("status") == "RETIRED"
        ),
        "stale_cf2x_v15_replay_retired": (
            stale_l1_replay.get("public_input_boundary") == "legacy_failed"
            and stale_l1_replay.get("status") == "RETIRED"
            and stale_l1_replay.get("formal_score_eligible") is False
        ),
        "historical_cf2x_v16_replay_superseded": (
            historical_v16_replay.get("phase") == "retired"
            and historical_v16_replay.get("status") == "RETIRED"
            and historical_v16_replay.get("public_input_boundary") == "enforced"
            and historical_v16_replay.get("task_contract_status") == "obsolete"
            and historical_v16_replay.get("result_adaptive_change") == "not_applicable"
            and historical_v16_replay.get("formal_score_eligible") is False
            and _historical_v16_calibration_evidence_is_verified(
                historical_v16_replay, resolver
            )
        ),
        "formal_main_matrix_blocked": formal_matrix.get("status") == "BLOCKED",
        "learning_training_blocked": training.get("status") == "BLOCKED",
        "no_development_record_promoted": not any(
            record.get("formal_score_eligible") is True
            for record in records
            if isinstance(record, dict) and record.get("phase") != "formal"
        ),
    }
    if not all(containment_checks.values()):
        for name, value in containment_checks.items():
            if not value:
                issues.append({"id": "registry", "issue": f"containment check failed: {name}"})

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "registry_hash": registry["registry_hash"],
        "external_evidence": external_evidence_report,
        "formal_score_eligible": False,
        "overall_status": (
            "CONTAINMENT_PASS_FORMAL_NO_GO" if not issues else "CONTAINMENT_FAIL_FORMAL_NO_GO"
        ),
        "covered_experiment_kinds": sorted(kinds),
        "missing_experiment_kinds": missing_kinds,
        "containment_checks": containment_checks,
        "records": sorted(normalized, key=lambda record: (record["kind"], record["id"])),
        "issues": issues,
        "formal_blockers": [
            "G2-I A gate and the v16 public CF2X panel are calibration evidence, "
            "not a formal task contract.",
            "The v12 public CF2X replay used a legacy public artifact and is retired.",
            "The v15 CF2X panel is retired: its historical public artifact exposed "
            "private target-count metadata and its mission-sector schema is stale.",
            "The v16 panel is historical calibration evidence for an older execution-contract "
            "hash; the current planning cadence requires a fresh A gate and B-gate panel.",
            "The locked MARVEL transfer has an L0 smoke and a 12-second L1 process "
            "interface diagnostic only; it is a 2-D transfer, issued no OBSERVE action, "
            "and is not a comparable external G2-I method.",
        "The ancestor-level protocol is implemented, but five calibration ancestors provide "
        "insufficient power for the planned 0.10 recall MDE (29 independent ancestors estimated).",
        "Development-only environment and clean-wheel evidence is source-marked "
        "UNCOMMITTED-DEVELOPMENT; a clean-source release rerun, recovery drill, and "
        "publication closure remain open.",
        ],
        "next_authorized_step": "RERUN_METHOD_INDEPENDENT_A_GATE_FOR_CURRENT_CONTRACT",
    }
    report["report_hash"] = content_hash(report)
    return report


def load_and_audit_experiment_governance(
    registry_path: Path,
    *,
    repository_root: Path,
    external_evidence_root: Path | None = None,
    external_evidence_manifest: Path | None = None,
) -> dict[str, Any]:
    return audit_experiment_governance(
        read_json(registry_path),
        repository_root=repository_root,
        registry_path=registry_path,
        external_evidence_root=external_evidence_root,
        external_evidence_manifest=external_evidence_manifest,
    )
