"""Build the sole machine-readable readiness status from hash-bound evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.behavioral_distinctness import COHORT_PANEL_AUDIT_SCHEMA
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.ordinary_config import load_ordinary_config

A_GATE_SCHEMA = "org.aerocity.bench.g2-i-a-gate-freeze.v1"
B_GATE_SCHEMA = "org.aerocity.bench.cf2x-b-gate-freeze.v1"
PUBLIC_REPLAY_SCHEMAS = {
    "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
    "org.aerocity.bench.cf2x-l1-fleet-preflight.v5",
}
STATUS_SCHEMA = "org.aerocity.bench.authoritative-readiness-status.v1"
GOVERNANCE_SCHEMA = "org.aerocity.bench.experiment-governance-registry.v1"
SEARCH_METHODS = {"atlas-region-greedy", "atlas-surface-inspector"}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--a-gate", type=Path, required=True)
    parser.add_argument("--legacy-b-gate", type=Path, required=True)
    parser.add_argument("--governance-registry", type=Path, required=True)
    parser.add_argument(
        "--recent-public-replay",
        type=Path,
        action="append",
        required=True,
        help="later replay evidence; repeat for every preserved outcome",
    )
    parser.add_argument("--behavior-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _hashed_report(path: Path, schema: str) -> dict[str, Any]:
    report = read_json(path.resolve())
    if not isinstance(report, dict) or report.get("schema") != schema:
        raise ValueError(f"readiness evidence schema differs: {path}")
    payload = dict(report)
    supplied = str(payload.pop("report_hash", ""))
    if supplied != content_hash(payload):
        raise ValueError(f"readiness evidence report hash differs: {path}")
    return report


def _public_replay(path: Path) -> dict[str, Any]:
    report = read_json(path.resolve())
    if not isinstance(report, dict) or report.get("schema") not in PUBLIC_REPLAY_SCHEMAS:
        raise ValueError(f"readiness public replay schema differs: {path}")
    payload = dict(report)
    supplied = str(payload.pop("public_report_sha256", ""))
    if supplied != content_hash(payload):
        raise ValueError(f"readiness public replay hash differs: {path}")
    if report.get("formal_score_eligible") is not False:
        raise ValueError("readiness accepts only non-formal calibration replay evidence")
    bindings = report.get("input_bindings")
    progress = report.get("policy_progress")
    timing = report.get("planning_timing")
    final = report.get("final")
    route = report.get("route_budget_audit")
    if not all(isinstance(value, dict) for value in (bindings, progress, timing, final, route)):
        raise ValueError("readiness public replay lacks required bound summaries")
    return report


def _source(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "file_sha256": file_hash(path.resolve())}


def _route_summary_hash(report: dict[str, Any]) -> str:
    route = dict(report["route_budget_audit"])
    route.pop("method_id", None)
    return content_hash(route)


def _behavior_state(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "MISSING",
            "context_count": 0,
            "mechanism_groups": [],
            "distinct_search_mechanism_count": 0,
            "source": None,
        }
    audit = _hashed_report(path, COHORT_PANEL_AUDIT_SCHEMA)
    groups = audit.get("mechanism_groups")
    if (
        int(audit.get("context_count", 0)) < 3
        or not isinstance(groups, list)
        or audit.get("requires_stochastic_repeat_adjudication") is True
    ):
        raise ValueError("readiness behavior audit is incomplete or unresolved")
    search_groups = [group for group in groups if set(group) & SEARCH_METHODS]
    return {
        "status": str(audit["status"]),
        "context_count": int(audit["context_count"]),
        "mechanism_groups": groups,
        "distinct_search_mechanism_count": len(search_groups),
        "source": _source(path),
        "report_hash": audit["report_hash"],
    }


def _governance_state(path: Path) -> dict[str, Any]:
    registry = read_json(path.resolve())
    if not isinstance(registry, dict) or registry.get("schema") != GOVERNANCE_SCHEMA:
        raise ValueError("readiness governance registry schema differs")
    payload = dict(registry)
    supplied = str(payload.pop("registry_hash", ""))
    if supplied != content_hash(payload):
        raise ValueError("readiness governance registry hash differs")
    records = registry.get("records")
    if not isinstance(records, list):
        raise ValueError("readiness governance registry lacks records")
    by_kind = {
        str(record.get("kind", "")): record
        for record in records
        if isinstance(record, dict)
    }
    formal = by_kind.get("formal_main_matrix", {})
    training = by_kind.get("learning_method_training", {})
    formal_authorized = (
        formal.get("phase") == "formal"
        and formal.get("status") == "FORMAL"
        and formal.get("task_contract_status") == "formal_frozen"
        and formal.get("result_adaptive_change") == "forbidden"
        and formal.get("formal_score_eligible") is True
    )
    training_authorized = (
        training.get("phase") == "formal"
        and training.get("status") == "FORMAL"
        and training.get("task_contract_status") == "formal_frozen"
        and training.get("result_adaptive_change") == "forbidden"
        and training.get("formal_score_eligible") is True
    )
    return {
        "formal_main_matrix_authorized": formal_authorized,
        "long_training_authorized": training_authorized,
        "formal_main_matrix_blockers": list(formal.get("blocking_conditions", [])),
        "long_training_blockers": list(training.get("blocking_conditions", [])),
        "registry_hash": supplied,
        "source": _source(path),
    }


def build_authoritative_readiness_status(
    *,
    release_config_path: Path,
    a_gate_path: Path,
    legacy_b_gate_path: Path,
    governance_registry_path: Path,
    recent_public_replay_paths: list[Path],
    behavior_audit_path: Path | None = None,
) -> dict[str, Any]:
    if len(recent_public_replay_paths) < 1:
        raise ValueError("readiness needs at least one later public replay")
    config = load_ordinary_config(release_config_path.resolve())
    current_execution_hash = content_hash(config.raw["execution_contract"])
    a_gate = _hashed_report(a_gate_path, A_GATE_SCHEMA)
    legacy_b_gate = _hashed_report(legacy_b_gate_path, B_GATE_SCHEMA)
    replays = [_public_replay(path) for path in recent_public_replay_paths]
    indexed: dict[tuple[str, str], dict[str, tuple[Path, dict[str, Any]]]] = {}
    for path, report in zip(recent_public_replay_paths, replays, strict=True):
        method = str(report.get("method", ""))
        context = (
            str(report["input_bindings"].get("layout_hash", "")),
            str(report["input_bindings"].get("public_episode_sha256", "")),
        )
        if not method or not all(len(value) == 64 for value in context):
            raise ValueError("readiness recent replay lacks method or public context bindings")
        if method in indexed.setdefault(context, {}):
            raise ValueError("readiness recent replay duplicates a method/public context")
        indexed[context][method] = (path, report)
    method_sets = {tuple(sorted(by_method)) for by_method in indexed.values()}
    if len(method_sets) != 1:
        raise ValueError("readiness recent replay panel is incomplete across public contexts")
    methods = next(iter(method_sets))
    contexts = sorted(indexed)

    replay_rows = []
    for path, report in zip(recent_public_replay_paths, replays, strict=True):
        progress = report["policy_progress"]
        timing = report["planning_timing"]
        final = report["final"]
        execution_hash = str(report["input_bindings"].get("execution_contract_hash", ""))
        closed = (
            progress.get("status") == "CALIBRATION_EPISODE_CLOSED"
            and int(timing.get("deadline_miss_tick_count", -1)) == 0
            and final.get("safe_completion") is True
            and final.get("all_returned_home") is True
        )
        replay_rows.append(
            {
                "method_id": str(report["method"]),
                "public_context": [
                    str(report["input_bindings"].get("layout_hash", "")),
                    str(report["input_bindings"].get("public_episode_sha256", "")),
                ],
                "execution_contract_hash": execution_hash,
                "matches_current_execution_contract": execution_hash
                == current_execution_hash,
                "calibration_episode_closed": closed,
                "deadline_miss_tick_count": int(
                    timing.get("deadline_miss_tick_count", -1)
                ),
                "confirmation_receipt_count": int(
                    progress.get("confirmation_receipt_count", 0)
                ),
                "safe_completion": final.get("safe_completion") is True,
                "all_returned_home": final.get("all_returned_home") is True,
                "route_summary_hash": _route_summary_hash(report),
                "source": _source(path),
            }
        )
    route_groups_by_context = []
    for context in contexts:
        route_groups: dict[str, list[str]] = {}
        for row in replay_rows:
            if tuple(row["public_context"]) == context:
                route_groups.setdefault(str(row["route_summary_hash"]), []).append(
                    str(row["method_id"])
                )
        groups = sorted(sorted(group) for group in route_groups.values())
        route_groups_by_context.append(
            {"public_context": list(context), "route_summary_groups": groups}
        )
    behavior = _behavior_state(behavior_audit_path)
    governance = _governance_state(governance_registry_path)
    current_closed_ancestors = set()
    for context in contexts:
        rows = [row for row in replay_rows if tuple(row["public_context"]) == context]
        if all(
            row["matches_current_execution_contract"]
            and row["calibration_episode_closed"]
            for row in rows
        ):
            current_closed_ancestors.add(context[0])
    current_search_successes = {
        str(row["method_id"])
        for row in replay_rows
        if row["method_id"] in SEARCH_METHODS
        and row["matches_current_execution_contract"]
        and row["calibration_episode_closed"]
        and int(row["confirmation_receipt_count"]) > 0
    }
    successful_search_groups = [
        group
        for group in behavior["mechanism_groups"]
        if set(group) & current_search_successes
    ]
    checks = {
        "a_gate_binds_current_execution_contract": a_gate["frozen_contract"].get(
            "execution_contract_hash"
        )
        == current_execution_hash,
        "current_contract_has_three_closed_l1_ancestors": len(current_closed_ancestors)
        >= 3,
        "three_context_behavior_audit_complete": behavior["context_count"] >= 3,
        "two_distinct_public_search_mechanisms": behavior[
            "distinct_search_mechanism_count"
        ]
        >= 2,
        "two_current_public_search_mechanisms_closed_and_nonzero": len(
            successful_search_groups
        )
        >= 2,
        "governance_authorizes_formal_main_matrix": governance[
            "formal_main_matrix_authorized"
        ],
    }
    blockers = []
    if not checks["a_gate_binds_current_execution_contract"]:
        blockers.append(
            "The verified A gate freezes an older execution contract; current cadence changes "
            "require method-independent recalibration before B-gate replay."
        )
    if not checks["current_contract_has_three_closed_l1_ancestors"]:
        blockers.append(
            "The current execution contract lacks three independent, closed four-CF2X L1 "
            "calibration ancestors."
        )
    if not checks["three_context_behavior_audit_complete"]:
        blockers.append(
            "No complete three-context public-action behavior audit is bound to the current "
            "method panel."
        )
    if not checks["two_distinct_public_search_mechanisms"]:
        blockers.append(
            "Two behaviorally distinct public search mechanisms have not been established."
        )
    if not checks["two_current_public_search_mechanisms_closed_and_nonzero"]:
        blockers.append(
            "Two distinct public search mechanisms have not both closed with nonzero anonymous "
            "confirmation under the current execution contract."
        )
    if not checks["governance_authorizes_formal_main_matrix"]:
        blockers.extend(str(value) for value in governance["formal_main_matrix_blockers"])
    formal_ready = all(checks.values())
    report: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "status": "FORMAL_READY" if formal_ready else "FORMAL_NO_GO",
        "formal_ready": formal_ready,
        "formal_test_access_authorized": formal_ready,
        "long_training_authorized": bool(
            formal_ready and governance["long_training_authorized"]
        ),
        "current_contract": {
            "release_version": config.version,
            "execution_contract_hash": current_execution_hash,
            "planning_cadence": config.raw["execution_contract"].get("planning"),
            "source": _source(release_config_path),
        },
        "a_gate": {
            "status": a_gate["status"],
            "frozen_execution_contract_hash": a_gate["frozen_contract"][
                "execution_contract_hash"
            ],
            "matches_current_execution_contract": checks[
                "a_gate_binds_current_execution_contract"
            ],
            "source": _source(a_gate_path),
        },
        "legacy_b_gate": {
            "status": legacy_b_gate["status"],
            "historical_replay_count": int(legacy_b_gate.get("replay_count", 0)),
            "superseded_for_current_readiness": True,
            "reason": (
                "It predates the current planning cadence and newer preserved failure evidence."
            ),
            "source": _source(legacy_b_gate_path),
        },
        "later_replay_panel": {
            "public_context_count": len(contexts),
            "method_ids": list(methods),
            "records": replay_rows,
            "route_summary_groups_by_context": route_groups_by_context,
            "route_summary_equivalence_is_not_general_behavior_equivalence": True,
            "preserved_incomplete_replay_count": sum(
                not bool(row["calibration_episode_closed"]) for row in replay_rows
            ),
        },
        "behavior_audit": behavior,
        "governance": governance,
        "checks": checks,
        "blockers": blockers,
        "evidence_precedence": [
            "current versioned release configuration",
            "later hash-bound replay outcomes including failures",
            "historical v16 calibration report",
        ],
        "failure_count": sum(not value for value in checks.values()),
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    report = build_authoritative_readiness_status(
        release_config_path=args.release_config,
        a_gate_path=args.a_gate,
        legacy_b_gate_path=args.legacy_b_gate,
        governance_registry_path=args.governance_registry,
        recent_public_replay_paths=args.recent_public_replay,
        behavior_audit_path=args.behavior_audit,
    )
    write_json(args.output.resolve(), report)
    return 0 if report["formal_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
