from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.ordinary_config import load_ordinary_config
from tools.build_authoritative_readiness_status import (
    A_GATE_SCHEMA,
    B_GATE_SCHEMA,
    STATUS_SCHEMA,
    build_authoritative_readiness_status,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"
GOVERNANCE_PATH = ROOT / "configs" / "experiment-governance-v1.json"


def _write_hashed(path: Path, payload: dict[str, object]) -> Path:
    payload["report_hash"] = content_hash(payload)
    write_json(path, payload)
    return path


def _a_gate(tmp_path: Path, execution_hash: str) -> Path:
    return _write_hashed(
        tmp_path / "a-gate.json",
        {
            "schema": A_GATE_SCHEMA,
            "status": "VERIFIED",
            "frozen_contract": {"execution_contract_hash": execution_hash},
        },
    )


def _b_gate(tmp_path: Path) -> Path:
    return _write_hashed(
        tmp_path / "legacy-b-gate.json",
        {
            "schema": B_GATE_SCHEMA,
            "status": "VERIFIED",
            "replay_count": 9,
        },
    )


def _public_replay(
    tmp_path: Path,
    *,
    method: str,
    execution_hash: str,
    deadline_misses: int,
    confirmations: int,
    route_points: int,
    context_index: int = 1,
) -> Path:
    closed = deadline_misses == 0
    payload: dict[str, object] = {
        "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight.v4",
        "formal_score_eligible": False,
        "method": method,
        "input_bindings": {
            "layout_hash": f"{context_index:064x}",
            "public_episode_sha256": f"{context_index + 100:064x}",
            "execution_contract_hash": execution_hash,
        },
        "policy_progress": {
            "status": (
                "CALIBRATION_EPISODE_CLOSED"
                if closed
                else "CALIBRATION_EPISODE_INCOMPLETE"
            ),
            "confirmation_receipt_count": confirmations,
        },
        "planning_timing": {"deadline_miss_tick_count": deadline_misses},
        "final": {"safe_completion": True, "all_returned_home": True},
        "route_budget_audit": {
            "method_id": method,
            "route_point_count": route_points,
            "status": "LOWER_BOUND_FITS",
        },
    }
    payload["public_report_sha256"] = content_hash(payload)
    path = tmp_path / f"{method}-{context_index}.public.json"
    write_json(path, payload)
    return path


def test_authoritative_status_supersedes_success_with_current_contract_and_failures(
    tmp_path: Path,
) -> None:
    current_hash = content_hash(load_ordinary_config(CONFIG_PATH).raw["execution_contract"])
    old_hash = "a" * 64
    replay_paths = [
        _public_replay(
            tmp_path,
            method="atlas-region-greedy",
            execution_hash=old_hash,
            deadline_misses=1,
            confirmations=4,
            route_points=67,
        ),
        _public_replay(
            tmp_path,
            method="atlas-surface-inspector",
            execution_hash=old_hash,
            deadline_misses=1,
            confirmations=4,
            route_points=67,
        ),
        _public_replay(
            tmp_path,
            method="sweep-3d",
            execution_hash=old_hash,
            deadline_misses=0,
            confirmations=0,
            route_points=4,
        ),
    ]
    report = build_authoritative_readiness_status(
        release_config_path=CONFIG_PATH,
        a_gate_path=_a_gate(tmp_path, old_hash),
        legacy_b_gate_path=_b_gate(tmp_path),
        governance_registry_path=GOVERNANCE_PATH,
        recent_public_replay_paths=replay_paths,
    )
    assert report["schema"] == STATUS_SCHEMA
    assert report["status"] == "FORMAL_NO_GO"
    assert report["formal_ready"] is False
    assert report["current_contract"]["execution_contract_hash"] == current_hash
    assert report["a_gate"]["matches_current_execution_contract"] is False
    assert report["legacy_b_gate"]["superseded_for_current_readiness"] is True
    assert report["behavior_audit"]["status"] == "MISSING"
    assert report["later_replay_panel"]["route_summary_groups_by_context"] == [
        {
            "public_context": [f"{1:064x}", f"{101:064x}"],
            "route_summary_groups": [
                ["atlas-region-greedy", "atlas-surface-inspector"],
                ["sweep-3d"],
            ],
        }
    ]
    assert sum(
        not row["calibration_episode_closed"]
        for row in report["later_replay_panel"]["records"]
    ) == 2
    assert report["report_hash"] == content_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )


def test_authoritative_status_rejects_tampered_replay_hash(tmp_path: Path) -> None:
    current_hash = content_hash(load_ordinary_config(CONFIG_PATH).raw["execution_contract"])
    replay = _public_replay(
        tmp_path,
        method="sweep-3d",
        execution_hash=current_hash,
        deadline_misses=0,
        confirmations=0,
        route_points=4,
    )
    payload = json.loads(replay.read_text(encoding="utf-8"))
    payload["policy_progress"]["confirmation_receipt_count"] = 99
    write_json(replay, payload)
    with pytest.raises(ValueError, match="public replay hash differs"):
        build_authoritative_readiness_status(
            release_config_path=CONFIG_PATH,
            a_gate_path=_a_gate(tmp_path, current_hash),
            legacy_b_gate_path=_b_gate(tmp_path),
            governance_registry_path=GOVERNANCE_PATH,
            recent_public_replay_paths=[replay],
        )


def test_authoritative_status_accepts_complete_multi_context_panel_without_requiring_failures(
    tmp_path: Path,
) -> None:
    current_hash = content_hash(load_ordinary_config(CONFIG_PATH).raw["execution_contract"])
    replay_paths = []
    for context_index in range(1, 4):
        for method, route_points in (
            ("atlas-region-greedy", 60 + context_index),
            ("atlas-surface-inspector", 70 + context_index),
        ):
            replay_paths.append(
                _public_replay(
                    tmp_path,
                    method=method,
                    execution_hash=current_hash,
                    deadline_misses=0,
                    confirmations=1,
                    route_points=route_points,
                    context_index=context_index,
                )
            )
    behavior = _write_hashed(
        tmp_path / "behavior.json",
        {
            "schema": "org.aerocity.bench.method-panel-behavior-cohort-audit.v2",
            "status": "PASS",
            "context_count": 3,
            "mechanism_groups": [
                ["atlas-region-greedy"],
                ["atlas-surface-inspector"],
            ],
            "requires_stochastic_repeat_adjudication": False,
        },
    )
    report = build_authoritative_readiness_status(
        release_config_path=CONFIG_PATH,
        a_gate_path=_a_gate(tmp_path, current_hash),
        legacy_b_gate_path=_b_gate(tmp_path),
        governance_registry_path=GOVERNANCE_PATH,
        recent_public_replay_paths=replay_paths,
        behavior_audit_path=behavior,
    )
    assert report["checks"]["current_contract_has_three_closed_l1_ancestors"] is True
    assert report["checks"]["two_distinct_public_search_mechanisms"] is True
    assert (
        report["checks"]["two_current_public_search_mechanisms_closed_and_nonzero"]
        is True
    )
    assert report["later_replay_panel"]["preserved_incomplete_replay_count"] == 0
    assert report["status"] == "FORMAL_NO_GO"
    assert report["checks"]["governance_authorizes_formal_main_matrix"] is False


def test_checked_in_authoritative_status_is_hash_bound_to_current_sources() -> None:
    status_path = ROOT / "configs" / "authoritative-readiness-status-v1.json"
    status = read_json(status_path)
    governance = read_json(GOVERNANCE_PATH)
    current_hash = content_hash(load_ordinary_config(CONFIG_PATH).raw["execution_contract"])
    assert status["report_hash"] == content_hash(
        {key: value for key, value in status.items() if key != "report_hash"}
    )
    assert status["status"] == "FORMAL_NO_GO"
    assert status["formal_test_access_authorized"] is False
    assert status["long_training_authorized"] is False
    assert status["current_contract"]["execution_contract_hash"] == current_hash
    assert status["current_contract"]["source"]["file_sha256"] == file_hash(CONFIG_PATH)
    assert status["governance"]["registry_hash"] == governance["registry_hash"]
    assert status["governance"]["source"]["file_sha256"] == file_hash(GOVERNANCE_PATH)
