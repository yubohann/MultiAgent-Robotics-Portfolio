"""Hardware-independent competition safety and protocol fault matrix."""

from __future__ import annotations

from robocon_game_supervisor.protocol import Deduplicator, MessageEnvelope
from robocon_game_supervisor.supervisor import GameSupervisor, SafetySnapshot, SupervisorState


def _fresh_supervisor() -> GameSupervisor:
    supervisor = GameSupervisor()
    supervisor.enter_preflight()
    supervisor.preflight_result(True)
    supervisor.start()
    return supervisor


def _result(name: str, passed: bool, observed: object, expected: object, reason: str = "") -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed.value if isinstance(observed, SupervisorState) else observed,
        "expected": expected.value if isinstance(expected, SupervisorState) else expected,
        "reason": reason,
        "evidence_level": "synthetic",
    }


def run_matrix() -> dict:
    cases: list[dict] = []
    supervisor = _fresh_supervisor()
    decision = supervisor.monitor_safety(SafetySnapshot(False, True, 0.05, True, 0.05, True, True, True))
    cases.append(_result("localization_loss", supervisor.state == SupervisorState.RECOVERY,
                         supervisor.state, SupervisorState.RECOVERY, decision.reason))

    supervisor = _fresh_supervisor()
    decision = supervisor.request_fire_shot(SafetySnapshot(True, False, 0.05, True, 0.05, True, True, True))
    cases.append(_result("map_unlock_blocks_shot", not decision.accepted,
                         decision.reason, "map_locked is false", decision.reason))

    supervisor = _fresh_supervisor()
    decision = supervisor.monitor_safety(SafetySnapshot(True, True, 0.05, True, 0.05, True, True, False), require_teammate=True)
    cases.append(_result("teammate_loss", supervisor.state == SupervisorState.RECOVERY,
                         supervisor.state, SupervisorState.RECOVERY, decision.reason))

    supervisor = _fresh_supervisor()
    decision = supervisor.handle_action_failure("ExecutePass", "synthetic feedback timeout")
    cases.append(_result("action_timeout", supervisor.state == SupervisorState.RECOVERY,
                         supervisor.state, SupervisorState.RECOVERY, decision.reason))

    supervisor = _fresh_supervisor()
    supervisor.emergency_stop("synthetic estop")
    cases.append(_result("emergency_stop_latched", supervisor.state == SupervisorState.ESTOP,
                         supervisor.state, SupervisorState.ESTOP, supervisor.last_failure_reason))

    envelope = MessageEnvelope(1, "heartbeat", "task-1", "robot-a", 1, 100, 200, {})
    dedup = Deduplicator()
    first = dedup.accept(envelope, 150)
    duplicate = dedup.accept(envelope, 150)
    expired = dedup.accept(MessageEnvelope(1, "heartbeat", "task-1", "robot-a", 2, 100, 200, {}), 201)
    cases.append(_result("duplicate_and_expired_team_messages",
                         first and not duplicate and not expired,
                         {"first": first, "duplicate": duplicate, "expired": expired},
                         {"first": True, "duplicate": False, "expired": False},
                         f"duplicates={dedup.duplicate_count}, stale={dedup.stale_count}"))
    return {
        "schema_version": 1,
        "evidence_level": "synthetic",
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "passed_count": sum(case["passed"] for case in cases),
        "case_count": len(cases),
    }
