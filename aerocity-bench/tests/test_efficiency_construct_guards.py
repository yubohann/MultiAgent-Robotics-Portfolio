from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aerocity_bench.adapters import GymnasiumFleetWrapper, PettingZooParallelWrapper
from aerocity_bench.behavioral_distinctness import (
    BEHAVIOR_SUMMARY_SCHEMA,
    audit_method_panel_behavior,
    audit_method_panel_behavior_cohort,
    summarize_public_action_trace,
)
from aerocity_bench.canonical import (
    content_hash,
    read_json,
    write_json_atomic,
    write_json_atomic_compact,
)
from aerocity_bench.contracts import ActionPacket, ObservationPacket, Pose3D
from aerocity_bench.planning_cadence import (
    PlanningCadenceController,
    rebind_held_action,
    validate_planning_cadence,
)
from tools.audit_g2_i_behavior_panel import behavior_distinct_count


def _action(*, drone_id: str, invocation: int, x: float) -> ActionPacket:
    return ActionPacket(
        episode_id="episode-a",
        drone_id=drone_id,
        sequence=invocation,
        issued_at_s=invocation * 0.2,
        kind="WAYPOINT",
        waypoint=Pose3D(position=(x, 1.0, 2.0), yaw_deg=90.0),
    )


def _summary(xs: tuple[float, ...]) -> dict[str, object]:
    trace = [
        {"uav-00": _action(drone_id="uav-00", invocation=index, x=x)}
        for index, x in enumerate(xs)
    ]
    return summarize_public_action_trace(trace)


def _report(method: str, summaries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "formal_score_eligible": False,
        "method_id": method,
        "layout_hash": "a" * 64,
        "episode_hash": "b" * 64,
        "replicates": [{"public_action_behavior": summary} for summary in summaries],
    }


def test_compact_atomic_json_preserves_value_and_content_hash(tmp_path: Path) -> None:
    value = {
        "execution_receipts": [
            {"drone_id": f"uav-{index:02d}", "values": list(range(30))}
            for index in range(100)
        ]
    }
    pretty = tmp_path / "pretty.json"
    compact = tmp_path / "compact.json"
    write_json_atomic(pretty, value)
    write_json_atomic_compact(compact, value)

    assert read_json(pretty) == read_json(compact) == value
    assert content_hash(read_json(compact)) == content_hash(value)
    assert compact.stat().st_size < pretty.stat().st_size * 0.6

    replacement = {"status": "replacement", "failures": ["kept"]}
    write_json_atomic_compact(compact, replacement)
    assert read_json(compact) == replacement
    assert not list(tmp_path.glob("*.tmp"))


def test_public_action_signature_ignores_binding_identity_but_not_behavior() -> None:
    first = _summary((1.0, 2.0))
    same_semantics = summarize_public_action_trace(
        [
            {
                "uav-00": ActionPacket(
                    episode_id="another-episode",
                    drone_id="uav-00",
                    sequence=10 + index,
                    issued_at_s=50.0 + index,
                    kind="WAYPOINT",
                    waypoint=Pose3D(position=(x, 1.0, 2.0), yaw_deg=90.0),
                )
            }
            for index, x in enumerate((1.0, 2.0))
        ]
    )
    changed_route = _summary((1.0, 3.0))

    assert first["schema"] == BEHAVIOR_SUMMARY_SCHEMA
    assert first["mission_action_semantics_sha256"] == same_semantics[
        "mission_action_semantics_sha256"
    ]
    assert first["mission_action_semantics_sha256"] != changed_route[
        "mission_action_semantics_sha256"
    ]


def test_behavior_panel_flags_equivalence_and_nondeterminism_without_censoring() -> None:
    shared = _summary((1.0, 2.0))
    distinct = _summary((1.0, 3.0))
    audit = audit_method_panel_behavior(
        [_report("method-a", [shared]), _report("method-b", [shared])]
    )
    assert audit["status"] == "REVIEW_EXACT_EQUIVALENCE"
    assert audit["exact_equivalence_groups"] == [["method-a", "method-b"]]
    assert audit["may_count_all_methods_as_behaviorally_distinct"] is False
    assert audit["does_not_delete_or_censor_replays"] is True

    nondeterministic = audit_method_panel_behavior(
        [_report("method-a", [shared, distinct]), _report("method-b", [distinct])]
    )
    assert nondeterministic["status"] == "REVIEW_NONDETERMINISM"
    assert nondeterministic["nondeterministic_methods"] == ["method-a"]

    with pytest.raises(ValueError, match="same layout and episode"):
        other_context = _report("method-b", [distinct])
        other_context["episode_hash"] = "c" * 64
        audit_method_panel_behavior([_report("method-a", [shared]), other_context])


def test_behavior_cohort_deduplicates_only_equivalence_across_every_context() -> None:
    reports = []
    for context_index in range(3):
        shared = _summary((1.0, 2.0 + context_index))
        distinct = _summary((1.0, 8.0 + context_index))
        for method, summary in (
            ("method-a", shared),
            ("method-b", shared),
            ("method-c", distinct),
        ):
            report = _report(method, [summary])
            report["layout_hash"] = f"{context_index + 1:064x}"
            report["episode_hash"] = f"{context_index + 11:064x}"
            reports.append(report)
    audit = audit_method_panel_behavior_cohort(reports)
    assert audit["status"] == "REVIEW_EXACT_EQUIVALENCE_ACROSS_COHORT"
    assert audit["mechanism_groups"] == [
        ["method-a", "method-b"],
        ["method-c"],
    ]
    assert audit["l1_representative_method_ids"] == ["method-a", "method-c"]
    assert audit["excluded_redundant_method_ids"] == ["method-b"]
    assert audit["context_count"] == 3

    incomplete = reports[:-1]
    with pytest.raises(ValueError, match="complete method-by-context"):
        audit_method_panel_behavior_cohort(incomplete)


def test_behavior_audit_cli_count_supports_single_and_cohort_schemas() -> None:
    assert behavior_distinct_count({"distinct_deterministic_behavior_count": 2}) == 2
    assert behavior_distinct_count({"distinct_mechanism_lower_bound": 3}) == 3
    with pytest.raises(ValueError, match="lacks a distinct-count field"):
        behavior_distinct_count({})


@pytest.mark.parametrize("wrapper", [GymnasiumFleetWrapper, PettingZooParallelWrapper])
def test_legacy_rl_wrappers_fail_closed_for_g2_i(wrapper: type[object]) -> None:
    runtime = SimpleNamespace(public_task_spec={"task_track": "G2-I"})
    with pytest.raises(ValueError, match="versioned G2-I training wrapper"):
        wrapper(runtime)  # type: ignore[arg-type,call-arg]


def _observation(*, sequence: int, drone_id: str = "uav-00") -> ObservationPacket:
    return ObservationPacket(
        episode_id="episode-a",
        observation_id=f"observation-{sequence}",
        drone_id=drone_id,
        sequence=sequence,
        timestamp_s=sequence * 0.2,
        pose=Pose3D(position=(0.0, 0.0, 2.0), yaw_deg=0.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=100.0,
    )


def _planning_contract() -> dict[str, object]:
    return {
        "schema": "org.aerocity.bench.planning-cadence.v1",
        "mode": "fixed-rate-with-public-events",
        "period_s": 1.0,
        "event_triggers": [
            "anonymous_confirmation",
            "safety_intervention",
            "fleet_roster_change",
            "return_reserve_entry",
        ],
        "held_action_rebinding": "latest-public-observation",
        "retransmit_messages_on_hold": False,
    }


def test_planning_cadence_decouples_control_ticks_and_replans_on_public_events() -> None:
    validate_planning_cadence(
        _planning_contract(), control_period_s=0.2, episode_duration_s=300.0
    )
    cadence = PlanningCadenceController(control_period_s=0.2, planning_period_s=1.0)
    first_observation = _observation(sequence=0)
    assert cadence.due_reasons(control_tick=0, active_drone_ids=["uav-00"]) == (
        "fixed_period",
        "initial",
    )
    action = ActionPacket(
        episode_id="episode-a",
        drone_id="uav-00",
        sequence=0,
        issued_at_s=0.0,
        kind="OBSERVE",
        source_observation_id=first_observation.observation_id,
    )
    cadence.approve({"uav-00": action})
    assert cadence.due_reasons(control_tick=1, active_drone_ids=["uav-00"]) == ()
    rebound = cadence.held_actions({"uav-00": _observation(sequence=1)})["uav-00"]
    assert rebound.sequence == 1
    assert rebound.source_observation_id == "observation-1"
    assert rebound.messages == ()

    cadence.request_event("anonymous_confirmation")
    assert cadence.due_reasons(control_tick=2, active_drone_ids=["uav-00"]) == (
        "anonymous_confirmation",
    )
    cadence.reject_planning_attempt()
    assert "safety_intervention" in cadence.due_reasons(
        control_tick=3, active_drone_ids=["uav-00"]
    )
    assert "fixed_period" in cadence.due_reasons(
        control_tick=5, active_drone_ids=["uav-00"]
    )


def test_held_action_rebinding_preserves_motion_but_not_stale_identity() -> None:
    action = _action(drone_id="uav-00", invocation=0, x=4.0)
    rebound = rebind_held_action(action, _observation(sequence=4))
    assert rebound.waypoint == action.waypoint
    assert rebound.sequence == 4
    assert rebound.issued_at_s == 0.8
    assert rebound.source_observation_id is None


def test_planning_cadence_rejects_control_rate_planning() -> None:
    contract = _planning_contract()
    contract["period_s"] = 0.2
    with pytest.raises(ValueError, match="slower than control"):
        validate_planning_cadence(
            contract, control_period_s=0.2, episode_duration_s=300.0
        )
