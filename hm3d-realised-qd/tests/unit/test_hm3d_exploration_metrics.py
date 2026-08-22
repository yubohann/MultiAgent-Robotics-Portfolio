from __future__ import annotations

import pytest

from aerocity_method.contracts.exploration import (
    AgentExecutionOutcome,
    AgentExplorationPlan,
    ExplorationExecutionOutcome,
    TeamExplorationCandidate,
)
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.evaluation.hm3d_exploration import (
    assemble_episode_ledger,
    build_decision_record,
    candidate_set_hash,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (
    ExplorationMetricSample,
    SceneMacroAggregate,
    evaluation_denominator_sha256,
    score_exploration_episode,
)


def _samples() -> tuple[ExplorationMetricSample, ...]:
    return (
        ExplorationMetricSample(0.0, 0.0, 10.0, 0.0),
        ExplorationMetricSample(5.0, 5.0, 10.0, 5.5, hallucinated_free_volume_m3=0.5),
        ExplorationMetricSample(10.0, 9.0, 10.0, 9.2, hallucinated_free_volume_m3=0.2),
    )


def test_exploration_metrics_score_auc_and_map_correctness():
    report = score_exploration_episode(
        episode_id="episode0",
        samples=_samples(),
        horizon_s=10.0,
        collision_count=1,
        energy_j=12.0,
        delivered_messages=3,
        attempted_messages=4,
    )
    assert report.explored_free_flight_volume_auc_time == pytest.approx(0.475)
    assert report.final_coverage_at_budget == pytest.approx(0.9)
    assert report.final_explored_free_volume_m3 == pytest.approx(9.0)
    assert report.evaluator_reachable_free_flight_volume_m3 == pytest.approx(10.0)
    assert report.mean_explored_free_volume_rate_m3_per_s == pytest.approx(0.9)
    assert report.communication_delivery_ratio == pytest.approx(0.75)
    assert report.report_hash == report.to_dict()["report_hash"]


def test_exploration_metrics_reject_a_shifting_private_denominator():
    with pytest.raises(ValueError, match="denominator must remain frozen"):
        score_exploration_episode(
            episode_id="episode0",
            samples=(
                ExplorationMetricSample(0.0, 0.0, 10.0, 0.0),
                ExplorationMetricSample(2.0, 1.0, 11.0, 1.0),
            ),
            horizon_s=2.0,
        )


def test_evaluator_denominator_digest_binds_p03_geometry_but_not_row_order():
    first = {
        "scene_id": "scene-a",
        "source_geometry_sha256": "a" * 64,
        "flight_space_manifest_hash": "b" * 64,
        "collision_geometry_sha256": "c" * 64,
        "resolution_m": 0.25,
        "vehicle_clearance_m": 0.30,
        "free_flight_volume_m3": 100.0,
    }
    second = {**first, "scene_id": "scene-b", "source_geometry_sha256": "d" * 64}
    digest = evaluation_denominator_sha256((first, second))
    assert digest == evaluation_denominator_sha256((second, first))
    assert digest != evaluation_denominator_sha256(
        ({**first, "free_flight_volume_m3": 101.0}, second)
    )


def test_exploration_metrics_reject_nonmonotone_unique_volume():
    with pytest.raises(ValueError, match="monotone non-decreasing"):
        score_exploration_episode(
            episode_id="episode0",
            samples=(
                ExplorationMetricSample(0.0, 0.0, 10.0, 0.0),
                ExplorationMetricSample(2.0, 2.0, 10.0, 2.0),
                ExplorationMetricSample(3.0, 1.0, 10.0, 2.0),
            ),
            horizon_s=3.0,
        )


def test_episode_ledger_binds_selected_candidate_to_outcome():
    belief_hash = canonical_sha256({"belief": 1})
    candidate = TeamExplorationCandidate(
        candidate_id="candidate0",
        context_sha256=canonical_sha256({"context": 1}),
        belief_version_sha256s=(belief_hash,),
        agent_plans=(
            AgentExplorationPlan(
                "uav0",
                "explore",
                ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                1.0,
                2.0,
                0.1,
                1.0,
                frontier_id="frontier0",
            ),
        ),
        planned_descriptor=(0.0, 1.0),
        feasible=True,
    )
    outcome = ExplorationExecutionOutcome(
        outcome_id="outcome0",
        episode_id="episode0",
        decision_id="decision0",
        candidate_sha256=candidate.digest,
        started_timestamp_s=0.0,
        ended_timestamp_s=1.0,
        agent_outcomes=(
            AgentExecutionOutcome(
                "uav0",
                ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                1.0,
                1.0,
                0,
                0,
                False,
            ),
        ),
        observation_ids=("obs0",),
        delivered_message_ids=(),
        observed_free_delta=2,
        observed_occupied_delta=0,
        realised_descriptor=(0.0, 1.0),
    )
    record = build_decision_record(
        decision_id="decision0",
        candidate_set=(candidate,),
        selected_candidate=candidate,
        outcome=outcome,
    )
    ledger = assemble_episode_ledger(
        episode_id="episode0",
        scene_id="scene0",
        horizon_s=10.0,
        decisions=(record,),
        samples=_samples(),
        collision_count=0,
        energy_j=1.0,
    )
    assert record.candidate_set_sha256 == candidate_set_hash((candidate,))
    assert ledger.status == "TASK_VALID"
    assert ledger.ledger_hash == ledger.to_dict()["ledger_hash"]


def test_episode_ledger_rejects_outcome_for_unselected_candidate():
    candidate = TeamExplorationCandidate(
        candidate_id="candidate0",
        context_sha256=canonical_sha256({"context": 1}),
        belief_version_sha256s=(canonical_sha256({"belief": 1}),),
        agent_plans=(
            AgentExplorationPlan(
                "uav0",
                "explore",
                ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                1.0,
                2.0,
                0.1,
                1.0,
                frontier_id="frontier0",
            ),
        ),
        planned_descriptor=(0.0, 1.0),
        feasible=True,
    )
    bad_outcome = ExplorationExecutionOutcome(
        outcome_id="outcome0",
        episode_id="episode0",
        decision_id="decision0",
        candidate_sha256=canonical_sha256({"other": 1}),
        started_timestamp_s=0.0,
        ended_timestamp_s=1.0,
        agent_outcomes=(
            AgentExecutionOutcome(
                "uav0",
                ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                1.0,
                1.0,
                0,
                0,
                False,
            ),
        ),
        observation_ids=("obs0",),
        delivered_message_ids=(),
        observed_free_delta=2,
        observed_occupied_delta=0,
        realised_descriptor=(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="outcome candidate hash"):
        build_decision_record(
            decision_id="decision0",
            candidate_set=(candidate,),
            selected_candidate=candidate,
            outcome=bad_outcome,
        )


def test_scene_macro_aggregate_reports_worst_decile_without_completion_threshold():
    reports = (
        score_exploration_episode(episode_id="episode0", samples=_samples(), horizon_s=10.0),
        score_exploration_episode(episode_id="episode1", samples=_samples()[:-1], horizon_s=10.0),
    )
    aggregate = SceneMacroAggregate("method0", reports).to_dict()
    assert aggregate["episode_count"] == 2
    assert aggregate["worst_decile_auc_time"] == pytest.approx(0.375)
