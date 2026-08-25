from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    GuardedPath,
    PUBLIC_ROUTE_CONTINUITY_BONUS_MAX,
    PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M,
    PublicAgentPose,
    PublicFrontier,
    PublicTaskReservation,
    PublicSearchState,
    _collision_avoidance_geometric_recovery_candidates,
    _assignment_route_tube_separation_m,
    _has_meaningful_multi_agent_routes,
    _manifest_for_assignment,
    _nonconverging_recovery_path,
    _traffic_reservation_variants,
    _vertical_access_count,
    build_public_candidate_pool,
    fixed_altitude_frontiers,
    identity_path_guard,
    is_non_alias_exploration_path,
    _public_gain_proxy,
    public_candidate_pool_hash,
    outcome_calibrated_path_length_budget_m,
    select_public_baseline,
    task_reservation_matches_frontier,
)
from aerocity_method.contracts.models import CandidateFragmentManifest, PublicMethodContext
from aerocity_method.runtime.hm3d_realised_qd import audit_public_candidate_intent_richness


def test_public_route_access_credit_is_bounded() -> None:
    short = PublicFrontier(
        "short",
        (2.0, 0.0, 1.0),
        1.0,
        0.0,
        access_paths_m=(("uav0", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0))),),
    )
    long = PublicFrontier(
        "long",
        (8.0, 0.0, 1.0),
        1.0,
        0.0,
        access_paths_m=(("uav0", ((0.0, 0.0, 1.0), (8.0, 0.0, 1.0))),),
    )

    assert _public_gain_proxy(short) == 1.0
    assert 1.0 < _public_gain_proxy(long) <= 1.0 + PUBLIC_ROUTE_CONTINUITY_BONUS_MAX


def _state(
    *,
    frontiers: tuple[PublicFrontier, ...] | None = None,
    task_reservations: tuple[PublicTaskReservation, ...] = (),
) -> PublicSearchState:
    context = PublicMethodContext(
        context_id="context0",
        episode_id="episode0",
        decision_id="decision0",
        agent_features=(("uav0", (0.0, 0.0)), ("uav1", (1.0, 0.0))),
        public_features=(("map_coverage", 0.2),),
        budget=(("time_remaining_s", 20.0),),
    )
    return PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 0.9, 1),
            PublicAgentPose("uav1", (3.0, 0.0, 2.0), 0.7, 1),
        ),
        frontiers=frontiers
        or (
            PublicFrontier("frontier0", (1.0, 1.0, 1.0), 0.8, 0.1),
            PublicFrontier("frontier1", (4.0, 1.0, 3.0), 0.9, 0.2),
            PublicFrontier("frontier2", (2.0, -2.0, 2.0), 0.4, 0.1),
        ),
        decision_start_s=0.0,
        decision_duration_s=10.0,
        transit_timing_model=ConservativeTransitTimingModel("unit-test", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
        task_reservations=task_reservations,
    )


def _task_reservation(
    agent_id: str,
    path: tuple[tuple[float, float, float], ...],
) -> PublicTaskReservation:
    return PublicTaskReservation.from_completed_public_exploration_transit(
        agent_id=agent_id,
        source_decision_id="decision0",
        source_manifest_hash="a" * 64,
        source_transit_outcome_sha256="b" * 64,
        public_path_m=path,
    )


def _single_agent_state(
    *,
    position_m: tuple[float, float, float],
    frontiers: tuple[PublicFrontier, ...],
    task_reservations: tuple[PublicTaskReservation, ...] = (),
) -> PublicSearchState:
    context = PublicMethodContext(
        context_id="single-route-continuity",
        episode_id="single-route-continuity-episode",
        decision_id="decision1",
        agent_features=(("uav0", (0.0,)),),
        public_features=(("map_coverage", 0.2),),
        budget=(("time_remaining_s", 20.0),),
    )
    return PublicSearchState(
        context=context,
        agents=(PublicAgentPose("uav0", position_m, 1.0, 1),),
        frontiers=frontiers,
        decision_start_s=0.0,
        decision_duration_s=10.0,
        transit_timing_model=ConservativeTransitTimingModel("single-route", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
        task_reservations=task_reservations,
    )


def _guard(agent_id: str, path_m: tuple[tuple[float, float, float], ...]) -> GuardedPath:
    if path_m[-1][0] == 4.0 and agent_id == "uav0":
        return GuardedPath(False, path_m, reason="blocked")
    if path_m[-1][0] == 2.0:
        return GuardedPath(True, (path_m[0], (1.5, -1.5, 2.0)), rewritten=True)
    return GuardedPath(True, path_m)


def test_all_weak_baselines_receive_exactly_the_same_public_candidate_pool():
    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=3)
    pool_hash = public_candidate_pool_hash(pool)
    selected_hashes = set()
    for strategy in ("random", "frontier_3d", "auction"):
        selected, selection = select_public_baseline(strategy, pool, random_key=17)
        assert selected.manifest_hash == selection.selected_manifest_hash
        assert selected.manifest_hash in {manifest.manifest_hash for manifest in pool}
        assert public_candidate_pool_hash(pool) == pool_hash
        selected_hashes.add(selected.manifest_hash)
    assert selected_hashes


@pytest.mark.parametrize("strategy", ("greedy", "strong_planner"))
def test_retired_public_baseline_selectors_fail_closed(strategy: str) -> None:
    """Removed legacy branches cannot silently re-enter a P07 comparison."""

    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=2)
    with pytest.raises(ValueError, match="unsupported weak-baseline strategy"):
        select_public_baseline(strategy, pool)


def test_legal_candidate_plans_only_the_minimum_observation_dwell():
    state = _state()
    manifest = build_public_candidate_pool(state, _guard, candidate_limit=1)[0]
    observations = [
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "observation"
    ]
    assert observations
    assert all(
        fragment.planned_end - fragment.planned_start == pytest.approx(state.observe_dwell_s)
        for fragment in observations
    )


def test_four_compatible_jobs_preserve_an_all_active_team_candidate() -> None:
    context = PublicMethodContext(
        context_id="four-active",
        episode_id="four-active-episode",
        decision_id="four-active-decision",
        agent_features=tuple((f"uav{index}", (0.0, 0.0)) for index in range(4)),
        public_features=(("map_coverage", 0.2),),
        budget=(("time_remaining_s", 20.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (float(index) * 3.0, 0.0, 1.0), 1.0, 1)
            for index in range(4)
        ),
        frontiers=tuple(
            PublicFrontier(
                f"frontier{index}",
                (float(index) * 3.0 + 1.0, 1.0, 1.0),
                1.0 - 0.1 * index,
                0.05,
            )
            for index in range(4)
        ),
        decision_start_s=0.0,
        decision_duration_s=20.0,
        transit_timing_model=ConservativeTransitTimingModel("four-active", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=8,
    )

    assert any(
        all(
            dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
        for manifest in pool
    )
    assert all(
        dict(fragment.type_signature.public_features).get("hold_reason", "") == ""
        for manifest in pool
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
        and dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
    )


def test_joint_conflict_fallback_uses_auditable_collision_avoidance_hold() -> None:
    """A jointly unsafe corridor must degrade explicitly, not fail the episode."""

    state = _state()

    def joint_guard(manifest):
        roles = {
            fragment.agent_id: dict(fragment.type_signature.public_features)["assignment_role"]
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        return "synchronized_fleet_separation" if all(role == "explore" for role in roles.values()) else None

    pool = build_public_candidate_pool(
        state,
        _guard,
        candidate_limit=2,
        joint_guard=joint_guard,
    )

    assert len([manifest for manifest in pool if manifest.feasible]) == 2
    for manifest in pool:
        if not manifest.feasible:
            continue
        holds = [
            dict(fragment.type_signature.public_features)["hold_reason"]
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
            and dict(fragment.type_signature.public_features)["assignment_role"] == "hold"
        ]
        assert holds == ["collision_avoidance"]


def test_joint_conflict_prefers_common_enforced_traffic_reservation_before_hold() -> None:
    """A serializable bottleneck keeps both explorers active before using hold."""

    state = _state()

    def joint_guard(manifest):
        transits = {
            fragment.agent_id: fragment
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        delayed = dict(transits["uav1"].type_signature.public_features)
        if (
            delayed["traffic_reservation_delay_s"] > 0.0
            and delayed["traffic_reservation_predecessor_agent_id"] == "uav0"
        ):
            return None
        return "synchronized_fleet_separation"

    pool = build_public_candidate_pool(
        state,
        _guard,
        candidate_limit=1,
        joint_guard=joint_guard,
    )

    manifest = next(manifest for manifest in pool if manifest.feasible)
    transits = {
        fragment.agent_id: fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    }
    assert {
        dict(fragment.type_signature.public_features)["assignment_role"]
        for fragment in transits.values()
    } == {"explore"}
    uav1_features = dict(transits["uav1"].type_signature.public_features)
    assert uav1_features["traffic_reservation_delay_s"] > 0.0
    assert uav1_features["traffic_reservation_predecessor_agent_id"] == "uav0"
    assert transits["uav1"].planned_start > transits["uav0"].planned_end


def test_traffic_reservation_variants_serialize_three_or_four_agent_corridor() -> None:
    """A multi-vehicle corridor needs a serial chain, not one delayed pair."""

    context = PublicMethodContext(
        context_id="corridor-chain",
        episode_id="corridor-chain-episode",
        decision_id="corridor-chain-decision",
        agent_features=tuple((f"uav{index}", (0.0, 0.0)) for index in range(4)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 40.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(
                f"uav{index}", (float(index) * 2.0, 0.0, 1.0), 1.0, 1
            )
            for index in range(4)
        ),
        frontiers=tuple(
            PublicFrontier(
                f"frontier{index}",
                (float(index) * 2.0 + 1.0, 0.0, 1.0),
                1.0,
                0.05,
            )
            for index in range(4)
        ),
        decision_start_s=0.0,
        decision_duration_s=40.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "corridor-chain", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    variants = _traffic_reservation_variants(
        state,
        ((0, 1, 2, 3),),
        identity_path_guard,
        candidate_limit=1,
    )
    chain_variants = [
        (delays, predecessors)
        for delays, predecessors in variants
        if len(delays) >= 3
    ]
    assert chain_variants
    delays, predecessors = chain_variants[0]
    assert set(delays) == set(predecessors)
    undelayed = {agent.agent_id for agent in state.agents} - set(delays)
    assert len(undelayed) == 1
    for delayed_id in delays:
        current = delayed_id
        visited: set[str] = set()
        while current in predecessors:
            assert current not in visited
            visited.add(current)
            current = predecessors[current]
        assert current in undelayed


def test_four_agent_corridor_chain_is_feasible_through_transitive_joint_guard() -> None:
    context = PublicMethodContext(
        context_id="corridor-chain-feasible",
        episode_id="corridor-chain-feasible-episode",
        decision_id="corridor-chain-feasible-decision",
        agent_features=tuple((f"uav{index}", (0.0, 0.0)) for index in range(4)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 40.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(
                f"uav{index}", (float(index) * 2.0, 0.0, 1.0), 1.0, 1
            )
            for index in range(4)
        ),
        frontiers=tuple(
            PublicFrontier(
                f"frontier{index}",
                (float(index) * 2.0 + 1.0, 0.0, 1.0),
                1.0,
                0.05,
            )
            for index in range(4)
        ),
        decision_start_s=0.0,
        decision_duration_s=40.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "corridor-chain-feasible", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    def joint_guard(manifest):
        transits = {
            fragment.agent_id: dict(fragment.type_signature.public_features)
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        if not transits or any(
            features["assignment_role"] != "explore"
            for features in transits.values()
        ):
            return "synchronized_fleet_separation"
        delayed = {
            agent_id: features
            for agent_id, features in transits.items()
            if float(features["traffic_reservation_delay_s"]) > 0.0
        }
        if not delayed:
            return "synchronized_fleet_separation"
        undelayed = set(transits) - set(delayed)
        if len(undelayed) != 1:
            return "synchronized_fleet_separation"
        for delayed_id in delayed:
            current = delayed_id
            visited: set[str] = set()
            while current in delayed:
                if current in visited:
                    return "synchronized_fleet_separation"
                visited.add(current)
                predecessor = delayed[current][
                    "traffic_reservation_predecessor_agent_id"
                ]
                current = predecessor
            if current not in undelayed:
                return "synchronized_fleet_separation"
        return None

    pool = build_public_candidate_pool(
        state,
        identity_path_guard,
        candidate_limit=1,
        joint_guard=joint_guard,
        minimum_multi_agent_route_candidates=1,
    )
    feasible = [manifest for manifest in pool if manifest.feasible]
    assert len(feasible) == 1
    transit_delays = [
        float(dict(fragment.type_signature.public_features)["traffic_reservation_delay_s"])
        for fragment in feasible[0].fragments
        if fragment.type_signature.fragment_type == "transit"
    ]
    assert sum(delay > 0.0 for delay in transit_delays) >= 3


def test_joint_conflict_uses_outcome_backtrack_when_stationary_yield_is_unsafe() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("explore-uav0", (1.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
            PublicFrontier(
                "outcome-backtrack-uav0",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav0",
                "backtrack",
                "uav0",
            ),
        )
    )

    def guard(_agent_id, path_m):
        return GuardedPath(True, path_m, reason="stationary_hold" if path_m[0] == path_m[-1] else "")

    def joint_guard(manifest):
        roles = {
            fragment.agent_id: dict(fragment.type_signature.public_features)["assignment_role"]
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        return None if roles["uav0"] == "backtrack" else "synchronized_fleet_separation"

    pool = build_public_candidate_pool(
        state,
        guard,
        candidate_limit=1,
        joint_guard=joint_guard,
    )

    manifest = next(manifest for manifest in pool if manifest.feasible)
    uav0 = next(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit" and fragment.agent_id == "uav0"
    )
    assert dict(uav0.type_signature.public_features)["assignment_role"] == "backtrack"
    assert uav0.context_bucket == "hm3d-outcome-backed-backtrack"


def test_envelope_recovery_is_explicit_and_cannot_be_emitted_as_exploration() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("explore-uav0", (1.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
            PublicFrontier(
                "outcome-backtrack-uav0",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav0",
                "backtrack",
                "uav0",
            ),
        )
    )

    def joint_guard(manifest):
        features = [
            dict(fragment.type_signature.public_features)
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        ]
        return (
            None
            if {row["safety_recovery_kind"] for row in features}
            == {"collision_avoidance_recovery"}
            else "synchronized_fleet_separation"
        )

    pool = build_public_candidate_pool(
        state,
        _guard,
        candidate_limit=1,
        joint_guard=joint_guard,
    )

    manifest = next(row for row in pool if row.feasible)
    transit_features = {
        fragment.agent_id: dict(fragment.type_signature.public_features)
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    }
    assert transit_features["uav0"]["assignment_role"] == "backtrack"
    assert transit_features["uav0"]["safety_recovery_agent_id"] == "uav0"
    assert transit_features["uav1"]["assignment_role"] == "hold"
    assert transit_features["uav1"]["hold_reason"] == "collision_avoidance_recovery"
    assert all(
        features["safety_recovery_kind"] == "collision_avoidance_recovery"
        for features in transit_features.values()
    )
    observation_features = {
        fragment.agent_id: dict(fragment.type_signature.public_features)
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "observation"
    }
    assert all(
        features["safety_recovery_kind"] == "collision_avoidance_recovery"
        for features in observation_features.values()
    )
    assert manifest.quality_hint == pytest.approx(0.0)


def test_geometric_recovery_requires_monotone_separation_from_stationary_agents() -> None:
    assert _nonconverging_recovery_path(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 1.0)),
        stationary_positions_m=((0.6, 0.0, 1.0),),
    )
    assert not _nonconverging_recovery_path(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        stationary_positions_m=((0.6, 0.0, 1.0),),
    )
    assert not _nonconverging_recovery_path(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 1.0)),
        stationary_positions_m=(),
    )


def test_geometric_recovery_resolves_sorted_synthetic_frontier_and_has_zero_gain() -> None:
    context = PublicMethodContext(
        context_id="geometric-recovery",
        episode_id="geometric-recovery-episode",
        decision_id="geometric-recovery-decision",
        agent_features=(("uav0", (0.0,)), ("uav1", (0.0,))),
        public_features=(("map_coverage", 0.2),),
        budget=(("time_remaining_s", 20.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
            PublicAgentPose("uav1", (0.6, 0.0, 1.0), 1.0, 1),
        ),
        # IDs intentionally sort after the synthetic collision-recovery ID.
        # The candidate builder must resolve the post-sort index by ID.
        frontiers=(
            PublicFrontier("z-away-uav0", (-1.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("z-away-uav1", (2.0, 0.0, 1.0), 1.0, 0.0),
        ),
        decision_start_s=0.0,
        decision_duration_s=10.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "geometric-recovery", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    def joint_guard(manifest: CandidateFragmentManifest) -> str | None:
        safety_kinds = {
            dict(fragment.type_signature.public_features).get("safety_recovery_kind", "")
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        return (
            None
            if safety_kinds == {"collision_avoidance_recovery"}
            else "synchronized_fleet_separation"
        )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
        joint_guard=joint_guard,
    )
    recovery = next(manifest for manifest in pool if manifest.feasible)
    transit = {
        fragment.agent_id: fragment
        for fragment in recovery.fragments
        if fragment.type_signature.fragment_type == "transit"
    }
    assert set(
        dict(fragment.type_signature.public_features)["safety_recovery_kind"]
        for fragment in transit.values()
    ) == {"collision_avoidance_recovery"}
    assert sum(
        dict(fragment.type_signature.public_features)["assignment_role"] == "backtrack"
        for fragment in transit.values()
    ) == 1
    assert recovery.quality_hint == pytest.approx(0.0)


def test_recovery_builder_rejects_a_backtrack_owned_by_another_agent() -> None:
    state = _state(
        frontiers=(
            PublicFrontier(
                "a-outcome-backtrack-uav1",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav1",
                "backtrack",
                "uav1",
            ),
            PublicFrontier("z-explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
        )
    )

    with pytest.raises(ValueError, match="owned backtrack route"):
        _manifest_for_assignment(
            state,
            (0, -1),
            _guard,
            candidate_index=0,
            hold_reason_overrides={"uav1": "collision_avoidance_recovery"},
            collision_avoidance_recovery_agent_id="uav0",
        )


def test_recovery_builder_requires_explicit_hold_metadata() -> None:
    state = _state(
        frontiers=(
            PublicFrontier(
                "a-outcome-backtrack-uav0",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav0",
                "backtrack",
                "uav0",
            ),
            PublicFrontier("z-explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
        )
    )

    with pytest.raises(ValueError, match="explicit recovery hold reason"):
        _manifest_for_assignment(
            state,
            (0, -1),
            _guard,
            candidate_index=0,
            collision_avoidance_recovery_agent_id="uav0",
        )


def test_outcome_backtrack_is_owner_only_and_replaces_hold_when_exploration_is_unavailable() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
            PublicFrontier(
                "outcome-backtrack-uav0",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav0",
                "backtrack",
                "uav0",
            ),
        )
    )

    def guard(agent_id, path_m):
        if path_m[0] == path_m[-1]:
            return GuardedPath(True, path_m, reason="stationary_hold")
        endpoint_x = path_m[-1][0]
        legal = (agent_id, endpoint_x) in {("uav0", -1.0), ("uav1", 4.0)}
        return GuardedPath(legal, path_m, reason="" if legal else "blocked")

    pool = build_public_candidate_pool(state, guard, candidate_limit=2)
    for manifest in pool:
        roles = {
            fragment.agent_id: dict(fragment.type_signature.public_features)["assignment_role"]
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        }
        assert roles == {"uav0": "backtrack", "uav1": "explore"}
        assert manifest.quality_hint == pytest.approx(1.0)
        backtrack = next(
            fragment
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit" and fragment.agent_id == "uav0"
        )
        assert backtrack.context_bucket == "hm3d-outcome-backed-backtrack"


def test_outcome_backtrack_is_withheld_when_ordinary_exploration_is_legal() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("explore-uav0", (1.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("explore-uav1", (4.0, 0.0, 2.0), 1.0, 0.0),
            PublicFrontier(
                "outcome-backtrack-uav0",
                (-1.0, 0.0, 1.0),
                0.0,
                0.0,
                "uav0",
                "backtrack",
                "uav0",
            ),
        )
    )

    def guard(agent_id, path_m):
        if path_m[0] == path_m[-1]:
            return GuardedPath(True, path_m, reason="stationary_hold")
        endpoint_x = path_m[-1][0]
        legal = (agent_id, endpoint_x) in {("uav0", 1.0), ("uav0", -1.0), ("uav1", 4.0)}
        return GuardedPath(legal, path_m, reason="" if legal else "blocked")

    pool = build_public_candidate_pool(state, guard, candidate_limit=2)
    assert all(
        dict(fragment.type_signature.public_features)["assignment_role"] != "backtrack"
        for manifest in pool
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )


def test_outcome_backtrack_requires_a_single_explicit_owner() -> None:
    with pytest.raises(ValueError, match="exclusive agent"):
        PublicFrontier("invalid-backtrack", (1.0, 0.0, 1.0), 0.0, 0.0, task_kind="backtrack")


@pytest.mark.parametrize(
    ("decision_count", "expected_budget_m"),
    ((5, 5.2), (8, 2.2), (10, 1.2005)),
)
def test_outcome_calibrated_frontier_step_shrinks_with_decision_density(
    decision_count: int, expected_budget_m: float
) -> None:
    timing = ConservativeTransitTimingModel("real-cf2x-profile", 1.0, 0.8, 0.55)

    budget_m = outcome_calibrated_path_length_budget_m(
        decision_duration_s=40.0 / decision_count,
        observe_dwell_s=1.0,
        transit_timing_model=timing,
    )

    assert budget_m == pytest.approx(expected_budget_m)
    predicted_s = timing.estimate_seconds(((0.0, 0.0, 0.0), (budget_m, 0.0, 0.0)))
    assert predicted_s + 1.0 == pytest.approx(40.0 / decision_count)


def test_guard_rewrite_is_recorded_and_illegal_assignments_are_not_selected():
    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=3)
    assert any(
        fragment.guard_rewritten
        for manifest in pool
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    assert all(manifest.feasible for manifest in pool)
    selected, _ = select_public_baseline("frontier_3d", pool)
    assert selected.feasible is True


@pytest.mark.parametrize("strategy", ("frontier_3d", "auction"))
def test_ranked_team_baselines_do_not_price_parallel_flight_as_sequential_time(
    strategy: str,
) -> None:
    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=8)

    def explorer_count(manifest) -> int:
        return sum(
            dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )

    selected, _ = select_public_baseline(strategy, pool)

    assert explorer_count(selected) == max(explorer_count(manifest) for manifest in pool)


def test_frontier_score_uses_shared_decision_window_for_concurrent_team_gain() -> None:
    """A high-gain multi-explorer row must beat a short low-gain hold row."""

    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=8)

    def explorer_count(manifest) -> int:
        return sum(
            dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )

    selected, selection = select_public_baseline("frontier_3d", pool)
    selected_score = dict(selection.scores)[selected.candidate_id]
    max_score = max(score for _, score in selection.scores)

    assert selected_score == pytest.approx(max_score)
    assert explorer_count(selected) == max(explorer_count(manifest) for manifest in pool)


def test_frontier_score_counts_a_public_frontier_cluster_once_per_team() -> None:
    """Several views of one cluster cannot imitate independent information."""

    state = _state(
        frontiers=(
            PublicFrontier("cluster-a-left", (1.0, 0.0, 1.0), 1.0, 0.0, frontier_cluster_id="a"),
            PublicFrontier("cluster-a-right", (4.0, 0.0, 2.0), 1.0, 0.0, frontier_cluster_id="a"),
            PublicFrontier("cluster-b", (1.0, 2.0, 1.0), 1.0, 0.0, frontier_cluster_id="b"),
        )
    )
    def cluster_ids(manifest) -> tuple[str, ...]:
        return tuple(
            str(dict(fragment.type_signature.public_features)["frontier_cluster_id"])
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )

    def guard(_agent_id, path_m) -> GuardedPath:
        return GuardedPath(True, path_m)

    duplicate = _manifest_for_assignment(state, (0, 1), guard, candidate_index=0)
    diverse = _manifest_for_assignment(state, (0, 2), guard, candidate_index=1)
    assert len(set(cluster_ids(duplicate))) == 1
    assert len(set(cluster_ids(diverse))) == 2
    # Reverse the old lexical preference deliberately. The public objective,
    # rather than generation order, must retain the distinct-cluster option.
    duplicate = replace(duplicate, candidate_id="zzz-duplicate")
    diverse = replace(diverse, candidate_id="aaa-diverse")

    selected, selection = select_public_baseline("frontier_3d", (duplicate, diverse))

    assert selected.candidate_id == "aaa-diverse"
    assert len(set(cluster_ids(selected))) == 2
    assert dict(selection.scores)["aaa-diverse"] > dict(selection.scores)["zzz-duplicate"]


def test_frontier_semantic_tie_break_does_not_depend_on_candidate_id() -> None:
    """Equal cluster value uses public formation semantics before an ID fallback."""

    state = _state(
        frontiers=(
            PublicFrontier("cluster-a-left", (1.0, 0.0, 1.0), 1.0, 0.0, frontier_cluster_id="a"),
            PublicFrontier("cluster-a-right", (4.0, 0.0, 2.0), 1.0, 0.0, frontier_cluster_id="a"),
            PublicFrontier("cluster-b", (1.0, 2.0, 1.0), 1.0, 0.0, frontier_cluster_id="b"),
        )
    )
    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path_m: GuardedPath(True, path_m),
        candidate_limit=8,
    )

    def cluster_count(manifest) -> int:
        return len(
            {
                str(dict(fragment.type_signature.public_features)["frontier_cluster_id"])
                for fragment in manifest.fragments
                if fragment.type_signature.fragment_type == "transit"
            }
        )

    diverse = [manifest for manifest in pool if cluster_count(manifest) == 2]
    lower_complementarity = min(diverse, key=lambda manifest: manifest.planned_descriptor[2])
    higher_complementarity = max(diverse, key=lambda manifest: manifest.planned_descriptor[2])
    assert higher_complementarity.planned_descriptor[2] > lower_complementarity.planned_descriptor[2]
    lower_complementarity = replace(lower_complementarity, candidate_id="zzz-low-complementarity")
    higher_complementarity = replace(higher_complementarity, candidate_id="aaa-high-complementarity")

    selected, selection = select_public_baseline(
        "frontier_3d", (lower_complementarity, higher_complementarity)
    )

    assert selected.candidate_id == "aaa-high-complementarity"
    assert dict(selection.scores)["aaa-high-complementarity"] == pytest.approx(
        dict(selection.scores)["zzz-low-complementarity"]
    )


def test_frontier_pool_rejects_a_settled_endpoint_alias_but_not_short_observation() -> None:
    """A current-pose alias cannot win by paying only dwell time."""

    state = _state(
        frontiers=(
            # This represents a snapped/repeated observation endpoint: high
            # nominal gain, but still inside the settled-position tolerance.
            PublicFrontier("micro", (0.01, 0.0, 1.0), 100.0, 0.0),
            PublicFrontier("uav0-long", (1.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("uav1-long", (4.0, 0.0, 2.0), 1.0, 0.0),
        )
    )
    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=8,
    )

    selected, _ = select_public_baseline("frontier_3d", pool)
    moving = [
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
        and dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
    ]

    assert len(moving) == 2
    assert all(
        is_non_alias_exploration_path(fragment.path)
        for fragment in moving
    )


@pytest.mark.parametrize(
    ("frontier_id", "position_m"),
    (
        ("short-doorway", (0.25, 0.0, 1.0)),
        ("short-vertical", (0.0, 0.0, 1.25)),
    ),
)
def test_short_non_alias_observation_remains_in_the_common_candidate_pool(
    frontier_id: str,
    position_m: tuple[float, float, float],
) -> None:
    state = _single_agent_state(
        position_m=(0.0, 0.0, 1.0),
        frontiers=(PublicFrontier(frontier_id, position_m, 1.0, 0.0),),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=1,
    )

    transit = next(
        fragment
        for fragment in pool[0].fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    assert dict(transit.type_signature.public_features)["assignment_role"] == "explore"
    assert is_non_alias_exploration_path(transit.path)
    assert sum(
        math.dist(left, right) for left, right in zip(transit.path, transit.path[1:], strict=False)
    ) == pytest.approx(0.25)


def test_current_public_access_polyline_survives_candidate_construction() -> None:
    path_m = ((0.0, 0.0, 1.0), (0.0, 2.0, 1.0), (2.0, 2.0, 1.0))
    state = _single_agent_state(
        position_m=path_m[0],
        frontiers=(
            PublicFrontier(
                "public-access-route",
                path_m[-1],
                1.0,
                0.0,
                access_paths_m=(("uav0", path_m),),
                frontier_cluster_id="cluster0",
            ),
        ),
    )
    received_paths: list[tuple[tuple[float, float, float], ...]] = []

    def guard(_agent_id: str, requested_path_m: tuple[tuple[float, float, float], ...]) -> GuardedPath:
        received_paths.append(requested_path_m)
        return GuardedPath(True, requested_path_m)

    pool = build_public_candidate_pool(state, guard, candidate_limit=1)
    transit = next(
        fragment for fragment in pool[0].fragments if fragment.type_signature.fragment_type == "transit"
    )

    assert path_m in received_paths
    assert transit.path == path_m
    features = dict(transit.type_signature.public_features)
    assert features["public_access_path_revalidated"] is True
    assert features["frontier_cluster_id"] == "cluster0"


def test_public_search_state_rejects_access_path_from_an_old_agent_pose() -> None:
    with pytest.raises(ValueError, match="start at the current public agent pose"):
        _single_agent_state(
            position_m=(0.0, 0.0, 1.0),
            frontiers=(
                PublicFrontier(
                    "stale-public-access-route",
                    (2.0, 2.0, 1.0),
                    1.0,
                    0.0,
                    access_paths_m=(
                        (
                            "uav0",
                            ((0.25, 0.0, 1.0), (0.25, 2.0, 1.0), (2.0, 2.0, 1.0)),
                        ),
                    ),
                ),
            ),
        )


def test_empty_task_reservations_preserve_common_candidate_pool_hash() -> None:
    state = _state()
    explicit_empty = _state(task_reservations=())

    default_pool = build_public_candidate_pool(state, _guard, candidate_limit=3)
    empty_pool = build_public_candidate_pool(explicit_empty, _guard, candidate_limit=3)

    assert state.to_dict() == explicit_empty.to_dict()
    assert [manifest.to_dict() for manifest in default_pool] == [
        manifest.to_dict() for manifest in empty_pool
    ]
    assert public_candidate_pool_hash(default_pool) == public_candidate_pool_hash(empty_pool)


def test_task_reservation_rejects_a_settled_endpoint_alias() -> None:
    with pytest.raises(ValueError, match="aliases its settled endpoint"):
        _task_reservation(
            "uav0",
            ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.0, 1.0)),
        )


def test_task_reservation_retains_equal_gain_forward_public_route() -> None:
    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("reverse", (0.0, 0.0, 1.0), 1.0, 0.0),
        ),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )
    selected, _ = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    features = dict(transit.type_signature.public_features)

    assert transit.path[-1] == (2.0, 0.0, 1.0)
    assert features["task_reservation_matched"] is True
    assert features["task_reservation_forward_compatible"] is True
    assert features["task_reservation_heading_alignment"] == pytest.approx(1.0)
    assert features["task_reservation_switch_cost"] == pytest.approx(0.0)
    assert features["task_reservation_source_decision_id"] == "decision0"


def test_task_reservation_allows_a_substantially_better_public_switch() -> None:
    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward-low", (2.0, 0.0, 1.0), 0.1, 0.0),
            PublicFrontier("reverse-high", (0.0, 0.0, 1.0), 1.0, 0.0),
        ),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )
    selected, selection = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    features = dict(transit.type_signature.public_features)

    assert transit.path[-1] == (0.0, 0.0, 1.0)
    assert features["task_reservation_matched"] is True
    assert features["task_reservation_forward_compatible"] is False
    assert features["task_reservation_heading_alignment"] == pytest.approx(-1.0)
    assert features["task_reservation_switch_cost"] > 0.0
    assert dict(selection.scores)[selected.candidate_id] == pytest.approx(
        max(score for _, score in selection.scores)
    )


def test_task_reservation_hysteresis_prefers_forward_route_inside_material_margin() -> None:
    """A small local-gain advantage must not cause an immediate reversal."""

    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            # The reverse task has a slightly higher raw gain, but remains
            # inside the frozen reservation switch margin after its reversal
            # cost is applied.
            PublicFrontier("reverse-slightly-better", (0.0, 0.0, 1.0), 1.3, 0.0),
        ),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )
    selected, selection = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path[-1] == (2.0, 0.0, 1.0)
    assert dict(selection.scores)[selected.candidate_id] < max(
        score for _, score in selection.scores
    )


def test_frontier_selection_uses_height_crossing_inside_continuity_margin() -> None:
    """A real public cross-height edge wins only after score equivalence filtering."""

    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("vertical", (1.0, 0.0, 1.6), 1.0, 0.0),
        ),
    )
    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )

    selected, selection = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path[-1] == (1.0, 0.0, 1.6)
    scores = dict(selection.scores)
    assert scores[selected.candidate_id] >= max(scores.values()) - 0.20 - 1.0e-9


def test_frontier_selection_uses_height_crossing_on_first_decision_without_reservation() -> None:
    """The first decision must not need stale history to access another floor."""

    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("vertical", (1.0, 0.0, 1.6), 1.0, 0.0),
        ),
    )
    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )

    selected, _ = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path[-1] == (1.0, 0.0, 1.6)


def test_vertical_access_threshold_is_inclusive_and_excludes_non_exploration() -> None:
    exact_state = _single_agent_state(
        position_m=(0.0, 0.0, 1.0),
        frontiers=(PublicFrontier("exact", (0.0, 0.0, 1.5), 1.0, 0.0),),
    )
    exact_pool = build_public_candidate_pool(
        exact_state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=1,
    )
    assert _vertical_access_count(exact_pool[0]) == 1

    below_state = _single_agent_state(
        position_m=(0.0, 0.0, 1.0),
        frontiers=(PublicFrontier("below", (0.0, 0.0, 1.499999), 1.0, 0.0),),
    )
    below_pool = build_public_candidate_pool(
        below_state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=1,
    )
    assert _vertical_access_count(below_pool[0]) == 0


def test_frontier_selection_does_not_trade_material_gain_for_height_crossing() -> None:
    """A materially worse vertical edge remains below the frozen gain margin."""

    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("vertical-low-gain", (1.0, 0.0, 1.6), 0.1, 0.0),
        ),
    )
    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )

    selected, selection = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path[-1] == (2.0, 0.0, 1.0)
    scores = dict(selection.scores)
    assert scores[selected.candidate_id] == pytest.approx(max(scores.values()))


def test_task_reservation_hysteresis_does_not_block_a_materially_better_switch() -> None:
    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier("forward", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("reverse-materially-better", (0.0, 0.0, 1.0), 2.0, 0.0),
        ),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )
    selected, _ = select_public_baseline("frontier_3d", pool)
    transit = next(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    assert transit.path[-1] == (0.0, 0.0, 1.0)


def test_task_reservation_rejects_an_opposite_public_frontier_normal() -> None:
    reservation = PublicTaskReservation.from_completed_public_exploration_transit(
        agent_id="uav0",
        source_decision_id="decision0",
        source_manifest_hash="a" * 64,
        source_transit_outcome_sha256="b" * 64,
        public_path_m=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        task_anchor_m=(1.0, 0.0, 1.0),
        task_normal_unit=(1.0, 0.0, 0.0),
    )
    frontier = PublicFrontier(
        "opposite-normal",
        (1.25, 0.0, 1.0),
        1.0,
        0.0,
        task_anchor_m=(1.0, 0.0, 1.0),
        task_normal_unit=(-1.0, 0.0, 0.0),
    )

    matched, anchor_distance, normal_alignment = task_reservation_matches_frontier(
        reservation,
        frontier,
    )

    assert matched is False
    assert anchor_distance == pytest.approx(0.0)
    assert normal_alignment == pytest.approx(-1.0)


def test_task_reservation_missing_current_public_match_is_released_by_match_contract() -> None:
    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    missing = PublicFrontier("far-task", (4.0, 0.0, 1.0), 1.0, 0.0)

    matched, anchor_distance, _normal_alignment = task_reservation_matches_frontier(
        reservation,
        missing,
    )

    assert matched is False
    assert anchor_distance > PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M


def test_route_progress_reservation_keeps_public_viewpoint_provenance() -> None:
    reservation = PublicTaskReservation.from_completed_public_exploration_transit(
        agent_id="uav0",
        source_decision_id="decision-progress",
        source_manifest_hash="c" * 64,
        source_transit_outcome_sha256="d" * 64,
        public_path_m=((0.0, 0.0, 1.0), (0.75, 0.0, 1.0)),
        task_anchor_m=(1.0, 0.0, 1.0),
        task_normal_unit=(0.0, 1.0, 0.0),
        source_frontier_cluster_id="cluster-progress",
        source_viewpoint_kind="route_progress",
    )

    assert reservation.source_viewpoint_kind == "route_progress"
    assert reservation.task_anchor_m == pytest.approx((1.0, 0.0, 1.0))
    assert reservation.task_normal_unit == pytest.approx((0.0, 1.0, 0.0))
    assert reservation.source_frontier_cluster_id == "cluster-progress"


def test_outcome_backtrack_has_no_task_reservation_privilege() -> None:
    reservation = _task_reservation("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)))
    state = _single_agent_state(
        position_m=(1.0, 0.0, 1.0),
        task_reservations=(reservation,),
        frontiers=(
            PublicFrontier(
                "owned-backtrack",
                (0.0, 0.0, 1.0),
                0.0,
                0.0,
                source_agent_id="uav0",
                task_kind="backtrack",
                exclusive_agent_id="uav0",
                viewpoint_kind="outcome_backtrack",
            ),
        ),
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=1,
    )
    transit = next(
        fragment
        for fragment in pool[0].fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    features = dict(transit.type_signature.public_features)

    assert features["assignment_role"] == "backtrack"
    assert features["task_reservation_matched"] is False
    assert features["task_reservation_switch_cost"] == pytest.approx(0.0)


def test_joint_admission_rejection_stays_in_the_common_candidate_denominator():
    pool = build_public_candidate_pool(
        _state(),
        _guard,
        candidate_limit=3,
        joint_guard=lambda manifest: (
            "synchronized_fleet_separation"
            if manifest.candidate_id == "hm3d-public-candidate-0"
            else None
        ),
    )
    assert len(pool) == 3
    assert all(row.feasible for row in pool)
    assert all(row.candidate_id != "hm3d-public-candidate-0" for row in pool)
    selected, _ = select_public_baseline("random", pool, random_key=1)
    assert selected.feasible is True


def test_candidate_pool_rejects_single_legal_row_when_a_selector_comparison_needs_headroom():
    with pytest.raises(ValueError, match="lacks strategy headroom"):
        build_public_candidate_pool(
            _state(),
            _guard,
            candidate_limit=3,
            joint_guard=lambda manifest: (
                "synchronized_fleet_separation"
                if manifest.candidate_id != "hm3d-public-candidate-0"
                else None
            ),
            minimum_feasible_candidates=2,
        )


def test_candidate_pool_rejects_an_impossible_headroom_requirement():
    with pytest.raises(ValueError, match="cannot exceed"):
        build_public_candidate_pool(
            _state(),
            _guard,
            candidate_limit=2,
            minimum_feasible_candidates=3,
        )


def test_public_candidates_contain_no_evaluator_truth_or_future_observation_id():
    pool = build_public_candidate_pool(_state(), _guard, candidate_limit=3)
    serialized = repr([manifest.to_dict() for manifest in pool]).casefold()
    assert "target" not in serialized
    assert "private" not in serialized
    assert all(
        fragment.source_observation_id is None
        for manifest in pool
        for fragment in manifest.fragments
    )


def test_fixed_altitude_control_projects_every_frontier_and_keeps_guard_authority():
    state = _state(
        frontiers=(
            PublicFrontier(
                "route-progress",
                (1.0, 1.0, 1.0),
                0.8,
                0.1,
                viewpoint_kind="route_progress",
            ),
            PublicFrontier("observation", (4.0, 1.0, 3.0), 0.9, 0.2),
        )
    )
    fixed = fixed_altitude_frontiers(state.frontiers, altitude_m=1.25)
    assert {frontier.position_m[2] for frontier in fixed} == {1.25}
    assert {frontier.frontier_id: frontier.viewpoint_kind for frontier in fixed} == {
        "route-progress": "route_progress",
        "observation": "observation",
    }
    fixed_state = _state(frontiers=fixed)
    pool = build_public_candidate_pool(fixed_state, _guard, candidate_limit=3)
    assert pool


def test_public_frontier_viewpoint_kind_is_explicit_and_task_consistent() -> None:
    observation = PublicFrontier("observation", (1.0, 0.0, 1.0), 1.0, 0.0)
    assert observation.viewpoint_kind == "observation"
    assert (
        PublicFrontier(
            "route-progress",
            (1.0, 0.0, 1.0),
            1.0,
            0.0,
            viewpoint_kind="route_progress",
        ).viewpoint_kind
        == "route_progress"
    )
    # Legacy outcome routes used the default field value. Preserve that call
    # shape while serializing the distinct recovery semantics explicitly.
    backtrack = PublicFrontier(
        "backtrack",
        (1.0, 0.0, 1.0),
        0.0,
        0.0,
        source_agent_id="uav0",
        task_kind="backtrack",
        exclusive_agent_id="uav0",
    )
    assert backtrack.viewpoint_kind == "outcome_backtrack"
    assert backtrack.to_dict()["viewpoint_kind"] == "outcome_backtrack"
    with pytest.raises(ValueError, match="invalid viewpoint kind"):
        PublicFrontier(
            "invalid-explore",
            (1.0, 0.0, 1.0),
            1.0,
            0.0,
            viewpoint_kind="outcome_backtrack",
        )
    with pytest.raises(ValueError, match="outcome-backed frontier"):
        PublicFrontier(
            "invalid-backtrack",
            (1.0, 0.0, 1.0),
            0.0,
            0.0,
            source_agent_id="uav0",
            task_kind="backtrack",
            exclusive_agent_id="uav0",
            viewpoint_kind="route_progress",
        )


def _transit_viewpoint_kinds(manifest) -> dict[str, str]:
    return {
        fragment.agent_id: str(
            dict(fragment.type_signature.public_features)["viewpoint_kind"]
        )
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    }


def test_primary_candidate_keeps_continuous_route_prefixes_with_observations() -> None:
    state = _state(
        frontiers=(
            # Deliberately poor gain/time scores: a route prefix is still a
            # valid primary exploration action when it is public and guarded.
            PublicFrontier("a-observation-uav0", (2.0, 0.0, 1.0), 0.01, 0.0),
            PublicFrontier("b-observation-uav1", (5.0, 0.0, 2.0), 0.01, 0.0),
            PublicFrontier(
                "c-prefix-uav0",
                (0.60, 0.0, 1.0),
                100.0,
                0.0,
                viewpoint_kind="route_progress",
            ),
            PublicFrontier(
                "d-prefix-uav1",
                (3.60, 0.0, 2.0),
                100.0,
                0.0,
                viewpoint_kind="route_progress",
            ),
        )
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=4,
    )

    assert "route_progress" in set(_transit_viewpoint_kinds(pool[0]).values())
    assert any(
        "route_progress" in set(_transit_viewpoint_kinds(manifest).values())
        for manifest in pool
    )
    assert all(
        dict(fragment.type_signature.public_features)["viewpoint_kind"]
        == _transit_viewpoint_kinds(pool[0])[fragment.agent_id]
        for fragment in pool[0].fragments
        if fragment.type_signature.fragment_type == "observation"
    )


def test_route_progress_remains_available_when_complete_observations_cannot_match() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("a-observation", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier(
                "b-prefix",
                (5.0, 0.0, 2.0),
                1.0,
                0.0,
                viewpoint_kind="route_progress",
            ),
        )
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
    )

    assert set(_transit_viewpoint_kinds(pool[0]).values()) == {
        "observation",
        "route_progress",
    }


def test_route_progress_is_admitted_only_after_joint_safety_rejects_observation_tier() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("a-observation-uav0", (2.0, 0.0, 1.0), 1.0, 0.0),
            PublicFrontier("b-observation-uav1", (5.0, 0.0, 2.0), 1.0, 0.0),
            PublicFrontier(
                "c-prefix-uav0",
                (0.60, 0.0, 1.0),
                1.0,
                0.0,
                viewpoint_kind="route_progress",
            ),
            PublicFrontier(
                "d-prefix-uav1",
                (3.60, 0.0, 2.0),
                1.0,
                0.0,
                viewpoint_kind="route_progress",
            ),
        )
    )

    def reject_complete_observation_team(manifest) -> str | None:
        kinds = set(_transit_viewpoint_kinds(manifest).values())
        return "synchronized_fleet_separation" if kinds == {"observation"} else None

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=4,
        joint_guard=reject_complete_observation_team,
        minimum_feasible_candidates=2,
    )

    feasible = tuple(manifest for manifest in pool if manifest.feasible)
    assert len(feasible) >= 2
    assert all(
        "route_progress" in set(_transit_viewpoint_kinds(manifest).values())
        for manifest in feasible
    )


def test_every_public_agent_requires_a_matching_context_agent():
    state = _state()
    with pytest.raises(ValueError, match="match PublicMethodContext"):
        PublicSearchState(
            context=state.context,
            agents=(PublicAgentPose("other", (0.0, 0.0, 1.0), 1.0, 0),),
            frontiers=state.frontiers,
            decision_start_s=0.0,
            decision_duration_s=1.0,
            transit_timing_model=ConservativeTransitTimingModel("unit-test", 1.0, 1.0, 0.0),
            observe_dwell_s=0.1,
        )


def test_delivered_frontier_provenance_does_not_lock_task_assignment():
    state = _state(
        frontiers=(
            PublicFrontier("uav0-r0", (1.0, 0.0, 1.0), 1.0, 0.1, "uav0"),
            PublicFrontier("uav0-r1", (1.0, 1.0, 1.0), 0.9, 0.1, "uav0"),
            PublicFrontier("uav1-r0", (4.0, 0.0, 2.0), 1.0, 0.1, "uav1"),
            PublicFrontier("uav1-r1", (4.0, 1.0, 2.0), 0.9, 0.1, "uav1"),
        )
    )
    pool = build_public_candidate_pool(state, _guard, candidate_limit=8)
    transits = tuple(
        (manifest, fragment)
        for manifest in pool
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    # The source field records who observed a public frontier; it is not a
    # permanent ownership constraint.  Cross-agent proposals remain in the
    # common denominator so every selector can allocate the same tasks.
    assert any(
        (fragment.agent_id == "uav1" and fragment.path[-1][0] == 1.0)
        or (fragment.agent_id == "uav0" and fragment.path[-1][0] == 4.0)
        for _, fragment in transits
    )

    # Cross-agent proposals pass through the same runtime guard before the
    # team pool is formed.  The blocked uav0 -> x=4 edge cannot leak into an
    # executable assignment, while legal cross-agent assignments remain.
    assert not any(
        fragment.agent_id == "uav0" and fragment.path[-1][0] == 4.0
        for _, fragment in transits
    )


def test_feasibility_first_matching_recovers_legal_edges_outside_old_prefix() -> None:
    state = replace(
        _state(
            frontiers=(
                PublicFrontier("frontier0", (1.0, 0.0, 1.0), 1.0, 0.1),
                PublicFrontier("frontier1", (2.0, 0.0, 1.0), 1.0, 0.1),
                PublicFrontier("frontier2", (3.0, 4.0, 1.0), 1.0, 0.1),
                PublicFrontier("frontier3", (4.0, 4.0, 1.0), 1.0, 0.1),
            )
        ),
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 0.9, 1),
            PublicAgentPose("uav1", (3.0, 4.0, 2.0), 0.7, 1),
        ),
    )

    def sparse_guard(agent_id, path_m):
        endpoint_x = path_m[-1][0]
        if path_m[0] == path_m[-1]:
            return GuardedPath(True, path_m, reason="stationary_hold")
        legal = (agent_id, endpoint_x) in {("uav0", 4.0), ("uav1", 3.0)}
        return GuardedPath(legal, path_m, reason="" if legal else "blocked")

    pool = build_public_candidate_pool(
        state,
        sparse_guard,
        candidate_limit=3,
        minimum_feasible_candidates=1,
    )

    assert len(pool) == 1
    assert all(row.feasible for row in pool)
    assert any(
        sum(
            dict(fragment.type_signature.public_features)["assignment_role"] == "explore"
            for fragment in row.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
        == 2
        for row in pool
    )


def test_route_tube_prefilter_rejects_crossing_team_routes() -> None:
    agents = (
        PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
        PublicAgentPose("uav1", (3.0, 0.0, 2.0), 1.0, 1),
    )
    crossing_edges = {
        (0, 0): GuardedPath(True, ((0.0, 0.0, 1.0), (4.0, 0.0, 1.0))),
        (1, 0): GuardedPath(True, ((3.0, 0.0, 2.0), (3.0, 0.0, 1.0))),
    }
    separated_edges = {
        (0, 0): GuardedPath(True, ((0.0, 0.0, 1.0), (4.0, 0.0, 1.0))),
        (1, 0): GuardedPath(True, ((3.0, 2.0, 2.0), (3.0, 2.0, 1.0))),
    }

    crossing = _assignment_route_tube_separation_m(crossing_edges, (0, 0), agents)
    separated = _assignment_route_tube_separation_m(separated_edges, (0, 0), agents)

    assert crossing < 0.5
    assert separated >= 0.5


def test_owned_frontier_emitter_offers_six_distinct_public_qd_intent_modes() -> None:
    state = _state(
        frontiers=(
            PublicFrontier("uav0-up", (1.0, 0.0, 3.0), 1.0, 0.1, "uav0"),
            PublicFrontier("uav0-down", (1.0, 1.0, 0.2), 0.9, 0.1, "uav0"),
            PublicFrontier("uav0-level", (3.0, 0.0, 1.0), 0.8, 0.1, "uav0"),
            PublicFrontier("uav0-far", (5.0, -1.0, 1.0), 0.7, 0.1, "uav0"),
            PublicFrontier("uav1-up", (4.0, 0.0, 4.0), 1.0, 0.1, "uav1"),
            PublicFrontier("uav1-down", (4.0, 1.0, 0.4), 0.9, 0.1, "uav1"),
            PublicFrontier("uav1-level", (6.0, 0.0, 2.0), 0.8, 0.1, "uav1"),
            PublicFrontier("uav1-far", (8.0, -1.0, 2.0), 0.7, 0.1, "uav1"),
        )
    )
    pool = build_public_candidate_pool(state, _guard, candidate_limit=8)

    audit = audit_public_candidate_intent_richness(pool)

    assert audit.status == "QD_CANDIDATE_INTENT_ADMITTED"
    assert audit.feasible_candidate_count >= 6
    assert audit.joint_effective_cells >= 6


def test_four_agent_assignment_search_is_bounded_with_many_legal_frontiers() -> None:
    context = PublicMethodContext(
        context_id="bounded-search",
        episode_id="bounded-search-episode",
        decision_id="bounded-search-decision",
        agent_features=tuple((f"uav{index}", (0.0, 0.0)) for index in range(4)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 100.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (float(index), 0.0, 1.0), 1.0, 1)
            for index in range(4)
        ),
        frontiers=tuple(
            PublicFrontier(
                f"frontier{index}",
                (float(index % 8), float(index // 8), 0.5 + 0.1 * (index % 6)),
                1.0 - 0.005 * index,
                0.05,
            )
            for index in range(64)
        ),
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "bounded-search", 10.0, 10.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    started = time.perf_counter()
    pool = build_public_candidate_pool(
        state,
        lambda _agent, path: GuardedPath(True, path),
        candidate_limit=8,
        minimum_feasible_candidates=8,
    )
    elapsed_s = time.perf_counter() - started

    assert len(pool) == 8
    assert all(candidate.feasible for candidate in pool)
    assert elapsed_s < 2.0


def test_common_pool_preserves_a_jointly_legal_long_route_extreme() -> None:
    """A bounded pool must expose a legal long option beside short gain rows.

    The route is intentionally low-gain and appears after many high-gain
    frontiers.  Before the route-extreme reservation, ``candidate_limit=2``
    filled with short assignments even though the four-agent long matching
    was individually and jointly legal.  Every selector receives this same
    row; no selector-specific route is synthesized here.
    """

    agent_count = 4
    context = PublicMethodContext(
        context_id="long-route-extreme",
        episode_id="long-route-extreme-episode",
        decision_id="long-route-extreme-decision",
        agent_features=tuple((f"uav{index}", (0.0,)) for index in range(agent_count)),
        public_features=(
            ("map_coverage", 0.1),
        ),
        budget=(("time_remaining_s", 100.0),),
    )
    high_gain_frontiers = tuple(
        PublicFrontier(
            f"high-gain-{index}",
            (float(index % 5) + 1.0, float(index // 5) * 0.1, 1.0),
            10.0,
            0.0,
        )
        for index in range(20)
    )
    long_frontiers = tuple(
        PublicFrontier(
            f"long-route-{index}",
            (50.0 + 10.0 * index, 0.0, 1.0),
            0.001,
            0.0,
        )
        for index in range(agent_count)
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(
                f"uav{index}",
                (2.0 * index, 0.0, 1.0),
                1.0,
                1,
            )
            for index in range(agent_count)
        ),
        frontiers=high_gain_frontiers + long_frontiers,
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "long-route-extreme", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
        joint_guard=lambda _manifest: None,
        minimum_feasible_candidates=2,
    )

    long_rows = []
    for manifest in pool:
        transit = tuple(
            fragment
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
        total_length = sum(
            math.dist(start, end)
            for fragment in transit
            for start, end in zip(fragment.path, fragment.path[1:], strict=False)
        )
        if total_length > 50.0:
            long_rows.append((manifest, transit, total_length))

    assert len(long_rows) == 1
    manifest, transit, total_length = long_rows[0]
    assert manifest.feasible
    assert len(transit) == agent_count
    assert total_length == pytest.approx(248.0)


def test_common_pool_prefers_separated_long_route_extreme_over_closer_winner() -> None:
    """Route-extreme enumeration must not spend slots on jointly illegal endpoints.

    The longest individual matching can place all four endpoints close together
    and therefore fail the endpoint separation contract.  The bounded enumerator
    should still expose a slightly shorter long-route matching whose endpoints
    are separated.
    """

    agent_count = 4
    context = PublicMethodContext(
        context_id="separated-long-route-extreme",
        episode_id="separated-long-route-extreme-episode",
        decision_id="separated-long-route-extreme-decision",
        agent_features=tuple((f"uav{index}", (0.0,)) for index in range(agent_count)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 100.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (2.0 * index, 0.0, 1.0), 1.0, 1)
            for index in range(agent_count)
        ),
        frontiers=tuple(
            PublicFrontier(f"close-{index}", (30.0, 0.0, 1.0), 0.001, 0.0)
            for index in range(agent_count)
        )
        + tuple(
            PublicFrontier(
                f"separated-{index}",
                (20.0 + 4.0 * index, 0.0, 1.0),
                0.001,
                0.0,
            )
            for index in range(agent_count)
        ),
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "separated-long-route-extreme", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    def endpoint_guard(manifest: CandidateFragmentManifest) -> str | None:
        endpoints = [
            tuple(fragment.path[-1])
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        ]
        if any(
            math.dist(left, right) < 0.95 - 1.0e-9
            for left_index, left in enumerate(endpoints)
            for right in endpoints[left_index + 1 :]
        ):
            return "planned_endpoint_separation_margin"
        return None

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=4,
        joint_guard=endpoint_guard,
        minimum_feasible_candidates=1,
        minimum_multi_agent_route_candidates=1,
    )

    long_feasible = []
    for manifest in pool:
        if not manifest.feasible:
            continue
        lengths = [
            sum(
                math.dist(left, right)
                for fragment in manifest.fragments
                if fragment.type_signature.fragment_type == "transit"
                for left, right in zip(
                    fragment.path, fragment.path[1:], strict=False
                )
            )
        ]
        if sum(lengths) >= 4.0:
            long_feasible.append(manifest)

    assert long_feasible


def test_pool_repairs_joint_conflict_into_a_feasible_multi_agent_long_route() -> None:
    """Numeric feasibility must not hide a one-long-route degenerate pool."""

    agent_count = 4
    context = PublicMethodContext(
        context_id="multi-agent-route-repair",
        episode_id="multi-agent-route-repair-episode",
        decision_id="multi-agent-route-repair-decision",
        agent_features=tuple((f"uav{index}", (0.0,)) for index in range(agent_count)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 100.0),),
    )
    high_gain_frontiers = tuple(
        PublicFrontier(
            f"high-gain-{index}",
            (float(index % 5) + 1.0, float(index // 5) * 0.1, 1.0),
            10.0,
            0.0,
        )
        for index in range(24)
    )
    long_frontiers = tuple(
        PublicFrontier(
            f"long-route-{index}",
            (50.0 + 10.0 * index, 0.0, 1.0),
            0.001,
            0.0,
        )
        for index in range(agent_count)
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(
                f"uav{index}",
                (2.0 * index, 0.0, 1.0),
                1.0,
                1,
            )
            for index in range(agent_count)
        ),
        frontiers=high_gain_frontiers + long_frontiers,
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "multi-agent-route-repair", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    def joint_guard(manifest: CandidateFragmentManifest) -> str | None:
        transit_lengths: list[float] = []
        delayed_agents: set[str] = set()
        for fragment in manifest.fragments:
            if fragment.type_signature.fragment_type != "transit":
                continue
            transit_lengths.append(
                sum(
                    math.dist(left, right)
                    for left, right in zip(
                        fragment.path, fragment.path[1:], strict=False
                    )
                )
            )
            features = dict(fragment.type_signature.public_features)
            if (
                fragment.agent_id in {"uav1", "uav3"}
                and float(features["traffic_reservation_delay_s"]) > 0.0
            ):
                delayed_agents.add(fragment.agent_id)
        meaningful = (
            sum(length >= 1.0 for length in transit_lengths) >= 2
            and sum(transit_lengths) >= 4.0
        )
        if meaningful and not delayed_agents:
            return "synchronized_fleet_separation"
        return None

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=4,
        joint_guard=joint_guard,
        minimum_feasible_candidates=4,
        minimum_multi_agent_route_candidates=1,
    )

    long_feasible = []
    for manifest in pool:
        if not manifest.feasible:
            continue
        transit = tuple(
            fragment
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
        lengths = [
            sum(
                math.dist(left, right)
                for left, right in zip(fragment.path, fragment.path[1:], strict=False)
            )
            for fragment in transit
        ]
        if sum(length >= 1.0 for length in lengths) >= 2 and sum(lengths) >= 4.0:
            long_feasible.append(manifest)

    assert long_feasible
    delayed = {
        fragment.agent_id
        for manifest in long_feasible[:1]
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
        and dict(fragment.type_signature.public_features)[
            "traffic_reservation_delay_s"
        ]
        > 0.0
    }
    assert delayed & {"uav1", "uav3"}


def test_common_pool_route_extreme_can_be_disabled_for_differential_audit() -> None:
    """The engineering switch restores the pre-route-extreme pool shape."""

    context = PublicMethodContext(
        context_id="route-extreme-differential",
        episode_id="route-extreme-differential-episode",
        decision_id="route-extreme-differential-decision",
        agent_features=tuple((f"uav{index}", (0.0,)) for index in range(4)),
        public_features=(("map_coverage", 0.1),),
        budget=(("time_remaining_s", 100.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (2.0 * index, 0.0, 1.0), 1.0, 1)
            for index in range(4)
        ),
        frontiers=tuple(
            PublicFrontier(
                f"frontier-{index}",
                (1.0 + float(index), 0.1 * (index % 4), 1.0),
                10.0 - float(index),
                0.0,
            )
            for index in range(4)
        )
        + tuple(
            PublicFrontier(
                f"long-route-{index}",
                (50.0 + 10.0 * index, 0.0, 1.0),
                0.001,
                0.0,
            )
            for index in range(4)
        ),
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "route-extreme-differential", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    pool = build_public_candidate_pool(
        state,
        lambda _agent_id, path: GuardedPath(True, path),
        candidate_limit=2,
        joint_guard=lambda _manifest: None,
        minimum_feasible_candidates=2,
        include_route_extreme=False,
    )

    transit_lengths = [
        sum(
            math.dist(start, end)
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
            for start, end in zip(fragment.path, fragment.path[1:], strict=False)
        )
        for manifest in pool
    ]
    assert len(pool) == 2
    assert all(length < 20.0 for length in transit_lengths)


def test_partial_active_route_extreme_recovers_when_all_agent_routes_jointly_illegal() -> None:
    """All-agent routes can be jointly unsafe while two/three long routes are safe.

    The pool must keep maximal-participation rows first, then explicitly add
    partial-active long-route rows labelled as waiting for team completion.
    It must not silently downgrade to three short collision-avoidance holds.
    """

    agent_count = 4
    context = PublicMethodContext(
        context_id="partial-route-extreme",
        episode_id="partial-route-extreme-episode",
        decision_id="partial-route-extreme-decision",
        agent_features=tuple((f"uav{index}", (0.0,)) for index in range(agent_count)),
        public_features=(("map_coverage", 0.1),),
        budget=((("time_remaining_s", 100.0),)),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (2.0 * index, 4.0 * index, 1.0), 1.0, 1)
            for index in range(agent_count)
        ),
        frontiers=tuple(
            PublicFrontier(f"close-{index}", (30.0, 4.0 * index, 1.0), 0.001, 0.0)
            for index in range(agent_count)
        )
        + tuple(
            PublicFrontier(
                f"separated-{index}",
                (20.0 + 4.0 * index, 4.0 * index + 2.0, 1.0),
                0.001,
                0.0,
            )
            for index in range(agent_count)
        ),
        decision_start_s=0.0,
        decision_duration_s=100.0,
        transit_timing_model=ConservativeTransitTimingModel(
            "partial-route-extreme", 2.0, 2.0, 0.0
        ),
        observe_dwell_s=0.5,
    )

    def joint_guard(manifest: CandidateFragmentManifest) -> str | None:
        roles = [
            dict(fragment.type_signature.public_features)["assignment_role"]
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        ]
        if all(role == "explore" for role in roles):
            return "synchronized_fleet_separation"
        return None

    pool = build_public_candidate_pool(
        state,
        identity_path_guard,
        candidate_limit=8,
        joint_guard=joint_guard,
        minimum_feasible_candidates=2,
        minimum_multi_agent_route_candidates=1,
    )

    recovered = [
        manifest
        for manifest in pool
        if manifest.feasible
        and _has_meaningful_multi_agent_routes(manifest)
        and any(
            dict(fragment.type_signature.public_features).get("hold_reason")
            == "waiting_for_team_completion"
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
    ]
    assert recovered

def test_frontier_3d_prefers_region_access_over_slightly_higher_local_gain() -> None:
    from aerocity_method.contracts.models import FragmentInstance, FragmentTypeSignature
    state = _state()

    def make_manifest(
        candidate_id: str,
        kind: str,
        gain: float,
        path: tuple[tuple[float, float, float], ...],
    ) -> CandidateFragmentManifest:
        features = (
            ('frontier_rank', 0),
            ('frontier_id', 'f'),
            ('viewpoint_kind', kind),
            ('assignment_role', 'explore'),
            ('task_kind', 'explore'),
            ('frontier_cluster_id', 'c'),
            ('expected_public_gain_proxy', gain),
            ('task_reservation_switch_cost', 0.0),
            ('vertical_delta_m', 0.0),
        )
        transit = FragmentInstance(
            instance_fragment_id=f'{candidate_id}-transit',
            type_signature=FragmentTypeSignature('transit', features),
            episode_id=state.context.episode_id,
            decision_id=state.context.decision_id,
            agent_id='uav0',
            planned_start=0.0,
            planned_end=1.0,
            path=path,
            pose_mode='guarded_waypoint',
            context_bucket='hm3d-public-frontier',
        )
        observation = FragmentInstance(
            instance_fragment_id=f'{candidate_id}-observe',
            type_signature=FragmentTypeSignature('observation', features),
            episode_id=state.context.episode_id,
            decision_id=state.context.decision_id,
            agent_id='uav0',
            planned_start=1.0,
            planned_end=1.5,
            path=(path[-1],),
            pose_mode='dwell',
            context_bucket='hm3d-public-frontier',
        )
        return CandidateFragmentManifest(
            candidate_id=candidate_id,
            context_hash=state.context.digest,
            fragments=(transit, observation),
            planned_descriptor=(0.0, 1.0, 0.0),
            feasible=True,
            quality_hint=gain,
            cost_hint=1.0,
            source='unit',
        )

    local = make_manifest(
        'local', 'observation', 0.3, ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0))
    )
    access = make_manifest(
        'access', 'region_access', 0.1, ((0.0, 0.0, 1.0), (3.0, 0.0, 1.0))
    )
    selected, selection = select_public_baseline('frontier_3d', (local, access))
    assert selected.candidate_id == 'access'
    assert dict(selection.scores)['access'] > dict(selection.scores)['local']
