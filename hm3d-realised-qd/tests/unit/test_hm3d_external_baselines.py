from __future__ import annotations

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    build_public_candidate_pool,
    identity_path_guard,
)
from aerocity_method.adapters.hm3d_external_baselines import select_gvp_mrep_port
from aerocity_method.contracts.models import PublicMethodContext


def _state() -> PublicSearchState:
    context = PublicMethodContext(
        context_id="gvp-context",
        episode_id="gvp-episode",
        decision_id="decision0",
        agent_features=(("uav0", (1.0, 1.0)), ("uav1", (1.0, 1.0))),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", 40.0),),
    )
    return PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
            PublicAgentPose("uav1", (4.0, 0.0, 1.0), 1.0, 1),
        ),
        frontiers=(
            PublicFrontier("uav0-low", (1.0, 0.0, 1.0), 0.5, 0.1, "uav0"),
            PublicFrontier("uav0-high", (1.0, 0.0, 3.0), 1.0, 0.1, "uav0"),
            PublicFrontier("uav1-low", (5.0, 0.0, 1.0), 0.5, 0.1, "uav1"),
            PublicFrontier("uav1-high", (5.0, 0.0, 3.0), 1.0, 0.1, "uav1"),
        ),
        decision_start_s=0.0,
        decision_duration_s=20.0,
        transit_timing_model=ConservativeTransitTimingModel("unit", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
        communication_range_m=8.0,
    )


def test_gvp_port_selects_only_a_legal_common_pool_candidate_and_records_graph_diagnostics(
) -> None:
    state = _state()
    pool = build_public_candidate_pool(state, identity_path_guard, candidate_limit=4)
    selected, selection = select_gvp_mrep_port(state, pool)
    payload = selection.to_dict()

    assert selected.feasible
    assert selected.manifest_hash == payload["selected_manifest_hash"]
    assert payload["strategy"] == "gvp_mrep_port"
    assert payload["adaptation_status"] == "controlled_transfer_not_original_ros_reproduction"
    assert payload["author_source"]["commit"] == "f5865b9c9c39e9d85095555f3e04b4fa349fce40"
    diagnostics = payload["selected_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["public_graph_node_count"] >= 4
    assert diagnostics["public_graph_edge_count"] >= 1
    assert diagnostics["author_parameters"] == {
        "lambda": 0.2,
        "allowance": 0.1,
        "tau": 0.3,
    }
    assert diagnostics["author_graph_partition_sha256"] == (
        "9eb02ce91f6e49184b224649ab6d6563139a82de58031ba5a60797ff36cbc846"
    )
    serialized = repr(payload).casefold()
    assert "target_id" not in serialized
    assert "target_position" not in serialized


def test_gvp_port_handles_four_agents_and_real_vertical_frontier_assignments() -> None:
    context = PublicMethodContext(
        context_id="gvp-4uav-context",
        episode_id="gvp-4uav-episode",
        decision_id="decision0",
        agent_features=tuple((f"uav{index}", (1.0, 1.0)) for index in range(4)),
        public_features=(("sparse_range_schedule_hz", 10.0),),
        budget=(("time_remaining_s", 40.0),),
    )
    state = PublicSearchState(
        context=context,
        agents=tuple(
            PublicAgentPose(f"uav{index}", (4.0 * index, 0.0, 1.0), 1.0, 1)
            for index in range(4)
        ),
        # Every currently observed frontier is above the launch band.  This
        # checks that the controlled transfer can allocate genuine 3-D tasks;
        # it does not force an ascent when a closer planar frontier is better.
        frontiers=tuple(
            PublicFrontier(
                f"uav{index}-upper",
                (4.0 * index + 1.0, 0.0, 3.0 + index),
                1.0,
                0.1,
                f"uav{index}",
            )
            for index in range(4)
        ),
        decision_start_s=0.0,
        decision_duration_s=20.0,
        transit_timing_model=ConservativeTransitTimingModel("unit", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
        communication_range_m=20.0,
    )
    pool = build_public_candidate_pool(
        state,
        identity_path_guard,
        candidate_limit=8,
        minimum_feasible_candidates=2,
    )
    selected, selection = select_gvp_mrep_port(state, pool)
    transits = tuple(
        fragment
        for fragment in selected.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    diagnostics = selection.selected_diagnostics

    assert len(transits) == 4
    assert {fragment.agent_id for fragment in transits} == {f"uav{index}" for index in range(4)}
    assert any(abs(fragment.path[-1][2] - fragment.path[0][2]) > 1.0 for fragment in transits)
    assert len(diagnostics["assigned_frontiers"]) == 4
    assert diagnostics["public_graph_node_count"] >= 8
    assert diagnostics["predicted_connected_agent_fraction"] == 1.0


def test_candidate_builder_skips_route_queries_that_fail_the_direct_time_lower_bound() -> None:
    state = _state()
    short_window_state = PublicSearchState(
        context=state.context,
        agents=state.agents,
        frontiers=state.frontiers
        + (
            PublicFrontier("far-0", (50.0, 0.0, 1.0), 1.0, 0.1, "uav0"),
            PublicFrontier("far-1", (-50.0, 0.0, 1.0), 1.0, 0.1, "uav1"),
        ),
        decision_start_s=0.0,
        decision_duration_s=2.0,
        transit_timing_model=ConservativeTransitTimingModel("short-window", 4.0, 4.0, 0.0),
        observe_dwell_s=0.5,
        communication_range_m=state.communication_range_m,
    )
    queried_endpoints: list[tuple[float, float, float]] = []

    def counting_guard(agent_id, path_m):
        del agent_id
        queried_endpoints.append(path_m[-1])
        return identity_path_guard("guarded-agent", path_m)

    pool = build_public_candidate_pool(
        short_window_state,
        counting_guard,
        candidate_limit=2,
    )

    assert pool
    assert (50.0, 0.0, 1.0) not in queried_endpoints
    assert (-50.0, 0.0, 1.0) not in queried_endpoints
