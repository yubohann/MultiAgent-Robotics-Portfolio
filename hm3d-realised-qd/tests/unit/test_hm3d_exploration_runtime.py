from __future__ import annotations

import pytest

from aerocity_method.runtime.hm3d_belief import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    PublicRangeRayOutcome,
    SparseVoxelBelief,
    public_free_voxel_transition,
)
from aerocity_method.runtime.hm3d_candidates import (
    ExplorationCandidateBudget,
    PublicExplorationAgentState,
    build_exploration_candidate_pool,
)
from aerocity_method.runtime.hm3d_frontiers import (
    FrontierExtractionConfig,
    extract_frontier_clusters,
)
from aerocity_method.runtime.hm3d_trajectory import (
    TrajectoryTimingConfig,
    plan_continuous_trajectory,
)


def _belief() -> SparseVoxelBelief:
    belief = SparseVoxelBelief("scene0", "uav0", 1.0)
    belief.integrate_ray(
        PublicRangeRayOutcome(
            "obs0",
            "uav0",
            0.0,
            (0.1, 0.1, 0.1),
            (3.1, 0.1, 0.1),
            hit_occupied=True,
        )
    )
    return belief


def test_sparse_belief_ray_replay_is_idempotent_and_occupied_wins():
    belief = _belief()
    before = belief.content_sha256
    assert (
        belief.integrate_ray(
            PublicRangeRayOutcome(
                "obs0",
                "uav0",
                0.0,
                (0.1, 0.1, 0.1),
                (3.1, 0.1, 0.1),
                hit_occupied=True,
            )
        )
        is False
    )
    assert belief.content_sha256 == before
    assert belief.state((0, 0, 0)) == FREE
    assert belief.state((3, 0, 0)) == OCCUPIED


def test_public_free_voxel_transition_accounts_for_occupied_revisions():
    newly_free, revised_away = public_free_voxel_transition(
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (2, 0, 0), (3, 0, 0)),
    )

    assert newly_free == frozenset({(2, 0, 0), (3, 0, 0)})
    assert revised_away == frozenset({(0, 0, 0)})
    assert 3 - 2 == len(newly_free) - len(revised_away)


def test_frontier_extraction_finds_vertical_unknown_boundaries():
    belief = _belief()
    clusters = extract_frontier_clusters(belief)
    assert clusters
    assert any(
        belief.state((key[0], key[1], key[2] + dz)) == UNKNOWN
        for cluster in clusters
        for viewpoint in cluster.viewpoint_candidates_m
        for key in (belief.world_to_voxel(viewpoint),)
        for dz in (-1, 1)
    )
    assert all(cluster.expected_gain_m3 > 0.0 for cluster in clusters)


def test_frontier_viewpoint_budget_represents_cluster_extent() -> None:
    belief = SparseVoxelBelief("scene", "uav0", 0.25)
    for index in range(1, 21):
        belief.integrate_ray(
            PublicRangeRayOutcome(
                observation_id=f"ray-{index}",
                agent_id="uav0",
                timestamp_s=float(index),
                origin_m=(0.125, 0.125, 0.125),
                endpoint_m=(index * 0.25 + 0.125, 0.125, 0.125),
                hit_occupied=False,
            )
        )

    clusters = extract_frontier_clusters(
        belief,
        config=FrontierExtractionConfig(max_viewpoints_per_cluster=4),
    )
    largest = max(clusters, key=lambda cluster: cluster.unknown_voxel_count)
    xs = [point[0] for point in largest.viewpoint_candidates_m]

    assert len(xs) == 4
    assert max(xs) - min(xs) >= 4.0


def test_continuous_trajectory_settles_at_intermediate_waypoints():
    trajectory = plan_continuous_trajectory(
        "uav0",
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)),
        start_time_s=2.0,
        config=TrajectoryTimingConfig(1.0, 2.0, tracking_margin_s=0.2),
    )
    assert trajectory.intermediate_stops == 1
    assert trajectory.waypoints[-1].stop_required is True
    assert trajectory.duration_s > 2.0


def test_exploration_candidates_use_public_frontiers_and_reject_private_shortcuts():
    belief = _belief()
    clusters = extract_frontier_clusters(belief)
    pool = build_exploration_candidate_pool(
        context_payload={"episode_id": "episode0", "decision_id": "decision0"},
        belief_version_sha256s=(belief.version().digest,),
        agents=(
            PublicExplorationAgentState("uav0", (0.0, 0.0, 1.0), 100.0, 1),
            PublicExplorationAgentState("uav1", (1.0, 0.0, 1.0), 100.0, 1),
        ),
        frontiers=clusters,
        budget=ExplorationCandidateBudget(0.0, 30.0),
        timing=TrajectoryTimingConfig(1.5, 2.0),
        candidate_limit=2,
    )
    assert any(candidate.feasible for candidate in pool)
    serialized = repr([candidate.to_dict() for candidate in pool]).casefold()
    assert "target" not in serialized
    assert "private" not in serialized


def test_candidate_generation_fails_closed_when_all_guards_reject():
    belief = _belief()
    clusters = extract_frontier_clusters(belief)
    with pytest.raises(ValueError, match="no feasible"):
        build_exploration_candidate_pool(
            context_payload={"episode_id": "episode0", "decision_id": "decision0"},
            belief_version_sha256s=(belief.version().digest,),
            agents=(PublicExplorationAgentState("uav0", (0.0, 0.0, 1.0), 100.0, 1),),
            frontiers=clusters,
            budget=ExplorationCandidateBudget(0.0, 30.0),
            timing=TrajectoryTimingConfig(1.5, 2.0),
            guard=lambda _agent_id, path: (False, path, "blocked_by_public_guard"),
        )
