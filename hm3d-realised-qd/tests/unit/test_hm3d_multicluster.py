from copy import deepcopy

import pytest

from aerocity_method.runtime.hm3d_multicluster import (
    HM3DClusterLayout,
    audit_reference_cluster_invariance,
    cluster_seed,
    ordered_qd_outcome_updates,
    validate_cluster_start_sets,
)


def test_layout_round_trips_local_world_and_flattens_four_agent_clusters() -> None:
    layout = HM3DClusterLayout(((0.0, 0.0, 0.0), (100.0, -50.0, 2.0)))
    assert layout.cluster_count == 2
    assert layout.total_agent_count == 8
    assert layout.flat_agent_index(1, 3) == 7
    local = (1.5, 2.0, 3.0)
    assert layout.to_local(1, layout.to_world(1, local)) == pytest.approx(local)
    flat = [(0.0, 0.0, 0.0)] * 4 + [layout.to_world(1, local)] * 4
    assert all(
        row == pytest.approx(local) for row in layout.local_team_from_flat_world(1, flat)
    )


def test_cluster_random_streams_are_stable_and_independent() -> None:
    first = cluster_seed(scene_id="scene", cluster_id=0, episode_id="episode", base_seed=17)
    assert first == cluster_seed(
        scene_id="scene", cluster_id=0, episode_id="episode", base_seed=17
    )
    assert first != cluster_seed(
        scene_id="scene", cluster_id=1, episode_id="episode", base_seed=17
    )


def test_vectorized_training_rejects_identical_start_sets_after_agent_permutation() -> None:
    first = ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (2.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="distinct local start sets"):
        validate_cluster_start_sets((first, tuple(reversed(first))))


def test_isolation_probe_can_explicitly_reuse_one_start_set() -> None:
    first = ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (2.0, 2.0, 1.0))
    result = validate_cluster_start_sets(
        (first, first), allow_identical_for_isolation_probe=True
    )
    assert result == (first, first)


def test_qd_outcomes_merge_in_pre_registered_order() -> None:
    rows = [
        {"scene_id": "b", "cluster_id": 0, "episode_id": "e", "decision_id": "d0"},
        {"scene_id": "a", "cluster_id": 1, "episode_id": "e", "decision_id": "d0"},
        {"scene_id": "a", "cluster_id": 0, "episode_id": "e", "decision_id": "d1"},
    ]
    ordered = ordered_qd_outcome_updates(rows)
    assert [(row["scene_id"], row["cluster_id"]) for row in ordered] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]


def test_reference_cluster_invariance_separates_peer_behavior_from_leakage() -> None:
    reference = {
        "selected_candidate_ids": ["a"],
        "action_hashes": ["1"],
        "outcome_hashes": ["2"],
        "local_root_trace_m": [[[0.0, 0.0, 1.0]]],
        "cross_cluster_contact_count": 0,
        "cross_cluster_message_count": 0,
        "cross_cluster_map_delta_count": 0,
    }
    peer_changed = deepcopy(reference)
    peer_changed["local_root_trace_m"] = [[[0.0, 0.0, 1.0 + 1.0e-7]]]
    assert audit_reference_cluster_invariance(
        reference, peer_changed, tolerance_m=1.0e-6
    )["passed"] is True
    peer_changed["cross_cluster_message_count"] = 1
    assert audit_reference_cluster_invariance(
        reference, peer_changed, tolerance_m=1.0e-6
    )["passed"] is False
