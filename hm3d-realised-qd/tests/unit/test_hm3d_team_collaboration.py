from __future__ import annotations

from aerocity_method.runtime.hm3d_team_collaboration import (
    audit_translation_invariant_team_trajectories,
)


def test_translated_copies_are_rejected_even_when_world_positions_differ() -> None:
    audit = audit_translation_invariant_team_trajectories(
        {
            "uav0": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 0.0, 0.2)),
            "uav1": ((4.0, -3.0, 2.0), (4.5, -3.0, 2.1), (5.0, -3.0, 2.2)),
        },
        scope="planned",
    )

    assert audit.status == "TEAM_TRAJECTORY_DIVERSITY_NOT_ADMITTED"
    assert audit.duplicate_pair_agent_ids == (("uav0", "uav1"),)
    assert audit.pair_audits[0].translated_rmse_m < 1.0e-12


def test_different_directions_are_admitted_after_translation_alignment() -> None:
    audit = audit_translation_invariant_team_trajectories(
        {
            "uav0": ((0.0, 0.0, 0.0), (0.8, 0.0, 0.2)),
            "uav1": ((4.0, -3.0, 2.0), (4.0, -2.2, 1.8)),
            "uav2": ((1.0, 2.0, 1.0), (0.4, 2.0, 1.1)),
        },
        scope="realised_physx",
    )

    assert audit.status == "TEAM_TRAJECTORY_DIVERSITY_ADMITTED"
    assert not audit.has_translated_duplicate
    assert len(audit.pair_audits) == 3


def test_stationary_agent_is_not_mistaken_for_a_copied_explorer() -> None:
    audit = audit_translation_invariant_team_trajectories(
        {
            "uav0": ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
            "uav1": ((2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            "uav2": ((0.0, 2.0, 0.0), (0.0, 2.8, 0.0)),
        },
        roles_by_agent={"uav0": "explore", "uav1": "hold", "uav2": "explore"},
        scope="planned",
    )

    assert audit.status == "TEAM_TRAJECTORY_DIVERSITY_ADMITTED"
    assert audit.excluded_from_explorer_pair_audit_agent_ids == ("uav1",)
    assert len(audit.pair_audits) == 1


def test_one_explorer_is_reported_as_unobservable_not_diverse() -> None:
    audit = audit_translation_invariant_team_trajectories(
        {
            "uav0": ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
            "uav1": ((2.0, 0.0, 0.0),),
            "uav2": ((0.0, 2.0, 0.0),),
            "uav3": ((1.0, 2.0, 0.0),),
        },
        roles_by_agent={
            "uav0": "explore",
            "uav1": "hold",
            "uav2": "hold",
            "uav3": "hold",
        },
        scope="realised_physx",
    )

    assert audit.status == "TEAM_TRAJECTORY_DIVERSITY_UNOBSERVABLE"
    assert audit.reasons == ("FEWER_THAN_TWO_MOVING_EXPLORERS",)
    assert not audit.has_translated_duplicate
