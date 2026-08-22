from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentOutcome,
    FragmentTypeSignature,
    PublicMethodContext,
)
from aerocity_method.runtime.tokens import authorize_manifest


def _context() -> PublicMethodContext:
    return PublicMethodContext(
        context_id="hm3d-exploration-test-context",
        episode_id="hm3d-exploration-test-episode",
        decision_id="hm3d-exploration-test-decision",
        agent_features=(
            ("uav-1", (0.0, 0.0, 1.0)),
            ("uav-2", (1.0, 0.0, 1.0)),
        ),
        public_features=(
            ("confirmed_free_volume_m3", 0.0),
            ("remaining_physical_time_s", 10.0),
        ),
        preferences=(("energy", 0.1), ("explored_free_volume", 1.0)),
        budget=(("planner_calls_remaining", 8.0),),
    )


def _manifest(
    context: PublicMethodContext,
    candidate_id: str,
    agent_id: str,
    x_offset: float,
    descriptor: tuple[float, float],
    quality: float,
) -> CandidateFragmentManifest:
    transit = FragmentInstance(
        instance_fragment_id=f"{candidate_id}-transit",
        type_signature=FragmentTypeSignature("transit", (("vertical_mode", "free_height"),)),
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id=agent_id,
        planned_start=0.0,
        planned_end=1.0,
        path=((x_offset, 0.0, 1.0), (x_offset + 1.0, 0.0, 1.2)),
        pose_mode="clearance_guarded",
    )
    observation = FragmentInstance(
        instance_fragment_id=f"{candidate_id}-sparse-range",
        type_signature=FragmentTypeSignature("observation", (("sensor", "sparse_range_3d"),)),
        episode_id=context.episode_id,
        decision_id=context.decision_id,
        agent_id=agent_id,
        planned_start=1.0,
        planned_end=2.0,
        path=((x_offset + 1.0, 0.0, 1.2),),
        pose_mode="range_scan",
    )
    return CandidateFragmentManifest(
        candidate_id=candidate_id,
        context_hash=context.digest,
        fragments=(transit, observation),
        planned_descriptor=descriptor,
        feasible=True,
        quality_hint=quality,
        cost_hint=0.1 + x_offset * 0.01,
        source="test_sparse_range_emitter",
    )


@pytest.fixture
def context():
    return _context()


@pytest.fixture
def manifests(context):
    return (
        _manifest(context, "candidate-A", "uav-1", 0.0, (0.2, 0.2), 0.8),
        _manifest(context, "candidate-B", "uav-2", 2.0, (0.8, 0.8), 0.9),
        _manifest(context, "candidate-C", "uav-1", 4.0, (0.2, 0.8), 0.7),
    )


@pytest.fixture
def token(context, manifests):
    return authorize_manifest(
        context,
        manifests,
        (True, False, False),
        0,
        token_id="token-A",
        issued_at=0.0,
        duration=2.0,
    )


@pytest.fixture
def outcomes(manifests, token):
    manifest = manifests[0]
    rows = []
    for index, planned in enumerate(manifest.fragments):
        is_observation = planned.type_signature.fragment_type == "observation"
        source_id = "public-range-frame-A" if is_observation else None
        applied = replace(planned, executed=True, source_observation_id=source_id)
        rows.append(
            FragmentOutcome(
                outcome_id=f"outcome-A-{index}",
                token_hash=token.digest,
                manifest_hash=manifest.manifest_hash,
                episode_id=planned.episode_id,
                decision_id=planned.decision_id,
                agent_id=planned.agent_id,
                planned_fragment_hash=planned.digest,
                executed=True,
                actual_start=planned.planned_start,
                actual_end=planned.planned_end,
                applied_fragment=applied,
                outcome_fields=(("explored_free_volume_m3", 0.2 + index),),
                cost_fields=(("energy", 0.1),),
                source_observation_id=source_id,
                source_observation_episode_id=(planned.episode_id if is_observation else None),
                source_observation_agent_id=(planned.agent_id if is_observation else None),
                range_ok=True if is_observation else None,
                fov_ok=True if is_observation else None,
                los_ok=True if is_observation else None,
                orientation_ok=True if is_observation else None,
                dwell_ok=True if is_observation else None,
            )
        )
    return tuple(rows)
