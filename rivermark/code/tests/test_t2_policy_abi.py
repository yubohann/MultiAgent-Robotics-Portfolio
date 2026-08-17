from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.isaac_transfer import FixedDecisionCadence, WorldCommandBounds
from rivermark_benchmark.t2_policy_abi import (
    T2_CLAIM_BOUNDARY,
    T2CandidateDetection,
    T2CandidateEventJournal,
    T2NativeStepEvidence,
    T2PolicyAbiError,
    T2PolicyRunner,
    T2PublicFleetObservation,
    T2PublicSensorObservation,
)


def _observation(*, physics_step: int = 4, command_time_ns: int = 20_000_000) -> T2PublicFleetObservation:
    return T2PublicFleetObservation.from_rigid_body_state(
        physics_step=physics_step,
        command_time_ns=command_time_ns,
        position_w_m=np.column_stack((np.arange(8, dtype=np.float64), np.zeros(8), np.ones(8))),
        linear_velocity_w_mps=np.zeros((8, 3), dtype=np.float64),
        quaternion_wxyz=np.tile(np.array((1.0, 0.0, 0.0, 0.0)), (8, 1)),
        angular_velocity_b_radps=np.zeros((8, 3), dtype=np.float64),
    )


def _decision() -> object:
    runner = T2PolicyRunner(
        lambda observation: np.tile(np.array((9.0, -9.0, 9.0, 9.0)), (8, 1)),
        cadence=FixedDecisionCadence(2),
        bounds=WorldCommandBounds(
            max_horizontal_speed_mps=2.0,
            max_vertical_speed_mps=1.0,
            max_yaw_rate_rad_s=0.5,
        ),
    )
    return runner.decide(_observation())


def test_public_observation_is_canonical_and_hash_bound() -> None:
    observation = _observation()
    payload = observation.public_dict()
    assert payload["claim_boundary"] == T2_CLAIM_BOUNDARY
    assert payload["state_fields"] == [
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_mps",
        "velocity_y_mps",
        "velocity_z_mps",
        "yaw_rad",
        "yaw_rate_radps",
    ]
    assert len(observation.sha256) == 64
    assert "target" not in payload
    assert "private" not in payload


def test_runner_applies_world_command_bounds_and_fixed_cadence() -> None:
    decision = _decision()
    emitted = decision.action.emitted_velocity_yaw_command
    assert np.allclose(np.linalg.norm(emitted[:, :2], axis=1), 2.0)
    assert np.allclose(emitted[:, 2], 1.0)
    assert np.allclose(emitted[:, 3], 0.5)
    assert decision.public_dict()["command_before_step"] is True
    with pytest.raises(T2PolicyAbiError, match="not on the fixed decision cadence"):
        T2PolicyRunner(lambda _: np.zeros((8, 4)), cadence=FixedDecisionCadence(2)).decide(
            _observation(physics_step=3)
        )


def test_non_finite_or_wrong_shape_policy_action_fails_closed() -> None:
    observation = _observation()
    for action in (np.zeros((7, 4)), np.full((8, 4), np.nan)):
        runner = T2PolicyRunner(lambda _, action=action: action, cadence=FixedDecisionCadence(2))
        with pytest.raises(T2PolicyAbiError):
            runner.decide(observation)


def test_candidate_event_journal_binds_public_observation_and_rejects_duplicates() -> None:
    journal = T2CandidateEventJournal(episode_id="t2-canary-001")
    sensor = T2PublicSensorObservation(
        agent_id=2,
        capture_frame_index=7,
        sensor_time_ns=25_000_000,
    )
    events = journal.append(
        sensor,
        [T2CandidateDetection(agent_id=2, position_w_m=(3.0, 4.0, 2.0), confidence=0.8)],
    )
    assert events[0]["source_observation_id"] == "obs-a02-f00000007"
    assert events[0]["timestamp_s"] == 0.025
    submission = journal.submission()
    assert submission["episode_id"] == "t2-canary-001"
    assert "target" not in str(submission).lower()
    with pytest.raises(T2PolicyAbiError, match="must match"):
        journal.append(
            sensor,
            [
                T2CandidateDetection(agent_id=1, position_w_m=(3.0, 4.0, 2.0), confidence=0.8),
            ],
        )


def test_candidate_event_journal_uses_the_frozen_rollout_time_origin() -> None:
    journal = T2CandidateEventJournal(
        episode_id="t2-canary-001", event_time_origin_ns=20_000_000
    )
    sensor = T2PublicSensorObservation(
        agent_id=0, capture_frame_index=0, sensor_time_ns=25_000_000
    )
    event = journal.append(
        sensor,
        [T2CandidateDetection(agent_id=0, position_w_m=(1.0, 2.0, 3.0), confidence=0.5)],
    )[0]
    assert event["timestamp_s"] == 0.005
    assert journal.public_dict()["event_time_origin_ns"] == 20_000_000
    with pytest.raises(T2PolicyAbiError, match="precedes"):
        journal.append(
            T2PublicSensorObservation(
                agent_id=0, capture_frame_index=1, sensor_time_ns=19_999_999
            ),
            [],
        )


def test_native_step_evidence_requires_actual_post_step_causality() -> None:
    decision = _decision()
    evidence = T2NativeStepEvidence(
        decision=decision,
        applied_physics_step=5,
        physical_command_time_ns=20_000_000,
        effective_time_ns=25_000_000,
        requested_thrust_n=np.full((8, 4), 0.09),
        applied_thrust_n=np.full((8, 4), 0.07),
        applied_wrench_body=np.tile(np.array((0.0, 0.0, 0.28, 0.0, 0.0, 0.0)), (8, 1)),
        post_step_state_8d=decision.observation.state.values + 0.001,
    )
    payload = evidence.public_dict()
    assert payload["decision_command_time_ns"] == decision.observation.command_time_ns
    assert payload["physical_command_time_ns"] == decision.observation.command_time_ns
    assert payload["effective_time_ns"] > payload["physical_command_time_ns"]
    assert payload["decision_sha256"] == decision.sha256
    with pytest.raises(T2PolicyAbiError, match="after physical_command_time_ns"):
        T2NativeStepEvidence(
            decision=decision,
            applied_physics_step=5,
            physical_command_time_ns=20_000_000,
            effective_time_ns=decision.observation.command_time_ns,
            requested_thrust_n=np.full((8, 4), 0.09),
            applied_thrust_n=np.full((8, 4), 0.07),
            applied_wrench_body=np.zeros((8, 6)),
            post_step_state_8d=decision.observation.state.values,
        )
    with pytest.raises(T2PolicyAbiError, match="must not precede"):
        T2NativeStepEvidence(
            decision=decision,
            applied_physics_step=decision.observation.physics_step - 1,
            physical_command_time_ns=20_000_000,
            effective_time_ns=25_000_000,
            requested_thrust_n=np.full((8, 4), 0.09),
            applied_thrust_n=np.full((8, 4), 0.07),
            applied_wrench_body=np.zeros((8, 6)),
            post_step_state_8d=decision.observation.state.values,
        )


def test_native_step_evidence_separates_held_decision_from_physical_command_time() -> None:
    decision = _decision()
    evidence = T2NativeStepEvidence(
        decision=decision,
        applied_physics_step=9,
        physical_command_time_ns=40_000_000,
        effective_time_ns=45_000_000,
        requested_thrust_n=np.full((8, 4), 0.09),
        applied_thrust_n=np.full((8, 4), 0.07),
        applied_wrench_body=np.tile(np.array((0.0, 0.0, 0.28, 0.0, 0.0, 0.0)), (8, 1)),
        post_step_state_8d=decision.observation.state.values + 0.001,
    )
    payload = evidence.public_dict()
    assert payload["decision_command_time_ns"] == decision.observation.command_time_ns
    assert payload["physical_command_time_ns"] == 40_000_000
    assert payload["effective_time_ns"] == 45_000_000
    with pytest.raises(T2PolicyAbiError, match="must not precede decision"):
        T2NativeStepEvidence(
            decision=decision,
            applied_physics_step=9,
            physical_command_time_ns=19_999_999,
            effective_time_ns=25_000_000,
            requested_thrust_n=np.full((8, 4), 0.09),
            applied_thrust_n=np.full((8, 4), 0.07),
            applied_wrench_body=np.zeros((8, 6)),
            post_step_state_8d=decision.observation.state.values,
        )


def test_candidate_event_journal_never_accepts_evaluator_owned_fields() -> None:
    journal = T2CandidateEventJournal(episode_id="t2-canary-001")
    sensor = T2PublicSensorObservation(
        agent_id=0,
        capture_frame_index=8,
        sensor_time_ns=30_000_000,
    )
    journal.append(
        sensor,
        [T2CandidateDetection(agent_id=0, position_w_m=(1.0, 2.0, 3.0), confidence=0.5)],
    )
    submission = journal.submission()
    malicious = copy.deepcopy(submission)
    malicious["events"][0]["target_id"] = "forbidden"
    with pytest.raises(T2PolicyAbiError, match="candidate event journal"):
        from rivermark_benchmark.search_event_evaluator import parse_candidate_events

        try:
            parse_candidate_events(malicious, expected_episode_id="t2-canary-001", agent_count=8)
        except ValueError as exc:
            raise T2PolicyAbiError(f"candidate event journal is invalid: {exc}") from exc


def test_candidate_event_journal_does_not_serialize_capture_local_deduplication_key() -> None:
    journal = T2CandidateEventJournal(episode_id="t2-canary-001")
    sensor = T2PublicSensorObservation(agent_id=0, capture_frame_index=8, sensor_time_ns=30_000_000)
    journal.append(
        sensor,
        [
            T2CandidateDetection(
                agent_id=0,
                position_w_m=(1.0, 2.0, 3.0),
                confidence=0.5,
                deduplication_key="search_target_slot_000",
            )
        ],
    )
    serialized = journal.submission()
    assert "deduplication_key" not in serialized["events"][0]
    assert "search_target_slot" not in str(serialized)
