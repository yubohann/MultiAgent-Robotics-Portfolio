from __future__ import annotations

from dataclasses import dataclass

import pytest

from aerocity_method.adapters.hm3d_baselines import (
    ConservativeTransitTimingModel,
    PublicAgentPose,
    PublicFrontier,
    PublicSearchState,
    build_public_candidate_pool,
    identity_path_guard,
)
from aerocity_method.adapters.hm3d_execution import (
    FragmentExecutionSample,
    execute_hm3d_manifest,
)
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import (
    ActionToken,
    CandidateFragmentManifest,
    PublicMethodContext,
)
from aerocity_method.evaluation.hm3d_safety import (
    TimedPolyline,
    assess_synchronized_separation,
)
from aerocity_method.runtime import hm3d_cf2x_execution as cf2x
from aerocity_method.runtime.tokens import authorize_manifest


def _context() -> PublicMethodContext:
    return PublicMethodContext(
        context_id="execution-context",
        episode_id="execution-episode",
        decision_id="execution-decision",
        agent_features=(("uav0", (0.0,)), ("uav1", (0.0,))),
    )


def test_joint_route_reserves_two_vehicle_tracking_envelopes() -> None:
    """An ideal 0.70 m pass is not enough for two 0.20 m tracking envelopes."""

    assert cf2x.PLANNED_INTER_AGENT_SEPARATION_M == pytest.approx(0.90)
    assert cf2x.PLANNED_INTER_AGENT_SEPARATION_M == pytest.approx(
        cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M
        + 2.0 * cf2x.TRACKING_CLEARANCE_MARGIN_M
    )
    routes = (
        TimedPolyline("uav0", ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)), 0.0, 1.0),
        TimedPolyline("uav1", ((0.0, 0.70, 1.0), (1.0, 0.70, 1.0)), 0.0, 1.0),
    )

    assert assess_synchronized_separation(
        routes,
        minimum_separation_m=cf2x.CF2X_MIN_INTER_AGENT_SEPARATION_M,
    ).admitted
    assert not assess_synchronized_separation(
        routes,
        minimum_separation_m=cf2x.PLANNED_INTER_AGENT_SEPARATION_M,
    ).admitted


def _manifest_and_token() -> tuple[CandidateFragmentManifest, ActionToken]:
    context = _context()
    state = PublicSearchState(
        context=context,
        agents=(
            PublicAgentPose("uav0", (0.0, 0.0, 1.0), 1.0, 1),
            PublicAgentPose("uav1", (2.0, 0.0, 1.0), 1.0, 1),
        ),
        frontiers=(
            PublicFrontier("frontier0", (1.0, 0.0, 1.0), 1.0, 0.1),
            PublicFrontier("frontier1", (3.0, 0.0, 2.0), 0.8, 0.1),
        ),
        decision_start_s=0.0,
        decision_duration_s=4.0,
        transit_timing_model=ConservativeTransitTimingModel("unit-test", 2.0, 2.0, 0.0),
        observe_dwell_s=0.5,
    )
    manifest = build_public_candidate_pool(state, identity_path_guard, candidate_limit=1)[0]
    token = authorize_manifest(
        context,
        (manifest,),
        (True,),
        0,
        token_id="execution-token",
        issued_at=0.0,
        duration=4.0,
    )
    return manifest, token


def _sample(fragment, *, observation_source: bool = True, **changes) -> FragmentExecutionSample:
    source = f"observation-{fragment.instance_fragment_id}" if observation_source else None
    values = dict(
        planned_fragment_hash=fragment.digest,
        executed=True,
        actual_start_s=fragment.planned_start,
        actual_end_s=fragment.planned_end,
        command_path_m=fragment.path,
        actual_path_m=fragment.path,
        execution_trace_hash=canonical_sha256({"fragment": fragment.digest, "trace": 1}),
        minimum_clearance_m=0.4,
        energy_used_j=1.5,
        communication_connected_at_every_telemetry_tick=True,
        source_observation_id=(
            source if fragment.type_signature.fragment_type == "observation" else None
        ),
        source_observation_episode_id=(
            fragment.episode_id
            if fragment.type_signature.fragment_type == "observation" and source
            else None
        ),
        source_observation_agent_id=(
            fragment.agent_id
            if fragment.type_signature.fragment_type == "observation" and source
            else None
        ),
        range_ok=(True if fragment.type_signature.fragment_type == "observation" else None),
        fov_ok=(True if fragment.type_signature.fragment_type == "observation" else None),
        los_ok=(True if fragment.type_signature.fragment_type == "observation" else None),
        orientation_ok=(True if fragment.type_signature.fragment_type == "observation" else None),
        dwell_ok=(True if fragment.type_signature.fragment_type == "observation" else None),
    )
    values.update(changes)
    return FragmentExecutionSample(**values)


@dataclass
class _Backend:
    backend_id: str = "deterministic-test-backend"
    evidence_class: str = "deterministic_test_only"
    missing_observation_source: bool = False
    collision: bool = False
    out_of_bounds: bool = False
    guard_intervened: bool = False
    static_clearance_contract_violation: bool = False
    inter_agent_separation_violation: bool = False

    def execute_manifest(self, manifest, token):
        del token
        return tuple(
            _sample(
                fragment,
                observation_source=not self.missing_observation_source,
                collision=self.collision,
                out_of_bounds=self.out_of_bounds,
                guard_intervened=self.guard_intervened,
                static_clearance_contract_violation=self.static_clearance_contract_violation,
                inter_agent_separation_violation=self.inter_agent_separation_violation,
            )
            for fragment in manifest.fragments
        )


def test_successful_concurrent_manifest_keeps_trace_provenance_and_observation_outcomes():
    manifest, token = _manifest_and_token()
    ledger = execute_hm3d_manifest(manifest, token, _Backend())
    assert ledger.executed_fragment_count == len(manifest.fragments)
    assert ledger.failed_fragment_count == 0
    assert ledger.reusable_fragment_count == len(manifest.fragments)
    assert ledger.engineering_only is True
    assert len(ledger.trace_hashes) == len(manifest.fragments)
    assert all(outcome.executed for outcome in ledger.outcomes)
    assert all(decision.allowed for decision in ledger.provenance_decisions)


def test_explicit_replay_exclusion_retains_real_outcomes_but_prevents_reuse() -> None:
    """A safety-only recovery stays in the physical ledger but cannot seed OGFR."""

    manifest, token = _manifest_and_token()
    ledger = execute_hm3d_manifest(
        manifest,
        token,
        _Backend(),
        replay_exclusion_reason="COLLISION_AVOIDANCE_RECOVERY",
    )

    assert ledger.executed_fragment_count == len(manifest.fragments)
    assert ledger.failed_fragment_count == 0
    assert ledger.total_energy_used_j == pytest.approx(1.5 * len(manifest.fragments))
    assert len(ledger.trace_hashes) == len(manifest.fragments)
    assert all(outcome.executed for outcome in ledger.outcomes)
    assert ledger.reusable_fragment_count == 0
    assert all(record is None for record in ledger.replay_records)
    assert all(not decision.allowed for decision in ledger.provenance_decisions)
    assert {decision.reason_code for decision in ledger.provenance_decisions} == {
        "COLLISION_AVOIDANCE_RECOVERY"
    }
    assert {
        decision.evidence_hash for decision in ledger.provenance_decisions
    } == {outcome.digest for outcome in ledger.outcomes}


def test_missing_observation_outcome_fails_closed_but_the_attempt_stays_in_ledger():
    manifest, token = _manifest_and_token()
    ledger = execute_hm3d_manifest(manifest, token, _Backend(missing_observation_source=True))
    decisions = {
        outcome.planned_fragment_hash: decision.reason_code
        for outcome, decision in zip(ledger.outcomes, ledger.provenance_decisions, strict=True)
    }
    observation_hashes = {
        fragment.digest
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "observation"
    }
    assert all(
        decisions[digest] == "MISSING_SOURCE_OBSERVATION_ID" for digest in observation_hashes
    )
    assert ledger.executed_fragment_count == len(manifest.fragments)
    assert ledger.reusable_fragment_count == len(manifest.fragments) - len(observation_hashes)


def test_started_timeout_keeps_its_real_trace_in_the_denominator_but_never_enters_replay():
    manifest, token = _manifest_and_token()
    timed_out = next(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    @dataclass
    class TimeoutBackend(_Backend):
        def execute_manifest(self, manifest, token):
            samples = list(super().execute_manifest(manifest, token))
            index = next(
                index
                for index, sample in enumerate(samples)
                if sample.planned_fragment_hash == timed_out.digest
            )
            samples[index] = _sample(
                timed_out,
                actual_end_s=token.duration,
                actual_path_m=(timed_out.path[0],),
                failure_reason="transit_timeout",
                minimum_clearance_m=0.42,
                energy_used_j=0.8,
            )
            return tuple(samples)

    ledger = execute_hm3d_manifest(manifest, token, TimeoutBackend())
    outcome, decision = next(
        (outcome, decision)
        for outcome, decision in zip(ledger.outcomes, ledger.provenance_decisions, strict=True)
        if outcome.planned_fragment_hash == timed_out.digest
    )
    assert outcome.executed is True
    assert outcome.actual_end == pytest.approx(token.duration)
    assert dict(outcome.outcome_fields)["transit_timeout"] == 1.0
    assert dict(outcome.outcome_fields)["execution_success"] == 0.0
    assert ledger.executed_fragment_count == len(manifest.fragments)
    assert ledger.failed_fragment_count == 0
    assert ledger.minimum_clearance_m == pytest.approx(0.4)
    assert decision.reason_code == "TRANSIT_TIMEOUT"
    assert ledger.reusable_fragment_count == len(manifest.fragments) - 1


def test_timeout_reason_has_priority_over_a_preflight_guard_rewrite() -> None:
    manifest, token = _manifest_and_token()
    timed_out = next(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )

    @dataclass
    class GuardedTimeoutBackend(_Backend):
        def execute_manifest(self, manifest, token):
            samples = list(super().execute_manifest(manifest, token))
            index = next(
                index
                for index, sample in enumerate(samples)
                if sample.planned_fragment_hash == timed_out.digest
            )
            samples[index] = _sample(
                timed_out,
                actual_end_s=token.duration,
                actual_path_m=(timed_out.path[0],),
                guard_intervened=True,
                failure_reason="transit_timeout",
            )
            return tuple(samples)

    ledger = execute_hm3d_manifest(manifest, token, GuardedTimeoutBackend())
    decision = next(
        decision
        for outcome, decision in zip(ledger.outcomes, ledger.provenance_decisions, strict=True)
        if outcome.planned_fragment_hash == timed_out.digest
    )
    assert decision.reason_code == "TRANSIT_TIMEOUT"


@pytest.mark.parametrize(
    "failure",
    (
        "collision",
        "out_of_bounds",
        "guard_intervened",
        "static_clearance_contract_violation",
        "inter_agent_separation_violation",
    ),
)
def test_physical_failures_remain_in_the_denominator_and_never_enter_replay(failure: str):
    manifest, token = _manifest_and_token()
    ledger = execute_hm3d_manifest(manifest, token, _Backend(**{failure: True}))
    assert ledger.executed_fragment_count == len(manifest.fragments)
    assert ledger.failed_fragment_count == 0
    assert ledger.reusable_fragment_count == 0
    if failure == "collision":
        assert ledger.collision_count == len(manifest.fragments)
        assert {row.reason_code for row in ledger.provenance_decisions} == {"PHYSICAL_COLLISION"}
    elif failure == "out_of_bounds":
        assert ledger.out_of_bounds_count == len(manifest.fragments)
        assert {row.reason_code for row in ledger.provenance_decisions} == {
            "FLIGHT_BOUNDS_VIOLATION"
        }
    elif failure == "inter_agent_separation_violation":
        assert ledger.inter_agent_separation_violation_count == len(manifest.fragments)
        assert {row.reason_code for row in ledger.provenance_decisions} == {
            "INTER_AGENT_SEPARATION_VIOLATION"
        }
    else:
        if failure == "guard_intervened":
            assert ledger.guard_intervention_count == len(manifest.fragments)
            assert {row.reason_code for row in ledger.provenance_decisions} == {"GUARD_REWRITTEN"}
        else:
            assert ledger.static_clearance_contract_violation_count == len(manifest.fragments)
            assert {row.reason_code for row in ledger.provenance_decisions} == {
                "STATIC_CLEARANCE_CONTRACT_VIOLATION"
            }


def test_token_manifest_mismatch_is_rejected_before_backend_execution():
    manifest, token = _manifest_and_token()
    forged = ActionToken(
        token_id=token.token_id,
        episode_id=token.episode_id,
        decision_id=token.decision_id,
        context_hash=token.context_hash,
        manifest_hash=canonical_sha256({"forged": True}),
        legal_mask_hash=token.legal_mask_hash,
        planned_fragment_hashes=token.planned_fragment_hashes,
        issued_at=token.issued_at,
        duration=token.duration,
    )
    with pytest.raises(ValueError, match="does not authorize"):
        execute_hm3d_manifest(manifest, forged, _Backend())


def test_backend_must_return_exactly_one_result_per_authorized_fragment():
    manifest, token = _manifest_and_token()

    @dataclass
    class IncompleteBackend(_Backend):
        def execute_manifest(self, manifest, token):
            return super().execute_manifest(manifest, token)[:-1]

    with pytest.raises(ValueError, match="exactly one sample"):
        execute_hm3d_manifest(manifest, token, IncompleteBackend())


def test_ledger_has_no_target_or_evaluator_private_payload():
    manifest, token = _manifest_and_token()
    serialized = repr(
        execute_hm3d_manifest(manifest, token, _Backend()).to_public_dict()
    ).casefold()
    assert "target" not in serialized
    assert "evaluator_private" not in serialized
