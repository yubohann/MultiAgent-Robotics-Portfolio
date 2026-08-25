"""Async-completion (per-agent rolling) unit tests for the P07 executor."""

from __future__ import annotations

from aerocity_method.contracts.models import FragmentInstance, FragmentTypeSignature
from aerocity_method.runtime.hm3d_cf2x_execution import _finalize_fragment_pair_into


def _fragment(fragment_id: str, fragment_type: str, path: tuple[tuple[float, float, float], ...]) -> FragmentInstance:
    return FragmentInstance(
        instance_fragment_id=fragment_id,
        type_signature=FragmentTypeSignature(
            fragment_type, (("assignment_role", "explore"),)
        ),
        episode_id="ep",
        decision_id="d0",
        agent_id="uav0",
        planned_start=0.0,
        planned_end=5.0,
        path=path,
        pose_mode="guarded_waypoint",
        context_bucket="hm3d-test",
        guard_rewritten=False,
    )


def test_finalize_fragment_pair_records_transit_and_observation_provenance() -> None:
    ledger: list[object] = []
    transit = _fragment("t1", "transit", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)))
    observe = _fragment("o1", "observation", ((2.0, 0.0, 1.0),))
    _finalize_fragment_pair_into(
        ledger,
        index=0,
        transit=transit,
        observe=observe,
        transit_completed=True,
        observation_completed=True,
        transit_trace=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)),
        observation_trace=((2.0, 0.0, 1.0),),
        transit_release_s=0.5,
        transit_end_s=4.0,
        observation_start_s=4.0,
        observation_end_s=5.0,
        execution_horizon_s=10.0,
        energy_j=3.5,
        collision=False,
        out_of_bounds=False,
        separation_violation=False,
        static_clearance_contract_violation=False,
        minimum_clearance_m=0.9,
        connected_at_every_tick=True,
        last_sensor_source_id="range-abc-uav0-0001",
        rolling=True,
    )
    assert len(ledger) == 2
    transit_sample, observe_sample = ledger
    assert transit_sample.executed is True
    assert transit_sample.planned_fragment_hash == transit.digest
    assert transit_sample.actual_path_m == (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
    )
    assert transit_sample.energy_used_j == 3.5
    assert observe_sample.executed is True
    assert observe_sample.planned_fragment_hash == observe.digest
    assert observe_sample.source_observation_id == "range-abc-uav0-0001"
    assert observe_sample.source_observation_agent_id == "uav0"
    assert observe_sample.range_ok is True
    assert observe_sample.fov_ok is True
    assert observe_sample.los_ok is True
    assert observe_sample.orientation_ok is True
    assert observe_sample.dwell_ok is True


def test_finalize_fragment_pair_timeout_marks_transit_and_skips_observation() -> None:
    ledger: list[object] = []
    transit = _fragment("t2", "transit", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)))
    observe = _fragment("o2", "observation", ((2.0, 0.0, 1.0),))
    _finalize_fragment_pair_into(
        ledger,
        index=0,
        transit=transit,
        observe=observe,
        transit_completed=False,
        observation_completed=False,
        transit_trace=((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
        observation_trace=(),
        transit_release_s=0.0,
        transit_end_s=None,
        observation_start_s=None,
        observation_end_s=None,
        execution_horizon_s=10.0,
        energy_j=1.0,
        collision=False,
        out_of_bounds=False,
        separation_violation=False,
        static_clearance_contract_violation=False,
        minimum_clearance_m=0.9,
        connected_at_every_tick=True,
        last_sensor_source_id=None,
        rolling=True,
    )
    assert len(ledger) == 2
    transit_sample, observe_sample = ledger
    assert transit_sample.executed is True
    assert transit_sample.failure_reason == "transit_timeout"
    assert observe_sample.executed is False
    assert observe_sample.failure_reason == "observation_not_reached"


def test_finalize_fragment_pair_collision_marks_both_samples_failed() -> None:
    from aerocity_method.runtime.hm3d_cf2x_execution import _finalize_fragment_pair_into
    ledger: list[object] = []
    transit = _fragment("t3", "transit", ((0.0, 0.0, 1.0), (2.0, 0.0, 1.0)))
    observe = _fragment("o3", "observation", ((2.0, 0.0, 1.0),))
    _finalize_fragment_pair_into(
        ledger,
        index=0, transit=transit, observe=observe,
        transit_completed=False, observation_completed=False,
        transit_trace=((0.0, 0.0, 1.0), (0.5, 0.0, 1.0)),
        observation_trace=(),
        transit_release_s=0.0, transit_end_s=None,
        observation_start_s=None, observation_end_s=None,
        execution_horizon_s=10.0, energy_j=1.0,
        collision=True, out_of_bounds=False, separation_violation=False,
        static_clearance_contract_violation=False, minimum_clearance_m=0.9,
        connected_at_every_tick=True, last_sensor_source_id=None,
        rolling=True,
    )
    assert len(ledger) == 2
    assert ledger[0].collision is True
    assert ledger[0].out_of_bounds is False
    assert ledger[1].collision is True
    assert ledger[1].failure_reason is not None
    assert "collision" in ledger[1].failure_reason
