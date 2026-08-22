from __future__ import annotations

from types import SimpleNamespace

from tools import run_cf2x_b_gate_l0_pairing as pairing


def test_l0_pairing_passes_the_public_policy_to_the_runtime(monkeypatch) -> None:
    """Keep the paired L0 run on the same public-policy execution path as L1."""

    policy = object()
    received: list[object] = []

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run_policy(self, supplied_policy: object) -> dict[str, object]:
            received.append(supplied_policy)
            return {
                "returned_home": {"uav-00": True},
                "budget_ledger": {
                    "collisions": 0,
                    "out_of_bounds_actions": 0,
                    "deadline_misses": 0,
                },
                "task_time_s": 12.0,
            }

    monkeypatch.setattr(pairing, "create_baseline", lambda *_args: policy)
    monkeypatch.setattr(pairing, "L0FleetRuntime", FakeRuntime)
    monkeypatch.setattr(
        pairing,
        "evaluate_run",
        lambda *_args: {"quality": {"confirmed_count": 2}},
    )

    score, execution = pairing._run_method(
        method_id="sweep-3d",
        config=SimpleNamespace(raw={"execution_contract": {"episode": {"duration_s": 300.0}}}),
        city={},
        private_episode={},
        public_episode={},
        task_spec={},
    )

    assert received == [policy]
    assert score == 2.0
    assert execution == {
        "all_returned_home": True,
        "collision_count": 0,
        "out_of_bounds_actions": 0,
        "deadline_miss_tick_count": 0,
        "task_time_s": 12.0,
    }
