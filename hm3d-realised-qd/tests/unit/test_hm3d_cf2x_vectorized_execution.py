from types import SimpleNamespace

import pytest

from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentTypeSignature,
    PublicMethodContext,
)
from aerocity_method.runtime.hm3d_cf2x_vectorized_execution import (
    IsaacCF2XVectorizedExecutionBackend,
)
from aerocity_method.runtime.hm3d_multicluster import HM3DClusterLayout


def _manifest() -> CandidateFragmentManifest:
    context = PublicMethodContext(
        context_id="timeout-probe",
        episode_id="timeout-probe",
        decision_id="decision0",
        agent_features=tuple((f"uav{index}", (1.0,)) for index in range(4)),
    )
    fragments = []
    for index in range(4):
        agent_id = f"uav{index}"
        start = (float(index), 0.0, 1.0)
        endpoint = (float(index), 1.0, 1.0)
        fragments.extend(
            (
                FragmentInstance(
                    instance_fragment_id=f"{agent_id}-transit",
                    type_signature=FragmentTypeSignature("transit"),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=0.0,
                    planned_end=3.0,
                    path=(start, endpoint),
                ),
                FragmentInstance(
                    instance_fragment_id=f"{agent_id}-observation",
                    type_signature=FragmentTypeSignature("observation"),
                    episode_id=context.episode_id,
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    planned_start=3.0,
                    planned_end=5.0,
                    path=(endpoint,),
                ),
            )
        )
    return CandidateFragmentManifest(
        candidate_id="timeout-probe",
        context_hash=context.digest,
        fragments=tuple(fragments),
        planned_descriptor=(0.5, 0.5, 0.5),
        feasible=True,
    )


def _backend(timeout_s: float | None) -> IsaacCF2XVectorizedExecutionBackend:
    return IsaacCF2XVectorizedExecutionBackend(
        sim=None,
        robot=None,
        contact=None,
        scene_query=None,
        static_clearance_oracle=None,
        layout=HM3DClusterLayout(((0.0, 0.0, 0.0),)),
        bounds_min_m=(-10.0, -10.0, -10.0),
        bounds_max_m=(10.0, 10.0, 10.0),
        arrival_tolerance_m=0.1,
        calibration_timeout_probe_s=timeout_s,
    )


def test_calibration_timeout_preserves_token_budget_and_records_short_deadline() -> None:
    runtime = _backend(1.5)._new_runtime(
        _manifest(), SimpleNamespace(duration=5.0)
    )
    assert runtime.horizon_s == 5.0
    assert runtime.execution_deadline_s == 1.5


def test_calibration_timeout_must_end_before_token_budget() -> None:
    with pytest.raises(ValueError, match="must end before"):
        _backend(5.0)._new_runtime(_manifest(), SimpleNamespace(duration=5.0))
