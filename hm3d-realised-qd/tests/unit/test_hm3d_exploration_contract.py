from __future__ import annotations

from copy import deepcopy

import pytest

from aerocity_method.evaluation.hm3d_exploration_contract import (
    HM3DExplorationObservationContract,
    load_exploration_observation_contract,
)


def test_frozen_p04_contract_declares_private_geometry_boundary_and_absolute_metrics() -> None:
    contract = load_exploration_observation_contract()
    payload = contract.to_dict()
    assert len(contract.digest) == 64
    assert "evaluator_truth_map" in payload["method_forbidden"]
    assert "final_explored_free_volume_m3" in payload["evaluation"]["required_report_fields"]
    assert (
        "evaluator_reachable_free_flight_volume_m3"
        in payload["evaluation"]["required_report_fields"]
    )


def test_p04_contract_rejects_observe_only_sensor_schedule() -> None:
    payload = deepcopy(load_exploration_observation_contract().to_dict())
    payload["sensor_profile"]["enabled_windows"] = ["observe", "dwell"]
    with pytest.raises(ValueError, match="sensor windows"):
        HM3DExplorationObservationContract(payload)
