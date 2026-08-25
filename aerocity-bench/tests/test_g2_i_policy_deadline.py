from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool_module():
    path = Path(__file__).parents[1] / "tools" / "diagnose_g2_i_policy_deadline.py"
    spec = importlib.util.spec_from_file_location("g2_i_policy_deadline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deadline_diagnosis_replays_the_recorded_sensor_pitch() -> None:
    observation = _tool_module()._observation_from_dict(  # noqa: SLF001 - tool contract
        {
            "episode_id": "episode",
            "observation_id": "observation",
            "drone_id": "uav-00",
            "sequence": 4,
            "timestamp_s": 1.0,
            "pose": {
                "position": [1.0, 2.0, 3.0],
                "yaw_deg": 10.0,
                "pitch_deg": 5.0,
                "roll_deg": 0.0,
            },
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_speed_deg_s": 0.0,
            "energy_remaining_j": 12.0,
            "local_occupancy": [[0, 0, 0]],
            "local_occupancy_origin_world_m": [0.0, 0.0, 0.0],
            "local_occupancy_resolution_m": 1.0,
            "local_occupancy_radius_m": 1.0,
            "teammate_states": [],
            "health": "nominal",
            "sensor_pitch_deg": -90.0,
        }
    )

    assert observation.sensor_pitch_deg == -90.0
