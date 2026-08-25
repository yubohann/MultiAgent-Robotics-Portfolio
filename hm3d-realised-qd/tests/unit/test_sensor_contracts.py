from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.runtime.sensors import (
    SENSOR_PILOT_MODES,
    SensorEntitlement,
    SensorFairnessAdmission,
    SensorProfile,
    SensorThroughputRecord,
    audit_sensor_throughput_pilot,
)


def profile(mode: str) -> SensorProfile:
    if mode == "physics_only":
        return SensorProfile("physics", mode, 0.0, (), ())
    if mode == "sparse_range_3d":
        return SensorProfile(
            "range",
            mode,
            10.0,
            ("transit", "observe", "dwell", "map_update"),
            ("range_points", "source_observation_id"),
            range_enabled=True,
        )
    raise ValueError(f"unsupported formal H15 mode: {mode}")


def throughput_row(fleet_size: int, mode: str) -> SensorThroughputRecord:
    sensor = profile(mode)
    observations = (0,) * fleet_size if mode == "physics_only" else (10,) * fleet_size
    return SensorThroughputRecord(
        comparison_id="pilot-1",
        scene_id="scene-a",
        episode_id="episode-a",
        fleet_size=fleet_size,
        profile=sensor,
        physics_dt_s=0.01,
        planned_episodes=2,
        executed_episodes=2,
        failed_episodes=0,
        physics_real_time_factor=1.0,
        environment_steps_per_s=100.0,
        sensor_frames_per_s=0.0 if mode == "physics_only" else 20.0,
        render_time_s=0.0 if mode == "physics_only" else 0.2,
        transfer_time_s=0.0 if mode == "physics_only" else 0.1,
        gpu_memory_mb=100.0,
        cpu_memory_mb=200.0,
        dropped_frames=0,
        observations_per_agent=observations,
        measurement_scope="throughput_only",
        wall_clock_s=2.0,
    )


def test_formal_sensor_contract_rejects_retired_camera_profiles():
    with pytest.raises(ValueError, match="unsupported sensor pilot mode"):
        SensorProfile("retired-camera", "retired_camera", 5.0, ("transit",), ("depth",))


def test_sensor_profile_rejects_private_geometry_and_target_distance():
    with pytest.raises(ValueError, match="private geometry/truth"):
        SensorProfile(
            "leaking",
            "sparse_range_3d",
            10.0,
            ("transit",),
            ("target_distance",),
            range_enabled=True,
        )


def test_sensor_fairness_requires_identical_entitlements():
    sensor = profile("sparse_range_3d")
    SensorFairnessAdmission(
        sensor,
        (
            SensorEntitlement("ours", sensor.entitlement_hash),
            SensorEntitlement("frontier", sensor.entitlement_hash),
        ),
    )
    with pytest.raises(ValueError, match="unequal sensor"):
        SensorFairnessAdmission(
            sensor,
            (
                SensorEntitlement("ours", sensor.entitlement_hash),
                SensorEntitlement("frontier", profile("physics_only").entitlement_hash),
            ),
        )


def test_sensor_pilot_requires_the_camera_free_matrix_for_the_formal_four_cf2x_fleet():
    rows = tuple(throughput_row(4, mode) for mode in SENSOR_PILOT_MODES)
    assert audit_sensor_throughput_pilot(rows)["status"] == "PASS"
    incomplete = audit_sensor_throughput_pilot(rows[:-1])
    assert incomplete["status"] == "RUNTIME_NOT_READY"
    assert incomplete["missing"] == [{"fleet_size": 4, "mode": "sparse_range_3d"}]


def test_sensor_pilot_keeps_failures_in_denominator():
    row = throughput_row(4, "sparse_range_3d")
    with pytest.raises(ValueError, match="denominator"):
        replace(row, planned_episodes=3)


def test_sensor_pilot_rejects_task_quality_semantics():
    row = throughput_row(4, "sparse_range_3d")
    with pytest.raises(ValueError, match="throughput_only"):
        replace(row, measurement_scope="task_quality")
