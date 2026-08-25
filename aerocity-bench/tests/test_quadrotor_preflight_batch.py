from __future__ import annotations

from pathlib import Path

import pytest

from aerocity_bench.canonical import content_hash, write_json
from tools.quadrotor_physics_preflight_batch import validate_preflight_report


def _receipt(*, valid_checks: bool = True) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.quadrotor-physx-preflight.v2",
        "formal": False,
        "formal_score_eligible": False,
        "vehicle_execution_model": (
            "cf2x_multirotor_per_rotor_thrust_geometry_allocated_root_wrench_physx"
        ),
        "steps": 240,
        "controller": {"profile": "shared-hold"},
        "checks": {"native": valid_checks},
        "runtime_quality": {"root_height_below_0_15m": False},
        "runtime": {
            "preflight_script_sha256": "a" * 64,
            "dynamics_contract_sha256": "b" * 64,
            "cf2x_contract_sha256": "c" * 64,
            "cf2x_native_sha256": "d" * 64,
        },
        "multirotor": {
            "contact_evidence": "IsaacLab ContactSensor.net_forces_w",
            "direct_root_state_writes_during_loop": False,
            "wrench_application_model": "derived_geometry_allocation_to_root_body_physx",
            "prop_link_forces_applied_directly": False,
        },
        "asset": {
            "asset_kind": "cf2x_local_runtime_dependency",
            "usd_sha256": "e" * 64,
            "schema_sha256": "f" * 64,
        },
        "final_state": {
            "position_w_m": [0.0, 0.0, 1.5],
            "linear_velocity_w_mps": [0.0, 0.0, 0.0],
            "applied_rotor_thrust_n": [0.1, 0.1, 0.1, 0.1],
        },
    }
    report["preflight_hash"] = content_hash(report)
    return report


def test_batch_validator_accepts_complete_nonformal_preflight(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    write_json(path, _receipt())
    verified = validate_preflight_report(
        path, expected_profile="shared-hold", expected_steps=240
    )
    assert verified["preflight_hash"]
    assert verified["final_position_w_m"] == [0.0, 0.0, 1.5]


def test_batch_validator_rejects_failed_or_rehashed_preflight(tmp_path: Path) -> None:
    failed = tmp_path / "failed.json"
    write_json(failed, _receipt(valid_checks=False))
    with pytest.raises(ValueError, match="failed or incomplete"):
        validate_preflight_report(failed, expected_profile="shared-hold", expected_steps=240)

    tampered = _receipt()
    tampered["steps"] = 241
    path = tmp_path / "tampered.json"
    write_json(path, tampered)
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_preflight_report(path, expected_profile="shared-hold", expected_steps=240)


def test_batch_validator_requires_explicit_long_horizon_evidence(tmp_path: Path) -> None:
    path = tmp_path / "long-hold-missing-metrics.json"
    report = _receipt()
    report["controller"] = {"profile": "shared-long-hold"}
    report["preflight_hash"] = content_hash(
        {key: value for key, value in report.items() if key != "preflight_hash"}
    )
    write_json(path, report)

    with pytest.raises(ValueError, match="lacks trend metrics or thresholds"):
        validate_preflight_report(path, expected_profile="shared-long-hold", expected_steps=240)

    report["controller"] = {"profile": "shared-long-lateral-hold"}
    report["preflight_hash"] = content_hash(
        {key: value for key, value in report.items() if key != "preflight_hash"}
    )
    write_json(path, report)
    with pytest.raises(ValueError, match="lacks trend metrics or thresholds"):
        validate_preflight_report(
            path, expected_profile="shared-long-lateral-hold", expected_steps=240
        )
