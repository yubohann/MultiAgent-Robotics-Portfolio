from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.cf2x_runtime_calibration import (
    CF2X_RUNTIME_CALIBRATION_PRELAUNCH_FAILURE_SCHEMA,
    CF2X_RUNTIME_CALIBRATION_SCHEMA,
    _as_float_vector,
    _locked_contrib_source_root,
    _persist_prelaunch_failure,
    _runtime_body_physics,
    calibration_report_sha256,
    prelaunch_failure_report_sha256,
    validate_calibration_report,
)


def _report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema": CF2X_RUNTIME_CALIBRATION_SCHEMA,
        "status": "passed",
        "claim_boundary": {
            "formal_episode": False,
            "city_lite_capture": False,
            "benchmark_score": False,
            "sensor_payload_retained": False,
        },
        "asset": {"usd_sha256": "a" * 64},
        "source": {
            "source_revision": "b" * 40,
            "source_tree_sha256": "c" * 64,
            "source_worktree_dirty": False,
        },
        "runtime_lock_sha256": "d" * 64,
        "runtime_audit": {"status": "passed"},
        "static_usd": {
            "usd_sha256": "a" * 64,
            "bodies": [
                {
                    "body_name": "body",
                    "mass_kg": 0.025,
                    "diagonal_inertia_kg_m2": [0.00001, 0.00001, 0.00002],
                }
            ],
        },
        "runtime": {
            "thruster_names": ["m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            "allocation_matrix": [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-0.046, 0.046, 0.046, -0.046],
                [-0.046, -0.046, 0.046, 0.046],
                [0.006, -0.006, 0.006, -0.006],
            ],
            "rotor_directions": [1, -1, 1, -1],
            "thrust_axis": {"all_positive_body_z": True},
            "bodies": [
                {
                    "body_name": "body",
                    "mass_kg": 0.025,
                    "diagonal_inertia_kg_m2": [0.00001, 0.00001, 0.00002],
                    "inertia_matrix_kg_m2": [
                        [0.00001, 0.0, 0.0],
                        [0.0, 0.00001, 0.0],
                        [0.0, 0.0, 0.00002],
                    ],
                }
            ],
            "actuator": {
                "control_dt_s": 0.005,
                "actuator_dt_s": 0.005,
                "max_thrust_rate_n_per_s": 100000.0,
                "thrust_range_n": [0.001, 0.18],
                "thrust_constant_range_n_per_rps_squared": [1.0e-6, 1.0e-6],
                "tau_increase_range_s": [0.04, 0.06],
                "tau_decrease_range_s": [0.02, 0.03],
                "torque_to_thrust_ratio_nm_per_n": 0.006,
                "sampled_tau_increase_s": [0.05] * 4,
                "sampled_tau_decrease_s": [0.025] * 4,
                "sampled_thrust_constant_n_per_rps_squared": [1.0e-6] * 4,
            },
        },
        "static_runtime_cross_check": {
            "status": "passed",
            "bodies": [{"body_name": "body", "status": "matched"}],
        },
        "actuation_probe": {
            "step_order": [
                "set_thrust_target",
                "write_data_to_sim",
                "simulation_step",
                "robot_update",
            ],
            "command_before_step": True,
            "requested_thrust_n": [0.09, 0.09, 0.09, 0.09],
            "target_thrust_after_set_n": [0.09, 0.09, 0.09, 0.09],
            "applied_thrust_after_write_n": [0.07, 0.07, 0.07, 0.07],
            "applied_wrench_after_write_body": [0.0, 0.0, 0.28, 0.0, 0.0, 0.0],
            "initial_root_position_w_m": [0.0, 0.0, 1.0],
            "final_root_position_w_m": [0.0, 0.0, 1.01],
            "samples": [
                {
                    "physics_step": 1,
                    "root_position_w_m": [0.0, 0.0, 1.01],
                    "root_linear_velocity_w_mps": [0.0, 0.0, 0.1],
                    "applied_thrust_n": [0.07] * 4,
                }
            ],
        },
    }
    report["report_sha256"] = calibration_report_sha256(report)
    return report


def test_valid_report_is_self_hash_bound() -> None:
    report = _report()
    assert validate_calibration_report(report) == ()
    assert report["report_sha256"] == calibration_report_sha256(report)


def test_changing_bound_runtime_parameter_breaks_self_hash() -> None:
    report = _report()
    changed = copy.deepcopy(report)
    changed["runtime"]["actuator"]["actuator_dt_s"] = 0.01
    issues = validate_calibration_report(changed)
    assert "runtime actuator dt differs from control dt" in issues
    assert "report self-hash does not match" in issues


def test_invalid_rotor_order_and_force_axis_fail_closed() -> None:
    report = _report()
    report["runtime"]["thruster_names"] = ["m1_prop"] * 4
    report["runtime"]["thrust_axis"]["all_positive_body_z"] = False
    report["report_sha256"] = calibration_report_sha256(report)
    issues = validate_calibration_report(report)
    assert "runtime rotor order is invalid" in issues
    assert "runtime thrust axis is not positive body z" in issues


def test_failed_static_runtime_cross_check_cannot_pass() -> None:
    report = _report()
    report["static_runtime_cross_check"]["status"] = "failed"
    report["report_sha256"] = calibration_report_sha256(report)
    assert "static/runtime cross-check did not pass" in validate_calibration_report(report)


def test_zero_applied_thrust_cannot_be_called_a_calibration() -> None:
    report = _report()
    report["actuation_probe"]["applied_thrust_after_write_n"] = [0.0] * 4
    report["actuation_probe"]["applied_wrench_after_write_body"][2] = 0.0
    report["report_sha256"] = calibration_report_sha256(report)
    issues = validate_calibration_report(report)
    assert "probe did not apply positive thrust" in issues
    assert "probe did not apply positive body-z force" in issues


def test_missing_response_or_clean_source_evidence_cannot_pass() -> None:
    report = _report()
    report["source"]["source_worktree_dirty"] = True
    report["runtime"]["actuator"].pop("sampled_tau_increase_s")
    report["actuation_probe"]["samples"][0].pop("root_linear_velocity_w_mps")
    report["report_sha256"] = calibration_report_sha256(report)
    issues = validate_calibration_report(report)
    assert "calibration source provenance is not clean and hash-bound" in issues
    assert "runtime actuator response evidence is invalid" in issues
    assert "probe has no post-step state samples" in issues


def test_zero_lower_thrust_bound_is_valid_but_degenerate_or_reversed_ranges_fail() -> None:
    report = _report()
    report["runtime"]["actuator"]["thrust_range_n"] = [0.0, 0.18]
    report["report_sha256"] = calibration_report_sha256(report)
    assert validate_calibration_report(report) == ()

    for invalid_range in ([0.0, 0.0], [0.18, 0.0], [-0.01, 0.18]):
        invalid = copy.deepcopy(report)
        invalid["runtime"]["actuator"]["thrust_range_n"] = invalid_range
        invalid["report_sha256"] = calibration_report_sha256(invalid)
        assert "runtime actuator response evidence is invalid" in validate_calibration_report(invalid)


def test_locked_contrib_path_resolves_beside_the_audited_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source" / "isaaclab"
    lock = {"isaaclab_contrib_source": {"relative_path": "isaaclab_contrib"}}
    assert _locked_contrib_source_root(source, lock) == source.parent.resolve() / "isaaclab_contrib"
    unsafe_lock = {"isaaclab_contrib_source": {"relative_path": "../isaaclab_contrib"}}
    with pytest.raises(RuntimeError, match="unsafe"):
        _locked_contrib_source_root(source, unsafe_lock)


def test_prelaunch_failure_receipt_is_hash_bound_and_never_overwritten(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed-calibration"
    args = SimpleNamespace(output_dir=output_dir)
    path = _persist_prelaunch_failure(args, RuntimeError("locked source is unavailable"))
    assert path == output_dir / "cf2x_runtime_calibration.prelaunch_failure.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == CF2X_RUNTIME_CALIBRATION_PRELAUNCH_FAILURE_SCHEMA
    assert report["status"] == "failed"
    assert report["claim_boundary"]["app_launcher_started"] is False
    assert report["report_sha256"] == prelaunch_failure_report_sha256(report)
    original = path.read_bytes()
    assert _persist_prelaunch_failure(args, RuntimeError("different failure")) is None
    assert path.read_bytes() == original


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = value

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> object:
        return self.value


class _PhysicsView:
    def __init__(self, masses: object, inertias: object) -> None:
        self.masses = masses
        self.inertias = inertias
        self.mass_reads = 0
        self.inertia_reads = 0

    def get_masses(self) -> _Tensor:
        self.mass_reads += 1
        return _Tensor(self.masses)

    def get_inertias(self) -> _Tensor:
        self.inertia_reads += 1
        return _Tensor(self.inertias)


def test_float_vector_materializes_cuda_style_tensor_before_numpy_conversion() -> None:
    assert _as_float_vector(_Tensor([[0.01, 0.02], [0.03, 0.04]])) == [0.01, 0.02, 0.03, 0.04]


def test_runtime_body_physics_reads_live_physx_view_not_empty_multirotor_cache() -> None:
    view = _PhysicsView(
        masses=[[0.025, 0.001]],
        inertias=[
            [
                [1.0e-5, 0.0, 0.0, 0.0, 1.1e-5, 0.0, 0.0, 0.0, 2.0e-5],
                [2.0e-6, 0.0, 0.0, 0.0, 2.1e-6, 0.0, 0.0, 0.0, 3.0e-6],
            ]
        ],
    )
    robot = SimpleNamespace(
        body_names=["body", "rotor"],
        root_physx_view=view,
        data=SimpleNamespace(default_mass=None, default_inertia=None),
    )

    assert _runtime_body_physics(robot) == [
        {
            "body_name": "body",
            "mass_kg": 0.025,
            "inertia_matrix_kg_m2": [[1.0e-5, 0.0, 0.0], [0.0, 1.1e-5, 0.0], [0.0, 0.0, 2.0e-5]],
            "diagonal_inertia_kg_m2": [1.0e-5, 1.1e-5, 2.0e-5],
        },
        {
            "body_name": "rotor",
            "mass_kg": 0.001,
            "inertia_matrix_kg_m2": [[2.0e-6, 0.0, 0.0], [0.0, 2.1e-6, 0.0], [0.0, 0.0, 3.0e-6]],
            "diagonal_inertia_kg_m2": [2.0e-6, 2.1e-6, 3.0e-6],
        },
    ]
    assert (view.mass_reads, view.inertia_reads) == (1, 1)


@pytest.mark.parametrize(
    ("robot", "message"),
    [
        (SimpleNamespace(body_names=["body"], root_physx_view=None), "no PhysX articulation view"),
        (
            SimpleNamespace(body_names=["body"], root_physx_view=SimpleNamespace()),
            "cannot read mass and inertia",
        ),
        (
            SimpleNamespace(
                body_names=["body"],
                root_physx_view=_PhysicsView([[0.025, 0.001]], [[[1.0] * 9]]),
            ),
            "mass shape disagrees",
        ),
    ],
)
def test_runtime_body_physics_rejects_missing_or_malformed_live_data(
    robot: SimpleNamespace, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _runtime_body_physics(robot)
