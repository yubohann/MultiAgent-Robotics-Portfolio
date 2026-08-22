from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from aerocity_method.runtime.hm3d_realised_qd import (
    RealisedQDDescriptor,
    audit_realised_qd_calibration_mode_contrasts,
    audit_realised_qd_reproducibility,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_hm3d_qd_replay_calibration.py"
SPEC = importlib.util.spec_from_file_location("hm3d_qd_replay_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _case(case_id: str, scene_id: str, random_key: int) -> object:
    return MODULE.CalibrationCase.from_dict(
        {
            "case_id": case_id,
            "scene_id": scene_id,
            "random_key": random_key,
            "runner_arguments": [
                "--scene-id",
                scene_id,
                "--collision-usd",
                "E:/asset/example.usd",
            ],
        }
    )


def test_qd_replay_calibration_plans_two_real_replays_for_all_six_public_modes(
    tmp_path: Path,
) -> None:
    plans = MODULE.build_calibration_plan(
        cases=(_case("case_a", "scene_a", 101), _case("case_b", "scene_b", 202)),
        output_dir=tmp_path / "outputs",
    )

    assert len(plans) == 24
    assert {plan.intent_mode for plan in plans} == set(MODULE.CALIBRATION_INTENT_MODES)
    assert {plan.repetition for plan in plans} == {0, 1}
    command = plans[0].command(python=Path("python"), runner=Path("runner.py"))
    assert "--split" in command and command[command.index("--split") + 1] == "train"
    assert "--strategy" in command and command[command.index("--strategy") + 1] == "qd_calibration"


def test_qd_replay_calibration_refuses_a_case_that_overrides_fixed_replay_fields() -> None:
    with pytest.raises(ValueError, match="may not override"):
        MODULE.CalibrationCase.from_dict(
            {
                "case_id": "bad_case",
                "scene_id": "scene_a",
                "random_key": 101,
                "runner_arguments": [
                    "--scene-id",
                    "scene_a",
                    "--strategy",
                    "random",
                ],
            }
        )


def test_qd_replay_calibration_requires_independent_axis_control_as_well_as_stability() -> None:
    replay = audit_realised_qd_reproducibility(
        {
            f"{index + 1:064x}": (
                RealisedQDDescriptor(0.20 + index * 0.05, 0.30, 0.40),
                RealisedQDDescriptor(0.20 + index * 0.05, 0.30, 0.40),
            )
            for index in range(3)
        }
    )
    assert replay.status == "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED"
    labels = tuple(mode for mode in MODULE.CALIBRATION_INTENT_MODES for _ in range(2))
    collapsed = audit_realised_qd_calibration_mode_contrasts(
        labels,
        tuple(RealisedQDDescriptor(0.50, 0.50, 0.50) for _ in labels),
        tuple("scene_a" if index % 2 == 0 else "scene_b" for index in range(len(labels))),
    )
    assert collapsed.status == "QD_CALIBRATION_MODE_CONTRAST_NOT_ADMITTED"

    assert not MODULE._calibration_admitted(
        replay.status,
        collapsed.status,
        "QD_DESCRIPTOR_FAMILY_CURRENT_ADMITTED",
    )
    assert not MODULE._calibration_admitted(
        replay.status,
        "QD_CALIBRATION_MODE_CONTRAST_ADMITTED",
        "QD_DESCRIPTOR_FAMILY_REDESIGN_REQUIRED",
    )
