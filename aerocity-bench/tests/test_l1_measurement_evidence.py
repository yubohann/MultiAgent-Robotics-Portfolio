from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.contracts import ObservationPacket, Pose3D
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.measurement_evidence import (
    L1MeasurementEvidence,
    validate_measurement_evidence_snapshot,
)
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.targets_v3 import derive_support_sites_v3, sample_episode_v3

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "releases" / "ordinary-v1-mini.json"


@pytest.fixture(scope="module")
def public_g2_i_inputs():
    config = load_ordinary_config(CONFIG_PATH)
    assets = list(config.raw["assets"]["allowlist"])
    for attempt in range(32):
        try:
            city = generate_city_v3(config, "calibration", 0, attempt, assets)
            task = compile_g2_i_task_spec(
                city, config.raw["execution_contract"], config.raw["fleet"]
            )
            episode = sample_episode_v3(
                config,
                city,
                derive_support_sites_v3(city, config),
                0,
                public_task_spec=task,
            )
            return city, task, episode
        except GenerationRejected:
            continue
    raise AssertionError("expected an admitted G2-I calibration input")


def _first_selected_cell(task: dict[str, object], episode: dict[str, object]) -> dict[str, object]:
    selected = set(episode["mission_sector"]["selected_cell_ids"])  # type: ignore[index]
    for region in task["inspection_atlas"]["regions"]:  # type: ignore[index]
        for cell in region["cells"]:
            if cell["cell_id"] in selected:
                return cell
    raise AssertionError("expected a selected public atlas cell")


def _observation(
    *,
    episode: dict[str, object],
    cell: dict[str, object],
    timestamp_s: float,
) -> ObservationPacket:
    pose = cell["pose"]
    return ObservationPacket(
        episode_id=str(episode["episode_id"]),
        observation_id=f"obs-{timestamp_s:.1f}",
        drone_id=str(episode["starts"][0]["drone_id"]),  # type: ignore[index]
        sequence=int(round(timestamp_s * 5)),
        timestamp_s=timestamp_s,
        pose=Pose3D(
            tuple(float(value) for value in pose["position"]),  # type: ignore[index]
            float(pose["yaw_deg"]),  # type: ignore[index]
        ),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1.0,
        sensor_pitch_deg=float(pose["pitch_deg"]),  # type: ignore[index]
    )


def test_l1_evidence_requires_accepted_safe_continuous_legal_observation(
    public_g2_i_inputs,
) -> None:
    city, task, episode = public_g2_i_inputs
    evidence = L1MeasurementEvidence(city=city, task_spec=task, public_episode=episode)
    cell = _first_selected_cell(task, episode)
    drone_id = str(episode["starts"][0]["drone_id"])
    position = tuple(float(value) for value in cell["pose"]["position"])
    for timestamp_s in (0.0, 0.2, 0.4, 0.6):
        evidence.record_observe(
            _observation(episode=episode, cell=cell, timestamp_s=timestamp_s),
            evaluator_accepted=True,
            runtime_safe=True,
        )
        evidence.record_measured_positions(
            timestamp_s + 0.2,
            {drone_id: position},
            safe_drone_ids={drone_id},
        )
    snapshot = evidence.snapshot(measured_state_trace=[{"tick": 0}], input_bindings_hash="a" * 64)
    assert snapshot["inspection_cell_count_trace"][-1][1] >= 1
    assert snapshot["inspection_coverage_trace"][-1][1] > 0.0
    assert snapshot["coverage_trace"][-1][2] >= 1


def test_l1_evidence_drops_unsafe_observation_credit(public_g2_i_inputs) -> None:
    city, task, episode = public_g2_i_inputs
    evidence = L1MeasurementEvidence(city=city, task_spec=task, public_episode=episode)
    cell = _first_selected_cell(task, episode)
    drone_id = str(episode["starts"][0]["drone_id"])
    position = tuple(float(value) for value in cell["pose"]["position"])
    evidence.record_observe(
        _observation(episode=episode, cell=cell, timestamp_s=0.0),
        evaluator_accepted=True,
        runtime_safe=True,
    )
    evidence.record_observe(
        _observation(episode=episode, cell=cell, timestamp_s=0.2),
        evaluator_accepted=True,
        runtime_safe=False,
    )
    evidence.record_measured_positions(0.2, {drone_id: position}, safe_drone_ids=set())
    snapshot = evidence.snapshot(measured_state_trace=[{"tick": 0}], input_bindings_hash="b" * 64)
    assert snapshot["inspection_cell_count_trace"][-1][1] == 0


def test_l1_evidence_snapshot_validator_rejects_trace_tampering(public_g2_i_inputs) -> None:
    city, task, episode = public_g2_i_inputs
    evidence = L1MeasurementEvidence(city=city, task_spec=task, public_episode=episode)
    drone_id = str(episode["starts"][0]["drone_id"])
    position = tuple(float(value) for value in episode["starts"][0]["position"])
    trace = [{"action_sequence": 0, "positions_w_m": {drone_id: list(position)}}]
    evidence.record_measured_positions(0.2, {drone_id: position}, safe_drone_ids={drone_id})
    snapshot = evidence.snapshot(measured_state_trace=trace, input_bindings_hash="c" * 64)

    validate_measurement_evidence_snapshot(
        snapshot,
        measured_state_trace=trace,
        input_bindings_hash="c" * 64,
    )
    tampered = copy.deepcopy(snapshot)
    tampered["inspection_coverage_trace"][0][0] = 0.3
    with pytest.raises(ValueError, match="timestamps disagree"):
        validate_measurement_evidence_snapshot(
            tampered,
            measured_state_trace=trace,
            input_bindings_hash="c" * 64,
        )
