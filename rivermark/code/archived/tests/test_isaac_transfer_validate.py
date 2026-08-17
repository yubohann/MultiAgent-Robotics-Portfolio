from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_scene import AGENT_COUNT
from rivermark_benchmark.isaac_runtime_safety import (
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RUNTIME_SAFETY_PHASE_CODES,
    RUNTIME_SAFETY_SCHEMA,
    RUNTIME_SAFETY_TRACE_SCHEMA,
    physics_time_ns,
)
from rivermark_benchmark.isaac_transfer import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    EXCLUDED_POLICY_INPUTS,
    STATE_FIELDS,
    STATE_ONLY_PROFILE,
    TRANSFER_SCHEMA,
    TRANSFER_SOURCE,
    CityLiteRouteAnchorTransform,
    WorldCommandBounds,
    derive_physical_state_8d,
)
from rivermark_benchmark.isaac_transfer_validate import (
    RUNTIME_SAFETY_PATH,
    STATE_ACTION_PATH,
    TRACE_FIELDS,
    TRACE_PATH,
    TRACE_PROVENANCE_PATH,
    TRANSFER_VALIDATION_SCHEMA,
    validate_isaac_state_only_transfer,
    write_transfer_validation_receipt,
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _refresh_capture_bindings(capture: Path) -> None:
    """Rebind the synthetic evidence after deliberate semantic tampering."""

    trace = capture / TRACE_PATH
    provenance_path = capture / TRACE_PROVENANCE_PATH
    scene_path = capture / "scene.json"
    receipt_path = capture / "capture_receipt.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["state_action_sha256"] = _sha256(capture / STATE_ACTION_PATH)
    provenance["trace_sha256"] = _sha256(trace)
    _write_json(provenance_path, provenance)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["control_transfer_trace_sha256"] = _sha256(trace)
    scene["control_transfer_provenance_sha256"] = _sha256(provenance_path)
    _write_json(scene_path, scene)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["task"]["decision_trace_sha256"] = _sha256(trace)
    artifacts = {}
    for relative in ("scene.json", STATE_ACTION_PATH, TRACE_PATH, TRACE_PROVENANCE_PATH, RUNTIME_SAFETY_PATH):
        path = capture / Path(*relative.split("/"))
        artifacts[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt["artifact_hashes"] = artifacts
    _write_json(receipt_path, receipt)
    (capture / "capture_receipt.sha256").write_text(
        f"{_sha256(receipt_path)}  capture_receipt.json\n", encoding="ascii"
    )


def _fixture(root: Path) -> Path:
    capture = root / "capture"
    capture.mkdir(parents=True)
    steps = 5
    warmup_steps = 1
    dt_s = 0.005
    cadence = 2
    transform = CityLiteRouteAnchorTransform.from_public_routes()
    mean = np.asarray([0.10, -0.20, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    standard_deviation = np.asarray([1.0, 1.1, 0.9, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float64)
    action_scale = np.asarray([0.75, 0.75, 0.05, 0.8], dtype=np.float64)
    bounds = WorldCommandBounds(0.75, 0.05, 0.8)

    agent_offsets = np.arange(AGENT_COUNT, dtype=np.float64)
    pre_pos = np.empty((steps, AGENT_COUNT, 3), dtype=np.float64)
    pre_quat = np.empty((steps, AGENT_COUNT, 4), dtype=np.float64)
    pre_lin = np.empty((steps, AGENT_COUNT, 3), dtype=np.float64)
    pre_ang = np.empty((steps, AGENT_COUNT, 3), dtype=np.float64)
    for step in range(steps):
        pre_pos[step] = transform.anchors_w_m
        pre_pos[step, :, 0] += 0.04 * step
        pre_pos[step, :, 1] += 0.01 * agent_offsets
        yaw = 0.03 * step + 0.01 * agent_offsets
        pre_quat[step, :, 0] = np.cos(yaw / 2.0)
        pre_quat[step, :, 1] = 0.0
        pre_quat[step, :, 2] = 0.0
        pre_quat[step, :, 3] = np.sin(yaw / 2.0)
        pre_lin[step, :, 0] = 0.04
        pre_lin[step, :, 1] = 0.01
        pre_lin[step, :, 2] = 0.0
        pre_ang[step, :, 0] = 0.0
        pre_ang[step, :, 1] = 0.0
        pre_ang[step, :, 2] = 0.02

    decision_steps = np.asarray(range(0, steps, cadence), dtype=np.int64)
    trace_rows: dict[str, list[np.ndarray | int]] = {key: [] for key in TRACE_FIELDS}
    emitted_by_decision: list[np.ndarray] = []
    altitude_reference = pre_pos[0, :, 2].copy()
    for decision_index, step in enumerate(decision_steps):
        physical = derive_physical_state_8d(
            pre_pos[step],
            pre_lin[step],
            pre_quat[step],
            pre_ang[step],
            agent_ids=range(AGENT_COUNT),
        )
        pilot = transform.physical_to_pilot(physical)
        normalized_observation = (pilot.values - mean) / standard_deviation
        raw_action = np.column_stack(
            (
                np.full(AGENT_COUNT, 1.20 + 0.03 * decision_index),
                np.full(AGENT_COUNT, -1.30),
                np.full(AGENT_COUNT, 1.10),
                np.full(AGENT_COUNT, 1.40),
            )
        )
        normalized_action = np.clip(raw_action, -1.0, 1.0)
        local_command = normalized_action * action_scale
        world_velocity = transform.pilot_velocity_to_world(local_command[:, :3], agent_ids=range(AGENT_COUNT))
        bounded_velocity, bounded_yaw_rate = bounds.apply(world_velocity, local_command[:, 3])
        prebound = np.concatenate((world_velocity, local_command[:, 3:4]), axis=1)
        emitted = np.concatenate((bounded_velocity, bounded_yaw_rate[:, None]), axis=1)
        emitted_by_decision.append(emitted)
        values: dict[str, np.ndarray | int] = {
            "rollout_physics_step": int(step),
            "command_time_ns": physics_time_ns(warmup_steps + int(step), dt_s),
            "effective_time_ns": physics_time_ns(warmup_steps + int(step) + 1, dt_s),
            "decision_index": decision_index,
            "physical_state_8d": physical.values,
            "pilot_state_8d": pilot.values,
            "normalized_observation_8d": normalized_observation,
            "raw_action": raw_action,
            "normalized_action": normalized_action,
            "local_velocity_yaw_command": local_command,
            "prebound_world_velocity_yaw_command": prebound,
            "emitted_world_velocity_yaw_command": emitted,
            "altitude_reference_w_m": altitude_reference,
        }
        for key, value in values.items():
            trace_rows[key].append(value)
    trace_payload = {
        key: np.asarray(values, dtype=np.int64)
        if key in {"rollout_physics_step", "command_time_ns", "effective_time_ns", "decision_index"}
        else np.stack(values, axis=0)
        for key, values in trace_rows.items()
    }
    trace_path = capture / TRACE_PATH
    trace_path.parent.mkdir(parents=True)
    np.savez_compressed(trace_path, **trace_payload)

    held_emitted = np.stack(
        [emitted_by_decision[step // cadence] for step in range(steps)], axis=0
    )
    state_payload = {
        "command_time_ns": np.asarray(
            [physics_time_ns(warmup_steps + step, dt_s) for step in range(steps)], dtype=np.int64
        ),
        "effective_time_ns": np.asarray(
            [physics_time_ns(warmup_steps + step + 1, dt_s) for step in range(steps)], dtype=np.int64
        ),
        "root_pos_w_m": pre_pos + np.asarray([0.002, 0.0, 0.0]),
        "root_quat_wxyz": pre_quat.copy(),
        "root_lin_vel_w_mps": pre_lin.copy(),
        "root_ang_vel_b_radps": pre_ang.copy(),
        "desired_pos_w_m": np.zeros((steps, AGENT_COUNT, 3), dtype=np.float64),
        "desired_vel_w_mps": np.zeros((steps, AGENT_COUNT, 3), dtype=np.float64),
        "target_thrust_n": np.zeros((steps, AGENT_COUNT, 4), dtype=np.float64),
        "applied_thrust_n": np.zeros((steps, AGENT_COUNT, 4), dtype=np.float64),
        "pre_command_root_pos_w_m": pre_pos,
        "pre_command_root_quat_wxyz": pre_quat,
        "pre_command_root_lin_vel_w_mps": pre_lin,
        "pre_command_root_ang_vel_b_radps": pre_ang,
        "emitted_world_velocity_yaw_command": held_emitted,
    }
    state_path = capture / STATE_ACTION_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, **state_payload)

    frame_count = 1 + warmup_steps + steps
    runtime_positions = np.concatenate(
        (
            pre_pos[0:1],
            pre_pos[0:1],
            state_payload["root_pos_w_m"],
        ),
        axis=0,
    )
    runtime_path = capture / RUNTIME_SAFETY_PATH
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        runtime_path,
        physics_step=np.arange(frame_count, dtype=np.int64),
        physics_time_ns=np.asarray([physics_time_ns(step, dt_s) for step in range(frame_count)], dtype=np.int64),
        phase_code=np.asarray(
            [RUNTIME_SAFETY_PHASE_CODES["post_reset"]]
            + [RUNTIME_SAFETY_PHASE_CODES["warmup"]] * warmup_steps
            + [RUNTIME_SAFETY_PHASE_CODES["rollout"]] * steps,
            dtype=np.int8,
        ),
        frame_outcome_code=np.full(frame_count, RUNTIME_SAFETY_FRAME_OUTCOME_CODES["passed"], dtype=np.uint8),
        root_pos_w_m=runtime_positions,
        net_contact_forces_w_n=np.zeros((frame_count, AGENT_COUNT, 1, 3), dtype=np.float64),
        max_contact_force_n=np.zeros((frame_count,), dtype=np.float64),
    )

    transfer = {
        "schema": TRANSFER_SCHEMA,
        "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        "formal_benchmark_admission": False,
        "physical_training": False,
        "isaac_training": False,
        "information_profile": STATE_ONLY_PROFILE,
        "policy_input_fields": list(STATE_FIELDS),
        "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
        "observation_mean": mean.tolist(),
        "observation_std": standard_deviation.tolist(),
        "action_scale": action_scale.tolist(),
        "action_source": TRANSFER_SOURCE,
        "decision_cadence_physics_steps": cadence,
        "coordinate_transform": transform.provenance(),
        "world_command_bounds": {
            "max_horizontal_speed_mps": 0.75,
            "max_vertical_speed_mps": 0.05,
            "max_yaw_rate_rad_s": 0.8,
        },
        "policy": {
            "method_id": "sb3_checkpoint_policy",
            "implementation_kind": "trained_sb3_pilot_checkpoint",
            "external_dependency": "stable_baselines3",
            "checkpoint": "C:/external/pilot.zip",
            "checkpoint_sha256": "a" * 64,
            "adapter_metadata": "C:/external/pilot.rivermark.json",
            "adapter_metadata_sha256": "b" * 64,
            "algorithm": "ppo",
            "parameter_sharing": "independent_shared_policy_per_agent",
        },
    }
    provenance = {
        "schema": "org.rivermark.isaac-sb3-state-transfer-trace.v1",
        "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
        "formal_benchmark_admission": False,
        "dataset_episode": False,
        "task_kind": "state_only_control_transfer_smoke",
        "task_variant_id": "isaac-eight-agent-sb3-state-only-control-transfer-smoke-v1",
        "control_mode": "sb3_state_only_transfer",
        "state_phase": "pre_sim_command_state",
        "state_action_state_phase": "pre_sim_command_state",
        "state_action_path": STATE_ACTION_PATH,
        "state_action_sha256": _sha256(state_path),
        "trace_path": TRACE_PATH,
        "trace_sha256": _sha256(trace_path),
        "trace_decision_count": len(decision_steps),
        "trace_fields": list(TRACE_FIELDS),
        "transfer": transfer,
    }
    provenance_path = capture / TRACE_PROVENANCE_PATH
    _write_json(provenance_path, provenance)
    scene_path = capture / "scene.json"
    _write_json(
        scene_path,
        {
            "capture_control_mode": "sb3_state_only_transfer",
            "control_transfer_task_kind": "state_only_control_transfer_smoke",
            "control_transfer_state_phase": "pre_sim_command_state",
            "control_transfer_policy_input": "state_only_8d",
            "control_transfer_trace_sha256": _sha256(trace_path),
            "control_transfer_provenance_sha256": _sha256(provenance_path),
        },
    )
    receipt = {
        "schema": "org.rivermark.isaac-swarm-capture.v1",
        "status": "captured",
        "ok": True,
        "task_kind": "state_only_control_transfer_smoke",
        "information_profile": "state_only",
        "command": {
            "steps": steps,
            "warmup_steps": warmup_steps,
            "dt_s": dt_s,
            "control_mode": "sb3_state_only_transfer",
            "sb3_state_only_transfer": {
                "checkpoint": "C:/external/pilot.zip",
                "metadata": "C:/external/pilot.rivermark.json",
                "decision_stride_physics_steps": cadence,
                "world_command_bounds": {
                    "max_horizontal_speed_mps": 0.75,
                    "max_vertical_speed_mps": 0.05,
                    "max_yaw_rate_rad_s": 0.8,
                },
            },
        },
        "task": {
            "task_kind": "state_only_control_transfer_smoke",
            "task_variant_id": "isaac-eight-agent-sb3-state-only-control-transfer-smoke-v1",
            "evaluation": "not_a_search_result",
            "private_targets_present": False,
            "decision_trace": TRACE_PATH,
            "decision_trace_sha256": _sha256(trace_path),
        },
        "claim_boundary": {
            "formal_benchmark_admission": False,
            "development_control_transfer": True,
            "isaac_training": False,
            "physical_training": False,
            "hardware_validated": False,
            "radar_profile_eligible": False,
            "foundation_model_executed": False,
            "semantic_labels_policy_visible": False,
        },
        "modalities": {
            "rtx_radar": "not_captured",
            "hardware_radar": "not_captured",
            "real_flight": "not_captured",
            "body_state": "captured_state_only_policy_input",
        },
        "state_only_transfer": transfer,
        "runtime_safety_guard": {
            "schema": RUNTIME_SAFETY_SCHEMA,
            "enabled": True,
            "fail_closed": True,
            "status": "passed",
            "evidence": {
                "schema": RUNTIME_SAFETY_TRACE_SCHEMA,
                "path": RUNTIME_SAFETY_PATH,
                "sha256": _sha256(runtime_path),
                "physics_frame_count": frame_count,
            },
            "checks": {
                "warmup_physics_steps_checked": warmup_steps,
                "rollout_physics_steps_checked": steps,
                "contact_samples_checked": frame_count,
                "contact_abort_count": 0,
            },
        },
    }
    _write_json(capture / "capture_receipt.json", receipt)
    _refresh_capture_bindings(capture)
    return capture


class IsaacTransferValidateTests(unittest.TestCase):
    def test_valid_pure_python_replay_writes_development_only_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _fixture(Path(temporary))
            report = validate_isaac_state_only_transfer(capture)
            self.assertTrue(report.valid, [issue.code for issue in report.issues])
            destination = write_transfer_validation_receipt(report, capture / "independent_validation.json")
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], TRANSFER_VALIDATION_SCHEMA)
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(payload["development_only"])
            self.assertFalse(payload["formal_benchmark_admission"])
            self.assertFalse(payload["dataset_episode"])
            self.assertEqual(payload["issues"], [])

    def test_rejects_rebound_trace_action_that_fails_command_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _fixture(Path(temporary))
            trace_path = capture / TRACE_PATH
            trace = _load_npz(trace_path)
            trace["raw_action"][1, 0, 2] = 0.20
            np.savez_compressed(trace_path, **trace)
            _refresh_capture_bindings(capture)
            report = validate_isaac_state_only_transfer(capture)
            self.assertFalse(report.valid)
            self.assertIn("trace_normalized_action", {issue.code for issue in report.issues})

    def test_rejects_rebound_trace_timing_without_matching_precommand_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _fixture(Path(temporary))
            trace_path = capture / TRACE_PATH
            trace = _load_npz(trace_path)
            trace["command_time_ns"][1] += 1
            np.savez_compressed(trace_path, **trace)
            _refresh_capture_bindings(capture)
            report = validate_isaac_state_only_transfer(capture)
            self.assertFalse(report.valid)
            codes = {issue.code for issue in report.issues}
            self.assertIn("trace_timing", codes)
            self.assertIn("trace_state_timing_binding", codes)

    def test_rejects_invalid_development_claim_after_receipt_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _fixture(Path(temporary))
            receipt_path = capture / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["claim_boundary"]["isaac_training"] = True
            _write_json(receipt_path, receipt)
            _refresh_capture_bindings(capture)
            report = validate_isaac_state_only_transfer(capture)
            self.assertFalse(report.valid)
            self.assertIn("claim_boundary", {issue.code for issue in report.issues})

    def test_rejects_artifact_tamper_before_receipt_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture = _fixture(Path(temporary))
            trace_path = capture / TRACE_PATH
            trace = _load_npz(trace_path)
            trace["raw_action"][0, 0, 0] += 0.10
            np.savez_compressed(trace_path, **trace)
            report = validate_isaac_state_only_transfer(capture)
            self.assertFalse(report.valid)
            self.assertIn("artifact_hash", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
