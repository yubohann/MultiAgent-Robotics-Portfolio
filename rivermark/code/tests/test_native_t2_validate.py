from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from rivermark_benchmark import native_t2_validate
from rivermark_benchmark.cf2x_runtime_calibration import calibration_report_sha256
from rivermark_benchmark.frame_archive import (
    ChunkedFrameArchive,
    write_chunked_frame_archive,
)
from rivermark_benchmark.isaac_capture import (
    CONTROL_MODE_NATIVE_T2_CANARY,
    NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH,
    NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
    NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
    NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
    NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
    NATIVE_T2_TASK_KIND,
    _overview_archive_frame_indices,
)
from rivermark_benchmark.isaac_transfer import (
    FixedDecisionCadence,
    WorldCommandBounds,
    derive_physical_state_8d,
)
from rivermark_benchmark.isaac_validate import (
    IsaacValidationReport,
    _native_t2_validator_sha256,
    validate_isaac_capture,
    write_validation_receipt,
)
from rivermark_benchmark.native_t2_canary import (
    NATIVE_T2_EVENTS_SCHEMA,
    NATIVE_T2_TRACE_SCHEMA,
    PublicRouteCoveragePolicy,
    SpatialCandidateDeduplicator,
    bind_native_t2_calibration,
    native_semantic_rgbd_candidates,
)
from rivermark_benchmark.native_t2_validate import (
    NATIVE_T2_EXPECTED_ARTIFACTS,
    NativeT2ValidationResult,
    validate_native_t2_capture,
)
from rivermark_benchmark.collection_protocol import (
    native_t2_v2_motion_contract,
    native_t2_v3_motion_contract,
)
from rivermark_benchmark.private_evaluator_manifest import (
    NATIVE_T2_TASK_VARIANT_ID,
    NATIVE_T2_V2_TASK_VARIANT_ID,
    NATIVE_T2_V3_TASK_VARIANT_ID,
)
from rivermark_benchmark.runtime_lock import runtime_lock_sha256
from rivermark_benchmark.t2_policy_abi import (
    T2CandidateEventJournal,
    T2NativeStepEvidence,
    T2PolicyRunner,
    T2PublicFleetObservation,
    T2PublicSensorObservation,
)
from rivermark_benchmark.video import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _native_evidence_summary(root: Path) -> dict[str, object]:
    trace_records = [
        json.loads(line)
        for line in (root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    event_payload = _read_json(root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH)
    source_observations = event_payload["source_observations"]
    journal = event_payload["candidate_event_journal"]
    assert isinstance(source_observations, list)
    assert isinstance(journal, dict)
    submission = journal["submission"]
    assert isinstance(submission, dict)
    events = submission["events"]
    assert isinstance(events, list)
    with np.load(root / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH, allow_pickle=False) as archive:
        frame_count = int(archive["timestamps_ns"].shape[0])
    return {
        "task_variant_id": NATIVE_T2_TASK_VARIANT_ID,
        "claim_boundary": "development_native_t2_canary_only",
        "decision_trace": {
            "path": NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
            "sha256": sha256_file(root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH),
            "decision_count": sum(record["record_type"] == "decision" for record in trace_records),
            "physical_step_count": sum(
                record["record_type"] == "physical_step" for record in trace_records
            ),
        },
        "candidate_events": {
            "path": NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
            "sha256": sha256_file(root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH),
            "source_observation_count": len(source_observations),
            "event_count": len(events),
        },
        "camera_extrinsics": {
            "path": NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH,
            "sha256": sha256_file(root / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH),
            "frame_count": frame_count,
            "world_camera_closure": "T_world_camera_from_verified_render_facing_usd_pose_converted_to_ros",
        },
    }


def _rebind_native_fixture(root: Path) -> None:
    receipt_path = root / "capture_receipt.json"
    receipt = _read_json(receipt_path)
    receipt["native_t2_evidence"] = _native_evidence_summary(root)
    receipt["artifact_hashes"] = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(NATIVE_T2_EXPECTED_ARTIFACTS)
    }
    _write_json(receipt_path, receipt)
    (root / "capture_receipt.sha256").write_text(
        f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
    )


def _native_t2_fixture(
    tmp_path: Path, *, decision_stride_physics_steps: int = 1
) -> tuple[Path, Path, Path, Path]:
    """Build a CPU-only native-T2 artifact from the checked common fixture.

    The imported fixture is used solely for the already-tested City-Lite scene,
    runtime-safety, contact, and sensor-phase artifacts.  This helper replaces
    the policy/action/event path with the native-T2 contract under test.
    """

    from test_cf2x_runtime_calibration import _report
    from test_isaac_validate import _capture_fixture
    from test_runtime_lock import _lock

    root = tmp_path / "native-t2-capture"
    private_manifest = tmp_path / "native-t2-private.json"
    _capture_fixture(root, private_manifest, steps=4, capture_stride=1)

    manifest = _read_json(private_manifest)
    manifest["task_variant_id"] = NATIVE_T2_TASK_VARIANT_ID
    canary_binding = {
        "protocol_id": "citylite-native-t2-canary-v1",
        "protocol_sha256": "a" * 64,
        "cell_id": "native-t2-canary-inner-dev-v1",
        "split": "inner_dev",
        "episode_index": 0,
        "episode_seed": 17,
    }
    manifest["collection_binding"] = canary_binding
    _write_json(private_manifest, manifest)

    for relative in ("streams/public_task.npz", "streams/public_messages.npz"):
        (root / relative).unlink()

    state_path = root / "streams/state_action.npz"
    with np.load(state_path, allow_pickle=False) as archive:
        state = {name: archive[name].copy() for name in archive.files}
    steps = int(state["command_time_ns"].shape[0])
    state["pre_command_root_pos_w_m"] = state["root_pos_w_m"].copy()
    state["pre_command_root_quat_wxyz"] = state["root_quat_wxyz"].copy()
    state["pre_command_root_lin_vel_w_mps"] = state["root_lin_vel_w_mps"].copy()
    state["pre_command_root_ang_vel_b_radps"] = state["root_ang_vel_b_radps"].copy()
    state["emitted_world_velocity_yaw_command"] = np.zeros((steps, 8, 4), dtype=np.float64)

    original_task = _read_json(root / "public_task.json")
    routes = np.asarray(original_task["routes_w_m"], dtype=np.float64)
    dt_s = 0.005
    bounds = WorldCommandBounds(
        max_horizontal_speed_mps=0.75,
        max_vertical_speed_mps=0.10,
        max_yaw_rate_rad_s=0.80,
    )
    policy = PublicRouteCoveragePolicy(
        routes, waypoint_segment_seconds=5.0, route_start_time_ns=0
    )
    runner = T2PolicyRunner(
        policy,
        cadence=FixedDecisionCadence(decision_stride_physics_steps),
        bounds=bounds,
    )

    public_task = {
        "schema": "org.rivermark.public-search-task.v1",
        "task_kind": NATIVE_T2_TASK_KIND,
        "task_variant_id": NATIVE_T2_TASK_VARIANT_ID,
        "agent_count": 8,
        "nominal_object_count": 4,
        "route_generation": "fixed-public-cell-coverage-v1",
        "route_conditioning": "public_only",
        "route_family_id": original_task["route_family_id"],
        "start_anchor_id": original_task["start_anchor_id"],
        "waypoint_segment_seconds": 5.0,
        "routes_w_m": routes.tolist(),
        "route_contract": original_task["route_contract"],
        "policy_abi": {
            "kind": "bounded_public_state_velocity_yaw",
            "state_visible_to_policy": True,
            "semantic_target_ids_visible_to_policy": False,
            "private_evaluator_inputs_visible_to_policy": False,
            "action_fields": [
                "velocity_x_mps",
                "velocity_y_mps",
                "velocity_z_mps",
                "yaw_rate_radps",
            ],
            "decision_stride_physics_steps": decision_stride_physics_steps,
        },
        "candidate_event_abi": {
            "kind": "native_rgbd_semantic_anonymous_candidate",
            "minimum_pixels": NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
            "merge_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
            "private_target_ids_released": False,
        },
        "evaluator_contract": {
            "schema": "org.rivermark.native-t2-private-evaluation-contract.v1",
            "event_time_origin": "post_warmup_physics_time",
            "time_budget_s": steps * dt_s,
            "match_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
            "maximum_false_confirmations": 0,
            "minimum_verified_matches": 1,
            "observation_time_tolerance_s": 0.0,
            "target_count_source": "external_private_evaluator_manifest",
        },
        "object_coordinates_in_policy_inputs": False,
    }
    _write_json(root / "public_task.json", public_task)
    scene = _read_json(root / "scene.json")
    scene["public_task_sha256"] = sha256_file(root / "public_task.json")
    scene["private_evaluator_manifest_sha256"] = sha256_file(private_manifest)
    _write_json(root / "scene.json", scene)

    report = _report()
    runtime_lock = _lock()
    runtime_lock_path = tmp_path / "runtime-lock.json"
    _write_json(runtime_lock_path, runtime_lock)
    lock_sha256 = runtime_lock_sha256(runtime_lock)
    usd_sha256 = "e" * 64
    report["asset"]["usd_sha256"] = usd_sha256
    report["static_usd"]["usd_sha256"] = usd_sha256
    report["runtime_lock_sha256"] = lock_sha256
    report["report_sha256"] = calibration_report_sha256(report)
    calibration_path = tmp_path / "cf2x-calibration.json"
    _write_json(calibration_path, report)
    allocation = np.asarray(report["runtime"]["allocation_matrix"], dtype=np.float64)

    decisions = []
    physical = []
    current_decision = None
    for step in range(steps):
        observation = T2PublicFleetObservation.from_rigid_body_state(
            physics_step=step,
            command_time_ns=int(state["command_time_ns"][step]),
            position_w_m=state["pre_command_root_pos_w_m"][step],
            linear_velocity_w_mps=state["pre_command_root_lin_vel_w_mps"][step],
            quaternion_wxyz=state["pre_command_root_quat_wxyz"][step],
            angular_velocity_b_radps=state["pre_command_root_ang_vel_b_radps"][step],
        )
        if step % decision_stride_physics_steps == 0:
            current_decision = runner.decide(observation)
            decisions.append(
                {
                    "schema": NATIVE_T2_TRACE_SCHEMA,
                    "record_type": "decision",
                    "rollout_physics_step": step,
                    "decision_sha256": current_decision.sha256,
                    "decision": current_decision.public_dict(),
                }
            )
        assert current_decision is not None
        state["emitted_world_velocity_yaw_command"][step] = (
            current_decision.action.emitted_velocity_yaw_command
        )
        evidence = T2NativeStepEvidence(
            decision=current_decision,
            applied_physics_step=step + 1,
            physical_command_time_ns=int(state["command_time_ns"][step]),
            effective_time_ns=int(state["effective_time_ns"][step]),
            requested_thrust_n=state["target_thrust_n"][step],
            applied_thrust_n=state["applied_thrust_n"][step],
            applied_wrench_body=state["applied_thrust_n"][step] @ allocation.T,
            post_step_state_8d=derive_physical_state_8d(
                state["root_pos_w_m"][step],
                state["root_lin_vel_w_mps"][step],
                state["root_quat_wxyz"][step],
                state["root_ang_vel_b_radps"][step],
                agent_ids=range(8),
            ).values,
        )
        physical.append(
            {
                "schema": NATIVE_T2_TRACE_SCHEMA,
                "record_type": "physical_step",
                "rollout_physics_step": step,
                "global_applied_physics_step": step + 1,
                "evidence": evidence.public_dict(),
            }
        )
    _write_npz(state_path, state)
    provenance = {
        "schema": NATIVE_T2_TRACE_SCHEMA,
        "record_type": "provenance",
        "claim_boundary": "development_native_t2_canary_only",
        "capture_attempt_id": "attempt-native-t2",
        "policy": policy.provenance(),
        "policy_abi": runner.provenance(),
    }
    trace_path = root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
    trace_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in (provenance, *decisions, *physical)
        ),
        encoding="utf-8",
    )

    onboard_path = root / "sensors/onboard_rgbd.npz"
    semantic_path = root / "learning_labels/semantic_segmentation.npz"
    with np.load(onboard_path, allow_pickle=False) as archive:
        onboard = {name: archive[name].copy() for name in archive.files}
    with np.load(semantic_path, allow_pickle=False) as archive:
        semantic = {name: archive[name].copy() for name in archive.files}
    timestamps = onboard["timestamps_ns"]
    depth = np.full_like(onboard["distance_to_image_plane_m"], 20.0)
    onboard["distance_to_image_plane_m"] = depth
    with (root / "learning_labels/semantic_frame_metadata.jsonl").open(
        "r", encoding="utf-8"
    ) as stream:
        semantic_metadata = [json.loads(line) for line in stream]
    targets = manifest["targets"]
    assert isinstance(targets, list)
    # The common fixture labels one target slot per camera, while this canary
    # fixture deliberately owns only the four externally committed targets.
    # Keep the remaining camera-local numeric IDs as public building labels,
    # not phantom target classes.
    for row in semantic_metadata:
        mapping = row["onboard_replicator_info"]["per_camera"]
        for agent_id in range(len(targets), 8):
            labels = mapping[agent_id]["id_to_labels"]
            labels.pop(str(agent_id + 1), None)
        mapping[4]["id_to_labels"]["5"] = {"class": "building"}
    (root / "learning_labels/semantic_frame_metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in semantic_metadata),
        encoding="utf-8",
    )
    per_camera = semantic_metadata[0]["onboard_replicator_info"]
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 8, axis=0)
    quaternions = np.zeros((8, 4), dtype=np.float64)
    quaternions[:, 0] = 1.0
    zero_positions = np.zeros((8, 3), dtype=np.float64)
    initial_points = native_t2_validate._native_world_points(
        depth[0], intrinsics, zero_positions, quaternions
    )
    initial_candidates = native_semantic_rgbd_candidates(
        semantic["semantic_segmentation"][0],
        per_camera,
        initial_points,
        minimum_pixels=NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
    )
    camera_positions = np.zeros((len(timestamps), 8, 3), dtype=np.float64)
    for agent_id, target in enumerate(targets):
        assert isinstance(target, dict)
        candidate = initial_candidates[agent_id][0]
        camera_positions[:, agent_id] = (
            np.asarray(target["position_w_m"], dtype=np.float64)
            - np.asarray(candidate.position_w_m, dtype=np.float64)
        )
    extrinsics_path = root / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH
    _write_npz(
        extrinsics_path,
        {
            "timestamps_ns": timestamps,
            "pos_w_m": camera_positions,
            "quat_w_ros": np.repeat(quaternions[None, :, :], len(timestamps), axis=0),
            "intrinsic_matrices": np.repeat(intrinsics[None, :, :, :], len(timestamps), axis=0),
        },
    )
    write_chunked_frame_archive(
        onboard_path,
        timestamps_ns=timestamps,
        inline_fields={},
        frame_fields={
            "rgb": onboard["rgb"],
            "distance_to_image_plane_m": onboard["distance_to_image_plane_m"],
        },
    )
    write_chunked_frame_archive(
        semantic_path,
        timestamps_ns=timestamps,
        inline_fields={},
        frame_fields={"semantic_segmentation": semantic["semantic_segmentation"]},
    )
    lidar_path = root / "sensors/lidar.npz"
    with np.load(lidar_path, allow_pickle=False) as archive:
        lidar = {name: archive[name].copy() for name in archive.files}
    lidar["ranges_m"].fill(35.0)
    _write_npz(lidar_path, lidar)

    overview_path = root / "sensors/overview_rgb.npz"
    with np.load(overview_path, allow_pickle=False) as archive:
        overview = {name: archive[name].copy() for name in archive.files}
    overview_indices = np.asarray(
        _overview_archive_frame_indices(len(timestamps)), dtype=np.int64
    )
    write_chunked_frame_archive(
        overview_path,
        timestamps_ns=overview["timestamps_ns"][overview_indices],
        inline_fields={
            "camera_pos_w_m": overview["camera_pos_w_m"][overview_indices],
            "camera_quat_wxyz": overview["camera_quat_wxyz"][overview_indices],
            "target_w_m": overview["target_w_m"][overview_indices],
        },
        frame_fields={
            "rgb": overview["rgb"][overview_indices],
            "semantic_segmentation": overview["semantic_segmentation"][overview_indices],
        },
    )

    journal = T2CandidateEventJournal(
        episode_id="attempt-native-t2", event_time_origin_ns=0
    )
    deduplicator = SpatialCandidateDeduplicator(NATIVE_T2_CANDIDATE_MERGE_RADIUS_M)
    source_observations = []
    for frame_index, timestamp in enumerate(timestamps):
        points = native_t2_validate._native_world_points(
            depth[frame_index],
            intrinsics,
            camera_positions[frame_index],
            quaternions,
        )
        rows = native_semantic_rgbd_candidates(
            semantic["semantic_segmentation"][frame_index],
            semantic_metadata[frame_index]["onboard_replicator_info"],
            points,
            minimum_pixels=NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
        )
        for agent_id, candidates in enumerate(rows):
            observation = T2PublicSensorObservation(
                agent_id=agent_id,
                capture_frame_index=frame_index,
                sensor_time_ns=int(timestamp),
            )
            source_observations.append(observation.public_dict())
            journal.append(observation, deduplicator.filter(candidates))
    event_payload = {
        "schema": NATIVE_T2_EVENTS_SCHEMA,
        "claim_boundary": "development_native_t2_canary_only",
        "formal_benchmark_admission": False,
        "capture_attempt_id": "attempt-native-t2",
        "decision_trace": NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
        "decision_trace_sha256": sha256_file(trace_path),
        "source_observations": source_observations,
        "candidate_event_journal": journal.public_dict(),
        "event_time_origin_ns": 0,
        "candidate_detection_is_public_rgbd_semantic_only": True,
        "private_evaluator_payload_released": False,
    }
    _write_json(root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH, event_payload)

    receipt = _read_json(root / "capture_receipt.json")
    receipt.update(
        {
            "task_kind": NATIVE_T2_TASK_KIND,
            "information_profile": "state_only_control_plus_rgbd_semantic_events",
            "source_worktree_dirty": False,
            "capture_attempt_id": "attempt-native-t2",
            "command": {
                "control_mode": CONTROL_MODE_NATIVE_T2_CANARY,
                "steps": steps,
                "warmup_steps": 0,
                "capture_stride": 1,
                "dt_s": dt_s,
                "drone_usd_sha256": usd_sha256,
            },
            "claim_boundary": {
                "formal_benchmark_admission": False,
                "development_native_t2_canary": True,
            },
            "runtime_lock": {"sha256": lock_sha256},
            "city_lite_scene": {
                "scene_contract_sha256": manifest["city_lite_scene_contract_sha256"],
                "scene_contract_payload_sha256": manifest[
                    "city_lite_scene_payload_sha256"
                ],
            },
            "evaluator_manifest_sha256": sha256_file(private_manifest),
            "collection_binding": canary_binding,
            "cf2x_runtime_calibration": bind_native_t2_calibration(
                report,
                expected_usd_sha256=usd_sha256,
                expected_runtime_lock_sha256=lock_sha256,
                expected_control_dt_s=dt_s,
            ),
        }
    )
    _write_json(root / "task_outcome.json", {"status": "development_only"})
    _write_json(root / "capture_receipt.json", receipt)
    _rebind_native_fixture(root)
    return root, private_manifest, calibration_path, runtime_lock_path


def _write_capture_receipt(root: Path, *, task_kind: str, control_mode: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "capture_receipt.json"
    path.write_text(
        json.dumps(
            {
                "task_kind": task_kind,
                "command": {"control_mode": control_mode},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_native_t2_receipt_is_dispatched_before_legacy_t1_validation(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "capture"
    _write_capture_receipt(
        root,
        task_kind="native_t2_search_canary",
        control_mode="native_t2_canary",
    )
    observed: dict[str, object] = {}

    def fake_native_validator(capture_root: Path, **kwargs: object) -> NativeT2ValidationResult:
        observed["root"] = capture_root
        observed.update(kwargs)
        return NativeT2ValidationResult({"native_fixture": True}, ())

    monkeypatch.setattr(
        native_t2_validate, "validate_native_t2_capture", fake_native_validator
    )
    report = validate_isaac_capture(
        root,
        evaluator_manifest=tmp_path / "private.json",
        cf2x_runtime_calibration=tmp_path / "calibration.json",
        runtime_lock_path=tmp_path / "runtime-lock.json",
        require_clean_source=True,
    )

    assert report.valid
    assert report.checks["validation_profile"] == "native_t2_canary"
    assert report.checks["native_fixture"] is True
    assert observed["root"] == root.resolve()
    assert observed["require_clean_source"] is True
    assert observed["evaluator_manifest"] == tmp_path / "private.json"
    assert observed["cf2x_runtime_calibration"] == tmp_path / "calibration.json"
    assert observed["runtime_lock_path"] == tmp_path / "runtime-lock.json"


def test_native_control_mode_dispatches_even_if_task_kind_is_tampered(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "capture"
    _write_capture_receipt(root, task_kind="search3d", control_mode="native_t2_canary")
    monkeypatch.setattr(
        native_t2_validate,
        "validate_native_t2_capture",
        lambda *_args, **_kwargs: NativeT2ValidationResult({"tamper_path": True}, ()),
    )

    report = validate_isaac_capture(root, evaluator_manifest=tmp_path / "private.json")

    assert report.valid
    assert report.checks["validation_profile"] == "native_t2_canary"
    assert report.checks["tamper_path"] is True


def test_native_t2_validator_requires_external_private_truth_and_calibration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    _write_capture_receipt(
        root,
        task_kind="native_t2_search_canary",
        control_mode="native_t2_canary",
    )

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=None,
        cf2x_runtime_calibration=None,
        runtime_lock_path=None,
    )

    codes = {issue.code for issue in result.issues}
    assert "evaluator_manifest_required" in codes
    assert "native_t2_calibration_required" in codes
    assert result.checks["native_t2_independent_replay"] is False


def test_native_validation_receipt_binds_the_dispatch_and_replay_sources(
    tmp_path: Path,
) -> None:
    report = IsaacValidationReport(
        root=tmp_path,
        receipt_sha256="a" * 64,
        checks={"validation_profile": "native_t2_canary"},
        issues=(),
    )
    destination = tmp_path / "native-validation.json"

    write_validation_receipt(report, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["validator_id"] == "rivermark-independent-native-t2-canary-validator-v1"
    assert payload["validator_source_sha256"] == _native_t2_validator_sha256()
    assert payload["formal_benchmark_admission"] is False


def test_native_t2_validator_replays_a_complete_cpu_fixture(tmp_path: Path) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
        require_clean_source=True,
    )

    assert result.issues == ()
    assert result.checks["native_t2_independent_replay"] is True
    assert result.checks["private_event_evaluation_verified"] is True
    assert result.checks["private_event_evaluation"]["matched_count"] >= 1


def test_native_t2_validator_replays_held_decision_with_per_step_command_times(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(
        tmp_path, decision_stride_physics_steps=2
    )

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
        require_clean_source=True,
    )

    assert result.issues == ()
    records = [
        json.loads(line)
        for line in (root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    decisions = [record for record in records if record["record_type"] == "decision"]
    physical = [record for record in records if record["record_type"] == "physical_step"]
    assert len(decisions) == 2
    assert len(physical) == 4
    assert (
        physical[1]["evidence"]["decision_command_time_ns"]
        == decisions[0]["decision"]["observation"]["command_time_ns"]
    )
    assert (
        physical[1]["evidence"]["physical_command_time_ns"]
        > physical[0]["evidence"]["physical_command_time_ns"]
    )


def test_native_t2_v2_rejects_missing_or_nonrealized_motion_contract(tmp_path: Path) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    receipt_path = root / "capture_receipt.json"
    receipt = _read_json(receipt_path)
    binding = receipt["collection_binding"]
    assert isinstance(binding, dict)
    binding.update(
        {
            "protocol_id": "citylite-native-t2-canary-v2",
            "cell_id": "native-t2-canary-inner-dev-v2",
        }
    )
    receipt["collection_binding"] = binding
    _write_json(receipt_path, receipt)
    _rebind_native_fixture(root)

    missing = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )
    assert "native_t2_motion_contract" in {issue.code for issue in missing.issues}

    receipt = _read_json(receipt_path)
    command = receipt["command"]
    assert isinstance(command, dict)
    command["native_t2_canary"] = {
        "decision_stride_physics_steps": 1,
        "world_command_bounds": {
            "max_horizontal_speed_mps": 0.75,
            "max_vertical_speed_mps": 0.10,
            "max_yaw_rate_rad_s": 0.80,
        },
        "motion_contract": native_t2_v2_motion_contract(),
        "route_timing_feasibility": {},
    }
    receipt["command"] = command
    _write_json(receipt_path, receipt)
    task_path = root / "public_task.json"
    task = _read_json(task_path)
    task["task_variant_id"] = NATIVE_T2_V2_TASK_VARIANT_ID
    task["motion_contract"] = native_t2_v2_motion_contract()
    _write_json(task_path, task)
    scene_path = root / "scene.json"
    scene = _read_json(scene_path)
    scene["public_task_sha256"] = sha256_file(task_path)
    _write_json(scene_path, scene)
    _rebind_native_fixture(root)

    nonrealized = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )
    codes = {issue.code for issue in nonrealized.issues}
    assert "native_t2_motion_command" in codes
    assert "native_t2_route_timing_feasibility" in codes


def test_native_t2_v3_receipt_cannot_borrow_v2_motion_contract() -> None:
    receipt = {"command": {"native_t2_canary": {"motion_contract": native_t2_v2_motion_contract()}}}
    assert (
        native_t2_validate._native_t2_motion_contract_from_receipt(
            receipt, task_variant_id=NATIVE_T2_V3_TASK_VARIANT_ID
        )
        is None
    )
    receipt["command"]["native_t2_canary"]["motion_contract"] = native_t2_v3_motion_contract()
    assert native_t2_validate._native_t2_motion_contract_from_receipt(
        receipt, task_variant_id=NATIVE_T2_V3_TASK_VARIANT_ID
    ) == native_t2_v3_motion_contract()


def test_native_t2_validator_rejects_rebound_allocation_wrench_tampering(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    trace_path = root / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    physical = next(record for record in records if record["record_type"] == "physical_step")
    physical["evidence"]["applied_wrench_body"][0][2] += 0.01
    trace_path.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    event_path = root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
    event_payload = _read_json(event_path)
    event_payload["decision_trace_sha256"] = sha256_file(trace_path)
    _write_json(event_path, event_payload)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "native_t2_allocation_wrench" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_rebound_event_replay_tampering(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    event_path = root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
    payload = _read_json(event_path)
    journal = payload["candidate_event_journal"]
    assert isinstance(journal, dict)
    submission = journal["submission"]
    assert isinstance(submission, dict)
    events = submission["events"]
    assert isinstance(events, list) and events
    events[0]["position_w_m"][0] += 1.0
    _write_json(event_path, payload)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "native_t2_event_replay" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_rebound_zero_match_journal(tmp_path: Path) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    event_path = root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
    payload = _read_json(event_path)
    journal = payload["candidate_event_journal"]
    assert isinstance(journal, dict)
    submission = journal["submission"]
    assert isinstance(submission, dict)
    submission["events"] = []
    journal["submission_sha256"] = native_t2_validate._canonical_sha256(submission)
    _write_json(event_path, payload)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "native_t2_event_replay" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_event_debug_private_manifest_path(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    event_path = root / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
    payload = _read_json(event_path)
    # This field does not change the replayed submission, so a hash-only or
    # replay-only validator would accept it after the receipt is rebound.
    payload["debug"] = {"private_manifest_path": str(private_manifest)}
    _write_json(event_path, payload)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "public_private_leakage" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_rebound_public_private_truth_leakage(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    private_payload = _read_json(private_manifest)
    targets = private_payload["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], dict)
    public_task_path = root / "public_task.json"
    public_task = _read_json(public_task_path)
    public_task["leaked_truth"] = {
        "target_id": targets[0]["target_id"],
        "position_w_m": targets[0]["position_w_m"],
    }
    _write_json(public_task_path, public_task)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "public_private_leakage" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_rebound_semantic_private_id_leakage(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    private_payload = _read_json(private_manifest)
    targets = private_payload["targets"]
    assert isinstance(targets, list) and isinstance(targets[0], dict)
    metadata_path = root / "learning_labels/semantic_frame_metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    labels = rows[0]["onboard_replicator_info"]["per_camera"][0]["id_to_labels"]
    labels["99"] = {"class": targets[0]["target_id"]}
    metadata_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    codes = {issue.code for issue in result.issues}
    assert "semantic_private_id_leakage" in codes
    assert "public_private_leakage" in codes


def test_native_t2_validator_rejects_rebound_moving_overview_camera(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    overview_path = root / "sensors/overview_rgb.npz"
    with ChunkedFrameArchive(overview_path) as overview:
        timestamps = overview.timestamps_ns.copy()
        positions = overview.array("camera_pos_w_m").copy()
        quaternions = overview.array("camera_quat_wxyz").copy()
        targets = overview.array("target_w_m").copy()
        rgb = np.stack([overview.frame("rgb", index) for index in range(len(timestamps))])
        semantic = np.stack(
            [overview.frame("semantic_segmentation", index) for index in range(len(timestamps))]
        )
    positions[1, 0] += 0.5
    write_chunked_frame_archive(
        overview_path,
        timestamps_ns=timestamps,
        inline_fields={
            "camera_pos_w_m": positions,
            "camera_quat_wxyz": quaternions,
            "target_w_m": targets,
        },
        frame_fields={"rgb": rgb, "semantic_segmentation": semantic},
    )
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert "route_witness" in {issue.code for issue in result.issues}


def test_native_t2_validator_rejects_rebound_sensor_phase_timestamp_tampering(
    tmp_path: Path,
) -> None:
    root, private_manifest, calibration, runtime_lock = _native_t2_fixture(tmp_path)
    phase_path = root / "sensors/sensor_phase.npz"
    with np.load(phase_path, allow_pickle=False) as archive:
        phase = {name: archive[name].copy() for name in archive.files}
    phase["physics_time_ns"][0] += 1
    _write_npz(phase_path, phase)
    _rebind_native_fixture(root)

    result = validate_native_t2_capture(
        root,
        evaluator_manifest=private_manifest,
        cf2x_runtime_calibration=calibration,
        runtime_lock_path=runtime_lock,
    )

    assert any(issue.code.startswith("sensor_phase") for issue in result.issues)
