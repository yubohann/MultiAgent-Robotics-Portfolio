"""Fail-closed aggregation of real G2-I L1 evidence into claim records."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json
from .cf2x_fleet_preflight_contract import (
    EXECUTION_FAILURE_CATEGORIES,
    validate_fleet_preflight_reports,
)
from .contracts import ObservationPacket, Pose3D
from .measurement_evidence import (
    EVIDENCE_SCHEMA,
    L1MeasurementEvidence,
)

MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-l1-measurement-evidence-manifest.v1"
AGGREGATION_SCHEMA = "org.aerocity.bench.g2-i-l1-measurement-records.v1"
_ALLOWED_FAILURES = EXECUTION_FAILURE_CATEGORIES


def _finite_vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result  # type: ignore[return-value]


def _observation(node: dict[str, Any]) -> ObservationPacket:
    pose_node = node.get("pose")
    if not isinstance(pose_node, dict):
        raise ValueError("source observation lacks its pose")
    return ObservationPacket(
        episode_id=str(node["episode_id"]),
        observation_id=str(node["observation_id"]),
        drone_id=str(node["drone_id"]),
        sequence=int(node["sequence"]),
        timestamp_s=float(node["timestamp_s"]),
        pose=Pose3D.from_dict(pose_node),
        linear_velocity_world_mps=_finite_vector(
            node["linear_velocity_world_mps"], "observation linear velocity"
        ),
        angular_speed_deg_s=float(node["angular_speed_deg_s"]),
        energy_remaining_j=float(node["energy_remaining_j"]),
        local_occupancy=tuple(
            tuple(int(item) for item in cell) for cell in node["local_occupancy"]
        ),
        local_occupancy_origin_world_m=_finite_vector(
            node["local_occupancy_origin_world_m"], "observation occupancy origin"
        ),
        local_occupancy_resolution_m=float(node["local_occupancy_resolution_m"]),
        local_occupancy_radius_m=float(node["local_occupancy_radius_m"]),
        teammate_states=tuple(node.get("teammate_states", [])),
        health=str(node.get("health", "nominal")),
        sensor_pitch_deg=float(node["sensor_pitch_deg"]),
    )


def _auc(trace: list[list[Any]], index: int, duration_s: float, denominator: float) -> float:
    if duration_s <= 0.0 or denominator <= 0.0 or not trace:
        return 0.0
    area = 0.0
    previous_time = 0.0
    previous_value = 0.0
    for item in trace:
        timestamp = max(previous_time, min(duration_s, float(item[0])))
        value = max(previous_value, min(1.0, float(item[index]) / denominator))
        area += (timestamp - previous_time) * previous_value
        previous_time, previous_value = timestamp, value
    return (area + max(0.0, duration_s - previous_time) * previous_value) / duration_s


def _terminal_status(private: dict[str, Any]) -> str:
    failures = private.get("failure_records")
    if not isinstance(failures, list):
        raise ValueError("L1 private evidence lacks failure records")
    categories: list[str] = []
    for failure in failures:
        if not isinstance(failure, dict) or failure.get("category") not in _ALLOWED_FAILURES:
            raise ValueError("L1 evidence contains an unknown failure category")
        categories.append(str(failure["category"]))
    final = private.get("final")
    if not isinstance(final, dict):
        raise ValueError("L1 private evidence lacks final safety state")
    if bool(final.get("collision_detected")) or "collision" in categories:
        return "collision"
    if bool(final.get("out_of_bounds_detected")) or "out_of_bounds_failure" in categories:
        return "out_of_bounds_failure"
    timing = private.get("planning_timing")
    if not isinstance(timing, dict):
        raise ValueError("L1 private evidence lacks planning timing")
    if int(timing.get("deadline_miss_tick_count", 0)) > 0 or {
        "deadline_failure",
        "planner_timeout",
    }.intersection(categories):
        return "planner_timeout"
    if "external_adapter_failure" in categories or "planner_crash" in categories:
        return "planner_crash"
    if not bool(final.get("all_returned_home")) or "return_failure" in categories:
        return "return_failure"
    if not bool(final.get("safe_completion")):
        if "controller_failure" in categories:
            return "controller_failure"
        if "reset_failure" in categories:
            return "reset_failure"
        if "method_failure" in categories:
            return "controller_failure"
        if "energy_exhausted" in categories:
            return "controller_failure"
        raise ValueError("L1 evidence is unsafe but has no recognized terminal cause")
    return "completed"


def _validate_manifest(manifest: object) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported L1 evidence manifest schema")
    expected = {
        "schema",
        "formal_score_eligible",
        "purpose",
        "protocol_hash",
        "panel_manifest_hash",
        "precommitted_before_execution",
        "episodes",
        "manifest_hash",
    }
    if set(manifest) != expected:
        raise ValueError("L1 evidence manifest fields differ")
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if content_hash(payload) != manifest["manifest_hash"]:
        raise ValueError("L1 evidence manifest hash mismatch")
    if manifest["formal_score_eligible"] is not False:
        raise ValueError("L1 evidence manifest cannot claim formal eligibility")
    if manifest["purpose"] != "precommitted_calibration_l1_evidence":
        raise ValueError("L1 evidence manifest purpose differs")
    if manifest["precommitted_before_execution"] is not True:
        raise ValueError("L1 evidence manifest was not precommitted")
    episodes = manifest["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("L1 evidence manifest has no episodes")
    keys: set[tuple[str, str, str]] = set()
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("L1 evidence manifest episode must be an object")
        required = {
            "layout_ancestor",
            "method_id",
            "episode_id",
            "episode_name",
            "layout_root",
            "release_config",
            "public_report",
            "private_report",
        }
        if set(item) != required:
            raise ValueError("L1 evidence manifest episode fields differ")
        key = (str(item["layout_ancestor"]), str(item["method_id"]), str(item["episode_id"]))
        if any(not value for value in key) or key in keys:
            raise ValueError("L1 evidence manifest repeats an episode")
        keys.add(key)
        for field in (
            "episode_name",
            "layout_root",
            "release_config",
            "public_report",
            "private_report",
        ):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(f"L1 evidence manifest path {field} is invalid")
    return manifest


def _recompute_evidence(
    *,
    city: dict[str, Any],
    task: dict[str, Any],
    episode: dict[str, Any],
    public: dict[str, Any],
    private: dict[str, Any],
) -> dict[str, Any]:
    evidence_node = private.get("measurement_evidence")
    state_trace = private.get("measured_state_trace_private")
    if not isinstance(evidence_node, dict) or evidence_node.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("L1 private evidence lacks measurement evidence")
    if not isinstance(state_trace, list):
        raise ValueError("L1 private evidence lacks measured state trace")
    if content_hash(state_trace) != evidence_node.get("measured_state_trace_hash"):
        raise ValueError("L1 measured state trace hash mismatch")
    input_bindings = private.get("input_bindings")
    if not isinstance(input_bindings, dict):
        raise ValueError("L1 private evidence lacks input bindings")
    if evidence_node.get("input_bindings_hash") != content_hash(input_bindings):
        raise ValueError("L1 measurement evidence input binding hash mismatch")
    if public.get("input_bindings") != input_bindings:
        raise ValueError("L1 public/private input bindings differ")
    if input_bindings.get("layout_hash") != city.get("layout_hash"):
        raise ValueError("L1 evidence layout hash differs from CitySpec")
    if input_bindings.get("task_spec_hash") != task.get("task_spec_hash"):
        raise ValueError("L1 evidence task hash differs from public task")
    if input_bindings.get("episode_id") != episode.get("episode_id"):
        raise ValueError("L1 evidence episode binding differs")
    if input_bindings.get("task_track") != "G2-I":
        raise ValueError("L1 evidence task track differs")
    if private.get("execution_purpose") != "complete-calibration-episode":
        raise ValueError("measurement aggregation accepts complete calibration episodes only")
    if private.get("execution_mode") not in {"public-policy", "external-process-policy"}:
        raise ValueError("measurement aggregation rejects private or fixture methods")
    execution = private.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("L1 private evidence lacks execution summary")
    duration = float(task["execution_contract"]["episode"]["duration_s"])
    if not math.isclose(float(execution.get("simulated_time_s", -1.0)), duration, abs_tol=1.0e-6):
        raise ValueError("L1 evidence did not run the frozen episode duration")

    bindings = private.get("execution_bindings_public")
    receipts = private.get("execution_receipts")
    observations = private.get("observation_receipts")
    if (
        not isinstance(bindings, list)
        or not isinstance(receipts, list)
        or not isinstance(observations, list)
    ):
        raise ValueError("L1 evidence lacks receipt-bound public packets")
    receipt_by_pair = {
        (str(item["drone_id"]), int(item["action_sequence"])): item for item in receipts
    }
    observation_receipt_by_id = {str(item["observation_id"]): item for item in observations}
    if len(receipt_by_pair) != len(receipts) or len(observation_receipt_by_id) != len(observations):
        raise ValueError("L1 evidence repeats a receipt identity")
    by_sequence: dict[int, list[dict[str, Any]]] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("L1 execution binding is malformed")
        action = binding.get("action")
        observation = binding.get("source_observation")
        if not isinstance(action, dict) or not isinstance(observation, dict):
            raise ValueError("L1 execution binding lacks public action or observation")
        sequence = int(binding["action_sequence"])
        pair = (str(binding["drone_id"]), sequence)
        receipt = receipt_by_pair.get(pair)
        if receipt is None:
            raise ValueError("L1 execution binding lacks its receipt")
        by_sequence.setdefault(sequence, []).append(binding)
    if sorted(by_sequence) != list(range(len(by_sequence))):
        raise ValueError("L1 evidence action sequence has gaps")
    if len(state_trace) != len(by_sequence):
        raise ValueError("L1 measured state trace length differs from control ticks")

    replay = L1MeasurementEvidence(city=city, task_spec=task, public_episode=episode)
    expected_ids = {str(item["drone_id"]) for item in episode["starts"]}
    for sequence in sorted(by_sequence):
        tick = by_sequence[sequence]
        if {str(item["drone_id"]) for item in tick} != expected_ids:
            raise ValueError("L1 evidence tick does not contain exactly the public fleet")
        for binding in sorted(tick, key=lambda item: str(item["drone_id"])):
            drone_id = str(binding["drone_id"])
            action = binding["action"]
            observation_node = binding["source_observation"]
            observation = _observation(observation_node)
            receipt = receipt_by_pair[(drone_id, sequence)]
            if action.get("kind") == "OBSERVE":
                observation_receipt = observation_receipt_by_id.get(
                    str(observation_node["observation_id"])
                )
                if observation_receipt is None:
                    raise ValueError("L1 OBSERVE action lacks its evaluator receipt")
                replay.record_observe(
                    observation,
                    evaluator_accepted=bool(observation_receipt.get("accepted")),
                    runtime_safe=not any(
                        bool(receipt.get(field))
                        for field in (
                            "collision",
                            "out_of_bounds",
                            "safety_intervention",
                            "deadline_miss",
                        )
                    ),
                )
            else:
                replay.end_observe(drone_id)
        state = state_trace[sequence]
        if int(state.get("action_sequence", -1)) != sequence:
            raise ValueError("L1 measured state trace sequence differs")
        positions = state.get("positions_w_m")
        safe_ids = state.get("safe_drone_ids")
        if not isinstance(positions, dict) or not isinstance(safe_ids, list):
            raise ValueError("L1 measured state trace is malformed")
        if set(positions) != expected_ids:
            raise ValueError("L1 measured state trace does not preserve the fleet roster")
        expected_safe_ids = {
            drone_id
            for drone_id in expected_ids
            if not any(
                bool(receipt_by_pair[(drone_id, sequence)].get(field))
                for field in ("collision", "out_of_bounds", "safety_intervention", "deadline_miss")
            )
        }
        if {str(value) for value in safe_ids} != expected_safe_ids:
            raise ValueError("L1 measured state trace safety mask differs from receipts")
        receipt_times = {
            float(receipt_by_pair[(drone_id, sequence)]["task_time_end_s"])
            for drone_id in expected_ids
        }
        if len(receipt_times) != 1 or not math.isclose(
            float(state["task_time_s"]), receipt_times.pop(), abs_tol=1.0e-6
        ):
            raise ValueError("L1 measured state trace time differs from receipts")
        replay.record_measured_positions(
            float(state["task_time_s"]),
            {
                str(drone_id): _finite_vector(position, "measured state position")
                for drone_id, position in positions.items()
            },
            safe_drone_ids={str(value) for value in safe_ids},
        )
    snapshot = replay.snapshot(
        measured_state_trace=state_trace, input_bindings_hash=content_hash(input_bindings)
    )
    for field in (
        "coverage_trace",
        "inspection_coverage_trace",
        "inspection_cell_count_trace",
        "coverage_denominators",
        "coverage_semantics",
        "inspection_footprint_semantics",
    ):
        if snapshot.get(field) != evidence_node.get(field):
            raise ValueError(f"L1 measurement evidence {field} does not replay")
    return snapshot


def aggregate_measurement_records(
    manifest_path: Path,
    *,
    protocol_hash: str,
    panel_manifest_hash: str,
) -> dict[str, Any]:
    """Read only hash-bound L1 evidence and return machine-built claim records."""

    manifest = _validate_manifest(read_json(manifest_path))
    if manifest["protocol_hash"] != protocol_hash:
        raise ValueError("L1 evidence manifest is bound to another measurement protocol")
    if manifest["panel_manifest_hash"] != panel_manifest_hash:
        raise ValueError("L1 evidence manifest is bound to another method panel")
    sources: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    root = manifest_path.resolve().parent
    for item in manifest["episodes"]:
        layout_root = (root / item["layout_root"]).resolve()
        release_config = (root / item["release_config"]).resolve()
        public_path = (root / item["public_report"]).resolve()
        private_path = (root / item["private_report"]).resolve()
        for path in (layout_root, release_config, public_path, private_path):
            if not path.exists():
                raise FileNotFoundError(f"L1 evidence source is missing: {path}")
        # Calibration evidence is a denominator, not a success-only filter.
        # The aggregation-only path still verifies every immutable binding,
        # receipt chain, measured trace, and public/private boundary while
        # retaining complete replays that ended in a planner or safety
        # failure.  Formal runners keep the strict default validator.
        validation = validate_fleet_preflight_reports(
            public_path,
            private_path,
            allow_execution_failure=True,
        )
        public = read_json(public_path)
        private = read_json(private_path)
        city = read_json(layout_root / "scene_authority" / "cityspec.json")
        task = read_json(layout_root / "method_public" / "task_spec.json")
        episode = read_json(
            layout_root / "method_public" / "episodes" / item["episode_name"]
        )
        input_bindings = private["input_bindings"]
        expected_files = {
            "stage_sha256": layout_root / "scene_authority" / "stage.usda",
            "cityspec_sha256": layout_root / "scene_authority" / "cityspec.json",
            "task_spec_sha256": layout_root / "method_public" / "task_spec.json",
            "public_episode_sha256": layout_root
            / "method_public"
            / "episodes"
            / item["episode_name"],
        }
        for field, path in expected_files.items():
            if file_hash(path) != input_bindings.get(field):
                raise ValueError(f"L1 evidence {field} does not match its source file")
        if file_hash(release_config) != input_bindings.get("release_config_sha256"):
            raise ValueError("L1 release configuration hash differs from evidence binding")
        if str(private.get("method")) != str(item["method_id"]):
            raise ValueError("L1 evidence method differs from the precommitted manifest")
        if str(private.get("input_bindings", {}).get("episode_id")) != str(item["episode_id"]):
            raise ValueError("L1 evidence episode differs from the precommitted manifest")
        if (
            "oracle" in str(item["method_id"]).lower()
            or "witness" in str(item["method_id"]).lower()
        ):
            raise ValueError("private-truth methods cannot enter the measurement panel")
        replayed = _recompute_evidence(
            city=city,
            task=task,
            episode=episode,
            public=public,
            private=private,
        )
        duration = float(task["execution_contract"]["episode"]["duration_s"])
        audit = private["evaluator_private_audit"]
        target_count = int(audit["target_count"])
        if target_count <= 0:
            raise ValueError("L1 evaluator target denominator must be positive")
        terminal = _terminal_status(private)
        denominators = replayed["coverage_denominators"]
        record = {
            "layout_ancestor": str(item["layout_ancestor"]),
            "method_id": str(item["method_id"]),
            "method_uses_private_truth": False,
            "episode_count": 1,
            "source_run_report_hashes": [str(private["private_report_content_sha256"])],
            "failure_included": True,
            "terminal_status_counts": {terminal: 1},
            "mean_final_confirmed_recall": float(audit["confirmed_count"]) / target_count,
            "free_space_coverage_auc": _auc(
                replayed["coverage_trace"],
                2,
                duration,
                float(denominators["coverage_3d_free_cells"]),
            ),
            "inspection_footprint_auc": _auc(
                replayed["inspection_coverage_trace"],
                1,
                duration,
                float(denominators["inspection_atlas_area_m2"]),
            ),
        }
        grouped.setdefault((record["layout_ancestor"], record["method_id"]), []).append(record)
        sources.append(
            {
                "layout_ancestor": record["layout_ancestor"],
                "method_id": record["method_id"],
                "episode_id": str(item["episode_id"]),
                "public_report_file_sha256": validation["public_report_file_sha256"],
                "private_report_file_sha256": validation["private_report_file_sha256"],
                "private_report_content_sha256": private["private_report_content_sha256"],
                "replayed_measurement_evidence_hash": content_hash(replayed),
                "structural_validation_status": validation["status"],
                "terminal_status": terminal,
            }
        )
    records: list[dict[str, Any]] = []
    for (ancestor, method), rows in sorted(grouped.items()):
        if not rows:
            raise ValueError("L1 evidence method/ancestor group is empty")
        records.append(
            {
                "layout_ancestor": ancestor,
                "method_id": method,
                "method_uses_private_truth": False,
                "episode_count": len(rows),
                "source_run_report_hashes": sorted(
                    hash_value for row in rows for hash_value in row["source_run_report_hashes"]
                ),
                "failure_included": True,
                "terminal_status_counts": {
                    status: sum(row["terminal_status_counts"].get(status, 0) for row in rows)
                    for status in sorted(
                        {status for row in rows for status in row["terminal_status_counts"]}
                    )
                },
                "mean_final_confirmed_recall": statistics.fmean(
                    row["mean_final_confirmed_recall"] for row in rows
                ),
                "free_space_coverage_auc": statistics.fmean(
                    row["free_space_coverage_auc"] for row in rows
                ),
                "inspection_footprint_auc": statistics.fmean(
                    row["inspection_footprint_auc"] for row in rows
                ),
            }
        )
    return {
        "schema": AGGREGATION_SCHEMA,
        "formal_score_eligible": False,
        "protocol_hash": protocol_hash,
        "panel_manifest_hash": panel_manifest_hash,
        "source_manifest_hash": manifest["manifest_hash"],
        "records": records,
        "sources": sources,
        "record_set_hash": content_hash(records),
        "source_set_hash": content_hash(sources),
    }
