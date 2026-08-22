"""Attribute a non-formal external G2-I planning deadline miss without Isaac.

The source L1 trace is protected because it preserves evaluator evidence, but
the replayed packets are the public packets that were sent to the planner.
This tool emits only hashes, action-equivalence counts, and timing scalars. It
never writes observations, actions, target data, or local paths to its report.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from aerocity_bench.adapters import (
    ExternalProcessPlannerBridge,
    arbitrate_public_fleet_actions,
    load_external_l1_adapter_manifest,
)
from aerocity_bench.canonical import file_hash, read_json, write_json
from aerocity_bench.contracts import ObservationPacket, Pose3D
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.public_boundary import assert_public_fields, validate_public_task_spec

REPORT_SCHEMA = "org.aerocity.bench.external-g2-i-process-timing-diagnosis.v1"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--public-episode", type=Path, required=True)
    parser.add_argument("--recorded-private-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args(argv)


def _packet(node: dict[str, Any]) -> ObservationPacket:
    if node.get("received_messages"):
        raise ValueError("timing diagnosis refuses non-empty communication payloads")
    return ObservationPacket(
        episode_id=str(node["episode_id"]),
        observation_id=str(node["observation_id"]),
        drone_id=str(node["drone_id"]),
        sequence=int(node["sequence"]),
        timestamp_s=float(node["timestamp_s"]),
        pose=Pose3D.from_dict(node["pose"]),
        linear_velocity_world_mps=tuple(
            float(value) for value in node["linear_velocity_world_mps"]
        ),
        angular_speed_deg_s=float(node["angular_speed_deg_s"]),
        energy_remaining_j=float(node["energy_remaining_j"]),
        local_occupancy=tuple(
            tuple(int(value) for value in cell) for cell in node["local_occupancy"]
        ),
        local_occupancy_origin_world_m=tuple(
            float(value) for value in node["local_occupancy_origin_world_m"]
        ),
        local_occupancy_resolution_m=float(node["local_occupancy_resolution_m"]),
        local_occupancy_radius_m=float(node["local_occupancy_radius_m"]),
        teammate_states=tuple(node["teammate_states"]),
        health=str(node["health"]),  # type: ignore[arg-type]
        sensor_pitch_deg=float(node["sensor_pitch_deg"]),
    )


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("timing samples must be non-empty, finite, and non-negative")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "call_count": len(ordered),
        "p50_s": percentile(0.50),
        "p95_s": percentile(0.95),
        "p99_s": percentile(0.99),
        "max_s": ordered[-1],
    }


def _public_trace(
    recorded: dict[str, Any], terminal_sequence: int
) -> list[tuple[dict[str, ObservationPacket], dict[str, dict[str, Any]]]]:
    bindings = recorded.get("execution_bindings_public")
    if not isinstance(bindings, list):
        raise ValueError("recorded report lacks protected public execution bindings")
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in bindings:
        if not isinstance(item, dict):
            raise ValueError("recorded public execution binding is not an object")
        sequence = int(item.get("action_sequence", -1))
        if 0 <= sequence <= terminal_sequence:
            groups.setdefault(sequence, []).append(item)
    if set(groups) != set(range(terminal_sequence + 1)):
        raise ValueError(
            "recorded trace has a missing public decision before the requested sequence"
        )
    if not groups:
        raise ValueError("requested action sequence does not exist in the protected trace")
    trace: list[tuple[dict[str, ObservationPacket], dict[str, dict[str, Any]]]] = []
    expected_roster: set[str] | None = None
    for sequence in range(terminal_sequence + 1):
        selected = groups[sequence]
        observations = {
            str(item["drone_id"]): _packet(item["source_observation"])
            for item in selected
            if isinstance(item.get("source_observation"), dict)
        }
        actions = {
            str(item["drone_id"]): item["action"]
            for item in selected
            if isinstance(item.get("action"), dict)
        }
        if (
            not observations
            or set(observations) != set(actions)
            or len(observations) != len(selected)
        ):
            raise ValueError("recorded sequence has an incomplete public fleet roster")
        if any(packet.sequence != sequence for packet in observations.values()):
            raise ValueError("recorded observation sequence differs from its trace position")
        if expected_roster is None:
            expected_roster = set(observations)
        elif set(observations) != expected_roster:
            raise ValueError("recorded public fleet roster changes inside one replay")
        trace.append((observations, actions))
    return trace


def diagnose(
    *,
    adapter_manifest_path: Path,
    release_config_path: Path,
    task_spec_path: Path,
    public_episode_path: Path,
    recorded_private_report_path: Path,
    sequence: int,
    repetitions: int,
) -> dict[str, Any]:
    if sequence < 0 or repetitions <= 0:
        raise ValueError("sequence must be non-negative and repetitions must be positive")
    manifest = load_external_l1_adapter_manifest(adapter_manifest_path)
    config = load_ordinary_config(release_config_path)
    task_spec = read_json(task_spec_path)
    public_episode = read_json(public_episode_path)
    recorded = read_json(recorded_private_report_path)
    validate_public_task_spec(task_spec)
    assert_public_fields(public_episode, path="public_episode")
    if task_spec.get("task_track") != "G2-I" or manifest.declaration.capability_profile != "G2-I":
        raise ValueError("external timing diagnosis requires a G2-I task and adapter")
    if recorded.get("formal_score_eligible") is not False:
        raise ValueError("timing diagnosis refuses formal-score evidence")
    if recorded.get("execution_mode") != "external-process-policy":
        raise ValueError("timing diagnosis requires an external-process L1 trace")
    bindings = recorded.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("recorded report lacks immutable input bindings")
    if bindings.get("task_spec_sha256") != file_hash(task_spec_path):
        raise ValueError("recorded report is not bound to the supplied public task")
    if bindings.get("public_episode_sha256") != file_hash(public_episode_path):
        raise ValueError("recorded report is not bound to the supplied public episode")
    if bindings.get("release_config_sha256") != file_hash(release_config_path):
        raise ValueError("recorded report is not bound to the supplied release configuration")
    adapter = recorded.get("external_adapter")
    if not isinstance(adapter, dict) or adapter.get("adapter_manifest_sha256") != file_hash(
        adapter_manifest_path
    ):
        raise ValueError("recorded report is not bound to the supplied external adapter")
    if recorded.get("method") != manifest.declaration.method_id:
        raise ValueError("recorded report method differs from the external adapter")
    timing = recorded.get("planning_timing")
    if not isinstance(timing, dict) or not isinstance(
        timing.get("planning_deadline_s"), (int, float)
    ):
        raise ValueError("recorded report lacks a finite planning deadline")
    deadline_s = float(timing["planning_deadline_s"])
    if not math.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError("recorded planning deadline is invalid")
    trace = _public_trace(recorded, sequence)
    vehicle_radius_m = float(config.raw["execution_contract"]["vehicle"]["radius_m"])
    if not math.isfinite(vehicle_radius_m) or vehicle_radius_m <= 0.0:
        raise ValueError("release configuration vehicle radius is invalid")

    bridge = ExternalProcessPlannerBridge(
        manifest.declaration,
        manifest.launch_command(),
        cwd=Path(__file__).resolve().parents[1],
        # This diagnostic measures an already-recorded event. It never grants
        # a deadline exemption to the benchmark run it diagnoses.
        response_timeout_s=max(5.0, deadline_s),
        initialization_timeout_s=10.0,
        maximum_line_bytes=2_000_000,
    )
    try:
        bridge_wall_samples: list[float] = []
        bridge_cpu_samples: list[float] = []
        arbitration_wall_samples: list[float] = []
        arbitration_cpu_samples: list[float] = []
        total_wall_samples: list[float] = []
        total_cpu_samples: list[float] = []
        for repetition in range(repetitions):
            bridge.reset(public_episode, public_task_spec=task_spec)
            for prefix_sequence, (observations, expected_actions) in enumerate(trace[:-1]):
                actions, _ = bridge.act(observations)
                actions = arbitrate_public_fleet_actions(
                    actions, observations, vehicle_radius_m=vehicle_radius_m
                )
                actual_actions = {
                    drone_id: action.to_dict() for drone_id, action in actions.items()
                }
                if actual_actions != expected_actions:
                    raise ValueError(
                        "stateful public-prefix replay diverged at sequence "
                        f"{prefix_sequence} during repetition {repetition}"
                    )
            observations, expected_actions = trace[-1]
            total_wall_started = time.perf_counter()
            total_cpu_started = time.process_time()
            bridge_wall_started = time.perf_counter()
            bridge_cpu_started = time.process_time()
            actions, _ = bridge.act(observations)
            bridge_wall = time.perf_counter() - bridge_wall_started
            bridge_cpu = time.process_time() - bridge_cpu_started
            arbitration_wall_started = time.perf_counter()
            arbitration_cpu_started = time.process_time()
            actions = arbitrate_public_fleet_actions(
                actions, observations, vehicle_radius_m=vehicle_radius_m
            )
            arbitration_wall = time.perf_counter() - arbitration_wall_started
            arbitration_cpu = time.process_time() - arbitration_cpu_started
            total_wall = time.perf_counter() - total_wall_started
            total_cpu = time.process_time() - total_cpu_started
            actual_actions = {drone_id: action.to_dict() for drone_id, action in actions.items()}
            if actual_actions != expected_actions:
                raise ValueError(
                    "stateful terminal replay diverged at requested sequence "
                    f"{sequence} during repetition {repetition}"
                )
            bridge_wall_samples.append(bridge_wall)
            bridge_cpu_samples.append(bridge_cpu)
            arbitration_wall_samples.append(arbitration_wall)
            arbitration_cpu_samples.append(arbitration_cpu)
            total_wall_samples.append(total_wall)
            total_cpu_samples.append(total_cpu)
    finally:
        bridge.close()
    return {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "diagnostic_only": True,
        "source": {
            "recorded_private_report_sha256": file_hash(recorded_private_report_path),
            "release_config_sha256": file_hash(release_config_path),
            "task_spec_sha256": file_hash(task_spec_path),
            "public_episode_sha256": file_hash(public_episode_path),
            "adapter_manifest_sha256": file_hash(adapter_manifest_path),
        },
        "replay": {
            "sequence": sequence,
            "stateful_prefix_replayed": True,
            "prefix_decision_count": sequence,
            "repetitions": repetitions,
            "fleet_size": len(trace[-1][0]),
            "action_mismatch_count": 0,
        },
        "planning_deadline_s": deadline_s,
        "timing": {
            "bridge_act_wall_clock": _summary(bridge_wall_samples),
            "bridge_act_process_cpu": _summary(bridge_cpu_samples),
            "fleet_arbitration_wall_clock": _summary(arbitration_wall_samples),
            "fleet_arbitration_process_cpu": _summary(arbitration_cpu_samples),
            "total_planner_wall_clock": _summary(total_wall_samples),
            "total_planner_process_cpu": _summary(total_cpu_samples),
            "total_planner_deadline_miss_count": sum(
                value > deadline_s for value in total_wall_samples
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    report = diagnose(
        adapter_manifest_path=args.adapter_manifest.resolve(),
        release_config_path=args.release_config.resolve(),
        task_spec_path=args.task_spec.resolve(),
        public_episode_path=args.public_episode.resolve(),
        recorded_private_report_path=args.recorded_private_report.resolve(),
        sequence=args.sequence,
        repetitions=args.repetitions,
    )
    write_json(args.output, report)
    print(
        "external timing diagnosis written: "
        f"sequence={report['replay']['sequence']} calls={report['replay']['repetitions']} "
        f"mismatches={report['replay']['action_mismatch_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
