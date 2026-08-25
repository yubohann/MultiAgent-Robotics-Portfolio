"""Replay recorded public observations to attribute a calibration policy overrun."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import create_baseline
from aerocity_bench.canonical import file_hash, read_json, write_json
from aerocity_bench.contracts import ObservationPacket, Pose3D
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.public_boundary import assert_public_fields, validate_public_task_spec

REPORT_SCHEMA = "org.aerocity.bench.g2-i-policy-deadline-diagnosis.v1"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--public-episode", type=Path, required=True)
    parser.add_argument("--recorded-private-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _observation_from_dict(node: dict[str, Any]) -> ObservationPacket:
    if node.get("received_messages"):
        raise ValueError("deadline diagnosis does not accept non-empty message payloads")
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
        sensor_pitch_deg=(
            None if node.get("sensor_pitch_deg") is None else float(node["sensor_pitch_deg"])
        ),
    )


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def diagnose(
    *,
    release_config_path: Path,
    task_spec_path: Path,
    public_episode_path: Path,
    recorded_private_report_path: Path,
) -> dict[str, Any]:
    config = load_ordinary_config(release_config_path)
    task_spec = read_json(task_spec_path)
    public_episode = read_json(public_episode_path)
    recorded = read_json(recorded_private_report_path)
    validate_public_task_spec(task_spec)
    assert_public_fields(public_episode, path="public_episode")
    if bool(recorded.get("formal_score_eligible")):
        raise ValueError("deadline diagnosis refuses formal-score evidence")
    if str(task_spec.get("split", "calibration")) in FORMAL_SPLITS:
        raise ValueError("deadline diagnosis refuses formal splits")
    if recorded.get("execution_purpose") != "complete-calibration-episode":
        raise ValueError("deadline diagnosis requires a complete calibration candidate")
    method_id = str(recorded.get("method", ""))
    if method_id != "atlas-region-greedy":
        raise ValueError("this diagnosis is frozen to the failed atlas-region-greedy case")
    if recorded.get("input_bindings", {}).get("task_spec_sha256") != file_hash(task_spec_path):
        raise ValueError("recorded report is not bound to the supplied task specification")
    if recorded.get("input_bindings", {}).get("public_episode_sha256") != file_hash(
        public_episode_path
    ):
        raise ValueError("recorded report is not bound to the supplied public episode")
    if recorded.get("input_bindings", {}).get("release_config_sha256") != file_hash(
        release_config_path
    ):
        raise ValueError("recorded report is not bound to the supplied release configuration")

    policy = create_baseline(method_id, config, task_spec, public_episode)
    refinement_events: list[dict[str, Any]] = []
    original_refine = policy._refine_scan_pose  # noqa: SLF001 - diagnostic attribution

    def instrumented_refine(observation: ObservationPacket, target: Pose3D) -> Pose3D:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        result = original_refine(observation, target)
        refinement_events.append(
            {
                "sequence": observation.sequence,
                "task_time_s": observation.timestamp_s,
                "drone_id": observation.drone_id,
                "local_occupancy_cell_count": len(observation.local_occupancy),
                "wall_clock_s": time.perf_counter() - wall_started,
                "process_cpu_s": time.process_time() - cpu_started,
                "input_pose": target.to_dict(),
                "output_pose": result.to_dict(),
            }
        )
        return result

    policy._refine_scan_pose = instrumented_refine  # type: ignore[method-assign]  # noqa: SLF001

    bindings = recorded.get("execution_bindings_public")
    receipts = recorded.get("execution_receipts")
    if not isinstance(bindings, list) or not isinstance(receipts, list):
        raise ValueError("recorded report lacks execution trace arrays")
    if len(bindings) != len(receipts):
        raise ValueError("recorded action bindings and receipts differ in length")

    grouped: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for binding, receipt in zip(bindings, receipts, strict=True):
        if not isinstance(binding, dict) or not isinstance(receipt, dict):
            raise ValueError("recorded execution trace contains a non-object")
        sequence = int(binding["action_sequence"])
        grouped.setdefault(sequence, []).append((binding, receipt))

    invocation_events: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for sequence in sorted(grouped):
        records = grouped[sequence]
        observations = {
            str(binding["drone_id"]): _observation_from_dict(binding["source_observation"])
            for binding, _ in records
        }
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        actions = policy(observations)
        cpu_elapsed = time.process_time() - cpu_started
        wall_elapsed = time.perf_counter() - wall_started
        recorded_latencies = {float(receipt["planning_latency_s"]) for _, receipt in records}
        if len(recorded_latencies) != 1:
            raise ValueError("fleet receipts disagree on centralized policy latency")
        for binding, _ in records:
            drone_id = str(binding["drone_id"])
            actual = actions[drone_id].to_dict()
            if actual != binding["action"]:
                mismatches.append(
                    {
                        "sequence": sequence,
                        "drone_id": drone_id,
                        "expected": binding["action"],
                        "actual": actual,
                    }
                )
        invocation_events.append(
            {
                "sequence": sequence,
                "task_time_s": min(item.timestamp_s for item in observations.values()),
                "active_drone_count": len(observations),
                "total_local_occupancy_cell_count": sum(
                    len(item.local_occupancy) for item in observations.values()
                ),
                "recorded_wall_clock_s": recorded_latencies.pop(),
                "replay_wall_clock_s": wall_elapsed,
                "replay_process_cpu_s": cpu_elapsed,
            }
        )

    replay_samples = [float(item["replay_wall_clock_s"]) for item in invocation_events]
    deadline_s = float(config.raw["execution_contract"]["planning_deadline_s"])
    return {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "diagnostic_only": True,
        "method_id": method_id,
        "source_private_report_sha256": file_hash(recorded_private_report_path),
        "source_baseline_sha256": file_hash(
            Path(__file__).resolve().parents[1] / "src" / "aerocity_bench" / "baselines.py"
        ),
        "planner_deadline_s": deadline_s,
        "action_replay": {
            "invocation_count": len(invocation_events),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:10],
        },
        "replay_timing": {
            "p50_s": _percentile(replay_samples, 0.50),
            "p95_s": _percentile(replay_samples, 0.95),
            "p99_s": _percentile(replay_samples, 0.99),
            "max_s": max(replay_samples),
            "deadline_miss_count": sum(value > deadline_s for value in replay_samples),
        },
        "top_invocations": sorted(
            invocation_events, key=lambda item: float(item["replay_wall_clock_s"]), reverse=True
        )[:20],
        "refinement_events": refinement_events,
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    report = diagnose(
        release_config_path=args.release_config.resolve(),
        task_spec_path=args.task_spec.resolve(),
        public_episode_path=args.public_episode.resolve(),
        recorded_private_report_path=args.recorded_private_report.resolve(),
    )
    write_json(args.output, report)
    print(
        f"deadline diagnosis written: actions={report['action_replay']['invocation_count']} "
        f"mismatches={report['action_replay']['mismatch_count']} "
        f"refinements={len(report['refinement_events'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
