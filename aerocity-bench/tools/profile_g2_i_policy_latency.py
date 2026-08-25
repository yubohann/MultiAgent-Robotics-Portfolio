"""Profile public G2-I policy latency on one frozen development episode."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import BASELINES, create_baseline
from aerocity_bench.behavioral_distinctness import summarize_public_action_trace
from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.host_guard import foreign_isaac_processes, host_snapshot
from aerocity_bench.inspection_atlas import validate_public_mission_sector
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import public_episode_projection

MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-public-policy-latency-profile.v1"
MINIMUM_CONTROLLED_REPEATS = 3
P95_HEADROOM_FRACTION = 0.75


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--method", default="atlas-region-greedy", choices=sorted(BASELINES))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.record_index < 0:
        parser.error("--record-index must be non-negative")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


def _local_path(root: Path, value: object, field: str) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative path inside the manifest root")
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{field} escapes the manifest root")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {candidate}")
    return resolved


def summarize_latencies(
    latencies_s: list[float], *, deadline_s: float
) -> dict[str, float | int | list[float]]:
    """Summarize already-recorded timing without exposing evaluator-private data."""

    if not latencies_s:
        raise ValueError("latency profile requires at least one execution receipt")
    if not math.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError("planner deadline must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in latencies_s):
        raise ValueError("recorded planning latency must be finite and non-negative")
    ordered = sorted(float(value) for value in latencies_s)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "receipt_count": len(ordered),
        "deadline_miss_count": sum(value > deadline_s for value in ordered),
        "mean_planning_latency_s": round(sum(ordered) / len(ordered), 8),
        "p95_planning_latency_s": round(ordered[p95_index], 8),
        "max_planning_latency_s": round(ordered[-1], 8),
        "top_ten_planning_latency_s": [round(value, 8) for value in reversed(ordered[-10:])],
    }


def summarize_invocations(samples: list[dict[str, Any]], *, deadline_s: float) -> dict[str, Any]:
    """Summarize centralized calls without counting each drone as a call."""

    if not samples:
        raise ValueError("latency profile requires at least one planner invocation")
    wall_summary = summarize_latencies(
        [float(sample["wall_clock_latency_s"]) for sample in samples],
        deadline_s=deadline_s,
    )
    cpu_summary = summarize_latencies(
        [float(sample["process_cpu_latency_s"]) for sample in samples],
        deadline_s=deadline_s,
    )
    delay_summary = summarize_latencies(
        [float(sample["non_cpu_delay_s"]) for sample in samples],
        deadline_s=deadline_s,
    )
    overruns = []
    for sample in samples:
        wall_latency = float(sample["wall_clock_latency_s"])
        if wall_latency <= deadline_s:
            continue
        cpu_latency = float(sample["process_cpu_latency_s"])
        overruns.append(
            {
                "invocation_index": int(sample["invocation_index"]),
                "task_time_s": float(sample["task_time_s"]),
                "active_drone_count": int(sample["active_drone_count"]),
                "wall_clock_latency_s": round(wall_latency, 8),
                "process_cpu_latency_s": round(cpu_latency, 8),
                "non_cpu_delay_s": round(float(sample["non_cpu_delay_s"]), 8),
                "deadline_miss_receipt_count": int(sample["deadline_miss_receipt_count"]),
                "attribution": (
                    "algorithm_compute_overrun"
                    if cpu_latency > deadline_s
                    else "non_cpu_delay_candidate"
                ),
            }
        )
    return {
        "planner_invocation_count": len(samples),
        "deadline_overrun_invocation_count": len(overruns),
        "deadline_miss_receipt_count": sum(
            int(sample["deadline_miss_receipt_count"]) for sample in samples
        ),
        "wall_clock": wall_summary,
        "process_cpu": cpu_summary,
        "non_cpu_delay": delay_summary,
        "overrun_invocations": overruns,
    }


def adjudicate_controlled_repeats(
    replicates: list[dict[str, Any]],
    *,
    deadline_s: float,
    minimum_repeats: int = MINIMUM_CONTROLLED_REPEATS,
    p95_headroom_fraction: float = P95_HEADROOM_FRACTION,
    host_quiescent: bool = True,
) -> dict[str, Any]:
    """Apply a frozen no-cherry-picking rule before a full calibration rerun."""

    if minimum_repeats < 1:
        raise ValueError("minimum controlled repeats must be positive")
    if not 0.0 < p95_headroom_fraction < 1.0:
        raise ValueError("P95 headroom fraction must lie in (0, 1)")
    if not math.isfinite(deadline_s) or deadline_s <= 0.0:
        raise ValueError("planner deadline must be finite and positive")
    enough_repeats = len(replicates) >= minimum_repeats
    zero_overruns = bool(replicates) and all(
        int(item["timing"]["deadline_overrun_invocation_count"]) == 0
        and int(item["timing"]["deadline_miss_receipt_count"]) == 0
        for item in replicates
    )
    safety_pass = bool(replicates) and all(
        item["safety"]["returned_home_all"]
        and int(item["safety"]["collision_count"]) == 0
        and int(item["safety"]["out_of_bounds_actions"]) == 0
        for item in replicates
    )
    p95_limit_s = deadline_s * p95_headroom_fraction
    p95_headroom_pass = bool(replicates) and all(
        float(item["timing"]["wall_clock"]["p95_planning_latency_s"]) <= p95_limit_s
        for item in replicates
    )
    maximum_below_deadline = bool(replicates) and all(
        float(item["timing"]["wall_clock"]["max_planning_latency_s"]) <= deadline_s
        for item in replicates
    )
    checks = {
        "host_quiescent_before_and_after": host_quiescent,
        "minimum_consecutive_repeats_met": enough_repeats,
        "zero_wall_clock_overruns": zero_overruns,
        "zero_safety_failures": safety_pass,
        "wall_clock_p95_has_25_percent_headroom": p95_headroom_pass,
        "wall_clock_maximum_within_deadline": maximum_below_deadline,
    }
    if not enough_repeats:
        status = "DIAGNOSTIC_ONLY"
    else:
        status = "PASS" if all(checks.values()) else "NO_GO"
    return {
        "status": status,
        "checks": checks,
        "repeat_count": len(replicates),
        "minimum_repeat_count": minimum_repeats,
        "planner_deadline_s": deadline_s,
        "wall_clock_p95_limit_s": round(p95_limit_s, 8),
        "permits_full_calibration_rerun": status == "PASS",
        "failed_runs_are_never_replaced_or_deleted": True,
    }


def _run_profile_replicate(
    *,
    config: Any,
    city: dict[str, Any],
    episode: dict[str, Any],
    task_spec: dict[str, Any],
    method_id: str,
    private_episode: dict[str, Any] | None,
    repeat_index: int,
) -> dict[str, Any]:
    policy = create_baseline(
        method_id,
        config,
        task_spec,
        public_episode_projection(episode),
        private_episode=private_episode,
    )
    runtime = L0FleetRuntime(
        config,
        city,
        episode,
        receipt_secret=b"g2-i-public-policy-latency-profile-v1",
        public_task_spec=task_spec,
        public_episode=public_episode_projection(episode),
    )
    raw_samples: list[dict[str, Any]] = []
    public_action_trace: list[dict[str, Any]] = []

    def instrumented_policy(observations: dict[str, Any]) -> dict[str, Any]:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        actions = policy(observations)
        public_action_trace.append(dict(actions))
        cpu_elapsed = time.process_time() - cpu_started
        wall_elapsed = time.perf_counter() - wall_started
        raw_samples.append(
            {
                "invocation_index": len(raw_samples),
                "task_time_s": min(
                    float(observation.timestamp_s) for observation in observations.values()
                ),
                "active_drone_count": len(observations),
                "inner_wall_clock_latency_s": wall_elapsed,
                "process_cpu_latency_s": cpu_elapsed,
            }
        )
        return actions

    result = runtime.run_policy(instrumented_policy)
    receipts = result["execution_receipts"]
    invoked_receipts = [receipt for receipt in receipts if receipt.get("planner_invoked") is True]
    held_receipts = [receipt for receipt in receipts if receipt.get("planner_invoked") is False]
    if any(
        float(receipt["planning_latency_s"]) != 0.0 or bool(receipt["deadline_miss"])
        for receipt in held_receipts
    ):
        raise RuntimeError("held-action receipts incorrectly contain planner timing")
    cursor = 0
    samples = []
    for raw_sample in raw_samples:
        active_count = int(raw_sample["active_drone_count"])
        step_receipts = invoked_receipts[cursor : cursor + active_count]
        cursor += active_count
        if len(step_receipts) != active_count:
            raise RuntimeError("profile trace cannot be bound to execution receipts")
        wall_values = {round(float(receipt["planning_latency_s"]), 12) for receipt in step_receipts}
        if len(wall_values) != 1:
            raise RuntimeError("centralized planner receipts disagree on call latency")
        wall_latency = float(step_receipts[0]["planning_latency_s"])
        cpu_latency = float(raw_sample["process_cpu_latency_s"])
        samples.append(
            {
                "invocation_index": raw_sample["invocation_index"],
                "task_time_s": raw_sample["task_time_s"],
                "active_drone_count": active_count,
                "wall_clock_latency_s": wall_latency,
                "process_cpu_latency_s": cpu_latency,
                "non_cpu_delay_s": max(0.0, wall_latency - cpu_latency),
                "deadline_miss_receipt_count": sum(
                    bool(receipt["deadline_miss"]) for receipt in step_receipts
                ),
            }
        )
    if cursor != len(invoked_receipts):
        raise RuntimeError("profile trace leaves unbound planner-invocation receipts")
    deadline_s = float(config.raw["execution_contract"]["planning_deadline_s"])
    return {
        "repeat_index": repeat_index,
        "timing": summarize_invocations(samples, deadline_s=deadline_s),
        "safety": {
            "returned_home_all": all(bool(value) for value in result["returned_home"].values()),
            "collision_count": int(result["budget_ledger"]["collisions"]),
            "out_of_bounds_actions": int(result["budget_ledger"]["out_of_bounds_actions"]),
        },
        "wall_clock_run_s": round(float(result["wall_clock_s"]), 8),
        "public_action_behavior": summarize_public_action_trace(public_action_trace),
    }


def profile_policy_latency(
    manifest_path: Path, *, method_id: str, record_index: int, repeat_count: int = 1
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported G2-I calibration manifest")
    if manifest.get("purpose") != "method-independent-task-calibration":
        raise ValueError("latency profile requires method-independent calibration inputs")
    if manifest.get("self_method_results_used") is not False:
        raise ValueError("latency profile cannot consume self-method results")
    records = manifest.get("records")
    if not isinstance(records, list) or record_index >= len(records):
        raise ValueError("record index is outside the calibration manifest")
    record = records[record_index]
    if not isinstance(record, dict):
        raise ValueError("calibration record must be an object")
    root = manifest_path.parent
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    city = read_json(_local_path(root, record["city_path"], "city_path"))
    episode = read_json(_local_path(root, record["private_episode_path"], "private_episode_path"))
    if str(city.get("split")) in FORMAL_SPLITS:
        raise ValueError("latency profile must not inspect a formal split")
    if episode.get("layout_id") != city.get("layout_id") or episode.get("layout_hash") != city.get(
        "layout_hash"
    ):
        raise ValueError("frozen episode is not bound to its city")
    task_spec = compile_g2_i_task_spec(city, config.raw["execution_contract"], config.raw["fleet"])
    sector = episode.get("mission_sector")
    if not isinstance(sector, dict):
        raise ValueError("frozen episode lacks a mission sector")
    validate_public_mission_sector(
        sector,
        task_spec["inspection_atlas"],
        episode.get("starts"),
        config.raw["execution_contract"],
    )
    descriptor = BASELINES[method_id]
    if repeat_count < 1:
        raise ValueError("latency profile repeat count must be positive")
    host_before = host_snapshot()
    foreign_before = foreign_isaac_processes()
    # Keep this lookup aligned with ``L0FleetRuntime.step``.  The clock block
    # controls the overrun policy, while the deadline itself is a top-level
    # execution-contract field.
    deadline_s = float(config.raw["execution_contract"]["planning_deadline_s"])
    replicates = [
        _run_profile_replicate(
            config=config,
            city=city,
            episode=episode,
            task_spec=task_spec,
            method_id=method_id,
            private_episode=episode if descriptor.requires_private_truth else None,
            repeat_index=repeat_index,
        )
        for repeat_index in range(repeat_count)
    ]
    foreign_after = foreign_isaac_processes()
    host_after = host_snapshot()
    host_quiescent = not foreign_before and not foreign_after
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "execution_level": "L0",
        "method_id": method_id,
        "method_requires_private_truth": descriptor.requires_private_truth,
        "manifest_hash": content_hash(manifest),
        "layout_hash": city["layout_hash"],
        "episode_hash": episode["episode_hash"],
        "planner_deadline_s": deadline_s,
        "replicates": replicates,
        "controlled_repeat_adjudication": adjudicate_controlled_repeats(
            replicates,
            deadline_s=deadline_s,
            host_quiescent=host_quiescent,
        ),
        "host_context": {
            "foreign_isaac_process_count_before": len(foreign_before),
            "foreign_isaac_process_count_after": len(foreign_after),
            "commit_fraction_before": host_before.commit_fraction,
            "commit_fraction_after": host_after.commit_fraction,
            "process_details_omitted": True,
        },
        "timing_contract": {
            "authoritative_deadline_basis": "wall_clock",
            "process_cpu_is_diagnostic_only": True,
            "non_cpu_delay_does_not_erase_a_wall_clock_failure": True,
            "centralized_call_counted_once_per_invocation": True,
            "receipt_deadline_misses_count_active_drones": True,
            "built_in_policy_performs_no_intentional_io": True,
        },
        "private_truth_omitted": not descriptor.requires_private_truth,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    write_json(
        args.output,
        profile_policy_latency(
            args.manifest,
            method_id=args.method,
            record_index=args.record_index,
            repeat_count=args.repeat,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
