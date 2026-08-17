"""Run a bounded, provenance-labelled baseline suite on the CPU pilot runtime.

This module is preparation for the native Isaac baseline package.  It executes
the existing public-policy runtime end to end, records a fixed evaluation
contract, resource observations, and every failed attempt, but it deliberately
does not claim Isaac, hardware, or benchmark evidence.  The report never
stores evaluator-private target coordinates or the private truth digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .methods import NATIVE_DESCRIPTORS, create_native_policy
from .provenance import detect_source_provenance
from .resource_telemetry import ResourceTelemetry
from .runtime import PilotRuntimeConfig, PilotSwarmRuntime


BASELINE_SUITE_SCHEMA = "org.rivermark.benchmark.baseline-suite.v1"
BASELINE_REPORT_SCHEMA = "org.rivermark.benchmark.baseline-report.v1"
PILOT_BACKEND = "rivermark-kinematic-pilot-v1"
PRIMARY_METRIC = "normalized_confirmed_auc"
_ALLOWED_FAMILIES = {"classical", "rl", "marl", "quality_diversity", "vlm", "vln", "vla", "world_model"}


class BaselineConfigError(ValueError):
    """Raised when a baseline suite would not be comparable or auditable."""


class BaselineReportError(ValueError):
    """Raised when a baseline report is incomplete or has been tampered with."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineConfigError(f"cannot read baseline config: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineConfigError("baseline config root must be an object")
    return value


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineConfigError(f"{name} must be an object")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BaselineConfigError(f"{name} must be a non-negative integer")
    return value


def validate_baseline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-compatible copy of a suite definition."""

    allowed = {
        "schema", "suite_id", "backend", "formal_benchmark_admission", "agent_count",
        "runtime", "train", "tune", "evaluate", "methods", "budget",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise BaselineConfigError("unknown config fields: " + ", ".join(unknown))
    if config.get("schema") != BASELINE_SUITE_SCHEMA:
        raise BaselineConfigError(f"schema must be {BASELINE_SUITE_SCHEMA}")
    suite_id = config.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id or len(suite_id) > 80:
        raise BaselineConfigError("suite_id must be a non-empty short string")
    if config.get("backend") != PILOT_BACKEND:
        raise BaselineConfigError("only the explicitly labelled CPU pilot backend is supported")
    if config.get("formal_benchmark_admission") is not False:
        raise BaselineConfigError("baseline pilot reports must set formal_benchmark_admission=false")
    agent_count = config.get("agent_count", 8)
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or not 2 <= agent_count <= 32:
        raise BaselineConfigError("agent_count must be an integer in [2, 32]")

    runtime = dict(_required_mapping(config.get("runtime", {}), "runtime"))
    runtime_allowed = {"dt_s", "world_size_xy_m", "camera_width", "camera_height", "max_speed_mps"}
    if set(runtime) - runtime_allowed:
        raise BaselineConfigError("unknown runtime fields: " + ", ".join(sorted(set(runtime) - runtime_allowed)))
    dt_s = runtime.get("dt_s", 0.2)
    if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(float(dt_s)) or float(dt_s) <= 0:
        raise BaselineConfigError("runtime.dt_s must be a positive finite number")
    world = runtime.get("world_size_xy_m", [32.0, 24.0])
    if not isinstance(world, list) or len(world) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) <= 6.0 for v in world):
        raise BaselineConfigError("runtime.world_size_xy_m must contain two values greater than 6")
    for key in ("camera_width", "camera_height"):
        value = runtime.get(key, 96 if key == "camera_width" else 72)
        if isinstance(value, bool) or not isinstance(value, int) or value < (32 if key == "camera_width" else 24):
            raise BaselineConfigError(f"runtime.{key} is below the minimum supported resolution")

    train = dict(_required_mapping(config.get("train", {}), "train"))
    tune = dict(_required_mapping(config.get("tune", {}), "tune"))
    evaluate = dict(_required_mapping(config.get("evaluate"), "evaluate"))
    for name, section, allowed_keys in (
        ("train", train, {"enabled", "seed", "episodes"}),
        ("tune", tune, {"enabled", "seed", "max_trials"}),
        ("evaluate", evaluate, {"seeds", "episodes_per_seed"}),
    ):
        unknown_section = sorted(set(section) - allowed_keys)
        if unknown_section:
            raise BaselineConfigError(f"unknown {name} fields: {', '.join(unknown_section)}")
    for name, section, count_key in (("train", train, "episodes"), ("tune", tune, "max_trials")):
        if not isinstance(section.get("enabled", False), bool):
            raise BaselineConfigError(f"{name}.enabled must be boolean")
        _nonnegative_int(section.get("seed", 0), f"{name}.seed")
        _nonnegative_int(section.get(count_key, 0), f"{name}.{count_key}")
        if section["enabled"]:
            raise BaselineConfigError(f"{name} execution is not implemented by the bounded harness; set enabled=false")
    seeds = evaluate.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise BaselineConfigError("evaluate.seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise BaselineConfigError("evaluate.seeds must be unique")
    episodes_per_seed = evaluate.get("episodes_per_seed", 1)
    if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int) or not 1 <= episodes_per_seed <= 64:
        raise BaselineConfigError("evaluate.episodes_per_seed must be in [1, 64]")

    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise BaselineConfigError("methods must be a non-empty list")
    normalized_methods: list[dict[str, Any]] = []
    method_ids: set[str] = set()
    for index, raw in enumerate(methods):
        item = _required_mapping(raw, f"methods[{index}]")
        if set(item) - {"method_id", "family", "information_profile"}:
            raise BaselineConfigError(f"unknown methods[{index}] fields")
        method_id = item.get("method_id")
        if not isinstance(method_id, str) or method_id in method_ids:
            raise BaselineConfigError(f"methods[{index}].method_id must be unique")
        descriptor = NATIVE_DESCRIPTORS.get(method_id)
        if descriptor is None:
            raise BaselineConfigError(f"{method_id} is not a runnable native pilot method")
        family = item.get("family", descriptor.family)
        profile = item.get("information_profile", descriptor.information_profile)
        if family != descriptor.family or family not in _ALLOWED_FAMILIES:
            raise BaselineConfigError(f"{method_id} family does not match the registered descriptor")
        if profile != descriptor.information_profile:
            raise BaselineConfigError(f"{method_id} information_profile does not match the registered descriptor")
        method_ids.add(method_id)
        normalized_methods.append({"method_id": method_id, "family": family, "information_profile": profile})
    budget = dict(_required_mapping(config.get("budget"), "budget"))
    if set(budget) - {"max_steps", "timeout_s", "max_failures"}:
        raise BaselineConfigError("unknown budget fields")
    max_steps = budget.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100_000:
        raise BaselineConfigError("budget.max_steps must be in [1, 100000]")
    timeout_s = budget.get("timeout_s")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0:
        raise BaselineConfigError("budget.timeout_s must be a positive finite number")
    max_failures = _nonnegative_int(budget.get("max_failures", 0), "budget.max_failures")
    normalized = dict(config)
    normalized.update({
        "agent_count": agent_count,
        "runtime": runtime,
        "train": train,
        "tune": tune,
        "evaluate": {"seeds": list(seeds), "episodes_per_seed": episodes_per_seed},
        "methods": normalized_methods,
        "budget": {"max_steps": max_steps, "timeout_s": float(timeout_s), "max_failures": max_failures},
    })
    return normalized


def _public_metrics(evaluation: Any) -> dict[str, Any]:
    """Copy evaluator output without exposing the private truth digest."""

    raw = evaluation.as_dict()
    return {
        key: value
        for key, value in raw.items()
        if key != "evaluator_truth_sha256"
    } | {"evaluator_truth_bound": True, "private_truth_digest_emitted": False}


def _run_id(suite_hash: str, method_id: str, seed: int, episode_index: int) -> str:
    return _sha256_bytes(f"{suite_hash}:{method_id}:{seed}:{episode_index}".encode("utf-8"))[:24]


def validate_baseline_report(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic issues for a generated baseline report."""

    issues: list[str] = []
    if not isinstance(report, Mapping):
        return ("report root must be an object",)
    if report.get("schema") != BASELINE_REPORT_SCHEMA:
        issues.append("schema mismatch")
    if report.get("backend") != PILOT_BACKEND:
        issues.append("backend is not the bounded CPU pilot")
    if report.get("formal_benchmark_admission") is not False:
        issues.append("formal_benchmark_admission must be false")
    if report.get("scientific_benchmark_claims_permitted") is not False:
        issues.append("scientific_benchmark_claims_permitted must be false")
    try:
        config = validate_baseline_config(_required_mapping(report.get("config"), "report.config"))
    except (BaselineConfigError, TypeError) as exc:
        issues.append(f"invalid embedded config: {exc}")
        config = None
    config_hash = report.get("config_sha256")
    if config is not None:
        expected_hash = _sha256_bytes(_canonical_bytes(config))
        if config_hash != expected_hash:
            issues.append("config_sha256 does not match embedded config")
    elif not isinstance(config_hash, str):
        issues.append("config_sha256 is missing")
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        issues.append("attempts must be a list")
        attempts = []
    expected_count = report.get("attempt_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count != len(attempts):
        issues.append("attempt_count does not match attempts")
    passed_count = sum(isinstance(row, Mapping) and row.get("status") == "passed" for row in attempts)
    failed_count = sum(isinstance(row, Mapping) and row.get("status") != "passed" for row in attempts)
    if report.get("passed_count") != passed_count:
        issues.append("passed_count does not match attempts")
    if report.get("failed_count") != failed_count:
        issues.append("failed_count does not match attempts")
    method_ids = {item["method_id"] for item in config["methods"]} if config is not None else set()
    seen_ids: set[str] = set()
    private_keys = {"evaluator_truth_sha256", "hidden_target", "target_positions", "target_coordinates", "private_evaluator"}
    for index, row in enumerate(attempts):
        if not isinstance(row, Mapping):
            issues.append(f"attempts[{index}] must be an object")
            continue
        method_id, seed, episode_index = row.get("method_id"), row.get("seed"), row.get("episode_index")
        run_id = row.get("run_id")
        if method_id not in method_ids:
            issues.append(f"attempts[{index}] has an unknown method_id")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            issues.append(f"attempts[{index}] has an invalid seed")
        if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
            issues.append(f"attempts[{index}] has an invalid episode_index")
        if config is not None and isinstance(method_id, str) and isinstance(seed, int) and isinstance(episode_index, int):
            expected_id = _run_id(str(config_hash), method_id, seed, episode_index)
            if run_id != expected_id:
                issues.append(f"attempts[{index}] run_id does not bind method/seed/config")
        if not isinstance(run_id, str) or run_id in seen_ids:
            issues.append(f"attempts[{index}] run_id is missing or duplicated")
        if isinstance(run_id, str):
            seen_ids.add(run_id)
        if row.get("status") not in {"passed", "failed"}:
            issues.append(f"attempts[{index}] has an invalid status")
        metrics = row.get("metrics")
        if isinstance(metrics, Mapping) and private_keys.intersection(metrics):
            issues.append(f"attempts[{index}] exposes private evaluator fields")
        if row.get("formal_benchmark_admission") is not False:
            issues.append(f"attempts[{index}] changes formal admission boundary")
    summaries = report.get("summaries")
    if not isinstance(summaries, list) or (config is not None and len(summaries) != len(config["methods"])):
        issues.append("summaries do not cover the configured methods")
    if report.get("status") not in {"completed", "stopped_failure_budget"}:
        issues.append("invalid report status")
    return tuple(issues)


def verify_baseline_report(path: Path) -> dict[str, Any]:
    """Load and verify one report without executing a policy."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineReportError(f"cannot read baseline report: {path}") from exc
    issues = validate_baseline_report(report)
    if issues:
        raise BaselineReportError("; ".join(issues))
    return dict(report)


def run_baseline_suite(config_path: Path, output_path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Execute the configured CPU suite and atomically write its report."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing baseline report: {output_path}")
    config = validate_baseline_config(_load_json(config_path))
    config_bytes = _canonical_bytes(config)
    config_hash = _sha256_bytes(config_bytes)
    source = detect_source_provenance()
    telemetry = ResourceTelemetry()
    attempts: list[dict[str, Any]] = []
    failures = 0
    stop_reason: str | None = None
    eval_seeds = config["evaluate"]["seeds"]
    episodes_per_seed = config["evaluate"]["episodes_per_seed"]
    runtime_values = config["runtime"]
    for method_spec in config["methods"]:
        method_id = method_spec["method_id"]
        descriptor = NATIVE_DESCRIPTORS[method_id]
        for seed in eval_seeds:
            for episode_index in range(episodes_per_seed):
                run_id = _run_id(config_hash, method_id, seed, episode_index)
                started = time.perf_counter()
                telemetry.sample(f"before:{run_id}")
                attempt: dict[str, Any] = {
                    "run_id": run_id,
                    "method_id": method_id,
                    "family": descriptor.family,
                    "information_profile": descriptor.information_profile,
                    "seed": seed,
                    "episode_index": episode_index,
                    "status": "failed",
                    "backend": PILOT_BACKEND,
                    "formal_benchmark_admission": False,
                }
                try:
                    runtime = PilotSwarmRuntime(
                        PilotRuntimeConfig(
                            agent_count=config["agent_count"],
                            seed=seed,
                            dt_s=float(runtime_values.get("dt_s", 0.2)),
                            max_steps=config["budget"]["max_steps"],
                            world_size_xy_m=tuple(float(v) for v in runtime_values.get("world_size_xy_m", [32.0, 24.0])),
                            camera_width=int(runtime_values.get("camera_width", 96)),
                            camera_height=int(runtime_values.get("camera_height", 72)),
                            max_speed_mps=float(runtime_values.get("max_speed_mps", 2.8)),
                        ),
                        information_profile=descriptor.information_profile,
                    )
                    policy = create_native_policy(method_id)
                    observations = runtime.reset()
                    policy.reset(
                        runtime.mission,
                        runtime.config.agent_count,
                        public_geometry=runtime.public_geometry if descriptor.information_profile == "geometry_state" else None,
                    )
                    steps = 0
                    while not runtime.done:
                        observations, _ = runtime.step(policy.act(observations))
                        steps += 1
                    elapsed_s = time.perf_counter() - started
                    evaluation = runtime.evaluate()
                    attempt.update({
                        "status": "passed" if elapsed_s <= config["budget"]["timeout_s"] else "failed",
                        "steps": steps,
                        "wall_time_s": elapsed_s,
                        "policy_provenance": policy.provenance(),
                        "metrics": _public_metrics(evaluation),
                    })
                    if elapsed_s > config["budget"]["timeout_s"]:
                        attempt["failure"] = {
                            "code": "timeout",
                            "exception_type": "TimeoutBudgetExceeded",
                            "message": "rollout completed after the declared wall-clock budget",
                        }
                        failures += 1
                except Exception as exc:  # retain every failed attempt and continue within the cap
                    failures += 1
                    attempt.update({
                        "wall_time_s": time.perf_counter() - started,
                        "failure": {
                            "code": "exception",
                            "exception_type": type(exc).__name__,
                            "message": str(exc)[:240],
                        },
                    })
                finally:
                    telemetry.sample(f"after:{run_id}")
                attempts.append(attempt)
                if failures > config["budget"]["max_failures"]:
                    stop_reason = (
                        f"max_failures={config['budget']['max_failures']} exceeded after {run_id}"
                    )
                    break
            if stop_reason is not None:
                break
        if stop_reason is not None:
            break

    summaries: list[dict[str, Any]] = []
    metric_names = ("normalized_confirmed_auc", "confirmed_count", "confirmation_precision", "false_confirmation_count", "collision_count")
    for method_spec in config["methods"]:
        method_id = method_spec["method_id"]
        rows = [row for row in attempts if row["method_id"] == method_id]
        passed = [row for row in rows if row["status"] == "passed"]
        summary: dict[str, Any] = {
            "method_id": method_id,
            "family": method_spec["family"],
            "attempt_count": len(rows),
            "passed_count": len(passed),
            "failed_count": len(rows) - len(passed),
            "failure_rate": (len(rows) - len(passed)) / len(rows) if rows else 1.0,
        }
        for metric_name in metric_names:
            values = [row["metrics"].get(metric_name) for row in passed if isinstance(row.get("metrics", {}).get(metric_name), (int, float))]
            summary[f"mean_{metric_name}"] = sum(values) / len(values) if values else None
        summaries.append(summary)
    report = {
        "schema": BASELINE_REPORT_SCHEMA,
        "suite_id": config["suite_id"],
        "backend": PILOT_BACKEND,
        "formal_benchmark_admission": False,
        "scientific_benchmark_claims_permitted": False,
        "status": "stopped_failure_budget" if stop_reason is not None else "completed",
        "stop_reason": stop_reason,
        "config": config,
        "config_sha256": config_hash,
        "source_provenance": source.as_dict(),
        "evaluator": {
            "id": "kinematic-private-search-evaluator-v1",
            "primary_metric": PRIMARY_METRIC,
            "private_truth_used": True,
            "private_truth_digest_emitted": False,
            "public_evaluator_equivalence": False,
        },
        "train": {**config["train"], "executed": False, "reason": "CPU harness evaluates fixed native policies; no training is implied"},
        "tune": {**config["tune"], "executed": False, "reason": "No hyperparameter search is run by the bounded pilot harness"},
        "evaluate": config["evaluate"],
        "attempt_count": len(attempts),
        "passed_count": sum(row["status"] == "passed" for row in attempts),
        "failed_count": sum(row["status"] != "passed" for row in attempts),
        "attempts": attempts,
        "summaries": summaries,
        "resource_telemetry": telemetry.as_dict(),
        "limitations": [
            "not Isaac Sim or Isaac Lab",
            "not hardware or sim-to-real evidence",
            "native pilot policies are not third-party foundation-model results",
            "no formal dataset episode or ranking is produced",
            "timeout is checked at episode boundaries; no worker is force-killed",
        ],
    }
    report_issues = validate_baseline_report(report)
    if report_issues:
        raise BaselineReportError("generated baseline report failed validation: " + "; ".join(report_issues))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(report))
    temporary.replace(output_path)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_report is not None:
        report = verify_baseline_report(args.verify_report)
        print(json.dumps({"status": "valid", "suite_id": report["suite_id"], "report": str(args.verify_report)}, indent=2, sort_keys=True))
        return 0
    if args.config is None or args.output is None:
        raise SystemExit("--config and --output are required unless --verify-report is used")
    report = run_baseline_suite(args.config, args.output, overwrite=args.overwrite)
    print(json.dumps({
        "suite_id": report["suite_id"],
        "report": str(args.output),
        "attempt_count": report["attempt_count"],
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
        "formal_benchmark_admission": report["formal_benchmark_admission"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
