"""Precommit and repeat the development-only v10 external CF2X calibration.

The earlier v10 replay repaired an implementation defect in the v9 route
adapter, but was not generated from an outcome-blind execution plan.  This
tool binds the three public calibration layouts before launching Isaac and
then runs every planned pair once under the host guard.  It never reads
private target truth to select, retry, or omit a replay.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_SOURCE = _REPOSITORY / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from aerocity_bench.adapters import load_external_l1_adapter_manifest  # noqa: E402
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json  # noqa: E402
from aerocity_bench.cf2x_fleet_preflight_contract import (  # noqa: E402
    COMPLETE_CALIBRATION_PURPOSE,
    EXTERNAL_PROCESS_POLICY_MODE,
    validate_fleet_preflight_reports,
)
from aerocity_bench.host_guard import isaac_host_lock, run_guarded_process  # noqa: E402
from aerocity_bench.public_boundary import audit_public_layout  # noqa: E402

PLAN_SCHEMA = "org.aerocity.bench.ortools-v10-grouped-safe-sky-replication-plan.v1"
SUMMARY_SCHEMA = "org.aerocity.bench.ortools-v10-grouped-safe-sky-replication-summary.v1"
METHOD_ID = "ortools-public-atlas-routing-baseline"
ADAPTER_ID = "ortools-public-atlas-routing-v10-grouped-safe-sky"
ANCESTORS = (
    "g2-i-calibration-ancestor-00",
    "g2-i-calibration-ancestor-03",
    "g2-i-calibration-ancestor-05",
)
_PRIVATE_SUFFIX = ".private.json"
_PUBLIC_SUFFIX = ".public.json"


def _relative(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(_REPOSITORY).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the benchmark repository") from exc


def _repository_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes the benchmark repository")
    path = (_REPOSITORY / relative).resolve()
    if _REPOSITORY not in path.parents:
        raise ValueError(f"{label} escapes the benchmark repository")
    return path


def _mapping(values: list[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        ancestor, separator, raw_path = value.partition("=")
        if not separator or not ancestor or not raw_path or ancestor in result:
            raise ValueError(f"--{label} must use unique ANCESTOR=PATH values")
        result[ancestor] = Path(raw_path).resolve()
    if tuple(sorted(result)) != ANCESTORS:
        raise ValueError("replication requires exactly the frozen three calibration ancestors")
    return result


def _hash_bound_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected v10 replication plan schema")
    supplied = plan.get("plan_hash")
    hashed = dict(plan)
    hashed.pop("plan_hash", None)
    if not isinstance(supplied, str) or supplied != content_hash(hashed):
        raise ValueError("v10 replication plan hash differs")
    return plan


def _expected_output_paths(plan_root: Path, record: dict[str, Any]) -> tuple[Path, Path, Path]:
    public = (plan_root / str(record["public_report"])).resolve()
    private = (plan_root / str(record["private_report"])).resolve()
    runtime = (plan_root / str(record["runtime_root"])).resolve()
    for path, label in (
        (public, "public report"),
        (private, "private report"),
        (runtime, "runtime"),
    ):
        if plan_root not in path.parents:
            raise ValueError(f"{label} escapes the replication plan directory")
    return public, private, runtime


def build_plan(
    *,
    layouts: dict[str, Path],
    release_config: Path,
    cf2x_usd: Path,
    isaac_python: Path,
    adapter_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Write an outcome-blind replay plan without launching Isaac."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replication plan: {output}")
    if not all(path.is_file() for path in (release_config, cf2x_usd, isaac_python)):
        raise FileNotFoundError("release config, CF2X USD, and Isaac Python must exist")
    if "5_in_drone" in str(cf2x_usd).casefold() or "five_in_drone" in str(cf2x_usd).casefold():
        raise ValueError("replication refuses the prohibited 5_in_drone asset")
    adapter_manifest = load_external_l1_adapter_manifest(adapter_manifest_path)
    if (
        adapter_manifest.declaration.method_id != METHOD_ID
        or adapter_manifest.declaration.adapter_id != ADAPTER_ID
        or adapter_manifest.declaration.capability_profile != "G2-I"
        or adapter_manifest.declaration.process_boundary != "process"
        or adapter_manifest.declaration.training_allowed
        or adapter_manifest.task_domain != "3d_geometry_search"
        or adapter_manifest.comparability_claim != "transfer_diagnostic"
    ):
        raise ValueError("adapter manifest does not declare the v10 external G2-I diagnostic")

    records: list[dict[str, Any]] = []
    for ancestor in ANCESTORS:
        layout = layouts[ancestor]
        public_audit = audit_public_layout(layout)
        task_path = layout / "method_public" / "task_spec.json"
        episode_paths = sorted((layout / "method_public" / "episodes").glob("*.json"))
        if len(episode_paths) != 1:
            raise ValueError("each frozen layout must expose exactly one public episode")
        task = read_json(task_path)
        episode = read_json(episode_paths[0])
        if (
            not isinstance(task, dict)
            or not isinstance(episode, dict)
            or task.get("task_track") != "G2-I"
            or public_audit.get("status") != "PASS"
        ):
            raise ValueError("replication input is not a valid public G2-I layout")
        stem = f"{ancestor}__{METHOD_ID}"
        records.append(
            {
                "layout_ancestor": ancestor,
                "layout_root": _relative(layout, label="layout root"),
                "layout_id": public_audit["layout_id"],
                "public_task_sha256": file_hash(task_path),
                "public_episode_sha256": file_hash(episode_paths[0]),
                "public_execution_contract_hash": public_audit[
                    "public_execution_contract_hash"
                ],
                "public_report": f"replays/{stem}{_PUBLIC_SUFFIX}",
                "private_report": f"replays/{stem}{_PRIVATE_SUFFIX}",
                "runtime_root": f"runtime/{stem}",
            }
        )
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "formal_score_eligible": False,
        "status": "PRECOMMITTED_UNRUN",
        "purpose": "development-only-v10-grouped-safe-sky-replication",
        "method_id": METHOD_ID,
        "adapter_id": ADAPTER_ID,
        "layout_ancestors": list(ANCESTORS),
        "release_config_path": _relative(release_config, label="release config"),
        "release_config_sha256": file_hash(release_config),
        "cf2x_usd_sha256": file_hash(cf2x_usd),
        "isaac_python_sha256": file_hash(isaac_python),
        "external_adapter_manifest_path": _relative(
            adapter_manifest_path, label="external adapter manifest"
        ),
        "external_adapter_manifest_sha256": adapter_manifest.manifest_file_sha256,
        "external_adapter": adapter_manifest.public_provenance(),
        "records": records,
        "execution": {
            "duration_s": 300.0,
            "device": "cuda:0",
            "execution_mode": EXTERNAL_PROCESS_POLICY_MODE,
            "run_purpose": COMPLETE_CALIBRATION_PURPOSE,
            "one_attempt_per_pair": True,
            "retry_decision_reads_outcome": False,
            "failure_denominator_policy": "retain_every_precommitted_pair",
        },
        "runner_source_sha256": file_hash(Path(__file__).resolve()),
    }
    plan["plan_hash"] = content_hash(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return plan


def _validate_run_inputs(
    plan: dict[str, Any], *, isaac_python: Path, cf2x_usd: Path
) -> tuple[Path, Path]:
    if (
        plan.get("status") != "PRECOMMITTED_UNRUN"
        or plan.get("formal_score_eligible") is not False
        or plan.get("method_id") != METHOD_ID
        or plan.get("adapter_id") != ADAPTER_ID
        or tuple(plan.get("layout_ancestors", ())) != ANCESTORS
        or plan.get("runner_source_sha256") != file_hash(Path(__file__).resolve())
    ):
        raise ValueError("replication plan differs from this runner or the frozen v10 cohort")
    release_config = _repository_path(plan["release_config_path"], label="release config")
    adapter_manifest = _repository_path(
        plan["external_adapter_manifest_path"], label="external adapter manifest"
    )
    for path, expected_hash, label in (
        (release_config, plan["release_config_sha256"], "release config"),
        (cf2x_usd.resolve(), plan["cf2x_usd_sha256"], "CF2X USD"),
        (isaac_python.resolve(), plan["isaac_python_sha256"], "Isaac Python"),
        (adapter_manifest, plan["external_adapter_manifest_sha256"], "adapter manifest"),
    ):
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"frozen {label} differs before replication")
    manifest = load_external_l1_adapter_manifest(adapter_manifest)
    if manifest.public_provenance() != plan.get("external_adapter"):
        raise ValueError("external adapter provenance differs before replication")
    return release_config, adapter_manifest


def run_plan(
    *, plan_path: Path, isaac_python: Path, cf2x_usd: Path, timeout_s: float
) -> dict[str, Any]:
    """Execute every precommitted pair exactly once and preserve all receipts."""

    if timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    plan_path = plan_path.resolve()
    plan_root = plan_path.parent
    plan = _hash_bound_plan(plan_path)
    release_config, adapter_manifest = _validate_run_inputs(
        plan,
        isaac_python=isaac_python,
        cf2x_usd=cf2x_usd,
    )
    isaac_python = isaac_python.resolve()
    cf2x_usd = cf2x_usd.resolve()
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != len(ANCESTORS):
        raise ValueError("replication plan records differ from the frozen cohort")
    summary_path = plan_root / "replication-summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite replication summary")

    summary_records: list[dict[str, Any]] = []
    with isaac_host_lock():
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("replication plan contains a non-object record")
            ancestor = record.get("layout_ancestor")
            if ancestor not in ANCESTORS:
                raise ValueError("replication plan contains an unknown ancestor")
            layout = _repository_path(record.get("layout_root"), label="layout root")
            public, private, runtime = _expected_output_paths(plan_root, record)
            failure = public.with_name(f"{public.stem}.failure.json")
            if any(path.exists() for path in (public, private, runtime, failure)):
                raise FileExistsError(f"replication evidence path already exists: {ancestor}")
            runtime.mkdir(parents=True)
            command = [
                str(isaac_python),
                str(_REPOSITORY / "tools" / "cf2x_l1_fleet_preflight.py"),
                "--layout-root",
                str(layout),
                "--release-config",
                str(release_config),
                "--output",
                str(public),
                "--private-output",
                str(private),
                "--cf2x-usd",
                str(cf2x_usd),
                "--execution-mode",
                EXTERNAL_PROCESS_POLICY_MODE,
                "--external-adapter-manifest",
                str(adapter_manifest),
                "--run-purpose",
                COMPLETE_CALIBRATION_PURPOSE,
                "--max-sim-time-s",
                "300",
                "--device",
                str(plan["execution"]["device"]),
                "--headless",
            ]
            result = run_guarded_process(
                command,
                cwd=_REPOSITORY,
                environment=dict(os.environ),
                log_path=runtime / "isaac.log",
                report_path=runtime / "host_guard.json",
                timeout_s=timeout_s,
                evidence_binding=content_hash(
                    {"plan_hash": plan["plan_hash"], "layout_ancestor": ancestor}
                ),
            )
            outcome: dict[str, Any] = {
                "layout_ancestor": ancestor,
                "process_returncode": result.returncode,
                "host_guard_receipt_sha256": file_hash(runtime / "host_guard.json"),
            }
            if public.is_file() and private.is_file():
                validate_fleet_preflight_reports(public, private)
                report = read_json(public)
                if not isinstance(report, dict):
                    raise ValueError("replication public report is not an object")
                outcome.update(
                    {
                        "status": "COMPLETED_PAIR",
                        "public_report_sha256": file_hash(public),
                        "private_report_sha256": file_hash(private),
                        "safe_completion": report["final"]["safe_completion"],
                    }
                )
            elif failure.is_file():
                outcome.update(
                    {
                        "status": "PRELAUNCH_OR_EXECUTOR_FAILURE",
                        "failure_receipt_sha256": file_hash(failure),
                    }
                )
            else:
                outcome["status"] = "MISSING_EXECUTION_RECEIPT"
            summary_records.append(outcome)
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "formal_score_eligible": False,
        "plan_hash": plan["plan_hash"],
        "method_id": METHOD_ID,
        "records": summary_records,
        "result": (
            "COMPLETE_DEVELOPMENT_REPLICATION"
            if all(record["status"] == "COMPLETED_PAIR" for record in summary_records)
            else "INCOMPLETE_REPLICATION_RECEIPTS"
        ),
        "limitations": [
            "Development/calibration evidence only; never a formal score or method ranking.",
            (
                "The v10 external solver is a routing diagnostic, not an upstream "
                "3-D hidden-target-search method."
            ),
            "Every precommitted pair remains in the denominator regardless of outcome.",
        ],
    }
    summary["summary_hash"] = content_hash(summary)
    write_json(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--layout", action="append", required=True)
    prepare.add_argument("--release-config", type=Path, required=True)
    prepare.add_argument("--cf2x-usd", type=Path, required=True)
    prepare.add_argument("--isaac-python", type=Path, required=True)
    prepare.add_argument("--adapter-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--isaac-python", type=Path, required=True)
    run.add_argument("--cf2x-usd", type=Path, required=True)
    run.add_argument("--timeout-s", type=float, default=7200.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        plan = build_plan(
            layouts=_mapping(args.layout, label="layout"),
            release_config=args.release_config.resolve(),
            cf2x_usd=args.cf2x_usd.resolve(),
            isaac_python=args.isaac_python.resolve(),
            adapter_manifest_path=args.adapter_manifest.resolve(),
            output=args.output.resolve(),
        )
        print(f"ORTOOLS_V10_REPLICATION_PLAN={plan['plan_hash']}")
    else:
        summary = run_plan(
            plan_path=args.plan,
            isaac_python=args.isaac_python,
            cf2x_usd=args.cf2x_usd,
            timeout_s=args.timeout_s,
        )
        print(f"ORTOOLS_V10_REPLICATION_SUMMARY={summary['summary_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
