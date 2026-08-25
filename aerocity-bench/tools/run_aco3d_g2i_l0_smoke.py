"""Run the source-locked ACO3D translation through the public G2-I ABI.

This calibration smoke test proves only that the locked source translation can
consume public inspection sectors through an isolated JSONL process.  It does
not claim native MATLAB execution, upstream four-UAV allocation, or a formal
hidden-target-search score.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.adapters import (  # noqa: E402
    AdapterDeclaration,
    ExternalProcessPlannerBridge,
    arbitrate_public_fleet_actions,
)
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json  # noqa: E402
from aerocity_bench.ordinary_config import load_ordinary_config  # noqa: E402
from aerocity_bench.public_boundary import audit_public_layout  # noqa: E402
from aerocity_bench.runtime import L0FleetRuntime  # noqa: E402
from aerocity_bench.targets_v3 import public_episode_projection  # noqa: E402

UPSTREAM_URL = "https://github.com/duynamrcv/aco_3d_ipp.git"
UPSTREAM_COMMIT = "c395f5b61f6746b2d39310dbc55a7ec3e1eae2d5"
UPSTREAM_LICENSE = "MIT"
ADAPTER_VERSION = "aco3d-source-translation-v1"
MAXIMUM_RESET_BYTES = 2_000_000
PASS_SEMANTICS = (
    "safety_and_abi_integrity; coverage and confirmation are never success criteria; "
    "return is required only when return_closure_required=true"
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--require-return", action="store_true")
    return parser.parse_args(argv)


def _required(path: Path, *, directory: bool = False) -> Path:
    resolved = path.resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise ValueError(f"required {expected} does not exist: {resolved}")
    return resolved


def _verify_source_translation(upstream_source: Path, adapter_path: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(adapter_path),
            "--upstream-source",
            str(upstream_source),
            "--version",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ACO3D source-lock verification failed: {detail}")
    version = completed.stdout.strip()
    if version != ADAPTER_VERSION:
        raise RuntimeError(
            f"ACO3D source-translation version drift: expected {ADAPTER_VERSION}, got {version}"
        )
    return version


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    ledger = result["budget_ledger"]
    returned_home = result["returned_home"]
    coverage_denominators = result.get("coverage_denominators", {})
    inspection_area_total = float(coverage_denominators.get("inspection_atlas_area_m2", 0.0))
    inspection_cell_total = int(coverage_denominators.get("inspection_atlas_cells", 0))
    inspection_area_trace = list(result.get("inspection_coverage_trace", []))
    inspection_cell_trace = list(result.get("inspection_cell_count_trace", []))
    inspection_area_covered = float(inspection_area_trace[-1][1]) if inspection_area_trace else 0.0
    inspection_cells_covered = int(inspection_cell_trace[-1][1]) if inspection_cell_trace else 0
    receipts = list(result["execution_receipts"])
    return {
        "task_time_s": float(result["task_time_s"]),
        "receipt_count": len(receipts),
        "confirmation_count": len(result["confirmations"]),
        "observe_request_count": sum(
            receipt.get("action_requested") == "OBSERVE" for receipt in receipts
        ),
        "inspection_coverage": {
            "covered_area_m2": inspection_area_covered,
            "total_area_m2": inspection_area_total,
            "area_fraction": (
                inspection_area_covered / inspection_area_total
                if inspection_area_total > 0.0
                else None
            ),
            "covered_cell_count": inspection_cells_covered,
            "total_cell_count": inspection_cell_total,
            "cell_fraction": (
                inspection_cells_covered / inspection_cell_total
                if inspection_cell_total > 0
                else None
            ),
        },
        "failure_categories": sorted({str(record["category"]) for record in result["failures"]}),
        "collision_count": int(ledger["collisions"]),
        "out_of_bounds_actions": int(ledger["out_of_bounds_actions"]),
        "deadline_miss_tick_count": int(ledger["deadline_misses"]),
        "returned_home": returned_home,
        "all_returned_home": bool(returned_home) and all(returned_home.values()),
        "formal_score_eligible": bool(result["formal_score_eligible"]),
    }


def run_smoke(
    *,
    city_root: Path,
    release_config: Path,
    upstream_source: Path,
    max_steps: int,
    require_return: bool = False,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("ACO3D smoke max_steps must be positive")
    city_root = city_root.resolve()
    city_path = _required(city_root / "scene_authority" / "cityspec.json")
    task_path = _required(city_root / "method_public" / "task_spec.json")
    public_episode_path = _required(city_root / "method_public" / "episodes" / "episode-0000.json")
    private_episode_path = _required(
        city_root / "evaluator_private" / "episodes" / "episode-0000.json"
    )
    adapter_path = _required(Path(__file__).with_name("aco3d_g2i_process_adapter.py"))
    source_lock_path = _required(_REPOSITORY_ROOT / "external" / "aco3d" / "source-lock.json")
    upstream_source = _required(upstream_source, directory=True)
    external_version = _verify_source_translation(upstream_source, adapter_path)
    city = read_json(city_path)
    task_spec = read_json(task_path)
    public_episode = read_json(public_episode_path)
    private_episode = read_json(private_episode_path)
    config = load_ordinary_config(release_config.resolve())
    public_boundary_audit = audit_public_layout(city_root)
    if content_hash(public_episode_projection(private_episode)) != content_hash(public_episode):
        raise ValueError("ACO3D smoke input is not the public episode projection")
    declaration = AdapterDeclaration(
        adapter_id="aco3d-public-atlas-ordering-translation-v1",
        method_id="aco3d-public-atlas-inspection-ordering",
        capability_profile="G2-I",
        upstream_url=UPSTREAM_URL,
        upstream_commit=UPSTREAM_COMMIT,
        upstream_license=UPSTREAM_LICENSE,
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    bridge = ExternalProcessPlannerBridge(
        declaration,
        [
            sys.executable,
            "-u",
            str(adapter_path),
            "--upstream-source",
            str(upstream_source),
        ],
        cwd=_REPOSITORY_ROOT,
        response_timeout_s=20.0,
        maximum_line_bytes=MAXIMUM_RESET_BYTES,
    )
    try:
        bridge.reset(public_episode, public_task_spec=task_spec)
        runtime = L0FleetRuntime(
            config,
            city,
            private_episode,
            receipt_secret=b"aco3d-g2i-process-smoke-v1",
            public_task_spec=task_spec,
            public_episode=public_episode,
        )

        def policy(observations: dict[str, Any]) -> dict[str, Any]:
            actions, _ = bridge.act(observations)
            return arbitrate_public_fleet_actions(
                actions,
                observations,
                vehicle_radius_m=float(config.raw["execution_contract"]["vehicle"]["radius_m"]),
            )

        result = runtime.run_policy(policy, max_steps=max_steps)
        adapter_tax = bridge.adapter_tax_report()
    finally:
        bridge.close()
    summary = _summary(result)
    report = {
        "schema": "org.aerocity.bench.aco3d-public-atlas-smoke.v1",
        "scope": "calibration_only_source_locked_translation_smoke",
        "pass_semantics": PASS_SEMANTICS,
        "formal_score_eligible": False,
        "return_closure_required": require_return,
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "source_lock_sha256": file_hash(source_lock_path),
            "source_checkout_verified": True,
            "upstream_runtime_executed": False,
            "adapter_version": external_version,
        },
        "adapter": {
            "declaration": declaration.to_dict(),
            "runner_source_sha256": file_hash(Path(__file__).resolve()),
            "adapter_source_sha256": file_hash(adapter_path),
            "public_safety_arbiter_source_sha256": file_hash(
                _SOURCE_ROOT / "aerocity_bench" / "adapters.py"
            ),
            "public_safety_arbiter": "deterministic_segment_yield_v1",
            "adapter_tax": adapter_tax,
            "maximum_reset_bytes": MAXIMUM_RESET_BYTES,
        },
        "public_input_hashes": {
            "city": file_hash(city_path),
            "task_spec": file_hash(task_path),
            "public_episode": file_hash(public_episode_path),
            "release_config": file_hash(release_config.resolve()),
            "public_boundary_audit": content_hash(public_boundary_audit),
        },
        "private_evaluator_commitment": content_hash(private_episode),
        "execution": summary,
        "pass": (
            summary["receipt_count"] == max_steps * len(public_episode["starts"])
            and not summary["failure_categories"]
            and summary["collision_count"] == 0
            and summary["out_of_bounds_actions"] == 0
            and summary["deadline_miss_tick_count"] == 0
            and (not require_return or summary["all_returned_home"])
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    report = run_smoke(
        city_root=args.city_root,
        release_config=args.release_config,
        upstream_source=args.upstream_source,
        max_steps=args.max_steps,
        require_return=args.require_return,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"ACO3D_G2I_SMOKE={'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
