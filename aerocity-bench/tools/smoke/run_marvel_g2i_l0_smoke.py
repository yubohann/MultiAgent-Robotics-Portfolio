"""Execute a bounded public-process smoke test for the frozen MARVEL adapter.

This tool is intentionally a calibration/integration diagnostic.  It runs the
private evaluator in the parent process but never places the private episode on
the adapter wire or writes it to the output report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.core.canonical import read_json, write_json  # noqa: E402
from aerocity_bench.generation.ordinary_config import load_ordinary_config  # noqa: E402
from aerocity_bench.generation.targets_v3 import public_episode_projection  # noqa: E402
from aerocity_bench.release.public_boundary import audit_public_layout  # noqa: E402
from aerocity_bench.runtime.adapters import (  # noqa: E402
    AdapterDeclaration,
    ExternalProcessPlannerBridge,
    arbitrate_public_fleet_actions,
)
from aerocity_bench.runtime.runtime import L0FleetRuntime  # noqa: E402

UPSTREAM_URL = "https://github.com/marmotlab/MARVEL.git"
UPSTREAM_COMMIT = "318c2a6016d0f2d1dbb0dd08b3f8f8224b361e4c"
UPSTREAM_LICENSE = "MIT"
# Public G2I atlases exceed the legacy 1 MB G1 default; 2 MB covers the
# largest compressed calibration reset (about 1.34 MB) without leaving the
# channel unbounded.
MAXIMUM_RESET_BYTES = 2_000_000


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--marvel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--require-return",
        action="store_true",
        help="fail unless every vehicle closes the calibration episode at home",
    )
    return parser.parse_args(argv)


def _required(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required file does not exist: {resolved}")
    return resolved


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    ledger = result["budget_ledger"]
    returned_home = result["returned_home"]
    return {
        "task_time_s": float(result["task_time_s"]),
        "receipt_count": len(result["execution_receipts"]),
        "confirmation_count": len(result["confirmations"]),
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
    marvel_root: Path,
    max_steps: int,
    device: str,
    require_return: bool = False,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("MARVEL smoke test max_steps must be positive")
    city_root = city_root.resolve()
    city_path = _required(city_root / "scene_authority" / "cityspec.json")
    task_path = _required(city_root / "method_public" / "task_spec.json")
    public_episode_path = _required(city_root / "method_public" / "episodes" / "episode-0000.json")
    private_episode_path = _required(
        city_root / "evaluator_private" / "episodes" / "episode-0000.json"
    )
    checkpoint = _required(marvel_root / "load_model" / "MARVEL" / "checkpoint.pth")
    adapter_path = _required(
        Path(__file__).resolve().parents[1] / "adapters" / "marvel_g2i_process_adapter.py"
    )
    city = read_json(city_path)
    task_spec = read_json(task_path)
    public_episode = read_json(public_episode_path)
    private_episode = read_json(private_episode_path)
    config = load_ordinary_config(release_config.resolve())
    public_boundary_audit = audit_public_layout(city_root)
    if public_episode_projection(private_episode) != public_episode:
        raise ValueError(
            "MARVEL smoke input public episode is not the projection of its "
            "private evaluator episode"
        )
    declaration = AdapterDeclaration(
        adapter_id="marvel-frozen-checkpoint-g2i-transfer-v1",
        method_id="marvel-2d-to-g2i-transfer-diagnostic",
        capability_profile="G2I",
        upstream_url=UPSTREAM_URL,
        upstream_commit=UPSTREAM_COMMIT,
        upstream_license=UPSTREAM_LICENSE,
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    command = [
        sys.executable,
        "-u",
        str(adapter_path),
        "--marvel-root",
        str(marvel_root.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
    ]
    bridge = ExternalProcessPlannerBridge(
        declaration,
        command,
        cwd=Path(__file__).resolve().parents[2],
        response_timeout_s=20.0,
        maximum_line_bytes=MAXIMUM_RESET_BYTES,
    )
    try:
        bridge.reset(public_episode, public_task_spec=task_spec)
        runtime = L0FleetRuntime(
            config,
            city,
            private_episode,
            receipt_secret=b"marvel-g2i-process-smoke-v1",
            public_task_spec=task_spec,
            public_episode=public_episode,
        )

        def policy(observations: dict[str, Any]) -> dict[str, Any]:
            actions, _ = bridge.act(observations)
            return arbitrate_public_fleet_actions(
                actions,
                observations,
                vehicle_radius_m=float(
                    config.raw["execution_contract"]["vehicle"]["radius_m"]
                ),
            )

        result = runtime.run_policy(policy, max_steps=max_steps)
        adapter_tax = bridge.adapter_tax_report()
    finally:
        bridge.close()
    summary = _summary(result)
    report = {
        "schema": "org.aerocity.bench.upstream-marvel-g2i-smoke.v1",
        "scope": "calibration_only_cross_environment_transfer_diagnostic",
        "formal_score_eligible": False,
        "return_closure_required": require_return,
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "checkpoint": checkpoint.name,
        },
        "adapter": {
            "declaration": declaration.to_dict(),
            "runner": Path(__file__).name,
            "adapter_source": adapter_path.name,
            "public_safety_arbiter": "deterministic_segment_yield_v1",
            "adapter_tax": adapter_tax,
            "maximum_reset_bytes": MAXIMUM_RESET_BYTES,
        },
        "public_inputs": {
            "city": city_path.name,
            "task_spec": task_path.name,
            "public_episode": public_episode_path.name,
            "release_config": release_config.name,
            "public_boundary_audit": public_boundary_audit["status"],
        },
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
    return report


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    report = run_smoke(
        city_root=args.city_root,
        release_config=args.release_config,
        marvel_root=args.marvel_root,
        max_steps=args.max_steps,
        device=args.device,
        require_return=args.require_return,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"MARVEL_G2I_SMOKE={'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
