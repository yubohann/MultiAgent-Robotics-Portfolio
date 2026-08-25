"""Write an immutable public-only record for the OR-Tools v10 repair."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_SOURCE = _REPOSITORY / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json  # noqa: E402

SCHEMA = "org.aerocity.bench.ortools-grouped-safe-sky-fix-record.v1"
OLD_ADAPTER = _REPOSITORY / "tools" / "ortools_g2i_process_adapter.py"
NEW_ADAPTER = _REPOSITORY / "tools" / "ortools_g2i_process_adapter_v10_grouped_safe_sky.py"


def _load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("_ortools_v10_record", NEW_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the v10 adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_inputs(layout_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    public = layout_root / "splits" / "calibration"
    cities = sorted(path for path in public.glob("city-*") if path.is_dir())
    if len(cities) != 1:
        raise ValueError("layout root must contain exactly one calibration city")
    city = cities[0]
    task = read_json(city / "method_public" / "task_spec.json")
    episodes = sorted((city / "method_public" / "episodes").glob("*.json"))
    if len(episodes) != 1:
        raise ValueError("public layout must contain exactly one episode")
    episode = read_json(episodes[0])
    if not isinstance(task, dict) or not isinstance(episode, dict):
        raise ValueError("public task or episode is not an object")
    return task, episode


def build(*, layout_root: Path, old_aggregate: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fix record: {output}")
    if not OLD_ADAPTER.is_file() or not NEW_ADAPTER.is_file():
        raise FileNotFoundError("the v9 and v10 adapter sources must exist")
    old = read_json(old_aggregate)
    if (
        not isinstance(old, dict)
        or old.get("schema") != "org.aerocity.bench.cf2x-l1-calibration-aggregate.v1"
        or old.get("method_id") != "ortools-public-atlas-routing-baseline"
        or old.get("totals", {}).get("confirmation_receipt_count") != 0
    ):
        raise ValueError("old aggregate is not the preserved zero-confirmation v9 evidence")
    task, episode = _public_inputs(layout_root.resolve())
    adapter = _load_adapter()
    if file_hash(OLD_ADAPTER) != adapter.LEGACY_ADAPTER_SHA256:
        raise ValueError("v10 does not lock the preserved v9 source")

    # A pure public probe fixes every local group to canonical order.  It
    # validates task semantics without pulling private truth into this record.
    original = adapter.GroupedSafeSkyORToolsPlanner._solve_local_group
    adapter.GroupedSafeSkyORToolsPlanner._solve_local_group = (
        lambda self, *, origin, group: list(group)
    )
    try:
        planner = adapter.GroupedSafeSkyORToolsPlanner.from_public_reset(episode, task)
    finally:
        adapter.GroupedSafeSkyORToolsPlanner._solve_local_group = original
    assignments = episode["mission_sector"]["cell_assignment_by_drone"]
    if not isinstance(assignments, dict):
        raise ValueError("public episode has no per-drone assignment")
    route_counts = {
        drone_id: len(state.ordered_cell_ids)
        for drone_id, state in planner.routes.items()
    }
    expected_counts = {drone_id: len(cell_ids) for drone_id, cell_ids in assignments.items()}
    if route_counts != expected_counts or any(count == 0 for count in route_counts.values()):
        raise ValueError("v10 public route probe failed to retain the full assignment")

    certificate = episode["mission_sector"].get("capacity_certificate")
    if not isinstance(certificate, dict):
        raise ValueError("public episode has no capacity certificate")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "formal_score_eligible": False,
        "status": "IMPLEMENTATION_DEFECT_CONFIRMED_V10_RERUN_REQUIRED",
        "old_v9_evidence_preserved": {
            "aggregate_path": old_aggregate.resolve().relative_to(_REPOSITORY).as_posix(),
            "aggregate_sha256": file_hash(old_aggregate),
            "adapter_source_path": OLD_ADAPTER.relative_to(_REPOSITORY).as_posix(),
            "adapter_source_sha256": file_hash(OLD_ADAPTER),
            "anonymous_confirmation_receipt_count": 0,
        },
        "v10_adapter": {
            "adapter_id": adapter.ADAPTER_ID,
            "adapter_source_path": NEW_ADAPTER.relative_to(_REPOSITORY).as_posix(),
            "adapter_source_sha256": file_hash(NEW_ADAPTER),
            "locks_v9_source_sha256": adapter.LEGACY_ADAPTER_SHA256,
            "route_model": adapter.GROUPED_ROUTE_MODEL,
        },
        "public_probe": {
            "layout_root": layout_root.resolve().relative_to(_REPOSITORY).as_posix(),
            "public_task_sha256": file_hash(
                next((layout_root / "splits" / "calibration").glob("city-*")).resolve()
                / "method_public"
                / "task_spec.json"
            ),
            "public_route_count_by_drone": route_counts,
            "assigned_cell_count_by_drone": expected_counts,
            "local_successor_count_by_drone": {
                drone_id: len(planner.direct_successors_by_drone[drone_id])
                for drone_id in sorted(planner.routes)
            },
            "certificate_model": certificate.get("model"),
            "certificate_per_drone_lower_bound_s": certificate.get(
                "per_drone_required_lower_bound_s"
            ),
        },
        "scope": {
            "task_contract_changed": False,
            "target_process_changed": False,
            "private_evaluator_read": False,
            "repair": [
                "make assigned cells mandatory rather than optional solver nodes",
                "match public grouped facade scan transitions",
                "retain safe-sky transitions across groups and around top-down cells",
            ],
            "next_authorized_step": "new_precommitted_development_only_three_ancestor_L1_replay",
        },
    }
    payload["record_hash"] = content_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--old-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build(
        layout_root=args.layout_root,
        old_aggregate=args.old_aggregate,
        output=args.output,
    )
    print(f"ORTOOLS_V10_FIX_RECORD={record['record_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
