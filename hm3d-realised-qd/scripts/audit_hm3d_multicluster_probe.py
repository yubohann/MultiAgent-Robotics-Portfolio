"""Compare one-cluster throughput and two peer modes for multi-cluster isolation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic  # noqa: E402
from aerocity_method.runtime.hm3d_multicluster import (  # noqa: E402
    audit_reference_cluster_invariance,
)


def _cluster_zero(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("clusters")
    if not isinstance(rows, list):
        raise ValueError("probe record lacks clusters")
    for row in rows:
        if isinstance(row, Mapping) and row.get("cluster_id") == 0:
            return row
    raise ValueError("probe record lacks reference cluster zero")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", required=True, type=Path)
    parser.add_argument("--dual-hover", required=True, type=Path)
    parser.add_argument("--dual-random", required=True, type=Path)
    parser.add_argument("--trace-tolerance-m", type=float, default=1.0e-5)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    single = read_json_object(args.single)
    dual_hover = read_json_object(args.dual_hover)
    dual_random = read_json_object(args.dual_random)
    if single.get("cluster_count") != 1:
        raise ValueError("single probe must contain one cluster")
    if dual_hover.get("cluster_count") != 2 or dual_random.get("cluster_count") != 2:
        raise ValueError("dual probes must contain two clusters")
    invariance = audit_reference_cluster_invariance(
        _cluster_zero(dual_hover),
        _cluster_zero(dual_random),
        tolerance_m=args.trace_tolerance_m,
    )
    single_rate = float(single.get("real_decisions_per_wall_hour"))
    dual_rate = min(
        float(dual_hover.get("real_decisions_per_wall_hour")),
        float(dual_random.get("real_decisions_per_wall_hour")),
    )
    speedup = dual_rate / single_rate
    reasons = list(invariance["reasons"])
    for payload, label in ((dual_hover, "DUAL_HOVER"), (dual_random, "DUAL_RANDOM")):
        for field in (
            "cross_cluster_contact_count",
            "cross_cluster_message_count",
            "cross_cluster_map_delta_count",
        ):
            if payload.get(field) != 0:
                reasons.append(f"{label}_{field.upper()}_NONZERO")
    if speedup < args.minimum_speedup:
        reasons.append("DUAL_CLUSTER_THROUGHPUT_BELOW_THRESHOLD")
    report = {
        "schema_version": "hm3d-multicluster-probe-audit-v1",
        "passed": not reasons,
        "expand_to_four_clusters": not reasons,
        "single_decisions_per_wall_hour": single_rate,
        "dual_decisions_per_wall_hour_conservative": dual_rate,
        "dual_cluster_speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "reference_cluster_invariance": invariance,
        "reasons": sorted(set(reasons)),
        "statistical_warning": (
            "Parallel clusters on one HM3D asset are rollout accelerators, not independent maps. "
            "Aggregate starts within scene before averaging across scenes."
        ),
    }
    write_json_atomic(args.output.expanduser().resolve(), report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
