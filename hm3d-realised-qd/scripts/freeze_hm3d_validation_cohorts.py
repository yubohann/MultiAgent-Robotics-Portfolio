"""Freeze disjoint representative and QD-mechanism HM3D validation cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from screen_hm3d_p07_scene_candidates import (  # noqa: E402
    _write_new_json,
    build_stratified_admission_queue,
    installed_split_candidates,
)

SCHEMA_VERSION = "hm3d-validation-cohort-freeze-v1"
_PROXY_FIELDS = (
    "num_floors",
    "num_rooms",
    "navigation_complexity",
    "scene_clutter",
    "room_density_per_100_navigable_m2",
)


def _rank(values: list[dict[str, Any]], field: str) -> dict[str, float]:
    ordered = sorted(
        values,
        key=lambda row: (float(row.get(field) or 0.0), str(row["scene_id"])),
    )
    if len(ordered) == 1:
        return {str(ordered[0]["scene_id"]): 0.5}
    return {
        str(row["scene_id"]): index / (len(ordered) - 1)
        for index, row in enumerate(ordered)
    }


def _annotate_qd_opportunity_proxy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranks = {field: _rank(rows, field) for field in _PROXY_FIELDS}
    annotated: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        scene_id = str(row["scene_id"])
        vector = {field: ranks[field][scene_id] for field in _PROXY_FIELDS}
        score = (
            0.30 * vector["num_floors"]
            + 0.15 * vector["num_rooms"]
            + 0.25 * vector["navigation_complexity"]
            + 0.15 * vector["scene_clutter"]
            + 0.15 * vector["room_density_per_100_navigable_m2"]
        )
        row["qd_opportunity_metadata_proxy"] = round(score, 8)
        row["qd_opportunity_proxy_vector"] = {
            key: round(value, 8) for key, value in vector.items()
        }
        row["qd_opportunity_proxy_caveat"] = (
            "Hypothesis-targeting metadata only. It does not prove vertical free-flight, "
            "planned-realised gap, archive diversity, or a method advantage."
        )
        annotated.append(row)
    return annotated


def _selected(queue: dict[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("primary_queue")
    if not isinstance(rows, list):
        raise ValueError("stratified queue lacks primary rows")
    return rows


def _without(rows: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded_ids = {str(row["scene_id"]) for row in excluded}
    return [row for row in rows if str(row["scene_id"]) not in excluded_ids]


def _stable_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['scene_id']}".encode()).hexdigest(),
    )


def freeze_validation_cohorts(
    metadata_rows: list[dict[str, str]],
    val_glb_root: Path,
    *,
    main_holdout_size: int = 36,
    qd_high_size: int = 12,
    qd_low_size: int = 12,
    dev_size: int = 12,
    selection_seed: str = "hm3d-four-uav-validation-v1",
) -> dict[str, Any]:
    candidates = installed_split_candidates(metadata_rows, val_glb_root, split="val")
    requested = main_holdout_size + qd_high_size + qd_low_size + dev_size
    if requested >= len(candidates):
        raise ValueError("validation cohorts must leave frozen same-protocol reserves")

    main = build_stratified_admission_queue(
        candidates,
        cohort_size=main_holdout_size,
        selection_seed=f"{selection_seed}:main",
        minimum_per_floor_stratum=1,
    )
    remaining = _without(candidates, _selected(main))
    annotated = _annotate_qd_opportunity_proxy(remaining)
    ordered = sorted(
        annotated,
        key=lambda row: (-float(row["qd_opportunity_metadata_proxy"]), str(row["scene_id"])),
    )
    multi_floor = [row for row in ordered if int(row["num_floors"]) >= 2]
    single_floor = [row for row in reversed(ordered) if int(row["num_floors"]) == 1]
    if len(multi_floor) < qd_high_size or len(single_floor) < qd_low_size:
        raise ValueError("official validation split cannot fill QD high/low opportunity cohorts")

    high_pool_size = max(qd_high_size, math.ceil(len(multi_floor) / 2))
    high = build_stratified_admission_queue(
        multi_floor[:high_pool_size],
        cohort_size=qd_high_size,
        selection_seed=f"{selection_seed}:qd-high",
        minimum_per_floor_stratum=1,
    )
    remaining = _without(remaining, _selected(high))

    low_candidates = [
        row
        for row in _annotate_qd_opportunity_proxy(remaining)
        if int(row["num_floors"]) == 1
    ]
    low_candidates.sort(
        key=lambda row: (float(row["qd_opportunity_metadata_proxy"]), str(row["scene_id"]))
    )
    low_pool_size = max(qd_low_size, math.ceil(len(low_candidates) / 2))
    low = build_stratified_admission_queue(
        low_candidates[:low_pool_size],
        cohort_size=qd_low_size,
        selection_seed=f"{selection_seed}:qd-low",
        minimum_per_floor_stratum=1,
    )
    remaining = _without(remaining, _selected(low))

    dev = build_stratified_admission_queue(
        remaining,
        cohort_size=dev_size,
        selection_seed=f"{selection_seed}:dev",
        minimum_per_floor_stratum=1,
    )
    remaining = _without(remaining, _selected(dev))
    reserves = _stable_order(remaining, f"{selection_seed}:reserves")

    cohorts = {
        "main_representative_holdout": main,
        "qd_high_opportunity_holdout": high,
        "qd_low_opportunity_counterfactual": low,
        "validation_dev": dev,
    }
    all_selected = [row for queue in cohorts.values() for row in _selected(queue)]
    scene_ids = [str(row["scene_id"]) for row in all_selected]
    if len(scene_ids) != len(set(scene_ids)):
        raise AssertionError("validation cohorts overlap")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_METADATA_COHORTS_REQUIRE_FLIGHT_ADMISSION",
        "official_split": "val",
        "installed_scene_count": len(candidates),
        "selection_seed": selection_seed,
        "selection_contract": {
            "main_claim": (
                "The representative holdout estimates overall HM3D generalization; it is "
                "selected before and independently of the QD mechanism cohorts."
            ),
            "mechanism_claim": (
                "The paired QD cohorts test the preregistered interaction: realised-QD should "
                "help more when vertical, topological and execution-divergence opportunities exist."
            ),
            "flight_admission": (
                "Metadata only proposes queues. P03 must confirm connected flight volume, legal "
                "four-CF2X starts and high/low mechanism opportunity without method outcomes."
            ),
            "forbidden_selection_inputs": [
                "method reward",
                "method coverage or AUC",
                "planned-QD or realised-QD occupancy",
                "RL value estimate or checkpoint score",
                "baseline ranking",
                "manual trajectory preference after running a method",
            ],
        },
        "cohorts": cohorts,
        "same_protocol_reserves": reserves,
        "cohort_scene_ids": {
            name: [str(row["scene_id"]) for row in _selected(queue)]
            for name, queue in cohorts.items()
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--val-glb-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--main-holdout-size", type=int, default=36)
    parser.add_argument("--qd-high-size", type=int, default=12)
    parser.add_argument("--qd-low-size", type=int, default=12)
    parser.add_argument("--dev-size", type=int, default=12)
    parser.add_argument("--selection-seed", default="hm3d-four-uav-validation-v1")
    args = parser.parse_args()
    with args.metadata.expanduser().resolve().open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    payload = freeze_validation_cohorts(
        rows,
        args.val_glb_root.expanduser().resolve(),
        main_holdout_size=args.main_holdout_size,
        qd_high_size=args.qd_high_size,
        qd_low_size=args.qd_low_size,
        dev_size=args.dev_size,
        selection_seed=args.selection_seed,
    )
    payload["freeze_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_new_json(args.output.expanduser().resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cohort_sizes": {
                    key: len(value) for key, value in payload["cohort_scene_ids"].items()
                },
                "reserves": len(payload["same_protocol_reserves"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
