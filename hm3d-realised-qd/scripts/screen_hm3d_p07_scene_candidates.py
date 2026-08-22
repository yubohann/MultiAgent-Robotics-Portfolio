"""Rank official HM3D *train* scenes for later 3-D multi-UAV admission.

This is deliberately a cheap preliminary screen.  Official Matterport fields
describe ground-navigation annotations; neither their floor count nor their
area authorizes free flight.  A listed scene must still pass collision USD,
3-D ESDF, public sparse-range receiver placement, vertical free-flight
opportunity, control-boundary, and four-CF2X
runtime admission before it enters P07.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


def _positive_int(row: dict[str, str], name: str) -> int:
    value = int(row[name])
    if value < 0:
        raise ValueError(f"metadata has negative {name} for {row.get('scene')}")
    return value


def _positive_float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if value < 0.0:
        raise ValueError(f"metadata has negative {name} for {row.get('scene')}")
    return value


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "").strip()
    if not value:
        return None
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"metadata has negative {name} for {row.get('scene')}")
    return parsed


def _candidate_row(row: dict[str, str], glb: Path) -> dict[str, Any]:
    num_floors = _positive_int(row, "num_floors")
    num_rooms = _positive_int(row, "num_rooms")
    navigable_area_m2 = _positive_float(row, "navigable_area")
    floor_space_m2 = _positive_float(row, "floor_space")
    return {
        "scene_id": row["scene"],
        "official_split": row["split"],
        "source_glb": str(glb.resolve()),
        "source_glb_bytes": glb.stat().st_size,
        "num_floors": num_floors,
        "num_rooms": num_rooms,
        "navigable_area_m2": navigable_area_m2,
        "floor_space_m2": floor_space_m2,
        "navigation_complexity": _optional_float(row, "navigation_complexity"),
        "scene_clutter": _optional_float(row, "scene_clutter"),
        "overall_scene_quality": _optional_float(row, "overall_scene_quality"),
        "official_diversity_label": _optional_float(row, "diversity"),
        "floor_space_per_floor_m2": floor_space_m2 / max(num_floors, 1),
        "room_density_per_100_navigable_m2": 100.0 * num_rooms / max(navigable_area_m2, 1e-9),
        "navigable_to_floor_space_ratio": navigable_area_m2 / max(floor_space_m2, 1e-9),
        "metadata_caveat": (
            "Ground-navigation metadata only; flight volume, vertical connectivity, "
            "control margin and four-CF2X capacity remain unmeasured."
        ),
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["navigable_area_m2"]),
            -float(row["floor_space_m2"]),
            -int(row["num_rooms"]),
            str(row["scene_id"]),
        ),
    )


_FLOOR_STRATA = ("1", "2", "3", "4_plus")
_QUANTILE_LABELS = ("low", "medium", "high")
_DIVERSITY_FEATURES = (
    "num_floors",
    "num_rooms",
    "navigable_area_m2",
    "floor_space_m2",
    "navigation_complexity",
    "scene_clutter",
    "floor_space_per_floor_m2",
    "room_density_per_100_navigable_m2",
    "navigable_to_floor_space_ratio",
)


def _floor_stratum(num_floors: int) -> str:
    if num_floors <= 0:
        raise ValueError("floor count must be positive")
    return str(num_floors) if num_floors <= 3 else "4_plus"


def _annotate_quantile(rows: list[dict[str, Any]], field: str, output_field: str) -> None:
    valid = sorted(
        (row for row in rows if row.get(field) is not None),
        key=lambda row: (float(row[field]), str(row["scene_id"])),
    )
    for index, row in enumerate(valid):
        bucket = min(len(_QUANTILE_LABELS) - 1, index * len(_QUANTILE_LABELS) // len(valid))
        row[output_field] = _QUANTILE_LABELS[bucket]
    for row in rows:
        row.setdefault(output_field, "unknown")


def _percentile_ranks(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    valid = sorted(
        (row for row in rows if row.get(field) is not None),
        key=lambda row: (float(row[field]), str(row["scene_id"])),
    )
    ranks: dict[str, float] = {}
    index = 0
    while index < len(valid):
        end = index + 1
        value = float(valid[index][field])
        while end < len(valid) and float(valid[end][field]) == value:
            end += 1
        if len(valid) == 1:
            percentile = 0.5
        else:
            percentile = ((index + end - 1) / 2.0) / (len(valid) - 1)
        for row in valid[index:end]:
            ranks[str(row["scene_id"])] = percentile
        index = end
    return ranks


def _annotate_diversity_vectors(rows: list[dict[str, Any]]) -> None:
    for floor_stratum in _FLOOR_STRATA:
        floor_rows = [row for row in rows if row["floor_stratum"] == floor_stratum]
        if not floor_rows:
            continue
        feature_ranks = {
            feature: _percentile_ranks(floor_rows, feature) for feature in _DIVERSITY_FEATURES
        }
        for row in floor_rows:
            scene_id = str(row["scene_id"])
            row["metadata_diversity_vector"] = {
                feature: round(feature_ranks[feature].get(scene_id, 0.5), 8)
                for feature in _DIVERSITY_FEATURES
            }


def _metadata_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_vector = left["metadata_diversity_vector"]
    right_vector = right["metadata_diversity_vector"]
    squared = [
        (float(left_vector[feature]) - float(right_vector[feature])) ** 2
        for feature in _DIVERSITY_FEATURES
    ]
    return math.sqrt(sum(squared) / len(squared))


def _annotate_sampling_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = [dict(row) for row in rows]
    for row in annotated:
        row["floor_stratum"] = _floor_stratum(int(row["num_floors"]))
    for floor_stratum in _FLOOR_STRATA:
        floor_rows = [row for row in annotated if row["floor_stratum"] == floor_stratum]
        if not floor_rows:
            continue
        _annotate_quantile(floor_rows, "navigable_area_m2", "area_quantile_within_floor")
        _annotate_quantile(
            floor_rows,
            "navigation_complexity",
            "complexity_quantile_within_floor",
        )
    _annotate_diversity_vectors(annotated)
    return annotated


def _allocate_floor_quotas(
    counts: dict[str, int], cohort_size: int, minimum_per_floor_stratum: int
) -> dict[str, int]:
    if cohort_size < 1 or cohort_size > sum(counts.values()):
        raise ValueError("training cohort size must fit the installed official train split")
    if minimum_per_floor_stratum < 1:
        raise ValueError("minimum per floor stratum must be positive")
    nonempty = [name for name in _FLOOR_STRATA if counts.get(name, 0) > 0]
    if cohort_size < len(nonempty):
        raise ValueError("training cohort is too small to represent every installed floor stratum")

    quotas = {name: 0 for name in nonempty}
    for _ in range(minimum_per_floor_stratum):
        for name in nonempty:
            if sum(quotas.values()) >= cohort_size:
                break
            if quotas[name] < counts[name]:
                quotas[name] += 1

    total = float(sum(counts[name] for name in nonempty))
    while sum(quotas.values()) < cohort_size:
        available = [name for name in nonempty if quotas[name] < counts[name]]
        if not available:
            raise ValueError("could not allocate the requested training cohort")
        name = max(
            available,
            key=lambda candidate: (
                cohort_size * counts[candidate] / total - quotas[candidate],
                counts[candidate] - quotas[candidate],
                -_FLOOR_STRATA.index(candidate),
            ),
        )
        quotas[name] += 1
    return quotas


def _stable_scene_order(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['scene_id']}".encode()).hexdigest(),
    )


def _round_robin_stratified_queue(
    rows: list[dict[str, Any]], quota: int, seed: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cell = (
            str(row["area_quantile_within_floor"]),
            str(row["complexity_quantile_within_floor"]),
        )
        cells.setdefault(cell, []).append(row)
    ordered_cells = {
        cell: _stable_scene_order(cell_rows, f"{seed}:{cell[0]}:{cell[1]}")
        for cell, cell_rows in sorted(cells.items())
    }
    selected: list[dict[str, Any]] = []
    area_counts = {label: 0 for label in (*_QUANTILE_LABELS, "unknown")}
    complexity_counts = {label: 0 for label in (*_QUANTILE_LABELS, "unknown")}
    cell_counts = {cell: 0 for cell in ordered_cells}
    cell_ties = {
        cell: hashlib.sha256(f"{seed}\0{cell[0]}\0{cell[1]}".encode()).hexdigest()
        for cell in ordered_cells
    }
    while len(selected) < quota:
        available = [cell for cell, cell_rows in ordered_cells.items() if cell_rows]
        if not available:
            raise ValueError("stratified queue exhausted before satisfying its quota")
        cell = min(
            available,
            key=lambda candidate: (
                cell_counts[candidate],
                area_counts[candidate[0]] + complexity_counts[candidate[1]],
                area_counts[candidate[0]],
                complexity_counts[candidate[1]],
                cell_ties[candidate],
            ),
        )
        if not selected:
            chosen_index = 0
        else:
            chosen_index = max(
                range(len(ordered_cells[cell])),
                key=lambda index: (
                    min(
                        _metadata_distance(ordered_cells[cell][index], existing)
                        for existing in selected
                    ),
                    hashlib.sha256(
                        f"{seed}\0{ordered_cells[cell][index]['scene_id']}".encode()
                    ).hexdigest(),
                ),
            )
        selected.append(ordered_cells[cell].pop(chosen_index))
        cell_counts[cell] += 1
        area_counts[cell[0]] += 1
        complexity_counts[cell[1]] += 1
    reserves = [row for cell in sorted(ordered_cells) for row in ordered_cells[cell]]
    return selected, reserves


def _diversity_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_floor: dict[str, Any] = {}
    duplicate_pairs = 0
    all_distances: list[float] = []
    for floor_stratum in _FLOOR_STRATA:
        floor_rows = [row for row in rows if row["floor_stratum"] == floor_stratum]
        distances = [
            _metadata_distance(floor_rows[left], floor_rows[right])
            for left in range(len(floor_rows))
            for right in range(left + 1, len(floor_rows))
        ]
        duplicates = sum(distance <= 1e-12 for distance in distances)
        duplicate_pairs += duplicates
        all_distances.extend(distances)
        per_floor[floor_stratum] = {
            "scene_count": len(floor_rows),
            "pair_count": len(distances),
            "exact_duplicate_vector_pairs": duplicates,
            "minimum_pairwise_distance": min(distances) if distances else None,
            "median_pairwise_distance": statistics.median(distances) if distances else None,
            "mean_pairwise_distance": statistics.fmean(distances) if distances else None,
        }
    return {
        "distance_definition": (
            "RMS distance over within-floor percentile ranks of method-independent HM3D "
            "metadata features; this is a diversity audit, not flight-admission evidence."
        ),
        "feature_axes": list(_DIVERSITY_FEATURES),
        "exact_duplicate_vector_pairs": duplicate_pairs,
        "minimum_within_floor_pairwise_distance": min(all_distances) if all_distances else None,
        "median_within_floor_pairwise_distance": statistics.median(all_distances)
        if all_distances
        else None,
        "mean_within_floor_pairwise_distance": statistics.fmean(all_distances)
        if all_distances
        else None,
        "per_floor_stratum": per_floor,
    }


def _build_training_admission_queue(
    rows: list[dict[str, Any]],
    *,
    cohort_size: int,
    selection_seed: str,
    minimum_per_floor_stratum: int,
) -> dict[str, Any]:
    annotated = _annotate_sampling_strata(rows)
    counts = {
        name: sum(row["floor_stratum"] == name for row in annotated) for name in _FLOOR_STRATA
    }
    quotas = _allocate_floor_quotas(counts, cohort_size, minimum_per_floor_stratum)
    selected: list[dict[str, Any]] = []
    reserves: list[dict[str, Any]] = []
    for floor_stratum in _FLOOR_STRATA:
        floor_rows = [row for row in annotated if row["floor_stratum"] == floor_stratum]
        if not floor_rows:
            continue
        floor_selected, floor_reserves = _round_robin_stratified_queue(
            floor_rows,
            quotas[floor_stratum],
            f"{selection_seed}:{floor_stratum}",
        )
        selected.extend(floor_selected)
        reserves.extend(floor_reserves)
    for index, row in enumerate(selected):
        row["metadata_admission_rank"] = index
    for index, row in enumerate(reserves):
        row["metadata_reserve_rank"] = index
    return {
        "status": "METADATA_STRATIFIED_QUEUE_REQUIRES_FLIGHT_ADMISSION",
        "selection_seed": selection_seed,
        "requested_scene_count": cohort_size,
        "floor_population_counts": counts,
        "floor_selection_quotas": quotas,
        "stratification_axes": [
            "official floor-count stratum: 1, 2, 3, or 4_plus",
            "navigable-area tertile within floor stratum",
            "navigation-complexity tertile within floor stratum",
            "deterministic maximin spread over method-independent metadata features",
        ],
        "primary_queue": selected,
        "same_protocol_reserves": reserves,
        "metadata_diversity_audit": _diversity_audit(selected),
        "replacement_rule": (
            "A flight-ineligible primary scene may only be replaced by the earliest frozen "
            "reserve from the same floor, area-quantile and complexity-quantile cell."
        ),
        "forbidden_selection_inputs": [
            "method reward",
            "method coverage",
            "QD occupancy",
            "RL value estimate",
            "baseline ranking",
        ],
    }


def build_stratified_admission_queue(
    rows: list[dict[str, Any]],
    *,
    cohort_size: int,
    selection_seed: str,
    minimum_per_floor_stratum: int,
) -> dict[str, Any]:
    """Build a deterministic method-independent queue for one official split."""

    return _build_training_admission_queue(
        rows,
        cohort_size=cohort_size,
        selection_seed=selection_seed,
        minimum_per_floor_stratum=minimum_per_floor_stratum,
    )


def installed_split_candidates(
    metadata_rows: list[dict[str, str]], glb_root: Path, *, split: str
) -> list[dict[str, Any]]:
    """Bind official metadata rows to exact locally installed scene assets."""

    if split not in {"train", "val"}:
        raise ValueError("only official HM3D train or val splits are supported")
    if not glb_root.is_dir():
        raise FileNotFoundError(glb_root)
    candidates: list[dict[str, Any]] = []
    for row in metadata_rows:
        if row.get("split") != split:
            continue
        scene_id = row.get("scene", "")
        scene_token = scene_id.split("-", 1)
        if len(scene_token) != 2 or not scene_token[1]:
            raise ValueError(f"invalid official scene ID: {scene_id!r}")
        glb = glb_root / scene_id / f"{scene_token[1]}.glb"
        if glb.is_file():
            candidates.append(_candidate_row(row, glb))
    if not candidates:
        raise ValueError(f"no official {split} metadata row matches an installed GLB")
    return candidates


def screen_train_candidates(
    metadata_rows: list[dict[str, str]],
    train_glb_root: Path,
    *,
    minimum_room_count: int,
    first_admission_max_navigable_area_m2: float,
    shortlist_size: int,
    training_cohort_size: int | None = None,
    selection_seed: str = "hm3d-four-uav-training-v1",
    minimum_per_floor_stratum: int = 8,
) -> dict[str, Any]:
    """Create a metadata-only shortlist from exact locally installed train GLBs."""

    if minimum_room_count < 1 or shortlist_size < 1:
        raise ValueError("minimum room count and shortlist size must be positive")
    if first_admission_max_navigable_area_m2 <= 0.0:
        raise ValueError("first-admission area cap must be positive")
    candidates = installed_split_candidates(metadata_rows, train_glb_root, split="train")

    single_floor = _rank([row for row in candidates if row["num_floors"] == 1])
    multi_floor = _rank([row for row in candidates if row["num_floors"] >= 2])
    if not single_floor or not multi_floor:
        raise ValueError("installed train assets lack required single- or multi-floor candidates")
    practical_single = [row for row in single_floor if row["num_rooms"] >= minimum_room_count]
    practical_multi = [
        row
        for row in multi_floor
        if row["num_rooms"] >= minimum_room_count
        and row["navigable_area_m2"] <= first_admission_max_navigable_area_m2
    ]
    if not practical_single or not practical_multi:
        raise ValueError("screening criteria leave no practical first runtime candidate")

    report = {
        "schema_version": "hm3d-p07-candidate-screen-v2",
        "status": "METADATA_ONLY_PRELIMINARY_SCREEN_NOT_FLIGHT_ADMISSION",
        "source_scope": "official_hm3d_v0.2_train_only",
        "installed_train_scene_count": len(candidates),
        "selection_contract": {
            "minimum_room_count": minimum_room_count,
            "first_admission_max_navigable_area_m2": first_admission_max_navigable_area_m2,
            "shortlist_size": shortlist_size,
            "training_cohort_size": training_cohort_size,
            "selection_seed": selection_seed,
            "minimum_per_floor_stratum": minimum_per_floor_stratum,
            "forbidden_inference": [
                "floor count is not vertical free-flight evidence",
                "navigable area is not a scene side length",
                "metadata admission is not four-CF2X capacity evidence",
            ],
        },
        "next_runtime_admission_order": {
            "single_floor_controller_calibration": practical_single[0],
            "multi_floor_primary": practical_multi[0],
            "largest_multi_floor_scale_stress": multi_floor[0],
        },
        "single_floor_shortlist": single_floor[:shortlist_size],
        "multi_floor_shortlist": multi_floor[:shortlist_size],
        "required_before_p07": [
            "collision USD and A-B-A reset",
            "3-D ESDF with control-boundary margin",
            "public sparse-range receiver and vertical free-flight admission",
            "four legal, separated CF2X start poses and simultaneous execution",
        ],
    }
    if training_cohort_size is not None:
        report["metadata_training_admission_queue"] = _build_training_admission_queue(
            candidates,
            cohort_size=training_cohort_size,
            selection_seed=selection_seed,
            minimum_per_floor_stratum=minimum_per_floor_stratum,
        )
    return report


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train-glb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-room-count", type=int, default=12)
    parser.add_argument("--first-admission-max-navigable-area-m2", type=float, default=750.0)
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--training-cohort-size", type=int, default=145)
    parser.add_argument("--selection-seed", default="hm3d-four-uav-training-v1")
    parser.add_argument("--minimum-per-floor-stratum", type=int, default=8)
    args = parser.parse_args()
    with args.metadata.expanduser().resolve().open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    payload = screen_train_candidates(
        rows,
        args.train_glb_root.expanduser().resolve(),
        minimum_room_count=args.minimum_room_count,
        first_admission_max_navigable_area_m2=args.first_admission_max_navigable_area_m2,
        shortlist_size=args.shortlist_size,
        training_cohort_size=args.training_cohort_size,
        selection_seed=args.selection_seed,
        minimum_per_floor_stratum=args.minimum_per_floor_stratum,
    )
    _write_new_json(args.output.expanduser().resolve(), payload)
    admission_order = payload["next_runtime_admission_order"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "multi_floor_primary": admission_order["multi_floor_primary"]["scene_id"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
