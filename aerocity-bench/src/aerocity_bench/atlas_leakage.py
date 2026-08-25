"""Grouped statistical shortcut probes for target-independent G2-I priors."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .canonical import content_hash
from .inspection_atlas import (
    MISSION_SECTOR_SCHEMA,
    validate_public_inspection_atlas,
    validate_public_mission_sector,
)

LEAKAGE_REPORT_SCHEMA = "org.aerocity.bench.g2-i-leakage-audit.v1"
_REGION_CLASSES = ("roof", "facade", "entrance", "rubble")
_ALTITUDE_BANDS = ("near_ground", "lower", "mid", "elevated", "highrise")


def _surface_point(cell: dict[str, Any]) -> tuple[float, float, float]:
    x_value, y_value, z_value = (float(value) for value in cell["surface_point"])
    return (x_value, y_value, z_value)


def _atlas_index(
    atlas: dict[str, Any],
) -> list[tuple[str, tuple[float, float, float], tuple[float, ...]]]:
    validate_public_inspection_atlas(atlas)
    graph_degree: Counter[str] = Counter()
    for edge in atlas["transit_graph"]["edges"]:
        graph_degree[str(edge["start_node_id"])] += 1
        graph_degree[str(edge["end_node_id"])] += 1
    degree_by_region = {
        str(node["region_id"]): graph_degree[str(node["node_id"])]
        for node in atlas["transit_graph"]["nodes"]
    }
    total_cells = sum(len(region["cells"]) for region in atlas["regions"])
    result = []
    for region in atlas["regions"]:
        region_class = str(region["region_class"])
        altitude_band = str(region["altitude_band"])
        feature = tuple(float(region_class == value) for value in _REGION_CLASSES) + tuple(
            float(altitude_band == value) for value in _ALTITUDE_BANDS
        ) + (
            math.log1p(float(region["represented_area_m2"])),
            math.log1p(len(region["cells"])),
            float(degree_by_region[str(region["region_id"])]),
            math.log1p(total_cells),
            math.log1p(len(atlas["regions"])),
        )
        for cell in region["cells"]:
            result.append((str(cell["cell_id"]), _surface_point(cell), feature))
    return result


def _nearest_feature(
    point: list[float],
    index: list[tuple[str, tuple[float, float, float], tuple[float, ...]]],
) -> tuple[str, tuple[float, ...]]:
    query = tuple(float(value) for value in point)
    if len(query) != 3 or not all(math.isfinite(value) for value in query):
        raise ValueError("private audit point must be a finite three-vector")
    _, cell_id, feature = min(
        (
            (math.dist(query, surface), cell_id, feature)
            for cell_id, surface, feature in index
        ),
        key=lambda item: item[0],
    )
    return cell_id, feature


def _standardize(
    train: list[tuple[float, ...]], test: list[tuple[float, ...]]
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
    dimensions = len(train[0])
    means = [sum(row[index] for row in train) / len(train) for index in range(dimensions)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in train) / len(train)
        scales.append(max(math.sqrt(variance), 1.0e-9))

    def transform(rows: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
        return [
            tuple(
                (row[index] - means[index]) / scales[index]
                for index in range(dimensions)
            )
            for row in rows
        ]

    return transform(train), transform(test)


def _grouped_binary_scores(
    features: list[tuple[float, ...]], labels: list[int], groups: list[str]
) -> list[float]:
    scores = [0.5] * len(features)
    for held_out in sorted(set(groups)):
        train_indices = [index for index, group in enumerate(groups) if group != held_out]
        test_indices = [index for index, group in enumerate(groups) if group == held_out]
        if not train_indices:
            continue
        train_features, test_features = _standardize(
            [features[index] for index in train_indices],
            [features[index] for index in test_indices],
        )
        k_value = min(7, len(train_indices))
        for test_index, test_feature in zip(test_indices, test_features, strict=True):
            neighbours = sorted(
                (
                    (math.dist(test_feature, train_feature), labels[train_index])
                    for train_index, train_feature in zip(
                        train_indices, train_features, strict=True
                    )
                ),
                key=lambda item: item[0],
            )[:k_value]
            scores[test_index] = sum(label for _, label in neighbours) / len(neighbours)
    return scores


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        raise ValueError("binary leakage probe requires both labels")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _grouped_multiclass_accuracy(
    features: list[tuple[float, ...]], labels: list[str], groups: list[str]
) -> float:
    correct = 0
    for held_out in sorted(set(groups)):
        train_indices = [index for index, group in enumerate(groups) if group != held_out]
        test_indices = [index for index, group in enumerate(groups) if group == held_out]
        train_features, test_features = _standardize(
            [features[index] for index in train_indices],
            [features[index] for index in test_indices],
        )
        k_value = min(7, len(train_indices))
        for test_index, test_feature in zip(test_indices, test_features, strict=True):
            neighbours = sorted(
                (
                    (math.dist(test_feature, train_feature), labels[train_index])
                    for train_index, train_feature in zip(
                        train_indices, train_features, strict=True
                    )
                ),
                key=lambda item: item[0],
            )[:k_value]
            counts = Counter(label for _, label in neighbours)
            predicted = min(
                (label for label, count in counts.items() if count == max(counts.values())),
                default="",
            )
            correct += int(predicted == labels[test_index])
    return correct / len(labels)


def _atlas_summary_feature(atlas: dict[str, Any]) -> tuple[float, ...]:
    region_classes = Counter(str(region["region_class"]) for region in atlas["regions"])
    altitude_bands = Counter(str(region["altitude_band"]) for region in atlas["regions"])
    cell_count = sum(len(region["cells"]) for region in atlas["regions"])
    area = sum(float(region["represented_area_m2"]) for region in atlas["regions"])
    return (
        *(float(region_classes[value]) for value in _REGION_CLASSES),
        *(float(altitude_bands[value]) for value in _ALTITUDE_BANDS),
        math.log1p(len(atlas["regions"])),
        math.log1p(cell_count),
        math.log1p(area),
        float(len(atlas["transit_graph"]["edges"])),
    )


def _multiclass_probe(
    features: list[tuple[float, ...]],
    labels: list[str],
    groups: list[str],
    *,
    permutation_count: int,
    seed_tag: str,
    permutation_scheme: str,
) -> dict[str, Any]:
    if len(set(groups)) < 3 or len(set(labels)) < 2 or len(labels) < 12:
        return {
            "status": "INSUFFICIENT_DATA",
            "sample_count": len(labels),
            "group_count": len(set(groups)),
            "class_count": len(set(labels)),
            "minimum_required_samples": 12,
        }
    accuracy = _grouped_multiclass_accuracy(features, labels, groups)
    majority = max(Counter(labels).values()) / len(labels)
    rng = random.Random(
        int(content_hash([seed_tag, len(labels), sorted(set(groups))])[:16], 16)
    )
    permutation_accuracies = []
    for _ in range(permutation_count):
        permuted = list(labels)
        if permutation_scheme == "within-layout-ancestor":
            for group in sorted(set(groups)):
                indices = [index for index, value in enumerate(groups) if value == group]
                group_labels = [labels[index] for index in indices]
                rng.shuffle(group_labels)
                for index, label in zip(indices, group_labels, strict=True):
                    permuted[index] = label
        elif permutation_scheme == "between-layout-ancestors":
            group_ids = sorted(set(groups))
            group_labels = []
            for group in group_ids:
                values = {
                    labels[index]
                    for index, value in enumerate(groups)
                    if value == group
                }
                if len(values) != 1:
                    raise ValueError(
                        "between-ancestor permutation requires one label per ancestor"
                    )
                group_labels.append(next(iter(values)))
            rng.shuffle(group_labels)
            label_by_group = dict(zip(group_ids, group_labels, strict=True))
            permuted = [label_by_group[group] for group in groups]
        else:
            raise ValueError("unsupported grouped permutation scheme")
        permutation_accuracies.append(
            _grouped_multiclass_accuracy(features, permuted, groups)
        )
    exceedances = sum(value >= accuracy - 1.0e-12 for value in permutation_accuracies)
    p_value = (exceedances + 1) / (permutation_count + 1)
    detected = accuracy > majority and p_value <= 0.05
    return {
        "status": "FAIL_DETECTED_SIGNAL" if detected else "PASS_NO_DETECTED_SIGNAL",
        "sample_count": len(labels),
        "group_count": len(set(groups)),
        "class_count": len(set(labels)),
        "grouped_accuracy": round(accuracy, 6),
        "majority_accuracy": round(majority, 6),
        "excess_accuracy": round(accuracy - majority, 6),
        "permutation_count": permutation_count,
        "permutation_scheme": permutation_scheme,
        "permutation_p_value": round(p_value, 6),
    }


def _sector_summary_feature(
    sector: dict[str, Any], atlas: dict[str, Any]
) -> tuple[tuple[float, ...], set[str], dict[str, dict[str, Any]]]:
    if sector.get("schema") != MISSION_SECTOR_SCHEMA:
        raise ValueError("mission sector schema differs")
    declared_hash = str(sector.get("sector_hash", ""))
    payload = {key: value for key, value in sector.items() if key != "sector_hash"}
    if content_hash(payload) != declared_hash:
        raise ValueError("mission-sector content hash mismatch")
    if (
        sector.get("atlas_hash") != atlas.get("atlas_hash")
        or sector.get("truth_independent") is not True
        or sector.get("frozen_before_sampling") is not True
    ):
        raise ValueError("mission sector is not target-independent or atlas-bound")
    selected_region_values = sector.get("selected_region_ids", [])
    selected_cell_values = sector.get("selected_cell_ids", [])
    if (
        not isinstance(selected_region_values, list)
        or not isinstance(selected_cell_values, list)
        or selected_region_values != sorted(selected_region_values)
        or selected_cell_values != sorted(selected_cell_values)
    ):
        raise ValueError("mission sector obligations must use canonical ordering")
    selected_regions = {str(value) for value in selected_region_values}
    selected_cells = {str(value) for value in selected_cell_values}
    region_lookup = {str(region["region_id"]): region for region in atlas["regions"]}
    cell_lookup = {
        str(cell["cell_id"]): (str(region["region_id"]), cell)
        for region in atlas["regions"]
        for cell in region["cells"]
    }
    if (
        not selected_regions
        or not selected_cells
        or selected_regions - set(region_lookup)
        or selected_cells - set(cell_lookup)
        or {cell_lookup[cell_id][0] for cell_id in selected_cells} != selected_regions
    ):
        raise ValueError("mission sector references invalid public obligations")
    assignment = sector.get("cell_assignment_by_drone")
    if not isinstance(assignment, dict) or not assignment:
        raise ValueError("mission sector cell assignment is missing")
    assigned_cells: list[str] = []
    for drone_id in sorted(assignment):
        cells = assignment[drone_id]
        if not isinstance(cells, list) or not cells or any(
            not isinstance(cell_id, str) for cell_id in cells
        ):
            raise ValueError("mission sector cell assignment is invalid")
        assigned_cells.extend(cells)
    if len(assigned_cells) != len(set(assigned_cells)) or set(assigned_cells) != selected_cells:
        raise ValueError("mission sector assignment does not cover selected cells")
    certificate = sector.get("capacity_certificate")
    if not isinstance(certificate, dict) or certificate.get("model") != (
        "public_grouped_safe_sky_scan_dwell_return_lower_bound"
    ):
        raise ValueError("mission sector lacks a public capacity certificate")
    limit = float(certificate.get("capacity_limit_s", 0.0))
    required = certificate.get("per_drone_required_lower_bound_s")
    if (
        limit <= 0.0
        or not math.isfinite(limit)
        or certificate.get("capacity_fraction") != 1.0
        or certificate.get("all_lower_bounds_fit") is not True
        or certificate.get("native_cf2x_validation_required") is not True
        or not isinstance(required, dict)
        or set(required) != set(assignment)
        or not required
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > limit
            for value in required.values()
        )
    ):
        raise ValueError("mission-sector capacity certificate is invalid")
    normalized = [float(value) / limit for value in required.values()]
    expected_area = sum(
        float(cell_lookup[cell_id][1]["represented_area_m2"])
        for cell_id in selected_cells
    )
    if not math.isclose(
        float(sector.get("represented_area_m2", -1.0)), expected_area, abs_tol=1.0e-5
    ):
        raise ValueError("mission sector represented area differs from selected cells")
    region_classes = Counter(
        str(region_lookup[cell_lookup[cell_id][0]]["region_class"]) for cell_id in selected_cells
    )
    altitude_bands = Counter(
        str(region_lookup[cell_lookup[cell_id][0]]["altitude_band"]) for cell_id in selected_cells
    )
    feature = (
        *(float(region_classes[value]) for value in _REGION_CLASSES),
        *(float(altitude_bands[value]) for value in _ALTITUDE_BANDS),
        math.log1p(len(selected_regions)),
        math.log1p(len(selected_cells)),
        math.log1p(float(sector["represented_area_m2"])),
        float(certificate["capacity_fraction"]),
        min(normalized),
        sum(normalized) / len(normalized),
        max(normalized),
    )
    return (
        feature,
        selected_cells,
        {region_id: region_lookup[region_id] for region_id in selected_regions},
    )


def _private_site_in_selected_regions(
    site: dict[str, Any], selected_regions: dict[str, dict[str, Any]]
) -> bool:
    class_mapping = {
        "roof": "roof",
        "facade": "facade_marker_site",
        "entrance": "entrance",
        "rubble": "rubble",
    }
    position = tuple(float(value) for value in site["position"])
    normal = tuple(float(value) for value in site["normal"])
    for region in selected_regions.values():
        if class_mapping.get(str(region["region_class"])) != site.get("support_class"):
            continue
        cells = region["cells"]
        region_normal = tuple(float(value) for value in cells[0]["surface_normal"])
        if sum(first * second for first, second in zip(normal, region_normal, strict=True)) < 0.999:
            continue
        lower = tuple(float(value) for value in region["bounds"]["minimum"])
        upper = tuple(float(value) for value in region["bounds"]["maximum"])
        if all(
            low - 0.25 <= value <= high + 0.25
            for value, low, high in zip(position, lower, upper, strict=True)
        ):
            return True
    return False


def audit_atlas_leakage(
    records: list[dict[str, Any]],
    *,
    execution_contract: dict[str, Any] | None = None,
    permutation_count: int = 64,
) -> dict[str, Any]:
    """Run grouped probes while keeping all authority labels inside the process."""

    if permutation_count < 16:
        raise ValueError("leakage audit requires at least 16 deterministic permutations")
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    groups: list[str] = []
    pair_indices: list[tuple[int, int]] = []
    summary_features: list[tuple[float, ...]] = []
    process_labels: list[str] = []
    split_labels: list[str] = []
    summary_groups: list[str] = []
    sector_features: list[tuple[float, ...]] = []
    sector_process_labels: list[str] = []
    sector_split_labels: list[str] = []
    sector_groups: list[str] = []
    sector_membership_checks = 0
    atlas_by_group: dict[str, str] = {}
    group_by_atlas: dict[str, str] = {}

    for record in records:
        atlas = record["atlas"]
        episode = record["private_episode"]
        validate_public_inspection_atlas(atlas)
        declared_episode_hash = str(episode.get("episode_hash", ""))
        episode_payload = {
            key: value for key, value in episode.items() if key != "episode_hash"
        }
        if content_hash(episode_payload) != declared_episode_hash:
            raise ValueError("private episode content hash mismatch")
        group = str(record["layout_ancestor"])
        if not group:
            raise ValueError("leakage records require a layout ancestor")
        atlas_hash = str(atlas["atlas_hash"])
        if group in atlas_by_group and atlas_by_group[group] != atlas_hash:
            raise ValueError("one layout ancestor maps to multiple public atlases")
        if atlas_hash in group_by_atlas and group_by_atlas[atlas_hash] != group:
            raise ValueError("one public atlas is claimed by multiple layout ancestors")
        atlas_by_group[group] = atlas_hash
        group_by_atlas[atlas_hash] = group
        targets = {str(item["site_id"]): item for item in episode["targets"]}
        distractors = {str(item["site_id"]): item for item in episode["distractors"]}
        pairs = episode["counterfactual_pairs"]
        if len(pairs) != len(targets) or len(pairs) != len(distractors):
            raise ValueError("counterfactual pair cardinality differs from private sites")
        atlas_cells = _atlas_index(atlas)
        sector = episode.get("mission_sector")
        selected_sector_cells: set[str] | None = None
        selected_sector_regions: dict[str, dict[str, Any]] | None = None
        if sector is not None:
            if not isinstance(sector, dict):
                raise ValueError("private episode mission sector must be an object")
            if execution_contract is None:
                raise ValueError("mission-sector leakage audit requires an execution contract")
            starts = episode.get("starts")
            if not isinstance(starts, list):
                raise ValueError("mission-sector leakage audit requires public starts")
            validate_public_mission_sector(sector, atlas, starts, execution_contract)
            (
                sector_feature,
                selected_sector_cells,
                selected_sector_regions,
            ) = _sector_summary_feature(sector, atlas)
            sector_features.append(sector_feature)
            sector_process_labels.append(str(episode["target_process"]))
            sector_split_labels.append(str(record["split_label"]))
            sector_groups.append(group)
        seen_targets: set[str] = set()
        seen_distractors: set[str] = set()
        for pair in pairs:
            target_site = str(pair["target_site_id"])
            distractor_site = str(pair["distractor_site_id"])
            if target_site not in targets or distractor_site not in distractors:
                raise ValueError("counterfactual pair references an unknown private site")
            first_index = len(features)
            _, target_feature = _nearest_feature(
                targets[target_site]["position"], atlas_cells
            )
            features.append(target_feature)
            labels.append(1)
            groups.append(group)
            second_index = len(features)
            _, distractor_feature = _nearest_feature(
                distractors[distractor_site]["position"], atlas_cells
            )
            features.append(distractor_feature)
            labels.append(0)
            groups.append(group)
            pair_indices.append((first_index, second_index))
            if selected_sector_cells is not None and selected_sector_regions is not None:
                if (
                    not _private_site_in_selected_regions(
                        targets[target_site], selected_sector_regions
                    )
                    or not _private_site_in_selected_regions(
                        distractors[distractor_site], selected_sector_regions
                    )
                ):
                    raise ValueError("private pair falls outside its frozen mission sector")
                sector_membership_checks += 1
            seen_targets.add(target_site)
            seen_distractors.add(distractor_site)
        if seen_targets != set(targets) or seen_distractors != set(distractors):
            raise ValueError("counterfactual pairs do not cover each private site exactly once")
        summary_features.append(_atlas_summary_feature(atlas))
        process_labels.append(str(episode["target_process"]))
        split_labels.append(str(record["split_label"]))
        summary_groups.append(group)

    group_count = len(set(groups))
    if group_count < 3 or len(pair_indices) < 20:
        paired_probe: dict[str, Any] = {
            "status": "INSUFFICIENT_DATA",
            "pair_count": len(pair_indices),
            "group_count": group_count,
            "minimum_required_pairs": 20,
            "minimum_required_groups": 3,
        }
    else:
        observed_auc = _auc(labels, _grouped_binary_scores(features, labels, groups))
        permutation_aucs = []
        seed = int(content_hash([len(records), len(pair_indices), sorted(set(groups))])[:16], 16)
        rng = random.Random(seed)
        for _ in range(permutation_count):
            permuted = list(labels)
            for first, second in pair_indices:
                if rng.random() < 0.5:
                    permuted[first], permuted[second] = permuted[second], permuted[first]
            permutation_aucs.append(
                _auc(permuted, _grouped_binary_scores(features, permuted, groups))
            )
        observed_delta = abs(observed_auc - 0.5)
        exceedances = sum(abs(value - 0.5) >= observed_delta for value in permutation_aucs)
        p_value = (exceedances + 1) / (permutation_count + 1)
        paired_probe = {
            "status": "FAIL_DETECTED_SIGNAL" if p_value <= 0.05 else "PASS_NO_DETECTED_SIGNAL",
            "pair_count": len(pair_indices),
            "group_count": group_count,
            "grouped_auc": round(observed_auc, 6),
            "absolute_auc_delta_from_chance": round(observed_delta, 6),
            "permutation_count": permutation_count,
            "permutation_p_value": round(p_value, 6),
            "counterfactual_swap_preserves_public_atlas": True,
        }

    report: dict[str, Any] = {
        "schema": LEAKAGE_REPORT_SCHEMA,
        "formal_score_eligible": False,
        "record_count": len(records),
        "paired_label_probe": paired_probe,
        "process_label_probe": _multiclass_probe(
            summary_features,
            process_labels,
            summary_groups,
            permutation_count=permutation_count,
            seed_tag="atlas-process",
            permutation_scheme="within-layout-ancestor",
        ),
        "split_label_probe": _multiclass_probe(
            summary_features,
            split_labels,
            summary_groups,
            permutation_count=permutation_count,
            seed_tag="atlas-split",
            permutation_scheme="between-layout-ancestors",
        ),
        "sector_process_label_probe": _multiclass_probe(
            sector_features,
            sector_process_labels,
            sector_groups,
            permutation_count=permutation_count,
            seed_tag="sector-process",
            permutation_scheme="within-layout-ancestor",
        ),
        "sector_split_label_probe": _multiclass_probe(
            sector_features,
            sector_split_labels,
            sector_groups,
            permutation_count=permutation_count,
            seed_tag="sector-split",
            permutation_scheme="between-layout-ancestors",
        ),
        "sector_contract": {
            "sector_record_count": len(sector_features),
            "target_distractor_pairs_checked_inside_sector": sector_membership_checks,
            "sector_features_exclude_private_truth": True,
        },
        "privacy_contract": {
            "contains_private_coordinates": False,
            "contains_private_instance_ids": False,
            "contains_label_names": False,
            "grouped_by_layout_ancestor": True,
        },
    }
    report["report_hash"] = content_hash(report)
    return report
