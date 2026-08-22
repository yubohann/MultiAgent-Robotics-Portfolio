"""Resumable, private-safe static admission audit for one development layout.

This module deliberately audits one layout at a time.  Support-site witness
compilation is more expensive than raw CitySpec generation; a per-layout report
keeps the future 10--20 city quality review observable and recoverable instead
of hiding host pressure in one opaque batch process.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .canonical import content_hash
from .errors import GenerationRejected
from .generator_v3 import generate_city_v3
from .geometry import AABB
from .ordinary_config import FORMAL_SPLITS, OrdinaryReleaseConfig
from .targets_v3 import derive_support_sites_v3, sample_episode_v3

SCENE_AUDIT_SCHEMA = "org.aerocity.bench.scene-audit.ordinary.v1"
SCENE_AUDIT_COHORT_SCHEMA = "org.aerocity.bench.scene-audit-cohort.ordinary.v1"
DEFAULT_DEVELOPMENT_AUDIT_ATTEMPTS = 8
DEVELOPMENT_AUDIT_SPLITS = ("train", "validation", "calibration")


def _positive_volume_overlap(first: AABB, second: AABB) -> bool:
    """Treat face/edge contact as legal and only reject positive volume."""

    return all(
        min(first_high, second_high) - max(first_low, second_low) > 0.0
        for first_low, first_high, second_low, second_high in zip(
            first.minimum,
            first.maximum,
            second.minimum,
            second.maximum,
            strict=True,
        )
    )


def _component_boxes(city: dict[str, Any]) -> list[AABB]:
    return [
        AABB.from_center_size(
            f"{building['id']}/{component['id']}",
            component["center"],
            component["size"],
            "building",
        )
        for building in city["buildings"]
        for component in building["components"]
    ]


def audit_generated_city(
    config: OrdinaryReleaseConfig, city: dict[str, Any]
) -> dict[str, Any]:
    """Audit a generated development city without exposing target truth.

    The report intentionally contains counts, layout/task hashes, and failure
    categories only.  It never serializes target positions, support-site IDs,
    distractors, target-process labels, or legal witness poses.
    """

    split = str(city["split"])
    if split in FORMAL_SPLITS:
        raise ValueError("scene audit must not sample a formal split")

    errors: list[str] = []
    boxes = _component_boxes(city)
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            if _positive_volume_overlap(first, second):
                errors.append("building_component_positive_volume_overlap")

    for obstacle in city["obstacles"]:
        obstacle_box = AABB.from_center_size(
            str(obstacle["id"]), obstacle["center"], obstacle["size"], "obstacle"
        )
        if any(_positive_volume_overlap(obstacle_box, component) for component in boxes):
            errors.append("semantic_obstacle_positive_volume_overlap")

    non_support_components = {
        f"{building['id']}/{component['id']}"
        for building in city["buildings"]
        for component in building["components"]
        if component.get("target_support", True) is not True
    }
    structural_counts = Counter(
        str(component["structural_role"])
        for building in city["buildings"]
        for component in building["components"]
        if component.get("structural_role") is not None
    )

    support_sites = derive_support_sites_v3(city, config)
    if not support_sites:
        errors.append("no_legal_support_sites")
    if any(site["owner_collider_id"] in non_support_components for site in support_sites):
        errors.append("target_support_false_component_leaked_into_support_sites")
    if any(int(site["legal_witness_count"]) < 1 for site in support_sites):
        errors.append("support_site_without_legal_witness")
    if any(int(site["surrounding_collider_count"]) < 1 for site in support_sites):
        errors.append("support_site_without_structural_context")

    # Sampling every development episode checks that the complete task contract
    # remains executable.  The report must not retain *any* target or support
    # cardinality: target totals and legal-site counts are evaluator-private
    # task information, even when no coordinates are written.
    episode_count = config.episodes(split)
    for episode_index in range(episode_count):
        episode = sample_episode_v3(config, city, support_sites, episode_index)
        if str(episode["layout_hash"]) != str(city["layout_hash"]):
            errors.append("episode_layout_hash_mismatch")
        for target in episode["targets"]:
            if target["owner_collider_id"] in non_support_components:
                errors.append("target_support_false_component_leaked_into_targets")
            if int(target["legal_witness_count"]) < 1:
                errors.append("target_without_legal_witness")

    report = {
        "schema": SCENE_AUDIT_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "split": split,
        "layout_id": str(city["layout_id"]),
        "layout_hash": str(city["layout_hash"]),
        "task_geometry_hash": str(city["task_geometry_hash"]),
        "generator_version": str(city["generator_version"]),
        "scene_counts": {
            "buildings": len(city["buildings"]),
            "building_colliders": len(boxes),
            "semantic_obstacles": len(city["obstacles"]),
            "visual_decorations": len(city["decorations"]),
            "structural_details": dict(sorted(structural_counts.items())),
            "episodes": episode_count,
        },
        # A city supplied directly to this function has already been accepted
        # by its caller.  The resumable generator below replaces this with the
        # actual retry count, keeping the public audit schema stable without
        # exposing individual admission failures.
        "generation_rejections_before_acceptance": 0,
        "error_categories": sorted(set(errors)),
    }
    report["report_hash"] = content_hash(report)
    return report


def audit_development_layout(
    config: OrdinaryReleaseConfig,
    split: str,
    index: int,
    asset_ids: list[str],
    *,
    max_attempts: int = DEFAULT_DEVELOPMENT_AUDIT_ATTEMPTS,
) -> dict[str, Any]:
    """Generate and audit one development layout with bounded retry evidence."""

    if split not in {"train", "validation", "calibration"}:
        raise ValueError("scene audit accepts only train, validation, or calibration")
    if index < 0:
        raise ValueError("index must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    # Generation rejection text is intentionally not reported.  It can contain
    # scene-specific observability/admission details that a user should not be
    # able to mine from a supposedly private-safe audit receipt.  The bounded
    # retry count still makes host cost and admission instability observable.
    rejection_count = 0
    for attempt in range(max_attempts):
        try:
            city = generate_city_v3(config, split, index, attempt, asset_ids)
            report = audit_generated_city(config, city)
        except GenerationRejected:
            rejection_count += 1
            continue
        report["attempt"] = attempt
        report["generation_rejections_before_acceptance"] = rejection_count
        report["report_hash"] = content_hash(
            {key: value for key, value in report.items() if key != "report_hash"}
        )
        return report

    report = {
        "schema": SCENE_AUDIT_SCHEMA,
        "status": "FAIL",
        "split": split,
        "index": index,
        "max_attempts": max_attempts,
        "error_categories": ["no_complete_city_episode_candidate"],
        "generation_rejection_count": rejection_count,
    }
    report["report_hash"] = content_hash(report)
    return report


def development_scene_audit_plan(per_split: int) -> tuple[tuple[str, int], ...]:
    """Return a deterministic, balanced development-only audit cohort.

    The procedural generator has an unbounded deterministic index space.  This
    sampling plan audits generator quality; it does not expand a release split
    or make any result eligible for formal scoring.
    """

    if per_split < 1:
        raise ValueError("per_split must be positive")
    return tuple(
        (split, index)
        for split in DEVELOPMENT_AUDIT_SPLITS
        for index in range(per_split)
    )


def _cohort_receipt_view(
    split: str,
    index: int,
    report: object,
    *,
    generator_version: str,
) -> dict[str, Any]:
    """Validate a cached private-safe receipt and retain its public fields."""

    if not isinstance(report, dict):
        raise ValueError(f"scene audit receipt is not an object: {split}/{index}")
    expected_hash = report.get("report_hash")
    unhashed = {key: value for key, value in report.items() if key != "report_hash"}
    if not isinstance(expected_hash, str) or expected_hash != content_hash(unhashed):
        raise ValueError(f"scene audit receipt hash mismatch: {split}/{index}")
    if report.get("schema") != SCENE_AUDIT_SCHEMA:
        raise ValueError(f"scene audit receipt schema mismatch: {split}/{index}")
    if report.get("split") != split:
        raise ValueError(f"scene audit receipt split mismatch: {split}/{index}")

    if report["status"] not in {"PASS", "FAIL"}:
        raise ValueError(f"scene audit status is invalid: {split}/{index}")

    if report["status"] == "FAIL":
        required = {
            "status",
            "error_categories",
            "generation_rejection_count",
            "report_hash",
        }
        if not required.issubset(report):
            raise ValueError(f"failed scene audit receipt is incomplete: {split}/{index}")
        return {
            "split": split,
            "index": index,
            "status": "FAIL",
            "layout_id": None,
            "layout_hash": None,
            "task_geometry_hash": None,
            "generator_version": generator_version,
            "scene_counts": None,
            "generation_rejections_before_acceptance": report[
                "generation_rejection_count"
            ],
            "error_categories": report["error_categories"],
            "scene_audit_report_hash": report["report_hash"],
        }

    required = {
        "status",
        "layout_id",
        "layout_hash",
        "task_geometry_hash",
        "generator_version",
        "scene_counts",
        "generation_rejections_before_acceptance",
        "error_categories",
        "report_hash",
    }
    if not required.issubset(report):
        raise ValueError(f"scene audit receipt is incomplete: {split}/{index}")
    if report["generator_version"] != generator_version:
        raise ValueError(f"scene audit generator mismatch: {split}/{index}")

    # This is deliberately an allow-list.  The cohort summary cannot acquire a
    # private target field merely because a future individual receipt grows.
    return {
        "split": split,
        "index": index,
        "status": report["status"],
        "layout_id": report["layout_id"],
        "layout_hash": report["layout_hash"],
        "task_geometry_hash": report["task_geometry_hash"],
        "generator_version": report["generator_version"],
        "scene_counts": report["scene_counts"],
        "generation_rejections_before_acceptance": report[
            "generation_rejections_before_acceptance"
        ],
        "error_categories": report["error_categories"],
        "scene_audit_report_hash": report["report_hash"],
    }


def summarize_development_scene_audit_cohort(
    config: OrdinaryReleaseConfig,
    receipts: dict[tuple[str, int], object],
    *,
    per_split: int,
) -> dict[str, Any]:
    """Create a fail-closed summary for a stratified generator-quality audit."""

    plan = development_scene_audit_plan(per_split)
    expected = set(plan)
    supplied = set(receipts)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            f"scene audit cohort members differ; missing={missing}, unexpected={unexpected}"
        )

    members = [
        _cohort_receipt_view(
            split,
            index,
            receipts[(split, index)],
            generator_version=config.generator_version,
        )
        for split, index in plan
    ]
    layout_hashes = [
        str(member["layout_hash"])
        for member in members
        if member["layout_hash"] is not None
    ]
    duplicate_layout_hashes = sorted(
        layout_hash
        for layout_hash, count in Counter(layout_hashes).items()
        if count > 1
    )
    failed_members = [
        {
            "split": member["split"],
            "index": member["index"],
            "error_categories": member["error_categories"],
        }
        for member in members
        if member["status"] != "PASS"
    ]
    report: dict[str, Any] = {
        "schema": SCENE_AUDIT_COHORT_SCHEMA,
        "status": "PASS" if not failed_members and not duplicate_layout_hashes else "FAIL",
        "formal_score_eligible": False,
        "scope": "development_generator_scene_admission_only",
        "release_config_hash": config.config_hash,
        "generator_version": config.generator_version,
        "sampling": {
            "splits": list(DEVELOPMENT_AUDIT_SPLITS),
            "layouts_per_split": per_split,
            "layout_count": len(members),
            "selection": "deterministic split/index cartesian cohort; not a release expansion",
            "configured_release_layout_counts": {
                split: config.count(split) for split in DEVELOPMENT_AUDIT_SPLITS
            },
        },
        "all_layout_hashes_unique": not duplicate_layout_hashes,
        "duplicate_layout_hashes": duplicate_layout_hashes,
        "failed_members": failed_members,
        "members": members,
    }
    report["report_hash"] = content_hash(report)
    return report
