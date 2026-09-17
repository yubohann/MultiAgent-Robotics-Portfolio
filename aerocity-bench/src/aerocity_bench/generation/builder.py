"""Immutable release construction with deterministic rejection sampling."""

from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from aerocity_bench.core.canonical import write_json
from aerocity_bench.core.config import EXPECTED_SPLITS, ReleaseConfig
from aerocity_bench.core.errors import GenerationRejected
from aerocity_bench.generation.compiler import write_compiled_public
from aerocity_bench.generation.generator import generate_city
from aerocity_bench.generation.targets import derive_support_sites, sample_episode
from aerocity_bench.release.assets import load_asset_lock, stage_assets
from aerocity_bench.release.audit import (
    audit_city_candidate,
    build_layout_manifest,
    validate_release,
)

MAX_ATTEMPTS_PER_LAYOUT = 200


def _write_private(
    config: ReleaseConfig,
    city: dict[str, Any],
    private_dir: Path,
    episode_count: int,
) -> dict[str, int]:
    sites = derive_support_sites(city)
    write_json(
        private_dir / "support_sites.json",
        {
            "schema": "org.aerocity.bench.support-sites-private.v2",
            "layout_id": city["layout_id"],
            "support_site_count": len(sites),
            "support_sites": sites,
        },
    )
    target_total = 0
    for episode_index in range(episode_count):
        episode = sample_episode(config, city, sites, episode_index)
        target_total += int(episode["target_count"])
        write_json(private_dir / "episodes" / f"episode-{episode_index:04d}.json", episode)
    return {"support_site_count": len(sites), "target_instance_count": target_total}


def build_release(
    config: ReleaseConfig,
    asset_root: Path,
    output: Path,
    selected_splits: tuple[str, ...] = EXPECTED_SPLITS,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    unknown = sorted(set(selected_splits) - set(EXPECTED_SPLITS))
    if unknown or not selected_splits:
        raise ValueError(f"invalid selected splits: {unknown}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        visual = config.raw["visual_assets"]
        bundle = str(visual["bundle"])
        standard_assets = [str(value) for value in visual["standard"]]
        requested = set(standard_assets)
        lock = load_asset_lock(asset_root.resolve(), bundle, requested)
        stage_assets(lock, asset_root.resolve(), staging)
        layouts: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        seen_layouts: set[str] = set()
        seen_topology: set[str] = set()
        for split in selected_splits:
            asset_ids = standard_assets
            for index in range(config.count(split)):
                accepted_city: dict[str, Any] | None = None
                private_metrics: dict[str, int] | None = None
                for attempt in range(MAX_ATTEMPTS_PER_LAYOUT):
                    candidate_dir: Path | None = None
                    try:
                        candidate = generate_city(config, split, index, attempt, asset_ids)
                        audit_city_candidate(candidate, config.raw["admission"])
                        if candidate["layout_id"] in seen_layouts:
                            raise GenerationRejected("duplicate layout id")
                        if candidate["topology_signature"] in seen_topology:
                            raise GenerationRejected("duplicate audited topology signature")
                        # Generate private truth before admitting a layout, so impossible target
                        # configurations do not leave partial public artifacts.
                        candidate_dir = staging / "splits" / split / candidate["layout_id"]
                        private_dir = candidate_dir / "evaluator_private"
                        private_metrics = _write_private(
                            config, candidate, private_dir, config.episodes(split)
                        )
                        accepted_city = candidate
                        break
                    except GenerationRejected as exc:
                        if candidate_dir is not None and candidate_dir.exists():
                            shutil.rmtree(candidate_dir)
                        rejections.append(
                            {
                                "split": split,
                                "index": index,
                                "attempt": attempt,
                                "reason": str(exc),
                            }
                        )
                if accepted_city is None or private_metrics is None:
                    raise GenerationRejected(
                        f"failed to admit {split}[{index}] after {MAX_ATTEMPTS_PER_LAYOUT} attempts"
                    )
                city = accepted_city
                seen_layouts.add(str(city["layout_id"]))
                seen_topology.add(str(city["topology_signature"]))
                layout_dir = staging / "splits" / split / city["layout_id"]
                public_dir = layout_dir / "public"
                write_compiled_public(city, public_dir, lock)
                manifest = build_layout_manifest(public_dir, city)
                write_json(public_dir / "layout_manifest.json", manifest)
                layouts.append(
                    {
                        "split": split,
                        "layout_id": city["layout_id"],
                        "topology_signature": city["topology_signature"],
                        "asset_set_id": city["asset_set_id"],
                        "size_m": city["size_m"],
                        "family": city["family"],
                        "generation_seed": city["generation_seed"],
                        **private_metrics,
                    }
                )
        write_json(
            staging / "audit" / "rejections.json",
            {
                "schema": "org.aerocity.bench.rejections.v1",
                "rejection_count": len(rejections),
                "reasons": dict(sorted(Counter(item["reason"] for item in rejections).items())),
                "candidates": rejections,
            },
        )
        effective = dict(config.raw)
        effective["split_counts"] = {
            split: config.count(split) if split in selected_splits else 0
            for split in EXPECTED_SPLITS
        }
        index = {
            "schema": "org.aerocity.bench.release-index.v2",
            "release_version": config.version,
            "generator_version": config.generator_version,
            "release_config": config.config_id,
            "selected_splits": list(selected_splits),
            "effective_release_config": effective,
            "layouts": layouts,
            "scientific_status": "pilot_only",
            "native_isaac_gate": "not_run",
        }
        write_json(staging / "release_index.json", index)
        report = validate_release(staging)
        os.replace(staging, output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
