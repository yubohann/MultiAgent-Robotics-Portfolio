"""Materialize one development-only G2-I layout for native CF2X replay.

This is deliberately separate from the official release builder.  It binds a
development CitySpec and one private episode to the public G2-I task contract,
but marks the resulting layout ineligible for formal scoring.  The tool never
chooses targets or witnesses; those remain in the supplied private episode.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

BENCH_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = BENCH_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# ruff: noqa: E402
from aerocity_bench.assets import stage_assets
from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec, write_compiled_public_v3
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.public_boundary import audit_public_layout
from aerocity_bench.supply_chain import load_official_cc0_lock
from aerocity_bench.targets_v3 import (
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
    validate_frozen_g2_i_episode,
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--development-split",
        choices=("calibration", "train", "validation"),
        default=None,
        help=(
            "required when the input is a staged public CitySpec, which deliberately "
            "does not carry its split label"
        ),
    )
    parser.add_argument(
        "--private-episode",
        type=Path,
        default=None,
        help="frozen evaluator-private calibration episode; never exposed to methods",
    )
    parser.add_argument(
        "--prior-level", choices=("full-cells", "coarse-regions"), default="full-cells"
    )
    return parser.parse_args(argv)


def materialize(
    city_path: Path,
    release_config_path: Path,
    asset_root: Path,
    output: Path,
    *,
    episode_index: int,
    prior_level: str,
    private_episode_path: Path | None = None,
    development_split: str | None = None,
) -> dict[str, Any]:
    if episode_index < 0:
        raise ValueError("episode-index must be non-negative")
    city = read_json(city_path.resolve())
    config = load_ordinary_config(release_config_path.resolve())
    if not isinstance(city, dict) or not city.get("layout_id"):
        raise ValueError("city must be a valid CitySpec")
    city_split = city.get("split")
    if city_split is not None and not isinstance(city_split, str):
        raise ValueError("CitySpec split must be a string when present")
    if development_split is not None and development_split in FORMAL_SPLITS:
        raise ValueError("native G2-I development layout cannot consume a formal split")
    if city_split is not None and development_split is not None and city_split != development_split:
        raise ValueError("explicit development split differs from CitySpec split")
    split = city_split if isinstance(city_split, str) else development_split
    if not isinstance(split, str) or not split:
        raise ValueError(
            "staged public CitySpec has no split; supply --development-split "
            "for a development replay"
        )
    if split in {"test_iid", "test_topology", "test_process_ood"}:
        raise ValueError("native G2-I development layout cannot consume a formal split")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    # A staged public CitySpec deliberately omits its development split.  The
    # private deterministic episode sampler still needs that label to select
    # the configured target process; keep this reconstructed copy in-process
    # and continue emitting the original public CitySpec unchanged.
    episode_city = dict(city)
    episode_city["split"] = split
    output.mkdir(parents=True)
    try:
        allowlist = [str(value) for value in config.raw["assets"]["allowlist"]]
        lock, _, _ = load_official_cc0_lock(
            asset_root.resolve(), str(config.raw["assets"]["bundle"]), allowlist
        )
        authority_task = compile_g2_i_task_spec(
            city,
            config.raw["execution_contract"],
            config.raw["fleet"],
        )
        task = (
            authority_task
            if prior_level == "full-cells"
            else compile_g2_i_task_spec(
                city,
                config.raw["execution_contract"],
                config.raw["fleet"],
                inspection_prior_level=prior_level,
            )
        )
        support_sites = derive_support_sites_v3(city, config)
        if private_episode_path is None:
            private_episode = sample_episode_v3(
                config,
                episode_city,
                support_sites,
                episode_index,
                public_task_spec=authority_task,
            )
            episode_source = "development-resample"
        else:
            private_episode = read_json(private_episode_path.resolve())
            if not isinstance(private_episode, dict):
                raise ValueError("frozen private episode must be an object")
            validate_frozen_g2_i_episode(
                private_episode,
                city,
                authority_task,
                config.raw["execution_contract"],
            )
            episode_source = "frozen-calibration-input"
        staged_assets = stage_assets(lock, asset_root.resolve(), output)
        layout_relative_root = Path("splits") / split / str(city["layout_id"])
        layout_root = output / layout_relative_root
        scene_dir = layout_root / "scene_authority"
        public_dir = layout_root / "method_public"
        private_dir = layout_root / "evaluator_private"
        scene_dir.mkdir(parents=True)
        (public_dir / "episodes").mkdir(parents=True)
        (private_dir / "episodes").mkdir(parents=True)
        write_compiled_public_v3(city, scene_dir, lock)
        materialized_cityspec = read_json(scene_dir / "cityspec.json")
        write_json(public_dir / "task_spec.json", task)
        public_episode = public_episode_projection(private_episode)
        if prior_level == "coarse-regions":
            public_episode.pop("mission_sector", None)
            public_episode.pop("mission_sector_hash", None)
        write_json(
            public_dir / "episodes" / f"episode-{episode_index:04d}.json",
            public_episode,
        )
        # Reject a boundary violation before this layout can become an L0/L1 input.
        public_boundary_audit = audit_public_layout(layout_root)
        write_json(private_dir / "episodes" / f"episode-{episode_index:04d}.json", private_episode)
        write_json(private_dir / "task_spec_authority.json", authority_task)
        write_json(
            private_dir / "support_sites.json",
            {
                "schema": "org.aerocity.bench.support-sites-private.ordinary.v3",
                "layout_id": city["layout_id"],
                "layout_hash": city["layout_hash"],
                "support_site_count": len(support_sites),
                "support_sites": support_sites,
            },
        )
        atlas = task.get("inspection_atlas")
        projection = task.get("inspection_atlas_projection")
        source_atlas_hash = (
            str(atlas["atlas_hash"])
            if isinstance(atlas, dict)
            else str(projection["source_atlas_hash"])
        )
        write_json(output / "development_layout_manifest.json", {
            "schema": "org.aerocity.bench.g2-i-development-layout.v1",
            "formal_score_eligible": False,
            "task_track": "G2-I",
            "split": split,
            "layout_id": city["layout_id"],
            "layout_hash": city["layout_hash"],
            "layout_relative_root": layout_relative_root.as_posix(),
            "task_spec_hash": task["task_spec_hash"],
            "authority_task_spec_hash": authority_task["task_spec_hash"],
            "atlas_hash": source_atlas_hash,
            "inspection_prior_level": prior_level,
            "asset_lock_hash": staged_assets["asset_lock_hash"],
            "episode_index": episode_index,
            "private_episode_source": episode_source,
            "city_source_sha256": content_hash(city),
            "materialized_cityspec_sha256": content_hash(materialized_cityspec),
            "private_episode_sha256": content_hash(private_episode),
            "public_boundary_audit": public_boundary_audit,
            "public_boundary_audit_hash": content_hash(public_boundary_audit),
        })
        return read_json(output / "development_layout_manifest.json")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    materialize(
        args.city,
        args.release_config,
        args.asset_root,
        args.output,
        episode_index=args.episode_index,
        prior_level=args.prior_level,
        private_episode_path=args.private_episode,
        development_split=args.development_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
