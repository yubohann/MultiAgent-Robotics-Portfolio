"""Freeze target-independent G2-I atlas-density calibration inputs.

Each density condition recompiles its public atlas on the exact public region
cohort that the source calibration froze before target sampling. The already
admitted private target realization is replayed verbatim; only the public
sector cells and their content hash change. The generated manifest is therefore
an input artifact, not a post-hoc denominator attached to method results.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.inspection_atlas import (
    compile_public_mission_sector,
    inspection_sampling_policy,
)
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.targets_v3 import (
    validate_frozen_g2_i_episode,
)

try:
    from tools.run_g2_i_l0_calibration import MANIFEST_SCHEMA, _local_path
except ModuleNotFoundError:
    from run_g2_i_l0_calibration import MANIFEST_SCHEMA, _local_path


DENSITY_MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-atlas-density-calibration.v1"
DEFAULT_POLICY_IDS = (
    "g2-i-geometric-sampling-density-sparse-v1",
    "g2-i-geometric-sampling-calibration-candidate-v2",
    "g2-i-geometric-sampling-density-dense-v1",
)


def _matched_region_sectors(
    task_specs: dict[str, dict[str, Any]],
    starts: list[dict[str, Any]],
    execution_contract: dict[str, Any],
    required_region_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Compile every density on the source sector's pre-sampling region cohort."""

    if not required_region_ids:
        raise GenerationRejected("density source mission sector has no public regions")
    sectors = {
        policy_id: compile_public_mission_sector(
            task_spec["inspection_atlas"],
            starts,
            execution_contract,
            region_allowlist=required_region_ids,
            require_all_allowed_regions=True,
        )
        for policy_id, task_spec in task_specs.items()
    }
    if any(
        {str(value) for value in sector["selected_region_ids"]} != required_region_ids
        for sector in sectors.values()
    ):
        raise GenerationRejected("density candidates changed the source public region cohort")
    return sectors, required_region_ids


def _private_truth_projection(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_process": episode["target_process"],
        "target_count": episode["target_count"],
        "targets": episode["targets"],
        "distractors": episode["distractors"],
        "counterfactual_pairs": episode["counterfactual_pairs"],
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy-ids",
        default=",".join(DEFAULT_POLICY_IDS),
        help="comma-separated recognized public sampling policy IDs",
    )
    return parser.parse_args(argv)


def _source_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("source manifest schema differs")
    if manifest.get("purpose") != "method-independent-task-calibration":
        raise ValueError("source manifest is not method-independent calibration")
    if manifest.get("self_method_results_used") is not False:
        raise ValueError("density calibration cannot consume self-method results")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source manifest has no records")
    by_ancestor: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("source record must be an object")
        ancestor = str(record.get("layout_ancestor", ""))
        city_path = str(record.get("city_path", ""))
        episode_path = str(record.get("private_episode_path", ""))
        split = str(record.get("split_label", ""))
        if not ancestor or not city_path or not episode_path or split in FORMAL_SPLITS:
            raise ValueError("source density panel must contain development city records")
        prior = by_ancestor.setdefault(
            ancestor,
            {
                "city_path": city_path,
                "split_label": split,
                "private_episode_paths": [],
            },
        )
        if prior["city_path"] != city_path or prior["split_label"] != split:
            raise ValueError("one layout ancestor maps to conflicting source cities")
        if episode_path in prior["private_episode_paths"]:
            raise ValueError("source density panel repeats a private episode")
        prior["private_episode_paths"].append(episode_path)
    if len(by_ancestor) < 3:
        raise ValueError("density calibration requires at least three ancestors")
    return by_ancestor


def build_density_manifest(
    source_manifest_path: Path,
    output_path: Path,
    *,
    policy_ids: tuple[str, ...] = DEFAULT_POLICY_IDS,
) -> dict[str, Any]:
    """Materialize density-specific sectors and private episodes before replay."""

    source_manifest_path = source_manifest_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"density manifest already exists: {output_path}")
    if not policy_ids or len(set(policy_ids)) != len(policy_ids):
        raise ValueError("density calibration policy IDs must be non-empty and unique")
    policies = {policy_id: inspection_sampling_policy(policy_id) for policy_id in policy_ids}
    source = read_json(source_manifest_path)
    if not isinstance(source, dict):
        raise ValueError("source manifest must be an object")
    panel = _source_records(source)
    root = output_path.parent
    source_root = source_manifest_path.parent
    config_path = _local_path(
        source_root, source["release_config_path"], "release_config_path"
    )
    config = load_ordinary_config(config_path)
    input_root = root / f"{output_path.stem}-inputs"
    if input_root.exists():
        stale_entries = list(input_root.iterdir())
        if len(stale_entries) == 1 and stale_entries[0].name == "release_config.json":
            # A terminated builder can leave only this copied input behind.  It
            # contains no generated scene or private episode, so removing this
            # exact verified directory prevents a false permanent blockage.
            shutil.rmtree(input_root)
        else:
            raise FileExistsError(f"density input directory already exists: {input_root}")
    records: list[dict[str, Any]] = []
    try:
        input_root.mkdir(parents=True)
        shutil.copyfile(config_path, input_root / "release_config.json")
        process_count = len(config.target_processes("calibration"))
        if process_count < 2:
            raise ValueError("density calibration needs at least two target processes")
        for ancestor, source_record in sorted(panel.items()):
            city = read_json(
                _local_path(source_root, source_record["city_path"], "city_path")
            )
            if not isinstance(city, dict) or str(city.get("split")) in FORMAL_SPLITS:
                raise ValueError("density calibration source city must be a development city")
            task_specs = {
                policy_id: compile_g2_i_task_spec(
                    city,
                    config.raw["execution_contract"],
                    config.raw["fleet"],
                    inspection_sampling_policy=policy,
                )
                for policy_id, policy in policies.items()
            }
            source_episodes = [
                read_json(
                    _local_path(source_root, value, "private_episode_path")
                )
                for value in source_record["private_episode_paths"]
            ]
            if any(not isinstance(episode, dict) for episode in source_episodes):
                raise ValueError("source density private episode must be an object")
            source_episodes.sort(key=lambda episode: str(episode["target_process"]))
            if {str(episode["target_process"]) for episode in source_episodes} != set(
                config.target_processes("calibration")
            ):
                raise ValueError("source density panel lacks the target-process pairing")
            start_hashes = {content_hash(episode["starts"]) for episode in source_episodes}
            source_region_sets = {
                tuple(str(value) for value in episode["mission_sector"]["selected_region_ids"])
                for episode in source_episodes
            }
            if len(start_hashes) != 1 or len(source_region_sets) != 1:
                raise ValueError("source target-process pair changed starts or mission regions")
            starts = copy.deepcopy(source_episodes[0]["starts"])
            required_region_ids = set(next(iter(source_region_sets)))
            sectors, common_regions = _matched_region_sectors(
                task_specs,
                starts,
                config.raw["execution_contract"],
                required_region_ids,
            )
            episodes_by_policy: dict[str, dict[int, dict[str, Any]]] = {}
            truth_hashes: dict[int, str] = {}
            for policy_id in policies:
                task_spec = task_specs[policy_id]
                sector = sectors[policy_id]
                episodes_by_policy[policy_id] = {}
                for episode_index, source_episode in enumerate(source_episodes):
                    episode = copy.deepcopy(source_episode)
                    episode["mission_sector"] = copy.deepcopy(sector)
                    episode["mission_sector_hash"] = sector["sector_hash"]
                    episode.pop("episode_hash", None)
                    episode["episode_hash"] = content_hash(episode)
                    validate_frozen_g2_i_episode(
                        episode,
                        city,
                        task_spec,
                        config.raw["execution_contract"],
                    )
                    episodes_by_policy[policy_id][episode_index] = episode
                for episode_index, episode in episodes_by_policy[policy_id].items():
                    truth_hash = content_hash(_private_truth_projection(episode))
                    prior_hash = truth_hashes.setdefault(episode_index, truth_hash)
                    if truth_hash != prior_hash:
                        raise GenerationRejected(
                            "density conditions changed a paired private target realization"
                        )
            local_city_path = input_root / f"{ancestor}-city.json"
            write_json(local_city_path, city)
            for policy_id in policies:
                task_spec = task_specs[policy_id]
                atlas = task_spec["inspection_atlas"]
                for episode_index in range(process_count):
                    episode = episodes_by_policy[policy_id][episode_index]
                    episode_path = input_root / (
                        f"{ancestor}-{policy_id}-episode-{episode_index:02d}.json"
                    )
                    write_json(episode_path, episode)
                    records.append(
                        {
                            "layout_ancestor": ancestor,
                            "split_label": str(source_record["split_label"]),
                            "city_panel_origin": "source-manifest",
                            "sampling_policy_id": policy_id,
                            "city_path": local_city_path.relative_to(root).as_posix(),
                            "private_episode_path": episode_path.relative_to(root).as_posix(),
                            "city_hash": content_hash(city),
                            "task_spec_hash": task_spec["task_spec_hash"],
                            "atlas_hash": atlas["atlas_hash"],
                            "mission_sector_hash": episode["mission_sector_hash"],
                            "private_episode_hash": episode["episode_hash"],
                            "common_region_set_hash": content_hash(sorted(common_regions)),
                            "paired_private_truth_hash": truth_hashes[episode_index],
                        }
                    )
        manifest: dict[str, Any] = {
            "schema": DENSITY_MANIFEST_SCHEMA,
            "purpose": "method-independent-atlas-density-calibration",
            "formal_score_eligible": False,
            "self_method_results_used": False,
            "source_manifest_hash": content_hash(source),
            "release_config_path": (input_root / "release_config.json")
            .relative_to(root)
            .as_posix(),
            "policy_ids": list(policy_ids),
            "target_processes": list(config.target_processes("calibration")),
            "records": records,
            "frozen_before_execution": True,
            "private_targets_reused_from_source_manifest": True,
            "source_sector_frozen_before_original_private_sampling": True,
            "source_city_panel_matched_across_density_conditions": True,
            "public_region_cohort_matched_across_density_conditions": True,
            "private_target_realizations_matched_across_density_conditions": True,
        }
        manifest["manifest_hash"] = content_hash(manifest)
        write_json(output_path, manifest)
        return manifest
    except Exception:
        if input_root.exists():
            shutil.rmtree(input_root)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    policy_ids = tuple(item.strip() for item in args.policy_ids.split(",") if item.strip())
    build_density_manifest(
        args.source_manifest,
        args.output,
        policy_ids=policy_ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
