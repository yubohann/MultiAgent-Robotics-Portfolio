"""Freeze fresh, target-private G2-I L1 calibration inputs before replay.

Historical private episodes must not be reused after a task-schema correction.
This tool deterministically resamples one private episode per already selected
development CitySpec, validates it against the current G2-I task contract, and
binds every input hash before any L0 or L1 method run.  It deliberately has no
method, score, replay, or Isaac dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, read_json, write_json  # noqa: E402
from aerocity_bench.compiler import compile_g2_i_task_spec  # noqa: E402
from aerocity_bench.errors import GenerationRejected  # noqa: E402
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config  # noqa: E402
from aerocity_bench.targets_v3 import (  # noqa: E402
    derive_support_sites_v3,
    sample_episode_v3,
    validate_frozen_g2_i_episode,
)

SOURCE_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
FROZEN_INPUT_SCHEMA = "org.aerocity.bench.g2-i-frozen-l1-inputs.v1"
FROZEN_INPUT_SOURCE = "method-independent-deterministic-resample-before-replays"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "optional read-only root for the relative CitySpec paths in the source "
            "manifest; defaults to the manifest directory"
        ),
    )
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition-index",
        type=int,
        default=None,
        help="optionally freeze one target-process condition; default scans all frozen conditions",
    )
    parser.add_argument("--max-resample-groups", type=int, default=16)
    return parser.parse_args(argv)


def _read_hashed_source(path: Path) -> dict[str, Any]:
    source = read_json(path.resolve())
    if not isinstance(source, dict) or source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema differs")
    supplied_hash = source.get("manifest_hash")
    payload = {key: value for key, value in source.items() if key != "manifest_hash"}
    if not isinstance(supplied_hash, str) or supplied_hash != content_hash(payload):
        raise ValueError("source manifest hash is invalid")
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("source manifest records are invalid")
    return source


def _source_city_path(
    manifest_path: Path, value: object, source_root: Path | None = None
) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source city path must stay below the declared source root")
    root = manifest_path.resolve().parent if source_root is None else source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root is absent: {root}")
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("source city path escapes the declared source root")
    if not resolved.is_file():
        raise FileNotFoundError(f"source CitySpec is absent: {relative}")
    return resolved


def _selected_calibration_sources(
    source: dict[str, Any], manifest_path: Path, source_root: Path | None = None
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in source["records"]:
        if not isinstance(record, dict) or record.get("split_label") != "calibration":
            continue
        ancestor = record.get("layout_ancestor")
        if not isinstance(ancestor, str) or not ancestor.startswith("g2-i-calibration-ancestor-"):
            raise ValueError("calibration ancestor identifier is invalid")
        split = record.get("split_label")
        if not isinstance(split, str) or not split:
            raise ValueError("source record split label is invalid")
        public_city_path = _source_city_path(
            manifest_path, record.get("city_path"), source_root
        )
        private_value = record.get("private_city_source_path")
        private_city_path = (
            public_city_path
            if private_value is None
            else _source_city_path(manifest_path, private_value, source_root)
        )
        expected_private_hash = record.get("private_city_source_sha256")
        if expected_private_hash is not None and (
            not isinstance(expected_private_hash, str) or len(expected_private_hash) != 64
        ):
            raise ValueError("private CitySpec hash in source manifest is invalid")
        candidate = {
            "public_city_path": public_city_path,
            "private_city_path": private_city_path,
            "split": split,
            "public_city_source_path": str(record["city_path"]),
            "private_city_source_path": (
                None if private_value is None else str(private_value)
            ),
            "expected_private_city_sha256": expected_private_hash,
        }
        prior = selected.setdefault(ancestor, candidate)
        if prior != candidate:
            raise ValueError("one calibration ancestor maps to multiple CitySpecs")
    if len(selected) != 3:
        raise ValueError("current L1 input freeze requires exactly three calibration ancestors")
    return [
        {"layout_ancestor": ancestor, **item}
        for ancestor, item in sorted(selected.items())
    ]


def freeze_inputs(
    source_manifest_path: Path,
    release_config_path: Path,
    output: Path,
    *,
    condition_index: int | None = None,
    max_resample_groups: int = 16,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Write three frozen private episodes without observing any policy outcome."""

    if (condition_index is not None and condition_index < 0) or max_resample_groups <= 0:
        raise ValueError("condition-index and max-resample-groups are invalid")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen L1 inputs: {output}")
    source_manifest_path = source_manifest_path.resolve()
    source = _read_hashed_source(source_manifest_path)
    config = load_ordinary_config(release_config_path.resolve())
    selected = _selected_calibration_sources(source, source_manifest_path, source_root)
    process_count = len(config.target_processes("calibration"))
    if condition_index is not None and condition_index >= process_count:
        raise ValueError("condition-index exceeds the frozen calibration process set")
    condition_indices = (
        (condition_index,) if condition_index is not None else tuple(range(process_count))
    )

    output.mkdir(parents=True)
    try:
        records: list[dict[str, Any]] = []
        seen_layout_hashes: set[str] = set()
        for source_record in selected:
            ancestor = str(source_record["layout_ancestor"])
            split = str(source_record["split"])
            public_city = read_json(source_record["public_city_path"])
            private_city = read_json(source_record["private_city_path"])
            if not isinstance(public_city, dict) or not isinstance(private_city, dict):
                raise ValueError(f"source CitySpec is invalid: {ancestor}")
            expected_private_hash = source_record["expected_private_city_sha256"]
            if (
                expected_private_hash is not None
                and content_hash(private_city) != expected_private_hash
            ):
                raise ValueError(f"private CitySpec hash differs from source manifest: {ancestor}")
            if (
                public_city.get("layout_id") != private_city.get("layout_id")
                or public_city.get("layout_hash") != private_city.get("layout_hash")
            ):
                raise ValueError(f"public/private CitySpecs differ in layout binding: {ancestor}")
            city_split = private_city.get("split")
            if city_split is not None and city_split != split:
                raise ValueError(f"source CitySpec split differs from source manifest: {ancestor}")
            if split in FORMAL_SPLITS:
                raise ValueError("frozen L1 inputs must not consume a formal split")
            if "spawn_grammar" not in private_city:
                raise ValueError(
                    "private CitySpec lacks spawn_grammar required for deterministic "
                    "episode sampling"
                )
            # Release CitySpecs deliberately omit their split label. Restore
            # the manifest label only while making the private episode.
            episode_city = dict(private_city)
            episode_city["split"] = split
            layout_hash = public_city.get("layout_hash")
            if not isinstance(layout_hash, str) or layout_hash in seen_layout_hashes:
                raise ValueError("calibration CitySpecs must have distinct layout hashes")
            seen_layout_hashes.add(layout_hash)
            task_spec = compile_g2_i_task_spec(
                public_city, config.raw["execution_contract"], config.raw["fleet"]
            )
            support_sites = derive_support_sites_v3(private_city, config)
            episode: dict[str, Any] | None = None
            selected_episode_index: int | None = None
            selected_resample_group: int | None = None
            for resample_group in range(max_resample_groups):
                for candidate_condition in condition_indices:
                    candidate_episode_index = candidate_condition + resample_group * process_count
                    try:
                        episode = sample_episode_v3(
                            config,
                            episode_city,
                            support_sites,
                            candidate_episode_index,
                            public_task_spec=task_spec,
                        )
                    except GenerationRejected:
                        continue
                    selected_episode_index = candidate_episode_index
                    selected_resample_group = resample_group
                    break
                if episode is not None:
                    break
            if episode is None or selected_episode_index is None or selected_resample_group is None:
                raise GenerationRejected(
                    f"no valid frozen episode in {max_resample_groups} attempts for {ancestor}"
                )
            validate_frozen_g2_i_episode(
                episode,
                episode_city,
                task_spec,
                config.raw["execution_contract"],
            )
            relative_episode = (
                Path("episodes") / ancestor / f"episode-{selected_episode_index:04d}.json"
            )
            write_json(output / relative_episode, episode)
            records.append(
                {
                    "layout_ancestor": ancestor,
                    "city_source_path": source_record["public_city_source_path"],
                    "city_source_sha256": content_hash(public_city),
                    "private_city_source_path": source_record["private_city_source_path"],
                    "private_city_source_sha256": content_hash(private_city),
                    "layout_id": public_city["layout_id"],
                    "layout_hash": layout_hash,
                    "task_spec_hash": task_spec["task_spec_hash"],
                    "condition_index": selected_episode_index % process_count,
                    "resample_group_index": selected_resample_group,
                    "episode_index": selected_episode_index,
                    "private_episode_path": relative_episode.as_posix(),
                    "private_episode_sha256": content_hash(episode),
                }
            )
        report: dict[str, Any] = {
            "schema": FROZEN_INPUT_SCHEMA,
            "purpose": "current-public-boundary-cf2x-l1-input-freeze",
            "formal_score_eligible": False,
            "private_episode_source": FROZEN_INPUT_SOURCE,
            "source_manifest_sha256": content_hash(source),
            "release_config_sha256": content_hash(config.raw),
            "process_selection_rule": (
                "first-valid-by-resample-group-then-frozen-condition-v1"
                if condition_index is None
                else "first-valid-fixed-frozen-condition-v1"
            ),
            "requested_condition_index": condition_index,
            "process_count": process_count,
            "max_resample_groups": max_resample_groups,
            "records": records,
        }
        report["manifest_hash"] = content_hash(report)
        write_json(output / "frozen_inputs_manifest.json", report)
        return report
    except Exception:
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    freeze_inputs(
        args.source_manifest,
        args.release_config,
        args.output,
        condition_index=args.condition_index,
        max_resample_groups=args.max_resample_groups,
        source_root=args.source_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
