"""Build a small method-independent G2-I sector calibration manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aerocity_bench.canonical import content_hash, write_json
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.errors import GenerationRejected
from aerocity_bench.generator_v3 import generate_city_v3
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.targets_v3 import derive_support_sites_v3, sample_episode_v3

MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ancestor-count", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=160)
    parser.add_argument(
        "--splits",
        default="calibration",
        help="comma-separated development splits; formal test splits are rejected",
    )
    return parser.parse_args(argv)


def build_manifest(
    config_path: Path,
    output_path: Path,
    *,
    ancestor_count: int,
    max_attempts: int,
    splits: tuple[str, ...] = ("calibration",),
) -> dict[str, object]:
    if ancestor_count < 3 or max_attempts < 1:
        raise ValueError("ancestor-count must be at least three and max-attempts positive")
    allowed_splits = {"train", "validation", "calibration"}
    if not splits or any(split not in allowed_splits for split in splits):
        raise ValueError(
            "splits must contain only train, validation, and calibration development splits"
        )
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")
    config = load_ordinary_config(config_path.resolve())
    root = output_path.resolve().parent
    input_root = root / f"{output_path.stem}-inputs"
    if input_root.exists():
        raise FileExistsError(f"calibration input directory already exists: {input_root}")
    records: list[dict[str, object]] = []
    accepted = 0
    try:
        input_root.mkdir(parents=True)
        shutil.copyfile(config_path.resolve(), input_root / "release_config.json")
        for split in splits:
            for ancestor_index in range(ancestor_count):
                accepted_city = None
                for attempt in range(max_attempts):
                    try:
                        city = generate_city_v3(
                            config,
                            split,
                            ancestor_index,
                            attempt,
                            list(config.raw["assets"]["allowlist"]),
                        )
                        task_spec = compile_g2_i_task_spec(
                            city,
                            config.raw["execution_contract"],
                            config.raw["fleet"],
                        )
                        support_sites = derive_support_sites_v3(city, config)
                        # Freeze the sector and prove all paired process instances
                        # can be sampled before admitting the ancestor.
                        for episode_index in range(3):
                            sample_episode_v3(
                                config,
                                city,
                                support_sites,
                                episode_index,
                                public_task_spec=task_spec,
                            )
                        accepted_city = city
                        break
                    except (GenerationRejected, ValueError):
                        continue
                if accepted_city is None:
                    raise GenerationRejected(
                        f"could not admit G2-I {split} ancestor {ancestor_index} "
                        f"after {max_attempts} attempts"
                    )
                city = accepted_city
                city_name = f"{split}-ancestor-{ancestor_index:02d}"
                city_path = input_root / f"{city_name}-city.json"
                write_json(city_path, city)
                for episode_index in range(3):
                    selector_path = input_root / f"{city_name}-episode-{episode_index:02d}.json"
                    task_spec = compile_g2_i_task_spec(
                        city,
                        config.raw["execution_contract"],
                        config.raw["fleet"],
                    )
                    support_sites = derive_support_sites_v3(city, config)
                    episode = sample_episode_v3(
                        config,
                        city,
                        support_sites,
                        episode_index,
                        public_task_spec=task_spec,
                    )
                    write_json(selector_path, episode)
                    records.append(
                        {
                            "city_path": city_path.relative_to(root).as_posix(),
                            "private_episode_path": selector_path.relative_to(root).as_posix(),
                            "layout_ancestor": f"g2-i-{split}-ancestor-{ancestor_index:02d}",
                            "split_label": split,
                        }
                    )
                accepted += 1
        manifest: dict[str, object] = {
            "schema": MANIFEST_SCHEMA,
            "purpose": "method-independent-task-calibration",
            "self_method_results_used": False,
            "release_config_path": (
                input_root / "release_config.json"
            ).relative_to(root).as_posix(),
            "records": records,
            "sector_policy": "g2-i-budgeted-public-sector-v1",
            "formal_score_eligible": False,
            "accepted_ancestor_count": accepted,
            "development_splits": list(splits),
        }
        manifest["manifest_hash"] = content_hash(manifest)
        write_json(output_path.resolve(), manifest)
        return manifest
    except Exception:
        # This is a development manifest.  Remove only the exact temporary
        # input directory just created; no prior evidence directory is touched.
        if input_root.exists():
            shutil.rmtree(input_root)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    build_manifest(
        args.config,
        args.output,
        ancestor_count=args.ancestor_count,
        max_attempts=args.max_attempts,
        splits=tuple(item.strip() for item in args.splits.split(",") if item.strip()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
