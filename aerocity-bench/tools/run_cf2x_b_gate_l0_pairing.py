"""Run exact L0 counterparts for a frozen CF2X B-gate panel.

This tool deliberately runs the same frozen development city and private episode
as the B-gate L1 panel.  It is not a formal score and it never exposes private
truth; the private episode is represented only by the evaluator commitment that
the L1 receipt already publishes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench import baselines as baselines_module
from aerocity_bench import compiler as compiler_module
from aerocity_bench import contracts as contracts_module
from aerocity_bench import evaluator as evaluator_module
from aerocity_bench import geometry as geometry_module
from aerocity_bench import inspection_atlas as inspection_atlas_module
from aerocity_bench import metrics as metrics_module
from aerocity_bench import ordinary_config as ordinary_config_module
from aerocity_bench import runtime as runtime_module
from aerocity_bench import targets_v3 as targets_module
from aerocity_bench.baselines import BASELINES, create_baseline
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.cf2x_l0_pairing_contract import (
    L0_PAIRING_METHODS,
    L0_PAIRING_SCHEMA,
    L0_PAIRING_SCOPE,
    SHARED_BINDING_FIELDS,
    is_sha256,
    l0_pair_record_evidence_hash,
    private_evaluator_commitment,
)
from aerocity_bench.compiler import compile_g2_i_task_spec
from aerocity_bench.metrics import evaluate_run
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.public_boundary import audit_public_layout
from aerocity_bench.runtime import L0FleetRuntime
from aerocity_bench.targets_v3 import public_episode_projection, validate_frozen_g2_i_episode

B_GATE_MANIFEST_SCHEMA = "org.aerocity.bench.cf2x-b-gate-manifest.v1"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--layouts-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _manifest_payload(path: Path) -> dict[str, Any]:
    payload = read_json(path.resolve())
    if not isinstance(payload, dict) or payload.get("schema") != B_GATE_MANIFEST_SCHEMA:
        raise ValueError("L0 pairing requires a B-gate manifest")
    claimed = payload.get("report_hash")
    unhashed = {key: value for key, value in payload.items() if key != "report_hash"}
    if not is_sha256(claimed) or claimed != content_hash(unhashed):
        raise ValueError("B-gate manifest hash is invalid")
    methods = tuple(payload.get("method_ids", ()))
    ancestors = tuple(payload.get("layout_ancestors", ()))
    records = payload.get("records")
    expected = payload.get("expected_input_bindings")
    if (
        methods != L0_PAIRING_METHODS
        or len(ancestors) != 3
        or len(set(ancestors)) != len(ancestors)
        or not isinstance(records, list)
        or len(records) != len(methods) * len(ancestors)
        or not isinstance(expected, dict)
        or set(expected) != {
            "baseline_source_sha256",
            "geometry_source_sha256",
            "controller_spec_hash",
            "cf2x_usd_sha256",
            "release_config_sha256",
        }
        or any(not is_sha256(value) for value in expected.values())
    ):
        raise ValueError("B-gate manifest is not a frozen bound three-by-three panel")
    pairs = {
        (str(item.get("layout_ancestor", "")), str(item.get("method_id", "")))
        for item in records
        if isinstance(item, dict)
    }
    if pairs != {(ancestor, method) for ancestor in ancestors for method in methods}:
        raise ValueError("B-gate manifest pairs differ from the frozen panel")
    return payload


def _implementation_hash() -> str:
    paths = {
        "runner": Path(__file__),
        "baselines": Path(str(baselines_module.__file__)),
        "compiler": Path(str(compiler_module.__file__)),
        "contracts": Path(str(contracts_module.__file__)),
        "evaluator": Path(str(evaluator_module.__file__)),
        "geometry": Path(str(geometry_module.__file__)),
        "inspection_atlas": Path(str(inspection_atlas_module.__file__)),
        "metrics": Path(str(metrics_module.__file__)),
        "ordinary_config": Path(str(ordinary_config_module.__file__)),
        "runtime": Path(str(runtime_module.__file__)),
        "targets": Path(str(targets_module.__file__)),
    }
    return content_hash({name: file_hash(path) for name, path in sorted(paths.items())})


def _city_root(layouts_root: Path, ancestor: str) -> Path:
    suffix = ancestor.rsplit("-", 1)[-1]
    candidates_root = layouts_root.resolve() / f"ancestor-{suffix}" / "splits" / "calibration"
    if not candidates_root.is_dir():
        raise ValueError(f"L0 pairing cannot find calibration directory for {ancestor}")
    cities = sorted(
        path
        for path in candidates_root.iterdir()
        if path.is_dir() and path.name.startswith("city-")
    )
    if len(cities) != 1:
        raise ValueError(f"L0 pairing requires one frozen city for {ancestor}")
    return cities[0]


def _shared_bindings(
    *,
    city: dict[str, Any],
    city_path: Path,
    stage_path: Path,
    task_path: Path,
    public_episode_path: Path,
    task_spec: dict[str, Any],
    public_episode: dict[str, Any],
    expected_global: dict[str, str],
    release_config: Path,
) -> dict[str, str]:
    bindings = {
        "layout_hash": content_hash(city),
        "stage_sha256": file_hash(stage_path),
        "cityspec_sha256": file_hash(city_path),
        "task_spec_sha256": file_hash(task_path),
        "task_spec_hash": str(task_spec.get("task_spec_hash", "")),
        "public_episode_sha256": file_hash(public_episode_path),
        "mission_sector_hash": str(public_episode.get("mission_sector_hash", "")),
        "atlas_hash": str(task_spec.get("inspection_atlas", {}).get("atlas_hash", "")),
        "execution_contract_hash": content_hash(task_spec["execution_contract"]),
        "release_config_sha256": file_hash(release_config),
        "baseline_source_sha256": file_hash(Path(str(baselines_module.__file__))),
        "geometry_source_sha256": file_hash(Path(str(geometry_module.__file__))),
    }
    if set(bindings) != SHARED_BINDING_FIELDS or any(
        not is_sha256(value) for value in bindings.values()
    ):
        raise ValueError("L0 pairing shared input bindings are incomplete")
    for field in ("release_config_sha256", "baseline_source_sha256", "geometry_source_sha256"):
        if bindings[field] != expected_global[field]:
            raise ValueError(f"L0 pairing {field} differs from the frozen B-gate panel")
    return bindings


def _run_method(
    *,
    method_id: str,
    config: Any,
    city: dict[str, Any],
    private_episode: dict[str, Any],
    public_episode: dict[str, Any],
    task_spec: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    descriptor = BASELINES[method_id]
    if descriptor.requires_private_truth:
        raise ValueError("B-gate L0 pairing only permits target-agnostic public methods")
    policy = create_baseline(method_id, config, task_spec, public_episode)
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        receipt_secret=b"cf2x-b-gate-l0-pairing-v1",
        public_task_spec=task_spec,
        public_episode=public_episode,
    )
    result = runtime.run_policy(policy)
    metrics = evaluate_run(
        result,
        private_episode,
        float(config.raw["execution_contract"]["episode"]["duration_s"]),
    )
    ledger = result["budget_ledger"]
    execution = {
        "all_returned_home": all(bool(value) for value in result["returned_home"].values()),
        "collision_count": int(ledger["collisions"]),
        "out_of_bounds_actions": int(ledger["out_of_bounds_actions"]),
        "deadline_miss_tick_count": int(ledger["deadline_misses"]),
        "task_time_s": float(result["task_time_s"]),
    }
    return float(metrics["quality"]["confirmed_count"]), execution


def run_pairing(
    *,
    manifest_path: Path,
    layouts_root: Path,
    release_config_path: Path,
) -> dict[str, Any]:
    manifest = _manifest_payload(manifest_path)
    expected_global = dict(manifest["expected_input_bindings"])
    config = load_ordinary_config(release_config_path.resolve())
    method_ids = tuple(manifest["method_ids"])
    records: list[dict[str, Any]] = []
    for ancestor in manifest["layout_ancestors"]:
        city_root = _city_root(layouts_root, str(ancestor))
        city_path = city_root / "scene_authority" / "cityspec.json"
        stage_path = city_root / "scene_authority" / "stage.usda"
        task_path = city_root / "evaluator_private" / "task_spec_authority.json"
        private_episode_path = city_root / "evaluator_private" / "episodes" / "episode-0000.json"
        public_episode_path = city_root / "method_public" / "episodes" / "episode-0000.json"
        required = (city_path, stage_path, task_path, private_episode_path, public_episode_path)
        if any(not path.is_file() for path in required):
            raise ValueError(f"L0 pairing input is incomplete for {ancestor}")
        audit_public_layout(city_root)
        city = read_json(city_path)
        private_episode = read_json(private_episode_path)
        public_episode = read_json(public_episode_path)
        if str(city.get("split")) in FORMAL_SPLITS:
            raise ValueError("B-gate L0 pairing must not read a formal split")
        task_spec = compile_g2_i_task_spec(
            city,
            config.raw["execution_contract"],
            config.raw["fleet"],
        )
        stored_task_spec = read_json(task_path)
        if content_hash(task_spec) != content_hash(stored_task_spec):
            raise ValueError("stored task authority differs from the compiled frozen task")
        validate_frozen_g2_i_episode(
            private_episode,
            city,
            task_spec,
            config.raw["execution_contract"],
        )
        if content_hash(public_episode_projection(private_episode)) != content_hash(public_episode):
            raise ValueError("public episode no longer projects from the frozen private episode")
        bindings = _shared_bindings(
            city=city,
            city_path=city_path,
            stage_path=stage_path,
            task_path=task_path,
            public_episode_path=public_episode_path,
            task_spec=task_spec,
            public_episode=public_episode,
            expected_global=expected_global,
            release_config=release_config_path.resolve(),
        )
        private_sha256 = file_hash(private_episode_path)
        commitment = private_evaluator_commitment(
            private_sha256,
            bindings["layout_hash"],
            bindings["execution_contract_hash"],
        )
        for method_id in method_ids:
            score, execution = _run_method(
                method_id=method_id,
                config=config,
                city=city,
                private_episode=private_episode,
                public_episode=public_episode,
                task_spec=task_spec,
            )
            record = {
                "layout_ancestor": str(ancestor),
                "method_id": method_id,
                "score": score,
                "execution_level": "L0",
                "input_bindings": bindings,
                "private_episode_sha256": private_sha256,
                "private_evaluator_commitment": commitment,
                "execution": execution,
            }
            record["evidence_hash"] = l0_pair_record_evidence_hash(
                record, str(manifest["report_hash"])
            )
            records.append(record)
    report: dict[str, Any] = {
        "schema": L0_PAIRING_SCHEMA,
        "evidence_scope": L0_PAIRING_SCOPE,
        "formal_score_eligible": False,
        "status": "VERIFIED_L0_PAIRING",
        "b_gate_manifest_report_hash": manifest["report_hash"],
        "b_gate_manifest_file_sha256": file_hash(manifest_path.resolve()),
        "l0_implementation_hash": _implementation_hash(),
        "layout_ancestors": list(manifest["layout_ancestors"]),
        "method_ids": list(method_ids),
        "expected_input_bindings": expected_global,
        "records": records,
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite L0 pairing evidence: {args.output}")
    report = run_pairing(
        manifest_path=args.manifest,
        layouts_root=args.layouts_root,
        release_config_path=args.release_config,
    )
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
