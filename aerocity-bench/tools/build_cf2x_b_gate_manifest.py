"""Freeze the public CF2X B-gate replay panel before any Isaac run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.behavioral_distinctness import COHORT_PANEL_AUDIT_SCHEMA
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json

A_GATE_SCHEMA = "org.aerocity.bench.g2-i-a-gate-freeze.v1"
SOURCE_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
MANIFEST_SCHEMA = "org.aerocity.bench.cf2x-b-gate-manifest.v1"
MANIFEST_SCHEMA_V2 = "org.aerocity.bench.cf2x-b-gate-manifest.v2"
METHOD_IDS = (
    "sweep-3d",
    "atlas-surface-inspector",
    "atlas-region-greedy",
)


def _evidence_pipeline_bindings() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    paths = {
        "manifest_builder_source_sha256": Path(__file__).resolve(),
        "replay_runner_source_sha256": repository / "tools" / "run_cf2x_b_gate_replays.py",
        "fleet_preflight_source_sha256": repository
        / "tools"
        / "cf2x_l1_fleet_preflight.py",
        "final_verifier_source_sha256": repository / "tools" / "verify_cf2x_b_gate.py",
        "host_guard_source_sha256": repository / "src" / "aerocity_bench" / "host_guard.py",
        "behavior_audit_source_sha256": repository
        / "src"
        / "aerocity_bench"
        / "behavioral_distinctness.py",
    }
    return {field: file_hash(path) for field, path in paths.items()}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-gate", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-root", default="replays")
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--expected-controller-spec-hash", required=True)
    parser.add_argument("--cf2x-usd-sha256", required=True)
    parser.add_argument("--release-config-sha256", required=True)
    parser.add_argument("--baseline-source-sha256", required=True)
    parser.add_argument("--geometry-source-sha256", required=True)
    parser.add_argument("--infrastructure-attempt-limit", type=int, default=1)
    parser.add_argument("--retry-archive-root", default=None)
    parser.add_argument("--retry-quiescence-s", type=float, default=180.0)
    parser.add_argument("--retry-max-wait-s", type=float, default=3600.0)
    parser.add_argument(
        "--behavior-audit",
        type=Path,
        required=True,
        help="complete multi-context L0 behavior audit required before expensive L1",
    )
    return parser.parse_args(argv)


def _hashed(path: Path, schema: str, hash_field: str = "report_hash") -> dict[str, Any]:
    value = read_json(path.resolve())
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"evidence schema differs: {path}")
    supplied = str(value.get(hash_field, ""))
    payload = dict(value)
    payload.pop(hash_field, None)
    if content_hash(payload) != supplied:
        raise ValueError(f"evidence hash mismatch: {path}")
    return value


def _select_ancestors(records: list[dict[str, Any]], source_root: Path) -> list[dict[str, Any]]:
    by_ancestor: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("calibration manifest record must be an object")
        ancestor = str(record.get("layout_ancestor", ""))
        if not ancestor:
            raise ValueError("calibration manifest record lacks layout ancestor")
        by_ancestor.setdefault(ancestor, []).append(record)
    ordered = sorted(by_ancestor)
    if len(ordered) < 3:
        raise ValueError("B-gate replay plan needs at least three independent ancestors")
    positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
    selected: list[dict[str, Any]] = []
    for position in positions:
        ancestor = ordered[position]
        candidates = sorted(
            by_ancestor[ancestor], key=lambda item: str(item.get("private_episode_path", ""))
        )
        chosen = candidates[0]
        city_path = (source_root / str(chosen["city_path"])).resolve()
        episode_path = (source_root / str(chosen["private_episode_path"])).resolve()
        if not city_path.is_file() or not episode_path.is_file():
            raise FileNotFoundError(f"B-gate source input is absent for {ancestor}")
        selected.append(
            {
                "layout_ancestor": ancestor,
                "city_path": str(chosen["city_path"]),
                "private_episode_path": str(chosen["private_episode_path"]),
                "city_file_sha256": file_hash(city_path),
                "private_episode_file_sha256": file_hash(episode_path),
            }
        )
    return selected


def _validated_relative_root(value: str, field: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{field} must be a non-empty path below the B-gate manifest")
    return path.as_posix()


def _default_runtime_root(replay_root: str) -> str:
    replay_path = Path(replay_root)
    if replay_path.as_posix() == "replays":
        return "runtime"
    return (replay_path.parent / f"{replay_path.name}-runtime").as_posix()


def _validated_sha256(value: str, field: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return normalized


def _behavior_preflight(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    audit = _hashed(path, COHORT_PANEL_AUDIT_SCHEMA)
    methods = tuple(str(value) for value in audit.get("method_ids", ()))
    representatives = tuple(
        str(value) for value in audit.get("l1_representative_method_ids", ())
    )
    groups = audit.get("mechanism_groups")
    canonical_groups = (
        isinstance(groups, list)
        and all(isinstance(group, list) and group for group in groups)
        and groups == sorted(groups)
        and all(group == sorted(group) for group in groups)
    )
    expected_representatives = (
        tuple(str(group[0]) for group in groups) if canonical_groups else ()
    )
    expected_excluded = (
        sorted(str(method) for group in groups for method in group[1:])
        if canonical_groups
        else []
    )
    if (
        methods != tuple(sorted(METHOD_IDS))
        or int(audit.get("context_count", 0)) < 3
        or not representatives
        or not canonical_groups
        or sorted(value for group in groups for value in group) != sorted(METHOD_IDS)
        or representatives != expected_representatives
        or audit.get("excluded_redundant_method_ids") != expected_excluded
        or audit.get("does_not_delete_or_censor_candidate_methods") is not True
    ):
        raise ValueError("B-gate behavior audit does not bind the complete three-method cohort")
    if audit.get("nondeterministic_methods"):
        raise ValueError(
            "B-gate behavior audit requires controlled stochastic-repeat adjudication"
        )
    if set(representatives) - set(METHOD_IDS):
        raise ValueError("B-gate behavior representatives are unknown")
    preflight = {
        "schema": "org.aerocity.bench.cf2x-behavior-preflight-binding.v1",
        "audit_report_hash": audit["report_hash"],
        "audit_file_sha256": file_hash(path.resolve()),
        "context_count": int(audit["context_count"]),
        "candidate_method_ids": list(METHOD_IDS),
        "mechanism_groups": groups,
        "l1_representative_method_ids": list(representatives),
        "excluded_redundant_method_ids": list(audit["excluded_redundant_method_ids"]),
        "candidate_methods_are_not_deleted": True,
        "redundant_methods_do_not_count_as_independent_mechanisms": True,
    }
    preflight["binding_hash"] = content_hash(preflight)
    return preflight, representatives


def build_manifest(
    a_gate_path: Path,
    calibration_manifest_path: Path,
    output_path: Path,
    *,
    replay_root: str = "replays",
    runtime_root: str | None = None,
    expected_controller_spec_hash: str | None = None,
    cf2x_usd_sha256: str | None = None,
    release_config_sha256: str | None = None,
    baseline_source_sha256: str | None = None,
    geometry_source_sha256: str | None = None,
    infrastructure_attempt_limit: int = 1,
    retry_archive_root: str | None = None,
    retry_quiescence_s: float = 180.0,
    retry_max_wait_s: float = 3600.0,
    behavior_audit_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite B-gate manifest: {output_path}")
    replay_root = _validated_relative_root(replay_root, "replay-root")
    runtime_root = _validated_relative_root(
        runtime_root if runtime_root is not None else _default_runtime_root(replay_root),
        "runtime-root",
    )
    if Path(runtime_root) == Path(replay_root):
        raise ValueError("runtime-root must differ from replay-root")
    if infrastructure_attempt_limit <= 0:
        raise ValueError("infrastructure-attempt-limit must be positive")
    if retry_quiescence_s < 0.0 or retry_max_wait_s <= 0.0:
        raise ValueError("retry quiescence and maximum wait must be non-negative and positive")
    retry_policy: dict[str, Any] | None = None
    if infrastructure_attempt_limit > 1:
        if retry_archive_root is None:
            raise ValueError("retry-archive-root is required for infrastructure retries")
        retry_archive_root = _validated_relative_root(
            retry_archive_root, "retry-archive-root"
        )
        if Path(retry_archive_root) in {Path(replay_root), Path(runtime_root)}:
            raise ValueError("retry-archive-root must differ from replay and runtime roots")
        if retry_max_wait_s < retry_quiescence_s:
            raise ValueError("retry maximum wait must cover the quiescence window")
        retry_policy = {
            "schema": "org.aerocity.bench.infrastructure-censoring-policy.v1",
            "selection_rule": "first-host-isolated-complete-attempt",
            "decision_reads_method_outcome": False,
            "max_attempts_per_pair": infrastructure_attempt_limit,
            "retry_archive_root": retry_archive_root,
            "allowed_host_guard_triggers": [
                "foreign_runtime",
                "foreign_runtime_during_attempt",
                "residual_runtime",
            ],
            "required_quiescence_s": retry_quiescence_s,
            "maximum_quiescence_wait_s": retry_max_wait_s,
            "method_or_safety_failure_retry_allowed": False,
            "preserve_every_censored_attempt": True,
        }
    expected_bindings: dict[str, str] | None = None
    supplied_bindings = (
        expected_controller_spec_hash,
        cf2x_usd_sha256,
        release_config_sha256,
        baseline_source_sha256,
        geometry_source_sha256,
    )
    if any(value is not None for value in supplied_bindings):
        if any(value is None for value in supplied_bindings):
            raise ValueError("all expected B-gate input binding hashes must be supplied together")
        expected_bindings = {
            "controller_spec_hash": _validated_sha256(
                str(expected_controller_spec_hash), "expected-controller-spec-hash"
            ),
            "cf2x_usd_sha256": _validated_sha256(str(cf2x_usd_sha256), "cf2x-usd-sha256"),
            "release_config_sha256": _validated_sha256(
                str(release_config_sha256), "release-config-sha256"
            ),
            "baseline_source_sha256": _validated_sha256(
                str(baseline_source_sha256), "baseline-source-sha256"
            ),
            "geometry_source_sha256": _validated_sha256(
                str(geometry_source_sha256), "geometry-source-sha256"
            ),
        }
    a_gate = _hashed(a_gate_path, A_GATE_SCHEMA)
    if a_gate.get("status") != "VERIFIED" or a_gate.get("authorizes_next_gate") is not True:
        raise ValueError("B-gate plan requires a verified A gate")
    source = _hashed(calibration_manifest_path, SOURCE_SCHEMA, hash_field="manifest_hash")
    selected = _select_ancestors(source["records"], calibration_manifest_path.resolve().parent)
    behavior_preflight: dict[str, Any] | None = None
    l1_method_ids = METHOD_IDS
    manifest_schema = MANIFEST_SCHEMA
    if behavior_audit_path is not None:
        behavior_preflight, l1_method_ids = _behavior_preflight(behavior_audit_path.resolve())
        manifest_schema = MANIFEST_SCHEMA_V2
    records: list[dict[str, str]] = []
    for item in selected:
        ancestor = item["layout_ancestor"]
        for method in l1_method_ids:
            stem = f"{ancestor}__{method}"
            records.append(
                {
                    "layout_ancestor": ancestor,
                    "method_id": method,
                    "public_report": f"{replay_root}/{stem}.public.json",
                    "private_report": f"{replay_root}/{stem}.private.json",
                }
            )
    manifest: dict[str, Any] = {
        "schema": manifest_schema,
        "formal_score_eligible": False,
        "purpose": "development-only-public-four-cf2x-l1-calibration",
        "a_gate_report_hash": a_gate["report_hash"],
        "source_calibration_manifest_hash": source["manifest_hash"],
        "selection_policy": "sorted-ancestor-even-quantiles-v1",
        "precommitted_before_replays": True,
        "method_ids": list(l1_method_ids),
        "layout_ancestors": [item["layout_ancestor"] for item in selected],
        "selected_source_inputs": selected,
        "replay_root": replay_root,
        "runtime_root": runtime_root,
        "records": records,
    }
    if expected_bindings is not None:
        manifest["expected_input_bindings"] = expected_bindings
    if behavior_preflight is not None:
        manifest["behavior_preflight"] = behavior_preflight
    if retry_policy is not None:
        manifest["infrastructure_censoring_policy"] = retry_policy
        manifest["evidence_pipeline_bindings"] = _evidence_pipeline_bindings()
    manifest["report_hash"] = content_hash(manifest)
    write_json(output_path.resolve(), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    build_manifest(
        args.a_gate,
        args.calibration_manifest,
        args.output,
        replay_root=args.replay_root,
        runtime_root=args.runtime_root,
        expected_controller_spec_hash=args.expected_controller_spec_hash,
        cf2x_usd_sha256=args.cf2x_usd_sha256,
        release_config_sha256=args.release_config_sha256,
        baseline_source_sha256=args.baseline_source_sha256,
        geometry_source_sha256=args.geometry_source_sha256,
        infrastructure_attempt_limit=args.infrastructure_attempt_limit,
        retry_archive_root=args.retry_archive_root,
        retry_quiescence_s=args.retry_quiescence_s,
        retry_max_wait_s=args.retry_max_wait_s,
        behavior_audit_path=args.behavior_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
