"""Run the precommitted public CF2X B-gate panel without cherry-picking.

The runner is deliberately thin: it resolves and validates the already frozen
3-by-3 panel, then launches the existing single-replay executable under the
shared Isaac host guard.  It never chooses a city or method from outcomes.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.cf2x_fleet_preflight_contract import validate_fleet_preflight_reports
from aerocity_bench.errors import HostGuardError
from aerocity_bench.host_guard import (
    HOST_GUARD_SCHEMA,
    foreign_isaac_processes,
    isaac_host_lock,
    run_guarded_process,
    validate_host_guard_pass_receipt,
)
from aerocity_bench.public_boundary import audit_public_layout

MANIFEST_SCHEMA = "org.aerocity.bench.cf2x-b-gate-manifest.v1"
MANIFEST_SCHEMA_V2 = "org.aerocity.bench.cf2x-b-gate-manifest.v2"
A_GATE_SCHEMA = "org.aerocity.bench.g2-i-a-gate-freeze.v1"
METHOD_IDS = ("sweep-3d", "atlas-surface-inspector", "atlas-region-greedy")
RETRY_POLICY_SCHEMA = "org.aerocity.bench.infrastructure-censoring-policy.v1"


def _current_evidence_pipeline_bindings() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    paths = {
        "manifest_builder_source_sha256": repository
        / "tools"
        / "build_cf2x_b_gate_manifest.py",
        "replay_runner_source_sha256": Path(__file__).resolve(),
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a-gate", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--layouts-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _hashed(path: Path, schema: str | tuple[str, ...]) -> dict[str, Any]:
    value = read_json(path.resolve())
    schemas = (schema,) if isinstance(schema, str) else schema
    if not isinstance(value, dict) or value.get("schema") not in schemas:
        raise ValueError(f"evidence schema differs: {path}")
    supplied = str(value.get("report_hash", ""))
    payload = dict(value)
    payload.pop("report_hash", None)
    if content_hash(payload) != supplied:
        raise ValueError(f"evidence hash mismatch: {path}")
    return value


def _relative_path(root: Path, value: object, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must remain below its declared root")
    return (root / relative).resolve()


def _layout_directory_name(ancestor: str) -> str:
    marker = "g2-i-calibration-"
    if not ancestor.startswith(marker):
        raise ValueError(f"unsupported B-gate ancestor identifier: {ancestor}")
    suffix = ancestor.removeprefix(marker)
    if not suffix.startswith("ancestor-"):
        raise ValueError(f"unsupported B-gate ancestor suffix: {ancestor}")
    return suffix


def _validate_completed_replay(
    public_path: Path,
    private_path: Path,
    *,
    method_id: str,
    expected_layout_hash: str,
    expected_input_bindings: dict[str, str],
) -> None:
    validate_fleet_preflight_reports(public_path, private_path)
    public = read_json(public_path)
    if not isinstance(public, dict):
        raise ValueError("completed public replay must be an object")
    bindings = public.get("input_bindings")
    if public.get("method") != method_id:
        raise ValueError("completed replay method differs from the precommitted panel")
    if not isinstance(bindings, dict) or bindings.get("layout_hash") != expected_layout_hash:
        raise ValueError("completed replay layout differs from the precommitted panel")
    for field, expected in expected_input_bindings.items():
        if bindings.get(field) != expected:
            raise ValueError(f"completed replay {field} differs from the precommitted binding")


def _validated_retry_policy(
    manifest_root: Path, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    value = manifest.get("infrastructure_censoring_policy")
    if value is None:
        return None
    if manifest.get("evidence_pipeline_bindings") != _current_evidence_pipeline_bindings():
        raise ValueError("B-gate evidence pipeline differs from its precommitted source hashes")
    required_fields = {
        "schema",
        "selection_rule",
        "decision_reads_method_outcome",
        "max_attempts_per_pair",
        "retry_archive_root",
        "allowed_host_guard_triggers",
        "required_quiescence_s",
        "maximum_quiescence_wait_s",
        "method_or_safety_failure_retry_allowed",
        "preserve_every_censored_attempt",
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise ValueError("B-gate infrastructure censoring policy fields differ")
    allowed = tuple(value["allowed_host_guard_triggers"])
    if (
        value["schema"] != RETRY_POLICY_SCHEMA
        or value["selection_rule"] != "first-host-isolated-complete-attempt"
        or value["decision_reads_method_outcome"] is not False
        or value["method_or_safety_failure_retry_allowed"] is not False
        or value["preserve_every_censored_attempt"] is not True
        or allowed
        != (
            "foreign_runtime",
            "foreign_runtime_during_attempt",
            "residual_runtime",
        )
        or isinstance(value["max_attempts_per_pair"], bool)
        or int(value["max_attempts_per_pair"]) <= 1
        or float(value["required_quiescence_s"]) < 0.0
        or float(value["maximum_quiescence_wait_s"])
        < float(value["required_quiescence_s"])
    ):
        raise ValueError("B-gate infrastructure censoring policy is invalid")
    return {
        **value,
        "allowed_host_guard_triggers": allowed,
        "archive_root": _relative_path(
            manifest_root, value["retry_archive_root"], "retry_archive_root"
        ),
    }


def _retryable_host_failure(
    report_path: Path,
    *,
    policy: dict[str, Any],
    evidence_binding: str,
) -> dict[str, Any]:
    receipt = read_json(report_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != HOST_GUARD_SCHEMA
        or receipt.get("status") != "FAIL"
        or receipt.get("trigger") not in policy["allowed_host_guard_triggers"]
        or receipt.get("evidence_binding") != evidence_binding
    ):
        raise ValueError("runtime attempt is not an outcome-blind infrastructure censoring event")
    return receipt


def _attempt_directories(archive_root: Path, stem: str) -> list[Path]:
    pair_root = archive_root / stem
    if not pair_root.exists():
        return []
    return sorted(
        path
        for path in pair_root.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")
    )


def _archive_retryable_attempt(item: dict[str, Any]) -> Path:
    policy = item["retry_policy"]
    if not isinstance(policy, dict):
        raise ValueError("infrastructure retry was not precommitted")
    runtime_root = Path(item["runtime_root"])
    receipt = _retryable_host_failure(
        runtime_root / "host_guard.json",
        policy=policy,
        evidence_binding=str(item["attempt_binding"]),
    )
    stem = f"{item['layout_ancestor']}__{item['method_id']}"
    existing = _attempt_directories(Path(policy["archive_root"]), stem)
    attempt_number = len(existing) + 1
    maximum_attempts = int(policy["max_attempts_per_pair"])
    if attempt_number > maximum_attempts:
        raise RuntimeError(f"infrastructure attempt limit already exhausted for {stem}")
    attempt_root = Path(policy["archive_root"]) / stem / f"attempt-{attempt_number:03d}"
    if attempt_root.exists():
        raise FileExistsError(f"retry archive attempt already exists: {attempt_root}")
    replay_archive = attempt_root / "replay"
    runtime_archive = attempt_root / "runtime"
    replay_archive.mkdir(parents=True)
    artifacts: dict[str, str] = {}
    public_path = Path(item["public_path"])
    private_path = Path(item["private_path"])
    candidates = (
        public_path,
        private_path,
        public_path.with_suffix(".progress.json"),
        public_path.with_name(f"{public_path.stem}.failure.json"),
    )
    for source in candidates:
        if source.exists():
            destination = replay_archive / source.name
            source.replace(destination)
            artifacts[f"replay/{destination.name}"] = file_hash(destination)
    runtime_root.replace(runtime_archive)
    for path in sorted(runtime_archive.rglob("*")):
        if path.is_file():
            relative = path.relative_to(attempt_root).as_posix()
            artifacts[relative] = file_hash(path)
    ledger: dict[str, Any] = {
        "schema": "org.aerocity.bench.infrastructure-censored-attempt.v1",
        "layout_ancestor": item["layout_ancestor"],
        "method_id": item["method_id"],
        "attempt_number": attempt_number,
        "evidence_binding": item["attempt_binding"],
        "host_guard_trigger": receipt["trigger"],
        "decision_reads_method_outcome": False,
        "retry_authorized": attempt_number < maximum_attempts,
        "artifacts": artifacts,
    }
    ledger["report_hash"] = content_hash(ledger)
    write_json(attempt_root / "attempt.json", ledger)
    return attempt_root


def _archived_attempt_authorizes_retry(attempt_root: Path) -> bool:
    ledger = read_json(attempt_root / "attempt.json")
    if not isinstance(ledger, dict) or ledger.get("retry_authorized") not in {True, False}:
        raise ValueError(f"censored attempt ledger is invalid: {attempt_root}")
    return bool(ledger["retry_authorized"])


def _is_retryable_host_failure(
    report_path: Path,
    *,
    policy: dict[str, Any] | None,
    evidence_binding: str,
) -> bool:
    if policy is None or not report_path.is_file():
        return False
    try:
        _retryable_host_failure(
            report_path,
            policy=policy,
            evidence_binding=evidence_binding,
        )
    except (OSError, TypeError, ValueError):
        return False
    return True


def _wait_for_retry_quiescence(policy: dict[str, Any]) -> None:
    required = float(policy["required_quiescence_s"])
    if required <= 0.0:
        return
    started = time.monotonic()
    quiet_started: float | None = None
    while True:
        now = time.monotonic()
        if now - started > float(policy["maximum_quiescence_wait_s"]):
            raise TimeoutError("host did not reach the precommitted retry quiescence window")
        if foreign_isaac_processes():
            quiet_started = None
        elif quiet_started is None:
            quiet_started = now
        elif now - quiet_started >= required:
            return
        time.sleep(min(5.0, max(0.1, required)))


def build_replay_plan(
    manifest_path: Path,
    a_gate_path: Path,
    source_root: Path,
    layouts_root: Path,
    release_config: Path,
    isaac_python: Path,
    cf2x_usd: Path,
    *,
    device: str,
    resume: bool,
) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest_root = manifest_path.parent
    manifest = _hashed(manifest_path, (MANIFEST_SCHEMA, MANIFEST_SCHEMA_V2))
    a_gate = _hashed(a_gate_path.resolve(), A_GATE_SCHEMA)
    if a_gate.get("status") != "VERIFIED" or a_gate.get("authorizes_next_gate") is not True:
        raise ValueError("B-gate execution requires a verified A gate")
    if a_gate.get("authorizes_formal_test_access") is not False:
        raise ValueError("B-gate execution must not authorize formal-test access")
    if manifest.get("a_gate_report_hash") != a_gate.get("report_hash"):
        raise ValueError("B-gate manifest is bound to another A-gate report")
    ancestors = tuple(manifest.get("layout_ancestors", ()))
    methods = tuple(manifest.get("method_ids", ()))
    records = manifest.get("records")
    if len(ancestors) != 3 or not isinstance(records, list):
        raise ValueError("B-gate manifest is not a frozen three-ancestor panel")
    if manifest.get("schema") == MANIFEST_SCHEMA:
        if methods != METHOD_IDS:
            raise ValueError("legacy B-gate manifest method plan differs")
    else:
        preflight = manifest.get("behavior_preflight")
        if not isinstance(preflight, dict):
            raise ValueError("B-gate v2 manifest lacks its behavior preflight")
        supplied_binding = str(preflight.get("binding_hash", ""))
        preflight_payload = dict(preflight)
        preflight_payload.pop("binding_hash", None)
        groups = preflight.get("mechanism_groups")
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
            supplied_binding != content_hash(preflight_payload)
            or preflight.get("candidate_method_ids") != list(METHOD_IDS)
            or tuple(preflight.get("l1_representative_method_ids", ())) != methods
            or tuple(preflight.get("l1_representative_method_ids", ()))
            != expected_representatives
            or preflight.get("excluded_redundant_method_ids") != expected_excluded
            or not canonical_groups
            or sorted(method for group in groups for method in group)
            != sorted(METHOD_IDS)
            or preflight.get("candidate_methods_are_not_deleted") is not True
            or preflight.get(
                "redundant_methods_do_not_count_as_independent_mechanisms"
            )
            is not True
        ):
            raise ValueError("B-gate v2 behavior preflight binding is invalid")
    expected_pairs = {(str(ancestor), method) for ancestor in ancestors for method in methods}
    actual_pairs = {
        (str(record.get("layout_ancestor", "")), str(record.get("method_id", "")))
        for record in records
        if isinstance(record, dict)
    }
    if len(records) != len(ancestors) * len(methods) or actual_pairs != expected_pairs:
        raise ValueError("B-gate records do not form the frozen representative panel")
    replay_root_value = str(manifest.get("replay_root", ""))
    runtime_root_value = manifest.get("runtime_root")
    if runtime_root_value is None:
        if replay_root_value != "replays":
            raise ValueError("B-gate manifest lacks a precommitted runtime root")
        runtime_root_value = "runtime"
    replay_root = _relative_path(manifest_root, replay_root_value, "replay_root")
    runtime_root = _relative_path(manifest_root, runtime_root_value, "runtime_root")
    if replay_root == runtime_root:
        raise ValueError("B-gate replay and runtime roots must differ")
    retry_policy = _validated_retry_policy(manifest_root, manifest)
    expected_input_bindings = manifest.get("expected_input_bindings")
    if expected_input_bindings is None:
        if replay_root_value != "replays":
            raise ValueError("versioned B-gate replay lacks precommitted input bindings")
        expected_input_bindings = {}
    required_binding_fields = {
        "baseline_source_sha256",
        "geometry_source_sha256",
        "controller_spec_hash",
        "cf2x_usd_sha256",
        "release_config_sha256",
    }
    if (
        not isinstance(expected_input_bindings, dict)
        or set(expected_input_bindings) != required_binding_fields
    ):
        if expected_input_bindings:
            raise ValueError("B-gate expected input bindings are incomplete")
    selected = {
        str(item.get("layout_ancestor", "")): item
        for item in manifest.get("selected_source_inputs", ())
        if isinstance(item, dict)
    }
    if set(selected) != set(ancestors):
        raise ValueError("B-gate selected inputs differ from the frozen ancestors")
    for path, label in (
        (release_config, "release config"),
        (isaac_python, "Isaac Python"),
        (cf2x_usd, "CF2X USD"),
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(f"{label} is absent: {path}")
    if "5_in_drone" in str(cf2x_usd).casefold() or "five_in_drone" in str(cf2x_usd).casefold():
        raise ValueError("B-gate runner refuses the prohibited 5_in_drone dependency")
    baseline_source = (
        Path(__file__).resolve().parents[1] / "src" / "aerocity_bench" / "baselines.py"
    )
    geometry_source = (
        Path(__file__).resolve().parents[1] / "src" / "aerocity_bench" / "geometry.py"
    )
    for path, field in (
        (baseline_source, "baseline_source_sha256"),
        (geometry_source, "geometry_source_sha256"),
        (release_config, "release_config_sha256"),
        (cf2x_usd, "cf2x_usd_sha256"),
    ):
        expected = expected_input_bindings.get(field)
        if expected is not None and file_hash(path.resolve()) != expected:
            raise ValueError(f"B-gate {field} differs from the precommitted file")

    plan: list[dict[str, Any]] = []
    layout_by_ancestor: dict[str, tuple[Path, dict[str, Any]]] = {}
    for ancestor in ancestors:
        ancestor = str(ancestor)
        source = selected[ancestor]
        city_path = _relative_path(source_root.resolve(), source.get("city_path"), "city_path")
        episode_path = _relative_path(
            source_root.resolve(), source.get("private_episode_path"), "private_episode_path"
        )
        city = read_json(city_path)
        episode = read_json(episode_path)
        layout_bundle = layouts_root.resolve() / _layout_directory_name(ancestor)
        development = read_json(layout_bundle / "development_layout_manifest.json")
        if not isinstance(development, dict):
            raise ValueError(f"development layout manifest is invalid for {ancestor}")
        if (
            development.get("formal_score_eligible") is not False
            or development.get("task_track") != "G2-I"
            or development.get("inspection_prior_level") != "full-cells"
            or development.get("private_episode_source") != "frozen-calibration-input"
            or development.get("split") != "calibration"
        ):
            raise ValueError(f"development layout contract differs for {ancestor}")
        source_city_hash = content_hash(city)
        if development.get("city_source_sha256") != source_city_hash:
            # A public layout stores a projected cityspec rather than the
            # generator's private source record.  Accept that projection only
            # when its own content hash was explicitly attested at materialize
            # time; the original source hash remains in the manifest.
            if development.get("materialized_cityspec_sha256") != source_city_hash:
                raise ValueError(f"materialized city differs from precommitted source: {ancestor}")
        if development.get("private_episode_sha256") != content_hash(episode):
            raise ValueError(
                f"materialized private episode differs from precommitted source: {ancestor}"
            )
        layout_root = _relative_path(
            layout_bundle, development.get("layout_relative_root"), "layout_relative_root"
        )
        if not layout_root.is_dir():
            raise FileNotFoundError(f"materialized layout root is absent: {layout_root}")
        audit_public_layout(layout_root)
        layout_by_ancestor[ancestor] = (layout_root, development)

    for record in records:
        ancestor = str(record["layout_ancestor"])
        method = str(record["method_id"])
        layout_root, development = layout_by_ancestor[ancestor]
        public_path = _relative_path(manifest_root, record["public_report"], "public_report")
        private_path = _relative_path(manifest_root, record["private_report"], "private_report")
        stem = f"{ancestor}__{method}"
        if public_path != replay_root / f"{stem}.public.json":
            raise ValueError("public replay path differs from the precommitted replay root")
        if private_path != replay_root / f"{stem}.private.json":
            raise ValueError("private replay path differs from the precommitted replay root")
        attempt_runtime_root = runtime_root / stem
        attempt_binding = content_hash(
            {
                "manifest_report_hash": manifest["report_hash"],
                "layout_ancestor": ancestor,
                "method_id": method,
            }
        )
        failure_path = public_path.with_name(f"{public_path.stem}.failure.json")
        public_exists = public_path.exists()
        private_exists = private_path.exists()
        runtime_exists = attempt_runtime_root.exists()
        archived_attempt_count = (
            len(_attempt_directories(Path(retry_policy["archive_root"]), stem))
            if retry_policy is not None
            else 0
        )
        if retry_policy is not None and archived_attempt_count >= int(
            retry_policy["max_attempts_per_pair"]
        ):
            raise RuntimeError(f"infrastructure attempt limit exhausted for {stem}")
        retryable_runtime = resume and _is_retryable_host_failure(
            attempt_runtime_root / "host_guard.json",
            policy=retry_policy,
            evidence_binding=attempt_binding,
        )
        status = "PENDING"
        if retryable_runtime:
            status = "RETRYABLE_INFRASTRUCTURE_FAILURE"
        elif failure_path.exists():
            raise FileExistsError(
                f"failed replay evidence already exists and will not be overwritten: {failure_path}"
            )
        elif public_exists or private_exists:
            if not resume or not (public_exists and private_exists):
                raise FileExistsError(
                    f"partial or unapproved existing replay evidence: {public_path}, {private_path}"
                )
            _validate_completed_replay(
                public_path,
                private_path,
                method_id=method,
                expected_layout_hash=str(development["layout_hash"]),
                expected_input_bindings=expected_input_bindings,
            )
            validate_host_guard_pass_receipt(
                attempt_runtime_root / "host_guard.json",
                expected_evidence_binding=attempt_binding,
            )
            status = "VERIFIED_EXISTING"
        elif runtime_exists:
            raise FileExistsError(
                "failed or incomplete runtime attempt already exists and cannot be resumed: "
                f"{attempt_runtime_root}"
            )
        command = [
            str(isaac_python.resolve()),
            str(Path(__file__).resolve().parent / "cf2x_l1_fleet_preflight.py"),
            "--layout-root",
            str(layout_root),
            "--release-config",
            str(release_config.resolve()),
            "--output",
            str(public_path),
            "--private-output",
            str(private_path),
            "--cf2x-usd",
            str(cf2x_usd.resolve()),
            "--execution-mode",
            "public-policy",
            "--run-purpose",
            "complete-calibration-episode",
            "--max-sim-time-s",
            "300",
            "--method",
            method,
            "--device",
            device,
            "--headless",
        ]
        plan.append(
            {
                "layout_ancestor": ancestor,
                "method_id": method,
                "status": status,
                "public_path": public_path,
                "private_path": private_path,
                "runtime_root": attempt_runtime_root,
                "attempt_binding": attempt_binding,
                "command": command,
                "expected_layout_hash": str(development["layout_hash"]),
                "expected_input_bindings": expected_input_bindings,
                "retry_policy": retry_policy,
            }
        )
    return plan


def run(args: argparse.Namespace) -> int:
    if args.timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    plan = build_replay_plan(
        args.manifest,
        args.a_gate,
        args.source_root,
        args.layouts_root,
        args.release_config,
        args.isaac_python,
        args.cf2x_usd,
        device=str(args.device),
        resume=bool(args.resume),
    )
    pending = [item for item in plan if item["status"] != "VERIFIED_EXISTING"]
    selected = pending[: args.limit] if args.limit is not None else pending
    print(
        f"B-gate plan verified: total={len(plan)} existing={len(plan) - len(pending)} "
        f"selected_pending={len(selected)}"
    )
    if args.prepare_only:
        for item in selected:
            print(f"PENDING {item['layout_ancestor']} {item['method_id']}")
        return 0
    if selected:
        initial_policy = selected[0]["retry_policy"]
        if initial_policy is not None:
            _wait_for_retry_quiescence(initial_policy)
    environment = dict(os.environ)
    with isaac_host_lock():
        for item in selected:
            runtime_root = Path(item["runtime_root"])
            policy = item["retry_policy"]
            if item["status"] == "RETRYABLE_INFRASTRUCTURE_FAILURE":
                archived = _archive_retryable_attempt(item)
                print(f"CENSORED {item['layout_ancestor']} {item['method_id']} {archived.name}")
                if not _archived_attempt_authorizes_retry(archived):
                    raise RuntimeError(
                        "precommitted infrastructure attempt limit exhausted after preserving "
                        f"{archived}"
                    )
                _wait_for_retry_quiescence(policy)
            while True:
                if runtime_root.exists():
                    raise FileExistsError(
                        "runtime attempt already exists and will not be overwritten: "
                        f"{runtime_root}"
                    )
                runtime_root.mkdir(parents=True)
                try:
                    guarded = run_guarded_process(
                        item["command"],
                        cwd=Path(__file__).resolve().parents[1],
                        environment=environment,
                        log_path=runtime_root / "isaac.log",
                        report_path=runtime_root / "host_guard.json",
                        timeout_s=float(args.timeout_s),
                        evidence_binding=str(item["attempt_binding"]),
                    )
                except HostGuardError:
                    if not _is_retryable_host_failure(
                        runtime_root / "host_guard.json",
                        policy=policy,
                        evidence_binding=str(item["attempt_binding"]),
                    ):
                        raise
                    archived = _archive_retryable_attempt(item)
                    print(
                        f"CENSORED {item['layout_ancestor']} {item['method_id']} "
                        f"{archived.name}"
                    )
                    if not _archived_attempt_authorizes_retry(archived):
                        raise RuntimeError(
                            "precommitted infrastructure attempt limit exhausted after preserving "
                            f"{archived}"
                        ) from None
                    _wait_for_retry_quiescence(policy)
                    continue
                break
            if guarded.returncode != 0:
                raise RuntimeError(
                    f"B-gate replay exited with {guarded.returncode}: "
                    f"{item['layout_ancestor']} {item['method_id']}"
                )
            _validate_completed_replay(
                item["public_path"],
                item["private_path"],
                method_id=str(item["method_id"]),
                expected_layout_hash=str(item["expected_layout_hash"]),
                expected_input_bindings=dict(item["expected_input_bindings"]),
            )
            print(f"VERIFIED {item['layout_ancestor']} {item['method_id']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
