"""Build, validate, and safely export ordinary-v3 authority releases."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .assets import stage_assets
from .canonical import content_hash, file_hash, read_json, write_json
from .compiler import compile_method_task_spec, write_compiled_public_v3
from .errors import GenerationRejected, ValidationError
from .generator_v3 import generate_city_v3
from .inspection_atlas import TASK_TRACK_G1_U
from .isaac_bridge import (
    FORMAL_L1_EVIDENCE_SCOPE,
    REQUIRED_NATIVE_CHECKS,
    validate_native_gate_report,
)
from .native_gate_contract import load_native_gate_inputs
from .ordinary_config import (
    FORMAL_SPLITS,
    ORDINARY_SPLITS,
    OrdinaryReleaseConfig,
    load_ordinary_config,
    load_public_runtime_contract,
    public_execution_contract,
)
from .public_boundary import assert_public_fields, validate_public_task_spec
from .supply_chain import (
    load_official_cc0_lock,
    validate_release_legal_materials,
    write_release_legal_materials,
)
from .targets_v3 import (
    derive_support_sites_v3,
    public_episode_projection,
    sample_episode_v3,
    start_contract_errors,
)

MAX_ATTEMPTS_PER_LAYOUT = 160
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PRIVATE_DIRECTORY_NAMES = frozenset({"evaluator_private", "authority_private"})
PRIVATE_KEY_TOKENS = (
    "target_id",
    "site_id",
    "target_process",
    "validity_hash",
    "legal_witness",
    "counterfactual",
    "distractor",
    "generation_seed",
    "family_private",
)
SCIENTIFIC_GATES = (
    "three_dimensionality",
    "difficulty_calibration",
    "shortcut_red_team",
    "coverage_to_search_pilot",
    "baseline_vertical_slice",
)


def _native_input_bindings_for_layout(
    root: Path, layout: dict[str, Any]
) -> dict[str, str]:
    layout_id = str(layout["layout_id"])
    layout_root = root / "splits" / str(layout["split"]) / layout_id
    public_episodes = sorted((layout_root / "method_public" / "episodes").glob("*.json"))
    if not public_episodes:
        raise ValidationError(f"layout has no public native-gate episode: {layout_id}")
    _, _, _, _, bindings = load_native_gate_inputs(
        root / "authority_private" / "release_config.json",
        layout_root / "method_public" / "task_spec.json",
        public_episodes[0],
        layout_root / "scene_authority" / "cityspec.json",
    )
    return bindings


def _sha256_text(value: object, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValidationError(f"{name} must be a lowercase SHA-256 value")
    return text


def _all_file_hashes(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash(path)})
    return records


def _write_directory_manifest(
    root: Path,
    destination: Path,
    schema: str,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    manifest = {
        "schema": schema,
        "files": _all_file_hashes(root, exclude=excluded_relative_paths),
    }
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(destination, manifest)
    return manifest


def _write_private_layout(
    config: OrdinaryReleaseConfig,
    city: dict[str, Any],
    layout_dir: Path,
) -> dict[str, Any]:
    private_dir = layout_dir / "evaluator_private"
    public_episode_dir = layout_dir / "method_public" / "episodes"
    support_sites = derive_support_sites_v3(city, config)
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
    target_total = 0
    validity_hashes = []
    process_histogram: Counter[str] = Counter()
    for episode_index in range(config.episodes(str(city["split"]))):
        episode = sample_episode_v3(config, city, support_sites, episode_index)
        target_total += int(episode["target_count"])
        process_histogram[str(episode["target_process"])] += 1
        validity_hashes.append(str(episode["target_validity"]["validity_hash"]))
        write_json(
            private_dir / "episodes" / f"episode-{episode_index:04d}.json",
            episode,
        )
        write_json(
            public_episode_dir / f"episode-{episode_index:04d}.json",
            public_episode_projection(episode),
        )
    return {
        "support_site_count_private": len(support_sites),
        "target_instance_count_private": target_total,
        "target_process_histogram_private": dict(sorted(process_histogram.items())),
        "validity_set_hash_private": content_hash(sorted(validity_hashes)),
    }


def _write_layout_manifest(layout_dir: Path, city: dict[str, Any]) -> dict[str, Any]:
    destination = layout_dir / "authority_manifest.json"
    relative = destination.relative_to(layout_dir).as_posix()
    manifest = _write_directory_manifest(
        layout_dir,
        destination,
        "org.aerocity.bench.layout-authority-manifest.v1",
        excluded_relative_paths=frozenset({relative}),
    )
    manifest.update(
        {
            "layout_id": city["layout_id"],
            "layout_hash": city["layout_hash"],
            "topology_signature_private": city["topology_signature"],
        }
    )
    manifest["manifest_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    write_json(destination, manifest)
    return manifest


def build_ordinary_release(
    config: OrdinaryReleaseConfig,
    asset_root: Path,
    output: Path,
    selected_splits: tuple[str, ...] = ORDINARY_SPLITS,
    *,
    source_commit: str,
    allow_uncommitted_development: bool = False,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not selected_splits or set(selected_splits) - set(ORDINARY_SPLITS):
        raise ValueError("selected ordinary-v3 splits are invalid")
    if tuple(split for split in ORDINARY_SPLITS if split in selected_splits) != selected_splits:
        raise ValueError("selected splits must preserve the canonical order")
    source_commit = source_commit.lower()
    publishable_source = bool(COMMIT_PATTERN.fullmatch(source_commit))
    if config.raw["release_kind"] == "OFFICIAL" and not publishable_source:
        if not allow_uncommitted_development:
            raise ValueError("OFFICIAL builds require a frozen 40-character source commit")
        source_commit = "UNCOMMITTED-DEVELOPMENT"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        assets = config.raw["assets"]
        allowlist = [str(value) for value in assets["allowlist"]]
        lock, evidence, dependency_closure = load_official_cc0_lock(
            asset_root.resolve(), str(assets["bundle"]), allowlist
        )
        staged_assets = stage_assets(lock, asset_root.resolve(), staging)
        legal_manifest = write_release_legal_materials(
            staging,
            lock,
            evidence,
            dependency_closure,
            project_version=config.version,
            source_commit=source_commit,
        )
        layouts: list[dict[str, Any]] = []
        private_layouts: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        seen_layouts: set[str] = set()
        seen_topologies: set[str] = set()
        for split in selected_splits:
            for index in range(config.count(split)):
                accepted: dict[str, Any] | None = None
                private_summary: dict[str, Any] | None = None
                for attempt in range(MAX_ATTEMPTS_PER_LAYOUT):
                    candidate_dir: Path | None = None
                    try:
                        city = generate_city_v3(config, split, index, attempt, allowlist)
                        if city["layout_hash"] in seen_layouts:
                            raise GenerationRejected("duplicate layout hash")
                        if city["topology_signature"] in seen_topologies:
                            raise GenerationRejected("duplicate audited topology signature")
                        candidate_dir = staging / "splits" / split / city["layout_id"]
                        private_summary = _write_private_layout(config, city, candidate_dir)
                        write_compiled_public_v3(city, candidate_dir / "scene_authority", lock)
                        write_json(
                            candidate_dir / "method_public" / "task_spec.json",
                            compile_method_task_spec(
                                city,
                                config.raw["execution_contract"],
                                config.raw["fleet"],
                            ),
                        )
                        _write_layout_manifest(candidate_dir, city)
                        accepted = city
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
                if accepted is None or private_summary is None:
                    raise GenerationRejected(
                        f"failed to admit {split}[{index}] after {MAX_ATTEMPTS_PER_LAYOUT} attempts"
                    )
                seen_layouts.add(str(accepted["layout_hash"]))
                seen_topologies.add(str(accepted["topology_signature"]))
                layouts.append(
                    {
                        "split": split,
                        "layout_id": accepted["layout_id"],
                        "layout_hash": accepted["layout_hash"],
                        "size_m": accepted["size_m"],
                        "scene_authority_path": (
                            f"splits/{split}/{accepted['layout_id']}/scene_authority"
                        ),
                        "method_public_path": (
                            f"splits/{split}/{accepted['layout_id']}/method_public"
                        ),
                    }
                )
                private_layouts.append(
                    {
                        "split": split,
                        "layout_id": accepted["layout_id"],
                        "topology_signature": accepted["topology_signature"],
                        "family": accepted["family_private"],
                        "generation_seed": accepted["generation_seed"],
                        **private_summary,
                    }
                )
        write_json(
            staging / "authority_private" / "layout_index.json",
            {
                "schema": "org.aerocity.bench.layout-index-private.ordinary.v1",
                "layouts": private_layouts,
            },
        )
        write_json(
            staging / "authority_private" / "rejections.json",
            {
                "schema": "org.aerocity.bench.rejections-private.ordinary.v1",
                "rejection_count": len(rejections),
                "reason_histogram": dict(
                    sorted(Counter(item["reason"] for item in rejections).items())
                ),
                "candidates": rejections,
            },
        )
        write_json(staging / "authority_private" / "release_config.json", config.raw)
        index = {
            "schema": "org.aerocity.bench.authority-release-index.ordinary.v1",
            "release_version": config.version,
            "generator_version": config.generator_version,
            "release_config_sha256": config.config_hash,
            "source_commit": source_commit,
            "selected_splits": list(selected_splits),
            "formal_splits": list(FORMAL_SPLITS),
            "layouts": layouts,
            "asset_lock_hash": staged_assets["asset_lock_hash"],
            "legal_manifest_hash": legal_manifest["legal_manifest_hash"],
            "execution_contract_hash": content_hash(config.raw["execution_contract"]),
            "public_execution_contract_hash": content_hash(
                public_execution_contract(config.raw["execution_contract"])
            ),
            "scientific_status": "pilot_only",
            "formal_execution_level": "L1",
            "native_isaac_gate": "not_run",
            "public_release_gate": (
                "eligible_after_native_and_scientific_gates"
                if publishable_source and selected_splits == ORDINARY_SPLITS
                else (
                    "blocked_partial_build"
                    if selected_splits != ORDINARY_SPLITS
                    else "blocked_uncommitted_development"
                )
            ),
        }
        index["release_index_hash"] = content_hash(index)
        write_json(staging / "release_index.json", index)
        _write_directory_manifest(
            staging,
            staging / "AUTHORITY_MANIFEST.json",
            "org.aerocity.bench.authority-package-manifest.v1",
            excluded_relative_paths=frozenset({"AUTHORITY_MANIFEST.json"}),
        )
        report = validate_ordinary_release(staging)
        os.replace(staging, output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_file_manifest(root: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    expected_hash = str(manifest.get("manifest_hash", ""))
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if content_hash(payload) != expected_hash:
        raise ValidationError(f"manifest content hash mismatch: {manifest_path}")
    declared: set[str] = set()
    for record in manifest.get("files", []):
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValidationError(f"manifest contains an unsafe path: {record['path']}")
        relative_text = relative.as_posix()
        if relative_text in declared:
            raise ValidationError(f"manifest repeats a file path: {relative_text}")
        declared.add(relative_text)
        path = root.joinpath(*relative.parts).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValidationError(f"manifest path escapes its package: {relative_text}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or file_hash(path) != str(record["sha256"])
        ):
            raise ValidationError(f"manifest file failed validation: {path}")
    manifest_relative = manifest_path.resolve().relative_to(root.resolve()).as_posix()
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != manifest_relative
    }
    if actual != declared:
        missing = sorted(declared - actual)
        extra = sorted(actual - declared)
        raise ValidationError(
            f"manifest file set differs; missing={missing[:8]}, extra={extra[:8]}"
        )


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _validate_public_episode(path: Path) -> None:
    episode = read_json(path)
    try:
        assert_public_fields(episode, path=f"episode:{path.name}")
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    keys = {key.lower() for key in _walk_keys(episode)}
    leaked = sorted(key for key in keys if any(token in key for token in PRIVATE_KEY_TOKENS))
    # Explicit negative booleans are part of the public contract, not leaked values.
    leaked = [key for key in leaked if key not in {"target_count_public", "target_process_public"}]
    if leaked:
        raise ValidationError(f"method-public episode leaks evaluator keys: {path}: {leaked}")
    if (
        episode.get("target_count_public") is not False
        or episode.get("target_process_public") is not False
    ):
        raise ValidationError(f"method-public episode exposes target truth: {path}")


def _validate_task_spec(
    path: Path,
    *,
    layout_id: str,
    public_execution_contract_hash: str,
    fleet: dict[str, Any],
) -> dict[str, Any]:
    task_spec = read_json(path)
    try:
        validate_public_task_spec(task_spec)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    expected_hash = str(task_spec.pop("task_spec_hash", ""))
    if content_hash(task_spec) != expected_hash:
        raise ValidationError(f"method task spec hash mismatch: {path}")
    if task_spec.get("schema") != "org.aerocity.bench.task-spec-public.ordinary.v1":
        raise ValidationError(f"method task spec schema is invalid: {path}")
    if task_spec.get("task_track") != TASK_TRACK_G1_U or "inspection_atlas" in task_spec:
        raise ValidationError(f"ordinary-v3 method task spec is not G1-U: {path}")
    if task_spec.get("layout_id") != layout_id:
        raise ValidationError(f"method task spec belongs to another layout: {path}")
    if task_spec.get("public_execution_contract_hash") != public_execution_contract_hash:
        raise ValidationError(f"method task spec public execution-contract hash is invalid: {path}")
    if content_hash(task_spec.get("execution_contract")) != public_execution_contract_hash:
        raise ValidationError(f"method task spec changed the execution contract: {path}")
    if task_spec.get("fleet_profile") != fleet:
        raise ValidationError(f"method task spec changed the fleet contract: {path}")
    if any(
        task_spec.get(key) is not False
        for key in (
            "exact_cityspec_public",
            "target_count_public",
            "target_process_public",
            "formal_split_label_public",
        )
    ):
        raise ValidationError(f"method task spec exposes forbidden truth: {path}")
    prior = dict(task_spec.get("coarse_prior", {}))
    prior_hash = str(prior.pop("prior_hash", ""))
    if content_hash(prior) != prior_hash or prior.get("layout_id") != layout_id:
        raise ValidationError(f"method task spec has a corrupt coarse prior: {path}")
    return task_spec


def _validate_asset_lock(
    root: Path, index: dict[str, Any], *, expected_bundle: str, expected_ids: set[str]
) -> None:
    lock_path = root / "_assets" / "asset_lock.json"
    lock = read_json(lock_path)
    expected_hash = str(lock.pop("asset_lock_hash", ""))
    if content_hash(lock) != expected_hash or expected_hash != index.get("asset_lock_hash"):
        raise ValidationError("asset lock hash differs from the release index")
    if lock.get("bundle") != expected_bundle:
        raise ValidationError("asset lock bundle differs from the frozen configuration")
    records = lock.get("assets", [])
    if {str(record.get("asset_id")) for record in records} != expected_ids:
        raise ValidationError("asset lock IDs differ from the frozen allowlist")
    for record in records:
        if record.get("spdx") != "CC0-1.0":
            raise ValidationError("ordinary-v3 asset lock contains a non-CC0 asset")
        for file_record in record.get("files", []):
            relative = PurePosixPath(str(file_record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError("asset lock contains an unsafe file path")
            staged = root / "_assets" / expected_bundle / Path(*relative.parts)
            if not staged.is_file() or file_hash(staged) != file_record.get("sha256"):
                raise ValidationError(f"asset lock staged file differs: {relative.as_posix()}")


def validate_ordinary_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    index_path = root / "release_index.json"
    manifest_path = root / "AUTHORITY_MANIFEST.json"
    if not index_path.is_file() or not manifest_path.is_file():
        raise ValidationError("ordinary-v3 authority release lacks its index or manifest")
    index = read_json(index_path)
    expected_index_hash = str(index.pop("release_index_hash", ""))
    if content_hash(index) != expected_index_hash:
        raise ValidationError("ordinary-v3 release index hash mismatch")
    if index.get("schema") != "org.aerocity.bench.authority-release-index.ordinary.v1":
        raise ValidationError("authority release index schema is not ordinary-v1")
    if index.get("formal_execution_level") != "L1":
        raise ValidationError("ordinary-v3 formal execution level changed")
    _verify_file_manifest(root, manifest_path)
    config = load_ordinary_config(root / "authority_private" / "release_config.json")
    if (
        config.config_hash != index.get("release_config_sha256")
        or config.version != index.get("release_version")
        or config.generator_version != index.get("generator_version")
    ):
        raise ValidationError("authority release differs from its frozen release configuration")
    execution_contract_hash = content_hash(config.raw["execution_contract"])
    public_execution_contract_hash = content_hash(
        public_execution_contract(config.raw["execution_contract"])
    )
    if execution_contract_hash != index.get("execution_contract_hash"):
        raise ValidationError("authority execution contract hash mismatch")
    if public_execution_contract_hash != index.get("public_execution_contract_hash"):
        raise ValidationError("authority public execution-contract hash mismatch")
    selected_splits = tuple(str(value) for value in index.get("selected_splits", []))
    if (
        not selected_splits
        or any(split not in ORDINARY_SPLITS for split in selected_splits)
        or tuple(split for split in ORDINARY_SPLITS if split in selected_splits) != selected_splits
    ):
        raise ValidationError("authority selected splits are invalid or out of canonical order")
    if tuple(index.get("formal_splits", [])) != FORMAL_SPLITS:
        raise ValidationError("authority formal split contract changed")
    if len({str(layout.get("layout_id")) for layout in index["layouts"]}) != len(index["layouts"]):
        raise ValidationError("authority release repeats a layout ID")
    layout_histogram = Counter(str(layout.get("split")) for layout in index["layouts"])
    expected_layout_histogram = Counter({split: config.count(split) for split in selected_splits})
    if layout_histogram != expected_layout_histogram:
        raise ValidationError("authority layout counts differ from the frozen configuration")
    legal = validate_release_legal_materials(root)
    if legal["legal_manifest_hash"] != index.get("legal_manifest_hash"):
        raise ValidationError("legal manifest hash differs from the release index")
    if legal["source_commit"] != index.get("source_commit"):
        raise ValidationError("legal materials and release index use different source commits")
    assets = config.raw["assets"]
    _validate_asset_lock(
        root,
        index,
        expected_bundle=str(assets["bundle"]),
        expected_ids={str(value) for value in assets["allowlist"]},
    )
    layout_count = 0
    episode_count = 0
    target_count = 0
    for layout in index["layouts"]:
        layout_id = str(layout["layout_id"])
        if not re.fullmatch(r"city-[0-9a-f]{16}", layout_id):
            raise ValidationError(f"authority layout ID is unsafe: {layout_id}")
        layout_root = root / "splits" / layout["split"] / layout["layout_id"]
        if not layout_root.is_dir():
            raise ValidationError(f"release index layout is absent: {layout_root}")
        _verify_file_manifest(layout_root, layout_root / "authority_manifest.json")
        city = read_json(layout_root / "scene_authority" / "cityspec.json")
        if city.get("layout_hash") != layout.get("layout_hash"):
            raise ValidationError("layout hash differs between index and CitySpec")
        if city.get("layout_id") != layout_id:
            raise ValidationError("layout ID differs between index and CitySpec")
        forbidden_city_keys = {"generation_seed", "split", "family_private", "spawn_grammar"}
        if forbidden_city_keys & set(_walk_keys(city)):
            raise ValidationError("scene-authority CitySpec retained private generator fields")
        _validate_task_spec(
            layout_root / "method_public" / "task_spec.json",
            layout_id=layout_id,
            public_execution_contract_hash=public_execution_contract_hash,
            fleet=config.raw["fleet"],
        )
        public_episode_paths = sorted((layout_root / "method_public" / "episodes").glob("*.json"))
        private_episode_paths = sorted(
            (layout_root / "evaluator_private" / "episodes").glob("*.json")
        )
        expected_episode_names = {
            f"episode-{index:04d}.json" for index in range(config.episodes(str(layout["split"])))
        }
        if {path.name for path in public_episode_paths} != expected_episode_names or {
            path.name for path in private_episode_paths
        } != expected_episode_names:
            raise ValidationError("layout episode files differ from the frozen configuration")
        for public_episode_path in public_episode_paths:
            _validate_public_episode(public_episode_path)
        for private_episode_path in private_episode_paths:
            episode = read_json(private_episode_path)
            expected_episode_hash = str(episode.pop("episode_hash", ""))
            if content_hash(episode) != expected_episode_hash:
                raise ValidationError(f"private episode hash mismatch: {private_episode_path}")
            validity = dict(episode["target_validity"])
            expected_validity_hash = str(validity.pop("validity_hash", ""))
            if content_hash(validity) != expected_validity_hash:
                raise ValidationError(f"target validity hash mismatch: {private_episode_path}")
            if episode["target_count"] != len(episode["targets"]):
                raise ValidationError(f"target count mismatch: {private_episode_path}")
            if len(episode["distractors"]) != len(episode["targets"]):
                raise ValidationError(f"distractor count mismatch: {private_episode_path}")
            target_ids = [str(target.get("target_id")) for target in episode["targets"]]
            if len(target_ids) != len(set(target_ids)):
                raise ValidationError(f"private episode repeats target IDs: {private_episode_path}")
            if any(
                not target.get("valid_before_run") or not target.get("legal_witnesses")
                for target in episode["targets"]
            ):
                raise ValidationError(
                    f"private episode contains an invalid target: {private_episode_path}"
                )
            start_errors = start_contract_errors(config, city, list(episode.get("starts", [])))
            if start_errors:
                raise ValidationError(
                    f"private episode violates the spawn contract: {private_episode_path}: "
                    + "; ".join(start_errors)
                )
            episode_count += 1
            target_count += int(episode["target_count"])
        layout_count += 1
    if index.get("native_isaac_gate") == "verified":
        native_hashes = []
        runtime_fingerprints = []
        for layout in index["layouts"]:
            layout_id = str(layout["layout_id"])
            stage_path = (
                root
                / "splits"
                / str(layout["split"])
                / layout_id
                / "scene_authority"
                / "stage.usda"
            )
            evidence = validate_native_gate_report(
                root / "release_evidence" / "native" / f"{layout_id}.json",
                stage_path,
                _native_input_bindings_for_layout(root, layout),
            )
            if (
                evidence.execution_level != "L1"
                or not evidence.formal_score_eligible
                or evidence.evidence_scope != FORMAL_L1_EVIDENCE_SCOPE
            ):
                raise ValidationError(
                    "promoted ordinary release contains capability-only native evidence"
                )
            native_hashes.append(evidence.report_hash)
            runtime_fingerprints.append(evidence.runtime_fingerprint)
        if content_hash(sorted(native_hashes)) != index.get("native_gate_set_hash"):
            raise ValidationError("native gate set hash mismatch")
        if not runtime_fingerprints or any(
            fingerprint != runtime_fingerprints[0] for fingerprint in runtime_fingerprints[1:]
        ):
            raise ValidationError("native runtime fingerprints differ within the release")
        if content_hash(runtime_fingerprints[0]) != index.get("runtime_fingerprint_hash"):
            raise ValidationError("native runtime fingerprint hash mismatch")
        source_stub = {
            "release_index_hash": index.get("promoted_from_release_index_hash"),
            "execution_contract_hash": index.get("execution_contract_hash"),
        }
        scientific_hash = validate_scientific_gate_report(
            root / "release_evidence" / "scientific_gate.json", source_stub
        )
        if scientific_hash != index.get("scientific_gate_hash"):
            raise ValidationError("scientific gate hash differs from the release index")
        if (
            index.get("scientific_status") != "release_candidate"
            or index.get("public_release_gate") != "ready_for_public_export"
        ):
            raise ValidationError("verified evidence is not reflected in release status")
    return {
        "schema": "org.aerocity.bench.validation-report.ordinary.v1",
        "status": "PASS",
        "release_version": index["release_version"],
        "layout_count": layout_count,
        "episode_count": episode_count,
        "target_count_private": target_count,
        "legal_gate": legal,
        "native_isaac_gate": index["native_isaac_gate"],
        "scientific_status": index["scientific_status"],
        "public_release_gate": index["public_release_gate"],
    }


def validate_scientific_gate_report(path: Path, source_index: dict[str, Any]) -> str:
    report = read_json(path)
    expected_hash = str(report.pop("scientific_gate_hash", ""))
    if content_hash(report) != expected_hash:
        raise ValidationError("scientific gate report hash mismatch")
    if report.get("schema") != "org.aerocity.bench.scientific-gate.ordinary.v1":
        raise ValidationError("scientific gate report schema is not ordinary-v1")
    if report.get("status") != "PASS":
        raise ValidationError("scientific gate report did not pass")
    if report.get("formal_results_accessed") is not False:
        raise ValidationError("scientific gate must be frozen before formal-result access")
    if report.get("calibration_split") != "calibration":
        raise ValidationError("scientific gate used a non-canonical calibration split")
    if report.get("source_release_index_hash") != source_index["release_index_hash"]:
        raise ValidationError("scientific gate belongs to another authority release")
    if report.get("execution_contract_hash") != source_index["execution_contract_hash"]:
        raise ValidationError("scientific gate used another execution contract")
    gates = report.get("gates", {})
    if set(gates) != set(SCIENTIFIC_GATES):
        raise ValidationError("scientific report does not contain every required gate")
    for name, result in gates.items():
        if result.get("status") != "PASS":
            raise ValidationError(f"scientific gate failed: {name}")
        _sha256_text(result.get("evidence_hash"), f"scientific gate {name} evidence_hash")
    if (
        not str(report.get("approved_by", "")).strip()
        or not str(report.get("approved_at", "")).strip()
    ):
        raise ValidationError("scientific gate lacks approval identity or time")
    return _sha256_text(expected_hash, "scientific_gate_hash")


def promote_ordinary_release(
    authority_root: Path,
    output: Path,
    *,
    native_report_dir: Path,
    scientific_report_path: Path,
) -> dict[str, Any]:
    """Create a new sealed release candidate; never mutate the pilot authority package."""

    authority_root = authority_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    validate_ordinary_release(authority_root)
    source_index = read_json(authority_root / "release_index.json")
    if source_index.get("public_release_gate") != ("eligible_after_native_and_scientific_gates"):
        raise ValidationError("only a complete, committed authority build can be promoted")
    if tuple(source_index.get("selected_splits", ())) != ORDINARY_SPLITS:
        raise ValidationError("promotion requires every canonical ordinary split")
    scientific_hash = validate_scientific_gate_report(
        scientific_report_path.resolve(), source_index
    )
    native_sources: dict[str, tuple[Path, str]] = {}
    runtime_fingerprints: list[dict[str, str]] = []
    for layout in source_index["layouts"]:
        layout_id = str(layout["layout_id"])
        report_path = native_report_dir.resolve() / f"{layout_id}.json"
        stage_path = (
            authority_root
            / "splits"
            / str(layout["split"])
            / layout_id
            / "scene_authority"
            / "stage.usda"
        )
        evidence = validate_native_gate_report(
            report_path,
            stage_path,
            _native_input_bindings_for_layout(authority_root, layout),
        )
        if (
            evidence.execution_level != "L1"
            or not evidence.formal_score_eligible
            or evidence.evidence_scope != FORMAL_L1_EVIDENCE_SCOPE
        ):
            raise ValidationError(
                f"ordinary promotion requires formal L1 episode evidence: {layout_id}"
            )
        native_sources[layout_id] = (report_path, evidence.report_hash)
        runtime_fingerprints.append(evidence.runtime_fingerprint)
    if not runtime_fingerprints or any(
        fingerprint != runtime_fingerprints[0] for fingerprint in runtime_fingerprints[1:]
    ):
        raise ValidationError("all native reports must use one frozen runtime fingerprint")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(authority_root, staging)
        evidence_root = staging / "release_evidence"
        (evidence_root / "native").mkdir(parents=True)
        shutil.copy2(scientific_report_path, evidence_root / "scientific_gate.json")
        promoted_native_hashes = []
        for layout in source_index["layouts"]:
            layout_id = str(layout["layout_id"])
            source_report = read_json(native_sources[layout_id][0])
            source_report.pop("native_gate_hash", None)
            relative_stage = (
                Path("splits") / str(layout["split"]) / layout_id / "scene_authority" / "stage.usda"
            )
            source_report["stage_path"] = relative_stage.as_posix()
            source_report["stage_sha256"] = file_hash(staging / relative_stage)
            source_report["native_gate_hash"] = content_hash(source_report)
            write_json(evidence_root / "native" / f"{layout_id}.json", source_report)
            promoted_native_hashes.append(source_report["native_gate_hash"])
        promoted_index = dict(source_index)
        source_index_hash = str(promoted_index.pop("release_index_hash"))
        promoted_index.update(
            {
                "promoted_from_release_index_hash": source_index_hash,
                "native_isaac_gate": "verified",
                "native_gate_set_hash": content_hash(sorted(promoted_native_hashes)),
                "runtime_fingerprint_hash": content_hash(runtime_fingerprints[0]),
                "scientific_status": "release_candidate",
                "scientific_gate_hash": scientific_hash,
                "public_release_gate": "ready_for_public_export",
            }
        )
        promoted_index["release_index_hash"] = content_hash(promoted_index)
        write_json(staging / "release_index.json", promoted_index)
        _write_directory_manifest(
            staging,
            staging / "AUTHORITY_MANIFEST.json",
            "org.aerocity.bench.authority-package-manifest.v1",
            excluded_relative_paths=frozenset({"AUTHORITY_MANIFEST.json"}),
        )
        os.replace(staging, output)
        report = validate_ordinary_release(output)
        return {
            **report,
            "native_gate_set_hash": promoted_index["native_gate_set_hash"],
            "scientific_gate_hash": scientific_hash,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _copy_public_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in PRIVATE_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.name not in {"AUTHORITY_MANIFEST.json", "authority_manifest.json"}:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def validate_public_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    index_path = root / "release_index.json"
    manifest_path = root / "PUBLIC_MANIFEST.json"
    if not index_path.is_file() or not manifest_path.is_file():
        raise ValidationError("public release lacks its index or manifest")
    index = read_json(index_path)
    expected_hash = str(index.pop("release_index_hash", ""))
    if content_hash(index) != expected_hash:
        raise ValidationError("public release index hash mismatch")
    if index.get("schema") != "org.aerocity.bench.public-release-index.ordinary.v1":
        raise ValidationError("public release schema is not ordinary-v1")
    _verify_file_manifest(root, manifest_path)
    legal = validate_release_legal_materials(root)
    if legal["legal_manifest_hash"] != index.get("legal_manifest_hash"):
        raise ValidationError("public legal manifest differs from the release index")
    if legal["source_commit"] != index.get("source_commit"):
        raise ValidationError("public legal materials use another source commit")
    if any(path.name in PRIVATE_DIRECTORY_NAMES for path in root.rglob("*")):
        raise ValidationError("public release retained an authority-private directory")
    if (root / "release_config.json").exists() or list(root.rglob("authority_manifest.json")):
        raise ValidationError("public release retained an authority-only configuration or manifest")
    runtime_contract = load_public_runtime_contract(root / "benchmark_contract.json")
    if (
        runtime_contract.version != index.get("release_version")
        or runtime_contract.contract_hash != index.get("runtime_contract_hash")
        or runtime_contract.raw["generator_version"] != index.get("generator_version")
        or content_hash(runtime_contract.raw["execution_contract"])
        != index.get("public_execution_contract_hash")
    ):
        raise ValidationError("public runtime contract differs from the release index")
    if index.get("formal_execution_level") != "L1":
        raise ValidationError("public release changed the formal execution level")
    _validate_asset_lock(
        root,
        index,
        expected_bundle=str(index.get("asset_bundle")),
        expected_ids={str(value) for value in index.get("asset_ids", [])},
    )
    development_count = 0
    layout_ids: set[str] = set()
    for layout in index["layouts"]:
        split = str(layout["split"])
        layout_id = str(layout["layout_id"])
        if split in FORMAL_SPLITS:
            raise ValidationError("public package exposes a formal blind layout")
        if split not in ORDINARY_SPLITS or not re.fullmatch(r"city-[0-9a-f]{16}", layout_id):
            raise ValidationError("public development layout has an invalid identity")
        if layout_id in layout_ids:
            raise ValidationError("public release repeats a development layout")
        layout_ids.add(layout_id)
        layout_root = root / "splits" / split / layout_id
        if not layout_root.is_dir() or (layout_root / "scene_authority").is_dir() is False:
            raise ValidationError("public development layout is incomplete")
        for episode_path in sorted((layout_root / "method_public" / "episodes").glob("*.json")):
            _validate_public_episode(episode_path)
        _validate_task_spec(
            layout_root / "method_public" / "task_spec.json",
            layout_id=layout_id,
            public_execution_contract_hash=str(index["public_execution_contract_hash"]),
            fleet=runtime_contract.raw["fleet"],
        )
        evaluator_root = layout_root / "development_evaluator"
        development_count += 1
        if not evaluator_root.is_dir():
            raise ValidationError("development split lacks its local evaluator data")
        if layout.get("formal_blind_required") is not False:
            raise ValidationError("development split is incorrectly marked blind")
    for split in FORMAL_SPLITS:
        if (root / "splits" / split).exists():
            raise ValidationError("public package contains a formal split directory")
    commitments = index.get("formal_blind_commitments", [])
    if not commitments or any(
        set(record) != {"split", "layout_commitment"}
        or record.get("split") not in FORMAL_SPLITS
        or len(str(record.get("layout_commitment", ""))) != 64
        for record in commitments
    ):
        raise ValidationError("formal blind layout commitments are incomplete")
    if set(index.get("formal_blind_splits", [])) != set(FORMAL_SPLITS):
        raise ValidationError("public formal blind split set changed")
    native_summary = read_json(root / "release_evidence" / "native_gate_summary.json")
    native_summary_hash = str(native_summary.pop("native_summary_hash", ""))
    if (
        content_hash(native_summary) != native_summary_hash
        or native_summary.get("status") != "PASS"
        or native_summary.get("execution_level") != "L1"
        or native_summary.get("native_gate_set_hash") != index.get("native_gate_set_hash")
        or native_summary.get("runtime_fingerprint_hash") != index.get("runtime_fingerprint_hash")
        or set(native_summary.get("required_checks", [])) != set(REQUIRED_NATIVE_CHECKS)
    ):
        raise ValidationError("public native gate summary is corrupt or incomplete")
    source_stub = {
        "release_index_hash": index.get("promoted_from_release_index_hash"),
        "execution_contract_hash": index.get("execution_contract_hash"),
    }
    scientific_hash = validate_scientific_gate_report(
        root / "release_evidence" / "scientific_gate.json", source_stub
    )
    if scientific_hash != index.get("scientific_gate_hash"):
        raise ValidationError("public scientific evidence differs from the release index")
    return {
        "schema": "org.aerocity.bench.public-validation-report.ordinary.v1",
        "status": "PASS",
        "release_version": index["release_version"],
        "layout_count": len(index["layouts"]),
        "development_layout_count": development_count,
        "formal_blind_layout_count": len(commitments),
        "evaluator_private_included": False,
        "formal_geometry_included": False,
    }


def export_public_release(authority_root: Path, output: Path) -> dict[str, Any]:
    authority_root = authority_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    report = validate_ordinary_release(authority_root)
    index = read_json(authority_root / "release_index.json")
    if report["public_release_gate"] != "ready_for_public_export":
        raise ValidationError(
            "authority release is not eligible for public export; native Isaac and "
            "scientific gates must pass on a complete committed build"
        )
    if (
        report["native_isaac_gate"] != "verified"
        or report["scientific_status"] != "release_candidate"
    ):
        raise ValidationError("public export requires verified native Isaac and scientific gates")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        for directory_name in ("_assets", "LICENSES"):
            shutil.copytree(authority_root / directory_name, staging / directory_name)
        for file_name in (
            "ASSET_BOM.json",
            "SBOM.cdx.json",
            "THIRD_PARTY_NOTICES.md",
            "DATA_LICENSE.md",
            "LEGAL_MANIFEST.json",
        ):
            shutil.copy2(authority_root / file_name, staging / file_name)
        private_config = load_ordinary_config(
            authority_root / "authority_private" / "release_config.json"
        )
        runtime_contract = {
            "schema": "org.aerocity.bench.runtime-contract-public.ordinary.v1",
            "release_version": index["release_version"],
            "generator_version": index["generator_version"],
            "fleet": private_config.raw["fleet"],
            "execution_contract": public_execution_contract(
                private_config.raw["execution_contract"]
            ),
            "authority_release_commitment": index["release_index_hash"],
        }
        runtime_contract["contract_hash"] = content_hash(runtime_contract)
        write_json(staging / "benchmark_contract.json", runtime_contract)
        public_layouts = []
        blind_commitments = []
        for layout in index["layouts"]:
            if layout["split"] in FORMAL_SPLITS:
                blind_commitments.append(
                    {
                        "split": layout["split"],
                        "layout_commitment": content_hash(
                            {
                                "layout_hash": layout["layout_hash"],
                                "promoted_release_index_hash": index["release_index_hash"],
                                "scientific_gate_hash": index["scientific_gate_hash"],
                            }
                        ),
                    }
                )
                continue
            public_layout = dict(layout)
            layout_root = staging / "splits" / layout["split"] / layout["layout_id"]
            source_layout_root = authority_root / "splits" / layout["split"] / layout["layout_id"]
            _copy_public_tree(source_layout_root, layout_root)
            source = source_layout_root / "evaluator_private"
            destination = layout_root / "development_evaluator"
            shutil.copytree(source, destination)
            public_contract_hash = content_hash(runtime_contract["execution_contract"])
            for episode_path in sorted((destination / "episodes").glob("*.json")):
                episode = read_json(episode_path)
                if not isinstance(episode, dict):
                    raise ValidationError(
                        f"development evaluator episode is not an object: {episode_path}"
                    )
                episode["execution_contract_hash"] = public_contract_hash
                episode_payload = dict(episode)
                episode_payload.pop("episode_hash", None)
                episode["episode_hash"] = content_hash(episode_payload)
                write_json(episode_path, episode)
            public_layout["development_evaluator_path"] = (
                f"splits/{layout['split']}/{layout['layout_id']}/development_evaluator"
            )
            public_layout["formal_blind_required"] = False
            public_layouts.append(public_layout)
        evidence_root = staging / "release_evidence"
        evidence_root.mkdir(parents=True)
        shutil.copy2(
            authority_root / "release_evidence" / "scientific_gate.json",
            evidence_root / "scientific_gate.json",
        )
        native_summary = {
            "schema": "org.aerocity.bench.native-gate-summary-public.v1",
            "status": "PASS",
            "execution_level": "L1",
            "layout_count_private": len(index["layouts"]),
            "native_gate_set_hash": index["native_gate_set_hash"],
            "runtime_fingerprint_hash": index["runtime_fingerprint_hash"],
            "required_checks": list(REQUIRED_NATIVE_CHECKS),
        }
        native_summary["native_summary_hash"] = content_hash(native_summary)
        write_json(evidence_root / "native_gate_summary.json", native_summary)
        asset_config = private_config.raw["assets"]
        public_index = {
            "schema": "org.aerocity.bench.public-release-index.ordinary.v1",
            "release_version": index["release_version"],
            "generator_version": index["generator_version"],
            "source_commit": index["source_commit"],
            "layouts": public_layouts,
            "formal_blind_commitments": blind_commitments,
            "formal_execution_level": "L1",
            "asset_lock_hash": index["asset_lock_hash"],
            "asset_bundle": asset_config["bundle"],
            "asset_ids": asset_config["allowlist"],
            "legal_manifest_hash": index["legal_manifest_hash"],
            "execution_contract_hash": index["execution_contract_hash"],
            "public_execution_contract_hash": content_hash(runtime_contract["execution_contract"]),
            "runtime_contract_hash": runtime_contract["contract_hash"],
            "native_gate_set_hash": index["native_gate_set_hash"],
            "runtime_fingerprint_hash": index["runtime_fingerprint_hash"],
            "scientific_gate_hash": index["scientific_gate_hash"],
            "promoted_from_release_index_hash": index["promoted_from_release_index_hash"],
            "development_evaluator_splits": [
                split for split in ORDINARY_SPLITS if split not in FORMAL_SPLITS
            ],
            "formal_blind_splits": list(FORMAL_SPLITS),
            "evaluator_private_included": False,
        }
        public_index["release_index_hash"] = content_hash(public_index)
        write_json(staging / "release_index.json", public_index)
        public_manifest = _write_directory_manifest(
            staging,
            staging / "PUBLIC_MANIFEST.json",
            "org.aerocity.bench.public-package-manifest.v1",
            excluded_relative_paths=frozenset({"PUBLIC_MANIFEST.json"}),
        )
        if any(path.name in PRIVATE_DIRECTORY_NAMES for path in staging.rglob("*")):
            raise ValidationError("public export retained a private directory")
        os.replace(staging, output)
        report = validate_public_release(output)
        return {**report, "file_count": len(public_manifest["files"])}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
