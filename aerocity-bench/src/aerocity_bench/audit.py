"""Release integrity, split, privacy, target-process, and fault-pair audits."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json, write_json
from .config import EXPECTED_SPLITS, configured_fleet_count
from .errors import GenerationRejected, ValidationError

FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "split",
        "family",
        "generation_seed",
        "support_site_rules",
        "targets",
        "target_count",
        "target_seed",
        "target_process",
        "episode_seed",
        "episode_hash",
        "condition_group_id",
        "fault_spec",
        "affected_drone_ids",
        "witnesses",
    }
)


def _walk_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def audit_city_candidate(city: dict[str, Any], admission: dict[str, Any]) -> None:
    built_low, built_high = [float(value) for value in admission["built_ratio"]]
    road_low, road_high = [float(value) for value in admission["road_ratio"]]
    built = float(city["metrics"]["built_ratio"])
    road = float(city["metrics"]["road_ratio"])
    if not built_low <= built <= built_high:
        raise GenerationRejected(f"built ratio {built:.4f} failed admission")
    if not road_low <= road <= road_high:
        raise GenerationRejected(f"road ratio {road:.4f} failed admission")
    heights = {int(float(item["height_m"]) // 12) for item in city["buildings"]}
    if len(heights) < 3:
        raise GenerationRejected("city has fewer than three structural height bands")
    if not city["obstacles"]:
        raise GenerationRejected("city has no rubble support domain")
    if len(city["blocks"]) < 2 or len(city["buildings"]) < 2:
        raise GenerationRejected("city grammar is structurally degenerate")


def build_layout_manifest(public_dir: Path, city: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name in (
        "cityspec.json",
        "scene.usda",
        "collision.usda",
        "stage.usda",
        "public_catalogue.json",
        "coarse_prior.json",
    ):
        path = public_dir / name
        files[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return {
        "schema": "org.aerocity.bench.layout-manifest.v2",
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "topology_signature": city["topology_signature"],
        "asset_set_hash": city["asset_set_hash"],
        "public_files": files,
    }


def _validate_hashes(public_dir: Path, manifest: dict[str, Any]) -> None:
    for relative, node in manifest["public_files"].items():
        path = public_dir / relative
        if not path.is_file():
            raise ValidationError(f"missing public artifact: {path}")
        if file_hash(path) != node["sha256"]:
            raise ValidationError(f"public artifact hash mismatch: {path}")


def _episode_without_hash(episode: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in episode.items() if key != "episode_hash"}


def _paired_payload(episode: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_seed",
        "condition_group_id",
        "target_process",
        "target_count",
        "targets",
        "fleet_profile",
        "dynamics_profile",
        "starts",
        "smoke",
        "communication",
        "energy_budget_j",
    )
    return {key: episode[key] for key in keys}


def _target_process_control_payload(episode: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_seed",
        "condition_group_id",
        "target_count",
        "fleet_profile",
        "dynamics_profile",
        "starts",
        "fault_spec",
        "smoke",
        "communication",
        "energy_budget_j",
    )
    return {key: episode[key] for key in keys}


def _validate_fault(
    episode: dict[str, Any],
    starts: list[dict[str, Any]],
    split: str,
    release_config: dict[str, Any],
    path: Path,
) -> None:
    fault = episode.get("fault_spec")
    if not isinstance(fault, dict):
        raise ValidationError(f"missing fault specification: {path}")
    profile_name = fault.get("profile")
    allowed = release_config["faults"]["by_split"][split]
    profiles = release_config["faults"]["profiles"]
    if profile_name not in allowed or profile_name not in profiles:
        raise ValidationError(f"fault profile is not admitted in {split}: {path}")
    profile = profiles[profile_name]
    if fault.get("type") != profile["type"]:
        raise ValidationError(f"fault type/profile mismatch: {path}")
    if fault.get("method_observability") != "health_signals_only_no_evaluator_truth":
        raise ValidationError(f"fault observability contract failed: {path}")
    start_ids = {start["drone_id"] for start in starts}
    affected = fault.get("affected_drone_ids", [])
    if len(affected) != int(profile["affected_count"]) or not set(affected) <= start_ids:
        raise ValidationError(f"fault affected-agent contract failed: {path}")
    if profile["type"] == "none":
        if fault.get("onset_fraction") is not None:
            raise ValidationError(f"no-fault episode has an onset: {path}")
    else:
        onset = float(fault.get("onset_fraction", -1.0))
        low, high = [float(value) for value in profile["onset_fraction"]]
        if not low <= onset <= high:
            raise ValidationError(f"fault onset lies outside the committed interval: {path}")
    if profile["type"] == "temporary_communication_isolation":
        duration = float(fault.get("duration_fraction", -1.0))
        low, high = [float(value) for value in profile["duration_fraction"]]
        if not low <= duration <= high or fault.get("recoverable") is not True:
            raise ValidationError(f"temporary communication fault contract failed: {path}")
    elif profile["type"] == "observation_channel_loss":
        if fault.get("channel") != profile["channel"] or fault.get("recoverable") is not False:
            raise ValidationError(f"observation-channel fault contract failed: {path}")
    elif profile["type"] == "hard_loss" and fault.get("recoverable") is not False:
        raise ValidationError(f"hard loss cannot be recoverable: {path}")


def _validate_start(city: dict[str, Any], start: dict[str, Any], path: Path) -> None:
    position = start.get("position", [])
    if not isinstance(position, list) or len(position) != 3:
        raise ValidationError(f"start has an invalid 3-D position: {path}")
    x, y, z = [float(value) for value in position]
    lower = [float(value) for value in city["flight_bounds"]["minimum"]]
    upper = [float(value) for value in city["flight_bounds"]["maximum"]]
    if not all(
        low <= value <= high for low, value, high in zip(lower, (x, y, z), upper, strict=True)
    ):
        raise ValidationError(f"start lies outside flight bounds: {path}")
    on_vertical = any(
        road["axis"] == "x" and abs(x - float(road["x"])) <= float(road["width_m"]) / 2
        for road in city["roads"]
    )
    on_horizontal = any(
        road["axis"] == "y" and abs(y - float(road["y"])) <= float(road["width_m"]) / 2
        for road in city["roads"]
    )
    if not on_vertical or not on_horizontal:
        raise ValidationError(f"start does not lie in an audited road intersection: {path}")
    for building in city["buildings"]:
        for component in building["components"]:
            cx, cy, cz = [float(value) for value in component["center"]]
            sx, sy, sz = [float(value) for value in component["size"]]
            if (
                abs(x - cx) <= sx / 2 + 0.3
                and abs(y - cy) <= sy / 2 + 0.3
                and abs(z - cz) <= sz / 2 + 0.3
            ):
                raise ValidationError(f"start intersects a building component: {path}")


def _validate_private(
    layout_dir: Path,
    city: dict[str, Any],
    record: dict[str, Any],
    release_config: dict[str, Any],
) -> tuple[int, int]:
    private_dir = layout_dir / "evaluator_private"
    support_path = private_dir / "support_sites.json"
    if not support_path.is_file():
        raise ValidationError(f"missing private support sites for {city['layout_id']}")
    support = read_json(support_path)
    sites = support.get("support_sites", [])
    sites_by_id = {item["site_id"]: item for item in sites}
    if len(sites_by_id) != len(sites):
        raise ValidationError(f"duplicate private support-site IDs for {city['layout_id']}")
    episodes = sorted((private_dir / "episodes").glob("*.json"))
    if not episodes:
        raise ValidationError(f"missing private episodes for {city['layout_id']}")
    split = str(record["split"])
    admission = release_config["admission"]
    minimum_ratio = float(admission["minimum_support_to_target_ratio"])
    minimum_separation = float(admission["minimum_target_separation_m"])
    fleet_count = configured_fleet_count(release_config, split)
    allowed_processes = set(release_config["target_processes"]["by_split"][split])
    process_profiles = release_config["target_processes"]["profiles"]
    paired_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paired_faults: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    target_process_groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    target_process_targets: dict[str, set[str]] = defaultdict(set)
    target_total = 0
    for path in episodes:
        episode = read_json(path)
        if episode.get("schema") != "org.aerocity.bench.episode-private.v2":
            raise ValidationError(f"private episode has the wrong schema: {path}")
        if episode.get("layout_hash") != city["layout_hash"]:
            raise ValidationError(f"episode layout mismatch: {path}")
        if episode.get("episode_hash") != content_hash(_episode_without_hash(episode)):
            raise ValidationError(f"episode hash mismatch: {path}")
        process_name = episode.get("target_process")
        if process_name not in allowed_processes:
            raise ValidationError(f"target process is not admitted in {split}: {path}")
        targets = episode.get("targets", [])
        if int(episode.get("target_count", -1)) != len(targets):
            raise ValidationError(f"target count mismatch: {path}")
        if len(sites) < math.ceil(minimum_ratio * len(targets)):
            raise ValidationError(f"support-to-target ratio failed: {path}")
        for target in targets:
            site = sites_by_id.get(target.get("site_id"))
            if site is None or target.get("position") != site.get("position"):
                raise ValidationError(f"invalid 3-D target support reference: {path}")
            if target.get("support_class") != site.get("support_class"):
                raise ValidationError(f"target support class mismatch: {path}")
        for first_index, first in enumerate(targets):
            for second in targets[first_index + 1 :]:
                if math.dist(first["position"], second["position"]) + 1e-9 < minimum_separation:
                    raise ValidationError(f"target separation failed: {path}")
        if process_name == "height_stratified":
            required_bands = min(int(process_profiles[process_name]["minimum_bands"]), len(targets))
            if len({target["altitude_band"] for target in targets}) < required_bands:
                raise ValidationError(f"height-stratified band contract failed: {path}")
        starts = episode.get("starts", [])
        if len(starts) != fleet_count or len({item["drone_id"] for item in starts}) != fleet_count:
            raise ValidationError(f"fleet/start count mismatch: {path}")
        for start in starts:
            _validate_start(city, start, path)
        fleet_profile = episode.get("fleet_profile", {})
        if int(fleet_profile.get("count", -1)) != fleet_count:
            raise ValidationError(f"episode fleet profile mismatch: {path}")
        _validate_fault(episode, starts, split, release_config, path)
        if split == "test_resilience":
            group = str(episode["condition_group_id"])
            profile = str(episode["fault_spec"]["profile"])
            if profile in paired_groups[group]:
                raise ValidationError(f"duplicate fault profile in resilience pair: {path}")
            paired_groups[group][profile] = _paired_payload(episode)
            paired_faults[group][profile] = episode["fault_spec"]
        if split == "test_target_process":
            group = str(episode["condition_group_id"])
            process = str(episode["target_process"])
            if process in target_process_groups[group]:
                raise ValidationError(f"duplicate target process in intervention group: {path}")
            target_process_groups[group][process] = _target_process_control_payload(episode)
            target_process_targets[group].add(content_hash(episode["targets"]))
        target_total += len(targets)
    if split == "test_resilience":
        expected_profiles = set(release_config["faults"]["by_split"][split])
        fault_profiles = release_config["faults"]["profiles"]
        hard_one_name = next(
            name
            for name in expected_profiles
            if fault_profiles[name]["type"] == "hard_loss"
            and int(fault_profiles[name]["affected_count"]) == 1
        )
        hard_two_name = next(
            name
            for name in expected_profiles
            if fault_profiles[name]["type"] == "hard_loss"
            and int(fault_profiles[name]["affected_count"]) == 2
        )
        for group, payloads in paired_groups.items():
            if set(payloads) != expected_profiles:
                raise ValidationError(f"incomplete resilience pair block {group}")
            hashes = {content_hash(payload) for payload in payloads.values()}
            if len(hashes) != 1:
                raise ValidationError(f"resilience interventions are not paired in {group}")
            hard_one = paired_faults[group][hard_one_name]
            hard_two = paired_faults[group][hard_two_name]
            if not set(hard_one["affected_drone_ids"]) <= set(hard_two["affected_drone_ids"]):
                raise ValidationError(f"hard-loss affected sets are not nested in {group}")
            if hard_one["onset_fraction"] != hard_two["onset_fraction"]:
                raise ValidationError(f"hard-loss onsets are not paired in {group}")
    if split == "test_target_process":
        expected_processes = set(release_config["target_processes"]["by_split"][split])
        for group, payloads in target_process_groups.items():
            if set(payloads) != expected_processes:
                raise ValidationError(f"incomplete target-process intervention group {group}")
            if len({content_hash(payload) for payload in payloads.values()}) != 1:
                raise ValidationError(f"target-process controls differ inside {group}")
            if len(target_process_targets[group]) != len(expected_processes):
                raise ValidationError(
                    f"target-process intervention did not change targets in {group}"
                )
    return len(episodes), target_total


def validate_release(root: Path, write_report: bool = True) -> dict[str, Any]:
    root = root.resolve()
    index_path = root / "release_index.json"
    if not index_path.is_file():
        raise ValidationError(f"not an AeroCityBench release: {root}")
    index = read_json(index_path)
    expected_index_hash = index.get("release_index_hash")
    index_without_hash = {key: value for key, value in index.items() if key != "release_index_hash"}
    if expected_index_hash != content_hash(index_without_hash):
        raise ValidationError("release index hash mismatch")
    release_config = index["effective_release_config"]
    seen_layouts: set[str] = set()
    seen_seeds: set[int] = set()
    split_topologies: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    sizes: Counter[int] = Counter()
    target_total = 0
    episode_total = 0
    for record in index["layouts"]:
        split = record["split"]
        layout_dir = root / "splits" / split / record["layout_id"]
        public_dir = layout_dir / "public"
        manifest = read_json(public_dir / "layout_manifest.json")
        _validate_hashes(public_dir, manifest)
        city = read_json(public_dir / "cityspec.json")
        if city.get("schema") != "org.aerocity.bench.cityspec.v2":
            raise ValidationError(f"public CitySpec has the wrong schema: {record['layout_id']}")
        if (
            city["layout_hash"] != record["layout_hash"]
            or manifest["layout_hash"] != city["layout_hash"]
        ):
            raise ValidationError(f"layout lineage mismatch: {record['layout_id']}")
        if city["layout_hash"] in seen_layouts:
            raise ValidationError(f"duplicate layout hash: {city['layout_hash']}")
        generation_seed = int(record["generation_seed"])
        if generation_seed in seen_seeds:
            raise ValidationError(f"duplicate generation seed: {generation_seed}")
        seen_layouts.add(city["layout_hash"])
        seen_seeds.add(generation_seed)
        public_keys: set[str] = set()
        for public_json in public_dir.glob("*.json"):
            public_keys.update(_walk_keys(read_json(public_json)))
        leaked = sorted(FORBIDDEN_PUBLIC_KEYS & public_keys)
        if leaked:
            raise ValidationError(f"private truth keys leaked in {public_dir}: {leaked}")
        private_episodes, private_targets = _validate_private(
            layout_dir, city, record, release_config
        )
        episode_total += private_episodes
        target_total += private_targets
        counts[split] += 1
        families[str(record["family"])] += 1
        sizes[int(city["size_m"])] += 1
        split_topologies[split].add(str(record["topology_signature"]))
    topology_ood = split_topologies["test_topology"]
    for split in EXPECTED_SPLITS:
        if split != "test_topology" and topology_ood & split_topologies[split]:
            raise ValidationError(f"topology OOD signature leaks into {split}")
    expected_counts = release_config["split_counts"]
    for split in index["selected_splits"]:
        if counts[split] != int(expected_counts[split]):
            raise ValidationError(
                f"split {split} contains {counts[split]} layouts, expected {expected_counts[split]}"
            )
    report = {
        "schema": "org.aerocity.bench.validation-report.v2",
        "status": "passed",
        "release_version": index["release_version"],
        "layout_count": sum(counts.values()),
        "episode_count": episode_total,
        "target_instance_count": target_total,
        "split_counts": dict(sorted(counts.items())),
        "family_counts": dict(sorted(families.items())),
        "size_counts": {str(key): sizes[key] for key in sorted(sizes)},
        "privacy_gate": "passed",
        "integrity_gate": "passed",
        "target_process_gate": "passed",
        "fault_pairing_gate": "passed",
        "native_isaac_gate": "not_run",
        "scientific_status": "pilot_only",
    }
    if write_report:
        write_json(root / "audit" / "validation_report.json", report)
    return report
