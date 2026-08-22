"""Fail-closed configuration for the ordinary-paper benchmark contract.

The legacy v2 loader remains available for historical pilot artifacts.  New
development uses this module and the ``org.aerocity.bench.release.ordinary.v3``
schema so resilience, perception, and formal-test semantics cannot silently
leak back into the ordinary-paper release.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_hash, read_json
from .planning_cadence import validate_planning_cadence

ORDINARY_SCHEMA = "org.aerocity.bench.release.ordinary.v3"
ORDINARY_SPLITS = (
    "train",
    "validation",
    "calibration",
    "test_iid",
    "test_topology",
    "test_process_ood",
)
FORMAL_SPLITS = ("test_iid", "test_topology", "test_process_ood")
DEVELOPMENT_PROCESSES = (
    "uniform_surface",
    "clustered_surface",
    "height_stratified",
)
OOD_PROCESSES = ("anisotropic_clustered_surface",)
_PUBLIC_EXECUTION_FORBIDDEN_KEY_FRAGMENTS = (
    "private",
    "target",
    "support",
    "witness",
    "evaluator",
    "split_label",
)


def _exact_keys(node: object, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - set(node))
    extra = sorted(set(node) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields differ; missing={missing}, extra={extra}")
    return node


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return number


def _range(node: object, name: str, *, lower: float | None = None) -> tuple[float, float]:
    if not isinstance(node, list) or len(node) != 2:
        raise ValueError(f"{name} must be a two-element list")
    low, high = float(node[0]), float(node[1])
    if low > high:
        raise ValueError(f"{name} must be non-decreasing")
    if lower is not None and low < lower:
        raise ValueError(f"{name} must be at least {lower}")
    return low, high


def _split_mapping(node: object, name: str) -> dict[str, Any]:
    value = _exact_keys(node, set(ORDINARY_SPLITS), name)
    return value


@dataclass(frozen=True)
class OrdinaryReleaseConfig:
    path: Path
    raw: dict[str, Any]
    config_hash: str

    @property
    def version(self) -> str:
        return str(self.raw["release_version"])

    @property
    def generator_version(self) -> str:
        return str(self.raw["generator_version"])

    @property
    def master_seed(self) -> int:
        return int(self.raw["master_seed"])

    @property
    def fleet_count(self) -> int:
        return int(self.raw["fleet"]["count"])

    @property
    def total_layouts(self) -> int:
        return sum(self.count(split) for split in ORDINARY_SPLITS)

    def count(self, split: str) -> int:
        if split not in ORDINARY_SPLITS:
            raise ValueError(f"unknown ordinary split: {split}")
        return int(self.raw["split_counts"][split])

    def episodes(self, split: str) -> int:
        if split not in ORDINARY_SPLITS:
            raise ValueError(f"unknown ordinary split: {split}")
        return int(self.raw["episodes_per_layout"][split])

    def target_processes(self, split: str) -> tuple[str, ...]:
        if split not in ORDINARY_SPLITS:
            raise ValueError(f"unknown ordinary split: {split}")
        return tuple(str(item) for item in self.raw["target_processes"]["by_split"][split])

    def target_range(self, size_m: int) -> tuple[int, int]:
        low, high = self.raw["target_count_ranges"][str(size_m)]
        return int(low), int(high)


@dataclass(frozen=True)
class PublicRuntimeConfig:
    """Redacted configuration sufficient for local development execution."""

    path: Path
    raw: dict[str, Any]
    contract_hash: str

    @property
    def version(self) -> str:
        return str(self.raw["release_version"])

    @property
    def fleet_count(self) -> int:
        return int(self.raw["fleet"]["count"])


def _validate_execution(node: object) -> None:
    execution = _exact_keys(
        node,
        {
            "canonical_profile",
            "formal_execution_level",
            "control_period_s",
            "planning_deadline_s",
            "planning",
            "clock",
            "episode",
            "observe",
            "sensor_rig",
            "vehicle",
            "communication",
            "safety",
        },
        "execution_contract",
    )
    if execution["canonical_profile"] != "G1_occupancy_voxel":
        raise ValueError("ordinary release must use G1_occupancy_voxel")
    if execution["formal_execution_level"] != "L1":
        raise ValueError("ordinary formal scores must use L1")
    period = _positive(execution["control_period_s"], "control_period_s")
    deadline = _positive(execution["planning_deadline_s"], "planning_deadline_s")
    if deadline > period:
        raise ValueError("planning_deadline_s cannot exceed the control period")
    validate_planning_cadence(
        execution["planning"],
        control_period_s=period,
        episode_duration_s=float(execution["episode"]["duration_s"]),
    )

    clock = _exact_keys(
        execution["clock"],
        {"basis", "overrun_policy", "max_consecutive_deadline_misses"},
        "execution_contract.clock",
    )
    if clock["basis"] != "simulated_execution_plus_compute_overrun":
        raise ValueError("planning compute must not receive a free task-clock pause")
    if clock["overrun_policy"] != "hold_last_safe_or_hover":
        raise ValueError("deadline overrun must hold the last safe action or hover")
    if int(clock["max_consecutive_deadline_misses"]) < 1:
        raise ValueError("max_consecutive_deadline_misses must be positive")

    episode = _exact_keys(
        execution["episode"],
        {"duration_s", "return_reserve_s", "fixed_target_count_private"},
        "execution_contract.episode",
    )
    duration = _positive(episode["duration_s"], "episode.duration_s")
    reserve = _positive(episode["return_reserve_s"], "episode.return_reserve_s")
    if reserve >= duration:
        raise ValueError("return reserve must be shorter than the episode")
    if episode["fixed_target_count_private"] is not True:
        raise ValueError("target count must remain evaluator-private")

    observe = _exact_keys(
        execution["observe"],
        {
            "exclusive_with_translation",
            "continuous_dwell_s",
            "cooldown_s",
            "max_linear_speed_mps",
            "max_angular_speed_deg_s",
            "max_pose_drift_m",
            "source_freshness_s",
            "max_range_m",
            "horizontal_fov_deg",
            "vertical_fov_deg",
            "surface_facing_min_cosine",
        },
        "execution_contract.observe",
    )
    if observe["exclusive_with_translation"] is not True:
        raise ValueError("OBSERVE must be exclusive with translation")
    for key in (
        "continuous_dwell_s",
        "cooldown_s",
        "max_linear_speed_mps",
        "max_angular_speed_deg_s",
        "max_pose_drift_m",
        "source_freshness_s",
        "max_range_m",
        "horizontal_fov_deg",
        "vertical_fov_deg",
    ):
        _positive(observe[key], f"observe.{key}")
    if float(observe["horizontal_fov_deg"]) > 180 or float(observe["vertical_fov_deg"]) > 180:
        raise ValueError("observe FoV cannot exceed 180 degrees")
    facing = float(observe["surface_facing_min_cosine"])
    if not 0 < facing <= 1:
        raise ValueError("surface_facing_min_cosine must lie in (0, 1]")

    rig_node = execution["sensor_rig"]
    if not isinstance(rig_node, dict):
        raise ValueError("execution_contract.sensor_rig must be an object")
    gimbal_mode = rig_node.get("gimbal_mode")
    expected_rig_keys = {
        "translation_body_m",
        "forward_axis",
        "up_axis",
        "gimbal_mode",
    }
    if gimbal_mode == "bounded":
        expected_rig_keys |= {"pitch_limits_deg", "max_pitch_rate_deg_s"}
    rig = _exact_keys(rig_node, expected_rig_keys, "execution_contract.sensor_rig")
    if not isinstance(rig["translation_body_m"], list) or len(rig["translation_body_m"]) != 3:
        raise ValueError("sensor translation must be a 3-vector")
    if rig["forward_axis"] != "+X" or rig["up_axis"] != "+Z":
        raise ValueError("ordinary virtual inspection sensor uses +X forward and +Z up")
    if rig["gimbal_mode"] not in {"fixed", "bounded"}:
        raise ValueError("gimbal_mode must be fixed or bounded")
    if rig["gimbal_mode"] == "bounded":
        limits = rig["pitch_limits_deg"]
        if (
            not isinstance(limits, list)
            or len(limits) != 2
            or not all(isinstance(value, (int, float)) for value in limits)
        ):
            raise ValueError("bounded gimbal pitch_limits_deg must be a numeric two-vector")
        lower, upper = (float(value) for value in limits)
        if not -180.0 <= lower < upper <= 180.0:
            raise ValueError("bounded gimbal pitch limits must lie in [-180, 180]")
        _positive(rig["max_pitch_rate_deg_s"], "sensor_rig.max_pitch_rate_deg_s")

    vehicle = _exact_keys(
        execution["vehicle"],
        {
            "radius_m",
            "horizontal_speed_mps",
            "vertical_speed_mps",
            "acceleration_mps2",
            "yaw_rate_deg_s",
            "minimum_clearance_m",
            "home_radius_m",
            "energy_budget_j",
            "energy_per_meter_j",
            "hover_power_w",
        },
        "execution_contract.vehicle",
    )
    for key, value in vehicle.items():
        _positive(value, f"vehicle.{key}")
    minimum_hover_endurance_j = duration * float(vehicle["hover_power_w"])
    if float(vehicle["energy_budget_j"]) + 1.0e-9 < minimum_hover_endurance_j:
        raise ValueError(
            "vehicle.energy_budget_j must cover full-episode hover until a LAND action "
            "is part of the execution contract"
        )

    communication = _exact_keys(
        execution["communication"],
        {"range_m", "bandwidth_bytes_s", "payload_bytes", "latency_s", "drop_probability", "ttl_s"},
        "execution_contract.communication",
    )
    for key in ("range_m", "bandwidth_bytes_s", "payload_bytes", "latency_s", "ttl_s"):
        _positive(communication[key], f"communication.{key}")
    drop = float(communication["drop_probability"])
    if not 0 <= drop < 1:
        raise ValueError("communication drop_probability must lie in [0, 1)")

    safety = _exact_keys(
        execution["safety"],
        {
            "hard_collision_agent_terminal",
            "hard_collision_rank_ineligible",
            "out_of_bounds_policy",
            "max_out_of_bounds_actions",
        },
        "execution_contract.safety",
    )
    if (
        safety["hard_collision_agent_terminal"] is not True
        or safety["hard_collision_rank_ineligible"] is not True
    ):
        raise ValueError("hard collisions must terminate the agent and invalidate safety ranking")
    if safety["out_of_bounds_policy"] != "reject_and_hover":
        raise ValueError("out-of-bounds actions must be rejected and replaced with hover")
    if int(safety["max_out_of_bounds_actions"]) < 1:
        raise ValueError("max_out_of_bounds_actions must be positive")


def _public_execution_key_errors(node: object, *, path: str = "execution_contract") -> list[str]:
    """Return semantic key paths that cannot cross a public method boundary."""

    errors: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            normalized = str(key).casefold().replace("-", "_")
            child_path = f"{path}.{key}"
            if any(
                fragment in normalized
                for fragment in _PUBLIC_EXECUTION_FORBIDDEN_KEY_FRAGMENTS
            ):
                errors.append(child_path)
            errors.extend(_public_execution_key_errors(value, path=child_path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            errors.extend(_public_execution_key_errors(value, path=f"{path}[{index}]"))
    return errors


def validate_public_execution_contract(node: object) -> None:
    """Validate the public projection of an ordinary execution contract.

    The full authority contract owns the private target-process invariant.  A
    method is allowed to know every operational limit needed for fair planning,
    but it must not receive that invariant or any future evaluator-private
    execution field.  Reconstructing the validated authority shape after the
    public-boundary check keeps the numerical/safety semantics identical.
    """

    forbidden = _public_execution_key_errors(node)
    if forbidden:
        raise ValueError(
            "public execution contract contains non-public fields: " + ", ".join(forbidden)
        )
    if not isinstance(node, dict):
        raise ValueError("public execution contract must be an object")
    authority_shape = deepcopy(node)
    episode = authority_shape.get("episode")
    if not isinstance(episode, dict):
        raise ValueError("public execution contract lacks an episode contract")
    episode["fixed_target_count_private"] = True
    _validate_execution(authority_shape)


def public_execution_contract(execution_contract: dict[str, Any]) -> dict[str, Any]:
    """Project an authority execution contract into a method-visible contract.

    This is intentionally centralized so task specs, public packages and
    process adapters cannot independently decide which authority fields are
    harmless to expose.
    """

    _validate_execution(execution_contract)
    projected = deepcopy(execution_contract)
    episode = projected.get("episode")
    if not isinstance(episode, dict) or "fixed_target_count_private" not in episode:
        raise ValueError("authority execution contract lacks its private target-count invariant")
    episode.pop("fixed_target_count_private")
    validate_public_execution_contract(projected)
    return projected


def _validate_processes(raw: dict[str, Any]) -> None:
    node = _exact_keys(raw["target_processes"], {"profiles", "by_split"}, "target_processes")
    expected = set(DEVELOPMENT_PROCESSES + OOD_PROCESSES)
    profiles = _exact_keys(node["profiles"], expected, "target_processes.profiles")
    if profiles["uniform_surface"] != {}:
        raise ValueError("uniform_surface accepts no parameters")
    for name in ("clustered_surface", "anisotropic_clustered_surface"):
        profile = profiles[name]
        required = {"cluster_count", "bandwidth_m"}
        if name == "anisotropic_clustered_surface":
            required |= {"vertical_scale"}
        _exact_keys(profile, required, f"target_processes.profiles.{name}")
        low, high = _range(profile["cluster_count"], f"{name}.cluster_count", lower=1)
        if int(low) != low or int(high) != high:
            raise ValueError(f"{name}.cluster_count must contain integers")
        _positive(profile["bandwidth_m"], f"{name}.bandwidth_m")
        if name == "anisotropic_clustered_surface":
            _positive(profile["vertical_scale"], f"{name}.vertical_scale")
    height = _exact_keys(
        profiles["height_stratified"],
        {"minimum_bands", "band_weights"},
        "target_processes.profiles.height_stratified",
    )
    if int(height["minimum_bands"]) < 3:
        raise ValueError("height_stratified must require at least three altitude bands")
    bands = {"near_ground", "lower", "mid", "elevated", "highrise"}
    weights = _exact_keys(height["band_weights"], bands, "height_stratified.band_weights")
    weight_sum = sum(
        _positive(value, f"height weight {key}", allow_zero=True) for key, value in weights.items()
    )
    if weight_sum <= 0:
        raise ValueError("height-stratified weights must have positive mass")

    by_split = _split_mapping(node["by_split"], "target_processes.by_split")
    development = set(DEVELOPMENT_PROCESSES)
    for split in ORDINARY_SPLITS:
        names = by_split[split]
        if not isinstance(names, list) or not names or len(names) != len(set(names)):
            raise ValueError(f"{split} target process list is empty or duplicated")
        if split == "test_process_ood":
            if set(names) != set(OOD_PROCESSES):
                raise ValueError("test_process_ood must contain only unseen process families")
        elif set(names) != development:
            raise ValueError(f"{split} must expose all development processes")
        if int(raw["episodes_per_layout"][split]) % len(names) != 0:
            raise ValueError(f"episodes for {split} must form complete process blocks")


def load_ordinary_config(path: Path) -> OrdinaryReleaseConfig:
    raw = read_json(path)
    required = {
        "schema",
        "release_kind",
        "release_version",
        "generator_version",
        "master_seed",
        "core_sizes_m",
        "topology_ood_size_m",
        "split_counts",
        "episodes_per_layout",
        "target_count_ranges",
        "city_grammar",
        "assets",
        "fleet",
        "execution_contract",
        "target_processes",
        "admission",
        "governance",
    }
    root = _exact_keys(raw, required, "ordinary release")
    if root["schema"] != ORDINARY_SCHEMA:
        raise ValueError(f"ordinary release must use {ORDINARY_SCHEMA}")
    if root["release_kind"] not in {"OFFICIAL", "CUSTOM"}:
        raise ValueError("release_kind must be OFFICIAL or CUSTOM")
    if not isinstance(root["master_seed"], int) or int(root["master_seed"]) < 0:
        raise ValueError("master_seed must be a non-negative integer")

    counts = _split_mapping(root["split_counts"], "split_counts")
    episodes = _split_mapping(root["episodes_per_layout"], "episodes_per_layout")
    for split in ORDINARY_SPLITS:
        if int(counts[split]) < 1 or int(episodes[split]) < 1:
            raise ValueError(f"{split} must contain at least one layout and episode")

    sizes = [int(item) for item in root["core_sizes_m"]]
    if not sizes or len(sizes) != len(set(sizes)) or any(not 48 <= item <= 224 for item in sizes):
        raise ValueError("core_sizes_m must be unique values in [48, 224]")
    topology_size = int(root["topology_ood_size_m"])
    if not 48 <= topology_size <= 224:
        raise ValueError("topology_ood_size_m must lie in [48, 224]")
    ranges = root["target_count_ranges"]
    expected_range_keys = {str(item) for item in sizes + [topology_size]}
    if not isinstance(ranges, dict) or set(ranges) != expected_range_keys:
        raise ValueError("target_count_ranges must match all configured sizes")
    for size, pair in ranges.items():
        low, high = _range(pair, f"target_count_ranges.{size}", lower=1)
        if int(low) != low or int(high) != high:
            raise ValueError("target count ranges must contain integers")

    grammar = _exact_keys(
        root["city_grammar"],
        {"development_families", "topology_ood_families", "spawn_grammars"},
        "city_grammar",
    )
    for key in ("development_families", "topology_ood_families", "spawn_grammars"):
        values = grammar[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"city_grammar.{key} must be non-empty and unique")
    if set(grammar["development_families"]) & set(grammar["topology_ood_families"]):
        raise ValueError("development and topology-OOD grammar families must be disjoint")

    assets = _exact_keys(
        root["assets"],
        {
            "bundle",
            "registry",
            "allowlist",
            "spdx_policy",
            "redistribute",
            "custom_official_forbidden",
        },
        "assets",
    )
    if not isinstance(assets["allowlist"], list) or not assets["allowlist"]:
        raise ValueError("assets.allowlist must be non-empty")
    if root["release_kind"] == "OFFICIAL":
        if assets["spdx_policy"] != "CC0-only" or assets["custom_official_forbidden"] is not True:
            raise ValueError("OFFICIAL releases must be CC0-only and reject custom assets")
        if assets["redistribute"] is not True:
            raise ValueError("OFFICIAL asset manifest must explicitly permit redistribution")

    fleet = _exact_keys(root["fleet"], {"profile", "count"}, "fleet")
    if int(fleet["count"]) != 4:
        raise ValueError("ordinary canonical fleet must contain four UAVs")
    _validate_execution(root["execution_contract"])
    _validate_processes(root)

    admission = _exact_keys(
        root["admission"],
        {
            "built_ratio",
            "road_ratio",
            "street_width_m",
            "minimum_support_to_target_ratio",
            "minimum_target_separation_m",
            "minimum_surrounding_colliders",
            "witness_distance_m",
            "minimum_vertical_span_m",
            "maximum_single_observation_target_fraction",
        },
        "admission",
    )
    for key in ("built_ratio", "road_ratio", "street_width_m"):
        _range(admission[key], f"admission.{key}", lower=0)
    for key in (
        "minimum_support_to_target_ratio",
        "minimum_target_separation_m",
        "minimum_surrounding_colliders",
        "witness_distance_m",
        "minimum_vertical_span_m",
        "maximum_single_observation_target_fraction",
    ):
        _positive(admission[key], f"admission.{key}")
    fraction = float(admission["maximum_single_observation_target_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("maximum_single_observation_target_fraction must lie in (0, 1]")

    governance = _exact_keys(
        root["governance"],
        {
            "plan_version",
            "calibration_split",
            "formal_splits",
            "calibration_formal_disjoint",
            "method_results_may_tune_benchmark",
            "formal_access",
        },
        "governance",
    )
    if governance["plan_version"] != "ordinary-paper-plan-v2":
        raise ValueError("ordinary config must bind ordinary-paper-plan-v2")
    if governance["calibration_split"] != "calibration":
        raise ValueError("calibration split must be explicit")
    if tuple(governance["formal_splits"]) != FORMAL_SPLITS:
        raise ValueError("formal_splits must use the canonical order")
    if governance["calibration_formal_disjoint"] is not True:
        raise ValueError("calibration and formal tests must be disjoint")
    if governance["method_results_may_tune_benchmark"] is not False:
        raise ValueError("method results cannot tune the benchmark")
    if governance["formal_access"] != "private_until_contract_freeze":
        raise ValueError("formal test must remain private until contract freeze")

    return OrdinaryReleaseConfig(path=path.resolve(), raw=root, config_hash=content_hash(root))


def load_public_runtime_contract(path: Path) -> PublicRuntimeConfig:
    root = _exact_keys(
        read_json(path),
        {
            "schema",
            "release_version",
            "generator_version",
            "fleet",
            "execution_contract",
            "authority_release_commitment",
            "contract_hash",
        },
        "public runtime contract",
    )
    expected_hash = str(root.pop("contract_hash", ""))
    if content_hash(root) != expected_hash:
        raise ValueError("public runtime contract hash mismatch")
    if root["schema"] != "org.aerocity.bench.runtime-contract-public.ordinary.v1":
        raise ValueError("public runtime contract schema is not ordinary-v1")
    fleet = _exact_keys(root["fleet"], {"profile", "count"}, "public runtime fleet")
    if int(fleet["count"]) != 4:
        raise ValueError("public runtime contract must use four UAVs")
    validate_public_execution_contract(root["execution_contract"])
    if len(str(root["authority_release_commitment"])) != 64:
        raise ValueError("public runtime contract lacks an authority commitment")
    runtime_raw = {
        "release_version": root["release_version"],
        "generator_version": root["generator_version"],
        "fleet": fleet,
        "execution_contract": root["execution_contract"],
    }
    return PublicRuntimeConfig(path=path.resolve(), raw=runtime_raw, contract_hash=expected_hash)
