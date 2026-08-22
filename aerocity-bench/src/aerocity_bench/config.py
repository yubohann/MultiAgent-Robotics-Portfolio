"""Release configuration loading and fail-closed validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_hash, read_json

EXPECTED_SPLITS = (
    "train",
    "validation",
    "test_iid",
    "test_topology",
    "test_target_process",
    "test_resilience",
    "test_scale",
)
TARGET_PROCESS_NAMES = frozenset({"uniform_surface", "clustered_surface", "height_stratified"})
TRAINING_FAMILY_NAMES = frozenset({"grid_corridor", "mixed_blocks", "industrial_spine"})
FAULT_TYPES = frozenset(
    {
        "none",
        "temporary_communication_isolation",
        "hard_loss",
        "observation_channel_loss",
    }
)
TASK_VIEWS = {
    "exploration-3d": "supported",
    "geometry-search-3d": "primary",
    "perception-search-3d": "optional",
}
OBSERVATION_TIERS = (
    "T0_state_prior",
    "T1_occupancy_voxel",
    "T2_range_fov",
    "T3_rgbd",
    "T4_full_perception",
)


@dataclass(frozen=True)
class ReleaseConfig:
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
    def total_layouts(self) -> int:
        return sum(int(self.raw["split_counts"][name]) for name in EXPECTED_SPLITS)

    @property
    def fleet_count(self) -> int:
        return int(self.raw["fleet"]["count"])

    def fleet_count_for_split(self, split: str) -> int:
        return configured_fleet_count(self.raw, split)

    @property
    def dynamics_profile(self) -> str:
        return str(self.raw["dynamics"]["active_profile"])

    def count(self, split: str) -> int:
        return int(self.raw["split_counts"][split])

    def episodes(self, split: str) -> int:
        return int(self.raw["episodes_per_layout"][split])

    def target_range(self, size_m: int) -> tuple[int, int]:
        low, high = self.raw["target_count_ranges"][str(size_m)]
        return int(low), int(high)

    def target_processes(self, split: str) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["target_processes"]["by_split"][split])

    def fault_profiles(self, split: str) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["faults"]["by_split"][split])


def configured_fleet_count(raw: dict[str, Any], split: str) -> int:
    """Return the official fleet size for a generated split.

    The full baseline matrix uses the core fleet.  Resilience is deliberately
    generated with a larger fleet so one- and two-agent loss measure
    recoordination instead of almost entirely measuring lost capacity.
    """

    if split not in EXPECTED_SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test_resilience":
        return int(raw["evaluation_tracks"]["fleet"]["resilience_count"])
    return int(raw["fleet"]["count"])


def _number_pair(value: object, name: str, *, allow_equal: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list")
    low, high = float(value[0]), float(value[1])
    if low > high or (low == high and not allow_equal):
        raise ValueError(f"{name} must be {'non-decreasing' if allow_equal else 'increasing'}")
    return low, high


def _validate_ordered_split_mapping(node: object, name: str) -> dict[str, Any]:
    if not isinstance(node, dict) or tuple(node.keys()) != EXPECTED_SPLITS:
        raise ValueError(f"{name} must contain the canonical split order")
    return node


def _validate_target_processes(raw: dict[str, Any]) -> None:
    node = raw["target_processes"]
    if not isinstance(node, dict) or set(node) != {"profiles", "by_split"}:
        raise ValueError("target_processes must contain profiles and by_split")
    profiles = node["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != TARGET_PROCESS_NAMES:
        raise ValueError("target-process profiles must define the canonical three processes")
    clustered = profiles["clustered_surface"]
    if profiles["uniform_surface"] != {}:
        raise ValueError("uniform_surface does not accept parameters")
    if not isinstance(clustered, dict) or set(clustered) != {"cluster_count", "bandwidth_m"}:
        raise ValueError("clustered_surface has the wrong fields")
    cluster_low, cluster_high = _number_pair(clustered.get("cluster_count"), "cluster_count")
    if int(cluster_low) != cluster_low or int(cluster_high) != cluster_high or cluster_low < 1:
        raise ValueError("cluster_count must contain positive integers")
    if float(clustered.get("bandwidth_m", 0.0)) <= 0:
        raise ValueError("clustered_surface bandwidth_m must be positive")
    height_stratified = profiles["height_stratified"]
    if not isinstance(height_stratified, dict) or set(height_stratified) != {
        "minimum_bands",
        "band_weights",
    }:
        raise ValueError("height_stratified has the wrong fields")
    if not 1 <= int(height_stratified["minimum_bands"]) <= 5:
        raise ValueError("height_stratified minimum_bands must lie in [1, 5]")
    weights = height_stratified.get("band_weights")
    if not isinstance(weights, dict) or set(weights) != {
        "near_ground",
        "lower",
        "mid",
        "elevated",
        "highrise",
    }:
        raise ValueError("height_stratified must define all altitude-band weights")
    if (
        any(float(value) < 0 for value in weights.values())
        or sum(map(float, weights.values())) <= 0
    ):
        raise ValueError("height-stratified weights must be non-negative and non-zero")
    by_split = _validate_ordered_split_mapping(node["by_split"], "target_processes.by_split")
    for split, names in by_split.items():
        if not isinstance(names, list) or not names or not set(names) <= TARGET_PROCESS_NAMES:
            raise ValueError(f"invalid target processes for {split}")
        if int(raw["episodes_per_layout"][split]) % len(names) != 0:
            raise ValueError(f"episodes in {split} must form complete target-process blocks")
    if "height_stratified" in set(by_split["train"]):
        raise ValueError("height_stratified is reserved for target-process evaluation")
    if set(by_split["test_target_process"]) != TARGET_PROCESS_NAMES:
        raise ValueError("test_target_process must compare all three target processes")
    if len(by_split["test_resilience"]) != 1:
        raise ValueError("test_resilience must isolate faults under one target process")


def _validate_evaluation_tracks(raw: dict[str, Any]) -> None:
    node = raw["evaluation_tracks"]
    if not isinstance(node, dict) or set(node) != {
        "primary_task",
        "task_views",
        "fleet",
    }:
        raise ValueError("evaluation_tracks has the wrong fields")
    if node["primary_task"] != "geometry-search-3d":
        raise ValueError("geometry-search-3d must remain the Paper I primary task")
    if node["task_views"] != TASK_VIEWS:
        raise ValueError("evaluation task views or statuses differ from the benchmark contract")
    fleet = node["fleet"]
    if not isinstance(fleet, dict) or set(fleet) != {
        "core_count",
        "scaling_counts",
        "resilience_count",
        "resilience_loss_counts",
        "conditional_stress_counts",
        "custom_count_range",
    }:
        raise ValueError("evaluation_tracks.fleet has the wrong fields")
    core = int(fleet["core_count"])
    resilience = int(fleet["resilience_count"])
    if core != int(raw["fleet"]["count"]):
        raise ValueError("evaluation core_count must equal fleet.count")
    if not 1 <= core <= 32 or not 1 <= resilience <= 32:
        raise ValueError("official fleet counts must lie in [1, 32]")
    scaling = [int(value) for value in fleet["scaling_counts"]]
    if (
        not scaling
        or scaling != sorted(set(scaling))
        or any(value < 1 or value > 32 for value in scaling)
    ):
        raise ValueError("scaling_counts must be sorted unique values in [1, 32]")
    if core not in scaling:
        raise ValueError("scaling_counts must include the core fleet")
    loss_counts = [int(value) for value in fleet["resilience_loss_counts"]]
    if loss_counts != sorted(set(loss_counts)) or not loss_counts or loss_counts[0] != 0:
        raise ValueError("resilience_loss_counts must be sorted, unique, and include zero")
    if any(value < 0 or value >= resilience for value in loss_counts):
        raise ValueError("resilience loss counts must leave at least one survivor")
    stress = [int(value) for value in fleet["conditional_stress_counts"]]
    if stress != sorted(set(stress)) or any(
        value <= max(scaling) or value > 32 for value in stress
    ):
        raise ValueError("conditional stress counts must exceed scaling counts and be at most 32")
    custom_low, custom_high = _number_pair(
        fleet["custom_count_range"], "custom_count_range", allow_equal=True
    )
    if (
        int(custom_low) != custom_low
        or int(custom_high) != custom_high
        or custom_low < 1
        or custom_high > 32
    ):
        raise ValueError("custom_count_range must contain integers in [1, 32]")


def _validate_observation_contract(raw: dict[str, Any]) -> None:
    node = raw["observation_contract"]
    if not isinstance(node, dict) or set(node) != {
        "tiers",
        "leaderboards",
        "geometry_confirmation",
        "perception_confirmation",
        "cross_tier_ranking",
    }:
        raise ValueError("observation_contract has the wrong fields")
    if tuple(node["tiers"]) != OBSERVATION_TIERS:
        raise ValueError("observation tiers must use the canonical capability order")
    leaderboards = node["leaderboards"]
    if not isinstance(leaderboards, dict) or set(leaderboards) != set(TASK_VIEWS):
        raise ValueError("leaderboards must define every task view")
    known_tiers = set(OBSERVATION_TIERS)
    for task, tiers in leaderboards.items():
        if not isinstance(tiers, list) or not tiers or not set(tiers) <= known_tiers:
            raise ValueError(f"invalid observation tiers for {task}")
        if len(tiers) != len(set(tiers)):
            raise ValueError(f"duplicate observation tiers for {task}")
    if "T4_full_perception" in leaderboards["geometry-search-3d"]:
        raise ValueError("full perception submissions belong on perception-search-3d")
    if leaderboards["perception-search-3d"] != ["T4_full_perception"]:
        raise ValueError("perception-search-3d must isolate T4_full_perception")
    geometry = node["geometry_confirmation"]
    if not isinstance(geometry, dict) or set(geometry) != {
        "max_range_m",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "minimum_dwell_s",
        "require_line_of_sight",
        "require_surface_facing",
        "require_source_observation_id",
        "allow_distance_only",
    }:
        raise ValueError("geometry_confirmation has the wrong fields")
    for key in (
        "max_range_m",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "minimum_dwell_s",
    ):
        if float(geometry[key]) <= 0:
            raise ValueError(f"geometry_confirmation.{key} must be positive")
    if float(geometry["horizontal_fov_deg"]) > 180 or float(geometry["vertical_fov_deg"]) > 180:
        raise ValueError("geometry confirmation FoV must not exceed 180 degrees")
    for key in (
        "require_line_of_sight",
        "require_surface_facing",
        "require_source_observation_id",
    ):
        if geometry[key] is not True:
            raise ValueError(f"geometry_confirmation.{key} must fail closed")
    if geometry["allow_distance_only"] is not False:
        raise ValueError("distance-only target confirmation is forbidden")
    perception = node["perception_confirmation"]
    if perception != {
        "requires_detector_output": True,
        "separate_leaderboard": True,
        "geometry_oracle_autofill": False,
    }:
        raise ValueError("perception confirmation must remain detector-grounded and separate")
    if node["cross_tier_ranking"] != "forbidden":
        raise ValueError("cross-tier ranking must be forbidden")


def _validate_faults(raw: dict[str, Any]) -> None:
    node = raw["faults"]
    if not isinstance(node, dict) or set(node) != {"profiles", "by_split"}:
        raise ValueError("faults must contain profiles and by_split")
    profiles = node["profiles"]
    if not isinstance(profiles, dict) or "none" not in profiles:
        raise ValueError("fault profiles must define none")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("type") not in FAULT_TYPES:
            raise ValueError(f"invalid fault profile: {name}")
        expected_fields = {
            "none": {"type", "affected_count"},
            "temporary_communication_isolation": {
                "type",
                "affected_count",
                "onset_fraction",
                "duration_fraction",
            },
            "hard_loss": {"type", "affected_count", "onset_fraction"},
            "observation_channel_loss": {
                "type",
                "affected_count",
                "onset_fraction",
                "channel",
            },
        }[str(profile["type"])]
        if set(profile) != expected_fields:
            raise ValueError(f"fault profile {name} has the wrong fields")
        affected = int(profile.get("affected_count", -1))
        if affected < 0 or affected > configured_fleet_count(raw, "test_resilience"):
            raise ValueError(f"invalid affected_count for fault profile {name}")
        if profile["type"] == "none" and affected != 0:
            raise ValueError("none fault profile cannot affect an agent")
        if profile["type"] != "none" and affected == 0:
            raise ValueError(f"fault profile {name} must affect at least one agent")
        if profile["type"] != "none":
            low, high = _number_pair(
                profile.get("onset_fraction"), f"{name}.onset_fraction", allow_equal=True
            )
            if low < 0 or high > 1:
                raise ValueError(f"{name}.onset_fraction must lie in [0, 1]")
        if "duration_fraction" in profile:
            low, high = _number_pair(
                profile["duration_fraction"],
                f"{name}.duration_fraction",
                allow_equal=True,
            )
            if low <= 0 or high > 1:
                raise ValueError(f"{name}.duration_fraction must lie in (0, 1]")
    by_split = _validate_ordered_split_mapping(node["by_split"], "faults.by_split")
    for split, names in by_split.items():
        if not isinstance(names, list) or not names or not set(names) <= set(profiles):
            raise ValueError(f"invalid fault profiles for {split}")
        if split != "test_resilience" and names != ["none"]:
            raise ValueError(f"fault injection is only permitted in test_resilience, not {split}")
    resilience = by_split["test_resilience"]
    if "none" not in resilience or not any(
        profiles[name]["type"] == "hard_loss" for name in resilience
    ):
        raise ValueError("test_resilience needs paired no-fault and hard-loss profiles")
    if int(raw["episodes_per_layout"]["test_resilience"]) % len(resilience) != 0:
        raise ValueError("test_resilience episodes must form complete paired fault blocks")
    if profiles["none"] != {"type": "none", "affected_count": 0}:
        raise ValueError("the canonical none profile must be an empty intervention")
    resilience_types = {profiles[name]["type"] for name in resilience}
    if resilience_types != {
        "none",
        "temporary_communication_isolation",
        "hard_loss",
        "observation_channel_loss",
    }:
        raise ValueError("test_resilience must include all mandatory v1 fault types")
    hard_counts = {
        int(profiles[name]["affected_count"])
        for name in resilience
        if profiles[name]["type"] == "hard_loss"
    }
    if hard_counts != {1, 2}:
        raise ValueError("test_resilience must include one- and two-agent hard loss")


def _validate_dynamics(raw: dict[str, Any]) -> None:
    node = raw["dynamics"]
    profiles = node.get("profiles") if isinstance(node, dict) else None
    active = node.get("active_profile") if isinstance(node, dict) else None
    if not isinstance(profiles, dict) or active not in profiles:
        raise ValueError("dynamics must select a defined active_profile")
    for name, profile in profiles.items():
        required = {
            "horizontal_speed_mps",
            "vertical_speed_mps",
            "acceleration_mps2",
            "yaw_rate_deg_s",
            "camera_rate_hz",
            "minimum_clearance_m",
        }
        if not isinstance(profile, dict) or set(profile) != required:
            raise ValueError(f"dynamics profile {name} has the wrong fields")
        if any(float(profile[key]) <= 0 for key in required):
            raise ValueError(f"dynamics profile {name} values must be positive")


def load_release_config(path: Path) -> ReleaseConfig:
    raw = read_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != "org.aerocity.bench.release.v2":
        raise ValueError("release configuration must use org.aerocity.bench.release.v2")
    required = {
        "schema",
        "release_version",
        "generator_version",
        "master_seed",
        "core_sizes_m",
        "scale_ood_size_m",
        "split_counts",
        "episodes_per_layout",
        "target_count_ranges",
        "training_families",
        "topology_holdout_family",
        "visual_assets",
        "fleet",
        "dynamics",
        "target_processes",
        "faults",
        "evaluation_tracks",
        "observation_contract",
        "admission",
    }
    missing = sorted(required - raw.keys())
    extra = sorted(raw.keys() - required)
    if missing or extra:
        raise ValueError(f"release configuration fields differ; missing={missing}, extra={extra}")
    split_counts = _validate_ordered_split_mapping(raw["split_counts"], "split_counts")
    episodes = _validate_ordered_split_mapping(raw["episodes_per_layout"], "episodes_per_layout")
    for name in EXPECTED_SPLITS:
        if int(split_counts[name]) <= 0 or int(episodes[name]) <= 0:
            raise ValueError(f"split {name} must contain layouts and episodes")
    sizes = [int(value) for value in raw["core_sizes_m"]]
    if (
        not sizes
        or len(set(sizes)) != len(sizes)
        or any(value < 48 or value > 160 for value in sizes)
    ):
        raise ValueError("core_sizes_m must be unique integers in [48, 160]")
    scale_ood = int(raw["scale_ood_size_m"])
    if scale_ood <= max(sizes) or scale_ood > 224:
        raise ValueError("scale_ood_size_m must be larger than core sizes and at most 224")
    target_ranges = raw["target_count_ranges"]
    all_sizes = set(sizes) | {scale_ood}
    if set(target_ranges) != {str(size) for size in all_sizes}:
        raise ValueError("target_count_ranges must exactly match configured sizes")
    for size in all_sizes:
        low, high = _number_pair(target_ranges.get(str(size)), f"target range {size}")
        if int(low) != low or int(high) != high or low < 1:
            raise ValueError("target-count ranges must contain positive integers")
    fleet = raw["fleet"]
    if not isinstance(fleet, dict) or set(fleet) != {"profile", "count"}:
        raise ValueError("fleet must contain profile and count")
    if not 1 <= int(fleet["count"]) <= 32:
        raise ValueError("fleet count must lie in [1, 32]")
    _validate_evaluation_tracks(raw)
    _validate_observation_contract(raw)
    visual = raw["visual_assets"]
    if not isinstance(visual, dict) or set(visual) != {"bundle", "standard"}:
        raise ValueError("visual_assets must contain bundle and standard")
    if not visual["standard"]:
        raise ValueError("at least one standard visual asset is required")
    if len(set(visual["standard"])) != len(visual["standard"]):
        raise ValueError("visual asset IDs must be unique")
    families = raw["training_families"]
    if (
        not isinstance(families, list)
        or not families
        or len(set(families)) != len(families)
        or not set(families) <= TRAINING_FAMILY_NAMES
    ):
        raise ValueError("training_families contains unknown or duplicate values")
    if raw["topology_holdout_family"] != "topology_holdout":
        raise ValueError("topology_holdout_family must be topology_holdout")
    _number_pair(raw["admission"].get("built_ratio"), "built_ratio")
    _number_pair(raw["admission"].get("road_ratio"), "road_ratio")
    _number_pair(raw["admission"].get("street_width_m"), "street_width_m")
    if float(raw["admission"].get("minimum_support_to_target_ratio", 0.0)) <= 1:
        raise ValueError("minimum_support_to_target_ratio must exceed one")
    if float(raw["admission"].get("minimum_target_separation_m", 0.0)) <= 0:
        raise ValueError("minimum_target_separation_m must be positive")
    _validate_dynamics(raw)
    _validate_target_processes(raw)
    _validate_faults(raw)
    return ReleaseConfig(path=path.resolve(), raw=raw, config_hash=content_hash(raw))
