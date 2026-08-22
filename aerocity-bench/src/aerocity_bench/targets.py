"""Evaluator-private three-dimensional target processes and fault interventions."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

from .canonical import content_hash, derived_seed
from .config import ReleaseConfig
from .errors import GenerationRejected

SUPPORT_CLASSES = ("roof", "opening", "entrance", "rubble")
ALTITUDE_BANDS = ("near_ground", "lower", "mid", "elevated", "highrise")


def altitude_band(z: float) -> str:
    if z < 4.0:
        return "near_ground"
    if z < 12.0:
        return "lower"
    if z < 25.0:
        return "mid"
    if z < 40.0:
        return "elevated"
    return "highrise"


def _site(
    support_class: str,
    owner_id: str,
    position: list[float],
    normal: list[float],
) -> dict[str, Any]:
    payload = {
        "support_class": support_class,
        "owner_id": owner_id,
        "position": [round(value, 4) for value in position],
        "normal": [round(value, 4) for value in normal],
    }
    payload["altitude_band"] = altitude_band(float(position[2]))
    payload["site_id"] = f"site-{content_hash(payload)[:16]}"
    return payload


def _positions(start: float, stop: float, spacing: float) -> Iterable[float]:
    span = max(0.0, stop - start)
    count = max(1, int(span // spacing) + 1)
    if count == 1:
        yield (start + stop) / 2
        return
    for index in range(count):
        yield start + span * index / (count - 1)


def _component_bounds(component: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    cx, cy, cz = [float(value) for value in component["center"]]
    sx, sy, sz = [float(value) for value in component["size"]]
    return (
        cx - sx / 2,
        cx + sx / 2,
        cy - sy / 2,
        cy + sy / 2,
        cz - sz / 2,
        cz + sz / 2,
    )


def _covered_by_other_component(
    position: list[float],
    source_index: int,
    components: list[dict[str, Any]],
    tolerance: float = 0.3,
) -> bool:
    px, py, pz = position
    for index, component in enumerate(components):
        if index == source_index:
            continue
        x0, x1, y0, y1, z0, z1 = _component_bounds(component)
        if (
            x0 - tolerance <= px <= x1 + tolerance
            and y0 - tolerance <= py <= y1 + tolerance
            and z0 - tolerance <= pz <= z1 + tolerance
        ):
            return True
    return False


def derive_support_sites(city: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive sites from actual component faces, never from a footprint bounding box."""

    rules = city["support_site_rules"]
    roof_spacing = float(rules["roof_spacing_m"])
    opening_spacing = float(rules["opening_spacing_m"])
    opening_altitudes = [float(value) for value in rules["opening_altitudes_m"]]
    sites: list[dict[str, Any]] = []
    for building in city["buildings"]:
        components = building["components"]
        building_id = str(building["id"])
        for component_index, component in enumerate(components):
            component_id = f"{building_id}/{component['id']}"
            x0, x1, y0, y1, z0, z1 = _component_bounds(component)
            for px in _positions(x0 + (x1 - x0) * 0.16, x1 - (x1 - x0) * 0.16, roof_spacing):
                for py in _positions(y0 + (y1 - y0) * 0.16, y1 - (y1 - y0) * 0.16, roof_spacing):
                    position = [px, py, z1 + 0.25]
                    if not _covered_by_other_component(position, component_index, components):
                        sites.append(_site("roof", component_id, position, [0.0, 0.0, 1.0]))
            levels = [value for value in opening_altitudes if z0 + 1.0 <= value <= z1 - 0.8]
            for level in levels:
                for px in _positions(x0 + 0.8, x1 - 0.8, opening_spacing):
                    for position, normal in (
                        ([px, y0 - 0.2, level], [0.0, -1.0, 0.0]),
                        ([px, y1 + 0.2, level], [0.0, 1.0, 0.0]),
                    ):
                        if not _covered_by_other_component(position, component_index, components):
                            sites.append(_site("opening", component_id, position, normal))
                for py in _positions(y0 + 0.8, y1 - 0.8, opening_spacing):
                    for position, normal in (
                        ([x0 - 0.2, py, level], [-1.0, 0.0, 0.0]),
                        ([x1 + 0.2, py, level], [1.0, 0.0, 0.0]),
                    ):
                        if not _covered_by_other_component(position, component_index, components):
                            sites.append(_site("opening", component_id, position, normal))
        footprint_x, footprint_y = [float(value) for value in building["footprint"][:2]]
        for entrance_index, entrance in enumerate(building["entrances"]):
            entrance_position = [float(value) for value in entrance]
            dx = entrance_position[0] - footprint_x
            dy = entrance_position[1] - footprint_y
            normal = (
                [math.copysign(1.0, dx), 0.0, 0.0]
                if abs(dx) > abs(dy)
                else [0.0, math.copysign(1.0, dy), 0.0]
            )
            sites.append(
                _site(
                    "entrance",
                    f"{building_id}/entrance-{entrance_index:02d}",
                    entrance_position,
                    normal,
                )
            )
    for obstacle in city["obstacles"]:
        if not obstacle.get("support_domain"):
            continue
        x, y, z = [float(value) for value in obstacle["center"]]
        height = float(obstacle["size"][2])
        sites.append(
            _site(
                "rubble",
                str(obstacle["id"]),
                [x, y, z + height / 2 + 0.2],
                [0.0, 0.0, 1.0],
            )
        )
    deduplicated = {item["site_id"]: item for item in sites}
    return [deduplicated[key] for key in sorted(deduplicated)]


def _distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.dist(first["position"], second["position"])


def _admissible(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    minimum_separation_m: float,
) -> list[dict[str, Any]]:
    selected_ids = {item["site_id"] for item in selected}
    return [
        candidate
        for candidate in candidates
        if candidate["site_id"] not in selected_ids
        and all(_distance(candidate, current) >= minimum_separation_m for current in selected)
    ]


def _weighted_choice(
    candidates: list[dict[str, Any]], weights: list[float], rng: random.Random
) -> dict[str, Any]:
    if not candidates or len(candidates) != len(weights) or sum(weights) <= 0:
        raise GenerationRejected("target process has no positive-weight admissible site")
    return rng.choices(candidates, weights=weights, k=1)[0]


def _sample_uniform(
    sites: list[dict[str, Any]], count: int, separation: float, rng: random.Random
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _ in range(count):
        candidates = _admissible(sites, selected, separation)
        if not candidates:
            raise GenerationRejected("uniform_surface exhausted separated support sites")
        selected.append(rng.choice(candidates))
    return selected


def _sample_clustered(
    sites: list[dict[str, Any]],
    count: int,
    separation: float,
    profile: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    low, high = [int(value) for value in profile["cluster_count"]]
    cluster_count = min(count, rng.randint(low, high))
    bandwidth = float(profile["bandwidth_m"])
    anchors = [rng.choice(sites)]
    while len(anchors) < cluster_count:
        nearby = [
            site
            for site in sites
            if site not in anchors
            and min(_distance(site, anchor) for anchor in anchors) <= 2.0 * bandwidth
        ]
        anchors.append(rng.choice(nearby or [site for site in sites if site not in anchors]))
    selected: list[dict[str, Any]] = []
    for _ in range(count):
        candidates = _admissible(sites, selected, separation)
        weights = [
            math.exp(-min(_distance(site, anchor) for anchor in anchors) / bandwidth)
            for site in candidates
        ]
        selected.append(_weighted_choice(candidates, weights, rng))
    return selected


def _sample_height_stratified(
    sites: list[dict[str, Any]],
    count: int,
    separation: float,
    profile: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    weights_by_band = {key: float(value) for key, value in profile["band_weights"].items()}
    available_bands = [
        band
        for band in ALTITUDE_BANDS
        if weights_by_band[band] > 0 and any(site["altitude_band"] == band for site in sites)
    ]
    minimum_bands = min(int(profile["minimum_bands"]), count)
    if len(available_bands) < minimum_bands:
        raise GenerationRejected("height_stratified lacks the required altitude support")
    selected: list[dict[str, Any]] = []
    coverage_order = sorted(available_bands, key=lambda band: (-weights_by_band[band], band))
    for band in coverage_order[:minimum_bands]:
        candidates = _admissible(
            [site for site in sites if site["altitude_band"] == band], selected, separation
        )
        if not candidates:
            raise GenerationRejected("height_stratified cannot satisfy separated band coverage")
        selected.append(rng.choice(candidates))
    while len(selected) < count:
        candidates = _admissible(sites, selected, separation)
        selected.append(
            _weighted_choice(
                candidates,
                [weights_by_band[site["altitude_band"]] for site in candidates],
                rng,
            )
        )
    return selected


def _starts(city: dict[str, Any], fleet_count: int) -> list[dict[str, Any]]:
    vertical = min(
        (road for road in city["roads"] if road["axis"] == "x"),
        key=lambda road: abs(float(road["x"])),
    )
    horizontal = min(
        (road for road in city["roads"] if road["axis"] == "y"),
        key=lambda road: abs(float(road["y"])),
    )
    center_x, center_y = float(vertical["x"]), float(horizontal["y"])
    radius = (
        0.0
        if fleet_count == 1
        else min(
            1.5,
            min(float(vertical["width_m"]), float(horizontal["width_m"])) * 0.16,
        )
    )
    starts = []
    for drone_index in range(fleet_count):
        angle = 2 * math.pi * drone_index / fleet_count
        starts.append(
            {
                "drone_id": f"uav-{drone_index:02d}",
                "position": [
                    round(center_x + math.cos(angle) * radius, 4),
                    round(center_y + math.sin(angle) * radius, 4),
                    2.5,
                ],
                "yaw_deg": round(math.degrees(angle), 3),
            }
        )
    return starts


def _fault_spec(
    config: ReleaseConfig,
    profile_name: str,
    starts: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    profile = config.raw["faults"]["profiles"][profile_name]
    fault_type = str(profile["type"])
    # Faults of the same type share an ordering and onset stream. This makes
    # hard_loss_1 a strict affected-agent subset of hard_loss_2 in a paired block.
    rng = random.Random(derived_seed(seed, "fault", fault_type))
    drone_ids = [str(start["drone_id"]) for start in starts]
    rng.shuffle(drone_ids)
    affected = sorted(drone_ids[: int(profile["affected_count"])])
    result: dict[str, Any] = {
        "profile": profile_name,
        "type": fault_type,
        "affected_drone_ids": affected,
        "method_observability": "health_signals_only_no_evaluator_truth",
    }
    if fault_type == "none":
        result.update({"onset_fraction": None, "recoverable": True})
        return result
    onset_low, onset_high = [float(value) for value in profile["onset_fraction"]]
    result["onset_fraction"] = round(rng.uniform(onset_low, onset_high), 6)
    result["recoverable"] = fault_type == "temporary_communication_isolation"
    if "duration_fraction" in profile:
        low, high = [float(value) for value in profile["duration_fraction"]]
        result["duration_fraction"] = round(rng.uniform(low, high), 6)
    if "channel" in profile:
        result["channel"] = str(profile["channel"])
    return result


def _episode_conditions(
    config: ReleaseConfig, city: dict[str, Any], episode_index: int
) -> tuple[str, str, int, str]:
    split = str(city["split"])
    processes = config.target_processes(split)
    faults = config.fault_profiles(split)
    if split == "test_resilience":
        pair_index = episode_index // len(faults)
        fault_name = faults[episode_index % len(faults)]
        process_name = processes[pair_index % len(processes)]
        condition_seed = derived_seed(config.master_seed, city["layout_hash"], "pair", pair_index)
        group_id = f"pair-{pair_index:04d}"
    else:
        process_name = processes[episode_index % len(processes)]
        fault_name = "none"
        group_index = episode_index // len(processes)
        condition_seed = derived_seed(
            config.master_seed,
            city["layout_hash"],
            "condition",
            group_index,
        )
        group_id = f"condition-{group_index:04d}"
    return process_name, fault_name, condition_seed, group_id


def sample_episode(
    config: ReleaseConfig,
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    episode_index: int,
) -> dict[str, Any]:
    process_name, fault_name, seed, group_id = _episode_conditions(config, city, episode_index)
    count_rng = random.Random(derived_seed(seed, "target_count"))
    target_rng = random.Random(derived_seed(seed, "target_process", process_name))
    environment_rng = random.Random(derived_seed(seed, "environment"))
    low, high = config.target_range(int(city["size_m"]))
    target_count = count_rng.randint(low, high)
    required_ratio = float(config.raw["admission"]["minimum_support_to_target_ratio"])
    if len(support_sites) < math.ceil(required_ratio * target_count):
        raise GenerationRejected(
            f"layout {city['layout_id']} has {len(support_sites)} support sites "
            f"for {target_count} targets"
        )
    separation = float(config.raw["admission"]["minimum_target_separation_m"])
    profile = config.raw["target_processes"]["profiles"][process_name]
    if process_name == "uniform_surface":
        selected = _sample_uniform(support_sites, target_count, separation, target_rng)
    elif process_name == "clustered_surface":
        selected = _sample_clustered(support_sites, target_count, separation, profile, target_rng)
    elif process_name == "height_stratified":
        selected = _sample_height_stratified(
            support_sites, target_count, separation, profile, target_rng
        )
    else:  # pragma: no cover - load_release_config rejects this path
        raise GenerationRejected(f"unknown target process: {process_name}")
    selected.sort(key=lambda item: item["site_id"])
    fleet_count = config.fleet_count_for_split(str(city["split"]))
    starts = _starts(city, fleet_count)
    size_m = float(city["size_m"])
    episode = {
        "schema": "org.aerocity.bench.episode-private.v2",
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "episode_index": episode_index,
        "episode_seed": seed,
        "condition_group_id": group_id,
        "target_process": process_name,
        "target_count": target_count,
        "targets": [
            {
                "target_id": f"target-{index:03d}",
                "site_id": site["site_id"],
                "support_class": site["support_class"],
                "altitude_band": site["altitude_band"],
                "position": site["position"],
                "normal": site["normal"],
            }
            for index, site in enumerate(selected)
        ],
        "fleet_profile": {
            "name": str(config.raw["fleet"]["profile"]),
            "count": fleet_count,
        },
        "dynamics_profile": config.dynamics_profile,
        "starts": starts,
        "fault_spec": _fault_spec(config, fault_name, starts, seed),
        "smoke": {
            "density": round(environment_rng.uniform(0.0, 0.35), 5),
            "seed": derived_seed(seed, "smoke"),
        },
        "communication": {
            "latency_ms": round(environment_rng.uniform(15.0, 120.0), 4),
            "drop_probability": round(environment_rng.uniform(0.0, 0.08), 5),
        },
        "energy_budget_j": round(size_m * environment_rng.uniform(24.0, 34.0), 4),
    }
    episode["episode_hash"] = content_hash(episode)
    return episode
