"""Deterministic contracts shared by the CPU tests and native Isaac gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json
from .geometry import (
    Vec3,
    colliders_from_city,
    distance,
    minimum_segment_clearance,
    segment_segment_distance,
)
from .inspection_atlas import TASK_TRACK_G1_U
from .ordinary_config import public_execution_contract
from .public_boundary import assert_public_fields, validate_public_task_spec

NATIVE_INPUT_BINDING_KEYS = frozenset(
    {
        "release_config_sha256",
        "task_spec_sha256",
        "public_episode_sha256",
        "cityspec_sha256",
        "execution_contract_hash",
        "layout_id",
        "episode_id",
    }
)


def load_native_gate_inputs(
    release_config_path: Path,
    task_spec_path: Path,
    public_episode_path: Path,
    cityspec_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load and bind the public L1 inputs without requiring evaluator-private data."""

    paths = (
        release_config_path,
        task_spec_path,
        public_episode_path,
        cityspec_path,
    )
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"native gate public input is absent: {missing}")
    config = read_json(release_config_path)
    task_spec = read_json(task_spec_path)
    episode = read_json(public_episode_path)
    city = read_json(cityspec_path)
    validate_public_task_spec(task_spec)
    assert_public_fields(episode, path="native_gate_public_episode")
    if config.get("schema") != "org.aerocity.bench.release.ordinary.v3":
        raise ValueError("native gate requires an ordinary-v3 release config")
    if task_spec.get("schema") != "org.aerocity.bench.task-spec-public.ordinary.v1":
        raise ValueError("native gate requires an ordinary public task spec")
    if task_spec.get("task_track") != TASK_TRACK_G1_U or "inspection_atlas" in task_spec:
        raise ValueError("current native gate is limited to the G1-U exploration task spec")
    if episode.get("schema") != "org.aerocity.bench.episode-public.ordinary.v1":
        raise ValueError("native gate requires an ordinary public episode")
    if city.get("schema") != "org.aerocity.bench.cityspec.ordinary.v3":
        raise ValueError("native gate requires an ordinary-v3 CitySpec")
    layout_ids = {
        str(task_spec.get("layout_id", "")),
        str(episode.get("layout_id", "")),
        str(city.get("layout_id", "")),
    }
    if len(layout_ids) != 1 or not next(iter(layout_ids)):
        raise ValueError("native gate public inputs belong to different layouts")
    authority_contract = config.get("execution_contract")
    if not isinstance(authority_contract, dict):
        raise ValueError("native gate release config lacks an execution contract")
    contract = public_execution_contract(authority_contract)
    if task_spec.get("execution_contract") != contract:
        raise ValueError("task spec execution contract differs from the release config")
    if task_spec.get("public_execution_contract_hash") != content_hash(contract):
        raise ValueError("task spec public execution-contract hash is invalid")
    fleet_count = int(config.get("fleet", {}).get("count", 0))
    starts = episode.get("starts")
    if not isinstance(starts, list) or len(starts) != fleet_count or fleet_count != 4:
        raise ValueError("canonical native gate requires exactly four public start poses")
    drone_ids = [str(start.get("drone_id", "")) for start in starts]
    if any(not drone_id for drone_id in drone_ids) or len(set(drone_ids)) != fleet_count:
        raise ValueError("native gate public start identities are invalid")
    forbidden_public_keys = {
        "targets",
        "distractors",
        "target_process",
        "target_validity",
        "counterfactual_pairs",
    }
    leaked = sorted(forbidden_public_keys & set(episode))
    if leaked:
        raise ValueError(f"native gate public episode contains private fields: {leaked}")
    bindings = {
        "release_config_sha256": file_hash(release_config_path),
        "task_spec_sha256": file_hash(task_spec_path),
        "public_episode_sha256": file_hash(public_episode_path),
        "cityspec_sha256": file_hash(cityspec_path),
        "execution_contract_hash": content_hash(contract),
        "layout_id": next(iter(layout_ids)),
        "episode_id": str(episode.get("episode_id", "")),
    }
    if set(bindings) != NATIVE_INPUT_BINDING_KEYS or not bindings["episode_id"]:
        raise ValueError("native gate input bindings are incomplete")
    return config, task_spec, episode, city, bindings


def _within_flight_bounds(
    point: Vec3,
    bounds: dict[str, list[float]],
    margin_m: float,
) -> bool:
    return all(
        float(low) + margin_m <= value <= float(high) - margin_m
        for value, low, high in zip(
            point, bounds["minimum"], bounds["maximum"], strict=True
        )
    )


def select_native_test_directions(
    city: dict[str, Any],
    starts: list[dict[str, Any]],
    *,
    travel_distance_m: float,
    clearance_m: float,
    body_radius_m: float,
) -> dict[str, Vec3]:
    """Choose deterministic collision-clear horizontal test segments for four UAVs."""

    if travel_distance_m <= 0.0 or clearance_m <= 0.0 or body_radius_m <= 0.0:
        raise ValueError("native test distance, clearance, and body radius must be positive")
    colliders = colliders_from_city(city)
    diagonal = math.sqrt(0.5)
    candidates: tuple[Vec3, ...] = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (diagonal, diagonal, 0.0),
        (-diagonal, diagonal, 0.0),
        (diagonal, -diagonal, 0.0),
        (-diagonal, -diagonal, 0.0),
    )
    selected: dict[str, Vec3] = {}
    segments: list[tuple[Vec3, Vec3]] = []
    for start in sorted(starts, key=lambda item: str(item["drone_id"])):
        origin = tuple(float(value) for value in start["position"])
        if len(origin) != 3:
            raise ValueError("native gate start position must be a three-vector")
        chosen: Vec3 | None = None
        chosen_end: Vec3 | None = None
        for direction in candidates:
            end = tuple(
                origin[index] + travel_distance_m * direction[index] for index in range(3)
            )
            if not _within_flight_bounds(end, city["flight_bounds"], clearance_m):
                continue
            obstacle_clearance, _ = minimum_segment_clearance(origin, end, colliders)
            if obstacle_clearance + 1.0e-9 < clearance_m:
                continue
            if any(
                segment_segment_distance(origin, end, other_start, other_end)
                < 2.0 * body_radius_m
                for other_start, other_end in segments
            ):
                continue
            chosen, chosen_end = direction, end  # type: ignore[assignment]
            break
        if chosen is None or chosen_end is None:
            raise ValueError(f"no collision-clear native test segment for {start['drone_id']}")
        selected[str(start["drone_id"])] = chosen
        segments.append((origin, chosen_end))
    return selected


def _speed_ramp(limit: float, delta: float) -> tuple[list[float], list[float]]:
    if limit <= 0.0 or delta <= 0.0:
        raise ValueError("speed ramp values must be positive")
    rising: list[float] = []
    speed = 0.0
    while speed + 1.0e-9 < limit:
        speed = min(limit, speed + delta)
        rising.append(speed)
    falling: list[float] = []
    speed = limit
    while speed > 1.0e-9:
        speed = max(0.0, speed - delta)
        falling.append(speed)
    return rising, falling


def build_native_action_transcript(
    execution_contract: dict[str, Any], directions: dict[str, Vec3]
) -> list[dict[str, Any]]:
    """Build the frozen acceleration-limited four-UAV L1 gate transcript."""

    period = float(execution_contract["control_period_s"])
    vehicle = execution_contract["vehicle"]
    acceleration = float(vehicle["acceleration_mps2"])
    horizontal_limit = float(vehicle["horizontal_speed_mps"])
    vertical_limit = float(vehicle["vertical_speed_mps"])
    yaw_limit = float(vehicle["yaw_rate_deg_s"])
    horizontal_up, horizontal_down = _speed_ramp(
        horizontal_limit, acceleration * period
    )
    vertical_up, vertical_down = _speed_ramp(vertical_limit, acceleration * period)
    commands: list[dict[str, Any]] = []

    def append(
        phase: str,
        *,
        horizontal_speed: float = 0.0,
        vertical_speed: float = 0.0,
        yaw_rate_deg_s: float = 0.0,
        observe_case: str | None = None,
    ) -> None:
        by_drone = {}
        for drone_id, direction in sorted(directions.items()):
            by_drone[drone_id] = {
                "linear_velocity_world_mps": [
                    direction[0] * horizontal_speed,
                    direction[1] * horizontal_speed,
                    vertical_speed,
                ],
                "yaw_rate_deg_s": yaw_rate_deg_s,
            }
        commands.append(
            {
                "index": len(commands),
                "phase": phase,
                "observe_case": observe_case,
                "commands": by_drone,
            }
        )

    for speed in horizontal_up:
        append("horizontal_acceleration", horizontal_speed=speed)
    for _ in range(2):
        append("horizontal_cruise", horizontal_speed=horizontal_limit)
    for speed in horizontal_down:
        append("horizontal_braking", horizontal_speed=speed)
    for speed in vertical_up:
        append("vertical_acceleration", vertical_speed=speed)
    append("vertical_cruise", vertical_speed=vertical_limit)
    for speed in vertical_down:
        append("vertical_braking", vertical_speed=speed)
    for _ in range(2):
        append("yaw_tracking", yaw_rate_deg_s=yaw_limit)
    append("yaw_stop")
    for _ in range(4):
        append("observe_stable", observe_case="positive")
    for _ in range(2):
        append("observe_interrupt_pre", observe_case="interrupted")
    interrupt_speed = max(
        2.0 * float(execution_contract["observe"]["max_linear_speed_mps"]),
        acceleration * period,
    )
    append(
        "observe_interrupt_motion",
        horizontal_speed=interrupt_speed,
        observe_case="interrupted",
    )
    append("observe_interrupt_stop", observe_case="interrupted")
    for _ in range(2):
        append("observe_interrupt_post", observe_case="interrupted")
    return commands


def evaluate_native_dwell_samples(
    samples: list[dict[str, Any]], observe_contract: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate positive dwell and movement-interruption traces from measured states."""

    required = float(observe_contract["continuous_dwell_s"])
    max_linear = float(observe_contract["max_linear_speed_mps"])
    max_angular = float(observe_contract["max_angular_speed_deg_s"])
    max_drift = float(observe_contract["max_pose_drift_m"])
    accepted: dict[str, float | None] = {"positive": None, "interrupted": None}
    session: dict[str, dict[str, Any] | None] = {"positive": None, "interrupted": None}
    interruption_reset = False
    for sample in samples:
        case = sample.get("observe_case")
        if case not in session:
            continue
        stable = (
            float(sample["linear_speed_mps"]) <= max_linear + 1.0e-9
            and float(sample["angular_speed_deg_s"]) <= max_angular + 1.0e-9
        )
        current = session[case]
        if not stable:
            if case == "interrupted" and current is not None:
                interruption_reset = True
            session[case] = None
            continue
        position = tuple(float(value) for value in sample["position"])
        timestamp = float(sample["task_time_s"])
        if current is None:
            current = {"started_at_s": timestamp, "initial_position": position}
            session[case] = current
        elif distance(current["initial_position"], position) > max_drift + 1.0e-9:
            if case == "interrupted":
                interruption_reset = True
            current = {"started_at_s": timestamp, "initial_position": position}
            session[case] = current
        dwell = timestamp - float(current["started_at_s"])
        if dwell + 1.0e-9 >= required and accepted[case] is None:
            accepted[case] = timestamp
    status = (
        "PASS"
        if accepted["positive"] is not None
        and accepted["interrupted"] is None
        and interruption_reset
        else "FAIL"
    )
    return {
        "status": status,
        "required_continuous_dwell_s": required,
        "positive_accepted_at_s": accepted["positive"],
        "interrupted_accepted_at_s": accepted["interrupted"],
        "movement_interruption_reset_observed": interruption_reset,
    }


def compare_native_replays(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    position_tolerance_m: float,
    velocity_tolerance_mps: float,
    orientation_tolerance: float,
) -> dict[str, Any]:
    """Compare all sampled states from two executions of one frozen transcript."""

    if len(first) != len(second) or not first:
        return {"status": "FAIL", "reason": "sample count differs or is empty"}
    maxima = {
        "position_error_m": 0.0,
        "linear_velocity_error_mps": 0.0,
        "orientation_component_error": 0.0,
    }
    for left, right in zip(first, second, strict=True):
        if (
            left.get("command_index") != right.get("command_index")
            or left.get("drone_id") != right.get("drone_id")
        ):
            return {"status": "FAIL", "reason": "sample identities differ"}
        maxima["position_error_m"] = max(
            maxima["position_error_m"],
            distance(tuple(left["position"]), tuple(right["position"])),
        )
        maxima["linear_velocity_error_mps"] = max(
            maxima["linear_velocity_error_mps"],
            distance(tuple(left["linear_velocity_mps"]), tuple(right["linear_velocity_mps"])),
        )
        direct_orientation_error = max(
            abs(float(a) - float(b))
            for a, b in zip(
                left["orientation_wxyz"], right["orientation_wxyz"], strict=True
            )
        )
        negated_orientation_error = max(
            abs(float(a) + float(b))
            for a, b in zip(
                left["orientation_wxyz"], right["orientation_wxyz"], strict=True
            )
        )
        maxima["orientation_component_error"] = max(
            maxima["orientation_component_error"],
            min(direct_orientation_error, negated_orientation_error),
        )
    status = (
        "PASS"
        if maxima["position_error_m"] <= position_tolerance_m
        and maxima["linear_velocity_error_mps"] <= velocity_tolerance_mps
        and maxima["orientation_component_error"] <= orientation_tolerance
        else "FAIL"
    )
    return {
        "status": status,
        **maxima,
        "position_tolerance_m": position_tolerance_m,
        "velocity_tolerance_mps": velocity_tolerance_mps,
        "orientation_component_tolerance": orientation_tolerance,
        "sample_count": len(first),
    }


def commanded_braking_distance(
    transcript: list[dict[str, Any]], control_period_s: float
) -> float:
    """Return the discrete stopping distance implied by the frozen speed commands."""

    first_drone = sorted(transcript[0]["commands"])[0]
    return sum(
        math.sqrt(
            sum(
                float(value) ** 2
                for value in step["commands"][first_drone]["linear_velocity_world_mps"][:2]
            )
        )
        * control_period_s
        for step in transcript
        if step["phase"] == "horizontal_braking"
    )
