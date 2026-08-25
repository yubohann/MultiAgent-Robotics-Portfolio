"""Experiment 3 8-drone formation demo helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


FormationShape = Literal["line", "triangle", "rectangle", "diamond", "circle"]
FormationMode = Literal["line", "triangle", "rectangle", "diamond", "circle", "mixed_route"]
ReasoningMode = Literal["fast_only", "slow_shadow", "slow_active", "guidance_shadow", "guidance_visible"]
StageKind = Literal[
    "handoff_replay",
    "prep_hold",
    "hold",
    "micro_travel",
    "mid_travel",
    "full_travel",
    "morph_hold",
    "route",
]
RouteMarkerRole = Literal["start", "waypoint", "goal"]


@dataclass(frozen=True)
class FormationDemoWaypoint:
    name: str
    x_m: float
    y_m: float

    @property
    def xy(self) -> tuple[float, float]:
        return (float(self.x_m), float(self.y_m))


@dataclass(frozen=True)
class FormationDemoStageSpec:
    stage_index: int
    stage_name: str
    kind: StageKind
    formation: FormationMode
    train_steps: int
    center_start_xy: tuple[float, float]
    center_goal_xy: tuple[float, float]
    path_waypoints_xy: tuple[tuple[float, float], ...]
    route_segment_formations: tuple[FormationShape, ...]
    reasoning_mode: ReasoningMode
    notes: str


@dataclass(frozen=True)
class FormationDemoRouteMarkerSpec:
    name: str
    x_m: float
    y_m: float
    role: RouteMarkerRole
    highlighted: bool
    color_name: str
    radius_m: float

    @property
    def xy(self) -> tuple[float, float]:
        return (float(self.x_m), float(self.y_m))


@dataclass(frozen=True)
class FormationDemoSwitchSnapshot:
    waypoint_name: str
    center_xy: tuple[float, float]
    heading_xy: tuple[float, float]
    formation: FormationShape
    slots_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FormationDemoMorphTransitionSpec:
    waypoint_name: str
    center_xy: tuple[float, float]
    from_formation: FormationShape
    to_formation: FormationShape
    arrival_heading_xy: tuple[float, float]
    departure_heading_xy: tuple[float, float]
    assignment_to_next_slot_indices: tuple[int, ...]
    staging_scale: float
    arrival_slots_xy: tuple[tuple[float, float], ...]
    next_start_slots_xy: tuple[tuple[float, float], ...]
    drone_target_slots_xy: tuple[tuple[float, float], ...]
    drone_staging_slots_xy: tuple[tuple[float, float], ...]
    drone_paths_xy: tuple[tuple[tuple[float, float], ...], ...]
    min_sampled_clearance_m: float


@dataclass(frozen=True)
class FormationDemoManifest:
    demo_name: str
    source_stage_name: str
    source_checkpoint_hint: str
    scene_mode: str
    fixed_height_m: float
    world_x_bounds_m: tuple[float, float]
    world_y_bounds_m: tuple[float, float]
    waypoints: tuple[FormationDemoWaypoint, ...]
    route_markers: tuple[FormationDemoRouteMarkerSpec, ...]
    switch_snapshots: tuple[FormationDemoSwitchSnapshot, ...]
    morph_transitions: tuple[FormationDemoMorphTransitionSpec, ...]
    stages: tuple[FormationDemoStageSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FormationDemoStageExecutionPlan:
    """Resolved manifest context for one demo8 formation stage."""

    manifest: FormationDemoManifest
    stage: FormationDemoStageSpec
    route_morph_transitions: tuple[FormationDemoMorphTransitionSpec, ...]


WAYPOINTS: tuple[FormationDemoWaypoint, ...] = (
    FormationDemoWaypoint("P1", -36.0, 0.0),
    FormationDemoWaypoint("P2", -20.0, 0.0),
    FormationDemoWaypoint("P3", -8.0, 9.0),
    FormationDemoWaypoint("P4", 8.0, 9.0),
    FormationDemoWaypoint("P5", 20.0, 0.0),
    FormationDemoWaypoint("P6", 36.0, 0.0),
)


FORMATION_BY_SEGMENT: tuple[FormationShape, ...] = (
    "line",
    "triangle",
    "rectangle",
    "diamond",
    "circle",
)


# Each tuple maps D1-D8 at a waypoint to the next formation's slot index.
MORPH_ASSIGNMENT_BY_WAYPOINT: dict[str, tuple[int, ...]] = {
    "P2": (6, 3, 7, 4, 1, 2, 0, 5),
    "P3": (3, 1, 2, 0, 6, 7, 4, 5),
    "P4": (1, 0, 2, 4, 3, 5, 7, 6),
    "P5": (7, 6, 0, 5, 1, 4, 2, 3),
}


MORPH_STAGING_SCALE_BY_WAYPOINT: dict[str, float] = {
    "P2": 1.6,
    "P3": 1.2,
    "P4": 1.2,
    "P5": 1.1,
}


def build_exp3_formation_demo_manifest() -> FormationDemoManifest:
    """Return the isolated 8-drone demo branch manifest."""

    stages = tuple(_build_demo_stages())
    return FormationDemoManifest(
        demo_name="exp3_demo8_branch_from_stage02d",
        source_stage_name="stage02d_empty_8_full",
        source_checkpoint_hint=(
            "runtime/paper_runs/"
            "exp3_stage02d_from_stage02c_v58_empty8_full_nobc_shortbridge/"
            "stages/20_stage02d_empty_8_full/checkpoints/latest_agent.pt"
        ),
        scene_mode="empty_fixed_height_isaaclab_shell_3d",
        fixed_height_m=4.0,
        world_x_bounds_m=(-44.0, 44.0),
        world_y_bounds_m=(-16.0, 16.0),
        waypoints=WAYPOINTS,
        route_markers=tuple(build_route_marker_specs()),
        switch_snapshots=tuple(build_switch_snapshots()),
        morph_transitions=tuple(build_morph_transition_specs()),
        stages=stages,
    )


def build_stage_execution_plan(stage_name: str) -> FormationDemoStageExecutionPlan:
    """Resolve one demo8 stage and the route morphs it needs."""

    manifest = build_exp3_formation_demo_manifest()
    normalized_name = str(stage_name).strip()
    by_name = {stage.stage_name: stage for stage in manifest.stages}
    if normalized_name not in by_name:
        raise ValueError(f"Unknown demo8 formation stage: {stage_name}")
    stage = by_name[normalized_name]
    path = tuple(stage.path_waypoints_xy)
    internal_waypoint_names = {
        waypoint.name
        for waypoint in manifest.waypoints
        if waypoint.xy in path[1:-1]
    }
    route_morphs = tuple(
        transition
        for transition in manifest.morph_transitions
        if transition.waypoint_name in internal_waypoint_names
    )
    return FormationDemoStageExecutionPlan(
        manifest=manifest,
        stage=stage,
        route_morph_transitions=route_morphs,
    )


def resolve_route_slot_permutations_and_morph_paths(
    plan: FormationDemoStageExecutionPlan,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[tuple[float, float], ...], ...], ...],
]:
    """Return cumulative route slot permutations and per-drone morph paths."""

    identity = tuple(range(8))
    route_slot_permutations: list[tuple[int, ...]] = [identity]
    route_morph_paths_xy: list[tuple[tuple[tuple[float, float], ...], ...]] = []
    cumulative_permutation = identity
    for transition in plan.route_morph_transitions:
        raw_assignment = tuple(int(idx) for idx in transition.assignment_to_next_slot_indices)
        route_morph_paths_xy.append(
            tuple(
                tuple((float(x), float(y)) for x, y in transition.drone_paths_xy[slot_index])
                for slot_index in cumulative_permutation
            )
        )
        cumulative_permutation = tuple(raw_assignment[slot_index] for slot_index in cumulative_permutation)
        route_slot_permutations.append(cumulative_permutation)
    return (tuple(route_slot_permutations), tuple(route_morph_paths_xy))


def build_route_marker_specs() -> list[FormationDemoRouteMarkerSpec]:
    """Return static IsaacSim marker specs for the six demo route waypoints."""

    specs: list[FormationDemoRouteMarkerSpec] = []
    for index, waypoint in enumerate(WAYPOINTS):
        if index == 0:
            role: RouteMarkerRole = "start"
            highlighted = True
            color_name = "amber_start"
            radius_m = 1.55
        elif index == len(WAYPOINTS) - 1:
            role = "goal"
            highlighted = True
            color_name = "green_goal"
            radius_m = 1.65
        else:
            role = "waypoint"
            highlighted = False
            color_name = "cyan_waypoint"
            radius_m = 1.15
        specs.append(
            FormationDemoRouteMarkerSpec(
                name=waypoint.name,
                x_m=waypoint.x_m,
                y_m=waypoint.y_m,
                role=role,
                highlighted=highlighted,
                color_name=color_name,
                radius_m=radius_m,
            )
        )
    return specs


def build_switch_snapshots() -> list[FormationDemoSwitchSnapshot]:
    """Return the desired 8-drone slots at each shape switch waypoint."""

    snapshots: list[FormationDemoSwitchSnapshot] = []
    for index, waypoint in enumerate(WAYPOINTS):
        if index == 0:
            formation = FORMATION_BY_SEGMENT[0]
            heading = _heading(WAYPOINTS[0].xy, WAYPOINTS[1].xy)
        elif index < len(WAYPOINTS) - 1:
            formation = FORMATION_BY_SEGMENT[index]
            heading = _heading(WAYPOINTS[index].xy, WAYPOINTS[index + 1].xy)
        else:
            formation = FORMATION_BY_SEGMENT[-1]
            heading = _heading(WAYPOINTS[index - 1].xy, WAYPOINTS[index].xy)
        snapshots.append(
            FormationDemoSwitchSnapshot(
                waypoint_name=waypoint.name,
                center_xy=waypoint.xy,
                heading_xy=heading,
                formation=formation,
                slots_xy=world_formation_slots(formation, waypoint.xy, heading),
            )
        )
    return snapshots


def build_morph_transition_specs() -> list[FormationDemoMorphTransitionSpec]:
    """Return conflict-aware per-drone slot-return plans at P2-P5.

    The terminal P6 has no downstream formation, so it is excluded.
    """

    transitions: list[FormationDemoMorphTransitionSpec] = []
    for waypoint_index in range(1, len(WAYPOINTS) - 1):
        waypoint = WAYPOINTS[waypoint_index]
        from_formation = FORMATION_BY_SEGMENT[waypoint_index - 1]
        to_formation = FORMATION_BY_SEGMENT[waypoint_index]
        arrival_heading = _heading(WAYPOINTS[waypoint_index - 1].xy, waypoint.xy)
        departure_heading = _heading(waypoint.xy, WAYPOINTS[waypoint_index + 1].xy)
        arrival_slots = world_formation_slots(from_formation, waypoint.xy, arrival_heading)
        next_start_slots = world_formation_slots(to_formation, waypoint.xy, departure_heading)
        assignment = MORPH_ASSIGNMENT_BY_WAYPOINT[waypoint.name]
        staging_scale = float(MORPH_STAGING_SCALE_BY_WAYPOINT[waypoint.name])
        unassigned_staging_slots = scaled_world_formation_slots(
            to_formation,
            waypoint.xy,
            departure_heading,
            scale=staging_scale,
        )
        drone_target_slots = tuple(next_start_slots[target_index] for target_index in assignment)
        drone_staging_slots = tuple(unassigned_staging_slots[target_index] for target_index in assignment)
        drone_paths = tuple(
            (
                arrival_slots[drone_index],
                drone_staging_slots[drone_index],
                drone_target_slots[drone_index],
            )
            for drone_index in range(8)
        )
        transitions.append(
            FormationDemoMorphTransitionSpec(
                waypoint_name=waypoint.name,
                center_xy=waypoint.xy,
                from_formation=from_formation,
                to_formation=to_formation,
                arrival_heading_xy=arrival_heading,
                departure_heading_xy=departure_heading,
                assignment_to_next_slot_indices=assignment,
                staging_scale=staging_scale,
                arrival_slots_xy=arrival_slots,
                next_start_slots_xy=next_start_slots,
                drone_target_slots_xy=drone_target_slots,
                drone_staging_slots_xy=drone_staging_slots,
                drone_paths_xy=drone_paths,
                min_sampled_clearance_m=sampled_morph_path_clearance_m(drone_paths),
            )
        )
    return transitions


def local_formation_offsets(formation: FormationShape) -> tuple[tuple[float, float], ...]:
    """Return local ``(forward, lateral)`` offsets for the 8-drone shape."""

    if formation == "line":
        spacing = 1.7
        return tuple((0.0, (idx - 3.5) * spacing) for idx in range(8))
    if formation == "triangle":
        return (
            (3.6, 0.0),
            (1.2, -1.7),
            (1.2, 1.7),
            (-1.2, -3.4),
            (-1.2, 0.0),
            (-1.2, 3.4),
            (-3.6, -1.7),
            (-3.6, 1.7),
        )
    if formation == "rectangle":
        return tuple((forward, lateral) for forward in (2.0, -2.0) for lateral in (-3.0, -1.0, 1.0, 3.0))
    if formation == "diamond":
        return (
            (4.0, 0.0),
            (2.0, -2.2),
            (2.0, 2.2),
            (0.0, -4.0),
            (0.0, 4.0),
            (-2.0, -2.2),
            (-2.0, 2.2),
            (-4.0, 0.0),
        )
    if formation == "circle":
        radius = 3.2
        return tuple(
            (
                radius * math.cos(2.0 * math.pi * idx / 8.0),
                radius * math.sin(2.0 * math.pi * idx / 8.0),
            )
            for idx in range(8)
        )
    raise ValueError(f"Unsupported demo formation: {formation}")


def world_formation_slots(
    formation: FormationShape,
    center_xy: tuple[float, float],
    heading_xy: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Project local 8-drone formation offsets into world coordinates."""

    heading = _normalize(heading_xy)
    lateral = (-heading[1], heading[0])
    center_x, center_y = center_xy
    slots: list[tuple[float, float]] = []
    for forward, lateral_offset in local_formation_offsets(formation):
        x = center_x + heading[0] * forward + lateral[0] * lateral_offset
        y = center_y + heading[1] * forward + lateral[1] * lateral_offset
        slots.append((_round3(x), _round3(y)))
    return tuple(slots)


def scaled_world_formation_slots(
    formation: FormationShape,
    center_xy: tuple[float, float],
    heading_xy: tuple[float, float],
    *,
    scale: float,
) -> tuple[tuple[float, float], ...]:
    """Project scaled local offsets into world coordinates for morph staging."""

    heading = _normalize(heading_xy)
    lateral = (-heading[1], heading[0])
    center_x, center_y = center_xy
    slots: list[tuple[float, float]] = []
    for forward, lateral_offset in local_formation_offsets(formation):
        scaled_forward = float(forward) * float(scale)
        scaled_lateral = float(lateral_offset) * float(scale)
        x = center_x + heading[0] * scaled_forward + lateral[0] * scaled_lateral
        y = center_y + heading[1] * scaled_forward + lateral[1] * scaled_lateral
        slots.append((_round3(x), _round3(y)))
    return tuple(slots)


def sampled_morph_path_clearance_m(
    drone_paths_xy: tuple[tuple[tuple[float, float], ...], ...],
    *,
    samples_per_leg: int = 31,
) -> float:
    """Return the minimum pair distance while all drones follow staged paths."""

    if not drone_paths_xy:
        return 0.0
    leg_count = min(len(path) for path in drone_paths_xy) - 1
    if leg_count <= 0:
        return min_pair_distance(tuple(path[0] for path in drone_paths_xy))
    best = float("inf")
    for leg_index in range(leg_count):
        for sample_index in range(max(int(samples_per_leg), 2)):
            alpha = sample_index / max(int(samples_per_leg) - 1, 1)
            points = tuple(
                _interpolate_xy(path[leg_index], path[leg_index + 1], alpha)
                for path in drone_paths_xy
            )
            best = min(best, min_pair_distance(points))
    return _round3(best)


def min_pair_distance(slots_xy: tuple[tuple[float, float], ...]) -> float:
    """Return the minimum pairwise distance across one slot snapshot."""

    best = float("inf")
    for i, (x_i, y_i) in enumerate(slots_xy):
        for x_j, y_j in slots_xy[i + 1 :]:
            best = min(best, math.hypot(x_i - x_j, y_i - y_j))
    return _round3(best)


def _build_demo_stages() -> list[FormationDemoStageSpec]:
    stages: list[FormationDemoStageSpec] = []

    def add(
        stage_name: str,
        kind: StageKind,
        formation: FormationMode,
        train_steps: int,
        start_waypoint_index: int,
        goal_waypoint_index: int,
        reasoning_mode: ReasoningMode,
        notes: str,
        *,
        travel_fraction: float = 1.0,
    ) -> None:
        stages.append(
            _make_stage(
                len(stages),
                stage_name,
                kind,
                formation,
                train_steps,
                start_waypoint_index,
                goal_waypoint_index,
                reasoning_mode,
                notes,
                travel_fraction=travel_fraction,
            )
        )

    def add_route(
        stage_name: str,
        train_steps: int,
        start_waypoint_index: int,
        goal_waypoint_index: int,
        reasoning_mode: ReasoningMode,
        notes: str,
    ) -> None:
        stages.append(
            _make_stage(
                len(stages),
                stage_name,
                "route",
                "mixed_route",
                train_steps,
                start_waypoint_index,
                goal_waypoint_index,
                reasoning_mode,
                notes,
            )
        )

    add("demo8_00_stage02d_handoff_replay", "handoff_replay", "line", 0, 0, 0, "fast_only", "Replay the source checkpoint at P1 and verify 8-drone shell placement before branch training.")
    add("demo8_01_p1_line_slot_lock", "prep_hold", "line", 768, 0, 0, "fast_only", "Lock the P1 line slots before any forward command; this catches handoff slot explosions early.")
    add("demo8_02_p1_line_low_speed_hold", "hold", "line", 1024, 0, 0, "fast_only", "Hold the wide line with low acceleration limits so the first travel stage does not look like instant formation loss.")
    add("demo8_03_p1_p2_line_2m_creep", "micro_travel", "line", 1024, 0, 1, "fast_only", "Move only the first 2 m of the 16 m straight segment before allowing longer travel.", travel_fraction=0.125)
    add("demo8_04_p1_p2_line_half", "mid_travel", "line", 1536, 0, 1, "fast_only", "Extend the line segment to 8 m after the 2 m creep is stable.", travel_fraction=0.5)
    add("demo8_05_p1_p2_line_full", "full_travel", "line", 3072, 0, 1, "fast_only", "Complete P1 to P2 while keeping the one-line formation.")

    add("demo8_06_p2_line_to_triangle_morph_hold", "morph_hold", "triangle", 1536, 1, 1, "fast_only", "Switch slots at P2 from line to triangle without forward travel.")
    add("demo8_07_p2_triangle_hold", "hold", "triangle", 1536, 1, 1, "fast_only", "Stabilize triangle slots at P2 before diagonal travel.")
    add("demo8_08_p2_p3_triangle_3m_creep_slow_shadow", "micro_travel", "triangle", 1536, 1, 2, "slow_shadow", "Move about 3 m on the diagonal with global route shadow only.", travel_fraction=0.2)
    add("demo8_09_p2_p3_triangle_half_slow_shadow", "mid_travel", "triangle", 2048, 1, 2, "slow_shadow", "Extend the triangle diagonal to half distance while global route remains non-controlling.", travel_fraction=0.5)
    add("demo8_10_p2_p3_triangle_full_slow_active", "full_travel", "triangle", 3072, 1, 2, "slow_active", "Use global route waypoints as visible guidance for the full triangle diagonal.")

    add("demo8_11_p3_triangle_to_rectangle_morph_hold", "morph_hold", "rectangle", 1536, 2, 2, "slow_active", "Switch from triangle to rectangle at P3 while holding the center.")
    add("demo8_12_p3_rectangle_hold", "hold", "rectangle", 1536, 2, 2, "slow_active", "Stabilize the rectangle before the straight P3 to P4 segment.")
    add("demo8_13_p3_p4_rectangle_3m_creep", "micro_travel", "rectangle", 1536, 2, 3, "slow_active", "Move about 3 m in rectangle before the 16 m straight segment.", travel_fraction=0.1875)
    add("demo8_14_p3_p4_rectangle_half", "mid_travel", "rectangle", 2048, 2, 3, "slow_active", "Extend rectangle travel to 8 m after creep stability.", travel_fraction=0.5)
    add("demo8_15_p3_p4_rectangle_full", "full_travel", "rectangle", 3072, 2, 3, "slow_active", "Complete P3 to P4 in rectangle formation.")

    add("demo8_16_p4_rectangle_to_diamond_morph_hold", "morph_hold", "diamond", 1536, 3, 3, "slow_active", "Switch from rectangle to diamond at P4.")
    add("demo8_17_p4_diamond_hold", "hold", "diamond", 1536, 3, 3, "slow_active", "Stabilize diamond slots before the descending diagonal.")
    add("demo8_18_p4_p5_diamond_3m_creep", "micro_travel", "diamond", 1536, 3, 4, "slow_active", "Move about 3 m in diamond before the full descending diagonal.", travel_fraction=0.2)
    add("demo8_19_p4_p5_diamond_half", "mid_travel", "diamond", 2048, 3, 4, "slow_active", "Extend diamond travel to half distance before full bridge.", travel_fraction=0.5)
    add("demo8_20_p4_p5_diamond_full", "full_travel", "diamond", 3072, 3, 4, "slow_active", "Complete P4 to P5 in diamond formation.")

    add("demo8_21_p5_diamond_to_circle_morph_hold", "morph_hold", "circle", 1536, 4, 4, "slow_active", "Switch from diamond to circle at P5.")
    add("demo8_22_p5_circle_hold", "hold", "circle", 1536, 4, 4, "slow_active", "Stabilize circle slots at P5 before final straight travel.")
    add("demo8_23_p5_p6_circle_2m_creep_guidance_shadow", "micro_travel", "circle", 1536, 4, 5, "guidance_shadow", "Start route guidance in shadow mode on a 2 m final straight creep only.", travel_fraction=0.125)
    add("demo8_24_p5_p6_circle_half_guidance_shadow", "mid_travel", "circle", 2048, 4, 5, "guidance_shadow", "Extend final circle travel to 8 m while guidance remains non-controlling.", travel_fraction=0.5)
    add("demo8_25_p5_p6_circle_full_guidance_shadow", "full_travel", "circle", 3072, 4, 5, "guidance_shadow", "Complete P5 to P6 with circle formation while guidance remains non-controlling.")

    # Keep each route segment tied to its own formation.
    add_route("demo8_26_route_p4_p6_diamond_circle_suffix", 3072, 3, 5, "slow_active", "Train the suffix P4->P6 so diamond-to-circle transition is stable before full-route training.")
    add_route("demo8_27_route_p3_p6_rectangle_diamond_circle_suffix", 4096, 2, 5, "slow_active", "Train P3->P6 to chain rectangle, diamond, and circle after isolated segment success.")
    add_route("demo8_28_route_p2_p6_triangle_rectangle_diamond_circle_suffix", 5120, 1, 5, "slow_active", "Train P2->P6 so triangle can hand off through every downstream formation.")
    add_route("demo8_29_route_p1_p3_line_triangle_prefix", 4096, 0, 2, "slow_active", "Train the route prefix P1->P3 to verify line-to-triangle transition under travel.")
    add_route("demo8_30_route_p1_p4_line_triangle_rectangle_prefix", 5120, 0, 3, "slow_active", "Extend the route prefix through rectangle before adding downstream shapes.")
    add_route("demo8_31_route_p1_p5_line_triangle_rectangle_diamond_prefix", 6144, 0, 4, "slow_active", "Extend the route prefix through diamond before final circle.")
    add_route("demo8_32_full_route_mixed_planner_only", 8192, 0, 5, "slow_active", "Run all five segments end-to-end with deterministic planner guidance only.")
    add_route("demo8_33_full_route_mixed_guidance_shadow", 6144, 0, 5, "guidance_shadow", "Run the full route with guidance telemetry visible but not policy-controlling.")
    add_route("demo8_34_full_route_mixed_guidance_visible_low_budget", 6144, 0, 5, "guidance_visible", "Expose low-frequency route guidance after deterministic full-route success.")
    add_route("demo8_35_full_route_mixed_isaaclab_render", 2048, 0, 5, "guidance_visible", "Render/export IsaacLab replay for visual acceptance of one-shot start-to-goal morphing.")
    return stages


def _make_stage(
    index: int,
    stage_name: str,
    kind: StageKind,
    formation: FormationMode,
    train_steps: int,
    start_waypoint_index: int,
    goal_waypoint_index: int,
    reasoning_mode: ReasoningMode,
    notes: str,
    *,
    travel_fraction: float = 1.0,
) -> FormationDemoStageSpec:
    start = WAYPOINTS[start_waypoint_index]
    goal = WAYPOINTS[goal_waypoint_index]
    if kind == "route":
        if start_waypoint_index >= goal_waypoint_index:
            path_waypoints = (start.xy,)
            route_segment_formations: tuple[FormationShape, ...] = tuple()
        else:
            path_waypoints = tuple(waypoint.xy for waypoint in WAYPOINTS[start_waypoint_index : goal_waypoint_index + 1])
            route_segment_formations = FORMATION_BY_SEGMENT[start_waypoint_index:goal_waypoint_index]
    elif start_waypoint_index == goal_waypoint_index:
        path_waypoints = (start.xy,)
        route_segment_formations = tuple()
    else:
        clipped_fraction = max(0.0, min(float(travel_fraction), 1.0))
        goal_xy = _interpolate_xy(start.xy, goal.xy, clipped_fraction)
        path_waypoints = (start.xy, goal_xy)
        route_segment_formations = (formation,) if formation != "mixed_route" else tuple()
    return FormationDemoStageSpec(
        stage_index=index,
        stage_name=stage_name,
        kind=kind,
        formation=formation,
        train_steps=int(train_steps),
        center_start_xy=start.xy,
        center_goal_xy=path_waypoints[-1],
        path_waypoints_xy=path_waypoints,
        route_segment_formations=route_segment_formations,
        reasoning_mode=reasoning_mode,
        notes=notes,
    )


def _heading(start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> tuple[float, float]:
    return _normalize((goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]))


def _interpolate_xy(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    fraction: float,
) -> tuple[float, float]:
    return (
        _round3(start_xy[0] + (goal_xy[0] - start_xy[0]) * float(fraction)),
        _round3(start_xy[1] + (goal_xy[1] - start_xy[1]) * float(fraction)),
    )


def _normalize(vector_xy: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector_xy[0], vector_xy[1])
    if norm <= 1e-9:
        return (1.0, 0.0)
    return (_round6(vector_xy[0] / norm), _round6(vector_xy[1] / norm))


def _round3(value: float) -> float:
    return round(float(value), 3)


def _round6(value: float) -> float:
    return round(float(value), 6)

