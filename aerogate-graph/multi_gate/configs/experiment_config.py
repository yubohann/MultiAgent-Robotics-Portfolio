"""Stage-D config entry for the multi-agent aerogate_graph experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from shared.configs.global_config import GLOBAL_CONFIG
from shared.core.dynamic_gate_density_2d import (
    DynamicGateDensity2DConfig,
    default_dynamic_gate_density_config,
)


EXP3_EMPTY_SCENE_MODES: tuple[str, ...] = (
    "empty_fixed_height",
    "empty_bridge_23_d13_fixed_height",
    "empty_bridge_23_d23_fixed_height",
)
GATE_2D_SCENE_MODES: tuple[str, ...] = ("gate_2d", "legacy_2d")
EXP3_GATE_SCENE_MODES: tuple[str, ...] = (
    "real_3d_fixed_height",
    "gate_bridge_23_d12_fixed_height",
)
EXP3_KINEMATIC_3D_SCENE_MODES: tuple[str, ...] = EXP3_EMPTY_SCENE_MODES + EXP3_GATE_SCENE_MODES
DYNAMIC_GATE_DENSITY_SCENE_MODES: tuple[str, ...] = ("dynamic_gate_density_8d_v1",)
FORMAL_MULTI_TEAM_SIZES: tuple[int, ...] = (2, 3, 5, 7, 8, 9)
HISTORICAL_DIAGNOSTIC_TEAM_SIZES: tuple[int, ...] = (12,)


def is_exp3_empty_scene_mode(scene_mode: str | None) -> bool:
    """Return whether one Experiment 3 scene mode belongs to the empty-scene family."""

    resolved_scene_mode = str(scene_mode or "").strip().lower().replace("-", "_")
    return resolved_scene_mode in EXP3_EMPTY_SCENE_MODES


def is_exp3_gate_scene_mode(scene_mode: str | None) -> bool:
    """Return whether one Experiment 3 scene mode belongs to the gate-scene family."""

    resolved_scene_mode = str(scene_mode or "").strip().lower().replace("-", "_")
    return resolved_scene_mode in EXP3_GATE_SCENE_MODES


def is_gate_2d_scene_mode(scene_mode: str | None) -> bool:
    """Return whether one scene mode uses the fixed-height gate 2D shell."""

    resolved_scene_mode = str(scene_mode or "").strip().lower().replace("-", "_")
    return resolved_scene_mode in GATE_2D_SCENE_MODES


def is_exp3_kinematic_3d_scene_mode(scene_mode: str | None) -> bool:
    """Return whether one Experiment 3 scene mode uses the fixed-height 3D shell stack."""

    resolved_scene_mode = str(scene_mode or "").strip().lower().replace("-", "_")
    return resolved_scene_mode in EXP3_KINEMATIC_3D_SCENE_MODES


def is_dynamic_gate_density_scene_mode(scene_mode: str | None) -> bool:
    """Return whether one scene mode uses the shared dynamic gate-density layout."""

    resolved_scene_mode = str(scene_mode or "").strip().lower().replace("-", "_")
    return resolved_scene_mode in DYNAMIC_GATE_DENSITY_SCENE_MODES


@dataclass(frozen=True)
class MultiFormationConfig:
    """Formation layout settings for the variable-size team."""

    max_columns: int = 4
    lateral_spacing_m: float = 2.4
    longitudinal_spacing_m: float = 3.0
    bootstrap_templates_enabled: bool = False
    bootstrap_lateral_spacing_m: float | None = None
    bootstrap_longitudinal_spacing_m: float | None = None
    bootstrap_three_agent_layout: str = "vee"
    bootstrap_shape_name: str | None = None
    bootstrap_initial_shape_name: str | None = None
    bootstrap_route_shape_names: tuple[str, ...] = ()
    bootstrap_route_slot_permutations: tuple[tuple[int, ...], ...] = ()
    bootstrap_slot_permutation: tuple[int, ...] = ()
    bootstrap_morph_paths_xy: tuple[tuple[tuple[float, float], ...], ...] = ()
    bootstrap_route_morph_paths_xy: tuple[tuple[tuple[tuple[float, float], ...], ...], ...] = ()
    goal_slot_tolerance_m: float = 1.4


@dataclass(frozen=True)
class MultiPlannerConfig:
    """Global route planner settings."""

    grid_resolution_m: float = 3.0
    safety_margin_m: float = 0.7
    waypoint_stride: int = 2
    max_search_iterations: int = 20_000


@dataclass(frozen=True)
class MultiGateEnvConfig:
    """Environment settings for the 2D multi-drone gate task."""

    fixed_height_m: float = GLOBAL_CONFIG.fixed_flight_height_m
    dt_s: float = GLOBAL_CONFIG.planar_dt_s
    max_command_speed_mps: float = GLOBAL_CONFIG.planar_max_speed_mps
    max_command_forward_speed_mps: float | None = None
    max_command_lateral_speed_mps: float | None = None
    max_accel_mps2: float = GLOBAL_CONFIG.planar_max_accel_mps2
    max_forward_accel_mps2: float | None = None
    max_lateral_accel_mps2: float | None = None
    drone_radius_m: float = 0.35
    inter_agent_safe_distance_m: float = 1.2
    goal_radius_m: float = 2.2
    goal_termination_enabled: bool = True
    goal_requires_slot_tolerance: bool = True
    preparation_hold_mode: bool = False
    timeout_counts_as_success: bool = False
    max_episode_steps: int = 280
    start_x_m: float = -46.0
    goal_x_m: float = 46.0
    start_y_range_m: tuple[float, float] = (-8.0, 8.0)
    goal_y_range_m: tuple[float, float] = (-8.0, 8.0)
    fixed_team_start_goal_y_m: tuple[tuple[int, float, float], ...] = ()
    path_waypoints_xy: tuple[tuple[float, float], ...] = ()
    world_x_bounds_m: tuple[float, float] = (-55.0, 55.0)
    world_y_bounds_m: tuple[float, float] = (-20.0, 20.0)
    gate_post_radius_scale: float = 1.0
    safety_clearance_m: float = 1.5
    slot_anchor_blend: float = 0.0
    guidance_tracking_penalty_scale: float = 0.0
    guidance_escape_soft_margin_m: float = 1.0
    guidance_escape_penalty_scale: float = 0.0
    boundary_soft_margin_m: float = 2.25
    boundary_proximity_penalty_scale: float = 0.0
    progress_reward_scale: float = 7.0
    ungated_progress_reward_fraction: float = 0.0
    goal_proximity_reward_scale: float = 0.0
    goal_proximity_sigma_m: float = 6.0
    slot_improvement_scale: float = 3.0
    slot_error_penalty_scale: float = 1.5
    max_slot_error_penalty_scale: float = 0.0
    max_slot_escape_soft_margin_ratio: float = 0.0
    max_slot_escape_penalty_scale: float = 0.0
    survival_reward: float = 0.1
    clearance_penalty_scale: float = 2.25
    separation_penalty_scale: float = 2.0
    separation_warning_ratio: float = 1.35
    separation_proximity_penalty_scale: float = 0.0
    progress_separation_hard_stop_margin_m: float = 0.0
    progress_max_slot_hard_stop_ratio: float = 0.0
    action_l2_penalty_scale: float = 0.015
    action_smoothness_penalty_scale: float = 0.03
    action_safety_shield_enabled: bool = False
    action_safety_shield_separation_margin_m: float = 0.0
    action_safety_shield_brake_scale: float = 0.0
    action_safety_shield_pair_closing_brake_only: bool = False
    action_safety_shield_pair_time_horizon_s: float = 0.0
    action_safety_shield_repulsion_scale: float = 0.0
    action_safety_shield_outward_slot_bias_scale: float = 0.0
    action_safety_shield_priority_team_size_limit: int = 3
    action_safety_shield_boundary_margin_m: float = 0.0
    action_safety_shield_boundary_brake_scale: float = 0.0
    action_safety_shield_boundary_inward_scale: float = 0.0
    action_safety_shield_guidance_margin_m: float = 0.0
    action_safety_shield_guidance_inward_scale: float = 0.0
    action_safety_shield_gate_channel_enabled: bool = False
    action_safety_shield_gate_channel_lookahead_m: float = 0.0
    action_safety_shield_gate_channel_behind_m: float = 0.0
    action_safety_shield_gate_channel_lateral_gain: float = 0.0
    action_safety_shield_gate_channel_max_lateral_mps: float = 0.0
    action_safety_shield_gate_channel_slowdown_scale: float = 0.0
    action_safety_shield_post_gate_cruise_enabled: bool = False
    action_safety_shield_post_gate_cruise_min_forward_mps: float = 0.0
    action_safety_shield_post_gate_cruise_gate_behind_m: float = 0.0
    action_safety_shield_post_gate_cruise_goal_margin_m: float = 0.0
    action_safety_shield_post_gate_cruise_min_pair_distance_m: float = 0.0
    action_safety_shield_post_gate_cruise_min_clearance_m: float = 0.0
    action_safety_shield_obstacle_margin_m: float = 0.0
    action_safety_shield_obstacle_brake_scale: float = 0.0
    action_safety_shield_obstacle_repulsion_scale: float = 0.0
    action_safety_shield_obstacle_time_horizon_s: float = 0.0
    goal_bonus: float = 90.0
    gate_post_collision_penalty: float = -90.0
    agent_collision_penalty: float = -70.0
    out_of_bounds_penalty: float = -40.0
    height_escape_penalty: float = -140.0
    side_bypass_penalty: float = -140.0
    corridor_miss_penalty: float = -140.0
    formation_line_collapse_min_lateral_bands: int = 0
    formation_line_collapse_band_width_m: float = 0.50
    formation_line_collapse_task_ratio: float = 0.70
    formation_line_collapse_terminal: bool = False
    formation_line_collapse_penalty_scale: float = 0.0
    formation_line_collapse_terminal_penalty: float = -120.0
    timeout_penalty: float = -12.0
    timeout_goal_distance_penalty_scale: float = 0.0


@dataclass(frozen=True)
class MultiSceneConfig:
    """Scene semantics for paper-oriented experiment variants."""

    scene_mode: str = "gate_2d"
    render_backend: str = "vector_2d"
    render_real_gate: bool = False
    render_real_drone_shell: bool = False
    kinematic_only: bool = True
    disable_motors: bool = True
    fixed_height_locked: bool = True
    drone_asset: str = "5_in_drone.usd"
    notes: str = "Fixed-height gate 2D training scene."


@dataclass(frozen=True)
class MultiReasoningConfig:
    """High-level reasoning stack toggles for the paper experiment matrix."""

    global_planner_enabled: bool = True
    route_guidance_enabled: bool = False
    guidance_shadow_mode: bool = False
    guidance_async_enabled: bool = False
    guidance_cache_enabled: bool = False
    guidance_provider: str = "none"
    guidance_base_url: str = "http://127.0.0.1:11434"
    guidance_model_name: str = "local-guidance-model"
    guidance_timeout_s: float = 30.0
    guidance_temperature: float = 0.1
    guidance_prompt_version: str = "exp3_v1"
    guidance_stage_name: str = ""
    guidance_node_enabled: bool = True
    inference_budget_hz: float = 0.0
    notes: str = "Default multi-agent graph line with global route guidance."


@dataclass(frozen=True)
class MultiGraphObservationConfig:
    """Fixed-size graph observation layout for the multi-agent task."""

    nearest_obstacle_count: int = 8
    lookahead_waypoint_count: int = 6
    guidance_node_count: int = 0
    node_feature_dim: int = 16
    adjacency_distance_scale_m: float = 24.0
    max_agents_for_nodes: int | None = None

    @property
    def max_nodes(self) -> int:
        max_agents = int(self.max_agents_for_nodes or GLOBAL_CONFIG.max_agents_soft)
        return (
            max_agents
            + max_agents
            + self.lookahead_waypoint_count
            + self.nearest_obstacle_count
            + self.guidance_node_count
            + 1
        )


@dataclass(frozen=True)
class MultiGraphMASACConfig:
    """Training settings for the lightweight Graph-FlashSAC agent."""

    algorithm_name: str = "graph_flashsac"
    action_dim: int = 2
    graph_hidden_dim: int = 160
    actor_hidden_dim: int = 160
    critic_hidden_dim: int = 192
    message_passing_steps: int = 2
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    init_alpha: float = 0.02
    target_entropy_per_agent: float = -2.0
    batch_size: int = 512
    replay_buffer_capacity: int = 30_000
    learning_starts: int = 2048
    updates_per_step: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    flash_update_interval: int = 8
    reward_scale_ema_decay: float = 0.995
    reward_scale_min: float = 1.0
    reward_scale_max: float = 400.0
    target_value_clip: float = 350.0
    max_grad_norm: float = 10.0
    actor_weight_decay: float = 1.0e-5
    critic_weight_decay: float = 1.0e-4
    alpha_weight_decay: float = 0.0
    weight_norm_bound: float = 180.0
    feature_norm_bound: float = 0.0
    min_alpha: float = 1.0e-5
    max_alpha: float = 1.0
    enable_safety_critic: bool = False
    safety_penalty_scale: float = 0.08
    actor_head_mode: str = "single"
    modular_gate_entropy_scale: float = 0.0
    behavior_anchor_loss_scale: float = 0.0
    behavior_anchor_non_failure_only: bool = True
    checkpoint_name: str = "graph_flashsac_multi.pt"


@dataclass(frozen=True)
class MultiBehaviorCloningConfig:
    """Behavior-cloning warm-start settings for the multi-agent actor."""

    expert_episodes: int = 12
    max_steps_per_episode: int = 320
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    validation_split: float = 0.1
    target_log_std: float = -2.0
    log_std_penalty_scale: float = 0.01
    dataset_name: str = "multi_expert_dataset.npz"
    actor_checkpoint_name: str = "graph_flashsac_multi_bc_actor.pt"


@dataclass(frozen=True)
class MultiFailureReplayConfig:
    """Failure-aware replay settings for off-policy Graph-FlashSAC updates."""

    enabled: bool = True
    failure_replay_ratio: float = 0.3
    near_miss_clearance_m: float = 1.0
    near_miss_pair_distance_m: float = 1.45
    slot_error_spike_m: float = 3.0
    hard_failure_reasons: tuple[str, ...] = (
        "gate_post_collision",
        "agent_collision",
        "out_of_bounds",
        "formation_line_collapse_failure",
    )
    safety_cost_scale: float = 1.0


@dataclass(frozen=True)
class MultiSizeInvarianceConfig:
    """Variable-size policy settings and bucketed evaluation defaults."""

    enabled: bool = True
    team_size_sampling_mode: str = "uniform_buckets"
    bucket_team_sizes: tuple[int, ...] = FORMAL_MULTI_TEAM_SIZES
    bucket_eval_episodes: int = 0
    min_bucket_success_rate: float = 0.0


@dataclass(frozen=True)
class MultiDaggerConfig:
    """DAgger-style online correction settings for BC warm start."""

    enabled: bool = True
    iterations: int = 2
    rollout_episodes_per_iteration: int = 2
    max_steps_per_episode: int = 220
    query_every_n_steps: int = 5
    max_corrections_per_iteration: int = 512
    bc_epochs_per_iteration: int = 2
    risk_clearance_m: float = 1.2
    risk_pair_distance_m: float = 1.55
    risk_slot_error_m: float = 3.2


@dataclass(frozen=True)
class MultiBenchmarkConfig:
    """Default benchmark dimensions for difficulty x team-size reporting."""

    difficulty_names: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    team_sizes: tuple[int, ...] = FORMAL_MULTI_TEAM_SIZES
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class MultiCheckpointPolicyConfig:
    """Checkpoint cadence and best-alias policy for multi-agent training."""

    checkpoint_interval_steps: int = 128
    selection_eval_episodes: int = 4
    best_alias_name: str = "best_agent.pt"


@dataclass(frozen=True)
class MultiEvaluationGateConfig:
    """Deterministic promotion gate for multi-agent runs."""

    enabled: bool = True
    eval_episodes: int = 4
    min_success_rate: float = 0.5
    max_agent_collision_rate: float = 0.49
    max_hard_failure_rate: float = 0.75
    max_safety_violation_rate: float | None = None
    min_bucket_success_rate: float | None = None
    min_mean_episode_reward: float | None = None


@dataclass(frozen=True)
class MultiResumePolicyConfig:
    """Safe resume defaults for multi-agent training."""

    default_mode: str = "reset_train_state"
    strict_experiment_id: bool = True
    strict_observation_shapes: bool = True
    reset_optimizer_state: bool = True
    reset_entropy_state: bool = True


@dataclass(frozen=True)
class MultiExperimentConfig:
    experiment_id: str = "multi_gate_gate_2d"
    paper_track: str = "legacy_stage_d"
    paper_variant: str = "variable"
    min_agents: int = 2
    default_agents: int = 4
    max_agents_soft: int = 12
    fixed_height_m: float = GLOBAL_CONFIG.fixed_flight_height_m
    control_mode: str = "graph_flashsac"
    planner_mode: str = "global_route_planner_plus_graph_policy"
    scene: MultiSceneConfig = field(default_factory=MultiSceneConfig)
    reasoning: MultiReasoningConfig = field(default_factory=MultiReasoningConfig)
    formation: MultiFormationConfig = field(default_factory=MultiFormationConfig)
    planner: MultiPlannerConfig = field(default_factory=MultiPlannerConfig)
    environment: MultiGateEnvConfig = field(default_factory=MultiGateEnvConfig)
    observation: MultiGraphObservationConfig = field(default_factory=MultiGraphObservationConfig)
    algorithm: MultiGraphMASACConfig = field(default_factory=MultiGraphMASACConfig)
    imitation: MultiBehaviorCloningConfig = field(default_factory=MultiBehaviorCloningConfig)
    failure_replay: MultiFailureReplayConfig = field(default_factory=MultiFailureReplayConfig)
    size_invariance: MultiSizeInvarianceConfig = field(default_factory=MultiSizeInvarianceConfig)
    dagger: MultiDaggerConfig = field(default_factory=MultiDaggerConfig)
    benchmark: MultiBenchmarkConfig = field(default_factory=MultiBenchmarkConfig)
    checkpoint_policy: MultiCheckpointPolicyConfig = field(default_factory=MultiCheckpointPolicyConfig)
    evaluation_gate: MultiEvaluationGateConfig = field(default_factory=MultiEvaluationGateConfig)
    resume_policy: MultiResumePolicyConfig = field(default_factory=MultiResumePolicyConfig)
    dynamic_gate_density: DynamicGateDensity2DConfig = field(default_factory=default_dynamic_gate_density_config)
    notes: str = (
        "Adds the variable-size multi-agent 2D gate environment, formation "
        "slots, a global route global planner, and a fast-think graph policy."
    )


def build_multi_experiment_config(
    *,
    experiment_id: str = "multi_gate_gate_2d",
    paper_track: str = "legacy_stage_d",
    paper_variant: str = "variable",
    min_agents: int = 2,
    default_agents: int = 4,
    max_agents_soft: int = 12,
    notes: str | None = None,
    checkpoint_name: str | None = None,
    bc_actor_checkpoint_name: str | None = None,
    control_mode: str = "graph_flashsac",
    planner_mode: str = "global_route_planner_plus_graph_policy",
    scene_config: MultiSceneConfig | None = None,
    reasoning_config: MultiReasoningConfig | None = None,
    observation_config: MultiGraphObservationConfig | None = None,
    dynamic_gate_density_config: DynamicGateDensity2DConfig | None = None,
    size_invariance_config: MultiSizeInvarianceConfig | None = None,
    evaluation_gate_config: MultiEvaluationGateConfig | None = None,
) -> MultiExperimentConfig:
    """Build one multi-agent experiment config entry."""

    resolved_notes = notes or (
        "Adds the variable-size multi-agent 2D gate environment, formation "
        "slots, a global route global planner, and a fast-think graph policy."
    )
    algorithm = MultiGraphMASACConfig(
        checkpoint_name=checkpoint_name or "graph_flashsac_multi.pt",
    )
    imitation = MultiBehaviorCloningConfig(
        actor_checkpoint_name=bc_actor_checkpoint_name or "graph_flashsac_multi_bc_actor.pt",
    )
    return MultiExperimentConfig(
        experiment_id=experiment_id,
        paper_track=paper_track,
        paper_variant=paper_variant,
        min_agents=min_agents,
        default_agents=default_agents,
        max_agents_soft=max_agents_soft,
        control_mode=control_mode,
        planner_mode=planner_mode,
        scene=scene_config or MultiSceneConfig(),
        reasoning=reasoning_config or MultiReasoningConfig(),
        observation=observation_config or MultiGraphObservationConfig(),
        dynamic_gate_density=dynamic_gate_density_config or default_dynamic_gate_density_config(),
        algorithm=algorithm,
        imitation=imitation,
        size_invariance=size_invariance_config or MultiSizeInvarianceConfig(),
        evaluation_gate=evaluation_gate_config or MultiEvaluationGateConfig(),
        notes=resolved_notes,
    )


def build_fixed_team_experiment_config(team_size: int) -> MultiExperimentConfig:
    """Build a fixed-team-size multi-agent config preset."""

    if team_size < 2:
        raise ValueError(f"Fixed multi-agent presets require at least 2 agents, got {team_size}")
    max_fixed_team_agents = int(GLOBAL_CONFIG.max_fixed_team_agents)
    if team_size > max_fixed_team_agents:
        raise ValueError(
            "Fixed multi-agent presets cannot exceed the fixed-team cap "
            f"({max_fixed_team_agents}), got {team_size}"
        )
    return build_multi_experiment_config(
        experiment_id=f"multi_gate_gate_2d_fixed_{team_size:02d}",
        paper_variant=f"fixed_{team_size:02d}",
        min_agents=team_size,
        default_agents=team_size,
        max_agents_soft=team_size,
        checkpoint_name=f"graph_flashsac_multi_{team_size:02d}d.pt",
        bc_actor_checkpoint_name=f"graph_flashsac_multi_{team_size:02d}d_bc_actor.pt",
        size_invariance_config=MultiSizeInvarianceConfig(
            enabled=False,
            team_size_sampling_mode="fixed",
            bucket_team_sizes=(team_size,),
        ),
        notes=(
            f"Fixed-team preset for {team_size} agents in the 2D gate multi-agent "
            "Graph-FlashSAC experiment."
        ),
    )


def build_dynamic_gate_density_8d_config() -> MultiExperimentConfig:
    """Build the 8-drone dynamic gate-density continuation curriculum preset."""

    from shared.runtime.exp3_formation_demo import (
        FORMATION_BY_SEGMENT,
        build_stage_execution_plan,
        resolve_route_slot_permutations_and_morph_paths,
    )

    demo_plan = build_stage_execution_plan("demo8_35_full_route_mixed_isaaclab_render")
    demo_stage = demo_plan.stage
    demo_path_waypoints = tuple((float(x), float(y)) for x, y in demo_stage.path_waypoints_xy)
    demo_route_shapes = tuple(str(shape) for shape in (demo_stage.route_segment_formations or FORMATION_BY_SEGMENT))
    demo_route_permutations, demo_route_morph_paths = resolve_route_slot_permutations_and_morph_paths(demo_plan)
    gate_cfg = replace(
        default_dynamic_gate_density_config(),
        world_x_bounds_m=demo_plan.manifest.world_x_bounds_m,
        world_y_bounds_m=demo_plan.manifest.world_y_bounds_m,
        start_x_m=float(demo_path_waypoints[0][0]),
        goal_x_m=float(demo_path_waypoints[-1][0]),
        fixed_height_m=float(demo_plan.manifest.fixed_height_m),
        gate_region_x_m=(-18.0, 24.0),
        moving_clip_x_m=(-40.0, 40.0),
        moving_clip_y_m=(-13.0, 13.0),
    )
    scene_config = MultiSceneConfig(
        scene_mode=gate_cfg.scene_mode,
        render_backend="vector_2d_shared_dynamic_gate",
        render_real_gate=False,
        render_real_drone_shell=True,
        kinematic_only=True,
        disable_motors=True,
        fixed_height_locked=True,
        drone_asset="5_in_drone.usd",
        notes=(
            "2D fixed-height 8-drone dynamic gate-density scene. Training, eval, "
            "and replay must use shared.core.dynamic_gate_density_2d for live "
            "gate centers, velocities, non-overlap projection, and terminal collisions."
        ),
    )
    env_config = MultiGateEnvConfig(
        fixed_height_m=gate_cfg.fixed_height_m,
        dt_s=0.1,
        max_command_speed_mps=1.15,
        max_accel_mps2=0.75,
        max_episode_steps=1474,
        start_x_m=gate_cfg.start_x_m,
        goal_x_m=gate_cfg.goal_x_m,
        start_y_range_m=(float(demo_path_waypoints[0][1]), float(demo_path_waypoints[0][1])),
        goal_y_range_m=(float(demo_path_waypoints[-1][1]), float(demo_path_waypoints[-1][1])),
        fixed_team_start_goal_y_m=((8, float(demo_path_waypoints[0][1]), float(demo_path_waypoints[-1][1])),),
        path_waypoints_xy=demo_path_waypoints,
        world_x_bounds_m=gate_cfg.world_x_bounds_m,
        world_y_bounds_m=gate_cfg.world_y_bounds_m,
        drone_radius_m=gate_cfg.drone_radius_m,
        inter_agent_safe_distance_m=1.2,
        goal_radius_m=2.2,
        slot_anchor_blend=1.0,
        guidance_tracking_penalty_scale=1.8,
        boundary_soft_margin_m=2.25,
        boundary_proximity_penalty_scale=12.0,
        progress_reward_scale=5.0,
        goal_proximity_reward_scale=2.0,
        slot_error_penalty_scale=2.2,
        max_slot_error_penalty_scale=0.8,
        safety_clearance_m=0.8,
        separation_warning_ratio=1.35,
        separation_proximity_penalty_scale=2.0,
        action_safety_shield_enabled=True,
        action_safety_shield_separation_margin_m=0.62,
        action_safety_shield_brake_scale=0.9,
        action_safety_shield_repulsion_scale=0.65,
        action_safety_shield_boundary_margin_m=2.0,
        action_safety_shield_boundary_brake_scale=0.9,
        action_safety_shield_boundary_inward_scale=0.6,
        gate_post_collision_penalty=-120.0,
        agent_collision_penalty=-90.0,
        formation_line_collapse_min_lateral_bands=3,
        formation_line_collapse_band_width_m=0.50,
        formation_line_collapse_task_ratio=0.70,
        formation_line_collapse_terminal=True,
        formation_line_collapse_penalty_scale=18.0,
        formation_line_collapse_terminal_penalty=-120.0,
    )
    observation_config = MultiGraphObservationConfig(
        # Keep the graph tensor shape compatible with the rt8 demo8 formation
        # handoff checkpoint: node_features=(85, 18), action_mask=(34,).
        # Dynamic gate-density stages need more lookahead gate posts than the
        # original empty-formation handoff. Reducing the graph-only agent node
        # budget from 34 to 30 keeps max_nodes at 85 while exposing 16 obstacle
        # nodes; action_mask remains 34 via max_agents_soft.
        nearest_obstacle_count=16,
        lookahead_waypoint_count=6,
        guidance_node_count=2,
        node_feature_dim=18,
        max_agents_for_nodes=30,
        adjacency_distance_scale_m=18.0,
    )
    reasoning_config = MultiReasoningConfig(
        global_planner_enabled=True,
        route_guidance_enabled=True,
        guidance_shadow_mode=False,
        guidance_async_enabled=True,
        guidance_cache_enabled=True,
        guidance_provider="local_http",
        guidance_model_name="local-guidance-model",
        guidance_prompt_version="dynamic_gate_density_8d_v1",
        guidance_node_enabled=True,
        inference_budget_hz=0.5,
        notes="Slow/fast/safety stack for dynamic moving-gate active avoidance.",
    )
    base_config = build_multi_experiment_config(
        experiment_id="multi_gate_dynamic_gate_density_8d_v1",
        paper_track="e2d_dynamic_gate_density",
        paper_variant="dynamic_gate_density_8d_v1",
        min_agents=2,
        default_agents=8,
        max_agents_soft=34,
        control_mode="graph_flashsac_dynamic_gate_density_2d",
        planner_mode="route_plan_guidance_guided_dynamic_gate_planner_plus_fast_graph_policy",
        checkpoint_name="graph_flashsac_multi_dynamic_gate_density_8d.pt",
        bc_actor_checkpoint_name="graph_flashsac_multi_dynamic_gate_density_8d_bc_actor.pt",
        scene_config=scene_config,
        reasoning_config=reasoning_config,
        observation_config=observation_config,
        dynamic_gate_density_config=gate_cfg,
        size_invariance_config=MultiSizeInvarianceConfig(
            enabled=True,
            team_size_sampling_mode="bucket",
            bucket_team_sizes=FORMAL_MULTI_TEAM_SIZES,
            bucket_eval_episodes=5,
            min_bucket_success_rate=0.8,
        ),
        evaluation_gate_config=MultiEvaluationGateConfig(
            enabled=True,
            eval_episodes=10,
            min_success_rate=0.8,
            max_agent_collision_rate=0.1,
            max_hard_failure_rate=0.25,
            max_safety_violation_rate=0.25,
            min_bucket_success_rate=0.8,
        ),
        notes=(
            "E2D-2/E2D-3 continuation from the demo8 8-drone formation-morphing "
            "checkpoint (line/triangle/rectangle/diamond/circle route) into "
            "tight-opening dynamic gates up to 60 obstacles and 2 m/s."
        ),
    )
    return replace(
        base_config,
        environment=env_config,
        formation=replace(
            base_config.formation,
            max_columns=8,
            lateral_spacing_m=1.7,
            longitudinal_spacing_m=3.0,
            bootstrap_templates_enabled=True,
            bootstrap_shape_name=str(demo_route_shapes[0]),
            bootstrap_initial_shape_name=None,
            bootstrap_route_shape_names=demo_route_shapes,
            bootstrap_route_slot_permutations=demo_route_permutations,
            bootstrap_route_morph_paths_xy=demo_route_morph_paths,
            goal_slot_tolerance_m=2.4,
        ),
        algorithm=replace(
            base_config.algorithm,
            enable_safety_critic=False,
            checkpoint_name="graph_flashsac_multi_dynamic_gate_density_8d.pt",
        ),
        resume_policy=replace(
            base_config.resume_policy,
            strict_experiment_id=False,
            strict_observation_shapes=True,
        ),
    )


def build_exp3_paper_experiment_config(variant: str) -> MultiExperimentConfig:
    """Build the paper-oriented Experiment 3 presets."""

    normalized_variant = str(variant).strip().lower().replace("-", "_")
    if normalized_variant not in {"e3_baseline", "e3_main", "e3_guidance"}:
        raise ValueError(f"Unsupported Experiment 3 paper variant: {variant}")

    global_planner_enabled = normalized_variant != "e3_baseline"
    route_guidance_enabled = normalized_variant == "e3_guidance"
    observation_config = MultiGraphObservationConfig(
        nearest_obstacle_count=8,
        lookahead_waypoint_count=6,
        guidance_node_count=0 if normalized_variant == "e3_baseline" else (2 if route_guidance_enabled else 1),
        node_feature_dim=16 if normalized_variant == "e3_baseline" else 18,
        adjacency_distance_scale_m=24.0,
    )
    scene_config = MultiSceneConfig(
        scene_mode="real_3d_fixed_height",
        render_backend="isaaclab_shell_3d",
        render_real_gate=True,
        render_real_drone_shell=True,
        kinematic_only=True,
        disable_motors=True,
        fixed_height_locked=True,
        drone_asset="5_in_drone.usd",
        notes=(
            "Paper Experiment 3 scene: real 3D gate, fixed 4m height, "
            "kinematic shell-only drones, motors disabled."
        ),
    )
    reasoning_config = MultiReasoningConfig(
        global_planner_enabled=global_planner_enabled,
        route_guidance_enabled=route_guidance_enabled,
        guidance_shadow_mode=False,
        guidance_async_enabled=route_guidance_enabled,
        guidance_cache_enabled=route_guidance_enabled,
        guidance_provider="local_http" if route_guidance_enabled else "none",
        guidance_base_url="http://127.0.0.1:11434",
        guidance_model_name="local-guidance-model",
        guidance_timeout_s=30.0,
        guidance_temperature=0.1,
        guidance_prompt_version="exp3_v1",
        guidance_stage_name="",
        guidance_node_enabled=global_planner_enabled or route_guidance_enabled,
        inference_budget_hz=0.5 if route_guidance_enabled else (1.0 if global_planner_enabled else 0.0),
        notes=(
            "Fast-think only baseline."
            if normalized_variant == "e3_baseline"
            else (
                "Main evaluation line with planner guidance plus graph policy."
                if normalized_variant == "e3_main"
                else "Enhanced evaluation line with low-frequency asynchronous route guidance."
            )
        ),
    )
    planner_mode = (
        "graph_policy_only"
        if normalized_variant == "e3_baseline"
        else (
            "global_route_planner_plus_graph_policy"
            if normalized_variant == "e3_main"
            else "route_plan_guidance_guided_global_planner_plus_graph_policy"
        )
    )
    base = build_multi_experiment_config(
        experiment_id=f"exp3_real_3d_kinematic_{normalized_variant}",
        paper_track="exp3_real_3d_kinematic",
        paper_variant=normalized_variant,
        min_agents=2,
        default_agents=5,
        max_agents_soft=12,
        control_mode="graph_flashsac_kinematic_3d",
        planner_mode=planner_mode,
        checkpoint_name=f"{normalized_variant}_graph_flashsac_multi.pt",
        bc_actor_checkpoint_name=f"{normalized_variant}_graph_flashsac_multi_bc_actor.pt",
        scene_config=scene_config,
        reasoning_config=reasoning_config,
        observation_config=observation_config,
        size_invariance_config=MultiSizeInvarianceConfig(
            enabled=True,
            team_size_sampling_mode="uniform_buckets",
            bucket_team_sizes=FORMAL_MULTI_TEAM_SIZES,
            bucket_eval_episodes=2,
            min_bucket_success_rate=0.2,
        ),
        evaluation_gate_config=MultiEvaluationGateConfig(
            enabled=True,
            eval_episodes=4,
            min_success_rate=0.45,
            max_agent_collision_rate=0.45,
            max_hard_failure_rate=0.7,
            max_safety_violation_rate=0.7,
            min_bucket_success_rate=0.2,
            min_mean_episode_reward=None,
        ),
        notes=(
            "Paper-oriented Experiment 3 preset: real 3D gate, fixed-height 4m "
            "kinematic training, shell-visible drones, variable-size Graph-FlashSAC."
        ),
    )
    return replace(
        base,
        algorithm=replace(
            base.algorithm,
            critic_lr=2.0e-4,
            batch_size=64,
            learning_starts=1024,
        ),
        failure_replay=replace(
            base.failure_replay,
            failure_replay_ratio=0.15,
        ),
        environment=replace(
            base.environment,
            slot_anchor_blend=1.0,
            guidance_tracking_penalty_scale=1.8,
            boundary_soft_margin_m=2.25,
            boundary_proximity_penalty_scale=12.0,
            separation_warning_ratio=1.35,
            separation_proximity_penalty_scale=6.0,
            max_slot_error_penalty_scale=0.75,
        ),
    )


def build_exp3_curriculum_experiment_config(
    variant: str,
    *,
    scene_type: str,
    scene_mode_override: str | None = None,
    team_sizes: Sequence[int],
    environment_overrides: dict[str, object] | None = None,
    formation_overrides: dict[str, object] | None = None,
    observation_overrides: dict[str, object] | None = None,
    reasoning_overrides: dict[str, object] | None = None,
    algorithm_overrides: dict[str, object] | None = None,
    stage_name: str | None = None,
    notes: str | None = None,
) -> MultiExperimentConfig:
    """Build one Experiment 3 stage config for the staged paper curriculum."""

    base = build_exp3_paper_experiment_config(variant)
    resolved_scene_type = str(scene_type).strip().lower().replace("-", "_")
    resolved_scene_mode = (
        resolved_scene_type
        if scene_mode_override is None
        else str(scene_mode_override).strip().lower().replace("-", "_")
    )
    if not is_exp3_kinematic_3d_scene_mode(resolved_scene_mode):
        raise ValueError(f"Unsupported Experiment 3 curriculum scene_type/scene_mode: {scene_type}, {scene_mode_override}")

    resolved_team_sizes = tuple(sorted({int(size) for size in team_sizes}))
    if not resolved_team_sizes:
        raise ValueError("Experiment 3 curriculum config requires at least one team size.")
    if resolved_team_sizes[0] < 2 or resolved_team_sizes[-1] > int(base.max_agents_soft):
        raise ValueError(
            "Experiment 3 curriculum team sizes must stay within "
            f"[2, {int(base.max_agents_soft)}], got {resolved_team_sizes}"
        )

    default_agents = 5 if 5 in resolved_team_sizes else resolved_team_sizes[0]
    if is_exp3_empty_scene_mode(resolved_scene_mode):
        scene_notes = (
            "Paper Experiment 3 empty-scene bridge stage: fixed 4m height, kinematic shell-only drones, "
            "motors disabled, no gate_post obstacles."
            if resolved_scene_mode != "empty_fixed_height"
            else (
                "Paper Experiment 3 empty-scene stage: fixed 4m height, kinematic shell-only drones, "
                "motors disabled, no gate_post obstacles."
            )
        )
    else:
        scene_notes = (
            "Paper Experiment 3 gate bridge stage: real 3D gate, fixed 4m height, "
            "kinematic shell-only drones, motors disabled."
            if resolved_scene_mode != "real_3d_fixed_height"
            else (
                "Paper Experiment 3 gate stage: real 3D gate, fixed 4m height, "
                "kinematic shell-only drones, motors disabled."
            )
        )
    scene_config = replace(
        base.scene,
        scene_mode=resolved_scene_mode,
        render_backend="isaaclab_shell_3d",
        render_real_gate=is_exp3_gate_scene_mode(resolved_scene_mode),
        render_real_drone_shell=True,
        notes=scene_notes,
    )
    environment_config = (
        base.environment
        if not environment_overrides
        else replace(base.environment, **dict(environment_overrides))
    )
    formation_config = (
        base.formation
        if not formation_overrides
        else replace(base.formation, **dict(formation_overrides))
    )
    observation_config = base.observation
    if stage_name is not None:
        base_reasoning = replace(base.reasoning, guidance_stage_name=str(stage_name))
    else:
        base_reasoning = base.reasoning
    if observation_overrides:
        observation_config = replace(base.observation, **dict(observation_overrides))
    reasoning_config = (
        base_reasoning
        if not reasoning_overrides
        else replace(base_reasoning, **dict(reasoning_overrides))
    )
    algorithm_config = (
        base.algorithm
        if not algorithm_overrides
        else replace(base.algorithm, **dict(algorithm_overrides))
    )
    size_invariance_config = replace(
        base.size_invariance,
        bucket_team_sizes=resolved_team_sizes,
    )
    resolved_notes = notes or (
        "Experiment 3 staged curriculum config: "
        f"scene_type={resolved_scene_type}, team_sizes={resolved_team_sizes}."
    )
    return replace(
        base,
        default_agents=int(default_agents),
        scene=scene_config,
        reasoning=reasoning_config,
        formation=formation_config,
        environment=environment_config,
        observation=observation_config,
        algorithm=algorithm_config,
        size_invariance=size_invariance_config,
        notes=resolved_notes,
    )


def override_multi_scene_config(
    experiment_config: MultiExperimentConfig,
    *,
    scene_mode: str | None = None,
    render_real_gate: bool | None = None,
    render_real_drone_shell: bool | None = None,
    notes: str | None = None,
) -> MultiExperimentConfig:
    """Return a copy of one multi-agent config with scene-only overrides applied."""

    resolved_scene = replace(
        experiment_config.scene,
        scene_mode=experiment_config.scene.scene_mode if scene_mode is None else str(scene_mode),
        render_real_gate=(
            bool(experiment_config.scene.render_real_gate)
            if render_real_gate is None
            else bool(render_real_gate)
        ),
        render_real_drone_shell=(
            bool(experiment_config.scene.render_real_drone_shell)
            if render_real_drone_shell is None
            else bool(render_real_drone_shell)
        ),
        notes=experiment_config.scene.notes if notes is None else str(notes),
    )
    return replace(experiment_config, scene=resolved_scene)


MULTI_EXPERIMENT_CONFIG = build_multi_experiment_config(
    size_invariance_config=MultiSizeInvarianceConfig(
        bucket_eval_episodes=2,
        min_bucket_success_rate=0.25,
    ),
    evaluation_gate_config=MultiEvaluationGateConfig(
        min_bucket_success_rate=0.25,
        max_safety_violation_rate=0.75,
    ),
)

