"""Stage-C config entry for the single-agent aerogate_graph experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.configs.global_config import GLOBAL_CONFIG


@dataclass(frozen=True)
class SingleGateEnvConfig:
    """Environment settings for the 2D single-drone gate task."""

    fixed_height_m: float = GLOBAL_CONFIG.fixed_flight_height_m
    dt_s: float = GLOBAL_CONFIG.planar_dt_s
    max_command_speed_mps: float = GLOBAL_CONFIG.planar_max_speed_mps
    max_accel_mps2: float = GLOBAL_CONFIG.planar_max_accel_mps2
    drone_radius_m: float = 0.35
    goal_radius_m: float = 1.5
    max_episode_steps: int = 240
    start_x_m: float = -46.0
    goal_x_m: float = 46.0
    start_y_range_m: tuple[float, float] = (-8.0, 8.0)
    goal_y_range_m: tuple[float, float] = (-8.0, 8.0)
    world_x_bounds_m: tuple[float, float] = (-55.0, 55.0)
    world_y_bounds_m: tuple[float, float] = (-18.0, 18.0)
    safety_clearance_m: float = 1.5
    progress_reward_scale: float = 6.0
    survival_reward: float = 0.05
    clearance_penalty_scale: float = 1.75
    action_l2_penalty_scale: float = 0.02
    action_smoothness_penalty_scale: float = 0.05
    goal_bonus: float = 60.0
    collision_penalty: float = -50.0
    out_of_bounds_penalty: float = -25.0
    timeout_penalty: float = -5.0


@dataclass(frozen=True)
class SingleGraphObservationConfig:
    """Fixed-size graph observation layout for the single-agent task."""

    nearest_obstacle_count: int = 6
    lookahead_waypoint_count: int = 4
    sensor_range_m: float = 28.0
    adjacency_distance_scale_m: float = 20.0
    node_feature_dim: int = 12

    @property
    def max_nodes(self) -> int:
        return 2 + self.lookahead_waypoint_count + self.nearest_obstacle_count


@dataclass(frozen=True)
class SingleGraphSACConfig:
    """Training settings for the lightweight single-agent Graph-FlashSAC agent."""

    algorithm_name: str = "graph_flashsac"
    action_dim: int = 2
    graph_hidden_dim: int = 128
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    message_passing_steps: int = 2
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    init_alpha: float = 0.02
    target_entropy: float = -2.0
    batch_size: int = 256
    replay_buffer_capacity: int = 50_000
    learning_starts: int = 1024
    updates_per_step: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    flash_update_interval: int = 8
    reward_scale_ema_decay: float = 0.995
    reward_scale_min: float = 1.0
    reward_scale_max: float = 200.0
    target_value_clip: float = 250.0
    max_grad_norm: float = 10.0
    actor_weight_decay: float = 1.0e-5
    critic_weight_decay: float = 1.0e-4
    alpha_weight_decay: float = 0.0
    weight_norm_bound: float = 150.0
    feature_norm_bound: float = 0.0
    min_alpha: float = 1.0e-5
    max_alpha: float = 1.0
    checkpoint_name: str = "graph_flashsac_single.pt"


@dataclass(frozen=True)
class SingleCheckpointPolicyConfig:
    """Checkpoint cadence and best-alias policy for single-agent training."""

    checkpoint_interval_steps: int = 128
    selection_eval_episodes: int = 8
    best_alias_name: str = "best_agent.pt"


@dataclass(frozen=True)
class SingleEvaluationGateConfig:
    """Deterministic promotion gate for single-agent runs."""

    enabled: bool = True
    eval_episodes: int = 5
    min_success_rate: float = 0.6
    max_collision_out_of_bounds_rate: float = 0.4
    min_mean_episode_reward: float | None = None


@dataclass(frozen=True)
class SingleResumePolicyConfig:
    """Safe resume defaults for single-agent training."""

    default_mode: str = "reset_train_state"
    strict_experiment_id: bool = True
    strict_observation_shapes: bool = True
    reset_optimizer_state: bool = True
    reset_entropy_state: bool = True


@dataclass(frozen=True)
class SingleExperimentConfig:
    experiment_id: str = "single_gate_gate_2d"
    num_agents: int = 1
    fixed_height_m: float = GLOBAL_CONFIG.fixed_flight_height_m
    control_mode: str = "graph_flashsac"
    planner_mode: str = "fast_only"
    environment: SingleGateEnvConfig = field(default_factory=SingleGateEnvConfig)
    observation: SingleGraphObservationConfig = field(default_factory=SingleGraphObservationConfig)
    algorithm: SingleGraphSACConfig = field(default_factory=SingleGraphSACConfig)
    checkpoint_policy: SingleCheckpointPolicyConfig = field(default_factory=SingleCheckpointPolicyConfig)
    evaluation_gate: SingleEvaluationGateConfig = field(default_factory=SingleEvaluationGateConfig)
    resume_policy: SingleResumePolicyConfig = field(default_factory=SingleResumePolicyConfig)
    notes: str = (
        "Adds the fixed-height 2D gate environment, graph observations, and a "
        "lightweight Graph-FlashSAC entry point."
    )


SINGLE_EXPERIMENT_CONFIG = SingleExperimentConfig()

