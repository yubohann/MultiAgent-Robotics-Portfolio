"""Frozen constants for collection protocols, split certificates, and coverage."""

from __future__ import annotations

import re

COLLECTION_PROTOCOL_SCHEMA = "org.rivermark.benchmark.collection-protocol.v1"
T1_COLLECTION_PROTOCOL_SCHEMA = "org.rivermark.benchmark.t1-collection-protocol.v2"
NATIVE_T2_CANARY_PROTOCOL_SCHEMA = (
    "org.rivermark.benchmark.native-t2-canary-protocol.v1"
)
NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA = (
    "org.rivermark.benchmark.native-t2-canary-protocol.v2"
)
NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA = (
    "org.rivermark.benchmark.native-t2-canary-protocol.v3"
)
COVERAGE_REPORT_SCHEMA = "org.rivermark.benchmark.coverage-report.v1"
T1_COVERAGE_REPORT_SCHEMA = "org.rivermark.benchmark.t1-coverage-report.v2"
SEED_DERIVATION = "sha256_utf8_lines_uint32_v1"
POWER_METHOD = "paired_normal_approximation_bonferroni"
COLLECTION_BINDING_KEYS = frozenset(
    {"protocol_id", "protocol_sha256", "cell_id", "split", "episode_index", "episode_seed"}
)
COLLECTION_SPLITS = frozenset({"train", "inner_dev", "validation", "blind_test", "ood_test"})
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_VALUE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SPLITS = COLLECTION_SPLITS
_EVALUATION_SPLITS = frozenset({"validation", "blind_test", "ood_test"})
_AXES = (
    "layout",
    "route",
    "route_family",
    "target_count",
    "height",
    "region",
    "occlusion",
    "density",
    "appearance",
    "dynamics",
    "lighting",
    "weather",
    "initial_condition",
    "start_anchor",
    "target_region",
    "visibility_bucket",
    "communication",
    "control_latency",
    "agent_dropout",
)
_AXIS_SET = frozenset(_AXES)
_SAME_LAYOUT_HOLDOUT_AXES = frozenset(
    {"route_family", "start_anchor", "target_region", "visibility_bucket"}
)
_PRIVATE_TOKENS = ("evaluator", "private", "hidden_target", "target_truth", "ground_truth")
_PROTOCOL_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "version",
        "dataset_version",
        "scene_identity",
        "track",
        "agent_count",
        "axes",
        "cells",
        "randomization",
        "power_analysis",
        "exclusion_rules",
    }
)
_AXIS_KEYS = frozenset({"axis_id", "values", "split_role"})
_CELL_KEYS = frozenset({"cell_id", "split", "conditions", "minimum_attempts", "minimum_admitted"})
_RANDOMIZATION_KEYS = frozenset({"seed_derivation", "episode_seed_start", "paired_initial_conditions"})
_POWER_KEYS = frozenset(
    {
        "method",
        "primary_metric",
        "familywise_alpha",
        "power",
        "minimum_effect_size",
        "difference_standard_deviation",
        "comparison_count",
        "evaluation_split",
        "required_evaluation_episodes",
    }
)
_T1_PROTOCOL_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "version",
        "dataset_version",
        "scene_identity",
        "track",
        "purpose",
        "scoring_status",
        "agent_count",
        "statistical_unit",
        "scope",
        "axes",
        "cells",
        "randomization",
        "analysis_plan",
        "split_certificate",
        "overview_retention",
        "quality_acceptance",
        "exclusion_rules",
    }
)
_T1_REQUIRED_AXES = frozenset(
    {
        "layout",
        "route",
        "route_family",
        "target_count",
        "height",
        "region",
        "dynamics",
        "initial_condition",
        "start_anchor",
        "target_region",
        "visibility_bucket",
        "communication",
        "control_latency",
        "agent_dropout",
    }
)
_T1_GEOMETRY_HOLDOUT_AXES = frozenset(
    {"route_family", "start_anchor", "target_region"}
)
_T1_QUALITY_GATES = frozenset(
    {
        "independent_validation_passed",
        "sensor_timestamps_synchronized",
        "action_before_step_causality",
        "camera_pose_closure",
        "visual_intrusion_absent",
        "physical_safety_passed",
        "condition_realization_passed",
        "artifact_hash_binding_passed",
    }
)
_T2_CANARY_PROTOCOL_KEYS = frozenset(
    {
        "schema",
        "protocol_id",
        "version",
        "scene_identity",
        "track",
        "purpose",
        "scoring_status",
        "agent_count",
        "claim_boundary",
        "execution_contract",
        "axes",
        "cells",
        "randomization",
        "overview_retention",
        "quality_acceptance",
        "exclusion_rules",
    }
)
_T2_CANARY_V2_PROTOCOL_KEYS = _T2_CANARY_PROTOCOL_KEYS | frozenset({"motion_contract"})
_T2_CANARY_CELL_KEYS = frozenset(
    {"cell_id", "split", "conditions", "planned_independent_attempts"}
)
_T2_CANARY_CONDITIONS = {
    "layout": "citylite-v1",
    "route": "fixed-public-route-v1",
    "route_family": "citylite-route-family-a-v1",
    "target_count": "object-count-4-v1",
    "height": "citylite-command-altitude-v1",
    "region": "citylite-command-volume-v1",
    "dynamics": "cf2x-nominal-v1",
    "initial_condition": "public-route-anchor-v1",
    "start_anchor": "citylite-start-anchor-a-v1",
    "target_region": "citylite-target-region-b-v1",
    "visibility_bucket": "direct-visible-v1",
    "communication": "synchronous-public-broadcast-v1",
    "control_latency": "one-step-command-latency-v1",
    "agent_dropout": "no-agent-dropout-v1",
}
_T2_CANARY_AXIS_ROLES = {
    "layout": "scene",
    "initial_condition": "episode",
}
_T2_CANARY_CLAIM_BOUNDARY = {
    "development_only": True,
    "formal_episode": False,
    "benchmark_score": False,
    "policy_ranking": False,
}
_T2_CANARY_EXECUTION_CONTRACT = {
    "control_mode": "native_t2_canary",
    "task_variant_id": "isaac-eight-agent-native-t2-search-canary-v1",
    "requires_cf2x_runtime_calibration": True,
    "requires_full_sensor_smoke": True,
    "required_independent_passes": 2,
}
_T2_CANARY_OVERVIEW_RETENTION = {
    "selection_rule": "first_each_fixed_retained_frame_stride_and_final",
    "frame_index_stride": 10,
    "fixed_world_camera": True,
    "outcome_independent": True,
    "stored_modalities": ["rgb", "semantic", "world_pose"],
    "runtime_only_modalities": ["depth"],
}
_T2_CANARY_QUALITY_GATES = [
    "cf2x_runtime_calibration_passed",
    "full_sensor_smoke_bound",
    "action_causality_passed",
    "sensor_sync_passed",
    "camera_pose_closure_passed",
    "visual_intrusion_absent",
    "fixed_world_witness_passed",
    "private_event_evaluation_passed",
    "native_t2_replay_passed",
]
_T2_CANARY_EXCLUSION_RULES = [
    "retain_failed_attempt_in_ledger",
    "reject_missing_external_prerequisite",
    "reject_private_manifest_binding_mismatch",
    "reject_native_t2_validation_failure",
]
_T2_CANARY_V2_MOTION_CONTRACT = {
    "schema": "org.rivermark.native-t2-motion-contract.v1",
    "waypoint_segment_seconds": 6.0,
    "dt_s": 0.005,
    "warmup_steps": 120,
    "rollout_steps": 2400,
    "capture_stride": 10,
    "decision_stride_physics_steps": 40,
    "max_horizontal_speed_mps": 2.0,
    "max_vertical_speed_mps": 0.4,
    "max_yaw_rate_rad_s": 0.8,
    "position_feedback_gain": 0.8,
    "yaw_feedback_gain": 1.2,
    "route_speed_utilization_limit": 0.9,
    "camera_heading_model": "segment_horizontal_heading_yaw_limited_v1",
    "yaw_stability_error_rad": 0.2,
    "yaw_settle_margin_s": 0.4,
}
_T2_CANARY_V3_MOTION_CONTRACT = {
    **_T2_CANARY_V2_MOTION_CONTRACT,
    # v2's 6 s schedule needed 0.6405 m/s vertical velocity while its
    # conservative action budget was 0.36 m/s.  v3 keeps the verified action
    # envelope and doubles the public route time instead of asserting an
    # unmeasured faster CF2X vertical track.
    "waypoint_segment_seconds": 12.0,
    "rollout_steps": 4800,
}
NATIVE_T2_CANARY_PROTOCOL_SCHEMAS = frozenset(
    (
        NATIVE_T2_CANARY_PROTOCOL_SCHEMA,
        NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
        NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    )
)
