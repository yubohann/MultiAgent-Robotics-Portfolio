"""Frozen public-observation contract for target-free HM3D exploration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerocity_method.contracts.io import canonical_sha256, read_json_object
from aerocity_method.runtime.range_sensing import (
    DENSE_26_RAY_PATTERN,
    LEGACY_SIX_AXIS_PATTERN,
    validate_public_range_directions,
)

SCHEMA_VERSION = "hm3d-exploration-observation-contract-v3"
CONTRACT_ID = "hm3d-sparse-range-public-exploration-v3"
TASK_INTERFACE = "hm3d-derived-multi-uav-exploration-3d"
PRIMARY_METRIC = "Explored-Free-Flight-Volume-AUC_time"
DEFAULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "external"
    / "hm3d_exploration_observation_contract.json"
)


@dataclass(frozen=True, slots=True)
class HM3DExplorationObservationContract:
    """The public sensor and private-evaluator boundary for every ranked method."""

    payload: dict[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "contract_id",
            "task_interface",
            "status",
            "sensor_profile",
            "method_visible",
            "method_forbidden",
            "public_belief",
            "evaluation",
            "fairness",
        }
        if set(self.payload) != expected:
            raise ValueError("exploration observation contract fields mismatch")
        if self.payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("exploration observation contract schema mismatch")
        if self.payload["contract_id"] != CONTRACT_ID:
            raise ValueError("unexpected exploration observation contract ID")
        if self.payload["task_interface"] != TASK_INTERFACE:
            raise ValueError("exploration observation task interface mismatch")
        if self.payload["status"] != "ACTIVE_PROTOCOL_NOT_FORMAL_RESULT":
            raise ValueError("observation contract cannot claim a formal result")
        sensor = self._object("sensor_profile")
        if sensor.get("profile_id") != "sparse-range-3d-vfov90":
            raise ValueError("P04 requires the sparse-range public sensor profile")
        ray_pattern = sensor.get("ray_pattern")
        if ray_pattern == DENSE_26_RAY_PATTERN:
            ray_directions = sensor.get("ray_directions")
            if not isinstance(ray_directions, list):
                raise ValueError("P04 dense range pattern must declare ray_directions")
            validate_public_range_directions(ray_directions)
        elif ray_pattern == LEGACY_SIX_AXIS_PATTERN:
            pass
        else:
            raise ValueError("P04 range-ray geometry changed")
        if sensor.get("source_observation_id_required") is not True:
            raise ValueError("P04 requires source_observation_id binding")
        if sensor.get("enabled_windows") != ["reset_bootstrap", "transit", "observe", "dwell"]:
            raise ValueError("P04 sensor windows drifted from the H15 entitlement")
        if (
            not isinstance(sensor.get("maximum_range_m"), (int, float))
            or float(sensor["maximum_range_m"]) <= 0.0
        ):
            raise ValueError("P04 range limit must be positive")
        if (
            not isinstance(sensor.get("update_hz"), (int, float))
            or float(sensor["update_hz"]) <= 0.0
        ):
            raise ValueError("P04 sensor rate must be positive")
        visible = self._identifiers("method_visible")
        forbidden = self._identifiers("method_forbidden")
        if "public_sparse_range_ray_outcomes" not in visible:
            raise ValueError("P04 must expose only real public sensor outcomes")
        required_forbidden = {
            "complete_hm3d_mesh",
            "private_esdf_or_distance_field",
            "evaluator_free_flight_mask",
            "evaluator_truth_map",
        }
        if not required_forbidden.issubset(forbidden):
            raise ValueError("P04 evaluator-private geometry boundary is incomplete")
        belief = self._object("public_belief")
        if belief.get("representation") != "sparse_occupancy_voxels":
            raise ValueError("P04 belief representation mismatch")
        if belief.get("merge_rule") != "own_outcomes_and_delivered_peer_deltas_only":
            raise ValueError("P04 map merge would bypass communication outcomes")
        if belief.get("ray_replay") != "idempotent_by_observation_id":
            raise ValueError("P04 outcome replay semantics mismatch")
        evaluation = self._object("evaluation")
        if evaluation.get("primary_metric") != PRIMARY_METRIC:
            raise ValueError("P04 primary exploration metric mismatch")
        required_metrics = {
            "explored_free_flight_volume_auc_time",
            "final_coverage_at_budget",
            "final_explored_free_volume_m3",
            "evaluator_reachable_free_flight_volume_m3",
            "mean_explored_free_volume_rate_m3_per_s",
        }
        if set(evaluation.get("required_report_fields", ())) != required_metrics:
            raise ValueError("P04 metric report is missing an absolute-volume diagnostic")
        if (
            evaluation.get("completion_diagnostic")
            != "Threshold_free_coverage_curve_diagnostics_do_not_replace_primary_metric"
        ):
            raise ValueError("P04 coverage diagnostics cannot replace the primary metric")
        fairness = self._object("fairness")
        self._identifiers(
            "fairness.all_ranked_methods_share", fairness.get("all_ranked_methods_share")
        )
        if (
            fairness.get("disallowed_shortcut")
            != "no_method_may_receive_extra_sensor_frames_or_evaluator_geometry"
        ):
            raise ValueError("P04 sensor fairness rule mismatch")

    def _object(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        return value

    def _identifiers(self, name: str, value: Any | None = None) -> tuple[str, ...]:
        raw = self.payload.get(name) if value is None else value
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(row, str) and row for row in raw)
        ):
            raise ValueError(f"{name} must be a non-empty string list")
        if len(raw) != len(set(raw)):
            raise ValueError(f"{name} contains duplicate values")
        return tuple(raw)

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload)


def load_exploration_observation_contract(
    path: str | Path = DEFAULT_PATH,
) -> HM3DExplorationObservationContract:
    return HM3DExplorationObservationContract(read_json_object(path))


__all__ = [
    "CONTRACT_ID",
    "DEFAULT_PATH",
    "HM3DExplorationObservationContract",
    "PRIMARY_METRIC",
    "SCHEMA_VERSION",
    "TASK_INTERFACE",
    "load_exploration_observation_contract",
]
