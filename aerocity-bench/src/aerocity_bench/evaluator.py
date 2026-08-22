"""Evaluator-private geometric confirmation with observation-bound receipts."""

from __future__ import annotations

import hashlib
import hmac
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, content_hash
from .contracts import (
    ActionPacket,
    ConfirmationReceipt,
    ObservationPacket,
    ObservationReceipt,
)
from .geometry import (
    Vec3,
    colliders_from_city,
    distance,
    in_field_of_view,
    line_of_sight,
    sensor_pose,
    surface_facing,
)
from .ordinary_config import OrdinaryReleaseConfig


@dataclass
class _DwellState:
    started_at_s: float
    last_seen_at_s: float
    initial_position: Vec3
    observation_ids: list[str]


class PrivateEvaluator:
    """Own target truth and emit only anonymous, signed confirmation receipts."""

    def __init__(
        self,
        config: OrdinaryReleaseConfig,
        city: dict[str, Any],
        private_episode: dict[str, Any],
        *,
        receipt_secret: bytes,
    ) -> None:
        if len(receipt_secret) < 16:
            raise ValueError("receipt_secret must contain at least 16 bytes")
        if private_episode.get("layout_hash") != city.get("layout_hash"):
            raise ValueError("private episode and CitySpec layout hashes differ")
        self.config = config
        self.city = city
        self.episode = private_episode
        self._secret = receipt_secret
        self._colliders = colliders_from_city(city)
        self._targets = {str(item["target_id"]): item for item in private_episode["targets"]}
        self._confirmed: dict[str, ConfirmationReceipt] = {}
        self._processed_observation_ids: set[str] = set()
        self._dwell: dict[tuple[str, str], _DwellState] = {}
        self._last_observation_time: dict[str, float] = {}
        self._observe_session_active: set[str] = set()
        self._last_observe_end_s: dict[str, float] = {}
        self._redundant_target_agent_pairs: set[tuple[str, str]] = set()
        self._simultaneous_confirmation_ids: set[str] = set()
        self._private_failure_counts: Counter[str] = Counter()
        self._observation_count = 0
        self._validate_private_contract()

    def _validate_private_contract(self) -> None:
        if int(self.episode["target_count"]) != len(self._targets):
            raise ValueError("private target count differs from target records")
        if any(target.get("valid_before_run") is not True for target in self._targets.values()):
            raise ValueError("all targets must be validated before execution")
        manifest = dict(self.episode["target_validity"])
        expected_hash = str(manifest.pop("validity_hash"))
        if content_hash(manifest) != expected_hash:
            raise ValueError("target-validity manifest hash mismatch")
        if manifest.get("frozen_before_execution") is not True:
            raise ValueError("target validity was not frozen before execution")
        if set(manifest["target_ids"]) != set(self._targets):
            raise ValueError("target-validity IDs differ from episode targets")
        expected_contract_hash = content_hash(self.config.raw["execution_contract"])
        if self.episode.get("execution_contract_hash") != expected_contract_hash:
            raise ValueError("episode execution contract hash differs from active config")

    @property
    def target_count_private(self) -> int:
        return len(self._targets)

    @property
    def confirmed_count_private(self) -> int:
        return len(self._confirmed)

    def _token(self, purpose: str, payload: dict[str, Any]) -> str:
        body = canonical_bytes([purpose, self.episode["episode_id"], payload])
        return hmac.new(self._secret, body, hashlib.sha256).hexdigest()

    def _anonymous_handle(self, target_id: str) -> str:
        digest = self._token("anonymous-target", {"target_id": target_id})
        return f"found-{digest[:20]}"

    def _visibility(
        self, observation: ObservationPacket, target: dict[str, Any]
    ) -> tuple[bool, str]:
        contract = self.config.raw["execution_contract"]
        observe = contract["observe"]
        rig = contract["sensor_rig"]
        camera_pose = sensor_pose(
            observation.pose,
            rig["translation_body_m"],
            sensor_pitch_deg=(
                observation.sensor_pitch_deg if rig["gimbal_mode"] == "bounded" else None
            ),
        )
        target_position = tuple(float(value) for value in target["position"])
        target_normal = tuple(float(value) for value in target["normal"])
        target_distance = distance(camera_pose.position, target_position)
        if target_distance > float(observe["max_range_m"]):
            return False, "range"
        in_view, _, _ = in_field_of_view(
            camera_pose,
            target_position,
            float(observe["horizontal_fov_deg"]),
            float(observe["vertical_fov_deg"]),
        )
        if not in_view:
            return False, "fov"
        facing, _ = surface_facing(
            camera_pose.position,
            target_position,
            target_normal,  # type: ignore[arg-type]
            float(observe["surface_facing_min_cosine"]),
        )
        if not facing:
            return False, "surface_facing"
        visible, _ = line_of_sight(
            camera_pose.position,
            target_position,
            self._colliders,
            ignored_ids=frozenset({str(target["owner_collider_id"])}),
        )
        if not visible:
            return False, "occlusion"
        return True, "visible"

    def _reset_drone_dwell(self, drone_id: str, except_targets: set[str] | None = None) -> None:
        keep = except_targets or set()
        for key in list(self._dwell):
            if key[0] == drone_id and key[1] not in keep:
                del self._dwell[key]

    def end_observe(self, drone_id: str, timestamp_s: float | None = None) -> None:
        if drone_id in self._observe_session_active:
            end_time = (
                float(timestamp_s)
                if timestamp_s is not None
                else self._last_observation_time.get(drone_id)
            )
            if end_time is not None:
                self._last_observe_end_s[drone_id] = end_time
            self._observe_session_active.discard(drone_id)
        self._reset_drone_dwell(drone_id)

    def process(
        self,
        observation: ObservationPacket,
        action: ActionPacket,
    ) -> tuple[ObservationReceipt, tuple[ConfirmationReceipt, ...]]:
        if observation.episode_id != self.episode["episode_id"]:
            raise ValueError("observation belongs to another episode")
        if action.episode_id != observation.episode_id or action.drone_id != observation.drone_id:
            raise ValueError("action and observation identity differ")
        if action.kind != "OBSERVE":
            self.end_observe(observation.drone_id, observation.timestamp_s)
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "not_an_observe_action",
                ),
                (),
            )
        if action.source_observation_id != observation.observation_id:
            self._private_failure_counts["source_mismatch"] += 1
            self.end_observe(observation.drone_id, observation.timestamp_s)
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "source_observation_mismatch",
                ),
                (),
            )
        if observation.observation_id in self._processed_observation_ids:
            self._private_failure_counts["replayed_source"] += 1
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "source_observation_replayed",
                ),
                (),
            )
        observe = self.config.raw["execution_contract"]["observe"]
        if action.issued_at_s < observation.timestamp_s:
            raise ValueError("action cannot predate its source observation")
        if action.issued_at_s - observation.timestamp_s > float(observe["source_freshness_s"]):
            self._private_failure_counts["stale_source"] += 1
            self.end_observe(observation.drone_id, observation.timestamp_s)
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "source_observation_stale",
                ),
                (),
            )
        self._processed_observation_ids.add(observation.observation_id)
        self._observation_count += 1
        linear_speed = math.sqrt(
            sum(value * value for value in observation.linear_velocity_world_mps)
        )
        if linear_speed > float(observe["max_linear_speed_mps"]):
            self._private_failure_counts["moving"] += 1
            self.end_observe(observation.drone_id, observation.timestamp_s)
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "observe_requires_stability",
                ),
                (),
            )
        if observation.angular_speed_deg_s > float(observe["max_angular_speed_deg_s"]):
            self._private_failure_counts["rotating"] += 1
            self.end_observe(observation.drone_id, observation.timestamp_s)
            return (
                ObservationReceipt.create(
                    observation.observation_id,
                    observation.drone_id,
                    observation.timestamp_s,
                    False,
                    "observe_requires_stability",
                ),
                (),
            )
        previous_time = self._last_observation_time.get(observation.drone_id)
        control_period = float(self.config.raw["execution_contract"]["control_period_s"])
        if (
            previous_time is not None
            and observation.timestamp_s - previous_time > control_period * 1.6
        ):
            self.end_observe(observation.drone_id, previous_time)
        if observation.drone_id not in self._observe_session_active:
            last_end = self._last_observe_end_s.get(observation.drone_id)
            if last_end is not None and observation.timestamp_s - last_end + 1.0e-9 < float(
                observe["cooldown_s"]
            ):
                self._private_failure_counts["observe_cooldown"] += 1
                return (
                    ObservationReceipt.create(
                        observation.observation_id,
                        observation.drone_id,
                        observation.timestamp_s,
                        False,
                        "observe_cooldown_active",
                    ),
                    (),
                )
            self._observe_session_active.add(observation.drone_id)
        self._last_observation_time[observation.drone_id] = observation.timestamp_s

        visible_target_ids: set[str] = set()
        for target_id, target in self._targets.items():
            if target_id in self._confirmed:
                visible, _ = self._visibility(observation, target)
                if visible:
                    self._redundant_target_agent_pairs.add((observation.drone_id, target_id))
                    stored = self._confirmed[target_id]
                    state = self._dwell.get((observation.drone_id, target_id))
                    control_period = float(
                        self.config.raw["execution_contract"]["control_period_s"]
                    )
                    if (
                        state is not None
                        and math.isclose(
                            stored.confirmed_at_s,
                            observation.timestamp_s,
                            abs_tol=1.0e-9,
                        )
                        and observation.timestamp_s - state.last_seen_at_s
                        <= control_period * 1.6
                        and observation.timestamp_s - state.started_at_s + 1.0e-9
                        >= float(observe["continuous_dwell_s"])
                        and distance(state.initial_position, observation.pose.position)
                        <= float(observe["max_pose_drift_m"])
                    ):
                        self._simultaneous_confirmation_ids.add(stored.confirmation_id)
                continue
            visible, reason = self._visibility(observation, target)
            self._private_failure_counts[reason] += 1
            if visible:
                visible_target_ids.add(target_id)
        self._reset_drone_dwell(observation.drone_id, visible_target_ids)
        confirmations: list[ConfirmationReceipt] = []
        for target_id in sorted(visible_target_ids):
            key = (observation.drone_id, target_id)
            state = self._dwell.get(key)
            if state is None:
                state = _DwellState(
                    started_at_s=observation.timestamp_s,
                    last_seen_at_s=observation.timestamp_s,
                    initial_position=observation.pose.position,
                    observation_ids=[observation.observation_id],
                )
                self._dwell[key] = state
            else:
                pose_drift = distance(state.initial_position, observation.pose.position)
                if pose_drift > float(observe["max_pose_drift_m"]):
                    state = _DwellState(
                        started_at_s=observation.timestamp_s,
                        last_seen_at_s=observation.timestamp_s,
                        initial_position=observation.pose.position,
                        observation_ids=[observation.observation_id],
                    )
                    self._dwell[key] = state
                    self._private_failure_counts["pose_drift"] += 1
                else:
                    state.last_seen_at_s = observation.timestamp_s
                    state.observation_ids.append(observation.observation_id)
            dwell_time = state.last_seen_at_s - state.started_at_s
            if dwell_time + 1.0e-9 < float(observe["continuous_dwell_s"]):
                continue
            payload = {
                "target_id": target_id,
                "drone_id": observation.drone_id,
                "confirmed_at_s": observation.timestamp_s,
                "source_observation_id": observation.observation_id,
                "dwell_observation_ids": state.observation_ids,
                "validity_hash": self.episode["target_validity"]["validity_hash"],
            }
            confirmation_id = f"confirmation-{content_hash(payload)[:18]}"
            receipt = ConfirmationReceipt(
                confirmation_id=confirmation_id,
                anonymous_target_handle=self._anonymous_handle(target_id),
                drone_id=observation.drone_id,
                confirmed_at_s=observation.timestamp_s,
                source_observation_id=observation.observation_id,
                receipt_token=self._token("confirmation", payload),
            )
            # Global target de-duplication makes simultaneous multi-UAV finds count once.
            if target_id not in self._confirmed:
                self._confirmed[target_id] = receipt
                confirmations.append(receipt)
            self._dwell.pop(key, None)
        return (
            ObservationReceipt.create(
                observation.observation_id,
                observation.drone_id,
                observation.timestamp_s,
                True,
                "accepted",
            ),
            tuple(confirmations),
        )

    def verify_confirmation(self, receipt: ConfirmationReceipt) -> bool:
        matches = [
            target_id
            for target_id, stored in self._confirmed.items()
            if stored.confirmation_id == receipt.confirmation_id
        ]
        return len(matches) == 1 and self._confirmed[matches[0]] == receipt

    def private_audit_snapshot(self) -> dict[str, Any]:
        """Return evaluator-owner diagnostics; never pass this object to a method."""

        confirmations = sorted(
            self._confirmed.items(),
            key=lambda item: (item[1].confirmed_at_s, item[1].confirmation_id),
        )
        return {
            "schema": "org.aerocity.bench.evaluator-private-audit.v1",
            "episode_id": self.episode["episode_id"],
            "validity_hash": self.episode["target_validity"]["validity_hash"],
            "target_count": len(self._targets),
            "confirmed_count": len(confirmations),
            "confirmation_times_s": [item.confirmed_at_s for _, item in confirmations],
            "confirmation_ids": [item.confirmation_id for _, item in confirmations],
            "confirmation_records_private": [
                {
                    "target_id": target_id,
                    "confirmation_id": item.confirmation_id,
                    "drone_id": item.drone_id,
                    "confirmed_at_s": item.confirmed_at_s,
                }
                for target_id, item in confirmations
            ],
            "observation_count": self._observation_count,
            "redundant_target_agent_pair_count": len(self._redundant_target_agent_pairs),
            "simultaneous_confirmation_ids": sorted(self._simultaneous_confirmation_ids),
            "simultaneous_confirmation_tie_count": len(
                self._simultaneous_confirmation_ids
            ),
            "visibility_diagnostics": dict(sorted(self._private_failure_counts.items())),
        }
