"""Pure fixed-map tracking state transitions used by the ROS adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackingState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    TRACKING = "TRACKING"
    RELOCALIZING = "RELOCALIZING"
    LOST = "LOST"


@dataclass
class TrackingStateMachine:
    state: TrackingState = TrackingState.UNINITIALIZED
    anchor_exists: bool = False
    verified_anchor: bool = False
    relocalization_requested: bool = False

    def request_relocalization(self) -> None:
        self.relocalization_requested = True
        self.verified_anchor = False
        self.state = TrackingState.RELOCALIZING

    def set_provisional_anchor(self) -> None:
        """Store an initial pose without claiming scan-to-map lock."""
        self.anchor_exists = True
        self.verified_anchor = False
        self.relocalization_requested = True
        self.state = TrackingState.RELOCALIZING

    def accept_anchor(self) -> None:
        """Accept a correction that passed the fixed-map quality gate."""
        self.anchor_exists = True
        self.verified_anchor = True
        self.relocalization_requested = False
        self.state = TrackingState.TRACKING

    def update(self, pose_valid: bool, odom_fresh: bool) -> TrackingState:
        if not self.anchor_exists:
            self.state = TrackingState.UNINITIALIZED
        elif self.relocalization_requested or not self.verified_anchor:
            self.state = TrackingState.RELOCALIZING
        elif not pose_valid or not odom_fresh:
            self.state = TrackingState.LOST
        else:
            self.state = TrackingState.TRACKING
        return self.state

    @property
    def map_locked(self) -> bool:
        return self.state == TrackingState.TRACKING
