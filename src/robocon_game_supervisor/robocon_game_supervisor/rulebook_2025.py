"""Executable constraints from the ABU ROBOCON 2025 Robot Basketball rulebook.

The rule engine is deliberately independent of Gazebo and ROS messages.  A simulator,
vision module, or a real hardware adapter must supply the physical evidence named by each
method.  A boolean claim is never inferred from a command being sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Team(str, Enum):
    RED = "red"
    BLUE = "blue"


class MatchPhase(str, Enum):
    PRELIMINARY = "preliminary"
    KNOCKOUT = "knockout"


class MatchState(str, Enum):
    WAITING_START = "WAITING_START"
    ACTIVE = "ACTIVE"
    REFEREE_STOP = "REFEREE_STOP"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class RuleDecision:
    accepted: bool
    code: str
    reason: str
    points: int = 0


class ABURobocon2025RuleEngine:
    """Track the rulebook constraints that can be evaluated from explicit evidence.

    Source: ABU ROBOCON 2025 Rulebook (2024-08-14), sections 2, 6, 7, 10 and 12.
    Referee-only decisions, contact/foul judgement and zone geometry that lack an official
    numeric definition remain external evidence inputs rather than guessed calculations.
    """

    FIELD_LENGTH_M = 15.0
    FIELD_WIDTH_M = 8.0
    FENCE_HEIGHT_M = 0.10
    FENCE_WIDTH_M = 0.05
    BASKET_HEIGHT_M = 2.43
    BACKBOARD_LENGTH_M = 1.80
    BACKBOARD_WIDTH_M = 1.05
    BASKET_INNER_DIAMETER_MIN_M = 0.450
    BASKET_INNER_DIAMETER_MAX_M = 0.459
    POSSESSION_DURATION_SEC = 20.0
    ADVANCE_DURATION_SEC = 8.0
    MIN_PASS_DISTANCE_M = 1.0
    MIN_DRIBBLE_HAND_HEIGHT_M = 0.70

    def __init__(self, phase: MatchPhase = MatchPhase.PRELIMINARY) -> None:
        self.phase = MatchPhase(phase)
        self.state = MatchState.WAITING_START
        self.game_duration_sec = 120.0 if self.phase is MatchPhase.PRELIMINARY else 160.0
        self.start_time_sec: float | None = None
        self.possession_start_sec: float | None = None
        self.offensive_team: Team | None = None
        self.score = {Team.RED: 0, Team.BLUE: 0}
        self._advanced_to_offense = False
        self._offensive_side_dribble = False
        self._valid_pass_to_offense = False
        self._has_ball_control = False

    def start_game(self, first_possession: Team, now_sec: float) -> RuleDecision:
        if self.state is not MatchState.WAITING_START:
            return RuleDecision(False, "invalid_state", "game has already started")
        self.start_time_sec = self._validate_time(now_sec)
        return self._begin_possession(Team(first_possession), now_sec, "first possession")

    def resume_possession(self, team: Team, now_sec: float) -> RuleDecision:
        if self.state is not MatchState.REFEREE_STOP:
            return RuleDecision(False, "invalid_state", "possession can only resume after a referee stop")
        if self._game_expired(now_sec):
            return RuleDecision(False, "game_time_expired", "game time has expired")
        return self._begin_possession(Team(team), now_sec, "referee resumed possession")

    def tick(self, now_sec: float) -> RuleDecision:
        now_sec = self._validate_time(now_sec)
        if self.state is MatchState.FINISHED:
            return RuleDecision(False, "game_finished", "game is already finished")
        if self.start_time_sec is None:
            return RuleDecision(False, "not_started", "game has not started")
        if self._game_expired(now_sec):
            self.state = MatchState.FINISHED
            return RuleDecision(False, "game_time_expired", "official game duration has elapsed")
        if self.state is MatchState.ACTIVE and self.possession_start_sec is not None:
            if now_sec - self.possession_start_sec > self.POSSESSION_DURATION_SEC:
                self.state = MatchState.REFEREE_STOP
                return RuleDecision(False, "shot_clock_expired", "20-second possession limit exceeded")
        return RuleDecision(True, "clock_running", "game and possession clocks are within limits")

    def record_dribble(
        self,
        now_sec: float,
        *,
        drop_height_m: float,
        pickup_height_m: float,
        ball_contacted_offensive_side: bool,
    ) -> RuleDecision:
        failure = self._require_active_clock(now_sec)
        if failure is not None:
            return failure
        if min(drop_height_m, pickup_height_m) < self.MIN_DRIBBLE_HAND_HEIGHT_M:
            return RuleDecision(False, "invalid_dribble_height", "both dribble hand heights must be at least 0.700 m")
        if not ball_contacted_offensive_side:
            return RuleDecision(False, "dribble_not_on_offensive_side", "advance evidence requires a dribble on the offensive side")
        self._offensive_side_dribble = True
        self._has_ball_control = True
        return RuleDecision(True, "dribble_confirmed", "rulebook dribble evidence accepted")

    def record_pass(
        self,
        now_sec: float,
        *,
        nearest_robot_distance_m: float,
        receiver_fully_on_offensive_side: bool,
        receipt_confirmed: bool,
    ) -> RuleDecision:
        failure = self._require_active_clock(now_sec)
        if failure is not None:
            return failure
        if nearest_robot_distance_m < self.MIN_PASS_DISTANCE_M:
            return RuleDecision(False, "pass_distance_too_short", "valid passes require at least 1.000 m robot separation")
        if not receiver_fully_on_offensive_side:
            return RuleDecision(False, "receiver_not_on_offensive_side", "receiver must be fully on the offensive side")
        if not receipt_confirmed:
            return RuleDecision(False, "receipt_not_confirmed", "a sent pass is not proof of team ball control")
        self._valid_pass_to_offense = True
        self._has_ball_control = True
        return RuleDecision(True, "pass_confirmed", "valid pass and receipt evidence accepted")

    def enter_offensive_side(self, now_sec: float) -> RuleDecision:
        failure = self._require_active_clock(now_sec)
        if failure is not None:
            return failure
        assert self.possession_start_sec is not None
        if now_sec - self.possession_start_sec > self.ADVANCE_DURATION_SEC:
            return RuleDecision(False, "advance_timeout", "offensive side was not reached within 8 seconds")
        if not (self._offensive_side_dribble or self._valid_pass_to_offense):
            return RuleDecision(False, "invalid_advance", "advance requires confirmed offensive-side dribble or pass")
        self._advanced_to_offense = True
        return RuleDecision(True, "advance_confirmed", "offensive-side entry is rulebook compliant")

    def record_shot(
        self,
        now_sec: float,
        *,
        shooter_on_offensive_side: bool,
        zone: str,
        immediately_after_gain_without_moving: bool = False,
        is_dunk: bool = False,
        dunk_in_paint_zone: bool = False,
        independent_jump: bool = False,
        released_ball_follows_fall: bool = False,
        ball_entered_basket: bool = True,
    ) -> RuleDecision:
        failure = self._require_active_clock(now_sec)
        if failure is not None:
            return failure
        if not shooter_on_offensive_side or not self._advanced_to_offense:
            return RuleDecision(False, "shot_from_defensive_side", "shots must be taken from the offensive side")
        if not self._has_ball_control:
            return RuleDecision(False, "ball_control_missing", "shot requires confirmed team ball control")
        if not self._offensive_side_dribble and not immediately_after_gain_without_moving:
            return RuleDecision(False, "dribble_required", "a shot requires a prior valid dribble unless it is immediate on gain")
        if not ball_entered_basket:
            return RuleDecision(True, "missed_shot", "shot was legal but did not score")
        if is_dunk:
            dunk_checks = (
                (dunk_in_paint_zone, "dunk_not_in_paint_zone"),
                (independent_jump, "dunk_not_independently_jumped"),
                (released_ball_follows_fall, "dunk_release_not_verified"),
            )
            for passed, code in dunk_checks:
                if not passed:
                    return RuleDecision(False, code, "dunk rule evidence is incomplete")
            return self._score(7, "dunk_scored")
        if zone == "three_point":
            return self._score(3, "three_point_scored")
        if zone in {"two_point", "paint"}:
            return self._score(2, "two_point_scored")
        return RuleDecision(False, "unknown_shot_zone", "zone must be two_point, three_point, or paint")

    def finish_game(self, now_sec: float) -> RuleDecision:
        self._validate_time(now_sec)
        if self.state is MatchState.WAITING_START:
            return RuleDecision(False, "not_started", "game has not started")
        self.state = MatchState.FINISHED
        return RuleDecision(True, "game_finished", "game was ended by referee or operator")

    def _begin_possession(self, team: Team, now_sec: float, reason: str) -> RuleDecision:
        self.offensive_team = team
        self.possession_start_sec = self._validate_time(now_sec)
        self.state = MatchState.ACTIVE
        self._advanced_to_offense = False
        self._offensive_side_dribble = False
        self._valid_pass_to_offense = False
        self._has_ball_control = True
        return RuleDecision(True, "possession_started", reason)

    def _score(self, points: int, code: str) -> RuleDecision:
        assert self.offensive_team is not None
        self.score[self.offensive_team] += points
        self._has_ball_control = False
        self.state = MatchState.REFEREE_STOP
        return RuleDecision(True, code, f"{points} points awarded", points=points)

    def _require_active_clock(self, now_sec: float) -> RuleDecision | None:
        clock = self.tick(now_sec)
        if not clock.accepted or self.state is not MatchState.ACTIVE:
            return RuleDecision(False, clock.code, clock.reason)
        return None

    def _game_expired(self, now_sec: float) -> bool:
        assert self.start_time_sec is not None
        return now_sec - self.start_time_sec >= self.game_duration_sec

    @staticmethod
    def _validate_time(now_sec: float) -> float:
        value = float(now_sec)
        if value < 0.0:
            raise ValueError("time must be non-negative")
        return value
