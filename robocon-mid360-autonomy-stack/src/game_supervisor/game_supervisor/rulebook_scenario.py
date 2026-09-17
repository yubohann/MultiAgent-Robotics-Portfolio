"""Run a deterministic, evidence-labelled ABU ROBOCON 2025 rulebook scenario."""

from __future__ import annotations

import json

from .rulebook_2025 import ABURobocon2025RuleEngine, MatchPhase, Team


def run_nominal_pass_and_three_point() -> dict[str, object]:
    """Exercise the official rule gates with synthetic evidence, not hardware claims."""
    engine = ABURobocon2025RuleEngine(MatchPhase.PRELIMINARY)
    decisions = [
        engine.start_game(Team.BLUE, 0.0),
        engine.record_pass(
            1.0,
            nearest_robot_distance_m=1.20,
            receiver_fully_on_offensive_side=True,
            receipt_confirmed=True,
        ),
        engine.enter_offensive_side(2.0),
        engine.record_dribble(
            3.0,
            drop_height_m=0.70,
            pickup_height_m=0.72,
            ball_contacted_offensive_side=True,
        ),
        engine.record_shot(4.0, shooter_on_offensive_side=True, zone="three_point"),
    ]
    return {
        "profile_id": "abu_robocon_2025_robot_basketball_rulebook_20240814",
        "evidence_level": "synthetic",
        "decisions": [decision.__dict__ for decision in decisions],
        "state": engine.state.value,
        "score": {team.value: points for team, points in engine.score.items()},
    }


def main() -> None:
    print(json.dumps(run_nominal_pass_and_three_point(), separators=(",", ":")))


if __name__ == "__main__":
    main()
