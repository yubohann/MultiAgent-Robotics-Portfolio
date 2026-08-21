from robocon_game_supervisor.rulebook_2025 import (
    ABURobocon2025RuleEngine,
    MatchPhase,
    MatchState,
    Team,
)


def start_and_advance(engine: ABURobocon2025RuleEngine) -> None:
    assert engine.start_game(Team.BLUE, 0.0).accepted
    assert engine.record_dribble(
        1.0,
        drop_height_m=0.70,
        pickup_height_m=0.72,
        ball_contacted_offensive_side=True,
    ).accepted
    assert engine.enter_offensive_side(2.0).accepted


def test_official_timing_and_field_constants() -> None:
    engine = ABURobocon2025RuleEngine(MatchPhase.PRELIMINARY)
    assert engine.FIELD_LENGTH_M == 15.0
    assert engine.FIELD_WIDTH_M == 8.0
    assert engine.BASKET_HEIGHT_M == 2.43
    assert engine.game_duration_sec == 120.0
    assert ABURobocon2025RuleEngine(MatchPhase.KNOCKOUT).game_duration_sec == 160.0


def test_pass_requires_distance_receiver_and_receipt_evidence() -> None:
    engine = ABURobocon2025RuleEngine()
    assert engine.start_game(Team.RED, 0.0).accepted
    assert engine.record_pass(
        1.0,
        nearest_robot_distance_m=0.99,
        receiver_fully_on_offensive_side=True,
        receipt_confirmed=True,
    ).code == "pass_distance_too_short"
    assert engine.record_pass(
        2.0,
        nearest_robot_distance_m=1.0,
        receiver_fully_on_offensive_side=True,
        receipt_confirmed=False,
    ).code == "receipt_not_confirmed"
    assert engine.record_pass(
        3.0,
        nearest_robot_distance_m=1.0,
        receiver_fully_on_offensive_side=True,
        receipt_confirmed=True,
    ).accepted
    assert engine.enter_offensive_side(4.0).accepted


def test_advance_and_shot_clock_violations_are_rejected() -> None:
    engine = ABURobocon2025RuleEngine()
    assert engine.start_game(Team.RED, 0.0).accepted
    assert engine.record_dribble(
        1.0,
        drop_height_m=0.69,
        pickup_height_m=0.70,
        ball_contacted_offensive_side=True,
    ).code == "invalid_dribble_height"
    assert engine.enter_offensive_side(8.1).code == "advance_timeout"
    assert engine.tick(20.1).code == "shot_clock_expired"
    assert engine.state is MatchState.REFEREE_STOP


def test_scoring_distinguishes_three_point_and_dunk_evidence() -> None:
    engine = ABURobocon2025RuleEngine()
    start_and_advance(engine)
    three = engine.record_shot(3.0, shooter_on_offensive_side=True, zone="three_point")
    assert three.accepted and three.points == 3
    assert engine.score[Team.BLUE] == 3
    assert engine.resume_possession(Team.BLUE, 5.0).accepted
    start_and_advance_after_resume = engine.record_dribble(
        6.0,
        drop_height_m=0.70,
        pickup_height_m=0.70,
        ball_contacted_offensive_side=True,
    )
    assert start_and_advance_after_resume.accepted
    assert engine.enter_offensive_side(7.0).accepted
    missing = engine.record_shot(
        8.0,
        shooter_on_offensive_side=True,
        zone="paint",
        is_dunk=True,
        dunk_in_paint_zone=True,
        independent_jump=False,
        released_ball_follows_fall=True,
    )
    assert missing.code == "dunk_not_independently_jumped"
    dunk = engine.record_shot(
        9.0,
        shooter_on_offensive_side=True,
        zone="paint",
        is_dunk=True,
        dunk_in_paint_zone=True,
        independent_jump=True,
        released_ball_follows_fall=True,
    )
    assert dunk.accepted and dunk.points == 7
    assert engine.score[Team.BLUE] == 10
