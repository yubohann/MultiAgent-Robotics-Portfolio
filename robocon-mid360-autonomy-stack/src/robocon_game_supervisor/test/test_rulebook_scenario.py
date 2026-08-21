from robocon_game_supervisor.rulebook_scenario import run_nominal_pass_and_three_point


def test_nominal_rulebook_scenario_is_explicitly_synthetic_and_scores_three() -> None:
    result = run_nominal_pass_and_three_point()
    assert result["evidence_level"] == "synthetic"
    assert all(decision["accepted"] for decision in result["decisions"])
    assert result["score"] == {"red": 0, "blue": 3}
    assert result["state"] == "REFEREE_STOP"
