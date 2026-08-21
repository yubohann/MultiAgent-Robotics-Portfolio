from robocon_game_supervisor.fault_matrix import run_matrix


def test_synthetic_fault_matrix_is_complete_and_passes():
    result = run_matrix()
    assert result["evidence_level"] == "synthetic"
    assert result["case_count"] >= 6
    assert result["passed"]
