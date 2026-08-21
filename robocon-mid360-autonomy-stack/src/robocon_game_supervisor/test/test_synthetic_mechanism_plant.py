from robocon_game_supervisor.simulator import SyntheticMechanismPlant


def test_synthetic_plant_requires_state_evidence_for_pass_and_shot_chain():
    plant = SyntheticMechanismPlant()

    succeeded, reason, evidence = plant.apply("PreparePass")
    assert not succeeded
    assert reason == "receiver-ready sensor is false"
    assert not evidence["pass_armed"]

    for action in ("PrepareReceive", "PreparePass", "ExecutePass", "CollectBall", "PrepareShot", "FireShot"):
        succeeded, _, evidence = plant.apply(action)
        assert succeeded, action

    assert evidence["receipt_confirmed"]
    assert evidence["shot_released"]
    assert not evidence["ball_present"]
    assert evidence["evidence_level"] == "synthetic"


def test_synthetic_plant_latches_estop_and_rejects_later_actions():
    plant = SyntheticMechanismPlant()
    succeeded, _, evidence = plant.apply("EmergencyStop")
    assert succeeded
    assert evidence["estop_latched"]

    succeeded, reason, _ = plant.apply("PrepareReceive")
    assert not succeeded
    assert reason == "synthetic estop is latched"
