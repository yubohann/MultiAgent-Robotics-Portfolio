from __future__ import annotations


def test_cli_defaults_are_parseable_without_runtime_dependencies() -> None:
    from fraud_ml_engineering.main import parse_args

    args = parse_args([])

    assert args.dataset == "elliptic"
    assert args.federated_rounds > 0
    assert args.transformer_activation_checkpointing is True
    assert args.planner_mode == "deterministic"


def test_cli_preserves_legacy_aliases_and_dataset_specific_flags() -> None:
    from fraud_ml_engineering.main import _build_training_kwargs, _normalize_active_mainline_args, parse_args

    args = parse_args(
        [
            "--dataset",
            "ieee",
            "--rounds",
            "3",
            "--local_epochs",
            "2",
            "--ieee_build_light_cache_only",
        ]
    )

    assert args.dataset == "ieee"
    assert args.federated_rounds == 3
    assert args.base_local_epochs == 2
    assert args.ieee_build_light_cache_only is True

    _normalize_active_mainline_args(args, ["ieee"])

    assert args.ieee_build_cache_only is True
    assert args.ieee_skip_training is True
    assert args.controller_timesteps == 0
    assert args.planner_mode == "deterministic"
    assert args.disable_federated is True

    training_kwargs = _build_training_kwargs(args)

    assert training_kwargs["federated_rounds"] == 3
    assert training_kwargs["local_epochs"] == 2
    assert training_kwargs["planner_mode"] == "deterministic"
    assert training_kwargs["ieee_build_cache_only"] is True
    assert training_kwargs["ieee_skip_training"] is True
