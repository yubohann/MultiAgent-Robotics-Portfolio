"""Config entry points and preset registry for the multi-agent experiment."""

from __future__ import annotations

from pathlib import Path

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG as VARIABLE_MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    build_exp3_curriculum_experiment_config,
    build_exp3_paper_experiment_config,
    build_dynamic_gate_density_8d_config,
    build_fixed_team_experiment_config,
    build_multi_experiment_config,
    override_multi_scene_config,
)
from multi_gate.configs.experiment_config_fixed_02 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_02
from multi_gate.configs.experiment_config_fixed_03 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_03
from multi_gate.configs.experiment_config_fixed_05 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_05
from multi_gate.configs.experiment_config_fixed_07 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_07
from multi_gate.configs.experiment_config_fixed_08 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_08
from multi_gate.configs.experiment_config_fixed_09 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_09
from multi_gate.configs.experiment_config_fixed_12 import MULTI_EXPERIMENT_CONFIG as FIXED_MULTI_EXPERIMENT_CONFIG_12

EXP3_PAPER_CONFIG_BASELINE = build_exp3_paper_experiment_config("e3_baseline")
EXP3_PAPER_CONFIG_MAIN = build_exp3_paper_experiment_config("e3_main")
EXP3_PAPER_CONFIG_GUIDANCE = build_exp3_paper_experiment_config("e3_guidance")
DYNAMIC_GATE_DENSITY_8D_CONFIG = build_dynamic_gate_density_8d_config()

MULTI_EXPERIMENT_CONFIG_CANONICAL_NAMES: tuple[str, ...] = (
    "variable",
    "fixed_02",
    "fixed_03",
    "fixed_05",
    "fixed_07",
    "fixed_08",
    "fixed_09",
    "fixed_12",
    "e3_baseline",
    "e3_main",
    "e3_guidance",
    "dynamic_gate_density_8d_v1",
)

MULTI_EXPERIMENT_CONFIG_REGISTRY: dict[str, MultiExperimentConfig] = {
    "variable": VARIABLE_MULTI_EXPERIMENT_CONFIG,
    "fixed_02": FIXED_MULTI_EXPERIMENT_CONFIG_02,
    "fixed_03": FIXED_MULTI_EXPERIMENT_CONFIG_03,
    "fixed_05": FIXED_MULTI_EXPERIMENT_CONFIG_05,
    "fixed_07": FIXED_MULTI_EXPERIMENT_CONFIG_07,
    "fixed_08": FIXED_MULTI_EXPERIMENT_CONFIG_08,
    "fixed_09": FIXED_MULTI_EXPERIMENT_CONFIG_09,
    "fixed_12": FIXED_MULTI_EXPERIMENT_CONFIG_12,
    "e3_baseline": EXP3_PAPER_CONFIG_BASELINE,
    "e3_main": EXP3_PAPER_CONFIG_MAIN,
    "e3_guidance": EXP3_PAPER_CONFIG_GUIDANCE,
    "dynamic_gate_density_8d_v1": DYNAMIC_GATE_DENSITY_8D_CONFIG,
    "2": FIXED_MULTI_EXPERIMENT_CONFIG_02,
    "3": FIXED_MULTI_EXPERIMENT_CONFIG_03,
    "5": FIXED_MULTI_EXPERIMENT_CONFIG_05,
    "7": FIXED_MULTI_EXPERIMENT_CONFIG_07,
    "8": FIXED_MULTI_EXPERIMENT_CONFIG_08,
    "9": FIXED_MULTI_EXPERIMENT_CONFIG_09,
    "12": FIXED_MULTI_EXPERIMENT_CONFIG_12,
}

MULTI_EXPERIMENT_CONFIG_ALIASES: dict[str, str] = {
    "variable": "variable",
    "fixed_02": "fixed_02",
    "fixed_03": "fixed_03",
    "fixed_05": "fixed_05",
    "fixed_07": "fixed_07",
    "fixed_08": "fixed_08",
    "fixed_09": "fixed_09",
    "fixed_12": "fixed_12",
    "e3_baseline": "e3_baseline",
    "e3_main": "e3_main",
    "e3_guidance": "e3_guidance",
    "dynamic_gate_density_8d_v1": "dynamic_gate_density_8d_v1",
    "gate_density_8d": "dynamic_gate_density_8d_v1",
    "dynamic_gate_density": "dynamic_gate_density_8d_v1",
    "exp3_baseline": "e3_baseline",
    "exp3_main": "e3_main",
    "exp3_guidance": "e3_guidance",
    "2": "fixed_02",
    "3": "fixed_03",
    "5": "fixed_05",
    "7": "fixed_07",
    "8": "fixed_08",
    "9": "fixed_09",
    "12": "fixed_12",
}

MULTI_EXPERIMENT_ID_TO_CONFIG_NAME: dict[str, str] = {
    VARIABLE_MULTI_EXPERIMENT_CONFIG.experiment_id: "variable",
    FIXED_MULTI_EXPERIMENT_CONFIG_02.experiment_id: "fixed_02",
    FIXED_MULTI_EXPERIMENT_CONFIG_03.experiment_id: "fixed_03",
    FIXED_MULTI_EXPERIMENT_CONFIG_05.experiment_id: "fixed_05",
    FIXED_MULTI_EXPERIMENT_CONFIG_07.experiment_id: "fixed_07",
    FIXED_MULTI_EXPERIMENT_CONFIG_08.experiment_id: "fixed_08",
    FIXED_MULTI_EXPERIMENT_CONFIG_09.experiment_id: "fixed_09",
    FIXED_MULTI_EXPERIMENT_CONFIG_12.experiment_id: "fixed_12",
    EXP3_PAPER_CONFIG_BASELINE.experiment_id: "e3_baseline",
    EXP3_PAPER_CONFIG_MAIN.experiment_id: "e3_main",
    EXP3_PAPER_CONFIG_GUIDANCE.experiment_id: "e3_guidance",
    DYNAMIC_GATE_DENSITY_8D_CONFIG.experiment_id: "dynamic_gate_density_8d_v1",
}


def get_multi_experiment_config(config_name: str | None = None) -> MultiExperimentConfig:
    """Resolve one named config preset for the multi-agent experiment."""

    normalized_name = normalize_multi_experiment_config_name(config_name)
    if normalized_name not in MULTI_EXPERIMENT_CONFIG_REGISTRY:
        valid = ", ".join(MULTI_EXPERIMENT_CONFIG_CANONICAL_NAMES)
        raise KeyError(f"Unknown multi-agent config preset '{config_name}'. Valid names: {valid}")
    return MULTI_EXPERIMENT_CONFIG_REGISTRY[normalized_name]


def list_multi_experiment_config_names() -> list[str]:
    """Return the canonical preset names exposed through the CLI."""

    return list(MULTI_EXPERIMENT_CONFIG_CANONICAL_NAMES)


def normalize_multi_experiment_config_name(config_name: str | None = None) -> str:
    """Resolve aliases such as '12' to the canonical registry key."""

    normalized_name = (config_name or "variable").strip().lower().replace("-", "_")
    return MULTI_EXPERIMENT_CONFIG_ALIASES.get(normalized_name, normalized_name)


def infer_multi_config_name_from_experiment_id(experiment_id: str | None) -> str | None:
    """Infer the canonical config name from one saved experiment identifier."""

    resolved_experiment_id = str(experiment_id or "").strip()
    if not resolved_experiment_id:
        return None
    return MULTI_EXPERIMENT_ID_TO_CONFIG_NAME.get(resolved_experiment_id)


def infer_multi_config_name_from_checkpoint(checkpoint_path: str | Path) -> str | None:
    """Best-effort inference of the multi-agent config preset from checkpoint metadata."""

    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    signature = metadata.get("training_signature")
    summary = metadata.get("summary")
    experiment_config = metadata.get("experiment_config")
    experiment_id = (
        (signature.get("experiment_id") if isinstance(signature, dict) else None)
        or metadata.get("experiment_id")
        or (summary.get("experiment_id") if isinstance(summary, dict) else None)
        or (experiment_config.get("experiment_id") if isinstance(experiment_config, dict) else None)
    )
    return infer_multi_config_name_from_experiment_id(str(experiment_id or ""))


def resolve_multi_experiment_config(
    config_name: str | None = None,
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[str, MultiExperimentConfig]:
    """Resolve one config preset, optionally auto-inferring it from checkpoint metadata."""

    normalized_name = normalize_multi_experiment_config_name(config_name)
    if normalized_name and normalized_name != "auto":
        return normalized_name, get_multi_experiment_config(normalized_name)

    inferred_name = infer_multi_config_name_from_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    resolved_name = inferred_name or "variable"
    return resolved_name, get_multi_experiment_config(resolved_name)

