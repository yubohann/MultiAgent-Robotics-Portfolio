from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    tag: str | None = None
    dataset: str | None = None

    def identifier(self) -> str:
        if self.name:
            return self.name
        if self.tag:
            return self.tag
        raise ValueError("Candidate config requires either `name` or `tag`.")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class CandidateListConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: list[CandidateConfig]
    selection_policy: str | None = None
    test_policy: str | None = None


class DatasetCandidatesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    datasets: dict[str, list[CandidateConfig]]
    selection_policy: str | None = None
    test_policy: str | None = None


class StageDatasetCandidatesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    stages: dict[str, dict[str, list[CandidateConfig]]]
    selection_policy: str | None = None
    test_policy: str | None = None
    objective: dict[str, Any] = Field(default_factory=dict)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    with open(resolved_path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in config file: {resolved_path}")
    return payload


def load_candidate_list_config(path: str | Path) -> CandidateListConfig:
    return CandidateListConfig.model_validate(_load_yaml(path))


def load_dataset_candidates_config(path: str | Path) -> DatasetCandidatesConfig:
    return DatasetCandidatesConfig.model_validate(_load_yaml(path))


def load_stage_dataset_candidates_config(path: str | Path) -> StageDatasetCandidatesConfig:
    return StageDatasetCandidatesConfig.model_validate(_load_yaml(path))
