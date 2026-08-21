"""Dependency-light constants shared by the training CLI and runtime pipeline."""

from __future__ import annotations

from .runtime_dataset_policy import active_runtime_datasets


DEFAULT_DEVICE_REQUEST = "cuda"
DEFAULT_HYBRID_MAINLINE_ROUNDS = 24
LEGACY_BATCH_DATASETS = active_runtime_datasets()
SUPPORTED_HYBRID_DATASETS = LEGACY_BATCH_DATASETS
DATASET_SELECTION_ALIASES = {
    "all": LEGACY_BATCH_DATASETS,
    "all_supported": SUPPORTED_HYBRID_DATASETS,
}
DATASET_SELECTION_CHOICES = tuple(DATASET_SELECTION_ALIASES) + SUPPORTED_HYBRID_DATASETS
