from __future__ import annotations

ACTIVE_RUNTIME_DATASETS: tuple[str, ...] = (
    "amazon",
    "yelp",
    "comp",
    "ieee",
    "archive",
    "ccfd",
    "amlsim",
    "elliptic",
    "ethereum_phishing",
    "ethereum_ponzi",
    "defi_rug_pull",
)
ACTIVE_RUNTIME_DATASET_SET = frozenset(str(name).strip().lower() for name in ACTIVE_RUNTIME_DATASETS)


def active_runtime_datasets() -> tuple[str, ...]:
    return ACTIVE_RUNTIME_DATASETS


def is_dataset_enabled(dataset_name: str) -> bool:
    return str(dataset_name).strip().lower() in ACTIVE_RUNTIME_DATASET_SET


def ensure_dataset_enabled(dataset_name: str, *, context: str = "") -> None:
    normalized_name = str(dataset_name).strip().lower()
    if normalized_name in ACTIVE_RUNTIME_DATASET_SET:
        return
    location = f"{context}: " if str(context).strip() else ""
    active_names = ", ".join(ACTIVE_RUNTIME_DATASETS)
    raise RuntimeError(
        f"{location}dataset '{dataset_name}' is disabled by the current runtime policy. "
        f"Active dataset(s): {active_names}. "
        "Enable the dataset in runtime_dataset_policy.py before using it in the main pipeline."
    )
