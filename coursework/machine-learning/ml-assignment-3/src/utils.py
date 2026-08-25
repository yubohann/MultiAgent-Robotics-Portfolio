"""Shared helpers for the assignment tasks.

Data loading, standardization, and the stratified split all live here so the
four task scripts don't each re-implement them.
"""

# Author: Bohan Yu
# Machine learning course, assignment 3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# one seed everywhere, keeps every run reproducible
RANDOM_STATE = 42

IRIS_LABELS = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
IRIS_LABEL_NAMES = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

WINE_COLUMNS = [
    "Class",
    "Alcohol",
    "Malic_Acid",
    "Ash",
    "Alcalinity_of_Ash",
    "Magnesium",
    "Total_Phenols",
    "Flavanoids",
    "Nonflavanoid_Phenols",
    "Proanthocyanins",
    "Color_Intensity",
    "Hue",
    "OD280_OD315_of_Diluted_Wines",
    "Proline",
]


def load_iris(file_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read iris.data and turn the string labels into 0/1/2."""
    path = Path(file_path) if file_path else DATA_DIR / "iris.data"
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, :-1].to_numpy(dtype=float)
    y = np.array([IRIS_LABELS[label] for label in df.iloc[:, -1]])
    return x, y


def load_wine(file_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read wine.data; original labels are 1/2/3, shift them down to 0/1/2."""
    path = Path(file_path) if file_path else DATA_DIR / "wine.data"
    df = pd.read_csv(path, header=None)
    y = df.iloc[:, 0].to_numpy(dtype=int) - 1
    x = df.iloc[:, 1:].to_numpy(dtype=float)
    return x, y


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize with train-set statistics only.

    Using the test set's own mean/std here would leak information into the
    evaluation, so I never touch it.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def stratified_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split so each class keeps its proportion in both sets."""
    rng = np.random.default_rng(random_state)
    x_train, x_test, y_train, y_test = [], [], [], []
    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)
        split = int(len(indices) * test_size)
        x_train.extend(x[indices[split:]])
        x_test.extend(x[indices[:split]])
        y_train.extend(y[indices[split:]])
        y_test.extend(y[indices[:split]])
    return (
        np.array(x_train),
        np.array(x_test),
        np.array(y_train),
        np.array(y_test),
    )


def results_path(name: str) -> Path:
    """Build an output path under results/, creating the folder if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name