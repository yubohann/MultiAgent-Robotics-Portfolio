# Machine Learning Assignment 3

[English](README.md) | [简体中文](README.zh-CN.md)

This assignment builds a compact classical-machine-learning pipeline around the UCI Iris and Wine datasets: preprocessing, dimensionality reduction, from-scratch classification, evaluation, and visualization.

## Scope

| Item | Detail |
|---|---|
| Author | Bohan Yu |
| Course | Machine Learning, Assignment 3 |
| Datasets | UCI Iris and Wine |
| Shared protocol | Stratified 70/30 split with seed 42; training-set-only standardization |
| Dependencies | NumPy, pandas, scikit-learn, and matplotlib |

## Tasks and Saved Outputs

| # | Task | Implementation | Evidence |
|---:|---|---|---|
| 1 | PCA on Wine | scikit-learn baseline workflow | 2D projection and explained-variance output |
| 2 | LDA on Wine | scikit-learn baseline workflow | One-dimensional discriminative projection |
| 3 | kNN on Iris | From scratch: vectorized Euclidean distance and voting | Odd-`k` accuracy sweep |
| 4 | kNN and ID3 on Wine | From scratch: kNN and mean-threshold ID3 | Shared-test-set comparison |

## Results Preview

<p align="center">
  <a href="results/pca_visualization.png"><img src="results/pca_visualization.png" alt="PCA projection of Wine classes" width="49%" /></a>
  <a href="results/lda_visualization.png"><img src="results/lda_visualization.png" alt="LDA projection of Wine classes" width="49%" /></a>
</p>

<p align="center">
  <a href="results/knn_accuracy_vs_k.png"><img src="results/knn_accuracy_vs_k.png" alt="Iris kNN accuracy versus k" width="72%" /></a>
</p>

The saved Iris sweep selects `k=7` with test accuracy `0.9556` for the fixed split. The saved Wine comparison reports `0.9231` for both the selected kNN configuration and the mean-threshold ID3 tree. These values describe the checked-in split and artifacts only; they are not generalization claims beyond this coursework experiment.

## Design Choices

- `src/utils.py` centralizes loading, stratified splitting, and train-only standardization.
- The from-scratch kNN implementation uses vectorized pairwise distances rather than a nested Python loop.
- Only odd `k` values are evaluated to avoid tied votes.
- The ID3 implementation uses an information-gain split with a per-feature mean threshold. This is a documented coursework simplification, not an exhaustive continuous-feature tree search.

## Layout

```text
ml-assignment-3/
  data/       # Iris and Wine data plus metadata
  src/        # Four tasks and shared preprocessing helpers
  results/    # Regenerated figures, projections, and comparison summary
  requirements.txt
```

## Run

```bash
python -m pip install -r requirements.txt
cd src
python task01_pca_wine.py
python task02_lda_wine.py
python task03_knn_iris.py
python task04_knn_id3_wine.py
```

The task scripts regenerate their corresponding files in `results/`. See the parent [Machine Learning Coursework index](../README.md) for the companion from-scratch algorithm collection.
