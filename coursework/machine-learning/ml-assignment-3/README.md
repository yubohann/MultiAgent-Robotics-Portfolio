# Machine Learning Assignment 3

[English](README.md) | [简体中文](README.zh-CN.md)

**Four classical machine learning tasks on the UCI Iris and Wine datasets, running end to end from preprocessing to saved figures.**

The assignment builds a compact pipeline that covers preprocessing, dimensionality reduction, from-scratch classification, scoring and visualization on a fixed split.

## Scope

| Item | Detail |
|---|---|
| Author | Bohan Yu |
| Course | Machine Learning, Assignment 3 |
| Datasets | UCI Iris and Wine |
| Shared protocol | Stratified 70 to 30 split with seed 42 and training-set-only standardization |
| Dependencies | NumPy, pandas, scikit-learn and matplotlib |

**Status.** All four tasks complete, with figures, projections and comparison summaries saved under `results/`.

## My Role

Built the four tasks and the shared preprocessing around a fixed split: the from-scratch vectorized kNN classifier, the mean-threshold ID3 tree, the PCA and LDA runs, and the saved projections, accuracy sweep and comparison summaries under `results/`.

## Tasks and Saved Outputs

| # | Task | Implementation | Evidence |
|---:|---|---|---|
| 1 | PCA on Wine | scikit-learn baseline workflow | 2D projection and explained-variance output |
| 2 | LDA on Wine | scikit-learn baseline workflow | One-dimensional discriminative projection |
| 3 | kNN on Iris | From scratch implementation with vectorized Euclidean distance and voting | Odd `k` accuracy sweep |
| 4 | kNN and ID3 on Wine | From scratch kNN and mean-threshold ID3 | Shared-test-set comparison |

## Results Preview

<p align="center">
  <a href="results/pca_visualization.png"><img src="results/pca_visualization.png" alt="PCA projection of Wine classes" width="49%" /></a>
  <a href="results/lda_visualization.png"><img src="results/lda_visualization.png" alt="LDA projection of Wine classes" width="49%" /></a>
</p>

<p align="center">
  <a href="results/knn_accuracy_vs_k.png"><img src="results/knn_accuracy_vs_k.png" alt="Iris kNN accuracy versus k" width="72%" /></a>
</p>

The saved Iris sweep selects `k=7` at `0.9556` test accuracy for the fixed split. The saved Wine comparison reports `0.9231` for both the selected kNN configuration and the mean-threshold ID3 tree. These values describe the checked-in split and artifacts for this coursework experiment.

## Design Choices

- `src/utils.py` centralizes loading, stratified splitting and train-only standardization.
- The from-scratch kNN implementation computes vectorized pairwise distances in NumPy.
- The sweep covers odd `k` values, so votes always resolve.
- The ID3 implementation uses an information-gain split with a per-feature mean threshold, a documented coursework simplification of continuous-feature tree search.

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

The Iris and Wine datasets come from UCI and keep their original terms.

*Bohan Yu, Machine Learning Assignment 3.*
