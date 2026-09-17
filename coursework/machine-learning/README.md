# Machine Learning Coursework

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="ml-assignment-3/results/pca_visualization.png"><img src="ml-assignment-3/results/pca_visualization.png" alt="PCA projection of the Wine classes" width="49%" /></a>
  <a href="ml-assignment-3/results/knn_accuracy_vs_k.png"><img src="ml-assignment-3/results/knn_accuracy_vs_k.png" alt="Iris kNN accuracy sweep" width="49%" /></a>
</p>

**Two machine learning coursework projects, ten classical algorithms written from scratch and a compact dimensionality and classification pipeline on UCI data.**

The first project implements AdaBoost, Apriori, C4.5, CART, EM, K-means, kNN, Naive Bayes, PageRank and SVM with NumPy and the Python standard library, and every algorithm keeps its own dataset, lab report and runnable command. The second project builds a small pipeline around the Iris and Wine datasets, with PCA, LDA, a vectorized kNN classifier and an ID3 tree comparison on a fixed split.

**Status.** Coursework complete. Sources, datasets, figures and saved results are all in the tree.

## My Role

Implemented all ten classical algorithms from the numerical core to the CLI entry point (`classic-ml-algorithms/*/src/`), ran each on its checked-in dataset and wrote the per-algorithm lab reports. Built the Assignment 3 pipeline end to end: data loading and fixed-split preprocessing, from-scratch vectorized kNN and mean-threshold ID3, the PCA and LDA runs, and the saved figures and comparison summaries under `ml-assignment-3/results/`.

## Projects

| Project | Scope | Stack |
|---|---|---|
| [Classic ML Algorithms](classic-ml-algorithms/) | Ten from-scratch implementations spanning ensembles, trees, clustering, association rules and PageRank | Python, NumPy, public datasets |
| [ML Assignment 3](ml-assignment-3/) | PCA and LDA on Wine, an Iris kNN sweep and a Wine comparison of kNN against ID3 | Python, NumPy, pandas, scikit-learn, matplotlib |

## Results Preview

The saved Iris sweep selects `k=7` at `0.9556` test accuracy. The saved Wine comparison reports `0.9231` for both the selected kNN configuration and the mean-threshold ID3 tree on the checked-in split.

## Documentation

- [Classic ML Algorithms](classic-ml-algorithms/README.md), implementation notes, datasets and run commands for the ten from-scratch algorithms.
- [ML Assignment 3](ml-assignment-3/README.md), tasks, design choices and saved outputs for the PCA, LDA, kNN and ID3 work.

## Layout

```text
machine-learning/
  README.md
  README.zh-CN.md
  classic-ml-algorithms/
  ml-assignment-3/
```

## License

The classic algorithm collection is released into the public domain under the [Unlicense](classic-ml-algorithms/LICENSE). The datasets come from public sources and keep their original terms.

*Bohan Yu, Machine Learning coursework.*
