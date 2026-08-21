# Classic Machine Learning Algorithms

[English](README.md) | [简体中文](README.zh-CN.md)

Ten classical machine-learning algorithms implemented from scratch for coursework. The implementations use NumPy or the Python standard library for the algorithm logic; they do not wrap equivalent estimator implementations from an ML library.

## Scope

| Item | Detail |
|---|---|
| Author | Bohan Yu |
| Course | Machine Learning |
| Focus | Supervised learning, clustering, association-rule mining, and graph algorithms |
| Runtime | Python 3.8+ and NumPy |
| Evidence | Per-algorithm source code, public datasets, lab reports, and runnable scripts |

## Algorithms

| # | Algorithm | Category | Dataset | Core idea |
|---:|---|---|---|---|
| 1 | [AdaBoost](AdaBoost/) | Ensemble learning | Magic Gamma | Decision stumps with adaptive sample weighting |
| 2 | [Apriori](Apriori/) | Association rules | Groceries | Frequent-itemset mining and confidence-based rules |
| 3 | [C4.5](C4.5/) | Decision tree | WDBC | Gain-ratio splitting, continuous-feature handling, and pruning |
| 4 | [CART](CART/) | Decision tree | Wine Quality | Gini-impurity splitting for classification and regression trees |
| 5 | [EM](EM/) | Clustering | Congressional Voting | Expectation-maximization for Gaussian mixtures |
| 6 | [K-means](K-means/) | Clustering | Iris | k-means++ initialization and iterative convergence |
| 7 | [kNN](KNN/) | Classification | Iris | Distance metrics and majority voting |
| 8 | [Naive Bayes](Naive_Bayes/) | Classification | 20 Newsgroups | Multinomial model with Laplace smoothing |
| 9 | [PageRank](PageRank/) | Graph algorithm | Sample web graph | Power iteration with damping |
| 10 | [SVM](SVM/) | Classification | Iris | Multiclass margin-based optimization |

## Repository Layout

```text
classic-ml-algorithms/
  <algorithm>/
    src/        # Implementation and CLI entry point
    data/       # Public dataset or metadata
    README.md   # Algorithm-specific lab report
    *.pdf       # Original course report, where available
```

## Run Examples

Run commands from this directory:

```bash
python AdaBoost/src/AdaBoost.py -f AdaBoost/data/magic04.data
python Apriori/src/Apriori.py -f Apriori/data/archive/Groceries_dataset.csv
python C4.5/src/C4.5.py -f C4.5/data/wdbc.data
python CART/src/CART.py -f CART/data/winequality-red.csv
python EM/src/EM.py -f EM/data/house-votes-84.data
python K-means/src/K-means.py -f K-means/data/iris.data
python KNN/src/KNN.py -f KNN/data/iris.data
python PageRank/src/PageRank.py --sample
python SVM/src/SVM.py -f SVM/data/iris.data
```

The 20 Newsgroups corpus used by Naive Bayes is not committed in full. Supply a local dataset path or use the project's documented data-loading route.

## Notes

- Each algorithm is independently runnable and documented in its own directory.
- The reports and datasets are coursework artifacts, not claims of production-ready or state-of-the-art implementations.
- See the parent [Machine Learning Coursework index](../README.md) for the related PCA/LDA and classifier assignment.
