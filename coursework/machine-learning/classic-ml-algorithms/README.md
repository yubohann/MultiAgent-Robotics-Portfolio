# Classic Machine Learning Algorithms

[English](README.md) | [简体中文](README.zh-CN.md)

**Ten classical machine learning algorithms implemented from scratch, each with its own dataset, lab report and runnable command.**

The implementations use NumPy and the Python standard library for the algorithm logic, and every estimator is written directly in the project.

**Status.** All ten implementations run on the checked-in public datasets with Python 3.8 or newer and NumPy.

## My Role

Implemented every algorithm from scratch with NumPy and the Python standard library, including the optimization, pruning, smoothing and convergence logic, ran each implementation on its checked-in public dataset, and wrote the per-algorithm lab reports that document the runs and results.

## Algorithms

| # | Algorithm | Category | Dataset | Core idea |
|---:|---|---|---|---|
| 1 | [AdaBoost](AdaBoost/) | Ensemble learning | Magic Gamma | Decision stumps with adaptive sample weighting |
| 2 | [Apriori](Apriori/) | Association rules | Groceries | Frequent-itemset mining and confidence-based rules |
| 3 | [C4.5](C4.5/) | Decision tree | WDBC | Gain-ratio splitting, continuous-feature handling and pruning |
| 4 | [CART](CART/) | Decision tree | Wine Quality | Gini-impurity splitting for classification and regression trees |
| 5 | [EM](EM/) | Clustering | Congressional Voting | Expectation-maximization for Gaussian mixtures |
| 6 | [K-means](K-means/) | Clustering | Iris | k-means++ initialization and iterative convergence |
| 7 | [kNN](KNN/) | Classification | Iris | Distance metrics and majority voting |
| 8 | [Naive Bayes](Naive-Bayes/) | Classification | 20 Newsgroups | Multinomial model with Laplace smoothing |
| 9 | [PageRank](PageRank/) | Graph algorithm | Sample web graph | Power iteration with damping |
| 10 | [SVM](SVM/) | Classification | Iris | Multiclass margin-based optimization |

## Run Examples

Run commands from this directory.

```bash
python AdaBoost/src/AdaBoost.py -f AdaBoost/data/magic04.data
python Apriori/src/Apriori.py -f Apriori/data/archive/Groceries_dataset.csv
python C4.5/src/C4.5.py -f C4.5/data/wdbc.data
python CART/src/CART.py -f CART/data/winequality-red.csv
python EM/src/EM.py -f EM/data/house-votes-84.data
python K-means/src/K-means.py -f K-means/data/iris.data
python KNN/src/KNN.py -f KNN/data/iris.data
python Naive-Bayes/src/Naive_Bayes.py
python PageRank/src/PageRank.py --sample
python SVM/src/SVM.py -f SVM/data/iris.data
```

Naive Bayes reads the 20 Newsgroups corpus from a local directory and downloads it automatically through scikit-learn when the directory is missing.

## Repository Layout

```text
classic-ml-algorithms/
  <algorithm>/
    src/        # Implementation and CLI entry point
    data/       # Public dataset or metadata
    README.md   # Algorithm-specific lab report
    *.pdf       # Original course report, where available
```

## Notes

- Each algorithm is independently runnable and documented in its own directory.
- The per-algorithm reports document the coursework runs and results.
- The companion PCA, LDA and classifier work is in the [Machine Learning Coursework index](../README.md).

## License

Released into the public domain under the [Unlicense](LICENSE). The datasets come from public sources and keep their original terms.

*Bohan Yu, Machine Learning coursework.*
