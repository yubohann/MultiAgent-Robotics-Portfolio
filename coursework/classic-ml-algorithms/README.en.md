# Classic Machine Learning Algorithms — From Scratch

**Ten classic machine learning algorithms implemented from scratch** with NumPy / the Python standard library, covering supervised learning, unsupervised learning, association-rule mining, and graph algorithms.

This is coursework for the *Machine Learning* course at [REDACTED]. Every implementation is my own work — the algorithms themselves do not use any ML library — and each one ships with source code, a public dataset (UCI etc.), and a detailed lab report.

- **Author**: Bohan Yu (Bohan Yu)
- **Course**: Machine Learning — ten classic algorithms

## Algorithm list

| # | Algorithm | Category | Dataset | Core idea |
| --- | --- | --- | --- | --- |
| 1 | [AdaBoost](./AdaBoost) | Ensemble learning | Magic Gamma | Decision-stump weak learners + adaptive sample weighting |
| 2 | [Apriori](./Apriori) | Association rules | Groceries | Frequent itemset mining + confidence-based rule generation |
| 3 | [C4.5](./C4.5) | Decision tree | WDBC | Gain-ratio splitting + continuous-feature discretization + pruning |
| 4 | [CART](./CART) | Decision tree | Wine Quality | Gini-impurity splitting (classification & regression trees) |
| 5 | [EM](./EM) | Clustering | Congressional voting | Expectation-maximization for Gaussian mixtures |
| 6 | [K-means](./K-means) | Clustering | Iris | kmeans++ initialization + iterative convergence |
| 7 | [KNN](./KNN) | Classification | Iris | Euclidean / Manhattan / Chebyshev distance + majority vote |
| 8 | [Naive_Bayes](./Naive_Bayes) | Classification | 20 Newsgroups | Multinomial naive Bayes + Laplace smoothing |
| 9 | [PageRank](./PageRank) | Graph algorithm | Web link graph | Power iteration + damping factor |
| 10 | [SVM](./SVM) | Classification | Iris | Perceptron-style loss + gradient descent + multiclass |

## Why from scratch

Implementing classic algorithms by hand forces you to work through the math — entropy, information gain, gradient updates, EM iterations — and produces reproducible, auditable experiments. Only `numpy` / the standard library are used for the algorithms themselves.

## Layout

```text
<Algorithm>/
├── src/          # from-scratch implementation (CLI included)
├── data/         # public dataset (UCI, etc.)
├── README.md     # lab report: theory, implementation, experiments, analysis
└── *.pdf         # course lab report
```

Each algorithm runs independently, e.g.:

```bash
python KNN/src/KNN.py -f KNN/data/iris.data
python PageRank/src/PageRank.py --sample
python Naive_Bayes/src/Naive_Bayes.py -f Naive_Bayes/data/20news-18828
```

Requires Python 3.8+ and `numpy` only.

*Bohan Yu — Machine Learning course.*