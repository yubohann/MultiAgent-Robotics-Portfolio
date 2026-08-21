# Classic Machine Learning Algorithms — From Scratch



**Ten classic machine learning algorithms implemented from scratch** with NumPy / the Python standard library: AdaBoost, Apriori, C4.5, CART, EM, K-means, KNN, Naive Bayes, PageRank, and SVM.



This is my coursework for the *Machine Learning* course at [REDACTED]. Each algorithm was written by me — not wrapped from a library — and comes with its own lab report (in the per-algorithm `README.md`), a public dataset, and a runnable script.



## About this work



- **Author**: Bohan Yu (Bohan Yu)

- **Course**: Machine Learning — ten classic algorithms (机器学习十大算法)

- **What it is**: ten independent implementations, one per folder, each with source code, dataset, and a detailed lab report covering the math, the implementation, the experiments, and the analysis.

- **Why from scratch**: writing entropy, information gain, gradient updates, and EM iterations by hand is the fastest way to actually understand when an algorithm works and when it breaks.



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

python AdaBoost/src/AdaBoost.py -f AdaBoost/data/magic04.data

python Apriori/src/Apriori.py -f Apriori/data/archive/Groceries_dataset.csv

python C4.5/src/C4.5.py -f C4.5/data/wdbc.data

python CART/src/CART.py -f CART/data/winequality-red.csv

python EM/src/EM.py -f EM/data/house-votes-84.data

python K-means/src/K-means.py -f K-means/data/iris.data

python KNN/src/KNN.py -f KNN/data/iris.data

python Naive_Bayes/src/Naive_Bayes.py -f Naive_Bayes/data/20news-18828

python PageRank/src/PageRank.py --sample

python SVM/src/SVM.py -f SVM/data/iris.data

```



> Note: the 20 Newsgroups dataset (~18k documents) for Naive_Bayes is not

> committed; point `-f` at a local copy or let the script fetch it via

> scikit-learn.



Requires Python 3.8+ and `numpy` (a few algorithms use only the standard

library).



---



## 中文说明



十个经典机器学习算法的从零实现（AdaBoost / Apriori / C4.5 / CART / EM /

K-means / KNN / 朴素贝叶斯 / PageRank / SVM），仅依赖 NumPy 或标准库。

每个算法目录内含源码、数据集和完整实验报告。课程：机器学习十大算法，

，Bohan Yu。



*Bohan Yu — Machine Learning course.*