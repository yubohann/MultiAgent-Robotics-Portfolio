# 经典机器学习算法实现

[English](README.md) | [简体中文](README.zh-CN.md)

**十个从零实现的经典机器学习算法，每个算法都有独立数据集，实验报告和可运行命令。**

算法逻辑使用 NumPy 与 Python 标准库编写，从决策树桩到幂迭代，所有估计器都在项目内直接实现。

**状态。** 十个实现均可基于仓库内公开数据集运行，环境为 Python 3.8 或更高版本与 NumPy。

## 算法目录

| # | 算法 | 类别 | 数据集 | 核心要点 |
|---:|---|---|---|---|
| 1 | [AdaBoost](AdaBoost/) | 集成学习 | Magic Gamma | 决策树桩与自适应样本权重 |
| 2 | [Apriori](Apriori/) | 关联规则 | Groceries | 频繁项集挖掘与置信度规则 |
| 3 | [C4.5](C4.5/) | 决策树 | WDBC | 增益率划分，连续特征处理与剪枝 |
| 4 | [CART](CART/) | 决策树 | Wine Quality | 基尼不纯度划分 |
| 5 | [EM](EM/) | 聚类 | Congressional Voting | 高斯混合模型的期望最大化 |
| 6 | [K-means](K-means/) | 聚类 | Iris | k-means++ 初始化与迭代收敛 |
| 7 | [kNN](KNN/) | 分类 | Iris | 距离度量与多数投票 |
| 8 | [Naive Bayes](Naive_Bayes/) | 分类 | 20 Newsgroups | 多项式模型与拉普拉斯平滑 |
| 9 | [PageRank](PageRank/) | 图算法 | 示例网页图 | 阻尼因子的幂迭代 |
| 10 | [SVM](SVM/) | 分类 | Iris | 多分类间隔优化 |

## 运行示例

在本目录下执行。

```bash
python AdaBoost/src/AdaBoost.py -f AdaBoost/data/magic04.data
python Apriori/src/Apriori.py -f Apriori/data/archive/Groceries_dataset.csv
python C4.5/src/C4.5.py -f C4.5/data/wdbc.data
python CART/src/CART.py -f CART/data/winequality-red.csv
python EM/src/EM.py -f EM/data/house-votes-84.data
python K-means/src/K-means.py -f K-means/data/iris.data
python KNN/src/KNN.py -f KNN/data/iris.data
python Naive_Bayes/src/Naive_Bayes.py
python PageRank/src/PageRank.py --sample
python SVM/src/SVM.py -f SVM/data/iris.data
```

朴素贝叶斯从本地目录读取 20 Newsgroups 语料，目录缺失时会通过 scikit-learn 自动下载。

## 目录结构

```text
classic-ml-algorithms/
  <algorithm>/
    src/        # 算法实现与命令行入口
    data/       # 公开数据集或元数据
    README.md   # 分算法实验报告
    *.pdf       # 可用的原始课程报告
```

## 说明

- 每个算法都可以独立运行，并在各自目录中配有实验报告。
- 分算法报告记录了课程实验的过程与结果。
- 相关的 PCA，LDA 与分类器作业见父目录的 [Machine Learning Coursework](../README.md)。

## 许可

本项目以 Unlicense 协议进入公有领域，见 [LICENSE](LICENSE)。数据集来自公开来源，沿用各自原始条款。

*Bohan Yu，机器学习课程作业。*
