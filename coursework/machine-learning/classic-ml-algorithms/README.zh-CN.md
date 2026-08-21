# 经典机器学习算法实现

[English](README.md) | [简体中文](README.zh-CN.md)

这是机器学习课程中的十个经典算法实现。算法核心仅使用 NumPy 或 Python 标准库编写，不调用等价的机器学习库分类器或聚类器作为实现替代。

## 项目范围

| 项目 | 说明 |
|---|---|
| 作者 | Bohan Yu |
| 课程 | 机器学习 |
| 覆盖内容 | 监督学习、聚类、关联规则挖掘与图算法 |
| 运行环境 | Python 3.8+ 与 NumPy |
| 证据 | 分算法源码、公开数据集、实验报告与可运行脚本 |

## 算法目录

| # | 算法 | 类别 | 数据集 | 核心要点 |
|---:|---|---|---|---|
| 1 | [AdaBoost](AdaBoost/) | 集成学习 | Magic Gamma | 决策树桩与自适应样本权重 |
| 2 | [Apriori](Apriori/) | 关联规则 | Groceries | 频繁项集挖掘与置信度规则 |
| 3 | [C4.5](C4.5/) | 决策树 | WDBC | 增益率划分、连续特征处理与剪枝 |
| 4 | [CART](CART/) | 决策树 | Wine Quality | 基尼不纯度划分 |
| 5 | [EM](EM/) | 聚类 | Congressional Voting | 高斯混合模型的期望最大化 |
| 6 | [K-means](K-means/) | 聚类 | Iris | k-means++ 初始化与迭代收敛 |
| 7 | [kNN](KNN/) | 分类 | Iris | 距离度量与多数投票 |
| 8 | [Naive Bayes](Naive_Bayes/) | 分类 | 20 Newsgroups | 多项式模型与拉普拉斯平滑 |
| 9 | [PageRank](PageRank/) | 图算法 | 示例网页图 | 阻尼因子的幂迭代 |
| 10 | [SVM](SVM/) | 分类 | Iris | 多分类间隔优化 |

## 目录与运行

每个算法目录都保留 `src/`、`data/`、算法实验报告和原始课程 PDF（如有）。从当前目录运行：

```bash
python KNN/src/KNN.py -f KNN/data/iris.data
python PageRank/src/PageRank.py --sample
```

朴素贝叶斯使用的完整 20 Newsgroups 语料未提交；请传入本地数据路径或按项目文档中的数据加载方式执行。

相关的 PCA/LDA 与分类作业见父目录的 [Machine Learning Coursework](../README.md)。
