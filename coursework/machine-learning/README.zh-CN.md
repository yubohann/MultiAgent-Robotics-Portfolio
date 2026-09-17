# 机器学习课程项目

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <a href="ml-assignment-3/results/pca_visualization.png"><img src="ml-assignment-3/results/pca_visualization.png" alt="Wine 类别 PCA 投影" width="49%" /></a>
  <a href="ml-assignment-3/results/knn_accuracy_vs_k.png"><img src="ml-assignment-3/results/knn_accuracy_vs_k.png" alt="Iris kNN 准确率扫描" width="49%" /></a>
</p>

**两个机器学习课程项目，十个从零实现的经典算法，以及一条围绕 UCI 数据的降维与分类小流程。**

第一个项目使用 NumPy 和 Python 标准库实现 AdaBoost，Apriori，C4.5，CART，EM，K-means，kNN，朴素贝叶斯，PageRank 和 SVM，每个算法都有自己的数据集，实验报告和可运行命令。第二个项目围绕 Iris 与 Wine 数据集构建完整小流程，包含 PCA，LDA，向量化 kNN 分类器和固定划分下的 ID3 决策树比较。

**状态。** 课程作业已完成，源码，数据集，图表和保存的结果都在仓库中。

## 项目导航

| 项目 | 内容 | 技术栈 |
|---|---|---|
| [经典机器学习算法](classic-ml-algorithms/README.zh-CN.md) | 十个从零实现，覆盖集成学习，决策树，聚类，关联规则和 PageRank | Python，NumPy，公开数据集 |
| [机器学习作业三](ml-assignment-3/README.zh-CN.md) | Wine 上的 PCA 和 LDA，Iris kNN 扫描，Wine 上 kNN 与 ID3 的比较 | Python，NumPy，pandas，scikit-learn，matplotlib |

## 结果预览

已保存的 Iris 扫描在固定划分下选择 `k=7`，测试准确率为 `0.9556`。已保存的 Wine 比较中，选定 kNN 配置与均值阈值 ID3 的准确率均为 `0.9231`。

## 文档入口

- [经典机器学习算法](classic-ml-algorithms/README.zh-CN.md)，十个从零算法的实现说明，数据集和运行命令。
- [机器学习作业三](ml-assignment-3/README.zh-CN.md)，PCA，LDA，kNN 与 ID3 任务的设计与保存产物。

## 目录结构

```text
machine-learning/
  README.md
  README.zh-CN.md
  classic-ml-algorithms/
  ml-assignment-3/
```

## 许可

经典算法合集以 Unlicense 协议进入公有领域，见 [LICENSE](classic-ml-algorithms/LICENSE)。数据集来自公开来源，沿用各自原始条款。

*Bohan Yu，机器学习课程作业。*
