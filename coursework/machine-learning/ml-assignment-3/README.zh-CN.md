# 机器学习作业三

[English](README.md) | [简体中文](README.zh-CN.md)

**围绕 UCI Iris 与 Wine 数据集的四个经典机器学习任务，从数据预处理一直跑到保存的图表产物。**

本作业实现一条紧凑流程，在固定划分下完成预处理，降维，从零实现的分类器，计分与可视化。

## 项目范围

| 项目 | 说明 |
|---|---|
| 作者 | Bohan Yu |
| 课程 | 机器学习作业三 |
| 数据集 | UCI Iris 与 Wine |
| 统一协议 | 固定随机种子 42 的分层 70 比 30 划分。只用训练集统计量做标准化 |
| 依赖 | NumPy，pandas，scikit-learn，matplotlib |

**状态。** 四项任务全部完成，图表，投影和比较摘要均保存在 `results/` 下。

## 任务与产物

| # | 任务 | 实现方式 | 保存证据 |
|---:|---|---|---|
| 1 | Wine PCA | scikit-learn 标准工作流 | 二维投影和解释方差结果 |
| 2 | Wine LDA | scikit-learn 标准工作流 | 一维判别投影 |
| 3 | Iris kNN | 从零实现向量化欧氏距离和投票 | 奇数 `k` 的准确率扫描 |
| 4 | Wine kNN 与 ID3 | 从零实现 kNN 和均值阈值 ID3 | 相同测试集上的比较 |

## 结果预览

<p align="center">
  <a href="results/pca_visualization.png"><img src="results/pca_visualization.png" alt="Wine PCA 投影" width="49%" /></a>
  <a href="results/lda_visualization.png"><img src="results/lda_visualization.png" alt="Wine LDA 投影" width="49%" /></a>
</p>

<p align="center">
  <a href="results/knn_accuracy_vs_k.png"><img src="results/knn_accuracy_vs_k.png" alt="Iris kNN 准确率与 k 的关系" width="72%" /></a>
</p>

已保存的 Iris 扫描在固定划分下选择 `k=7`，测试准确率为 `0.9556`。已保存的 Wine 比较中，选定 kNN 配置与均值阈值 ID3 的准确率均为 `0.9231`。这些数值描述当前仓库中的课程实验结果。

## 设计选择

- `src/utils.py` 统一实现数据加载，分层划分和只基于训练集的标准化。
- 从零实现的 kNN 在 NumPy 中完成向量化距离计算。
- 扫描覆盖奇数 `k`，投票结果保持确定。
- ID3 使用信息增益和单特征均值阈值，这是明确记录的课程简化版本，用于本次实验的连续特征处理。

## 目录与运行

```text
ml-assignment-3/
  data/       # Iris、Wine 数据及元数据
  src/        # 四项任务和共享预处理工具
  results/    # 可再生成的图、投影和比较摘要
  requirements.txt
```

```bash
python -m pip install -r requirements.txt
cd src
python task01_pca_wine.py
python task02_lda_wine.py
python task03_knn_iris.py
python task04_knn_id3_wine.py
```

任务脚本会把对应文件重新生成到 `results/` 中。相关的十个从零算法实现见父目录的 [Machine Learning Coursework](../README.md)。

Iris 与 Wine 数据集来自 UCI，沿用各自原始条款。

*Bohan Yu，机器学习作业三。*
