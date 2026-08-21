# SVM (线性 SVM — Pegasos)

本目录包含一个不依赖第三方机器学习库（如 scikit-learn、numpy）的线性 SVM 实现，使用 Pegasos（随机子梯度下降）优化器，并通过 one-vs-rest 策略支持多分类（用于 Iris 数据集示例）。

**文件结构**：

```
SVM/
├── src/
│   └── SVM.py        # 主要实现（Pegasos + one-vs-rest）
├── data/
│   └── iris.data     # 可放置 Iris 数据集（也可自动从 UCI 下载）
└── README.md         # 本文件
```

## 算法简介

- 使用 Pegasos（随机子梯度下降）训练线性 SVM（hinge loss + L2 正则化）
- 对于多类问题（Iris），采用 one-vs-rest：为每个类别训练一个二元分类器，预测时选择评分最高的类别
- 特点：实现轻量、易理解，适合作为教学/实验用途

## 数据集：Iris（鸢尾花）

- **来源**：UCI Machine Learning Repository
  - URL: https://archive.ics.uci.edu/ml/datasets/iris
  - 数据文件直接链接：`https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`

- **基本情况**：
  - 记录数（实例数）：150
  - 属性数（特征数）：4
    - sepal length in cm
    - sepal width in cm
    - petal length in cm
    - petal width in cm
  - 类别数：3（Iris-setosa, Iris-versicolor, Iris-virginica）
  - 数据格式：CSV，每行 5 列（前 4 列为浮点数特征，第 5 列为类别标签）

- **备注**：本实现会尝试在 `SVM/data/iris.data` 找到数据文件；若不存在并启用 `--sample`，脚本会尝试从 UCI 自动下载。

## 使用示例

1. 使用内置样例（会自动下载 Iris 数据集到 `SVM/data/iris.data`）

```powershell
cd "d:\360MoveData\Users\21281\Desktop\机器学习作业四"
& C:/Users/21281/.conda/envs/py310/python.exe SVM/src/SVM.py --sample
```

2. 指定本地数据文件并调整参数

```powershell
& C:/Users/21281/.conda/envs/py310/python.exe SVM/src/SVM.py -f SVM/data/iris.data --epochs 100 --lambda_reg 0.0001 --batch_size 1
```

3. 限制样本数用于快速调试

```powershell
& C:/Users/21281/.conda/envs/py310/python.exe SVM/src/SVM.py --sample --max_samples 50
```

## 参数说明

- `-f, --file`：Iris 数据文件路径（默认 `SVM/data/iris.data`）
- `--sample`：使用示例（自动下载 Iris 数据）
- `--epochs`：训练轮数，默认 `50`
- `--batch_size`：批量大小，默认 `1`（即随机梯度）
- `--lambda_reg`：正则化参数，默认 `0.0001`
- `--test_size`：测试集占比，默认 `0.2`
- `--max_samples`：限制载入的样本数（用于快速调试）

## 输出

脚本会打印训练/测试样本数、类别信息、以及测试准确率，并展示若干预测示例。

## 参考

- Shai Shalev-Shwartz, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", 2007
- UCI Iris dataset: https://archive.ics.uci.edu/ml/datasets/iris

---

**创建日期**：2025年11月28日  
**作者**：23计算1Bohan Yu  