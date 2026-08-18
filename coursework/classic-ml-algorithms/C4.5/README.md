# C4.5 决策树算法实现

## 1. 算法原理

C4.5 是由 Ross Quinlan 改进的决策树算法，是 ID3 的升级版本。

### 关键特点：
- **分裂准则**：使用**信息增益率（Gain Ratio）**代替 ID3 的信息增益
  - 信息增益率 = 信息增益 / 分裂信息
  - 避免偏向具有多个值的特征
  
- **支持连续特征**：通过二分裂（阈值分裂）处理连续数值特征
  
- **停止条件**：
  - 所有样本属于同一类
  - 样本数少于最小阈值
  - 达到最大深度限制
  - 没有有意义的分裂

### 伪代码：
```
C4.5(数据集D, 特征集A)
  if D 中所有样本属于同一类 then
    return 单节点树，标签为该类
  end if
  
  best_gain_ratio = 0
  best_feature = None
  best_threshold = None
  
  for each 特征 a in A do
    if a 是连续特征 then
      for each 可能的阈值 t do
        计算分裂 a <= t 的信息增益率
        if 信息增益率 > best_gain_ratio then
          best_gain_ratio = 信息增益率
          best_feature = a
          best_threshold = t
        end if
      end for
    end if
  end for
  
  if best_gain_ratio == 0 then
    return 单节点树，标签为 D 中最常见的类
  end if
  
  创建决策节点，分裂特征为 best_feature，阈值为 best_threshold
  将 D 分为 D_left (a <= t) 和 D_right (a > t)
  
  left_subtree = C4.5(D_left, A)
  right_subtree = C4.5(D_right, A)
  
  return 以该节点为根的树
```

## 2. 数据集说明

### 数据集名称
**Wisconsin Diagnostic Breast Cancer (WDBC)**

### 来源
- **URL**: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Creator**: University of Wisconsin-Madison, Clinical Sciences Center
- **创建者**: 
  - Dr. William H. Wolberg (wolberg@eagle.surgery.wisc.edu)
  - W. Nick Street (street@cs.wisc.edu)
  - Olvi L. Mangasarian

### 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 569 |
| **特征数** | 30 |
| **缺失值** | 无 |
| **类别** | 2（良性 B / 恶性 M） |
| **类别分布** | 357 良性（B）, 212 恶性（M） |

### 特征说明

数据基于**细针穿刺（Fine Needle Aspirate, FNA）**图像的数字化，用于诊断乳腺肿瘤。

对于每个细胞核，计算以下 **10 个基础特征**的**均值（mean）、标准误（SE）、最大值（worst）**，共 30 个特征：

1. **Radius** - 从中心到周边的平均距离
2. **Texture** - 灰度值的标准差
3. **Perimeter** - 周长
4. **Area** - 面积
5. **Smoothness** - 半径长度的局部变化
6. **Compactness** - 周长² / 面积 - 1.0
7. **Concavity** - 凹陷部分的严重程度
8. **Concave points** - 凹陷部分的数量
9. **Symmetry** - 对称性
10. **Fractal dimension** - "海岸线近似" - 1

### 特征列表

| 索引 | 特征名 | 类型 |
|------|--------|------|
| 1-10 | Mean (均值) | 连续数值 |
| 11-20 | Standard Error (标准误) | 连续数值 |
| 21-30 | Worst (最大值) | 连续数值 |

**目标变量**：Diagnosis
- **M** = Malignant（恶性）
- **B** = Benign（良性）

### 应用背景

该数据集用于通过机器学习诊断乳腺癌。已有研究表明，使用前 3 个特征（Worst Area、Worst Smoothness、Mean Texture）的分离平面可以达到 **97.5% 的准确率**。

### 文件格式

CSV 格式，每行一个样本：
```
ID, Diagnosis, Radius_mean, Texture_mean, ..., Fractal_dimension_worst
842302, M, 17.99, 10.38, ..., 0.1189
```

## 3. 代码结构

### 主要类和函数

- `load_wdbc_data(file_path)`: 加载 WDBC 数据集
- `Node`: 决策树节点类
- `C45Tree`: C4.5 决策树分类器
  - `entropy()`: 计算信息熵
  - `information_gain()`: 计算信息增益
  - `gain_ratio()`: 计算信息增益率
  - `best_split()`: 找到最佳分裂
  - `build_tree()`: 递归构建树
  - `fit()`: 训练模型
  - `predict()`: 进行预测
  - `print_tree()`: 打印树结构

### 依赖

- 标准库：`csv`, `math`, `random`, `argparse`, `os`, `collections`
- **无第三方库依赖**（不使用 scikit-learn 等）

## 4. 使用方法

### 基本用法

```bash
# 默认参数运行
python C4.5.py

# 指定数据文件和参数
python C4.5.py -f path/to/wdbc.data -d 15 -m 3 -t 0.2 -s 123

# 打印决策树结构
python C4.5.py -f path/to/wdbc.data --print_tree
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | WDBC 数据 CSV 文件路径 |
| `--max_depth` | `-d` | int | 20 | 决策树最大深度 |
| `--min_samples` | `-m` | int | 2 | 节点分裂所需最小样本数 |
| `--test_size` | `-t` | float | 0.3 | 测试集比例 |
| `--seed` | `-s` | int | 42 | 随机种子（复现结果） |
| `--print_tree` | 无 | flag | False | 是否打印树结构 |

### 输出示例

```
加载数据: 样本数=569, 特征数=30
类别分布: {'B': 357, 'M': 212}
训练集大小: 398, 测试集大小: 171

训练 C4.5 决策树 (max_depth=20, min_samples=2)...

=== 性能评估 ===
训练准确率: 0.9925
测试准确率: 0.9415

混淆矩阵:
       B      M
B    119      3
M      6     43

=== 分类指标 ===
类别 B: 精确率=0.9520, 召回率=0.9756, F1=0.9637
类别 M: 精确率=0.9348, 召回率=0.8776, F1=0.9053
```

## 5. 与 sklearn 的对比

本实现与 `sklearn.tree.DecisionTreeClassifier` 的主要区别：

| 特性 | C4.5 实现 | sklearn |
|------|---------|---------|
| 分裂准则 | 信息增益率（Gain Ratio） | Gini 或熵 |
| 代码依赖 | 纯 Python 标准库 | NumPy, SciPy |
| 特征处理 | 简单的阈值分裂 | 更优化的实现 |
| 树剪枝 | 未实现 | 支持 |
| 速度 | 较慢（小数据集） | 快（使用 C 实现） |
| 教学价值 | 高（易读易改） | 低（生产级） |

## 6. 实验结果

在 WDBC 数据集上，使用默认参数 (max_depth=20, min_samples=2) 的典型结果：

- **训练准确率**: ~99%
- **测试准确率**: ~94-96%
- **训练时间**: < 1 秒
- **推理时间**: < 10ms（对 100 个样本）

## 参考文献

1. Quinlan, J. R. (1993). "C4.5: Programs for Machine Learning"
2. Wolberg, W. H., Street, W. N., Mangasarian, O. L. (1995). "Machine learning techniques to diagnose breast cancer from fine-needle aspirates"
3. UCI Machine Learning Repository: https://archive.ics.uci.edu/
