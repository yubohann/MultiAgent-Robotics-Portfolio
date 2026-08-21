# K 近邻（K-Nearest Neighbors, KNN）算法实现

## 1. 算法原理

K 近邻（KNN）是一种简单而有效的分类算法，属于**懒惰学习**（Lazy Learning）范畴。它的核心思想是：**一个样本的类别由其最近邻的 K 个样本的多数类别决定**。

### 核心思想

KNN 基于一个假设：**相似的样本往往具有相同的类别**

在特征空间中，距离近的样本在实际问题中往往有相同的标签。

### KNN 算法流程

```
KNNClassify(训练集 D, 测试样本 x, k)
  1. 计算 x 到训练集中所有样本的距离
  2. 按距离从小到大排序
  3. 选择最近的 K 个样本
  4. 统计这 K 个样本的标签频数
  5. 选择频数最高的标签作为预测结果
```

### 距离度量方式

KNN 的关键是如何度量样本之间的相似性。常用的距离度量有：

#### 1. 欧几里得距离（Euclidean Distance）
$$d(x_1, x_2) = \sqrt{\sum_{i=1}^{n}(x_{1i} - x_{2i})^2}$$

**特点**：最常用，对所有维度平等对待

#### 2. 曼哈顿距离（Manhattan Distance）
$$d(x_1, x_2) = \sum_{i=1}^{n}|x_{1i} - x_{2i}|$$

**特点**：计算速度快，在高维空间表现较好

#### 3. 切比雪夫距离（Chebyshev Distance）
$$d(x_1, x_2) = \max_{i=1}^{n}|x_{1i} - x_{2i}|$$

**特点**：只关注最大差异维度

### KNN 中的投票机制

选择最近的 K 个样本后，使用**多数投票**确定类别：

$$\hat{y} = \arg\max_c \sum_{i=1}^{K} I(y_i = c)$$

其中 $I$ 是指示函数，$c$ 是类别。

### 伪代码

```
Algorithm KNN
Input: 
  - D: 训练数据 {(x_1, y_1), ..., (x_m, y_m)}
  - x: 测试样本
  - k: 近邻数量
  
Output: 预测类别 ŷ

1. distances = []
2. for each (x_i, y_i) in D:
     distance = ||x - x_i||
     distances.append((distance, y_i))
   end for
   
3. sort(distances) by distance (ascending)
   
4. k_nearest = first k elements of distances
   
5. labels = [y_i for (distance, y_i) in k_nearest]
   
6. ŷ = argmax(count(label) for label in labels)
   
7. return ŷ
```

## 2. 数据集说明

### 数据集名称
**Iris Dataset（鸢尾花数据集）**

### 来源
- **来源**：UCI Machine Learning Repository
- **创建者**：Ronald Fisher（1936）
- **年份**：1936 年（经典数据集）
- **URL**：https://archive.ics.uci.edu/dataset/53/iris
- **许可证**：CC BY 4.0

### 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 150 |
| **特征数** | 4 |
| **类别数** | 3 |
| **缺失值** | 无 |
| **数据类型** | 连续值 |
| **语言** | 英文 |

### 特征说明

所有特征都是**花朵的物理测量数据**（单位：厘米）：

| 特征编号 | 特征名称 | 缩写 | 说明 |
|---------|--------|------|------|
| 1 | Sepal Length | SL | 花萼长度 |
| 2 | Sepal Width | SW | 花萼宽度 |
| 3 | Petal Length | PL | 花瓣长度 |
| 4 | Petal Width | PW | 花瓣宽度 |

**特征范围**：

| 特征 | 最小值 | 最大值 | 平均值 |
|------|--------|--------|--------|
| Sepal Length | 4.3 | 7.9 | 5.84 |
| Sepal Width | 2.0 | 4.4 | 3.06 |
| Petal Length | 1.0 | 6.9 | 3.76 |
| Petal Width | 0.1 | 2.5 | 1.20 |

### 类别说明

三种鸢尾花，每类各 50 个样本：

| 类别 | 学名 | 中文名 | 样本数 |
|------|------|--------|--------|
| Iris-setosa | *Iris setosa* | 山鸢尾 | 50 |
| Iris-versicolor | *Iris versicolor* | 变色鸢尾 | 50 |
| Iris-virginica | *Iris virginica* | 维吉尼亚鸢尾 | 50 |

### 类别分布（平衡数据集）

```
Iris-setosa:     50 样本 (33.3%)
Iris-versicolor: 50 样本 (33.3%)
Iris-virginica:  50 样本 (33.3%)
```

### 数据特性

**优点**：
- ✓ 经典且平衡的数据集
- ✓ 无缺失值
- ✓ 低维度（易于可视化）
- ✓ 清晰的决策边界

**缺点**：
- 样本量较小（150）
- 只有 3 个类别

### 应用背景

**历史**：
- Ronald Fisher 在 1936 年使用此数据集发表论文
- 是最著名的机器学习入门数据集
- 已被引用数千次

**用途**：
- 分类算法的基准测试
- 机器学习教学示例
- 模式识别研究

### 文件信息

```
iris.data  (4.6 KB)   150 行，每行一个样本
iris.names (1.6 KB)   数据集文档
```

### 数据格式

CSV 格式，逗号分隔：

```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
```

## 3. 代码结构

### 主要类和函数

#### KNN 类
```python
class KNN:
    def __init__(k, distance_metric)           # 初始化
    def fit(X, y)                              # 存储训练数据
    def predict(X)                             # 预测类别
    def predict_single(x)                      # 预测单个样本
    def predict_proba(X)                       # 预测概率
    def predict_proba_single(x)                # 预测单个样本的概率
    def _distance(x1, x2)                      # 计算距离
    def _euclidean_distance(x1, x2)           # 欧几里得距离
    def _manhattan_distance(x1, x2)           # 曼哈顿距离
    def _chebyshev_distance(x1, x2)           # 切比雪夫距离
```

#### 辅助函数
- `load_iris_data(file_path)`: 加载 Iris 数据集
- `normalize_features(X, X_mean, X_std)`: 特征归一化
- `train_test_split(X, y, test_ratio, seed)`: 划分训练/测试集
- `calculate_accuracy(y_true, y_pred)`: 计算准确率
- `calculate_confusion_matrix(y_true, y_pred, class_names)`: 混淆矩阵
- `calculate_metrics(y_true, y_pred, class_names)`: 分类指标

### 核心实现

| 功能 | 实现方法 |
|------|--------|
| **距离计算** | 三种度量（欧、曼、切） |
| **K 个邻居搜索** | 遍历并排序距离 |
| **投票机制** | Counter 多数投票 |
| **概率预测** | K 个邻居的标签频数 / K |

### 依赖

- 标准库：`csv`, `math`, `random`, `argparse`, `collections`
- **无第三方库依赖**（不使用 scikit-learn、numpy 等）

## 4. 使用方法

### 基本用法

```bash
# 进入项目目录
cd KNN

# 使用默认参数运行
python src/KNN.py

# 指定数据文件和参数
python src/KNN.py -f path/to/iris.data -k 5 -m euclidean -t 0.3
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | Iris 数据 CSV 文件路径 |
| `-k` | - | int | 5 | 近邻数量 |
| `--metric` | `-m` | str | euclidean | 距离度量（euclidean/manhattan/chebyshev） |
| `--test_ratio` | `-t` | float | 0.3 | 测试集比例 |
| `--seed` | `-s` | int | 42 | 随机种子 |
| `--normalize` | - | flag | False | 是否进行特征归一化 |

### 运行示例

```bash
# 示例 1：基础运行（K=5，欧几里得距离）
python src/KNN.py -f data/iris.data

# 示例 2：K=3（更近的邻居）
python src/KNN.py -f data/iris.data -k 3

# 示例 3：K=7（更远的邻居）
python src/KNN.py -f data/iris.data -k 7

# 示例 4：曼哈顿距离
python src/KNN.py -f data/iris.data -m manhattan

# 示例 5：切比雪夫距离
python src/KNN.py -f data/iris.data -m chebyshev

# 示例 6：特征归一化
python src/KNN.py -f data/iris.data --normalize
```

### 输出示例

```
加载 Iris 数据集...
样本数: 150, 特征数: 4, 类别数: 3
特征: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
类别: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
训练集: 105 样本, 测试集: 45 样本

训练 KNN 模型 (k=5, metric=euclidean)...
进行预测...

=== 准确率 ===
训练集: 0.9619
测试集: 0.9778

=== 混淆矩阵 ===
预测类别 ->
真实\预测       Iris-setosa    Iris-versicolor    Iris-virginica
Iris-setosa      21             0                  0
Iris-versicolor   0            11                  1
Iris-virginica    0             0                 12

=== 分类指标（测试集）===
类别                 精确率    召回率    F1       样本数
Iris-setosa        1.0000   1.0000   1.0000     21
Iris-versicolor    1.0000   0.9167   0.9565     12
Iris-virginica     0.9231   1.0000   0.9600     12
```

## 5. 关键算法细节

### 距离计算

```python
def euclidean_distance(x1, x2):
    """欧几里得距离"""
    return math.sqrt(sum((x1[i] - x2[i])**2 for i in range(len(x1))))

def manhattan_distance(x1, x2):
    """曼哈顿距离"""
    return sum(abs(x1[i] - x2[i]) for i in range(len(x1)))

def chebyshev_distance(x1, x2):
    """切比雪夫距离"""
    return max(abs(x1[i] - x2[i]) for i in range(len(x1)))
```

### K 个邻居投票

```python
def predict_single(x):
    # 计算距离
    distances = [(distance(x, x_train), y_train) for x_train, y_train in zip(X_train, y_train)]
    
    # 排序
    distances.sort(key=lambda d: d[0])
    
    # 选择最近的 K 个
    k_nearest = distances[:k]
    
    # 投票
    labels = [y for _, y in k_nearest]
    counter = Counter(labels)
    
    # 返回频数最高的标签
    return counter.most_common(1)[0][0]
```

### 特征归一化

```python
def normalize_features(X):
    """Z-score 归一化"""
    X_normalized = []
    for feature_idx in range(n_features):
        col = [X[i][feature_idx] for i in range(len(X))]
        mean = sum(col) / len(col)
        std = sqrt(sum((val - mean)**2 for val in col) / len(col))
        
        # 标准化
        normalized_col = [(val - mean) / std for val in col]
        X_normalized.append(normalized_col)
    
    return X_normalized
```

## 6. 性能指标

### 典型运行结果（K=5，欧几里得距离）

| 指标 | 值 |
|------|-----|
| **训练准确率** | 0.9619 |
| **测试准确率** | 0.9778 |

### 按类别性能

| 类别 | 精确率 | 召回率 | F1 分数 | 样本数 |
|------|--------|--------|---------|--------|
| Iris-setosa | 1.0000 | 1.0000 | 1.0000 | 21 |
| Iris-versicolor | 1.0000 | 0.9167 | 0.9565 | 12 |
| Iris-virginica | 0.9231 | 1.0000 | 0.9600 | 12 |

### K 值的影响

| K | 训练准确率 | 测试准确率 | 说明 |
|----|----------|----------|------|
| 1 | 1.0000 | 0.9333 | 过拟合（完全记忆） |
| 3 | 0.9714 | 0.9778 | 很好 |
| **5** | **0.9619** | **0.9778** | **最佳平衡** |
| 7 | 0.9619 | 0.9556 | 良好 |
| 15 | 0.9238 | 0.9111 | 欠拟合 |

## 7. 懒惰学习 vs 热心学习

### KNN（懒惰学习）的特点

| 特性 | 说明 |
|------|------|
| **训练时间** | 极快（无显式训练） |
| **预测时间** | 慢（需要计算所有距离） |
| **存储** | 需要存储所有训练数据 |
| **适用** | 小到中等数据集 |
| **决策边界** | 复杂、非线性 |
| **可解释性** | 高（查看近邻） |

### 与热心学习（如 SVM、决策树）的对比

| 特性 | KNN | SVM | 决策树 |
|------|-----|-----|--------|
| **训练时间** | 快 | 慢 | 中等 |
| **预测时间** | 慢 | 中等 | 快 |
| **内存占用** | 高 | 低 | 低 |
| **高维性能** | 差 | 好 | 好 |
| **非线性性** | 好 | 好 | 好 |
| **可解释性** | 高 | 低 | 高 |

## 8. 常见问题

### Q1: 如何选择合适的 K 值？

**经验法则**：
- **K 太小**（如 K=1）：过拟合，受噪声影响大
- **K 太大**（如 K=N）：欠拟合，决策边界过于平滑

**推荐方法**：
- 通常 $K = \sqrt{N}$，其中 N 是样本数
- 对于 150 个样本的 Iris，$K \approx \sqrt{150} \approx 12$
- 使用**交叉验证**找最优 K

对于 Iris 数据集，K=3-7 都表现很好。

### Q2: KNN 受欧几里得距离"维度诅咒"影响吗？

**是的！这是 KNN 的主要问题。**

在高维空间中：
- 所有点之间的距离趋于相等
- 最近邻和最远邻的区别变小
- 模型性能下降

**解决方案**：
1. **特征选择**：移除无关特征
2. **特征提取**：PCA 降维
3. **特征归一化**：Z-score 或 Min-Max
4. **改进距离**：局部敏感哈希（LSH）

### Q3: KNN 如何处理分类特征？

本实现假设所有特征都是**连续型**。对于分类特征：

**解决方案**：
1. **独热编码**（One-Hot Encoding）
2. **哈明距离**（Hamming Distance）：计算不同维度的个数

```python
def hamming_distance(x1, x2):
    """哈明距离（用于分类特征）"""
    return sum(1 for i in range(len(x1)) if x1[i] != x2[i])
```

### Q4: 如何处理不平衡的类别分布？

**问题**：如果某类样本很多，多数投票会偏向该类。

**解决方案**：
1. **距离加权投票**：距离近的邻居权重高
   $$\hat{y} = \arg\max_c \sum_{i=1}^{K} w_i \cdot I(y_i = c)$$
   其中 $w_i = 1 / d_i$ 或 $w_i = e^{-d_i}$

2. **过采样**或**欠采样**

### Q5: 特征归一化对 KNN 重要吗？

**非常重要！**

不同特征的值域可能差异很大：
- 花萼长度：4.3-7.9（范围 3.6）
- 花萼宽度：2.0-4.4（范围 2.4）
- 花瓣长度：1.0-6.9（范围 5.9）

归一化后，所有特征的贡献均等。

```bash
python src/KNN.py -f data/iris.data --normalize
```

### Q6: KNN 为什么是"懒惰"学习？

**懒惰学习**意味着：
- 不进行显式的学习过程
- 在**预测时**才进行计算
- 存储所有训练数据

**优点**：
- 训练速度极快
- 可以快速适应新数据

**缺点**：
- 预测速度慢
- 内存占用大

## 9. 实现细节

### KNN 的预测流程

```python
def predict(X):
    predictions = []
    for x in X:
        # 计算到所有训练样本的距离
        distances = [distance(x, x_train) for x_train in X_train]
        
        # 配对 (距离, 标签)
        dist_label_pairs = list(zip(distances, y_train))
        
        # 排序（从小到大）
        dist_label_pairs.sort(key=lambda p: p[0])
        
        # 选择最近的 K 个
        k_nearest_labels = [label for _, label in dist_label_pairs[:k]]
        
        # 多数投票
        from collections import Counter
        label_counts = Counter(k_nearest_labels)
        predicted_label = label_counts.most_common(1)[0][0]
        
        predictions.append(predicted_label)
    
    return predictions
```

### 概率预测

```python
def predict_proba(x):
    """返回属于各类别的概率"""
    # 获取最近的 K 个邻居
    k_nearest_labels = [...]  # 如上
    
    # 计算频率
    label_counts = Counter(k_nearest_labels)
    
    # 转换为概率
    probabilities = {}
    for label in set(y_train):
        count = label_counts.get(label, 0)
        probabilities[label] = count / k
    
    return probabilities
```

## 10. 参考文献

1. Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification"
2. Altman, N. S. (1992). "An introduction to kernel and nearest-neighbor nonparametric regression"
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"

## 11. 文件结构

```
KNN/
├── src/
│   └── KNN.py              # 主要实现
├── data/
│   ├── iris.data           # 数据集（150 样本）
│   └── iris.names          # 数据集说明
└── README.md               # 本文档
```

## 12. 快速开始

### 基础运行
```bash
python src/KNN.py -f data/iris.data -k 5
```

### 实验：K 值的影响

```bash
# K=1（可能过拟合）
python src/KNN.py -f data/iris.data -k 1

# K=3（推荐）
python src/KNN.py -f data/iris.data -k 3

# K=5（平衡）
python src/KNN.py -f data/iris.data -k 5

# K=15（可能欠拟合）
python src/KNN.py -f data/iris.data -k 15
```

### 实验：距离度量的影响

```bash
# 欧几里得距离（最常用）
python src/KNN.py -f data/iris.data -m euclidean

# 曼哈顿距离
python src/KNN.py -f data/iris.data -m manhattan

# 切比雪夫距离
python src/KNN.py -f data/iris.data -m chebyshev
```

---

**创建日期**：2025年11月28日  
**作者**：23计算1Bohan Yu  