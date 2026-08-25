# CART 决策树（Classification And Regression Tree）实现

## 1. 算法原理

CART（分类回归树）是由 Breiman 等人在 1984 年提出的二叉树算法。它既可以用于分类问题（CART 分类树），也可以用于回归问题（CART 回归树）。本实现为**分类树**。

### 核心思想

CART 是一种**递归二分裂**算法：
- 每次分裂都产生两个子节点（二叉树）
- 每次选择能最小化子节点不纯度的特征和阈值
- 重复这个过程，直到满足停止条件

### 分裂准则：基尼指数（Gini Index）

**基尼指数**衡量节点的不纯度：

$$\text{Gini}(D) = 1 - \sum_{i=1}^{K} p_i^2$$

其中：
- **$p_i$**：类别 $i$ 的样本比例
- **$K$**：类别数量

**性质**：
- 当所有样本属于同一类时，$\text{Gini} = 0$（纯节点）
- 当类别分布均匀时，$\text{Gini}$ 最大

### 基尼增益（Gini Gain）

对于特征 $a$ 和阈值 $t$，分裂后的基尼增益为：

$$\Delta\text{Gini}(D, a, t) = \text{Gini}(D) - \frac{|D_L|}{|D|}\text{Gini}(D_L) - \frac{|D_R|}{|D|}\text{Gini}(D_R)$$

其中：
- **$D_L$**：满足 $a \leq t$ 的样本集合
- **$D_R$**：满足 $a > t$ 的样本集合

**选择规则**：选择使基尼增益最大的特征和阈值

### CART 树构建算法

```
CART_BUILD(数据集 D, 深度 depth)
  // 停止条件
  if depth >= max_depth or |D| < min_samples_split or 所有样本同一类 then
    return 叶子节点(最多类别)
  end if
  
  best_gain = -1
  best_feature = None
  best_threshold = None
  
  // 遍历所有特征
  for 每个特征 a do
    // 生成候选阈值（特征值的中点）
    for 特征值中点 t do
      // 计算基尼增益
      gain = 计算基尼增益(D, a, t)
      
      if gain > best_gain then
        best_gain = gain
        best_feature = a
        best_threshold = t
      end if
    end for
  end for
  
  // 如果未找到有效分裂
  if best_gain == -1 then
    return 叶子节点(最多类别)
  end if
  
  // 分裂样本集合
  D_L = {(x, y) ∈ D : x[best_feature] <= best_threshold}
  D_R = {(x, y) ∈ D : x[best_feature] > best_threshold}
  
  // 递归构建子树
  left_subtree = CART_BUILD(D_L, depth + 1)
  right_subtree = CART_BUILD(D_R, depth + 1)
  
  return 内部节点(best_feature, best_threshold, left_subtree, right_subtree)
```

### 与 ID3/C4.5 的对比

| 特性 | CART | C4.5 | ID3 |
|------|------|------|-----|
| **树类型** | 二叉树 | 多叉树 | 多叉树 |
| **分裂准则** | 基尼指数 | 信息增益率 | 信息增益 |
| **处理连续特征** | ✓ 二分裂 | ✓ 二分裂 | ✗ 不支持 |
| **处理缺失值** | ✓ 代理分裂 | ✓ 特殊处理 | ✗ 不支持 |
| **后剪枝** | ✓ 成本复杂度 | ✓ 错误率 | ✗ |
| **适用范围** | 分类+回归 | 分类+回归 | 仅分类 |

## 2. 数据集说明

### 数据集名称
**Wine Quality Dataset（葡萄酒质量数据集）**

### 来源
- **来源**：UCI Machine Learning Repository
- **创建者**：Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis（2009）
- **年份**：2009 年
- **URL**：https://archive.ics.uci.edu/dataset/186/wine+quality
- **许可证**：CC BY 4.0

### 基本信息

| 属性 | 值 |
|------|-----|
| **数据集类型** | 两个数据集（红葡萄酒 + 白葡萄酒）|
| **红葡萄酒样本数** | 1,599 |
| **白葡萄酒样本数** | 4,898 |
| **总样本数** | 6,497 |
| **特征数** | 11 |
| **目标变量** | 质量评分（0-10） |
| **缺失值** | 无 |
| **语言** | 英文 |

### 特征说明

所有特征都是**物理化学测试**的结果：

| 特征编号 | 特征名称 | 单位 | 说明 |
|---------|--------|------|------|
| 1 | fixed acidity | g/dm³ | 固定酸度（主要是酒石酸） |
| 2 | volatile acidity | g/dm³ | 挥发性酸度（醋酸） |
| 3 | citric acid | g/dm³ | 柠檬酸含量 |
| 4 | residual sugar | g/dm³ | 残留糖分 |
| 5 | chlorides | g/dm³ | 氯化物（盐分） |
| 6 | free sulfur dioxide | mg/dm³ | 游离二氧化硫 |
| 7 | total sulfur dioxide | mg/dm³ | 总二氧化硫 |
| 8 | density | g/cm³ | 密度 |
| 9 | pH | - | 酸碱度 |
| 10 | sulphates | g/dm³ | 硫酸盐 |
| 11 | alcohol | % vol | 酒精度 |
| **12** | **quality** | **0-10** | **感官质量评分（目标）** |

### 质量评分分布

**红葡萄酒**（不平衡）：

| 评分 | 样本数 | 比例 |
|------|--------|------|
| 3 | 10 | 0.63% |
| 4 | 53 | 3.31% |
| 5 | 681 | 42.59% |
| **6** | **638** | **39.90%** |
| 7 | 199 | 12.44% |
| 8 | 18 | 1.13% |
| 总计 | 1,599 | 100% |

**特点**：
- 大多数样本评分为 5 或 6（"中等"质量）
- 低分（3-4）和高分（7-8）样本很少（类别不平衡）
- 评分呈近似正态分布，中心在 5-6

### 应用背景

**研究背景**：
- 来自葡萄牙北部的"绿葡萄酒"（Vinho Verde）
- 由于隐私和物流原因，数据集仅包含物理化学特征和感官评分
- 不包含葡萄品种、品牌、价格等信息

**任务类型**：
- 可作为回归问题（预测评分值）
- 可作为分类问题（预测评分等级）
- 本实现采用分类方法

### 数据文件

```
winequality-red.csv    (84.2 KB)   红葡萄酒，1599 样本
winequality-white.csv  (258.2 KB)  白葡萄酒，4898 样本
winequality.names      (3.3 KB)    数据集文档说明
```

## 3. 代码结构

### 主要类和函数

#### Node 类
```python
class Node:
    feature        # 分裂特征索引
    threshold      # 分裂阈值
    left           # 左子树
    right          # 右子树
    value          # 叶子节点预测值
```

#### CARTTree 类
```python
class CARTTree:
    def __init__(max_depth, min_samples_split, min_samples_leaf)
    def fit(X, y, feature_names)              # 训练树
    def predict(X)                            # 预测
    def gini_impurity(y)                      # 计算基尼指数
    def gini_gain(y, left, right)             # 计算基尼增益
    def best_split(X, y)                      # 寻找最优分裂
    def build_tree(X, y, depth)               # 递归构建树
    def print_tree(node, depth, prefix)       # 打印树结构
```

#### 辅助函数
- `load_wine_quality_data(file_path)`: 加载 CSV 数据集
- `train_test_split(X, y, test_ratio, seed)`: 划分训练/测试集
- `evaluate_classification(y_true, y_pred)`: 计算分类指标

### 核心实现

| 功能 | 实现方法 |
|------|--------|
| **基尼指数计算** | Gini = 1 - Σ(p_i²) |
| **基尼增益** | ΔGini = Gini(parent) - weighted_Gini(children) |
| **候选阈值** | 特征值的中点 |
| **最优分裂搜索** | 遍历所有特征和阈值 |
| **递归构建** | 深度优先 |
| **预测** | 沿树路径到叶子节点 |

### 依赖

- 标准库：`csv`, `math`, `random`, `argparse`, `collections`
- **无第三方库依赖**（不使用 scikit-learn、numpy 等）

## 4. 使用方法

### 基本用法

```bash
# 进入项目目录
cd CART

# 使用默认参数运行
python src/CART.py

# 指定数据文件和参数
python src/CART.py -f path/to/winequality-red.csv -d 10 -m 2 -t 0.2 -s 42

# 打印树结构
python src/CART.py -f path/to/winequality-red.csv --print_tree
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | 葡萄酒数据 CSV 文件路径 |
| `--max_depth` | `-d` | int | 10 | 决策树最大深度 |
| `--min_samples_split` | `-m` | int | 2 | 分裂所需最少样本数 |
| `--test_ratio` | `-t` | float | 0.2 | 测试集比例 |
| `--seed` | `-s` | int | 42 | 随机种子（用于划分） |
| `--print_tree` | - | flag | False | 是否打印树结构 |

### 运行示例

```bash
# 示例 1：基础运行（红葡萄酒）
python src/CART.py -f data/winequality-red.csv

# 示例 2：深度为 8（欠拟合）
python src/CART.py -f data/winequality-red.csv -d 8

# 示例 3：深度为 15（过拟合）
python src/CART.py -f data/winequality-red.csv -d 15

# 示例 4：打印树结构（深度 5）
python src/CART.py -f data/winequality-red.csv -d 5 --print_tree

# 示例 5：白葡萄酒数据
python src/CART.py -f data/winequality-white.csv -d 10
```

### 输出示例

```
加载葡萄酒质量数据...
样本数: 1599, 特征数: 11, 标签类数: 6
质量评分分布: {3: 10, 4: 53, 5: 681, 6: 638, 7: 199, 8: 18}
训练集: 1279 样本, 测试集: 320 样本

训练 CART 决策树 (max_depth=10)...
评估模型...

=== 训练集评估 ===
准确率: 0.8944

=== 测试集评估 ===
准确率: 0.6062

=== 按类别评估（测试集）===
质量 3: 精确率=0.0000, 召回率=0.0000, F1=0.0000
质量 4: 精确率=0.0000, 召回率=0.0000, F1=0.0000
质量 5: 精确率=0.7209, 召回率=0.6739, F1=0.6966
质量 6: 精确率=0.5682, 召回率=0.5859, F1=0.5769
质量 7: 精确率=0.5417, 召回率=0.5652, F1=0.5532
质量 8: 精确率=0.0000, 召回率=0.0000, F1=0.0000
```

## 5. 关键算法细节

### 基尼指数计算

```python
def gini_impurity(y):
    """Gini = 1 - Σ(p_i²)"""
    counter = Counter(y)
    impurity = 1.0
    for count in counter.values():
        prob = count / len(y)
        impurity -= prob * prob
    return impurity

# 示例：2 类均衡分布
# y = [0, 0, 1, 1]
# p(0) = 0.5, p(1) = 0.5
# Gini = 1 - 0.5² - 0.5² = 0.5（最不纯）

# 示例：纯节点
# y = [0, 0, 0, 0]
# p(0) = 1, p(1) = 0
# Gini = 1 - 1² - 0² = 0（最纯）
```

### 基尼增益计算

```python
def gini_gain(y, left, right):
    n = len(y)
    n_left = len(left)
    n_right = len(right)
    
    # 父节点基尼
    gini_parent = gini_impurity(y)
    
    # 子节点加权基尼
    gini_left = gini_impurity(left)
    gini_right = gini_impurity(right)
    gini_children = (n_left/n)*gini_left + (n_right/n)*gini_right
    
    # 增益 = 父 - 子
    return gini_parent - gini_children
```

### 候选阈值生成

```python
# 方法：特征值的中点
unique_values = sorted(set(feature_values))
thresholds = []
for i in range(len(unique_values) - 1):
    threshold = (unique_values[i] + unique_values[i+1]) / 2
    thresholds.append(threshold)

# 示例：特征值 [3.0, 5.2, 7.1, 9.0]
# 阈值为：[4.1, 6.15, 8.05]
```

### 最优分裂搜索

```
for 每个特征 a:
  for 每个候选阈值 t:
    分裂样本集合为 D_L 和 D_R
    计算基尼增益
    如果增益最大，记录分裂方案
    
选择增益最大的分裂方案
```

### 递归构建

```python
def build_tree(X, y, depth=0):
    # 停止条件
    if 纯节点 or 深度过深 or 样本太少:
        return 叶子节点
    
    # 寻找最优分裂
    best_split = best_split(X, y)
    
    if 无有效分裂:
        return 叶子节点
    
    # 分裂数据
    left_X, left_y = 左子集
    right_X, right_y = 右子集
    
    # 递归构建
    left_tree = build_tree(left_X, left_y, depth+1)
    right_tree = build_tree(right_X, right_y, depth+1)
    
    return 内部节点
```

## 6. 性能指标

### 典型运行结果（红葡萄酒，max_depth=10）

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| **准确率** | 0.8944 | 0.6062 |

**按类别性能**（测试集）：

| 质量 | 精确率 | 召回率 | F1 分数 |
|------|--------|--------|---------|
| 3 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.7209 | 0.6739 | 0.6966 |
| 6 | 0.5682 | 0.5859 | 0.5769 |
| 7 | 0.5417 | 0.5652 | 0.5532 |
| 8 | 0.0000 | 0.0000 | 0.0000 |

### 性能分析

**观察**：
1. **训练集 vs 测试集**：训练准确率 89.44%，测试准确率 60.62%，存在明显过拟合
2. **少数类性能差**：质量 3、4、8 的样本太少，模型未能正确识别
3. **主要类性能**：对多数类（5、6、7）有一定识别能力
4. **类别不平衡问题**：数据集中样本分布不均

### 改进方向

1. **处理类别不平衡**
   - 过采样少数类（SMOTE）
   - 欠采样多数类
   - 调整类别权重

2. **超参调优**
   - 减小 max_depth（减少过拟合）
   - 增加 min_samples_split
   - 使用后剪枝

3. **特征工程**
   - 特征选择（移除不相关特征）
   - 特征归一化
   - 特征交互

## 7. 与 sklearn 的对比

```python
# sklearn 实现
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=10, criterion='gini')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

**本实现与 sklearn 的区别**：

| 特性 | 本实现 | sklearn |
|------|------|---------|
| 代码依赖 | 纯 Python | NumPy/Cython |
| 实现复杂度 | 简单易懂 | 优化复杂 |
| 性能 | 慢（教学） | 快（生产） |
| 特征处理 | 手动 | 自动 |
| 后剪枝 | 无 | 有 |
| 可视化 | 无 | graphviz |
| 教学价值 | 高 | 低 |
| 生产环境 | 不推荐 | 推荐 |

## 8. 常见问题

### Q1: 基尼指数和熵的区别？

**基尼指数**（本实现）：
$$\text{Gini}(D) = 1 - \sum_{i=1}^{K} p_i^2$$

**熵**（ID3/C4.5 使用）：
$$\text{Entropy}(D) = -\sum_{i=1}^{K} p_i \log_2 p_i$$

**对比**：

| 特性 | 基尼指数 | 熵 |
|------|---------|-----|
| 计算速度 | 快（平方） | 慢（对数） |
| 分裂倾向 | 平衡分裂 | 可能不平衡 |
| 结果相似 | 通常类似 | - |
| 应用 | CART | C4.5/ID3 |

**实验结果**：在大多数数据集上，基尼指数和熵产生的树性能相似。

### Q2: 为什么训练准确率远高于测试准确率？

**原因**：过拟合

树的深度太大，导致：
- 在训练数据上拟合得很好（89.44%）
- 在新数据上泛化能力差（60.62%）

**解决方案**：
```bash
# 减小树的深度
python src/CART.py -f data/winequality-red.csv -d 5

# 增加分裂所需样本数
python src/CART.py -f data/winequality-red.csv -m 10
```

### Q3: 为什么某些类别（3、4、8）的精确率为 0？

**原因**：
- 这些类别样本数很少（3 类仅 10 个，8 类仅 18 个）
- 训练集中可能完全没有这些类别的样本
- 模型学不到这些稀有类别的特征

**解决方案**：
1. **数据平衡**：过采样少数类
2. **阈值调整**：降低分裂的最小样本数
3. **重新标签化**：将质量等级合并（如 3-4 为"差"，5-6 为"中"，7-8 为"好"）

### Q4: 特征重要性如何计算？

**特征重要性** = 该特征在所有分裂中产生的总基尼增益

```python
def feature_importance(tree, feature_idx):
    """计算特征重要性"""
    def traverse(node, importance):
        if node.value is not None:
            return
        if node.feature == feature_idx:
            importance += node.gini_gain  # 该分裂的增益
        traverse(node.left, importance)
        traverse(node.right, importance)
    
    importance = [0]
    traverse(tree, importance)
    return importance[0]
```

### Q5: 如何处理缺失值？

CART 的**代理分裂方法**（未在本实现中实现）：
1. 对于主分裂特征，找到最好的替代特征
2. 当主特征缺失时，使用代理特征

**本实现的简单方法**：
- 在加载数据时移除含缺失值的样本
- 或用中位数/众数填充

### Q6: 如何选择最优的 max_depth？

**经验法则**：

| max_depth | 结果 |
|-----------|------|
| 1-3 | 欠拟合（太简单） |
| **5-10** | **适中（推荐）** |
| 15-20 | 过拟合（太复杂） |

**方法**：使用**交叉验证**找最优深度：

```python
for depth in range(1, 20):
    tree = CARTTree(max_depth=depth)
    tree.fit(X_train, y_train)
    cv_score = 交叉验证评分(tree, X_val, y_val)
    记录(depth, cv_score)
```

## 9. 实现细节

### 数据加载

```python
def load_wine_quality_data(file_path):
    """加载 CSV 数据集
    
    格式：分号分隔，最后一列为标签
    """
    X, y = [], []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # 跳过头部
        for row in reader:
            features = [float(val) for val in row[:-1]]
            label = int(float(row[-1]))
            X.append(features)
            y.append(label)
    return X, y, header[:-1]
```

### 模型评估

```python
def evaluate_classification(y_true, y_pred):
    """计算分类指标"""
    # 准确率
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # 混淆矩阵
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    
    # 精确率、召回率、F1
    for cls in set(y_true):
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[other][cls] for other in ... if other != cls)
        fn = sum(confusion_matrix[cls][other] for other in ...)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```

## 10. 参考文献

1. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). "Classification and Regression Trees"
2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). "Modeling wine preferences by data mining from physicochemical properties"
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"

## 11. 文件结构

```
CART/
├── src/
│   └── CART.py                    # 主要实现
├── data/
│   ├── winequality-red.csv        # 红葡萄酒（1599 样本）
│   ├── winequality-white.csv      # 白葡萄酒（4898 样本）
│   └── winequality.names          # 数据集说明
└── README.md                      # 本文档
```

## 12. 快速开始

### 基础运行
```bash
python src/CART.py -f data/winequality-red.csv -d 10
```

### 实验：深度对性能的影响

```bash
# 浅树（欠拟合）
python src/CART.py -f data/winequality-red.csv -d 3

# 中等树（推荐）
python src/CART.py -f data/winequality-red.csv -d 10

# 深树（过拟合）
python src/CART.py -f data/winequality-red.csv -d 20
```

### 尝试白葡萄酒数据
```bash
python src/CART.py -f data/winequality-white.csv -d 10
```

---

**创建日期**：2025年11月28日  
**作者**：23计算1Bohan Yu  

