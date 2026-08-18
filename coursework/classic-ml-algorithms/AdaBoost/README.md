# AdaBoost 算法实现

## 1. 算法原理

AdaBoost（Adaptive Boosting）是一种集成学习算法，通过迭代训练多个弱分类器，根据分类错误重新分配样本权重，最终将所有弱分类器加权组合成强分类器。

### 核心思想

1. **初始化权重**：所有样本的权重相等，$w_i = \frac{1}{N}$
2. **迭代训练**：
   - 在第 $t$ 次迭代中，使用当前权重分布训练一个弱分类器 $h_t$
   - 计算弱分类器的加权错误率：$\epsilon_t = \sum_{i=1}^{N} w_i \cdot \mathbb{1}(h_t(x_i) \neq y_i)$
   - 计算分类器权重：$\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$
   - 更新样本权重：$w_i \leftarrow w_i \cdot \exp(-\alpha_t \cdot y_i \cdot h_t(x_i))$
   - 归一化权重：$w_i \leftarrow \frac{w_i}{\sum_j w_j}$
3. **最终预测**：$f(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$

### 特点

- **自适应性**：权重自动分配给难分类的样本
- **弱学习者**：基学习器（如决策桩）准确率略优于随机猜测即可
- **指数加权**：错误分类样本权重呈指数增长
- **二分类**：标签为 $\{+1, -1\}$

### 伪代码

```
AdaBoost(训练集 S, 弱学习算法 WeakLearner, 迭代数 T)
  初始化权重: w_i = 1/N, i=1,...,N
  
  for t = 1 to T do
    训练弱分类器: h_t = WeakLearner(S, w)
    计算加权错误率: ε_t = Σ w_i * I(h_t(x_i) ≠ y_i)
    
    if ε_t > 0.5 or ε_t <= 0 then
      break  // 停止迭代
    end if
    
    计算分类器权重: α_t = 0.5 * ln((1 - ε_t) / ε_t)
    
    for i = 1 to N do
      更新权重: w_i = w_i * exp(-α_t * y_i * h_t(x_i))
    end for
    
    归一化权重: w_i = w_i / Σ_j w_j
  end for
  
  最终分类器: f(x) = sign(Σ_t α_t * h_t(x))
  return f
```

## 2. 数据集说明

### 数据集名称
**Magic Gamma Telescope Dataset**

### 来源
- **URL**: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
- **UCI Machine Learning Repository**
- 天文学应用：区分真实伽马射线事件与背景噪声

### 基本信息

| 属性 | 值 |
|------|-----|
| **样本数** | 19,020 |
| **特征数** | 10 |
| **缺失值** | 无 |
| **类别** | 2（伽马/强子） |
| **类别分布** | 不均衡（约 64% g, 36% h） |

### 特征说明

数据来自 **MAGIC（Major Atmospheric Gamma Imaging Cherenkov）** 望远镜，用于检测伽马射线：

| 索引 | 特征名 | 说明 |
|------|--------|------|
| 1 | Length | 主轴长度 |
| 2 | Width | 次轴长度 |
| 3 | Size | 像素总数 |
| 4 | Concentration | 核心像素比例 |
| 5 | Asymmetry | 不对称参数 |
| 6 | Conc1 | 1st Hillas moment |
| 7 | Conc2 | 2nd Hillas moment |
| 8 | Conc3 | 3rd Hillas moment |
| 9 | M3Long | 3rd moment along主轴 |
| 10 | M3Trans | 3rd moment along次轴 |

### 目标变量

- **g** (Gamma): 真实伽马射线事件 → 标签 +1
- **h** (Hadron): 强子背景噪声 → 标签 -1

### 应用背景

MAGIC 是位于加那利群岛的大气切伦可夫伽马射线望远镜，通过机器学习分类高能物理事件，具有重要的天文学价值。

## 3. 代码结构

### 主要类和函数

#### `DecisionStump` 类
弱分类器：基于单个特征的阈值分裂决策桩

**方法**：
- `fit(X, y, weights)`: 训练决策桩，选择最优特征和阈值
- `predict(X)`: 对样本进行预测

#### `AdaBoost` 类
集成分类器：组合多个决策桩

**方法**：
- `__init__(n_clf)`: 初始化，指定弱分类器数量
- `fit(X_train, y_train)`: 训练 AdaBoost 模型
- `predict(X_test)`: 对测试集进行预测

#### 辅助函数
- `load_magic_data(file_path)`: 加载 Magic 数据集
- `accuracy_score(y_true, y_pred)`: 计算准确率
- `demo_on_toy()`: 玩具示例演示

### 依赖

- 标准库：`csv`, `math`, `random`, `argparse`, `os`
- **无第三方库依赖**（不使用 scikit-learn 等）

## 4. 使用方法

### 基本用法

```bash
# 默认参数运行（查找默认路径）
python AdaBoost.py

# 指定数据文件
python AdaBoost.py -f path/to/magic04.data

# 自定义参数
python AdaBoost.py -f magic04.data -n 30 -t 0.25 -s 123
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | Magic 数据 CSV 文件路径 |
| `--n_estimators` | `-n` | int | 50 | 弱分类器（决策桩）数量 |
| `--test_size` | `-t` | float | 0.3 | 测试集比例（0-1） |
| `--seed` | `-s` | int | 42 | 随机种子（复现结果） |

### 输出示例

```
加载数据: 样本数=19020, 特征数=10, 训练=13314, 测试=5706
AdaBoost 测试准确率: 0.7762 (n_estimators=10)
```

## 5. 关键优化

### 阈值采样策略

由于 Magic 数据集有 19,020 个样本和 10 个特征，原始的"所有中点阈值"枚举会导致训练极慢。

**优化方案**：
```python
# 而不是尝试每对唯一值之间的所有中点，采样均匀分布的阈值
n_thresholds = 15  # 每个特征仅采样 15 个候选阈值
min_val = min(feature_values)
max_val = max(feature_values)
thresholds = [min_val + (max_val - min_val) * i / (n_thresholds - 1) 
              for i in range(n_thresholds)]
```

**效果**：
- 原始方案：无法在合理时间内完成
- 优化方案：n=10 时约 30 秒，n=30 时约 5 分钟

## 6. 性能指标

### 典型结果（n_estimators=10）

在 Magic 数据集上，随机 70/30 划分：

- **训练集大小**: 13,314
- **测试集大小**: 5,706
- **测试准确率**: ~77.6%

### 与完整实现的对比

| 指标 | 优化版 (n=15 阈值) | 完整版 (全中点) |
|------|------------------|----------------|
| n=10 训练时间 | ~30s | 无法完成 |
| n=10 准确率 | 77.6% | 预期 ~80% |
| 代码复杂度 | 低 | 高 |
| 适合场景 | 大数据集 | 小数据集 |

## 7. 实现细节

### DecisionStump 的工作流程

```python
1. 对每个特征采样候选阈值（均匀分布）
2. 对每个阈值和 polarity 组合：
   a. 计算 x[feature] < threshold 的预测
   b. 计算加权误差（考虑当前样本权重）
   c. 保存最低误差的分裂
3. 返回最优的特征、阈值和 polarity
```

### 权重更新的指数性

```python
# 错误分类的样本权重呈指数增长
w_i *= exp(-alpha * y_i * pred_i)

# 结果：
# - 如果 y_i == pred_i（正确分类）：权重减小
# - 如果 y_i != pred_i（错误分类）：权重增大
```

这使得后续分类器专注于前面分类器出错的样本。

## 8. 与 sklearn 的对比

```python
# sklearn 版本
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0
)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

**本实现与 sklearn 的主要区别**：

| 特性 | 本实现 | sklearn |
|------|------|---------|
| 基学习器 | 简单决策桩 | 可自定义 |
| 阈值选择 | 均匀采样 | 全枚举 |
| 代码依赖 | 纯 Python | NumPy, SciPy |
| 速度 | 较慢 | 快（C 实现） |
| 教学价值 | 高 | 低 |
| 生产环境 | 不推荐 | 推荐 |

## 9. 常见问题

### Q1: 为什么要用 -1/+1 而不是 0/1？
**A**: AdaBoost 的权重更新公式中，错误分类项 $y_i \neq h_t(x_i)$ 需要转换为实数乘法。使用 ±1 标签时，$-y_i \cdot h_t(x_i) = +1$ 当错误，更方便计算。

### Q2: polarity 是什么？
**A**: DecisionStump 计算 $\text{sign}(x_{\text{feature}} < \text{threshold}) \times \text{polarity}$。polarity ∈ {-1, +1} 允许翻转分裂方向，提高灵活性。

### Q3: 为什么准确率不到 80%？
**A**: Magic 数据集本身具有天然的类别重叠，完全分离困难。同时，决策桩是非常弱的学习器。使用更多迭代 (-n 50) 或更好的弱学习器可以改进。

### Q4: 如何加速训练？
**A**: 
- 减少 n_estimators (-n 10)
- 减少候选阈值数量（修改代码中的 `n_thresholds`）
- 使用子样本训练（未实现）

## 10. 参考文献

1. Freund, Y., & Schapire, R. E. (1997). "A decision-theoretic generalization of on-line learning and an application to boosting"
2. Schapire, R. E. (2013). "Explaining AdaBoost"
3. UCI Machine Learning Repository - Magic Gamma Telescope: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

## 11. 文件结构

```
AdaBoost/
├── src/
│   └── AdaBoost.py          # 主要实现
├── data/
│   ├── magic04.data         # 数据集文件
│   └── magic04.names        # 数据集说明
└── README.md                # 本文档
```

## 12. 快速开始

```bash
# 进入项目目录
cd AdaBoost

# 运行默认配置（查找 data/magic04.data）
python src/AdaBoost.py

# 自定义参数
python src/AdaBoost.py -n 20 -t 0.2 -s 999
```

预期输出：
```
加载数据: 样本数=19020, 特征数=10, 训练=13314, 测试=5706
AdaBoost 测试准确率: 0.xxxx (n_estimators=20)
```

---

**创建日期**: 2025年11月28日  
**作者**: 23计算1Bohan Yu  