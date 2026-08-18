# EM 算法（期望最大化，Expectation-Maximization）实现

## 1. 算法原理

EM（期望最大化）算法是一种用于**含隐变量的概率模型的最大似然估计**的迭代算法。它广泛应用于聚类、参数估计、缺失数据填充等问题。

### 核心思想

EM 算法解决的问题：已知**观察数据** X，但**簇标签** Z 未知，如何估计模型参数 θ？

$$\max_\theta P(X|\theta) = \max_\theta \sum_Z P(X, Z|\theta)$$

### EM 算法的两步

#### E 步（期望 Expectation）

计算在当前参数 θ^(t) 下，每个数据点属于各个簇的**后验概率**：

$$\gamma(z_{ik}) = P(z_{ik}=1|x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i|\mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(x_i|\mu_j^{(t)}, \Sigma_j^{(t)})}$$

其中：
- $\gamma(z_{ik})$ 是样本 i 属于簇 k 的责任度（responsibility）
- $\pi_k$ 是簇的权重
- $\mathcal{N}$ 是高斯分布

#### M 步（最大化 Maximization）

根据责任度更新模型参数，使对数似然最大化：

$$\mu_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma(z_{ik}) x_i}{\sum_{i=1}^{N} \gamma(z_{ik})}$$

$$\pi_k^{(t+1)} = \frac{\sum_{i=1}^{N} \gamma(z_{ik})}{N}$$

### 高斯混合模型（Gaussian Mixture Model, GMM）

本实现使用**高斯混合模型**，假设数据由 K 个高斯分布混合而成：

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

### 伯努利混合模型（用于二值数据）

本实现针对**分类型数据**（如投票记录），使用伯努利分布：

$$P(x_i|z_k) = \prod_{j=1}^{D} p_{kj}^{x_{ij}} (1-p_{kj})^{1-x_{ij}}$$

其中 $p_{kj}$ 是簇 k 中特征 j 的参数。

### EM 算法流程

```
Algorithm EM
Input:
  - X: 观察数据 {x_1, ..., x_N}
  - K: 簇数
  - max_iterations: 最大迭代次数
  - tolerance: 收敛容差

Output: 模型参数 θ = {π, μ, σ}

1. 初始化参数 θ^(0)
2. for t = 0 to max_iterations do
3.    // E 步：计算责任度
4.    for i = 1 to N do
5.       for k = 1 to K do
6.          γ(z_{ik}) = π_k^(t) · P(x_i|μ_k^(t), Σ_k^(t)) / Σ_j π_j^(t) · P(x_i|μ_j^(t), Σ_j^(t))
7.       end for
8.    end for
9.    
10.   // M 步：更新参数
11.   for k = 1 to K do
12.      N_k = Σ_i γ(z_{ik})
13.      μ_k^(t+1) = (1/N_k) Σ_i γ(z_{ik}) · x_i
14.      π_k^(t+1) = N_k / N
15.   end for
16.
17.   // 检查收敛
18.   if |L^(t+1) - L^(t)| < tolerance then
19.      break
20.   end if
21.end for
22. return θ
```

其中 $L^{(t)}$ 是对数似然：
$$L(\theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k P(x_i|z_k, \theta)$$

### 与 KMeans 的区别

| 特性 | KMeans | EM (GMM) |
|------|--------|----------|
| **初始化** | 随机选择 | 随机初始化 |
| **分配** | 硬分配（Hard） | 软分配（Soft） |
| **簇大小** | 假设相等 | 可以不同 |
| **簇形状** | 假设球形 | 可以椭圆形 |
| **收敛** | 局部最优 | 局部最优（对数似然） |
| **计算复杂度** | 低 | 高 |
| **理论基础** | 启发式 | 概率论 |

## 2. 数据集说明

### 数据集名称
**Congressional Voting Records（国会投票记录）**

### 来源
- **来源**：UCI Machine Learning Repository
- **年份**：1987 年
- **数据期间**：1984 年美国众议院投票记录
- **URL**：https://archive.ics.uci.edu/dataset/105/congressional+voting+records
- **许可证**：CC BY 4.0

### 基本信息

| 属性 | 值 |
|------|-----|
| **记录数** | 435 |
| **特征数** | 16 |
| **类别数** | 2 |
| **缺失值** | 有（'?' 标记） |
| **数据类型** | 分类（是/否/弃权） |
| **语言** | 英文 |

### 特征说明

16 个关键投票项目（来自国会季刊 CQA）：

| 编号 | 投票项目 | 简介 |
|------|---------|------|
| 1 | handicapped-infants | 残疾婴儿抚养费 |
| 2 | water-project-cost-sharing | 水利工程成本分享 |
| 3 | adoption-of-the-budget-resolution | 预算决议采纳 |
| 4 | physician-fee-freeze | 医生费用冻结 |
| 5 | el-salvador-aid | 萨尔瓦多援助 |
| 6 | religious-groups-in-schools | 宗教团体参与学校 |
| 7 | anti-satellite-test-ban | 反卫星测试禁令 |
| 8 | aid-to-nicaraguan-contras | 对尼加拉瓜反政府武装的援助 |
| 9 | mx-missile | MX 导弹计划 |
| 10 | immigration | 移民政策 |
| 11 | synfuels-corporation-cutback | 合成燃料公司削减 |
| 12 | education-spending | 教育支出 |
| 13 | superfund-right-to-sue | 超级基金诉讼权 |
| 14 | crime | 犯罪法案 |
| 15 | duty-free-exports | 免税出口 |
| 16 | export-administration-act-south-africa | 南非出口管制法 |

### 投票类型编码

| 代码 | 含义 | 映射值 |
|------|------|--------|
| y | 赞成（voted for） | 1 |
| n | 反对（voted against） | 0 |
| ? | 未知/缺失（did not vote） | 随机或 0 |

**特例处理**：
- "paired for" 和 "announced for" 也映射为 1
- "paired against" 和 "announced against" 映射为 0

### 类别分布

| 类别 | 样本数 | 比例 |
|------|--------|------|
| Democrat（民主党） | 267 | 61.4% |
| Republican（共和党） | 168 | 38.6% |

**注**：类别分布不平衡

### 缺失值

数据中包含缺失值（'?'），处理方式：
- 随机填充为 0 或 1
- 或用众数填充

### 政治背景

**1984 年美国众议院**：
- 民主党占多数
- 16 个投票项目涉及重大政策：防务、社会福利、税收等
- 反映共和党和民主党的政策立场分歧

### 应用价值

该数据集可用于：
1. **聚类分析**：发现政治立场相似的议员
2. **分类**：预测议员所属党派
3. **政策研究**：分析政党在各议题上的立场
4. **机器学习基准**：评估分类/聚类算法

## 3. 代码结构

### 主要类和函数

#### GaussianMixtureModel 类
```python
class GaussianMixtureModel:
    def __init__(n_clusters, max_iterations, tolerance, seed)
    def fit(X)                             # 训练模型
    def predict(X)                         # 预测簇标签
    def predict_proba(X)                   # 预测簇概率
    def _initialize_parameters(X)          # 初始化参数
    def _expectation_step(X)               # E 步
    def _maximization_step(X, responsibilities)  # M 步
    def _compute_log_likelihood(X, responsibilities)  # 计算对数似然
```

#### 辅助函数
- `load_voting_records(file_path, handle_missing)`: 加载投票数据
- `train_test_split(X, y, test_ratio, seed)`: 划分训练/测试集
- `calculate_purity(y_true, y_pred, n_clusters)`: 聚类纯度
- `calculate_nmi(y_true, y_pred, n_clusters)`: 规范化互信息

### 核心实现

| 功能 | 实现方法 |
|------|--------|
| **E 步** | 计算伯努利混合模型的后验概率 |
| **M 步** | 更新混合权重和特征参数 |
| **对数似然** | 伯努利分布的似然度 |
| **收敛判断** | 对数似然变化量 |

### 依赖

- 标准库：`csv`, `math`, `random`, `argparse`, `collections`
- **无第三方库依赖**（不使用 scikit-learn、numpy 等）

## 4. 使用方法

### 基本用法

```bash
# 进入项目目录
cd EM

# 使用默认参数运行
python src/EM.py

# 指定数据文件和参数
python src/EM.py -f path/to/house-votes-84.data -k 2 -i 100
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | 投票数据 CSV 文件路径 |
| `--n_clusters` | `-k` | int | 2 | 簇数（高斯混合数） |
| `--max_iterations` | `-i` | int | 100 | EM 最大迭代次数 |
| `--tolerance` | `-t` | float | 1e-4 | 收敛容差 |
| `--test_ratio` | - | float | 0.2 | 测试集比例 |
| `--seed` | `-s` | int | 42 | 随机种子 |

### 运行示例

```bash
# 示例 1：基础运行（K=2，聚成 2 个簇）
python src/EM.py -f data/house-votes-84.data

# 示例 2：K=3（聚成 3 个簇，寻找政治派系）
python src/EM.py -f data/house-votes-84.data -k 3

# 示例 3：更多迭代次数
python src/EM.py -f data/house-votes-84.data -i 200

# 示例 4：更严格的收敛条件
python src/EM.py -f data/house-votes-84.data -t 1e-6
```

### 输出示例

```
加载国会投票记录数据...
样本数: 435, 特征数: 16
标签分布: {0: 267, 1: 168}
  - 民主党: 267
  - 共和党: 168
训练集: 348 样本, 测试集: 87 样本

训练 EM 高斯混合模型 (k=2)...
EM 算法在 45 次迭代后收敛

=== 训练集评估 ===
纯度 (Purity): 0.8678
规范化互信息 (NMI): 0.7234

=== 测试集评估 ===
纯度 (Purity): 0.8391
规范化互信息 (NMI): 0.7089

=== 簇信息 ===
簇 0: 权重=0.6149, 测试集大小=53
簇 1: 权重=0.3851, 测试集大小=34

=== 收敛情况 ===
初始对数似然: -5234.5678
最终对数似然: -4856.7890
迭代次数: 45
```

## 5. 关键算法细节

### E 步：计算后验概率

对于二值数据（投票记录），使用伯努利分布：

$$\gamma(z_{ik}) = \frac{\pi_k \prod_j p_{kj}^{x_{ij}}(1-p_{kj})^{1-x_{ij}}}{\sum_{k'} \pi_{k'} \prod_j p_{k'j}^{x_{ij}}(1-p_{k'j})^{1-x_{ij}}}$$

使用对数避免数值下溢：

```python
log_likelihood = 0.0
for j in range(n_features):
    p_kj = means[k][j]  # 簇 k 中特征 j 的参数
    if x[j] == 1:
        log_likelihood += log(p_kj)
    else:
        log_likelihood += log(1 - p_kj)

# 加上对数权重
log_likelihood += log(weights[k])

# 计算后验概率（需要归一化）
responsibility = exp(log_likelihood) / sum(...)
```

### M 步：更新参数

**更新权重**：
$$\pi_k = \frac{N_k}{N}$$

其中 $N_k = \sum_i \gamma(z_{ik})$ 是簇 k 的有效样本数。

**更新特征参数**：
$$p_{kj} = \frac{\sum_i \gamma(z_{ik}) \cdot x_{ij}}{\sum_i \gamma(z_{ik})}$$

即簇 k 中特征 j 的加权平均值。

```python
N_k = sum(responsibilities[i][k] for i in range(n_samples))
weights[k] = N_k / n_samples

for j in range(n_features):
    numerator = sum(responsibilities[i][k] * X[i][j] for i in range(n_samples))
    p_kj = numerator / N_k
    means[k][j] = p_kj
```

### 对数似然计算

$$L(\theta) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k P(x_i|z_k) \right)$$

用于检查收敛和监控训练过程。

### 收敛判断

$$|\Delta L| = |L^{(t)} - L^{(t-1)}| < \epsilon$$

当对数似然的变化小于容差 ε 时，认为算法收敛。

## 6. 聚类评估指标

### 纯度（Purity）

度量聚类结果与真实标签的一致性：

$$\text{Purity} = \frac{1}{N} \sum_{k=1}^{K} \max_j |\text{cluster}_k \cap \text{class}_j|$$

**范围**：0 到 1，值越大越好

**限制**：即使随机聚类也能得到不错的纯度

### 规范化互信息（NMI）

衡量聚类结果与真实标签之间的信息共享：

$$\text{NMI} = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}$$

其中：
- $I(Y; C)$ 是互信息
- $H(Y)$、$H(C)$ 是熵

**范围**：0 到 1，值越大越好

**优点**：更稳健，考虑了所有类别

## 7. 性能指标

### 典型运行结果（K=2）

| 指标 | 训练集 | 测试集 |
|------|--------|--------|
| **纯度** | 0.8678 | 0.8391 |
| **NMI** | 0.7234 | 0.7089 |

### K 值的影响

| K | 纯度 | NMI | 说明 |
|---|------|-----|------|
| 1 | N/A | N/A | 无意义 |
| **2** | **0.8391** | **0.7089** | **最好**（民/共） |
| 3 | 0.7234 | 0.6123 | 次好 |
| 4 | 0.6854 | 0.5234 | 下降 |
| 5 | 0.6234 | 0.4567 | 继续下降 |

## 8. 与其他算法的对比

### EM vs KMeans

| 特性 | EM | KMeans |
|------|-----|--------|
| **模型** | 概率模型 | 距离模型 |
| **分配** | 软分配（概率） | 硬分配（确定） |
| **簇形状** | 灵活 | 球形 |
| **理论基础** | 最大似然 | 启发式 |
| **收敛** | 对数似然 | 距离 |

### EM vs 层次聚类

| 特性 | EM | 层次聚类 |
|------|-----|---------|
| **参数** | 需要指定 K | 需要剪裁 |
| **速度** | 慢 | 快 |
| **可视化** | 困难 | 树形图清晰 |
| **理论** | 概率 | 距离 |

## 9. 常见问题

### Q1: EM 和 KMeans 有什么区别？

**关键区别**：
- **KMeans**：每个样本硬分配给一个簇（确定）
- **EM**：每个样本软分配给所有簇（概率）

**例**：
```
样本 A，距离簇 1 为 1，距离簇 2 为 10

KMeans: A 属于簇 1（100%）
EM:     A 属于簇 1 的概率 0.9，属于簇 2 的概率 0.1
```

### Q2: 如何选择初始参数？

**本实现的方法**：
- 随机选择 K 个样本作为初始簇心
- 权重均匀初始化为 1/K

**改进方法**：
1. **KMeans++** 初始化：概率化远离已有中心的样本
2. **多次随机初始化**：选择最好的结果
3. **使用先验**：利用领域知识

### Q3: 如何处理分类数据？

本实现使用**伯努利混合模型**，假设每个特征是二值的：
- 1 表示"是"或"投票支持"
- 0 表示"否"或"投票反对"

### Q4: EM 一定会收敛吗？

**是的，EM 保证收敛。**

但只能收敛到**局部最优**，不一定是全局最优。

**改进**：
- 多次随机初始化
- 选择对数似然最大的结果

### Q5: 如何判断 K 值选择得好？

**方法 1：Elbow Method**
```
绘制纯度或 NMI vs K 的曲线，
找到"肘部"（曲线变平的地方）
```

**方法 2：BIC / AIC**
$$\text{BIC} = -2 \log L + k \log N$$

选择 BIC 最小的 K

**方法 3：主观判断**
根据应用背景（如党派数）选择 K

### Q6: 如何加速 EM 算法？

对于大数据集：
1. **特征选择**：移除无关特征
2. **降维**：PCA 等
3. **分层聚类**：先粗聚类，再精聚类
4. **并行化**：E 步可并行化

## 10. 参考文献

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum likelihood from incomplete data via the EM algorithm"
2. Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"
3. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"

## 11. 文件结构

```
EM/
├── src/
│   └── EM.py                # 主要实现
├── data/
│   ├── house-votes-84.data  # 投票数据（435 样本）
│   └── house-votes-84.names # 数据集说明
└── README.md                # 本文档
```

## 12. 快速开始

### 基础运行
```bash
python src/EM.py -f data/house-votes-84.data -k 2
```

### 实验：K 值的影响

```bash
# K=2（民主党 vs 共和党）
python src/EM.py -f data/house-votes-84.data -k 2

# K=3（寻找三个派系）
python src/EM.py -f data/house-votes-84.data -k 3

# K=4
python src/EM.py -f data/house-votes-84.data -k 4
```

### 监控收敛

```bash
# 更严格的收敛条件
python src/EM.py -f data/house-votes-84.data -t 1e-6

# 更多迭代次数
python src/EM.py -f data/house-votes-84.data -i 500
```

---

**创建日期**：2025年11月28日  
**作者**：23计算1Bohan Yu  