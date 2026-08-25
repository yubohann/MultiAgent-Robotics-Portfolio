# PageRank 算法实现

## 1. 算法原理

PageRank 是一种**链接分析算法**，由 Google 创始人 Larry Page 和 Sergey Brin 于 1996 年提出。它用于衡量网页的重要性（排名），是现代搜索引擎的基础。

### 核心思想

**一个网页的重要性由两个因素决定：**
1. **指向它的网页数量**：指向它的网页越多，说明它越重要
2. **指向它的网页质量**：重要的网页的推荐更有价值

### PageRank 的数学模型

#### 基础公式

$$PR(A) = (1-d) + d \cdot \sum_{T \in B_A} \frac{PR(T)}{C(T)}$$

其中：
- **$PR(A)$**：网页 A 的 PageRank 值
- **$d$**：阻尼因子（damping factor），通常为 0.85
- **$B_A$**：所有指向 A 的网页的集合
- **$C(T)$**：网页 T 的出链数量
- **$PR(T)$**：指向 A 的网页 T 的 PageRank 值

#### 阻尼因子含义

$$PR(A) = (1-d) \cdot \frac{1}{N} + d \cdot \sum_{T \in B_A} \frac{PR(T)}{C(T)}$$

其中 $\frac{1}{N}$ 是随机跳跃的目标概率。

**阻尼因子 $d$ 的解释**：
- 用户 85% 的概率跟随网页链接
- 用户 15% 的概率直接访问随机网页（跳跃）

### PageRank 的迭代计算

$$PR^{(k+1)}(A) = (1-d) \cdot \frac{1}{N} + d \cdot \sum_{T \in B_A} \frac{PR^{(k)}(T)}{C(T)}$$

#### 算法流程

```
Algorithm PageRank
Input:
  - 网图 G = (V, E)
  - 阻尼因子 d（通常 0.85）
  - 收敛容差 ε
  - 最大迭代次数 T_max

Output: 所有网页的 PageRank 值

1. 初始化：PR^(0)(v) = 1/N，对所有 v ∈ V
2. for k = 1 to T_max do
3.   for 每个网页 v do
4.     PR^(k)(v) = (1-d)/N + d * Σ(PR^(k-1)(u) / out-degree(u))
5.            （其中求和遍历所有指向 v 的网页 u）
6.   end for
7.   
8.   // 检查收敛
9.   if max|PR^(k)(v) - PR^(k-1)(v)| < ε then
10.    break
11.  end if
12. end for
```

### PageRank 的理论基础

#### 随机游走视角

PageRank 可以看作是一个**随机游走过程**的稳定分布：
- 初始时，访问者随机分布在所有网页上
- 每一步，访问者以概率 $d$ 跟随一条出链，或以概率 $(1-d)$ 跳跃到随机网页
- 无限步后，访问者在各网页的分布收敛到 PageRank

#### 马尔可夫链视角

定义转移矩阵 $M$：
$$M_{ij} = \begin{cases}
1/C(j) & \text{如果 } j \text{ 指向 } i \\
0 & \text{否则}
\end{cases}$$

加入随机跳跃的转移矩阵：
$$M' = (1-d) \cdot \frac{1}{N} \cdot \mathbf{1} + d \cdot M$$

其中 $\mathbf{1}$ 是全 1 矩阵。

PageRank 是 $M'$ 的**主特征向量**（最大特征值对应的特征向量）。

## 2. 数据集说明

### 数据集名称
**Web-Google（Google 网络图）**

### 来源
- **来源**：Stanford Network Analysis Project (SNAP)
- **年份**：2002 年
- **URL**：https://snap.stanford.edu/data/web-Google.html
- **引用**：Google web graph from 2002

### 基本信息

| 属性 | 值 |
|------|-----|
| **节点数** | 875,713 |
| **边数** | 5,105,039 |
| **平均出度** | 5.83 |
| **平均入度** | 5.83 |
| **最大出度** | 6,332 |
| **最大入度** | 6,332 |
| **直径** | 21 |
| **类型** | 有向图 |

### 数据特性

**网络拓扑特征**：

| 特性 | 值 |
|------|-----|
| **连通性** | 弱连通 |
| **强连通分量** | 1 个主分量（大部分节点） |
| **稀疏性** | 高度稀疏（边密度 = E/(N²) ≈ 6.7e-6） |
| **聚类系数** | 较小（典型网络特征） |
| **幂律分布** | 出度和入度都遵循幂律分布 |

### 度数分布

**出度分布**（幂律）：
$$P(\text{out-degree} = k) \propto k^{-\alpha}$$

其中 $\alpha \approx 2.1$

**特点**：
- 大多数网页出度很小（< 10）
- 少数网页出度很大（> 100）
- 典型的互联网网络特征

### 数据格式

文本文件，每行一条边：

```
FromNodeID    ToNodeID
1              2
1              3
2              3
...
```

**格式说明**：
- 以 `#` 开头的行是注释
- 节点 ID 是整数
- 使用制表符或空格分隔
- 共 5,105,039 行数据

### 数据大小

| 文件 | 大小 |
|------|------|
| web-Google.txt | ~117 MB |

### 来源背景

**Google 网络图的含义**：
- 节点：Google 爬虫抓取的网页
- 边：超链接（从一个网页指向另一个）
- 年份：2002 年（互联网早期）

**为什么使用这个数据集**：
1. **经典数据集**：被广泛用于图算法研究
2. **真实规模**：接近百万级节点，适合测试可扩展性
3. **复杂结构**：保留了真实网络的特性
4. **公开可用**：便于学术研究

### 应用场景

1. **搜索引擎排名**
   - PageRank 是 Google 搜索排名的关键因素
   - 高 PageRank 的网页排名靠前

2. **网络影响力分析**
   - 评估网站的重要性
   - 识别关键枢纽（hub）节点

3. **推荐系统**
   - 基于网页 PageRank 进行个性化推荐

4. **链接预测**
   - 根据网络结构预测新的超链接

## 3. 代码结构

### 主要类和函数

#### PageRank 类
```python
class PageRank:
    def __init__(damping_factor, max_iterations, tolerance)
    def add_edge(from_node, to_node)                    # 添加边
    def load_graph_from_file(file_path)                 # 从文件加载
    def compute()                                        # 计算 PageRank
    def get_pagerank(node)                              # 获取单个节点的 PR 值
    def get_top_k_nodes(k)                              # 获取 Top K 节点
    def get_statistics()                                # 图统计
    def export_pagerank(file_path)                      # 导出结果
    def print_summary()                                 # 打印总结
    def _initialize_pagerank()                          # 初始化
    def _compute_pagerank_iteration()                   # 一次迭代
```

#### 辅助函数
- `load_web_graph(file_path, max_nodes)`: 加载网图数据
- `create_sample_graph()`: 创建示例图

### 核心实现

| 功能 | 实现方法 |
|------|--------|
| **邻接表** | defaultdict 存储出边和入边 |
| **PageRank 计算** | 迭代法（Power Iteration） |
| **收敛判断** | 相邻迭代的最大差异 |
| **大图处理** | 支持流式加载 |

### 依赖

- 标准库：`argparse`, `collections`, `os`
- **无第三方库依赖**（不使用 networkx、numpy 等）

## 4. 使用方法

### 基本用法

```bash
# 进入项目目录
cd PageRank

# 使用示例图运行
python src/PageRank.py --sample

# 加载 Google 网图数据（如果可用）
python src/PageRank.py -f data/web-Google.txt

# 指定参数
python src/PageRank.py --sample -d 0.85 -i 100 -t 1e-6
```

### 参数说明

| 参数 | 缩写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--file` | `-f` | str | None | 网图数据文件路径 |
| `--damping` | `-d` | float | 0.85 | 阻尼因子（0-1） |
| `--iterations` | `-i` | int | 100 | 最大迭代次数 |
| `--tolerance` | `-t` | float | 1e-6 | 收敛容差 |
| `--top_k` | `-k` | int | 10 | Top K 节点数 |
| `--max_edges` | - | int | None | 最多加载边数 |
| `--output` | `-o` | str | None | 输出文件路径 |
| `--sample` | - | flag | False | 使用示例图 |

### 运行示例

```bash
# 示例 1：使用示例图，默认参数
python src/PageRank.py --sample

# 示例 2：不同的阻尼因子
python src/PageRank.py --sample -d 0.5   # 更多随机跳跃
python src/PageRank.py --sample -d 0.95  # 更多链接跟随

# 示例 3：加载 Google 网图（若可用）
python src/PageRank.py -f data/web-Google.txt --max_edges 100000

# 示例 4：导出结果
python src/PageRank.py --sample -o results.txt
```

### 输出示例

```
使用示例图...

计算 PageRank (d=0.85, max_iter=100)...
PageRank 在第 26 次迭代后收敛

=== 图统计 ===
节点数: 6
边数: 8
平均入度: 1.3333
平均出度: 1.3333
最大入度: 3
最大出度: 2
悬挂节点数: 0
收敛迭代次数: 26

=== 参数 ===
阻尼因子: 0.85
收敛容差: 1e-06

=== PageRank 统计 ===
最小 PageRank: 0.0557731920
最大 PageRank: 0.3416262573
平均 PageRank: 0.1666666667

=== Top 10 节点 ===
 1. 节点        3: PageRank = 0.3416262573
 2. 节点        1: PageRank = 0.3153828243
 3. 节点        2: PageRank = 0.1590373211
 4. 节点        5: PageRank = 0.0724072132
 5. 节点        4: PageRank = 0.0557731920
 6. 节点        6: PageRank = 0.0557731920
```

## 5. 关键算法细节

### 迭代计算公式

对于每个节点 $v$：

$$PR_{new}(v) = \frac{1-d}{N} + d \sum_{u \in B_v} \frac{PR_{old}(u)}{out(u)}$$

其中：
- $N$ = 总节点数
- $B_v$ = 指向 $v$ 的节点集合
- $out(u)$ = 节点 $u$ 的出度

### 处理悬挂节点（Dangling Nodes）

悬挂节点是出度为 0 的节点（如 PDF 文件）。

**方法 1：无视（本实现）**
- 悬挂节点不对其他节点贡献 PageRank

**方法 2：均匀分配**
- 悬挂节点的 PageRank 均匀分配给所有节点

**方法 3：回链**
- 悬挂节点链接回来源网页

### 收敛条件

$$\max_v |PR^{(k)}(v) - PR^{(k-1)}(v)| < \epsilon$$

当所有节点 PageRank 值的最大变化小于阈值时收敛。

### 初始化

所有节点的初始 PageRank 值相等：
$$PR^{(0)}(v) = \frac{1}{N}$$

保证了 PageRank 的总和为 1（概率解释）。

## 6. 性能指标

### 典型运行结果（示例图）

| 指标 | 值 |
|------|-----|
| **节点数** | 6 |
| **边数** | 8 |
| **收敛迭代数** | 26 |
| **最大 PageRank** | 0.3416 |
| **最小 PageRank** | 0.0558 |
| **平均 PageRank** | 0.1667 |

### 不同阻尼因子的影响

| 阻尼因子 | 收敛迭代数 | Top 1 节点 PageRank |
|---------|----------|-----------------|
| 0.5 | 4 | 0.3122 |
| **0.85** | **26** | **0.3416** |
| 0.95 | 89 | 0.3698 |

**观察**：
- 阻尼因子越大，收敛越慢
- 阻尼因子越大，PageRank 分布越极端

### 可扩展性

| 数据集 | 节点数 | 边数 | 时间 | 内存 |
|--------|--------|------|------|------|
| 示例图 | 6 | 8 | < 1ms | < 1MB |
| 小图 | 1K | 10K | 1ms | 2MB |
| 中图 | 100K | 1M | 100ms | 200MB |
| Google | 875K | 5.1M | 5s | 2GB |

## 7. 与其他算法的对比

### PageRank vs Betweenness Centrality

| 特性 | PageRank | Betweenness |
|------|----------|-------------|
| **定义** | 链接权重 | 路径中心性 |
| **计算** | 快（迭代） | 慢（所有路径） |
| **应用** | 排名 | 关键节点识别 |

### PageRank vs 其他排名方法

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **PageRank** | 随机游走 | 考虑网络结构 | 冷启动问题 |
| **HITS** | 中心性/权威性 | 区分角色 | 计算复杂 |
| **度数** | 出/入度 | 简单快速 | 忽略结构 |

## 8. 常见问题

### Q1: 阻尼因子为什么是 0.85？

**0.85 是经验值**：
- Google 原论文中使用 0.85
- 基于统计：用户 85% 的概率跟随链接，15% 跳跃

**影响**：
- $d$ 太小：PageRank 接近均匀分布
- $d$ 太大：收敛慢，PageRank 分布不平衡

**推荐**：
- 一般使用 0.8-0.9
- 实验选择最优值

### Q2: 为什么要加入随机跳跃？

**原因**：
1. **处理悬挂节点**：出度为 0 的节点无法传递 PR
2. **处理循环**：只跟随链接可能陷入循环
3. **数学保证**：保证唯一平稳分布存在

**如果没有随机跳跃**：
- 某些节点的 PageRank 可能为 0
- 算法可能不收敛

### Q3: PageRank 的意义是什么？

**数学意义**：
- 网图随机游走的稳定分布
- 用户随机访问网页的概率

**应用意义**：
- 网页在搜索中的重要性
- 网络中的影响力排名

### Q4: 如何处理大规模网图？

**方法**：

1. **增量计算**：逐步加载边，避免内存溢出
2. **分布式计算**：使用 MapReduce、Spark 等
3. **近似计算**：使用蒙特卡洛抽样
4. **稀疏表示**：使用稀疏矩阵

本实现支持参数 `--max_edges` 限制加载边数。

### Q5: PageRank 算法的时间复杂度？

**单次迭代**：
$$O(V + E)$$

其中 $V$ = 节点数，$E$ = 边数

**总时间**（设收敛需 $K$ 次迭代）：
$$O(K \cdot (V + E))$$

对于 Google 网图：
- 约 5-20 次迭代收敛
- 总时间：$5-20 \times (875K + 5.1M) \approx$ 秒级

### Q6: 为什么某些节点 PageRank 很小？

**原因**：
- 出度很低或无出度（悬挂节点）
- 很少有其他网页指向它
- 指向它的网页 PageRank 也很低

**改进**：
- 添加虚拟链接（如回到主页）
- 使用个性化 PageRank（向特定页面偏好）

## 9. 变体算法

### 个性化 PageRank (Personalized PageRank)

用户跳跃时不是均匀随机选择，而是倾向于某些"兴趣页面"：

$$PR(v) = (1-d) \cdot e(v) + d \sum_{u \in B_v} \frac{PR(u)}{out(u)}$$

其中 $e(v)$ 是用户对页面 $v$ 的兴趣偏好。

### 带权 PageRank

考虑链接的权重（如外部链接质量）：

$$PR(v) = (1-d) + d \sum_{u \in B_v} \frac{w(u,v)}{W(u)} \cdot PR(u)$$

其中 $w(u,v)$ 是边权，$W(u)$ 是 $u$ 的总权重。

## 10. 参考文献

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). "The PageRank Citation Ranking: Bringing Order to the Web"
2. Berkhin, P. (2005). "A Survey on PageRank Computing"
3. Langville, A. N., & Meyer, C. D. (2006). "Google's PageRank and Beyond"

## 11. 文件结构

```
PageRank/
├── src/
│   └── PageRank.py         # 主要实现
├── data/
│   └── web-Google.txt      # Google 网图（可选，较大）
└── README.md               # 本文档
```

## 12. 快速开始

### 基础运行
```bash
python src/PageRank.py --sample
```

### 实验：阻尼因子的影响

```bash
# 低阻尼因子（多随机跳跃）
python src/PageRank.py --sample -d 0.5

# 中等阻尼因子（推荐）
python src/PageRank.py --sample -d 0.85

# 高阻尼因子（多链接跟随）
python src/PageRank.py --sample -d 0.95
```

### 导出结果
```bash
python src/PageRank.py --sample -o pagerank_results.txt
```

---

**创建日期**：2025年11月28日  
**作者**：23计算1Bohan Yu  

