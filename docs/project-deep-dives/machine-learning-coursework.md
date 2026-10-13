# Machine Learning Coursework：项目技术说明

## 300 字项目概括

Machine Learning Coursework 将两个课程项目整理到同一稳定路径：一组从零实现的经典算法，以及围绕 PCA/LDA、Iris kNN 和 Wine kNN-versus-ID3 的综合作业。经典算法目录覆盖 AdaBoost、Apriori、C4.5、CART、EM、K-means、KNN、Naive Bayes、PageRank 和 SVM；每个子目录保留自己的数据、`src/` 入口、README、运行参数和图表。实现重点是把算法步骤写成可读代码，例如 K-means++ 初始化与迭代更新、EM 的 E/M 步和对数似然、C4.5 的信息增益率、CART 的基尼指数、AdaBoost 的样本权重、Apriori 的候选集剪枝以及线性 SVM 的 Pegasos/one-vs-rest。Assignment 3 则把降维、分类和可视化串成实验流程，输出 PCA 投影和 kNN 随 k 变化曲线，并保留课程报告。它适合展示数学推导到 Python 实现、数据预处理、实验可视化和结果解释的基础能力；这些是教学与课程成果，不应包装成生产 ML 库或经过严格基准协议的研究系统。

## 项目定位

- **经典算法**：强调从零实现和算法机制理解，主要依赖 Python/NumPy 与公开数据。
- **Assignment 3**：强调实验设计、降维可视化和模型比较，使用 NumPy、pandas、scikit-learn、matplotlib。
- **组织方式**：保留原始项目、数据、结果和报告，只在 portfolio 层统一导航。

## 技术架构

```text
dataset -> parser / preprocessing
        -> from-scratch algorithm or sklearn comparison
        -> fit / predict / cluster / rule mining
        -> metrics and plots
        -> README, report and result figures
```

经典算法互相独立，便于单独阅读和运行；Assignment 3 的 PCA/LDA、Iris kNN、Wine kNN/ID3 通过脚本和结果目录形成实验链。每个算法的 README 通常解释定义、伪代码、参数、示例输出、复杂度和局限。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `classic-ml-algorithms/<algorithm>/src/` | 十个算法的主要 Python 实现和 CLI |
| `classic-ml-algorithms/<algorithm>/data/` | 对应示例/公开数据集 |
| `classic-ml-algorithms/<algorithm>/README.md` | 算法推导、参数、命令和结果说明 |
| `ml-assignment-3/` | PCA/LDA、Iris/Wine 实验、脚本、报告和结果 |
| `ml-assignment-3/results/` | PCA 投影、准确率曲线等可视化产物 |

## 关键算法知识

- **分类/集成**：KNN 距离投票、Naive Bayes 条件独立、SVM 间隔与正则、AdaBoost 弱分类器加权。
- **树模型**：ID3 信息增益、C4.5 gain ratio、CART 基尼指数和递归划分。
- **聚类/潜变量**：K-means/K-means++、EM 高斯混合和收敛历史。
- **无监督/图算法**：Apriori 支持度/置信度/提升度、PageRank 迭代和阻尼。
- **降维/评估**：PCA 特征分解、LDA 类间/类内散度、kNN 的 k 扫描和可视化。

## 开发过程

先为每个算法建立独立 README 和可执行入口，再补充数据读取、参数命令和示例结果；Assignment 3 在此基础上增加实验脚本、图表和课程报告。项目合并到 portfolio 时保留原始图片路径和中英文导航，避免把课程成果误写成一个统一框架。

## 运行入口

```powershell
cd coursework/machine-learning/classic-ml-algorithms/<algorithm>
python src/<algorithm>.py

cd coursework/machine-learning/ml-assignment-3
python <对应实验脚本>.py
```

具体文件名、输入格式和参数以子项目 README 为准；Assignment 3 结果图位于 `ml-assignment-3/results/`。

## 学术申请中的项目价值

该项目最适合证明基础扎实和能够独立完成课程实验：从公式、伪代码到可运行实现，再用图表解释模型行为。它可以作为机器学习、数据挖掘或机器人学习申请中的基础项目，与更大型的 FraudGraph、Rivermark 或 CBG-WM 形成从经典方法到系统研究的梯度。

## 证据边界

这些内容属于课程工程和教学实现。结果依赖数据文件、脚本参数和本地运行环境；README 中的典型输出不等于统一随机种子、多次重复或论文级统计。不要把单个准确率或课程报告包装为生产模型、SOTA 或独立科研贡献。

## 源码审计记录

| 算法/模块 | 主要阅读对象 | 实现要点 |
| --- | --- | --- |
| AdaBoost | `AdaBoost/src/AdaBoost.py` | 初始化样本权重、弱分类器误差、alpha、权重更新和投票预测。 |
| Apriori | `Apriori/src/Apriori.py` | 事务解析、候选集生成、支持度剪枝、规则置信度/提升度。 |
| C4.5 | `C4.5/src/C4.5.py` | 熵、信息增益率、连续/离散属性分裂和递归树。 |
| CART | `CART/src/CART.py` | 基尼指数、二叉划分、停止条件、叶节点预测。 |
| EM | `EM/src/EM.py` | 高斯参数、E-step 后验、M-step 更新、对数似然和收敛。 |
| K-means | `K-means/src/K-means.py` | K-means++/随机初始化、assignment、centroid 更新、inertia。 |
| KNN | `KNN/src/KNN.py` | 距离、k 邻居选择、投票、分类/回归边界。 |
| Naive Bayes | `Naive_Bayes/src/Naive_Bayes.py` | 先验、条件概率、平滑、对数概率和预测。 |
| PageRank | `PageRank/src/PageRank.py` | 邻接、阻尼、迭代、dangling node 和收敛阈值。 |
| SVM | `SVM/src/SVM.py` | Pegasos/线性间隔、正则、one-vs-rest 和推理。 |
| Assignment 3 | `ml-assignment-3/` | PCA/LDA 变换、Iris kNN 扫描、Wine kNN/ID3 和图表。 |

## 统一实验复核

每个算法都应按同一顺序阅读：先找 CLI 和输入 parser，再找核心 class/function，再追踪 fit/train 到 predict/cluster/rules 的数据形状，最后核对 README 中的公式、复杂度和示例输出。对无监督方法，标签只能用于外部评估或可视化，不能偷偷进入训练；对分类方法，要确认预处理、距离尺度、类别编码和随机种子；对 PageRank/Apriori，要区分图结构或事务频次与监督标签。

## 数学到代码的检查点

- 公式中的求和、归一化和对数是否与数组维度一致。
- 终止条件是否同时考虑最大迭代和数值收敛。
- 空簇、零方差、零支持度、dangling node 和缺失值是否有分支。
- 训练/测试划分、标准化和可视化是否发生在正确阶段。
- 示例输出是否来自当前脚本和数据，而非手工复制。

## 学习路径

先用 KNN、K-means 理解距离和迭代，再读 Naive Bayes、树和 SVM；随后读 AdaBoost 的重加权和 EM 的隐变量优化；Apriori/PageRank 用来补充频繁模式与图迭代；最后用 Assignment 3 的 PCA/LDA 把数学、可视化和比较串起来。完成后再对照 FraudGraph 的图/序列融合，理解从单文件算法到实验工程的差异。

## 原理与理论基础

### 1. 监督分类与经验风险

分类算法都在有限样本上最小化经验风险 1/n sum_i l(f(x_i),y_i)，但损失、假设空间和归纳偏置不同。KNN 不显式训练参数，依赖局部距离；Naive Bayes 假设给定类别后特征条件独立；SVM 通过最大化间隔控制泛化；树模型通过递归划分降低节点不纯度；AdaBoost 用加权弱学习器降低分类错误。理解假设比背模型名称更重要，因为数据尺度、噪声和类别结构会决定方法是否适合。

### 2. 树模型的分裂准则

熵 H(Y)=-sum_c p_c log p_c 描述不确定性，信息增益是父节点熵减子节点加权熵；C4.5 用 split information 归一化得到 gain ratio，减轻高基数属性偏好；CART 用基尼指数 1-sum_c p_c^2 并进行二叉分裂。代码中的递归停止、空子集、连续阈值、叶节点多数类和剪枝/深度参数决定了理论公式能否落地。

### 3. 聚类与潜变量

K-means 近似最小化簇内平方和 sum_k sum_{x_i in C_k} ||x_i-mu_k||^2，交替 assignment 和 centroid update 只能保证到达局部最优；K-means++ 用距离平方比例采样改善初始质心。EM 对含隐变量的似然下界做 coordinate ascent：E-step 计算后验责任度，M-step 在责任度加权下更新参数；高斯协方差退化、数值下溢和局部最优需要显式处理。

### 4. 降维、关联规则与 PageRank

PCA 寻找最大方差正交方向，等价于对中心化协方差矩阵做特征分解；LDA 最大化类间散度相对类内散度。Apriori 依赖 support 的 downward closure：频繁项集的所有子集都频繁，因此可以剪枝候选；PageRank 迭代 r=alpha P^T r +(1-alpha)v，阻尼项处理 dangling node 和随机跳转。三者分别体现线性代数、组合剪枝和图上的固定点。

### 5. 理论假设和失效模式

距离模型假设尺度可比，概率模型假设分布近似成立，树模型假设分裂能表达决策边界，聚类假设簇结构与距离度量匹配。缺失值、异常点、类别不平衡、特征泄漏、随机初始化和评估口径会改变结果；课程项目因此应把数据、参数、seed、图表和典型输出一起阅读，而非只看算法名。
