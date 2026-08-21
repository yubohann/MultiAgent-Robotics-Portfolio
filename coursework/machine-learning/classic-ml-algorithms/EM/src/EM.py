# 作者：23计算1Bohan Yu

import os
import csv
import math
import random
import argparse
from collections import defaultdict, Counter

class GaussianMixtureModel:
    def __init__(self, n_clusters=2, max_iterations=100, tolerance=1e-4, seed=42):
        """
        Args:
            n_clusters: 混合模型的簇数（高斯分布个数）
            max_iterations: EM 算法最大迭代次数
            tolerance: 收敛容差（对数似然变化）
            seed: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        random.seed(seed)
        
        # 模型参数
        self.weights = None           # π_k，各簇的混合权重
        self.means = None             # μ_k，各簇的均值（特征概率）
        self.converged = False
        self.log_likelihood_history = []
        self.labels = None            # 最终聚类标签

    def _initialize_parameters(self, X):
        """随机初始化模型参数
        
        Args:
            X: 输入数据，形状 (n_samples, n_features)
        """
        n_samples, n_features = len(X), len(X[0])
        
        # 初始化权重为均匀分布
        self.weights = [1.0 / self.n_clusters] * self.n_clusters
        
        # 初始化均值：随机选择 n_clusters 个样本的特征均值
        self.means = []
        selected_indices = random.sample(range(n_samples), self.n_clusters)
        
        for idx in selected_indices:
            # 使用选中样本的特征作为初始均值
            # 特征值为 0 或 1，表示投票为"否"或"是"或"弃权"
            self.means.append(X[idx][:])

    def _expectation_step(self, X):
        """E 步：计算后验概率 P(z_k | x_i)
        
        Args:
            X: 输入数据
            
        Returns:
            responsibilities: 形状 (n_samples, n_clusters)，每个样本属于各簇的概率
        """
        n_samples = len(X)
        
        # 计算 P(x_i | z_k)，使用伯努利分布（用于二值数据）
        # P(x_i | z_k) = ∏_j [p_kj^(x_ij) * (1-p_kj)^(1-x_ij)]
        # 其中 p_kj 是簇 k 中特征 j 的均值（参数）
        
        responsibilities = []
        
        for i in range(n_samples):
            x = X[i]
            responsibilities_i = []
            
            for k in range(self.n_clusters):
                # 计算 P(x_i | z_k) * π_k
                log_likelihood = 0.0
                
                for j in range(len(x)):
                    # x[j] 为 0 或 1（0 表示"否"或"缺失"，1 表示"是"）
                    p_kj = self.means[k][j]
                    
                    # 处理边界情况
                    p_kj = max(1e-6, min(1 - 1e-6, p_kj))
                    
                    if x[j] == 1:
                        log_likelihood += math.log(p_kj)
                    else:
                        log_likelihood += math.log(1 - p_kj)
                
                # 加上权重的对数
                log_likelihood += math.log(self.weights[k])
                responsibilities_i.append(math.exp(log_likelihood))
            
            # 归一化：P(z_k | x_i) = P(x_i | z_k) * π_k / P(x_i)
            sum_resp = sum(responsibilities_i)
            if sum_resp > 0:
                responsibilities_i = [r / sum_resp for r in responsibilities_i]
            else:
                responsibilities_i = [1.0 / self.n_clusters] * self.n_clusters
            
            responsibilities.append(responsibilities_i)
        
        return responsibilities

    def _maximization_step(self, X, responsibilities):
        """M 步：根据后验概率更新模型参数
        
        Args:
            X: 输入数据
            responsibilities: E 步计算的后验概率
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # 计算每个簇的有效样本数
        N_k = [sum(responsibilities[i][k] for i in range(n_samples)) 
               for k in range(self.n_clusters)]
        
        # 更新权重：π_k = N_k / N
        total_N = sum(N_k)
        self.weights = [N_k[k] / total_N for k in range(self.n_clusters)]
        
        # 更新均值：μ_kj = Σ_i (r_ik * x_ij) / Σ_i r_ik
        new_means = []
        for k in range(self.n_clusters):
            new_mean_k = []
            for j in range(n_features):
                numerator = sum(responsibilities[i][k] * X[i][j] 
                               for i in range(n_samples))
                denominator = N_k[k]
                
                if denominator > 0:
                    p_kj = numerator / denominator
                else:
                    p_kj = 0.5
                
                new_mean_k.append(p_kj)
            new_means.append(new_mean_k)
        
        self.means = new_means

    def _compute_log_likelihood(self, X, responsibilities):
        """计算模型的对数似然
        
        Log L = Σ_i log(P(x_i)) = Σ_i log(Σ_k π_k * P(x_i | z_k))
        """
        n_samples = len(X)
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # P(x_i) = Σ_k π_k * P(x_i | z_k)
            prob_xi = 0.0
            for k in range(self.n_clusters):
                # 计算 P(x_i | z_k)
                log_prob_xi_zk = 0.0
                for j in range(len(X[i])):
                    p_kj = self.means[k][j]
                    p_kj = max(1e-6, min(1 - 1e-6, p_kj))
                    
                    if X[i][j] == 1:
                        log_prob_xi_zk += math.log(p_kj)
                    else:
                        log_prob_xi_zk += math.log(1 - p_kj)
                
                prob_xi += self.weights[k] * math.exp(log_prob_xi_zk)
            
            if prob_xi > 0:
                log_likelihood += math.log(prob_xi)
        
        return log_likelihood

    def fit(self, X):
        """训练 EM 模型
        
        Args:
            X: 输入数据，形状 (n_samples, n_features)
        """
        self._initialize_parameters(X)
        self.log_likelihood_history = []
        
        for iteration in range(self.max_iterations):
            # E 步
            responsibilities = self._expectation_step(X)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X, responsibilities)
            self.log_likelihood_history.append(log_likelihood)
            
            # 检查收敛
            if len(self.log_likelihood_history) > 1:
                delta = abs(self.log_likelihood_history[-1] - 
                           self.log_likelihood_history[-2])
                if delta < self.tolerance:
                    self.converged = True
                    break
            
            # M 步
            self._maximization_step(X, responsibilities)
        
        # 获得最终聚类标签
        responsibilities = self._expectation_step(X)
        self.labels = [max(range(self.n_clusters), 
                          key=lambda k: responsibilities[i][k])
                      for i in range(len(X))]

    def predict(self, X):
        """预测样本的簇标签
        
        Args:
            X: 输入数据
            
        Returns:
            labels: 簇标签
        """
        n_samples = len(X)
        labels = []
        
        for i in range(n_samples):
            x = X[i]
            probs = []
            
            for k in range(self.n_clusters):
                log_likelihood = 0.0
                for j in range(len(x)):
                    p_kj = self.means[k][j]
                    p_kj = max(1e-6, min(1 - 1e-6, p_kj))
                    
                    if x[j] == 1:
                        log_likelihood += math.log(p_kj)
                    else:
                        log_likelihood += math.log(1 - p_kj)
                
                log_likelihood += math.log(self.weights[k])
                probs.append(math.exp(log_likelihood))
            
            # 选择概率最大的簇
            label = max(range(self.n_clusters), key=lambda k: probs[k])
            labels.append(label)
        
        return labels

    def predict_proba(self, X):
        """预测样本属于各个簇的概率
        
        Args:
            X: 输入数据
            
        Returns:
            probabilities: 形状 (n_samples, n_clusters)
        """
        n_samples = len(X)
        probabilities = []
        
        for i in range(n_samples):
            x = X[i]
            probs = []
            
            for k in range(self.n_clusters):
                log_likelihood = 0.0
                for j in range(len(x)):
                    p_kj = self.means[k][j]
                    p_kj = max(1e-6, min(1 - 1e-6, p_kj))
                    
                    if x[j] == 1:
                        log_likelihood += math.log(p_kj)
                    else:
                        log_likelihood += math.log(1 - p_kj)
                
                log_likelihood += math.log(self.weights[k])
                probs.append(math.exp(log_likelihood))
            
            # 归一化
            sum_probs = sum(probs)
            if sum_probs > 0:
                probs = [p / sum_probs for p in probs]
            else:
                probs = [1.0 / self.n_clusters] * self.n_clusters
            
            probabilities.append(probs)
        
        return probabilities


def load_voting_records(file_path, handle_missing=True):
    """加载国会投票记录数据集
    
    数据格式：第一列为类别（democrat/republican），后续列为投票（y/n/?)
    
    Args:
        file_path: 数据文件路径
        handle_missing: 是否处理缺失值（'?' 标记）
        
    Returns:
        X: 特征数据，二值化 (0 或 1)
        y: 类别标签 (0=democrat, 1=republican)
        feature_names: 特征名称
    """
    X = []
    y = []
    feature_names = None
    
    # 16 个投票项目名称
    vote_names = [
        'handicapped-infants',
        'water-project-cost-sharing',
        'adoption-of-the-budget-resolution',
        'physician-fee-freeze',
        'el-salvador-aid',
        'religious-groups-in-schools',
        'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras',
        'mx-missile',
        'immigration',
        'synfuels-corporation-cutback',
        'education-spending',
        'superfund-right-to-sue',
        'crime',
        'duty-free-exports',
        'export-administration-act-south-africa'
    ]
    feature_names = vote_names
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # 跳过空行和注释
                if not row or row[0].startswith(';'):
                    continue
                
                try:
                    # 第一列是类别
                    class_label = row[0].strip()
                    if class_label.lower() == 'democrat':
                        y.append(0)
                    elif class_label.lower() == 'republican':
                        y.append(1)
                    else:
                        continue
                    
                    # 后续 16 列是投票
                    votes = []
                    for vote in row[1:17]:
                        vote = vote.strip()
                        if vote == 'y':
                            votes.append(1)
                        elif vote == 'n':
                            votes.append(0)
                        elif vote == '?':
                            # 处理缺失值：随机填充或用 0.5
                            if handle_missing:
                                votes.append(random.randint(0, 1))
                            else:
                                votes.append(0)
                        else:
                            continue
                    
                    if len(votes) == 16:
                        X.append(votes)
                
                except (ValueError, IndexError):
                    continue
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None, None, None
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        return None, None, None
    
    return X, y, feature_names


def calculate_purity(y_true, y_pred, n_clusters):
    """计算聚类纯度
    
    纯度 = (正确聚类的样本数) / (总样本数)
    """
    # 对于每个聚类，找到与其最匹配的真实标签
    matched = 0
    
    for cluster_id in range(n_clusters):
        # 该聚类中的样本
        cluster_indices = [i for i, pred in enumerate(y_pred) if pred == cluster_id]
        
        if len(cluster_indices) == 0:
            continue
        
        # 该聚类中各真实标签的计数
        label_counts = Counter([y_true[i] for i in cluster_indices])
        
        # 选择最频繁的标签作为匹配
        most_common_label = label_counts.most_common(1)[0][0]
        most_common_count = label_counts.most_common(1)[0][1]
        
        matched += most_common_count
    
    purity = matched / len(y_true)
    return purity


def calculate_nmi(y_true, y_pred, n_clusters):
    """计算规范化互信息 (Normalized Mutual Information)
    
    NMI = 2 * I(Y; C) / (H(Y) + H(C))
    其中 Y 是真实标签，C 是聚类标签
    """
    n = len(y_true)
    
    # 计算熵 H(Y)
    y_counts = Counter(y_true)
    H_Y = sum(-count / n * math.log(count / n) for count in y_counts.values())
    
    # 计算熵 H(C)
    c_counts = Counter(y_pred)
    H_C = sum(-count / n * math.log(count / n) for count in c_counts.values())
    
    # 计算互信息 I(Y; C)
    # I(Y; C) = Σ_y Σ_c P(y, c) * log(P(y, c) / (P(y) * P(c)))
    MI = 0.0
    for true_label in set(y_true):
        for pred_label in set(y_pred):
            # 计数
            count_yc = sum(1 for y, c in zip(y_true, y_pred) 
                          if y == true_label and c == pred_label)
            count_y = sum(1 for y in y_true if y == true_label)
            count_c = sum(1 for c in y_pred if c == pred_label)
            
            if count_yc > 0:
                p_yc = count_yc / n
                p_y = count_y / n
                p_c = count_c / n
                MI += p_yc * math.log(p_yc / (p_y * p_c))
    
    # 计算 NMI
    if H_Y + H_C > 0:
        NMI = 2 * MI / (H_Y + H_C)
    else:
        NMI = 0
    
    return NMI


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """划分训练集和测试集"""
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_ratio))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EM 算法进行无监督聚类')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='国会投票记录数据 CSV 文件路径')
    parser.add_argument('-k', '--n_clusters', type=int, default=2,
                        help='簇数（混合高斯数）')
    parser.add_argument('-i', '--max_iterations', type=int, default=100,
                        help='EM 最大迭代次数')
    parser.add_argument('-t', '--tolerance', type=float, default=1e-4,
                        help='收敛容差')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 确定数据文件路径
    if args.file:
        data_file = args.file
    else:
        # 默认路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, '..', 'data', 'house-votes-84.data')
    
    # 加载数据
    print("加载国会投票记录数据...")
    X, y, feature_names = load_voting_records(data_file)
    
    if X is None:
        print("无法加载数据，程序退出")
        exit(1)
    
    print(f"样本数: {len(X)}, 特征数: {len(X[0])}")
    print(f"标签分布: {dict(sorted(Counter(y).items()))}")
    print(f"  - 民主党: {sum(1 for label in y if label == 0)}")
    print(f"  - 共和党: {sum(1 for label in y if label == 1)}")
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 训练 EM 模型
    print(f"\n训练 EM 高斯混合模型 (k={args.n_clusters})...")
    model = GaussianMixtureModel(
        n_clusters=args.n_clusters,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        seed=args.seed
    )
    model.fit(X_train)
    
    if model.converged:
        print(f"EM 算法在 {len(model.log_likelihood_history)} 次迭代后收敛")
    else:
        print(f"EM 算法未收敛（达到最大迭代次数 {args.max_iterations}）")
    
    # 预测
    print("进行预测...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估
    print(f"\n=== 训练集评估 ===")
    train_purity = calculate_purity(y_train, y_train_pred, args.n_clusters)
    train_nmi = calculate_nmi(y_train, y_train_pred, args.n_clusters)
    print(f"纯度 (Purity): {train_purity:.4f}")
    print(f"规范化互信息 (NMI): {train_nmi:.4f}")
    
    print(f"\n=== 测试集评估 ===")
    test_purity = calculate_purity(y_test, y_test_pred, args.n_clusters)
    test_nmi = calculate_nmi(y_test, y_test_pred, args.n_clusters)
    print(f"纯度 (Purity): {test_purity:.4f}")
    print(f"规范化互信息 (NMI): {test_nmi:.4f}")
    
    # 混合权重
    print(f"\n=== 簇信息 ===")
    for k in range(args.n_clusters):
        weight = model.weights[k]
        size = sum(1 for label in y_test_pred if label == k)
        print(f"簇 {k}: 权重={weight:.4f}, 测试集大小={size}")
    
    # 显示对数似然变化
    if len(model.log_likelihood_history) > 1:
        print(f"\n=== 收敛情况 ===")
        print(f"初始对数似然: {model.log_likelihood_history[0]:.4f}")
        print(f"最终对数似然: {model.log_likelihood_history[-1]:.4f}")
        print(f"迭代次数: {len(model.log_likelihood_history)}")
