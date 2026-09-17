# Author: Bohan Yu

import os
import csv
import math
import random
import argparse
from collections import defaultdict, Counter

class GaussianMixtureModel:
    def __init__(self, n_clusters=2, max_iterations=100, tolerance=1e-4, seed=42):
        """Initialize the mixture with cluster count, iteration limit and tolerance."""
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        random.seed(seed)
        

        self.weights = None           # pi_k, mixture weight per cluster.
        self.means = None             # mu_k, per-feature mean per cluster.
        self.converged = False
        self.log_likelihood_history = []
        self.labels = None            # final cluster labels.

    def _initialize_parameters(self, X):
        """Initialize the mixture weights and means."""
        n_samples, n_features = len(X), len(X[0])
        

        self.weights = [1.0 / self.n_clusters] * self.n_clusters
        

        self.means = []
        selected_indices = random.sample(range(n_samples), self.n_clusters)
        
        for idx in selected_indices:

            # Vote features take 0 or 1 values.
            self.means.append(X[idx][:])

    def _expectation_step(self, X):
        """E step, compute the posterior responsibility P(z_k | x_i)."""
        n_samples = len(X)
        
        # Compute P(x_i | z_k) with a Bernoulli model for binary features.
        # P(x_i | z_k) = prod_j p_kj^x_ij * (1 - p_kj)^(1 - x_ij)
        # p_kj is the mean of feature j inside cluster k.
        
        responsibilities = []
        
        for i in range(n_samples):
            x = X[i]
            responsibilities_i = []
            
            for k in range(self.n_clusters):

                log_likelihood = 0.0
                
                for j in range(len(x)):
                    # x[j] is 0 for no or missing and 1 for yes.
                    p_kj = self.means[k][j]
                    

                    p_kj = max(1e-6, min(1 - 1e-6, p_kj))
                    
                    if x[j] == 1:
                        log_likelihood += math.log(p_kj)
                    else:
                        log_likelihood += math.log(1 - p_kj)
                

                log_likelihood += math.log(self.weights[k])
                responsibilities_i.append(math.exp(log_likelihood))
            
            # Normalize to the responsibility P(z_k | x_i).
            sum_resp = sum(responsibilities_i)
            if sum_resp > 0:
                responsibilities_i = [r / sum_resp for r in responsibilities_i]
            else:
                responsibilities_i = [1.0 / self.n_clusters] * self.n_clusters
            
            responsibilities.append(responsibilities_i)
        
        return responsibilities

    def _maximization_step(self, X, responsibilities):
        """M step, update the mixture weights and means from the responsibilities."""
        n_samples = len(X)
        n_features = len(X[0])
        

        N_k = [sum(responsibilities[i][k] for i in range(n_samples)) 
               for k in range(self.n_clusters)]
        

        total_N = sum(N_k)
        self.weights = [N_k[k] / total_N for k in range(self.n_clusters)]
        

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
        """Compute the model log likelihood."""
        n_samples = len(X)
        log_likelihood = 0.0
        
        for i in range(n_samples):

            prob_xi = 0.0
            for k in range(self.n_clusters):

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
        """Train the EM model until convergence."""
        self._initialize_parameters(X)
        self.log_likelihood_history = []
        
        for iteration in range(self.max_iterations):

            responsibilities = self._expectation_step(X)
            

            log_likelihood = self._compute_log_likelihood(X, responsibilities)
            self.log_likelihood_history.append(log_likelihood)
            

            if len(self.log_likelihood_history) > 1:
                delta = abs(self.log_likelihood_history[-1] - 
                           self.log_likelihood_history[-2])
                if delta < self.tolerance:
                    self.converged = True
                    break
            

            self._maximization_step(X, responsibilities)
        

        responsibilities = self._expectation_step(X)
        self.labels = [max(range(self.n_clusters), 
                          key=lambda k: responsibilities[i][k])
                      for i in range(len(X))]

    def predict(self, X):
        """Predict the cluster label of each sample."""
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
            

            label = max(range(self.n_clusters), key=lambda k: probs[k])
            labels.append(label)
        
        return labels

    def predict_proba(self, X):
        """Predict the cluster probabilities of each sample."""
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
            

            sum_probs = sum(probs)
            if sum_probs > 0:
                probs = [p / sum_probs for p in probs]
            else:
                probs = [1.0 / self.n_clusters] * self.n_clusters
            
            probabilities.append(probs)
        
        return probabilities


def load_voting_records(file_path, handle_missing=True):
    """Load the congressional voting dataset."""
    X = []
    y = []
    feature_names = None
    
    # The 16 congressional vote columns.
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

                if not row or row[0].startswith(';'):
                    continue
                
                try:
                    # Column 0 holds the class label.
                    class_label = row[0].strip()
                    if class_label.lower() == 'democrat':
                        y.append(0)
                    elif class_label.lower() == 'republican':
                        y.append(1)
                    else:
                        continue
                    
                    # The next 16 columns hold the votes.
                    votes = []
                    for vote in row[1:17]:
                        vote = vote.strip()
                        if vote == 'y':
                            votes.append(1)
                        elif vote == 'n':
                            votes.append(0)
                        elif vote == '?':
                            # Missing votes are filled randomly or with 0.5.
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
    """Compute clustering purity against the true labels."""

    matched = 0
    
    for cluster_id in range(n_clusters):

        cluster_indices = [i for i, pred in enumerate(y_pred) if pred == cluster_id]
        
        if len(cluster_indices) == 0:
            continue
        

        label_counts = Counter([y_true[i] for i in cluster_indices])
        

        most_common_label = label_counts.most_common(1)[0][0]
        most_common_count = label_counts.most_common(1)[0][1]
        
        matched += most_common_count
    
    purity = matched / len(y_true)
    return purity


def calculate_nmi(y_true, y_pred, n_clusters):
    """Compute normalized mutual information against the true labels."""
    n = len(y_true)
    

    y_counts = Counter(y_true)
    H_Y = sum(-count / n * math.log(count / n) for count in y_counts.values())
    

    c_counts = Counter(y_pred)
    H_C = sum(-count / n * math.log(count / n) for count in c_counts.values())
    


    MI = 0.0
    for true_label in set(y_true):
        for pred_label in set(y_pred):

            count_yc = sum(1 for y, c in zip(y_true, y_pred) 
                          if y == true_label and c == pred_label)
            count_y = sum(1 for y in y_true if y == true_label)
            count_c = sum(1 for c in y_pred if c == pred_label)
            
            if count_yc > 0:
                p_yc = count_yc / n
                p_y = count_y / n
                p_c = count_c / n
                MI += p_yc * math.log(p_yc / (p_y * p_c))
    

    if H_Y + H_C > 0:
        NMI = 2 * MI / (H_Y + H_C)
    else:
        NMI = 0
    
    return NMI


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split records into train and test sets."""
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
    

    if args.file:
        data_file = args.file
    else:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, '..', 'data', 'house-votes-84.data')
    

    print("加载国会投票记录数据...")
    X, y, feature_names = load_voting_records(data_file)
    
    if X is None:
        print("无法加载数据，程序退出")
        exit(1)
    
    print(f"样本数: {len(X)}, 特征数: {len(X[0])}")
    print(f"标签分布: {dict(sorted(Counter(y).items()))}")
    print(f"  - 民主党: {sum(1 for label in y if label == 0)}")
    print(f"  - 共和党: {sum(1 for label in y if label == 1)}")
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    

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
    

    print("进行预测...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    

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
    

    print(f"\n=== 簇信息 ===")
    for k in range(args.n_clusters):
        weight = model.weights[k]
        size = sum(1 for label in y_test_pred if label == k)
        print(f"簇 {k}: 权重={weight:.4f}, 测试集大小={size}")
    

    if len(model.log_likelihood_history) > 1:
        print(f"\n=== 收敛情况 ===")
        print(f"初始对数似然: {model.log_likelihood_history[0]:.4f}")
        print(f"最终对数似然: {model.log_likelihood_history[-1]:.4f}")
        print(f"迭代次数: {len(model.log_likelihood_history)}")
