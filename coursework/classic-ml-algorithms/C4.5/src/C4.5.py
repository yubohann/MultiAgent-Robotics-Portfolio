# 作者：23计算1Bohan Yu

import csv
import math
import random
import argparse
import os
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter


def load_wdbc_data(file_path: str) -> Tuple[List[List[float]], List[str], List[str]]:
    """从 CSV 加载 Wisconsin Diagnostic Breast Cancer 数据。
    
    返回：
        X: 特征列表，每个样本是一个浮点数列表
        y: 标签列表（'M' 或 'B'）
        feature_names: 特征名称列表
    """
    X = []
    y = []
    feature_names = [
        'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean',
        'Compactness_mean', 'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean', 'Fractal_dimension_mean',
        'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se', 'Smoothness_se',
        'Compactness_se', 'Concavity_se', 'Concave_points_se', 'Symmetry_se', 'Fractal_dimension_se',
        'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst', 'Smoothness_worst',
        'Compactness_worst', 'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
    ]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 32:
                continue
            try:
                # 跳过 ID 列（第0列），标签是第1列
                label = row[1].strip()
                # 特征从第2列开始
                features = [float(v) for v in row[2:]]
                if len(features) == 30:  # 确保有30个特征
                    X.append(features)
                    y.append(label)
            except (ValueError, IndexError):
                continue
    
    return X, y, feature_names


class Node:
    """决策树节点。"""
    
    def __init__(self, is_leaf: bool = False, class_label: Optional[str] = None,
                 feature_idx: Optional[int] = None, threshold: Optional[float] = None):
        self.is_leaf = is_leaf
        self.class_label = class_label  # 叶子节点的预测类别
        self.feature_idx = feature_idx  # 分裂的特征索引
        self.threshold = threshold      # 分裂的阈值（连续特征）
        self.left = None                # 左子树
        self.right = None               # 右子树


class C45Tree:
    """C4.5 决策树分类器。"""
    
    def __init__(self, max_depth: int = 20, min_samples: int = 2):
        """
        初始化 C4.5 决策树。
        
        Args:
            max_depth: 树的最大深度
            min_samples: 分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.feature_names = None
    
    @staticmethod
    def entropy(y: List[str]) -> float:
        """计算标签列表的熵。"""
        if not y:
            return 0.0
        counts = Counter(y)
        ent = 0.0
        n = len(y)
        for count in counts.values():
            if count > 0:
                p = count / n
                ent -= p * math.log2(p)
        return ent
    
    @staticmethod
    def information_gain(parent_entropy: float, weighted_child_entropy: float) -> float:
        """计算信息增益。"""
        return parent_entropy - weighted_child_entropy
    
    @staticmethod
    def split_information(split_sizes: List[int], total: int) -> float:
        """计算分裂信息（用于信息增益率）。"""
        si = 0.0
        for size in split_sizes:
            if size > 0:
                p = size / total
                si -= p * math.log2(p)
        return si
    
    @staticmethod
    def gain_ratio(parent_entropy: float, weighted_child_entropy: float,
                   split_sizes: List[int], total: int) -> float:
        """计算信息增益率（C4.5 的关键准则）。"""
        gain = C45Tree.information_gain(parent_entropy, weighted_child_entropy)
        si = C45Tree.split_information(split_sizes, total)
        
        # 避免除以零
        if si == 0.0:
            return 0.0
        return gain / si
    
    def best_split(self, X: List[List[float]], y: List[str]) -> Tuple[Optional[int], Optional[float], float]:
        """
        找到最佳的分裂特征和阈值（使用信息增益率）。
        
        返回：(best_feature_idx, best_threshold, best_gain_ratio)
        """
        parent_entropy = self.entropy(y)
        best_gain_ratio = -1.0
        best_feature = None
        best_threshold = None
        n = len(X)
        n_features = len(X[0]) if X else 0
        
        for feature_idx in range(n_features):
            feature_values = [X[i][feature_idx] for i in range(n)]
            unique_vals = sorted(set(feature_values))
            
            # 若特征只有一个值，跳过
            if len(unique_vals) == 1:
                continue
            
            # 对于连续特征，尝试中点作为候选阈值
            for i in range(len(unique_vals) - 1):
                threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
                
                # 分裂数据集
                left_indices = [j for j in range(n) if X[j][feature_idx] <= threshold]
                right_indices = [j for j in range(n) if X[j][feature_idx] > threshold]
                
                # 若某一侧为空，跳过
                if not left_indices or not right_indices:
                    continue
                
                # 计算分裂后的加权熵
                y_left = [y[j] for j in left_indices]
                y_right = [y[j] for j in right_indices]
                left_weight = len(y_left) / n
                right_weight = len(y_right) / n
                weighted_entropy = left_weight * self.entropy(y_left) + right_weight * self.entropy(y_right)
                
                # 计算信息增益率
                split_sizes = [len(y_left), len(y_right)]
                gr = self.gain_ratio(parent_entropy, weighted_entropy, split_sizes, n)
                
                if gr > best_gain_ratio:
                    best_gain_ratio = gr
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain_ratio
    
    def build_tree(self, X: List[List[float]], y: List[str], depth: int = 0) -> Node:
        """递归构建决策树。"""
        
        # 停止条件1：纯叶子（所有样本属于同一类）
        if len(set(y)) == 1:
            return Node(is_leaf=True, class_label=y[0])
        
        # 停止条件2：样本数太少
        if len(y) < self.min_samples:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(is_leaf=True, class_label=most_common)
        
        # 停止条件3：达到最大深度
        if depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(is_leaf=True, class_label=most_common)
        
        # 寻找最优分裂
        best_feature, best_threshold, best_gr = self.best_split(X, y)
        
        # 若没有有意义的分裂，创建叶子节点
        if best_feature is None or best_gr <= 0.0:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(is_leaf=True, class_label=most_common)
        
        # 分裂数据集
        left_indices = [i for i in range(len(X)) if X[i][best_feature] <= best_threshold]
        right_indices = [i for i in range(len(X)) if X[i][best_feature] > best_threshold]
        
        X_left = [X[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_left = [y[i] for i in left_indices]
        y_right = [y[i] for i in right_indices]
        
        # 创建内部节点
        node = Node(is_leaf=False, feature_idx=best_feature, threshold=best_threshold)
        node.left = self.build_tree(X_left, y_left, depth + 1)
        node.right = self.build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(self, X: List[List[float]], y: List[str], feature_names: Optional[List[str]] = None):
        """训练决策树。"""
        self.feature_names = feature_names
        self.root = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x: List[float], node: Optional[Node] = None) -> str:
        """预测单个样本的类别。"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return node.class_label
        
        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X: List[List[float]]) -> List[str]:
        """预测多个样本的类别。"""
        return [self.predict_sample(x) for x in X]
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0, prefix: str = ""):
        """打印决策树结构（用于可视化）。"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            print(f"{prefix}└─ 类别: {node.class_label}")
        else:
            feature_name = self.feature_names[node.feature_idx] if self.feature_names else f"特征{node.feature_idx}"
            print(f"{prefix}├─ {feature_name} <= {node.threshold:.4f}?")
            print(f"{prefix}│  (是)")
            self.print_tree(node.left, depth + 1, prefix + "│  ")
            print(f"{prefix}│  (否)")
            self.print_tree(node.right, depth + 1, prefix + "│  ")


def accuracy_score(y_true: List[str], y_pred: List[str]) -> float:
    """计算准确率。"""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) if y_true else 0.0


def confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict:
    """计算混淆矩阵。"""
    classes = sorted(set(y_true))
    matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    
    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1
    
    return matrix


def print_confusion_matrix(cm: Dict):
    """打印混淆矩阵。"""
    classes = sorted(cm.keys())
    print("\n混淆矩阵:")
    print("   " + "  ".join(f"{c:>5}" for c in classes))
    for c1 in classes:
        print(f"{c1}  " + "  ".join(f"{cm[c1][c2]:>5}" for c2 in classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C4.5 决策树（纯 Python 实现）')
    parser.add_argument('-f', '--file', help='WDBC 数据 CSV 文件路径', default=None)
    parser.add_argument('-d', '--max_depth', type=int, default=20, help='树的最大深度')
    parser.add_argument('-m', '--min_samples', type=int, default=2, help='分裂所需的最小样本数')
    parser.add_argument('-t', '--test_size', type=float, default=0.3, help='测试集比例（0-1）')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--print_tree', action='store_true', help='打印决策树结构')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.file is None:
        # 尝试默认位置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'wdbc.data'))
        if os.path.exists(default_path):
            args.file = default_path

    if args.file is None or not os.path.exists(args.file):
        print('未找到 WDBC 数据文件。')
        print('数据集下载地址: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic')
        raise SystemExit(1)

    # 加载数据
    X, y, feature_names = load_wdbc_data(args.file)
    print(f'\n加载数据: 样本数={len(X)}, 特征数={len(X[0]) if X else 0}')
    print(f'类别分布: {dict(Counter(y))}')

    # 划分训练/测试集
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    
    n_test = int(len(X) * args.test_size)
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]
    
    print(f'训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}')

    # 训练模型
    print(f'\n训练 C4.5 决策树 (max_depth={args.max_depth}, min_samples={args.min_samples})...')
    model = C45Tree(max_depth=args.max_depth, min_samples=args.min_samples)
    model.fit(X_train, y_train, feature_names)

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 评估
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f'\n=== 性能评估 ===')
    print(f'训练准确率: {train_acc:.4f}')
    print(f'测试准确率: {test_acc:.4f}')

    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    print_confusion_matrix(cm)

    # 计算精确率、召回率、F1 分数
    classes = sorted(set(y_test))
    print(f'\n=== 分类指标 ===')
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[c][cls] for c in classes if c != cls)
        fn = sum(cm[cls][c] for c in classes if c != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f'类别 {cls}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}')

    # 打印树结构（可选）
    if args.print_tree:
        print('\n=== 决策树结构 ===')
        model.print_tree()
