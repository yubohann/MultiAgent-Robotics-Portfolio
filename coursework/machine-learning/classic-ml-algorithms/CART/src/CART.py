# Author: Bohan Yu

import os
import csv

import random
import argparse
from collections import defaultdict, Counter

class Node:
    """Decision tree node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CARTTree:
    """CART decision tree for classification with Gini impurity."""
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.n_features = None
        self.feature_names = None

    def gini_impurity(self, y):
        """Compute the Gini impurity of a label list."""
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        impurity = 1.0
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob * prob
        return impurity

    def gini_gain(self, y, left, right):
        """Compute the Gini gain of a split."""
        n = len(y)
        if n == 0:
            return 0
        
        n_left = len(left)
        n_right = len(right)
        if n_left == 0 or n_right == 0:
            return 0
        
        gini_parent = self.gini_impurity(y)
        gini_left = self.gini_impurity(left)
        gini_right = self.gini_impurity(right)
        
        # Weighted child Gini impurity.
        gini_children = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        # Gini gain is the parent impurity minus the weighted child impurity.
        return gini_parent - gini_children

    def best_split(self, X, y):
        """Find the split with the largest Gini gain."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None
        
        for feature_idx in range(self.n_features):

            feature_values = [X[i][feature_idx] for i in range(len(X))]
            
            # Candidate thresholds are midpoints between feature values.
            unique_values = sorted(set(feature_values))
            if len(unique_values) < 2:
                continue
            
            thresholds = []
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                thresholds.append(threshold)
            

            for threshold in thresholds:
                left_indices = []
                right_indices = []
                
                for i in range(len(X)):
                    if X[i][feature_idx] <= threshold:
                        left_indices.append(i)
                    else:
                        right_indices.append(i)
                

                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                

                left_y = [y[i] for i in left_indices]
                right_y = [y[i] for i in right_indices]
                
                gain = self.gini_gain(y, left_y, right_y)
                

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left = left_indices
                    best_right = right_indices
        
        return {
            'gain': best_gain,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': best_left,
            'right': best_right
        }

    def build_tree(self, X, y, depth=0):
        """Build the decision tree recursively."""
        n_samples = len(y)
        n_classes = len(set(y))
        


        if n_classes == 1:
            most_common = y[0]
            return Node(value=most_common)
        

        if self.max_depth is not None and depth >= self.max_depth:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        

        if n_samples < self.min_samples_split:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        

        split_info = self.best_split(X, y)
        

        if split_info['gain'] == -1 or split_info['feature'] is None:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)
        

        left_X = [X[i] for i in split_info['left']]
        left_y = [y[i] for i in split_info['left']]
        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        
        right_X = [X[i] for i in split_info['right']]
        right_y = [y[i] for i in split_info['right']]
        right_subtree = self.build_tree(right_X, right_y, depth + 1)
        

        return Node(
            feature=split_info['feature'],
            threshold=split_info['threshold'],
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X, y, feature_names=None):
        """Train the decision tree."""
        self.n_features = len(X[0])
        self.feature_names = feature_names or [f"Feature {i}" for i in range(self.n_features)]
        self.tree = self.build_tree(X, y)
        return self

    def predict_sample(self, x, node):
        """Predict the value of one sample."""
        if node.value is not None:

            return node.value
        

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        """Predict the values of many samples."""
        predictions = []
        for x in X:
            pred = self.predict_sample(x, self.tree)
            predictions.append(pred)
        return predictions

    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """Print the tree structure."""
        if node is None:
            node = self.tree
        
        indent = "  " * depth
        
        if node.value is not None:

            print(f"{indent}{prefix}质量 = {node.value}")
        else:

            feature_name = self.feature_names[node.feature]
            print(f"{indent}{prefix}{feature_name} <= {node.threshold:.3f}?")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "Left: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "Right: ")


def load_wine_quality_data(file_path):
    """Load the wine quality dataset from CSV."""
    X = []
    y = []
    feature_names = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            

            header = next(reader)
            feature_names = header[:-1]  # The last column holds the quality label.
            

            for row in reader:
                try:
                    features = [float(val) for val in row[:-1]]
                    label = int(float(row[-1]))
                    X.append(features)
                    y.append(label)
                except (ValueError, IndexError):
                    continue
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None, None, None
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        return None, None, None
    
    return X, y, feature_names


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split features and labels into train and test sets."""
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


def evaluate_classification(y_true, y_pred):
    """Score a classifier with accuracy, precision, recall and F1."""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    

    classes = sorted(set(y_true) | set(y_pred))
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    

    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    
    for cls in classes:
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[other][cls] for other in classes if other != cls)
        fn = sum(confusion_matrix[cls][other] for other in classes if other != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores[cls] = precision
        recall_scores[cls] = recall
        f1_scores[cls] = f1
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision': precision_scores,
        'recall': recall_scores,
        'f1': f1_scores
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CART 决策树分类')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='葡萄酒质量数据 CSV 文件路径')
    parser.add_argument('-d', '--max_depth', type=int, default=10,
                        help='最大树深度')
    parser.add_argument('-m', '--min_samples_split', type=int, default=2,
                        help='分裂的最小样本数')
    parser.add_argument('-t', '--test_ratio', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--print_tree', action='store_true',
                        help='是否打印决策树结构')
    
    args = parser.parse_args()
    

    if args.file:
        data_file = args.file
    else:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, '..', 'data', 'winequality-red.csv')
    

    print("加载葡萄酒质量数据...")
    X, y, feature_names = load_wine_quality_data(data_file)
    
    if X is None:
        print("无法加载数据，程序退出")
        exit(1)
    
    print(f"样本数: {len(X)}, 特征数: {len(X[0])}, 标签类数: {len(set(y))}")
    print(f"质量评分分布: {dict(sorted(Counter(y).items()))}")
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    

    print(f"\n训练 CART 决策树 (max_depth={args.max_depth})...")
    tree = CARTTree(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=1
    )
    tree.fit(X_train, y_train, feature_names=feature_names)
    

    print("评估模型...")
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    

    train_results = evaluate_classification(y_train, y_train_pred)
    test_results = evaluate_classification(y_test, y_test_pred)
    
    print("\n=== 训练集评估 ===")
    print(f"准确率: {train_results['accuracy']:.4f}")
    
    print("\n=== 测试集评估 ===")
    print(f"准确率: {test_results['accuracy']:.4f}")
    
    print("\n=== 按类别评估（测试集）===")
    for cls in sorted(set(y_test)):
        precision = test_results['precision'].get(cls, 0)
        recall = test_results['recall'].get(cls, 0)
        f1 = test_results['f1'].get(cls, 0)
        print(f"质量 {cls}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
    

    if args.print_tree:
        print("\n=== 决策树结构 ===")
        tree.print_tree()
