# 作者：23计算1Bohan Yu

import os
import csv
import math
import random
import argparse
from collections import Counter

class KNN:
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        Args:
            k: 近邻数量
            distance_metric: 距离度量方式 ('euclidean', 'manhattan', 'chebyshev')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.n_features = None

    def _euclidean_distance(self, x1, x2):
        """欧几里得距离
        
        d(x1, x2) = √(Σ(x1_i - x2_i)²)
        """
        if len(x1) != len(x2):
            raise ValueError("特征维度不匹配")
        
        sum_squared_diff = sum((x1[i] - x2[i]) ** 2 for i in range(len(x1)))
        return math.sqrt(sum_squared_diff)

    def _manhattan_distance(self, x1, x2):
        """曼哈顿距离（城市街区距离）
        
        d(x1, x2) = Σ|x1_i - x2_i|
        """
        if len(x1) != len(x2):
            raise ValueError("特征维度不匹配")
        
        return sum(abs(x1[i] - x2[i]) for i in range(len(x1)))

    def _chebyshev_distance(self, x1, x2):
        """切比雪夫距离（棋盘距离）
        
        d(x1, x2) = max(|x1_i - x2_i|)
        """
        if len(x1) != len(x2):
            raise ValueError("特征维度不匹配")
        
        return max(abs(x1[i] - x2[i]) for i in range(len(x1)))

    def _distance(self, x1, x2):
        """根据选定的距离度量计算距离"""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'chebyshev':
            return self._chebyshev_distance(x1, x2)
        else:
            raise ValueError(f"未知的距离度量: {self.distance_metric}")

    def fit(self, X, y):
        """存储训练数据（KNN 不进行显式训练）
        
        Args:
            X: 训练特征，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
        """
        if len(X) != len(y):
            raise ValueError("特征数和标签数不匹配")
        
        self.X_train = X
        self.y_train = y
        self.n_features = len(X[0]) if X else 0

    def predict_single(self, x):
        """预测单个样本的类别
        
        Args:
            x: 单个样本的特征向量
            
        Returns:
            预测的类别标签
        """
        if self.X_train is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        # 计算到所有训练样本的距离
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # 排序并选择最近的 K 个
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 获取 K 个最近邻的标签
        k_labels = [label for _, label in k_nearest]
        
        # 投票：选择最频繁的标签
        label_counts = Counter(k_labels)
        predicted_label = label_counts.most_common(1)[0][0]
        
        return predicted_label

    def predict(self, X):
        """预测多个样本的类别
        
        Args:
            X: 测试特征，形状 (n_samples, n_features)
            
        Returns:
            预测的类别标签列表
        """
        predictions = []
        for x in X:
            pred = self.predict_single(x)
            predictions.append(pred)
        
        return predictions

    def predict_proba_single(self, x):
        """预测单个样本属于各类别的概率
        
        Args:
            x: 单个样本的特征向量
            
        Returns:
            各类别的概率字典 {label: probability}
        """
        if self.X_train is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        # 计算到所有训练样本的距离
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # 排序并选择最近的 K 个
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 获取 K 个最近邻的标签
        k_labels = [label for _, label in k_nearest]
        
        # 计算概率：每个类别的频数 / K
        label_counts = Counter(k_labels)
        probabilities = {}
        for label in set(self.y_train):
            count = label_counts.get(label, 0)
            probabilities[label] = count / self.k
        
        return probabilities

    def predict_proba(self, X):
        """预测多个样本属于各类别的概率
        
        Args:
            X: 测试特征，形状 (n_samples, n_features)
            
        Returns:
            各样本的概率列表 [{'label1': prob, 'label2': prob, ...}, ...]
        """
        probabilities = []
        for x in X:
            proba = self.predict_proba_single(x)
            probabilities.append(proba)
        
        return probabilities


def load_iris_data(file_path):
    """加载 Iris 数据集
    
    数据格式：CSV 文件，最后一列为类别标签
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        X: 特征数据，形状 (n_samples, n_features)
        y: 类别标签，形状 (n_samples,)
        feature_names: 特征名称列表
        class_names: 类别名称列表
    """
    X = []
    y = []
    feature_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width'
    ]
    class_names = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # 跳过空行
                if not row:
                    continue
                
                try:
                    # 前 4 列是特征
                    features = [float(val) for val in row[:4]]
                    # 最后一列是类别
                    label = row[4].strip()
                    
                    X.append(features)
                    y.append(label)
                    
                    if label not in class_names:
                        class_names.append(label)
                
                except (ValueError, IndexError):
                    continue
    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None, None, None, None
    except Exception as e:
        print(f"错误：读取文件失败 - {e}")
        return None, None, None, None
    
    # 标准化类别名称
    class_names = sorted(class_names)
    
    return X, y, feature_names, class_names


def normalize_features(X, X_mean=None, X_std=None):
    """特征归一化 (Z-score 归一化)
    
    x_normalized = (x - mean) / std
    
    Args:
        X: 特征数据
        X_mean: 每个特征的均值（如果为 None，则计算）
        X_std: 每个特征的标准差（如果为 None，则计算）
        
    Returns:
        X_normalized: 归一化后的特征
        X_mean: 每个特征的均值
        X_std: 每个特征的标准差
    """
    n_features = len(X[0])
    
    # 计算均值和标准差
    if X_mean is None:
        X_mean = []
        for j in range(n_features):
            col = [X[i][j] for i in range(len(X))]
            mean = sum(col) / len(col)
            X_mean.append(mean)
    
    if X_std is None:
        X_std = []
        for j in range(n_features):
            col = [X[i][j] for i in range(len(X))]
            mean = X_mean[j]
            variance = sum((col[i] - mean) ** 2 for i in range(len(col))) / len(col)
            std = math.sqrt(variance)
            X_std.append(std if std > 0 else 1)
    
    # 归一化
    X_normalized = []
    for i in range(len(X)):
        normalized_sample = []
        for j in range(n_features):
            normalized_val = (X[i][j] - X_mean[j]) / X_std[j]
            normalized_sample.append(normalized_val)
        X_normalized.append(normalized_sample)
    
    return X_normalized, X_mean, X_std


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """划分训练集和测试集
    
    Args:
        X: 特征数据
        y: 标签
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        X_train, X_test, y_train, y_test
    """
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


def calculate_accuracy(y_true, y_pred):
    """计算准确率"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def calculate_confusion_matrix(y_true, y_pred, class_names):
    """计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        
    Returns:
        confusion_matrix: 混淆矩阵字典
    """
    confusion_matrix = {}
    for true_class in class_names:
        confusion_matrix[true_class] = {}
        for pred_class in class_names:
            confusion_matrix[true_class][pred_class] = 0
    
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    
    return confusion_matrix


def print_confusion_matrix(confusion_matrix, class_names):
    """打印混淆矩阵"""
    print("\n=== 混淆矩阵 ===")
    print("预测类别 ->")
    
    # 打印头部
    print("真实\\预测", end="")
    for class_name in class_names:
        print(f"\t{class_name[:10]}", end="")
    print()
    
    # 打印矩阵
    for true_class in class_names:
        print(f"{true_class[:10]}", end="")
        for pred_class in class_names:
            print(f"\t{confusion_matrix[true_class][pred_class]}", end="")
        print()


def calculate_metrics(y_true, y_pred, class_names):
    """计算分类指标（精确率、召回率、F1）"""
    metrics = {}
    
    for class_name in class_names:
        # 真正例 (TP)
        tp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == class_name and pred == class_name)
        
        # 假正例 (FP)
        fp = sum(1 for true, pred in zip(y_true, y_pred) 
                if true != class_name and pred == class_name)
        
        # 假负例 (FN)
        fn = sum(1 for true, pred in zip(y_true, y_pred) 
                if true == class_name and pred != class_name)
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for true in y_true if true == class_name)
        }
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN 分类算法')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='Iris 数据 CSV 文件路径')
    parser.add_argument('-k', type=int, default=5,
                        help='近邻数量 K')
    parser.add_argument('-m', '--metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'chebyshev'],
                        help='距离度量方式')
    parser.add_argument('-t', '--test_ratio', type=float, default=0.3,
                        help='测试集比例')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--normalize', action='store_true',
                        help='是否进行特征归一化')
    
    args = parser.parse_args()
    
    # 确定数据文件路径
    if args.file:
        data_file = args.file
    else:
        # 默认路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(script_dir, '..', 'data', 'iris.data')
    
    # 加载数据
    print("加载 Iris 数据集...")
    X, y, feature_names, class_names = load_iris_data(data_file)
    
    if X is None:
        print("无法加载数据，程序退出")
        exit(1)
    
    print(f"样本数: {len(X)}, 特征数: {len(X[0])}, 类别数: {len(class_names)}")
    print(f"特征: {feature_names}")
    print(f"类别: {class_names}")
    print(f"标签分布: {dict(sorted(Counter(y).items()))}")
    
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 特征归一化
    if args.normalize:
        print("进行特征归一化...")
        X_train, X_mean, X_std = normalize_features(X_train)
        X_test, _, _ = normalize_features(X_test, X_mean, X_std)
    
    # 训练 KNN 模型
    print(f"\n训练 KNN 模型 (k={args.k}, metric={args.metric})...")
    knn = KNN(k=args.k, distance_metric=args.metric)
    knn.fit(X_train, y_train)
    
    # 预测
    print("进行预测...")
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    # 评估
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    
    print(f"\n=== 准确率 ===")
    print(f"训练集: {train_accuracy:.4f}")
    print(f"测试集: {test_accuracy:.4f}")
    
    # 混淆矩阵
    confusion_matrix = calculate_confusion_matrix(y_test, y_test_pred, class_names)
    print_confusion_matrix(confusion_matrix, class_names)
    
    # 分类指标
    metrics = calculate_metrics(y_test, y_test_pred, class_names)
    print("\n=== 分类指标（测试集）===")
    print(f"{'类别':<20} {'精确率':<10} {'召回率':<10} {'F1':<10} {'样本数':<10}")
    print("-" * 60)
    for class_name in class_names:
        m = metrics[class_name]
        print(f"{class_name:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['support']:<10}")
    
    # 示例：预测单个样本并显示概率
    if len(X_test) > 0:
        print("\n=== 示例预测 ===")
        sample_idx = 0
        sample = X_test[sample_idx]
        true_label = y_test[sample_idx]
        pred_label = knn.predict_single(sample)
        probabilities = knn.predict_proba_single(sample)
        
        print(f"样本特征: {[f'{v:.2f}' for v in sample]}")
        print(f"真实标签: {true_label}")
        print(f"预测标签: {pred_label}")
        print(f"各类别概率:")
        for class_name in sorted(probabilities.keys()):
            prob = probabilities[class_name]
            print(f"  {class_name}: {prob:.4f}")
