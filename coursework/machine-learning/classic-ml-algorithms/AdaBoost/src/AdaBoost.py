# 作者：23计算1Bohan Yu

import csv
import math
import random
import argparse
import os
from typing import List, Tuple


def load_magic_data(file_path: str) -> Tuple[List[List[float]], List[int]]:
    """从 CSV 加载 Magic Gamma Telescope 数据。

    假定每行末列为标签（例如 'g' 或 'h'），其它列为数值特征。
    将标签映射为 +1（gamma）和 -1（hadron）。
    """
    X = []
    y = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # 跳过注释或非数据行（如果有）
            try:
                feats = [float(v) for v in row[:-1]]
            except ValueError:
                # 可能是 header/注释
                continue
            label_raw = row[-1].strip()
            # 常见数据集中用 'g'/'h'，也可能直接用 1/0 或 1/-1
            if label_raw.lower() in ('g', 'gamma'):
                label = 1
            elif label_raw.lower() in ('h', 'hadron'):
                label = -1
            else:
                # 尝试数值转换
                try:
                    lv = float(label_raw)
                    label = 1 if lv > 0 else -1
                except Exception:
                    # 默认映射：常见 'g'->1, 其它->-1
                    label = 1 if label_raw == '1' else -1
            X.append(feats)
            y.append(label)
    return X, y


class DecisionStump:
    """简单的决策桩：基于某个特征和阈值进行二分类（+1 / -1）。"""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1  # polarity: if polarity * x[f] < polarity * threshold => predict 1

    def predict(self, X: List[List[float]]) -> List[int]:
        preds = []
        for x in X:
            val = x[self.feature_index]
            pred = 1 if val < self.threshold else -1
            preds.append(pred * self.polarity)
        return preds

    def fit(self, X: List[List[float]], y: List[int], weights: List[float]):
        n_samples = len(X)
        n_features = len(X[0])
        best_error = float('inf')
        n_thresholds = 15  # 限制候选阈值数量以加快速度
        
        # 遍历每个特征，尝试分位点采样的阈值
        for feature in range(n_features):
            feature_values = [x[feature] for x in X]
            min_val = min(feature_values)
            max_val = max(feature_values)
            
            # 若最大值等于最小值，跳过
            if min_val == max_val:
                continue
            
            # 生成均匀分布的候选阈值
            thresholds = [min_val + (max_val - min_val) * i / (n_thresholds - 1) for i in range(n_thresholds)]

            for thr in thresholds:
                # 用单个polarity尝试（简化：不再尝试两个polarity）
                error = sum(weights[i] for i in range(n_samples) if (1 if X[i][feature] < thr else -1) != y[i])
                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = thr
                    self.polarity = 1
        return self


class AdaBoost:
    def __init__(self, n_clf: int = 50):
        self.n_clf = n_clf
        self.clfs = []
        self.alphas = []

    def fit(self, X: List[List[float]], y: List[int]):
        n_samples = len(X)
        # 初始化权重
        w = [1.0 / n_samples] * n_samples

        for _ in range(self.n_clf):
            stump = DecisionStump()
            stump.fit(X, y, w)
            preds = stump.predict(X)
            # 计算误差
            error = sum(wi for wi, pi, yi in zip(w, preds, y) if pi != yi)
            # 避免数值问题
            error = max(1e-10, min(error, 1 - 1e-10))
            alpha = 0.5 * math.log((1 - error) / error)

            # 更新权重
            for i in range(n_samples):
                w[i] = w[i] * math.exp(-alpha * y[i] * preds[i])
            # 归一化
            s = sum(w)
            w = [wi / s for wi in w]

            self.clfs.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        n_samples = len(X)
        agg = [0.0] * n_samples
        for alpha, clf in zip(self.alphas, self.clfs):
            preds = clf.predict(X)
            for i, p in enumerate(preds):
                agg[i] += alpha * p
        return [1 if a >= 0 else -1 for a in agg]


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def demo_on_toy():
    # 简单演示：一个线性可分但弱学习难以直接分开的集合
    X = [[0.1], [0.2], [0.4], [0.6], [0.8], [1.0]]
    y = [1, 1, 1, -1, -1, -1]
    model = AdaBoost(n_clf=5)
    model.fit(X, y)
    preds = model.predict(X)
    print('toy true:', y)
    print('toy pred:', preds)
    print('toy acc:', accuracy_score(y, preds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AdaBoost (纯 Python 实现)')
    parser.add_argument('-f', '--file', help='Magic Gamma 数据 CSV 文件路径', default=None)
    parser.add_argument('-n', '--n_estimators', type=int, default=50, help='弱分类器数量')
    parser.add_argument('-t', '--test_size', type=float, default=0.3, help='测试集比例（0-1）')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.file is None:
        # 尝试默认位置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'magic04.data'))
        if os.path.exists(default_path):
            args.file = default_path

    if args.file is None or not os.path.exists(args.file):
        print('未找到 Magic 数据文件，运行内置示例以演示 AdaBoost。')
        demo_on_toy()
        raise SystemExit(0)

    X, y = load_magic_data(args.file)
    # 简单划分训练/测试
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    n_test = int(len(X) * args.test_size)
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    print(f'加载数据: 样本数={len(X)}, 特征数={len(X[0]) if X else 0}, 训练={len(X_train)}, 测试={len(X_test)}')

    model = AdaBoost(n_clf=args.n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'AdaBoost 测试准确率: {acc:.4f} (n_estimators={args.n_estimators})')
