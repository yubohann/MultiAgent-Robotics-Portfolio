# Author: Bohan Yu

import csv
import math
import random
import argparse
import os
from typing import List, Tuple


def load_magic_data(file_path: str) -> Tuple[List[List[float]], List[int]]:
    """Load the Magic Gamma Telescope dataset from CSV and map labels to +1 and -1."""
    X = []
    y = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            try:
                feats = [float(v) for v in row[:-1]]
            except ValueError:

                continue
            label_raw = row[-1].strip()
            # This dataset labels rows with 'g' or 'h'; numeric 1/0 forms also appear.
            if label_raw.lower() in ('g', 'gamma'):
                label = 1
            elif label_raw.lower() in ('h', 'hadron'):
                label = -1
            else:

                try:
                    lv = float(label_raw)
                    label = 1 if lv > 0 else -1
                except Exception:
                    # Fallback mapping, positive labels to 1 and everything else to -1.
                    label = 1 if label_raw == '1' else -1
            X.append(feats)
            y.append(label)
    return X, y


class DecisionStump:
    """Decision stump that splits one feature on a threshold into +1 and -1."""

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
        n_thresholds = 15  # Cap the candidate thresholds to keep the sweep fast.
        

        for feature in range(n_features):
            feature_values = [x[feature] for x in X]
            min_val = min(feature_values)
            max_val = max(feature_values)
            

            if min_val == max_val:
                continue
            

            thresholds = [min_val + (max_val - min_val) * i / (n_thresholds - 1) for i in range(n_thresholds)]

            for thr in thresholds:
                # A single polarity keeps the stump search simple.
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

        w = [1.0 / n_samples] * n_samples

        for _ in range(self.n_clf):
            stump = DecisionStump()
            stump.fit(X, y, w)
            preds = stump.predict(X)

            error = sum(wi for wi, pi, yi in zip(w, preds, y) if pi != yi)
            # Numerical guard.
            error = max(1e-10, min(error, 1 - 1e-10))
            alpha = 0.5 * math.log((1 - error) / error)


            for i in range(n_samples):
                w[i] = w[i] * math.exp(-alpha * y[i] * preds[i])

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
    # A simple linearly separable set that is hard for a single weak learner.
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

        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'magic04.data'))
        if os.path.exists(default_path):
            args.file = default_path

    if args.file is None or not os.path.exists(args.file):
        print('未找到 Magic 数据文件，运行内置示例以演示 AdaBoost。')
        demo_on_toy()
        raise SystemExit(0)

    X, y = load_magic_data(args.file)

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
