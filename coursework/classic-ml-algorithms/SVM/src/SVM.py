#!/usr/bin/env python3
# 作者：23计算1Bohan Yu

import argparse
import math
import random
import os
import sys
import urllib.request
from collections import defaultdict


def load_iris_from_file(path):
    X = []
    y = []
    if not os.path.exists(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            try:
                features = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            label = parts[4]
            X.append(features)
            y.append(label)
    return X, y


def download_iris(destination):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    try:
        print('尝试从 UCI 下载 Iris 数据集...')
        urllib.request.urlretrieve(url, destination)
        print('下载完成:', destination)
        return True
    except Exception as e:
        print('无法下载 Iris 数据集:', e)
        return False


def train_test_split(X, y, test_size=0.2, seed=42):
    combined = list(zip(X, y))
    random.Random(seed).shuffle(combined)
    n_test = max(1, int(len(combined) * test_size))
    test = combined[:n_test]
    train = combined[n_test:]
    X_train, y_train = zip(*train) if train else ([], [])
    X_test, y_test = zip(*test) if test else ([], [])
    return list(X_train), list(X_test), list(y_train), list(y_test)


def zscore_normalize(train_X, test_X=None):
    if not train_X:
        return train_X, test_X
    n_features = len(train_X[0])
    means = [0.0] * n_features
    stds = [0.0] * n_features
    for j in range(n_features):
        col = [x[j] for x in train_X]
        means[j] = sum(col) / len(col)
        var = sum((v - means[j]) ** 2 for v in col) / len(col)
        stds[j] = math.sqrt(var) if var > 0 else 1.0
    def norm(X):
        return [[(x[j] - means[j]) / stds[j] for j in range(n_features)] for x in X]
    train_norm = norm(train_X)
    test_norm = norm(test_X) if test_X is not None and test_X != [] else None
    return train_norm, test_norm


class LinearSVM:
    def __init__(self, dim, lambda_reg=0.0001):
        self.dim = dim
        self.lambda_reg = lambda_reg
        self.w = [0.0] * dim
        self.b = 0.0

    def dot(self, x):
        return sum(self.w[i] * x[i] for i in range(self.dim)) + self.b

    def predict_raw(self, x):
        return self.dot(x)

    def predict(self, x):
        return 1 if self.predict_raw(x) >= 0 else -1

    def update(self, x, y, eta):
        # y in {+1, -1}
        decision = self.dot(x)
        if y * decision < 1:
            # subgradient for hinge loss
            # w <- (1 - eta*lambda)*w + eta*y*x
            for i in range(self.dim):
                self.w[i] = (1 - eta * self.lambda_reg) * self.w[i] + eta * y * x[i]
            self.b = self.b + eta * y
        else:
            for i in range(self.dim):
                self.w[i] = (1 - eta * self.lambda_reg) * self.w[i]
            # bias decays slightly (optional)


def train_pegasos(X, y, lambda_reg=0.0001, epochs=50, batch_size=1, seed=42, lr0=None):
    # X: list of feature lists
    # y: list of labels in {+1, -1}
    random.seed(seed)
    dim = len(X[0])
    svm = LinearSVM(dim, lambda_reg=lambda_reg)
    t = 0
    n = len(X)
    for epoch in range(epochs):
        indices = list(range(n))
        random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            t += 1
            # learning rate following Pegasos: eta = 1/(lambda * t)
            eta = 1.0 / (lambda_reg * t) if lr0 is None else lr0
            for idx in batch_idx:
                xi = X[idx]
                yi = y[idx]
                svm.update(xi, yi, eta)
    return svm


class OneVsRestSVM:
    def __init__(self, labels, lambda_reg=0.0001):
        self.labels = list(labels)
        self.models = {lab: None for lab in self.labels}
        self.lambda_reg = lambda_reg

    def fit(self, X, y, **kwargs):
        # y are original labels (strings or ints)
        for lab in self.labels:
            y_binary = [1 if yy == lab else -1 for yy in y]
            model = train_pegasos(X, y_binary, lambda_reg=self.lambda_reg, **kwargs)
            self.models[lab] = model

    def predict(self, X):
        preds = []
        for x in X:
            scores = {lab: self.models[lab].predict_raw(x) for lab in self.labels}
            # choose label with largest score
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            preds.append(best)
        return preds


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def main():
    parser = argparse.ArgumentParser(description='Simple SVM (Pegasos) for Iris, no sklearn')
    parser.add_argument('-f', '--file', type=str, default=None, help='Iris data file path')
    parser.add_argument('--lambda_reg', type=float, default=0.0001, help='Regularization lambda')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample', action='store_true', help='Run built-in sample (download if needed)')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit samples for speed')
    args = parser.parse_args()

    data_file = args.file
    if args.sample or data_file is None:
        data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.data')
        data_file = os.path.normpath(data_file)
        if not os.path.exists(data_file):
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            ok = download_iris(data_file)
            if not ok:
                print('无法获取 Iris 数据集，请手动将 iris.data 放入', data_file)
                sys.exit(1)

    X, y = load_iris_from_file(data_file)
    if X is None:
        print('未找到 Iris 数据，请确保文件存在或使用 --sample 以自动下载。')
        sys.exit(1)

    if args.max_samples is not None:
        X = X[:args.max_samples]
        y = y[:args.max_samples]

    # Map labels to themselves (strings) and get label set
    labels = sorted(set(y))
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, seed=args.seed)
    X_train, X_test = zscore_normalize(X_train, X_test)

    print('训练样本数:', len(X_train), '测试样本数:', len(X_test))
    print('类别:', labels)

    ovr = OneVsRestSVM(labels, lambda_reg=args.lambda_reg)
    ovr.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    y_pred = ovr.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('测试准确率: {:.4f}'.format(acc))

    # show a few predictions
    print('\n示例预测（真实 -> 预测）:')
    for i in range(min(10, len(X_test))):
        print(y_test[i], '->', y_pred[i])


if __name__ == '__main__':
    main()
