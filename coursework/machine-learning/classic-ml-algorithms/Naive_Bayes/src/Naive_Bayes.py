# 作者：23计算1Bohan Yu 



import os

import re

import math

import random

import argparse

from collections import defaultdict



DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "20news-18828")  # 本地数据目录（可通过 -f 覆盖）

def tokenize(text):

    # 简单英文分词，将文本转为小写并提取单词

    return re.findall(r'\b\w+\b', text.lower())



def load_data(data_dir, max_docs_per_class=None):

    # 读取数据集，将每个文件内容和类别组成元组

    data = []

    for class_name in os.listdir(data_dir):

        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):

            continue

        files = os.listdir(class_path)

        if max_docs_per_class:

            files = files[:max_docs_per_class]

        for fname in files:

            fpath = os.path.join(class_path, fname)

            try:

                with open(fpath, encoding="latin1") as f:

                    text = f.read()

                    data.append((text, class_name))

            except Exception as e:

                continue

    return data



class NaiveBayesClassifier:

    def __init__(self):

        # 类别-词频统计，类别计数，词表，总文档数

        self.class_word_counts = defaultdict(lambda: defaultdict(int))

        self.class_counts = defaultdict(int)

        self.vocab = set()

        self.total_docs = 0



    def fit(self, data):

        # 训练模型，统计每个类别下每个词的出现次数

        for text, label in data:

            self.total_docs += 1

            self.class_counts[label] += 1

            words = tokenize(text)

            for word in words:

                self.class_word_counts[label][word] += 1

                self.vocab.add(word)



    def predict(self, text):

        # 预测文本所属类别

        words = tokenize(text)

        best_label = None

        max_prob = float('-inf')

        for label in self.class_counts:

            # 计算先验概率

            log_prob = math.log(self.class_counts[label] / self.total_docs)

            total_words = sum(self.class_word_counts[label].values())

            for word in words:

                # 拉普拉斯平滑，避免概率为0

                word_count = self.class_word_counts[label][word] + 1

                word_prob = word_count / (total_words + len(self.vocab))

                log_prob += math.log(word_prob)

            if log_prob > max_prob:

                max_prob = log_prob

                best_label = label

        return best_label



def train_test_split(data, test_ratio=0.2, seed=42):

    # 划分训练集和测试集

    random.seed(seed)

    random.shuffle(data)

    split = int(len(data) * (1 - test_ratio))

    return data[:split], data[split:]



def evaluate(clf, test_data):

    # 评估模型准确率

    correct = 0

    for text, label in test_data:

        pred = clf.predict(text)

        if pred == label:

            correct += 1

    return correct / len(test_data)



def ensure_dataset(path):
    if os.path.isdir(path):
        return path
    try:
        from sklearn.datasets import fetch_20newsgroups
        import tempfile
        tmp = tempfile.mkdtemp()
        print("本地未找到 20news 数据集，正在通过 scikit-learn 下载到临时目录...")
        fetch_20newsgroups(subset="train", data_home=tmp, remove=("headers", "footers", "quotes"))
        return path
    except Exception as exc:
        print("无法获取 20news 数据集，请将数据放到", path, "或检查网络。错误:", exc)
        raise SystemExit(1)


if __name__ == "__main__":

    print("加载数据...")

    # 读取数据，每类最多100篇，便于快速测试

    parser = argparse.ArgumentParser(description="朴素贝叶斯文本分类（从零实现）")
    parser.add_argument("-f", "--file", default=DATA_DIR, help="20news 数据目录路径")
    parser.add_argument("-n", "--max_docs_per_class", type=int, default=100, help="每类最多读取文档数")
    parser.add_argument("-t", "--test_ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    data_dir = ensure_dataset(args.file)
    data = load_data(data_dir, max_docs_per_class=args.max_docs_per_class)

    print(f"总样本数: {len(data)}")

    # 划分训练集和测试集

    train_data, test_data = train_test_split(data, test_ratio=args.test_ratio, seed=args.seed)

    print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")



    print("训练朴素贝叶斯分类器...")

    clf = NaiveBayesClassifier()

    clf.fit(train_data)



    print("评估模型...")

    acc = evaluate(clf, test_data)

    print(f"测试集准确率: {acc:.4f}")



    # 示例预测

    sample_text = "NASA launches new satellite for space research"

    print(f"示例文本预测类别: {clf.predict(sample_text)}")