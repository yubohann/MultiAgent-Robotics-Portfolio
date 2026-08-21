# 作者：23计算1Bohan Yu 

import os
import csv
import math
import random
import argparse
from collections import Counter, defaultdict


def load_iris_data(file_path=None):
	if file_path is None:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		file_path = os.path.normpath(os.path.join(script_dir, '..', 'data', 'iris.data'))

	X = []
	y = []
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"Iris 数据文件未找到: {file_path}")
	with open(file_path, 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue
			# 行格式：4 个数值 + label
			try:
				feats = [float(x) for x in row[:4]]
			except ValueError:
				continue
			label = row[4] if len(row) > 4 else ''
			X.append(feats)
			y.append(label)
	return X, y


def euclidean(a, b):
	return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KMeans:
	def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='kmeans++', random_state=None):
		self.n_clusters = int(n_clusters)
		self.max_iter = int(max_iter)
		self.tol = float(tol)
		assert init in ('random', 'kmeans++')
		self.init = init
		self.random_state = random_state
		if random_state is not None:
			random.seed(random_state)
		self.cluster_centers_ = []
		self.labels_ = []
		self.inertia_ = None

	def _init_centroids(self, X):
		n_samples = len(X)
		if self.init == 'random':
			return [list(X[i]) for i in random.sample(range(n_samples), self.n_clusters)]

		# k-means++ initialization
		centers = []
		# choose first center uniformly
		first_idx = random.randrange(n_samples)
		centers.append(list(X[first_idx]))
		# choose remaining
		for _ in range(1, self.n_clusters):
			# compute squared distances to nearest center
			dists = []
			for x in X:
				min_d = min((euclidean(x, c) ** 2) for c in centers)
				dists.append(min_d)
			total = sum(dists)
			if total == 0:
				# fallback: random choice
				centers.append(list(X[random.randrange(n_samples)]))
				continue
			# choose index with probability proportional to distance squared
			r = random.random() * total
			cum = 0.0
			for idx, val in enumerate(dists):
				cum += val
				if cum >= r:
					centers.append(list(X[idx]))
					break
		return centers

	def fit(self, X):
		if len(X) == 0:
			raise ValueError('Empty data')
		n_features = len(X[0])
		# initialize centers
		centers = self._init_centroids(X)

		for it in range(self.max_iter):
			clusters = [[] for _ in range(self.n_clusters)]
			labels = [None] * len(X)
			# assign
			for i, x in enumerate(X):
				dists = [euclidean(x, c) for c in centers]
				label = min(range(self.n_clusters), key=lambda j: dists[j])
				clusters[label].append(x)
				labels[i] = label

			# update
			new_centers = []
			for k in range(self.n_clusters):
				if clusters[k]:
					# mean
					mean = [sum(col) / len(clusters[k]) for col in zip(*clusters[k])]
					new_centers.append(mean)
				else:
					# empty cluster -> reinitialize to random point
					new_centers.append(list(X[random.randrange(len(X))]))

			# check convergence
			shifts = [euclidean(c, nc) for c, nc in zip(centers, new_centers)]
			centers = new_centers
			if max(shifts) <= self.tol:
				break

		# final assignment and inertia
		self.cluster_centers_ = centers
		self.labels_ = [min(range(self.n_clusters), key=lambda j: euclidean(x, centers[j])) for x in X]
		self.inertia_ = sum(euclidean(x, centers[self.labels_[i]]) ** 2 for i, x in enumerate(X))
		return self

	def predict(self, X):
		if not self.cluster_centers_:
			raise ValueError('Model not fitted')
		return [min(range(self.n_clusters), key=lambda j: euclidean(x, self.cluster_centers_[j])) for x in X]

	def fit_predict(self, X):
		self.fit(X)
		return self.labels_


def purity_score(true_labels, pred_labels):
	"""计算聚类纯度：将每个簇映射到最多数的真实标签，然后计算总体正确率"""
	cluster_to_labels = defaultdict(list)
	for cl, tl in zip(pred_labels, true_labels):
		cluster_to_labels[cl].append(tl)
	correct = 0
	for cl, labs in cluster_to_labels.items():
		if not labs:
			continue
		most_common, cnt = Counter(labs).most_common(1)[0]
		correct += cnt
	return correct / len(true_labels)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='K-means 聚类（纯 Python 实现）')
	parser.add_argument('-k', '--n_clusters', type=int, default=3, help='簇数 K')
	parser.add_argument('-i', '--init', choices=['random', 'kmeans++'], default='kmeans++', help='初始化方法')
	parser.add_argument('-m', '--max_iter', type=int, default=300, help='最大迭代次数')
	parser.add_argument('-t', '--tol', type=float, default=1e-4, help='收敛容差')
	parser.add_argument('-f', '--file', default=None, help='Iris 数据文件路径（默认使用 K-means/data/iris.data）')
	parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子')
	args = parser.parse_args()

	# 加载数据
	X, y = load_iris_data(args.file)
	print(f'样本数: {len(X)}, 特征数: {len(X[0]) if X else 0}, 标签类数: {len(set(y))}')

	km = KMeans(n_clusters=args.n_clusters, max_iter=args.max_iter, tol=args.tol, init=args.init, random_state=args.seed)
	km.fit(X)
	labels = km.labels_
	print(f'簇中心 (共 {len(km.cluster_centers_)} 个):')
	for idx, c in enumerate(km.cluster_centers_):
		print(f'  C{idx}: {[round(v, 4) for v in c]}')

	counts = Counter(labels)
	print('每簇样本数:')
	for k in range(args.n_clusters):
		print(f'  簇{k}: {counts.get(k,0)}')

	print(f'Inertia (簇内平方和): {km.inertia_:.4f}')
	if y:
		pur = purity_score(y, labels)
		print(f'聚类纯度 (purity): {pur:.4f}')

	# 显示每个簇中最常见的真实标签
	cluster_labels = defaultdict(list)
	for cl, tl in zip(labels, y):
		cluster_labels[cl].append(tl)
	print('每簇最常见真实标签:')
	for cl in sorted(cluster_labels.keys()):
		most = Counter(cluster_labels[cl]).most_common(3)
		print(f'  簇{cl}: {most}')
