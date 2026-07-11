# 文件名: HelloKMeans.py
# 目的: K-Means 聚类演示（鸢尾花数据集，不使用标签）
# 对比: 聚类的颜色分配 vs 真实的品种标签

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据（只取花瓣长、宽，便于二维可视化）
iris = load_iris()
X = iris.data[:, :2]   # 只有 X，没有 y（聚类不看标签）
y_true = iris.target   # 仅用于后续对比，K-Means 训练时不使用

# 2. 创建 K-Means 模型，指定 K=3
model = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = model.fit_predict(X)  # 返回每个样本所属的簇编号（0,1,2）

# 3. 对比真实标签和聚类结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：真实标签（有监督的“标准答案”）
colors = ['#FF0000', '#0088FF', '#00CC00']
cmap = ListedColormap(colors)
ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, edgecolor='k', s=30)
ax1.set_xlabel('花瓣长度')
ax1.set_ylabel('花瓣宽度')
ax1.set_title('真实标签（三种花）')

# 右图：K-Means 聚类结果（无监督）
ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=30)
# 画出质心位置（黑色大十字）
centers = model.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], marker='X', color='black', s=200, linewidths=3)
ax2.set_xlabel('花瓣长度')
ax2.set_ylabel('花瓣宽度')
ax2.set_title('K-Means 聚类结果（K=3）')

plt.tight_layout()
plt.show()

# 4. 打印簇中心坐标
print("三个簇的质心坐标（花瓣长, 花瓣宽）：")
for i, center in enumerate(centers):
    print(f"  簇 {i}: ({center[0]:.2f}, {center[1]:.2f})")