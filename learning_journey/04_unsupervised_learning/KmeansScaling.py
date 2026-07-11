# 目的: 演示特征缩放（StandardScaler）对 K-Means 聚类结果的影响
# 对比: 左图（原始数据，收入主导聚类）vs 右图（缩放后，双特征均衡）

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成两组特征：收入（范围大）和年龄（范围小）
np.random.seed(42)
n_samples = 300
income = np.random.normal(50000, 15000, n_samples)   # 均值5万，标准差1.5万
age = np.random.normal(35, 10, n_samples)            # 均值35岁，标准差10岁

X = np.column_stack((income, age))

# 2. 分别用原始数据和缩放后数据跑 K-Means（K=3）
kmeans_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_raw = kmeans_raw.fit_predict(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# 3. 画图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始数据
ax1.scatter(X[:, 0], X[:, 1], c=labels_raw, cmap='viridis', alpha=0.7)
ax1.scatter(kmeans_raw.cluster_centers_[:, 0], kmeans_raw.cluster_centers_[:, 1], 
            marker='X', color='red', s=200, label='质心')
ax1.set_xlabel('年收入（元）')
ax1.set_ylabel('年龄（岁）')
ax1.set_title('原始数据：收入范围大，年龄被忽略')
ax1.legend()

# 右图：缩放后数据（注意：坐标轴是缩放后的无量纲值，没有单位）
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled, cmap='viridis', alpha=0.7)
ax2.scatter(kmeans_scaled.cluster_centers_[:, 0], kmeans_scaled.cluster_centers_[:, 1], 
            marker='X', color='red', s=200, label='质心')
ax2.set_xlabel('年收入（标准化）')
ax2.set_ylabel('年龄（标准化）')
ax2.set_title('标准化后：收入与年龄共同决定聚类')
ax2.legend()

plt.tight_layout()
plt.show()