# 文件名: 06_kmeans_silhouette_limitations.py
# 目的: 完整演示 K-Means 局限性、轮廓系数选K、PCA局限性
# 包含: 长条形/月牙形数据失效、轮廓系数 vs 肘部法则、PCA vs t-SNE

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：K-Means 在非球形数据上的失效
# ============================================================
np.random.seed(42)

# 生成长条形数据（各向异性）
X_elongated, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0)
X_elongated[:, 0] = X_elongated[:, 0] * 4  # x轴拉伸4倍

# 生成月牙形数据（非线性）
X_moons, _ = make_moons(n_samples=300, noise=0.05)

# 分别对两种数据做 K-Means（K=2）
kmeans_elong = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_elong = kmeans_elong.fit_predict(X_elongated)

kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_moons = kmeans_moons.fit_predict(X_moons)

# ============================================================
# 第二部分：轮廓系数 vs 肘部法则（用鸢尾花数据）
# ============================================================
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data[:, :2]  # 只取前两维，便于可视化

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_iris)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_iris, labels))

# ============================================================
# 第三部分：PCA vs t-SNE 在非线性数据上的对比（瑞士卷）
# ============================================================
X_swiss, t_swiss = make_swiss_roll(n_samples=800, noise=0.0, random_state=42)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_swiss)

# t-SNE 降维（非线性方法）
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_swiss)

# ============================================================
# 画图：所有结果统一展示
# ============================================================
fig = plt.figure(figsize=(16, 10))

# 1. 长条形数据失效
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(X_elongated[:, 0], X_elongated[:, 1], c=labels_elong, cmap='viridis', alpha=0.7)
ax1.scatter(kmeans_elong.cluster_centers_[:, 0], kmeans_elong.cluster_centers_[:, 1],
            marker='X', color='red', s=200)
ax1.set_title('K-Means 在长条形数据上的分割（硬切）')

# 2. 月牙形数据失效
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', alpha=0.7)
ax2.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
            marker='X', color='red', s=200)
ax2.set_title('K-Means 在月牙形数据上的分割（硬切）')

# 3. 肘部法则曲线
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(K_range, inertias, 'bo-')
ax3.set_xlabel('K 值')
ax3.set_ylabel('总误差 (inertia)')
ax3.set_title('肘部法则：找拐点')

# 4. 轮廓系数曲线
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(K_range, silhouettes, 'ro-')
ax4.set_xlabel('K 值')
ax4.set_ylabel('轮廓系数')
ax4.set_title('轮廓系数：越高越好，选最高点')
best_k = K_range[np.argmax(silhouettes)]
ax4.axvline(x=best_k, color='green', linestyle='--', label=f'最佳 K={best_k}')
ax4.legend()

# 5. PCA 降维结果（瑞士卷）
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(X_pca[:, 0], X_pca[:, 1], c=t_swiss, cmap='viridis', alpha=0.7)
ax5.set_title('PCA 降维瑞士卷（线性 → 混成一团）')

# 6. t-SNE 降维结果（瑞士卷）
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t_swiss, cmap='viridis', alpha=0.7)
ax6.set_title('t-SNE 降维瑞士卷（非线性 → 清晰展开）')

plt.tight_layout()
plt.show()

# 打印结果
print("=" * 50)
print("【轮廓系数选K结果】")
print(f"最佳 K 值: {best_k}")
print(f"对应轮廓系数: {max(silhouettes):.4f}")
print("=" * 50)