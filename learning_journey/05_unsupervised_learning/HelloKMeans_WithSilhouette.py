# 文件名: HelloKMeans_WithSilhouette.py
# 目的: 用轮廓系数曲线辅助选择最佳 K 值

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]  # 只取花瓣长、宽

# ===== 循环计算 K=2 到 K=8 的轮廓系数 =====
silhouette_scores = []
K_range = range(2, 9)  # 轮廓系数要求 K>=2

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k} -> 轮廓系数: {score:.4f}")

# ===== 画曲线找最高点 =====
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('K 值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数曲线（越高越好）')
plt.grid(True)

# 标记最高点
best_k = K_range[np.argmax(silhouette_scores)]
plt.axvline(x=best_k, color='green', linestyle='--', label=f'最佳 K={best_k}')
plt.legend()
plt.show()

print(f"\n最佳 K 值为: {best_k}")