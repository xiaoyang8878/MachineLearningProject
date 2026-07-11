# 文件名: KMeansElbowMethod.py
# 目的: 用肘部法则辅助选择 K 值

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]

inertias = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)  # 总误差（越小越好，但要注意边际收益）

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K 值')
plt.ylabel('总误差（inertia）')
plt.title('肘部法则：误差随 K 值变化的曲线')
plt.grid(True)
plt.show()