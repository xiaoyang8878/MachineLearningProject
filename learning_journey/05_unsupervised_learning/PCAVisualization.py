# 文件名: PCAVisualization.py
# 目的: 绘制累计解释方差曲线，辅助判断保留多少个主成分
# 用法: 曲线拐点处即为推荐的主成分数量

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# 计算所有主成分（最多 min(样本数, 特征数) 个）
pca = PCA()
pca.fit(X_scaled)

# 累计方差解释率
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 绘制曲线
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差线')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释率')
plt.title('PCA 累计方差解释率曲线')
plt.legend()
plt.grid(True)
plt.show()