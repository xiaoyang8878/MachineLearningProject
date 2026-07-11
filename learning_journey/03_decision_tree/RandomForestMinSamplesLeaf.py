# 目的: 调整 min_samples_leaf 参数（默认1 → 5），观察"单间"是否消失，边界是否更平滑

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 唯一改动：增加 min_samples_leaf=5
model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

print(f"训练准确率: {model.score(X_train, y_train):.3f}")
print(f"测试准确率: {model.score(X_test, y_test):.3f}")

# 画边界
def plot_decision_boundary():
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = ['#FF0000', '#0088FF', '#00CC00']
    cmap = ListedColormap(colors)
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=20)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='^', edgecolor='k', s=20)
    plt.xlabel('花瓣长度')
    plt.ylabel('花瓣宽度')
    plt.title(f'随机森林 (100棵树, min_samples_leaf=5)')
    plt.show()

plot_decision_boundary()