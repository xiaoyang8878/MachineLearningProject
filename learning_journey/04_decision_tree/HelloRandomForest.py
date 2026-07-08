"""
随机森林的两个核心“随机”点（只讲事实）
    1.随机选样本（Bootstrap）：每棵树训练时，不是用全部训练数据，
      而是有放回地随机抽取一部分样本（大约 2/3）。这样每棵树看到的数据都略有不同，不会全部记住同一批噪声。

随机选特征：
    2.普通决策树在分裂时，会考察所有特征，选出最好的那个。
      随机森林强制每棵树在分裂时只随机考察一小部分特征（比如总共 4 个特征，每棵树只看 2 个），从中选最好的。这样树与树之间的差异更大，投票时更稳健。
"""

# 目的: 对比单棵树 vs 随机森林（10棵 vs 100棵），展示投票机制如何抹平"单间"、稳定边界

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练三个模型
model_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
model_tree.fit(X_train, y_train)

model_rf_10 = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
model_rf_10.fit(X_train, y_train)

model_rf_100 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_rf_100.fit(X_train, y_train)

# 计算准确率
acc_tree_train = model_tree.score(X_train, y_train)
acc_tree_test = model_tree.score(X_test, y_test)
acc_rf10_train = model_rf_10.score(X_train, y_train)
acc_rf10_test = model_rf_10.score(X_test, y_test)
acc_rf100_train = model_rf_100.score(X_train, y_train)
acc_rf100_test = model_rf_100.score(X_test, y_test)

def plot_decision_boundary(ax, model, title, train_acc, test_acc):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    colors = ['#FF0000', '#0088FF', '#00CC00']
    cmap = ListedColormap(colors)
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=15)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='^', edgecolor='k', s=15)
    ax.set_xlabel('花瓣长度')
    ax.set_ylabel('花瓣宽度')
    ax.set_title(f'{title}\n训练: {train_acc:.3f} | 测试: {test_acc:.3f}')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plot_decision_boundary(ax1, model_tree, '单棵决策树 (深度=10)', acc_tree_train, acc_tree_test)
plot_decision_boundary(ax2, model_rf_10, '随机森林 (10棵树)', acc_rf10_train, acc_rf10_test)
plot_decision_boundary(ax3, model_rf_100, '随机森林 (100棵树)', acc_rf100_train, acc_rf100_test)
plt.tight_layout()
plt.show()