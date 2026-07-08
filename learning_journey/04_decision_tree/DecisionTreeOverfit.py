# 目的: 演示深度（max_depth）如何影响过拟合。深度越大，训练准确率越高，测试准确率可能越低。

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据（只用花瓣长、宽）
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练两个不同深度的模型
model_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
model_shallow.fit(X_train, y_train)

model_deep = DecisionTreeClassifier(max_depth=20, random_state=42)  # 20层，几乎放开了切
model_deep.fit(X_train, y_train)

# 计算准确率
train_acc_shallow = model_shallow.score(X_train, y_train)
test_acc_shallow = model_shallow.score(X_test, y_test)
train_acc_deep = model_deep.score(X_train, y_train)
test_acc_deep = model_deep.score(X_test, y_test)

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
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=20)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='^', edgecolor='k', s=20)
    ax.set_xlabel('花瓣长度')
    ax.set_ylabel('花瓣宽度')
    ax.set_title(f'{title}\n训练准确率: {train_acc:.3f} | 测试准确率: {test_acc:.3f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_decision_boundary(ax1, model_shallow, '深度=3（保守）', train_acc_shallow, test_acc_shallow)
plot_decision_boundary(ax2, model_deep, '深度=20（放开切）', train_acc_deep, test_acc_deep)
plt.tight_layout()
plt.show()