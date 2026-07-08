# 目的: 并排展示决策树（矩形边界）与逻辑回归（斜线边界）的决策区域对比


import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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

# 训练两个模型
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

model_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
model_tree.fit(X_train, y_train)

# 定义画决策边界的函数
def plot_decision_boundary(ax, model, title):
    # 生成网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 颜色（红、蓝、绿）
    colors = ['#FF0000', '#0088FF', '#00CC00']
    cmap = ListedColormap(colors)
    
    # 画背景色块
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    
    # 画训练集和测试集
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=30, label='训练集')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='^', edgecolor='k', s=30, label='测试集')
    
    ax.set_xlabel('花瓣长度 (cm)')
    ax.set_ylabel('花瓣宽度 (cm)')
    ax.set_title(title)
    ax.legend()

# 创建左右并排的图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

plot_decision_boundary(ax1, model_lr, '逻辑回归（直线边界）')
plot_decision_boundary(ax2, model_tree, '决策树（矩形边界）')

plt.tight_layout()
plt.show()