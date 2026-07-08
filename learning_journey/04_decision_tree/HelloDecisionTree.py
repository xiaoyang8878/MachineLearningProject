# 目的: 绘制决策树结构图，展示每个节点的"是/否"判断路径


import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端，解决弹窗报错

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 加载数据集（鸢尾花数据集）
iris = load_iris()
X = iris.data[:, 2:]  # 只取花瓣长度和花瓣宽度两个特征，便于可视化
y = iris.target  # 目标变量（花的类别），0、1、2分别代表3种不同的花

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建决策树模型(限制深度为3，防止它疯狂细分)
model = DecisionTreeClassifier(max_depth=3, random_state=42)    # random_state=42: 随机种子，保证每次运行结果一致
model.fit(X_train, y_train)  # 训练模型

# 4. 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['花瓣长', '花瓣宽'], class_names=iris.target_names, filled=True)
plt.show()