
# 目的: 训练随机森林（全部4个特征），输出特征重要性分数，判断哪些特征对分类贡献最大

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据（这次用全部 4 个特征）
iris = load_iris()
X = iris.data          # 4个特征：花萼长、花萼宽、花瓣长、花瓣宽
y = iris.target
feature_names = iris.feature_names

# 2. 切分 + 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. 提取特征重要性
importances = model.feature_importances_

# 4. 打印
print("特征重要性排序（从高到低）：")
for name, score in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {score:.4f}")

# 5. 画柱状图
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('重要性分数')
plt.title('随机森林特征重要性 (100棵树)')
plt.gca().invert_yaxis()  # 让最重要的显示在最上面
plt.show()

# 6. 顺便看一眼准确率
print(f"\n测试集准确率: {model.score(X_test, y_test):.3f}")