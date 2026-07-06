# =============================================================================
# 第二课：线性回归（预测连续数字）
# =============================================================================
# 场景：根据房屋面积（平米）预测房价（万元）
# 核心目标：找到一条直线 y = kx + b，让所有点到直线的“竖直距离”最小
# =============================================================================
import matplotlib
matplotlib.use('TkAgg')  # 解决弹窗报错
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']

# ===== 1. 生成数据：两个特征 =====
np.random.seed(42)
n = 60
面积 = np.random.uniform(50, 100, n)
卧室数 = np.random.randint(1, 5, n)

# 真实规律：房价 = 2.5*面积 + 8*卧室数 + 20，再加点噪声
真实房价 = 2.5 * 面积 + 8 * 卧室数 + 20 + np.random.normal(0, 5, n)

# 构造 X 矩阵（两列）
X = np.column_stack((面积, 卧室数))
y = 真实房价


print("X.shape:", X.shape)
print("y.shape:", y.shape)
print('='*50)
# ===== 2. 训练模型 =====
model = LinearRegression()
model.fit(X, y)

# 打印结果
print(f"面积权重 (w1): {model.coef_[0]:.4f}")
print(f"卧室数权重 (w2): {model.coef_[1]:.4f}")
print(f"截距 (b): {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# ===== 3. 画 3D 图 =====
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 散点（实际数据）
ax.scatter(面积, 卧室数, y, color='blue', alpha=0.6, label='实际数据')

# 生成网格平面（模型预测的面）
面积_grid, 卧室_grid = np.meshgrid(np.linspace(50, 100, 10), np.linspace(1, 4, 10))
X_grid = np.column_stack((面积_grid.ravel(), 卧室_grid.ravel()))
y_grid = model.predict(X_grid).reshape(面积_grid.shape)

# 绘制平面
ax.plot_surface(面积_grid, 卧室_grid, y_grid, color='red', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('面积 (平米)')
ax.set_ylabel('卧室数 (个)')
ax.set_zlabel('房价 (万元)')
ax.set_title('线性回归拟合平面 (两个特征)')
plt.show()

'''
运行后需要观察的事实
权重：真实规律是 2.5*面积 + 8*卧室数 + 20，模型算出的两个权重会非常接近 2.5 和 8。

平面：图上的红色平面代表模型的预测。所有蓝色散点如果紧贴平面，说明模型有效。

核心认知：当特征从一个变成两个时，模型从寻找“最佳直线”变成了寻找“最佳平面”。如果特征继续增加到三个及以上，模型寻找的是“最佳超平面”（人类无法直接画图，但数学原理完全一样）。
'''