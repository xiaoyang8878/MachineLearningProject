# 文件名: PCAVisualization.py
# 目的: 用单个样本演示 PCA 投影值的计算过程（点积公式）

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备数据
iris = load_iris()
X = iris.data[:, :2]  # 只取前两个特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 只取第一个样本（蓝色大圆点），让它显眼
sample = X_scaled[0]  # 这是一个二维坐标 (x, y)
x0, y0 = sample

print(f"选中的样本坐标: x={x0:.2f}, y={y0:.2f}")

# 3. 设置画布
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('花萼长度 (标准化)')
ax.set_ylabel('花萼宽度 (标准化)')
ax.set_title('单个样本的投影过程 (投影值 = 点积)')
ax.grid(True, alpha=0.3)

# 画其他样本作为背景（灰色小点，降低透明度，不干扰视线）
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], color='lightgray', alpha=0.5, s=10)

# 高亮选中的样本（大蓝色圆点）
sample_point, = ax.plot(x0, y0, 'bo', markersize=12, label='选中的样本')

# 动态元素：旋转的直线（候选方向）
line, = ax.plot([], [], color='black', linewidth=2, label='候选方向 (旋转中)')

# 动态元素：投影点（样本在直线上的落点）
proj_point, = ax.plot([], [], 'ro', markersize=10, label='投影点')

# 动态元素：从样本到投影点的虚线（显示"投影"这个动作）
proj_line, = ax.plot([], [], 'r--', linewidth=1.5, alpha=0.7)

# 文本信息（显示当前角度和投影值）
info_text = ax.text(-2.8, 2.6, '', fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

ax.legend(loc='upper right')

# 4. 动画更新函数
def update(frame):
    # 角度从 0 到 180 度旋转
    theta = frame * np.pi / 180  # 转为弧度
    
    # 当前方向向量 (单位向量)
    direction = np.array([np.cos(theta), np.sin(theta)])
    
    # ---- 核心：计算投影值 ----
    # 投影值 = 样本点 · 方向向量 (点积公式)
    proj_value = x0 * np.cos(theta) + y0 * np.sin(theta)
    
    # 投影点的坐标 = 投影值 × 方向向量
    proj_x = proj_value * np.cos(theta)
    proj_y = proj_value * np.sin(theta)
    
    # ---- 更新图形元素 ----
    # 1) 旋转的直线（从原点延伸到正负方向）
    line.set_data([-3 * np.cos(theta), 3 * np.cos(theta)], 
                  [-3 * np.sin(theta), 3 * np.sin(theta)])
    
    # 2) 投影点
    proj_point.set_data([proj_x], [proj_y])
    
    # 3) 虚线（从样本点到投影点）
    proj_line.set_data([x0, proj_x], [y0, proj_y])
    
    # 4) 更新公式文本（展示具体的计算过程）
    info_text.set_text(
        f"角度: {theta * 180 / np.pi:.1f}°\n"
        f"方向向量: ({np.cos(theta):.2f}, {np.sin(theta):.2f})\n"
        f"投影值 = {x0:.2f}×{np.cos(theta):.2f} + {y0:.2f}×{np.sin(theta):.2f}\n"
        f"= {proj_value:.4f}"
    )
    
    return line, proj_point, proj_line, info_text

# 5. 生成动画
ani = FuncAnimation(fig, update, frames=180, interval=50, blit=True)

# 保存为 GIF
ani.save('pca_single_sample.gif', writer=PillowWriter(fps=20))
print("动画已保存为 pca_single_sample.gif")