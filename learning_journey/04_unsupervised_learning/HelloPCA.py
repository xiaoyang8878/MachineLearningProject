# 文件名: HelloPCA.py
# 目的: 展示 PCA 的内部降维过程（打印新坐标轴、手动验算投影点）
# 内容: 1. 输出主成分（新坐标轴）方向；2. 手动计算一个点的投影坐标；3. 验证与 sklearn 计算结果是否一致

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载并标准化数据
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

scaler = StandardScaler()
print("【标准化背后的秘密】")
print(f"花萼长度的均值: {scaler.mean_[0]:.4f}, 标准差: {scaler.scale_[0]:.4f}")
print(f"花萼宽度的均值: {scaler.mean_[1]:.4f}, 标准差: {scaler.scale_[1]:.4f}")
print(f"花瓣长度的均值: {scaler.mean_[2]:.4f}, 标准差: {scaler.scale_[2]:.4f}")
print(f"花瓣宽度的均值: {scaler.mean_[3]:.4f}, 标准差: {scaler.scale_[3]:.4f}")
X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("【步骤 1】查看标准化后的第一个样本")
print("原始特征值 (标准化后):", X_scaled[0])
print("这个数值表示该样本的四个特征（花萼长/宽、花瓣长/宽）在标准正态分布中的位置。")
print("-" * 60)

# 2. 显式执行"拟合"和"转换"，拆解过程
pca = PCA(n_components=2)
# 2.1 拟合（fit）：计算新坐标轴的方向（主成分）
pca.fit(X_scaled)

print("【步骤 2】PCA 计算出的'新坐标轴'方向")
print("主成分（Components）形状:", pca.components_.shape)  # (2, 4)
print("第一主成分（新 X 轴）方向向量:", pca.components_[0])
print("第二主成分（新 Y 轴）方向向量:", pca.components_[1])
print("解释：新坐标轴由原始特征的线性组合构成。例如：")
print("  PC1 = 0.52*花萼长 + (-0.27)*花萼宽 + 0.58*花瓣长 + 0.56*花瓣宽")
print("-" * 60)

# 2.2 转换（transform）：计算原始数据在新坐标轴上的投影值
X_pca = pca.transform(X_scaled)  # 这里等价于 fit_transform

print("【步骤 3】查看降维结果（投影值）")
print("第一个样本在 PC1 上的投影值 (新 X 坐标):", X_pca[0][0])
print("第一个样本在 PC2 上的投影值 (新 Y 坐标):", X_pca[0][1])
print("-" * 60)

# 3. 手动验算：用点积公式验证第一个样本的投影值
# 原理：新坐标 = 原始向量 · 新坐标轴方向的单位向量（主成分）
sample_0 = X_scaled[0]
pc1_vector = pca.components_[0]
pc2_vector = pca.components_[1]

manual_pc1 = np.dot(sample_0, pc1_vector)
manual_pc2 = np.dot(sample_0, pc2_vector)

print("【步骤 4】手动验算第一个样本的投影（点积公式）")
print(f"手动计算 PC1: {manual_pc1:.6f} (sklearn 结果: {X_pca[0][0]:.6f})")
print(f"手动计算 PC2: {manual_pc2:.6f} (sklearn 结果: {X_pca[0][1]:.6f})")
print("结论：降维就是把原始点'投影'到新坐标轴上，这个投影值就是新坐标。")
print("=" * 60)

# 4. 绘制降维后的散点图
colors = ['red', 'blue', 'green']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, label=iris.target_names[i], alpha=0.8)

plt.xlabel('第一主成分 (PC1)')
plt.ylabel('第二主成分 (PC2)')
plt.title('PCA 降维结果（4维 → 2维）')
plt.legend()
plt.grid(True)
plt.show()