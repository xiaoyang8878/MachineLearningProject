

# 📓 机器学习第四课 · 完整笔记：无监督学习（K-Means 与 PCA）

> **课程目标**：理解“无标签”数据的处理逻辑，掌握 K-Means 聚类和 PCA 降维的原理、详细操作步骤及完整代码实现。


## 1. 无监督学习的核心概念

### 1.1 什么叫“没有标签”？

- **有标签（监督学习）**：数据包含特征 X 和正确答案 y（如鸢尾花的品种名称）。模型学习 X → y 的映射。
- **没有标签（无监督学习）**：数据只有特征 X，没有正确答案 y。模型自己发现 X 内部的结构。

| | 有标签（前三课） | 无标签（第四课） |
| :--- | :--- | :--- |
| **输入** | X, y | 只有 X |
| **目标** | 预测 y | 发现 X 内部结构 |
| **评估** | 准确率、MSE 等客观数值 | 无绝对标尺，靠可视化或业务解释 |
| **代表算法** | 逻辑回归、线性回归、决策树 | K-Means、PCA |


## 2. K-Means 聚类

### 2.1 核心目标

把数据点按“距离远近”自动分成 K 堆（簇），使得：
- **簇内**（同一堆）的点尽量靠近（紧密）。
- **簇间**（不同堆）的点尽量远离（分开）。

**难度：⭐（基础概念）**


### 2.2 通俗解释：展厅挂画

你在展厅里看到 300 幅画，每幅画有“色彩风格评分”（X轴）和“笔触复杂度评分”（Y轴），但负责人没告诉你这些画属于哪个流派。

1. 你猜可能有 3 个流派，于是在展厅地板上随机插了 3 面旗子（初始质心）。
2. **分配**：把每幅画放到离它最近的那面旗子旁边。
3. **更新**：走到每一堆画的中间，算出这一堆画的“平均坐标”，把旗子挪到这个新位置。
4. **重复 2 和 3**，直到旗子不再移动。

最后你得到 3 堆画。至于这堆画叫什么名字，不是 K-Means 能告诉你的，而是你事后去观察和命名的。

**难度：⭐（基础概念）**


### 2.3 详细流程（7 步）

| 步骤 | 操作 | 说明 |
| :--- | :--- | :--- |
| 0 | 原始数据 | 150 个样本，每个 2 个特征（花瓣长、花瓣宽），无标签 |
| 1 | **标准化** | `(x - μ) / σ`，消除量纲差异 |
| 2 | **初始化质心** | 指定 K=3，随机选 3 个样本作为初始质心 |
| 3 | **分配** | 每个样本分给最近的质心（欧几里得距离） |
| 4 | **更新质心** | 取每个簇内所有点的均值坐标，作为新质心 |
| 5 | **判断收敛** | 质心变了？是→回第3步，否→停止 |
| 6 | **输出结果** | 簇标签、最终质心、总误差（`inertia_`） |
| 7 | **事后命名（人工）** | 根据业务含义给簇编号贴上名字 |

**难度：⭐⭐（核心操作，需掌握流程）**


### 2.4 核心操作：`.fit_predict()`

K-Means 的训练和分组是同时完成的。它不像逻辑回归那样需要先 `.fit()` 再 `.predict()`，而是用 `.fit_predict()` 一步完成训练和分组。因为聚类没有“标准答案”要预测，它只是把现有数据分成 K 堆，所以训练和分组是同一件事。

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)  # 同时完成训练和分组
```

**难度：⭐（基础操作）**


### 2.5 完整代码：HelloKMeans.py

```python
# 文件名: HelloKMeans.py
# 目的: K-Means 聚类演示（鸢尾花数据集，不使用标签）
# 对比: 聚类的颜色分配 vs 真实的品种标签

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]          # 只取花瓣长、宽
y_true = iris.target          # 真实标签（仅用于对比，K-Means训练时不用）

model = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = model.fit_predict(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#FF0000', '#0088FF', '#00CC00']
cmap = ListedColormap(colors)

# 左图：真实标签（有监督的“标准答案”）
ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, edgecolor='k', s=30)
ax1.set_xlabel('花瓣长度')
ax1.set_ylabel('花瓣宽度')
ax1.set_title('真实标签（三种花）')

# 右图：K-Means 聚类结果（无监督，不知道花名）
ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=30)
centers = model.cluster_centers_
ax2.scatter(centers[:, 0], centers[:, 1], marker='X', color='black', s=200, linewidths=3)
ax2.set_xlabel('花瓣长度')
ax2.set_ylabel('花瓣宽度')
ax2.set_title('K-Means 聚类结果（K=3）')

plt.tight_layout()
plt.show()

print("三个簇的质心坐标：")
for i, center in enumerate(centers):
    print(f"  簇 {i}: ({center[0]:.2f}, {center[1]:.2f})")
```

**难度：⭐⭐（核心操作）**


### 2.6 肘部法则：选 K 值的方法

**这个知识点在讲什么？**

K-Means 需要你提前指定 K 值（分成几堆），但真实数据没有“正确答案”。肘部法则是帮你选 K 值的经验方法。

**核心逻辑：**

你让 K 从 1 跑到 10，每次记录一个数：`inertia_`（总误差）。
- `inertia_` = 所有样本到其所属质心的距离的平方和。
- K=1 时，所有点在一个簇里，`inertia_` 最大。
- K 越大，每个簇里的点越少，点离自己的质心越近，`inertia_` 不断下降。

**关键问题：K 越大一定越好吗？**

不是。当 K 增加到接近样本数时，每个点都自成一簇，`inertia_` 变成 0，但这样毫无意义。不能只看 `inertia_` 是否下降，要看**下降的幅度**。

**怎么看曲线？**

画出 `inertia_` 随 K 变化的曲线。当降幅从“大幅”变成“缓慢”的那个**拐点**，就是肘部。这个点对应的 K 值就是相对合理的选择。

**适用场景：**

- ✅ 曲线有明显的“肘部” → K-Means 适用，选肘部对应的 K 值。
- ❌ 曲线平滑下降，没有明显拐点 → 数据本身没有天然的簇结构，K-Means 不适合。

**难度：⭐⭐⭐（易混淆概念，需理解惯性量与拐点的含义）**


### 2.7 完整代码：KMeansElbowMethod.py

```python
# 文件名: KMeansElbowMethod.py
# 目的: 用肘部法则辅助选择 K 值

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]

inertias = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)   # inertia_ = 总误差

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K 值')
plt.ylabel('总误差（inertia）')
plt.title('肘部法则：误差随 K 值变化的曲线')
plt.grid(True)
plt.show()
```

**难度：⭐⭐（核心操作）**


### 2.8 特征缩放与标准化的区别

**这个知识点在讲什么？**

K-Means 依赖欧几里得距离计算“远近”。如果特征量纲不同（如收入 0~10 万，年龄 20~50 岁），数值大的特征会主宰距离计算，导致聚类结果只看收入，年龄完全失效。

**标准化 vs 归一化：**

| | 标准化（Standardization） | 归一化（Min-Max） |
| :--- | :--- | :--- |
| **公式** | (x - μ) / σ | (x - min) / (max - min) |
| **结果范围** | 均值=0，标准差=1（无固定上下界） | [0, 1] |
| **用于 K-Means / PCA？** | ✅ 必须 | ❌ 通常不用 |

**结论：**

- **特征缩放**是一个大类，指“把所有特征调整到同一尺度”的所有操作。
- **标准化（Z-score）**是特征缩放中最常用的一种方法，K-Means 和 PCA 都必须先做标准化。

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)  # 必须做
```

**难度：⭐⭐⭐（易混淆概念，需区分标准化与归一化）**


### 2.9 完整代码：标准化对 K-Means 的影响

```python
# 文件名: 05_kmeans_scaling_20250709.py
# 目的: 演示标准化对 K-Means 的影响（左图：未标准化，右图：标准化）

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n_samples = 300
income = np.random.normal(50000, 15000, n_samples)   # 收入范围大
age = np.random.normal(35, 10, n_samples)            # 年龄范围小
X = np.column_stack((income, age))

# 左图：未标准化
kmeans_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_raw = kmeans_raw.fit_predict(X)

# 右图：标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_raw, cmap='viridis', alpha=0.7)
ax1.scatter(kmeans_raw.cluster_centers_[:, 0], kmeans_raw.cluster_centers_[:, 1],
            marker='X', color='red', s=200, label='质心')
ax1.set_xlabel('年收入（元）')
ax1.set_ylabel('年龄（岁）')
ax1.set_title('原始数据：收入范围大，年龄被忽略')
ax1.legend()

ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled, cmap='viridis', alpha=0.7)
ax2.scatter(kmeans_scaled.cluster_centers_[:, 0], kmeans_scaled.cluster_centers_[:, 1],
            marker='X', color='red', s=200, label='质心')
ax2.set_xlabel('年收入（标准化）')
ax2.set_ylabel('年龄（标准化）')
ax2.set_title('标准化后：收入与年龄共同决定聚类')
ax2.legend()

plt.tight_layout()
plt.show()
```

**难度：⭐⭐（核心操作 + 对比验证）**


### 2.10 K-Means 与分类的区别

| | 分类（有监督） | K-Means（无监督） |
| :--- | :--- | :--- |
| **输入** | X, y | 只有 X |
| **输出** | 类别名称（如“山鸢尾”） | 簇编号（0、1、2...） |
| **名字来源** | 训练数据自带 | 你事后观察命名 |

**难度：⭐（基础概念）**


### 2.11 轮廓系数：比肘部法则更可靠的选 K 方法 

**这个知识点在讲什么？**

肘部法则需要你“肉眼看拐点”，带有主观性。轮廓系数是一个**数值指标**，直接给你打分，让你客观比较不同 K 值的好坏。

**它是怎么算的？**

对于每一个样本点：
1. **a（簇内凝聚度）**：该点到其**所在簇**内所有其他点的平均距离。a 越小，说明簇内越紧密。
2. **b（簇间分离度）**：该点到**最近的其他簇**内所有点的平均距离。b 越大，说明该点离别的簇越远。

**该点的轮廓系数** = `(b - a) / max(a, b)`

**整体轮廓系数** = 所有样本的轮廓系数平均值。

**分数范围与解读：**

| 分数范围 | 含义 |
| :--- | :--- |
| **接近 +1** | 该点离自己簇很近，离别的簇很远 → 效果好 |
| **接近 0** | 该点在两个簇的边界上 |
| **接近 -1** | 该点应该被分到别的簇 → 效果差 |

**用法：** 计算 K=2, 3, 4... 的轮廓系数，**数值越高越好**，不需要找“拐点”。

```python
from sklearn.metrics import silhouette_score
labels = model.fit_predict(X)
score = silhouette_score(X, labels)  # 传入 X 和预测出的 labels
```

**注意：** 轮廓系数要求 `K >= 2`，因为 K=1 时没有“簇间”的概念可以计算。所以循环从 2 开始，不包含 1。

**难度：⭐⭐⭐（新概念，需理解 a 和 b 的定义）**


### 2.12 完整代码：轮廓系数曲线 

```python
# 文件名: 06_silhouette_score.py
# 目的: 用轮廓系数辅助选择 K 值（比肘部法则更客观）

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data[:, :2]

silhouette_scores = []
K_range = range(2, 11)  # 轮廓系数要求 K>=2

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k} -> 轮廓系数: {score:.4f}")

plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('K 值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数曲线（越高越好）')
plt.grid(True)

best_k = K_range[np.argmax(silhouette_scores)]
plt.axvline(x=best_k, color='green', linestyle='--', label=f'最佳 K={best_k}')
plt.legend()
plt.show()

print(f"\n最佳 K 值为: {best_k}, 轮廓系数: {max(silhouette_scores):.4f}")
```

**难度：⭐⭐⭐（完整实现，包含曲线绘制与自动选K）**


### 2.13 K-Means 的局限性 

#### 局限性一：球形假设

K-Means 最小化的是**欧几里得距离的平方和**，这导致它天然倾向于产生**圆形（球形）**的簇边界。其分割边界永远是**直线**（二维）或**超平面**（高维）。

**为什么是直线？**
K-Means 决定“这个点属于哪个簇”的唯一依据是“到哪个质心的直线距离最近”。两个质心之间的分界线是**垂直平分线**（一条直线）。多个质心形成的分界线是由直线段组成的多边形，所以永远画不出曲线边界。

**什么时候失效？**
- **长条形数据**：K-Means 会把一个连贯的簇硬切成两段。
- **月牙形数据**：K-Means 会把弧形的两端切开。
- **嵌套/环形数据**：K-Means 无法识别“环内”和“环外”。

**解决方案**：改用 DBSCAN（基于密度）或谱聚类（基于图）。

#### 局限性二：初始值敏感

K-Means 的初始质心是随机选的，可能收敛到局部最优而非全局最优。

**解决方法**：
- 设置 `n_init=10`（sklearn 默认）：跑 10 次随机初始化，取 `inertia_` 最小的结果。
- KMeans++（sklearn 默认开启）：初始质心尽量分散，减少“扎堆”概率。

```python
model = KMeans(n_clusters=3, n_init=10, random_state=42)  # 安全写法
```

**难度：⭐⭐（理解问题 + 知道如何规避）**


## 3. PCA 主成分分析（降维）

### 3.1 核心目标

把高维数据压缩到低维，同时尽量保留原始数据的方差（信息）。

通俗理解：给一个三维物体拍一张二维照片，你要找“信息损失最少”的拍摄角度——PCA 自动找到这个角度。

**难度：⭐（基础概念）**


### 3.2 详细流程（6 步）

| 步骤 | 操作 | 说明 |
| :--- | :--- | :--- |
| 0 | 原始数据 | 150 个样本，每个 4 个特征 |
| 1 | **标准化** | `(x - μ) / σ`，每列均值=0，标准差=1 |
| 2 | **计算协方差矩阵** | 4×4 矩阵，描述特征间的协同变化 |
| 3 | **SVD 分解** | 得到 4 个方向向量 + 对应方差值 |
| 4 | **按方差排序** | PC1 最大，PC4 最小 |
| 5 | **截断（选主成分）** | 指定 `n_components=2`，丢弃 PC3、PC4 |
| 6 | **投影（计算新坐标）** | 点积：新坐标 = 原始向量 · 方向向量 |

**难度：⭐⭐（核心流程）**


### 3.3 标准化值的计算过程

第一个样本的花萼长度标准化值 = `-0.90068117`，计算过程如下：

1. 取鸢尾花数据集中 **所有 150 个样本的花萼长度** 这一列数据。
2. 计算这一列的 **均值（μ）** 和 **标准差（σ）**：
   - 均值 μ ≈ 5.843
   - 标准差 σ ≈ 0.825
3. 取第一个样本的 **原始花萼长度**：5.1
4. 代入标准化公式：`(5.1 - 5.843) / 0.825 = -0.9006`

**解读**：`-0.9` 表示这朵花的花萼长度比所有花的平均花萼长度 **低了 0.9 个标准差**。

**代码验证：**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[0][0])  # 输出: -0.90068117
```

**难度：⭐⭐（基础计算，需理解公式含义）**


### 3.4 方向向量的含义

PCA 对标准化后的数据做 SVD 分解，得到 **4 个方向向量**（PC1、PC2、PC3、PC4），每个方向向量包含 **4 个数字**，代表这个新方向在原始 4 个特征上的倾斜程度。

代码中打印出的 PC1 方向向量为：`[0.521, -0.269, 0.580, 0.565]`

这 4 个数字分别对应：`[花萼长, 花萼宽, 花瓣长, 花瓣宽]` 的权重。

**排序的对象是什么？**

排序的对象是 **4 条不同的直线（PC1、PC2、PC3、PC4）**，排序的依据是每条直线对应的投影方差大小。方差最大的排第一（PC1），方差最小的排第四（PC4）。

**难度：⭐⭐⭐（核心概念，需理解方向向量的构成）**


### 3.5 投影值的计算（点积）

投影值 = **样本向量** · **方向向量**（点积）

对于第一个样本，标准化后的值为：`[-0.901, 1.019, -1.340, -1.315]`

在 PC1 上的投影值 =
`(-0.901 × 0.521) + (1.019 × -0.269) + (-1.340 × 0.580) + (-1.315 × 0.565)`
`= -0.469 - 0.274 - 0.777 - 0.743`
`= -2.263`

**几何含义：** 把样本点垂直拍到新坐标轴上，落点离原点的距离就是投影值。

**代码验证：**
```python
sample = X_scaled[0]
pc1 = pca.components_[0]
projection = np.dot(sample, pc1)  # 输出: -2.2647
```

**难度：⭐⭐⭐（核心计算，需理解点积与投影的几何关系）**


### 3.6 完整代码：HelloPCA.py

```python
# 文件名: HelloPCA.py
# 目的: PCA 降维 + 打印方向向量 + 手动验算投影值

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
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print("=" * 60)
print("【标准化后的第一个样本】")
print("原始特征值 (标准化后):", X_scaled[0])
print("-" * 60)

print("【PCA 计算出的新坐标轴方向】")
print("第一主成分（PC1）方向向量:", pca.components_[0])
print("第二主成分（PC2）方向向量:", pca.components_[1])
print("-" * 60)

print("【降维结果（投影值）】")
print("第一个样本在 PC1 上的投影值:", X_pca[0][0])
print("第一个样本在 PC2 上的投影值:", X_pca[0][1])
print("-" * 60)

sample_0 = X_scaled[0]
manual_pc1 = np.dot(sample_0, pca.components_[0])
manual_pc2 = np.dot(sample_0, pca.components_[1])
print(f"手动计算 PC1: {manual_pc1:.6f} (sklearn: {X_pca[0][0]:.6f})")
print(f"手动计算 PC2: {manual_pc2:.6f} (sklearn: {X_pca[0][1]:.6f})")
print("结论：降维就是把原始点投影到新坐标轴上，投影值就是新坐标。")
print("=" * 60)

colors = ['red', 'blue', 'green']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, label=iris.target_names[i], alpha=0.8)
plt.xlabel('第一主成分 (PC1)')
plt.ylabel('第二主成分 (PC2)')
plt.title('PCA 降维结果（4维→2维）')
plt.legend()
plt.grid(True)
plt.show()
```

**难度：⭐⭐⭐（完整代码，包含手动验算）**


### 3.7 累计方差解释率曲线：选主成分数量的方法

**这个知识点在讲什么？**

降维时你要决定“保留几个主成分”。这条曲线帮你看“保留多少个新特征才够用”。

**它是怎么算的？**

PCA 把 4 个原始特征重组成 4 个新方向（PC1、PC2、PC3、PC4），并按方差从大到小排序。每个主成分都有一个“方差解释率”（`explained_variance_ratio_`），表示它承载了原始数据总方差的百分之多少。累计方差解释率就是把这些值从 PC1 开始逐个累加。

**怎么看这条曲线？**
- **横轴**：主成分个数（1、2、3、4...）
- **纵轴**：累计方差解释率（0 ~ 1）
- **红线**：通常画在 0.95（95%）的位置作为参考线

**怎么用？**

找到曲线**第一次跨过 95% 红线**的位置。

在鸢尾花数据中（4个特征）：
- PC1 解释方差 ≈ 72%
- PC2 解释方差 ≈ 23%
- 累计到 PC2 ≈ 95%

结论：只用 2 个主成分，就能保留原始 4 个特征 95% 的信息。PC3 和 PC4 承载的是同类花内部的细微差异（噪音或冗余信息），丢掉它们不会破坏大类区分。

**难度：⭐⭐⭐（核心判断依据，需理解累计方差的含义）**


### 3.8 完整代码：PCAVisualization.py

```python
# 文件名: PCAVisualization.py
# 目的: 绘制累计解释方差曲线，辅助判断保留多少个主成分

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

pca = PCA()
pca.fit(X_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差线')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释率')
plt.title('PCA 累计方差解释率曲线')
plt.legend()
plt.grid(True)
plt.show()
```

**难度：⭐⭐（核心操作）**


### 3.9 PCA 的局限性 

#### 局限性一：线性假设

PCA 只寻找**线性**方向。如果数据是**非线性的**（瑞士卷、球面、环形），PCA 会失效——它试图用直线或平面去拟合曲线，信息损失极大。

**通俗理解**：PCA 就像用压路机把三维物体压扁。如果物体本身就是扁平的（线性结构），压出来效果很好。如果物体是弹簧（非线性结构），压出来会严重变形，邻居关系被破坏。无论你怎么选择投影角度，非线性结构都无法被线性投影无损保留。

**解决方案**：改用非线性降维方法，如 **t-SNE**（可视化专用）或 **UMAP**。

#### 局限性二：可解释性问题

PCA 的新坐标轴是原始特征的**线性组合**（混合）。例如：
> PC1 = 0.52×花萼长 - 0.27×花萼宽 + 0.58×花瓣长 + 0.56×花瓣宽

如果原始特征有 100 个，PC1 是 100 个特征的加权混合，无法给它一个业务含义。

**解决方案**：
- 需要解释 → 不做 PCA，改用**特征选择**（如随机森林的特征重要性）。
- 需要降维且可解释 → 使用**稀疏 PCA**（强制方向向量中大部分权重为0）。

**难度：⭐⭐⭐（理解业务与数学的权衡）**


### 3.10 PCA vs t-SNE（快速对比）

| 维度 | PCA | t-SNE |
| :--- | :--- | :--- |
| **能否处理非线性** | ❌ 不能（线性） | ✅ 能（非线性） |
| **是否可复现** | ✅ 确定性算法 | ❌ 随机性（需设 random_state） |
| **能否用于预测新数据** | ✅ 可以（有投影矩阵） | ❌ 不可以（只能做可视化） |
| **坐标轴含义** | 有（方差方向） | 无（纯粹为了视觉布局） |
| **计算速度** | 快 | 慢（数据量大时极慢） |


## 4. PCA 动图：投影值的几何含义

### 4.1 投影值的几何可视化

`HelloPCA.py` 算出了投影值（如 -2.25），但动图把“投影值 = 点积”这个公式拆解成几何上连续变化的动作，直观展示了投影值随方向旋转而变化的过程。

**动图中可观察到的现象：**
- 当黑线指向 0°（完全水平）时，投影值 = 样本的 x 坐标。
- 当黑线指向 90°（完全垂直）时，投影值 = 样本的 y 坐标。
- 当黑线指向 45° 时，投影值是 x 和 y 各取一半。

**几何含义：** 投影值 = 样本向量 · 方向向量，即“把样本点垂直拍到旋转直线上，落点离原点的距离”。

**难度：⭐⭐⭐（辅助理解，帮助建立投影的几何直觉）**


### 4.2 完整代码：pca_single_sample_animation.py

```python
# 文件名: pca_single_sample_animation.py
# 目的: 用单个样本演示 PCA 投影值的计算过程（点积公式）
# 输出: pca_single_sample.gif

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
X = iris.data[:, :2]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 只取第一个样本
sample = X_scaled[0]
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

# 画其他样本作为背景
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], color='lightgray', alpha=0.5, s=10)

# 高亮选中的样本
sample_point, = ax.plot(x0, y0, 'bo', markersize=12, label='选中的样本')

# 动态元素
line, = ax.plot([], [], color='black', linewidth=2, label='候选方向 (旋转中)')
proj_point, = ax.plot([], [], 'ro', markersize=10, label='投影点')
proj_line, = ax.plot([], [], 'r--', linewidth=1.5, alpha=0.7)

# 文本信息
info_text = ax.text(-2.8, 2.6, '', fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
ax.legend(loc='upper right')

# 4. 动画更新函数
def update(frame):
    theta = frame * np.pi / 180
    direction = np.array([np.cos(theta), np.sin(theta)])

    # 核心计算：投影值 = 点积
    proj_value = x0 * np.cos(theta) + y0 * np.sin(theta)
    proj_x = proj_value * np.cos(theta)
    proj_y = proj_value * np.sin(theta)

    line.set_data([-3 * np.cos(theta), 3 * np.cos(theta)],
                  [-3 * np.sin(theta), 3 * np.sin(theta)])
    proj_point.set_data([proj_x], [proj_y])
    proj_line.set_data([x0, proj_x], [y0, proj_y])

    info_text.set_text(
        f"角度: {theta * 180 / np.pi:.1f}°\n"
        f"方向向量: ({np.cos(theta):.2f}, {np.sin(theta):.2f})\n"
        f"投影值 = {x0:.2f}×{np.cos(theta):.2f} + {y0:.2f}×{np.sin(theta):.2f}\n"
        f"= {proj_value:.4f}"
    )
    return line, proj_point, proj_line, info_text

ani = FuncAnimation(fig, update, frames=180, interval=50, blit=True)
ani.save('pca_single_sample.gif', writer=PillowWriter(fps=20))
print("动画已保存为 pca_single_sample.gif")
```

**难度：⭐⭐⭐（辅助理解，帮助建立投影的几何直觉）**


## 5. PCA 与分类的区别

**知识点定义：** PCA 不分类，它只做坐标变换。带颜色的散点图是用真实标签 `y` 去上色，不是 PCA 自己算出了分类结果。

| | PCA | 分类（有监督） |
| :--- | :--- | :--- |
| **输入** | 只有 X | X, y |
| **输出** | 新坐标（投影值） | 类别名称 |
| **是否分类** | ❌ 否 | ✅ 是 |

**难度：⭐（基础概念）**


## 6. 第四课核心要点

1. **无监督学习没有标签（y）**，只有特征（X）。目标是发现数据内部结构，而非预测某个答案。

2. **K-Means 聚类**：把无标签数据按距离分成 K 堆。K 值用肘部法则选。输出是簇编号（0、1、2...），名字由你事后手动贴上去。

3. **K-Means 和 PCA 都必须先做标准化**（`StandardScaler`），否则量纲大的特征会主导结果。特征缩放是**大类**，标准化是其中一种具体方法（Z-score）。

4. **PCA 降维**：通过 SVD 找出新坐标轴，按方差排序后丢弃方差小的轴。保留方差大的方向（主干信号），丢掉方差小的方向（噪音或冗余信息）。

5. **PCA 不分类**：它只做坐标变换。带颜色的散点图是用真实标签上色，不是 PCA 算出来的。

6. **投影值 = 点积**：新坐标 = 原始向量 · 方向向量。几何上就是把样本点垂直拍到旋转直线上，落点离原点的距离。

7. **聚类输出编号（0、1、2）**，分类输出名字。编号对应的名字是你事后手工贴上的。


## 7. 完整代码文件清单 

| 文件名 | 对应的知识点 |
| :--- | :--- |
| `HelloKMeans.py` | K-Means 基础聚类 + 真实标签对比 |
| `KMeansElbowMethod.py` | 肘部法则选 K |
| `05_kmeans_scaling_20250709.py` | 标准化对 K-Means 的影响 |
| `06_silhouette_score.py` | 轮廓系数（客观选 K） |
| `06_kmeans_limitation_shapes.py` | K-Means 在长条形/月牙形数据上的失效 |
| `HelloPCA.py` | PCA 降维 + 方向向量 + 手动验算投影值 |
| `PCAVisualization.py` | 累计方差解释率曲线 |
| `06_pca_vs_tsne.py` | PCA vs t-SNE 在瑞士卷数据上的对比 |

---
