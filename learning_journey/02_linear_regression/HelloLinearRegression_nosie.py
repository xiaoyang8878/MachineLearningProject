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

# ===== 【B 实验】给房价加上随机波动（模拟真实世界） =====
# 真实规律依然是 房价 = 3 * 面积，但加一些随机的“运气成分”

# np.random.seed(42)  # 固定随机种子，让你我看到的波动是一样的
# X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)

# # 生成基础房价，然后加上 ±5 万元以内的随机误差
# true_price = 3 * X.flatten()
# noise = np.random.normal(0, 3, size=len(X))  # 均值为0，标准差为3的随机波动
# y = true_price + noise

# print("面积:", X.flatten())
# print("真实房价(带噪声):", np.round(y, 2))

# print('='*50)
# # 2. 切分数据（依然要闭卷考试）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # 3. 创建模型并训练（这就是在找 k 和 b）
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 4. 查看模型学到的规律（你的初中数学答案！）
# k = model.coef_[0]  # 斜率
# b = model.intercept_  # 截距
# print(f"模型学到的规律：房价 = {k:.2f} × 面积 + {b:.2f}")
# print(f"（真实规律是：房价 = 3.00 × 面积 + 0.00）")
# print('='*50)
# # 5. 预测（考试的时候，你只能用这个模型来预测，不能自己算）
# price_85 = model.predict([[85]])
# print(f"预测面积为 85 平米的房价为：{price_85[0]:.2f} 万元")
# print('='*50)
# # 6. 评估（考试的时候，你只能用这个模型来评估，不能自己算）
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"\n【测试集评估】")
# print(f"均方误差 MSE：{mse:.2f}（越小越好，0 表示完美）")
# print(f"决定系数 R²：{r2:.4f}（越接近 1 越好）")

# # 7. 画图：看那条直线
# plt.scatter(X_train, y_train, color='blue', label='训练集')
# plt.scatter(X_test, y_test, color='red', marker='^', label='测试集')
# plt.plot(X, model.predict(X), color='red', linewidth=3, label='拟合直线 y = kx + b')
# plt.xlabel('面积 (平米)')
# plt.ylabel('房价 (万元)')
# plt.legend()
# plt.title('线性回归：找一条直线尽可能穿过所有点')
# plt.grid(True)
# plt.show()


# ===== B - 1 区分残差与噪声

# ===== 固定随机种子，保证结果可复现 =====
np.random.seed(42)

# ===== 生成模拟数据（60个样本） =====
X = np.linspace(50, 100, 60).reshape(-1, 1)

# --- 左图数据：线性关系 + 纯噪声（模型选对了） ---
# 真实规律：y = 3x，加上标准差为8的随机波动
y_linear = 3 * X.flatten() + np.random.normal(0, 8, size=60)

# --- 右图数据：曲线关系（模型选错了） ---
# 真实规律：y = 0.04 * x^2，同样加上一些小噪声
# 注意：当 x=50 时，y≈100；当 x=100 时，y≈400，量级与左图相近，便于对比
y_curve = 0.04 * (X.flatten() ** 2) + np.random.normal(0, 5, size=60)

# ===== 左图：线性模型拟合线性数据 =====
model_linear = LinearRegression()
model_linear.fit(X, y_linear)
y_pred_linear = model_linear.predict(X)
residual_linear = y_linear - y_pred_linear  # 残差 = 真实值 - 预测值

# ===== 右图：线性模型拟合曲线数据 =====
model_curve = LinearRegression()
model_curve.fit(X, y_curve)
y_pred_curve = model_curve.predict(X)
residual_curve = y_curve - y_pred_curve

# ===== 画图 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左子图：残差随机分布（意味着模型已榨干规律） ---
axes[0].scatter(y_pred_linear, residual_linear, alpha=0.7, color='blue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('场景1：线性数据 + 噪声')
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('残差')
axes[0].set_ylim(-30, 30)  # 固定坐标轴范围，便于对比

# --- 右子图：残差呈U形（意味着直线没抓住曲线规律） ---
axes[1].scatter(y_pred_curve, residual_curve, alpha=0.7, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('场景2：曲线数据 + 用直线去拟合')
axes[1].set_xlabel('预测值')
axes[1].set_ylabel('残差')
axes[1].set_ylim(-30, 30)

plt.tight_layout()
plt.show()

# ===== 打印R²分数，辅助判断 =====
r2_linear = model_linear.score(X, y_linear)
r2_curve = model_curve.score(X, y_curve)
print(f"场景1（线性数据）的 R² = {r2_linear:.4f}")
print(f"场景2（曲线数据）的 R² = {r2_curve:.4f}") 