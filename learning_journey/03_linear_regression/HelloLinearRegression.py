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

# 1. 造数据（面积和房价，严格满足 房价 = 3 × 面积）
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)  # reshape(-1, 1) 表示将数组 X 变成 6 行 1 列的二维数组 （即 6 个样本，每个样本有 1 个特征 -> 房屋面积）
y = np.array([150, 180, 210, 240, 270, 300])    # 房价（万元）

print("X.shape:", X.shape)
print("y.shape:", y.shape)
print('='*50)
# 2. 切分数据（依然要闭卷考试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建模型并训练（这就是在找 k 和 b）
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 查看模型学到的规律（你的初中数学答案！）
k = model.coef_[0]  # 斜率
b = model.intercept_  # 截距
print(f"模型学到的规律：房价 = {k:.2f} × 面积 + {b:.2f}")
print(f"（真实规律是：房价 = 3.00 × 面积 + 0.00）")
print('='*50)
# 5. 预测（考试的时候，你只能用这个模型来预测，不能自己算）
price_85 = model.predict([[85]])
print(f"预测面积为 85 平米的房价为：{price_85[0]:.2f} 万元")
print('='*50)
# 6. 评估（考试的时候，你只能用这个模型来评估，不能自己算）
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n【测试集评估】")
print(f"均方误差 MSE：{mse:.2f}（越小越好，0 表示完美）")
print(f"决定系数 R²：{r2:.4f}（越接近 1 越好）")

# 7. 画图：看那条直线
plt.scatter(X_train, y_train, color='blue', label='训练集')
plt.scatter(X_test, y_test, color='red', marker='^', label='测试集')
plt.plot(X, model.predict(X), color='red', linewidth=3, label='拟合直线 y = kx + b')
plt.xlabel('面积 (平米)')
plt.ylabel('房价 (万元)')
plt.legend()
plt.title('线性回归：找一条直线尽可能穿过所有点')
plt.grid(True)
plt.show()