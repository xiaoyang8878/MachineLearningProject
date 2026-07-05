'''
阶段1 · 实战项目：使用线性回归预测共享单车的需求
数据集：bike-day.csv（共享单车日租赁量数据集）
特征包括：季节、年份、月份、天气、温度、湿度、风速等
目标：预测每天的共享单车总租赁量（cnt）
'''

import matplotlib
matplotlib.use('TkAgg')  # 解决弹窗报错

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === 解决 matplotlib 中文显示乱码问题 ===
# 原因：matplotlib 默认使用英文字体，不支持中文，会导致图表中的中文显示为方块（□□□）
# 解决方案：手动指定中文字体，并提供多个备选字体以兼容不同操作系统

import platform  # 用于判断当前操作系统
import os        # 用于构建相对路径

system = platform.system()
if system == 'Windows':
    # Windows 系统优先使用 SimHei（黑体），备选 Microsoft YaHei（微软雅黑）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
elif system == 'Darwin':
    # macOS 系统优先使用 PingFang SC（苹方），备选 STHeiti（华文黑体）
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Heiti TC']
else:
    # Linux 系统使用 WenQuanYi（文泉驿），备选 Noto Sans CJK
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

# 解决坐标轴上负号 '-' 显示为方块的问题（默认字体没有负号字形）
plt.rcParams['axes.unicode_minus'] = False

print(f"当前操作系统: {system}，使用中文字体: {plt.rcParams['font.sans-serif'][0]}")

# ============================================================
# 第1步：加载数据
# ============================================================
# 使用 pandas 读取 CSV 文件，将数据加载为 DataFrame
# 使用相对路径：从当前脚本所在目录向上两级到项目根目录，再进入 data/raw/
# 这样无论从哪里运行脚本，只要保持目录结构不变就能找到数据文件
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
project_root = os.path.dirname(os.path.dirname(script_dir))  # 向上两级到项目根目录
data_path = os.path.join(project_root, 'data', 'raw', 'bike-day.csv')
data = pd.read_csv(data_path)

# 打印数据的前几行，了解数据长什么样
print("========== 数据前5行 ==========")
print(data.head())

# 查看数据的基本信息（行数、列数、每列的数据类型）
print("\n========== 数据基本信息 ==========")
print(data.info())

# 查看数据的统计描述（均值、标准差、最大最小值等）
print("\n========== 数据统计描述 ==========")
print(data.describe())

# ============================================================
# 第2步：特征工程 —— 选择哪些列作为特征（X），哪列作为标签（y）
# ============================================================
# 数据集中各列的含义：
#   instant     - 序号（无意义，丢弃）
#   dteday      - 日期（字符串类型，丢弃）
#   season      - 季节（1=春, 2=夏, 3=秋, 4=冬）
#   yr          - 年份（0=2011, 1=2012）
#   mnth        - 月份（1~12）
#   holiday     - 是否节假日（0=否, 1=是）
#   weekday     - 星期几（0~6）
#   workingday  - 是否工作日（0=否, 1=是）
#   weathersit  - 天气状况（1=晴, 2=阴, 3=小雨雪, 4=恶劣天气）
#   temp        - 温度（归一化后的值，0~1）
#   atemp       - 体感温度（归一化后的值，0~1）
#   hum         - 湿度（归一化后的值，0~1）
#   windspeed   - 风速（归一化后的值，0~1）
#   casual      - 临时用户租赁数（这是结果的一部分，不能做特征，否则就是"作弊"）
#   registered  - 注册用户租赁数（同上，不能做特征）
#   cnt         - 总租赁量（= casual + registered，这就是我们要预测的目标）

# 选择特征列：去掉序号、日期、以及目标相关的 casual 和 registered
feature_columns = ['season', 'yr', 'mnth', 'holiday', 'weekday',
                   'workingday', 'weathersit', 'temp', 'atemp',
                   'hum', 'windspeed']

# X 是特征矩阵（自变量），y 是标签向量（因变量，要预测的目标）
X = data[feature_columns].values  # 转成 numpy 数组，形状为 (731, 11)
y = data['cnt'].values            # 每天的总租赁量，形状为 (731,)

print(f"\n特征矩阵 X 的形状: {X.shape}")  # 应该是 (731, 11)，即731条数据，11个特征
print(f"标签 y 的形状: {y.shape}")        # 应该是 (731,)

# ============================================================
# 第3步：划分训练集和测试集
# ============================================================
# test_size=0.2 表示 20% 的数据作为测试集，80% 作为训练集
# random_state=42 是随机种子，保证每次运行结果一致（可复现）
# 为什么要划分？
#   如果用全部数据训练，再拿同一批数据测试，就像考试考原题，满分毫无意义。
#   正确做法：训练时用80%的数据"做作业"，剩下20%藏起来当"考试卷"，最后考机器。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n========== 数据集划分完成 ==========")
print(f"训练集大小: {X_train.shape[0]} 条数据")
print(f"测试集大小: {X_test.shape[0]} 条数据")

# ============================================================
# 第4步：创建并训练线性回归模型
# ============================================================
# 创建线性回归模型对象
# 线性回归的核心思想：找到一组权重 w 和截距 b，使得 y_pred = X·w + b 尽可能接近真实的 y
# 内部使用"最小二乘法"来求解，即让所有样本的误差平方和最小
model = LinearRegression()

# 训练模型（fit = 让模型从训练数据中学习特征和标签之间的关系）
model.fit(X_train, y_train)

print("\n========== 模型训练完成 ==========")

# ============================================================
# 第5步：模型预测
# ============================================================
# 用训练好的模型对测试集进行预测
y_pred = model.predict(X_test)

# 打印前10条预测结果与真实值的对比
print("\n========== 预测结果 vs 真实值（前10条）==========")
print(f"{'序号':<5} {'预测值':<10} {'真实值':<10} {'误差':<10}")
print("-" * 45)
for i in range(10):
    error = y_pred[i] - y_test[i]
    print(f"{i+1:<5} {y_pred[i]:<10.1f} {y_test[i]:<10} {error:<10.1f}")

# ============================================================
# 第6步：模型评估 —— 判断模型好不好
# ============================================================
# MSE（均方误差）：预测值与真实值差值的平方的平均值，越小越好
# 公式：MSE = (1/n) * Σ(y_pred - y_true)^2
# 注意：这就是最小二乘法优化的目标函数（误差平方和）除以样本数取平均
#      最小二乘法找到的最优权重，就是让 MSE 最小的那组权重
mse = mean_squared_error(y_test, y_pred)
print(f"\n========== 模型评估指标 ==========")
print(f"均方误差 (MSE): {mse:.2f}")

# RMSE（均方根误差）：MSE 开根号，单位和 y 一致，更直观
# 比如 RMSE=1000 意味着预测平均偏差约1000辆车
rmse = np.sqrt(mse)
print(f"均方根误差 (RMSE): {rmse:.2f}")

# R^2（决定系数）：衡量模型对数据的拟合程度，范围通常 0~1，越接近 1 越好
# R^2 = 1 表示完美预测
# R^2 = 0 表示和直接取均值一样差
# R^2 < 0 表示比取均值还差（模型很差）
r2 = r2_score(y_test, y_pred)
print(f"决定系数 (R^2): {r2:.4f}")
print(f"解读：模型能解释 {r2*100:.1f}% 的租赁量变化")

# ============================================================
# 第7步：查看每个特征的权重（回归系数）
# ============================================================
# 线性回归的核心就是学到了一组权重：每个特征对预测结果的贡献有多大
# 最终模型：cnt = w1*season + w2*yr + ... + w11*windspeed + intercept
print("\n========== 各特征的回归系数（权重）==========")
print(f"{'特征名':<15} {'系数':<12}")
print("-" * 30)
for name, coef in zip(feature_columns, model.coef_):
    print(f"{name:<15} {coef:<12.2f}")

print(f"\n截距 (bias): {model.intercept_:.2f}")
print("\n系数含义：该特征每增加1单位，租赁量变化对应系数的值")
print("例如：yr 系数为正 → 2012年比2011年租赁量更高")
print("      temp 系数为正 → 温度越高，租赁量越大")

# ============================================================
# 第8步：可视化
# ============================================================

# --- 图1：预测值 vs 真实值的散点图 ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='black', s=20)
# 画一条"完美预测"的参考线（如果预测完全准确，所有点都落在这条线上）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='完美预测线')
plt.xlabel('真实租赁量')
plt.ylabel('预测租赁量')
plt.title(f'预测值 vs 真实值 (R$^2$={r2:.4f})')
plt.legend()

# --- 图2：特征重要性（回归系数的绝对值）柱状图 ---
plt.subplot(1, 2, 2)
# 取系数的绝对值来比较"影响力"大小（正负只影响方向，绝对值才代表影响大小）
abs_coefs = np.abs(model.coef_)
# 按影响力从小到大排序，方便横向柱状图从下到上递增显示
sorted_idx = np.argsort(abs_coefs)
sorted_names = np.array(feature_columns)[sorted_idx]
sorted_coefs = abs_coefs[sorted_idx]

plt.barh(sorted_names, sorted_coefs, color='steelblue')
plt.xlabel('回归系数绝对值（影响力大小）')
plt.title('各特征的重要性排序')
plt.tight_layout()

# 显示所有图表
plt.show()
