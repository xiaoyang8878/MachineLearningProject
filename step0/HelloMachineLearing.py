import matplotlib
matplotlib.use('TkAgg')  # 1. 解决弹窗报错

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === 解决中文乱码（这两行是核心！）===
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体（Windows/Mac通用）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# # 1. 造数据：假设这是“房子面积（X）和房价（Y）”
# X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)  # 特征（必须二维）==》 X1 = np.array([[50, 60, 70, 80, 90, 100]]).T
# y = np.array([150, 180, 210, 240, 270, 300])  # 标签
#
# # 2. 创建模型（机器学习核心对象）
# model = LinearRegression()
#
# # 3. 训练模型（俗称“拟合”，就是让模型找规律）
# model.fit(X, y)
#
# # 4. 预测：如果有一套85平的房子，卖多少钱
# price = model.predict([[85]])
# print(f"预测85平的房价为：{price[0]:.2f} 万元")
#
# # 画图看看这根“学习到的直线”
# plt.scatter(X, y, color="red", label='真实房价')
# plt.plot(X, model.predict(X), color="blue", label='机器学习拟合线')
# plt.ylabel('房价(万元)')
# plt.legend()
# plt.show()


"""
1. 机器学习到底在干嘛？
    给机器一堆“输入”和“正确答案”，让它自己总结出从输入到答案的映射规则。在代码里，机器总结出的规则就是：房价 = 3 × 面积（即斜率为3）。
    
2. 什么是 特征（X） 和 标签（y）？
    特征（X）：用来预测的依据。在代码里是 [50, 60, 70, 80, 90, 100]（房子面积）。
    标签（y）：我们想预测的真实结果。在代码里是 [150, 180, 210, 240, 270, 300]（对应房价）。
    一句话记忆法：X是“考题题干”，y是“标准答案”。
    
3. 什么是 训练（fit） 和 预测（predict）？
    model.fit(X, y)（训练）：把“考题+答案”一起扔给机器，让它埋头苦读，总结规律。此时机器内部算出了“斜率=3，截距=0”。
    model.predict([[85]])（预测）：给机器一道新题（85平），让它用刚学到的规律（斜率3）写出答案。机器算出 85×3=255 万。
    
4. 为什么必须把数据分成 训练集 和 测试集？
    如果你用 全部6套房（X） 去训练，又拿 这6套房中的某一套（比如50平） 去问机器，机器会直接背出答案150万。这就像期末考试考课本原题，满分毫无意义。
    正规做法是：把6套房分成两堆——
    4套拿去训练（做作业）。
    2套藏起来（当考试卷），训练时绝不用它，最后拿这2套来考机器，看它猜得准不准。
"""
from sklearn.model_selection import train_test_split  # 第1行新增：导入切分工具

# 1. 造数据：假设这是“房子面积（X）和房价（Y）”
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)  # 特征（必须二维）==》 X1 = np.array([[50, 60, 70, 80, 90, 100]]).T
y = np.array([150, 180, 210, 240, 270, 300])  # 标签

# 2. 创建模型（机器学习核心对象）
model = LinearRegression()

# 3. 训练模型（俗称“拟合”，就是让模型找规律）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("训练集面积：", X_train.ravel())  # 看看哪些去当作业题了
print("测试集面积：", X_test.ravel())  # 看看哪些被藏起来当考试题了

model.fit(X_train, y_train)

# 4. 预测：如果有一套85平的房子，卖多少钱
predictions = model.predict(X_test)
print("预测的房价：", predictions)
print("真实的房价：", y_test)