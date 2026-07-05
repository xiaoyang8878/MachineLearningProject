import matplotlib
matplotlib.use('TkAgg')  # 解决弹窗报错
# =============================================================================
# 机器学习第1课：鸢尾花分类（逻辑回归）—— 全流程解剖
# =============================================================================
# 【任务类型】有监督学习 -> 分类（预测花的品种）
# 【选用模型】逻辑回归（Logistic Regression）：名字带"回归"，实则是分类算法
# 【涉及笔记章节】第1章（概述）、第3.3节（逻辑回归）、第7.1节（数据集划分）、
#                   第7.2节（评估指标）、第10章（工具）
# 【数学难度】⭐（只用到了 Sigmoid 函数，把数字压成 0~1 之间的概率）
# =============================================================================


# =============================================================================
# 模块1：导入工具库（对应笔记第10章）
# =============================================================================
# 在干嘛：把 Python 的"工具箱"搬出来，每个库都有专属用途

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris                # 内置经典数据集（不用去网上下载）
from sklearn.model_selection import train_test_split  # 笔记7.1：切分练习册和考试卷
from sklearn.linear_model import LogisticRegression   # 笔记3.3：本次主角模型
from sklearn.metrics import accuracy_score            # 笔记7.2：分类任务的打分器（算猜对比例）
import matplotlib.patches as mpatches  # 这个专门用来画图例色块
from matplotlib.colors import ListedColormap


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']

# =============================================================================
# 模块2：加载数据（笔记1.2：特征 X 和标签 y）
# =============================================================================
# 在干嘛：把花的"体检报告"（X）和"品种名称"（y）读进内存
iris = load_iris()
# print(type(iris))
print(iris.data.shape)  # (150, 4)
print(type(iris.data))  # <class 'numpy.ndarray'>
# X：特征矩阵，共有4列（花萼长/宽、花瓣长/宽）
# 这里只取前两列（花瓣长度、花瓣宽度），目的是为了能在纸上画出二维散点图
X = iris.data[:, :2]
# 读取前五行数据
print(X[:5])

# y：标签向量，0=山鸢尾，1=杂色鸢尾，2=维吉尼亚鸢尾
y = iris.target
print(type(y))  # <class 'numpy.ndarray'>
print(y.shape)  # (150,)
print(y[:5])


# 【知识点】特征（X）是你量出的数据，标签（y）是植物学家告诉你的标准答案。

# =============================================================================
# 模块3：切分训练集和测试集（笔记7.1 的核心动作）
# =============================================================================
# 在干嘛：把150朵花随机打乱，70%（105朵）当作业题，30%（45朵）当期末考试卷。
# 为什么要切分：防止"伪学霸"效应（笔记1.3 过拟合）。
#   如果拿全部数据训练又拿全部数据考，机器直接背答案，毫无意义。

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,   # 30% 的数据留给测试集
    random_state=42  # 固定随机种子：保证每次运行被分去考试的都是同一批花。
                     # 如果不加这行，每次运行结果都不同，没法复现实验。
)  # 返回训练集和测试集的特征矩阵以及标签向量

print(X_train.shape)    # (105, 2)
print(X_test.shape)     # (45, 2)
print(y_train.shape)    # (105,)
print(y_test.shape)     # (45,)

# =============================================================================
# 模块4：创建模型并训练（机器学习最神圣的仪式）
# =============================================================================
# 在干嘛：
#   1. 创建了一个空白的"逻辑回归分类器"。
#   2. .fit()：把105朵花的特征和品种喂给机器。机器开始埋头苦算，找出最佳的"决策边界"。
#
# 【内部原理白话版】（对应笔记3.3 逻辑回归 + 笔记0.4 梯度下降）
#   - 虽然名字叫回归，但它最后加了一道"Sigmoid门"（1/(1+e^-z)），
#     把计算结果强行压到 0~1 之间，变成"这朵花属于第1类的概率是90%"。
#   - max_iter=1000：告诉机器"我给你1000次下山的机会去找最低点"。
#     数据简单时默认100次可能不够收敛，调大点保证成功。

model = LogisticRegression(max_iter=1000)   # 创建模型
model.fit(X_train, y_train)                 # <--- 这里在发生"梯度下降"数学运算，但你只需要知道它在找规律

print("="*50)
print("逻辑回归模型内部参数大揭秘")
print("="*50)

# 1. 查看权重（对应你理解的 k）
# coef_ 是一个二维数组：行数=类别数(3)，列数=特征数(2)
print(f"权重矩阵 (coef_) 的形状: {model.coef_.shape}")
print("【权重 k 的值】（分别对应 山鸢尾、杂色鸢尾、维吉尼亚鸢尾）:")
for i, class_name in enumerate(iris.target_names):
    # 每一行有2个数字：k1(花瓣长度权重), k2(花瓣宽度权重)
    k1, k2 = model.coef_[i]
    print(f"  类别 {i} ({class_name}): 花瓣长度权重={k1:.4f}, 花瓣宽度权重={k2:.4f}")

# 2. 查看偏置（对应你理解的 b）
# intercept_ 是一个一维数组，长度为类别数(3)
print(f"\n【偏置 b 的值】（分别对应 山鸢尾、杂色鸢尾、维吉尼亚鸢尾）:")
for i, class_name in enumerate(iris.target_names):
    b = model.intercept_[i]
    print(f"  类别 {i} ({class_name}): b = {b:.4f}")

# 3. 对第一个样本手工验算一下（看看公式 z = k1*x1 + k2*x2 + b 是否成立）
sample_index = 0  # 取测试集第一个样本
sample_X = X_test[sample_index]  # 比如 [5.0, 3.4]
sample_y_true = y_test[sample_index]  # 真实的标签数字

print(f"\n【手工验算 - 测试集第1个样本】")
print(f"  特征值: 花瓣长={sample_X[0]:.2f}, 花瓣宽={sample_X[1]:.2f}")
print(f"  真实标签: {sample_y_true} ({iris.target_names[sample_y_true]})")

# 分别计算 3 个类别的得分 z
for i, class_name in enumerate(iris.target_names):
    z = (model.coef_[i][0] * sample_X[0] +   # k1 * x1
         model.coef_[i][1] * sample_X[1] +   # k2 * x2
         model.intercept_[i])                # b
    print(f"  类别 {i} ({class_name}) 的线性得分 z = {z:.4f}")

# 模型最后会选 z 值最大的那个类别作为预测结果
predicted = model.predict([sample_X])[0]
print(f"\n  模型最终预测标签: {predicted} ({iris.target_names[predicted]})")
print("  (因为上面哪个类别的 z 值最大，模型就选哪一个)")

# =============================================================================
# 模块6：画决策边界（可视化魔法）
# =============================================================================
# 在干嘛：模型其实是一套"区域划分规则"。这段代码把整个平面切成无数小点，
#   让模型预测每个小点的类别，然后涂上不同颜色。
#   最终你看到的红黄蓝背景，就是模型学到的"领土疆界"。
# 【涉及知识点】决策边界（Decision Boundary）可视化。
#   红点和蓝点混在一起的地方，就是模型容易犯错的地方。

def plot_decision_boundary():
    # 生成网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                         np.arange(y_min, y_max, 0.001))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # print('Z shape:', Z.shape)
    # print('Z:', Z)
    # ===== 【核心改动】强制使用红、蓝、绿（没有任何中间色混淆） =====
    # 分别对应标签 0（山鸢尾）、1（杂色鸢尾）、2（维吉尼亚鸢尾）
    color_list = ['#FF0000', '#0088FF', '#00CC00']  # 纯红、亮蓝、翠绿
    cmap_bg = ListedColormap(color_list)

    # 背景色块 alpha 调到 0.8，几乎不透明，颜色非常实在
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_bg)

    # 散点强制使用同样的颜色列表（用 c 参数直接映射）
    # 注意：scatter 里如果传入颜色列表，需要用 c= 加上列表映射，或者直接给每个点赋颜色
    # 这里用最简单的方法：分别绘制三类数据点
    for i, color in enumerate(color_list):
        # 训练集散点（圆形）
        train_mask = (y_train == i)
        plt.scatter(X_train[train_mask, 0], X_train[train_mask, 1],
                    color=color, edgecolor='black', linewidth=0.5, s=40, label='训练集' if i == 0 else "")
        # 测试集散点（三角形）
        test_mask = (y_test == i)
        plt.scatter(X_test[test_mask, 0], X_test[test_mask, 1],
                    color=color, marker='^', edgecolor='black', linewidth=0.5, s=40, label='测试集' if i == 0 else "")

    # ===== 手动建立图例色块（确保和屏幕颜色 100% 一致） =====
    patch_list = [
        mpatches.Patch(color='#FF0000', label='山鸢尾 (Setosa)'),
        mpatches.Patch(color='#0088FF', label='杂色鸢尾 (Versicolour)'),
        mpatches.Patch(color='#00CC00', label='维吉尼亚鸢尾 (Virginica)')
    ]

    # 获取已有的训练集/测试集图例（但因为我们分开画了，这里简化处理）
    # 我们手动构造完整的图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='训练集'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='测试集'),
        *patch_list  # 直接拼接三个色块
    ]
    labels = ['训练集', '测试集', '山鸢尾 (红)', '杂色鸢尾 (蓝)', '维吉尼亚鸢尾 (绿)']

    plt.legend(handles=handles, labels=labels, loc='best')
    plt.xlabel('花瓣长度 (cm)')
    plt.ylabel('花瓣宽度 (cm)')
    plt.title('逻辑回归决策边界（红/蓝/绿严格对应三类花）')
    plt.show()

plot_decision_boundary()


# =============================================================================
# 模块7：模型评估（对应笔记7.2 评估指标）
# =============================================================================
# 目的：用"没见过的测试集"来检验模型的真实水平
# 注意：这里只评估分类任务的指标，回归任务的指标（MSE等）第二课再学
# =============================================================================

print("\n" + "="*50)
print("【模型评估】逻辑回归在测试集上的表现")
print("="*50)

# 1. 预测测试集（模型没见过这些数据）
y_pred = model.predict(X_test)

# 2. 计算准确率（Accuracy）：猜对的比例
acc = accuracy_score(y_test, y_pred)
print(f"✅ 准确率 (Accuracy): {acc:.4f}  ({acc*100:.2f}%)")
print(f"   → 意思是：{len(y_test)} 朵花里，模型猜对了 {int(acc * len(y_test))} 朵")

# 3. 手动数一数：哪些被分错了（对应你之前数“落入别人地盘的点”）
wrong_indices = np.where(y_pred != y_test)[0]
print(f"\n❌ 分类错误: 共 {len(wrong_indices)} 个样本")
if len(wrong_indices) > 0:
    print("   被分错的样本索引 (在测试集中的位置):", wrong_indices)
    for idx in wrong_indices:
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        print(f"     样本 {idx}: 真实={true_label} ({iris.target_names[true_label]}), "
              f"预测={pred_label} ({iris.target_names[pred_label]})")

# 4. 【进阶】打印混淆矩阵（直观看到“哪一类被误判成哪一类”）
print("\n📊 混淆矩阵 (Confusion Matrix):")
print("   行 = 真实类别, 列 = 预测类别")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("   ", cm)
print("   解读: 对角线上的数字是猜对的，非对角线是分错的")

# 5. 【进阶】打印分类报告（精确率、召回率、F1 分数）
print("\n📋 分类报告 (Classification Report):")
from sklearn.metrics import classification_report
# target_names 是花的英文名，我们手动改成中文方便阅读
report = classification_report(y_test, y_pred, target_names=['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾'])
print(report)

print("="*50)



# =============================================================================
# 📚 代码世界观 · 必记口诀（对应你笔记的核心套路）
# =============================================================================
# 1. load_data() + X, y  → 读入数据（X是特征列，y是目标列）
# 2. train_test_split()  → 闭卷考试原则（训练绝不许偷看测试集）
# 3. model.fit()          → 埋头苦读做练习（机器找规律）
# 4. model.predict()      → 上考场写答案（做推理）
# 5. accuracy_score()     → 批卷子打分（评估好坏）


# =============================================================================
# 思考题（请用大白话回答我，巩固理解）
# =============================================================================
# 1. 逻辑回归模型在 model.fit() 的时候，到底在"算"什么东西？
#    （提示：它是不是在找一条直线？还是找某个数学公式里的 k 和 b？）
'''
逻辑回归模型在model.fit()的时候，算的是k 和 b，通过梯度下降的方法，找到一个最接近的直线，使得损失函数最小。

逻辑回归确实就是通过梯度下降反复调整权重系数（k，即 w）和偏置（b），试图画出一条尽可能把点分干净的“分隔线”。
进阶补充：你用了“最接近的直线”这个词，在逻辑回归里，我们更专业的叫法是 “决策边界（Decision Boundary）”。但因为你们用的是二维数据（花瓣长/宽），这条决策边界在图上看起来确实是一条直线。你已经准确理解了它。

用初中的数学知识解释：             机器学习里面的解释
k =》 分割线的斜率               k（机器学习里叫权重 w）：决定这条分割线是“陡”还是“平”（斜率）。
b =》 分割线上下平移量            b（机器学习里叫偏置 bias）：决定这条分割线是往上挪还是往下挪（上下平移）。

逻辑回归的损失函数，本质就是 “正确答案概率的倒数”。正确答案的概率越接近 100%，损失越趋近于 0；正确答案的概率越接近 0%，损失趋近于无穷大。

    情景模拟（你是机器，你猜花）
        拿到一朵花：真实标签是“山鸢尾”（我们就叫它 类别 A）。
        机器内部算了一下：输出一个概率，比如 “我觉得这朵花是类别 A 的概率是 90%”。
        算损失（痛苦值）：
            如果概率是 90%：因为正确答案（类别A）已经非常接近 100% 了，机器非常接近真相。损失很小（不痛）。
            如果概率是 10%：正确答案明明是 A，机器却说“我觉得只有 10% 是 A”，这意味着机器严重看走了眼。损失极大（剧痛）。
'''
# 2. 代码里 model.predict(X_test) 返回的 y_pred，打印出来是 0、1、2 这种数字，
#    还是 "山鸢尾" 这样的中文名字？这说明了什么？
'''
打印的是0、1、2这样的数字，说明model.predict(X_test)返回的是模型预测的标签，而不是标签对应的中文名字。
'''
# 3. 如果我把 test_size=0.3 改成 test_size=0.5，你觉得准确率大概率会变高还是变低？为什么？
'''
准确率会变低，因为训练集变小了

    点评（80分）：你的直觉方向是正确的——一般情况下，训练数据变少，模型确实更容易“学不到位”，泛化能力下降的风险变大。
    但是，这里有个容易踩的坑（必须记住）：
    将 test_size 从 0.3 改成 0.5，准确率不一定会变低，它可能变高、变低，甚至不变。为什么？
        1.因为当你把测试集变大（从45个变成75个）时，相当于换了半张“新卷子”。
        2.如果这半张卷子里恰好都是简单的题（好分的花），准确率可能反而变高。
        3.如果这半张卷子里恰好都是模棱两可的题（边界线上的花），准确率就会变低。
    要真正防止这种“因为换卷子导致误判”的情况，我们未来会用到一个工具——这就是你笔记第7.1节里提到的 K折交叉验证（K-Fold Cross Validation）。它会洗牌多次取平均，这样就不会被某一次“坏运气”带偏。这也是你第2课要学的内容！
'''
# =============================================================================

