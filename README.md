# 机器学习学习项目

一个系统化的学习机器学习的入门项目，按学习顺序组织代码和笔记，方便复习。

## 项目结构

```
MachineLearningProject/
├── learning_journey/                          # 学习旅程（按顺序学习）
│   ├── 00_notes/                              # 学习笔记
│   │   ├── 机器学习学习笔记.md                 # 总笔记
│   │   ├── 01_basics/
│   │   │   └── L1_Homework_用线性回归预测共享单车的需求.ipynb
│   │   ├── 02_logistic_regression/
│   │   │   ├── LogisticRegressionNotes.md
│   │   │   └── images/                        # 逻辑回归可视化图片
│   │   ├── 03_linear_regression/
│   │   │   ├── LinearRegressionNotes.md
│   │   │   └── images/                        # 线性回归可视化图片
│   │   └── 04_decision_tree/
│   │       ├── DecisionTreeAndRandomForest.md  # 决策树与随机森林专题笔记
│   │       └── images/                        # 决策树可视化图片
│   │
│   ├── 01_basics/                             # 第1课：机器学习入门
│   │   ├── HelloMachineLearing.py             # 入门示例
│   │   └── BikeDemandPrediction.py            # 入门 demo
│   ├── 02_logistic_regression/                # 第2课：逻辑回归（鸢尾花分类）
│   │   └── HelloLogisticRegression.py
│   ├── 03_linear_regression/                  # 第3课：线性回归
│   │   ├── HelloLinearRegression.py           # 线性回归（无噪声）
│   │   ├── HelloLinearRegression_nosie.py     # 线性回归（带噪声 + 残差分析）
│   │   └── HelloLinearRegression_mutiLable.py # 多特征线性回归
│   └── 04_decision_tree/                      # 第4课：决策树与随机森林
│       ├── HelloDecisionTree.py               # 单棵决策树结构图
│       ├── DecisionTreeOverfit.py             # 过拟合演示（不同深度对比）
│       ├── DecisionTreeCompareLogisticRegression.py  # 决策树 vs 逻辑回归边界对比
│       ├── HelloRandomForest.py               # 随机森林（树的数量对比）
│       ├── RandomForestMinSamplesLeaf.py      # min_samples_leaf 参数调优
│       ├── RandomForestFeatureImportance.py   # 特征重要性分析（鸢尾花）
│       └── homework/                          # 课后作业：皮马印第安人糖尿病
│           ├── PimaRandomForest.py            # 随机森林训练 + 4 张结果图生成
│           └── results/                       # 输出结果统一存放
│               ├── PimaConfusionMatrix.png    # 混淆矩阵热力图
│               ├── PimaROC.png                # ROC 曲线 (AUC=0.805)
│               ├── PimaFeatureImportance.png  # 特征重要性排序图
│               └── PimaPCA.png                # PCA 降维散点图
│
├── data/                                      # 数据目录
│   └── raw/                                   # 原始数据
│       ├── bike-day.csv                       # 共享单车数据集
│       ├── insurance.csv                      # 保险数据集
│       └── Pima Indians Diabetes Datase.csv   # 皮马印第安人糖尿病数据集
│
├── requirements.txt                           # 依赖包列表
├── .gitignore                                 # Git忽略配置
└── README.md                                  # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 按顺序学习并运行代码

```bash
# 第1课：机器学习入门
python learning_journey/01_basics/HelloMachineLearing.py

# 第2课：逻辑回归 - 鸢尾花分类
python learning_journey/02_logistic_regression/HelloLogisticRegression.py

# 第3课：线性回归 - 房价预测
python learning_journey/03_linear_regression/HelloLinearRegression.py

# 第3课拓展A：多特征线性回归 - 面积+卧室数预测房价
python learning_journey/03_linear_regression/HelloLinearRegression_mutiLable.py

# 第3课拓展B：带噪声数据 + 残差分析
python learning_journey/03_linear_regression/HelloLinearRegression_nosie.py

# 第4课：决策树 - 绘制决策树结构图
python learning_journey/04_decision_tree/HelloDecisionTree.py

# 第4课：过拟合演示 - 不同深度对比
python learning_journey/04_decision_tree/DecisionTreeOverfit.py

# 第4课：决策树 vs 逻辑回归边界对比
python learning_journey/04_decision_tree/DecisionTreeCompareLogisticRegression.py

# 第4课：随机森林 - 树的数量对比
python learning_journey/04_decision_tree/HelloRandomForest.py

# 第4课：min_samples_leaf 参数调优
python learning_journey/04_decision_tree/RandomForestMinSamplesLeaf.py

# 第4课：随机森林特征重要性分析
python learning_journey/04_decision_tree/RandomForestFeatureImportance.py

# 第4课作业：皮马印第安人糖尿病 - 随机森林 + 特征重要性 + 分类结果
python learning_journey/04_decision_tree/homework/PimaRandomForest.py
```

### 3. 学习建议

- **有监督学习三件套**：逻辑回归（分类）→ 线性回归（回归）→ 决策树（非线性）构成了从线性到非线性的完整认知闭环
- **推荐学习顺序**：按目录编号 01 → 02 → 03 → 04 逐课推进
- **配合笔记**：每个专题目录下的笔记 `.md` 文件包含核心原理 + 类比理解，建议先读笔记再运行代码

## 学习路线

| 顺序 | 主题 | 代码文件 | 笔记 | 状态 |
|------|------|----------|------|------|
| 1 | 机器学习入门 | `HelloMachineLearing.py` / `BikeDemandPrediction.py` | - | ✅ 已完成 |
| 2 | 分类入门 - 逻辑回归（鸢尾花分类） | `HelloLogisticRegression.py` | `LogisticRegressionNotes.md` | ✅ 已完成 |
| 3 | 回归入门 - 线性回归 | `HelloLinearRegression.py` / `HelloLinearRegression_mutiLable.py` / `HelloLinearRegression_nosie.py` | `LinearRegressionNotes.md` | ✅ 已完成 |
| 4 | 非线性的力量 - 决策树与随机森林 | `HelloDecisionTree.py` / `DecisionTreeOverfit.py` / `DecisionTreeCompareLogisticRegression.py` / `HelloRandomForest.py` / `RandomForestMinSamplesLeaf.py` / `RandomForestFeatureImportance.py` | `DecisionTreeAndRandomForest.md` | ✅ 已完成 |
| 5 | 无监督探索 | - | - | ⏳ 待学习 |
| 6 | 全流程项目 | - | - | ⏳ 待学习 |

## 学习笔记

- [机器学习学习笔记.md](learning_journey/00_notes/机器学习学习笔记.md) - 总笔记（覆盖全部章节）
- [逻辑回归笔记](learning_journey/00_notes/02_logistic_regression/LogisticRegressionNotes.md) - 逻辑回归专题笔记（含 Softmax、梯度下降、超参数详解）
- [线性回归笔记](learning_journey/00_notes/03_linear_regression/LinearRegressionNotes.md) - 线性回归与残差分析笔记（含噪声 vs 残差区分、多特征扩展）
- [决策树与随机森林笔记](learning_journey/00_notes/04_decision_tree/DecisionTreeAndRandomForest.md) - 决策树与随机森林专题笔记（含过拟合、特征重要性、模型选择指南）

## 各课核心知识点

### 第1课：机器学习入门
- 机器学习基本概念（特征 X / 标签 y / 训练 / 预测）
- 过拟合与欠拟合
- 训练集 / 测试集切分

### 第2课：逻辑回归（分类）
- Sigmoid / Softmax 概率转换
- 交叉熵损失函数
- 梯度下降直觉理解
- 决策边界可视化
- 模型评估（准确率、混淆矩阵、分类报告）

### 第3课：线性回归（回归）
- 最小二乘法 / 正规方程
- 噪声 vs 残差的本质区别
- 残差图诊断（U形 → 模型选错）
- 多特征扩展（平面 / 超平面）
- 线性回归对异常值敏感、逻辑回归不敏感

### 第4课：决策树与随机森林
- 决策树的"连环问答"逻辑
- **横平竖直的矩形边界** vs 逻辑回归的斜线边界
- 过拟合与 `max_depth` 的关系
- 随机森林的**两个"随机"**：Bootstrap（随机选样本）+ 特征子集（随机选特征）
- `min_samples_leaf` 消除"单间"
- 特征重要性分析

### 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 需要解释预测逻辑 | 逻辑回归 / 单棵决策树 | 权重 / 树结构可直接解释 |
| 追求最高准确率，数据量大 | 随机森林 | 集成投票稳定，抗噪声 |
| 数据量小，快速实验 | 逻辑回归（基准模型） | 训练快，结果稳定 |
| 规则提取 / EDA | 单棵决策树 | 可视化树状图直观 |

## 技术栈

- **Python 3.9+**
- **NumPy / Pandas**: 数据处理
- **Matplotlib**: 数据可视化
- **Scikit-learn**: 机器学习算法（LinearRegression, LogisticRegression, DecisionTree, RandomForest）