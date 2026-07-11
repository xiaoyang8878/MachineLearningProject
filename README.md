# 机器学习学习项目

一个系统化的学习机器学习的入门项目，按学习顺序组织代码和笔记，方便复习。

## 项目结构

```
MachineLearningProject/
├── learning_journey/                          # 学习旅程（按顺序学习）
│   ├── notes/                                 # 学习笔记
│   │   ├── 机器学习学习笔记.md                 # 总笔记
│   │   ├── 00_intro/
│   │   │   └── L1_Homework_用线性回归预测共享单车的需求.ipynb
│   │   ├── 01_logistic_regression/
│   │   │   ├── LogisticRegressionNotes.md
│   │   │   └── images/                        # 逻辑回归可视化图片
│   │   ├── 02_linear_regression/
│   │   │   ├── LinearRegressionNotes.md
│   │   │   └── images/                        # 线性回归可视化图片
│   │   ├── 03_decision_tree/
│   │   │   ├── DecisionTreeAndRandomForest.md  # 决策树与随机森林专题笔记
│   │   │   └── images/                        # 决策树可视化图片
│   │   └── 04_unsupervised_learning/
│   │       └── UnsupervisedLearning.md         # 无监督学习笔记（K-Means 与 PCA）
│   │
│   ├── 00_intro/                              # 第1课：机器学习概述
│   │   ├── HelloMachineLearing.py             # 入门示例
│   │   └── BikeDemandPrediction.py            # 入门 demo
│   ├── 01_logistic_regression/                # 第1课：逻辑回归（分类）
│   │   └── HelloLogisticRegression.py
│   ├── 02_linear_regression/                  # 第2课：线性回归（回归）
│   │   ├── HelloLinearRegression.py           # 线性回归（无噪声）
│   │   ├── HelloLinearRegression_nosie.py     # 线性回归（带噪声 + 残差分析）
│   │   └── HelloLinearRegression_mutiLable.py # 多特征线性回归
│   ├── 03_decision_tree/                      # 第3课：决策树与随机森林
│   │   ├── HelloDecisionTree.py               # 单棵决策树结构图
│   │   ├── DecisionTreeOverfit.py             # 过拟合演示（不同深度对比）
│   │   ├── DecisionTreeCompareLogisticRegression.py  # 决策树 vs 逻辑回归边界对比
│   │   ├── HelloRandomForest.py               # 随机森林（树的数量对比）
│   │   ├── RandomForestMinSamplesLeaf.py      # min_samples_leaf 参数调优
│   │   ├── RandomForestFeatureImportance.py   # 特征重要性分析（鸢尾花）
│   │   └── homework/                          # 课后作业：皮马印第安人糖尿病
│   │       ├── PimaRandomForest.py            # 随机森林训练 + 4 张结果图生成
│   │       └── results/                       # 输出结果统一存放
│   │           ├── PimaConfusionMatrix.png    # 混淆矩阵热力图
│   │           ├── PimaROC.png                # ROC 曲线 (AUC=0.805)
│   │           ├── PimaFeatureImportance.png  # 特征重要性排序图
│   │           └── PimaPCA.png                # PCA 降维散点图
│   ├── 04_unsupervised_learning/              # 第4课：无监督学习（K-Means 与 PCA）
│   │   ├── HelloKMeans.py                     # K-Means 聚类演示
│   │   ├── KMeansElbowMethod.py               # 肘部法则选 K 值
│   │   ├── KmeansScaling.py                   # 标准化对 K-Means 的影响
│   │   ├── HelloKMeans_WithSilhouette.py      # 轮廓系数评估聚类
│   │   ├── kmeans_silhouette_limitations.py   # K-Means 局限性演示
│   │   ├── HelloPCA.py                        # PCA 降维 + 手动验算投影值
│   │   ├── PCAVisualization.py                # 累计方差解释率曲线
│   │   └── pca_single_sample_animation.py     # 投影值几何含义动图
│   ├── 05_model_evaluation/                   # 第5课：模型评估与诊断（待学习）
│   │   └── (待添加)
│   ├── 06_feature_engineering_project/        # 第6课：特征工程与综合实战（待学习）
│   │   └── (待添加)
│   └── 07_deployment/                         # 部署概念（待学习）
│       └── (待添加)
│
├── data/                                      # 数据目录
│   └── raw/                                   # 原始数据
│       ├── bike-day.csv                       # 共享单车数据集
│       ├── insurance.csv                      # 保险数据集
│       └── Pima Indians Diabetes Datase.csv   # 皮马印第安人糖尿病数据集
│
├── pca_single_sample.gif                      # PCA 投影动图输出
├── pca_animation.gif                          # PCA 动画输出
├── course_syllabus.md                         # 课程大纲
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
# 第1课：机器学习入门 + 逻辑回归
python learning_journey/00_intro/HelloMachineLearing.py
python learning_journey/01_logistic_regression/HelloLogisticRegression.py

# 第2课：线性回归
python learning_journey/02_linear_regression/HelloLinearRegression.py
python learning_journey/02_linear_regression/HelloLinearRegression_mutiLable.py
python learning_journey/02_linear_regression/HelloLinearRegression_nosie.py

# 第3课：决策树与随机森林
python learning_journey/03_decision_tree/HelloDecisionTree.py
python learning_journey/03_decision_tree/DecisionTreeOverfit.py
python learning_journey/03_decision_tree/DecisionTreeCompareLogisticRegression.py
python learning_journey/03_decision_tree/HelloRandomForest.py
python learning_journey/03_decision_tree/RandomForestMinSamplesLeaf.py
python learning_journey/03_decision_tree/RandomForestFeatureImportance.py
python learning_journey/03_decision_tree/homework/PimaRandomForest.py

# 第4课：无监督学习
python learning_journey/04_unsupervised_learning/HelloKMeans.py
python learning_journey/04_unsupervised_learning/KMeansElbowMethod.py
python learning_journey/04_unsupervised_learning/KmeansScaling.py
python learning_journey/04_unsupervised_learning/HelloKMeans_WithSilhouette.py
python learning_journey/04_unsupervised_learning/kmeans_silhouette_limitations.py
python learning_journey/04_unsupervised_learning/HelloPCA.py
python learning_journey/04_unsupervised_learning/PCAVisualization.py
python learning_journey/04_unsupervised_learning/pca_single_sample_animation.py
```

### 3. 学习建议

- **五部分递进路线**：基础（4课）→ 评估专项 → 特征工程与实战 → 部署概念 → 进阶选学
- **先学"工具箱"，再学"度量尺"**：第一部分掌握核心算法（工具箱），第二部分学习评估方法（度量尺），第三部分用完整的工具箱 + 度量尺做端到端实战
- **推荐学习顺序**：按课程编号 1 → 2 → 3 → 4 → 5 → 6 → 7 逐课推进
- **配合笔记**：每个专题目录下的 `notes/` 中包含对应笔记 `.md` 文件，包含核心原理 + 类比理解，建议先读笔记再运行代码
- **进阶选学**：完成第1-7课后，可根据兴趣选学 SVM、XGBoost 等进阶内容

## 课程大纲

| 部分 | 课程 | 内容 | 笔记 | 状态 |
|:---|:---|:---|:---|:---:|
| **第一部分** | 第1课 | 机器学习概述、逻辑回归、Sigmoid/Softmax、梯度下降 | `00_intro` / `01_logistic_regression` | ✅ 已完成 |
| | 第2课 | 线性回归、MSE/R²、多特征扩展、残差分析 | `02_linear_regression` | ✅ 已完成 |
| | 第3课 | 决策树、基尼系数、过拟合、随机森林、特征重要性 | `03_decision_tree` | ✅ 已完成 |
| | 第4课 | K-Means、肘部法则、PCA、轮廓系数、局限性、t-SNE | `04_unsupervised_learning` | ✅ 已完成 |
| **第二部分** | 第5课 | 模型评估与诊断专项 | `05_model_evaluation` | ⏳ 待学习 |
| **第三部分** | 第6课 | 特征工程与综合实战 | `06_feature_engineering_project` | ⏳ 待学习 |
| **第四部分** | 第7课 | 部署概念（API 基础） | `07_deployment` | ⏳ 待学习 |
| **第五部分** | 进阶选学 | SVM、XGBoost、朴素贝叶斯、特征选择系统版 | - | 📚 课程后选学 |

## 学习笔记

- [机器学习学习笔记.md](learning_journey/notes/机器学习学习笔记.md) - 总笔记（覆盖全部章节）
- [逻辑回归笔记](learning_journey/notes/01_logistic_regression/LogisticRegressionNotes.md) - 逻辑回归专题笔记（含 Softmax、梯度下降、超参数详解）
- [线性回归笔记](learning_journey/notes/02_linear_regression/LinearRegressionNotes.md) - 线性回归与残差分析笔记（含噪声 vs 残差区分、多特征扩展）
- [决策树与随机森林笔记](learning_journey/notes/03_decision_tree/DecisionTreeAndRandomForest.md) - 决策树与随机森林专题笔记（含过拟合、特征重要性、模型选择指南）
- [无监督学习笔记](learning_journey/notes/04_unsupervised_learning/UnsupervisedLearning.md) - 无监督学习笔记（K-Means 聚类与 PCA 降维，含肘部法则、轮廓系数、投影值几何动图）
- [课程大纲](course_syllabus.md) - 完整课程体系总览

## 各课知识点

### 第1课：逻辑回归（分类）
- 机器学习基本概念（特征 X / 标签 y / 训练 / 预测）
- 过拟合与欠拟合、训练集/测试集切分
- Sigmoid / Softmax 概率转换
- 交叉熵损失函数、梯度下降直觉
- 决策边界可视化
- ▶ 分类评估指标（混淆矩阵、F1、ROC/AUC）→ 第5课统一处理

### 第2课：线性回归（回归）
- 最小二乘法 / 正规方程
- 噪声 vs 残差的本质区别
- 残差图诊断（U形 → 模型选错）
- 多特征扩展（平面 / 超平面）
- 线性回归对异常值敏感、逻辑回归不敏感
- ✅ **已讲透**：回归评估（MSE、R²、残差图）

### 第3课：决策树与随机森林
- 决策树的"连环问答"逻辑
- **横平竖直的矩形边界** vs 逻辑回归的斜线边界
- 过拟合与 `max_depth` 的关系
- 随机森林的**两个"随机"**：Bootstrap（随机选样本）+ 特征子集（随机选特征）
- `min_samples_leaf` 消除"单间"
- 特征重要性分析
- ▶ 模型复杂度的判断（学习曲线）→ 第5课统一处理

### 第4课：无监督学习（K-Means 与 PCA）
- **无监督学习**：只有特征 X，没有标签 y，模型自己发现数据内部结构
- **K-Means 聚类**：按距离把数据分成 K 堆，簇内紧密、簇间远离
- **肘部法则**：通过 `inertia_` 曲线拐点辅助选择 K 值
- **轮廓系数（Silhouette Score）**：比肘部法则更可靠的选 K 方法，值域 [-1, 1]
- **K-Means 的局限性**：假设簇是球形的，对非凸形状数据失效；对初始质心和异常值敏感
- **标准化（StandardScaler）**：K-Means 和 PCA 都必须先做，否则量纲大的特征会主导结果
- **PCA 降维**：通过 SVD 找出新坐标轴，按方差排序后丢弃方差小的轴
- **累计方差解释率曲线**：辅助判断保留多少个主成分（通常选累计 ≥ 95%）
- **投影值 = 点积**：新坐标 = 原始向量 · 方向向量，几何上就是垂直投影到旋转直线上
- **PCA 不分类**：只做坐标变换，带颜色的散点图是用真实标签上色
- **t-SNE 基础**：非线性降维，擅长可视化高维数据的局部结构

### 第5课（待学习）：模型评估与诊断专项
| 模块 | 内容 |
|:---|:---|
| 5.1 混淆矩阵 | TP/TN/FP/FN 四格表 |
| 5.2 精确率/召回率/F1 | 业务场景配对：癌症筛查看召回率，垃圾邮件看精确率 |
| 5.3 ROC 曲线与 AUC | 模型排序能力的衡量标准 |
| 5.4 回归评估补充 | MAE、MAPE 的业务解读 |
| 5.5 学习曲线 | 量化判断过拟合与欠拟合 |
| 5.6 K-Fold 交叉验证 | 工业界标配验证方法 |

### 第6课（待学习）：特征工程与综合实战
| 模块 | 内容 |
|:---|:---|
| 6.1 数据探索与清洗 | 缺失值、异常值（IQR/Z-score）、分布与相关性 |
| 6.2a 数值特征 | 标准化/归一化、分箱、对数变换 |
| 6.2b 类别特征 | One-Hot 编码、标签编码、频数编码 |
| 6.2c 日期特征 | 提取年/月/日/星期/节假日 |
| 6.2d 缺失值处理 | 均值填充、模型预测填充 |
| 6.2e 特征构造 | 基于业务逻辑的组合特征 |
| 6.3 类别不平衡 | class_weight、SMOTE、欠采样 |
| 6.4 文本特征 | CountVectorizer / TfidfVectorizer |
| 6.5 调优与保存 | GridSearchCV、joblib 模型保存/加载 |
| 6.6 项目复盘 | 量化业务价值、模拟面试追问 |

### 第7课（待学习）：部署概念
| 模块 | 内容 |
|:---|:---|
| 7.1 部署基础 | 理解模型上线流程 |
| 7.2 极简 API 示例 | FastAPI / Flask 接口示例 |

## 模型选择指南

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 需要解释预测逻辑 | 逻辑回归 / 单棵决策树 | 权重 / 树结构可直接解释 |
| 追求最高准确率，数据量大 | 随机森林 | 集成投票稳定，抗噪声 |
| 数据量小，快速实验 | 逻辑回归（基准模型） | 训练快，结果稳定 |
| 规则提取 / EDA | 单棵决策树 | 可视化树状图直观 |
| 无标签数据，发现结构 | K-Means 聚类 | 自动分组，无需标签 |
| 高维数据可视化 / 去噪 | PCA 降维 | 保留主要方差，压缩维度 |

## 技术栈

- **Python 3.9+**
- **NumPy / Pandas**: 数据处理
- **Matplotlib**: 数据可视化
- **Scikit-learn**: 机器学习算法（LinearRegression, LogisticRegression, DecisionTree, RandomForest, KMeans, PCA）

## 进阶路线（课程后选学）

完成以上课程后，可根据兴趣选学以下内容：
- **SVM（支持向量机）** — 另一类分类模型
- **XGBoost** — 工业界最常用的树模型集成框架
- **朴素贝叶斯** — 文本分类的经典入门模型
- **特征选择（系统版）** — 过滤法/包裹法/嵌入法完整介绍