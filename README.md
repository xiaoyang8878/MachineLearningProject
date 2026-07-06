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
│   │   └── 02_logistic_regression/
│   │       └── LogisticRegressionNotes.md
│   │
│   ├── 01_basics/                             # 第1课：机器学习入门
│   │   ├── HelloMachineLearing.py             # 入门示例
│   │   └── BikeDemandPrediction.py            # 入门 demo
│   ├── 02_logistic_regression/                # 第2课：逻辑回归（鸢尾花分类）
│   │   └── HelloLogisticRegression.py
│   ├── 03_linear_regression/                  # 第3课：线性回归
│   │   ├── HelloLinearRegression.py           # 线性回归（无噪声）
│   │   └── HelloLinearRegression_mutiLable.py # 多特征线性回归
│   └── 04_advanced/                           # 后续进阶内容（待填充）
│
├── data/                                      # 数据目录
│   └── raw/                                   # 原始数据
│       ├── bike-day.csv                       # 共享单车数据集
│       └── insurance.csv                      # 保险数据集
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

# 第3课拓展：多特征线性回归 - 面积+卧室数预测房价
python learning_journey/03_linear_regression/HelloLinearRegression_mutiLable.py
```

## 学习路线

| 顺序 | 主题                 | 代码文件 | 笔记 | 状态 |
|------|--------------------|----------|------|------|
| 1 | 机器学习入门             | `HelloMachineLearing.py` | - | 已完成 |
| 2 | 分类入门 - 逻辑回归（鸢尾花分类） | `HelloLogisticRegression.py` | `LogisticRegressionNotes.md` | 已完成 |
| 3 | 回归入门 - 线性回归        | `HelloLinearRegression.py` / `HelloLinearRegression_mutiLable.py` | `LinearRegressionNotes.md` | 已完成 |
| 4 | 非线性的力量               | - | - | 待学习 |
| 5 | 无监督探索                 | - | - | 待学习 |
| 6 | 全流程项目             | - | - | 待学习 |



## 学习笔记

- [机器学习学习笔记.md](learning_journey/00_notes/机器学习学习笔记.md) - 总笔记
- [逻辑回归笔记](learning_journey/00_notes/02_logistic_regression/LogisticRegressionNotes.md) - 逻辑回归专题笔记
- [线性回归笔记](learning_journey/00_notes/03_linear_regression/LinearRegressionNotes.md) - 线性回归与残差分析笔记

## 技术栈

- **Python 3.9+**
- **NumPy/Pandas**: 数据处理
- **Matplotlib**: 数据可视化
- **Scikit-learn**: 机器学习算法
