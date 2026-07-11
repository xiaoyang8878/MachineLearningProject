# 目的: 使用皮马印第安人糖尿病数据集训练随机森林，输出4张结果图：
#       1. 混淆矩阵热力图  2. ROC曲线  3. 特征重要性排序图  4. PCA降维散点图

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
data_path = os.path.normpath(os.path.join(script_dir, '..', '..', '..',
                                          'data', 'raw', 'Pima Indians Diabetes Datase.csv'))
print("数据文件路径:", data_path)
df = pd.read_csv(data_path)

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_names].values
y = df['Outcome'].values

print("数据集形状:", df.shape)
print("正样本(患病)数量:", (y == 1).sum(), "  负样本(未患病)数量:", (y == 0).sum())

# ============================================================
# 2. 切分 + 标准化 + 训练
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # 正类概率

accuracy = model.score(X_test_scaled, y_test)
print(f"\n测试集准确率: {accuracy:.3f}")

# ============================================================
# 3. 图1: 混淆矩阵热力图
# ============================================================
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues',
                                         display_labels=['未患病(0)', '患病(1)'])
plt.title('随机森林混淆矩阵 (Confusion Matrix)')
cm_path = os.path.join(results_dir, 'PimaConfusionMatrix.png')
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
print(f"混淆矩阵热力图已保存: {cm_path}")
plt.close()

# ============================================================
# 4. 图2: ROC 曲线
# ============================================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='随机猜测 (AUC=0.5)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('随机森林 ROC 曲线')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
roc_path = os.path.join(results_dir, 'PimaROC.png')
plt.tight_layout()
plt.savefig(roc_path, dpi=150)
print(f"ROC 曲线已保存: {roc_path}")
plt.close()

# ============================================================
# 5. 图3: 特征重要性排序图
# ============================================================
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_scores = importances[sorted_idx]

print("\n特征重要性排序（从高到低）：")
for name, score in zip(sorted_names, sorted_scores):
    print(f"  {name}: {score:.4f}")

plt.figure(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_scores)))
bars = plt.barh(range(len(sorted_scores)), sorted_scores, color=colors)
plt.yticks(range(len(sorted_scores)), sorted_names)
plt.xlabel('重要性分数')
plt.title('随机森林特征重要性排序 (Feature Importance)')
plt.gca().invert_yaxis()

for bar, score in zip(bars, sorted_scores):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{score:.4f}', va='center', fontsize=10)

fi_path = os.path.join(results_dir, 'PimaFeatureImportance.png')
plt.tight_layout()
plt.savefig(fi_path, dpi=150)
print(f"特征重要性排序图已保存: {fi_path}")
plt.close()

# ============================================================
# 6. 图4: PCA 降维散点图
# ============================================================
# 对整个数据集做 PCA 降维到 2D
X_all_scaled = scaler.transform(X)  # 用训练集的 scaler 统一变换
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)

explained_var = pca.explained_variance_ratio_
print(f"\nPCA 前两个主成分解释方差比例: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
print(f"累计解释方差: {explained_var.sum():.3f}")

plt.figure(figsize=(9, 7))
colors = ['steelblue', 'coral']
labels = ['未患病 (0)', '患病 (1)']
for cls in [0, 1]:
    mask = (y == cls)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[cls], label=labels[cls], alpha=0.6,
                edgecolors='white', linewidth=0.3, s=40)

plt.xlabel(f'第一主成分 (PC1) - 解释方差 {explained_var[0]:.1%}')
plt.ylabel(f'第二主成分 (PC2) - 解释方差 {explained_var[1]:.1%}')
plt.title('PCA 降维散点图 (皮马印第安人糖尿病数据集)')
plt.legend()
plt.grid(alpha=0.3)
pca_path = os.path.join(results_dir, 'PimaPCA.png')
plt.tight_layout()
plt.savefig(pca_path, dpi=150)
print(f"PCA 降维散点图已保存: {pca_path}")
plt.close()

# ============================================================
# 7. 打印分类报告
# ============================================================
print("\n" + "="*60)
print("随机森林模型分类结果")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['未患病(0)', '患病(1)']))

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(f"           预测未患病  预测患病")
print(f"实际未患病    {cm[0,0]:3d}        {cm[0,1]:3d}")
print(f"实际患病      {cm[1,0]:3d}        {cm[1,1]:3d}")

print(f"\nROC AUC: {roc_auc:.3f}")
print("\n全部 4 张结果图已生成完毕！")
