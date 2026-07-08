# 目的: 使用皮马印第安人糖尿病数据集训练随机森林，输出特征重要性排序并绘制柱状图

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'Pima Indians Diabetes Datase.csv')
data_path = os.path.normpath(data_path)
print("数据文件路径:", data_path)
df = pd.read_csv(data_path)

# 查看列名
print("数据集形状:", df.shape)
print("列名:", list(df.columns))
print(df.head())

# 2. 划分特征和标签
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_names].values
y = df['Outcome'].values

# 3. 切分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 训练随机森林
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 提取特征重要性
importances = model.feature_importances_

# 6. 按重要性降序排序
sorted_idx = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_scores = importances[sorted_idx]

# 7. 打印特征重要性排序
print("\n特征重要性排序（从高到低）：")
for name, score in zip(sorted_names, sorted_scores):
    print(f"  {name}: {score:.4f}")

# 8. 画柱状图
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_scores)), sorted_scores, color='steelblue')
plt.yticks(range(len(sorted_scores)), sorted_names)
plt.xlabel('重要性分数')
plt.title('皮马印第安人糖尿病 - 随机森林特征重要性 (100棵树)')
plt.gca().invert_yaxis()  # 最重要的显示在最上面

# 在柱状图上标注数值
for bar, score in zip(bars, sorted_scores):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{score:.4f}', va='center', fontsize=10)

plt.tight_layout()
output_path = os.path.join(script_dir, 'PimaFeatureImportance.png')
plt.savefig(output_path, dpi=150)
print(f"\n柱状图已保存为 {output_path}")
# 不调用 plt.show()，避免在无 GUI 环境下阻塞
plt.close()

# 9. 模型分类结果输出
y_pred = model.predict(X_test)

print("\n" + "="*60)
print("随机森林模型分类结果")
print("="*60)

# 9a. 准确率
accuracy = model.score(X_test, y_test)
print(f"\n测试集准确率: {accuracy:.3f}")

# 9b. 分类报告（精确率、召回率、F1-score）
print("\n分类报告（Precision / Recall / F1-score）：")
print(classification_report(y_test, y_pred, target_names=['未患病(0)', '患病(1)']))

# 9c. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：")
print(f"           预测未患病  预测患病")
print(f"实际未患病    {cm[0,0]:3d}        {cm[0,1]:3d}")
print(f"实际患病      {cm[1,0]:3d}        {cm[1,1]:3d}")

# 9d. 显示前20条预测结果对比
print("\n前20条测试样本预测结果对比：")
print(f"{'序号':>4}  {'实际值':>6}  {'预测值':>6}  {'结果':>4}")
print("-" * 28)
for i in range(min(20, len(y_test))):
    mark = "OK" if y_test[i] == y_pred[i] else "NO"
    print(f"{i+1:>4}  {y_test[i]:>6}  {y_pred[i]:>6}  {mark:>4}")

# 9e. 保存混淆矩阵图
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues',
                                         display_labels=['未患病(0)', '患病(1)'])
plt.title('随机森林混淆矩阵')
cm_path = os.path.join(script_dir, 'PimaConfusionMatrix.png')
plt.savefig(cm_path, dpi=150)
print(f"\n混淆矩阵图已保存为 {cm_path}")
plt.close()
