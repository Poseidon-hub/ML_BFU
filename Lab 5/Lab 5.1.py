import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Разделение на признаки и целевую переменную
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Логистическая регрессия
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

# Решающее дерево (стандартные параметры)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:, 1]

# Метрики для логистической регрессии
print("Метрики для логистической регрессии:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_log):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_log):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred_log):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_log):.3f}")
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred_log))
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred_log))

# Метрики для решающего дерева
print("\nМетрики для решающего дерева:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_tree):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_tree):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred_tree):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_tree):.3f}")
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred_tree))
print("\nОтчет классификации:")
print(classification_report(y_test, y_pred_tree))

# Исследуем глубину дерева от 1 до 20
max_depths = range(1, 21)
train_scores = []
test_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    # Оценка на тренировочных данных
    y_train_pred = tree.predict(X_train)
    train_scores.append(f1_score(y_train, y_train_pred))

    # Оценка на тестовых данных
    y_test_pred = tree.predict(X_test)
    test_scores.append(f1_score(y_test, y_test_pred))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, label='Train F1-score', marker='o')
plt.plot(max_depths, test_scores, label='Test F1-score', marker='o')
plt.xlabel('Глубина дерева')
plt.ylabel('F1-score')
plt.title('Зависимость F1-score от глубины решающего дерева')
plt.legend()
plt.grid(True)
plt.xticks(max_depths)
plt.show()

# Найдем оптимальную глубину
optimal_depth = max_depths[np.argmax(test_scores)]
print(f"Оптимальная глубина дерева: {optimal_depth}")

# Оптимальное дерево
optimal_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
optimal_tree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(optimal_tree, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'],
          filled=True, rounded=True, proportion=True, max_depth=3)
plt.title(f"Решающее дерево (глубина={optimal_depth})")
plt.show()

# Важность признаков
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': optimal_tree.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Важность признаков в решающем дереве')
plt.show()

# Прогнозы оптимального дерева
y_prob_optimal = optimal_tree.predict_proba(X_test)[:, 1]

# PR кривая
precision, recall, _ = precision_recall_curve(y_test, y_prob_optimal)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# ROC кривая
fpr, tpr, _ = roc_curve(y_test, y_prob_optimal)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()