# Titanic Dataset Analysis with Extended Metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import label_binarize

# 1. Загрузка и предобработка данных
titanic = pd.read_csv('titanic.csv')

# Предобработка (как в предыдущей работе)
titanic_clean = titanic.dropna()
cols_to_drop = [col for col in titanic_clean.columns
                if titanic_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
titanic_clean = titanic_clean.drop(cols_to_drop, axis=1)
titanic_clean['Sex'] = titanic_clean['Sex'].map({'male': 0, 'female': 1})
titanic_clean['Embarked'] = titanic_clean['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
titanic_clean = titanic_clean.drop('PassengerId', axis=1)

X = titanic_clean.drop('Survived', axis=1)
y = titanic_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Функция для вычисления и визуализации метрик
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    # Основные метрики
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Кривая PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    # Кривая ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc
    }


# Часть 1: Логистическая регрессия с дополнительными метриками
print("\n" + "=" * 50)
print("Часть 1: Логистическая регрессия с расширенными метриками")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
logreg_metrics = evaluate_model(logreg, X_test, y_test, "Logistic Regression")

# Часть 2: Сравнение моделей
print("\n" + "=" * 50)
print("Часть 2: Сравнение моделей классификации")

# Метод опорных векторов
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_metrics = evaluate_model(svm, X_test, y_test, "SVM")

# Метод ближайших соседей
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_metrics = evaluate_model(knn, X_test, y_test, "KNN")

# Сравнение моделей
metrics_df = pd.DataFrame([logreg_metrics, svm_metrics, knn_metrics],
                          index=['Logistic Regression', 'SVM', 'KNN'])
print("\nСравнение метрик всех моделей:")
print(metrics_df)

# Визуализация сравнения
plt.figure(figsize=(10, 5))
metrics_df.plot(kind='bar', rot=0)
plt.title('Сравнение метрик моделей')
plt.ylabel('Значение метрики')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Вывод о наилучшей модели
best_model = metrics_df['f1'].idxmax()
print(f"\nНаилучшая модель по F1-score: {best_model}")