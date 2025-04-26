# Titanic Dataset Analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка данных
titanic = pd.read_csv('titanic.csv')  # Предполагаем, что файл titanic.csv в той же директории

# 1. Предобработка данных
print("Исходные размеры данных:", titanic.shape)

# 1.1. Удаление строк с пропусками
titanic_clean = titanic.dropna()
print("Размер после удаления строк с пропусками:", titanic_clean.shape)

# 1.2. Удаление нечисловых столбцов (кроме Sex и Embarked)
cols_to_drop = [col for col in titanic_clean.columns
               if titanic_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
titanic_clean = titanic_clean.drop(cols_to_drop, axis=1)
print("Столбцы после удаления нечисловых:", titanic_clean.columns.tolist())

# 1.3. Перекодировка категориальных признаков
titanic_clean['Sex'] = titanic_clean['Sex'].map({'male': 0, 'female': 1})
titanic_clean['Embarked'] = titanic_clean['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

# 1.4. Удаление PassengerId
titanic_clean = titanic_clean.drop('PassengerId', axis=1)

# 1.5. Расчет процента потерянных данных
initial_size = titanic.shape[0]
final_size = titanic_clean.shape[0]
data_loss = (initial_size - final_size) / initial_size * 100
print(f"\nПроцент потерянных данных: {data_loss:.2f}%")

# 2. Машинное обучение
# 2.1. Разделение данных
X = titanic_clean.drop('Survived', axis=1)
y = titanic_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2.2. Обучение модели
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 2.3. Оценка точности
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f}")

# 2.4. Влияние признака Embarked
X_no_embarked = X.drop('Embarked', axis=1)
X_train_ne, X_test_ne, y_train_ne, y_test_ne = train_test_split(X_no_embarked, y, test_size=0.3, random_state=42)

model_ne = LogisticRegression(max_iter=1000, random_state=42)
model_ne.fit(X_train_ne, y_train_ne)
y_pred_ne = model_ne.predict(X_test_ne)
accuracy_ne = accuracy_score(y_test_ne, y_pred_ne)

print(f"Точность без Embarked: {accuracy_ne:.4f}")
print(f"Разница в точности: {accuracy - accuracy_ne:.4f}")

# Вывод коэффициентов модели
print("\nКоэффициенты модели:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")