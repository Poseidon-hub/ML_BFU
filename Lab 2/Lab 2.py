import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('housing.csv', delim_whitespace=True, header=None)
data.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Разделим данные на 5 частей: 20%, 40%, 60%, 80%, 100%
sample_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
metrics = {'Size': [], 'MSE': [], 'R2': [], 'MAE': []}

for size in sample_sizes:
    if size < 1.0:
        # Для размеров меньше 100% используем train_test_split
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=size, random_state=42)
    else:
        # Для 100% используем все данные
        X_sample, y_sample = X, y

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics['Size'].append(size)
    metrics['MSE'].append(mean_squared_error(y_test, y_pred))
    metrics['R2'].append(r2_score(y_test, y_pred))
    metrics['MAE'].append(mean_absolute_error(y_test, y_pred))

# Визуализация результатов
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(metrics['Size'], metrics['MSE'], 'o-')
plt.title('MSE vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('MSE')

plt.subplot(1, 3, 2)
plt.plot(metrics['Size'], metrics['R2'], 'o-')
plt.title('R2 Score vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('R2 Score')

plt.subplot(1, 3, 3)
plt.plot(metrics['Size'], metrics['MAE'], 'o-')
plt.title('MAE vs Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

"""Выводы по количеству данных:
С увеличением количества данных MSE уменьшается, что указывает на улучшение точности модели.
R2-score увеличивается с ростом объема данных, достигая стабильного уровня при 60-80% данных.
MAE также уменьшается с увеличением объема данных.
Наибольший прирост точности наблюдается при увеличении данных с 20% до 60%."""

# Выберем разные комбинации признаков
feature_sets = [
    ['RM', 'LSTAT'],  # 2 признака
    ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'CRIM', 'AGE'],  # 7 признаков
    list(X.columns)  # все 13 признаков
]

metrics_features = {'Num Features': [], 'MSE': [], 'R2': [], 'MAE': []}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for features in feature_sets:
    model = LinearRegression()
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])

    metrics_features['Num Features'].append(len(features))
    metrics_features['MSE'].append(mean_squared_error(y_test, y_pred))
    metrics_features['R2'].append(r2_score(y_test, y_pred))
    metrics_features['MAE'].append(mean_absolute_error(y_test, y_pred))

# Визуализация результатов
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(metrics_features['Num Features'], metrics_features['MSE'], 'o-')
plt.title('MSE vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('MSE')

plt.subplot(1, 3, 2)
plt.plot(metrics_features['Num Features'], metrics_features['R2'], 'o-')
plt.title('R2 Score vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('R2 Score')

plt.subplot(1, 3, 3)
plt.plot(metrics_features['Num Features'], metrics_features['MAE'], 'o-')
plt.title('MAE vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

"""Выводы по количеству признаков:
Увеличение количества признаков с 2 до 7 значительно улучшает все метрики.
Добавление всех 13 признаков дает небольшое дополнительное улучшение по сравнению с 7 признаками.
Наиболее важными признаками являются RM (среднее количество комнат) и LSTAT (статус населения), но дополнительные признаки также содержат полезную информацию."""

# Выбираем 2 признака для визуализации
features = ['RM', 'LSTAT']
X_2d = X[features]

# Обучаем модель
model = LinearRegression()
model.fit(X_2d, y)

# Создаем сетку для визуализации плоскости
x1 = np.linspace(X_2d['RM'].min(), X_2d['RM'].max(), 10)
x2 = np.linspace(X_2d['LSTAT'].min(), X_2d['LSTAT'].max(), 10)
x1, x2 = np.meshgrid(x1, x2)
y_plane = model.intercept_ + model.coef_[0]*x1 + model.coef_[1]*x2

# Визуализация
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Точки данных
ax.scatter(X_2d['RM'], X_2d['LSTAT'], y, c='b', marker='o', alpha=0.5)

# Плоскость регрессии
ax.plot_surface(x1, x2, y_plane, color='r', alpha=0.3)

ax.set_xlabel('RM (среднее число комнат)')
ax.set_ylabel('LSTAT (% населения с низким статусом)')
ax.set_zlabel('MEDV (цена дома, $1000)')
plt.title('3D визуализация линейной регрессии')
plt.show()