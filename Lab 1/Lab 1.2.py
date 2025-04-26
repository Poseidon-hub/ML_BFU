import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Загрузка набора данных
diabetes = datasets.load_diabetes()
print(diabetes.DESCR)  # Выводим описание набора данных

# 2. Исследование данных и выбор столбца
# Преобразуем данные в DataFrame для удобства
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("\nПервые 5 строк данных:")
print(df.head())

print("\nСтатистика данных:")
print(df.describe())

# Выберем столбец 'bmi' (индекс массы тела), так как он имеет четкую линейную зависимость с target
X = df[['bmi']].values
y = df['target'].values

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Реализация линейной регрессии с помощью Scikit-Learn
# Создаем и обучаем модель
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

# Предсказания на тестовых данных
y_pred_sklearn = sklearn_model.predict(X_test)

# Выводим коэффициенты
print("\nScikit-Learn модель:")
print(f"Коэффициент (угол наклона): {sklearn_model.coef_[0]:.4f}")
print(f"Пересечение (intercept): {sklearn_model.intercept_:.4f}")
print(f"R^2 score: {r2_score(y_test, y_pred_sklearn):.4f}")


# 4. Реализация собственного алгоритма линейной регрессии
class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Добавляем столбец единиц для intercept
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]

        # Вычисляем коэффициенты по нормальному уравнению
        theta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X):
        return self.intercept_ + X.dot(self.coef_)


# Создаем и обучаем собственную модель
my_model = MyLinearRegression()
my_model.fit(X_train, y_train)

# Предсказания на тестовых данных
y_pred_my = my_model.predict(X_test)

# Выводим коэффициенты
print("\nСобственная модель:")
print(f"Коэффициент (угол наклона): {my_model.coef_[0]:.4f}")
print(f"Пересечение (intercept): {my_model.intercept_:.4f}")
print(f"R^2 score: {r2_score(y_test, y_pred_my):.4f}")

# 5. Визуализация данных и регрессионных прямых
plt.figure(figsize=(12, 6))

# Отображаем данные
plt.scatter(X_test, y_test, color='black', label='Реальные данные')

# Отображаем предсказания Scikit-Learn
plt.plot(X_test, y_pred_sklearn, color='blue', linewidth=2, label='Scikit-Learn регрессия')

# Отображаем предсказания собственной модели
plt.plot(X_test, y_pred_my, color='red', linestyle='dashed', linewidth=2, label='Моя регрессия')

plt.xlabel('Индекс массы тела (BMI)')
plt.ylabel('Зависимая переменная (target)')
plt.title('Линейная регрессия для набора данных Diabetes')
plt.legend()
plt.show()

# 6. Таблица с результатами предсказаний
results = pd.DataFrame({
    'Реальное значение': y_test,
    'Scikit-Learn предсказание': y_pred_sklearn,
    'Моя модель предсказание': y_pred_my,
    'Scikit-Learn ошибка': np.abs(y_test - y_pred_sklearn),
    'Моя модель ошибка': np.abs(y_test - y_pred_my)
})

print("\nТаблица с результатами предсказаний (первые 10 строк):")
print(results.head(10))