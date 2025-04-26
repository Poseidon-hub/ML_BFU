import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Функция для вычисления MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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


# Вычисление метрик для Scikit-Learn модели
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)
mape_sklearn = mean_absolute_percentage_error(y_test, y_pred_sklearn)

# Вычисление метрик для собственной модели
mae_my = mean_absolute_error(y_test, y_pred_my)
r2_my = r2_score(y_test, y_pred_my)
mape_my = mean_absolute_percentage_error(y_test, y_pred_my)

# Создаем таблицу с метриками
metrics_df = pd.DataFrame({
    'Метрика': ['MAE', 'R²', 'MAPE (%)'],
    'Scikit-Learn модель': [mae_sklearn, r2_sklearn, mape_sklearn],
    'Моя модель': [mae_my, r2_my, mape_my]
})

print("\nСравнение метрик качества моделей:")
print(metrics_df)

# Вывод о качестве моделей
print("\nВывод о качестве моделей:")
if np.allclose([mae_sklearn, r2_sklearn, mape_sklearn], [mae_my, r2_my, mape_my]):
    print("1. Обе модели показывают идентичные результаты, что подтверждает корректность собственной реализации.")
else:
    print("1. Модели показывают разные результаты, возможно есть ошибка в реализации.")

print("2. R² значение около", round(r2_sklearn, 2), "указывает на то, что модель объясняет примерно",
      round(r2_sklearn*100, 1), "% дисперсии целевой переменной.")

print("3. MAE в", round(mae_sklearn, 1), "означает, что в среднем модель ошибается на эту величину.")

print("4. MAPE в", round(mape_sklearn, 1), "% показывает среднюю процентную ошибку предсказаний.")

if r2_sklearn > 0.5:
    print("5. Модель можно считать относительно хорошей для этих данных.")
elif r2_sklearn > 0.3:
    print("5. Модель умеренного качества, есть место для улучшений.")
else:
    print("5. Модель низкого качества, возможно стоит попробовать другие признаки или методы.")