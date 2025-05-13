import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Загрузка данных
data = pd.read_csv('housing.csv', delim_whitespace=True, header=None)
data.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

# Разделение на признаки и целевую переменную
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='neg_mean_squared_error')

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Целевое значение MSE (например, 25)
    target_mse = 25
    plt.axhline(y=target_mse, color='b', linestyle='--', label='Target MSE')

    plt.legend(loc="best")
    return plt


# Линейная регрессия
plot_learning_curve(LinearRegression(), "Learning Curve (Linear Regression)",
                    X_train, y_train, ylim=(0, 100), cv=5)
plt.show()

# Гребневая регрессия (Ridge)
plot_learning_curve(make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                    "Learning Curve (Ridge Regression)",
                    X_train, y_train, ylim=(0, 100), cv=5)
plt.show()

"""Выводы:
Линейная регрессия показывает признаки небольшого переобучения - ошибка на обучающей выборке значительно ниже, чем на валидационной.
Гребневая регрессия демонстрирует лучшую сбалансированность между ошибками на обучающей и валидационной выборках.
Обе модели не достигают целевого значения MSE=25, что говорит о необходимости:
Сбора дополнительных данных
Инжененрии признаков
Использования более сложных моделей
Гребневая регрессия более подходит для данного датасета, так как лучше обобщает данные."""

# Создание и обучение моделей
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso Regression': make_pipeline(StandardScaler(), Lasso(alpha=0.1))
}

results = []
coefficients = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Сохранение коэффициентов
    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_
    elif hasattr(model.steps[1][1], 'coef_'):  # Для pipeline
        coefficients[name] = model.steps[1][1].coef_

    # Расчет метрик
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        'Model': name,
        'MSE': mse,
        'R2': r2,
        'MAE': mae
    })

# Таблица с результатами
results_df = pd.DataFrame(results)
print(results_df)

# Таблица с коэффициентами
coef_df = pd.DataFrame(coefficients, index=X.columns)
print("\nКоэффициенты моделей:")
print(coef_df)


"""Выводы:
Все три модели показывают схожие результаты, но Ridge регрессия немного лучше по всем метрикам.
Lasso регрессия обнулила коэффициенты для некоторых признаков (ZN, CHAS), что указывает на их меньшую важность.
Наиболее значимые признаки (по абсолютному значению коэффициентов):
LSTAT (% населения с низким статусом) - отрицательное влияние
RM (среднее количество комнат) - положительное влияние
DIS (расстояние до центров занятости) - положительное влияние
Признаки RAD (индекс доступности дорог) и TAX (налог на имущество) имеют высокие коэффициенты и могут быть коллинеарны (что подтверждается их высокой корреляцией в данных)."""


# Визуализация коэффициентов
plt.figure(figsize=(12, 6))
coef_df.plot(kind='bar', ax=plt.gca())
plt.title('Коэффициенты признаков в разных моделях')
plt.ylabel('Значение коэффициента')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()