import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings

# Загрузка данных
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""Подбор гиперпараметров с использованием Random Search (Scikit-Learn)"""

# Определение пространства параметров для Random Forest
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'max_features': randint(1, X.shape[1]+1),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'bootstrap': [True, False]
}

# Создание модели
rf = RandomForestClassifier(random_state=42)

# Настройка Random Search
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Запуск поиска и замер времени
start_time = time.time()
random_search.fit(X_train, y_train)
random_search_time = time.time() - start_time

# Лучшие параметры и точность
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print("Random Search результаты:")
print(f"Лучшие параметры: {best_params_random}")
print(f"Лучшая точность: {best_score_random:.4f}")
print(f"Время выполнения: {random_search_time:.2f} сек")

"""Подбор гиперпараметров с использованием TPE (Hyperopt)"""

# Определение пространства параметров для Hyperopt
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'max_features': hp.quniform('max_features', 1, X.shape[1], 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
    'bootstrap': hp.choice('bootstrap', [True, False])
}

warnings.filterwarnings("ignore", category=FutureWarning)

# Функция для оптимизации
def objective(params):
    # Преобразуем параметры в нужные типы
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'max_features': int(params['max_features']),
        'min_samples_split': int(params['min_samples_split']),
        'min_samples_leaf': int(params['min_samples_leaf']),
        'bootstrap': bool(params['bootstrap']),  # Явное преобразование в bool
        'random_state': 42
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()
start_time = time.time()

# Используем np.random.default_rng()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=np.random.default_rng(42)
)

hyperopt_time = time.time() - start_time

# Преобразование результатов
best_params_hyperopt = {
    'n_estimators': int(best['n_estimators']),
    'max_depth': int(best['max_depth']),
    'max_features': int(best['max_features']),
    'min_samples_split': int(best['min_samples_split']),
    'min_samples_leaf': int(best['min_samples_leaf']),
    'bootstrap': best['bootstrap']
}

# Оценка лучшей модели
rf_hyperopt = RandomForestClassifier(**best_params_hyperopt, random_state=42)
rf_hyperopt.fit(X_train, y_train)
y_pred_hyperopt = rf_hyperopt.predict(X_test)
best_score_hyperopt = accuracy_score(y_test, y_pred_hyperopt)

print("\nHyperopt (TPE) результаты:")
print(f"Лучшие параметры: {best_params_hyperopt}")
print(f"Лучшая точность: {best_score_hyperopt:.4f}")
print(f"Время выполнения: {hyperopt_time:.2f} сек")

"""Сравнительный анализ результатов"""

# Создание таблицы сравнения
comparison = pd.DataFrame({
    'Метод': ['Random Search', 'Hyperopt (TPE)'],
    'Точность': [best_score_random, best_score_hyperopt],
    'Время (сек)': [random_search_time, hyperopt_time],
    'n_estimators': [best_params_random['n_estimators'], best_params_hyperopt['n_estimators']],
    'max_depth': [best_params_random['max_depth'], best_params_hyperopt['max_depth']],
    'max_features': [best_params_random['max_features'], best_params_hyperopt['max_features']],
    'min_samples_split': [best_params_random['min_samples_split'], best_params_hyperopt['min_samples_split']],
    'min_samples_leaf': [best_params_random['min_samples_leaf'], best_params_hyperopt['min_samples_leaf']],
    'bootstrap': [best_params_random['bootstrap'], best_params_hyperopt['bootstrap']]
})

print("\nСравнительный анализ:")
print(comparison)

# Визуализация сравнения
plt.figure(figsize=(10, 5))
plt.bar(['Random Search', 'Hyperopt (TPE)'], [best_score_random, best_score_hyperopt])
plt.title('Сравнение точности методов оптимизации')
plt.ylabel('Точность')
plt.ylim(0.7, 0.85)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(['Random Search', 'Hyperopt (TPE)'], [random_search_time, hyperopt_time], color='orange')
plt.title('Сравнение времени выполнения методов оптимизации')
plt.ylabel('Время (сек)')
plt.grid(axis='y')
plt.show()