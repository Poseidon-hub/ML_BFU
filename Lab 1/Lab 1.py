import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_stats(data):
    return {
        'count': len(data),
        'min': min(data),
        'max': max(data),
        'mean': sum(data) / len(data)
    }

# Чтение данных из файла
filename = 'student_scores.csv'
x_col = int(input('Для X введите колонку 0 или 1: '))
y_col = int(input('Для Y введите колонку 0 или 1: '))
x_data = []
y_data = []

with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= max(x_col, y_col) + 1:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                x_data.append(x)
                y_data.append(y)
            except ValueError:
                continue

# Вывод статистики
stats_x = calculate_stats(x_data)
stats_y = calculate_stats(y_data)

print("Статистика для X:")
print(f"Количество: {stats_x['count']}")
print(f"Минимум: {stats_x['min']:.2f}")
print(f"Максимум: {stats_x['max']:.2f}")
print(f"Среднее: {stats_x['mean']:.2f}")

print("\nСтатистика для Y:")
print(f"Количество: {stats_y['count']}")
print(f"Минимум: {stats_y['min']:.2f}")
print(f"Максимум: {stats_y['max']:.2f}")
print(f"Среднее: {stats_y['mean']:.2f}")

# Создание фигуры с тремя subplot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Исходные данные
ax1.scatter(x_data, y_data, color='blue')
ax1.set_title('Исходные данные')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True)

# Вычисление параметров регрессии
n = len(x_data)
sum_x = sum(x_data)
sum_y = sum(y_data)
sum_x_squared = sum(xi ** 2 for xi in x_data)
sum_xy = sum(xi * yi for xi, yi in zip(x_data, y_data))

a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
b = (sum_y - a * sum_x) / n

x_min, x_max = min(x_data), max(x_data)
y_pred_min = a * x_min + b
y_pred_max = a * x_max + b

# Регрессионная прямая
ax2.scatter(x_data, y_data, color='blue')
ax2.plot([x_min, x_max], [y_pred_min, y_pred_max], color='red')
ax2.set_title('Регрессионная прямая')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True)

# Квадраты ошибок
ax3.scatter(x_data, y_data, color='red', zorder=3)  # Точки
ax3.plot([x_min, x_max], [y_pred_min, y_pred_max], color='blue', zorder=2)  # Линия регрессии

for xi, yi in zip(x_data, y_data):
    y_pred_i = a * xi + b
    error = abs(y_pred_i - yi)  # Величина ошибки (сторона квадрата)

    # Вертикальная линия от точки до регрессионной прямой
    ax3.vlines(xi, min(yi, y_pred_i), max(yi, y_pred_i), color='black', linestyle='--')

    # Нижний левый угол квадрата (смещаем влево от вертикальной линии на величину ошибки)
    rect_x = xi - error
    rect_y = min(yi, y_pred_i)  # Нижняя граница квадрата

    # Создаем квадрат (ширина = высота = error)
    square = patches.Rectangle((rect_x, rect_y), error, error, facecolor='red', alpha=0.3, edgecolor='black')

    # Добавляем квадрат на график
    ax3.add_patch(square)

ax3.set_title('Квадраты ошибок')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.grid(True)

plt.tight_layout()
plt.show()

