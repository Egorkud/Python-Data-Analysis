import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

# Ініціалізація змінних
# Параметри варіанту
g = 8
k = 9

t_start = 0 # Початковий момент
t_end = 1   # Кінцевий момент
step = 0.05  # Крок, можна змінювати
steps_num = int(1 / step) + 1 # Кількість кроків, включно з останнім, само підлаштовується
initial_conditions = [0, 0]  # Початкові значення змінних: x(0) = 0; y(0) = 0
accuracy = 0.1  # Цільова точність для RMSE

# Система рівнянь для повернення в інших методах
def euler_system(t, state_vector_Y):
    x, y = state_vector_Y
    dx_dt = g / k * t + x - y + k / g   # Похідна по х
    dy_dt = -x + k / g * y              # Похідна по у

    return np.array([dx_dt, dy_dt])     # Повернення масивом


# Реалізація методу Ейлера
def euler_method(function, t_start, t_end, initial_conditions, step):
    t_values = np.arange(t_start, t_end + step, step)
    n = len(t_values)
    y_values = np.zeros((n, len(initial_conditions)))
    y_values[0] = initial_conditions

    for i in range(1, n):
        y_values[i] = (y_values[i - 1] + step *
                       function(t_values[i - 1], y_values[i - 1]))

    return t_values, y_values           # Повертаємо час та значення змінних


# Реалізація методу Рунге-Кута 4-го порядку
def runge_kuta_4(function, t_start, t_end, initial_conditions, step):
    t_values = np.arange(t_start, t_end + step, step)
    n = len(t_values)
    y_values = np.zeros((n, len(initial_conditions)))
    y_values[0] = initial_conditions

    for i in range(1, n):
        t = t_values[i - 1]
    y = y_values[i - 1]

    k1 = step * function(t, y)
    k2 = step * function(t + step / 2, y + k1 / 2)
    k3 = step * function(t + step / 2, y + k2 / 2)
    k4 = step * function(t + step, y + k3)
    y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values           # Повертаємо час та значення змінних


# Використання вбудованого бібліотечного методу
def solve_with_library(steps_num, t_start, t_end, initial_conditions):
    # Метод 'RK45' — Рунге-Кута 4-го/5-го порядку
    solution = solve_ivp(euler_system, [t_start, t_end], initial_conditions,
                         method='RK45', t_eval=np.linspace(t_start, t_end, steps_num))

    return solution.t, solution.y.T


# Метод Річардсона для оцінки точності
def richardson_method(function, t_start, t_end, initial_conditions, step, target_rmse):
    # Розв'язок із кроком h
    t_step, y_step = runge_kuta_4(function, t_start, t_end, initial_conditions, step)

    # Розв'язок із кроком h/2
    t_step_05, y_step_05 = runge_kuta_4(function, t_start, t_end, initial_conditions, step * 0.5)

    # Інтерполяція значень із h/2 до кроку h
    y_step_inter = y_step_05[::2]       # Вибір точок, які відповідають кроку h

    # RMSE
    rmse = np.sqrt(np.mean((y_step_inter - y_step) ** 2))

    # Друк результату перевірки точності
    if rmse >= target_rmse:
        print(f"\nRMSE = {rmse:.4f} більший за цільове значення, необхідно зменшити крок")
    else:
        print(f"\nУспішно досягнуто точність RMSE = {rmse:.4f}, яка відповідає вимогам задачі")



# Розрахунок методами, отримуємо повернуті значення
t_euler_method, y_euler_method = euler_method(euler_system, t_start, t_end, initial_conditions, step)
t_rk4_method, y_rk4_method = runge_kuta_4(euler_system, t_start, t_end, initial_conditions, step)
t_lib_auto, y_lib_auto = solve_with_library(steps_num, t_start, t_end, initial_conditions)


# region Виведення числових результатів, відділив для зручності
print("\nМетод Ейлера:")
for t, y in zip(t_euler_method, y_euler_method):
    print(f"t = {t:.2f}, x(t) = {y[0]:.4f}, y(t) = {y[1]:.4f}")

print("\nМетод Рунге-Кута 4-го порядку:")
for t, y in zip(t_rk4_method, y_rk4_method):
    print(f"t = {t:.2f}, x(t) = {y[0]:.4f}, y(t) = {y[1]:.4f}")

print("\nБібліотечний метод:")
for t, y in zip(t_lib_auto, y_lib_auto):
    print(f"t = {t:.2f}, x(t) = {y[0]:.4f}, y(t) = {y[1]:.4f}")
# endregion

# Оцінка методом Річардсона
richardson_method(euler_system, t_start, t_end, initial_conditions, step, accuracy)

# region Побудова графіків бібліотекою matplotlib.pyplot (Частина даних для візуалізації графіка)
# Розмір вікна
plt.figure(figsize=(10, 6))

# Легенди тексти
plt.plot(t_euler_method, y_euler_method[:, 0], "d-", color="blue", label="Ейлер: x(t)", alpha=0.7, linewidth=1.5)
plt.plot(t_euler_method, y_euler_method[:, 1], "d--", color="blue", label="Ейлер: y(t)", alpha=0.7, linewidth=1.5)
plt.plot(t_rk4_method, y_rk4_method[:, 0], "p-", color="green", label="Рунге-Кутта: x(t)", alpha=0.7, linewidth=1.5)
plt.plot(t_rk4_method, y_rk4_method[:, 1], "p--", color="green", label="Рунге-Кутта: y(t)", alpha=0.7, linewidth=1.5)
plt.plot(t_lib_auto, y_lib_auto[:, 0], "x-", color="purple", label="SciPy: x(t)", linewidth=2)
plt.plot(t_lib_auto, y_lib_auto[:, 1], "x--", color="purple", label="SciPy: y(t)", linewidth=2)

# Гарні заголовки
plt.title(
    f"Порівняння методів розв’язків ДР\nКрок: {step}; Точки: {steps_num}",
    fontsize=16,
    fontweight="bold",
    color="darkred"
)
plt.xlabel("Час t", fontsize=12, fontweight="medium", color="darkblue")
plt.ylabel("Розв’язки x(t) та y(t)", fontsize=12, fontweight="medium", color="darkblue")

# Стиль сітки
plt.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

# Стиль легенди
plt.legend(loc="upper left", fontsize=10, edgecolor="black", shadow=True)

# Відображення графіка
plt.show()
# endregion