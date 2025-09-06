import matplotlib.pyplot as plt
import math as mat

def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return df_dx, df_dy

def gradient_ascent(x0, y0, learning_rate=0.01, iterations=100):
    x, y = x0, y0
    best_values = []

    x_range = [1, 2]
    y_range = [1, 2]

    for i in range(iterations):
        df_dx, df_dy = grad_f(x, y)
        x += learning_rate * df_dx
        y += learning_rate * df_dy
        x = max(min(x, x_range[1]), x_range[0])
        y = max(min(y, y_range[1]), y_range[0])
        best_values.append(f(x, y))
        print("Iteration: ",i," value: ",f(x, y))

    return x, y, best_values


x0, y0 = 0.1, -0.1
learning_rate = 0.1
iterations = 500

x_max, y_max, best_values = gradient_ascent(x0, y0, learning_rate, iterations)

print("Максимум в точке:", (x_max, y_max))
print("Значение функции:", f(x_max, y_max))

plt.plot(best_values)
plt.xlabel("Итерация")
plt.ylabel("Значение функции")
plt.title("Прогресс градиентного подъёма")
plt.savefig("progress.png")
