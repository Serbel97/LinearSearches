import inspect

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    # return x ** 2 - 4*x + 4  # x^2 − 4x + 4
    # return -x ** 2  # f(x) = -x^2
    # return x ** 2 - np.sin(x)  # x^2 − sin(x)
    return np.exp(x) - np.sin(2 * x)  # f(x) = e^x - sin(2x)


def df(x):
    # return 2 * x - 4  # f'(x) = 2x − 4
    # return -2 * x  # f(x) = -2*x
    # return 2 * x - np.cos(x)  # 2^x − cos(x)
    return np.exp(x) - 2 * np.cos(2 * x)  # f(x) = e^x - sin(2x)


def ddf(x):
    # return 2  # f''(x) = 2
    # return -2  # f(x) = -2
    # return 2 + np.sin(x)  # f''(x) = x^2 − sin(x)
    return np.exp(x) + 4 * np.sin(2 * x)  # f(x) = e^x - sin(2x)


def newton_method(x0, tolerance=1e-5, max_iterations=100, find_max=True):
    x = x0
    x_vals = [x]
    for _ in range(max_iterations):

        if find_max:
            derivation = f(x) / df(x)
        else:
            derivation = df(x) / ddf(x)
        x_new = x - derivation

        x_vals.append(x_new)

        if abs(x_new - x) < tolerance:
            break

        x = x_new

    return x, x_vals


newton_x0 = -2  # Initial guess for Newton's method
newton_result, x_vals = newton_method(newton_x0, find_max=False)

x = np.linspace(-3, 3, 400)
y = f(x)

# Extract the function's source code
f_equation = inspect.getsource(f).strip().split("#")[-1].strip()

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=rf'f(x) = {f_equation}', color='blue')

plt.scatter(x_vals, f(np.array(x_vals)), color='purple', marker='h', label='x values')
plt.scatter(newton_result, f(newton_result), color='green', label=f'Newton Method Minimum: x = {newton_result:.7f}')

plt.title(f'Local Minima and Maxima of f(x) = {f_equation}')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()


def print_x_vals_table():
    print(f"{'Iteration':<10}{'x Value':<20}")
    print("-" * 30)
    for i, x_val in enumerate(x_vals, start=1):
        print(f"{i:<10}{x_val:<20.10f}")


print_x_vals_table()
