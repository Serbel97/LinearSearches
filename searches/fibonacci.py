import inspect

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    # return x ** 2 - 4*x + 4  # x^2 − 4x + 4
    # return -x ** 2  # f(x) = -x^2
    # return x ** 2 - np.sin(x)  # x^2 − sin(x)
    return np.exp(x) - np.sin(2 * x)  # f(x) = e^x - sin(2x)


def fibonacci_search(f, a, b, n, find_max=True):
    """
    Perform Fibonacci search to find the maximum or minimum of function f in the interval [a, b].

    Parameters:
    f (function): The function to minimize.
    a (float): The start of the interval.
    b (float): The end of the interval.
    n (int): The number of iterations.

    Returns:
    float: The approximate location of the maximum or minimum.
    """

    def fibonacci_numbers(n):
        """Generate Fibonacci numbers up to F_n."""
        fib = [0, 1]
        for i in range(2, n + 1):
            fib.append(fib[i - 1] + fib[i - 2])
        return fib

    fib = fibonacci_numbers(n)

    # Initialize x1 and x2
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)

    # Evaluate function values at x1 and x2
    f_x1 = f(x1)
    f_x2 = f(x2)

    for i in range(n - 1):
        if f_x1 < f_x2 if find_max else f_x1 > f_x2:
            a = x1
            x1 = x2
            x2 = a + (fib[n - i - 2] / fib[n - i - 1]) * (b - a)
            f_x1 = f_x2
            f_x2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + (fib[n - i - 3] / fib[n - i - 1]) * (b - a)
            f_x2 = f_x1
            f_x1 = f(x1)

    # After n iterations, the midpoint of the interval is considered the optimal point (i.e., minimum)
    return (a + b) / 2


a = -3
b = 0.5
n = 10  # Number of iterations

minimum = fibonacci_search(f, a, b, n, find_max=True)
print(f"Approximate minimum: x = {minimum:.7f}")

x = np.linspace(-3, 1, 1000)
y = f(x)

# Extract the equation
f_equation = inspect.getsource(f).strip().split("#")[-1].strip()

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=rf'f(x) = {f_equation}', color='blue')

plt.scatter(minimum, f(minimum), color='green', label=f'Newton Method Minimum: x = {minimum:.7f}')

plt.title(f'Local Maxima or Minima of f(x) = {f_equation}')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
