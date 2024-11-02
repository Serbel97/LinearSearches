import inspect

import matplotlib.pyplot as plt
import numpy as np

# Starting intervals [a, b]
# a = 0
# b = np.pi
a = -3
b = 1


def f(x):

    # return x * 2 - np.cos(x)  # 2x − cos(x)
    # return x ** 2 - np.sin(x)  # x^2 − sin(x)
    # return np.sin(x)  # sin(x)
    return np.exp(x) - np.sin(2 * x)  # f(x) = e^x - sin(2x)


# Golden section constants
phi = (1 + np.sqrt(5)) / 2
invphi = 1 / phi
print('Golden section: ' + str(phi))
print('Inverse golden section: ' + str(invphi))

# Required precision
epsilon = 1e-4

# Create an array for storing iteration points
a_vals, b_vals, c_vals, d_vals = [], [], [], []


# Golden cut algorithm
def golden_section_search(a, b, find_max=True):
    # First approximation c and d
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    while (b - a) > epsilon:
        a_vals.append(a)
        b_vals.append(b)
        c_vals.append(c)
        d_vals.append(d)

        if f(c) < f(d) if find_max else f(c) > f(d):
            a = c
            c = d
            d = a + invphi * (b - a)
        else:
            b = d
            d = c
            c = b - invphi * (b - a)

    return (a + b) / 2


# Find the approximate minimum
x_min = golden_section_search(a, b, find_max=False)

f_equation = inspect.getsource(f).strip().split("#")[-1].strip()

# Visualization of the search process
x = np.linspace(-3, 1, 1000)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f_equation, color='blue')
plt.axvline(x_min, color='red', linestyle='--', label=f'Approximate minimum at x={x_min:.7f}')

plt.scatter(a_vals, f(np.array(a_vals)), color='blue', marker='o', label='a values')
plt.scatter(b_vals, f(np.array(b_vals)), color='purple', marker='h', label='b values')

plt.title(f'Golden Section Search - Finding Minimum of {f_equation} on [{a}, {b}]')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()


# Function to print table
def print_table():
    print(f"{'Iteration':<10} | {'a':^10} | {'b':^15} | {'c':^15} | {'d':^15}")
    print("-" * 70)
    for i, (a, b, c, d) in enumerate(zip(a_vals, b_vals, c_vals, d_vals)):
        print(f"{i:<10} | {a:<10.7f} | {b:<15.7f} | {c:<15.7f} | {d:<15.7f}")


print_table()
