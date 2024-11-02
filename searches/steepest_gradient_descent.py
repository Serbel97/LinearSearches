import inspect

import numpy as np
from matplotlib import pyplot as plt


# Example usage
def f(x):
    # return x**2 - 4*x + 4  # x^2 - 4x + 4
    return np.exp(x) - np.sin(2 * x)  # f(x) = e^x - sin(2x)

def df(x):
    # return 2*x - 4
    return np.exp(x) - 2 * np.cos(2 * x)  # f(x) = e^x - sin(2x)

def steepest_gradient_descent(f, df, x0, alpha, tol, max_iter, find_max=True):
    """
    Perform the steepest gradient descent to find the minimum of function f.

    Parameters:
    f (function): The function to minimize.
    grad_f (function): The gradient of the function f.
    x0 (array-like): The initial point for the descent.
    alpha (float): The step size (learning rate).
    tol (float): The tolerance for the stopping criterion.
    max_iter (int): The maximum number of iterations.

    Returns:
    tuple: The approximate location of the minimum, the function value at the minimum, and the number of iterations performed.
    """

    # Initialize the starting point
    x_k = x0
    k = 0

    # Repeat until convergence or maximum iterations are reached
    while k < max_iter:
        # Compute the gradient at the current point
        grad_f_k = df(x_k)

        # Update the current point by moving in the direction of the negative gradient
        alpha_grad = alpha * grad_f_k
        x_k_next = x_k + alpha_grad if find_max else x_k - alpha_grad

        # Check the stopping criterion (if the gradient's magnitude is less than the tolerance)
        if np.linalg.norm(grad_f_k) < tol:
            break

        # Update the current point to the next point and increment the iteration counter
        x_k = x_k_next
        k += 1

    # After exiting the loop, return the found minimum point and the function value
    x_min = x_k
    f_min = f(x_min)

    return x_min, f_min, k


x0 = -2  # np.array([0.0])  # Initial point
alpha = 0.1  # Step size (learning rate)
tol = 1e-6  # Tolerance
max_iter = 1000  # Maximum number of iterations

x_min, f_min, iterations = steepest_gradient_descent(f, df, x0, alpha, tol, max_iter, find_max=False)
print(f"Minimum point: {x_min}, Function value at minimum: {f_min}, Iterations: {iterations}")

# Extract the function's source code
f_equation = inspect.getsource(f).strip().split("#")[-1].strip()

x = np.linspace(-3, 1, 400)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=rf'f(x) = {f_equation}', color='blue')

plt.scatter(x_min, f(x_min), color='green', label=f'Newton Method Minimum: x = {x_min:.7f}')

plt.title(f'Local Minima and Maxima of f(x) = {f_equation}')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
