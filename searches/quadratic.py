import numpy as np
from matplotlib import pyplot as plt


def f(x):
    # return x ** 2 - np.sin(x)
    return np.exp(x) - np.sin(2 * x)


def df(x):
    # return 2 * x - np.cos(x)
    return np.exp(x) - 2*np.cos(2 * x)



def quadratic_search(f, x0, h_initial, tolerance=1e-5, max_iterations=100, find_max=True):
    h = h_initial
    x_vals = [x0]
    x0_vals, x1_vals, x2_vals = [], [], []  # Lists to store x0, x1, x2 values for each iteration

    for i in range(max_iterations):
        # Step 1: Select points x1 and x2
        x1 = x0 + h
        x2 = x0 + 2 * h

        # Step 2: Function values at points
        f0 = f(x0)
        f1 = f(x1)
        f2 = f(x2)

        # Append x0, x1, x2 for plotting
        x0_vals.append(x0)
        x1_vals.append(x1)
        x2_vals.append(x2)

        # Adjust the conditions for finding maximum or minimum
        if (f1 < f2) if find_max else (f1 > f2):
            h *= 2  # Increase h if the extremum is further away
        elif (f0 >= f1) if find_max else (f0 <= f1):
            h /= 2  # Decrease h if we've skipped over the extremum
        else:
            # Quadratic interpolation step
            numerator = ((x1 - x0) ** 2 * (f1 - f2)) - ((x1 - x2) ** 2 * (f1 - f0))
            denominator = 2 * ((x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0))

            if denominator == 0:  # Avoid division by zero
                break

            # Calculate new estimate for extreme
            x_extreme = x1 - numerator / denominator

            # Check convergence
            if abs(x_extreme - x1) < tolerance:
                return x_extreme, x_vals, x0_vals, x1_vals, x2_vals

            # Update x0 and h for next iteration
            x0 = x_extreme
            x_vals.append(x_extreme)
            h = h_initial

    return x0, x_vals, x0_vals, x1_vals, x2_vals


# Initial values
x0 = -3  # Start of interval
h_initial = 0.1  # Initial step size

# Find the maximum of f_max and minimum of f
max_x, x_vals_max, x0_vals_max, x1_vals_max, x2_vals_max = quadratic_search(
    f, x0, h_initial, tolerance=1e-5, find_max=True
)
min_x, x_vals_min, x0_vals_min, x1_vals_min, x2_vals_min = quadratic_search(
    f, x0, h_initial, tolerance=1e-5, find_max=False
)

# Plotting for the maximum search
x = np.linspace(-3, 1, 500)
y_max = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y_max, label=r'$f(x) = e^x - sin(2x)$', color='black')
plt.scatter(x2_vals_max, f(np.array(x2_vals_max)), color='green', marker='s', s=20, label='x2 values')
plt.scatter(x1_vals_max, f(np.array(x1_vals_max)), color='purple', marker='x', s=20, label='x1 values')
plt.scatter(x0_vals_max, f(np.array(x0_vals_max)), color='blue', marker='o', s=20, label='x0 values')
plt.plot(max_x, f(max_x), 'ro', label=f'Approximate maximum at x={max_x:.7f}')
plt.title('Quadratic Interpolation Search - Finding Maximum of $f(x) = e^x - sin(2x)$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print("Approximate maximum x:", max_x)
print("Function value at maximum f(x):", f(max_x))

# Plotting for the minimum search
y_min = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y_min, label=r'$f(x) = x^2 - sin(x)$', color='black')
plt.scatter(x2_vals_min, f(np.array(x2_vals_min)), color='green', marker='s', s=20, label='x2 values')
plt.scatter(x1_vals_min, f(np.array(x1_vals_min)), color='purple', marker='x', s=20, label='x1 values')
plt.scatter(x0_vals_min, f(np.array(x0_vals_min)), color='blue', marker='o', s=20, label='x0 values')
plt.plot(min_x, f(min_x), 'ro', label=f'Approximate minimum at x={min_x:.7f}')
plt.title('Quadratic Interpolation Search - Finding Minimum of $f(x) = x^2 - sin(x)$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print("Approximate minimum x:", min_x)
print("Function value at minimum f(x):", f(min_x))


# Function to print the table
def print_quadratic_search_table(x0_vals, x1_vals, x2_vals, x_vals):
    print(f"{'k':<10} | {'x0':^15} | {'x1':^15} | {'x2':^15} | {'Estimate':^15}")
    print("-" * 75)

    # Data rows
    for k, (x0, x1, x2, x_est) in enumerate(zip(x0_vals, x1_vals, x2_vals, x_vals), start=1):
        print(f"{k:<10} | {x0:<15.7f} | {x1:<15.7f} | {x2:<15.7f} | {x_est:<15.7f}")


# Print the tables for maximum and minimum searches
print("Quadratic Search Table - Maximum")
print_quadratic_search_table(x0_vals_max, x1_vals_max, x2_vals_max, x_vals_max)

print("\nQuadratic Search Table - Minimum")
print_quadratic_search_table(x0_vals_min, x1_vals_min, x2_vals_min, x_vals_min)
