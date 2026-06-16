"""Optimization algorithms demo."""

import math

from cds.optimization import adam, gradient_descent, line_search, newton_method

# --- Gradient descent ---
print("=== Gradient Descent ===")
result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(f"Minimize (x-3)²: x={result.x:.6f}, f(x)={result.value:.2e}")
print(f"Converged in {result.iterations} iterations")

# --- Newton's method (root finding) ---
print("\n=== Newton's Method ===")
result = newton_method(lambda x: x**3 - 2, x0=2.0)
print(f"Root of x³-2=0: x={result.x:.10f}")
print(f"Actual ³√2 = {2 ** (1 / 3):.10f}")

# --- Adam optimizer ---
print("\n=== Adam Optimizer ===")
result = adam(lambda x: math.sin(x) + 0.1 * x**2, x0=2.0, lr=0.05)
print(f"Min of sin(x)+0.1x²: x={result.x:.6f}, f(x)={result.value:.6f}")

# --- Golden section search ---
print("\n=== Line Search (Golden Section) ===")
result = line_search(lambda x: (x - 2.5) ** 4, a=0, b=5)
print(f"Min of (x-2.5)⁴ in [0,5]: x={result.x:.8f}")
