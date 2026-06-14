from cds.optimization import gradient_descent, newton_method

# Find minimum of (x-3)²
result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(f"x = {result.x:.6f}")  # ~3.0

# Find √2 using Newton's method
result = newton_method(lambda x: x ** 2 - 2, x0=1.5)
print(f"√2 = {result.x:.10f}")  # 1.4142135624
