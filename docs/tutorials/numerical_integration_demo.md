# 📈 Numerical Integration Tutorial

`cds.numerical_integration` provides deterministic quadrature rules. A canonical test: `∫_0^1 e^x dx = e - 1`.

## 1. Newton–Cotes Rules

```python
import math
from cds.numerical_integration import trapezoid, simpson, simpson_38

f = math.exp
print(trapezoid(f, 0, 1, 1000))  # O(h^2)
print(simpson(f, 0, 1, 100))      # O(h^4)
print(simpson_38(f, 0, 1, 99))    # 3/8 variant
```

## 2. Higher-Order & Adaptive

```python
from cds.numerical_integration import gaussian_quadrature, romberg, adaptive_simpson

print(gaussian_quadrature(f, 0, 1, 8).value)  # ~1e-16 error
print(romberg(f, 0, 1).value)                   # Richardson extrapolation
print(adaptive_simpson(f, 0, 1).value)          # tolerance-driven bisection
```

**Why it matters:** you can watch error drop from `1e-7` (trapezoid) to `1e-16` (Gauss-Legendre) on the same problem — a clear lesson in quadrature convergence.

Run the full demo with `python examples/numerical_integration_demo.py`.
