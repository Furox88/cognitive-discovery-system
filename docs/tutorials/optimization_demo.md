# Optimization Tutorial

`cds.optimization` provides four pure-Python optimizers that work on both scalar
and vector objectives. Each returns an `OptResult` dataclass reporting the
minimizer, the objective value, the iteration count, a convergence flag, and
(for Adam) a resumable state checkpoint.

## 1. Gradient Descent

First-order minimization. Pass a scalar or vector starting point — the return
type tracks the input (`OptResult[float]` for a scalar `x0`,
`OptResult[list[float]]` for a list).

```python
from cds.optimization import gradient_descent

# Scalar: minimize (x-3)^2 starting from x=10
res = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(f"x={res.x:.6f}, f(x)={res.value:.2e}, iters={res.iterations}, converged={res.converged}")
# x=3.000000, f(x)=1.89e-17, iters=95, converged=True

# Vector: minimize a bowl x0^2 + x1^2
res = gradient_descent(lambda v: v[0] ** 2 + v[1] ** 2, x0=[5.0, -3.0], lr=0.1)
print(res.x)  # ≈ [0.0, 0.0]
```

## 2. Newton's Method (root finding)

Second-order root finding for scalar `f`. Returns the `x` where `f(x) ≈ 0`.

```python
from cds.optimization import newton_method

res = newton_method(lambda x: x**3 - 2, x0=2.0)
print(f"cbrt(2) ≈ {res.x:.10f}  (true = {2 ** (1 / 3):.10f})")
# cbrt(2) ≈ 1.2599210499  (true = 1.2599210499)
```

## 3. Adam

Adaptive learning-rate optimizer. Stateful — pass the returned `state` back in
to resume a run, and supply an analytic gradient via `grad_f` when you have one.

```python
import math
from cds.optimization import adam

res = adam(lambda x: math.sin(x) + 0.1 * x**2, x0=2.0, lr=0.05)
print(f"x={res.x:.6f}, f(x)={res.value:.6f}")
# x=3.837467, f(x)=0.831559
```

## 4. Line Search (golden section)

Bracketed scalar minimization. Give it an interval `[a, b]` assumed to be
unimodal.

```python
from cds.optimization import line_search

res = line_search(lambda x: (x - 2.5) ** 4, a=0, b=5)
print(f"min at x={res.x:.8f}")  # min at x=2.50000000
```

---

Run the full demo with `python examples/optimization_demo.py`.
