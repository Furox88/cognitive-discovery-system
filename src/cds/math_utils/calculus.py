"""Numerical calculus — derivatives, integrals, gradients."""
from __future__ import annotations

from collections.abc import Callable


def derivative(f: Callable[[float], float], x: float, h_base: float = 1e-7) -> float:
    """Central difference approximation with adaptive step size."""
    h = h_base * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2 * h)


def integral(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """ Simpson's rule for numerical integration. """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        coeff = 4 if i % 2 != 0 else 2
        s += coeff * f(a + i * h)
    return s * h / 3


def gradient(f: Callable[..., float], point: list[float], h_base: float = 1e-7) -> list[float]:
    """Numerical gradient for multivariable functions with adaptive scaling."""
    grad = []
    for i in range(len(point)):
        # Scale step size h relative to point magnitude to maintain precision
        h = h_base * max(1.0, abs(point[i]))
        def partial(val: float, idx: int = i) -> float:
            p = point.copy()
            p[idx] = val
            return f(*p)
        grad.append((partial(point[i] + h) - partial(point[i] - h)) / (2 * h))
    return grad
