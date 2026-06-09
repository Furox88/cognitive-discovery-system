"""Numerical calculus — derivatives, integrals, gradients."""
from __future__ import annotations

from collections.abc import Callable


def derivative(f: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """Central difference approximation."""
    return (f(x + h) - f(x - h)) / (2 * h)


def integral(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Simpson's rule."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        coeff = 4 if i % 2 != 0 else 2
        s += coeff * f(a + i * h)
    return s * h / 3


def gradient(f: Callable[..., float], point: list[float], h: float = 1e-7) -> list[float]:
    """Numerical gradient for multivariable functions."""
    grad = []
    for i in range(len(point)):
        def partial(val: float, idx: int = i) -> float:
            p = point.copy()
            p[idx] = val
            return f(*p)
        grad.append(derivative(partial, point[i], h))
    return grad
