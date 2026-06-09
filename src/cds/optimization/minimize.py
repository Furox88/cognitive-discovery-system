"""Minimization algorithms for single and multi-variable functions."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class OptResult:
    """Result of an optimization run."""

    x: float | list[float]
    value: float
    iterations: int
    converged: bool


def gradient_descent(
    f: Callable[[float], float],
    x0: float,
    lr: float = 0.01,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-7,
) -> OptResult:
    """Minimize a scalar function using gradient descent.

    Args:
        f: objective function
        x0: starting point
        lr: learning rate
        tol: convergence tolerance on gradient magnitude
        max_iter: iteration limit
        h: step size for numerical gradient
    """
    x = x0
    for i in range(max_iter):
        grad = (f(x + h) - f(x - h)) / (2 * h)
        if abs(grad) < tol:
            return OptResult(x=x, value=f(x), iterations=i, converged=True)
        x -= lr * grad
    return OptResult(x=x, value=f(x), iterations=max_iter, converged=False)


def newton_method(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 1000,
    h: float = 1e-5,
) -> OptResult:
    """Find a root of f using Newton-Raphson method.

    Args:
        f: function whose root to find
        x0: starting point
        tol: convergence tolerance
        max_iter: iteration limit
        h: step for numerical derivative
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return OptResult(x=x, value=fx, iterations=i, converged=True)
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        if abs(dfx) < 1e-15:
            break
        x -= fx / dfx
    return OptResult(
        x=x, value=f(x), iterations=max_iter, converged=False,
    )


def adam(
    f: Callable[[float], float],
    x0: float,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-7,
) -> OptResult:
    """Minimize using Adam optimizer (adaptive learning rate).

    Args:
        f: objective function
        x0: starting point
        lr: learning rate
        beta1: first moment decay
        beta2: second moment decay
        eps: numerical stability constant
        tol: convergence tolerance
        max_iter: iteration limit
        h: step for numerical gradient
    """
    x = x0
    m = 0.0
    v = 0.0
    for i in range(1, max_iter + 1):
        grad = (f(x + h) - f(x - h)) / (2 * h)
        if abs(grad) < tol:
            return OptResult(
                x=x, value=f(x), iterations=i, converged=True,
            )
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        x -= lr * m_hat / (math.sqrt(v_hat) + eps)
    return OptResult(
        x=x, value=f(x), iterations=max_iter, converged=False,
    )


def line_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> OptResult:
    """Golden section search for minimum in [a, b].

    Args:
        f: unimodal function to minimize
        a: left bound
        b: right bound
        tol: convergence tolerance on interval width
        max_iter: iteration limit
    """
    phi = (math.sqrt(5) - 1) / 2
    for i in range(max_iter):
        if abs(b - a) < tol:
            mid = (a + b) / 2
            return OptResult(
                x=mid, value=f(mid), iterations=i, converged=True,
            )
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
    mid = (a + b) / 2
    return OptResult(
        x=mid, value=f(mid), iterations=max_iter, converged=False,
    )
