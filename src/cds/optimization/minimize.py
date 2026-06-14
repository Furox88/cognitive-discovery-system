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
    state: dict[str, Any] | None = None


def _compute_gradient(f: Callable, x: float | list[float], h_base: float = 1e-7) -> float | list[float]:
    """Compute numerical gradient with adaptive step size for precision."""
    if isinstance(x, (int, float)):
        # Adaptive h based on x scale
        h = h_base * max(1.0, abs(x))
        return (f(x + h) - f(x - h)) / (2 * h)
    
    grad = [0.0] * len(x)
    for i in range(len(x)):
        h = h_base * max(1.0, abs(x[i]))
        x_plus = x[:]
        x_plus[i] += h
        x_minus = x[:]
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def _update_x(x: float | list[float], grad: float | list[float], step: float) -> float | list[float]:
    """Apply gradient update step for scalar or vector inputs."""
    if isinstance(x, (int, float)):
        return x - step * grad
    return [xi - step * gi for xi, gi in zip(x, grad)]

def _magnitude(vec: float | list[float]) -> float:
    """Compute magnitude of a scalar or vector."""
    if isinstance(vec, (int, float)):
        return abs(vec)
    return math.sqrt(sum(vi * vi for vi in vec))


def gradient_descent(
    f: Callable,
    x0: float | list[float],
    lr: float = 0.01,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-7,
) -> OptResult:
    """Minimize a scalar or vector function using gradient descent.

    Args:
        f: objective function
        x0: starting point (scalar or list of floats)
        lr: learning rate
        tol: convergence tolerance on gradient magnitude
        max_iter: iteration limit
        h: step size for numerical gradient
    """
    x = x0 if isinstance(x0, (int, float)) else x0[:]
    for i in range(max_iter):
        grad = _compute_gradient(f, x, h)
        if _magnitude(grad) < tol:
            return OptResult(x=x, value=f(x), iterations=i, converged=True)
        x = _update_x(x, grad, lr)
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
    f: Callable,
    x0: float | list[float],
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-8,
    max_iter: int = 10000,
    h: float = 1e-7,
    state: dict[str, Any] | None = None,
) -> OptResult:
    """Minimize using Adam optimizer (adaptive learning rate) for scalars or vectors.

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
        state: optional dictionary to resume optimization (contains m, v, t)
    """
    x = x0 if isinstance(x0, (int, float)) else x0[:]
    
    if state is None:
        if isinstance(x, (int, float)):
            m = 0.0
            v = 0.0
        else:
            m = [0.0] * len(x)
            v = [0.0] * len(x)
        t_start = 1
    else:
        m = state["m"]
        v = state["v"]
        t_start = state["t"]

    if isinstance(x, (int, float)):
        for i in range(t_start, t_start + max_iter):
            grad = _compute_gradient(f, x, h)
            if abs(grad) < tol:
                return OptResult(
                    x=x, value=f(x), iterations=i - t_start, converged=True,
                    state={"m": m, "v": v, "t": i}
                )
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** i)
            v_hat = v / (1 - beta2 ** i)
            x -= lr * m_hat / (math.sqrt(v_hat) + eps)
    else:
        for i in range(t_start, t_start + max_iter):
            grad = _compute_gradient(f, x, h)
            if _magnitude(grad) < tol:
                return OptResult(
                    x=x, value=f(x), iterations=i - t_start, converged=True,
                    state={"m": m, "v": v, "t": i}
                )
            
            for j in range(len(x)):
                m[j] = beta1 * m[j] + (1 - beta1) * grad[j]
                v[j] = beta2 * v[j] + (1 - beta2) * grad[j] ** 2
                m_hat = m[j] / (1 - beta1 ** i)
                v_hat = v[j] / (1 - beta2 ** i)
                x[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    return OptResult(
        x=x, value=f(x), iterations=max_iter, converged=False,
        state={"m": m, "v": v, "t": t_start + max_iter}
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
