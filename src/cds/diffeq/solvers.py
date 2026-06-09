"""Numerical ODE solvers.

References:
    - Euler, L. (1768). Institutionum calculi integralis.
    - Runge, C. (1895). Über die numerische Auflösung von Differentialgleichungen.
    - Kutta, M.W. (1901). Beitrag zur näherungsweisen Integration totaler DGL.
    - Butcher, J.C. Numerical Methods for Ordinary Differential Equations (3rd ed.)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ODESolution:
    """Result of an ODE integration."""

    t: list[float]
    y: list[float]
    method: str
    steps: int


def euler_method(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = 0.01,
) -> ODESolution:
    """Euler's method for dy/dt = f(t, y).

    First-order explicit method. Local truncation error O(dt²),
    global error O(dt). [Euler 1768]

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time
        dt: time step
    """
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        y = y + h * f(t, y)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="euler", steps=steps)


def rk4(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = 0.01,
) -> ODESolution:
    """Classical 4th-order Runge-Kutta method.

    Local truncation error O(dt⁵), global error O(dt⁴). [Runge 1895, Kutta 1901]

    The standard Butcher tableau:
        0   |
        1/2 | 1/2
        1/2 | 0   1/2
        1   | 0   0   1
        ----|----------------
            | 1/6 1/3 1/3 1/6

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time
        dt: time step
    """
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="rk4", steps=steps)


def midpoint_method(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = 0.01,
) -> ODESolution:
    """Explicit midpoint method (2nd-order Runge-Kutta).

    Local truncation error O(dt³), global error O(dt²).

    Args:
        f: right-hand side function f(t, y)
        t0: initial time
        y0: initial value y(t0)
        t_end: end time
        dt: time step
    """
    t_vals = [t0]
    y_vals = [y0]
    t, y = t0, y0
    steps = 0

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        y = y + h * k2
        t = t + h
        t_vals.append(t)
        y_vals.append(y)
        steps += 1

    return ODESolution(t=t_vals, y=y_vals, method="midpoint", steps=steps)


def solve_system(
    f: Callable[[float, list[float]], list[float]],
    t0: float,
    y0: list[float],
    t_end: float,
    dt: float = 0.01,
) -> tuple[list[float], list[list[float]]]:
    """RK4 for systems of ODEs: dy/dt = f(t, y) where y is a vector.

    Args:
        f: right-hand side f(t, y) returning a list of derivatives
        t0: initial time
        y0: initial state vector
        t_end: end time
        dt: time step

    Returns:
        (t_values, y_values) where y_values[i] is the state vector at t_values[i]
    """
    n = len(y0)
    t_vals = [t0]
    y_vals = [list(y0)]
    t = t0
    y = list(y0)

    while t < t_end - 1e-12:
        h = min(dt, t_end - t)

        k1 = f(t, y)
        y_tmp = [y[i] + h * k1[i] / 2 for i in range(n)]
        k2 = f(t + h / 2, y_tmp)
        y_tmp = [y[i] + h * k2[i] / 2 for i in range(n)]
        k3 = f(t + h / 2, y_tmp)
        y_tmp = [y[i] + h * k3[i] for i in range(n)]
        k4 = f(t + h, y_tmp)

        y = [
            y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(n)
        ]
        t = t + h
        t_vals.append(t)
        y_vals.append(list(y))

    return t_vals, y_vals
