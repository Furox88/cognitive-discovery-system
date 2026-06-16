"""Numerical ODE solvers.

References:
    - Euler, L. (1768). Institutionum calculi integralis.
    - Runge, C. (1895). Über die numerische Auflösung von Differentialgleichungen.
    - Kutta, M.W. (1901). Beitrag zur näherungsweisen Integration totaler DGL.
    - Butcher, J.C. Numerical Methods for Ordinary Differential Equations (3rd ed.)
"""

from __future__ import annotations

import sys
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


def rk45(
    f: Callable[[float, float], float],
    t0: float,
    y0: float,
    t_end: float,
    dt: float = 0.01,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> ODESolution:
    """Dormand-Prince (RK45) adaptive step-size method.

    Computes 4th and 5th order estimates to approximate local error
    and adjust the step size automatically. [Dormand & Prince 1980]

    Args:
        f: right-hand side f(t, y)
        t0: initial time
        y0: initial value
        t_end: end time
        dt: initial time step
        atol: absolute tolerance
        rtol: relative tolerance
    """
    # Dormand-Prince Butcher Tableau coefficients
    a = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
    b = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    c5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
    c4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

    t, y = t0, y0
    t_vals = [t]
    y_vals = [y]
    h = dt
    steps = 0

    # Absolute step-size floor, scaled to the integration span, below which no
    # further progress can be made (a "machine precision floor"). Prevents the
    # adaptive loop from spinning forever on stiff/diverging problems.
    span = abs(t_end - t0) if t_end != t0 else 1.0
    eps_floor = 16 * sys.float_info.epsilon * max(abs(t), span)

    while t < t_end - 1e-12:
        if t + h > t_end:
            h = t_end - t

        k = [0.0] * 7
        k[0] = f(t, y)
        for i in range(1, 7):
            y_next = y + h * sum(b[i][j] * k[j] for j in range(i))
            k[i] = f(t + a[i] * h, y_next)

        # Estimate 5th and 4th order solutions
        y5 = y + h * sum(c5[i] * k[i] for i in range(7))
        y4 = y + h * sum(c4[i] * k[i] for i in range(7))

        # Local error estimate
        error = abs(y5 - y4)
        tolerance = atol + rtol * abs(y)

        if error <= tolerance:
            # Step accepted
            t += h
            y = y5
            t_vals.append(t)
            y_vals.append(y)
            steps += 1

        # Adjust step size
        if error > 0:
            h_opt = h * (tolerance / error) ** 0.2
            h = min(max(0.1 * h, 0.9 * h_opt), 10 * h)
        else:
            h *= 10.0  # Error is zero, aggressively increase step up to max scale

        # Precision floor to prevent infinite loop: either the step size has
        # shrunk below the span-scaled epsilon floor, or it has become so small
        # that adding it to t makes no progress (t + h == t).
        if h < eps_floor or t + h == t:
            raise RuntimeError("Step size h reached machine precision floor.")

    return ODESolution(t=t_vals, y=y_vals, method="rk45", steps=steps)


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

        y = [y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        t = t + h
        t_vals.append(t)
        y_vals.append(list(y))

    return t_vals, y_vals
