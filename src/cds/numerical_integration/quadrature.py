"""Deterministic numerical quadrature — Newton-Cotes, Romberg, Gauss-Legendre.

This module provides classical deterministic integration rules to complement the
stochastic integration in :mod:`cds.montecarlo` and the ODE solvers in
:mod:`cds.diffeq`. All routines are pure Python with no external dependencies.

References:
    - Newton, I. & Cotes, R. (1722). Harmonia Mensurarum (Newton-Cotes rules).
    - Simpson, T. (1743). Mathematical Dissertations on Physical Subjects.
    - Romberg, W. (1955). Vereinfachte numerische Integration (Richardson extrapolation).
    - Gauss, C.F. (1814). Methodus nova integralium valores per approximationem inveniendi.
    - Davis, P.J. & Rabinowitz, P. Methods of Numerical Integration (2nd ed.).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass

from cds.core._numeric import NEAR_ZERO


@dataclass
class QuadratureResult:
    """Result of an adaptive numerical integration.

    Attributes:
        value: computed approximation of the integral
        method: name of the quadrature rule used
        n_eval: number of integrand evaluations performed
        error_estimate: internal estimate of the truncation error (``nan`` if
            unavailable for the chosen rule)
    """

    value: float
    method: str
    n_eval: int
    error_estimate: float


def trapezoid(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Composite trapezoidal rule.

    Approximates ``∫_a^b f(x) dx`` with ``n`` equal panels. Closed Newton-Cotes
    of order 1; error ``O(h²)`` for twice-differentiable integrands. [Cotes 1722]

    Args:
        f: integrand
        a: lower limit
        b: upper limit (may be less than ``a``)
        n: number of panels (``n >= 1``)

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s


def simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1000,
) -> float:
    """Composite Simpson's 1/3 rule.

    Closed Newton-Cotes of order 2; error ``O(h⁴)``. Requires an even number of
    panels so that every group of two panels spans one parabola. [Simpson 1743]

    Args:
        f: integrand
        a: lower limit
        b: upper limit (may be less than ``a``)
        n: number of panels (must be even and ``>= 2``)

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: if ``n`` is not an even number ``>= 2``.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n must be an even integer >= 2")
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 == 1 else 2.0) * f(a + i * h)
    return (h / 3.0) * s


def simpson_38(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 999,
) -> float:
    """Composite Simpson's 3/8 rule.

    Closed Newton-Cotes of order 3 over groups of three panels; error ``O(h⁴)``.
    Useful as a companion to the 1/3 rule when ``n`` is a multiple of 3.

    Args:
        f: integrand
        a: lower limit
        b: upper limit (may be less than ``a``)
        n: number of panels (must be a multiple of 3 and ``>= 3``)

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: if ``n`` is not a multiple of 3 ``>= 3``.
    """
    if n < 3 or n % 3 != 0:
        raise ValueError("n must be a multiple of 3 and >= 3")
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (3.0 if i % 3 != 0 else 2.0) * f(a + i * h)
    return (3.0 * h / 8.0) * s


def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 20,
) -> QuadratureResult:
    """Romberg integration via Richardson extrapolation on the trapezoidal rule.

    Builds a triangular table where column ``k`` is ``O(h^{2k})`` accurate.
    Halting is driven by the relative/absolute change in the extrapolated
    diagonal. [Romberg 1955]

    Args:
        f: integrand
        a: lower limit
        b: upper limit
        tol: convergence tolerance on successive diagonal estimates
        max_iter: maximum number of extrapolation levels (each adds one row)

    Returns:
        :class:`QuadratureResult` with an internal error estimate.

    Raises:
        ValueError: if ``max_iter < 1``.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    width = b - a
    # Total integrand evaluations across all levels: 1 + 1 + 2 + 4 + ... + 2^(m-1)
    n_eval = 1

    # R[0][0]: single trapezoid
    r: list[list[float]] = [[0.5 * width * (f(a) + f(b))]]
    best = r[0][0]
    error_est = math.inf

    for k in range(1, max_iter):
        # Trapezoid with 2^k panels reusing the 2^(k-1) level
        panels = 1 << (k - 1)
        h = width / (1 << k)
        total = 0.0
        for i in range(1, panels + 1):
            total += f(a + (2 * i - 1) * h)
        n_eval += panels
        t_k = 0.5 * r[k - 1][0] + h * total

        row = [t_k]
        for j in range(1, k + 1):
            # Richardson extrapolation factor 4^j / (4^j - 1)
            factor = 1 << (2 * j)  # 4^j
            row.append((factor * row[j - 1] - r[k - 1][j - 1]) / (factor - 1))
        r.append(row)

        error_est = abs(row[k] - best)
        best = row[k]
        if error_est <= tol * max(1.0, abs(best)):
            break

    return QuadratureResult(
        value=best,
        method="romberg",
        n_eval=n_eval,
        error_estimate=error_est,
    )


@functools.cache
def _gauss_legendre_nodes(n: int) -> tuple[tuple[float, float], ...]:
    """Return ``n`` Gauss-Legendre (node, weight) pairs on ``[-1, 1]``.

    Nodes are the roots of the Legendre polynomial ``P_n``; weights follow
    ``w_i = 2 / ((1 - x_i^2) [P_n'(x_i)]^2)``. Roots are found by Newton's
    method seeded from the classical Chebyshev guesses. [Gauss 1814]
    """
    if n == 1:
        return ((0.0, 2.0),)
    if n < 1:
        raise ValueError("n must be >= 1")

    pairs: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        # Initial guess isolates the i-th root (largest to smallest).
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        dp = 0.0
        for _ in range(100):
            p, dp = _legendre(n, x)
            dx = p / dp
            x -= dx
            if abs(dx) < NEAR_ZERO:
                break
        w = 2.0 / ((1.0 - x * x) * dp * dp)
        pairs.append((x, w))
    pairs.sort(key=lambda t: t[0])
    return tuple(pairs)


def _legendre(n: int, x: float) -> tuple[float, float]:
    """Evaluate ``P_n(x)`` and ``P_n'(x)`` via the three-term recurrence.

    Uses ``(k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)`` and its
    term-by-term derivative.
    """
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0
    p_prev, p_curr = 1.0, x
    dp_prev, dp_curr = 0.0, 1.0
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        dp_next = ((2 * k + 1) * (p_curr + x * dp_curr) - k * dp_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
        dp_prev, dp_curr = dp_curr, dp_next
    return p_curr, dp_curr


def gaussian_quadrature(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 5,
) -> float:
    """Gauss-Legendre quadrature with ``n`` nodes.

    Exact for polynomials of degree up to ``2n - 1``. The ``[-1, 1]`` rule is
    affinely mapped onto ``[a, b]``. [Gauss 1814]

    Args:
        f: integrand
        a: lower limit
        b: upper limit (may be less than ``a``)
        n: number of Gauss-Legendre nodes (``>= 1``)

    Returns:
        Approximation of the integral.

    Raises:
        ValueError: if ``n < 1``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    total = 0.0
    for node, weight in _gauss_legendre_nodes(n):
        total += weight * f(half * node + mid)
    return half * total


def adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> QuadratureResult:
    """Adaptive recursive Simpson quadrature.

    Recursively bisects subintervals where the local error estimate (the
    difference between Simpson over the whole interval and over its halves)
    exceeds ``tol``, concentrating work where the integrand is hard. [Lyness 1969]

    Args:
        f: integrand
        a: lower limit
        b: upper limit
        tol: desired absolute tolerance
        max_depth: maximum recursion depth to bound cost on hard integrands

    Returns:
        :class:`QuadratureResult` carrying the number of integrand evaluations.

    Raises:
        ValueError: if ``max_depth < 1``.
        RuntimeError: if ``max_depth`` is exhausted before convergence.
    """

    def _simpson(fa: float, fm: float, fb: float, a: float, b: float) -> float:
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")

    counter = {"n": 0}

    def _eval(x: float) -> float:
        counter["n"] += 1
        return f(x)

    def _recurse(
        a: float, b: float, fa: float, fb: float, fm: float, whole: float, depth: int, eps: float
    ) -> float:
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = _eval(lm)
        frm = _eval(rm)
        left = _simpson(fa, flm, fm, a, m)
        right = _simpson(fm, frm, fb, m, b)
        diff = left + right - whole
        # A NaN diff means the integrand produced NaN on this subinterval
        # (e.g. a divergent/undefined integrand). Stop recursing immediately so
        # the NaN propagates to the top-level guard instead of branching until
        # max_depth is exhausted (2**max_depth calls -> hang).
        if math.isnan(diff):
            return left + right + diff / 15.0
        # Standard Lyness error estimate (scaled by 1/15).
        if depth <= 0 or abs(diff) <= 15.0 * eps:
            return left + right + diff / 15.0
        return _recurse(a, m, fa, fm, flm, left, depth - 1, 0.5 * eps) + _recurse(
            m, b, fm, fb, frm, right, depth - 1, 0.5 * eps
        )

    fa = _eval(a)
    fb = _eval(b)
    fm = _eval(0.5 * (a + b))
    whole = _simpson(fa, fm, fb, a, b)

    value = _recurse(a, b, fa, fb, fm, whole, max_depth, tol)
    if math.isnan(value):
        raise RuntimeError("adaptive_simpson produced NaN (likely divergent integrand)")

    return QuadratureResult(
        value=value,
        method="adaptive_simpson",
        n_eval=counter["n"],
        error_estimate=math.nan,
    )
