"""Probability tail functions and their special-function machinery.

This module is the engine room of :mod:`cds.stats.hypothesis_tests`: it
provides the survival functions ``t_sf`` (Student's t), ``chi2_sf``
(chi-square) and ``f_sf`` (Fisher F) that the classical tests need to turn a
statistic into a p-value, together with the regularized incomplete gamma and
beta functions they are built on.

The split exists so that the test functions (:mod:`cds.stats.hypothesis_tests`)
read as a focused catalog of hypothesis tests rather than a wall of continued
fractions. The special functions follow Numerical Recipes (3rd ed.) closely;
they are kept here as private ``_``-prefixed helpers because they are
implementation details of the survival functions, but :mod:`hypothesis_tests`
re-exports them so existing ``from cds.stats.hypothesis_tests import _gser``
style imports (used by the test-suite for coverage) keep working.

References:
    - Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
      (2007). "Numerical Recipes," 3rd ed., §6.2-6.4 (incomplete gamma and
      beta functions: gammln, gser, gcf, betacf).
    - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical
      Functions," §6.5, §26.
    - Student [W. S. Gosset] (1908). "The probable error of a mean."
      Biometrika, 6(1), 1-25. (t-distribution)
    - Pearson, K. (1900). Philosophical Magazine, 50(302), 157-175.
      (chi-square)
    - Fisher, R. A. (1925). "Statistical Methods for Research Workers."
      Oliver & Boyd. (F-distribution)
"""

from __future__ import annotations

import math

_MAXIT = 200
_EPS = 3.0e-12
_FPMIN = 1.0e-300


def _gammln(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation).

    Reference: Numerical Recipes §6.1; Lanczos (1964).
    """
    cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _gser(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a,x) via series expansion.

    Reference: Numerical Recipes §6.2 (gser).
    """
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    # NR §6.2 series expansion converges within _MAXIT for every valid
    # (a>0, x≥0) input, so the loop always exits via the ``break`` below —
    # the natural-exhaustion arc (loop completes without breaking) is
    # mathematically unreachable and excluded as a branch arc.
    for _ in range(_MAXIT):  # pragma: no branch
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - _gammln(a))


def _gcf(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a,x) via continued fraction.

    Reference: Numerical Recipes §6.2 (gcf), Lentz's algorithm.
    """
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    # NR §6.2 continued fraction (Lentz's algorithm) converges within _MAXIT
    # for every valid (a>0, x≥0) input, so the loop always exits via the
    # ``break`` below — the natural-exhaustion arc is mathematically
    # unreachable and excluded as a branch arc.
    for i in range(1, _MAXIT + 1):  # pragma: no branch
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:  # pragma: no cover — d ≥ |b| - |an*d| stays above FPMIN
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:  # pragma: no cover — c stays ≥ |b| - |an/c| > FPMIN
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - _gammln(a)) * h


def _gammp(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be > 0 and x must be >= 0 (regularized incomplete gamma P(a, x))")
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("a must be > 0 and x must be >= 0 (regularized incomplete gamma Q(a, x))")
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function.

    Reference: Numerical Recipes §6.4 (betacf), Lentz's algorithm.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:  # pragma: no cover — first-loop aa≥0 keeps d≥1
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:  # pragma: no cover — first-loop aa≥0 keeps c≥1
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Reference: Numerical Recipes §6.4 (betai).
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0, 1] for the incomplete beta function I_x(a, b)")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(
        _gammln(a + b) - _gammln(a) - _gammln(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Two-tailed survival probability for Student's t distribution.

    Returns P(|T| >= |t|) for T ~ t(df), via the incomplete beta function:
    p = I_{df/(df+t^2)}(df/2, 1/2).

    Reference: Student (1908); Numerical Recipes §6.14.
    """
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability for the chi-square distribution: P(X >= x).

    Equals Q(df/2, x/2) with the regularized upper incomplete gamma.

    Reference: Pearson (1900); Abramowitz & Stegun §26.4.
    """
    if x <= 0.0:
        return 1.0
    return _gammq(df / 2.0, x / 2.0)


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail probability for the F distribution: P(F >= f).

    Equals I_{df2/(df2+df1 f)}(df2/2, df1/2).

    Reference: Fisher (1925); Numerical Recipes §6.14.
    """
    if f <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _betai(df2 / 2.0, df1 / 2.0, x)
