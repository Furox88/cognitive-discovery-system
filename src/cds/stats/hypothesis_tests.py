"""Classical statistical hypothesis tests in pure Python.

Implements Student's t-test, Pearson's chi-square test, and Fisher's
one-way ANOVA. The required tail probabilities are computed from the
regularized incomplete beta and gamma functions, evaluated with the
series / continued-fraction methods of Numerical Recipes.

References:
    - Student [W. S. Gosset] (1908). "The probable error of a mean."
      Biometrika, 6(1), 1-25. (t-distribution / t-test)
    - Pearson, K. (1900). "On the criterion that a given system of
      deviations from the probable... is such that it can be reasonably
      supposed to have arisen from random sampling." Philosophical
      Magazine, 50(302), 157-175. (chi-square test)
    - Fisher, R. A. (1925). "Statistical Methods for Research Workers."
      Oliver & Boyd. (analysis of variance, F-distribution)
    - Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
      (2007). "Numerical Recipes," 3rd ed., §6.2-6.4 (incomplete gamma and
      beta functions: gammln, gser, gcf, betacf).
    - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical
      Functions," §6.5, §26.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from cds.stats.descriptive import mean, variance

_MAXIT = 200
_EPS = 3.0e-12
_FPMIN = 1.0e-300


def _gammln(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation).

    Reference: Numerical Recipes §6.1; Lanczos (1964).
    """
    cof = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
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
    for _ in range(_MAXIT):
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
    for i in range(1, _MAXIT + 1):
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
        raise ValueError("invalid arguments to gammp")
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to gammq")
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
        raise ValueError("x must be in [0, 1]")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(
        _gammln(a + b) - _gammln(a) - _gammln(b)
        + a * math.log(x) + b * math.log(1.0 - x)
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


@dataclass
class TestResult:
    """Result of a hypothesis test: test statistic, degrees of freedom, p."""

    statistic: float
    df: float
    p_value: float


def one_sample_ttest(data: list[float], popmean: float = 0.0) -> TestResult:
    """One-sample Student's t-test against a population mean.

    Tests H0: mean(data) == popmean. The statistic is
    t = (x_bar - mu) / (s / sqrt(n)) with n-1 degrees of freedom.

    Reference: Student [Gosset] (1908), Biometrika 6(1), 1-25.

    Args:
        data: sample observations (n >= 2)
        popmean: hypothesized population mean

    Returns:
        TestResult with t statistic, df = n-1, two-tailed p-value
    """
    n = len(data)
    if n < 2:
        raise ValueError("need at least 2 observations")
    df = n - 1
    se = math.sqrt(variance(data, ddof=1) / n)
    if se == 0.0:
        raise ValueError("zero variance; t-test undefined")
    t = (mean(data) - popmean) / se
    return TestResult(statistic=t, df=df, p_value=t_sf(t, df))


def two_sample_ttest(
    a: list[float], b: list[float], equal_var: bool = True,
) -> TestResult:
    """Two-sample t-test for equality of means.

    With ``equal_var=True`` uses the pooled-variance (Student) t-test; with
    ``equal_var=False`` uses Welch's t-test with the Welch-Satterthwaite
    degrees of freedom.

    References:
        - Student [Gosset] (1908), Biometrika 6(1), 1-25.
        - Welch, B. L. (1947). "The generalization of 'Student's' problem
          when several different population variances are involved."
          Biometrika, 34(1-2), 28-35.

    Args:
        a: first sample (n >= 2)
        b: second sample (n >= 2)
        equal_var: pooled-variance test if True, Welch's test otherwise

    Returns:
        TestResult with t statistic, degrees of freedom, two-tailed p-value
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("each sample needs at least 2 observations")
    va, vb = variance(a, ddof=1), variance(b, ddof=1)
    diff = mean(a) - mean(b)
    if equal_var:
        df = na + nb - 2
        sp2 = ((na - 1) * va + (nb - 1) * vb) / df
        se = math.sqrt(sp2 * (1.0 / na + 1.0 / nb))
        df_eff = float(df)
    else:
        se = math.sqrt(va / na + vb / nb)
        num = (va / na + vb / nb) ** 2
        den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        df_eff = num / den
    if se == 0.0:
        raise ValueError("zero variance; t-test undefined")
    t = diff / se
    return TestResult(statistic=t, df=df_eff, p_value=t_sf(t, df_eff))


def chi_square_gof(
    observed: list[float], expected: list[float],
) -> TestResult:
    """Pearson's chi-square goodness-of-fit test.

    Statistic chi2 = sum((O_i - E_i)^2 / E_i) with len-1 degrees of freedom.

    Reference: Pearson, K. (1900), Philosophical Magazine 50(302), 157-175.

    Args:
        observed: observed counts
        expected: expected counts (same length, all > 0)

    Returns:
        TestResult with chi2 statistic, df = k-1, upper-tail p-value
    """
    if len(observed) != len(expected):
        raise ValueError("observed and expected must have same length")
    if len(observed) < 2:
        raise ValueError("need at least 2 categories")
    if any(e <= 0 for e in expected):
        raise ValueError("expected counts must be positive")
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    df = len(observed) - 1
    return TestResult(statistic=chi2, df=df, p_value=chi2_sf(chi2, df))


def chi_square_independence(table: list[list[float]]) -> TestResult:
    """Pearson's chi-square test of independence for a contingency table.

    Expected counts E_ij = (row_i total)(col_j total) / grand total;
    degrees of freedom (rows-1)(cols-1).

    Reference: Pearson, K. (1900), Philosophical Magazine 50(302), 157-175.

    Args:
        table: r x c contingency table of non-negative counts

    Returns:
        TestResult with chi2 statistic, df, upper-tail p-value
    """
    rows = len(table)
    if rows < 2:
        raise ValueError("need at least 2 rows")
    cols = len(table[0])
    if cols < 2 or any(len(r) != cols for r in table):
        raise ValueError("need a rectangular table with at least 2 columns")
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    grand = sum(row_tot)
    if grand == 0:
        raise ValueError("table total must be positive")
    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            exp = row_tot[i] * col_tot[j] / grand
            if exp > 0:
                chi2 += (table[i][j] - exp) ** 2 / exp
    df = (rows - 1) * (cols - 1)
    return TestResult(statistic=chi2, df=df, p_value=chi2_sf(chi2, df))


def one_way_anova(*groups: list[float]) -> TestResult:
    """Fisher's one-way analysis of variance (ANOVA F-test).

    Partitions total variability into between-group and within-group sums of
    squares and forms F = MS_between / MS_within with (k-1, N-k) degrees of
    freedom.

    Reference: Fisher, R. A. (1925). "Statistical Methods for Research
    Workers," Oliver & Boyd.

    Args:
        *groups: two or more samples, each with at least one observation

    Returns:
        TestResult with F statistic, df = k-1 (stored), upper-tail p-value.
        The within-group degrees of freedom (N-k) are used internally for p.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("need at least 2 groups")
    if any(len(g) < 1 for g in groups):
        raise ValueError("each group needs at least one observation")
    n_total = sum(len(g) for g in groups)
    if n_total <= k:
        raise ValueError("need more observations than groups")
    grand_mean = sum(sum(g) for g in groups) / n_total
    ss_between = sum(len(g) * (mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(
        sum((x - mean(g)) ** 2 for x in g) for g in groups
    )
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0.0:
        raise ValueError("zero within-group variance; F undefined")
    f = ms_between / ms_within
    return TestResult(
        statistic=f, df=df_between, p_value=f_sf(f, df_between, df_within),
    )
