"""Classical statistical hypothesis tests in pure Python.

Implements Student's t-test, Pearson's chi-square test, and Fisher's
one-way ANOVA. The required tail probabilities are computed by the
regularized incomplete beta and gamma functions, which live in the sibling
:mod:`cds.stats._distributions` module and are re-exported here so the
long-standing ``from cds.stats.hypothesis_tests import t_sf`` (and the
private ``_gser`` / ``_gcf`` / ... coverage helpers) keep working.

References:
    - Student [W. S. Gosset] (1908). "The probable error of a mean."
      Biometrika, 6(1), 1-25. (t-distribution / t-test)
    - Pearson, K. (1900). "On the criterion that a given system of
      deviations from the probable... is such that it can be reasonably
      supposed to have arisen from random sampling." Philosophical
      Magazine, 50(302), 157-175. (chi-square test)
    - Fisher, R. A. (1925). "Statistical Methods for Research Workers."
      Oliver & Boyd. (analysis of variance, F-distribution)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cds.stats._distributions import (  # re-exported for compat, see __all__
    _EPS,
    _FPMIN,
    _MAXIT,
    _betacf,
    _betai,
    _gammln,
    _gammp,
    _gammq,
    _gcf,
    _gser,
    chi2_sf,
    f_sf,
    t_sf,
)
from cds.stats.descriptive import mean, variance

# Explicit re-export list. mypy in strict mode does not treat a bare
# ``from x import name`` import (even one suppressed for the unused-import
# lint rule) as an exported module attribute, so
# ``from cds.stats.hypothesis_tests import t_sf`` fails its attr-defined check.
# Declaring ``__all__`` marks these names as the module's public surface and
# satisfies both mypy and the linter that the imports are intentional
# re-exports rather than unused. The distribution helpers
# (``t_sf``/``chi2_sf``/``f_sf``) and the private incomplete-gamma/beta
# routines are kept re-exported for the long-standing import paths documented
# in the module docstring.
__all__ = [
    "TestResult",
    "one_sample_ttest",
    "two_sample_ttest",
    "chi_square_gof",
    "chi_square_independence",
    "one_way_anova",
    "cohens_d",
    "eta_squared_from_f",
    "cramers_v",
    "bonferroni_corrected_alpha",
    # Re-exported from cds.stats._distributions for backward compatibility.
    "t_sf",
    "chi2_sf",
    "f_sf",
    "_EPS",
    "_FPMIN",
    "_MAXIT",
    "_betacf",
    "_betai",
    "_gammln",
    "_gammp",
    "_gammq",
    "_gcf",
    "_gser",
]


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
    a: list[float],
    b: list[float],
    equal_var: bool = True,
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
    observed: list[float],
    expected: list[float],
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
    ss_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups)
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0.0:
        raise ValueError("zero within-group variance; F undefined")
    f = ms_between / ms_within
    return TestResult(
        statistic=f,
        df=df_between,
        p_value=f_sf(f, df_between, df_within),
    )


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d effect size for the difference of two independent means.

    Standardized mean difference using the pooled standard deviation:

        d = (mean_a - mean_b) / s_pooled
        s_pooled = sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    Conventional interpretation (Cohen, 1988): |d| ~ 0.2 small, ~0.5 medium,
    ~0.8 large. The sign reflects the direction (positive = group_a larger).

    Reference: Cohen, J. (1988). "Statistical Power Analysis for the
    Behavioral Sciences," 2nd ed., Lawrence Erlbaum, §2.5.

    Args:
        group_a: first sample (n >= 2)
        group_b: second sample (n >= 2)

    Returns:
        Cohen's d (signed). Zero when the group means are equal.

    Raises:
        ValueError: if either group has fewer than 2 observations, or if both
            groups are constant (pooled variance is zero).
    """
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        raise ValueError("each sample needs at least 2 observations")
    va, vb = variance(group_a, ddof=1), variance(group_b, ddof=1)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if sp2 == 0.0:
        raise ValueError("zero pooled variance; Cohen's d undefined")
    return (mean(group_a) - mean(group_b)) / math.sqrt(sp2)


def eta_squared_from_f(f: float, df1: int, df2: int) -> float:
    """Eta-squared effect size for a one-way ANOVA from its F statistic.

    Converts the F ratio of a one-way ANOVA into the proportion of variance
    accounted for by group membership:

        eta^2 = (F * df1) / (F * df1 + df2)

    where df1 = k - 1 (between groups) and df2 = N - k (within groups).
    Range [0, 1]; 0 when F = 0. Conventional interpretation (Cohen, 1988):
    ~0.01 small, ~0.06 medium, ~0.14 large.

    Reference: Cohen, J. (1988). "Statistical Power Analysis for the
    Behavioral Sciences," 2nd ed., Lawrence Erlbaum, §8.2.

    Args:
        f: the ANOVA F statistic (>= 0)
        df1: between-groups degrees of freedom (k - 1, >= 1)
        df2: within-groups degrees of freedom (N - k, >= 1)

    Returns:
        Eta-squared in [0, 1].

    Raises:
        ValueError: if f < 0, df1 < 1, or df2 < 1.
    """
    if f < 0.0:
        raise ValueError("F statistic must be non-negative")
    if df1 < 1 or df2 < 1:
        raise ValueError("df1 and df2 must be >= 1")
    return (f * df1) / (f * df1 + df2)


def cramers_v(table: list[list[float]]) -> float:
    """Cramer's V effect size for a chi-square test of independence.

    Normalizes the chi-square statistic of an r x c contingency table to
    [0, 1] by the sample size and the smaller dimension:

        V = sqrt(chi^2 / (N * min(r - 1, c - 1)))

    0 = no association, 1 = perfect association. Conventional interpretation
    (Cohen, 1988): ~0.1 small, ~0.3 medium, ~0.5 large.

    Reference: Cramer, H. (1946). "Mathematical Methods of Statistics,"
    Princeton University Press. Interpretation cutoffs from Cohen (1988).

    Args:
        table: r x c contingency table of non-negative counts (r, c >= 2).

    Returns:
        Cramer's V in [0, 1].

    Raises:
        ValueError: if the table is smaller than 2x2, non-rectangular, or has
            a non-positive total.
    """
    rows = len(table)
    if rows < 2:
        raise ValueError("need at least 2 rows")
    cols = len(table[0])
    if cols < 2 or any(len(r) != cols for r in table):
        raise ValueError("need a rectangular table with at least 2 columns")
    n = sum(sum(r) for r in table)
    if n <= 0:
        raise ValueError("table total must be positive")
    # Reuse chi_square_independence for the statistic rather than recomputing.
    chi2 = chi_square_independence(table).statistic
    return math.sqrt(chi2 / (n * min(rows - 1, cols - 1)))


def bonferroni_corrected_alpha(alpha: float, k: int) -> float:
    """Family-wise corrected significance level for k comparisons.

    Returns the Bonferroni-corrected per-test alpha that keeps the
    family-wise error rate at approximately ``alpha`` across ``k`` tests:

        alpha_corrected = alpha / k

    The corrected level is what each individual comparison must beat. For
    k = 1 it is a no-op (returns ``alpha``). Use it with ``evaluate_batch``
    so a single significant p-value among many does not overstate evidence.

    Reference: Bonferroni, C. E. (1936); applied as a multiple-comparison
    procedure by Dunn, O. J. (1961), "Multiple Comparisons Among Means,"
    Journal of the American Statistical Association, 56(293), 52-64.

    Args:
        alpha: desired family-wise error rate (e.g. 0.05), in (0, 1).
        k: number of comparisons in the family (>= 1).

    Returns:
        Corrected per-test alpha (= alpha / k).

    Raises:
        ValueError: if k < 1, or alpha is outside (0, 1).
    """
    if k < 1:
        raise ValueError("number of comparisons k must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")
    return alpha / k
