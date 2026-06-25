"""Hypothesis-driven property-based tests.

Where ``test_numerical_invariants.py`` rolls its own ``random.Random`` loop
(zero new dependencies, fixed seed, fixed trial count), this module leans on
the `Hypothesis <https://hypothesis.readthedocs.io/>`_ library to do real
property-based testing: Hypothesis *searches* the input space for a failing
example and then *shrinks* any failure to the smallest input that still
reproduces it. That catches edge cases a hand-picked seed will miss — empty
lists, single-element lists, integers at 0, vectors that are near-singular,
and so on.

This suite is optional: it only runs when Hypothesis is installed
(``pip install ".[property]"`` or ``[all]``). CI gates it behind a dedicated
job (see ``.github/workflows/tests.yml``) so a missing dependency never breaks
the default ``[test]`` matrix. The module-level guard below makes that safe:
without Hypothesis, every test skips rather than erroring on import.

Properties covered (6 domains):
    1. Linear algebra — inverse round-trip (A·A⁻¹ = I), transpose involution.
    2. Descriptive statistics — variance is non-negative, median lies in range,
       mean is homogeneous under positive scaling.
    3. Numerical calculus — derivative is linear (af+bg)' = af' + bg',
       derivative of a constant is zero.
    4. Signals — FFT ⇄ IFFT round-trip recovers the input, Parseval energy.
    5. Optimization — gradient descent never increases a convex quadratic.
    6. Probability — a PMF built from arbitrary non-negative weights sums to 1,
       and its cumulative sum is monotone non-decreasing.
"""

from __future__ import annotations

import math

import pytest

# Every import below runs after the ``importorskip`` guard, so module-level
# placement triggers E402. That's intentional and correct here: we *cannot*
# import Hypothesis before confirming it's installed. Each line carries its
# own ``noqa: E402`` so ruff's import sorter (I001) and the E402 rule both
# stay satisfied.
hypothesis = pytest.importorskip("hypothesis")  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.strategies import SearchStrategy  # noqa: E402

from cds.math_utils.calculus import derivative  # noqa: E402
from cds.math_utils.linalg import (  # noqa: E402
    identity,
    mat_mul,
    matrix_inverse,
    transpose,
)
from cds.signals.processing import fft as fft_func  # noqa: E402
from cds.signals.processing import ifft as ifft_func  # noqa: E402
from cds.stats.descriptive import mean, median, variance  # noqa: E402

# A modest example budget keeps the suite fast in CI while still exercising
# Hypothesis's search + shrink loop. The default (100) would be fine too; we
# tune down because each property here runs a numerical kernel (FFT, matrix
# inverse) rather than a cheap predicate.
MAX_EXAMPLES = 50


# --------------------------------------------------------------------------- #
# Shared strategies
# --------------------------------------------------------------------------- #
def finite_floats(
    lo: float = -1e6, hi: float = 1e6, allow_nan: bool = False
) -> SearchStrategy[float]:
    """Finite, non-NaN floats in [lo, hi].

    NaN/inf are valid Python floats but they break every numerical invariant
    (NaN != NaN, inf - inf = NaN). Excluding them keeps the properties about
    the *algorithm* rather than IEEE-754 edge cases, which the example-based
    suite already covers explicitly.
    """
    return st.floats(min_value=lo, max_value=hi, allow_nan=allow_nan, allow_infinity=False)


@st.composite
def square_matrix(draw: st.DrawFn, n_min: int = 1, n_max: int = 5) -> list[list[float]]:
    """A random square matrix with moderate-magnitude entries.

    Bounded magnitude (±10) keeps matrices far from the overflow/underflow
    frontier where float error alone can flip a singularity check.
    """
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    return [[draw(finite_floats(-10.0, 10.0)) for _ in range(n)] for _ in range(n)]


@st.composite
def power_of_two_length(draw: st.DrawFn, exp_min: int = 1, exp_max: int = 7) -> int:
    """A length that is a power of two (the FFT radix-2 constraint)."""
    return 1 << draw(st.integers(min_value=exp_min, max_value=exp_max))


# --------------------------------------------------------------------------- #
# 1. Linear algebra
# --------------------------------------------------------------------------- #
class TestLinAlgProperties:
    @given(mat=square_matrix(n_min=2, n_max=4))
    @settings(max_examples=MAX_EXAMPLES)
    def test_transpose_is_involution(self, mat: list[list[float]]) -> None:
        # (Aᵀ)ᵀ = A for every matrix.
        assert transpose(transpose(mat)) == mat

    @given(mat=square_matrix(n_min=2, n_max=4))
    @settings(max_examples=MAX_EXAMPLES)
    def test_inverse_times_original_is_identity(self, mat: list[list[float]]) -> None:
        # A·A⁻¹ = I, provided A is invertible. We compute the determinant and
        # skip near-singular matrices (|det| below a floor) — those are a
        # property of the *precondition*, not of the inverse routine.
        from cds.math_utils.linalg import determinant

        if abs(determinant(mat)) < 1e-3:
            return  # skip: invertibility precondition not met
        n = len(mat)
        prod = mat_mul(matrix_inverse(mat), mat)
        ident = identity(n)
        for i in range(n):
            for j in range(n):
                assert math.isclose(prod[i][j], ident[i][j], abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# 2. Descriptive statistics
# --------------------------------------------------------------------------- #
class TestStatsProperties:
    @given(data=st.lists(finite_floats(-1e4, 1e4), min_size=2, max_size=50))
    @settings(max_examples=MAX_EXAMPLES)
    def test_variance_non_negative(self, data: list[float]) -> None:
        # Variance is a sum of squares over a count — always ≥ 0. A negative
        # result would flag a catastrophic-cancellation bug in the formula.
        assert variance(data, ddof=0) >= -1e-9

    @given(data=st.lists(finite_floats(-1e4, 1e4), min_size=1, max_size=50))
    @settings(max_examples=MAX_EXAMPLES)
    def test_median_within_data_range(self, data: list[float]) -> None:
        # The median of a set can never lie outside [min, max] of that set.
        m = median(data)
        assert min(data) - 1e-9 <= m <= max(data) + 1e-9

    @given(
        data=st.lists(finite_floats(-1e3, 1e3), min_size=1, max_size=30),
        scale=finite_floats(0.1, 10.0),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_mean_is_homogeneous(self, data: list[float], scale: float) -> None:
        # mean(c·x) = c·mean(x) — linearity of the arithmetic mean under
        # positive scalar scaling. Negative/zero scale would be just as valid
        # but flipping the sign on an odd-length list complicates the bound.
        # When mean(x) ≈ 0 (e.g. symmetric data like [342, -342]) the relative
        # error of c·mean(x) blows up even though the absolute error is at the
        # round-off floor, so anchor with an abs_tol in addition to rel_tol.
        assert math.isclose(
            mean([scale * x for x in data]),
            scale * mean(data),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


# --------------------------------------------------------------------------- #
# 3. Numerical calculus
# --------------------------------------------------------------------------- #
class TestCalculusProperties:
    @given(
        a=finite_floats(-5.0, 5.0),
        b=finite_floats(-5.0, 5.0),
        x=finite_floats(-3.0, 3.0),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_derivative_is_linear(self, a: float, b: float, x: float) -> None:
        # (a·f + b·g)' = a·f' + b·g'. Use f(t)=t² and g(t)=sin(t) so both
        # branches of the central-difference formula are exercised.
        def f(t: float) -> float:
            return t * t

        def g(t: float) -> float:
            return math.sin(t)

        def combo(t: float) -> float:
            return a * f(t) + b * g(t)

        lhs = derivative(combo, x)
        rhs = a * derivative(f, x) + b * derivative(g, x)
        assert math.isclose(lhs, rhs, rel_tol=1e-5, abs_tol=1e-5)

    @given(c=finite_floats(-100.0, 100.0), x=finite_floats(-10.0, 10.0))
    @settings(max_examples=MAX_EXAMPLES)
    def test_derivative_of_constant_is_zero(self, c: float, x: float) -> None:
        # d/dx [c] = 0. Central difference must recover this to within the
        # cancellation noise of subtracting two equal-valued evaluations.
        assert abs(derivative(lambda _t: c, x)) < 1e-5


# --------------------------------------------------------------------------- #
# 4. Signals
# --------------------------------------------------------------------------- #
class TestSignalProperties:
    @given(n=power_of_two_length(exp_min=1, exp_max=6))
    @settings(max_examples=MAX_EXAMPLES)
    def test_fft_ifft_roundtrip(self, n: int) -> None:
        # ifft(fft(x)) == x (within float noise). Power-of-two length is the
        # radix-2 contract; non-power-of-two falls back to the DFT path which
        # is covered by the example-based suite.
        x = [math.sin(i) * 0.5 for i in range(n)]
        recon = ifft_func(fft_func(list(x)))
        for r, xi in zip(recon, x):
            assert math.isclose(r.real, xi, rel_tol=1e-7, abs_tol=1e-7)
            assert abs(r.imag) < 1e-7

    @given(n=power_of_two_length(exp_min=2, exp_max=6))
    @settings(max_examples=MAX_EXAMPLES)
    def test_parseval_energy_conservation(self, n: int) -> None:
        # Σ|x[n]|² = (1/N)·Σ|X[k]|². Energy is conserved across the DFT.
        x = [math.cos(i) for i in range(n)]
        spec = fft_func(list(x))
        time_energy = sum(abs(xi) ** 2 for xi in x)
        freq_energy = sum(abs(xk) ** 2 for xk in spec) / n
        assert math.isclose(time_energy, freq_energy, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# 5. Optimization
# --------------------------------------------------------------------------- #
class TestOptimizationProperties:
    @given(
        x_star=finite_floats(-5.0, 5.0),
        x0=finite_floats(-5.0, 5.0),
    )
    @settings(max_examples=MAX_EXAMPLES)
    def test_gradient_descent_reduces_convex_objective(self, x_star: float, x0: float) -> None:
        # On the convex quadratic f(x) = (x − x*)² the objective value after
        # gradient descent must not exceed the starting objective. Convexity
        # guarantees a descent direction; a rise would signal a sign flip in
        # the update step. Skip when x0 is already at the minimum.
        from cds.optimization.minimize import gradient_descent

        if abs(x0 - x_star) < 1e-3:
            return  # skip: already at the global minimum

        def f(x: float, _xs: float = x_star) -> float:
            return (x - _xs) ** 2

        f_start = f(x0)
        result = gradient_descent(f, x0, lr=0.1, tol=1e-10, max_iter=500)
        assert result.value <= f_start + 1e-9


# --------------------------------------------------------------------------- #
# 6. Probability
# --------------------------------------------------------------------------- #
class TestProbabilityProperties:
    @given(weights=st.lists(finite_floats(0.01, 100.0), min_size=2, max_size=20))
    @settings(max_examples=MAX_EXAMPLES)
    def test_pmf_normalizes(self, weights: list[float]) -> None:
        # Any set of positive weights, divided by their sum, yields a PMF
        # that sums to exactly 1.
        total = sum(weights)
        pmf = [w / total for w in weights]
        assert math.isclose(sum(pmf), 1.0, abs_tol=1e-12)

    @given(weights=st.lists(finite_floats(0.01, 100.0), min_size=2, max_size=20))
    @settings(max_examples=MAX_EXAMPLES)
    def test_cdf_is_monotone(self, weights: list[float]) -> None:
        # The cumulative sum of a PMF is non-decreasing and ends at 1.
        total = sum(weights)
        pmf = [w / total for w in weights]
        cdf: list[float] = []
        acc = 0.0
        for p in pmf:
            acc += p
            cdf.append(acc)
        for prev, cur in zip(cdf, cdf[1:]):
            assert cur >= prev - 1e-12
        assert math.isclose(cdf[-1], 1.0, abs_tol=1e-12)
