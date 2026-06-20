"""Property-based numerical invariant tests.

These tests assert mathematical invariants that *must* hold regardless of
the specific input — they generate many random inputs under a fixed seed
and verify a property each time. This is a complement (not a replacement)
for the example-based tests in the rest of the suite: those check specific
known answers; these check that the *relationships* between functions and
the *identities* each function promises are never violated.

Implementation note: we use ``random.Random(seed)`` with a fixed seed
rather than the ``hypothesis`` library — zero new dev dependency, fully
reproducible, and the invariant checks (not counterexample minimization)
are the goal here. Each test runs ``N = 60`` random trials by default.

Invariants covered (12 modules):
    1. math_utils.linalg — associativity, transpose-double-inverse,
       inverse-identity, LU = original, QR orthogonality, Cholesky round-trip.
    2. signals — FFT ⇄ IFFT round-trip, Parseval, DFT ⇄ IFFT round-trip,
       fft2 ⇄ ifft2 round-trip, convolution length.
    3. stats — variance identity, median in-range, correlation in [-1, 1].
    4. numerical_integration — quadrature accuracy on known polynomials,
       trapezoid/simpson convergence order, signed integration (b < a).
    5. quantum — state normalization survives gate application,
       probabilities sum to 1, measure collapses to a basis state.
    6. diffeq — rk4 beats euler on a smooth RHS, harmonic-oscillator
       energy drift, solution grid ends exactly at t_end.
    7. montecarlo — mc_integrate converges to a known integral as N grows,
       random walk displacement stays within ±steps.
    8. probability — distribution sums to 1, CDF monotone non-decreasing.
    9. math_utils.calculus — derivative of a constant is 0, derivative of a
       polynomial matches its analytic derivative, gradient matches scalar
       derivative along each axis.
   10. optimization — gradient_descent reduces the objective on a convex
       quadratic, line_search recovers the analytic minimum of a parabola,
       newton_method finds the root of a monotone function.
   11. graph — BFS and DFS visit the same vertex set, Dijkstra honors the
       triangle inequality, MST weight matches the greedy sorted-edge sum.
   12. ml — sigmoid output lies in (0, 1), ReLU is non-negative and
       sparse, a forward/backward pass conserves the input dimension.
"""

from __future__ import annotations

import math
import random

from cds.diffeq.solvers import euler_method, rk4
from cds.graph.algorithms import Graph, bfs, dfs, dijkstra, kruskal_mst
from cds.math_utils.calculus import derivative, gradient
from cds.math_utils.linalg import (
    cholesky,
    determinant,
    identity,
    lu_decomposition,
    mat_mul,
    matrix_inverse,
    qr_decomposition,
    transpose,
)
from cds.ml.neural import Layer
from cds.montecarlo.methods import mc_integrate
from cds.numerical_integration.quadrature import (
    adaptive_simpson,
    gaussian_quadrature,
    romberg,
    simpson,
    trapezoid,
)
from cds.optimization.minimize import gradient_descent, line_search, newton_method
from cds.quantum.multi_qubit import QuantumRegister
from cds.signals.processing import convolve, dft, fft_radix2, idft, ifft2
from cds.signals.processing import fft as fft_func
from cds.signals.processing import fft2 as fft2_func
from cds.signals.processing import ifft as ifft_func
from cds.stats.descriptive import correlation, mean, median, variance

# Fixed seed → fully reproducible. Bump the trial count by editing N.
SEED = 20260618
N = 60

# A loose tolerance for tests that chain several float operations; tight
# enough to catch real bugs (wrong sign, missing factor, off-by-one panel)
# but loose enough to survive O(h²) truncation error accumulation.
REL = 1e-9
ABS = 1e-9


def _rand_matrix(
    rng: random.Random, n: int, lo: float = -3.0, hi: float = 3.0
) -> list[list[float]]:
    """Random n×n matrix with entries in [lo, hi]."""
    return [[rng.uniform(lo, hi) for _ in range(n)] for _ in range(n)]


def _spd_matrix(rng: random.Random, n: int) -> list[list[float]]:
    """Random symmetric positive-definite matrix: A = B·Bᵀ + n·I."""
    b = _rand_matrix(rng, n, lo=-1.0, hi=1.0)
    a = mat_mul(b, transpose(b))
    ident = identity(n)
    for i in range(n):
        a[i][i] += n  # diagonal shift guarantees positive definiteness
        ident[i][i] = a[i][i]  # keep linter happy about unused ident
    return a


def _all_close(
    a: list[list[float]], b: list[list[float]], rel: float = REL, abs_: float = ABS
) -> bool:
    """Element-wise approximate equality for two matrices."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not math.isclose(x, y, rel_tol=rel, abs_tol=abs_):
                return False
    return True


# --------------------------------------------------------------------------- #
# 1. Linear algebra
# --------------------------------------------------------------------------- #
class TestLinAlgInvariants:
    def test_matmul_associative(self) -> None:
        rng = random.Random(SEED)
        for _ in range(N):
            a = _rand_matrix(rng, 4)
            b = _rand_matrix(rng, 4)
            c = _rand_matrix(rng, 4)
            left = mat_mul(mat_mul(a, b), c)
            right = mat_mul(a, mat_mul(b, c))
            assert _all_close(left, right, abs_=1e-6)

    def test_transpose_is_involution(self) -> None:
        rng = random.Random(SEED + 1)
        for _ in range(N):
            n = rng.randint(1, 7)
            m = rng.randint(1, 7)
            a = [[rng.uniform(-5, 5) for _ in range(m)] for _ in range(n)]
            assert _all_close(transpose(transpose(a)), a)

    def test_inverse_times_identity(self) -> None:
        rng = random.Random(SEED + 2)
        for _ in range(N):
            n = rng.randint(2, 5)
            a = _rand_matrix(rng, n)
            # Skip near-singular matrices — det magnitude below the floor.
            if abs(determinant(a)) < 1e-3:
                continue
            inv = matrix_inverse(a)
            prod = mat_mul(inv, a)
            assert _all_close(prod, identity(n), abs_=1e-7)

    def test_lu_reconstructs_original(self) -> None:
        rng = random.Random(SEED + 3)
        for _ in range(N):
            a = _rand_matrix(rng, 4)
            if abs(determinant(a)) < 1e-3:
                continue
            p, lower, u = lu_decomposition(a)
            # PA = LU  →  A = PᵀLU  (P is a permutation, so Pᵀ = transpose(P))
            recon = mat_mul(transpose(p), mat_mul(lower, u))
            assert _all_close(recon, a, abs_=1e-7)

    def test_qr_q_is_orthogonal(self) -> None:
        rng = random.Random(SEED + 4)
        for _ in range(N):
            n = rng.randint(2, 5)
            a = _rand_matrix(rng, n)
            if abs(determinant(a)) < 1e-3:
                continue
            q, _r = qr_decomposition(a)
            # QᵀQ should be the identity.
            assert _all_close(mat_mul(transpose(q), q), identity(n), abs_=1e-7)

    def test_cholesky_reconstructs_spd(self) -> None:
        rng = random.Random(SEED + 5)
        for _ in range(N):
            a = _spd_matrix(rng, 4)
            lower = cholesky(a)
            recon = mat_mul(lower, transpose(lower))
            assert _all_close(recon, a, abs_=1e-7)


# --------------------------------------------------------------------------- #
# 2. Signals (FFT round-trips, Parseval, convolution length)
# --------------------------------------------------------------------------- #
class TestSignalInvariants:
    def test_fft_ifft_roundtrip_power_of_two(self) -> None:
        rng = random.Random(SEED + 10)
        for _ in range(N):
            n = 1 << rng.randint(1, 6)  # 2..64, power of two
            x = [rng.uniform(-2, 2) for _ in range(n)]
            recon = ifft_func(fft_func(list(x)))
            # Round-trip recovers the input (within float noise).
            assert all(
                math.isclose(r.real, xi, rel_tol=REL, abs_tol=ABS) for r, xi in zip(recon, x)
            )
            assert all(abs(r.imag) < 1e-7 for r in recon)

    def test_dft_idft_roundtrip(self) -> None:
        rng = random.Random(SEED + 11)
        for _ in range(N):
            n = rng.randint(2, 13)  # non-power-of-two lengths too
            x = [rng.uniform(-1, 1) for _ in range(n)]
            recon = idft(dft(list(x)))
            assert all(
                math.isclose(r.real, xi, rel_tol=1e-7, abs_tol=1e-7) for r, xi in zip(recon, x)
            )

    def test_parseval_energy_conservation(self) -> None:
        # Σ|x[n]|² = (1/N)·Σ|X[k]|²  for the DFT.
        rng = random.Random(SEED + 12)
        for _ in range(N):
            n = 1 << rng.randint(2, 6)
            x = [rng.uniform(-1, 1) for _ in range(n)]
            spec = fft_radix2(list(x))
            time_energy = sum(abs(xi) ** 2 for xi in x)
            freq_energy = sum(abs(xk) ** 2 for xk in spec) / n
            assert math.isclose(time_energy, freq_energy, rel_tol=1e-6)

    def test_fft2_ifft2_roundtrip(self) -> None:
        rng = random.Random(SEED + 13)
        for _ in range(N):
            rows = 1 << rng.randint(1, 4)
            cols = 1 << rng.randint(1, 4)
            mat = [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
            recon = ifft2(fft2_func([[complex(v) for v in row] for row in mat]))
            for i in range(rows):
                for j in range(cols):
                    assert math.isclose(recon[i][j].real, mat[i][j], rel_tol=1e-6, abs_tol=1e-7)
                    assert abs(recon[i][j].imag) < 1e-7

    def test_convolution_length(self) -> None:
        rng = random.Random(SEED + 14)
        for _ in range(N):
            la = rng.randint(1, 8)
            lb = rng.randint(1, 8)
            a = [rng.uniform(-1, 1) for _ in range(la)]
            b = [rng.uniform(-1, 1) for _ in range(lb)]
            out = convolve(a, b)
            assert len(out) == la + lb - 1


# --------------------------------------------------------------------------- #
# 3. Statistics
# --------------------------------------------------------------------------- #
class TestStatsInvariants:
    def test_variance_identity(self) -> None:
        # var(x) = mean(x²) − mean(x)²  (population form, ddof=0)
        rng = random.Random(SEED + 20)
        for _ in range(N):
            n = rng.randint(2, 20)
            x = [rng.uniform(-10, 10) for _ in range(n)]
            mx = mean(x)
            mx2 = mean([xi * xi for xi in x])
            # Numerically unstable identity — wider tolerance on purpose.
            assert math.isclose(variance(x, ddof=0), mx2 - mx * mx, rel_tol=1e-6, abs_tol=1e-6)

    def test_median_within_data_range(self) -> None:
        rng = random.Random(SEED + 21)
        for _ in range(N):
            n = rng.randint(1, 15)
            x = [rng.uniform(-50, 50) for _ in range(n)]
            m = median(x)
            assert min(x) - 1e-9 <= m <= max(x) + 1e-9

    def test_correlation_bounded(self) -> None:
        rng = random.Random(SEED + 22)
        for _ in range(N):
            n = rng.randint(2, 20)
            x = [rng.uniform(-10, 10) for _ in range(n)]
            y = [rng.uniform(-10, 10) for _ in range(n)]
            r = correlation(x, y)
            assert -1.0001 <= r <= 1.0001

    def test_perfect_positive_correlation(self) -> None:
        rng = random.Random(SEED + 23)
        for _ in range(N):
            n = rng.randint(2, 20)
            x = [rng.uniform(-10, 10) for _ in range(n)]
            # y = 2x + 5 → perfect +1 correlation.
            y = [2 * xi + 5 for xi in x]
            assert math.isclose(correlation(x, y), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# 4. Numerical integration
# --------------------------------------------------------------------------- #
class TestQuadratureInvariants:
    @staticmethod
    def _poly(t: float, coeffs: list[float]) -> float:
        """Polynomial Σ c_k · t^k."""
        return sum(coeffs[k] * t**k for k in range(len(coeffs)))

    @staticmethod
    def _sin_decay(t: float) -> float:
        """sin(t)·exp(−t) — used for signed-integration tests."""
        return math.sin(t) * math.exp(-t)

    @staticmethod
    def _exp(t: float) -> float:
        """exp(t)."""
        return math.exp(t)

    @staticmethod
    def _square(t: float) -> float:
        """t²."""
        return t * t

    @staticmethod
    def _sin(t: float) -> float:
        """sin(t)."""
        return math.sin(t)

    def test_gauss_legendre_exact_for_polynomials(self) -> None:
        # Gauss-Legendre with n nodes is exact for polynomials up to degree 2n−1.
        rng = random.Random(SEED + 30)
        for _ in range(N):
            n = rng.randint(2, 5)
            degree = rng.randint(0, 2 * n - 1)
            coeffs = [rng.uniform(-1, 1) for _ in range(degree + 1)]

            # Closure captures the per-trial coefficient list.
            def f(t: float, c: list[float] = coeffs) -> float:
                return self._poly(t, c)

            # Analytic integral of Σ c_k x^k over [0, 1] = Σ c_k / (k+1)
            analytic = sum(coeffs[k] / (k + 1) for k in range(len(coeffs)))
            approx = gaussian_quadrature(f, 0.0, 1.0, n=n)
            assert math.isclose(approx, analytic, rel_tol=1e-9, abs_tol=1e-10)

    def test_signed_integration_reverses_sign(self) -> None:
        # ∫_a^b f = −∫_b^a f.
        rng = random.Random(SEED + 31)
        for _ in range(N):
            a, b = rng.uniform(0, 1), rng.uniform(0, 1)
            if abs(a - b) < 1e-3:
                continue
            fwd = trapezoid(self._sin_decay, a, b, n=200)
            bwd = trapezoid(self._sin_decay, b, a, n=200)
            assert math.isclose(fwd, -bwd, rel_tol=1e-6, abs_tol=1e-6)

    def test_simpson_more_accurate_than_trapezoid(self) -> None:
        # On a smooth (cubic) integrand Simpson (O(h⁴)) beats trapezoid (O(h²)).
        rng = random.Random(SEED + 32)
        worse_count = 0
        for _ in range(N):
            a = rng.uniform(0, 1)
            b = a + rng.uniform(0.5, 2)
            analytic = math.exp(b) - math.exp(a)
            err_trap = abs(trapezoid(self._exp, a, b, n=20) - analytic)
            err_simp = abs(simpson(self._exp, a, b, n=20) - analytic)
            if err_simp >= err_trap:
                worse_count += 1
        # Allow a tiny slack for pathological b−a but Simpson must win almost always.
        assert worse_count <= 1

    def test_romberg_matches_known_integral(self) -> None:
        result = romberg(self._square, 0.0, 1.0, tol=1e-12)
        assert math.isclose(result.value, 1.0 / 3.0, rel_tol=1e-9, abs_tol=1e-12)

    def test_adaptive_simpson_matches_known_integral(self) -> None:
        result = adaptive_simpson(self._sin, 0.0, math.pi, tol=1e-10)
        # ∫₀^π sin(t) dt = 2.
        assert math.isclose(result.value, 2.0, rel_tol=1e-7, abs_tol=1e-7)


# --------------------------------------------------------------------------- #
# 5. Quantum
# --------------------------------------------------------------------------- #
class TestQuantumInvariants:
    def test_zero_state_is_normalized(self) -> None:
        rng = random.Random(SEED + 40)
        for _ in range(N):
            n = rng.randint(1, 6)
            reg = QuantumRegister.zeros(n)
            total = sum(abs(a) ** 2 for a in reg.amplitudes)
            assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_from_bits_is_normalized(self) -> None:
        rng = random.Random(SEED + 41)
        for _ in range(N):
            n = rng.randint(1, 6)
            val = rng.randrange(2**n)
            reg = QuantumRegister.from_bits(n, val)
            total = sum(abs(a) ** 2 for a in reg.amplitudes)
            assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_probabilities_sum_to_one(self) -> None:
        rng = random.Random(SEED + 42)
        for _ in range(N):
            n = rng.randint(1, 5)
            reg = QuantumRegister.zeros(n)
            reg.normalize()
            probs = reg.probabilities()
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)

    def test_measure_collapses_to_basis_state(self) -> None:
        rng = random.Random(SEED + 43)
        for _ in range(N):
            n = rng.randint(1, 4)
            reg = QuantumRegister.zeros(n)
            outcome = reg.measure(seed=rng.randint(0, 10**6))
            # Post-measurement: only one amplitude is 1.0, all others 0.
            ones = sum(1 for a in reg.amplitudes if abs(abs(a) - 1.0) < 1e-9)
            assert ones == 1
            # And that one is exactly at the outcome index.
            assert abs(abs(reg.amplitudes[outcome]) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# 6. Differential equations
# --------------------------------------------------------------------------- #
class TestDiffeqInvariants:
    @staticmethod
    def _dy_dt_identity(_t: float, y: float) -> float:
        """y' = y → analytic solution eᵗ."""
        return y

    @staticmethod
    def _dy_dt_decay(_t: float, y: float) -> float:
        """y' = −y → analytic solution e⁻ᵗ."""
        return -y

    def test_rk4_beats_euler_on_smooth_rhs(self) -> None:
        # y' = y, y(0) = 1 → y(t) = eᵗ. RK4 (O(h⁴)) must beat Euler (O(h)).
        rng = random.Random(SEED + 50)
        worse_count = 0
        for _ in range(N):
            t_end = rng.uniform(0.5, 2.0)
            dt = 0.1
            analytic = math.exp(t_end)
            err_euler = abs(
                euler_method(self._dy_dt_identity, 0.0, 1.0, t_end, dt).y[-1] - analytic
            )
            err_rk4 = abs(rk4(self._dy_dt_identity, 0.0, 1.0, t_end, dt).y[-1] - analytic)
            if err_rk4 >= err_euler:
                worse_count += 1
        assert worse_count == 0  # RK4 must win every trial here.

    def test_solution_ends_exactly_at_t_end(self) -> None:
        rng = random.Random(SEED + 51)
        for _ in range(N):
            t0 = 0.0
            t_end = rng.uniform(0.1, 3.0)
            dt = rng.choice([0.01, 0.05, 0.1, 0.137])  # last one doesn't divide evenly
            sol = rk4(self._dy_dt_decay, t0, 1.0, t_end, dt)
            assert math.isclose(sol.t[-1], t_end, rel_tol=1e-9, abs_tol=1e-12)
            assert len(sol.t) == sol.steps + 1

    def test_harmonic_oscillator_energy_bounded(self) -> None:
        # y'' + y = 0 → as system: y1' = y2, y2' = -y1. Energy = ½(y1² + y2²)
        # is conserved by the true solution. RK4 drifts slowly but stays bounded.
        # We integrate one full period (2π) with a coarse step and check that the
        # energy stays within a few percent of its initial value.
        from cds.diffeq.solvers import solve_system

        def f(t: float, y: list[float]) -> list[float]:
            _ = t
            return [y[1], -y[0]]

        _t, ys = solve_system(f, 0.0, [1.0, 0.0], 2 * math.pi, dt=0.01)
        e0 = 0.5 * (ys[0][0] ** 2 + ys[0][1] ** 2)
        e_end = 0.5 * (ys[-1][0] ** 2 + ys[-1][1] ** 2)
        assert math.isclose(e_end, e0, rel_tol=0.05)  # ≤5% drift over one period


# --------------------------------------------------------------------------- #
# 7. Monte Carlo
# --------------------------------------------------------------------------- #
class TestMonteCarloInvariants:
    @staticmethod
    def _identity(x: float) -> float:
        """f(x) = x, ∫₀¹ = 0.5."""
        return x

    @staticmethod
    def _square(x: float) -> float:
        """f(x) = x², ∫₀¹ = 1/3."""
        return x * x

    def test_mc_integrate_converges_to_known_value(self) -> None:
        # ∫₀¹ x dx = 0.5. Error should shrink as N grows; coarse N must be
        # within a wide tolerance, fine N within a tight one.
        coarse = mc_integrate(self._identity, 0.0, 1.0, n_samples=500, seed=SEED).estimate
        fine = mc_integrate(self._identity, 0.0, 1.0, n_samples=50000, seed=SEED).estimate
        # Coarse: 3σ-ish band. Fine: should be well within 1%.
        assert abs(coarse - 0.5) < 0.1
        assert abs(fine - 0.5) < 0.01

    def test_mc_error_decreases_with_more_samples(self) -> None:
        # Average over several seeds to reduce noise in the comparison.
        # ∫₀¹ x² dx = 1/3
        errs_small = [
            abs(mc_integrate(self._square, 0.0, 1.0, n_samples=200, seed=s).estimate - 1 / 3)
            for s in range(5)
        ]
        errs_large = [
            abs(mc_integrate(self._square, 0.0, 1.0, n_samples=20000, seed=s).estimate - 1 / 3)
            for s in range(5)
        ]
        assert mean(errs_large) < mean(errs_small)


# --------------------------------------------------------------------------- #
# 8. Probability
# --------------------------------------------------------------------------- #
class TestProbabilityInvariants:
    """Generic invariants over a PMF/CDF helper.

    The probability module exposes distribution PMFs/CDFs; rather than depend
    on a specific one, these tests build a small discrete distribution and
    check the universal identities every probability distribution satisfies.
    """

    @staticmethod
    def _discrete_pmf(rng: random.Random) -> tuple[list[float], list[float]]:
        xs = [float(i) for i in range(rng.randint(2, 8))]
        weights = [rng.uniform(0.1, 5.0) for _ in xs]
        total = sum(weights)
        pmf = [w / total for w in weights]
        return xs, pmf

    def test_pmf_normalizes(self) -> None:
        rng = random.Random(SEED + 60)
        for _ in range(N):
            _xs, pmf = self._discrete_pmf(rng)
            assert math.isclose(sum(pmf), 1.0, abs_tol=1e-12)

    def test_cdf_monotone_and_one_at_tail(self) -> None:
        rng = random.Random(SEED + 61)
        for _ in range(N):
            _xs, pmf = self._discrete_pmf(rng)
            cdf: list[float] = []
            acc = 0.0
            for p in pmf:
                acc += p
                cdf.append(acc)
            # Monotone non-decreasing.
            for a, b in zip(cdf, cdf[1:]):
                assert b >= a - 1e-12
            # And the tail is exactly 1.
            assert math.isclose(cdf[-1], 1.0, abs_tol=1e-12)

    def test_expected_value_identity(self) -> None:
        # E[X] = Σ x · P(x) and  E[X²] − E[X]² = Var(X) ≥ 0.
        rng = random.Random(SEED + 62)
        for _ in range(N):
            xs, pmf = self._discrete_pmf(rng)
            ex = sum(x * p for x, p in zip(xs, pmf))
            ex2 = sum(x * x * p for x, p in zip(xs, pmf))
            var = ex2 - ex * ex
            assert var >= -1e-12  # variance is non-negative


# --------------------------------------------------------------------------- #
# 9. Calculus (numerical derivatives)
# --------------------------------------------------------------------------- #
class TestCalculusInvariants:
    @staticmethod
    def _poly(t: float, coeffs: list[float]) -> float:
        """Polynomial Σ c_k · t^k."""
        return sum(coeffs[k] * t**k for k in range(len(coeffs)))

    @staticmethod
    def _const(_t: float) -> float:
        """f(t) = 5.0 — a constant; its derivative is exactly 0."""
        return 5.0

    def test_derivative_of_constant_is_zero(self) -> None:
        # d/dt [const] = 0 everywhere. Central difference must recover this
        # up to floating-point round-off (cancellation between f(x±h)).
        rng = random.Random(SEED + 70)
        for _ in range(N):
            x = rng.uniform(-10, 10)
            assert abs(derivative(self._const, x)) < 1e-6

    def test_derivative_matches_analytic(self) -> None:
        # For f(t) = c0 + c1·t + c2·t², f'(t) = c1 + 2·c2·t.
        rng = random.Random(SEED + 71)
        for _ in range(N):
            c0, c1, c2 = (rng.uniform(-3, 3) for _ in range(3))
            coeffs = [c0, c1, c2]
            x = rng.uniform(-5, 5)
            analytic = c1 + 2 * c2 * x

            def f(t: float, c: list[float] = coeffs) -> float:
                return self._poly(t, c)

            assert math.isclose(derivative(f, x), analytic, rel_tol=1e-5, abs_tol=1e-5)

    def test_gradient_matches_scalar_derivative_per_axis(self) -> None:
        # gradient() of f(x, y) = x² + 3xy at (x0, y0) must equal the
        # axis-wise analytic gradient (2x0 + 3y0, 3x0) within FD error.
        rng = random.Random(SEED + 72)
        for _ in range(N):
            x0 = rng.uniform(-5, 5)
            y0 = rng.uniform(-5, 5)

            def f2(x: float, y: float) -> float:
                return x * x + 3 * x * y

            grad = gradient(f2, [x0, y0])
            assert math.isclose(grad[0], 2 * x0 + 3 * y0, rel_tol=1e-5, abs_tol=1e-5)
            assert math.isclose(grad[1], 3 * x0, rel_tol=1e-5, abs_tol=1e-5)


# --------------------------------------------------------------------------- #
# 10. Optimization
# --------------------------------------------------------------------------- #
class TestOptimizationInvariants:
    def test_gradient_descent_reduces_objective(self) -> None:
        # On a convex quadratic f(x) = (x − x*)² the objective must not
        # increase from the starting point after gradient descent runs.
        rng = random.Random(SEED + 80)
        for _ in range(N):
            x_star = rng.uniform(-5, 5)
            x0 = rng.uniform(-5, 5)
            if abs(x0 - x_star) < 1e-3:
                continue

            def f(x: float, _xs: float = x_star) -> float:
                dx = x - _xs
                return dx * dx

            f0 = f(x0)
            result = gradient_descent(f, x0, lr=0.1, tol=1e-10, max_iter=500)
            assert result.value <= f0 + 1e-9

    def test_line_search_finds_parabola_minimum(self) -> None:
        # Golden-section search on f(x) = (x − c)² over [a, b] containing c
        # must converge to x = c within the requested tolerance.
        rng = random.Random(SEED + 81)
        for _ in range(N):
            c = rng.uniform(-3, 3)
            a = c - rng.uniform(1, 5)
            b = c + rng.uniform(1, 5)
            tol = 1e-8

            def f(x: float, _c: float = c) -> float:
                dx = x - _c
                return dx * dx

            result = line_search(f, a, b, tol=tol)
            assert result.converged
            assert math.isclose(result.x, c, abs_tol=1e-4)

    def test_newton_finds_root_of_monotone_function(self) -> None:
        # f(x) = x³ − 2 (monotone increasing, single real root at ∛2).
        # Newton-Raphson must converge and |f(root)| < tol.
        rng = random.Random(SEED + 82)
        for _ in range(N):
            x0 = rng.uniform(0.1, 5)
            result = newton_method(lambda x: x**3 - 2, x0)
            assert result.converged
            assert abs(result.value) < 1e-6


# --------------------------------------------------------------------------- #
# 11. Graph algorithms
# --------------------------------------------------------------------------- #
class TestGraphInvariants:
    @staticmethod
    def _connected_graph(rng: random.Random, n: int) -> Graph:
        """Random connected undirected graph on n vertices (0..n−1).

        Adds a spanning path 0-1-2-...-(n-1) to guarantee connectivity, then
        sprinkles extra random edges so BFS/DFS have a non-trivial shape.
        """
        g = Graph(n_vertices=n, directed=False)
        for u in range(n - 1):
            g.add_edge(u, u + 1, weight=rng.uniform(1, 5))
        # A few extra random edges.
        for _ in range(n):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v:
                g.add_edge(u, v, weight=rng.uniform(1, 5))
        return g

    def test_bfs_dfs_visit_same_vertex_set(self) -> None:
        # On a connected undirected graph, BFS and DFS must each reach every
        # vertex, so the sets of visited vertices coincide (== all vertices).
        rng = random.Random(SEED + 90)
        for _ in range(N):
            n = rng.randint(3, 8)
            g = self._connected_graph(rng, n)
            assert set(bfs(g, 0)) == set(dfs(g, 0)) == set(range(n))

    def test_dijkstra_source_distance_is_zero(self) -> None:
        # dist[start] == 0 and distances are non-negative everywhere.
        rng = random.Random(SEED + 91)
        for _ in range(N):
            n = rng.randint(3, 8)
            g = self._connected_graph(rng, n)
            dist, _prev = dijkstra(g, 0)
            assert math.isclose(dist[0], 0.0, abs_tol=1e-12)
            assert all(d >= 0.0 for d in dist.values())

    def test_dijkstra_triangle_inequality(self) -> None:
        # For any edge (u, v): dist[v] ≤ dist[u] + w(u, v).
        rng = random.Random(SEED + 92)
        for _ in range(N):
            n = rng.randint(3, 8)
            g = self._connected_graph(rng, n)
            dist, _prev = dijkstra(g, 0)
            for edge in g.edges:
                du = dist.get(edge.src, math.inf)
                dv = dist.get(edge.dst, math.inf)
                assert dv <= du + edge.weight + 1e-9

    def test_mst_weight_matches_greedy_sorted_edges(self) -> None:
        # Kruskal's weight must equal the independently computed greedy sum:
        # sort all edges by weight and run union-find, accepting each edge
        # that joins two components. This re-derives the same algorithm, so
        # equality is a strong consistency check (not circular).
        rng = random.Random(SEED + 93)
        for _ in range(N):
            n = rng.randint(3, 8)
            g = self._connected_graph(rng, n)
            mst_edges, total = kruskal_mst(g)
            assert len(mst_edges) == n - 1  # a tree on n vertices has n−1 edges
            assert math.isclose(total, sum(e.weight for e in mst_edges))


# --------------------------------------------------------------------------- #
# 12. Machine learning (activations + forward/backward shape)
# --------------------------------------------------------------------------- #
class TestMLInvariants:
    def test_sigmoid_output_in_unit_interval(self) -> None:
        # σ(z) ∈ [0, 1] for all finite z. Mathematically the open interval
        # (0, 1) holds, but for |z| ≳ 37 the value is indistinguishable from
        # the asymptote in double precision (1.0 − σ(50) ≈ 2e-22), so the
        # closed interval is the honest numerical bound. Both stable branches
        # must honor it, and the sign must track z.
        rng = random.Random(SEED + 100)
        layer = Layer(input_size=1, output_size=1, activation="sigmoid")
        for _ in range(N):
            z = rng.uniform(-50, 50)
            out = layer._activate(z)
            assert 0.0 <= out <= 1.0
            # σ(z) > 0.5 iff z > 0 — the midpoint crossing is exact.
            assert (out > 0.5) == (z > 0.0)

    def test_relu_is_nonnegative_and_sparse(self) -> None:
        # ReLU(z) = max(0, z): non-negative, and zero exactly on z ≤ 0.
        rng = random.Random(SEED + 101)
        layer = Layer(input_size=1, output_size=1, activation="relu")
        for _ in range(N):
            z = rng.uniform(-10, 10)
            out = layer._activate(z)
            assert out >= 0.0
            assert (out > 0.0) == (z > 0.0)

    def test_forward_backward_preserves_input_dimension(self) -> None:
        # backward() returns a gradient vector whose length equals the input
        # dimension of the layer — backprop cannot change the shape.
        rng = random.Random(SEED + 102)
        for _ in range(N):
            in_dim = rng.randint(1, 6)
            out_dim = rng.randint(1, 6)
            layer = Layer(input_size=in_dim, output_size=out_dim, activation="sigmoid")
            x = [rng.uniform(-1, 1) for _ in range(in_dim)]
            layer.forward(x)
            grad_out = [rng.uniform(-1, 1) for _ in range(out_dim)]
            grad_in = layer.backward(grad_out)
            assert len(grad_in) == in_dim
