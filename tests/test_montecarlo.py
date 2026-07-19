"""Tests for Monte Carlo methods."""

import math

import pytest

from cds.montecarlo import (
    buffon_needle,
    estimate_pi,
    mc_integrate,
    random_walk_1d,
    random_walk_2d,
)


class TestEstimatePi:
    def test_pi_within_tolerance(self) -> None:
        result = estimate_pi(n_samples=50_000, seed=42)
        assert abs(result.estimate - math.pi) < 0.1

    def test_standard_error_positive(self) -> None:
        result = estimate_pi(n_samples=1000, seed=42)
        assert result.std_error > 0

    def test_samples_recorded(self) -> None:
        result = estimate_pi(n_samples=500, seed=42)
        assert result.samples == 500

    def test_reproducible_with_seed(self) -> None:
        r1 = estimate_pi(n_samples=1000, seed=123)
        r2 = estimate_pi(n_samples=1000, seed=123)
        assert r1.estimate == r2.estimate


class TestMCIntegrate:
    def test_integrate_x_squared(self) -> None:
        # ∫₀¹ x² dx = 1/3
        result = mc_integrate(lambda x: x**2, 0, 1, n_samples=50_000, seed=42)
        assert abs(result.estimate - 1 / 3) < 0.02

    def test_integrate_sine(self) -> None:
        # ∫₀^π sin(x) dx = 2
        result = mc_integrate(math.sin, 0, math.pi, n_samples=50_000, seed=42)
        assert abs(result.estimate - 2.0) < 0.1

    def test_standard_error(self) -> None:
        result = mc_integrate(lambda x: x, 0, 1, n_samples=10_000, seed=42)
        assert result.std_error > 0


class TestRandomWalk1D:
    def test_starts_at_zero(self) -> None:
        walk = random_walk_1d(10, seed=42)
        assert walk[0] == 0.0

    def test_correct_length(self) -> None:
        walk = random_walk_1d(100, seed=42)
        assert len(walk) == 101

    def test_step_size(self) -> None:
        walk = random_walk_1d(1, step_size=2.5, seed=42)
        assert abs(walk[1]) == 2.5

    def test_reproducible(self) -> None:
        w1 = random_walk_1d(50, seed=99)
        w2 = random_walk_1d(50, seed=99)
        assert w1 == w2


class TestRandomWalk2D:
    def test_starts_at_origin(self) -> None:
        walk = random_walk_2d(10, seed=42)
        assert walk[0] == (0.0, 0.0)

    def test_correct_length(self) -> None:
        walk = random_walk_2d(50, seed=42)
        assert len(walk) == 51

    def test_step_distance(self) -> None:
        walk = random_walk_2d(1, step_size=1.0, seed=42)
        x, y = walk[1]
        dist = math.hypot(x, y)
        assert abs(dist - 1.0) < 1e-10


class TestBuffonNeedle:
    def test_pi_estimate(self) -> None:
        result = buffon_needle(
            needle_length=1.0,
            line_spacing=2.0,
            n_throws=50_000,
            seed=42,
        )
        assert abs(result.estimate - math.pi) < 0.2

    def test_needle_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="shorter"):
            buffon_needle(needle_length=3.0, line_spacing=2.0)

    def test_reproducible(self) -> None:
        r1 = buffon_needle(n_throws=1000, seed=42)
        r2 = buffon_needle(n_throws=1000, seed=42)
        assert r1.estimate == r2.estimate


def test_mc_expectation() -> None:
    from cds.montecarlo import mc_expectation

    # E[x] on [0,1] ≈ 0.5
    r = mc_expectation(lambda x: x, n_samples=20_000, a=0.0, b=1.0, seed=1)
    assert abs(r.estimate - 0.5) < 0.05


def test_hit_or_miss_unit_disk() -> None:
    import math

    from cds.montecarlo import hit_or_miss

    r = hit_or_miss(
        lambda x, y: x * x + y * y <= 1.0,
        (-1.0, 1.0),
        (-1.0, 1.0),
        n_samples=50_000,
        seed=2,
    )
    assert abs(r.estimate - math.pi) < 0.15


def test_mc_new_errors() -> None:
    from cds.montecarlo import hit_or_miss, mc_expectation

    with pytest.raises(ValueError):
        mc_expectation(lambda x: x, n_samples=0)
    with pytest.raises(ValueError):
        hit_or_miss(lambda x, y: True, (1.0, 0.0), (0.0, 1.0), n_samples=10)


def test_mc_expectation_a_ge_b() -> None:
    from cds.montecarlo import mc_expectation

    with pytest.raises(ValueError):
        mc_expectation(lambda x: x, n_samples=10, a=1.0, b=0.0)


def test_hit_or_miss_n_samples() -> None:
    from cds.montecarlo import hit_or_miss

    with pytest.raises(ValueError):
        hit_or_miss(lambda x, y: True, (0.0, 1.0), (0.0, 1.0), n_samples=0)
