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
    def test_pi_within_tolerance(self):
        result = estimate_pi(n_samples=50_000, seed=42)
        assert abs(result.estimate - math.pi) < 0.1

    def test_standard_error_positive(self):
        result = estimate_pi(n_samples=1000, seed=42)
        assert result.std_error > 0

    def test_samples_recorded(self):
        result = estimate_pi(n_samples=500, seed=42)
        assert result.samples == 500

    def test_reproducible_with_seed(self):
        r1 = estimate_pi(n_samples=1000, seed=123)
        r2 = estimate_pi(n_samples=1000, seed=123)
        assert r1.estimate == r2.estimate


class TestMCIntegrate:
    def test_integrate_x_squared(self):
        # ∫₀¹ x² dx = 1/3
        result = mc_integrate(lambda x: x**2, 0, 1, n_samples=50_000, seed=42)
        assert abs(result.estimate - 1 / 3) < 0.02

    def test_integrate_sine(self):
        # ∫₀^π sin(x) dx = 2
        result = mc_integrate(math.sin, 0, math.pi, n_samples=50_000, seed=42)
        assert abs(result.estimate - 2.0) < 0.1

    def test_standard_error(self):
        result = mc_integrate(lambda x: x, 0, 1, n_samples=10_000, seed=42)
        assert result.std_error > 0


class TestRandomWalk1D:
    def test_starts_at_zero(self):
        walk = random_walk_1d(10, seed=42)
        assert walk[0] == 0.0

    def test_correct_length(self):
        walk = random_walk_1d(100, seed=42)
        assert len(walk) == 101

    def test_step_size(self):
        walk = random_walk_1d(1, step_size=2.5, seed=42)
        assert abs(walk[1]) == 2.5

    def test_reproducible(self):
        w1 = random_walk_1d(50, seed=99)
        w2 = random_walk_1d(50, seed=99)
        assert w1 == w2


class TestRandomWalk2D:
    def test_starts_at_origin(self):
        walk = random_walk_2d(10, seed=42)
        assert walk[0] == (0.0, 0.0)

    def test_correct_length(self):
        walk = random_walk_2d(50, seed=42)
        assert len(walk) == 51

    def test_step_distance(self):
        walk = random_walk_2d(1, step_size=1.0, seed=42)
        x, y = walk[1]
        dist = math.sqrt(x**2 + y**2)
        assert abs(dist - 1.0) < 1e-10


class TestBuffonNeedle:
    def test_pi_estimate(self):
        result = buffon_needle(
            needle_length=1.0,
            line_spacing=2.0,
            n_throws=50_000,
            seed=42,
        )
        assert abs(result.estimate - math.pi) < 0.2

    def test_needle_too_long_raises(self):
        with pytest.raises(ValueError, match="shorter"):
            buffon_needle(needle_length=3.0, line_spacing=2.0)

    def test_reproducible(self):
        r1 = buffon_needle(n_throws=1000, seed=42)
        r2 = buffon_needle(n_throws=1000, seed=42)
        assert r1.estimate == r2.estimate
