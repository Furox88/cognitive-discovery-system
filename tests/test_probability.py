"""Tests for probability distributions module."""

import math

import pytest

from cds.probability.distributions import (
    binomial_pmf,
    exponential_pdf,
    gaussian_pdf,
    poisson_pmf,
    uniform_pdf,
    uniform_sample,
)

# --- Gaussian ---


def test_gaussian_peak():
    # PDF peaks at mu
    assert gaussian_pdf(0.0) > gaussian_pdf(1.0)


def test_gaussian_symmetric():
    assert abs(gaussian_pdf(-1.0) - gaussian_pdf(1.0)) < 1e-12


def test_gaussian_custom_params():
    val = gaussian_pdf(5.0, mu=5.0, sigma=2.0)
    peak = 1 / (2 * math.sqrt(2 * math.pi))
    assert abs(val - peak) < 1e-9


def test_gaussian_invalid_sigma():
    with pytest.raises(ValueError):
        gaussian_pdf(0.0, sigma=0.0)


def test_gaussian_narrow():
    # smaller sigma -> taller peak
    narrow = gaussian_pdf(0.0, sigma=0.1)
    wide = gaussian_pdf(0.0, sigma=10.0)
    assert narrow > wide


# --- Uniform ---


def test_uniform_inside():
    assert abs(uniform_pdf(0.5) - 1.0) < 1e-9


def test_uniform_outside():
    assert uniform_pdf(-1.0) == 0.0
    assert uniform_pdf(2.0) == 0.0


def test_uniform_custom_range():
    val = uniform_pdf(5.0, a=0, b=10)
    assert abs(val - 0.1) < 1e-9


def test_uniform_invalid():
    with pytest.raises(ValueError):
        uniform_pdf(0.5, a=1, b=0)


# --- Exponential ---


def test_exponential_at_zero():
    assert abs(exponential_pdf(0.0, lam=2.0) - 2.0) < 1e-9


def test_exponential_negative():
    assert exponential_pdf(-1.0) == 0.0


def test_exponential_decreasing():
    assert exponential_pdf(1.0) > exponential_pdf(2.0)


def test_exponential_invalid_lambda():
    with pytest.raises(ValueError):
        exponential_pdf(1.0, lam=-1.0)


# --- Binomial ---


def test_binomial_fair_coin():
    # P(3 heads in 5 flips) = C(5,3) * 0.5^5
    val = binomial_pmf(3, 5, 0.5)
    expected = math.comb(5, 3) * 0.5**5
    assert abs(val - expected) < 1e-12


def test_binomial_certain():
    # p=1, k=n => probability 1
    assert abs(binomial_pmf(5, 5, 1.0) - 1.0) < 1e-12


def test_binomial_impossible():
    # p=0, k>0 => probability 0
    assert binomial_pmf(1, 5, 0.0) == 0.0


def test_binomial_out_of_range():
    assert binomial_pmf(6, 5, 0.5) == 0.0
    assert binomial_pmf(-1, 5, 0.5) == 0.0


def test_binomial_invalid_p():
    with pytest.raises(ValueError):
        binomial_pmf(1, 5, 1.5)


def test_binomial_sums_to_one():
    total = sum(binomial_pmf(k, 10, 0.3) for k in range(11))
    assert abs(total - 1.0) < 1e-9


# --- Poisson ---


def test_poisson_basic():
    # P(k=0, lam=1) = e^-1
    val = poisson_pmf(0, 1.0)
    assert abs(val - math.exp(-1)) < 1e-12


def test_poisson_mode():
    # mode of Poisson(5) is near 5
    probs = [poisson_pmf(k, 5.0) for k in range(15)]
    mode = probs.index(max(probs))
    assert mode in (4, 5)


def test_poisson_negative_k():
    assert poisson_pmf(-1, 3.0) == 0.0


def test_poisson_invalid_lambda():
    with pytest.raises(ValueError):
        poisson_pmf(0, -1.0)


def test_poisson_sums_approx_one():
    total = sum(poisson_pmf(k, 3.0) for k in range(50))
    assert abs(total - 1.0) < 1e-9


# --- Sampling ---


def test_uniform_sample_count():
    samples = uniform_sample(0, 1, 100, seed=42)
    assert len(samples) == 100


def test_uniform_sample_range():
    samples = uniform_sample(2, 5, 1000, seed=7)
    assert all(2 <= s <= 5 for s in samples)


def test_uniform_sample_reproducible():
    s1 = uniform_sample(0, 1, 10, seed=99)
    s2 = uniform_sample(0, 1, 10, seed=99)
    assert s1 == s2
