"""Common probability distributions — pure Python."""

from __future__ import annotations

import math
import random


def gaussian_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian (normal) probability density function.

    Args:
        x: point to evaluate
        mu: mean
        sigma: standard deviation

    Raises:
        ValueError: if sigma <= 0
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coeff = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def uniform_pdf(
    x: float,
    a: float = 0.0,
    b: float = 1.0,
) -> float:
    """Uniform distribution PDF on [a, b].

    Args:
        x: point to evaluate
        a: lower bound
        b: upper bound

    Raises:
        ValueError: if a >= b
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if a <= x <= b:
        return 1 / (b - a)
    return 0.0


def exponential_pdf(x: float, lam: float = 1.0) -> float:
    """Exponential distribution PDF.

    Args:
        x: point to evaluate (must be >= 0)
        lam: rate parameter (lambda)

    Raises:
        ValueError: if lam <= 0
    """
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial distribution probability mass function.

    P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    Args:
        k: number of successes
        n: number of trials
        p: probability of success per trial

    Raises:
        ValueError: if parameters are invalid
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if k < 0 or k > n:
        return 0.0
    coeff = math.comb(n, k)
    return coeff * (p**k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson distribution probability mass function.

    P(X=k) = (lambda^k * e^-lambda) / k!

    Args:
        k: number of events
        lam: expected rate (lambda)

    Raises:
        ValueError: if lam < 0 or k < 0
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if k < 0:
        return 0.0
    return (lam**k) * math.exp(-lam) / math.factorial(k)


def geometric_pmf(k: int, p: float) -> float:
    """Geometric PMF (number of trials until first success, ``k >= 1``).

    ``P(X=k) = (1-p)^{k-1} * p``.
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    if k < 1:
        return 0.0
    return ((1.0 - p) ** (k - 1)) * p


def gaussian_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian CDF via ``math.erf`` (accurate enough for teaching/numerics)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def uniform_cdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Uniform CDF on ``[a, b]``."""
    if a >= b:
        raise ValueError("a must be less than b")
    if x < a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def exponential_cdf(x: float, lam: float = 1.0) -> float:
    """Exponential CDF: ``1 - exp(-λx)`` for ``x >= 0``."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-lam * x)


def uniform_sample(
    a: float,
    b: float,
    n: int,
    seed: int | None = None,
) -> list[float]:
    """Generate n uniform random samples from [a, b].

    Args:
        a: lower bound
        b: upper bound
        n: number of samples
        seed: optional random seed
    """
    if a >= b:
        raise ValueError("a must be less than b")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    return [rng.uniform(a, b) for _ in range(n)]


def gaussian_sample(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` i.i.d. Gaussian samples (Box–Muller)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    out: list[float] = []
    while len(out) < n:
        u1 = rng.random()
        u2 = rng.random()
        # Avoid log(0)
        u1 = max(u1, 1e-12)
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(mu + sigma * r * math.cos(theta))
        if len(out) < n:
            out.append(mu + sigma * r * math.sin(theta))
    return out[:n]


def exponential_sample(
    n: int,
    lam: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Generate ``n`` exponential samples via inverse transform."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        u = max(rng.random(), 1e-12)
        out.append(-math.log(u) / lam)
    return out


def poisson_sample(
    n: int,
    lam: float = 1.0,
    seed: int | None = None,
) -> list[int]:
    """Generate ``n`` Poisson samples (Knuth algorithm)."""
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    if lam == 0.0:
        return [0] * n
    out: list[int] = []
    l_exp = math.exp(-lam)
    for _ in range(n):
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= l_exp:
                out.append(k - 1)
                break
    return out
