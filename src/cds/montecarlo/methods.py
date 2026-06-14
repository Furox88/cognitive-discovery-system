"""Monte Carlo simulation methods.

References:
    - Metropolis, N. & Ulam, S. (1949). The Monte Carlo Method.
    - Buffon, G.L. (1777). Essai d'arithmétique morale.
    - Robert, C.P. & Casella, G. Monte Carlo Statistical Methods (2nd ed.)
"""
from __future__ import annotations

import math
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class MCResult:
    """Result of a Monte Carlo estimation."""

    estimate: float
    samples: int
    std_error: float


def _pi_worker(samples_seed: tuple[int, int | None]) -> int:
    """Worker function for parallel pi estimation."""
    samples, seed = samples_seed
    if seed is not None:
        random.seed(seed)
    inside = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside

def estimate_pi(n_samples: int = 100_000, seed: int | None = None) -> MCResult:
    """Estimate π using the unit-circle method (Parallelized).

    Throw random points into the unit square [0,1]×[0,1].
    Fraction inside the quarter-circle ≈ π/4.

    Args:
        n_samples: number of random points
        seed: optional random seed
    """
    if n_samples <= 0:
        return MCResult(0.0, n_samples, 0.0)

    cores = min(multiprocessing.cpu_count(), n_samples)
    chunk_size = n_samples // cores
    chunks = [chunk_size] * cores
    chunks[-1] += n_samples - sum(chunks) # add remainder to last chunk
    
    if seed is None:
        import sys, os
        seed = int.from_bytes(os.urandom(4), sys.byteorder)

    seeds = [seed + i for i in range(cores)]
    tasks = list(zip(chunks, seeds))

    inside = 0
    with ProcessPoolExecutor(max_workers=cores) as executor:
        for result in executor.map(_pi_worker, tasks):
            inside += result

    p = inside / n_samples
    estimate = 4.0 * p
    se = 4.0 * math.sqrt(p * (1 - p) / n_samples) if n_samples > 1 else 0.0
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)


def mc_integrate(
    f: callable,
    a: float,
    b: float,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Monte Carlo integration of f over [a, b].

    E[f(X)] * (b-a) where X ~ Uniform(a, b).

    Args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n_samples: number of random evaluations
        seed: optional random seed
    """
    if seed is not None:
        random.seed(seed)
    total = 0.0
    total_sq = 0.0
    width = b - a
    for _ in range(n_samples):
        x = a + random.random() * width
        val = f(x)
        total += val
        total_sq += val * val

    mean_val = total / n_samples
    estimate = mean_val * width
    var = (total_sq / n_samples - mean_val ** 2) if n_samples > 1 else 0.0
    se = width * math.sqrt(var / n_samples) if var > 0 else 0.0
    return MCResult(estimate=estimate, samples=n_samples, std_error=se)


def random_walk_1d(
    steps: int, step_size: float = 1.0, seed: int | None = None,
) -> list[float]:
    """1D symmetric random walk.

    At each step, move +step_size or -step_size with equal probability.

    Args:
        steps: number of steps
        step_size: size of each step
        seed: optional random seed

    Returns:
        list of positions at each step (length = steps + 1)
    """
    if seed is not None:
        random.seed(seed)
    positions = [0.0]
    pos = 0.0
    for _ in range(steps):
        pos += step_size if random.random() < 0.5 else -step_size
        positions.append(pos)
    return positions


def random_walk_2d(
    steps: int, step_size: float = 1.0, seed: int | None = None,
) -> list[tuple[float, float]]:
    """2D random walk on a plane.

    At each step, move in a random direction (uniform angle).

    Args:
        steps: number of steps
        step_size: size of each step
        seed: optional random seed

    Returns:
        list of (x, y) positions at each step (length = steps + 1)
    """
    if seed is not None:
        random.seed(seed)
    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    for _ in range(steps):
        angle = random.uniform(0, 2 * math.pi)
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)
        positions.append((x, y))
    return positions


def buffon_needle(
    needle_length: float = 1.0,
    line_spacing: float = 2.0,
    n_throws: int = 100_000,
    seed: int | None = None,
) -> MCResult:
    """Buffon's needle experiment for estimating π.

    Drop a needle of length L onto parallel lines spaced D apart.
    P(crossing) = 2L / (πD), so π ≈ 2L / (D * P(crossing)).

    Reference: Buffon (1777).

    Args:
        needle_length: length of the needle (must be <= line_spacing)
        line_spacing: distance between parallel lines
        n_throws: number of needle drops
        seed: optional random seed

    Raises:
        ValueError: if needle_length > line_spacing
    """
    if needle_length > line_spacing:
        raise ValueError("needle must be shorter than line spacing")
    if seed is not None:
        random.seed(seed)

    crossings = 0
    for _ in range(n_throws):
        center = random.uniform(0, line_spacing / 2)
        angle = random.uniform(0, math.pi)
        tip = (needle_length / 2) * math.sin(angle)
        if tip >= center:
            crossings += 1

    if crossings == 0:
        return MCResult(estimate=0.0, samples=n_throws, std_error=0.0)

    p = crossings / n_throws
    estimate = (2 * needle_length) / (line_spacing * p)
    se_p = math.sqrt(p * (1 - p) / n_throws)
    se = (2 * needle_length * se_p) / (line_spacing * p * p) if p > 0 else 0.0
    return MCResult(estimate=estimate, samples=n_throws, std_error=se)
