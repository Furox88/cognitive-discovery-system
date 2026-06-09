"""Data transformations."""
from __future__ import annotations

from cds.stats.descriptive import mean, stdev


def normalize(data: list[float]) -> list[float]:
    """Min-max normalization to [0, 1]."""
    lo, hi = min(data), max(data)
    rng = hi - lo
    if rng == 0:
        return [0.0] * len(data)
    return [(x - lo) / rng for x in data]


def z_score(data: list[float]) -> list[float]:
    """Standardize to mean=0, std=1."""
    m = mean(data)
    s = stdev(data)
    if s == 0:
        return [0.0] * len(data)
    return [(x - m) / s for x in data]


def moving_average(data: list[float], window: int = 3) -> list[float]:
    if window < 1:
        raise ValueError("window must be >= 1")
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result
