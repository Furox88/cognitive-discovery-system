"""Basic descriptive statistics — no numpy needed."""
from __future__ import annotations

import math


def mean(data: list[float]) -> float:
    if not data:
        raise ValueError("empty dataset")
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    if not data:
        raise ValueError("empty dataset")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def variance(data: list[float], ddof: int = 1) -> float:
    """Sample variance (ddof=1) or population variance (ddof=0)."""
    if len(data) < 2:
        raise ValueError("need at least 2 values")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def stdev(data: list[float], ddof: int = 1) -> float:
    return math.sqrt(variance(data, ddof))


def correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points")
    mx, my = mean(x), mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)
