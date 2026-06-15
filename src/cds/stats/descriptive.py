"""Descriptive statistics — mean, median, variance, stdev."""
from __future__ import annotations

import math


def mean(data: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        data: List of numeric values.

    Returns:
        Arithmetic mean (sum / N).
    """
    if not data:
        return 0.0
    return sum(data) / len(data)


def median(data: list[float]) -> float:
    """Calculate the median (middle value) of a list of numbers.

    Args:
        data: List of numeric values.

    Returns:
        Median value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    return float(sorted_data[mid])


def variance(data: list[float], ddof: int = 1) -> float:
    """Calculate the sample variance of a list of numbers.

    Args:
        data: List of numeric values.
        ddof: Delta Degrees of Freedom (1 for sample, 0 for population).

    Returns:
        Sample or population variance.
    """
    if len(data) <= ddof:
        return 0.0
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def stdev(data: list[float], ddof: int = 1) -> float:
    """Calculate the standard deviation of a list of numbers.

    Args:
        data: List of numeric values.
        ddof: Delta Degrees of Freedom.

    Returns:
        Standard deviation.
    """
    return math.sqrt(variance(data, ddof))


def correlation(x: list[float], y: list[float]) -> float:
    """Calculate the Pearson correlation coefficient between two lists.

    Args:
        x: first list of values
        y: second list of values

    Returns:
        Pearson correlation coefficient.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    
    return num / den if den > 1e-15 else 0.0
