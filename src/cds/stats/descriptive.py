"""Descriptive statistics — mean, median, variance, stdev."""
from __future__ import annotations

import math


def mean(data: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        data: List of numeric values.

    Returns:
        Arithmetic mean (sum / N).
        
    Raises:
        ValueError: if data is empty.
    """
    if not data:
        raise ValueError("mean requires at least one data point")
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
        
    Raises:
        ValueError: if data size is <= ddof.
    """
    if len(data) <= ddof:
        raise ValueError(f"variance requires more than {ddof} data points")
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
        
    Raises:
        ValueError: if lengths mismatch or lists are too short.
    """
    if len(x) != len(y):
        raise ValueError("lists must be the same length")
    if len(x) < 2:
        raise ValueError("correlation requires at least two data points")
    
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    
    return num / den if den > 1e-15 else 0.0
