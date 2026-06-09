"""Tests for stats module."""
import math

import pytest

from cds.stats.descriptive import correlation, mean, median, stdev, variance
from cds.stats.regression import linear_regression


def test_mean():
    assert mean([1, 2, 3, 4, 5]) == 3.0


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_variance():
    v = variance([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(v - 4.571) < 0.01


def test_stdev():
    s = stdev([2, 4, 4, 4, 5, 5, 7, 9])
    assert abs(s - math.sqrt(4.571)) < 0.01


def test_correlation_perfect():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    assert abs(correlation(x, y) - 1.0) < 1e-9


def test_correlation_negative():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    assert abs(correlation(x, y) - (-1.0)) < 1e-9


def test_linear_regression_perfect():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1
    result = linear_regression(x, y)
    assert abs(result.slope - 2.0) < 1e-9
    assert abs(result.intercept - 1.0) < 1e-9
    assert abs(result.r_squared - 1.0) < 1e-9


def test_regression_predict():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    r = linear_regression(x, y)
    assert abs(r.predict(10) - 20.0) < 1e-6
