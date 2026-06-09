"""Statistical analysis tools."""
from cds.stats.descriptive import correlation, mean, median, stdev, variance
from cds.stats.regression import RegressionResult, linear_regression

__all__ = [
    "mean", "median", "stdev", "variance", "correlation",
    "linear_regression", "RegressionResult",
]
