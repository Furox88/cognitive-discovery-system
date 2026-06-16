"""Simple linear regression via least squares."""
from __future__ import annotations

from dataclasses import dataclass

from cds.stats.descriptive import mean


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


def linear_regression(x: list[float], y: list[float]) -> RegressionResult:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need matching lists with at least 2 points")
    mx = mean(x)
    my = mean(y)

    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    if den == 0:
        raise ValueError("all x values are identical")

    slope = num / den
    intercept = my - slope * mx

    # r-squared
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(slope=slope, intercept=intercept, r_squared=r_sq)
