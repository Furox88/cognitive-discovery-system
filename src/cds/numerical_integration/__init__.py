"""Deterministic numerical quadrature — Newton-Cotes, Romberg, Gauss-Legendre.

Complements :mod:`cds.montecarlo` (stochastic integration) and :mod:`cds.diffeq`
(ODE integration) with classical deterministic integration rules.
"""
from cds.numerical_integration.quadrature import (
    QuadratureResult,
    adaptive_simpson,
    gaussian_quadrature,
    romberg,
    simpson,
    simpson_38,
    trapezoid,
)

__all__ = [
    "QuadratureResult",
    "trapezoid",
    "simpson",
    "simpson_38",
    "romberg",
    "gaussian_quadrature",
    "adaptive_simpson",
]
