"""Deterministic numerical quadrature — Newton-Cotes, Romberg, Gauss-Legendre.

Complements :mod:`cds.montecarlo` (stochastic integration) and :mod:`cds.diffeq`
(ODE integration) with classical deterministic integration rules. Includes
1-D rules (trapezoid, Simpson 1/3 and 3/8, Romberg, Gauss-Legendre, adaptive
Simpson) and tensor-product 2-D rules over rectangular domains.
"""

from cds.numerical_integration.quadrature import (
    QuadratureResult,
    adaptive_simpson,
    gaussian_quadrature,
    gaussian_quadrature_2d,
    romberg,
    simpson,
    simpson_2d,
    simpson_38,
    trapezoid,
)

__all__ = [
    "QuadratureResult",
    "trapezoid",
    "simpson",
    "simpson_38",
    "simpson_2d",
    "romberg",
    "gaussian_quadrature",
    "gaussian_quadrature_2d",
    "adaptive_simpson",
]
