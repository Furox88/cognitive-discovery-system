"""Monte Carlo methods — estimation, integration, random walks."""

from cds.montecarlo.methods import (
    buffon_needle,
    estimate_pi,
    mc_integrate,
    random_walk_1d,
    random_walk_2d,
)

__all__ = [
    "estimate_pi",
    "mc_integrate",
    "random_walk_1d",
    "random_walk_2d",
    "buffon_needle",
]
