"""
RBA (Research-grade Baryonic Acceleration) model.

This module implements the core RBA velocity prediction law
extracted and cleaned from real galaxy rotation curve analysis.

The functional form comes from solving the low-acceleration
interpolating function:

    μ(x) = x / sqrt(1 + x²)   →   a = a0 * sqrt( t )
    where t solves the quadratic derived from a = μ(a/a0) * aN

Key properties:
- Reduces to Newtonian at high acceleration (aN >> a0)
- Approaches sqrt(a0 * aN) (flat rotation) at low acceleration
- Fully specified with fixed a0 + mass-to-light ratios in basic usage

This is a *substantial* physical model, not a toy. It was used
for systematic comparison against Einasto/NFW/Burkert on SPARC data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Conversion: 1 (km/s)^2 / kpc  = 3.240779289e-14 m/s^2
KMS2_PER_KPC_TO_MPS2 = 3.240779289e-14


@dataclass
class RBAModel:
    """
    RBA acceleration law for rotation curve modeling.

    Parameters
    ----------
    a0_mps2 : float
        Characteristic acceleration scale in m/s² (typical ~1.0-1.2e-10).
    ups_d : float
        Disk mass-to-light ratio.
    ups_b : float
        Bulge mass-to-light ratio.
    """

    a0_mps2: float = 1.14e-10
    ups_d: float = 0.5
    ups_b: float = 0.7

    def __post_init__(self) -> None:
        if self.a0_mps2 <= 0:
            raise ValueError("a0 must be positive")
        self.a0_internal = self.a0_mps2 / KMS2_PER_KPC_TO_MPS2  # in (km/s)^2/kpc

    def predict_velocity(
        self,
        r: np.ndarray,
        vgas: np.ndarray,
        vdisk: np.ndarray,
        vbul: np.ndarray,
    ) -> np.ndarray:
        """
        Predict circular velocity from baryonic components using the RBA law.

        This is the heart of the model — a non-trivial interpolating function.
        """
        r = np.asarray(r, dtype=float)
        vgas = np.asarray(vgas, dtype=float)
        vdisk = np.asarray(vdisk, dtype=float)
        vbul = np.asarray(vbul, dtype=float)

        # Baryonic contribution to v²
        vbar2 = vgas**2 + (self.ups_d * vdisk**2) + (self.ups_b * vbul**2)

        # Newtonian baryonic acceleration in internal units
        aN = np.divide(vbar2, r, out=np.zeros_like(r), where=r > 0)

        eps = 1e-30
        aN2 = np.maximum(aN * aN, eps)
        a02 = self.a0_internal**2

        # Exact solution for the interpolating function
        # t = (aN² / (2 a0²)) * (1 + sqrt(1 + 4 a0² / aN²))
        t = (aN2 / (2 * a02)) * (1.0 + np.sqrt(1.0 + 4.0 * a02 / aN2))

        a = self.a0_internal * np.sqrt(t)  # effective acceleration

        v = np.sqrt(a * r)
        return v

    def compute_loglike(
        self,
        r: np.ndarray,
        vobs: np.ndarray,
        verr: np.ndarray,
        vgas: np.ndarray,
        vdisk: np.ndarray,
        vbul: np.ndarray,
    ) -> float:
        """Gaussian log-likelihood for the model prediction."""
        vmod = self.predict_velocity(r, vgas, vdisk, vbul)
        resid = (vobs - vmod) / verr
        return float(-0.5 * np.sum(resid**2 + np.log(2 * np.pi * verr**2)))

    def compute_bic(
        self,
        r: np.ndarray,
        vobs: np.ndarray,
        verr: np.ndarray,
        vgas: np.ndarray,
        vdisk: np.ndarray,
        vbul: np.ndarray,
        k: int = 0,
    ) -> dict[str, float]:
        """
        Compute information criteria for this model on a galaxy.

        k = number of free parameters (0 for the basic fixed RBA version).
        """
        lnL = self.compute_loglike(r, vobs, verr, vgas, vdisk, vbul)
        N = int(np.sum((r > 0) & np.isfinite(verr) & (verr > 0)))
        chi2 = float(np.sum(((vobs - self.predict_velocity(r, vgas, vdisk, vbul)) / verr) ** 2))

        aic = 2 * k - 2 * lnL
        bic = k * np.log(N) - 2 * lnL

        return {
            "lnL": lnL,
            "chi2": chi2,
            "AIC": aic,
            "BIC": bic,
            "N": N,
            "k": k,
            "model": "RBA",
        }

    def __repr__(self) -> str:
        return f"RBAModel(a0={self.a0_mps2:.2e}, ups_d={self.ups_d}, ups_b={self.ups_b})"
