"""Central numeric constants for CDS solvers and numerical kernels.

These replace scattered magic numbers (``1e-15`` near-zero guards, ``1e-7``
finite-difference steps, ``0.01`` learning rates, ``-1e9`` masking sentinels)
with named, documented constants. The values mirror the literals they replace
so behavior is unchanged; only readability and consistency improve.

Notes
-----
- ``EPSILON`` is derived from :func:`math.ulp` rather than hardcoded, so it
  reflects the actual machine epsilon on the running interpreter.
- Module-local algorithm constants (e.g. the Lanczos ``cof`` coefficients and
  ``_MAXIT`` / ``_EPS`` / ``_FPMIN`` in ``stats.hypothesis_tests``) stay where
  they are: they are intrinsic to a specific Numerical-Recipes routine, not
  shared tuning knobs.
"""

from __future__ import annotations

import math

# Machine epsilon of the running interpreter (~2.220446e-16 on CPython/IEEE-754).
# Use this instead of the legacy hardcoded ``1e-15`` singularity guards.
EPSILON: float = math.ulp(1.0)

# Near-zero threshold for singularity / divide-by-zero guards
# (pivot checks, zero-vector guards, variance floors). Kept at ``1e-15`` to
# preserve the exact behavior of the original guards.
NEAR_ZERO: float = 1e-15

# Default step size for central finite-difference gradient estimates.
DEFAULT_FD_STEP: float = 1e-7

# Generic convergence tolerance for iterative solvers (gradient descent,
# Adam, golden-section line search).
DEFAULT_TOLERANCE: float = 1e-8

# Newton-Raphson root-finding tolerance (``abs(f(x)) < tol``).
NEWTON_TOLERANCE: float = 1e-10

# Base step for the Newton-Raphson numerical derivative.
NEWTON_DERIVATIVE_STEP: float = 1e-5

# RK45 adaptive integrator defaults.
RK45_DEFAULT_ATOL: float = 1e-6
RK45_DEFAULT_RTOL: float = 1e-3
RK45_DEFAULT_DT: float = 0.01

# Adaptive-step shrink / grow factors for the RK45 controller.
RK45_STEP_SHRINK: float = 0.1
RK45_STEP_SAFETY: float = 0.9
RK45_STEP_GROW: float = 10.0

# Loop-termination epsilon for ODE time stepping (``while t < t_end - eps``).
LOOP_EPSILON: float = 1e-12

# Near-minimum positive float (Numerical Recipes ``FPMIN``), used to avoid
# underflow in continued-fraction / series evaluations.
NEAR_MIN_FLOAT: float = 1e-300

# Adam-family optimizer defaults (Kingma & Ba 2015). Unifies the defaults
# previously split between ``optimization.minimize`` (lr=0.01) and
# ``nlp.optim`` (lr=1e-3) — both now read ``ADAM_DEFAULT_LR``.
ADAM_DEFAULT_LR: float = 1e-3
ADAM_DEFAULT_BETAS: tuple[float, float] = (0.9, 0.999)
ADAM_DEFAULT_EPS: float = 1e-8

# SGD learning rate default.
SGD_DEFAULT_LR: float = 0.01

# MLP / plain gradient-descent learning rate default.
GD_DEFAULT_LR: float = 0.01

# Layer normalization variance-floor epsilon.
LAYERNORM_EPS: float = 1e-5

# Sampling temperature divide-by-zero guard.
SAMPLING_TEMPERATURE_EPS: float = 1e-8

# Softmax sampling divide-by-zero guard (legacy ``1e-8``).
SOFTMAX_TEMPERATURE_EPS: float = 1e-8

# Masked-fill sentinel replacing the legacy ``-1e9`` literal in NLP attention
# masks. ``-inf`` propagates to a true zero after softmax; ``-1e9`` only
# underflows to ~0, leaving a small residual probability.
NEG_INF_MASK: float = float("-inf")

# Entanglement (concurrence) detection threshold.
CONCURRENCE_THRESHOLD: float = 1e-9

# Plotting normalization floor (avoids division by zero on flat signals).
PLOT_NORMALIZATION_FLOOR: float = 1e-10
