"""
cognitive-discovery-platform

Pure Python computational science platform for research, simulation,
and scientific discovery.

Key features:
- Zero heavy dependencies (pure Python)
- Quantum simulation (single & multi-qubit with entanglement)
- Signal processing (FFT, 2D FFT, convolution, filtering)
- Optimization, statistics, probability, linear algebra
- Hypothesis generation for structured research ideas
- CLI for quick calculations and discovery workflows

All modules are designed to be readable, testable, and usable
for education, research, and custom scientific discovery workflows.

Usage:
    import cds
    print(cds.__version__)

    from cds.quantum import ghz_state, is_entangled
    from cds.hypothesis import generate_hypotheses
"""

__version__ = "0.5.0"

# Convenient top-level re-exports for common scientific tools
from cds.scientific.constants import CONSTANTS, get_constant
from cds.scientific.formulas import (
    kinetic_energy, gravitational_force, wave_frequency, ideal_gas_pressure,
    schwarzschild_radius, de_broglie_wavelength, escape_velocity, photon_energy,
)

# Core modules
from cds import core
from cds import data_analysis
from cds import ml

# Scientific computing modules
from cds import diffeq
from cds import graph
from cds import hypothesis
from cds import knowledge
from cds import math_utils
from cds import montecarlo
from cds import optimization
from cds import probability
from cds import quantum
from cds import scientific
from cds import signals
from cds import stats

__all__ = [
    "__version__",
    "CONSTANTS",
    "get_constant",
    "kinetic_energy",
    "gravitational_force",
    "wave_frequency",
    "ideal_gas_pressure",
    "schwarzschild_radius",
    "de_broglie_wavelength",
    "escape_velocity",
    "photon_energy",
    "core",
    "data_analysis",
    "ml",
    "diffeq",
    "graph",
    "hypothesis",
    "knowledge",
    "math_utils",
    "montecarlo",
    "optimization",
    "probability",
    "quantum",
    "scientific",
    "signals",
    "stats",
]
