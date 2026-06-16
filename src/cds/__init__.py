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

__version__ = "0.6.0"

# Convenient top-level re-exports for common scientific tools
# Core modules
# Scientific computing modules
from cds import (
    core,
    data_analysis,
    diffeq,
    graph,
    hypothesis,
    math_utils,
    ml,
    montecarlo,
    optimization,
    probability,
    quantum,
    scientific,
    signals,
    stats,
)
from cds.scientific.constants import CONSTANTS, get_constant
from cds.scientific.formulas import (
    de_broglie_wavelength,
    escape_velocity,
    gravitational_force,
    ideal_gas_pressure,
    kinetic_energy,
    photon_energy,
    schwarzschild_radius,
    wave_frequency,
)

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
    "math_utils",
    "montecarlo",
    "optimization",
    "probability",
    "quantum",
    "scientific",
    "signals",
    "stats",
]
