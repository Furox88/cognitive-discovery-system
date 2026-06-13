"""
Cognitive Discovery System (CDS)

Pure Python computational science toolkit for research, simulation,
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

__version__ = "0.1.0"

# Convenient top-level re-exports for common scientific tools
# (full submodules remain importable as cds.xxx)
from cds.scientific.constants import CONSTANTS, get_constant

__all__ = [
    "__version__",
    "CONSTANTS",
    "get_constant",
]
