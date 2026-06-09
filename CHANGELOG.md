# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-09

### Added
- **Quantum computing** (`cds.quantum`)
  - Single-qubit gates: Hadamard, Pauli-X/Y/Z, phase, Rz rotation
  - Multi-qubit register with state vector simulation
  - Two-qubit gates: CNOT, CZ, SWAP
  - Three-qubit gate: Toffoli (CCNOT)
  - Bell state and GHZ state preparation
  - Entanglement detection via concurrence
  - Measurement with shot-based sampling
- **Optimization** (`cds.optimization`)
  - Gradient descent minimization
  - Newton-Raphson root finding
  - Adam optimizer (adaptive learning rate)
  - Golden section line search
- **Signal processing** (`cds.signals`)
  - Discrete Fourier Transform (DFT)
  - Fast Fourier Transform (Cooley-Tukey radix-2)
  - Linear convolution
  - Power spectrum computation
  - Frequency-domain low-pass filter
- **Probability** (`cds.probability`)
  - Gaussian, uniform, exponential PDF
  - Binomial and Poisson PMF
  - Uniform random sampling
- **Statistics** (`cds.stats`)
  - Mean, median, variance, standard deviation
  - Pearson correlation coefficient
  - Linear regression with R² and prediction
- **Numerical math** (`cds.math_utils`)
  - Derivatives (central difference)
  - Integrals (Simpson's rule)
  - Gradient computation
  - Matrix multiplication, transpose, determinant
- **Data analysis** (`cds.data_analysis`)
  - CSV file loader
  - Normalization, z-score standardization
  - Moving average
- **Scientific computing** (`cds.scientific`)
  - Physical constants (c, G, h, k_B, N_A, R, etc.)
  - Kinetic energy, gravitational force, wave frequency
  - Ideal gas law, photon energy, Schwarzschild radius
  - De Broglie wavelength, escape velocity
- **Hypothesis generation** (`cds.hypothesis`)
  - LLM-ready prompt templates
  - Offline placeholder generator for demos
- **CLI** (`cds` command)
  - `cds constants` — list physical constants
  - `cds calc` — interactive physics calculator
  - `cds hypothesize` — generate hypotheses
- Core data models (`cds.core.models`)
- 200+ tests with pytest
- Example scripts in `examples/`
- API reference documentation (`docs/api-reference.md`)
- Performance benchmarks (`docs/benchmarks.md`)
- Contributing guidelines, Code of Conduct, Security policy
- Issue and PR templates
- Getting started documentation
