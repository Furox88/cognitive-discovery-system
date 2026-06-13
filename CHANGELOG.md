# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `CITATION.cff` at repository root for standard academic citation of the toolkit in research papers and pipelines.
- `examples/hypothesis_custom_generator.py` — a small runnable demonstration of supplying a custom implementation of the HypothesisGenerator Protocol for a toy domain.

### Changed
- 

## [0.2.0] - 2026-06-13

### Added
- `cds hypothesis` command for quick demos of the hypothesis generation module.
- `examples/hypothesis_demo.py` runnable example for the cognitive/hypothesis features.
- GitHub community files: PR template, issue templates, CI workflow, FUNDING.yml, CODEOWNERS.
- Basic CLI smoke tests (`tests/test_cli.py`).
- **CLI**: New `cds modules` command that lists all available scientific modules in a clear table; improved `--version` / `-v` flag and help discoverability.
- **Public API**: Cleaner top-level package exports and documentation.
- **Hypothesis module**: Improved module and generator documentation for better clarity on providing custom generators.
- **Developer experience**: Replaced the placeholder setup script with a working one-command development setup (creates venv, installs the package, runs tests).
- **Documentation**: Added a short Vision section, expanded CONTRIBUTING.md with module/CLI/hypothesis guidance, and updated quick-start examples.

### Changed
- General improvements to documentation and usability.
- README now includes live CI badge and reference to the new hypothesis demo.

### Previous (from initial push)
- **2-D FFT** (`cds.signals`): `fft2` / `ifft2` via row-column decomposition (Cooley-Tukey 1965)
- **Hypothesis testing** (`cds.stats`): one-sample & two-sample t-tests (pooled / Welch),
  chi-square goodness-of-fit and independence, one-way ANOVA, with exact distribution
  tail functions (`t_sf`, `chi2_sf`, `f_sf`) built on regularized incomplete gamma/beta
  functions (Student 1908, Pearson 1900, Fisher 1925; Numerical Recipes §6.1–6.4)
- **Linear algebra** (`cds.math_utils`): QR decomposition via Householder reflections
  (Householder 1958) and Cholesky decomposition (Benoît/Cholesky 1924)
- New runnable demos: `examples/fft2_demo.py`, `examples/hypothesis_tests_demo.py`,
  `examples/linalg_demo.py`
- Project logo/banner (`assets/logo.svg`)
- Test suite expanded to **314 tests** (see CI)

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
  - LU decomposition (Doolittle's method)
  - Linear system solver (forward/back substitution)
  - Matrix inverse
  - Eigenvalue computation (power iteration — Von Mises 1929)
  - Gram-Schmidt orthonormalization
- **Graph theory** (`cds.graph`)
  - Breadth-first search (BFS)
  - Depth-first search (DFS)
  - Dijkstra's shortest path algorithm (1959)
  - Kruskal's minimum spanning tree (1956)
  - Topological sort (Kahn's algorithm)
  - Cycle detection (DFS coloring)
- **Monte Carlo methods** (`cds.montecarlo`)
  - π estimation (unit-circle method)
  - Monte Carlo integration
  - Buffon's needle experiment (1777)
  - 1D and 2D random walks
- **Differential equations** (`cds.diffeq`)
  - Euler's method (1768)
  - Classical 4th-order Runge-Kutta (RK4)
  - Explicit midpoint method (2nd-order RK)
  - System solver for coupled ODEs
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
  - prompt templates for hypothesis work
  - Offline placeholder generator for demos
- **CLI** (`cds` command)
  - `cds constants` — list physical constants
  - `cds calc` — interactive physics calculator
  - `cds hypothesize` — generate hypotheses
- Core data models (`cds.core.models`)
- 309 tests with pytest
- Example scripts in `examples/`
- API reference documentation (`docs/api-reference.md`)
- Performance benchmarks (`docs/benchmarks.md`)
- Contributing guidelines, Code of Conduct, Security policy
- Issue and PR templates
- Getting started documentation
