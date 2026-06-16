# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-06-14

### Added
- **Machine Learning** (`cds.ml`): Pure Python Neural Networks with `MLP` and `Layer` abstractions. Support for Adam-based training and common activation functions.
- **Interactive Dashboard**: Streamlit-based web dashboard for real-time scientific exploration, including a live Hypothesis Engine and Quantum Circuit Simulator.
- **Improved Monte Carlo**: Parallelized Multi-Core Monte Carlo engines for faster π estimation and integration.
- **Data Analysis Viz**: ASCII-based line plotting directly in the terminal via `cds plot`.

### Changed
- **Numerical Core**: Refactored `math_utils` with O(N³) pure-Python Partial Pivoting LU decomposition, improving stability and performance for linear systems and determinants.
- **Optimization**: Vectorized machine learning optimizers for improved training efficiency.
- **Test Coverage**: Achieved **95%+ code coverage** with 350+ tests.
- **CLI Refresh**: `cds info` now displays a comprehensive status panel with 14 core modules and platform health info.
- Version bumped to 0.5.0 (pyproject.toml, src/cds/__init__.py, README, CLI info).

## [0.3.0] - 2026-06-13

### Added
- `docs/research-workflows.md` — guidance on using CDS in real research and discovery pipelines: combining hypothesis generation with stats/optimization/Monte Carlo, implementing custom `HypothesisGenerator` implementations, and embedding modules in larger scripts.
- `CITATION.cff` at repository root for standard academic citation of the platform in research papers and pipelines.
- `examples/hypothesis_custom_generator.py` — a small runnable demonstration of supplying a custom implementation of the HypothesisGenerator Protocol for a toy domain.

### Changed
- All references to the test suite updated to exactly **350 tests** (README, docs, CONTRIBUTING.md, CLI).
- `docs/benchmarks.md` refreshed with fresh micro-benchmark timings (hypothesis generation ~0.1 ms for n=3, stats operations, quantum primitives, optimization).
- Cleaned Codecov placeholder comment from README (badge remains; live data requires connecting the repo on codecov.io).
- Version bumped to 0.3.0 (pyproject.toml, src/cds/__init__.py, README alpha notice, CLI info/status).
- Cross-references to the new research workflows documentation added throughout README, getting-started, and CONTRIBUTING.
- GitHub repository description and topics updated via gh for better visibility (topics include pure-python, scientific-computing, hypothesis-generation, etc.).

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
- Test suite expanded to **350 tests** (see CI)

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
- 350 tests with pytest
- Example scripts in `examples/`
- API reference documentation (`docs/api-reference.md`)
- Performance benchmarks (`docs/benchmarks.md`)
- Contributing guidelines, Code of Conduct, Security policy
- Issue and PR templates
- Getting started documentation
