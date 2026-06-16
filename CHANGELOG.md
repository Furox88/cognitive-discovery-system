# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.8.2] - 2026-06-16

### Added
- **Docstrings** for all public functions and classes (previously 137/168 functions and 22/26 classes were documented; now 100%).

### Fixed
- **Empty `src/cds/knowledge/` directory removed** (stale artifact from a v0.7.0 cleanup that left a leftover empty package).
- **CHANGELOG backfill**: added missing entries for v0.4.0 and v0.5.0 to align the file with the actual git tag history.

## [0.8.1] - 2026-06-16

### Added
- **Pre-commit hooks** (`.pre-commit-config.yaml`): Ruff (lint + format), Mypy (type check), and standard file checks run on every commit. CI mirrors the same checks.
- **Dependency lockfiles** (`requirements.lock`, `requirements-dev.lock`): fully pinned dependency snapshots for reproducible installs. Regenerate via `pip-compile` (see CONTRIBUTING.md).
- **Coverage threshold gate**: CI fails if coverage drops below 100% (`--cov-fail-under=100`). Configured in `pyproject.toml` under `[tool.coverage.*]`.

## [0.5.0] - 2026-06-15

Internal: package rename to `cognitive-discovery-toolkit`, then to `cognitive-discovery-platform`. 350 tests, full metadata synchronization, API finalization. (PyPI: yes)

## [0.4.0] - 2026-06-15

Internal: interim package rename to `cds-cognitive` to avoid a PyPI name collision. Superseded by 0.5.0. (PyPI: no)

## [0.3.3] - 2026-06-15
## [0.3.2] - 2026-06-15
## [0.3.1] - 2026-06-15
## [0.3.0] - 2026-06-14
## [0.2.0] - 2026-06-14
## [0.1.0] - 2026-06-09

### Added
- **Pre-commit hooks** (`.pre-commit-config.yaml`): Ruff (lint + format), Mypy (type check), and standard file checks run on every commit. CI mirrors the same checks.
- **Dependency lockfiles** (`requirements.lock`, `requirements-dev.lock`): fully pinned dependency snapshots for reproducible installs. Regenerate via `pip-compile` (see CONTRIBUTING.md).
- **Coverage threshold gate**: CI fails if coverage drops below 100% (`--cov-fail-under=100`). Configured in `pyproject.toml` under `[tool.coverage.*]`.
- **Pre-commit job in CI**: `.github/workflows/tests.yml` now has a dedicated `pre-commit` job alongside the existing test job.
- **Docs deploy workflow** (`.github/workflows/pages.yml`): builds and deploys the mkdocs site to GitHub Pages on every push to main.
- **Publish script** (`scripts/publish.ps1`): single-command local PyPI release workflow (verify, test, build, upload).
- **Coverage tests**: added `test_independence_zero_row_total` and `test_independence_zero_col_total` for the `if exp > 0` False branch in `chi_square_independence`.

### Changed
- `CONTRIBUTING.md` updated with pre-commit setup, lockfile regeneration, and `ruff format` instructions.
- `.github/workflows/tests.yml` extended: added `ruff format --check`, coverage threshold gate, and separate pre-commit job.
- Codebase reformatted with `ruff format` (85 files).
- Test count: 570 → 572.

## [0.8.0] - 2026-06-16

### Fixed
- **`adaptive_simpson` hang on NaN/divergent integrands**: when the integrand produced `NaN` (e.g. `float("nan")` or a divergent integrand), the Lyness error estimate `diff` was `NaN`, making `abs(diff) <= 15*eps` always `False`. The recursion then branched down to `max_depth`, making up to `2**max_depth` calls and effectively hanging the suite (the test ran for 10+ minutes). Added an early-exit: if `diff` is `NaN`, the subinterval stops recursing so the `NaN` propagates to the top-level `RuntimeError` guard. The previously hanging test `test_divergent_integrand_raises` now passes in <0.2 s. Full suite runtime dropped from 10 min (timeout) to ~19 s, and test coverage reached **100%** (570 tests).

### Changed
- README badges and stats updated: tests **565 → 570**, coverage **99% → 100%**.

## [0.7.0] - 2026-06-16

### Added
- **Numerical Integration** (`cds.numerical_integration`): Deterministic quadrature module with `trapezoid`, `simpson` (1/3), `simpson_38` (3/8), `romberg` (Richardson extrapolation), `gaussian_quadrature` (Gauss-Legendre), and `adaptive_simpson` (recursive adaptive). Complements the stochastic integration in `cds.montecarlo` and the ODE solvers in `cds.diffeq`. Gauss-Legendre nodes/weights are derived and cached; exact for polynomials up to degree 2n−1. Backed by 36 new tests (Newton-Cotes 1722, Simpson 1743, Romberg 1955, Gauss 1814, Lyness 1969).
- Top-level `cds.numerical_integration` export registered in package `__init__` and `__all__`.

### Changed
- README module table, module count (14 → 15), architecture tree, and test badge (440+ → 565) updated to reflect the new module. `docs/api.md` and `docs/api-reference.md` now document `cds.numerical_integration`.
- Version bumped to 0.7.0 (pyproject.toml, src/cds/__init__.py, CITATION.cff, README, docs/index.md).

### Fixed
- **Public API**: `rz_gate` was missing from the `cds.quantum` package exports (`__init__.py` / `__all__`) despite being documented and tested. Now importable via `from cds.quantum import rz_gate`.
- **Documentation accuracy**: `lu_decomposition` return type corrected across docs from `(L, U)` to `(P, L, U)` (the implementation uses partial pivoting, PA = LU), and the misleading "Doolittle" label removed.
- **Consistency sweep**: aligned test-count and coverage references across README, CLI (`cds info`), `docs/index.md`, `getting-started.md`, `CONTRIBUTING.md`, `ROADMAP.md`, and `benchmarks.md`. Coverage now reported as **99%** (previously stale 97% / 92%); test count reported as **565** (previously stale 350/380/440). Resolved the quantum speedup contradiction in `benchmarks.md` (61.1x → 60.3x, matching the visual proof and the actual 0.41s/0.0068s ratio). Added missing `cds.scientific` and `cds.probability` sections to `docs/api.md`.
- **api-reference.md vs source-code audit**: corrected 15 discrepancies — fixed `QuantumRegister.normalize()` return type (None, not QuantumRegister); fixed `cds.scientific` constants list (removed phantom `epsilon_0`/`mu_0`, added `hbar`/`pi`/`e_math`); added missing `rk45` (diffeq), `DataSet`/`plot_bar`/`plot_line` (data_analysis), `QuantumGate`/`Qubit` (quantum), and 6 hypothesis exports (`Domain`, `Hypothesis`, `HypothesisStatus`, `HypothesisGenerator`, `HypothesisEvaluator`, `EvaluationResult`); aligned parameter names across optimization (`h`, `h_base`, `state`, `grad_f`), stats (`data`), math_utils (`h_base`, `point`), scientific (`n_moles`, `temperature`, `volume`), and hypothesis (`research_question`). Added missing `numerical_integration` module to CLI `cds info` panel.
- **Error messages**: replaced 11 generic/internal messages with actionable guidance across `linalg` (5), `signals` (3), and `stats` (3). Internal helper names (`gammp`/`gammq`) replaced with mathematical domain descriptions. Parameters, offending values, and fix suggestions now included where applicable. All existing `pytest.raises(match=...)` substring assertions preserved.
- **ROADMAP**: all v0.7.0 checkboxes marked complete.

## [0.6.0] - 2026-06-14

### Added
- **Machine Learning** (`cds.ml`): Pure Python Neural Networks with `MLP` and `Layer` abstractions. Support for Adam-based training and common activation functions.
- **Interactive Dashboard**: Streamlit-based web dashboard for real-time scientific exploration, including a live Hypothesis Engine and Quantum Circuit Simulator.
- **Improved Monte Carlo**: Parallelized Multi-Core Monte Carlo engines for faster π estimation and integration.
- **Data Analysis Viz**: ASCII-based line plotting directly in the terminal via `cds plot`.

### Changed
- **Numerical Core**: Refactored `math_utils` with O(N³) pure-Python Partial Pivoting LU decomposition, improving stability and performance for linear systems and determinants.
- **Optimization**: Vectorized machine learning optimizers for improved training efficiency.
- **Test Coverage**: Achieved **92%+ code coverage** with 350 tests.
- **CLI Refresh**: `cds info` now displays a comprehensive status panel with 14 core modules and platform health info.
- Version bumped to 0.6.0 (pyproject.toml, src/cds/__init__.py, README, CLI info).

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
  - LU decomposition (partial pivoting)
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
