<p align="center">
  <img src="assets/logo.svg" alt="Cognitive Discovery System" width="640">
</p>

# Cognitive Discovery System (CDS)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-300%2B-brightgreen.svg)]()
[![CI](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/tests.yml/badge.svg)](https://github.com/Furox88/cognitive-discovery-system/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Furox88/cognitive-discovery-system/branch/main/graph/badge.svg)](https://codecov.io/gh/Furox88/cognitive-discovery-system)
<!-- Connect repo on codecov.io for live coverage data (badge is ready) -->

**Open-source computational science toolkit for research, simulation, and discovery.**

CDS brings together quantum computing simulation, statistical analysis, signal processing, optimization, probability, and scientific computing in a single, dependency-light Python package. Every module is pure Python — no NumPy or SciPy required.

The project also includes built-in support for structured hypothesis generation, making it easier to explore ideas and connect them to simulation or analysis tools.

> Currently in **alpha (v0.2.0)**. Contributions welcome!

## Why CDS?

- **Zero heavy dependencies** — pure Python implementations you can read and learn from
- **Quantum simulation** — single & multi-qubit circuits with entanglement
- **Built for discovery** — hypothesis generation with structured outputs (assumptions, predictions, confidence) plus a Protocol for custom implementations
- **Broad scope** — 12 modules covering math, physics, stats, signals, optimization, graph theory, ODEs, Monte Carlo
- **300+ tests** (see CI) — thoroughly tested with pytest
- **Practical automation** — workflows for PR checklists, dependency updates, and releases to keep maintenance manageable
- **CLI included** — interactive tools and demos from your terminal

## Citing CDS

If CDS is useful in your research or publications, please cite it using the information in `CITATION.cff` at the repository root. This helps give proper credit and track adoption in scientific work.

## Modules

| Module | Description |
|--------|-------------|
| `cds.quantum` | Single & multi-qubit simulation — Hadamard, Pauli, CNOT, SWAP, Toffoli, Bell/GHZ states, entanglement detection |
| `cds.optimization` | Gradient descent, Newton's method, Adam optimizer, golden section search |
| `cds.signals` | DFT, FFT (radix-2), 2-D FFT, convolution, power spectrum, low-pass filter |
| `cds.probability` | Gaussian, uniform, exponential, binomial, Poisson distributions |
| `cds.stats` | Descriptive stats, Pearson correlation, linear regression, t-test, chi-square, ANOVA |
| `cds.math_utils` | Numerical calculus, LU / QR / Cholesky decomposition, eigenvalue (power iteration), Gram-Schmidt, matrix inverse |
| `cds.data_analysis` | CSV loader, normalization, z-score, moving average |
| `cds.scientific` | Physical constants, formulas (KE, gravity, gas law, Schwarzschild, de Broglie, escape velocity) |
| `cds.graph` | BFS, DFS, Dijkstra shortest path, Kruskal MST, topological sort, cycle detection |
| `cds.montecarlo` | Monte Carlo integration, π estimation, Buffon's needle, random walks (1D/2D) |
| `cds.diffeq` | Euler method, RK4, midpoint method, ODE system solver |
| `cds.hypothesis` | Structured hypothesis generation with prompt templates for custom research workflows |

## Quick Start

```bash
git clone https://github.com/Furox88/cognitive-discovery-system.git
cd cognitive-discovery-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run CLI
cds --help
cds constants
cds calc ke
cds modules
cds hypothesis "What causes the Hubble tension?"
```

## Examples

See the [`examples/`](examples/) directory for runnable demos.

### Hypothesis Generation (cognitive discovery)

```bash
# Basic demo
python examples/hypothesis_demo.py

# With stats / experiment sketch example
python examples/hypothesis_with_stats_demo.py

# Custom generator implementation (using the HypothesisGenerator Protocol)
python examples/hypothesis_custom_generator.py

# Or via CLI
cds hypothesis "What causes the Hubble tension?"
```

### Quantum Circuit (single qubit)
```python
from cds.quantum import QuantumCircuit, hadamard, pauli_x, simulate

circuit = QuantumCircuit().add(hadamard()).add(pauli_x())
result = circuit.run()
print(result.probabilities())

counts = simulate(circuit, shots=1000)
print(counts)  # {0: ~500, 1: ~500}
```

### Multi-Qubit & Entanglement
```python
from cds.quantum import (
    QuantumRegister, h_gate, cnot, bell_state,
    ghz_state, is_entangled,
)

# Bell state (|00⟩ + |11⟩) / √2
reg = bell_state(0)
print(is_entangled(reg))  # True
print(reg.measure_shots(shots=1000))  # {'00': ~500, '11': ~500}

# 4-qubit GHZ state
ghz = ghz_state(4)
counts = ghz.measure_shots(shots=1000)
print(counts)  # {'0000': ~500, '1111': ~500}
```

### Optimization
```python
from cds.optimization import gradient_descent, newton_method

# Find minimum of (x-3)²
result = gradient_descent(lambda x: (x - 3) ** 2, x0=10.0, lr=0.1)
print(f"x = {result.x:.6f}")  # ~3.0

# Find √2 using Newton's method
result = newton_method(lambda x: x ** 2 - 2, x0=1.5)
print(f"√2 = {result.x:.10f}")  # 1.4142135624
```

### Signal Processing
```python
from cds.signals import dft, fft_radix2, convolve, low_pass_filter

# FFT of a signal
signal = [complex(i) for i in range(8)]
spectrum = fft_radix2(signal)

# Convolution
result = convolve([1.0, 2.0, 3.0], [0.5, 0.5])
print(result)  # [0.5, 1.5, 2.5, 1.5]
```

### Probability Distributions
```python
from cds.probability import gaussian_pdf, binomial_pmf, poisson_pmf

# Gaussian PDF at x=0
print(gaussian_pdf(0.0, mu=0, sigma=1))  # 0.3989...

# Binomial: P(3 heads in 5 fair flips)
print(binomial_pmf(3, 5, 0.5))  # 0.3125

# Poisson: P(k=2, λ=3)
print(poisson_pmf(2, 3.0))  # 0.2240...
```

### Statistics
```python
from cds.stats import mean, stdev, correlation, linear_regression

data = [12.5, 14.3, 11.8, 15.1, 13.7]
print(f"mean={mean(data):.2f}, std={stdev(data):.2f}")

x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
reg = linear_regression(x, y)
print(f"y = {reg.slope:.2f}x + {reg.intercept:.2f}, R²={reg.r_squared:.3f}")
```

### Scientific Computing
```python
from cds.scientific import kinetic_energy, escape_velocity, get_constant

print(get_constant("c"))          # speed of light
print(kinetic_energy(10, 5))      # 125.0 J
print(escape_velocity(5.972e24, 6.371e6))  # ~11186 m/s
```

### Graph Theory
```python
from cds.graph import Graph, dijkstra, kruskal_mst, bfs

g = Graph(n_vertices=4, directed=False)
g.add_edge(0, 1, 1.0)
g.add_edge(1, 2, 2.0)
g.add_edge(2, 3, 3.0)
g.add_edge(0, 3, 10.0)

dist, prev = dijkstra(g, 0)
print(dist)  # {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}

edges, total = kruskal_mst(g)
print(f"MST weight: {total}")  # 6.0
```

### Monte Carlo Simulation
```python
from cds.montecarlo import estimate_pi, mc_integrate
import math

result = estimate_pi(n_samples=100_000, seed=42)
print(f"π ≈ {result.estimate:.4f}")  # ~3.1416

area = mc_integrate(math.sin, 0, math.pi, n_samples=100_000)
print(f"∫sin(x)dx = {area.estimate:.4f}")  # ~2.0
```

### Differential Equations
```python
from cds.diffeq import rk4, solve_system
import math

# dy/dt = -y, y(0)=1  =>  y(t) = e^(-t)
sol = rk4(lambda t, y: -y, t0=0, y0=1.0, t_end=2.0)
print(f"y(2) = {sol.y[-1]:.6f}")  # ~0.135335 (e^-2)

# Harmonic oscillator: x'' = -x
def harmonic(t, y):
    return [y[1], -y[0]]
t_vals, y_vals = solve_system(harmonic, 0, [1.0, 0.0], math.pi)
print(f"x(π) = {y_vals[-1][0]:.4f}")  # ~-1.0
```

## Architecture

```
src/cds/
├── quantum/        # Quantum circuit simulation (single & multi-qubit)
├── optimization/   # Gradient descent, Newton, Adam, line search
├── signals/        # DFT, FFT, convolution, filtering
├── probability/    # Probability distributions & sampling
├── stats/          # Statistical analysis & regression
├── math_utils/     # Calculus, linear algebra, eigenvalues, Gram-Schmidt
├── data_analysis/  # CSV loading & data transforms
├── scientific/     # Physical constants & formulas
├── graph/          # Graph algorithms (Dijkstra, BFS, DFS, Kruskal MST)
├── montecarlo/     # Monte Carlo methods (π, integration, random walks)
├── diffeq/         # ODE solvers (Euler, RK4, midpoint)
├── hypothesis/     # Hypothesis generation
├── core/           # Shared models, config
└── cli.py          # Command-line interface

examples/           # Runnable demo scripts
tests/              # 300+ tests (see CI)
docs/               # Documentation (getting started, API reference, benchmarks)

.github/workflows/  # Automation for PRs (labels + checklist), releases, and dependency updates
```

## Vision

The long-term goal of CDS is to provide a lightweight, dependency-free toolkit for scientific exploration and discovery.

We aim to combine solid numerical foundations (quantum simulation, FFT, linear algebra, statistics, differential equations, etc.) with higher-level tools for hypothesis generation and research workflows.

A distinctive part is the `cds.hypothesis` module, which generates structured, falsifiable hypotheses with explicit assumptions and predictions. The `cds hypothesis` CLI command and `examples/hypothesis_demo.py` make this side immediately usable. Recent work has focused on making the CLI and docs more practical for day-to-day use while keeping everything readable pure Python.

The project is still early but is being actively developed with a focus on code quality, test coverage, documentation, and usability for researchers and students.

Run `cds modules` after installation to explore the current modules.

## Recent improvements

Recent updates have aimed to make it simpler to generate and explore ideas within the toolkit:

- New CLI commands for browsing available modules and experimenting with hypothesis generation
- A dedicated example showing how to use the hypothesis features end-to-end
- Automation around pull requests, dependency management, and releases to free up time for core scientific work

The goal is to lower the barrier for using the discovery-oriented parts of the project and reduce time spent on routine tasks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Looking for:
- Researchers with domain expertise
- People interested in pure-Python scientific computing
- Contributors for new modules (ML basics, PDE solvers, etc.)
- People who want to help make scientific tools easier to maintain and use


## Automation and Maintenance Workflows

A few GitHub Actions handle repetitive aspects of keeping the project running:

- Dependabot for regular updates to dependencies and GitHub Actions
- Automatic labeling and review checklists for pull requests
- A release process that builds and publishes on version tags

These help ensure that time spent on the project goes more toward developing new modules, improving hypothesis tools, and supporting research use cases rather than manual upkeep.

See `.github/workflows/` for the current setup.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Maintainer: [@Furox88](https://github.com/Furox88)
- Issues & Discussions: [GitHub](https://github.com/Furox88/cognitive-discovery-system/issues)


## Why These Automations Exist

The project is maintained by a small team (often solo). The workflows above exist so that routine tasks (labeling PRs, running checks, cutting releases, keeping dependencies fresh) take as little time as possible. This frees hours for actual research work: improving the hypothesis tools, adding new scientific modules, writing better examples, and exploring new discovery workflows.

If you're a researcher or educator using CDS, these automations mean you can focus on the science instead of repo housekeeping.
