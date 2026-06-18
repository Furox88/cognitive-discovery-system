# Cognitive Discovery Platform (CDS)

Welcome to the official documentation for the **Cognitive Discovery Platform (CDS)**.

CDS is an open-source computational science platform designed for research, simulation, and discovery. It provides a lightweight, dependency-free environment for scientific exploration, featuring 16 modules covering everything from quantum simulation to educational NLP primitives and automated hypothesis generation.

## Key Features

- **Pure Python:** Every module is implemented from scratch using the Python standard library. No heavy dependencies like NumPy or SciPy required.
- **Quantum Simulation:** Full state-vector simulation for single and multi-qubit circuits with entanglement and O(1) sampling.
- **Advanced Mathematics:** O(N³) Partial Pivoting LU decomposition, vectorized optimizers, and adaptive ODE solvers (RK45).
- **Hypothesis Engine:** Built-in tools for generating and statistically validating scientific hypotheses.
- **High Reliability:** 883 tests with 99% code coverage.
- **Interactive Tools:** Beautiful CLI and a Streamlit-based web dashboard.

## Overview of Modules

| Module | Description |
|--------|-------------|
| `cds.core` | Shared data models (`Domain`, `Hypothesis`, `HypothesisStatus`) |
| `cds.quantum` | Single & multi-qubit quantum circuit simulation |
| `cds.optimization` | Gradient-based and numerical optimizers |
| `cds.ml` | Pure Python Neural Networks (MLP, Adam-based training) |
| `cds.signals` | Fast signal processing (DFT, FFT/IFFT, convolution, filtering) |
| `cds.probability` | Probability distributions & sampling |
| `cds.stats` | Statistical analysis, regression & hypothesis testing |
| `cds.math_utils` | Numerical calculus, linear algebra, eigenvalues |
| `cds.data_analysis` | Structured data management and visualization |
| `cds.scientific` | Physical constants & scientific formulas |
| `cds.graph` | Graph algorithms (BFS, DFS, Dijkstra, Kruskal MST) |
| `cds.montecarlo` | Monte Carlo integration, π estimation, random walks |
| `cds.diffeq` | ODE solvers (Euler, RK4, midpoint) |
| `cds.numerical_integration` | Deterministic quadrature (trapezoid, Simpson, Romberg) |
| `cds.nlp` | Educational NLP from scratch (BPE, embeddings, attention, autograd, MiniGPT) |
| `cds.hypothesis` | Cognitive discovery and structured hypothesis generation |

## Quick Navigation

- [Getting Started](getting-started.md)
- [API Reference](api.md)
- [Case Studies](CASE_STUDY_HUBBLE.md)
- [Benchmarks](benchmarks.md)

---
*CDS v1.0.4 is stable and actively developed. Contributions are welcome!*
